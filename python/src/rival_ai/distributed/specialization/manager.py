"""
Specialization manager for handling different model specialties.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime
import numpy as np
import chess
import json
import os
import torch
from torch.utils.data import DataLoader

from rival_ai.distributed.config import ModelManagerConfig
from rival_ai.distributed.game_collector.collector import CollectedGame
from rival_ai.models.gnn import ChessGNN
from rival_ai.training.trainer import Trainer
from rival_ai.training.metrics import calculate_metrics
from rival_ai.training.loss import PolicyValueLoss
from .position_analyzer import PositionAnalyzer, PositionFeatures
from .encoder import ChessEncoder
from .opening_book import OpeningManager

logger = logging.getLogger(__name__)

@dataclass
class SpecialtyConfig:
    """Configuration for a model specialty."""
    name: str
    description: str
    training_weight: float = 1.0  # Weight in mixed training
    eval_threshold: float = 0.55  # Required win rate for promotion
    min_games: int = 100  # Minimum games before evaluation
    
    # Specialty-specific parameters
    opening_book: Optional[str] = None  # Path to opening book
    endgame_tablebase: Optional[str] = None  # Path to tablebase
    variant_rules: Optional[Dict] = None  # Custom rules for variants
    style_params: Optional[Dict] = None  # Style-specific parameters

@dataclass
class SpecialtyStats:
    """Statistics for a model specialty."""
    games_played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    avg_elo: float = 1500.0
    last_updated: float = field(default_factory=lambda: datetime.now().timestamp())
    
    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        if self.games_played == 0:
            return 0.0
        return (self.wins + 0.5 * self.draws) / self.games_played

@dataclass
class SpecializedModel:
    """A model specialized for a particular aspect of play."""
    model_id: str
    specialty: str
    base_version: str
    created: float = field(default_factory=lambda: datetime.now().timestamp())
    stats: SpecialtyStats = field(default_factory=SpecialtyStats)
    active: bool = True

class SpecializationManager:
    """Manages model specialization and training."""
    
    def __init__(self, config_path: str):
        """Initialize the specialization manager.
        
        Args:
            config_path: Path to specialization config file
        """
        self.config_path = config_path
        self.specialties: Dict[str, SpecialtyConfig] = {}
        self.models: Dict[str, SpecializedModel] = {}
        self.game_cache: Dict[str, List[CollectedGame]] = {}
        self.position_analyzer = PositionAnalyzer()
        self.encoder = ChessEncoder()
        self.opening_manager = OpeningManager()
        
        # Load configurations
        self._load_config()
        self._load_models()
        
    def _load_config(self):
        """Load specialty configurations."""
        try:
            with open(self.config_path) as f:
                configs = json.load(f)
                
            for name, config in configs.items():
                self.specialties[name] = SpecialtyConfig(
                    name=name,
                    **config
                )
                
        except Exception as e:
            logger.error(f"Error loading specialization config: {e}")
            # Load default specialties
            self._load_default_specialties()
            
    def _load_default_specialties(self):
        """Load default specialty configurations."""
        self.specialties.update({
            "opening": SpecialtyConfig(
                name="opening",
                description="Opening specialist (first 15 moves)",
                training_weight=1.2,
                style_params={"max_move": 15},
                opening_book="main_repertoire.json"
            ),
            "queens_gambit": SpecialtyConfig(
                name="queens_gambit",
                description="Queen's Gambit specialist",
                training_weight=0.8,
                opening_book="queens_gambit.json"
            ),
            "sicilian": SpecialtyConfig(
                name="sicilian",
                description="Sicilian Defense specialist",
                training_weight=0.8,
                opening_book="sicilian.json"
            ),
            "middlegame": SpecialtyConfig(
                name="middlegame",
                description="Middlegame specialist",
                training_weight=1.0,
                style_params={"min_move": 15, "max_move": 40}
            ),
            "endgame": SpecialtyConfig(
                name="endgame",
                description="Endgame specialist",
                training_weight=1.1,
                style_params={"min_pieces": 4, "max_pieces": 10}
            ),
            "tactical": SpecialtyConfig(
                name="tactical",
                description="Tactical play specialist",
                training_weight=1.2,
                style_params={"aggression": 0.8}
            ),
            "positional": SpecialtyConfig(
                name="positional",
                description="Positional play specialist",
                training_weight=1.0,
                style_params={"positional_weight": 0.7}
            ),
        })
        
    def _load_models(self):
        """Load existing specialized models."""
        model_dir = "models/specialized"
        if not os.path.exists(model_dir):
            return
            
        try:
            with open(os.path.join(model_dir, "metadata.json")) as f:
                metadata = json.load(f)
                
            for model_data in metadata["models"]:
                model = SpecializedModel(**model_data)
                self.models[model.model_id] = model
                
        except Exception as e:
            logger.error(f"Error loading specialized models: {e}")
            
    def _save_models(self):
        """Save specialized model metadata."""
        model_dir = "models/specialized"
        os.makedirs(model_dir, exist_ok=True)
        
        try:
            metadata = {
                "models": [vars(m) for m in self.models.values()]
            }
            
            with open(os.path.join(model_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving specialized models: {e}")
            
    async def create_specialist(
        self,
        specialty: str,
        base_model: ChessGNN,
        base_version: str,
        pgn_path: Optional[str] = None
    ) -> Optional[str]:
        """Create a new specialized model.
        
        Args:
            specialty: Type of specialization
            base_model: Base model to specialize
            base_version: Version of base model
            pgn_path: Path to PGN file for opening book
            
        Returns:
            ID of created model, or None if failed
        """
        if specialty not in self.specialties:
            logger.error(f"Unknown specialty: {specialty}")
            return None
            
        try:
            # Create specialized version
            model_id = f"{specialty}_{base_version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Initialize with base model
            specialized = SpecializedModel(
                model_id=model_id,
                specialty=specialty,
                base_version=base_version
            )
            
            # Save model
            model_path = f"models/specialized/{model_id}.pt"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(base_model.state_dict(), model_path)
            
            # Add to opening book if PGN provided
            config = self.specialties[specialty]
            if pgn_path and config.opening_book:
                self.opening_manager.add_pgn_file(
                    config.opening_book,
                    pgn_path,
                    max_moves=config.style_params.get("max_move", 20)
                )
                
            self.models[model_id] = specialized
            self._save_models()
            
            logger.info(f"Created specialized model {model_id}")
            return model_id
            
        except Exception as e:
            logger.error(f"Error creating specialist: {e}")
            return None
            
    def is_relevant_game(self, game: CollectedGame, specialty: str) -> bool:
        """Check if a game is relevant for a specialty.
        
        Args:
            game: Game to check
            specialty: Specialty to check for
            
        Returns:
            Whether the game is relevant
        """
        if specialty not in self.specialties:
            return False
            
        config = self.specialties[specialty]
        
        try:
            # Check opening book
            if config.opening_book:
                # Verify game starts with book moves
                board = chess.Board()
                for i, move in enumerate(game.moves[:10]):  # Check first 10 moves
                    book_move = self.opening_manager.get_move(
                        board,
                        {config.opening_book: 1.0},
                        temperature=1.0,
                        randomization=0.0  # Strict matching
                    )
                    if not book_move or book_move != move:
                        return False
                    board.push(move)
                return True
                
            # Analyze final position
            board = chess.Board()
            for move in game.moves:
                board.push(move)
            features = self.position_analyzer.analyze_position(board)
            
            # Check game phase
            if specialty == "opening":
                return len(game.moves) <= config.style_params["max_move"]
                
            elif specialty == "middlegame":
                move_num = len(game.moves)
                return (
                    move_num >= config.style_params["min_move"] and
                    move_num <= config.style_params["max_move"]
                )
                
            elif specialty == "endgame":
                piece_count = len(board.piece_map())
                return (
                    piece_count >= config.style_params["min_pieces"] and
                    piece_count <= config.style_params["max_pieces"]
                )
                
            elif specialty == "tactical":
                return features.tactical_score >= 0.7
                
            elif specialty == "positional":
                return features.positional_score >= 0.7
                
            elif specialty == "queens_gambit":
                # Check opening moves
                if len(game.moves) < 4:
                    return False
                    
                # Check characteristic moves
                moves = [move.uci() for move in game.moves[:4]]
                return (
                    moves[0] == "d2d4" and  # 1.d4
                    moves[2] == "c2c4" and  # 2.c4
                    "d7d5" in moves[1:4]    # Black plays ...d5
                )
                
            return True
            
        except Exception as e:
            logger.error(f"Error checking game relevance: {e}")
            return False
            
    async def process_game(self, game: CollectedGame):
        """Process a game for specialization.
        
        Args:
            game: Game to process
        """
        try:
            # Find relevant specialties
            for specialty, config in self.specialties.items():
                if not self.is_relevant_game(game, specialty):
                    continue
                    
                # Cache game for specialty
                if specialty not in self.game_cache:
                    self.game_cache[specialty] = []
                self.game_cache[specialty].append(game)
                
                # Update stats for specialized models
                for model in self.models.values():
                    if model.specialty != specialty:
                        continue
                        
                    if model.model_id == game.metadata.model_version:
                        model.stats.games_played += 1
                        if game.result == "1-0":
                            model.stats.wins += 1
                        elif game.result == "0-1":
                            model.stats.losses += 1
                        else:
                            model.stats.draws += 1
                            
                        model.stats.last_updated = datetime.now().timestamp()
                        
            self._save_models()
            
        except Exception as e:
            logger.error(f"Error processing game: {e}")
            
    async def get_training_data(
        self,
        specialty: str,
        max_games: int = 1000
    ) -> List[CollectedGame]:
        """Get training data for a specialty.
        
        Args:
            specialty: Specialty to get data for
            max_games: Maximum number of games to return
            
        Returns:
            List of relevant games
        """
        if specialty not in self.game_cache:
            return []
            
        return self.game_cache[specialty][:max_games]
        
    def get_best_specialist(
        self,
        specialty: str,
        min_games: Optional[int] = None
    ) -> Optional[str]:
        """Get the best model for a specialty.
        
        Args:
            specialty: Specialty to get model for
            min_games: Minimum games required (defaults to specialty config)
            
        Returns:
            ID of best model, or None if none qualified
        """
        if specialty not in self.specialties:
            return None
            
        config = self.specialties[specialty]
        min_games = min_games or config.min_games
        
        best_model = None
        best_win_rate = 0.0
        
        for model in self.models.values():
            if not model.active or model.specialty != specialty:
                continue
                
            if model.stats.games_played < min_games:
                continue
                
            if model.stats.win_rate > best_win_rate:
                best_model = model
                best_win_rate = model.stats.win_rate
                
        return best_model.model_id if best_model else None
        
    def get_stats(self) -> Dict:
        """Get specialization statistics.
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            "total_models": len(self.models),
            "active_models": len([m for m in self.models.values() if m.active]),
            "specialties": {}
        }
        
        for specialty in self.specialties:
            specialty_models = [m for m in self.models.values() if m.specialty == specialty]
            stats["specialties"][specialty] = {
                "models": len(specialty_models),
                "total_games": sum(m.stats.games_played for m in specialty_models),
                "best_win_rate": max((m.stats.win_rate for m in specialty_models), default=0.0)
            }
            
        return stats 

    async def train_specialist(
        self,
        model_id: str,
        trainer: Trainer,
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        validation_split: float = 0.2
    ) -> Dict:
        """Train a specialized model.
        
        Args:
            model_id: ID of model to train
            trainer: Training manager
            epochs: Number of epochs to train
            batch_size: Training batch size
            learning_rate: Learning rate
            validation_split: Fraction of data to use for validation
            
        Returns:
            Training metrics
        """
        if model_id not in self.models:
            raise ValueError(f"Unknown model ID: {model_id}")
            
        model = self.models[model_id]
        specialty = model.specialty
        
        # Get training data
        games = await self.get_training_data(specialty)
        if not games:
            raise ValueError(f"No training data for specialty: {specialty}")
            
        # Create dataset
        dataset = self._create_dataset(games)
        
        # Split into train/val
        train_size = int(len(dataset) * (1 - validation_split))
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, len(dataset) - train_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size
        )
        
        # Load model
        model_path = f"models/specialized/{model_id}.pt"
        model = ChessGNN()  # TODO: Load architecture from config
        model.load_state_dict(torch.load(model_path))
        
        # Configure optimizer and loss
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate
        )
        criterion = PolicyValueLoss()
        
        # Training loop
        best_val_loss = float('inf')
        metrics_history = []
        
        for epoch in range(epochs):
            # Train
            model.train()
            train_metrics = await trainer.train_epoch(
                model,
                train_loader,
                optimizer,
                criterion
            )
            
            # Validate
            model.eval()
            val_metrics = await trainer.validate(
                model,
                val_loader,
                criterion
            )
            
            # Save if best
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                torch.save(model.state_dict(), model_path)
                
            metrics = {
                'epoch': epoch,
                'train': train_metrics,
                'val': val_metrics
            }
            metrics_history.append(metrics)
            
            logger.info(
                f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f}, "
                f"val_loss={val_metrics['loss']:.4f}"
            )
            
        return metrics_history
        
    async def evaluate_specialist(
        self,
        model_id: str,
        opponent_id: Optional[str] = None,
        num_games: int = 100
    ) -> Dict:
        """Evaluate a specialized model.
        
        Args:
            model_id: ID of model to evaluate
            opponent_id: ID of opponent model (if None, use base model)
            num_games: Number of evaluation games to play
            
        Returns:
            Evaluation metrics
        """
        if model_id not in self.models:
            raise ValueError(f"Unknown model ID: {model_id}")
            
        model = self.models[model_id]
        specialty = model.specialty
        config = self.specialties[specialty]
        
        # Load models
        model_path = f"models/specialized/{model_id}.pt"
        specialist = ChessGNN()
        specialist.load_state_dict(torch.load(model_path))
        
        if opponent_id:
            opponent_path = f"models/specialized/{opponent_id}.pt"
            opponent = ChessGNN()
            opponent.load_state_dict(torch.load(opponent_path))
        else:
            # Use base model as opponent
            opponent = ChessGNN()  # TODO: Load base model
            
        # Play evaluation games
        results = []
        for i in range(num_games):
            # Alternate colors
            if i % 2 == 0:
                white, black = specialist, opponent
                is_white = True
            else:
                white, black = opponent, specialist
                is_white = False
                
            # Play game
            game = await self.play_evaluation_game(white, black)
            
            # Record result
            if game.result == "1-0":
                results.append(1.0 if is_white else 0.0)
            elif game.result == "0-1":
                results.append(0.0 if is_white else 1.0)
            else:
                results.append(0.5)
                
            # Process game for training
            await self.process_game(game)
            
        # Calculate metrics
        win_rate = sum(r == 1.0 for r in results) / len(results)
        draw_rate = sum(r == 0.5 for r in results) / len(results)
        loss_rate = sum(r == 0.0 for r in results) / len(results)
        
        metrics = {
            'games_played': len(results),
            'win_rate': win_rate,
            'draw_rate': draw_rate,
            'loss_rate': loss_rate,
            'avg_score': sum(results) / len(results)
        }
        
        # Update model stats
        model.stats.games_played += len(results)
        model.stats.wins += int(win_rate * len(results))
        model.stats.draws += int(draw_rate * len(results))
        model.stats.losses += int(loss_rate * len(results))
        model.stats.last_updated = datetime.now().timestamp()
        
        # Check for promotion/demotion
        if win_rate >= config.eval_threshold:
            logger.info(f"Model {model_id} promoted: {win_rate:.3f} win rate")
            # TODO: Implement promotion logic
        else:
            logger.info(f"Model {model_id} below threshold: {win_rate:.3f} win rate")
            
        self._save_models()
        return metrics
        
    async def play_evaluation_game(
        self,
        white: ChessGNN,
        black: ChessGNN,
        max_moves: int = 200
    ) -> CollectedGame:
        """Play a single evaluation game between two models.
        
        Args:
            white: White player model
            black: Black player model
            max_moves: Maximum number of moves before draw
            
        Returns:
            Completed game
        """
        board = chess.Board()
        moves = []
        
        for move_num in range(max_moves):
            if board.is_game_over():
                break
                
            # Get current player
            current_model = white if board.turn else black
            
            # Get model move
            move = await self._get_model_move(current_model, board)
            moves.append(move)
            board.push(move)
            
        # Create game record
        game = CollectedGame(
            moves=moves,
            result=board.result(),
            metadata={
                'white_id': white.model_id,
                'black_id': black.model_id,
                'termination': board.outcome().termination if board.outcome() else None
            }
        )
        
        return game
        
    async def _get_model_move(
        self,
        model: ChessGNN,
        board: chess.Board
    ) -> chess.Move:
        """Get a move from a model for the current position.
        
        Args:
            model: Model to get move from
            board: Current board position
            
        Returns:
            Selected move
        """
        try:
            # Check opening book first
            if board.fullmove_number <= 15:  # Only use book in opening
                specialty = self._get_model_specialty(model)
                if specialty in self.specialties:
                    config = self.specialties[specialty]
                    if config.opening_book:
                        # Get book move with temperature based on position
                        features = self.position_analyzer.analyze_position(board)
                        temperature = self._calculate_temperature(
                            features,
                            board.fullmove_number
                        )
                        book_move = self.opening_manager.get_move(
                            board,
                            {config.opening_book: 1.0},
                            temperature=temperature
                        )
                        if book_move:
                            return book_move
                            
            # Fallback to model inference
            features = self.position_analyzer.analyze_position(board)
            policy, value = await self._get_model_predictions(model, board, features)
            
            # Apply temperature
            temperature = self._calculate_temperature(features, board.fullmove_number)
            policy = self._apply_temperature(policy, temperature)
            
            # Filter legal moves
            legal_moves = list(board.legal_moves)
            legal_policy = {move: policy[move] for move in legal_moves}
            
            # Normalize probabilities
            total_prob = sum(legal_policy.values())
            if total_prob > 0:
                legal_policy = {
                    move: prob / total_prob
                    for move, prob in legal_policy.items()
                }
            else:
                # Fallback to uniform distribution
                legal_policy = {
                    move: 1.0 / len(legal_moves)
                    for move in legal_moves
                }
                
            # Sample move
            moves = list(legal_policy.keys())
            probs = list(legal_policy.values())
            selected_move = np.random.choice(moves, p=probs)
            
            return selected_move
            
        except Exception as e:
            logger.error(f"Error in model inference: {e}")
            # Fallback to random move
            return np.random.choice(list(board.legal_moves))
            
    def _get_model_specialty(self, model: ChessGNN) -> Optional[str]:
        """Get the specialty of a model."""
        for model_id, spec_model in self.models.items():
            if model_id == model.model_id:
                return spec_model.specialty
        return None
        
    async def _get_model_predictions(
        self,
        model: ChessGNN,
        board: chess.Board,
        features: PositionFeatures
    ) -> Tuple[Dict[chess.Move, float], float]:
        """Get policy and value predictions from model.
        
        Args:
            model: Model to use
            board: Current position
            features: Position features
            
        Returns:
            Move probabilities and position value
        """
        # Convert inputs to tensors
        board_tensor = self.encoder.encode_board(board)
        features_tensor = self.encoder.encode_features(features)
        
        # Add batch dimension
        board_tensor = board_tensor.unsqueeze(0)
        features_tensor = features_tensor.unsqueeze(0)
        
        # Get model predictions
        with torch.no_grad():
            policy_logits, value = model(board_tensor, features_tensor)
            
        # Convert policy logits to move probabilities
        move_probs = self.encoder.get_move_probabilities(
            policy_logits[0],  # Remove batch dimension
            board
        )
        
        return move_probs, value.item()
        
    def _calculate_temperature(
        self,
        features: PositionFeatures,
        move_number: int
    ) -> float:
        """Calculate sampling temperature based on position.
        
        Args:
            features: Position features
            move_number: Current move number
            
        Returns:
            Temperature for move sampling
        """
        # Base temperature
        temperature = 1.0
        
        # Adjust for game phase
        if move_number <= 15:  # Opening
            temperature *= 0.8  # More deterministic in opening
        elif move_number >= 40:  # Endgame
            temperature *= 1.2  # More exploratory in endgame
            
        # Adjust for tactical positions
        tactical_score = features.tactical_score
        if tactical_score > 0.7:  # Highly tactical
            temperature *= 0.7  # More deterministic
            
        # Adjust for positional positions
        positional_score = features.positional_score
        if positional_score > 0.7:  # Highly positional
            temperature *= 1.2  # More exploratory
            
        # Clamp temperature
        temperature = max(0.5, min(2.0, temperature))
        
        return temperature
        
    def _apply_temperature(
        self,
        policy: Dict[chess.Move, float],
        temperature: float
    ) -> Dict[chess.Move, float]:
        """Apply temperature to move probabilities.
        
        Args:
            policy: Move probabilities
            temperature: Sampling temperature
            
        Returns:
            Adjusted probabilities
        """
        # Apply temperature scaling
        scaled_policy = {
            move: prob ** (1 / temperature)
            for move, prob in policy.items()
        }
        
        # Normalize
        total = sum(scaled_policy.values())
        if total > 0:
            scaled_policy = {
                move: prob / total
                for move, prob in scaled_policy.items()
            }
            
        return scaled_policy
        
    def _board_to_tensor(self, board: chess.Board) -> torch.Tensor:
        """Convert board to model input tensor."""
        return self.encoder.encode_board(board)
        
    def _features_to_tensor(self, features: PositionFeatures) -> torch.Tensor:
        """Convert position features to model input tensor."""
        return self.encoder.encode_features(features)
        
    def _move_to_index(self, move: chess.Move) -> int:
        """Convert move to policy index."""
        return self.encoder.encode_move(move)
        
    async def create_ensemble(
        self,
        specialties: List[str],
        weights: Optional[Dict[str, float]] = None
    ) -> str:
        """Create an ensemble of specialized models.
        
        Args:
            specialties: List of specialties to include
            weights: Optional weights for each specialty
            
        Returns:
            ID of created ensemble
        """
        if not all(s in self.specialties for s in specialties):
            raise ValueError("Unknown specialty in list")
            
        # Get best model for each specialty
        models = {}
        for specialty in specialties:
            model_id = self.get_best_specialist(specialty)
            if not model_id:
                raise ValueError(f"No qualified model for specialty: {specialty}")
            models[specialty] = model_id
            
        # Create ensemble configuration
        if weights is None:
            weights = {s: 1.0 / len(specialties) for s in specialties}
            
        ensemble_id = f"ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ensemble_config = {
            'id': ensemble_id,
            'models': models,
            'weights': weights
        }
        
        # Save ensemble configuration
        os.makedirs("models/ensembles", exist_ok=True)
        config_path = f"models/ensembles/{ensemble_id}.json"
        with open(config_path, 'w') as f:
            json.dump(ensemble_config, f, indent=2)
            
        return ensemble_id
        
    async def get_ensemble_prediction(
        self,
        ensemble_id: str,
        board: chess.Board
    ) -> Tuple[chess.Move, float]:
        """Get a weighted prediction from an ensemble.
        
        Args:
            ensemble_id: ID of ensemble to use
            board: Current board position
            
        Returns:
            Selected move and confidence
        """
        # Load ensemble config
        config_path = f"models/ensembles/{ensemble_id}.json"
        with open(config_path) as f:
            ensemble_config = json.load(f)
            
        predictions = []
        weights = []
        
        # Get predictions from each model
        for specialty, model_id in ensemble_config['models'].items():
            weight = ensemble_config['weights'][specialty]
            
            # Load model
            model_path = f"models/specialized/{model_id}.pt"
            model = ChessGNN()
            model.load_state_dict(torch.load(model_path))
            
            # Get prediction
            move = await self._get_model_move(model, board)
            predictions.append(move)
            weights.append(weight)
            
        # Weight predictions
        # TODO: Implement proper move selection/combination
        selected_idx = np.random.choice(len(predictions), p=weights)
        selected_move = predictions[selected_idx]
        confidence = weights[selected_idx]
        
        return selected_move, confidence
        
    def _create_dataset(self, games: List[CollectedGame]) -> torch.utils.data.Dataset:
        """Create a dataset from games.
        
        Args:
            games: List of games to convert
            
        Returns:
            Dataset for training
        """
        class ChessDataset(torch.utils.data.Dataset):
            def __init__(self, positions, policies, values):
                self.positions = positions
                self.policies = policies
                self.values = values
                
            def __len__(self):
                return len(self.positions)
                
            def __getitem__(self, idx):
                return (
                    self.positions[idx],
                    self.policies[idx],
                    self.values[idx]
                )
                
        positions = []
        policies = []
        values = []
        
        for game in games:
            board = chess.Board()
            result = self._parse_result(game.result)
            
            for move in game.moves:
                # Get position features
                features = self.position_analyzer.analyze_position(board)
                
                # Encode position
                board_tensor = self.encoder.encode_board(board)
                features_tensor = self.encoder.encode_features(features)
                position = (board_tensor, features_tensor)
                
                # Create policy target (one-hot)
                policy = torch.zeros(len(self.encoder.index_to_move))
                move_idx = self.encoder.encode_move(move)
                policy[move_idx] = 1.0
                
                # Use game result as value target
                value = result if board.turn == chess.WHITE else -result
                
                positions.append(position)
                policies.append(policy)
                values.append(value)
                
                board.push(move)
                
        return ChessDataset(positions, policies, values)
        
    def _parse_result(self, result: str) -> float:
        """Convert game result string to value target."""
        if result == "1-0":
            return 1.0
        elif result == "0-1":
            return -1.0
        else:
            return 0.0 