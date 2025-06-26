"""
Self-play module for generating training data through self-play games.
"""

import os
import time
import logging
import torch
import numpy as np
import random
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
from tqdm import tqdm
from pathlib import Path
import chess
from torch.utils.data import Dataset
import pickle
from datetime import datetime
import traceback

from rival_ai.models import ChessGNN
from rival_ai.mcts import MCTS, MCTSConfig
from rival_ai.chess import GameResult
from rival_ai.training.training_types import GameRecord
from rival_ai.utils.board_conversion import board_to_hetero_data
from .losses import PolicyValueLoss  # Use the simpler version from losses.py

logger = logging.getLogger(__name__)

@dataclass
class SelfPlayConfig:
    """Configuration for self-play."""
    num_games: int = 10
    num_simulations: int = 800
    max_moves: int = 200  # Reduced from 400 to force more decisive play
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 256
    num_workers: int = 4
    prefetch_factor: int = 2
    save_dir: Optional[str] = None  # Changed to Optional, will be set by trainer
    use_tqdm: bool = True
    
    # PAG Integration
    use_dense_pag: bool = True  # Enable dense PAG feature extraction
    pag_fallback_to_python: bool = True  # Fallback to Python PAG if Rust fails
    
    # MCTS parameters - MUCH more aggressive exploration
    c_puct: float = 4.0  # Increased from 3.0 to encourage more exploration
    dirichlet_alpha: float = 1.5  # Increased from 1.2 for more noise
    dirichlet_weight: float = 0.8  # Increased from 0.6 for more exploration
    temperature: float = 3.0  # Increased from 2.0 for much more variety
    
    # Temperature settings - MUCH more aggressive to encourage decisive play
    opening_temperature: float = 3.5  # Increased from 2.5
    midgame_temperature: float = 3.0  # Increased from 2.2
    endgame_temperature: float = 2.5  # Increased from 1.8
    
    # Repetition prevention - EXTREMELY stronger penalties
    min_pieces_for_repetition_penalty: int = 8  # Decreased from 20 to apply earlier
    repetition_penalty: float = 20.0  # Increased from 8.0 to extreme penalty
    
    # Randomness - MUCH more to break patterns
    random_move_probability: float = 0.25  # Increased from 0.15
    random_move_temperature: float = 5.0  # Increased from 4.0
    
    # Forward progress bonus - MUCH stronger to encourage active play
    forward_progress_bonus: float = 1.2  # Increased from 0.8
    
    # Draw prevention - MUCH stronger penalties
    draw_penalty_scale: float = 10.0  # Increased from 6.0
    early_draw_penalty: float = 15.0  # Increased from 10.0
    
    # Material imbalance bonus - MUCH stronger to encourage decisive play
    material_imbalance_bonus: float = 1.2  # Increased from 0.8
    
    # Position complexity bonus - MUCH stronger to encourage interesting positions
    position_complexity_bonus: float = 0.6  # Increased from 0.4
    
    # NEW: Aggressive play settings
    capture_bonus: float = 1.5  # Increased from 1.0
    check_bonus: float = 0.8  # Increased from 0.5
    attack_bonus: float = 0.5  # Increased from 0.3
    development_bonus: float = 0.6  # Increased from 0.4
    
    # NEW: Game phase detection
    opening_moves: int = 15  # First 15 moves are opening
    midgame_moves: int = 50  # Moves 16-50 are middlegame
    # After move 50 is endgame
    
    # NEW: Dynamic move limit based on game phase
    max_opening_moves: int = 25  # Decreased from 30 for shorter opening
    max_middlegame_moves: int = 80  # Decreased from 100 for shorter middlegame
    max_endgame_moves: int = 120  # Decreased from 150 for shorter endgame
    
    # NEW: Forced decisive play settings
    force_capture_after_moves: int = 15  # Decreased from 20 to force earlier
    force_attack_after_moves: int = 25  # Decreased from 30 to force earlier
    force_win_attempt_after_moves: int = 40  # Decreased from 50 to force earlier

class SelfPlayGenerator:
    """Generator for self-play games."""
    
    def __init__(self, model: ChessGNN, config: SelfPlayConfig):
        """Initialize generator.
        
        Args:
            model: Neural network model
            config: Self-play configuration
        """
        self.model = model
        self.config = config
        self.mcts = MCTS(model, MCTSConfig(
            num_simulations=config.num_simulations,
            c_puct=config.c_puct,
            dirichlet_alpha=config.dirichlet_alpha,
            dirichlet_weight=config.dirichlet_weight,
            temperature=config.temperature,
            device=config.device
        ))
        
        # Create save directory if it doesn't exist
        os.makedirs(config.save_dir, exist_ok=True)
        
        # Initialize metrics with limited history
        self.metrics = defaultdict(lambda: deque(maxlen=1000))  # Limit metric history
        self.timing_metrics = defaultdict(float)
        self.position_counts = defaultdict(int)
        self.last_positions = defaultdict(lambda: deque(maxlen=15))  # Limit position history
        
        # Add memory cleanup interval
        self.games_since_cleanup = 0
        self.cleanup_interval = 50  # Clean up every 50 games
    
    def _cleanup_memory(self):
        """Clean up memory periodically."""
        if self.games_since_cleanup >= self.cleanup_interval:
            # Clear MCTS search tree
            self.mcts.clear_tree()
            
            # Clear metrics older than 1000 games
            self.metrics.clear()
            self.timing_metrics.clear()
            self.position_counts.clear()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear CUDA cache if using GPU
            if self.config.device == "cuda":
                torch.cuda.empty_cache()
            
            self.games_since_cleanup = 0
        else:
            self.games_since_cleanup += 1
    
    def _get_position_key(self, board: chess.Board) -> str:
        """Get a unique key for the current position."""
        return board.fen()

    def _apply_random_component(self, value: float) -> float:
        """Add small random component to break symmetry."""
        return value + np.random.normal(0, self.config.random_component_std)

    def _compute_draw_penalty(self, move_count: int, num_pieces: int) -> float:
        """Compute penalty for draws based on game phase."""
        if move_count < 20:  # Opening
            return 0.3
        elif move_count < 40:  # Middlegame
            return 0.2
        elif num_pieces <= 6:  # Endgame with few pieces
            return 0.0  # No penalty in endgame
        else:
            return 0.1  # Small penalty in other endgames

    def _apply_repetition_penalty(self, board: chess.Board, value: float, game_idx: int) -> float:
        """Apply penalty for repeated positions with dynamic scaling."""
        position_key = self._get_position_key(board)
        num_pieces = len(board.piece_map())
        
        # Apply penalty even with fewer pieces - repetition is bad at any stage
        if num_pieces >= self.config.min_pieces_for_repetition_penalty:
            # Check recent positions in this game
            recent_positions = self.last_positions[game_idx]
            if position_key in recent_positions:
                # Count how many times this position has occurred recently
                recent_count = recent_positions.count(position_key)
                # Scale penalty based on game phase and piece count - MUCH stronger scaling
                phase_scale = 2.0  # Increased from 1.0
                if board.fullmove_number() < 15:  # Very early repetition
                    phase_scale = 3.0  # Increased from 1.5
                elif board.fullmove_number() < 30:  # Early repetition
                    phase_scale = 2.5  # Increased from 1.2
                piece_scale = min(1.5, num_pieces / 12.0)  # Increased scaling
                penalty = self.config.repetition_penalty * recent_count * phase_scale * piece_scale
                value -= penalty
                
                # Additional penalty for multiple repetitions
                if recent_count >= 2:
                    value -= self.config.repetition_penalty * 2.0  # Extra penalty for multiple reps
            
            # Update recent positions (keep last 15 positions instead of 10)
            recent_positions.append(position_key)
            if len(recent_positions) > 15:
                recent_positions.pop(0)
        
        return value

    def _apply_forward_progress_bonus(self, board: chess.Board, value: float) -> float:
        """Apply enhanced bonus for pawn advancement and piece development."""
        bonus = 0.0
        move_count = board.fullmove_number
        
        # Pawn advancement bonus (stronger in opening)
        if move_count <= 20:  # Extended from 10 to 20 moves
            for rank in range(8):
                for file in range(8):
                    square = chess.square(file, rank)
                    piece = board.piece_at(square)
                    if piece and piece.piece_type == chess.PAWN:
                        # Calculate progress towards promotion
                        progress = rank / 7.0 if piece.color == chess.WHITE else (7 - rank) / 7.0
                        # Scale bonus based on game phase
                        phase_scale = 1.5 if move_count <= 10 else 1.0
                        bonus += progress * self.config.forward_progress_bonus * phase_scale
        
        # Piece development bonus (only in opening)
        if move_count <= 15:
            for piece_type in [chess.KNIGHT, chess.BISHOP]:
                for square in board.pieces(piece_type, board.turn):
                    # Bonus for pieces that have moved from their starting squares
                    if piece_type == chess.KNIGHT and square in [chess.B1, chess.G1, chess.B8, chess.G8]:
                        bonus += 0.05
                    elif piece_type == chess.BISHOP and square in [chess.C1, chess.F1, chess.C8, chess.F8]:
                        bonus += 0.05
        
        value += bonus
        return value

    def generate_games(self, num_games: Optional[int] = None, epoch: Optional[int] = None, save_games: bool = True) -> List[GameRecord]:
        """Generate self-play games.
        
        Args:
            num_games: Number of games to generate (defaults to config.num_games)
            epoch: Current training epoch
            save_games: Whether to save games to disk
            
        Returns:
            List of game records
        """
        num_games = num_games or self.config.num_games
        games = []
        
        # Use tqdm if configured
        game_iter = tqdm(range(num_games)) if self.config.use_tqdm else range(num_games)
        
        for i in game_iter:
            try:
                # Generate game
                game = self.play_game()
                games.append(game)
                
                # Save games in batches to reduce memory usage
                if save_games and len(games) >= 10:
                    self._save_games(games, epoch)
                    games = []  # Clear saved games from memory
                
                # Cleanup memory periodically
                self._cleanup_memory()
                
            except Exception as e:
                logger.error(f"Error generating game {i}: {e}")
                logger.error(traceback.format_exc())
                continue
        
        # Save any remaining games
        if save_games and games:
            self._save_games(games, epoch)
        
        return games
    
    def create_dataloader(self, games: List[GameRecord]):
        """Create a dataloader from generated games.
        
        Args:
            games: List of GameRecord objects
            
        Returns:
            DataLoader for training
        """
        # Import here to avoid circular dependency
        from rival_ai.data import PAGDataset, PAGDataLoader
        
        dataset = PAGDataset.from_game_records(games)
        return PAGDataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            prefetch_factor=self.config.prefetch_factor,
            persistent_workers=True
        )

class SelfPlay:
    """Manages self-play games for training data generation."""
    
    def __init__(
        self,
        model: ChessGNN,
        config: SelfPlayConfig,
        experiment_dir: Optional[Path] = None,  # Added experiment_dir parameter
    ):
        """Initialize self-play manager.
        
        Args:
            model: The GNN model to use for self-play
            config: Self-play configuration
            experiment_dir: Optional path to experiment directory
        """
        self.model = model.to(config.device)
        self.config = config
        self.device = config.device
        
        # Initialize PAG engine if enabled
        self.pag_engine = None
        if config.use_dense_pag:
            try:
                # Import the PAG engine from Rust bindings
                import rival_ai_engine as engine
                self.pag_engine = engine.PyPAGEngine()
                logger.info("✅ Dense PAG engine initialized for self-play")
            except Exception as e:
                if config.pag_fallback_to_python:
                    logger.warning(f"⚠️ Failed to initialize Rust PAG engine ({e}), falling back to Python PAG")
                    self.pag_engine = None
                else:
                    logger.error(f"❌ Failed to initialize PAG engine: {e}")
                    raise
        
        # Create MCTS config
        self.mcts_config = MCTSConfig(
            num_simulations=config.num_simulations,
            c_puct=config.c_puct,
            temperature=config.temperature,
            dirichlet_alpha=config.dirichlet_alpha,
            dirichlet_weight=config.dirichlet_weight,
            device=config.device
        )
        
        # Initialize MCTS
        self.mcts = MCTS(
            model=model,
            config=self.mcts_config
        )
        self.mcts.model = self.model  # Set the model after initialization
        
        # Set up save directory
        if experiment_dir is not None:
            self.save_dir = experiment_dir / 'self_play_data'
        else:
            self.save_dir = Path(config.save_dir) if config.save_dir else Path('self_play_data')
        
        # Create save directory if it doesn't exist
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics
        self.metrics = defaultdict(list)
        
        # Initialize timing metrics
        self.timing_metrics = {
            'total_time': 0.0,
            'mcts_time': 0.0,
            'model_time': 0.0,
            'move_selection_time': 0.0
        }
        
        # Initialize game counter
        self.game_counter = 0
        
        # Initialize last positions for repetition detection
        self.last_positions = defaultdict(list)
    
    def _log_timing(self, component: str, start_time: float):
        """Log timing for a component.
        
        Args:
            component: Name of the component being timed
            start_time: Start time of the operation
        """
        if not self.config.log_timing:
            return
            
        elapsed = time.time() - start_time
        self.timing_metrics[f'{component}_time'] += elapsed
        
        if component == 'move_execution':
            self.timing_metrics['num_moves'] += 1
    
    def _log_game_metrics(self):
        """Log current game metrics."""
        if not self.config.log_timing:
            return
            
        total_time = self.timing_metrics['total_time']
        if total_time == 0:
            return
            
        num_moves = self.timing_metrics['num_moves']
        num_games = self.timing_metrics['num_games']
        
        metrics = {
            'Total Game Time': f"{total_time:.2f}s",
            'Move Selection': f"{self.timing_metrics['move_selection_time']:.2f}s ({self.timing_metrics['move_selection_time']/total_time*100:.1f}%)",
            'Move Execution': f"{self.timing_metrics['move_execution_time']:.2f}s ({self.timing_metrics['move_execution_time']/total_time*100:.1f}%)",
            'State Recording': f"{self.timing_metrics['state_recording_time']:.2f}s ({self.timing_metrics['state_recording_time']/total_time*100:.1f}%)",
            'Moves per Game': f"{num_moves/max(1, num_games):.1f}",
            'Time per Move': f"{total_time/max(1, num_moves):.2f}s",
            'Time per Game': f"{total_time/max(1, num_games):.2f}s",
        }
        
        logger.info("Self-Play Performance Metrics:")
        for name, value in metrics.items():
            logger.info(f"  {name}: {value}")
    
    def _should_make_random_move(self, board: chess.Board, move_count: int) -> bool:
        """Determine if we should make a random move to break patterns."""
        # Much higher chance of random move in opening/middlegame
        if move_count < 10:  # Very early game
            return random.random() < self.config.random_move_probability * 3  # Triple chance
        elif move_count < 20:  # Early game
            return random.random() < self.config.random_move_probability * 2  # Double chance
        elif move_count < 40:  # Middlegame
            return random.random() < self.config.random_move_probability * 1.5  # 50% more chance
        elif move_count < 60:  # Late middlegame
            return random.random() < self.config.random_move_probability
        else:  # Endgame
            return random.random() < self.config.random_move_probability * 0.5  # Reduced in endgame
        
        # Additional check: if we're in a position that could lead to repetition, increase random chance
        legal_moves = list(board.legal_moves)
        repetition_moves = [move for move in legal_moves if self._would_cause_repetition(board, move)]
        if len(repetition_moves) > len(legal_moves) * 0.3:  # If more than 30% of moves would repeat
            return random.random() < 0.8  # 80% chance of random move to break repetition

    def _get_random_move(self, board: chess.Board) -> chess.Move:
        """Get a random move with higher temperature."""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
            
        # Get policy from model with higher temperature
        data = board_to_hetero_data(board)
        data = data.to(self.config.device)
        with torch.no_grad():
            policy, _ = self.model(data)
            policy = policy.squeeze(0)
            
        # Apply temperature
        policy = policy ** (1.0 / self.config.random_move_temperature)
        policy = policy / policy.sum()
        
        # Sample move
        move_idx = torch.multinomial(policy, 1).item()
        return self.mcts._index_to_move(move_idx, board)

    def _apply_material_evaluation(self, board: chess.Board, current_value: float) -> float:
        """
        Apply material-based evaluation to encourage capturing pieces and discourage losing material.
        
        Args:
            board: Current chess board position
            current_value: Current position evaluation value
            
        Returns:
            Modified value with material considerations
        """
        # Piece values (standard chess piece values)
        piece_values = {
            chess.PAWN: 1.0,
            chess.KNIGHT: 3.0,
            chess.BISHOP: 3.0,
            chess.ROOK: 5.0,
            chess.QUEEN: 9.0,
            chess.KING: 0.0  # King has no material value
        }
        
        # Calculate material balance
        white_material = 0.0
        black_material = 0.0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                value = piece_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    white_material += value
                else:
                    black_material += value
        
        # Material balance from white's perspective (positive = white advantage)
        material_balance = white_material - black_material
        
        # Apply material bonus/penalty based on whose turn it is
        if board.turn == chess.WHITE:
            # White's turn - reward positive material balance, penalize negative
            if material_balance > 0:
                # White has material advantage - bonus to encourage maintaining it
                material_bonus = material_balance * 1.0  # Increased from 0.5 to 1.0
            else:
                # White is behind in material - HEAVY penalty to discourage further losses
                material_bonus = material_balance * 3.0  # Increased from 1.0 to 3.0 - MUCH stronger penalty!
        else:
            # Black's turn - reward negative material balance (black advantage), penalize positive
            if material_balance < 0:
                # Black has material advantage - bonus
                material_bonus = abs(material_balance) * 1.0  # Increased from 0.5 to 1.0
            else:
                # Black is behind in material - HEAVY penalty
                material_bonus = -material_balance * 3.0  # Increased from 1.0 to 3.0 - MUCH stronger penalty!
        
        # Scale material bonus based on game phase
        move_count = board.fullmove_number
        if move_count < 10:
            # Opening - smaller material influence to allow for piece development
            material_bonus *= 0.7  # Increased from 0.5
        elif move_count < 30:
            # Middlegame - full material influence
            material_bonus *= 1.0
        else:
            # Endgame - stronger material influence
            material_bonus *= 2.0  # Increased from 1.5
        
        # Add material bonus to current value
        modified_value = current_value + material_bonus
        
        # Log material evaluation for debugging
        if abs(material_bonus) > 0.1:  # Lowered threshold to see more logging
            logger.info(f"Material evaluation: balance={material_balance:.2f}, "
                        f"bonus={material_bonus:.3f}, turn={'white' if board.turn else 'black'}")
        
        return modified_value

    def _apply_pag_tactical_evaluation(self, board: chess.Board, current_value: float) -> float:
        """
        Apply PAG-based tactical evaluation using ultra-dense features.
        This leverages our 308-dimensional piece features and tactical analysis.
        """
        try:
            # Extract PAG features for current position
            pag_features = self._extract_pag_features(board)
            if pag_features is None:
                return current_value  # Fallback if PAG extraction fails
            
            pag_penalty = 0.0
            pag_bonus = 0.0
            
            # Get PAG data from Rust engine
            import rival_ai_engine
            engine = rival_ai_engine.PyPAGEngine()
            pag_data = engine.fen_to_dense_pag(board.fen())
            
            # Extract piece vulnerability from PAG features
            if 'node_features' in pag_data and 'node_types' in pag_data:
                node_features = pag_data['node_features']
                node_types = pag_data['node_types']
                
                for i, (features, node_type) in enumerate(zip(node_features, node_types)):
                    if node_type == 'piece' and len(features) >= 308:
                        # Extract tactical features (first 76 dimensions are tactical)
                        tactical_features = features[:76]
                        
                        # Vulnerability status features (dimensions 60-75 in tactical features)
                        vulnerability_features = tactical_features[60:76] if len(tactical_features) >= 76 else tactical_features[-16:]
                        
                        # Check for hanging pieces (high vulnerability, low defense)
                        if len(vulnerability_features) >= 16:
                            hanging_score = vulnerability_features[0]  # is_hanging indicator
                            pinned_score = vulnerability_features[1]   # is_pinned indicator
                            overloaded_score = vulnerability_features[2]  # is_overloaded indicator
                            attack_defense_ratio = vulnerability_features[3]  # attack/defense ratio
                            
                            # Heavy penalty for hanging pieces
                            if hanging_score > 0.7:  # Piece is hanging
                                piece_value = self._estimate_piece_value_from_features(features)
                                pag_penalty += piece_value * 10.0  # MASSIVE penalty for hanging pieces
                                logger.warning(f"🚨 HANGING PIECE DETECTED! Penalty: {piece_value * 10.0:.2f}")
                            
                            # Penalty for pinned pieces
                            if pinned_score > 0.7:
                                piece_value = self._estimate_piece_value_from_features(features)
                                pag_penalty += piece_value * 2.0  # Strong penalty for pinned pieces
                            
                            # Penalty for overloaded pieces
                            if overloaded_score > 0.7:
                                piece_value = self._estimate_piece_value_from_features(features)
                                pag_penalty += piece_value * 1.5  # Penalty for overloaded pieces
                            
                            # Penalty for poor attack/defense ratio
                            if attack_defense_ratio > 2.0:  # Much more attacked than defended
                                piece_value = self._estimate_piece_value_from_features(features)
                                pag_penalty += piece_value * (attack_defense_ratio - 1.0)
                        
                        # Extract positional features (dimensions 76-156 are positional)
                        if len(features) >= 156:
                            positional_features = features[76:156]
                            
                            # Activity metrics (dimensions 52-65 in positional features)
                            if len(positional_features) >= 66:
                                activity_features = positional_features[52:66]
                                activity_score = sum(activity_features) / len(activity_features)
                                
                                # Bonus for active pieces
                                if activity_score > 0.6:
                                    pag_bonus += activity_score * 0.5
                                # Penalty for passive pieces
                                elif activity_score < 0.3:
                                    pag_penalty += (0.3 - activity_score) * 0.3
                        
                        # Extract strategic features (dimensions 156-216 are strategic)
                        if len(features) >= 216:
                            strategic_features = features[156:216]
                            
                            # King safety contribution (dimensions 12-23 in strategic features)
                            if len(strategic_features) >= 24:
                                king_safety_features = strategic_features[12:24]
                                king_safety_score = sum(king_safety_features) / len(king_safety_features)
                                
                                # Bonus for pieces contributing to king safety
                                if king_safety_score > 0.5:
                                    pag_bonus += king_safety_score * 0.8
                                # Penalty for pieces that weaken king safety
                                elif king_safety_score < 0.2:
                                    pag_penalty += (0.2 - king_safety_score) * 1.0
            
            # Apply PAG-based adjustments
            net_pag_adjustment = pag_bonus - pag_penalty
            modified_value = current_value + net_pag_adjustment
            
            # Log significant PAG adjustments
            if abs(net_pag_adjustment) > 0.5:
                logger.info(f"PAG evaluation: bonus={pag_bonus:.2f}, penalty={pag_penalty:.2f}, "
                           f"net_adjustment={net_pag_adjustment:.2f}")
            
            return modified_value
            
        except Exception as e:
            logger.warning(f"PAG tactical evaluation failed: {e}")
            return current_value  # Fallback to original value
    
    def _estimate_piece_value_from_features(self, features) -> float:
        """
        Estimate piece value from PAG features.
        This is a heuristic based on the feature patterns.
        """
        if len(features) < 308:
            return 1.0  # Default pawn value
        
        # The piece type information should be encoded in the features
        # For now, use a heuristic based on feature magnitudes
        feature_magnitude = sum(abs(f) for f in features[:20]) / 20  # Average of first 20 features
        
        if feature_magnitude > 0.8:
            return 9.0  # Likely queen
        elif feature_magnitude > 0.6:
            return 5.0  # Likely rook
        elif feature_magnitude > 0.4:
            return 3.0  # Likely knight/bishop
        else:
            return 1.0  # Likely pawn

    def _apply_capture_encouragement(self, board: chess.Board, current_value: float) -> float:
        """
        Specifically encourage capturing moves when they are advantageous.
        
        Args:
            board: Current chess board position
            current_value: Current position evaluation value
            
        Returns:
            Modified value with capture encouragement
        """
        # Get all legal moves
        legal_moves = list(board.legal_moves)
        
        # Check if any legal moves are captures
        capture_moves = [move for move in legal_moves if board.is_capture(move)]
        
        if not capture_moves:
            return current_value  # No captures available
        
        # Evaluate capture opportunities
        best_capture_value = 0.0
        capture_bonus = 0.0
        
        for move in capture_moves:
            # Make the move temporarily
            board.push(move)
            
            # Get the captured piece value
            captured_square = move.to_square
            captured_piece = board.piece_at(captured_square)
            
            if captured_piece is None:
                # This shouldn't happen for capture moves, but handle it
                board.pop()
                continue
            
            # Get the capturing piece value
            from_square = move.from_square
            capturing_piece = board.piece_at(from_square)
            
            if capturing_piece is None:
                # This shouldn't happen, but handle it
                board.pop()
                continue
            
            # Calculate material exchange value
            piece_values = {
                chess.PAWN: 1.0,
                chess.KNIGHT: 3.0,
                chess.BISHOP: 3.0,
                chess.ROOK: 5.0,
                chess.QUEEN: 9.0,
                chess.KING: 0.0
            }
            
            captured_value = piece_values[captured_piece.piece_type]
            capturing_value = piece_values[capturing_piece.piece_type]
            
            # Calculate exchange value (positive = good capture)
            exchange_value = captured_value - capturing_value
            
            # Add small bonus for equal captures to encourage activity
            if exchange_value == 0:
                exchange_value = 0.2  # Increased from 0.1
            
            # Update best capture value
            if exchange_value > best_capture_value:
                best_capture_value = exchange_value
            
            # Undo the move
            board.pop()
        
        # Apply capture bonus based on best available capture
        if best_capture_value > 0:
            # Good capture available - VERY strongly encourage it
            capture_bonus = best_capture_value * 2.0  # Increased from 1.0 to 2.0 - much stronger!
        elif best_capture_value < 0:
            # Only bad captures available - HEAVY penalty to discourage them
            capture_bonus = best_capture_value * 1.0  # Increased from 0.3 to 1.0 - stronger penalty!
        
        # Scale capture bonus based on game phase
        move_count = board.fullmove_number
        if move_count < 10:
            # Opening - be more cautious about captures
            capture_bonus *= 0.8  # Increased from 0.7
        elif move_count < 30:
            # Middlegame - normal capture evaluation
            capture_bonus *= 1.0
        else:
            # Endgame - more aggressive about captures
            capture_bonus *= 1.5  # Increased from 1.2
        
        modified_value = current_value + capture_bonus
        
        # Log capture evaluation for debugging
        if abs(capture_bonus) > 0.1:  # Lowered threshold to see more logging
            logger.info(f"Capture evaluation: best_capture={best_capture_value:.2f}, "
                        f"bonus={capture_bonus:.3f}")
        
        return modified_value

    def _apply_obvious_capture_bonus(self, board: chess.Board, current_value: float) -> float:
        """
        Apply very strong bonuses for obvious captures (like pawn capturing queen).
        
        Args:
            board: Current chess board position
            current_value: Current position evaluation value
            
        Returns:
            Modified value with obvious capture bonuses
        """
        # Get all legal moves
        legal_moves = list(board.legal_moves)
        
        # Check if any legal moves are captures
        capture_moves = [move for move in legal_moves if board.is_capture(move)]
        
        if not capture_moves:
            return current_value  # No captures available
        
        # Look for obvious captures (capturing higher value pieces)
        obvious_capture_bonus = 0.0
        piece_values = {
            chess.PAWN: 1.0,
            chess.KNIGHT: 3.0,
            chess.BISHOP: 3.0,
            chess.ROOK: 5.0,
            chess.QUEEN: 9.0,
            chess.KING: 0.0
        }
        
        for move in capture_moves:
            # Make the move temporarily
            board.push(move)
            
            # Get the captured piece value
            captured_square = move.to_square
            captured_piece = board.piece_at(captured_square)
            
            if captured_piece is None:
                board.pop()
                continue
            
            # Get the capturing piece value
            from_square = move.from_square
            capturing_piece = board.piece_at(from_square)
            
            if capturing_piece is None:
                board.pop()
                continue
            
            captured_value = piece_values[captured_piece.piece_type]
            capturing_value = piece_values[capturing_piece.piece_type]
            
            # Calculate exchange value
            exchange_value = captured_value - capturing_value
            
            # Apply very strong bonus for obvious captures
            if exchange_value >= 3.0:  # Capturing piece worth 3+ more than capturing piece
                # This is a very good capture (e.g., pawn capturing queen = +8)
                obvious_bonus = exchange_value * 5.0  # Increased from 2.0 to 5.0 - EXTREME bonus!
                if obvious_bonus > obvious_capture_bonus:
                    obvious_capture_bonus = obvious_bonus
                    logger.info(f"OBVIOUS CAPTURE FOUND: {capturing_piece.piece_type} capturing {captured_piece.piece_type} "
                               f"(value: {captured_value} - {capturing_value} = +{exchange_value})")
            elif exchange_value >= 1.0:  # Also bonus for smaller but still good captures
                # Good capture (e.g., pawn taking knight = +2)
                obvious_bonus = exchange_value * 3.0  # New bonus for smaller but good captures
                if obvious_bonus > obvious_capture_bonus:
                    obvious_capture_bonus = obvious_bonus
                    logger.info(f"GOOD CAPTURE FOUND: {capturing_piece.piece_type} capturing {captured_piece.piece_type} "
                               f"(value: {captured_value} - {capturing_value} = +{exchange_value})")
            
            # Undo the move
            board.pop()
        
        # Apply the obvious capture bonus
        modified_value = current_value + obvious_capture_bonus
        
        if obvious_capture_bonus > 0:
            logger.info(f"Applied obvious capture bonus: {obvious_capture_bonus:.2f}")
        
        return modified_value

    def _apply_aggressive_play_bonus(self, board: chess.Board, policy: torch.Tensor, move_count: int) -> torch.Tensor:
        """Apply bonuses for aggressive play to encourage decisive games."""
        policy = policy.clone()
        
        # Get legal moves
        legal_moves = list(board.legal_moves)
        
        for move in legal_moves:
            move_idx = self.mcts._move_to_move_idx(move)
            if move_idx is not None and 0 <= move_idx < 5312:
                bonus = 0.0
                
                # Capture bonus
                if board.is_capture(move):
                    bonus += self.config.capture_bonus
                    
                    # Extra bonus for capturing with less valuable pieces
                    captured_piece = board.piece_at(move.to_square)
                    moving_piece = board.piece_at(move.from_square)
                    if captured_piece and moving_piece:
                        piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, 
                                      chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
                        captured_value = piece_values.get(captured_piece.piece_type, 0)
                        moving_value = piece_values.get(moving_piece.piece_type, 0)
                        if moving_value < captured_value:
                            bonus += self.config.capture_bonus * 0.5  # Extra bonus for good captures
                
                # Check bonus
                if board.gives_check(move):
                    bonus += self.config.check_bonus
                
                # Attack bonus (moves that attack enemy pieces)
                if self._is_attacking_move(board, move):
                    bonus += self.config.attack_bonus
                
                # Development bonus (in opening)
                if move_count <= self.config.opening_moves:
                    if self._is_development_move(board, move):
                        bonus += self.config.development_bonus
                
                # Apply bonus
                policy[move_idx] += bonus
        
        return policy
    
    def _is_attacking_move(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if a move attacks enemy pieces."""
        # Make the move temporarily
        board.push(move)
        
        # Check if any enemy pieces are under attack
        is_attacking = False
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color != board.turn:
                # Check if this piece is under attack
                if board.is_attacked_by(board.turn, square):
                    is_attacking = True
                    break
        
        # Undo the move
        board.pop()
        return is_attacking
    
    def _is_development_move(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if a move develops a piece in the opening."""
        piece = board.piece_at(move.from_square)
        if not piece:
            return False
        
        # Consider knight and bishop moves as development
        if piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
            # Check if moving from starting square
            if piece.color == chess.WHITE:
                starting_squares = {
                    chess.KNIGHT: [chess.B1, chess.G1],
                    chess.BISHOP: [chess.C1, chess.F1]
                }
            else:
                starting_squares = {
                    chess.KNIGHT: [chess.B8, chess.G8],
                    chess.BISHOP: [chess.C8, chess.F8]
                }
            
            if move.from_square in starting_squares.get(piece.piece_type, []):
                return True
        
        return False
    
    def _force_decisive_play(self, board: chess.Board, policy: torch.Tensor, move_count: int) -> torch.Tensor:
        """Force more decisive play after certain move counts."""
        policy = policy.clone()
        
        # ALWAYS penalize moves that would cause repetition (not just after certain moves)
        legal_moves = list(board.legal_moves)
        for move in legal_moves:
            move_idx = self.mcts._move_to_move_idx(move)
            if move_idx is not None and 0 <= move_idx < 5312:
                # Heavily penalize moves that would cause repetition
                if self._would_cause_repetition(board, move):
                    policy[move_idx] *= 0.1  # Reduce to 10% of original probability
                
                # Boost moves that don't repeat positions
                else:
                    policy[move_idx] *= 1.2  # 20% bonus for non-repetition moves
        
        # Force captures after certain moves (earlier now)
        if move_count >= self.config.force_capture_after_moves:
            capture_moves = [move for move in legal_moves if board.is_capture(move)]
            
            if capture_moves:
                # Boost capture moves significantly
                for move in capture_moves:
                    move_idx = self.mcts._move_to_move_idx(move)
                    if move_idx is not None and 0 <= move_idx < 5312:
                        policy[move_idx] *= 4.0  # Quadruple the probability (increased from 3.0)
        
        # Force attacks after certain moves (earlier now)
        if move_count >= self.config.force_attack_after_moves:
            attacking_moves = [move for move in legal_moves if self._is_attacking_move(board, move)]
            
            if attacking_moves:
                # Boost attacking moves
                for move in attacking_moves:
                    move_idx = self.mcts._move_to_move_idx(move)
                    if move_idx is not None and 0 <= move_idx < 5312:
                        policy[move_idx] *= 3.0  # Triple the probability (increased from 2.0)
        
        # Force win attempts after certain moves (earlier now)
        if move_count >= self.config.force_win_attempt_after_moves:
            # Boost all moves that don't lead to immediate draws
            for move in legal_moves:
                move_idx = self.mcts._move_to_move_idx(move)
                if move_idx is not None and 0 <= move_idx < 5312:
                    # Boost moves that don't repeat positions even more
                    if not self._would_cause_repetition(board, move):
                        policy[move_idx] *= 2.0  # Double the probability (increased from 1.5)
        
        # Additional early game aggression
        if move_count < 10:  # Very early game
            # Boost all non-repetition moves in opening
            for move in legal_moves:
                move_idx = self.mcts._move_to_move_idx(move)
                if move_idx is not None and 0 <= move_idx < 5312:
                    if not self._would_cause_repetition(board, move):
                        policy[move_idx] *= 1.5  # 50% bonus for early non-repetition moves
        
        return policy
    
    def _apply_dynamic_move_limit(self, board: chess.Board, move_count: int) -> bool:
        """Apply dynamic move limits based on game phase."""
        num_pieces = len(board.piece_map())
        
        # Opening phase
        if move_count <= self.config.opening_moves:
            if move_count >= self.config.max_opening_moves:
                return True  # Force game end
        
        # Middlegame phase
        elif move_count <= self.config.midgame_moves:
            if move_count >= self.config.max_middlegame_moves:
                return True  # Force game end
        
        # Endgame phase
        else:
            if move_count >= self.config.max_endgame_moves:
                return True  # Force game end
            
            # In endgame, force decisive play if few pieces remain
            if num_pieces <= 8:
                if move_count >= self.config.max_endgame_moves // 2:
                    return True  # Force game end earlier in simple endgames
        
        return False

    def play_game(self) -> GameRecord:
        """Play a single self-play game.
        
        Returns:
            GameRecord containing the game data
        """
        # Initialize board
        board = chess.Board()
        game_record = GameRecord()
        
        # Initialize timing
        start_time = time.time()
        move_count = 0
        total_mcts_time = 0.0
        
        # Game loop
        while not board.is_game_over() and move_count < self.config.max_moves:
            try:
                # Record current state
                game_record.add_state(board.copy())
                
                # Extract PAG features for current position
                pag_features = None
                if self.config.use_dense_pag:
                    pag_features = self._extract_pag_features(board)
                
                # MCTS search with timing
                mcts_start = time.time()
                
                # 🚨 TEMPORARY FIX: Use direct model evaluation instead of MCTS to avoid infinite loop
                logger.info(f"🎯 TEMPORARY: Using direct model evaluation instead of MCTS (move {move_count})")
                data = board_to_hetero_data(board)
                data = data.to(self.config.device)
                with torch.no_grad():
                    policy, value = self.model(data)
                    policy = policy.squeeze(0)
                    value = value.squeeze()
                
                mcts_time = time.time() - mcts_start
                total_mcts_time += mcts_time
                
                # Apply enhanced bonuses and penalties
                value = self._apply_material_evaluation(board, value)
                value = self._apply_pag_tactical_evaluation(board, value)
                value = self._apply_repetition_penalty(board, value, self.game_counter)
                value = self._apply_forward_progress_bonus(board, value)
                
                # Apply aggressive play bonuses to policy
                policy = self._apply_aggressive_play_bonus(board, policy, move_count)
                
                # Apply PAG-based policy adjustments using ultra-dense features
                policy = self._apply_pag_policy_adjustments(board, policy, move_count)
                
                # Force decisive play based on move count
                policy = self._force_decisive_play(board, policy, move_count)
                
                # Temperature-based move selection with dynamic temperature
                temperature = self._get_dynamic_temperature(move_count)
                
                # Make random move if configured
                if self._should_make_random_move(board, move_count):
                    selected_move = self._get_random_move(board)
                    if selected_move is None:
                        # If random move selection fails, fall back to policy
                        selected_move = self._select_move_from_policy(board, policy, temperature)
                else:
                    selected_move = self._select_move_from_policy(board, policy, temperature)
                
                if selected_move is None:
                    logger.warning("No valid move found, ending game")
                    break
                
                # Record move with PAG features
                game_record.add_move(
                    move=selected_move,
                    policy=policy,
                    value=value,
                    pag_features=pag_features  # Include PAG features in game record
                )
                
                # Make the move
                board.push(selected_move)
                move_count += 1
                
                # Check dynamic move limit
                if self._apply_dynamic_move_limit(board, move_count):
                    logger.info(f"Dynamic move limit reached at move {move_count}")
                    break
                    
            except Exception as e:
                logger.error(f"Error during move {move_count}: {e}")
                logger.error(traceback.format_exc())
                break
        
        # Record final state
        game_record.add_state(board.copy())
        
        # Set game result
        if board.is_game_over():
            result = board.result()
            if result == "1-0":
                game_record.set_result(GameResult.WHITE_WINS)
            elif result == "0-1":
                game_record.set_result(GameResult.BLACK_WINS)
            else:
                game_record.set_result(GameResult.DRAW)
        else:
            # Game ended due to move limit
            game_record.set_result(GameResult.DRAW)
        
        # Log game statistics
        game_time = time.time() - start_time
        self.timing_metrics['total_time'] += game_time
        self.timing_metrics['num_games'] = self.timing_metrics.get('num_games', 0) + 1
        
        # Update metrics
        self.metrics['game_length'].append(move_count)
        self.metrics['game_time'].append(game_time)
        self.metrics['mcts_time'].append(total_mcts_time)
        self.metrics['avg_mcts_time_per_move'].append(total_mcts_time / max(1, move_count))
        
        # Increment game counter
        self.game_counter += 1
        
        # Log PAG usage statistics
        if self.config.use_dense_pag:
            pag_positions = sum(1 for move_data in game_record.moves if hasattr(move_data, 'pag_features') and move_data.pag_features is not None)
            logger.debug(f"Game {self.game_counter}: {pag_positions}/{move_count} positions with PAG features")
        
        return game_record
    
    def generate_games(self, num_games: Optional[int] = None, epoch: Optional[int] = None, save_games: bool = True) -> List[GameRecord]:
        """Generate self-play games for training.
        
        Args:
            num_games: Optional number of games to generate. If None, uses config value.
            epoch: Optional epoch number for logging.
            save_games: Whether to save the generated games to disk. Defaults to True.
            
        Returns:
            List of generated game records
        """
        logger.info("Starting self-play game generation...")
        logger.info(f"Configuration: {self.config}")
        
        try:
            # Initialize games and boards
            num_games = num_games or self.config.num_games
            games = []
            boards = [chess.Board() for _ in range(num_games)]
            game_records = [GameRecord() for _ in range(num_games)]  # Create game records upfront
            active_games = list(range(num_games))
            
            logger.info(f"Initialized {num_games} boards and game records")
            
            # Create progress bar
            pbar = tqdm(total=num_games, desc="Generating self-play games", 
                       disable=not self.config.use_tqdm)
            
            while active_games:  # Changed condition to only check active_games
                # Process each active game
                still_active = []  # New list to track games that remain active
                for game_idx in active_games:
                    try:
                        board = boards[game_idx]
                        game_record = game_records[game_idx]
                        
                        # Check if game is over first
                        if board.is_game_over() or len(game_record.moves) >= self.config.max_moves:
                            # Set result
                            if board.is_checkmate():
                                game_record.set_result(GameResult.WHITE_WINS if board.turn == chess.BLACK else GameResult.BLACK_WINS)
                            elif board.is_stalemate() or board.is_insufficient_material():
                                game_record.set_result(GameResult.DRAW)
                            elif board.can_claim_threefold_repetition():
                                # Check if we're in endgame (few pieces left)
                                num_pieces = len(board.piece_map())
                                if num_pieces <= 6:  # Endgame with 6 or fewer pieces
                                    game_record.set_result(GameResult.DRAW)  # Neutral reward in endgame
                                else:
                                    game_record.set_result(GameResult.REPETITION_DRAW)  # Penalty in opening/middlegame
                            else:
                                game_record.set_result(GameResult.DRAW)  # Draw by move limit
                            
                            games.append(game_record)
                            pbar.update(1)
                            logger.info(f"Game {game_idx + 1} completed: {game_record.result.name} after {len(game_record.moves)} moves")
                            continue  # Skip to next game
                        
                        # Record current state
                        game_record.add_state(board)
                        
                        # Convert board to graph data
                        data = board_to_hetero_data(board)
                        data = data.to(self.config.device)
                        
                        # 🚨 TEMPORARY FIX: Skip MCTS to avoid infinite loop, use direct model evaluation
                        try:
                            logger.info(f"🎯 TEMPORARY: Using direct model evaluation instead of MCTS to avoid infinite loop")
                            # Fallback to direct model evaluation
                            with torch.no_grad():
                                policy, value = self.model(data)
                                policy = policy.squeeze(0)
                                value = value.squeeze()
                        except Exception as e:
                            logger.error(f"❌ Even direct model evaluation failed: {e}")
                            raise
                        
                        # Convert policy to numpy if it's a tensor
                        if isinstance(policy, torch.Tensor):
                            policy = policy.detach().cpu().numpy()
                        
                        # Get legal moves and create mask
                        legal_moves = list(board.legal_moves)
                        legal_mask = np.zeros(5312, dtype=np.float32)
                        move_probs = {}
                        
                        for move in legal_moves:
                            move_idx = self.mcts._move_to_move_idx(move)
                            if move_idx is not None and 0 <= move_idx < 5312:
                                legal_mask[move_idx] = 1.0
                                move_probs[move] = float(policy[move_idx])
                        
                        # Apply aggressive play bonuses
                        policy_tensor = torch.tensor(policy, dtype=torch.float32, device=self.device)
                        move_count = len(game_record.moves)
                        policy_tensor = self._apply_aggressive_play_bonus(board, policy_tensor, move_count)
                        policy_tensor = self._force_decisive_play(board, policy_tensor, move_count)
                        
                        # Convert back to numpy
                        policy = policy_tensor.detach().cpu().numpy()
                        
                        # Update move probabilities with bonuses
                        for move in legal_moves:
                            move_idx = self.mcts._move_to_move_idx(move)
                            if move_idx is not None and 0 <= move_idx < 5312:
                                move_probs[move] = float(policy[move_idx])
                        
                        # Apply softmax to get valid probabilities
                        moves = list(move_probs.keys())
                        logits = np.array([move_probs[move] for move in moves])
                        
                        # Apply temperature before softmax (much higher temperature for more variety)
                        current_temperature = self._get_dynamic_temperature(move_count)
                        if current_temperature != 1.0:
                            logits = logits / current_temperature
                        
                        # Compute softmax
                        exp_logits = np.exp(logits - np.max(logits))
                        probs = exp_logits / exp_logits.sum()
                        
                        # Update move probabilities
                        move_probs = {move: float(prob) for move, prob in zip(moves, probs)}
                        
                        # Sample move
                        move_idx = np.random.choice(len(moves), p=probs)
                        selected_move = moves[move_idx]
                        
                        # Record move, policy, and value
                        policy_array = np.zeros(5312, dtype=np.float32)
                        for move, prob in move_probs.items():
                            move_idx = self.mcts._move_to_move_idx(move)
                            if move_idx is not None and 0 <= move_idx < 5312:
                                policy_array[move_idx] = prob
                        
                        policy_tensor = torch.tensor(policy_array, dtype=torch.float32)
                        game_record.add_move(selected_move, policy=policy_tensor, value=float(value))
                        
                        # Make move
                        board.push(selected_move)
                        
                        # Check if we should force game end based on dynamic limits
                        if self._apply_dynamic_move_limit(board, move_count):
                            # Force game end by declaring it a draw
                            game_record.set_result(GameResult.DRAW)
                            games.append(game_record)
                            pbar.update(1)
                            logger.info(f"Game {game_idx + 1} forced to end after {move_count} moves (dynamic limit)")
                            continue
                        
                        # Add to still active games
                        still_active.append(game_idx)
                        
                    except Exception as e:
                        logger.error(f"Error processing game {game_idx}: {str(e)}")
                        # Remove failed game from active games
                        continue
                
                # Update active games list
                active_games = still_active
                
                # Break if no games are active
                if not active_games:
                    break
            
            pbar.close()
            logger.info(f"Completed {len(games)} games in epoch {epoch if epoch is not None else 'N/A'}")
            logger.info(f"Total games played so far: {len(games)}")
            
            # Save games if we have any and save_games is True
            if games and save_games:
                self._save_games(games, epoch or 0)
            
            return games
        
        except Exception as e:
            logger.error(f"Error in generate_games: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def _save_games(self, games: List[GameRecord], epoch: int) -> str:
        """Save generated games to disk.
        
        Args:
            games: List of game records to save
            epoch: Current training epoch
            
        Returns:
            Path to saved games file
        """
        if not games:
            return None
            
        # Create filename with epoch number and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"games_epoch_{epoch}_{timestamp}.pkl"
        filepath = self.save_dir / filename
        
        # Save games
        with open(filepath, 'wb') as f:
            pickle.dump(games, f)
            
        logger.info(f"Saved {len(games)} games to {filepath}")
        return str(filepath)
    
    def create_dataloader(self, games: List[GameRecord]):
        """Create a dataloader from generated games.
        
        Args:
            games: List of GameRecord objects
            
        Returns:
            DataLoader for training
        """
        # Import here to avoid circular dependency
        from rival_ai.data import PAGDataset, PAGDataLoader
        
        dataset = PAGDataset.from_game_records(games)
        return PAGDataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
    
    def _log_metrics(self):
        """Log current self-play metrics."""
        if not self.metrics['num_moves']:
            return
        
        avg_moves = np.mean(self.metrics['num_moves'])
        white_wins = sum(1 for r in self.metrics['result'] if r == GameResult.WHITE_WINS)
        black_wins = sum(1 for r in self.metrics['result'] if r == GameResult.BLACK_WINS)
        draws = sum(1 for r in self.metrics['result'] if r == GameResult.DRAW)
        total = len(self.metrics['result'])
        
        logger.info(
            f"Self-play metrics: "
            f"avg_moves={avg_moves:.1f}, "
            f"white_wins={white_wins/total:.1%}, "
            f"black_wins={black_wins/total:.1%}, "
            f"draws={draws/total:.1%}"
        )
    
    def reset_metrics(self):
        """Reset all metrics."""
        self.metrics.clear()

    def _get_position_key(self, board: chess.Board) -> str:
        """Get a unique key for the current position.
        
        Args:
            board: Current chess board
            
        Returns:
            str: Unique key representing the position
        """
        return board.fen()

    def _apply_forward_progress_bonus(self, board: chess.Board, value: float) -> float:
        """Apply enhanced bonus for pawn advancement and piece development.
        
        Args:
            board: Current chess board
            value: Current position value
            
        Returns:
            float: Modified value with forward progress bonus applied
        """
        bonus = 0.0
        move_count = board.fullmove_number
        
        # Pawn advancement bonus (stronger in opening)
        if move_count <= 20:  # Extended from 10 to 20 moves
            for rank in range(8):
                for file in range(8):
                    square = chess.square(file, rank)
                    piece = board.piece_at(square)
                    if piece and piece.piece_type == chess.PAWN:
                        # Calculate progress towards promotion
                        progress = rank / 7.0 if piece.color == chess.WHITE else (7 - rank) / 7.0
                        # Scale bonus based on game phase
                        phase_scale = 1.5 if move_count <= 10 else 1.0
                        bonus += progress * self.config.forward_progress_bonus * phase_scale
        
        # Piece development bonus (only in opening)
        if move_count <= 15:
            for piece_type in [chess.KNIGHT, chess.BISHOP]:
                for square in board.pieces(piece_type, board.turn):
                    # Bonus for pieces that have moved from their starting squares
                    if piece_type == chess.KNIGHT and square in [chess.B1, chess.G1, chess.B8, chess.G8]:
                        bonus += 0.05
                    elif piece_type == chess.BISHOP and square in [chess.C1, chess.F1, chess.C8, chess.F8]:
                        bonus += 0.05
        
        value += bonus
        return value

    def _get_dynamic_temperature(self, move_count: int) -> float:
        """Get dynamic temperature based on game phase."""
        if move_count <= self.config.opening_moves:
            return self.config.opening_temperature
        elif move_count <= self.config.midgame_moves:
            return self.config.midgame_temperature
        else:
            return self.config.endgame_temperature

    def _would_cause_repetition(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if a move would cause a repetition."""
        # Make a copy of the board and try the move
        test_board = board.copy()
        test_board.push(move)
        new_pos = self._get_position_key(test_board)
        
        # Count how many times this position has occurred recently
        # We'll use a simple approach - check if the position exists in recent history
        # For now, we'll use the board's built-in repetition detection
        return test_board.is_repetition()

    def _extract_pag_features(self, board: chess.Board) -> Optional[torch.Tensor]:
        """Extract PAG features for the current position.
        
        Args:
            board: Chess board position
            
        Returns:
            PAG feature tensor or None if extraction fails
        """
        if not self.config.use_dense_pag or self.pag_engine is None:
            return None
            
        try:
            # Convert board to FEN
            fen = board.fen()
            
            # Extract dense PAG features using Rust engine
            pag_data = self.pag_engine.fen_to_dense_pag(fen)
            
            # Extract relevant features and convert to tensor
            if 'piece_features' in pag_data and 'square_features' in pag_data:
                piece_features = torch.tensor(pag_data['piece_features'], dtype=torch.float32)
                square_features = torch.tensor(pag_data['square_features'], dtype=torch.float32)
                
                # Combine piece and square features
                # This creates a comprehensive position representation
                all_features = torch.cat([
                    piece_features.flatten(),
                    square_features.flatten()
                ], dim=0)
                
                return all_features.to(self.device)
            
        except Exception as e:
            if self.config.pag_fallback_to_python:
                logger.debug(f"PAG extraction failed, using fallback: {e}")
                # Fallback to Python PAG implementation
                try:
                    from rival_ai.pag import PositionalAdjacencyGraph, PAGConfig
                    pag_config = PAGConfig()
                    pag = PositionalAdjacencyGraph(pag_config)
                    pag.build_from_board(board)
                    hetero_data = pag.to_hetero_data()
                    
                    # Extract features from hetero data
                    if 'piece' in hetero_data and hasattr(hetero_data['piece'], 'x'):
                        return hetero_data['piece'].x.flatten().to(self.device)
                    
                except Exception as fallback_e:
                    logger.warning(f"Both Rust and Python PAG extraction failed: {e}, {fallback_e}")
            else:
                logger.warning(f"PAG feature extraction failed: {e}")
        
        return None

    def _select_move_from_policy(self, board: chess.Board, policy: torch.Tensor, temperature: float) -> Optional[chess.Move]:
        """Select a move from the policy distribution.
        
        Args:
            board: Current chess board
            policy: Policy tensor with move probabilities
            temperature: Temperature for move selection
            
        Returns:
            Selected chess move or None if no valid move
        """
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        
        legal_indices = []
        legal_probs = []
        
        for move in legal_moves:
            # Convert move to policy index using FIXED compact encoding
            if move.promotion:
                # Use the same compact encoding as training_types.py
                piece_type = {
                    chess.KNIGHT: 0,
                    chess.BISHOP: 1, 
                    chess.ROOK: 2,
                    chess.QUEEN: 3
                }.get(move.promotion, 3)
                
                from_file = move.from_square % 8
                from_rank = move.from_square // 8
                to_file = move.to_square % 8
                to_rank = move.to_square // 8
                
                if to_file == from_file:
                    direction = 0
                elif to_file == from_file - 1:
                    direction = 1
                elif to_file == from_file + 1:
                    direction = 2
                else:
                    continue  # Skip invalid promotion
                
                if from_rank == 6 and to_rank == 7:
                    side_offset = 0
                elif from_rank == 1 and to_rank == 0:
                    side_offset = 96
                else:
                    continue  # Skip invalid promotion
                
                move_idx = 4096 + side_offset + (from_file * 12) + (direction * 4) + piece_type
            else:
                move_idx = move.from_square * 64 + move.to_square
            
            if 0 <= move_idx < len(policy):
                legal_indices.append(move_idx)
                legal_probs.append(float(policy[move_idx]))
        
        if not legal_probs:
            # Fallback to random legal move
            return legal_moves[0]
        
        # Apply temperature
        if temperature > 0:
            probs = np.array(legal_probs)
            if temperature != 1.0:
                probs = probs ** (1.0 / temperature)
            probs = probs / probs.sum()
            
            # Sample move
            selected_idx = np.random.choice(len(legal_indices), p=probs)
        else:
            # Greedy selection
            selected_idx = np.argmax(legal_probs)
        
        return legal_moves[selected_idx]

    def _apply_pag_policy_adjustments(self, board: chess.Board, policy: torch.Tensor, move_count: int) -> torch.Tensor:
        """Apply PAG-based policy adjustments using ultra-dense features."""
        try:
            import rival_ai_engine
            engine = rival_ai_engine.PyPAGEngine()
            
            # Get current position PAG features
            current_pag = engine.fen_to_dense_pag(board.fen())
            if 'node_features' not in current_pag or 'node_types' not in current_pag:
                return policy
            
            policy = policy.clone()
            legal_moves = list(board.legal_moves)
            
            for move in legal_moves:
                move_idx = self.mcts._move_to_move_idx(move)
                if move_idx is None or move_idx >= 5312:
                    continue
                
                # Simulate the move to get PAG features after the move
                board.push(move)
                try:
                    new_pag = engine.fen_to_dense_pag(board.fen())
                    board.pop()
                    
                    if 'node_features' not in new_pag or 'node_types' not in new_pag:
                        continue
                    
                    # Calculate PAG-based move evaluation
                    move_bonus = self._evaluate_move_with_pag(current_pag, new_pag, move, board)
                    
                    # Apply the bonus/penalty to the policy
                    if move_bonus != 0:
                        policy[move_idx] *= (1.0 + move_bonus)
                        
                        # Log significant adjustments
                        if abs(move_bonus) > 0.5:
                            logger.info(f"PAG policy adjustment for {move}: {move_bonus:.3f}")
                
                except Exception as e:
                    board.pop()
                    logger.debug(f"PAG evaluation failed for move {move}: {e}")
                    continue
            
            return policy
            
        except Exception as e:
            logger.warning(f"PAG policy adjustments failed: {e}")
            return policy
    
    def _evaluate_move_with_pag(self, before_pag, after_pag, move: chess.Move, board: chess.Board) -> float:
        """
        Evaluate a move based on PAG feature changes.
        Returns a bonus/penalty multiplier for the move probability.
        """
        move_bonus = 0.0
        
        try:
            before_features = before_pag['node_features']
            before_types = before_pag['node_types']
            after_features = after_pag['node_features']
            after_types = after_pag['node_types']
            
            # Count hanging pieces before and after
            hanging_before = self._count_hanging_pieces(before_features, before_types)
            hanging_after = self._count_hanging_pieces(after_features, after_types)
            
            # MASSIVE bonus for moves that capture hanging pieces
            if hanging_after < hanging_before:
                pieces_saved = hanging_before - hanging_after
                move_bonus += pieces_saved * 3.0  # Huge bonus for reducing hanging pieces
                logger.info(f"🎯 Move {move} reduces hanging pieces by {pieces_saved}")
            
            # MASSIVE penalty for moves that create hanging pieces
            elif hanging_after > hanging_before:
                pieces_hung = hanging_after - hanging_before
                move_bonus -= pieces_hung * 5.0  # Huge penalty for creating hanging pieces
                logger.warning(f"🚨 Move {move} creates {pieces_hung} hanging pieces!")
            
            # Evaluate piece activity changes
            activity_before = self._calculate_total_activity(before_features, before_types)
            activity_after = self._calculate_total_activity(after_features, after_types)
            activity_change = activity_after - activity_before
            
            # Bonus for improving piece activity
            if activity_change > 0.1:
                move_bonus += activity_change * 2.0
            elif activity_change < -0.1:
                move_bonus += activity_change * 1.0  # Penalty for reducing activity
            
            # Evaluate king safety changes
            king_safety_before = self._calculate_king_safety(before_features, before_types)
            king_safety_after = self._calculate_king_safety(after_features, after_types)
            safety_change = king_safety_after - king_safety_before
            
            # Strong bonus/penalty for king safety changes
            if safety_change > 0.1:
                move_bonus += safety_change * 3.0  # Strong bonus for improving king safety
            elif safety_change < -0.1:
                move_bonus += safety_change * 4.0  # Strong penalty for weakening king safety
            
            # Special bonuses for tactical moves
            if board.is_capture(move):
                captured_piece = board.piece_at(move.to_square)
                moving_piece = board.piece_at(move.from_square)
                
                if captured_piece and moving_piece:
                    piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, 
                                  chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
                    captured_value = piece_values.get(captured_piece.piece_type, 0)
                    moving_value = piece_values.get(moving_piece.piece_type, 0)
                    
                    # Extra bonus for good captures
                    if captured_value > moving_value:
                        move_bonus += (captured_value - moving_value) * 1.5
                    # Penalty for bad captures
                    elif captured_value < moving_value:
                        move_bonus -= (moving_value - captured_value) * 2.0
            
            # Bonus for checks that improve position
            if board.gives_check(move):
                move_bonus += 0.8
            
            return move_bonus
            
        except Exception as e:
            logger.debug(f"PAG move evaluation failed: {e}")
            return 0.0
    
    def _count_hanging_pieces(self, node_features, node_types) -> int:
        """Count hanging pieces from PAG features."""
        hanging_count = 0
        
        for features, node_type in zip(node_features, node_types):
            if node_type == 'piece' and len(features) >= 76:
                # Check vulnerability features (first feature in vulnerability is hanging indicator)
                tactical_features = features[:76]
                if len(tactical_features) >= 61:  # Ensure we have vulnerability features
                    vulnerability_features = tactical_features[60:76]
                    hanging_score = vulnerability_features[0] if len(vulnerability_features) > 0 else 0.0
                    
                    if hanging_score > 0.7:  # Threshold for considering piece hanging
                        hanging_count += 1
        
        return hanging_count
    
    def _calculate_total_activity(self, node_features, node_types) -> float:
        """Calculate total piece activity from PAG features."""
        total_activity = 0.0
        piece_count = 0
        
        for features, node_type in zip(node_features, node_types):
            if node_type == 'piece' and len(features) >= 156:
                # Activity metrics are in positional features (dimensions 52-65)
                positional_features = features[76:156]
                if len(positional_features) >= 66:
                    activity_features = positional_features[52:66]
                    activity_score = sum(activity_features) / len(activity_features)
                    total_activity += activity_score
                    piece_count += 1
        
        return total_activity / max(piece_count, 1)
    
    def _calculate_king_safety(self, node_features, node_types) -> float:
        """Calculate king safety from PAG features."""
        total_safety = 0.0
        piece_count = 0
        
        for features, node_type in zip(node_features, node_types):
            if node_type == 'piece' and len(features) >= 216:
                # King safety features are in strategic features (dimensions 12-23)
                strategic_features = features[156:216]
                if len(strategic_features) >= 24:
                    king_safety_features = strategic_features[12:24]
                    safety_score = sum(king_safety_features) / len(king_safety_features)
                    total_safety += safety_score
                    piece_count += 1
        
        return total_safety / max(piece_count, 1)

def generate_game(model: ChessGNN, config: SelfPlayConfig) -> GameRecord:
    """Generate a single game of self-play.
    
    Args:
        model: Neural network model
        config: Self-play configuration
        
    Returns:
        GameRecord containing the game history
    """
    board = chess.Board()
    game_record = GameRecord()
    move_count = 0
    # Create proper MCTS config for the standalone function
    mcts_config = MCTSConfig(
        num_simulations=config.num_simulations,
        c_puct=config.c_puct,
        temperature=config.temperature,
        dirichlet_alpha=config.dirichlet_alpha,
        dirichlet_weight=config.dirichlet_weight,
        device=config.device
    )
    mcts = MCTS(model, mcts_config)  # Create MCTS instance for move index conversion
    
    while not board.is_game_over() and move_count < config.max_moves:
        # Record current state
        game_record.add_state(board)
        
        # Get policy and value from model
        policy, value = model(board)
        
        # Get legal moves and create mask
        legal_moves = list(board.legal_moves)
        legal_mask = torch.zeros(5312, dtype=torch.float32, device=config.device)
        for move in legal_moves:
            # Use MCTS move index formula consistently
            move_idx = mcts._move_to_move_idx(move)
            if move_idx is not None and 0 <= move_idx < 5312:
                legal_mask[move_idx] = 1.0
            else:
                logger.warning(f"Invalid move index {move_idx} for move {move}")
        
        # Apply mask to policy
        masked_policy = policy * legal_mask
        masked_policy = masked_policy / (masked_policy.sum() + 1e-8)
        
        # Sample move from policy
        move_idx = torch.multinomial(masked_policy, 1).item()
        
        # Find corresponding move
        selected_move = None
        for move in legal_moves:
            idx = mcts._move_to_move_idx(move)
            if idx == move_idx:
                selected_move = move
                break
        
        if selected_move is None:
            logger.error(f"Could not find move for index {move_idx}")
            break
        
        # Record move and policy
        game_record.add_move(selected_move)
        game_record.add_policy(masked_policy)
        game_record.add_value(float(value))
        
        # Make move
        board.push(selected_move)
        move_count += 1
        
        # Log progress every 10 moves
        if move_count % 10 == 0:
            logger.info(f"Game progress: Move {move_count} | Last move: {selected_move}")
    
    # Record final state
    game_record.add_state(board)
    
    # Set game result
    if board.is_checkmate():
        game_record.set_result(GameResult.WHITE_WINS if board.turn else GameResult.BLACK_WINS)
    elif board.is_stalemate() or board.is_insufficient_material():
        game_record.set_result(GameResult.DRAW)
    elif board.can_claim_threefold_repetition():
        # Check if we're in endgame (few pieces left)
        num_pieces = len(board.piece_map())
        if num_pieces <= 6:  # Endgame with 6 or fewer pieces
            game_record.set_result(GameResult.DRAW)  # Neutral reward in endgame
        else:
            game_record.set_result(GameResult.REPETITION_DRAW)  # Penalty in opening/middlegame
    else:
        game_record.set_result(GameResult.DRAW)  # Draw by move limit
    
    return game_record 