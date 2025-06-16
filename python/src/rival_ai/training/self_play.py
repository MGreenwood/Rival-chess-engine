"""
Self-play module for generating training data through self-play games.
"""

import os
import time
import logging
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
import chess
from torch.utils.data import Dataset
import random
import pickle

from rival_ai.models import ChessGNN
from rival_ai.mcts import MCTS, MCTSConfig
from rival_ai.chess import GameResult
from rival_ai.training.types import GameRecord
from rival_ai.utils.board_conversion import board_to_hetero_data
from .loss import PolicyValueLoss  # Import directly from loss module instead of rival_ai.training

logger = logging.getLogger(__name__)

@dataclass
class SelfPlayConfig:
    """Configuration for self-play game generation."""
    
    num_games: int = 100
    num_simulations: int = 400
    max_moves: int = 100
    dirichlet_alpha: float = 0.3
    dirichlet_weight: float = 0.25
    temperature: float = 1.0
    c_puct: float = 1.0
    save_dir: str = 'self_play_data'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size: int = 32
    num_workers: int = 4
    use_tqdm: bool = True
    log_timing: bool = True
    num_parallel_games: int = 10
    prefetch_factor: int = 2
    opening_temperature: float = 1.0
    midgame_temperature: float = 0.5
    endgame_temperature: float = 0.1
    repetition_penalty: float = 0.2  # Penalty for repeated positions
    forward_progress_bonus: float = 0.1  # Bonus for pawn advancement
    min_pieces_for_repetition_penalty: int = 8  # Only apply repetition penalty when more pieces are on board

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
        
        # Initialize metrics
        self.metrics = defaultdict(list)
        self.timing_metrics = defaultdict(float)
        self.position_counts = defaultdict(int)  # Track position occurrences
        self.last_positions = defaultdict(list)  # Track recent positions for each game
    
    def _get_position_key(self, board: chess.Board) -> str:
        """Get a unique key for the current position."""
        return board.fen()

    def _apply_repetition_penalty(self, board: chess.Board, value: float, game_idx: int) -> float:
        """Apply penalty for repeated positions."""
        position_key = self._get_position_key(board)
        num_pieces = len(board.piece_map())
        
        # Only apply penalty if we have enough pieces on the board
        if num_pieces >= self.config.min_pieces_for_repetition_penalty:
            # Check recent positions in this game
            recent_positions = self.last_positions[game_idx]
            if position_key in recent_positions:
                # Count how many times this position has occurred recently
                recent_count = recent_positions.count(position_key)
                penalty = self.config.repetition_penalty * recent_count
                value -= penalty
            
            # Update recent positions (keep last 10 positions)
            recent_positions.append(position_key)
            if len(recent_positions) > 10:
                recent_positions.pop(0)
        
        return value

    def _apply_forward_progress_bonus(self, board: chess.Board, value: float) -> float:
        """Apply bonus for pawn advancement in the opening."""
        if board.fullmove_number() <= 10:  # Only in opening
            bonus = 0.0
            for rank in range(8):
                for file in range(8):
                    square = chess.square(file, rank)
                    piece = board.piece_at(square)
                    if piece and piece.piece_type == chess.PAWN:
                        # Calculate progress towards promotion
                        progress = rank / 7.0 if piece.color == chess.WHITE else (7 - rank) / 7.0
                        bonus += progress * self.config.forward_progress_bonus
            value += bonus
        return value

    def generate_games(self, epoch: Optional[int] = None) -> List[GameRecord]:
        """Generate self-play games.
        
        Args:
            epoch: Current training epoch (for logging)
            
        Returns:
            List[GameRecord]: List of game records
        """
        logger.info("Starting self-play game generation...")
        logger.info(f"Configuration: {self.config}")
        
        # Initialize games
        games = [GameRecord() for _ in range(self.config.num_games)]
        boards = [chess.Board() for _ in range(self.config.num_games)]
        active_games = list(range(self.config.num_games))
        move_count = 0
        completed_games = 0
        
        with tqdm(total=self.config.num_games, desc="Generating self-play games", 
                 disable=not self.config.use_tqdm) as pbar:
            
            while active_games and move_count < self.config.max_moves:
                # Record states for active games
                for game_idx in active_games:
                    games[game_idx].add_state(boards[game_idx])
                
                # Get policies and values for all active games
                policies = []
                values = []
                for game_idx in active_games:
                    policy, value = self.mcts.get_action_policy(boards[game_idx])
                    policies.append(policy)
                    values.append(value)
                
                # Process each active game
                still_active = []
                for i, game_idx in enumerate(active_games):
                    board = boards[game_idx]
                    policy = policies[i]
                    value = values[i]
                    
                    # Get legal moves and create mask
                    legal_moves = list(board.legal_moves)
                    legal_mask = np.zeros(5312, dtype=np.float32)
                    for move in legal_moves:
                        move_idx = self.mcts._move_to_move_idx(move)
                        if move_idx is not None and 0 <= move_idx < 5312:
                            legal_mask[move_idx] = 1.0
                    
                    # Apply mask to policy
                    masked_policy = policy * legal_mask
                    
                    # Get probabilities for legal moves
                    move_probs = {}
                    for move in legal_moves:
                        prob = masked_policy.get(move, 1e-8)
                        move_probs[move] = float(prob)
                    
                    # Add small epsilon to prevent zero probabilities
                    min_prob = 1e-8
                    for move in move_probs:
                        move_probs[move] = max(move_probs[move], min_prob)
                    
                    # Apply temperature if needed
                    if self.config.temperature > 0:
                        moves = list(move_probs.keys())
                        probs = torch.tensor([move_probs[move] for move in moves], device=self.config.device)
                        probs = probs ** (1.0 / self.config.temperature)
                        probs = probs / (probs.sum() + 1e-8)
                        move_probs = {move: float(prob) for move, prob in zip(moves, probs)}
                    
                    # Sample move
                    if self.config.temperature > 0:
                        moves = list(move_probs.keys())
                        probs = np.array([move_probs[move] for move in moves])
                        probs = np.array(probs, dtype=np.float64)
                        probs = np.clip(probs, 0, None)
                        total = probs.sum()
                        if total == 0 or not np.isfinite(total):
                            probs = np.ones_like(probs) / len(probs)
                        else:
                            probs = probs / total
                        move_idx = np.random.choice(len(moves), p=probs)
                        move = moves[move_idx]
                    else:
                        move = max(move_probs.items(), key=lambda x: x[1])[0]
                    
                    # Record move and policy
                    games[game_idx].add_move(move)
                    
                    # Convert policy to array format for storage
                    policy_array = np.zeros(5312, dtype=np.float32)
                    for m, prob in move_probs.items():
                        move_idx = self.mcts._move_to_move_idx(m)
                        if move_idx is not None and 0 <= move_idx < 5312:
                            policy_array[move_idx] = float(prob)
                    games[game_idx].add_policy(policy_array)
                    games[game_idx].add_value(value)
                    
                    # Make move
                    board.push(move)
                    
                    # Check if game is over
                    if board.is_game_over():
                        # Set game result
                        if board.is_checkmate():
                            games[game_idx].set_result(GameResult.WHITE_WINS if board.turn == chess.BLACK else GameResult.BLACK_WINS)
                        elif board.is_stalemate() or board.is_insufficient_material():
                            games[game_idx].set_result(GameResult.DRAW)
                        elif board.can_claim_threefold_repetition():
                            # Check if we're in endgame (few pieces left)
                            num_pieces = len(board.piece_map())
                            if num_pieces <= 6:  # Endgame with 6 or fewer pieces
                                games[game_idx].set_result(GameResult.DRAW)  # Neutral reward in endgame
                            else:
                                games[game_idx].set_result(GameResult.REPETITION_DRAW)  # Penalty in opening/middlegame
                        else:
                            games[game_idx].set_result(GameResult.DRAW)
                        completed_games += 1
                        pbar.update(1)
                        # Log game completion with result
                        logger.info(f"Game {game_idx + 1} completed: {games[game_idx].result.name} after {len(games[game_idx].moves)} moves")
                    else:
                        still_active.append(game_idx)
                
                active_games = still_active
                move_count += 1
            
            # Handle any remaining active games (they hit the move limit)
            for game_idx in active_games:
                games[game_idx].set_result(GameResult.DRAW)  # Draw by move limit
                completed_games += 1
                pbar.update(1)
                logger.info(f"Game {game_idx + 1} completed: Draw by move limit after {len(games[game_idx].moves)} moves")
            
            # Update game counter for next epoch
            self.game_counter += completed_games
            
            # Log completion status
            logger.info(f"Completed {completed_games} games in epoch {epoch + 1 if epoch is not None else 'unknown'}")
            logger.info(f"Total games played so far: {self.game_counter}")
        
        # Save games
        self._save_games(games, epoch or 0)
        
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
    ):
        """Initialize self-play manager.
        
        Args:
            model: The GNN model to use for self-play
            config: Self-play configuration
        """
        self.model = model.to(config.device)
        self.config = config
        self.mcts = MCTS(
            model=model,
            config=MCTSConfig(
                num_simulations=config.num_simulations,
                c_puct=config.c_puct
            )
        )
        self.mcts.model = self.model  # Set the model after initialization
        
        # Create save directory if it doesn't exist
        os.makedirs(config.save_dir, exist_ok=True)
        
        # Initialize metrics
        self.metrics = defaultdict(list)
        
        # Initialize timing metrics
        self.timing_metrics = {
            'total_time': 0.0,
            'move_selection_time': 0.0,
            'move_execution_time': 0.0,
            'state_recording_time': 0.0,
            'num_moves': 0,
            'num_games': 0,
        }
        
        # Initialize game counter for continuous numbering across epochs
        self.game_counter = 0
    
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
    
    def play_game(self) -> GameRecord:
        """Play a single game of self-play.
        
        Returns:
            GameRecord containing the game history and result
        """
        board = chess.Board()
        game_record = GameRecord()
        move_count = 0
        game_idx = len(self.last_positions)  # Unique game index
        
        while not board.is_game_over() and move_count < self.config.max_moves:
            # Record current state
            game_record.add_state(board)
            
            # Determine temperature based on game phase
            if move_count < 10:
                temperature = self.config.opening_temperature
            elif move_count < 30:
                temperature = self.config.midgame_temperature
            else:
                temperature = self.config.endgame_temperature
            
            # Run MCTS search with current temperature
            self.mcts.config.temperature = temperature
            self.mcts.search(board)
            root = self.mcts.root
            
            # Get policy from visit counts
            policy = root.get_policy()
            
            # Get value from neural network
            data = board_to_hetero_data(board)
            data = data.to(self.config.device)
            with torch.no_grad():
                _, value = self.model(data)
                value = float(value)
            
            # Apply repetition penalty and forward progress bonus
            value = self._apply_repetition_penalty(board, value, game_idx)
            value = self._apply_forward_progress_bonus(board, value)
            
            # Get legal moves and create mask
            legal_moves = list(board.legal_moves)
            legal_mask = np.zeros(5312, dtype=np.float32)
            for move in legal_moves:
                move_idx = self.mcts._move_to_move_idx(move)
                if move_idx is not None and 0 <= move_idx < 5312:
                    legal_mask[move_idx] = 1.0
            
            # Apply mask to policy
            masked_policy = policy * legal_mask
            
            # Get probabilities for legal moves
            move_probs = {}
            for move in legal_moves:
                prob = masked_policy.get(move, 1e-8)
                move_probs[move] = float(prob)
            
            # Add small epsilon to prevent zero probabilities
            min_prob = 1e-8
            for move in move_probs:
                move_probs[move] = max(move_probs[move], min_prob)
            
            # Apply temperature if needed
            if temperature > 0:
                moves = list(move_probs.keys())
                probs = torch.tensor([move_probs[move] for move in moves], device=self.config.device)
                probs = probs ** (1.0 / temperature)
                probs = probs / (probs.sum() + 1e-8)
                move_probs = {move: float(prob) for move, prob in zip(moves, probs)}
            
            # Sample move
            if temperature > 0:
                moves = list(move_probs.keys())
                probs = np.array([move_probs[move] for move in moves])
                probs = np.array(probs, dtype=np.float64)
                probs = np.clip(probs, 0, None)
                total = probs.sum()
                if total == 0 or not np.isfinite(total):
                    probs = np.ones_like(probs) / len(probs)
                else:
                    probs = probs / total
                move_idx = np.random.choice(len(moves), p=probs)
                move = moves[move_idx]
            else:
                move = max(move_probs.items(), key=lambda x: x[1])[0]
            
            # Record move and policy
            game_record.add_move(move)
            game_record.add_policy(masked_policy)
            game_record.add_value(value)
            
            # Make move
            board.push(move)
            move_count += 1
            
            # Log progress every 10 moves
            if move_count % 10 == 0:
                logger.info(f"Game progress: Move {move_count} | Last move: {move} | Temperature: {temperature:.2f}")
        
        # Record final state
        game_record.add_state(board)
        
        # Set game result with repetition handling
        if board.is_checkmate():
            game_record.set_result(GameResult.WHITE_WINS if board.turn else GameResult.BLACK_WINS)
        elif board.is_stalemate() or board.is_insufficient_material():
            game_record.set_result(GameResult.DRAW)
        elif board.can_claim_threefold_repetition():
            num_pieces = len(board.piece_map())
            if num_pieces <= 6:  # Endgame with 6 or fewer pieces
                game_record.set_result(GameResult.DRAW)  # Neutral reward in endgame
            else:
                game_record.set_result(GameResult.REPETITION_DRAW)  # Penalty in opening/middlegame
        else:
            game_record.set_result(GameResult.DRAW)  # Draw by move limit
        
        # Clear game history
        self.last_positions[game_idx] = []
        
        # Log game completion
        logger.info(
            f"Game completed: {game_record.result.name} | "
            f"Moves: {move_count} | "
            f"Final position: {board.fen()}"
        )
        
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
                        
                        # Get policy and value from model
                        with torch.no_grad():
                            policy, value = self.model(data)
                            policy = policy.squeeze(0)
                            value = value.squeeze()
                        
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
                        
                        # Apply softmax to get valid probabilities
                        moves = list(move_probs.keys())
                        logits = np.array([move_probs[move] for move in moves])
                        
                        # Apply temperature before softmax
                        if self.config.temperature != 1.0:
                            logits = logits / self.config.temperature
                        
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
            import traceback
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
            
        # Create filename with epoch number
        filename = f"games_epoch_{epoch}.pkl"
        filepath = os.path.join(self.config.save_dir, filename)
        
        # Save games
        with open(filepath, 'wb') as f:
            pickle.dump(games, f)
            
        logger.info(f"Saved {len(games)} games to {filepath}")
        return filepath
    
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
    mcts = MCTS(model, config)  # Create MCTS instance for move index conversion
    
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