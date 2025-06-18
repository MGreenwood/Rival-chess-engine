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
        
        # Initialize metrics
        self.metrics = defaultdict(list)
        self.timing_metrics = defaultdict(float)
        self.position_counts = defaultdict(int)  # Track position occurrences
        self.last_positions = defaultdict(list)  # Track recent positions for each game
    
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
        move_count = board.fullmove_number()
        
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
        
        # Create MCTS config
        self.mcts_config = MCTSConfig(
            num_simulations=config.num_simulations,
            c_puct=config.c_puct,
            temperature=config.temperature,
            dirichlet_alpha=config.dirichlet_alpha,
            dirichlet_weight=config.dirichlet_weight
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
                material_bonus = material_balance * 0.5  # Increased from 0.1
            else:
                # White is behind in material - penalty to discourage further losses
                material_bonus = material_balance * 1.0  # Increased from 0.2
        else:
            # Black's turn - reward negative material balance (black advantage), penalize positive
            if material_balance < 0:
                # Black has material advantage - bonus
                material_bonus = abs(material_balance) * 0.5  # Increased from 0.1
            else:
                # Black is behind in material - penalty
                material_bonus = -material_balance * 1.0  # Increased from 0.2
        
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
            # Good capture available - strongly encourage it
            capture_bonus = best_capture_value * 1.0  # Increased from 0.3
        elif best_capture_value < 0:
            # Only bad captures available - penalty to discourage them
            capture_bonus = best_capture_value * 0.3  # Increased from 0.1
        
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
                obvious_bonus = exchange_value * 2.0  # Very strong bonus
                if obvious_bonus > obvious_capture_bonus:
                    obvious_capture_bonus = obvious_bonus
                    logger.info(f"OBVIOUS CAPTURE FOUND: {capturing_piece.piece_type} capturing {captured_piece.piece_type} "
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
        """Play a single game of self-play."""
        board = chess.Board()
        game_record = GameRecord()
        game_record.moves = []
        game_record.result = None
        game_record.metadata = {
            'num_simulations': self.config.num_simulations,
            'c_puct': self.config.c_puct,
            'temperature': self.config.temperature,
            'dirichlet_alpha': self.config.dirichlet_alpha,
            'dirichlet_weight': self.config.dirichlet_weight
        }
        
        # Initialize MCTS for this game
        mcts = MCTS(
            model=self.model,
            config=self.mcts_config
        )
        
        # Track positions for repetition detection
        position_history = []
        move_count = 0
        
        while not board.is_game_over() and move_count < self.config.max_moves:
            try:
                # Get current position value
                position_value = mcts._get_value(board)
                
                # Apply repetition penalty
                position_value = self._apply_repetition_penalty(board, position_value, 0)  # Using game index 0 for single game
                
                # Apply forward progress bonus
                position_value = self._apply_forward_progress_bonus(board, position_value)
                
                # Apply material evaluation
                position_value = self._apply_material_evaluation(board, position_value)
                
                # Apply capture encouragement
                position_value = self._apply_capture_encouragement(board, position_value)
                
                # Apply obvious capture bonus (very strong for good captures)
                position_value = self._apply_obvious_capture_bonus(board, position_value)
                
                # Apply aggressive play bonuses
                policy_tensor = torch.tensor(position_value, dtype=torch.float32, device=self.device)
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
                game_record.add_move(selected_move, policy=policy_tensor, value=float(position_value))
                
                # Make move
                board.push(selected_move)
                move_count += 1
                
                # Track position for repetition detection
                position_key = self._get_position_key(board)
                position_history.append(position_key)
                
            except Exception as e:
                logger.error(f"Error during game play: {str(e)}")
                logger.error(f"Error type: {type(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                logger.error(f"Board state: {board.fen()}")
                logger.error(f"Move count: {move_count}")
                raise
        
        # Record game result
        if board.is_checkmate():
            game_record.result = GameResult.BLACK_WINS if board.turn else GameResult.WHITE_WINS
        elif board.is_stalemate():
            game_record.result = GameResult.DRAW
        elif board.is_insufficient_material():
            game_record.result = GameResult.DRAW
        elif board.is_fifty_moves():
            game_record.result = GameResult.DRAW
        elif board.is_repetition():
            game_record.result = GameResult.DRAW
        elif move_count >= self.config.max_moves:
            game_record.result = GameResult.DRAW
        else:
            game_record.result = GameResult.DRAW  # Default to draw for any other case
        
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