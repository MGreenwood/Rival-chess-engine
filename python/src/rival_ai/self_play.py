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
import chess

from rival_ai.models import ChessGNN
from rival_ai.mcts import MCTS, MCTSConfig
from rival_ai.chess import Move, GameResult
from rival_ai.data.dataset import PAGDataset
from rival_ai.types import GameRecord

logger = logging.getLogger(__name__)

@dataclass
class SelfPlayConfig:
    """Configuration for self-play."""
    num_games: int = 100
    mcts: MCTS = None  # MCTS instance to use for move selection
    max_moves: int = 200  # Maximum moves per game
    save_dir: str = "self_play_data"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 32
    num_workers: int = 4
    use_tqdm: bool = True

class GameRecord:
    """Record of a single game of self-play."""
    
    def __init__(self, states: Optional[List[chess.Board]] = None, moves: Optional[List[Move]] = None,
                 policies: Optional[List[torch.Tensor]] = None, values: Optional[List[float]] = None,
                 result: Optional[GameResult] = None):
        """Initialize a game record.
        
        Args:
            states: List of board states
            moves: List of moves made
            policies: List of policy tensors
            values: List of value predictions
            result: Game result
        """
        self.states = states or []
        self.moves = moves or []
        self.policies = policies or []
        self.values = values or []
        self.result = result
    
    def add_move(self, move: Move, policy: np.ndarray, value: float):
        """Add a move to the game record.
        
        Args:
            move: The move made
            policy: Policy tensor for the move
            value: Value prediction for the move
        """
        self.moves.append(move)
        self.policies.append(torch.tensor(policy, dtype=torch.float32))
        self.values.append(float(value))
    
    def add_state(self, state: chess.Board):
        """Add a board state to the game record.
        
        Args:
            state: The board state
        """
        self.states.append(state.copy())
    
    def set_result(self, result: GameResult):
        """Set the game result.
        
        Args:
            result: The game result
        """
        self.result = result
        
        # Adjust values based on game result
        final_value = 1.0 if result == GameResult.WHITE_WINS else -1.0 if result == GameResult.BLACK_WINS else 0.0
        self.values = [final_value * (-1) ** (i % 2) for i in range(len(self.values))]

def play_game(mcts: MCTS, max_moves: int = 200) -> GameRecord:
    """Play a single game of self-play using MCTS.
    
    Args:
        mcts: MCTS instance to use for move selection
        max_moves: Maximum number of moves before declaring a draw
        
    Returns:
        GameRecord containing the game history
    """
    board = chess.Board()
    game_record = GameRecord()
    move_count = 0
    
    while not board.is_checkmate() and move_count < max_moves:
        # Record current state
        game_record.add_state(board)
        
        # Get policy and value from MCTS
        policy, value = mcts.get_action_policy(board)
        
        # Convert policy to numpy array if it's a dict
        if isinstance(policy, dict):
            policy_array = np.zeros(5312, dtype=np.float32)  # Total number of possible moves
            for move, prob in policy.items():
                # Get move index from Move object
                move_idx = move.to_index(board)
                if 0 <= move_idx < 5312:
                    policy_array[move_idx] = float(prob)
                else:
                    logger.warning(f"Invalid move index {move_idx} for move {move}")
            policy = policy_array
        
        # Get legal moves and create mask
        legal_moves = board.legal_moves
        legal_mask = np.zeros(5312, dtype=np.float32)
        for move in legal_moves:
            # Calculate correct move index including promotions
            move_idx = move.from_square * 64 + move.to_square
            if move.promotion:
                move_idx += 4096 + (move.promotion - 2) * 64
            if 0 <= move_idx < 5312:
                legal_mask[move_idx] = 1.0
            else:
                logger.warning(f"Invalid move index {move_idx} for move {move}")
        
        # Apply mask to policy
        masked_policy = policy * legal_mask
        
        # Normalize policy
        policy_sum = masked_policy.sum()
        if policy_sum > 0:
            masked_policy /= policy_sum
        
        # Select move using temperature
        if move_count < 10:  # Use temperature for first 10 moves
            temperature = 1.0
        else:
            temperature = 0.0  # Deterministic after first 10 moves
        
        if temperature > 0:
            # Apply temperature
            masked_policy = masked_policy ** (1.0 / temperature)
            masked_policy /= masked_policy.sum()
            # Sample move
            move_idx = np.random.choice(5312, p=masked_policy)
        else:
            # Take best move
            move_idx = np.argmax(masked_policy)
        
        # Find the actual move corresponding to this index
        selected_move = None
        for move in legal_moves:
            if move.to_square == move_idx:
                selected_move = move
                break
        
        if selected_move is None:
            raise ValueError(f"Could not find move for index {move_idx}")
        
        # Record move before making it
        game_record.add_move(selected_move, masked_policy, value)
        
        # Make move
        board.push(selected_move)
        move_count += 1
    
    # Record final state
    game_record.add_state(board)
    
    # Set game result
    if move_count >= max_moves:
        game_record.set_result(GameResult.DRAW)
    else:
        game_record.set_result(board.result())
    
    return game_record

def generate_self_play_games(config: SelfPlayConfig) -> List[GameRecord]:
    """Generate self-play games.
    
    Args:
        config: Self-play configuration
        
    Returns:
        List of GameRecord objects containing the generated games
    """
    if config.mcts is None:
        raise ValueError("MCTS instance must be provided in config")
    
    games = []
    total_moves = 0
    start_time = time.time()
    
    # Use tqdm for progress bar if enabled
    game_range = tqdm(range(config.num_games), desc="Generating self-play games") if config.use_tqdm else range(config.num_games)
    
    for game_idx in game_range:
        try:
            # Generate single game
            game = play_game(config.mcts, config.max_moves)
            games.append(game)
            
            # Update statistics
            total_moves += game.num_moves
            avg_moves = total_moves / (game_idx + 1)
            elapsed = time.time() - start_time
            games_per_sec = (game_idx + 1) / elapsed
            moves_per_sec = total_moves / elapsed
            
            # Log progress
            progress_msg = (
                f"Game {game_idx + 1}/{config.num_games} | "
                f"Moves: {game.num_moves} | "
                f"Result: {game.result.name} | "
                f"Avg moves: {avg_moves:.1f} | "
                f"Speed: {games_per_sec:.1f} games/s, {moves_per_sec:.1f} moves/s"
            )
            
            if config.use_tqdm:
                tqdm.write(progress_msg)
            else:
                logger.info(progress_msg)
            
        except Exception as e:
            logger.error(f"Error generating game {game_idx + 1}: {e}")
            raise
    
    # Log final statistics
    logger.info(
        f"Generated {len(games)} games in {time.time() - start_time:.1f}s | "
        f"Total moves: {total_moves} | "
        f"Avg moves per game: {total_moves/len(games):.1f}"
    )
    
    return games 