"""
Self-play module using Rust MCTS engine for generating training data.
This replaces the problematic Python MCTS that was causing infinite loops.
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
from rival_ai.chess import GameResult
from rival_ai.training.training_types import GameRecord
from rival_ai.utils.board_conversion import board_to_hetero_data

logger = logging.getLogger(__name__)

@dataclass
class RustMCTSSelfPlayConfig:
    """Configuration for Rust MCTS self-play."""
    num_games: int = 10
    num_simulations: int = 500  # Reduced for faster generation
    max_moves: int = 150  # Reduced to encourage more decisive games
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 256
    num_workers: int = 4
    prefetch_factor: int = 2
    save_dir: Optional[str] = None
    use_tqdm: bool = True
    
    # MCTS parameters for Rust engine
    temperature: float = 1.0
    c_puct: float = 1.25
    max_time_per_move: float = 2.0  # Max 2 seconds per move for faster games
    
    # Aggressive play settings
    opening_moves: int = 15
    midgame_moves: int = 50
    force_capture_after_moves: int = 12
    force_attack_after_moves: int = 20
    force_win_attempt_after_moves: int = 35

class RustMCTSSelfPlay:
    """Self-play using Rust MCTS engine."""
    
    def __init__(self, model: ChessGNN, config: RustMCTSSelfPlayConfig, experiment_dir: Optional[Path] = None):
        """Initialize Rust MCTS self-play.
        
        Args:
            model: The GNN model to use
            config: Self-play configuration
            experiment_dir: Optional experiment directory
        """
        self.model = model.to(config.device)
        self.config = config
        self.device = config.device
        
        # Set up save directory
        if experiment_dir is not None:
            self.save_dir = experiment_dir / 'self_play_data'
        else:
            self.save_dir = Path(config.save_dir) if config.save_dir else Path('self_play_data')
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Rust MCTS engine
        self.mcts_engine = None
        self._initialize_rust_mcts()
        
        # Initialize metrics
        self.metrics = defaultdict(list)
        self.game_counter = 0
    
    def _initialize_rust_mcts(self):
        """Initialize the Rust MCTS engine with our model."""
        try:
            import rival_ai_engine
            
            # Create a model wrapper that the Rust engine can use
            class ModelWrapper:
                def __init__(self, pytorch_model, device):
                    self.pytorch_model = pytorch_model
                    self.device = device
                
                def predict_with_board(self, fen_string):
                    """Predict policy and value for a board position."""
                    try:
                        board = chess.Board(fen_string)
                        data = board_to_hetero_data(board)
                        data = data.to(self.device)
                        
                        with torch.no_grad():
                            policy, value = self.pytorch_model(data)
                            policy = policy.squeeze(0)
                            value = value.squeeze()
                        
                        # Convert to numpy for Rust interface
                        policy_np = policy.detach().cpu().numpy().astype(np.float32)
                        value_np = float(value.detach().cpu().numpy())
                        
                        return policy_np.tolist(), value_np
                    except Exception as e:
                        logger.error(f"Model prediction failed: {e}")
                        # Return uniform policy and neutral value as fallback
                        uniform_policy = [1.0/5312] * 5312
                        return uniform_policy, 0.0
            
            # Create model wrapper
            model_wrapper = ModelWrapper(self.model, self.device)
            
            # Initialize Rust MCTS engine
            self.mcts_engine = rival_ai_engine.PyMCTSEngine(model_wrapper)
            
            # Configure the engine
            self.mcts_engine.configure(
                num_simulations=self.config.num_simulations,
                temperature=self.config.temperature,
                c_puct=self.config.c_puct
            )
            
            logger.info(f"✅ Rust MCTS engine initialized with {self.config.num_simulations} simulations")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Rust MCTS engine: {e}")
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Cannot initialize Rust MCTS: {e}")
    
    def play_game(self) -> GameRecord:
        """Play a single self-play game using Rust MCTS."""
        # Reset the engine for a new game
        self.mcts_engine.reset()
        
        board = chess.Board()
        game_record = GameRecord()
        move_count = 0
        
        logger.info(f"🎮 Starting game {self.game_counter + 1} with Rust MCTS")
        
        while not board.is_game_over() and move_count < self.config.max_moves:
            try:
                # Record current state
                game_record.add_state(board.copy())
                
                # Get move from Rust MCTS engine
                fen = board.fen()
                start_time = time.time()
                
                # Use Rust MCTS to find the best move
                move_str = self.mcts_engine.search_move(
                    fen, 
                    max_time_seconds=self.config.max_time_per_move
                )
                
                search_time = time.time() - start_time
                
                # Parse the move
                try:
                    selected_move = chess.Move.from_uci(move_str)
                    if selected_move not in board.legal_moves:
                        logger.warning(f"Illegal move returned by Rust MCTS: {move_str}")
                        # Fallback to random legal move
                        selected_move = random.choice(list(board.legal_moves))
                except ValueError:
                    logger.warning(f"Invalid move format from Rust MCTS: {move_str}")
                    # Fallback to random legal move
                    selected_move = random.choice(list(board.legal_moves))
                
                # Get policy and value for training (from our model directly)
                try:
                    data = board_to_hetero_data(board)
                    data = data.to(self.device)
                    with torch.no_grad():
                        policy, value = self.model(data)
                        policy = policy.squeeze(0)
                        value = value.squeeze()
                except Exception as e:
                    logger.warning(f"Failed to get model policy/value: {e}")
                    # Create dummy policy and value
                    policy = torch.zeros(5312, dtype=torch.float32, device=self.device)
                    value = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                
                # Record the move with policy and value
                game_record.add_move(
                    move=selected_move,
                    policy=policy,
                    value=float(value)
                )
                
                # Make the move
                board.push(selected_move)
                move_count += 1
                
                # Log progress
                if move_count % 10 == 0:
                    logger.info(f"🎯 Game {self.game_counter + 1}, Move {move_count}: {selected_move} "
                               f"(search time: {search_time:.3f}s)")
                
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
        
        # Update metrics
        self.metrics['game_length'].append(move_count)
        self.game_counter += 1
        
        logger.info(f"✅ Game {self.game_counter} completed: {game_record.result.name} after {move_count} moves")
        
        return game_record
    
    def generate_games(self, num_games: Optional[int] = None, epoch: Optional[int] = None, save_games: bool = True) -> List[GameRecord]:
        """Generate self-play games using Rust MCTS.
        
        Args:
            num_games: Number of games to generate
            epoch: Training epoch number
            save_games: Whether to save games to disk
            
        Returns:
            List of game records
        """
        num_games = num_games or self.config.num_games
        games = []
        
        logger.info(f"🚀 Starting Rust MCTS self-play generation: {num_games} games")
        
        # Use tqdm if configured
        game_iter = tqdm(range(num_games), desc="Rust MCTS Self-Play") if self.config.use_tqdm else range(num_games)
        
        for i in game_iter:
            try:
                # Generate game using Rust MCTS
                game = self.play_game()
                games.append(game)
                
                # Save games in batches to reduce memory usage
                if save_games and len(games) >= 5:
                    self._save_games(games, epoch)
                    games = []  # Clear saved games from memory
                
            except Exception as e:
                logger.error(f"Error generating game {i}: {e}")
                logger.error(traceback.format_exc())
                continue
        
        # Save any remaining games
        if save_games and games:
            self._save_games(games, epoch)
        
        # Log final statistics
        avg_length = np.mean(self.metrics['game_length']) if self.metrics['game_length'] else 0
        logger.info(f"🏁 Rust MCTS self-play completed: {len(games)} games, avg length: {avg_length:.1f} moves")
        
        return games
    
    def _save_games(self, games: List[GameRecord], epoch: Optional[int]) -> str:
        """Save generated games to disk."""
        if not games:
            return None
        
        # Create filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        epoch_str = f"epoch_{epoch}_" if epoch is not None else ""
        filename = f"rust_mcts_games_{epoch_str}{timestamp}.pkl"
        filepath = self.save_dir / filename
        
        # Save games
        with open(filepath, 'wb') as f:
            pickle.dump(games, f)
        
        logger.info(f"💾 Saved {len(games)} games to {filepath}")
        return str(filepath)
    
    def create_dataloader(self, games: List[GameRecord]):
        """Create a dataloader from generated games."""
        from rival_ai.data import PAGDataset, PAGDataLoader
        
        dataset = PAGDataset.from_game_records(games)
        return PAGDataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )

# Convenience function for easy integration
def generate_rust_mcts_games(model: ChessGNN, num_games: int = 10, max_moves: int = 150, 
                            num_simulations: int = 500, save_dir: Optional[str] = None) -> List[GameRecord]:
    """Generate self-play games using Rust MCTS.
    
    Args:
        model: The neural network model
        num_games: Number of games to generate
        max_moves: Maximum moves per game
        num_simulations: MCTS simulations per move
        save_dir: Directory to save games
        
    Returns:
        List of game records
    """
    config = RustMCTSSelfPlayConfig(
        num_games=num_games,
        max_moves=max_moves,
        num_simulations=num_simulations,
        save_dir=save_dir
    )
    
    rust_self_play = RustMCTSSelfPlay(model, config)
    return rust_self_play.generate_games(save_games=save_dir is not None) 