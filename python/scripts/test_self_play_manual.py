"""
Manual test script for self-play implementation.
This script runs a small number of self-play games and provides detailed output
about the process, including game statistics and visualization.
"""

import os
import sys
import logging
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from rival_ai.models import ChessGNN
from rival_ai.training.self_play import SelfPlay, SelfPlayConfig
from rival_ai.chess import GameResult, Board

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_model():
    """Create a test model."""
    return ChessGNN(
        hidden_channels=64,  # Smaller for testing
        num_layers=2,
        heads=2,
        dropout=0.1,
    )

def print_game_statistics(games):
    """Print detailed statistics about the generated games."""
    num_games = len(games)
    total_moves = sum(game.num_moves for game in games)
    avg_moves = total_moves / num_games
    
    # Count results
    results = {
        GameResult.WHITE_WINS: 0,
        GameResult.BLACK_WINS: 0,
        GameResult.DRAW: 0
    }
    for game in games:
        results[game.result] += 1
    
    # Print statistics
    logger.info("\nGame Statistics:")
    logger.info(f"Total games: {num_games}")
    logger.info(f"Average moves per game: {avg_moves:.2f}")
    logger.info(f"White wins: {results[GameResult.WHITE_WINS]} ({results[GameResult.WHITE_WINS]/num_games*100:.1f}%)")
    logger.info(f"Black wins: {results[GameResult.BLACK_WINS]} ({results[GameResult.BLACK_WINS]/num_games*100:.1f}%)")
    logger.info(f"Draws: {results[GameResult.DRAW]} ({results[GameResult.DRAW]/num_games*100:.1f}%)")
    
    # Print move distribution
    move_counts = [game.num_moves for game in games]
    logger.info(f"\nMove count distribution:")
    logger.info(f"Min moves: {min(move_counts)}")
    logger.info(f"Max moves: {max(move_counts)}")
    logger.info(f"Median moves: {np.median(move_counts):.1f}")

def main():
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("test_output") / f"self_play_test_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model and config
    model = create_model()
    config = SelfPlayConfig(
        num_games=1,  # Just one game for testing
        num_simulations=10,  # Minimal simulations
        temperature=1.0,
        dirichlet_alpha=0.3,
        dirichlet_weight=0.25,
        max_moves=10,  # Very short game for testing
        save_dir=str(output_dir),
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=1,  # Minimal batch size
        num_workers=0,
        use_tqdm=True,
    )
    
    logger.info(f"Starting self-play test with {config.num_games} games")
    logger.info(f"Using device: {config.device}")
    logger.info(f"Output directory: {output_dir}")
    
    # Create self-play instance
    self_play = SelfPlay(model, config)
    
    try:
        # Generate games
        logger.info("Generating games...")
        games = self_play.generate_games()
        
        # Print statistics
        print_game_statistics(games)
        
        # Save games
        logger.info("\nSaving games...")
        self_play.save_games(games, epoch=0)
        
        # Test dataloader
        logger.info("\nTesting dataloader...")
        dataloader = self_play.create_dataloader(games)
        batch = next(iter(dataloader))
        
        logger.info(f"Batch size: {batch.num_graphs}")
        logger.info(f"Number of edges: {batch.edge_index.shape[1]}")
        logger.info(f"Policy target shape: {batch.policy_target.shape}")  # Should be [batch_size, num_moves]
        logger.info(f"Value target shape: {batch.value_target.shape}")    # Should be [batch_size, 1]
        logger.info(f"Node features shape: {batch.x.shape}")
        logger.info(f"Edge features shape: {batch.edge_attr.shape}")
        
        # Verify targets
        assert batch.policy_target.shape[0] == batch.num_graphs, f"Policy target batch size mismatch: {batch.policy_target.shape[0]} != {batch.num_graphs}"
        assert batch.value_target.shape[0] == batch.num_graphs, f"Value target batch size mismatch: {batch.value_target.shape[0]} != {batch.num_graphs}"
        assert batch.policy_target.shape[1] == 33, f"Policy target should have 33 moves, got {batch.policy_target.shape[1]}"
        assert torch.all(batch.policy_target >= 0) and torch.all(batch.policy_target <= 1), "Policy targets should be probabilities"
        assert torch.all(batch.value_target >= -1) and torch.all(batch.value_target <= 1), "Value targets should be in [-1, 1]"
        
        # Verify policy probabilities sum to 1
        policy_sums = batch.policy_target.sum(dim=1)
        assert torch.allclose(policy_sums, torch.ones_like(policy_sums)), "Policy probabilities should sum to 1"
        
        logger.info("All tests passed!")
        
    except Exception as e:
        logger.error(f"Error during self-play test: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 