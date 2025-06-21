#!/usr/bin/env python3
"""
Server Self-Play Generation Script
Called by the Rust server to generate self-play games.
"""

import sys
import os
import argparse
from pathlib import Path
import logging

# Add the src directory to the path
script_dir = Path(__file__).parent
python_src = script_dir.parent / 'src'
sys.path.insert(0, str(python_src))

# Import required modules
from rival_ai.models import ChessGNN
from rival_ai.training.self_play import SelfPlay, SelfPlayConfig
from rival_ai.utils.board_conversion import board_to_hetero_data
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SelfPlayRunner:
    def __init__(self, model_path, save_dir, device='cuda'):
        self.model_path = model_path
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        self.model = ChessGNN(hidden_dim=256, num_layers=4, num_heads=4, dropout=0.1)
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Loaded model from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
            
        # Create self-play config
        config = SelfPlayConfig(
            num_games=0,  # Will be set dynamically
            num_simulations=400,
            max_moves=150,
            device=self.device,
            batch_size=32,
            num_workers=2,
            save_dir=str(self.save_dir),
            use_tqdm=False,
            c_puct=2.5,
            dirichlet_alpha=0.8,
            dirichlet_weight=0.4,
            temperature=1.5,
            opening_temperature=1.8,
            midgame_temperature=1.5,
            endgame_temperature=1.2,
            repetition_penalty=5.0,
            random_move_probability=0.1,
            forward_progress_bonus=0.5,
        )
        
        # Create self-play instance
        self.self_play = SelfPlay(self.model, config)
        
    def generate_games(self, num_games):
        logger.info(f"Starting generation of {num_games} self-play games")
        try:
            games = self.self_play.generate_games(num_games=num_games, save_games=True)
            logger.info(f"Successfully generated {len(games)} games")
            return len(games)
        except Exception as e:
            logger.error(f"Error generating games: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Server Self-Play Generator')
    parser.add_argument('--model-path', required=True, help='Path to model checkpoint')
    parser.add_argument('--save-dir', required=True, help='Directory to save games')
    parser.add_argument('--num-games', type=int, required=True, help='Number of games to generate')
    parser.add_argument('--device', default='cuda', help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    runner = SelfPlayRunner(args.model_path, args.save_dir, args.device)
    games_generated = runner.generate_games(args.num_games)
    
    print(f"Generated {games_generated} games")  # Output for Rust server to capture
    
if __name__ == '__main__':
    main() 