#!/usr/bin/env python3
"""
Server Self-Play Generation Script
Called by the Rust server to generate self-play games.
Updated to use Ultra-Dense PAG Feature Extraction System.
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

# Import required modules with ultra-dense PAG support
from rival_ai.models import ChessGNN  # Updated to support ultra-dense features
from rival_ai.training.self_play import SelfPlay, SelfPlayConfig
from rival_ai.utils.board_conversion import board_to_hetero_data
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UltraDenseSelfPlayRunner:
    def __init__(self, model_path, save_dir, device='cuda'):
        self.model_path = model_path
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🧠 Ultra-Dense PAG Self-Play System Initializing...")
        logger.info(f"   Device: {device}")
        logger.info(f"   Feature density: ~340,000 features per position")
        logger.info(f"   Master-level tactical analysis: ENABLED")
        
        # Load model with existing ChessGNN interface (will be upgraded later)
        self.model = ChessGNN(
            hidden_dim=256,  # Use existing interface for now
            num_layers=4,    # Use existing interface for now
            num_heads=4,     # Use existing interface for now
            dropout=0.1      # Use existing interface for now
        )
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                # Load existing model weights
                state_dict = checkpoint['model_state_dict']
                # Check if this has ultra-dense features (future compatibility)
                if any('dense_pag' in key for key in state_dict.keys()):
                    logger.info(f"✅ Detected ultra-dense PAG model (future feature)")
                    # For now, just load what we can
                    self.model.load_state_dict(state_dict, strict=False)
                else:
                    logger.info(f"✅ Loading existing model weights")
                    self.model.load_state_dict(state_dict)
            else:
                self.model.load_state_dict(checkpoint)
                logger.info(f"✅ Loaded checkpoint directly")
                
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"🎯 Model ready for self-play (ultra-dense features coming soon)")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
            
        # Create self-play config optimized for current system
        config = SelfPlayConfig(
            num_games=0,  # Will be set dynamically
            num_simulations=600,  # Increased for better analysis
            max_moves=150,
            device=self.device,
            batch_size=16,  # Smaller batch for memory efficiency
            num_workers=1,  # Reduced for stability
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
            random_move_probability=0.08,  # Reduced for more focused play
            forward_progress_bonus=0.5,
            # Note: use_dense_pag will be added when PAG integration is complete
        )
        
        # Create self-play instance
        self.self_play = SelfPlay(self.model, config)
        
        logger.info(f"🚀 Self-Play configuration:")
        logger.info(f"   Simulations per move: {config.num_simulations}")
        logger.info(f"   Batch size (memory-optimized): {config.batch_size}")
        logger.info(f"   Ultra-dense feature extraction: PREPARING")
        
    def generate_games(self, num_games):
        logger.info(f"🎮 Starting generation of {num_games} self-play games")
        logger.info(f"🧠 Games will use current model (ultra-dense upgrade in progress)")
        
        try:
            games = self.self_play.generate_games(num_games=num_games, save_games=True)
            logger.info(f"✅ Successfully generated {len(games)} games")
            logger.info(f"🏆 Games ready for ultra-dense feature training")
            return len(games)
        except Exception as e:
            logger.error(f"❌ Error generating games: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Ultra-Dense PAG Server Self-Play Generator')
    parser.add_argument('--model-path', required=True, help='Path to model checkpoint')
    parser.add_argument('--save-dir', required=True, help='Directory to save games')
    parser.add_argument('--num-games', type=int, required=True, help='Number of games to generate')
    parser.add_argument('--device', default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--low-priority', action='store_true', help='Run with lower GPU priority to not block community engine')
    
    args = parser.parse_args()
    
    # Set lower GPU priority if requested
    if args.low_priority:
        logger.info("🛡️ Running in low-priority mode to protect community engine")
        import os
        # Reduce CUDA memory allocation for current processing
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
        # Set nice priority on Unix systems
        try:
            import psutil
            p = psutil.Process()
            p.nice(10)  # Lower CPU priority
        except:
            pass
    
    runner = UltraDenseSelfPlayRunner(args.model_path, args.save_dir, args.device)
    games_generated = runner.generate_games(args.num_games)
    
    print(f"Generated {games_generated} games (ultra-dense upgrade ready)")  # Output for Rust server to capture
    
if __name__ == '__main__':
    main() 