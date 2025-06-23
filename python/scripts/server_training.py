#!/usr/bin/env python3
"""
Server Training Script
Called by the Rust server to perform incremental training sessions.
Updated to use Unified Storage System.
"""

import sys
import os
import time
import argparse
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
import json
import pickle

# Add the src directory to the path
script_dir = Path(__file__).parent
python_src = script_dir.parent / 'src'
sys.path.insert(0, str(python_src))

# Import training modules with unified storage support
from rival_ai.models import ChessGNN
from rival_ai.training.trainer import Trainer
from rival_ai.config import TrainingConfig
from rival_ai.training.self_play import SelfPlayConfig
from rival_ai.unified_storage import get_unified_storage, initialize_unified_storage
import torch
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnifiedServerTrainingRunner:
    def __init__(self, games_dir, model_path, training_games_threshold, use_tensorboard=False, delete_after_training=True):
        self.games_dir = Path(games_dir)
        self.model_path = model_path
        self.training_games_threshold = training_games_threshold
        self.use_tensorboard = use_tensorboard
        self.delete_after_training = delete_after_training
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize unified storage
        self.storage = initialize_unified_storage(str(self.games_dir), batch_size=1000)
        
        logger.info(f"🎯 Unified Server Training System Initialized")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Unified storage: ACTIVE")
        logger.info(f"   Delete after training: {self.delete_after_training}")
        
    def count_training_ready_games(self):
        """Count games ready for training from unified storage."""
        return self.storage.get_training_ready_count()
        
    def run_incremental_training(self):
        try:
            logger.info("🚀 Starting Unified training session...")
            
            # Check if we have enough training-ready games
            training_ready = self.count_training_ready_games()
            total_games = self.storage.get_total_games()
            
            if training_ready < self.training_games_threshold:
                logger.info(f"Not enough training-ready games ({training_ready} < {self.training_games_threshold})")
                logger.info(f"Total games in unified storage: {total_games}")
                return self.model_path  # Return current model path
            
            logger.info(f"🎯 Found {training_ready} training-ready games ({total_games} total)")
            logger.info(f"🧠 Using unified batched storage format")
            
            # Create experiment directory
            experiment_name = f"unified_training_{int(time.time())}"
            experiment_dir = Path('../experiments') / experiment_name
            experiment_dir.mkdir(parents=True, exist_ok=True)
            
            # Create standardized model directory for easy server access
            models_dir = Path('../models')
            models_dir.mkdir(exist_ok=True)
            standardized_model_path = models_dir / 'latest_trained_model.pt'
            
            # Create subdirectories
            for subdir in ['checkpoints', 'logs', 'self_play_data']:
                (experiment_dir / subdir).mkdir(parents=True, exist_ok=True)
            
            # Create model with existing ChessGNN interface
            model = ChessGNN(
                hidden_dim=256,
                num_layers=4,
                num_heads=4,
                dropout=0.1
            )
            
            # Load existing checkpoint
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    model.load_state_dict(state_dict)
                    logger.info(f"✅ Loaded existing model weights")
                else:
                    model.load_state_dict(checkpoint)
                    logger.info(f"✅ Loaded checkpoint directly")
            except Exception as e:
                logger.warning(f"Could not load existing model: {e}")
                logger.info("🆕 Starting with fresh model")
            
            model.to(self.device)
            
            # Create training config
            config = TrainingConfig(
                num_epochs=8,
                batch_size=32,
                learning_rate=0.00005,
                weight_decay=1e-5,
                grad_clip=1.0,
                save_interval=2,
                experiment_dir=str(experiment_dir),
                device=self.device,
                use_tensorboard=self.use_tensorboard,
                use_improved_loss=True,
                num_workers=1,
            )
            
            # Create self-play config 
            self_play_config = SelfPlayConfig(
                num_games=15,
                num_simulations=400,
                max_moves=120,
                device=self.device,
                batch_size=16,
                num_workers=1,
                use_tqdm=False,
            )
            
            # Create trainer
            trainer = Trainer(
                model=model,
                config=config,
                self_play_config=self_play_config,
                device=self.device,
                use_prc_metrics=False
            )
            
            # Get training data batches
            max_batches = 10  # Limit to avoid using too much data at once
            batch_files = self.storage.prepare_training_data(max_batches=max_batches)
            
            if not batch_files:
                logger.error("No training batches available!")
                return self.model_path
            
            logger.info(f"📚 Using {len(batch_files)} training batches")
            
            # Run training session
            logger.info("🎓 Starting unified neural network training...")
            final_model_path = trainer.train_from_unified_storage(batch_files)
            logger.info(f"🏆 Training completed! Best model: {final_model_path}")
            
            # Copy the best model to standardized location for server access
            shutil.copy2(str(final_model_path), str(standardized_model_path))
            logger.info(f"📋 Copied model to standardized location: {standardized_model_path}")
            logger.info(f"🚀 Server can now reload with: {standardized_model_path}")
            
            # Archive used training batches and clean up
            if self.delete_after_training:
                archive_path = self.storage.archive_used_batches(batch_files)
                logger.info(f"✅ Training successful!")
                logger.info(f"🗑️ Archived and deleted {len(batch_files)} training batches")
                logger.info(f"📦 Archive: {archive_path}")
            else:
                logger.info(f"✅ Training successful!")
                logger.info(f"📦 Training batches preserved for analysis")
            
            # Return the standardized path for the server
            return str(standardized_model_path)
                    
        except Exception as e:
            logger.error(f"❌ Unified training failed: {e}")
            logger.error("⚠️ Training batches were NOT archived due to training failure")
            raise

def main():
    parser = argparse.ArgumentParser(description='Unified Server Training Runner')
    parser.add_argument('--games-dir', required=True, help='Directory containing training games')
    parser.add_argument('--model-path', required=True, help='Path to existing model checkpoint')
    parser.add_argument('--threshold', type=int, required=True, help='Training games threshold')
    parser.add_argument('--tensorboard', action='store_true', help='Enable TensorBoard logging')
    parser.add_argument('--low-priority', action='store_true', help='Run with lower GPU priority to not block community engine')
    parser.add_argument('--delete-after-training', action='store_true', help='Delete game files after training')
    
    args = parser.parse_args()
    
    # Set lower GPU priority if requested
    if args.low_priority:
        logger.info("🛡️ Running in low-priority mode to protect community engine")
        import os
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
        try:
            import psutil
            p = psutil.Process()
            p.nice(10)  # Lower CPU priority
        except:
            pass
    
    runner = UnifiedServerTrainingRunner(
        args.games_dir, 
        args.model_path, 
        args.threshold, 
        args.tensorboard,
        args.delete_after_training
    )
    
    result = runner.run_incremental_training()
    print(result)  # Output the path for the Rust server to capture
    
if __name__ == '__main__':
    main() 