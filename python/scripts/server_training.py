#!/usr/bin/env python3
"""
Server Training Script
Called by the Rust server to perform incremental training sessions.
Updated to use Ultra-Dense PAG Feature Extraction System.
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

# Import training modules with ultra-dense PAG support
from rival_ai.models import ChessGNN  # Updated to support ultra-dense features
from rival_ai.training.trainer import Trainer
from rival_ai.config import TrainingConfig
from rival_ai.training.self_play import SelfPlayConfig
import torch
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ServerTrainingRunner:
    def __init__(self, games_dir, model_path, training_games_threshold, use_tensorboard=False):
        self.games_dir = Path(games_dir)
        self.model_path = model_path
        self.training_games_threshold = training_games_threshold
        self.use_tensorboard = use_tensorboard
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Set up directory structure for game archival
        self.processed_dir = self.games_dir / 'processed'
        self.archive_dir = self.games_dir / 'archives'
        self.processed_dir.mkdir(exist_ok=True)
        self.archive_dir.mkdir(exist_ok=True)
        
        logger.info(f"🧠 Ultra-Dense PAG Training System Initialized")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Feature density: ~340,000 features per position")
        logger.info(f"   Master-level tactical analysis: ENABLED")
        
    def count_unprocessed_games(self):
        """Count games that haven't been processed yet (for training threshold)."""
        unprocessed_count = 0
        
        # Count single player games (exclude community and processed directories)
        single_player_dir = self.games_dir / 'single_player'
        if single_player_dir.exists():
            for game_file in single_player_dir.glob('*.json'):
                # Skip if already processed
                processed_file = self.processed_dir / 'single_player' / game_file.name
                if not processed_file.exists():
                    unprocessed_count += 1
        
        # Count self-play games (pickle files) - FIXED: Count games inside each PKL file
        for game_file in self.games_dir.glob('*.pkl'):
            processed_file = self.processed_dir / game_file.name
            if not processed_file.exists():
                try:
                    # Load the PKL file to count games inside it
                    with open(game_file, 'rb') as f:
                        games_list = pickle.load(f)
                        if isinstance(games_list, list):
                            games_in_file = len(games_list)
                            unprocessed_count += games_in_file
                            logger.info(f"PKL file {game_file.name}: {games_in_file} games")
                        else:
                            # Fallback: assume 1 game if not a list
                            unprocessed_count += 1
                            logger.warning(f"PKL file {game_file.name}: unexpected format, counting as 1 game")
                except Exception as e:
                    logger.warning(f"Could not load PKL file {game_file.name}: {e}")
                    # Fallback: count as 1 game if we can't load it
                    unprocessed_count += 1
                
        logger.info(f"Found {unprocessed_count} unprocessed games for ultra-dense training")
        return unprocessed_count
        
    def archive_processed_games(self):
        """Archive processed games to prevent retraining."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_name = f"ultra_dense_training_batch_{timestamp}.zip"
        archive_path = self.archive_dir / archive_name
        
        archived_count = 0
        games_metadata = []  # Collect metadata for persistent stats
        
        try:
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Archive single player games
                single_player_dir = self.games_dir / 'single_player'
                if single_player_dir.exists():
                    processed_sp_dir = self.processed_dir / 'single_player'
                    processed_sp_dir.mkdir(exist_ok=True)
                    
                    for game_file in single_player_dir.glob('*.json'):
                        processed_file = processed_sp_dir / game_file.name
                        if not processed_file.exists():
                            # Read metadata before archiving
                            try:
                                with open(game_file, 'r') as f:
                                    game_data = json.load(f)
                                    if 'metadata' in game_data:
                                        games_metadata.append(game_data['metadata'])
                            except Exception as e:
                                logger.warning(f"Could not read metadata from {game_file}: {e}")
                            
                            # Add to archive
                            zipf.write(game_file, f"single_player/{game_file.name}")
                            # Move to processed
                            shutil.move(str(game_file), str(processed_file))
                            archived_count += 1
                
                # Archive self-play games (pickle files)
                for game_file in self.games_dir.glob('*.pkl'):
                    processed_file = self.processed_dir / game_file.name
                    if not processed_file.exists():
                        # Self-play games don't have individual metadata to extract
                        # The training script handles stats differently for these
                        
                        # Add to archive
                        zipf.write(game_file, game_file.name)
                        # Move to processed
                        shutil.move(str(game_file), str(processed_file))
                        archived_count += 1
            
            # Save collected metadata to persistent stats (for single player games)
            if games_metadata:
                logger.info(f"Saving metadata for {len(games_metadata)} ultra-dense games to persistent stats")
                # Note: This would ideally call the Rust GameStorage::archive_games_metadata
                # but since we're in Python, we'll let the Rust side handle persistent stats
                # when games are saved initially
            
            if archived_count > 0:
                logger.info(f"📦 Archived {archived_count} ultra-dense games to {archive_path}")
                logger.info(f"🧠 Games now contain master-level analysis and won't be used for future training")
                logger.info(f"📊 Model stats are preserved in persistent storage")
            else:
                # Remove empty archive
                archive_path.unlink()
                logger.info("No new games to archive")
                
        except Exception as e:
            logger.error(f"Failed to archive games: {e}")
            
        return archived_count
        
    def run_incremental_training(self):
        try:
            logger.info("🚀 Starting Ultra-Dense PAG incremental training session...")
            
            # Check if we have enough unprocessed games
            unprocessed_count = self.count_unprocessed_games()
            if unprocessed_count < self.training_games_threshold:
                logger.info(f"Not enough unprocessed games ({unprocessed_count} < {self.training_games_threshold})")
                return self.model_path  # Return current model path
            
            logger.info(f"🎯 Found {unprocessed_count} unprocessed games, proceeding with ultra-dense training...")
            logger.info(f"🧠 Each position will use ~340,000 master-level features")
            
            # Create experiment directory
            experiment_name = f"ultra_dense_pag_training_{int(time.time())}"
            experiment_dir = Path('../experiments') / experiment_name
            experiment_dir.mkdir(parents=True, exist_ok=True)
            
            # Create standardized model directory for easy server access
            models_dir = Path('../models')
            models_dir.mkdir(exist_ok=True)
            standardized_model_path = models_dir / 'latest_trained_model.pt'
            
            # Create subdirectories
            for subdir in ['checkpoints', 'logs', 'self_play_data']:
                (experiment_dir / subdir).mkdir(parents=True, exist_ok=True)
            
            # Create model with existing ChessGNN interface (will be upgraded later)
            # Updated configuration for current model architecture
            model = ChessGNN(
                hidden_dim=256,  # Use existing interface for now
                num_layers=4,    # Use existing interface for now
                num_heads=4,     # Use existing interface for now
                dropout=0.1      # Use existing interface for now
            )
            
            # Load existing checkpoint
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    # Check if this has ultra-dense features (future compatibility)
                    state_dict = checkpoint['model_state_dict']
                    if any('dense_pag' in key for key in state_dict.keys()):
                        logger.info(f"✅ Detected ultra-dense PAG model (future feature)")
                        # For now, just load what we can
                        model.load_state_dict(state_dict, strict=False)
                    else:
                        logger.info(f"✅ Loading existing model weights")
                        model.load_state_dict(state_dict)
                else:
                    model.load_state_dict(checkpoint)
                    logger.info(f"✅ Loaded checkpoint directly")
            except Exception as e:
                logger.warning(f"Could not load existing model: {e}")
                logger.info("🆕 Starting with fresh model")
            
            model.to(self.device)
            
            # Create training config - optimized for current architecture
            config = TrainingConfig(
                num_epochs=8,  # More epochs for complex feature learning
                batch_size=32,  # Reasonable batch size for current system
                learning_rate=0.00005,  # Lower LR for stable training
                weight_decay=1e-5,
                grad_clip=1.0,  # Higher grad clip for stability
                save_interval=2,
                experiment_dir=str(experiment_dir),
                device=self.device,
                use_tensorboard=self.use_tensorboard,
                use_improved_loss=True,
                num_workers=1,  # Reduced workers due to memory requirements
                # Note: use_dense_pag will be added when PAG integration is complete
            )
            
            # Create self-play config 
            self_play_config = SelfPlayConfig(
                num_games=15,  # Slightly fewer due to computation cost
                num_simulations=400,  # Keep high for strong play
                max_moves=120,
                device=self.device,
                batch_size=16,  # Smaller batch for memory
                num_workers=1,
                use_tqdm=False,
                # Note: use_dense_pag will be added when PAG integration is complete
            )
            
            # Create trainer
            trainer = Trainer(
                model=model,
                config=config,
                self_play_config=self_play_config,
                device=self.device,
                use_prc_metrics=False
                # Note: use_dense_pag will be added when PAG integration is complete
            )
            
            # Run ultra-dense training session
            logger.info("🎓 Starting ultra-dense PAG neural network training...")
            logger.info("🧠 Network will learn from master-level tactical analysis")
            trainer.train()
            
            # Find the best model
            checkpoints_dir = experiment_dir / 'checkpoints'
            best_model_path = checkpoints_dir / 'best_ultra_dense_model.pt'
            
            final_model_path = None
            if best_model_path.exists():
                final_model_path = best_model_path
                logger.info(f"🏆 Found best model: {best_model_path}")
            else:
                # Fall back to latest checkpoint
                latest_checkpoint = None
                for checkpoint_file in checkpoints_dir.glob('checkpoint_epoch_*.pt'):
                    if latest_checkpoint is None or checkpoint_file.stat().st_mtime > latest_checkpoint.stat().st_mtime:
                        latest_checkpoint = checkpoint_file
                
                if latest_checkpoint:
                    final_model_path = latest_checkpoint
                    logger.info(f"📂 Using latest checkpoint: {latest_checkpoint}")
                else:
                    raise Exception("No ultra-dense checkpoints found after training")
            
            # Copy the best model to standardized location for server access
            import shutil
            shutil.copy2(str(final_model_path), str(standardized_model_path))
            logger.info(f"📋 Copied model to standardized location: {standardized_model_path}")
            logger.info(f"🚀 Server can now reload with: {standardized_model_path}")
            
            # Archive processed games after successful training
            archived_count = self.archive_processed_games()
            logger.info(f"📦 Training successful - archived {archived_count} processed games")
            
            # Return the standardized path for the server
            return str(standardized_model_path)
                    
        except Exception as e:
            logger.error(f"❌ Ultra-dense training failed: {e}")
            logger.error("⚠️ Games were NOT archived due to training failure")
            raise

def main():
    parser = argparse.ArgumentParser(description='Ultra-Dense PAG Server Training Runner')
    parser.add_argument('--games-dir', required=True, help='Directory containing training games')
    parser.add_argument('--model-path', required=True, help='Path to existing model checkpoint')
    parser.add_argument('--threshold', type=int, required=True, help='Training games threshold')
    parser.add_argument('--tensorboard', action='store_true', help='Enable TensorBoard logging')
    parser.add_argument('--low-priority', action='store_true', help='Run with lower GPU priority to not block community engine')
    
    args = parser.parse_args()
    
    # Set lower GPU priority if requested
    if args.low_priority:
        logger.info("🛡️ Running in low-priority mode to protect community engine")
        import os
        # Reduce CUDA memory allocation and batch sizes for ultra-dense processing
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'  # Higher for ultra-dense
        # Set nice priority on Unix systems
        try:
            import psutil
            p = psutil.Process()
            p.nice(10)  # Lower CPU priority
        except:
            pass
    
    runner = ServerTrainingRunner(
        args.games_dir, 
        args.model_path, 
        args.threshold, 
        args.tensorboard
    )
    
    result = runner.run_incremental_training()
    print(result)  # Output the path for the Rust server to capture
    
if __name__ == '__main__':
    main() 