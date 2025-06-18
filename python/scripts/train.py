#!/usr/bin/env python3
"""
Training script for RivalAI chess engine.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from rival_ai.models import ChessGNN
from rival_ai.training import Trainer, PolicyValueLoss
from rival_ai.training.checkpoint import save_checkpoint, load_checkpoint
from rival_ai.utils.logging import setup_logging
from rival_ai.data.dataset import ChessDataset, create_dataloader
from rival_ai.training.self_play import SelfPlayGenerator, SelfPlayConfig
from rival_ai.config import TrainingConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_experiment_dir(experiment_name: str) -> str:
    """Create and return the path to the experiment directory.
    
    Args:
        experiment_name: Base name for the experiment
        
    Returns:
        str: Path to the created experiment directory
    """
    # Create base experiments directory
    base_dir = Path('experiments')
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Create experiment directory (without timestamp)
    experiment_dir = base_dir / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped run directory within experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = experiment_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories within the run directory
    for subdir in ['self_play_data', 'checkpoints', 'logs']:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    return str(run_dir)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train the chess AI model')
    
    # Experiment settings
    parser.add_argument('--experiment-name', type=str, required=True,
                      help='Name of the experiment')
    parser.add_argument('--num-epochs', type=int, default=100,
                      help='Number of training epochs')
    parser.add_argument('--save-interval', type=int, default=5,
                      help='Save model every N epochs')
    
    # Training settings
    parser.add_argument('--batch-size', type=int, default=64,
                      help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                      help='Weight decay for regularization')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                      help='Gradient clipping value')
    
    # Model settings
    parser.add_argument('--hidden-dim', type=int, default=256,
                      help='Hidden dimension size')
    parser.add_argument('--num-layers', type=int, default=4,
                      help='Number of GNN layers')
    parser.add_argument('--num-heads', type=int, default=4,
                      help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                      help='Dropout rate')
    
    # Self-play settings
    parser.add_argument('--num-games', type=int, default=1000,
                      help='Number of self-play games per epoch')
    parser.add_argument('--num-simulations', type=int, default=500,
                      help='Number of MCTS simulations per move')
    parser.add_argument('--num-parallel-games', type=int, default=10,
                      help='Number of games to generate in parallel')
    parser.add_argument('--max-moves', type=int, default=200,
                      help='Maximum number of moves per game')
    
    # Hardware settings
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use (cuda/cpu)')
    parser.add_argument('--num-workers', type=int, default=4,
                      help='Number of data loading workers')
    
    # Logging settings
    parser.add_argument('--tensorboard', action='store_true',
                      help='Enable TensorBoard logging')
    parser.add_argument('--profile', action='store_true',
                      help='Enable PyTorch profiling for one batch')
    
    # Checkpoint settings
    parser.add_argument('--checkpoint', type=str,
                      help='Path to checkpoint file to resume training')
    parser.add_argument('--resume', action='store_true',
                      help='Automatically resume from the latest checkpoint')
    parser.add_argument('--resume-epoch', type=int,
                      help='Epoch to resume training from')
    parser.add_argument('--no-auto-resume', action='store_true',
                      help='Disable automatic resumption from latest checkpoint')
    
    return parser.parse_args()

def find_latest_experiment_checkpoint(experiment_name: str) -> Optional[Tuple[str, int]]:
    """Find the latest checkpoint for an experiment using standardized paths.
    
    Args:
        experiment_name: Name of the experiment (e.g. 'rival_ai_v1_Alice')
        
    Returns:
        Tuple of (checkpoint_path, epoch) if found, None otherwise
    """
    # Look in the experiments directory structure
    experiment_base_dir = Path('experiments') / experiment_name
    if not experiment_base_dir.exists():
        return None
    
    # Find all run directories
    run_dirs = [d for d in experiment_base_dir.iterdir() if d.is_dir() and d.name.startswith('run_')]
    if not run_dirs:
        return None
    
    # Sort run directories by modification time (newest first)
    run_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    # Look for checkpoints in each run directory
    for run_dir in run_dirs:
        checkpoint_dir = run_dir / 'checkpoints'
        if not checkpoint_dir.exists():
            continue
            
        # Find all checkpoints in this run directory
        checkpoints = []
        for f in checkpoint_dir.glob('*.pt'):
            try:
                # Try to extract epoch from filename
                if 'epoch' in f.stem:
                    epoch = int(f.stem.split('epoch_')[-1])
                    checkpoints.append((f, epoch))
                elif 'checkpoint' in f.stem:
                    # Fallback: try to extract epoch from checkpoint filename
                    parts = f.stem.split('_')
                    for i, part in enumerate(parts):
                        if part.isdigit() and i > 0:
                            epoch = int(part)
                            checkpoints.append((f, epoch))
                            break
            except (ValueError, IndexError):
                continue
        
        if checkpoints:
            # Sort by epoch (highest first) and modification time
            latest_checkpoint = max(checkpoints, key=lambda x: (x[1], x[0].stat().st_mtime))
            return str(latest_checkpoint[0]), latest_checkpoint[1]
    
    return None

def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Set up initial logging (without epoch)
    setup_logging(args.experiment_name)
    
    # Check for latest checkpoint if not explicitly disabled
    checkpoint_path = args.checkpoint
    resume_epoch = args.resume_epoch
    
    # Handle --resume flag
    if args.resume and not checkpoint_path:
        latest = find_latest_experiment_checkpoint(args.experiment_name)
        if latest:
            checkpoint_path, epoch = latest
            logger.info(f"Found latest checkpoint at epoch {epoch}")
            resume_epoch = epoch
            
    # Create model
    logger.info("Creating model...")
    model = ChessGNN(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout
    ).to(torch.device(args.device))
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=args.learning_rate * 0.01
    )
    
    # Create loss functions
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()
    
    # Create training config
    config = TrainingConfig(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        save_interval=args.save_interval,
        experiment_dir=create_experiment_dir(args.experiment_name),
        device=args.device,
        use_tensorboard=args.tensorboard,
        checkpoint_path=checkpoint_path,
        resume_epoch=resume_epoch,
        num_workers=args.num_workers
    )
    
    # Create self-play config
    self_play_config = SelfPlayConfig(
        num_games=args.num_games,
        num_simulations=args.num_simulations,
        max_moves=args.max_moves,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=2,
        save_dir=os.path.join(create_experiment_dir(args.experiment_name), 'self_play_data'),
        use_tqdm=True,
        
        # Temperature settings - increased to encourage more variety
        opening_temperature=1.2,
        midgame_temperature=1.1,
        endgame_temperature=0.9,
        
        # Repetition prevention
        min_pieces_for_repetition_penalty=12,
        repetition_penalty=2.0,
        
        # Randomness
        random_move_probability=0.05,
        random_move_temperature=2.0,
        
        # Forward progress bonus
        forward_progress_bonus=0.1,
        
        # Draw prevention
        draw_penalty_scale=1.5,
        early_draw_penalty=2.0
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        self_play_config=self_play_config,
        device=args.device,
        use_prc_metrics=True
    )
    
    # Enable profiling if requested
    if args.profile:
        logger.info("Enabling PyTorch profiling for one batch...")
        profile_dir = Path(config.experiment_dir) / 'profile'
        profile_dir.mkdir(parents=True, exist_ok=True)
        profile_path = profile_dir / 'pytorch_trace.json'
        
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=1,
                warmup=1,
                active=3,
                repeat=1
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            trainer.train(setup_logging_fn=lambda epoch: setup_logging(args.experiment_name, epoch))
            prof.step()
        
        # Export to Chrome trace format
        prof.export_chrome_trace(str(profile_path))
        logger.info(f"Profiling complete. Profile saved to: {profile_path}")
        logger.info("To view the profile:")
        logger.info("1. Open Chrome/Edge")
        logger.info("2. Go to chrome://tracing")
        logger.info("3. Click 'Load' and select the pytorch_trace.json file")
    else:
        # Normal training
        trainer.train(setup_logging_fn=lambda epoch: setup_logging(args.experiment_name, epoch))

if __name__ == '__main__':
    main() 