#!/usr/bin/env python3
"""
Training script for RivalAI chess engine.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path('experiments') / f"{experiment_name}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return str(experiment_dir)

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
    parser.add_argument('--num-simulations', type=int, default=400,
                      help='Number of MCTS simulations per move')
    parser.add_argument('--num-parallel-games', type=int, default=10,
                      help='Number of games to generate in parallel')
    
    # Hardware settings
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use (cuda/cpu)')
    parser.add_argument('--num-workers', type=int, default=4,
                      help='Number of data loading workers')
    
    # Logging settings
    parser.add_argument('--tensorboard', action='store_true',
                      help='Enable TensorBoard logging')
    parser.add_argument('--profile', action='store_true',
                      help='Enable profiling for first 10 iterations')
    
    # Checkpoint settings
    parser.add_argument('--checkpoint', type=str,
                      help='Path to checkpoint file to resume training')
    parser.add_argument('--resume-epoch', type=int,
                      help='Epoch to resume training from')
    
    return parser.parse_args()

def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    setup_logging(args.experiment_name)
    
    # Create or load experiment directory
    if args.checkpoint:
        # If resuming, use the checkpoint's experiment directory
        checkpoint_dir = Path(args.checkpoint).parent
        experiment_dir = str(checkpoint_dir)
        logger.info(f"Resuming training from checkpoint in: {experiment_dir}")
    else:
        # Create new experiment directory
        experiment_dir = create_experiment_dir(args.experiment_name)
        logger.info(f"Created new experiment directory: {experiment_dir}")
    
    # Set device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Create model
    logger.info("Creating model...")
    model = ChessGNN(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout
    ).to(device)
    
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
        eta_min=args.learning_rate * 0.1
    )
    
    # Create loss functions
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()
    
    # Create self-play config
    self_play_config = SelfPlayConfig(
        num_games=args.num_games,
        num_simulations=args.num_simulations,
        max_moves=100,
        dirichlet_alpha=0.3,
        dirichlet_weight=0.25,
        temperature=1.0,
        c_puct=1.0,
        save_dir=os.path.join(experiment_dir, 'self_play_data'),
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_tqdm=True,
        log_timing=True,
        num_parallel_games=args.num_parallel_games,
        prefetch_factor=2
    )
    
    # Create training config
    train_config = TrainingConfig(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        save_interval=args.save_interval,
        experiment_dir=experiment_dir,
        device=args.device,
        use_tensorboard=args.tensorboard,
        checkpoint_path=args.checkpoint,
        resume_epoch=args.resume_epoch,
        num_workers=args.num_workers
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        config=train_config,
        self_play_config=self_play_config,
        device=args.device
    )
    
    # Load checkpoint if specified
    if args.checkpoint:
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint)
    
    # Train model
    trainer.train()

if __name__ == '__main__':
    main() 