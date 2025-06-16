"""
Trainer module for training the chess model.
"""

import os
import time
import logging
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from rival_ai.config import TrainingConfig, SelfPlayConfig
from rival_ai.data.dataset import ChessDataset, create_dataloader

from rival_ai.models import ChessGNN as ChessModel
from rival_ai.training.loss import PolicyValueLoss
from rival_ai.training.visualizer import TrainingVisualizer
from rival_ai.mcts import MCTS
from rival_ai.training.self_play import SelfPlay, SelfPlayConfig, GameRecord
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class TrainerConfig:
    """Configuration for model training."""
    num_epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    save_interval: int = 10
    eval_interval: int = 5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 8
    use_tensorboard: bool = True
    experiment_name: str = "rival_ai_v1"
    checkpoint_dir: str = "checkpoints"

def train_epoch(
    model: ChessModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: Union[PolicyValueLoss, nn.Module],
    device: str,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    """Train the model for one epoch.
    
    Args:
        model: The model to train
        dataloader: DataLoader for training data
        optimizer: Optimizer for training
        criterion: Loss function
        device: Device to use for training
        grad_clip: Gradient clipping value
        
    Returns:
        Dictionary of training metrics
    """
    model.train()
    total_loss = 0
    loss_components = {}
    
    for batch in dataloader:
        # Move batch to device
        data = batch['data'].to(device)
        policy_target = batch['policy'].to(device)
        value_target = batch['value'].to(device)
        
        # Forward pass
        policy_pred, value_pred = model(data)
        
        # Compute loss
        if isinstance(criterion, PolicyValueLoss):
            loss, components = criterion(
                policy_pred=policy_pred,
                value_pred=value_pred,
                policy_target=policy_target,
                value_target=value_target,
                model=model
            )
        else:
            loss, components = criterion(
                policy_pred,
                value_pred,
                policy_target,
                value_target,
                model
            )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                grad_clip
            )
        
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        for k, v in components.items():
            loss_components[k] = loss_components.get(k, 0) + v
    
    # Average metrics
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    avg_components = {k: v / num_batches for k, v in loss_components.items()}
    
    return {'loss': avg_loss, **avg_components}

@dataclass
class Trainer:
    """Trainer for the chess model."""
    
    model: nn.Module  # Using nn.Module as the base type for the model
    config: TrainingConfig
    self_play_config: Optional[SelfPlayConfig] = None
    device: Optional[str] = None
    
    def __post_init__(self):
        """Initialize trainer after dataclass initialization."""
        # Set default self-play config if not provided
        if self.self_play_config is None:
            self.self_play_config = SelfPlayConfig()
            
        # Set default device if not provided
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Initialize learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.num_epochs,
            eta_min=self.config.learning_rate * 0.1
        )
        
        # Initialize loss function
        self.criterion = PolicyValueLoss()
        
        # Initialize tensorboard writer if enabled
        self.writer = None
        if self.config.use_tensorboard:
            log_dir = os.path.join('runs', os.path.basename(self.config.experiment_dir))
            self.writer = SummaryWriter(log_dir)
        
        # Initialize visualizer
        experiment_name = os.path.basename(self.config.experiment_dir)
        log_dir = os.path.join('runs', experiment_name)
        self.visualizer = TrainingVisualizer(
            log_dir=log_dir,
            experiment_name=experiment_name
        )
        
        # Initialize self-play manager (single instance for all epochs)
        self.self_play = SelfPlay(self.model, self.self_play_config)
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.patience_counter = 0
        
        # Load checkpoint if specified
        if self.config.checkpoint_path:
            self.load_checkpoint(self.config.checkpoint_path)
            if self.config.resume_epoch is not None:
                self.current_epoch = self.config.resume_epoch
    
    def train(self):
        """Train the model using self-play data."""
        logger.info("Starting training...")
        
        # Start from current_epoch and go until num_epochs
        while self.current_epoch < self.config.num_epochs:
            logger.info(f"Starting epoch {self.current_epoch + 1}/{self.config.num_epochs}")
            
            try:
                # Generate self-play games using the persistent self-play instance
                logger.info("Generating self-play games...")
                max_retries = 3  # Maximum number of retries for failed games
                all_games = []  # Accumulate all games for this epoch
                retry_count = 0
                
                # Generate all games for this epoch before training
                while len(all_games) < self.self_play_config.num_games and retry_count < max_retries:
                    try:
                        # Generate a batch of games
                        batch_size = min(self.self_play_config.num_parallel_games, self.self_play_config.num_games - len(all_games))  # Use configured parallel games
                        batch_games = self.self_play.generate_games(epoch=self.current_epoch, num_games=batch_size, save_games=False)  # Don't save individual batches
                        all_games.extend(batch_games)
                        retry_count = 0  # Reset retry count on success
                        logger.info(f"Generated batch of {len(batch_games)} games. Total games so far: {len(all_games)}/{self.self_play_config.num_games}")
                    except Exception as e:
                        retry_count += 1
                        logger.error(f"Error generating games (attempt {retry_count}/{max_retries}): {str(e)}")
                        if retry_count >= max_retries:
                            logger.error("Max retries reached, skipping this epoch")
                            raise  # Re-raise to be caught by outer try-except
                        time.sleep(1)  # Wait a bit before retrying
                
                if not all_games:
                    logger.error("Failed to generate any games in this epoch")
                    continue  # Don't increment epoch, try again
                
                logger.info(f"Generated total of {len(all_games)} games for epoch {self.current_epoch}")
                
                # Save all games for this epoch at once
                self.self_play._save_games(all_games, self.current_epoch)
                
                # Create dataset and dataloader from all games
                dataset = ChessDataset.from_game_records(all_games)
                dataloader = create_dataloader(
                    dataset,
                    batch_size=self.config.batch_size,
                    shuffle=True,
                    num_workers=self.config.num_workers,
                    pin_memory=True
                )
                
                # Train on the generated games
                logger.info("Training on self-play data...")
                metrics = self.train_epoch(dataloader)
                
                # Log metrics
                if self.writer is not None:
                    for name, value in metrics.items():
                        self.writer.add_scalar(f'train/{name}', value, self.current_epoch)
                
                # Save checkpoint
                if (self.current_epoch + 1) % self.config.save_interval == 0:
                    self.save_checkpoint(self.current_epoch, metrics)
                
                # Only increment epoch if everything was successful
                self.current_epoch += 1
                
                # Step the learning rate scheduler
                self.scheduler.step()
                
            except Exception as e:
                logger.error(f"Error during training epoch: {str(e)}")
                logger.error("Retrying epoch...")
                continue  # Don't increment epoch, try again
        
        # Save final model
        try:
            self.save_checkpoint(self.current_epoch, metrics)
            logger.info("Training complete!")
        except Exception as e:
            logger.error(f"Error saving final model: {str(e)}")
        
        if self.writer:
            self.writer.close()
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            dataloader: DataLoader for training data
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        loss_components = {}
        num_batches = 0
        
        # Add timing metrics
        data_load_time = 0.0
        gpu_transfer_time = 0.0
        forward_time = 0.0
        backward_time = 0.0
        
        for batch in dataloader:
            batch_start = time.time()
            
            # Move batch components to device
            data = batch['data'].to(self.device)
            policy_target = batch['policy'].to(self.device)
            value_target = batch['value'].to(self.device)
            gpu_transfer_time += time.time() - batch_start
            
            # Forward pass
            forward_start = time.time()
            policy_pred, value_pred = self.model(data)
            forward_time += time.time() - forward_start
            
            # Calculate loss using PolicyValueLoss
            loss, components = self.criterion(
                policy_pred=policy_pred,
                value_pred=value_pred,
                policy_target=policy_target,
                value_target=value_target,
                model=self.model
            )
            
            # Backward pass
            backward_start = time.time()
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            
            self.optimizer.step()
            backward_time += time.time() - backward_start
            
            # Update metrics
            total_loss += loss.item()
            for k, v in components.items():
                loss_components[k] = loss_components.get(k, 0) + v
            num_batches += 1
            
            # Log timing for this batch
            if num_batches % 10 == 0:  # Log every 10 batches
                logger.info(f"Batch {num_batches} timing:")
                logger.info(f"  Data load + GPU transfer: {gpu_transfer_time/num_batches:.3f}s")
                logger.info(f"  Forward pass: {forward_time/num_batches:.3f}s")
                logger.info(f"  Backward pass: {backward_time/num_batches:.3f}s")
                logger.info(f"  GPU memory: {torch.cuda.memory_allocated()/1e9:.1f}GB")
        
        # Calculate average metrics
        metrics = {
            'total_loss': total_loss / num_batches,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'avg_data_load_time': data_load_time / num_batches,
            'avg_gpu_transfer_time': gpu_transfer_time / num_batches,
            'avg_forward_time': forward_time / num_batches,
            'avg_backward_time': backward_time / num_batches
        }
        # Add averaged loss components
        metrics.update({k: v / num_batches for k, v in loss_components.items()})
        
        return metrics
    
    def _log_metrics(self, metrics: Dict[str, float], epoch: int):
        """Log training metrics.
        
        Args:
            metrics: Dictionary of metrics to log
            epoch: Current epoch number
        """
        # Log to console
        logger.info(f"Epoch {epoch + 1} metrics:")
        for name, value in metrics.items():
            logger.info(f"  {name}: {value:.4f}")
        
        # Log to TensorBoard
        if self.writer:
            for name, value in metrics.items():
                self.writer.add_scalar(f'train/{name}', value, epoch)
    
    def _should_stop_early(self, current_loss: float) -> bool:
        """Check if training should stop early.
        
        Args:
            current_loss: Current epoch's loss
            
        Returns:
            True if training should stop, False otherwise
        """
        if current_loss < self.best_loss - self.config.early_stopping_min_delta:
            self.best_loss = current_loss
            self.patience_counter = 0
            return False
        
        self.patience_counter += 1
        return self.patience_counter >= self.config.early_stopping_patience
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            metrics: Current training metrics
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.model.get_config(),
            'experiment_name': os.path.basename(self.config.experiment_dir)  # Store experiment name in checkpoint
        }
        
        # Get model version from experiment name (e.g. rival_ai_v1 from rival_ai_v1_20250613_215945)
        experiment_name = os.path.basename(self.config.experiment_dir)
        model_version = experiment_name.split('_')[0] + '_' + experiment_name.split('_')[1]  # e.g. rival_ai_v1
        
        # Create model version directory inside checkpoints
        model_checkpoint_dir = Path(self.config.checkpoint_dir) / model_version
        model_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save checkpoint with timestamp and epoch
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = model_checkpoint_dir / f'checkpoint_{timestamp}_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Also save as best model if this is the best so far
        if metrics.get('total_loss', float('inf')) < self.best_loss:
            best_model_path = model_checkpoint_dir / f'best_model.pt'
            torch.save(checkpoint, best_model_path)
            logger.info(f"Saved best model to {best_model_path}")
            self.best_loss = metrics['total_loss']
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_loss = checkpoint['metrics']['total_loss']
        
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}") 