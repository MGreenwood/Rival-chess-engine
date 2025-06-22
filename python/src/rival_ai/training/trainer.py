"""
Trainer module for training the chess model.
"""

import os
import time
import logging
import json
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from rival_ai.config import TrainingConfig, SelfPlayConfig
from rival_ai.data.dataset import ChessDataset, create_dataloader

from rival_ai.models import ChessGNN as ChessModel
from rival_ai.training.losses import ImprovedPolicyValueLoss, PolicyValueLoss
from rival_ai.training.visualizer import TrainingVisualizer
from rival_ai.mcts import MCTS
from rival_ai.training.self_play import SelfPlay, SelfPlayConfig, GameRecord
from datetime import datetime
from .metrics import PieceRelationshipMetrics, PRCScore
from collections import defaultdict
import chess
import pickle
import traceback  # Add at module level

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
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    use_improved_loss: bool = True  # Use improved loss function
    policy_weight: float = 1.0
    value_weight: float = 0.5  # Reduced value weight
    entropy_weight: float = 0.01
    l2_weight: float = 1e-4

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
    use_prc_metrics: bool = True
    
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
        
        # Initialize loss function - Use PAG tactical loss if configured
        if self.config.use_pag_tactical_loss:
            from .pag_tactical_loss import PAGTacticalLoss
            self.criterion = PAGTacticalLoss(**self.config.pag_tactical_config)
            logger.info("Using PAG Tactical Loss for preventing blunders")
        elif self.config.use_improved_loss:
            self.criterion = ImprovedPolicyValueLoss(
                policy_weight=self.config.policy_weight,
                value_weight=self.config.value_weight,
                entropy_weight=self.config.entropy_weight,
                l2_weight=self.config.l2_weight,
            )
        else:
            self.criterion = PolicyValueLoss()
        
        # Initialize tensorboard writer if enabled
        self.writer = None
        if self.config.use_tensorboard:
            log_dir = Path(self.config.experiment_dir) / 'logs'
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(str(log_dir))
        
        # Initialize visualizer
        experiment_name = Path(self.config.experiment_dir).name
        log_dir = Path(self.config.experiment_dir) / 'logs'
        self.visualizer = TrainingVisualizer(
            log_dir=str(log_dir),
            experiment_name=experiment_name
        )
        
        # Initialize self-play manager with experiment directory
        self.self_play = SelfPlay(
            self.model, 
            self.self_play_config,
            experiment_dir=Path(self.config.experiment_dir)
        )
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.patience_counter = 0
        
        # Load checkpoint if specified
        if self.config.checkpoint_path:
            self.load_checkpoint(self.config.checkpoint_path)
            if self.config.resume_epoch is not None:
                self.current_epoch = self.config.resume_epoch
        
        if self.use_prc_metrics:
            self.prc_metrics = PieceRelationshipMetrics(self.model)
    
    def _cleanup_old_self_play_data(self, max_files: int = 10):
        """Clean up old self-play data files, keeping only the most recent ones.
        
        Args:
            max_files: Maximum number of pickle files to keep
        """
        self_play_dir = Path(self.config.experiment_dir) / 'self_play_data'
        if not self_play_dir.exists():
            return
        
        # Find all pickle files
        pkl_files = list(self_play_dir.glob('*.pkl'))
        if len(pkl_files) <= max_files:
            return
        
        # Sort by modification time (newest first)
        pkl_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Delete older files
        for old_file in pkl_files[max_files:]:
            try:
                old_file.unlink()
                logger.info(f"Deleted old self-play data: {old_file}")
            except Exception as e:
                logger.warning(f"Failed to delete {old_file}: {e}")

    def train(self, setup_logging_fn=None):
        """Train the model.
        
        Args:
            setup_logging_fn: Optional function to set up logging for each epoch
        """
        logger.info("Starting training...")
        logger.info(f"Training configuration: {self.config}")
        logger.info(f"Using improved loss: {self.config.use_improved_loss}")
        
        # Initialize metrics
        best_loss = float('inf')
        metrics_history = []
        metrics = {}  # Initialize metrics at the start
        
        try:
            # Create experiment-specific directories
            experiment_dir = Path(self.config.experiment_dir)
            experiment_name = experiment_dir.name
            self_play_dir = experiment_dir / 'self_play_data'
            checkpoints_dir = experiment_dir / 'checkpoints'
            
            # Create all necessary directories
            for directory in [self_play_dir, checkpoints_dir]:
                directory.mkdir(parents=True, exist_ok=True)
            
            for epoch in range(self.current_epoch, self.config.num_epochs):
                # Set up epoch-specific logging if function provided
                if setup_logging_fn:
                    setup_logging_fn(epoch)
                    
                try:
                    logger.info(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
                    
                    # Generate self-play games
                    logger.info("Generating self-play games...")
                    try:
                        games = self.self_play.generate_games(epoch=epoch)
                        if not games:
                            logger.error("No games were generated in this epoch")
                            continue
                        logger.info(f"Generated {len(games)} games")
                    except Exception as e:
                        logger.error(f"Error generating self-play games: {str(e)}")
                        logger.error(f"Error type: {type(e)}")
                        logger.error(f"Traceback: {traceback.format_exc()}")
                        continue
                    
                    # Save games to pickle file
                    try:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        games_file = self_play_dir / f'games_epoch_{epoch}_{timestamp}.pkl'
                        
                        with open(games_file, 'wb') as f:
                            pickle.dump(games, f)
                        logger.info(f"Saved {len(games)} games to {games_file}")
                    except Exception as e:
                        logger.error(f"Error saving games: {str(e)}")
                        logger.error(f"Error type: {type(e)}")
                        logger.error(f"Traceback: {traceback.format_exc()}")
                        # Continue training even if saving fails
                    
                    # Clean up old self-play data files
                    try:
                        self._cleanup_old_self_play_data()
                    except Exception as e:
                        logger.warning(f"Error cleaning up old self-play data: {str(e)}")
                        # Continue training even if cleanup fails
                    
                    # Create dataset and dataloader
                    try:
                        if self.config.use_pag_tactical_loss:
                            # Use enhanced dataset for PAG tactical loss
                            from rival_ai.data.enhanced_dataset import create_enhanced_dataloader
                            
                            # Save games temporarily for enhanced dataset
                            temp_data_dir = Path(self.config.experiment_dir) / 'temp_training_data'
                            temp_data_dir.mkdir(exist_ok=True)
                            
                            # Convert games to JSON format for enhanced dataset
                            temp_positions = []
                            for game in games:
                                # Extract training data from GameRecord structure
                                if hasattr(game, 'states') and hasattr(game, 'policies') and hasattr(game, 'values'):
                                    for i in range(len(game.states)):
                                        if i < len(game.policies) and i < len(game.values):
                                            # Convert policy tensor to list if needed
                                            policy = game.policies[i]
                                            if hasattr(policy, 'tolist'):
                                                policy = policy.tolist()
                                            elif hasattr(policy, 'numpy'):
                                                policy = policy.numpy().tolist()
                                            else:
                                                policy = [0.0] * 4096  # Fallback
                                            
                                            # Convert value to float if needed
                                            value = game.values[i]
                                            if hasattr(value, 'item'):
                                                value = value.item()
                                            elif hasattr(value, 'numpy'):
                                                value = value.numpy().item()
                                            else:
                                                value = float(value)
                                            
                                            temp_positions.append({
                                                'fen': game.states[i].fen(),
                                                'policy': policy,
                                                'value': value
                                            })
                                else:
                                    logger.warning(f"Skipping game with unexpected structure: {type(game)}")
                            
                            # Save to temporary JSON file
                            temp_file = temp_data_dir / f'epoch_{epoch}_data.json'
                            with open(temp_file, 'w') as f:
                                json.dump(temp_positions, f)
                            
                            # Create enhanced dataloader
                            dataloader = create_enhanced_dataloader(
                                data_dir=temp_data_dir,
                                batch_size=self.config.batch_size,
                                shuffle=True,
                                num_workers=0,  # Use 0 for Windows compatibility with enhanced dataset
                                pin_memory=True,
                                extract_rust_pag=True
                            )
                            logger.info(f"Created enhanced dataset with PAG features: {len(temp_positions)} positions")
                        else:
                            # Use standard dataset
                            dataset = ChessDataset.from_game_records(games)
                            dataloader = create_dataloader(
                                dataset,
                                batch_size=self.config.batch_size,
                                shuffle=True,
                                num_workers=self.config.num_workers,
                                pin_memory=True
                            )
                            logger.info(f"Created standard dataset with {len(dataset)} positions")
                    except Exception as e:
                        logger.error(f"Error creating dataset/dataloader: {str(e)}")
                        logger.error(f"Error type: {type(e)}")
                        logger.error(f"Traceback: {traceback.format_exc()}")
                        continue
                    
                    # Train for one epoch
                    try:
                        train_metrics = self.train_epoch(dataloader)
                        logger.info(f"Completed training epoch {epoch + 1}")
                    except Exception as e:
                        logger.error(f"Error during training epoch: {str(e)}")
                        logger.error(f"Error type: {type(e)}")
                        logger.error(f"Traceback: {traceback.format_exc()}")
                        # Save checkpoint before continuing
                        self.save_checkpoint(epoch, {'error': str(e)})
                        continue
                    
                    # Log metrics
                    self._log_metrics(epoch, train_metrics)
                    
                    # Update learning rate
                    self.scheduler.step()
                    
                    # Update PAG tactical loss epoch for progressive difficulty
                    if self.config.use_pag_tactical_loss and hasattr(self.criterion, 'update_epoch'):
                        self.criterion.update_epoch(epoch)
                    
                    # Save checkpoint
                    if (epoch + 1) % self.config.save_interval == 0:
                        self.save_checkpoint(epoch, train_metrics)
                    
                    # Check for early stopping
                    current_loss = train_metrics.get('total_loss', float('inf'))
                    if self._should_stop_early(current_loss):
                        logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                        break
                    
                    # Update best loss
                    if current_loss < best_loss:
                        best_loss = current_loss
                        self.save_checkpoint(epoch, train_metrics, is_best=True)
                    
                    # Store metrics history
                    metrics_history.append(train_metrics)
                    
                except Exception as e:
                    logger.error(f"Error in epoch {epoch + 1}: {str(e)}")
                    logger.error(f"Error type: {type(e)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    continue
            
            logger.info("Training completed!")
            logger.info(f"Best loss achieved: {best_loss:.4f}")
            
        except Exception as e:
            logger.error(f"Critical error during training: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch with improved loss monitoring."""
        self.model.train()
        total_loss = 0.0
        loss_components = defaultdict(float)
        num_batches = 0
        
        # Initialize timing metrics
        timing_metrics = {
            'data_load_time': 0.0,
            'gpu_transfer_time': 0.0,
            'forward_time': 0.0,
            'backward_time': 0.0,
        }
        
        # Initialize PRC metrics if enabled
        prc_scores = defaultdict(list) if self.use_prc_metrics else None
        
        try:
            for batch_idx, batch in enumerate(dataloader):
                try:
                    # Data loading timing
                    data_load_start = time.time()
                    
                    # Move batch to device
                    data = batch['data'].to(self.device)
                    policy_target = batch['policy'].to(self.device)
                    value_target = batch['value'].to(self.device)
                    
                    timing_metrics['data_load_time'] += time.time() - data_load_start
                    timing_metrics['gpu_transfer_time'] += time.time() - data_load_start
                    
                    # Forward pass timing
                    forward_start = time.time()
                    
                    # Forward pass
                    policy_pred, value_pred = self.model(data)
                    
                    timing_metrics['forward_time'] += time.time() - forward_start
                    
                    # Validate predictions
                    if torch.isnan(policy_pred).any() or torch.isnan(value_pred).any():
                        logger.error(f"NaN detected in predictions at batch {batch_idx}")
                        logger.error(f"Policy pred range: [{policy_pred.min():.4f}, {policy_pred.max():.4f}]")
                        logger.error(f"Value pred range: [{value_pred.min():.4f}, {value_pred.max():.4f}]")
                        continue
                    
                    # Calculate loss
                    try:
                        if self.config.use_pag_tactical_loss:
                            # Extract PAG features for tactical loss
                            pag_features = batch.get('pag_features')
                            if pag_features is not None:
                                pag_features = pag_features.to(self.device)
                                loss_dict = self.criterion(
                                    policy_pred,
                                    value_pred,
                                    policy_target,
                                    value_target,
                                    pag_features
                                )
                                loss = loss_dict['total_loss']
                                components = {k: v.item() if hasattr(v, 'item') else v 
                                            for k, v in loss_dict.items() if k != 'total_loss'}
                            else:
                                # Fallback to regular loss if PAG features not available
                                logger.warning("PAG features not available, falling back to regular loss")
                                loss, components = self.criterion(
                                    policy_pred,
                                    value_pred,
                                    policy_target,
                                    value_target,
                                    self.model
                                )
                        else:
                            loss, components = self.criterion(
                                policy_pred,
                                value_pred,
                                policy_target,
                                value_target,
                                self.model
                            )
                    except Exception as e:
                        logger.error(f"Error calculating loss for batch {batch_idx}: {str(e)}")
                        logger.error(f"Shapes: policy_pred={policy_pred.shape}, value_pred={value_pred.shape}")
                        logger.error(f"Shapes: policy_target={policy_target.shape}, value_target={value_target.shape}")
                        logger.error(f"Policy target range: [{policy_target.min():.4f}, {policy_target.max():.4f}]")
                        logger.error(f"Value target range: [{value_target.min():.4f}, {value_target.max():.4f}]")
                        raise
                    
                    # Backward pass
                    try:
                        backward_start = time.time()
                        self.optimizer.zero_grad()
                        loss.backward()
                        
                        # Gradient clipping
                        if self.config.grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                        
                        self.optimizer.step()
                        timing_metrics['backward_time'] += time.time() - backward_start
                    except Exception as e:
                        logger.error(f"Error in backward pass for batch {batch_idx}: {str(e)}")
                        logger.error(f"Loss value: {loss.item()}")
                        raise
                    
                    # Update metrics
                    total_loss += loss.item()
                    for k, v in components.items():
                        loss_components[k] += v
                    num_batches += 1
                    
                    # Log progress every 100 batches
                    if batch_idx % 100 == 0:
                        avg_loss = total_loss / max(1, num_batches)
                        logger.info(f"Batch {batch_idx}: avg_loss={avg_loss:.4f}")
                        
                        # Log detailed loss components
                        if components:
                            logger.info(f"  Policy loss: {components.get('policy_loss', 0):.4f}")
                            logger.info(f"  Value loss: {components.get('value_loss', 0):.4f}")
                            logger.info(f"  Entropy: {components.get('entropy', 0):.4f}")
                    
                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx}: {str(e)}")
                    logger.error(f"Error type: {type(e)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    continue
            
            # Calculate average metrics
            try:
                metrics = {
                    'total_loss': total_loss / max(1, num_batches),
                    'learning_rate': self.scheduler.get_last_lr()[0],
                    'avg_data_load_time': timing_metrics['data_load_time'] / max(1, num_batches),
                    'avg_gpu_transfer_time': timing_metrics['gpu_transfer_time'] / max(1, num_batches),
                    'avg_forward_time': timing_metrics['forward_time'] / max(1, num_batches),
                    'avg_backward_time': timing_metrics['backward_time'] / max(1, num_batches)
                }
                # Add averaged loss components
                metrics.update({k: v / max(1, num_batches) for k, v in loss_components.items()})
                
                if self.use_prc_metrics and prc_scores:
                    for key, values in prc_scores.items():
                        if values:  # Only compute average if we have values
                            metrics[f'prc_{key}'] = sum(values) / len(values)
                
                return metrics
                
            except Exception as e:
                logger.error(f"Error calculating final metrics: {str(e)}")
                # Return basic metrics if detailed calculation fails
                return {
                    'total_loss': total_loss / max(1, num_batches),
                    'error': str(e)
                }
                
        except Exception as e:
            logger.error(f"Error in train_epoch: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def _log_metrics(self, epoch: int, metrics: Dict[str, float]):
        """Log training metrics.
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary of metrics to log
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
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save checkpoint with standardized path and naming.
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary of training metrics
            is_best: True if this is the best checkpoint so far
        """
        # Extract experiment name from experiment directory
        experiment_name = Path(self.config.experiment_dir).name
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'best_loss': self.best_loss,
            'prc_metrics_enabled': self.use_prc_metrics,
            'experiment_name': experiment_name
        }
        
        # Create standardized checkpoint directory within experiment directory
        checkpoint_dir = Path(self.config.experiment_dir) / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save checkpoint with standardized naming
        checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Also save as best model if this is the best so far
        if is_best:
            best_model_path = checkpoint_dir / 'best_model.pt'
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
        
        # Load model state (always present)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if present (may not be in emergency checkpoints)
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("Loaded optimizer state")
        else:
            logger.warning("No optimizer state found in checkpoint - starting with fresh optimizer")
        
        # Load scheduler state if present
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logger.info("Loaded scheduler state")
        else:
            logger.warning("No scheduler state found in checkpoint - starting with fresh scheduler")
        
        # Load training state
        self.current_epoch = checkpoint['epoch']
        
        # Load best loss if present
        if 'best_loss' in checkpoint:
            self.best_loss = checkpoint['best_loss']
        elif 'metrics' in checkpoint and 'total_loss' in checkpoint['metrics']:
            self.best_loss = checkpoint['metrics']['total_loss']
        else:
            logger.warning("No best loss found in checkpoint - starting with default")
        
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
        
        if self.use_prc_metrics:
            self.prc_metrics = PieceRelationshipMetrics(self.model)
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model with PRC metrics."""
        self.model.eval()
        metrics = {}
        
        # Generate a few games for evaluation
        eval_games = []
        for _ in range(5):  # Evaluate on 5 games
            game = self.self_play.play_game()
            eval_games.append(game)
        
        # Compute basic metrics
        total_moves = sum(len(game.moves) for game in eval_games)
        avg_game_length = total_moves / len(eval_games)
        
        # Count game results
        results = [game.result for game in eval_games]
        white_wins = sum(1 for r in results if r == GameResult.WHITE_WINS)
        black_wins = sum(1 for r in results if r == GameResult.BLACK_WINS)
        draws = sum(1 for r in results if r == GameResult.DRAW)
        
        # Add basic metrics
        metrics.update({
            'eval_avg_game_length': avg_game_length,
            'eval_white_win_rate': white_wins / len(eval_games),
            'eval_black_win_rate': black_wins / len(eval_games),
            'eval_draw_rate': draws / len(eval_games)
        })
        
        if self.use_prc_metrics:
            # Analyze games for PRC scores
            prc_scores = defaultdict(list)
            for game in eval_games:
                game_scores = self.prc_metrics.analyze_game(
                    game.moves,
                    chess.Board()
                )
                for key, values in game_scores.items():
                    prc_scores[key].extend(values)
            
            # Add average PRC scores to metrics
            for key, values in prc_scores.items():
                metrics[f'eval_prc_{key}'] = sum(values) / len(values)
        
        return metrics 

    def _log_final_metrics(self, metrics_history: List[Dict[str, float]]):
        """Log final training metrics summary.
        
        Args:
            metrics_history: List of metrics dictionaries from all epochs
        """
        if not metrics_history:
            logger.warning("No metrics history to log")
            return
        
        try:
            logger.info("=== FINAL TRAINING SUMMARY ===")
            
            # Calculate final statistics
            final_metrics = metrics_history[-1]
            logger.info(f"Final epoch metrics: {final_metrics}")
            
            # Calculate average metrics across all epochs
            avg_metrics = {}
            for key in final_metrics.keys():
                if key in ['learning_rate', 'epoch']:  # Skip non-numeric metrics
                    continue
                values = [m.get(key, 0) for m in metrics_history if key in m]
                if values:
                    avg_metrics[f'avg_{key}'] = sum(values) / len(values)
            
            logger.info(f"Average metrics across all epochs: {avg_metrics}")
            
            # Find best and worst epochs
            if 'total_loss' in final_metrics:
                losses = [m.get('total_loss', float('inf')) for m in metrics_history]
                best_epoch = losses.index(min(losses)) + 1
                worst_epoch = losses.index(max(losses)) + 1
                logger.info(f"Best loss at epoch {best_epoch}: {min(losses):.4f}")
                logger.info(f"Worst loss at epoch {worst_epoch}: {max(losses):.4f}")
            
            # Log training duration
            if hasattr(self, 'training_start_time'):
                duration = time.time() - self.training_start_time
                hours = int(duration // 3600)
                minutes = int((duration % 3600) // 60)
                seconds = int(duration % 60)
                logger.info(f"Total training time: {hours:02d}:{minutes:02d}:{seconds:02d}")
            
            logger.info("=== END TRAINING SUMMARY ===")
            
        except Exception as e:
            logger.error(f"Error logging final metrics: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}") 