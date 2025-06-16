"""
Training visualization utilities.
"""

import os
import logging
import json
from typing import Dict, List, Optional
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

class TrainingVisualizer:
    """Visualizes training progress using TensorBoard and matplotlib."""
    
    def __init__(
        self,
        log_dir: str,
        experiment_name: str,
    ):
        """Initialize visualizer.
        
        Args:
            log_dir: Directory to save logs
            experiment_name: Name of the experiment
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=log_dir)
        
        # Initialize metrics storage
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'policy_loss': [],
            'value_loss': [],
            'feature_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'learning_rate': [],
        }
        
        # Save experiment config
        self._save_config()
    
    def _save_config(self):
        """Save experiment configuration."""
        config = {
            'experiment_name': self.experiment_name,
            'log_dir': self.log_dir,
            'timestamp': str(torch.tensor(0).device),  # Save device info
        }
        
        config_path = os.path.join(self.log_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def log_metrics(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        policy_loss: Optional[float] = None,
        value_loss: Optional[float] = None,
        feature_loss: Optional[float] = None,
        train_accuracy: Optional[float] = None,
        val_accuracy: Optional[float] = None,
        learning_rate: Optional[float] = None,
    ):
        """Log training metrics.
        
        Args:
            epoch: Current epoch
            train_loss: Training loss
            val_loss: Validation loss
            policy_loss: Policy head loss
            value_loss: Value head loss
            feature_loss: Feature extraction loss
            train_accuracy: Training accuracy
            val_accuracy: Validation accuracy
            learning_rate: Current learning rate
        """
        # Update metrics
        self.metrics['train_loss'].append(train_loss)
        if val_loss is not None:
            self.metrics['val_loss'].append(val_loss)
        if policy_loss is not None:
            self.metrics['policy_loss'].append(policy_loss)
        if value_loss is not None:
            self.metrics['value_loss'].append(value_loss)
        if feature_loss is not None:
            self.metrics['feature_loss'].append(feature_loss)
        if train_accuracy is not None:
            self.metrics['train_accuracy'].append(train_accuracy)
        if val_accuracy is not None:
            self.metrics['val_accuracy'].append(val_accuracy)
        if learning_rate is not None:
            self.metrics['learning_rate'].append(learning_rate)
        
        # Log to TensorBoard
        self.writer.add_scalar('Loss/train', train_loss, epoch)
        if val_loss is not None:
            self.writer.add_scalar('Loss/val', val_loss, epoch)
        if policy_loss is not None:
            self.writer.add_scalar('Loss/policy', policy_loss, epoch)
        if value_loss is not None:
            self.writer.add_scalar('Loss/value', value_loss, epoch)
        if feature_loss is not None:
            self.writer.add_scalar('Loss/feature', feature_loss, epoch)
        if train_accuracy is not None:
            self.writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        if val_accuracy is not None:
            self.writer.add_scalar('Accuracy/val', val_accuracy, epoch)
        if learning_rate is not None:
            self.writer.add_scalar('LearningRate', learning_rate, epoch)
        
        # Save metrics to file
        metrics_path = os.path.join(self.log_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def plot_metrics(
        self,
        metrics: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ):
        """Plot training metrics.
        
        Args:
            metrics: List of metrics to plot (default: all)
            save_path: Path to save plot (default: show plot)
        """
        if metrics is None:
            metrics = list(self.metrics.keys())
        
        plt.figure(figsize=(12, 8))
        
        for metric in metrics:
            if metric in self.metrics and self.metrics[metric]:
                plt.plot(self.metrics[metric], label=metric)
        
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title(f'Training Metrics - {self.experiment_name}')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def log_model_graph(self, model: torch.nn.Module, input_data: torch.Tensor):
        """Log model graph to TensorBoard.
        
        Args:
            model: Model to visualize
            input_data: Sample input data
        """
        try:
            self.writer.add_graph(model, input_data)
        except Exception as e:
            logger.warning(f"Failed to log model graph: {e}")
    
    def log_histogram(self, name: str, values: torch.Tensor, epoch: int):
        """Log histogram to TensorBoard.
        
        Args:
            name: Name of the histogram
            values: Values to plot
            epoch: Current epoch
        """
        self.writer.add_histogram(name, values, epoch)
    
    def log_scalar(self, name: str, value: float, epoch: int):
        """Log scalar value to TensorBoard.
        
        Args:
            name: Name of the scalar
            value: Value to log
            epoch: Current epoch
        """
        self.writer.add_scalar(name, value, epoch)
    
    def close(self):
        """Close TensorBoard writer."""
        self.writer.close() 