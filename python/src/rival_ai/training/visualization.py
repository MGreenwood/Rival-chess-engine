"""
Visualization utilities for training metrics.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)

class TrainingVisualizer:
    """Handles visualization of training metrics."""
    
    def __init__(
        self,
        log_dir: Union[str, Path],
        experiment_name: Optional[str] = None,
    ):
        """Initialize visualizer.
        
        Args:
            log_dir: Directory to save logs and plots
            experiment_name: Name of the experiment (defaults to timestamp)
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = self.log_dir / self.experiment_name
        
        # Create directories
        self.tensorboard_dir = self.log_dir / "tensorboard"
        self.plots_dir = self.log_dir / "plots"
        self.metrics_dir = self.log_dir / "metrics"
        
        for directory in [self.tensorboard_dir, self.plots_dir, self.metrics_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(str(self.tensorboard_dir))
        
        # Set up plotting style
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (12, 6)
        
        # Initialize metrics storage
        self.metrics_history: Dict[str, List[float]] = {}
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        prefix: str = "",
    ):
        """Log metrics to TensorBoard and store for plotting.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Current training step
            prefix: Prefix for metric names (e.g., 'train_' or 'val_')
        """
        # Log to TensorBoard
        for name, value in metrics.items():
            full_name = f"{prefix}{name}" if prefix else name
            self.writer.add_scalar(full_name, value, step)
            
            # Store for plotting
            if full_name not in self.metrics_history:
                self.metrics_history[full_name] = []
            self.metrics_history[full_name].append(value)
        
        # Save metrics to JSON
        metrics_file = self.metrics_dir / f"metrics_step_{step}.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def plot_metrics(
        self,
        metrics: Optional[List[str]] = None,
        window_size: int = 10,
        save: bool = True,
    ):
        """Plot training metrics.
        
        Args:
            metrics: List of metric names to plot (None for all)
            window_size: Window size for moving average
            save: Whether to save the plot
        """
        if not metrics:
            metrics = list(self.metrics_history.keys())
        
        # Create subplots
        n_metrics = len(metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 4 * n_metrics))
        if n_metrics == 1:
            axes = [axes]
        
        # Plot each metric
        for ax, metric in zip(axes, metrics):
            if metric not in self.metrics_history:
                logger.warning(f"Metric {metric} not found in history")
                continue
            
            values = self.metrics_history[metric]
            steps = range(len(values))
            
            # Plot raw values
            ax.plot(steps, values, alpha=0.3, label='Raw')
            
            # Plot moving average
            if len(values) >= window_size:
                moving_avg = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
                ax.plot(range(window_size-1, len(values)), moving_avg, label=f'{window_size}-step MA')
            
            ax.set_title(metric)
            ax.set_xlabel('Step')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        
        if save:
            plot_file = self.plots_dir / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(plot_file)
            logger.info(f"Saved metrics plot to {plot_file}")
        
        plt.close()
    
    def plot_game_metrics(
        self,
        num_moves: List[int],
        results: List[str],
        window_size: int = 10,
        save: bool = True,
    ):
        """Plot self-play game metrics.
        
        Args:
            num_moves: List of game lengths
            results: List of game results ('white_win', 'black_win', 'draw')
            window_size: Window size for moving average
            save: Whether to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot game lengths
        steps = range(len(num_moves))
        ax1.plot(steps, num_moves, alpha=0.3, label='Raw')
        
        if len(num_moves) >= window_size:
            moving_avg = np.convolve(num_moves, np.ones(window_size)/window_size, mode='valid')
            ax1.plot(range(window_size-1, len(num_moves)), moving_avg, label=f'{window_size}-step MA')
        
        ax1.set_title('Game Length')
        ax1.set_xlabel('Game')
        ax1.set_ylabel('Number of Moves')
        ax1.legend()
        ax1.grid(True)
        
        # Plot game results
        result_counts = {
            'white_win': 0,
            'black_win': 0,
            'draw': 0
        }
        result_history = {
            'white_win': [],
            'black_win': [],
            'draw': []
        }
        
        for result in results:
            result_counts[result] += 1
            total = sum(result_counts.values())
            for key in result_history:
                result_history[key].append(result_counts[key] / total)
        
        for result, values in result_history.items():
            ax2.plot(steps, values, label=result.replace('_', ' ').title())
        
        ax2.set_title('Game Results')
        ax2.set_xlabel('Game')
        ax2.set_ylabel('Proportion')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save:
            plot_file = self.plots_dir / f"game_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(plot_file)
            logger.info(f"Saved game metrics plot to {plot_file}")
        
        plt.close()
    
    def close(self):
        """Close TensorBoard writer and clean up."""
        self.writer.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close() 