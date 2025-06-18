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
import sys
from pathlib import Path
from datetime import datetime
import chess
import networkx as nx

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
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        
        # Create log directory and subdirectories
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir = self.log_dir / 'plots'
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
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
            'log_dir': str(self.log_dir),
            'timestamp': str(torch.tensor(0).device),  # Save device info
        }
        
        config_path = self.log_dir / 'config.json'
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
        metrics_path = self.log_dir / 'metrics.json'
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
    
    def plot_game_metrics(
        self,
        num_moves: List[int],
        results: List[str],
        window_size: int = 10,
        save: bool = True,
    ):
        """Plot game metrics.
        
        Args:
            num_moves: List of number of moves per game
            results: List of game results
            window_size: Size of moving average window
            save: Whether to save the plot
        """
        # ... existing plotting code ...
        
        if save:
            # Save plot in experiment's plots directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_file = self.plots_dir / f"game_metrics_{timestamp}.png"
            plt.savefig(plot_file)
            logger.info(f"Saved game metrics plot to {plot_file}")
        
        plt.close()

    def plot_training_metrics(self, metrics: Dict[str, List[float]], title: str = "Training Metrics") -> None:
        """Plot training metrics over time.
        
        Args:
            metrics: Dictionary of metric names to lists of values
            title: Plot title
        """
        plt.figure(figsize=(12, 6))
        
        for metric_name, values in metrics.items():
            plt.plot(values, label=metric_name)
        
        plt.title(title)
        plt.xlabel("Training Step")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        
        # Save plot with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = self.plots_dir / f'training_metrics_{timestamp}.png'
        plt.savefig(plot_file)
        plt.close()
        
        logger.info(f"Saved training metrics plot to {plot_file}")
    
    def plot_position_evaluation(self, position: str, policy: Dict[str, float], value: float) -> None:
        """Plot position evaluation with move probabilities.
        
        Args:
            position: FEN string of the position
            policy: Dictionary of moves to probabilities
            value: Position evaluation
        """
        # Create chess board
        board = chess.Board(position)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot board
        chess.svg.board(board, size=400, ax=ax1)
        ax1.set_title("Position")
        
        # Plot move probabilities
        moves = list(policy.keys())
        probs = list(policy.values())
        
        # Sort by probability
        sorted_indices = np.argsort(probs)[::-1]
        moves = [moves[i] for i in sorted_indices]
        probs = [probs[i] for i in sorted_indices]
        
        # Take top 10 moves
        moves = moves[:10]
        probs = probs[:10]
        
        ax2.barh(range(len(moves)), probs)
        ax2.set_yticks(range(len(moves)))
        ax2.set_yticklabels(moves)
        ax2.set_xlabel("Probability")
        ax2.set_title(f"Top 10 Moves (Value: {value:.3f})")
        
        plt.tight_layout()
        
        # Save plot with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = self.plots_dir / f'position_eval_{timestamp}.png'
        plt.savefig(plot_file)
        plt.close()
        
        logger.info(f"Saved position evaluation plot to {plot_file}")
    
    def plot_game_tree(self, game: Dict, max_depth: int = 5) -> None:
        """Plot game tree visualization.
        
        Args:
            game: Dictionary containing game data
            max_depth: Maximum depth to plot
        """
        # Create chess board
        board = chess.Board()
        
        # Create figure
        plt.figure(figsize=(15, 10))
        
        # Plot initial position
        plt.subplot(1, 2, 1)
        chess.svg.board(board, size=400)
        plt.title("Initial Position")
        
        # Plot game tree
        plt.subplot(1, 2, 2)
        
        # Create tree structure
        tree = nx.DiGraph()
        
        # Add root node
        tree.add_node(0, pos=(0, 0), fen=board.fen())
        
        # Add moves
        for i, move in enumerate(game['moves'][:max_depth]):
            parent = i
            child = i + 1
            
            # Make move
            board.push(move)
            
            # Add node
            tree.add_node(child, pos=(i+1, 0), fen=board.fen())
            tree.add_edge(parent, child, move=move.uci())
        
        # Draw tree
        pos = nx.get_node_attributes(tree, 'pos')
        nx.draw(tree, pos, with_labels=True, node_color='lightblue', 
                node_size=2000, font_size=8)
        
        # Add move labels
        edge_labels = nx.get_edge_attributes(tree, 'move')
        nx.draw_networkx_edge_labels(tree, pos, edge_labels=edge_labels)
        
        plt.title("Game Tree")
        plt.tight_layout()
        
        # Save plot with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = self.plots_dir / f'game_tree_{timestamp}.png'
        plt.savefig(plot_file)
        plt.close()
        
        logger.info(f"Saved game tree plot to {plot_file}")

def main():
    """Main entry point for the visualizer script."""
    if len(sys.argv) != 3:
        print("Usage: python visualizer.py <log_dir> <experiment_name>")
        print("Example: python visualizer.py ./logs/my_experiment experiment_1")
        sys.exit(1)
        
    log_dir = sys.argv[1]
    experiment_name = sys.argv[2]
    
    # Create visualizer
    visualizer = TrainingVisualizer(log_dir=log_dir, experiment_name=experiment_name)
    
    # Try to load existing metrics
    metrics_path = visualizer.log_dir / 'metrics.json'
    if metrics_path.exists():
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                visualizer.metrics = metrics
            print(f"Loaded metrics from {metrics_path}")
            
            # Plot all metrics
            print("Plotting metrics...")
            visualizer.plot_metrics()
            
            # Print some summary statistics
            print("\nSummary Statistics:")
            for metric_name, values in metrics.items():
                if values:  # Only print if we have values
                    print(f"{metric_name}:")
                    print(f"  Latest value: {values[-1]:.4f}")
                    print(f"  Min: {min(values):.4f}")
                    print(f"  Max: {max(values):.4f}")
                    print(f"  Mean: {sum(values)/len(values):.4f}")
                    print()
        except Exception as e:
            print(f"Error loading metrics: {e}")
    else:
        print(f"No metrics found at {metrics_path}")
        print("This directory might be empty or the metrics haven't been saved yet.")

if __name__ == "__main__":
    main() 