"""
Metrics for monitoring model training and evaluation.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

class MetricTracker:
    """
    Tracks and computes various metrics during training.
    
    This class maintains running statistics for multiple metrics
    and provides methods for updating and computing them.
    """
    
    def __init__(self):
        """Initialize the metric tracker."""
        self.reset()
    
    def reset(self):
        """Reset all metrics to their initial state."""
        self.metrics = defaultdict(list)
        self.running_sums = defaultdict(float)
        self.running_counts = defaultdict(int)
    
    def update(self, metrics_dict: Dict[str, float], batch_size: int = 1):
        """
        Update metrics with new values.
        
        Args:
            metrics_dict: Dictionary of metric names and values
            batch_size: Size of the current batch
        """
        for name, value in metrics_dict.items():
            self.metrics[name].append(value)
            self.running_sums[name] += value * batch_size
            self.running_counts[name] += batch_size
    
    def compute(self) -> Dict[str, float]:
        """
        Compute current metric values.
        
        Returns:
            Dictionary of metric names and their current values
        """
        return {
            name: self.running_sums[name] / max(1, self.running_counts[name])
            for name in self.running_sums
        }
    
    def get_history(self, metric_name: str) -> List[float]:
        """
        Get history of values for a specific metric.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            List of historical values
        """
        return self.metrics[metric_name]

def compute_policy_accuracy(
    policy_logits: torch.Tensor,
    policy_target: torch.Tensor,
    top_k: int = 1
) -> float:
    """
    Compute policy accuracy (top-k).
    
    Args:
        policy_logits: Predicted move logits [batch_size, num_moves]
        policy_target: Target move indices [batch_size]
        top_k: Number of top predictions to consider
        
    Returns:
        Accuracy as a float between 0 and 1
    """
    _, top_k_indices = torch.topk(policy_logits, k=top_k, dim=-1)
    correct = torch.any(top_k_indices == policy_target.unsqueeze(-1), dim=-1)
    return correct.float().mean().item()

def compute_value_accuracy(
    value_pred: torch.Tensor,
    value_target: torch.Tensor,
    threshold: float = 0.5
) -> float:
    """
    Compute value prediction accuracy.
    
    Args:
        value_pred: Predicted values [batch_size, 1]
        value_target: Target values [batch_size, 1]
        threshold: Threshold for considering predictions correct
        
    Returns:
        Accuracy as a float between 0 and 1
    """
    pred_sign = torch.sign(value_pred)
    target_sign = torch.sign(value_target)
    correct = (pred_sign == target_sign).float()
    return correct.mean().item()

def compute_elo_rating(
    wins: int,
    losses: int,
    draws: int,
    k_factor: float = 32.0,
    initial_rating: float = 1500.0
) -> Tuple[float, float]:
    """
    Compute Elo rating from game results.
    
    Args:
        wins: Number of wins
        losses: Number of losses
        draws: Number of draws
        k_factor: K-factor for rating updates
        initial_rating: Initial Elo rating
        
    Returns:
        Tuple of:
        - Current Elo rating
        - Rating uncertainty
    """
    total_games = wins + losses + draws
    if total_games == 0:
        return initial_rating, float('inf')
    
    # Compute win rate
    win_rate = (wins + 0.5 * draws) / total_games
    
    # Compute expected score using logistic function
    expected_score = 1 / (1 + 10 ** (-initial_rating / 400))
    
    # Update rating
    rating_change = k_factor * (win_rate - expected_score)
    new_rating = initial_rating + rating_change
    
    # Compute rating uncertainty (standard error)
    uncertainty = k_factor * np.sqrt(win_rate * (1 - win_rate) / total_games)
    
    return new_rating, uncertainty

class TrainingMetrics:
    """
    Comprehensive metrics for model training.
    
    This class combines various metrics and provides methods
    for computing and tracking them during training.
    """
    
    def __init__(self):
        """Initialize the metrics tracker."""
        self.tracker = MetricTracker()
        self.best_metrics = {}
    
    def update(
        self,
        policy_logits: torch.Tensor,
        value_pred: torch.Tensor,
        policy_target: torch.Tensor,
        value_target: torch.Tensor,
        loss_dict: Dict[str, float],
        batch_size: int = 1
    ):
        """
        Update metrics with new batch of predictions.
        
        Args:
            policy_logits: Predicted move logits
            value_pred: Predicted position values
            policy_target: Target move probabilities
            value_target: Target position values
            loss_dict: Dictionary of loss components
            batch_size: Size of the current batch
        """
        # Compute accuracy metrics
        metrics = {
            'policy_accuracy_top1': compute_policy_accuracy(policy_logits, policy_target, top_k=1),
            'policy_accuracy_top3': compute_policy_accuracy(policy_logits, policy_target, top_k=3),
            'value_accuracy': compute_value_accuracy(value_pred, value_target),
        }
        
        # Add loss components
        metrics.update(loss_dict)
        
        # Update tracker
        self.tracker.update(metrics, batch_size)
        
        # Update best metrics
        for name, value in metrics.items():
            if name not in self.best_metrics:
                self.best_metrics[name] = value
            elif 'loss' in name:
                self.best_metrics[name] = min(self.best_metrics[name], value)
            else:
                self.best_metrics[name] = max(self.best_metrics[name], value)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute current metric values.
        
        Returns:
            Dictionary of metric names and their current values
        """
        return self.tracker.compute()
    
    def get_best_metrics(self) -> Dict[str, float]:
        """
        Get best metric values achieved so far.
        
        Returns:
            Dictionary of metric names and their best values
        """
        return self.best_metrics.copy()
    
    def reset(self):
        """Reset all metrics."""
        self.tracker.reset()
        self.best_metrics.clear() 