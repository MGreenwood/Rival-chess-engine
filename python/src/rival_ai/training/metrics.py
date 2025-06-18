"""
Metrics for monitoring model training and evaluation.
"""

import torch
import torch.nn as nn
import chess
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass
import logging
from rival_ai.utils.board_conversion import board_to_hetero_data

logger = logging.getLogger(__name__)

@dataclass
class PRCScore:
    """Piece Relationship Coherence Score components."""
    tactical_coherence: float  # How well model recognizes direct piece interactions
    strategic_coherence: float  # How well model recognizes piece formations/patterns
    move_coherence: float  # How well model's attention aligns with legal moves
    overall_score: float  # Combined weighted score

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

class PieceRelationshipMetrics:
    """Metrics for evaluating piece relationship coherence."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: Optional[str] = None,
        tactical_weight: float = 0.4,
        strategic_weight: float = 0.2,
        move_weight: float = 0.4
    ):
        """Initialize metrics.
        
        Args:
            model: Neural network model
            device: Device to use for computation
            tactical_weight: Weight for tactical coherence
            strategic_weight: Weight for strategic coherence
            move_weight: Weight for move coherence
        """
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tactical_weight = tactical_weight
        self.strategic_weight = strategic_weight
        self.move_weight = move_weight
        
        # Register hooks to collect activations
        self.activations = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks to collect model activations."""
        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach()
            return hook
        
        # Register hooks for each layer we want to analyze
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.MultiheadAttention):
                module.register_forward_hook(get_activation(f'attn_{name}'))
            elif isinstance(module, torch.nn.Linear):
                module.register_forward_hook(get_activation(f'linear_{name}'))
    
    def _get_legal_move_mask(self, board: chess.Board) -> torch.Tensor:
        """Get mask of legal moves."""
        mask = torch.zeros(64, 64, device=self.device)
        for move in board.legal_moves:
            mask[move.from_square, move.to_square] = 1.0
        return mask
    
    def _get_attack_defense_mask(self, board: chess.Board) -> torch.Tensor:
        """Get mask of attacking and defending relationships."""
        mask = torch.zeros(64, 64, device=self.device)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                continue
            
            # Get squares this piece attacks
            for target in chess.SQUARES:
                if board.is_attacked_by(piece.color, target):
                    mask[square, target] = 1.0
                if board.is_attacked_by(not piece.color, target):
                    mask[square, target] = -1.0
        return mask
    
    def _get_piece_formation_mask(self, board: chess.Board) -> torch.Tensor:
        """Get mask of piece formation relationships."""
        mask = torch.zeros(64, 64, device=self.device)
        piece_map = board.piece_map()
        
        # Consider piece types and their relative positions
        for square1, piece1 in piece_map.items():
            for square2, piece2 in piece_map.items():
                if square1 == square2:
                    continue
                
                # Same piece type and color
                if piece1.piece_type == piece2.piece_type and piece1.color == piece2.color:
                    mask[square1, square2] = 1.0
                
                # Pawn structure
                if piece1.piece_type == chess.PAWN and piece2.piece_type == chess.PAWN:
                    if piece1.color == piece2.color:
                        # Connected pawns
                        if abs(chess.square_file(square1) - chess.square_file(square2)) == 1:
                            mask[square1, square2] = 1.0
                
                # King and pawn relationships
                if (piece1.piece_type == chess.KING and piece2.piece_type == chess.PAWN) or \
                   (piece1.piece_type == chess.PAWN and piece2.piece_type == chess.KING):
                    if piece1.color == piece2.color:
                        # King protection of pawns
                        if abs(chess.square_file(square1) - chess.square_file(square2)) <= 1:
                            mask[square1, square2] = 1.0
        
        return mask
    
    def _compute_tactical_coherence(self) -> float:
        """Compute tactical coherence from model activations."""
        if not self.activations:
            return 0.0
        
        # Get attention patterns from the last attention layer
        attn_layers = [k for k in self.activations.keys() if k.startswith('attn_')]
        if not attn_layers:
            return 0.0
        
        last_attn = self.activations[attn_layers[-1]]
        
        # Compute attention to attacking/defending relationships
        attack_defense_mask = self._get_attack_defense_mask(self.current_board)
        attention = last_attn.mean(dim=1)  # Average over heads
        
        # Compute correlation
        correlation = torch.corrcoef(
            torch.stack([attention.flatten(), attack_defense_mask.flatten()])
        )[0, 1]
        
        return float(correlation) if not torch.isnan(correlation) else 0.0
    
    def _compute_strategic_coherence(self) -> float:
        """Compute strategic coherence from model activations."""
        if not self.activations:
            return 0.0
        
        # Get attention patterns from the last attention layer
        attn_layers = [k for k in self.activations.keys() if k.startswith('attn_')]
        if not attn_layers:
            return 0.0
        
        last_attn = self.activations[attn_layers[-1]]
        
        # Compute attention to piece formations
        formation_mask = self._get_piece_formation_mask(self.current_board)
        attention = last_attn.mean(dim=1)  # Average over heads
        
        # Compute correlation
        correlation = torch.corrcoef(
            torch.stack([attention.flatten(), formation_mask.flatten()])
        )[0, 1]
        
        return float(correlation) if not torch.isnan(correlation) else 0.0
    
    def _compute_move_coherence(self) -> float:
        """Compute move coherence from model activations."""
        if not self.activations:
            return 0.0
        
        # Get attention patterns from the last attention layer
        attn_layers = [k for k in self.activations.keys() if k.startswith('attn_')]
        if not attn_layers:
            return 0.0
        
        last_attn = self.activations[attn_layers[-1]]
        
        # Compute attention to legal moves
        legal_move_mask = self._get_legal_move_mask(self.current_board)
        attention = last_attn.mean(dim=1)  # Average over heads
        
        # Compute correlation
        correlation = torch.corrcoef(
            torch.stack([attention.flatten(), legal_move_mask.flatten()])
        )[0, 1]
        
        return float(correlation) if not torch.isnan(correlation) else 0.0
    
    def compute_prc_score(self, board: chess.Board) -> PRCScore:
        """Compute PRC score for a position.
        
        Args:
            board: Chess board position
            
        Returns:
            PRCScore object containing various coherence metrics
        """
        # Store current board for mask computation
        self.current_board = board
        
        # Clear previous activations
        self.activations.clear()
        
        # Convert board to HeteroData before passing to model
        data = board_to_hetero_data(board)
        data = data.to(self.device)
        
        # Get model predictions
        with torch.no_grad():
            self.model(data)  # This will trigger our hooks
        
        # Compute metrics from the collected activations
        tactical_coherence = self._compute_tactical_coherence()
        strategic_coherence = self._compute_strategic_coherence()
        move_coherence = self._compute_move_coherence()
        
        # Compute overall score
        overall_score = (
            self.tactical_weight * tactical_coherence +
            self.strategic_weight * strategic_coherence +
            self.move_weight * move_coherence
        ) / (self.tactical_weight + self.strategic_weight + self.move_weight)
        
        return PRCScore(
            tactical_coherence=tactical_coherence,
            strategic_coherence=strategic_coherence,
            move_coherence=move_coherence,
            overall_score=overall_score
        )
    
    def analyze_game(self, game_moves: List[chess.Move], initial_board: chess.Board) -> Dict[str, List[float]]:
        """Analyze a game and compute PRC scores for each position.
        
        Args:
            game_moves: List of moves in the game
            initial_board: Initial board position
            
        Returns:
            Dictionary of lists containing PRC scores for each position
        """
        board = initial_board.copy()
        scores = {
            'tactical': [],
            'strategic': [],
            'move': [],
            'overall': []
        }
        
        # Compute score for initial position
        score = self.compute_prc_score(board)
        scores['tactical'].append(score.tactical_coherence)
        scores['strategic'].append(score.strategic_coherence)
        scores['move'].append(score.move_coherence)
        scores['overall'].append(score.overall_score)
        
        # Compute scores for each position after a move
        for move in game_moves:
            board.push(move)
            score = self.compute_prc_score(board)
            scores['tactical'].append(score.tactical_coherence)
            scores['strategic'].append(score.strategic_coherence)
            scores['move'].append(score.move_coherence)
            scores['overall'].append(score.overall_score)
        
        return scores
    
    def __del__(self):
        """Clean up hooks when object is destroyed."""
        try:
            # Clear activations dictionary
            self.activations.clear()
            
            # Remove hooks from model if it still exists
            if hasattr(self, 'model') and self.model is not None:
                for name, module in self.model.named_modules():
                    if isinstance(module, torch.nn.MultiheadAttention) or isinstance(module, torch.nn.Linear):
                        # Get the hook handle if it exists
                        hook_handle = getattr(module, '_forward_hooks', {}).get(f'attn_{name}' if isinstance(module, torch.nn.MultiheadAttention) else f'linear_{name}')
                        if hook_handle is not None:
                            try:
                                hook_handle.remove()
                            except (AttributeError, RuntimeError):
                                # Hook may already be removed or model deleted
                                pass
        except Exception as e:
            # Log but don't raise - this is cleanup code
            logger.debug(f"Error during PieceRelationshipMetrics cleanup: {e}") 