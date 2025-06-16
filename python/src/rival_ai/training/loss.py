"""
Loss functions for training the chess model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Union, List
from dataclasses import dataclass
import chess

@dataclass
class LossConfig:
    """Configuration for loss functions."""
    policy_weight: float = 1.0
    value_weight: float = 1.0
    entropy_weight: float = 0.01
    mse_weight: float = 1.0
    kl_weight: float = 0.1

class PolicyValueLoss(nn.Module):
    """Combined policy and value loss for chess model training.
    
    This loss function combines:
    1. Policy loss (cross-entropy)
    2. Value loss (MSE)
    3. Entropy regularization
    4. KL divergence (if teacher model provided)
    """
    
    def __init__(
        self,
        policy_weight: float = 1.0,
        value_weight: float = 1.0,
        entropy_weight: float = 0.01,
        repetition_penalty_weight: float = 0.2,
        forward_progress_weight: float = 0.1,
        teacher_model: Optional[nn.Module] = None,
        temperature: float = 1.0,
        l2_weight: float = 1e-4
    ):
        """Initialize loss function.
        
        Args:
            policy_weight: Weight for policy loss
            value_weight: Weight for value loss
            entropy_weight: Weight for entropy regularization
            repetition_penalty_weight: Weight for repetition penalty
            forward_progress_weight: Weight for forward progress reward
            teacher_model: Optional teacher model for distillation
            temperature: Temperature for soft targets
            l2_weight: Weight for L2 regularization
        """
        super().__init__()
        self.policy_weight = policy_weight
        self.value_weight = value_weight
        self.entropy_weight = entropy_weight
        self.repetition_penalty_weight = repetition_penalty_weight
        self.forward_progress_weight = forward_progress_weight
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.l2_weight = l2_weight
        
        # Freeze teacher model if provided
        if teacher_model is not None:
            for param in teacher_model.parameters():
                param.requires_grad = False
    
    def _compute_repetition_penalty(
        self,
        policy_pred: torch.Tensor,
        board_states: List[chess.Board],
        move_count: int
    ) -> torch.Tensor:
        """Compute penalty for moves that lead to repeated positions."""
        penalty = torch.tensor(0.0, device=policy_pred.device)
        
        # Only apply penalty in opening/middlegame
        if move_count < 30:
            # Get current position
            current_board = board_states[-1]
            current_fen = current_board.fen()
            
            # Check last 10 positions for repetition
            for prev_board in board_states[-11:-1]:  # Skip current position
                if prev_board.fen() == current_fen:
                    # Find moves that would lead to this repeated position
                    for move in current_board.legal_moves:
                        move_idx = self._move_to_index(move)
                        if move_idx is not None:
                            # Apply penalty to moves that lead to repetition
                            penalty += policy_pred[move_idx] * self.repetition_penalty_weight
        
        return penalty

    def _compute_forward_progress_reward(
        self,
        policy_pred: torch.Tensor,
        board: chess.Board,
        move_count: int
    ) -> torch.Tensor:
        """Compute reward for moves that advance pawns in the opening."""
        reward = torch.tensor(0.0, device=policy_pred.device)
        
        # Only apply in opening
        if move_count < 10:
            for move in board.legal_moves:
                move_idx = self._move_to_index(move)
                if move_idx is not None:
                    # Check if move is a pawn advance
                    piece = board.piece_at(move.from_square)
                    if piece and piece.piece_type == chess.PAWN:
                        # Calculate progress
                        from_rank = chess.square_rank(move.from_square)
                        to_rank = chess.square_rank(move.to_square)
                        if piece.color == chess.WHITE:
                            progress = (to_rank - from_rank) / 7.0
                        else:
                            progress = (from_rank - to_rank) / 7.0
                        
                        if progress > 0:  # Only reward forward movement
                            reward += policy_pred[move_idx] * progress * self.forward_progress_weight
        
        return reward

    def _move_to_index(self, move: chess.Move) -> Optional[int]:
        """Convert a chess move to policy index."""
        from_sq = move.from_square
        to_sq = move.to_square
        promotion = move.promotion
        
        if promotion:
            # Promotion moves are encoded after regular moves
            piece_offset = promotion - 1  # promotion = 1-4, offset = 0-3
            base = 4096 + (from_sq * 64 + to_sq) * 4
            return base + piece_offset
        else:
            # Regular moves are encoded as from * 64 + to
            return from_sq * 64 + to_sq

    def forward(
        self,
        policy_pred: torch.Tensor,
        value_pred: torch.Tensor,
        policy_target: torch.Tensor,
        value_target: torch.Tensor,
        board_states: List[chess.Board],
        move_count: int,
        model: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the combined loss with repetition penalties and forward progress rewards."""
        # Policy loss (cross-entropy)
        policy_loss = F.cross_entropy(policy_pred, policy_target)
        
        # Value loss (MSE)
        value_loss = F.mse_loss(value_pred, value_target)
        
        # Entropy regularization
        entropy = 0.0
        if model is not None:
            with torch.no_grad():
                policy_probs = F.softmax(policy_pred, dim=-1)
                entropy = -torch.sum(policy_probs * torch.log(policy_probs + 1e-10))
        
        # Repetition penalty
        repetition_penalty = self._compute_repetition_penalty(
            policy_pred,
            board_states,
            move_count
        )
        
        # Forward progress reward
        forward_progress_reward = self._compute_forward_progress_reward(
            policy_pred,
            board_states[-1],
            move_count
        )
        
        # KL divergence loss (if teacher model provided)
        kl_loss = 0.0
        if self.teacher_model is not None and model is not None:
            with torch.no_grad():
                teacher_policy, teacher_value = self.teacher_model(model)
                teacher_probs = F.softmax(teacher_policy / self.temperature, dim=-1)
            
            student_log_probs = F.log_softmax(policy_pred / self.temperature, dim=-1)
            kl_loss = F.kl_div(
                student_log_probs,
                teacher_probs,
                reduction='batchmean'
            ) * (self.temperature ** 2)
        
        # Combine losses
        total_loss = (
            self.policy_weight * policy_loss +
            self.value_weight * value_loss -
            self.entropy_weight * entropy +
            repetition_penalty -
            forward_progress_reward +
            kl_loss
        )
        
        # Return loss components for monitoring
        loss_dict = {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'repetition_penalty': repetition_penalty.item(),
            'forward_progress_reward': forward_progress_reward.item(),
            'kl_loss': kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss,
        }
        
        return total_loss, loss_dict

class PAGPolicyValueLoss(PolicyValueLoss):
    """Policy and value loss with PAG (Positional Adjacency Graph) features.
    
    This loss function extends PolicyValueLoss to include:
    1. PAG feature prediction loss
    2. Graph structure preservation loss
    """
    
    def __init__(
        self,
        config: Optional[LossConfig] = None,
        teacher_model: Optional[nn.Module] = None,
        temperature: float = 1.0,
        pag_weight: float = 0.1,
        graph_weight: float = 0.1
    ):
        """Initialize PAG loss function.
        
        Args:
            config: Loss configuration
            teacher_model: Optional teacher model for distillation
            temperature: Temperature for soft targets
            pag_weight: Weight for PAG feature loss
            graph_weight: Weight for graph structure loss
        """
        super().__init__(config.policy_weight, config.value_weight, config.entropy_weight, 0.0, 0.0, teacher_model, temperature, 0.0)
        self.pag_weight = pag_weight
        self.graph_weight = graph_weight
    
    def forward(
        self,
        policy_pred: torch.Tensor,
        value_pred: torch.Tensor,
        pag_pred: Dict[str, torch.Tensor],
        policy_target: torch.Tensor,
        value_target: torch.Tensor,
        pag_target: Dict[str, torch.Tensor],
        model: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the combined loss with PAG features.
        
        Args:
            policy_pred: Predicted policy logits [batch_size, num_moves]
            value_pred: Predicted values [batch_size, 1]
            pag_pred: Predicted PAG features
            policy_target: Target policy probabilities [batch_size, num_moves]
            value_target: Target values [batch_size, 1]
            pag_target: Target PAG features
            model: Optional model for entropy computation
            
        Returns:
            Tuple of (total_loss, loss_components)
        """
        # Get base loss
        base_loss, components = super().forward(
            policy_pred=policy_pred,
            value_pred=value_pred,
            policy_target=policy_target,
            value_target=value_target,
            model=model
        )
        
        # PAG feature loss
        pag_loss = 0.0
        for key in pag_pred:
            if key in pag_target:
                pag_loss += F.mse_loss(pag_pred[key], pag_target[key])
        
        # Graph structure loss (if applicable)
        graph_loss = 0.0
        if 'edge_index' in pag_pred and 'edge_index' in pag_target:
            # Compute graph structure similarity
            pred_adj = torch.sparse_coo_tensor(
                pag_pred['edge_index'],
                torch.ones(pag_pred['edge_index'].size(1)),
                size=(pag_pred['edge_index'].max() + 1,) * 2
            ).to_dense()
            
            target_adj = torch.sparse_coo_tensor(
                pag_target['edge_index'],
                torch.ones(pag_target['edge_index'].size(1)),
                size=(pag_target['edge_index'].max() + 1,) * 2
            ).to_dense()
            
            graph_loss = F.mse_loss(pred_adj, target_adj)
        
        # Combine all losses
        total_loss = (
            base_loss +
            self.pag_weight * pag_loss +
            self.graph_weight * graph_loss
        )
        
        # Update components
        components.update({
            'pag_loss': pag_loss.item() if isinstance(pag_loss, torch.Tensor) else pag_loss,
            'graph_loss': graph_loss.item() if isinstance(graph_loss, torch.Tensor) else graph_loss,
            'total_loss': total_loss.item()
        })
        
        return total_loss, components

class PAGFeatureLoss(nn.Module):
    """
    Loss function for predicting PAG features.
    
    This loss function focuses on the accuracy of predicting various PAG features:
    - Piece relationships (attacks, defenses, etc.)
    - Square control
    - Mobility
    - Pawn structure
    """
    
    def __init__(
        self,
        feature_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.feature_weights = feature_weights or {
            'control': 1.0,
            'direct_relation': 1.0,
            'mobility': 1.0,
            'cooperation': 0.8,
            'obstruction': 0.8,
            'vulnerability': 0.8,
            'pawn_structure': 1.0,
        }
        
        # Loss functions for different feature types
        self.control_loss = nn.MSELoss()
        self.relationship_loss = nn.BCEWithLogitsLoss()
        self.mobility_loss = nn.MSELoss()
    
    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the PAG feature loss.
        
        Args:
            pred: Dictionary of predicted PAG features
            target: Dictionary of target PAG features
            
        Returns:
            Tuple of:
            - Total loss
            - Dictionary of individual feature losses
        """
        losses = {}
        
        # Control loss
        if 'control' in pred and 'control' in target:
            losses['control'] = self.control_loss(
                pred['control'],
                target['control']
            )
        
        # Relationship losses
        relationship_types = [
            'direct_relation',
            'cooperation',
            'obstruction',
            'vulnerability',
            'pawn_structure',
        ]
        for rel_type in relationship_types:
            if rel_type in pred and rel_type in target:
                losses[rel_type] = self.relationship_loss(
                    pred[rel_type],
                    target[rel_type]
                )
        
        # Mobility loss
        if 'mobility' in pred and 'mobility' in target:
            losses['mobility'] = self.mobility_loss(
                pred['mobility'],
                target['mobility']
            )
        
        # Weight and combine losses
        total_loss = sum(
            self.feature_weights.get(k, 1.0) * v
            for k, v in losses.items()
        )
        
        # Return total loss and individual components
        loss_components = {
            'total': total_loss.item(),
            **{k: v.item() for k, v in losses.items()}
        }
        
        return total_loss, loss_components 