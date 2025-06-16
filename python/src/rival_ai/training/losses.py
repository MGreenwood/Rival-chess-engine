"""
Loss functions for GNN training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class PolicyValueLoss(nn.Module):
    """
    Combined policy and value loss for chess position evaluation.
    
    This loss function combines:
    1. Policy loss (cross-entropy for move probabilities)
    2. Value loss (MSE for position evaluation)
    3. Optional regularization terms
    """
    
    def __init__(
        self,
        policy_weight: float = 1.0,
        value_weight: float = 1.0,
        l2_weight: float = 1e-4,
        entropy_weight: float = 0.01,
    ):
        """
        Initialize the loss function.
        
        Args:
            policy_weight: Weight for policy loss
            value_weight: Weight for value loss
            l2_weight: Weight for L2 regularization
            entropy_weight: Weight for policy entropy regularization
        """
        super().__init__()
        self.policy_weight = policy_weight
        self.value_weight = value_weight
        self.l2_weight = l2_weight
        self.entropy_weight = entropy_weight
        
    def forward(
        self,
        policy_logits: torch.Tensor,
        value_pred: torch.Tensor,
        policy_target: torch.Tensor,
        value_target: torch.Tensor,
        model: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the combined loss.
        
        Args:
            policy_logits: Predicted move logits [batch_size, num_moves]
            value_pred: Predicted position values [batch_size, 1]
            policy_target: Target move probabilities [batch_size, num_moves]
            value_target: Target position values [batch_size, 1]
            model: Optional model for L2 regularization
            
        Returns:
            Tuple of:
            - Total loss
            - Dictionary of individual loss components
        """
        # Policy loss (cross-entropy)
        policy_loss = F.cross_entropy(policy_logits, policy_target)
        
        # Value loss (MSE)
        value_loss = F.mse_loss(value_pred, value_target)
        
        # Policy entropy (for exploration)
        policy_probs = F.softmax(policy_logits, dim=-1)
        entropy = -torch.sum(policy_probs * torch.log(policy_probs + 1e-8), dim=-1).mean()
        
        # L2 regularization
        l2_reg = torch.tensor(0.0, device=policy_logits.device)
        if model is not None and self.l2_weight > 0:
            for param in model.parameters():
                l2_reg += torch.norm(param, p=2)
            l2_reg = self.l2_weight * l2_reg
        
        # Combine losses
        total_loss = (
            self.policy_weight * policy_loss +
            self.value_weight * value_loss -
            self.entropy_weight * entropy +
            l2_reg
        )
        
        # Return loss components for monitoring
        loss_dict = {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'l2_reg': l2_reg.item() if isinstance(l2_reg, torch.Tensor) else l2_reg,
        }
        
        return total_loss, loss_dict

class KLDivergenceLoss(nn.Module):
    """
    KL divergence loss for policy distillation.
    
    Used when training a student model to mimic a teacher model's
    policy distribution.
    """
    
    def __init__(self, temperature: float = 1.0):
        """
        Initialize the loss function.
        
        Args:
            temperature: Temperature for softmax scaling
        """
        super().__init__()
        self.temperature = temperature
        
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute KL divergence loss.
        
        Args:
            student_logits: Student model logits [batch_size, num_moves]
            teacher_logits: Teacher model logits [batch_size, num_moves]
            
        Returns:
            Tuple of:
            - KL divergence loss
            - Dictionary with loss components
        """
        # Scale logits by temperature
        student_logits = student_logits / self.temperature
        teacher_logits = teacher_logits / self.temperature
        
        # Compute softmax probabilities
        student_probs = F.log_softmax(student_logits, dim=-1)
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        
        # Compute KL divergence
        kl_div = F.kl_div(
            student_probs,
            teacher_probs,
            reduction='batchmean',
            log_target=False,
        ) * (self.temperature ** 2)
        
        # Return loss components
        loss_dict = {
            'kl_div': kl_div.item(),
            'temperature': self.temperature,
        }
        
        return kl_div, loss_dict 