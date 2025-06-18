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
        # Policy loss (cross-entropy) - FIXED: Use proper target format
        # policy_target should be class indices, not probabilities
        if policy_target.dim() == 2 and policy_target.size(1) == policy_logits.size(1):
            # If policy_target is probabilities, convert to class indices
            policy_target_indices = torch.argmax(policy_target, dim=1)
        else:
            # Assume policy_target is already class indices
            policy_target_indices = policy_target
            
        policy_loss = F.cross_entropy(policy_logits, policy_target_indices)
        
        # Value loss (MSE) - FIXED: Ensure proper scaling
        value_loss = F.mse_loss(value_pred.squeeze(), value_target.squeeze())
        
        # Policy entropy (for exploration) - FIXED: Proper entropy calculation
        policy_probs = F.softmax(policy_logits, dim=-1)
        entropy = -torch.sum(policy_probs * torch.log(policy_probs + 1e-8), dim=-1).mean()
        
        # L2 regularization
        l2_reg = torch.tensor(0.0, device=policy_logits.device)
        if model is not None and self.l2_weight > 0:
            for param in model.parameters():
                l2_reg += torch.norm(param, p=2)
            l2_reg = self.l2_weight * l2_reg
        
        # Combine losses with proper scaling
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

class ImprovedPolicyValueLoss(nn.Module):
    """
    Improved policy and value loss with better scaling and normalization.
    
    This version addresses the high loss issue by:
    1. Proper policy loss calculation
    2. Better value loss scaling
    3. Improved loss weighting
    4. Target validation
    """
    
    def __init__(
        self,
        policy_weight: float = 1.0,
        value_weight: float = 0.5,  # Reduced from 1.0
        l2_weight: float = 1e-4,
        entropy_weight: float = 0.01,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.policy_weight = policy_weight
        self.value_weight = value_weight
        self.l2_weight = l2_weight
        self.entropy_weight = entropy_weight
        self.temperature = temperature
        
    def forward(
        self,
        policy_logits: torch.Tensor,
        value_pred: torch.Tensor,
        policy_target: torch.Tensor,
        value_target: torch.Tensor,
        model: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute improved combined loss.
        """
        # Validate inputs
        if policy_logits.size(0) != value_pred.size(0):
            raise ValueError(f"Batch size mismatch: policy_logits {policy_logits.size(0)} vs value_pred {value_pred.size(0)}")
            
        # Policy loss with proper target handling
        if policy_target.dim() == 2:
            # Convert probability distribution to class indices
            policy_target_indices = torch.argmax(policy_target, dim=1)
        else:
            policy_target_indices = policy_target
            
        # Ensure targets are within valid range
        if policy_target_indices.max() >= policy_logits.size(1):
            raise ValueError(f"Policy target index {policy_target_indices.max()} >= num_classes {policy_logits.size(1)}")
            
        policy_loss = F.cross_entropy(policy_logits, policy_target_indices)
        
        # Value loss with proper scaling
        value_pred = value_pred.squeeze()
        value_target = value_target.squeeze()
        
        # Clip values to [-1, 1] range
        value_pred = torch.clamp(value_pred, -1.0, 1.0)
        value_target = torch.clamp(value_target, -1.0, 1.0)
        
        value_loss = F.mse_loss(value_pred, value_target)
        
        # Policy entropy for exploration
        policy_probs = F.softmax(policy_logits / self.temperature, dim=-1)
        entropy = -torch.sum(policy_probs * torch.log(policy_probs + 1e-8), dim=-1).mean()
        
        # L2 regularization
        l2_reg = torch.tensor(0.0, device=policy_logits.device)
        if model is not None and self.l2_weight > 0:
            for param in model.parameters():
                l2_reg += torch.norm(param, p=2)
            l2_reg = self.l2_weight * l2_reg
        
        # Combine losses with improved weighting
        total_loss = (
            self.policy_weight * policy_loss +
            self.value_weight * value_loss -
            self.entropy_weight * entropy +
            l2_reg
        )
        
        # Return detailed loss components
        loss_dict = {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'l2_reg': l2_reg.item() if isinstance(l2_reg, torch.Tensor) else l2_reg,
            'policy_weight': self.policy_weight,
            'value_weight': self.value_weight,
        }
        
        return total_loss, loss_dict 