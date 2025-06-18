#!/usr/bin/env python3
"""
Script to fix high loss issues by adjusting loss weights and normalization.
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
from pathlib import Path
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rival_ai.training.losses import ImprovedPolicyValueLoss
from rival_ai.models import ChessGNN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NormalizedPolicyValueLoss(ImprovedPolicyValueLoss):
    """
    Normalized policy and value loss to address high loss values.
    
    This loss function includes:
    1. Policy loss normalization
    2. Value loss scaling
    3. Dynamic weight adjustment
    4. Loss clipping
    """
    
    def __init__(
        self,
        policy_weight: float = 1.0,
        value_weight: float = 0.1,  # Much smaller value weight
        l2_weight: float = 1e-4,
        entropy_weight: float = 0.01,
        temperature: float = 1.0,
        max_loss: float = 10.0,  # Clip losses above this value
    ):
        super().__init__(policy_weight, value_weight, l2_weight, entropy_weight, temperature)
        self.max_loss = max_loss
        
    def forward(
        self,
        policy_logits: torch.Tensor,
        value_pred: torch.Tensor,
        policy_target: torch.Tensor,
        value_target: torch.Tensor,
        model: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute normalized combined loss.
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
            
        # Normalize policy logits to prevent extreme values
        policy_logits = torch.clamp(policy_logits, -10.0, 10.0)
        
        policy_loss = F.cross_entropy(policy_logits, policy_target_indices)
        
        # Value loss with proper scaling and normalization
        value_pred = value_pred.squeeze()
        value_target = value_target.squeeze()
        
        # Clip values to [-1, 1] range
        value_pred = torch.clamp(value_pred, -1.0, 1.0)
        value_target = torch.clamp(value_target, -1.0, 1.0)
        
        # Scale value loss to be much smaller
        value_loss = F.mse_loss(value_pred, value_target) * 0.1
        
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
        
        # Clip total loss to prevent extreme values
        total_loss = torch.clamp(total_loss, 0.0, self.max_loss)
        
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

def create_optimized_loss_config():
    """Create an optimized loss configuration for better training."""
    return {
        'policy_weight': 1.0,
        'value_weight': 0.05,  # Very small value weight
        'entropy_weight': 0.01,
        'l2_weight': 1e-4,
        'temperature': 1.0,
        'max_loss': 5.0,  # Clip at 5.0 instead of 10.0
    }

def test_loss_normalization():
    """Test the normalized loss function with sample data."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create sample data
    batch_size = 32
    num_moves = 5312
    
    policy_logits = torch.randn(batch_size, num_moves, device=device)
    value_pred = torch.randn(batch_size, 1, device=device)
    policy_target = torch.randint(0, num_moves, (batch_size,), device=device)
    value_target = torch.randn(batch_size, 1, device=device) * 0.5  # Scale down values
    
    # Test different loss functions
    loss_functions = {
        'Original': ImprovedPolicyValueLoss(),
        'Normalized': NormalizedPolicyValueLoss(),
        'Optimized': NormalizedPolicyValueLoss(**create_optimized_loss_config()),
    }
    
    logger.info("Testing loss functions with sample data:")
    logger.info(f"Policy logits shape: {policy_logits.shape}")
    logger.info(f"Value pred shape: {value_pred.shape}")
    logger.info(f"Policy target shape: {policy_target.shape}")
    logger.info(f"Value target shape: {value_target.shape}")
    
    for name, loss_fn in loss_functions.items():
        try:
            loss, components = loss_fn(policy_logits, value_pred, policy_target, value_target)
            logger.info(f"\n{name} Loss Function:")
            logger.info(f"  Total loss: {loss.item():.4f}")
            for comp_name, comp_value in components.items():
                if isinstance(comp_value, (int, float)):
                    logger.info(f"  {comp_name}: {comp_value:.4f}")
        except Exception as e:
            logger.error(f"Error with {name} loss function: {e}")

def update_training_config():
    """Update the training configuration for better loss handling."""
    config_updates = {
        'use_improved_loss': True,
        'policy_weight': 1.0,
        'value_weight': 0.05,  # Much smaller value weight
        'entropy_weight': 0.01,
        'l2_weight': 1e-4,
        'learning_rate': 0.0005,  # Slightly lower learning rate
        'grad_clip': 0.5,  # Smaller gradient clipping
    }
    
    logger.info("Recommended training configuration updates:")
    for key, value in config_updates.items():
        logger.info(f"  {key}: {value}")
    
    return config_updates

def main():
    """Main function to run loss optimization."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fix high loss issues in chess model training")
    parser.add_argument("--test", action="store_true", help="Test loss normalization")
    parser.add_argument("--config", action="store_true", help="Show optimized config")
    
    args = parser.parse_args()
    
    if args.test:
        test_loss_normalization()
    
    if args.config:
        update_training_config()
    
    if not args.test and not args.config:
        logger.info("Running loss optimization analysis...")
        test_loss_normalization()
        print("\n" + "="*50)
        update_training_config()

if __name__ == "__main__":
    main() 