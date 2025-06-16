"""
Checkpoint module for saving and loading model checkpoints.
"""

import os
import logging
import torch
from typing import Dict, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    save_dir: str,
    experiment_name: str,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    is_best: bool = False,
    additional_info: Optional[Dict[str, Any]] = None,
) -> str:
    """Save a model checkpoint.
    
    Args:
        model: The model to save
        optimizer: The optimizer to save
        epoch: Current epoch number
        metrics: Dictionary of training metrics
        save_dir: Directory to save checkpoints
        experiment_name: Name of the experiment
        scheduler: Optional learning rate scheduler
        is_best: Whether this is the best model so far
        additional_info: Optional additional information to save
        
    Returns:
        Path to the saved checkpoint
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Prepare checkpoint data
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'timestamp': datetime.now().isoformat(),
        'experiment_name': experiment_name,
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if additional_info is not None:
        checkpoint.update(additional_info)
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(
        save_dir,
        f'{experiment_name}_epoch_{epoch}.pt'
    )
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    # Save best model if this is the best so far
    if is_best:
        best_model_path = os.path.join(save_dir, f'{experiment_name}_best.pt')
        torch.save(checkpoint, best_model_path)
        logger.info(f"Saved best model to {best_model_path}")
    
    return checkpoint_path

def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Load a model checkpoint.
    
    Args:
        path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to load checkpoint to (if None, uses model's device)
        
    Returns:
        Dictionary containing checkpoint data
    """
    if device is None:
        device = next(model.parameters()).device
    
    checkpoint = torch.load(path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded model state from epoch {checkpoint['epoch']}")
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info("Loaded optimizer state")
    
    # Load scheduler state if provided
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info("Loaded scheduler state")
    
    return checkpoint

def get_latest_checkpoint(save_dir: str, experiment_name: str) -> Optional[str]:
    """Get the path to the latest checkpoint for an experiment.
    
    Args:
        save_dir: Directory containing checkpoints
        experiment_name: Name of the experiment
        
    Returns:
        Path to latest checkpoint if found, None otherwise
    """
    if not os.path.exists(save_dir):
        return None
    
    # Find all checkpoints for this experiment
    checkpoints = [
        f for f in os.listdir(save_dir)
        if f.startswith(f'{experiment_name}_epoch_') and f.endswith('.pt')
    ]
    
    if not checkpoints:
        return None
    
    # Sort by epoch number and get latest
    latest = sorted(
        checkpoints,
        key=lambda x: int(x.split('_')[-1].split('.')[0])
    )[-1]
    
    return os.path.join(save_dir, latest)

def get_best_checkpoint(save_dir: str, experiment_name: str) -> Optional[str]:
    """Get the path to the best checkpoint for an experiment.
    
    Args:
        save_dir: Directory containing checkpoints
        experiment_name: Name of the experiment
        
    Returns:
        Path to best checkpoint if found, None otherwise
    """
    best_path = os.path.join(save_dir, f'{experiment_name}_best.pt')
    return best_path if os.path.exists(best_path) else None 