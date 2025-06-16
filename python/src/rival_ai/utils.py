"""
Utility functions for RivalAI chess engine.
"""

import os
import json
import torch
from typing import Dict, Any, Optional
from pathlib import Path

def save_pag(
    pag_data: Dict[str, Any],
    filepath: str,
    device: Optional[torch.device] = None
) -> None:
    """Save PAG data to a JSON file.
    
    Args:
        pag_data: Dictionary containing PAG data
        filepath: Path to save the file
        device: Optional device to move tensors to before saving
    """
    # Convert tensors to CPU and numpy
    if device is not None:
        pag_data = {
            k: v.to(device).cpu().numpy() if torch.is_tensor(v) else v
            for k, v in pag_data.items()
        }
    else:
        pag_data = {
            k: v.cpu().numpy() if torch.is_tensor(v) else v
            for k, v in pag_data.items()
        }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save to JSON
    with open(filepath, 'w') as f:
        json.dump(pag_data, f, indent=2)

def load_pag(
    filepath: str,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """Load PAG data from a JSON file.
    
    Args:
        filepath: Path to the JSON file
        device: Optional device to move tensors to after loading
        
    Returns:
        Dictionary containing PAG data with tensors
    """
    # Load JSON
    with open(filepath, 'r') as f:
        pag_data = json.load(f)
    
    # Convert numpy arrays to tensors
    pag_data = {
        k: torch.tensor(v) if isinstance(v, (list, tuple)) else v
        for k, v in pag_data.items()
    }
    
    # Move tensors to device if specified
    if device is not None:
        pag_data = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in pag_data.items()
        }
    
    return pag_data 