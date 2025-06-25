#!/usr/bin/env python3
"""
Create a fresh model with correct dimensions to replace the mismatched one.
"""

import sys
import torch
from pathlib import Path

# Add the src directory to the path
script_dir = Path(__file__).parent
python_src = script_dir.parent / 'src'
sys.path.insert(0, str(python_src))

from rival_ai.models import ChessGNN

def create_fresh_model():
    """Create a fresh model with correct dimensions."""
    print("🚀 Creating fresh ChessGNN with correct dimensions...")
    
    # Create model with correct dimensions
    model = ChessGNN(
        hidden_dim=256,
        num_layers=10,
        num_heads=4,
        dropout=0.1,
        use_ultra_dense_pag=True,
        piece_dim=308,  # Correct piece dimension
        critical_square_dim=95  # Correct critical square dimension
    )
    
    print(f"✅ Model created with:")
    print(f"   Piece embedding: {model.piece_embedding.weight.shape}")
    print(f"   Square embedding: {model.square_embedding.weight.shape}")
    
    # Save the fresh model
    models_dir = Path('../models')
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / 'latest_trained_model.pt'
    
    # Save with proper checkpoint format
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': 0,
        'loss': float('inf'),
        'optimizer_state_dict': None,
        'dimensions': {
            'piece_dim': 308,
            'critical_square_dim': 95,
            'hidden_dim': 256
        }
    }
    
    torch.save(checkpoint, model_path)
    print(f"💾 Saved fresh model to: {model_path}")
    print("🎯 Model is now compatible with 308-dimensional PAG features!")
    
    return str(model_path)

if __name__ == '__main__':
    create_fresh_model() 