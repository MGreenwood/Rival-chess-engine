#!/usr/bin/env python3
"""
Create a fresh randomly initialized ChessGNN model for RivalAI
"""

import sys
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from rival_ai.models.gnn import ChessGNN
from rival_ai.config import TrainingConfig

def create_fresh_model():
    """Create a fresh, randomly initialized ChessGNN model"""
    print("🆕 Creating fresh ChessGNN model...")
    
    # Create model with default config
    config = {
        'hidden_dim': 256,
        'num_layers': 4,
        'num_heads': 4, 
        'dropout': 0.1
    }
    
    model = ChessGNN(**config)
    model.eval()
    
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create models directory
    models_dir = Path(__file__).parent / 'models'
    models_dir.mkdir(exist_ok=True)
    
    # Save the model
    model_path = models_dir / 'latest_trained_model.pt'
    
    # Create a proper checkpoint structure
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'epoch': 0,
        'training_loss': float('inf'),
        'validation_loss': float('inf'),
        'creation_type': 'fresh_random_init'
    }
    
    torch.save(checkpoint, model_path)
    print(f"💾 Fresh model saved to: {model_path}")
    print(f"🎯 Model is ready for self-play and training!")
    
    return str(model_path)

if __name__ == "__main__":
    model_path = create_fresh_model()
    print(f"\n🚀 Ready to use: {model_path}")
    print("🎮 You can now:")
    print("   1. Run self-play: python scripts/server_self_play.py")
    print("   2. Start server: cd ../engine && cargo run --bin server")
    print("   3. Generate tournament games: python scripts/uci_tournament.py") 