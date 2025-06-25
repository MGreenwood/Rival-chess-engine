#!/usr/bin/env python3
"""
Simple test script to verify model inference works without self-play.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'python', 'src'))

import torch
import chess
import logging
from rival_ai.models.gnn import ChessGNN
from rival_ai.utils.board_conversion import board_to_hetero_data

# Set up minimal logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_simple_model():
    """Test just the model inference without complex self-play."""
    try:
        logger.info("🧪 Testing simple model inference...")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Initialize model
        model = ChessGNN(
            hidden_dim=256,
            num_layers=10,  # Correct 10-layer architecture as specified
            num_heads=4,
            piece_dim=308,
            critical_square_dim=95,
            use_ultra_dense_pag=True
        )
        model = model.to(device)
        model.eval()
        
        # Test with a simple chess position
        board = chess.Board()  # Starting position
        logger.info(f"Testing position: {board.fen()}")
        
        # Convert board to data
        logger.info("Converting board to hetero data...")
        data = board_to_hetero_data(board, use_dense_pag=True)
        data = data.to(device)
        
        # Test model inference
        logger.info("Running model inference...")
        with torch.no_grad():
            policy, value = model(data)
            
        logger.info(f"✅ Success!")
        logger.info(f"Policy shape: {policy.shape}")
        logger.info(f"Value: {value.item():.4f}")
        
        # Test a few moves
        logger.info("Testing after a few moves...")
        board.push_san("e4")
        board.push_san("e5")
        
        data2 = board_to_hetero_data(board, use_dense_pag=True)
        data2 = data2.to(device)
        
        with torch.no_grad():
            policy2, value2 = model(data2)
            
        logger.info(f"After e4 e5 - Value: {value2.item():.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_simple_model()
    if success:
        print("✅ Model inference works correctly!")
        sys.exit(0)
    else:
        print("❌ Model inference has issues")
        sys.exit(1) 