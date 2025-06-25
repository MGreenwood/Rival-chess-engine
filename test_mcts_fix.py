#!/usr/bin/env python3
"""
Test script to verify the MCTS infinite loop fix.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'python', 'src'))

import torch
import chess
import logging
import time
from rival_ai.models.gnn import ChessGNN
from rival_ai.mcts import MCTS, MCTSConfig

# Set up minimal logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_mcts_fix():
    """Test MCTS with a timeout to ensure it doesn't loop infinitely."""
    try:
        logger.info("🧪 Testing MCTS infinite loop fix...")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Initialize model
        model = ChessGNN(
            hidden_dim=256,
            num_layers=10,  # Full 10-layer architecture 
            num_heads=4,
            piece_dim=308,
            critical_square_dim=95,
            use_ultra_dense_pag=True
        )
        model = model.to(device)
        model.eval()
        
        # Create MCTS config with minimal simulations for fast testing
        mcts_config = MCTSConfig(
            num_simulations=5,  # Very few simulations to test quickly
            c_puct=1.0,
            temperature=1.0,
            dirichlet_alpha=0.3,
            dirichlet_weight=0.25,
            device=device,
            max_time=10.0  # 10 second timeout
        )
        
        # Initialize MCTS
        mcts = MCTS(model, mcts_config)
        
        # Test with starting position
        board = chess.Board()
        logger.info(f"Testing MCTS on: {board.fen()}")
        
        # Test with timeout
        start_time = time.time()
        timeout = 15.0  # 15 second maximum timeout
        
        logger.info("Running MCTS search...")
        policy, value = mcts.get_action_policy(board)
        
        elapsed = time.time() - start_time
        logger.info(f"✅ MCTS completed in {elapsed:.2f} seconds")
        
        if elapsed > timeout:
            logger.error(f"❌ MCTS took too long: {elapsed:.2f}s > {timeout}s")
            return False
        
        # Verify results
        if isinstance(policy, dict) and len(policy) > 0:
            logger.info(f"Policy returned {len(policy)} legal moves")
            logger.info(f"Value: {value:.4f}")
        else:
            logger.error("❌ Invalid policy returned")
            return False
        
        # Test one more position to be sure
        logger.info("Testing after e4...")
        board.push_san("e4")
        
        start_time = time.time()
        policy2, value2 = mcts.get_action_policy(board)
        elapsed2 = time.time() - start_time
        
        logger.info(f"✅ Second search completed in {elapsed2:.2f} seconds")
        logger.info(f"Value after e4: {value2:.4f}")
        
        if elapsed2 > timeout:
            logger.error(f"❌ Second search took too long: {elapsed2:.2f}s > {timeout}s")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_mcts_fix()
    if success:
        print("✅ MCTS infinite loop fix works!")
        sys.exit(0)
    else:
        print("❌ MCTS still has issues")
        sys.exit(1) 