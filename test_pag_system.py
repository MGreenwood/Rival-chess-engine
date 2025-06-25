#!/usr/bin/env python3
"""
Test script to validate the ultra-dense PAG system end-to-end
"""
import sys
sys.path.insert(0, 'src')

import chess
import torch
import numpy as np

def test_pag_engine():
    """Test the Rust PAG engine directly"""
    print("🔥 Testing Rust PAG Engine...")
    try:
        import rival_ai_engine
        engine = rival_ai_engine.PyPAGEngine()
        
        # Test with starting position
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        result = engine.fen_to_dense_pag(fen)
        
        print(f"✅ PAG extraction successful!")
        print(f"   Node features shape: {result['node_features'].shape}")
        print(f"   Edge features shape: {result['edge_features'].shape}")
        print(f"   Metadata: {result['metadata']}")
        
        # Check if features are actually dense (not mostly zeros)
        node_features = result['node_features']
        non_zero_ratio = np.count_nonzero(node_features) / node_features.size
        print(f"   Non-zero feature ratio: {non_zero_ratio:.3f}")
        
        return True
    except Exception as e:
        print(f"❌ PAG engine test failed: {e}")
        return False

def test_board_conversion():
    """Test Python board conversion pipeline"""
    print("\n🔥 Testing Python Board Conversion...")
    try:
        from rival_ai.utils.board_conversion import board_to_hetero_data, PAG_ENGINE_AVAILABLE
        
        print(f"   PAG_ENGINE_AVAILABLE: {PAG_ENGINE_AVAILABLE}")
        
        board = chess.Board()
        data = board_to_hetero_data(board, use_dense_pag=True)
        
        print(f"✅ Board conversion successful!")
        print(f"   Node types: {data.node_types}")
        print(f"   Edge types: {data.edge_types}")
        
        if 'piece' in data:
            print(f"   Piece features shape: {data['piece'].x.shape}")
        if 'critical_square' in data:
            print(f"   Critical square features shape: {data['critical_square'].x.shape}")
            
        return True
    except Exception as e:
        print(f"❌ Board conversion test failed: {e}")
        return False

def test_model_compatibility():
    """Test if the GNN model can handle the ultra-dense features"""
    print("\n🔥 Testing Model Compatibility...")
    try:
        from rival_ai.models.gnn import ChessGNN
        from rival_ai.utils.board_conversion import board_to_hetero_data
        
        # Create model expecting ultra-dense features
        model = ChessGNN(
            hidden_dim=256,
            num_layers=4,
            num_heads=4,
            dropout=0.1,
            use_ultra_dense_pag=True,
            piece_dim=308,  # Should match actual PAG output
            critical_square_dim=308  # FIXED: Both pieces and squares have 308 features
        )
        
        # Test with real data
        board = chess.Board()
        data = board_to_hetero_data(board, use_dense_pag=True)
        
        print(f"   Model piece_dim: {model.piece_dim}")
        print(f"   Model critical_square_dim: {model.critical_square_dim}")
        print(f"   Actual piece features: {data['piece'].x.shape[1] if 'piece' in data else 'N/A'}")
        print(f"   Actual square features: {data['critical_square'].x.shape[1] if 'critical_square' in data else 'N/A'}")
        
        # Try forward pass
        policy, value = model(data)
        print(f"✅ Model forward pass successful!")
        print(f"   Policy shape: {policy.shape}")
        print(f"   Value shape: {value.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Model compatibility test failed: {e}")
        return False

def test_uci_integration():
    """Test if UCI engine can use real model"""
    print("\n🔥 Testing UCI Integration...")
    try:
        from rival_ai.models.gnn import ChessGNN
        
        # Test the predict_with_board function
        model = ChessGNN(use_ultra_dense_pag=True)
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        
        policy, value = model.predict_with_board(fen)
        
        print(f"✅ UCI prediction successful!")
        print(f"   Policy length: {len(policy)}")
        print(f"   Value: {value}")
        print(f"   Policy sum: {sum(policy):.3f} (should be ~1.0)")
        
        # Check if it's not just uniform random
        policy_variance = np.var(policy)
        print(f"   Policy variance: {policy_variance:.6f} (>0 means not uniform)")
        
        return True
    except Exception as e:
        print(f"❌ UCI integration test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 RivalAI Ultra-Dense PAG System Validation")
    print("=" * 50)
    
    results = []
    results.append(test_pag_engine())
    results.append(test_board_conversion()) 
    results.append(test_model_compatibility())
    results.append(test_uci_integration())
    
    print("\n" + "=" * 50)
    print("📊 FINAL RESULTS:")
    
    tests = ["PAG Engine", "Board Conversion", "Model Compatibility", "UCI Integration"]
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {i+1}. {test}: {status}")
    
    passed = sum(results)
    total = len(results)
    print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 ULTRA-DENSE PAG SYSTEM IS FULLY FUNCTIONAL!")
    else:
        print("⚠️  System has issues that need to be addressed.") 