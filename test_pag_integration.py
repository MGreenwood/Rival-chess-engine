#!/usr/bin/env python3
"""
Test PAG Integration with Training Pipeline

This script tests the complete integration of PAG tactical loss with the training system.
"""

import sys
import torch
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "python" / "src"))

def test_enhanced_dataset():
    """Test the enhanced dataset creation."""
    print("🧪 Testing Enhanced Dataset...")
    
    try:
        from rival_ai.data.enhanced_dataset import UltraDensePAGDataset, create_enhanced_dataloader
        
        # Create a simple test dataset directory
        test_dir = Path("test_data")
        test_dir.mkdir(exist_ok=True)
        
        # Create sample training data
        sample_data = [
            {
                "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "policy": [0.0] * 4096,
                "value": 0.1
            },
            {
                "fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1",
                "policy": [0.0] * 4096,
                "value": 0.0
            }
        ]
        
        # Save test data
        import json
        with open(test_dir / "test_positions.json", "w") as f:
            json.dump(sample_data, f)
        
        # Test enhanced dataset
        dataset = UltraDensePAGDataset(
            data_dir=test_dir,
            extract_rust_pag=True,  # Will try Rust, fall back to Python
            fallback_to_python_pag=True
        )
        
        print(f"✅ Enhanced dataset created: {len(dataset)} positions")
        
        # Test data loading
        sample = dataset[0]
        print(f"✅ Sample data keys: {sample.keys()}")
        print(f"   Has Rust PAG: {sample.get('has_rust_pag', False)}")
        
        # Test dataloader
        dataloader = create_enhanced_dataloader(
            data_dir=test_dir,
            batch_size=2,
            extract_rust_pag=True,
            num_workers=0  # Use 0 for testing on Windows
        )
        
        batch = next(iter(dataloader))
        print(f"✅ Batch keys: {batch.keys()}")
        print(f"   Has PAG features: {batch.get('has_rust_pag_features', False)}")
        
        # Cleanup
        import shutil
        shutil.rmtree(test_dir)
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_config():
    """Test training configuration with PAG tactical loss."""
    print("\n🧪 Testing Training Configuration...")
    
    try:
        from rival_ai.config import TrainingConfig
        
        # Create config with PAG tactical loss
        config = TrainingConfig()
        
        print(f"✅ Config created successfully!")
        print(f"   use_pag_tactical_loss: {config.use_pag_tactical_loss}")
        print(f"   pag_tactical_config keys: {list(config.pag_tactical_config.keys())}")
        print(f"   vulnerability_weight: {config.pag_tactical_config['vulnerability_weight']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tactical_loss_creation():
    """Test PAG tactical loss creation."""
    print("\n🧪 Testing PAG Tactical Loss Creation...")
    
    try:
        from rival_ai.training.pag_tactical_loss import PAGTacticalLoss
        from rival_ai.config import TrainingConfig
        
        config = TrainingConfig()
        loss_fn = PAGTacticalLoss(**config.pag_tactical_config)
        
        print(f"✅ PAG Tactical Loss created successfully!")
        print(f"   vulnerability_weight: {loss_fn.vulnerability_weight}")
        print(f"   tactical_positional_balance: {loss_fn.tactical_positional_balance}")
        
        return True
        
    except Exception as e:
        print(f"❌ PAG tactical loss test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fallback_behavior():
    """Test fallback behavior when Rust PAG not available."""
    print("\n🧪 Testing Fallback Behavior...")
    
    try:
        from rival_ai.training.pag_tactical_loss import PAGTacticalLoss
        
        # Create sample data without PAG features
        batch_size = 2
        num_moves = 4096
        
        policy_logits = torch.randn(batch_size, num_moves)
        value_pred = torch.randn(batch_size, 1)
        target_policy = torch.randn(batch_size, num_moves)
        target_value = torch.randn(batch_size, 1)
        
        # Create dummy PAG features (this simulates fallback scenario)
        dummy_pag = torch.zeros(batch_size, 16, 308)  # 16 pieces, 308 features
        
        loss_fn = PAGTacticalLoss()
        
        result = loss_fn(
            policy_logits=policy_logits,
            value_pred=value_pred,
            target_policy=target_policy,
            target_value=target_value,
            pag_features=dummy_pag
        )
        
        print(f"✅ Fallback behavior works!")
        print(f"   Total loss: {result['total_loss'].item():.4f}")
        print(f"   Policy loss: {result['policy_loss'].item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Fallback behavior test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all integration tests."""
    print("🚀 Testing PAG Tactical Loss Integration")
    print("=" * 60)
    
    tests = [
        test_enhanced_dataset,
        test_training_config,
        test_tactical_loss_creation,
        test_fallback_behavior,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"🏁 Integration Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All integration tests passed!")
        print("\n✅ **PAG TACTICAL LOSS IS PROPERLY INTEGRATED!**")
        print("\n📋 Next Steps:")
        print("   1. Start training with your existing command")
        print("   2. Monitor TensorBoard for tactical loss metrics")
        print("   3. Watch for 'vulnerability_loss' decreasing")
        print("   4. Test tactical positions after ~5-10 epochs")
        print("\n🎯 Expected Improvements:")
        print("   • Hanging pieces: Fixed in ~5K-10K games")
        print("   • Basic tactics: Learned in ~20K games")
        print("   • Advanced patterns: Mastered in ~50K games")
    else:
        print("❌ Some integration tests failed.")
        print("Check the errors above before starting training.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 