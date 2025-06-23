#!/usr/bin/env python3
"""
Unified System Usage Guide
Shows how to use the new unified game storage and training system.
"""

import sys
from pathlib import Path

# Add the src directory to the path
script_dir = Path(__file__).parent
python_src = script_dir.parent / 'src'
sys.path.insert(0, str(python_src))

def print_usage_guide():
    print("🎯 UNIFIED CHESS AI SYSTEM - USAGE GUIDE")
    print("=" * 50)
    print()
    
    print("📦 WHAT'S NEW:")
    print("  ✅ One Storage Location: training_games/unified/")
    print("  ✅ One Format: Compressed JSON batches (1000 games each)")
    print("  ✅ One Model: models/latest_trained_model.pt (all engines)")
    print("  ✅ Auto-Training: Games → Batches → Training → New Model → Reload")
    print("  ✅ No More Mess: PKL files and scattered JSON files eliminated")
    print()
    
    print("🔄 MIGRATION STEPS:")
    print("  1. Migrate existing games:")
    print("     python migrate_to_unified_storage.py --dry-run")
    print("     python migrate_to_unified_storage.py")
    print()
    print("  2. Clean up old files (optional):")
    print("     python cleanup_training_games.py")
    print()
    
    print("🚀 RUNNING THE SYSTEM:")
    print("  1. Start the server (uses unified system automatically):")
    print("     cd ../engine")
    print("     cargo run --bin server")
    print()
    print("  2. Server will automatically:")
    print("     - Save all games to unified storage")
    print("     - Train when 5000+ games ready")
    print("     - Reload both engines with new model")
    print("     - Clean up training batches after use")
    print()
    
    print("📊 MONITORING:")
    print("  - Check storage: ls ../training_games/unified/")
    print("  - Check archives: ls ../training_games/archives/")
    print("  - Check model: ls ../models/")
    print("  - Server stats: http://localhost:3000/stats")
    print()
    
    print("🎮 ENGINE BEHAVIOR:")
    print("  - Single-player: Fast direct policy")
    print("  - Community: Stronger MCTS search")
    print("  - Model: SAME for both (latest_trained_model.pt)")
    print("  - Training: Uses ALL game types together")
    print()
    
    print("💾 DISK SPACE SAVINGS:")
    print("  - Before: 100GB+ scattered files")
    print("  - After: ~10GB compressed batches")
    print("  - Compression: ~90% space reduction")
    print("  - Auto-cleanup: Training batches deleted after use")
    print()
    
    print("🔧 TROUBLESHOOTING:")
    print("  - Old games not training? Run migration script")
    print("  - Storage full? Check archives/ for old training batches")
    print("  - Training not starting? Check unified/ for batch files")
    print("  - Models out of sync? Server auto-reloads after training")

def check_system_status():
    print("\n🔍 SYSTEM STATUS CHECK:")
    print("-" * 30)
    
    # Check unified storage
    unified_dir = Path("../training_games/unified")
    if unified_dir.exists():
        batch_files = list(unified_dir.glob("batch_*.json.gz"))
        print(f"  📦 Unified batches: {len(batch_files)} files")
        
        # Estimate games
        total_games = len(batch_files) * 1000  # Approximate
        print(f"  🎮 Estimated games: ~{total_games}")
    else:
        print("  ❌ Unified storage not found - run migration first")
    
    # Check model
    model_file = Path("../models/latest_trained_model.pt")
    if model_file.exists():
        size_mb = model_file.stat().st_size / (1024 * 1024)
        print(f"  🧠 Unified model: {size_mb:.1f}MB")
    else:
        print("  ❌ Unified model not found")
    
    # Check old files
    old_pkl = list(Path("../training_games").glob("*.pkl"))
    old_json = list(Path("../training_games/single_player").glob("*.json")) if Path("../training_games/single_player").exists() else []
    
    if old_pkl or old_json:
        print(f"  ⚠️  Old files remaining: {len(old_pkl)} PKL, {len(old_json)} JSON")
        print("     Consider running migration script")
    else:
        print("  ✅ No old files found - system is clean")

if __name__ == '__main__':
    print_usage_guide()
    check_system_status()
    
    print("\n🎉 Ready to use the unified system!")
    print("   Run: python migrate_to_unified_storage.py --dry-run")
    print("   Then: python migrate_to_unified_storage.py") 