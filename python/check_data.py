#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')

from rival_ai.unified_storage import initialize_unified_storage

print("🔍 Checking unified training data...")
storage = initialize_unified_storage('training_games', batch_size=1000)
training_ready = storage.get_training_ready_count()
total_games = storage.get_total_games()

print(f"📊 Training-ready games: {training_ready}")
print(f"📊 Total games: {total_games}")

if training_ready >= 1000:
    print("✅ Enough data to start training!")
else:
    print(f"⚠️ Need {1000 - training_ready} more games for training") 