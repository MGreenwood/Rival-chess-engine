#!/usr/bin/env python3
"""
Debug batch file reading without PyTorch dependencies
"""
import gzip
import json
from pathlib import Path

def debug_batch_files():
    """Debug what's in each batch file"""
    unified_dir = Path('python/training_games/unified')
    batch_files = sorted(unified_dir.glob('batch_*.json.gz'))
    
    print(f"🔍 Found {len(batch_files)} batch files")
    
    total_games_metadata = 0
    total_games_actual = 0
    
    for batch_file in batch_files:
        try:
            with gzip.open(batch_file, 'rt', encoding='utf-8') as f:
                batch_data = json.load(f)
            
            # Get counts from metadata and actual games array
            metadata_count = batch_data.get("game_count", 0)
            actual_games = batch_data.get("games", [])
            actual_count = len(actual_games)
            
            total_games_metadata += metadata_count
            total_games_actual += actual_count
            
            print(f"📁 {batch_file.name}:")
            print(f"   📊 Metadata count: {metadata_count}")
            print(f"   📊 Actual games: {actual_count}")
            
            if actual_count > 0:
                first_game = actual_games[0]
                print(f"   🎯 First game source: {first_game.get('source', 'unknown')}")
                print(f"   🎯 First game ID: {first_game.get('game_id', 'unknown')}")
            
            if metadata_count != actual_count:
                print(f"   ⚠️ MISMATCH: metadata says {metadata_count}, actual array has {actual_count}")
                
        except Exception as e:
            print(f"❌ Error reading {batch_file.name}: {e}")
    
    print(f"\n📊 SUMMARY:")
    print(f"   📈 Total from metadata: {total_games_metadata}")
    print(f"   📈 Total from actual arrays: {total_games_actual}")
    
    return total_games_actual

if __name__ == "__main__":
    debug_batch_files() 