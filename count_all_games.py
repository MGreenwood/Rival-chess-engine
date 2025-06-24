#!/usr/bin/env python3
"""
Count all games that the Rust server sees vs Python system
"""
from pathlib import Path

def count_individual_files():
    """Count individual JSON files that Rust server sees"""
    base_dir = Path('python/training_games')
    
    directories = ['single_player', 'community', 'uci_matches']
    total_json_files = 0
    
    print("📁 Individual JSON files:")
    for directory in directories:
        dir_path = base_dir / directory
        if dir_path.exists():
            json_files = list(dir_path.glob('*.json'))
            count = len(json_files)
            total_json_files += count
            print(f"   {directory}: {count} files")
            
            # Show first few file names for debugging
            if count > 0 and count <= 5:
                for json_file in json_files:
                    print(f"      - {json_file.name}")
            elif count > 5:
                for json_file in json_files[:3]:
                    print(f"      - {json_file.name}")
                print(f"      ... and {count-3} more")
        else:
            print(f"   {directory}: directory not found")
    
    return total_json_files

def count_unified_batches():
    """Count games in unified batch files"""
    unified_dir = Path('python/training_games/unified')
    batch_files = list(unified_dir.glob('batch_*.json.gz'))
    
    total_games = 0
    print("\n📦 Unified batch files:")
    
    for batch_file in sorted(batch_files):
        import gzip
        import json
        try:
            with gzip.open(batch_file, 'rt', encoding='utf-8') as f:
                batch_data = json.load(f)
            game_count = batch_data.get("game_count", 0)
            total_games += game_count
            print(f"   {batch_file.name}: {game_count} games")
        except Exception as e:
            print(f"   {batch_file.name}: ERROR - {e}")
    
    return total_games

def main():
    print("🔍 Analyzing game storage discrepancy...\n")
    
    json_count = count_individual_files()
    batch_count = count_unified_batches()
    
    print(f"\n📊 SUMMARY:")
    print(f"   Individual JSON files: {json_count}")
    print(f"   Unified batch games: {batch_count}")
    print(f"   Total (Rust server sees): {json_count + batch_count}")
    print(f"   Python system sees: {batch_count} (only unified)")
    
    print(f"\n🎯 EXPLANATION:")
    print(f"   - Rust server reads BOTH individual files AND unified batches")
    print(f"   - Python training system only reads unified batches")
    print(f"   - That's why there's a {json_count} game difference!")

if __name__ == "__main__":
    main() 