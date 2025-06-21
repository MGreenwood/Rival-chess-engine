#!/usr/bin/env python3
"""
Monitor self-play game generation and training progress.
"""

import os
import time
import json
import pickle
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SelfPlayMonitor:
    def __init__(self, games_dir: str = "../python/training_games"):
        self.games_dir = Path(games_dir)
        self.last_check = datetime.now()
        
    def count_recent_games(self, hours: int = 24) -> Dict[str, int]:
        """Count games generated in the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        counts = {
            'json_games': 0,
            'pkl_games': 0,
            'total_positions': 0
        }
        
        # Check JSON games (from server)
        for json_file in self.games_dir.glob('*.json'):
            if datetime.fromtimestamp(json_file.stat().st_mtime) > cutoff_time:
                counts['json_games'] += 1
                try:
                    with open(json_file, 'r') as f:
                        game_data = json.load(f)
                        counts['total_positions'] += len(game_data.get('moves', []))
                except Exception:
                    pass
        
        # Check PKL games (from Python self-play)
        for pkl_file in self.games_dir.rglob('*.pkl'):
            if datetime.fromtimestamp(pkl_file.stat().st_mtime) > cutoff_time:
                try:
                    with open(pkl_file, 'rb') as f:
                        games = pickle.load(f)
                        counts['pkl_games'] += len(games) if isinstance(games, list) else 1
                except Exception:
                    pass
        
        return counts
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status."""
        experiments_dir = Path("../python/experiments")
        
        status = {
            'active_experiments': 0,
            'latest_checkpoint': None,
            'total_checkpoints': 0
        }
        
        if experiments_dir.exists():
            # Count active experiments (directories modified in last 24 hours)
            cutoff_time = datetime.now() - timedelta(hours=24)
            
            for exp_dir in experiments_dir.iterdir():
                if exp_dir.is_dir() and datetime.fromtimestamp(exp_dir.stat().st_mtime) > cutoff_time:
                    status['active_experiments'] += 1
                    
                    # Look for checkpoints
                    checkpoints_dir = exp_dir / 'checkpoints'
                    if checkpoints_dir.exists():
                        checkpoints = list(checkpoints_dir.glob('*.pt'))
                        status['total_checkpoints'] += len(checkpoints)
                        
                        # Find most recent checkpoint
                        if checkpoints:
                            latest = max(checkpoints, key=lambda x: x.stat().st_mtime)
                            if (status['latest_checkpoint'] is None or 
                                latest.stat().st_mtime > status['latest_checkpoint']['mtime']):
                                status['latest_checkpoint'] = {
                                    'path': str(latest),
                                    'mtime': latest.stat().st_mtime,
                                    'age_hours': (datetime.now() - datetime.fromtimestamp(latest.stat().st_mtime)).total_seconds() / 3600
                                }
        
        return status
    
    def print_status(self):
        """Print current status."""
        print(f"\n=== RivalAI Self-Play Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
        
        # Recent games
        games_24h = self.count_recent_games(24)
        games_1h = self.count_recent_games(1)
        
        print(f"\nGame Generation (Last 24 hours):")
        print(f"  JSON games (server): {games_24h['json_games']}")
        print(f"  PKL games (python): {games_24h['pkl_games']}")
        print(f"  Total positions: {games_24h['total_positions']}")
        
        print(f"\nRecent Activity (Last 1 hour):")
        print(f"  JSON games: {games_1h['json_games']}")
        print(f"  PKL games: {games_1h['pkl_games']}")
        
        # Training status
        training_status = self.get_training_status()
        print(f"\nTraining Status:")
        print(f"  Active experiments: {training_status['active_experiments']}")
        print(f"  Total checkpoints: {training_status['total_checkpoints']}")
        
        if training_status['latest_checkpoint']:
            checkpoint = training_status['latest_checkpoint']
            print(f"  Latest checkpoint: {Path(checkpoint['path']).name}")
            print(f"  Checkpoint age: {checkpoint['age_hours']:.1f} hours")
        else:
            print(f"  Latest checkpoint: None found")
        
        # Recommendations
        print(f"\nRecommendations:")
        if games_1h['json_games'] == 0 and games_1h['pkl_games'] == 0:
            print("  ⚠️  No games generated in the last hour - check self-play system")
        elif games_1h['json_games'] > 0:
            print("  ✓ Server self-play is active")
        
        if training_status['active_experiments'] == 0:
            print("  ⚠️  No active training experiments")
        else:
            print("  ✓ Training system is active")
        
        print("=" * 60)

def main():
    monitor = SelfPlayMonitor()
    
    try:
        while True:
            monitor.print_status()
            time.sleep(300)  # Update every 5 minutes
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

if __name__ == '__main__':
    main() 