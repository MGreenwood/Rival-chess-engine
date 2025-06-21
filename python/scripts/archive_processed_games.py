#!/usr/bin/env python3
"""
Archive Processed Games Script
Manually archive games that have been used for training to prevent retraining.
"""

import sys
import argparse
from pathlib import Path

# Add the src directory to the path
script_dir = Path(__file__).parent
python_src = script_dir.parent / 'src'
sys.path.insert(0, str(python_src))

from server_training import ServerTrainingRunner
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Archive processed games')
    parser.add_argument('--games-dir', required=True, help='Directory containing training games')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be archived without actually doing it')
    
    args = parser.parse_args()
    
    runner = ServerTrainingRunner(args.games_dir, '', 0)
    
    if args.dry_run:
        # Show what would be archived
        unprocessed_count = runner.count_unprocessed_games()
        logger.info(f"DRY RUN: Found {unprocessed_count} unprocessed games")
        
        # List files that would be archived
        games_dir = Path(args.games_dir)
        
        # Check single player games
        single_player_dir = games_dir / 'single_player'
        if single_player_dir.exists():
            for game_file in single_player_dir.glob('*.json'):
                processed_file = runner.processed_dir / 'single_player' / game_file.name
                if not processed_file.exists():
                    logger.info(f"Would archive: {game_file}")
        
        # Check self-play games
        for game_file in games_dir.glob('*.pkl'):
            processed_file = runner.processed_dir / game_file.name
            if not processed_file.exists():
                logger.info(f"Would archive: {game_file}")
    else:
        # Actually archive
        logger.info("Archiving processed games...")
        archived_count = runner.archive_processed_games()
        logger.info(f"Archived {archived_count} games")
        
        remaining_count = runner.count_unprocessed_games()
        logger.info(f"Remaining unprocessed games: {remaining_count}")

if __name__ == '__main__':
    main() 