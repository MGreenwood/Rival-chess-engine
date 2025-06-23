#!/usr/bin/env python3
"""
Migrate to Unified Storage
Convert all existing game files to unified batched format and clean up the mess.
"""

import sys
import os
from pathlib import Path
import argparse
import logging
from typing import List, Dict, Any
import shutil

# Add the src directory to the path
script_dir = Path(__file__).parent
python_src = script_dir.parent / 'src'
sys.path.insert(0, str(python_src))

from rival_ai.unified_storage import UnifiedGameStorage, GameSource, initialize_unified_storage

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GameMigrator:
    def __init__(self, games_dir: str = "../training_games", batch_size: int = 1000):
        self.games_dir = Path(games_dir)
        self.storage = initialize_unified_storage(str(self.games_dir), batch_size)
        
        # Track what we find
        self.stats = {
            "pkl_files": 0,
            "json_files": 0,
            "games_converted": 0,
            "space_before": 0,
            "space_after": 0,
            "errors": []
        }
        
        logger.info(f"🎯 Game Migrator initialized")
        logger.info(f"   Games directory: {self.games_dir}")
        logger.info(f"   Batch size: {batch_size}")
    
    def scan_existing_games(self) -> Dict[str, List[Path]]:
        """Scan for existing game files"""
        found = {
            "self_play_pkl": [],
            "single_player_json": [],
            "processed_pkl": [],
            "processed_json": [],
            "uci_tournament_pkl": [],
            "experiment_pkl": []
        }
        
        # Self-play PKL files in root
        for pkl_file in self.games_dir.glob("*.pkl"):
            found["self_play_pkl"].append(pkl_file)
            self.stats["space_before"] += pkl_file.stat().st_size
        
        # Single player JSON files
        single_player_dir = self.games_dir / "single_player"
        if single_player_dir.exists():
            for json_file in single_player_dir.glob("*.json"):
                found["single_player_json"].append(json_file)
                self.stats["space_before"] += json_file.stat().st_size
        
        # Processed PKL files
        processed_dir = self.games_dir / "processed"
        if processed_dir.exists():
            for pkl_file in processed_dir.glob("*.pkl"):
                found["processed_pkl"].append(pkl_file)
                self.stats["space_before"] += pkl_file.stat().st_size
            
            # Processed single player JSON
            processed_sp = processed_dir / "single_player"
            if processed_sp.exists():
                for json_file in processed_sp.glob("*.json"):
                    found["processed_json"].append(json_file)
                    self.stats["space_before"] += json_file.stat().st_size
        
        # UCI tournament PKL files
        uci_matches_dir = self.games_dir / "uci_matches"
        if uci_matches_dir.exists():
            for pkl_file in uci_matches_dir.rglob("*.pkl"):
                found["uci_tournament_pkl"].append(pkl_file)
                self.stats["space_before"] += pkl_file.stat().st_size
        
        # Experiment PKL files (the 95GB monster)
        experiments_dir = self.games_dir.parent / "experiments"
        if experiments_dir.exists():
            for pkl_file in experiments_dir.rglob("*.pkl"):
                found["experiment_pkl"].append(pkl_file)
                self.stats["space_before"] += pkl_file.stat().st_size
        
        # Count files
        self.stats["pkl_files"] = (len(found["self_play_pkl"]) + 
                                   len(found["processed_pkl"]) + 
                                   len(found["uci_tournament_pkl"]) +
                                   len(found["experiment_pkl"]))
        self.stats["json_files"] = (len(found["single_player_json"]) + 
                                    len(found["processed_json"]))
        
        return found
    
    def migrate_all_games(self, dry_run: bool = False) -> None:
        """Migrate all existing games to unified format"""
        found = self.scan_existing_games()
        
        total_files = sum(len(file_list) for file_list in found.values())
        space_gb = self.stats["space_before"] / (1024**3)
        
        logger.info(f"📊 Found {total_files} game files ({space_gb:.1f}GB)")
        logger.info(f"   - {len(found['self_play_pkl'])} self-play PKL files")
        logger.info(f"   - {len(found['single_player_json'])} single-player JSON files")
        logger.info(f"   - {len(found['processed_pkl'])} processed PKL files")
        logger.info(f"   - {len(found['processed_json'])} processed JSON files")
        logger.info(f"   - {len(found['uci_tournament_pkl'])} UCI tournament PKL files")
        logger.info(f"   - {len(found['experiment_pkl'])} experiment PKL files")
        
        if dry_run:
            logger.info("🔍 DRY RUN - No files will be migrated")
            return
        
        # Confirm before proceeding
        response = input(f"\n⚠️  Migrate {total_files} files ({space_gb:.1f}GB) to unified format? Type 'yes' to confirm: ")
        if response.lower() != 'yes':
            logger.info("❌ Migration cancelled")
            return
        
        # Migrate each type
        logger.info("🔄 Starting migration...")
        
        # 1. Self-play PKL files (highest quality data)
        self._migrate_pkl_files(found["self_play_pkl"], GameSource.SELF_PLAY)
        
        # 2. Single player JSON files
        self._migrate_json_files(found["single_player_json"], GameSource.SINGLE_PLAYER)
        
        # 3. Processed files (should be deleted after migration since they're duplicates)
        self._migrate_pkl_files(found["processed_pkl"], GameSource.SELF_PLAY, delete_after=True)
        self._migrate_json_files(found["processed_json"], GameSource.SINGLE_PLAYER, delete_after=True)
        
        # 4. UCI tournament games (convert and delete after migration)
        self._migrate_pkl_files(found["uci_tournament_pkl"], GameSource.UCI_TOURNAMENT, delete_after=True)
        
        # 5. Experiment files (convert but keep for now - user can delete manually)
        self._migrate_pkl_files(found["experiment_pkl"], GameSource.SELF_PLAY, delete_after=False)
        
        # Force save any remaining batch
        self.storage.force_save_current_batch()
        
        # Calculate final stats
        self._calculate_final_stats()
        
        logger.info("✅ Migration completed!")
        logger.info(f"🎯 Converted {self.stats['games_converted']} games to unified format")
        logger.info(f"💾 Space usage: {self.stats['space_before']/(1024**3):.1f}GB → {self.stats['space_after']/(1024**3):.1f}GB")
        
        if self.stats["errors"]:
            logger.warning(f"⚠️ {len(self.stats['errors'])} errors occurred during migration")
    
    def _migrate_pkl_files(self, pkl_files: List[Path], source: GameSource, delete_after: bool = False) -> None:
        """Migrate PKL files to unified format"""
        for pkl_file in pkl_files:
            try:
                logger.info(f"🔄 Converting {pkl_file.name}...")
                
                # Convert to unified format
                unified_games = self.storage.convert_legacy_game(pkl_file, source)
                
                if isinstance(unified_games, list):
                    # Multiple games from PKL
                    self.storage.store_multiple_games(unified_games)
                    self.stats["games_converted"] += len(unified_games)
                    logger.info(f"   ✅ Converted {len(unified_games)} games")
                elif unified_games:
                    # Single game
                    self.storage.store_game(unified_games)
                    self.stats["games_converted"] += 1
                    logger.info(f"   ✅ Converted 1 game")
                
                # Delete original if requested
                if delete_after:
                    pkl_file.unlink()
                    logger.info(f"   🗑️ Deleted original file")
                
            except Exception as e:
                error_msg = f"Failed to convert {pkl_file}: {e}"
                logger.error(error_msg)
                self.stats["errors"].append(error_msg)
    
    def _migrate_json_files(self, json_files: List[Path], source: GameSource, delete_after: bool = False) -> None:
        """Migrate JSON files to unified format"""
        for json_file in json_files:
            try:
                logger.info(f"🔄 Converting {json_file.name}...")
                
                # Convert to unified format
                unified_game = self.storage.convert_legacy_game(json_file, source)
                
                if unified_game:
                    self.storage.store_game(unified_game)
                    self.stats["games_converted"] += 1
                    logger.info(f"   ✅ Converted 1 game")
                
                # Delete original if requested
                if delete_after:
                    json_file.unlink()
                    logger.info(f"   🗑️ Deleted original file")
                
            except Exception as e:
                error_msg = f"Failed to convert {json_file}: {e}"
                logger.error(error_msg)
                self.stats["errors"].append(error_msg)
    
    def _calculate_final_stats(self) -> None:
        """Calculate final space usage"""
        # Calculate unified storage size
        unified_size = 0
        for batch_file in self.storage.unified_dir.glob("batch_*.json.gz"):
            unified_size += batch_file.stat().st_size
        
        self.stats["space_after"] = unified_size
    
    def cleanup_old_directories(self, confirm: bool = False) -> None:
        """Clean up old directory structure after migration"""
        if not confirm:
            logger.warning("Use --cleanup to actually delete old directories")
            return
        
        dirs_to_remove = [
            self.games_dir / "processed",
            self.games_dir / "single_player"
        ]
        
        space_freed = 0
        for dir_path in dirs_to_remove:
            if dir_path.exists():
                # Calculate space
                for file_path in dir_path.rglob("*"):
                    if file_path.is_file():
                        space_freed += file_path.stat().st_size
                
                # Remove directory
                shutil.rmtree(dir_path)
                logger.info(f"🗑️ Removed {dir_path}")
        
        logger.info(f"💾 Freed {space_freed/(1024**3):.1f}GB by cleaning up old directories")

def main():
    parser = argparse.ArgumentParser(description='Migrate existing games to unified storage format')
    parser.add_argument('--games-dir', default='../training_games',
                      help='Directory containing training games')
    parser.add_argument('--batch-size', type=int, default=1000,
                      help='Number of games per batch file')
    parser.add_argument('--dry-run', action='store_true',
                      help='Show what would be migrated without doing it')
    parser.add_argument('--cleanup', action='store_true',
                      help='Clean up old directories after migration')
    
    args = parser.parse_args()
    
    migrator = GameMigrator(args.games_dir, args.batch_size)
    
    try:
        # Migrate all games
        migrator.migrate_all_games(dry_run=args.dry_run)
        
        if not args.dry_run and args.cleanup:
            migrator.cleanup_old_directories(confirm=True)
    
    except KeyboardInterrupt:
        logger.info("\nMigration interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 