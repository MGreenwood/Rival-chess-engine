#!/usr/bin/env python3
"""
Cleanup script for training game files.
This helps clean up the accumulated PKL and JSON game files.
"""

import sys
import os
from pathlib import Path
import shutil
import zipfile
from datetime import datetime
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cleanup_training_games(games_dir, dry_run=False, keep_archives=True):
    """Clean up training game files.
    
    Args:
        games_dir: Path to the training games directory
        dry_run: If True, only show what would be deleted without actually deleting
        keep_archives: If True, create archives before deleting. If False, just delete.
    """
    games_path = Path(games_dir)
    
    if not games_path.exists():
        logger.error(f"Games directory does not exist: {games_dir}")
        return
    
    # Count files
    pkl_files = list(games_path.glob("*.pkl"))
    json_files = list((games_path / "single_player").glob("*.json")) if (games_path / "single_player").exists() else []
    processed_pkl = list((games_path / "processed").glob("*.pkl")) if (games_path / "processed").exists() else []
    processed_json = list((games_path / "processed" / "single_player").glob("*.json")) if (games_path / "processed" / "single_player").exists() else []
    
    total_files = len(pkl_files) + len(json_files) + len(processed_pkl) + len(processed_json)
    
    if total_files == 0:
        logger.info("✅ No game files to clean up!")
        return
    
    # Calculate sizes
    def get_size_mb(files):
        return sum(f.stat().st_size for f in files) / (1024 * 1024)
    
    pkl_size = get_size_mb(pkl_files)
    json_size = get_size_mb(json_files)
    processed_pkl_size = get_size_mb(processed_pkl)
    processed_json_size = get_size_mb(processed_json)
    total_size = pkl_size + json_size + processed_pkl_size + processed_json_size
    
    logger.info(f"📊 Found {total_files} game files totaling {total_size:.1f}MB:")
    logger.info(f"   - {len(pkl_files)} PKL files ({pkl_size:.1f}MB) in root")
    logger.info(f"   - {len(json_files)} JSON files ({json_size:.1f}MB) in single_player/")
    logger.info(f"   - {len(processed_pkl)} PKL files ({processed_pkl_size:.1f}MB) in processed/")
    logger.info(f"   - {len(processed_json)} JSON files ({processed_json_size:.1f}MB) in processed/single_player/")
    
    if dry_run:
        logger.info("🔍 DRY RUN - No files will be deleted")
    
    if not dry_run:
        response = input(f"\n⚠️  Delete {total_files} files ({total_size:.1f}MB)? Type 'yes' to confirm: ")
        if response.lower() != 'yes':
            logger.info("❌ Cleanup cancelled")
            return
    
    # Create archive if requested
    archive_path = None
    if keep_archives and not dry_run:
        archive_dir = games_path / "archives"
        archive_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = archive_dir / f"cleanup_backup_{timestamp}.zip"
        
        logger.info(f"📦 Creating backup archive: {archive_path}")
        
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Archive all files
            for f in pkl_files:
                zipf.write(f, f.name)
            for f in json_files:
                zipf.write(f, f"single_player/{f.name}")
            for f in processed_pkl:
                zipf.write(f, f"processed/{f.name}")
            for f in processed_json:
                zipf.write(f, f"processed/single_player/{f.name}")
        
        archive_size = archive_path.stat().st_size / (1024 * 1024)
        logger.info(f"✅ Archive created: {archive_size:.1f}MB (compressed from {total_size:.1f}MB)")
    
    # Delete files
    deleted_count = 0
    
    if not dry_run:
        # Delete PKL files in root
        for f in pkl_files:
            f.unlink()
            deleted_count += 1
        
        # Delete JSON files in single_player
        for f in json_files:
            f.unlink()
            deleted_count += 1
        
        # Delete processed files
        for f in processed_pkl:
            f.unlink()
            deleted_count += 1
        
        for f in processed_json:
            f.unlink()
            deleted_count += 1
        
        logger.info(f"🗑️ Deleted {deleted_count} game files")
        logger.info(f"💾 Freed up {total_size:.1f}MB of disk space!")
        
        if archive_path:
            logger.info(f"📋 Backup saved at: {archive_path}")
    else:
        logger.info(f"🔍 Would delete {total_files} files ({total_size:.1f}MB)")

def main():
    parser = argparse.ArgumentParser(description='Clean up training game files')
    parser.add_argument('--games-dir', default='../training_games', 
                      help='Directory containing training games')
    parser.add_argument('--dry-run', action='store_true',
                      help='Show what would be deleted without actually deleting')
    parser.add_argument('--no-archive', action='store_true',
                      help='Delete files without creating backup archive')
    
    args = parser.parse_args()
    
    cleanup_training_games(
        args.games_dir,
        dry_run=args.dry_run,
        keep_archives=not args.no_archive
    )

if __name__ == '__main__':
    main() 