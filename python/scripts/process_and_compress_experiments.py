#!/usr/bin/env python3
"""
Process and Compress Experiment PKL Files
Recursively process all pkl files in experiments directory, convert to training format, and compress to save space.
"""

import os
import sys
import json
import pickle
import gzip
import shutil
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add the src directory to the path
script_dir = Path(__file__).parent
python_src = script_dir.parent / 'src'
sys.path.insert(0, str(python_src))

try:
    import torch
    import chess
    from rival_ai.training.training_types import GameRecord
    from rival_ai.data.dataset import ChessDataset
    from rival_ai.pag import PAG
    from rival_ai.utils.board_conversion import board_to_hetero_data
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")
    print("Some features may not be available")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ExperimentProcessor:
    def __init__(self, experiments_dir: str, output_dir: str, compression_format: str = "gzip"):
        self.experiments_dir = Path(experiments_dir)
        self.output_dir = Path(output_dir)
        self.compression_format = compression_format
        self.processed_dir = self.output_dir / "processed"
        self.archived_dir = self.output_dir / "archived"
        self.training_dir = self.output_dir / "training_data"
        
        # Create output directories
        for directory in [self.processed_dir, self.archived_dir, self.training_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.stats = {
            "files_processed": 0,
            "total_games": 0,
            "total_positions": 0,
            "bytes_saved": 0,
            "compression_ratio": 0.0,
            "errors": []
        }
        self.stats_lock = threading.Lock()

    def find_all_pkl_files(self) -> List[Path]:
        """Find all pkl files recursively in the experiments directory."""
        pkl_files = []
        for pkl_file in self.experiments_dir.rglob("*.pkl"):
            pkl_files.append(pkl_file)
        
        logger.info(f"Found {len(pkl_files)} pkl files to process")
        return pkl_files

    def load_pkl_file(self, pkl_file: Path) -> Optional[List]:
        """Load a pkl file and return the games."""
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            
            if isinstance(data, list):
                return data
            else:
                return [data]  # Single game
        except Exception as e:
            logger.error(f"Error loading {pkl_file}: {e}")
            with self.stats_lock:
                self.stats["errors"].append(f"Load error {pkl_file}: {e}")
            return None

    def convert_game_record_to_training_data(self, game_record: Any) -> Dict[str, Any]:
        """Convert a GameRecord object to training data format."""
        try:
            # Handle game result safely
            result = getattr(game_record, 'result', None)
            if result is not None:
                # Convert result to string if it's not already JSON serializable
                try:
                    json.dumps(result)  # Test if it's serializable
                except TypeError:
                    result = str(result)  # Convert to string if not serializable
            
            training_data = {
                "positions": [],
                "game_metadata": {
                    "result": result,
                    "game_length": len(getattr(game_record, 'states', [])),
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            # Extract positions and moves
            if hasattr(game_record, 'states') and hasattr(game_record, 'moves') and hasattr(game_record, 'values'):
                states = game_record.states
                moves = game_record.moves
                values = game_record.values
                
                for i in range(len(states) - 1):
                    current_state = states[i]
                    move = moves[i] if i < len(moves) else None
                    value = values[i] if i < len(values) else 0.0
                    
                    if isinstance(current_state, chess.Board):
                        # Safely convert value to float
                        try:
                            if hasattr(value, 'item'):
                                value_float = float(value.item())
                            elif hasattr(value, 'numpy'):
                                value_float = float(value.numpy())
                            else:
                                value_float = float(value)
                        except (TypeError, ValueError):
                            value_float = 0.0
                        
                        position_data = {
                            'fen': current_state.fen(),
                            'move': move.uci() if move else None,
                            'value': value_float,
                            'turn': bool(current_state.turn),
                            'castling_rights': str(current_state.castling_rights),
                            'halfmove_clock': int(current_state.halfmove_clock),
                            'fullmove_number': int(current_state.fullmove_number)
                        }
                        training_data["positions"].append(position_data)
                
                # Add final position
                if len(states) > 0:
                    final_state = states[-1]
                    final_value = values[-1] if len(values) > 0 else 0.0
                    if isinstance(final_state, chess.Board):
                        # Safely convert final value to float
                        try:
                            if hasattr(final_value, 'item'):
                                final_value_float = float(final_value.item())
                            elif hasattr(final_value, 'numpy'):
                                final_value_float = float(final_value.numpy())
                            else:
                                final_value_float = float(final_value)
                        except (TypeError, ValueError):
                            final_value_float = 0.0
                        
                        training_data["positions"].append({
                            'fen': final_state.fen(),
                            'value': final_value_float,
                            'turn': bool(final_state.turn),
                            'castling_rights': str(final_state.castling_rights),
                            'halfmove_clock': int(final_state.halfmove_clock),
                            'fullmove_number': int(final_state.fullmove_number)
                        })
            
            return training_data
        except Exception as e:
            logger.error(f"Error converting game record: {e}")
            return None

    def process_pkl_file(self, pkl_file: Path) -> Dict[str, Any]:
        """Process a single pkl file."""
        result = {
            "original_file": str(pkl_file),
            "games_processed": 0,
            "positions_extracted": 0,
            "original_size": 0,
            "compressed_size": 0,
            "success": False,
            "error": None
        }
        
        try:
            # Get original file size
            result["original_size"] = pkl_file.stat().st_size
            
            # Load the pkl file
            games = self.load_pkl_file(pkl_file)
            if games is None:
                result["error"] = "Failed to load pkl file"
                return result
            
            # Process each game
            training_games = []
            for game in games:
                training_data = self.convert_game_record_to_training_data(game)
                if training_data:
                    training_games.append(training_data)
                    result["positions_extracted"] += len(training_data["positions"])
            
            result["games_processed"] = len(training_games)
            
            if training_games:
                # Create output filename
                relative_path = pkl_file.relative_to(self.experiments_dir)
                output_name = f"{relative_path.parent.name}_{relative_path.stem}"
                
                # Save to training directory as JSON
                training_file = self.training_dir / f"{output_name}_training.json"
                with open(training_file, 'w') as f:
                    json.dump(training_games, f, indent=2)
                
                # Save compressed version
                if self.compression_format == "gzip":
                    compressed_file = self.archived_dir / f"{output_name}.json.gz"
                    with open(training_file, 'rb') as f_in:
                        with gzip.open(compressed_file, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    result["compressed_size"] = compressed_file.stat().st_size
                    
                    # Remove uncompressed training file to save space
                    training_file.unlink()
                
                elif self.compression_format == "pkl_compressed":
                    compressed_file = self.archived_dir / f"{output_name}_compressed.pkl.gz"
                    with gzip.open(compressed_file, 'wb') as f:
                        pickle.dump(training_games, f)
                    result["compressed_size"] = compressed_file.stat().st_size
                
                # Create processed marker file
                processed_marker = self.processed_dir / f"{relative_path.parent.name}_{relative_path.stem}.processed"
                with open(processed_marker, 'w') as f:
                    json.dump({
                        "original_file": str(pkl_file),
                        "processed_at": datetime.now().isoformat(),
                        "games_count": result["games_processed"],
                        "positions_count": result["positions_extracted"],
                        "compression_ratio": result["original_size"] / max(result["compressed_size"], 1)
                    }, f, indent=2)
                
                result["success"] = True
            else:
                result["error"] = "No valid games found"
                
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Error processing {pkl_file}: {e}")
        
        return result

    def process_all_files(self, max_workers: int = 4, dry_run: bool = False) -> None:
        """Process all pkl files with threading."""
        pkl_files = self.find_all_pkl_files()
        
        if dry_run:
            logger.info("DRY RUN: Would process the following files:")
            total_size = 0
            for pkl_file in pkl_files:
                size_mb = pkl_file.stat().st_size / (1024 * 1024)
                total_size += size_mb
                logger.info(f"  {pkl_file} ({size_mb:.1f} MB)")
            logger.info(f"Total size: {total_size/1024:.2f} GB")
            return
        
        logger.info(f"Processing {len(pkl_files)} files with {max_workers} workers...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {executor.submit(self.process_pkl_file, pkl_file): pkl_file for pkl_file in pkl_files}
            
            # Process results as they complete
            for future in as_completed(future_to_file):
                pkl_file = future_to_file[future]
                try:
                    result = future.result()
                    
                    with self.stats_lock:
                        self.stats["files_processed"] += 1
                        if result["success"]:
                            self.stats["total_games"] += result["games_processed"]
                            self.stats["total_positions"] += result["positions_extracted"]
                            self.stats["bytes_saved"] += result["original_size"] - result["compressed_size"]
                        else:
                            self.stats["errors"].append(f"{pkl_file}: {result['error']}")
                    
                    if result["success"]:
                        compression_ratio = result["original_size"] / max(result["compressed_size"], 1)
                        logger.info(f"✅ Processed {pkl_file.name}: {result['games_processed']} games, "
                                  f"{result['positions_extracted']} positions, "
                                  f"compression {compression_ratio:.1f}x")
                    else:
                        logger.error(f"❌ Failed {pkl_file.name}: {result['error']}")
                        
                except Exception as e:
                    logger.error(f"❌ Exception processing {pkl_file}: {e}")
                    with self.stats_lock:
                        self.stats["errors"].append(f"{pkl_file}: {str(e)}")

    def cleanup_original_files(self, confirm: bool = False) -> None:
        """Remove original pkl files after successful processing."""
        if not confirm:
            logger.warning("Use --confirm-cleanup to actually delete original files")
            return
        
        pkl_files = self.find_all_pkl_files()
        deleted_count = 0
        space_freed = 0
        
        for pkl_file in pkl_files:
            # Check if this file was processed successfully
            relative_path = pkl_file.relative_to(self.experiments_dir)
            processed_marker = self.processed_dir / f"{relative_path.parent.name}_{relative_path.stem}.processed"
            
            if processed_marker.exists():
                try:
                    space_freed += pkl_file.stat().st_size
                    pkl_file.unlink()
                    deleted_count += 1
                    logger.info(f"🗑️  Deleted {pkl_file}")
                except Exception as e:
                    logger.error(f"Failed to delete {pkl_file}: {e}")
        
        logger.info(f"Cleanup complete: {deleted_count} files deleted, {space_freed / (1024**3):.2f} GB freed")

    def print_summary(self) -> None:
        """Print processing summary."""
        logger.info("\n" + "="*50)
        logger.info("PROCESSING SUMMARY")
        logger.info("="*50)
        logger.info(f"Files processed: {self.stats['files_processed']}")
        logger.info(f"Total games: {self.stats['total_games']}")
        logger.info(f"Total positions: {self.stats['total_positions']}")
        logger.info(f"Space saved: {self.stats['bytes_saved'] / (1024**3):.2f} GB")
        
        if self.stats['bytes_saved'] > 0:
            # Calculate overall compression ratio
            original_total = sum(pkl_file.stat().st_size for pkl_file in self.find_all_pkl_files())
            compression_ratio = original_total / max(original_total - self.stats['bytes_saved'], 1)
            logger.info(f"Overall compression ratio: {compression_ratio:.1f}x")
        
        if self.stats['errors']:
            logger.info(f"\nErrors encountered: {len(self.stats['errors'])}")
            for error in self.stats['errors'][:10]:  # Show first 10 errors
                logger.info(f"  {error}")
            if len(self.stats['errors']) > 10:
                logger.info(f"  ... and {len(self.stats['errors']) - 10} more")

def main():
    parser = argparse.ArgumentParser(description='Process and compress experiment pkl files')
    parser.add_argument('--experiments-dir', default='python/experiments',
                      help='Directory containing experiment pkl files')
    parser.add_argument('--output-dir', default='python/compressed_experiments',
                      help='Output directory for processed files')
    parser.add_argument('--compression', choices=['gzip', 'pkl_compressed'], default='gzip',
                      help='Compression format to use')
    parser.add_argument('--max-workers', type=int, default=4,
                      help='Maximum number of worker threads')
    parser.add_argument('--dry-run', action='store_true',
                      help='Show what would be processed without doing it')
    parser.add_argument('--cleanup', action='store_true',
                      help='Remove original pkl files after processing')
    parser.add_argument('--confirm-cleanup', action='store_true',
                      help='Actually delete original files (required with --cleanup)')
    
    args = parser.parse_args()
    
    processor = ExperimentProcessor(
        experiments_dir=args.experiments_dir,
        output_dir=args.output_dir,
        compression_format=args.compression
    )
    
    try:
        # Process all files
        processor.process_all_files(max_workers=args.max_workers, dry_run=args.dry_run)
        
        if not args.dry_run:
            processor.print_summary()
            
            # Cleanup if requested
            if args.cleanup:
                processor.cleanup_original_files(confirm=args.confirm_cleanup)
    
    except KeyboardInterrupt:
        logger.info("\nProcessing interrupted by user")
        processor.print_summary()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
