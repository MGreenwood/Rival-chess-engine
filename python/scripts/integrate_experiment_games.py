#!/usr/bin/env python3
"""
Integrate Experiment Games into Training Pipeline
Process pkl files from experiments and add them to the training games directory for immediate use.
"""

import os
import sys
import json
import pickle
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import shutil

# Add the src directory to the path
script_dir = Path(__file__).parent
python_src = script_dir.parent / 'src'
sys.path.insert(0, str(python_src))

try:
    import torch
    import chess
    from rival_ai.training.training_types import GameRecord
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")
    print("Some features may not be available")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrainingGameIntegrator:
    def __init__(self, experiments_dir: str, training_games_dir: str):
        self.experiments_dir = Path(experiments_dir)
        self.training_games_dir = Path(training_games_dir)
        # Put experiment games in the single_player subdirectory where server expects them
        self.single_player_dir = self.training_games_dir / "single_player"
        self.processed_dir = self.training_games_dir / "processed" / "experiments"
        
        # Create directories
        self.single_player_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.stats = {
            "files_processed": 0,
            "games_added": 0,
            "positions_added": 0,
            "errors": []
        }

    def find_all_pkl_files(self) -> List[Path]:
        """Find all pkl files recursively in the experiments directory."""
        pkl_files = []
        for pkl_file in self.experiments_dir.rglob("*.pkl"):
            # Skip if already processed
            relative_path = pkl_file.relative_to(self.experiments_dir)
            processed_marker = self.processed_dir / f"{relative_path.parent.name}_{relative_path.stem}.processed"
            if not processed_marker.exists():
                pkl_files.append(pkl_file)
        
        logger.info(f"Found {len(pkl_files)} unprocessed pkl files")
        return pkl_files

    def convert_game_record_to_rival_format(self, game_record: Any) -> Optional[Dict[str, Any]]:
        """Convert a GameRecord to RivalAI training format."""
        try:
            # Handle game result safely
            result = getattr(game_record, 'result', None)
            if result is not None:
                try:
                    json.dumps(result)  # Test if it's serializable
                except TypeError:
                    result = str(result)  # Convert to string if not serializable
            
            # Extract game metadata
            game_data = {
                "game_id": f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{id(game_record)}",
                "timestamp": datetime.now().isoformat(),
                "game_type": "self_play_experiment",
                "result": result,
                "moves": [],
                "positions": [],
                "final_position": None
            }
            
            # Extract positions and moves
            if hasattr(game_record, 'states') and hasattr(game_record, 'moves') and hasattr(game_record, 'values'):
                states = game_record.states
                moves = game_record.moves
                values = game_record.values
                
                for i in range(len(states)):
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
                        
                        # Add position
                        position_data = {
                            'fen': current_state.fen(),
                            'value': value_float,
                            'turn': bool(current_state.turn),
                            'castling_rights': str(current_state.castling_rights),
                            'halfmove_clock': int(current_state.halfmove_clock),
                            'fullmove_number': int(current_state.fullmove_number)
                        }
                        game_data["positions"].append(position_data)
                        
                        # Add move if available
                        if move:
                            move_data = {
                                'uci': move.uci(),
                                'san': current_state.san(move),
                                'value': value_float
                            }
                            game_data["moves"].append(move_data)
                
                # Set final position
                if len(states) > 0:
                    final_state = states[-1]
                    if isinstance(final_state, chess.Board):
                        game_data["final_position"] = {
                            'fen': final_state.fen(),
                            'result': result
                        }
            
            return game_data
        except Exception as e:
            logger.error(f"Error converting game record: {e}")
            return None

    def process_pkl_file(self, pkl_file: Path) -> Dict[str, Any]:
        """Process a single pkl file and add games to training directory."""
        result = {
            "original_file": str(pkl_file),
            "games_processed": 0,
            "positions_extracted": 0,
            "success": False,
            "error": None,
            "output_files": []
        }
        
        try:
            # Load the pkl file
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            
            if not isinstance(data, list):
                data = [data]  # Single game
            
            # Process each game
            for i, game in enumerate(data):
                game_data = self.convert_game_record_to_rival_format(game)
                if game_data:
                    # Create output filename
                    relative_path = pkl_file.relative_to(self.experiments_dir)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_filename = f"experiment_{relative_path.parent.name}_{relative_path.stem}_{i}_{timestamp}.json"
                    output_file = self.single_player_dir / output_filename
                    
                    # Save to single_player directory where server expects them
                    with open(output_file, 'w') as f:
                        json.dump(game_data, f, indent=2)
                    
                    result["games_processed"] += 1
                    result["positions_extracted"] += len(game_data["positions"])
                    result["output_files"].append(str(output_file))
                    
                    logger.info(f"Added game {i+1}/{len(data)} from {pkl_file.name} -> {output_filename}")
            
            # Create processed marker
            relative_path = pkl_file.relative_to(self.experiments_dir)
            processed_marker = self.processed_dir / f"{relative_path.parent.name}_{relative_path.stem}.processed"
            with open(processed_marker, 'w') as f:
                json.dump({
                    "original_file": str(pkl_file),
                    "processed_at": datetime.now().isoformat(),
                    "games_count": result["games_processed"],
                    "positions_count": result["positions_extracted"],
                    "output_files": result["output_files"]
                }, f, indent=2)
            
            result["success"] = True
            
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Error processing {pkl_file}: {e}")
        
        return result

    def integrate_all_games(self, max_files: Optional[int] = None) -> None:
        """Process all pkl files and integrate them into training."""
        pkl_files = self.find_all_pkl_files()
        
        if max_files:
            pkl_files = pkl_files[:max_files]
            logger.info(f"Processing first {max_files} files only")
        
        if not pkl_files:
            logger.info("No unprocessed pkl files found")
            return
        
        logger.info(f"Processing {len(pkl_files)} pkl files...")
        
        for i, pkl_file in enumerate(pkl_files, 1):
            logger.info(f"Processing file {i}/{len(pkl_files)}: {pkl_file.name}")
            
            result = self.process_pkl_file(pkl_file)
            
            self.stats["files_processed"] += 1
            if result["success"]:
                self.stats["games_added"] += result["games_processed"]
                self.stats["positions_added"] += result["positions_extracted"]
                logger.info(f"✅ Added {result['games_processed']} games, {result['positions_extracted']} positions")
            else:
                self.stats["errors"].append(f"{pkl_file}: {result['error']}")
                logger.error(f"❌ Failed: {result['error']}")

    def print_summary(self) -> None:
        """Print integration summary."""
        logger.info("\n" + "="*50)
        logger.info("INTEGRATION SUMMARY")
        logger.info("="*50)
        logger.info(f"Files processed: {self.stats['files_processed']}")
        logger.info(f"Games added to training: {self.stats['games_added']}")
        logger.info(f"Training positions added: {self.stats['positions_added']}")
        logger.info(f"Training games directory: {self.training_games_dir}")
        
        if self.stats['errors']:
            logger.info(f"\nErrors encountered: {len(self.stats['errors'])}")
            for error in self.stats['errors'][:5]:
                logger.info(f"  {error}")

def main():
    parser = argparse.ArgumentParser(description='Integrate experiment pkl files into training pipeline')
    parser.add_argument('--experiments-dir', default='../python/experiments',
                      help='Directory containing experiment pkl files')
    parser.add_argument('--training-dir', default='../python/training_games',
                      help='Training games directory')
    parser.add_argument('--max-files', type=int,
                      help='Maximum number of files to process (for testing)')
    parser.add_argument('--dry-run', action='store_true',
                      help='Show what would be processed without doing it')
    
    args = parser.parse_args()
    
    integrator = TrainingGameIntegrator(
        experiments_dir=args.experiments_dir,
        training_games_dir=args.training_dir
    )
    
    if args.dry_run:
        pkl_files = integrator.find_all_pkl_files()
        logger.info("DRY RUN: Would process the following files:")
        for pkl_file in pkl_files[:10]:  # Show first 10
            logger.info(f"  {pkl_file}")
        if len(pkl_files) > 10:
            logger.info(f"  ... and {len(pkl_files) - 10} more files")
        logger.info(f"Total files to process: {len(pkl_files)}")
        return
    
    try:
        integrator.integrate_all_games(max_files=args.max_files)
        integrator.print_summary()
        
        logger.info("\n🎯 NEXT STEPS:")
        logger.info("1. Your experiment games are now in the training_games directory")
        logger.info("2. Run your training script to start training on these games")
        logger.info("3. Use: python scripts/train.py --experiment-name your_experiment")
        
    except KeyboardInterrupt:
        logger.info("\nIntegration interrupted by user")
        integrator.print_summary()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 