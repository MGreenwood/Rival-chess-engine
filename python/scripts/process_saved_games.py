"""
Process saved games from the web interface and add them to the training queue.
"""

import os
import json
import logging
import chess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import shutil
import time

from rival_ai.training.training_types import GameRecord
from rival_ai.chess import GameResult

logger = logging.getLogger(__name__)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def load_saved_game(file_path: Path) -> Optional[Dict]:
    """Load a saved game from JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load game {file_path}: {e}")
        return None

def convert_to_game_record(game_data: Dict) -> Optional[GameRecord]:
    """Convert saved game data to GameRecord format."""
    try:
        # Create new game record
        record = GameRecord()
        
        # Set up board
        board = chess.Board()
        
        # Add initial state
        record.add_state(board.copy())
        
        # Process moves
        for move_str in game_data['moves']:
            try:
                move = chess.Move.from_uci(move_str)
                if move not in board.legal_moves:
                    logger.error(f"Illegal move {move_str} in game {game_data['game_id']}")
                    return None
                board.push(move)
                record.add_state(board.copy())
                record.add_move(move)
            except Exception as e:
                logger.error(f"Failed to process move {move_str}: {e}")
                return None
        
        # Set result
        result_map = {
            'white_wins': GameResult.WHITE_WINS,
            'black_wins': GameResult.BLACK_WINS,
            'draw_stalemate': GameResult.DRAW,
            'draw_insufficient': GameResult.DRAW,
            'draw_repetition': GameResult.REPETITION_DRAW,
            'draw_fifty_moves': GameResult.DRAW,
            'in_progress': None
        }
        
        result = result_map.get(game_data['result'])
        if result is None:
            logger.error(f"Invalid result {game_data['result']} in game {game_data['game_id']}")
            return None
            
        record.set_result(result)
        return record
        
    except Exception as e:
        logger.error(f"Failed to convert game {game_data['game_id']}: {e}")
        return None

def process_games_directory(games_dir: Path, processed_dir: Path, training_dir: Path) -> None:
    """
    Process all games in the games directory:
    1. Load each game JSON file
    2. Convert to GameRecord format
    3. Save to training directory
    4. Move processed files to processed directory
    """
    # Create directories if they don't exist
    processed_dir.mkdir(parents=True, exist_ok=True)
    training_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each game file
    for game_file in games_dir.glob('*.json'):
        try:
            # Load game data
            game_data = load_saved_game(game_file)
            if game_data is None:
                continue
                
            # Convert to GameRecord
            record = convert_to_game_record(game_data)
            if record is None:
                continue
                
            # Save to training directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            training_file = training_dir / f"game_{game_data['game_id']}_{timestamp}.json"
            
            # Save record
            record.save(training_file)
            logger.info(f"Saved game record to {training_file}")
            
            # Move original file to processed directory
            processed_file = processed_dir / game_file.name
            shutil.move(str(game_file), str(processed_file))
            logger.info(f"Moved {game_file.name} to processed directory")
            
        except Exception as e:
            logger.error(f"Failed to process {game_file}: {e}")

def main():
    setup_logging()
    
    # Set up directories
    base_dir = Path(__file__).parent.parent
    games_dir = base_dir / "training_games"
    processed_dir = games_dir / "processed"
    training_dir = base_dir / "experiments" / "current" / "training_queue"
    
    logger.info("Starting game processing loop...")
    
    while True:
        try:
            process_games_directory(games_dir, processed_dir, training_dir)
        except Exception as e:
            logger.error(f"Error in processing loop: {e}")
            
        # Sleep before next check
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    main() 