#!/usr/bin/env python3
"""
Script to convert PKL training data files to JSON format.
"""

import os
import sys
import json
import pickle
import argparse
from pathlib import Path
import logging
from typing import List, Dict, Any
import chess
import numpy as np
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_game_record_to_dict(game_record: Any) -> Dict[str, Any]:
    """Convert a GameRecord object to a dictionary.
    
    Args:
        game_record: GameRecord object from self-play
        
    Returns:
        Dictionary representation of the game record
    """
    try:
        # Extract basic game information
        positions = []
        
        # Convert states to positions with moves
        for i in range(len(game_record.states) - 1):  # -1 because last state has no move
            current_state = game_record.states[i]
            next_state = game_record.states[i + 1]
            
            # Get the move that was made
            if isinstance(current_state, chess.Board) and isinstance(next_state, chess.Board):
                # Find the move that was made
                move = None
                for legal_move in current_state.legal_moves:
                    temp_board = current_state.copy()
                    temp_board.push(legal_move)
                    if temp_board.fen() == next_state.fen():
                        move = legal_move
                        break
                
                if move is None:
                    logger.warning(f"Could not find move between states {i} and {i+1}")
                    continue
                
                # Create position dictionary
                position = {
                    'fen': current_state.fen(),
                    'move': move.uci(),
                    'value': float(game_record.values[i].item() if hasattr(game_record.values[i], 'item') else game_record.values[i])
                }
                positions.append(position)
        
        # Add the final position without a move
        final_state = game_record.states[-1]
        if isinstance(final_state, chess.Board):
            positions.append({
                'fen': final_state.fen(),
                'value': float(game_record.values[-1].item() if hasattr(game_record.values[-1], 'item') else game_record.values[-1])
            })
        
        return positions
    
    except Exception as e:
        logger.error(f"Error converting game record: {e}")
        raise

def convert_pkl_to_json(pkl_file: Path, output_dir: Path) -> None:
    """Convert a single PKL file to JSON format.
    
    Args:
        pkl_file: Path to the PKL file
        output_dir: Directory to save the JSON file
    """
    try:
        # Load PKL data
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        
        # Convert to JSON-serializable format
        if isinstance(data, list):
            # If it's a list of games
            json_data = []
            for item in data:
                if hasattr(item, 'states'):  # GameRecord object
                    json_data.extend(convert_game_record_to_dict(item))
                elif isinstance(item, dict):
                    # Convert any tensor values to lists
                    processed_item = {}
                    for key, value in item.items():
                        if isinstance(value, (torch.Tensor, np.ndarray)):
                            processed_item[key] = value.tolist() if hasattr(value, 'tolist') else value.cpu().numpy().tolist()
                        else:
                            processed_item[key] = value
                    json_data.append(processed_item)
                else:
                    json_data.append(str(item))  # Convert other objects to strings
        else:
            # If it's a single game
            if hasattr(data, 'states'):  # GameRecord object
                json_data = convert_game_record_to_dict(data)
            else:
                json_data = {}
                for key, value in data.items():
                    if isinstance(value, (torch.Tensor, np.ndarray)):
                        json_data[key] = value.tolist() if hasattr(value, 'tolist') else value.cpu().numpy().tolist()
                    else:
                        json_data[key] = value
        
        # Create output filename
        output_file = output_dir / f"{pkl_file.stem}.json"
        
        # Save as JSON
        with open(output_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        logger.info(f"Converted {pkl_file} to {output_file}")
        
    except Exception as e:
        logger.error(f"Error converting {pkl_file}: {e}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Convert PKL files to JSON format')
    parser.add_argument('--input-dir', type=str, required=True,
                      help='Directory containing PKL files')
    parser.add_argument('--output-dir', type=str, required=True,
                      help='Directory to save JSON files')
    parser.add_argument('--recursive', action='store_true',
                      help='Search for PKL files recursively')
    args = parser.parse_args()
    
    # Convert paths to Path objects
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find PKL files
    if args.recursive:
        pkl_files = list(input_dir.rglob('*.pkl'))
    else:
        pkl_files = list(input_dir.glob('*.pkl'))
    
    if not pkl_files:
        logger.error(f"No PKL files found in {input_dir}")
        sys.exit(1)
    
    logger.info(f"Found {len(pkl_files)} PKL files")
    
    # Convert each file
    for pkl_file in pkl_files:
        # Create corresponding subdirectory in output_dir if recursive
        if args.recursive:
            rel_path = pkl_file.relative_to(input_dir)
            file_output_dir = output_dir / rel_path.parent
            file_output_dir.mkdir(parents=True, exist_ok=True)
        else:
            file_output_dir = output_dir
        
        convert_pkl_to_json(pkl_file, file_output_dir)
    
    logger.info("Conversion complete!")

if __name__ == '__main__':
    main() 