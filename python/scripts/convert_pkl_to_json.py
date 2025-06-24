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
        
        # Safety checks for array lengths
        num_states = len(game_record.states) if hasattr(game_record, 'states') else 0
        num_moves = len(game_record.moves) if hasattr(game_record, 'moves') else 0
        num_values = len(game_record.values) if hasattr(game_record, 'values') else 0
        num_policies = len(game_record.policies) if hasattr(game_record, 'policies') else 0
        
        if num_states == 0:
            logger.warning("GameRecord has no states")
            return []
        
        # Convert states to positions with moves (use minimum length for safety)
        max_positions = min(num_states - 1, num_moves, num_values, num_policies) if num_states > 0 else 0
        
        for i in range(max_positions):
            current_state = game_record.states[i]
            next_state = game_record.states[i + 1] if i + 1 < num_states else None
            
            # Get the move that was made
            move = None
            if isinstance(current_state, chess.Board):
                # Try to get move from moves array first
                if i < num_moves and hasattr(game_record.moves[i], 'uci'):
                    move = game_record.moves[i]
                elif next_state and isinstance(next_state, chess.Board):
                    # Find the move that was made by comparing board states
                    for legal_move in current_state.legal_moves:
                        temp_board = current_state.copy()
                        temp_board.push(legal_move)
                        if temp_board.fen() == next_state.fen():
                            move = legal_move
                            break
                
                if move is None:
                    logger.warning(f"Could not find move for position {i}")
                    continue
                
                # Get policy data for this position
                policy = None
                if i < num_policies:
                    policy_tensor = game_record.policies[i]
                    if hasattr(policy_tensor, 'tolist'):
                        policy = policy_tensor.tolist()
                    elif hasattr(policy_tensor, 'numpy'):
                        policy = policy_tensor.numpy().tolist()
                    elif isinstance(policy_tensor, (list, np.ndarray)):
                        policy = policy_tensor
                
                # Get value for this position
                value = 0.0
                if i < num_values:
                    value_data = game_record.values[i]
                    if hasattr(value_data, 'item'):
                        value = float(value_data.item())
                    else:
                        value = float(value_data)
                
                # Create position dictionary (enhanced dataset format)
                position = {
                    'fen': current_state.fen(),
                    'move': move.uci(),
                    'policy': policy,  # Required for enhanced dataset
                    'value': value
                }
                positions.append(position)
        
        # Add the final position (if we have states remaining)
        if num_states > max_positions:
            final_state = game_record.states[-1]
            if isinstance(final_state, chess.Board):
                final_policy = None
                final_index = num_states - 1
                
                # Get final policy if available
                if final_index < num_policies:
                    policy_tensor = game_record.policies[final_index]
                    if hasattr(policy_tensor, 'tolist'):
                        final_policy = policy_tensor.tolist()
                    elif hasattr(policy_tensor, 'numpy'):
                        final_policy = policy_tensor.numpy().tolist()
                    elif isinstance(policy_tensor, (list, np.ndarray)):
                        final_policy = policy_tensor
                
                # Get final value
                final_value = 0.0
                if num_values > 0:
                    final_value_data = game_record.values[-1]
                    if hasattr(final_value_data, 'item'):
                        final_value = float(final_value_data.item())
                    else:
                        final_value = float(final_value_data)
                
                positions.append({
                    'fen': final_state.fen(),
                    'policy': final_policy,
                    'value': final_value
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