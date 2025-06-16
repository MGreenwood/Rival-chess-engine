"""
Convert pickle game files to JSON format for the game viewer.
"""

import os
import sys
import json
import pickle
import chess
import argparse
from pathlib import Path
from typing import List, Dict

def convert_game_to_json(game) -> List[Dict]:
    """Convert a single game record to JSON format."""
    positions = []
    board = chess.Board()
    
    for state, move, policy, value in zip(game.states, game.moves, game.policies, game.values):
        # Convert policy tensor to list if needed
        if hasattr(policy, 'cpu'):
            policy = policy.cpu().numpy().tolist()
        
        # Create position dictionary
        position = {
            'fen': state.fen(),
            'move': move.uci(),
            'value': float(value),
            'policy': policy
        }
        positions.append(position)
        
        # Make move to verify it's valid
        board.push(move)
    
    return positions

def convert_games_file(pickle_file: str, output_dir: str = None) -> str:
    """Convert a pickle games file to JSON format.
    
    Args:
        pickle_file: Path to pickle file containing games
        output_dir: Optional directory to save JSON files
        
    Returns:
        Path to the output JSON file
    """
    # Load games from pickle file
    with open(pickle_file, 'rb') as f:
        games = pickle.load(f)
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = Path(output_dir)
    else:
        output_path = Path(pickle_file).parent
    
    # Convert each game
    for i, game in enumerate(games):
        positions = convert_game_to_json(game)
        
        # Save to JSON file
        output_file = output_path / f'game_{i+1}.json'
        with open(output_file, 'w') as f:
            json.dump(positions, f, indent=2)
        
        print(f"Converted game {i+1} to {output_file}")
    
    return str(output_path)

def main():
    parser = argparse.ArgumentParser(description='Convert pickle game files to JSON format')
    parser.add_argument('pickle_file', help='Path to pickle file containing games')
    parser.add_argument('--output-dir', help='Directory to save JSON files')
    args = parser.parse_args()
    
    output_path = convert_games_file(args.pickle_file, args.output_dir)
    print(f"\nConversion complete. JSON files saved in: {output_path}")
    print("\nTo view a game, run:")
    print(f"python scripts/view_game.py {output_path}/game_1.json")

if __name__ == '__main__':
    main() 