"""
Script to analyze generated self-play games.
"""

import os
import json
import chess
import logging
from typing import List, Dict, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_game(filename: str) -> List[Dict]:
    """Load a game from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)

def analyze_material_balance(positions: List[Dict]) -> List[float]:
    """Calculate material balance throughout the game."""
    piece_values = {
        chess.PAWN: 1.0,
        chess.KNIGHT: 3.0,
        chess.BISHOP: 3.2,
        chess.ROOK: 5.0,
        chess.QUEEN: 9.0,
        chess.KING: 0.0
    }
    
    material_balance = []
    for pos in positions:
        board = chess.Board(pos['fen'])
        balance = 0.0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                value = piece_values[piece.piece_type]
                balance += value if piece.color == chess.WHITE else -value
        
        material_balance.append(balance)
    
    return material_balance

def analyze_move_types(positions: List[Dict]) -> Dict[str, int]:
    """Analyze types of moves made in the game."""
    move_types = defaultdict(int)
    
    for pos in positions:
        board = chess.Board(pos['fen'])
        move = chess.Move.from_uci(pos['move'])
        
        # Get the piece that moved
        piece = board.piece_at(move.from_square)
        if piece is None:
            continue
            
        # Check move type
        if board.is_capture(move):
            move_types['capture'] += 1
        if move.promotion:
            move_types['promotion'] += 1
        if board.is_castling(move):
            move_types['castling'] += 1
        if board.is_en_passant(move):
            move_types['en_passant'] += 1
        if board.is_check():
            move_types['check'] += 1
            
        # Count by piece type
        move_types[f'{piece.symbol().lower()}_moves'] += 1
    
    return dict(move_types)

def analyze_game_lengths(games: List[List[Dict]]) -> Tuple[float, int, int]:
    """Calculate statistics about game lengths."""
    lengths = [len(game) for game in games]
    return np.mean(lengths), min(lengths), max(lengths)

def analyze_position_values(positions: List[Dict]) -> Tuple[float, float, float]:
    """Analyze the distribution of position values."""
    values = [pos['value'] for pos in positions]
    return np.mean(values), np.std(values), np.max(np.abs(values))

def plot_material_balance(material_balance: List[float], game_num: int):
    """Plot material balance throughout the game."""
    plt.figure(figsize=(12, 4))
    plt.plot(material_balance)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    plt.title(f'Material Balance - Game {game_num}')
    plt.xlabel('Move Number')
    plt.ylabel('Material Balance (White - Black)')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'analysis/material_balance_game_{game_num}.png')
    plt.close()

def analyze_position(board: chess.Board) -> dict:
    """Analyze a chess position."""
    # Count material
    material = {chess.WHITE: 0, chess.BLACK: 0}
    piece_counts = {chess.WHITE: defaultdict(int), chess.BLACK: defaultdict(int)}
    
    for square in chess.SQUARES:
        piece = board.get_piece(square)
        if piece is not None:
            material[piece.color] += PIECE_VALUES[piece.piece_type]
            piece_counts[piece.color][piece.piece_type] += 1
    
    # Analyze move
    if board.move_stack:
        last_move = board.move_stack[-1]
        piece = board.get_piece(last_move.from_square)
        move_type = "capture" if board.is_capture(last_move) else "move"
        
        return {
            'material': material,
            'piece_counts': dict(piece_counts),
            'last_move': {
                'from': chess.square_name(last_move.from_square),
                'to': chess.square_name(last_move.to_square),
                'piece': piece.symbol() if piece else None,
                'type': move_type
            }
        }
    
    return {
        'material': material,
        'piece_counts': dict(piece_counts)
    }

def main():
    """Analyze generated games."""
    # Create analysis directory
    os.makedirs('analysis', exist_ok=True)
    
    # Find all game files
    game_files = [f for f in os.listdir('data') if f.startswith('positions_game_')]
    games = [load_game(os.path.join('data', f)) for f in game_files]
    
    # Analyze each game
    for i, (game, filename) in enumerate(zip(games, game_files), 1):
        logger.info(f"\nAnalyzing {filename}:")
        
        # Material balance
        material_balance = analyze_material_balance(game)
        plot_material_balance(material_balance, i)
        
        # Move types
        move_types = analyze_move_types(game)
        logger.info("Move types:")
        for move_type, count in move_types.items():
            logger.info(f"  {move_type}: {count}")
        
        # Position values
        mean_val, std_val, max_abs_val = analyze_position_values(game)
        logger.info(f"Position values - Mean: {mean_val:.3f}, Std: {std_val:.3f}, Max abs: {max_abs_val:.3f}")
        
        # Game length
        logger.info(f"Game length: {len(game)} moves")
        
        # Final position
        final_board = chess.Board(game[-1]['fen'])
        logger.info(f"Game result: {final_board.outcome().result() if final_board.is_game_over() else 'In progress'}")
    
    # Overall statistics
    avg_length, min_length, max_length = analyze_game_lengths(games)
    logger.info(f"\nOverall statistics:")
    logger.info(f"Average game length: {avg_length:.1f} moves")
    logger.info(f"Shortest game: {min_length} moves")
    logger.info(f"Longest game: {max_length} moves")
    logger.info(f"Total positions analyzed: {sum(len(game) for game in games)}")

if __name__ == '__main__':
    main() 