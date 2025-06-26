#!/usr/bin/env python3
"""
Debug RAW PAG Features
====================

Shows the actual raw PAG data without any interpretation or filtering.
Let's see what your 308-dimensional features really contain.
"""

import sys
from pathlib import Path
import chess
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    from rival_ai.utils.board_conversion import board_to_hetero_data, PAG_ENGINE_AVAILABLE
    print(f"✅ PAG engine available: {PAG_ENGINE_AVAILABLE}")
except ImportError as e:
    print(f"❌ Failed to import PAG components: {e}")
    sys.exit(1)

def analyze_raw_pag(fen):
    """Show raw PAG data without interpretation"""
    print(f"🔍 RAW PAG Analysis: {fen}")
    
    board = chess.Board(fen)
    print(f"\nBoard:\n{board}")
    
    # Extract PAG data
    data = board_to_hetero_data(board, use_dense_pag=True)
    
    piece_tensor = data['piece'].x
    square_tensor = data['critical_square'].x
    
    print(f"\n📊 RAW DATA:")
    print(f"Pieces: {piece_tensor.shape[0]} × {piece_tensor.shape[1]} features")
    print(f"Squares: {square_tensor.shape[0]} × {square_tensor.shape[1]} features")
    
    # Show actual piece mappings
    print(f"\n🔍 PIECE MAPPING:")
    piece_idx = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece_idx < piece_tensor.shape[0]:
            features = piece_tensor[piece_idx]
            
            # Show feature categories
            basic = features[0:10].mean().item()
            tactical = features[10:86].mean().item()
            positional = features[86:166].mean().item()
            strategic = features[166:226].mean().item()
            dynamic = features[226:268].mean().item()
            advanced = features[268:308].mean().item()
            
            # Show top individual features
            top_indices = features.topk(5).indices.tolist()
            top_values = features.topk(5).values.tolist()
            
            print(f"\nPiece {piece_idx}: {chess.piece_name(piece.piece_type)} on {chess.square_name(square)} ({'White' if piece.color else 'Black'})")
            print(f"  Categories: basic={basic:.3f}, tactical={tactical:.3f}, positional={positional:.3f}")
            print(f"              strategic={strategic:.3f}, dynamic={dynamic:.3f}, advanced={advanced:.3f}")
            print(f"  Top features: {[(i, f'{v:.3f}') for i, v in zip(top_indices, top_values)]}")
            
            # Show non-zero features count
            nonzero = (features > 0.01).sum().item()
            print(f"  Non-zero features: {nonzero}/308 ({nonzero/308*100:.1f}%)")
            
            piece_idx += 1
    
    # Show critical squares summary
    print(f"\n🎯 CRITICAL SQUARES SUMMARY:")
    for i in range(min(5, square_tensor.shape[0])):
        features = square_tensor[i]
        nonzero = (features > 0.01).sum().item()
        avg_value = features.mean().item()
        max_value = features.max().item()
        
        print(f"Square {i}: avg={avg_value:.3f}, max={max_value:.3f}, non-zero={nonzero}/95 ({nonzero/95*100:.1f}%)")
    
    # Show overall statistics
    print(f"\n📈 OVERALL STATISTICS:")
    all_piece_features = piece_tensor.flatten()
    all_square_features = square_tensor.flatten()
    
    print(f"Piece features - mean: {all_piece_features.mean().item():.3f}, std: {all_piece_features.std().item():.3f}")
    print(f"Square features - mean: {all_square_features.mean().item():.3f}, std: {all_square_features.std().item():.3f}")
    
    # Show feature distribution
    piece_nonzero = (all_piece_features > 0.01).sum().item()
    square_nonzero = (all_square_features > 0.01).sum().item()
    
    total_piece_features = len(all_piece_features)
    total_square_features = len(all_square_features)
    
    print(f"Non-zero features: pieces {piece_nonzero}/{total_piece_features} ({piece_nonzero/total_piece_features*100:.1f}%)")
    print(f"                   squares {square_nonzero}/{total_square_features} ({square_nonzero/total_square_features*100:.1f}%)")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fen', nargs='?', 
                       default='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
                       help='FEN to analyze')
    args = parser.parse_args()
    
    analyze_raw_pag(args.fen)

if __name__ == "__main__":
    main() 