#!/usr/bin/env python3
"""
Dense PAG Visualizer
===================

This script takes a FEN string and outputs the ultra-dense PAG features in a readable format.
Useful for debugging and understanding what the 308-dimensional piece features and 95-dimensional 
square features actually represent.

Usage:
    python visualize_dense_pag.py "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    python visualize_dense_pag.py --interactive
"""

import argparse
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
    print(f"❌ Failed to import board conversion: {e}")
    sys.exit(1)

def get_piece_name_from_board(board, piece_index):
    """Get the actual piece name from the chess board."""
    # Extract all pieces from the board in a consistent order
    pieces = []
    
    # Iterate through all squares to build piece list
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            color_name = "White" if piece.color == chess.WHITE else "Black"
            piece_name = piece.symbol().upper() if piece.color == chess.WHITE else piece.symbol().lower()
            
            # Convert piece symbol to name
            piece_names = {
                'P': 'Pawn', 'p': 'Pawn',
                'R': 'Rook', 'r': 'Rook', 
                'N': 'Knight', 'n': 'Knight',
                'B': 'Bishop', 'b': 'Bishop',
                'Q': 'Queen', 'q': 'Queen',
                'K': 'King', 'k': 'King'
            }
            
            piece_type = piece_names.get(piece.symbol(), piece.symbol())
            square_name = chess.square_name(square)
            
            pieces.append(f"{color_name} {piece_type} on {square_name}")
    
    # Return the piece name for the given index, with fallback
    if piece_index < len(pieces):
        return pieces[piece_index]
    else:
        return f"piece_{piece_index}"

def analyze_piece_features(piece_tensor, piece_types, board):
    """Analyze and interpret piece features."""
    print("=" * 80)
    print("PIECE FEATURE ANALYSIS")
    print("=" * 80)
    
    if piece_tensor.shape[1] != 308:
        print(f"⚠️ Expected 308-dimensional piece features, got {piece_tensor.shape[1]}")
        return
    
    # Feature ranges based on the Rust implementation
    feature_ranges = {
        'basic_info': (0, 10),           # piece type, color, position
        'tactical': (10, 86),            # attack patterns, defense, motifs, threats, vulnerability
        'positional': (86, 166),         # mobility, control, coordination, activity, structural role
        'strategic': (166, 226),         # development, king safety, pawn structure, space control
        'dynamic': (226, 268),           # tempo, initiative, pressure, compensation
        'advanced': (268, 308)           # pattern recognition, endgame, sacrificial potential
    }
    
    for i, piece_features in enumerate(piece_tensor):
        # Get the actual piece name from the board
        piece_name = get_piece_name_from_board(board, i)
        
        print(f"\n{piece_name}:")
        print("-" * 40)
        
        for category, (start, end) in feature_ranges.items():
            features = piece_features[start:end].numpy()
            non_zero = np.count_nonzero(features)
            max_val = np.max(features)
            min_val = np.min(features)
            avg_val = np.mean(features)
            
            print(f"  {category.upper():12} [{start:3d}-{end:3d}]: "
                  f"{non_zero:2d}/{end-start:2d} non-zero, "
                  f"range [{min_val:6.3f}, {max_val:6.3f}], "
                  f"avg {avg_val:6.3f}")
            
            # Show some specific interesting features
            if category == 'basic_info' and non_zero > 0:
                print(f"    Basic: {features[:6]} (piece type one-hot)")
            elif category == 'tactical' and max_val > 0.1:
                top_indices = np.argsort(features)[-3:][::-1]
                top_values = features[top_indices]
                print(f"    Top tactical: indices {top_indices + start} = {top_values}")
            elif category == 'positional' and max_val > 0.1:
                top_indices = np.argsort(features)[-3:][::-1] 
                top_values = features[top_indices]
                print(f"    Top positional: indices {top_indices + start} = {top_values}")

def get_square_name_from_index(square_index):
    """Get the actual square name from the index."""
    # Critical squares are typically key squares like center squares, etc.
    # For now, we'll use a simple mapping based on common critical squares
    if square_index < 64:
        # Map to actual chess squares (a1=0, b1=1, ..., h8=63)
        square_name = chess.square_name(square_index)
        return f"Square {square_name}"
    else:
        return f"Critical Square {square_index}"

def analyze_square_features(square_tensor, square_types):
    """Analyze and interpret critical square features."""
    print("\n" + "=" * 80)
    print("CRITICAL SQUARE FEATURE ANALYSIS")
    print("=" * 80)
    
    if square_tensor.shape[1] != 95:
        print(f"⚠️ Expected 95-dimensional square features, got {square_tensor.shape[1]}")
        return
    
    # Feature ranges for critical squares (estimated based on typical implementation)
    feature_ranges = {
        'basic_info': (0, 8),            # file, rank, color, occupancy
        'control': (8, 28),              # piece control, attack counts, defense
        'strategic': (28, 48),           # outpost potential, weakness, key squares
        'tactical': (48, 68),            # pin/fork potential, tactics involvement
        'positional': (68, 88),          # centrality, development, pawn structure
        'dynamic': (88, 95)              # tempo, pressure, activity
    }
    
    for i, square_features in enumerate(square_tensor):
        # Get the actual square name
        square_name = get_square_name_from_index(i)
        
        print(f"\n{square_name}:")
        print("-" * 45)
        
        for category, (start, end) in feature_ranges.items():
            features = square_features[start:end].numpy()
            non_zero = np.count_nonzero(features)
            max_val = np.max(features)
            min_val = np.min(features)
            avg_val = np.mean(features)
            
            print(f"  {category.upper():12} [{start:2d}-{end:2d}]: "
                  f"{non_zero:2d}/{end-start:2d} non-zero, "
                  f"range [{min_val:6.3f}, {max_val:6.3f}], "
                  f"avg {avg_val:6.3f}")
            
            # Show interesting features
            if max_val > 0.1:
                top_indices = np.argsort(features)[-2:][::-1]
                top_values = features[top_indices]
                print(f"    Top values: indices {top_indices + start} = {top_values}")

def analyze_edges(edge_index, edge_attr):
    """Analyze edge information."""
    print("\n" + "=" * 80)
    print("EDGE ANALYSIS")
    print("=" * 80)
    
    num_edges = edge_index.shape[1]
    edge_dim = edge_attr.shape[1] if edge_attr is not None else 0
    
    print(f"Total edges: {num_edges}")
    print(f"Edge feature dimensions: {edge_dim}")
    
    if edge_attr is not None and edge_dim > 0:
        # Analyze edge feature statistics
        edge_features = edge_attr.numpy()
        non_zero_edges = np.count_nonzero(edge_features, axis=1)
        
        print(f"Edges with features: {np.count_nonzero(non_zero_edges)}/{num_edges}")
        print(f"Average features per edge: {np.mean(non_zero_edges):.2f}")
        print(f"Max features in single edge: {np.max(non_zero_edges)}")
        
        # Show some example edges
        print("\nSample edges (source -> target):")
        for i in range(min(5, num_edges)):
            src, tgt = edge_index[:, i]
            features = edge_features[i]
            non_zero_count = np.count_nonzero(features)
            max_feature = np.max(features)
            print(f"  Edge {i}: {src} -> {tgt}, {non_zero_count} features, max={max_feature:.3f}")

def visualize_dense_pag(fen):
    """Main function to visualize dense PAG for a given FEN."""
    try:
        # Parse the FEN
        board = chess.Board(fen)
        print(f"Analyzing position: {fen}")
        print(f"Board state:\n{board}")
        print(f"Turn: {'White' if board.turn else 'Black'}")
        print(f"Castling rights: {board.castling_rights}")
        print(f"En passant: {board.ep_square}")
        
        # Convert to dense PAG
        print(f"\n🔍 Converting to dense PAG features...")
        try:
            data = board_to_hetero_data(board, use_dense_pag=True)
            print(f"✅ Successfully generated dense PAG data")
        except Exception as e:
            print(f"❌ Failed to generate dense PAG, trying fallback: {e}")
            data = board_to_hetero_data(board, use_dense_pag=False)
            print(f"⚠️ Using fallback (basic) features")
        
        # Extract data
        piece_tensor = data['piece'].x
        square_tensor = data['critical_square'].x  
        
        # Handle edge data properly
        edge_index = None
        edge_attr = None
        
        # Try to get edge information - PyTorch Geometric stores it differently
        edge_types = data.edge_types
        print(f"Available edge types: {edge_types}")
        
        if len(edge_types) > 0:
            # Get the first edge type (usually the main one)
            edge_type = edge_types[0]
            edge_store = data[edge_type]
            
            # Access edge_index and edge_attr from the edge store
            if hasattr(edge_store, 'edge_index'):
                edge_index = edge_store.edge_index
            if hasattr(edge_store, 'edge_attr'):
                edge_attr = edge_store.edge_attr
        
        print(f"\nData shapes:")
        print(f"  Pieces: {piece_tensor.shape}")
        print(f"  Critical squares: {square_tensor.shape}")
        print(f"  Edges: {edge_index.shape if edge_index is not None else 'None'}")
        print(f"  Edge attributes: {edge_attr.shape if edge_attr is not None else 'None'}")
        
        # Debug: Show what's actually in the data object
        print(f"\nData object info:")
        print(f"  Node types: {data.node_types}")
        print(f"  Edge types: {data.edge_types}")
        
        # Get node types if available - handle both dict and tensor cases
        piece_types = []
        square_types = []
        
        try:
            # Try to get node types from data
            if hasattr(data['piece'], 'node_type'):
                piece_types = data['piece'].node_type
            else:
                piece_types = [f"piece_{i}" for i in range(piece_tensor.shape[0])]
                
            if hasattr(data['critical_square'], 'node_type'):
                square_types = data['critical_square'].node_type  
            else:
                square_types = [f"square_{i}" for i in range(square_tensor.shape[0])]
        except Exception as e:
            print(f"  Warning: Could not get node types: {e}")
            piece_types = [f"piece_{i}" for i in range(piece_tensor.shape[0])]
            square_types = [f"square_{i}" for i in range(square_tensor.shape[0])]
        
        # Analyze each component
        analyze_piece_features(piece_tensor, piece_types, board)
        analyze_square_features(square_tensor, square_types)
        
        if edge_index is not None:
            analyze_edges(edge_index, edge_attr)
        else:
            print("\n" + "=" * 80)
            print("EDGE ANALYSIS")
            print("=" * 80)
            print("No edge data available in this PAG representation")
        
        # Summary statistics
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)
        
        total_piece_features = np.count_nonzero(piece_tensor.numpy())
        total_square_features = np.count_nonzero(square_tensor.numpy())
        
        print(f"Total active piece features: {total_piece_features}/{piece_tensor.numel()}")
        print(f"Total active square features: {total_square_features}/{square_tensor.numel()}")
        print(f"Piece feature density: {total_piece_features/piece_tensor.numel()*100:.1f}%")
        print(f"Square feature density: {total_square_features/square_tensor.numel()*100:.1f}%")
        
        if edge_attr is not None:
            total_edge_features = np.count_nonzero(edge_attr.numpy())
            print(f"Total active edge features: {total_edge_features}/{edge_attr.numel()}")
            print(f"Edge feature density: {total_edge_features/edge_attr.numel()*100:.1f}%")
        
    except Exception as e:
        print(f"❌ Error analyzing position: {e}")
        import traceback
        traceback.print_exc()

def interactive_mode():
    """Interactive mode for exploring different positions."""
    print("🎯 Interactive Dense PAG Explorer")
    print("Enter FEN strings to analyze, or 'quit' to exit")
    print("Try some examples:")
    print("  Starting position: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    print("  Tactical position: r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3")
    print("  Endgame: 8/8/8/8/8/8/8/K1k5 w - - 0 1")
    print()
    
    while True:
        try:
            fen = input("Enter FEN (or 'quit'): ").strip()
            if fen.lower() in ['quit', 'exit', 'q']:
                break
            
            if not fen:
                continue
                
            print("\n" + "="*100)
            visualize_dense_pag(fen)
            print("="*100 + "\n")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Visualize dense PAG features from FEN strings")
    parser.add_argument("fen", nargs="?", help="FEN string to analyze")
    parser.add_argument("--interactive", "-i", action="store_true", 
                       help="Run in interactive mode")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode()
    elif args.fen:
        visualize_dense_pag(args.fen)
    else:
        # Default to starting position
        starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        print("No FEN provided, analyzing starting position...")
        visualize_dense_pag(starting_fen)

if __name__ == "__main__":
    main() 