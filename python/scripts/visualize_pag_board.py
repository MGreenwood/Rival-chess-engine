#!/usr/bin/env python3
"""
PAG Board Visualizer
==================

Creates beautiful visualizations of the ultra-dense PAG features:
1. Chess board heatmaps showing feature intensities  
2. Piece feature radar charts
3. Network graph of piece-square connections
4. Feature distribution plots

Usage:
    python visualize_pag_board.py "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
"""

import argparse
import sys
from pathlib import Path
import chess
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    from rival_ai.utils.board_conversion import board_to_hetero_data, PAG_ENGINE_AVAILABLE
    print(f"✅ PAG engine available: {PAG_ENGINE_AVAILABLE}")
except ImportError as e:
    print(f"❌ Failed to import PAG components: {e}")
    sys.exit(1)

class PAGVisualizer:
    def __init__(self):
        self.piece_unicode = {
            'P': '♙', 'R': '♖', 'N': '♘', 'B': '♗', 'Q': '♕', 'K': '♔',
            'p': '♟', 'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚'
        }
        
        self.feature_categories = {
            'basic': (0, 10),
            'tactical': (10, 86), 
            'positional': (86, 166),
            'strategic': (166, 226), 
            'dynamic': (226, 268),
            'advanced': (268, 308)
        }

    def create_chess_board_heatmap(self, board, data, feature_type='tactical'):
        """Create a chess board heatmap showing feature intensities"""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Extract piece data
        piece_tensor = data['piece'].x
        
        # Create 8x8 board for visualization
        board_values = np.zeros((8, 8))
        piece_chars = [['' for _ in range(8)] for _ in range(8)]
        
        # Map pieces to board positions
        piece_idx = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                row, col = 7 - (square // 8), square % 8
                
                if feature_type in self.feature_categories:
                    start, end = self.feature_categories[feature_type]
                    if piece_idx < piece_tensor.shape[0]:
                        features = piece_tensor[piece_idx, start:end]
                        board_values[row, col] = features.mean().item()
                        piece_chars[row][col] = self.piece_unicode[piece.symbol()]
                        piece_idx += 1
        
        # Create heatmap
        im = ax.imshow(board_values, cmap='RdYlBu_r', aspect='equal', vmin=0, vmax=1)
        
        # Add piece symbols
        for i in range(8):
            for j in range(8):
                if piece_chars[i][j]:
                    ax.text(j, i, piece_chars[i][j], ha='center', va='center', 
                           fontsize=24, weight='bold',
                           color='white' if board_values[i][j] > 0.5 else 'black')
        
        # Add square colors for chess board pattern
        for i in range(8):
            for j in range(8):
                if (i + j) % 2 == 0 and board_values[i][j] == 0:  # Light squares
                    ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, 
                                             facecolor='lightgray', alpha=0.3))
        
        # Customize board
        ax.set_xlim(-0.5, 7.5)
        ax.set_ylim(-0.5, 7.5)
        ax.set_xticks(range(8))
        ax.set_yticks(range(8))
        ax.set_xticklabels(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
        ax.set_yticklabels(['8', '7', '6', '5', '4', '3', '2', '1'])
        ax.set_title(f'PAG {feature_type.title()} Features Heatmap', fontsize=16, pad=20)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Feature Intensity', rotation=270, labelpad=20)
        
        plt.tight_layout()
        plt.show()

    def create_feature_distribution(self, piece_tensor):
        """Create distribution plots for each feature category"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, (cat_name, (start, end)) in enumerate(self.feature_categories.items()):
            ax = axes[idx]
            
            # Extract features for this category across all pieces
            cat_features = piece_tensor[:, start:end].flatten().detach().numpy()
            
            # Create histogram
            ax.hist(cat_features, bins=50, alpha=0.7, color=plt.cm.Set3(idx))
            ax.set_title(f'{cat_name.title()} Features')
            ax.set_xlabel('Feature Value')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            mean_val = np.mean(cat_features)
            std_val = np.std(cat_features)
            ax.axvline(mean_val, color='red', linestyle='--', 
                      label=f'Mean: {mean_val:.3f}')
            ax.legend()
        
        plt.suptitle('PAG Feature Distributions by Category', fontsize=16)
        plt.tight_layout()
        plt.show()

    def create_piece_comparison(self, piece_tensor, board):
        """Create comparison chart of piece feature averages"""
        piece_count = piece_tensor.shape[0]
        categories = list(self.feature_categories.keys())
        
        # Calculate averages for each piece and category
        matrix_data = np.zeros((piece_count, len(categories)))
        
        for piece_idx in range(piece_count):
            for cat_idx, (cat_name, (start, end)) in enumerate(self.feature_categories.items()):
                features = piece_tensor[piece_idx, start:end]
                matrix_data[piece_idx, cat_idx] = features.mean().item()
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(matrix_data, 
                   xticklabels=[cat.title() for cat in categories],
                   yticklabels=[f'Piece {i}' for i in range(piece_count)],
                   annot=True, fmt='.3f', cmap='RdYlBu_r', center=0.5)
        
        plt.title('PAG Feature Averages by Piece and Category')
        plt.xlabel('Feature Categories')
        plt.ylabel('Pieces')
        plt.tight_layout()
        plt.show()

    def create_board_overview(self, board, data):
        """Create comprehensive overview of the position"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        piece_tensor = data['piece'].x
        square_tensor = data['critical_square'].x
        
        # 1. Tactical features heatmap
        self._create_mini_heatmap(ax1, board, piece_tensor, 'tactical', 'Tactical Features')
        
        # 2. Positional features heatmap  
        self._create_mini_heatmap(ax2, board, piece_tensor, 'positional', 'Positional Features')
        
        # 3. Feature category averages
        categories = list(self.feature_categories.keys())
        piece_count = piece_tensor.shape[0]
        
        category_avgs = []
        for cat_name, (start, end) in self.feature_categories.items():
            features = piece_tensor[:, start:end]
            category_avgs.append(features.mean().item())
        
        ax3.bar(categories, category_avgs, color=plt.cm.Set3(range(len(categories))))
        ax3.set_title('Average Feature Values by Category')
        ax3.set_xlabel('Categories')
        ax3.set_ylabel('Average Value')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Piece count by type
        piece_types = {'P': 0, 'R': 0, 'N': 0, 'B': 0, 'Q': 0, 'K': 0,
                      'p': 0, 'r': 0, 'n': 0, 'b': 0, 'q': 0, 'k': 0}
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                piece_types[piece.symbol()] += 1
        
        white_pieces = [piece_types[p] for p in 'PRNBQK']
        black_pieces = [piece_types[p] for p in 'prnbqk']
        
        x = np.arange(6)
        width = 0.35
        
        ax4.bar(x - width/2, white_pieces, width, label='White', color='lightblue')
        ax4.bar(x + width/2, black_pieces, width, label='Black', color='lightcoral')
        ax4.set_title('Piece Count by Type')
        ax4.set_xlabel('Piece Types')
        ax4.set_ylabel('Count')
        ax4.set_xticks(x)
        ax4.set_xticklabels(['Pawn', 'Rook', 'Knight', 'Bishop', 'Queen', 'King'])
        ax4.legend()
        
        plt.suptitle(f'PAG Analysis Overview\n{board.fen()}', fontsize=14)
        plt.tight_layout()
        plt.show()

    def _create_mini_heatmap(self, ax, board, piece_tensor, feature_type, title):
        """Helper to create mini heatmap"""
        board_values = np.zeros((8, 8))
        piece_chars = [['' for _ in range(8)] for _ in range(8)]
        
        piece_idx = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                row, col = 7 - (square // 8), square % 8
                
                if feature_type in self.feature_categories:
                    start, end = self.feature_categories[feature_type]
                    if piece_idx < piece_tensor.shape[0]:
                        features = piece_tensor[piece_idx, start:end]
                        board_values[row, col] = features.mean().item()
                        piece_chars[row][col] = self.piece_unicode[piece.symbol()]
                        piece_idx += 1
        
        im = ax.imshow(board_values, cmap='RdYlBu_r', aspect='equal', vmin=0, vmax=1)
        
        # Add piece symbols
        for i in range(8):
            for j in range(8):
                if piece_chars[i][j]:
                    ax.text(j, i, piece_chars[i][j], ha='center', va='center', 
                           fontsize=16, weight='bold',
                           color='white' if board_values[i][j] > 0.5 else 'black')
        
        ax.set_xlim(-0.5, 7.5)
        ax.set_ylim(-0.5, 7.5)
        ax.set_xticks(range(8))
        ax.set_yticks(range(8))
        ax.set_xticklabels(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
        ax.set_yticklabels(['8', '7', '6', '5', '4', '3', '2', '1'])
        ax.set_title(title)

def main():
    parser = argparse.ArgumentParser(description='Visualize dense PAG features')
    parser.add_argument('fen', nargs='?', 
                       default='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
                       help='FEN string to analyze')
    
    args = parser.parse_args()
    fen = args.fen
    
    print(f"🔍 Analyzing position: {fen}")
    
    try:
        board = chess.Board(fen)
        print("Board state:")
        print(board)
        
        # Generate PAG data
        print("\n🔍 Converting to dense PAG features...")
        data = board_to_hetero_data(board, use_dense_pag=True)
        print("✅ Successfully generated dense PAG data")
        
        # Initialize visualizer
        viz = PAGVisualizer()
        
        print("\n🎨 Creating visualizations...")
        
        # 1. Comprehensive overview
        print("  📊 Creating overview...")
        viz.create_board_overview(board, data)
        
        # 2. Individual feature heatmaps
        for feature_type in ['tactical', 'positional', 'strategic']:
            print(f"  🔥 Creating {feature_type} heatmap...")
            viz.create_chess_board_heatmap(board, data, feature_type)
        
        # 3. Feature distributions
        print("  📈 Creating feature distributions...")
        viz.create_feature_distribution(data['piece'].x)
        
        # 4. Piece comparison
        print("  🆚 Creating piece comparison...")
        viz.create_piece_comparison(data['piece'].x, board)
        
        print("\n✅ All visualizations complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 