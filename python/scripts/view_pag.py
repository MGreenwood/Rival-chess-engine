"""
Visualize a Positional Adjacency Graph (PAG) for a chess position.
"""

import sys
import chess
import networkx as nx
import matplotlib.pyplot as plt
from rival_ai.pag import PositionalAdjacencyGraph
import numpy as np

def visualize_pag(board: chess.Board, output_file: str = None):
    """Visualize the PAG for a given board position.
    
    Args:
        board: Chess board position
        output_file: Optional file to save the visualization
    """
    # Create PAG
    pag = PositionalAdjacencyGraph(board)
    
    # Create networkx graph
    G = nx.DiGraph()
    
    # Add piece nodes
    piece_colors = {
        chess.WHITE: 'lightblue',
        chess.BLACK: 'lightcoral'
    }
    
    piece_labels = {
        chess.PAWN: '♟',
        chess.KNIGHT: '♞',
        chess.BISHOP: '♝',
        chess.ROOK: '♜',
        chess.QUEEN: '♛',
        chess.KING: '♚'
    }
    
    # Add piece nodes with their features
    for square, features in pag.piece_nodes.items():
        piece = board.piece_at(square)
        if piece is not None:
            label = f"{piece_labels[piece.piece_type]}\n{chess.square_name(square)}"
            node_id = f"piece_{square}"
            G.add_node(
                node_id,
                label=label,
                color=piece_colors[piece.color],
                type='piece',
                features=features
            )
    
    # Add critical square nodes
    for square, features in pag.critical_square_nodes.items():
        if features['is_attacked'] or features['is_controlled']:
            label = chess.square_name(square)
            node_id = f"square_{square}"
            G.add_node(
                node_id,
                label=label,
                color='lightgreen',
                type='square',
                features=features
            )
    
    # Add edges with their features
    edge_colors = {
        'attack': 'red',
        'support': 'green',
        'control': 'blue',
        'mobility': 'purple',
        'cooperation': 'orange',
        'obstruction': 'brown',
        'vulnerability': 'pink'
    }
    
    # Add direct relation edges (attacks and supports)
    for (from_sq, to_sq), features in pag.direct_edges.items():
        # Ensure both nodes exist
        from_node = f"piece_{from_sq}"
        to_node = f"piece_{to_sq}"
        if from_node not in G:
            G.add_node(from_node, type='piece', label=chess.square_name(from_sq), color='gray')
        if to_node not in G:
            G.add_node(to_node, type='piece', label=chess.square_name(to_sq), color='gray')
            
        G.add_edge(
            from_node,
            to_node,
            color=edge_colors['attack' if features['type'] == 'attack' else 'support'],
            label=f"{features['type']}\n{features['distance']:.1f}",
            type='direct'
        )
    
    # Add control edges
    for (piece_sq, square), features in pag.control_edges.items():
        # Ensure both nodes exist
        piece_node = f"piece_{piece_sq}"
        square_node = f"square_{square}"
        if piece_node not in G:
            G.add_node(piece_node, type='piece', label=chess.square_name(piece_sq), color='gray')
        if square_node not in G:
            G.add_node(square_node, type='square', label=chess.square_name(square), color='lightgreen')
            
        G.add_edge(
            piece_node,
            square_node,
            color=edge_colors['control'],
            label=f"control\n{features['distance']:.1f}",
            type='control'
        )
    
    # Add mobility edges
    for (piece_sq, square), features in pag.mobility_edges.items():
        if features['distance'] <= 2:  # Only show close mobility for clarity
            # Ensure both nodes exist
            piece_node = f"piece_{piece_sq}"
            square_node = f"square_{square}"
            if piece_node not in G:
                G.add_node(piece_node, type='piece', label=chess.square_name(piece_sq), color='gray')
            if square_node not in G:
                G.add_node(square_node, type='square', label=chess.square_name(square), color='lightgreen')
                
            G.add_edge(
                piece_node,
                square_node,
                color=edge_colors['mobility'],
                label=f"mobility\n{features['distance']:.1f}",
                type='mobility'
            )
    
    # Create the visualization
    plt.figure(figsize=(15, 15))
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Draw nodes by type
    piece_nodes = []
    square_nodes = []
    other_nodes = []
    
    for node in G.nodes():
        node_data = G.nodes[node]
        if 'type' not in node_data:
            print(f"Warning: Node {node} has no type attribute")
            other_nodes.append(node)
        elif node_data['type'] == 'piece':
            piece_nodes.append(node)
        elif node_data['type'] == 'square':
            square_nodes.append(node)
        else:
            other_nodes.append(node)
    
    # Draw piece nodes
    if piece_nodes:
        piece_colors = [G.nodes[node]['color'] for node in piece_nodes]
        nx.draw_networkx_nodes(G, pos, nodelist=piece_nodes, node_color=piece_colors, node_size=1000)
    
    # Draw square nodes
    if square_nodes:
        square_colors = [G.nodes[node]['color'] for node in square_nodes]
        nx.draw_networkx_nodes(G, pos, nodelist=square_nodes, node_color=square_colors, node_size=500, node_shape='s')
    
    # Draw other nodes (if any) in gray
    if other_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=other_nodes, node_color='gray', node_size=300)
    
    # Draw node labels
    labels = {}
    for node in G.nodes():
        node_data = G.nodes[node]
        if 'label' in node_data:
            labels[node] = node_data['label']
        else:
            # Use node ID as fallback label
            labels[node] = node.split('_')[-1]  # Extract square number from node ID
            print(f"Warning: Node {node} has no label attribute, using fallback")
    
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    # Draw edges with different colors
    for edge_type, color in edge_colors.items():
        edge_list = [(u, v) for u, v, d in G.edges(data=True) if d['type'] == edge_type]
        nx.draw_networkx_edges(G, pos, edgelist=edge_list, edge_color=color, arrows=True, arrowsize=10)
    
    # Draw edge labels
    edge_labels = {(u, v): d['label'] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6)
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label)
        for label, color in [
            ('White Pieces', 'lightblue'),
            ('Black Pieces', 'lightcoral'),
            ('Critical Squares', 'lightgreen'),
            ('Attack', 'red'),
            ('Support', 'green'),
            ('Control', 'blue'),
            ('Mobility', 'purple')
        ]
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.title("Positional Adjacency Graph (PAG)")
    plt.axis('off')
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    else:
        plt.show()

def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python view_pag.py <fen> [output_file]")
        print("Example: python view_pag.py 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1' pag.png")
        sys.exit(1)
    
    # Parse FEN
    fen = sys.argv[1]
    board = chess.Board(fen)
    
    # Get output file if provided
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Visualize PAG
    visualize_pag(board, output_file)

if __name__ == '__main__':
    main() 