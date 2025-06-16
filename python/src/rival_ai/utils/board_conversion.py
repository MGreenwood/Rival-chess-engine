import torch
from torch_geometric.data import HeteroData
import chess
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

# Cache for piece features to avoid recomputation
_piece_feature_cache: Dict[Tuple[int, int, bool], np.ndarray] = {}
_critical_square_cache: Dict[int, np.ndarray] = {}

def get_piece_features(piece: chess.Piece, square: int, board: chess.Board) -> np.ndarray:
    """Get features for a piece with caching."""
    # Create cache key
    piece_type = piece.piece_type
    piece_color = piece.color
    cache_key = (piece_type, square, piece_color)
    
    # Check cache
    if cache_key in _piece_feature_cache:
        return _piece_feature_cache[cache_key]
    
    # Compute features
    features = np.zeros(12, dtype=np.float32)  # 6 piece types * 2 colors
    
    # One-hot encode piece type and color
    piece_idx = (piece_type - 1) * 2 + int(piece_color)
    features[piece_idx] = 1.0
    
    # Cache result
    _piece_feature_cache[cache_key] = features
    return features

def get_critical_square_features(square: int, board: chess.Board) -> np.ndarray:
    """Get features for a critical square with caching."""
    # Check cache
    if square in _critical_square_cache:
        return _critical_square_cache[square]
    
    # Compute features
    features = np.zeros(1, dtype=np.float32)
    
    # Check if square is controlled by any piece
    is_controlled = False
    for piece in board.piece_map().values():
        if piece.color == board.turn:  # Only consider current player's pieces
            moves = board.attacks(square)
            if moves:
                is_controlled = True
                break
    
    features[0] = float(is_controlled)
    
    # Cache result
    _critical_square_cache[square] = features
    return features

@torch.jit.script  # Use TorchScript for faster execution
def create_edge_index(piece_nodes: List[int], target_nodes: List[int]) -> torch.Tensor:
    """Create edge index tensor efficiently."""
    return torch.tensor([piece_nodes, target_nodes], dtype=torch.long)

def board_to_hetero_data(board: chess.Board) -> HeteroData:
    """Convert a chess board to heterogeneous graph data.
    
    Args:
        board: Chess board to convert
        
    Returns:
        HeteroData object containing:
        - piece nodes with features [num_pieces, piece_dim]
        - square nodes with features [num_squares, critical_square_dim]
        - edge_index_dict with 'piece_to_square' and 'square_to_square' edges
        - batch information for global pooling
    """
    # Initialize arrays for node features and edge indices
    piece_features = []
    square_features = []
    piece_to_square_edges = []
    square_to_square_edges = []
    
    # Maps to track node indices
    piece_to_node = {}  # Maps (piece_type, color) to node index
    square_to_node = {}  # Maps square index to node index
    
    # Process pieces
    piece_node_idx = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            # Create one-hot encoded piece features
            piece_feature = np.zeros(12, dtype=np.float32)  # 6 piece types * 2 colors
            piece_idx = piece.piece_type - 1 + (6 if piece.color else 0)  # 0-5 for white, 6-11 for black
            piece_feature[piece_idx] = 1.0
            
            # Add piece node
            piece_features.append(piece_feature)
            piece_to_node[(piece.piece_type, piece.color)] = piece_node_idx
            piece_node_idx += 1
    
    # Process squares
    square_node_idx = 0
    for square in chess.SQUARES:
        # Create square features (binary for now, can be extended)
        square_feature = np.array([1.0 if square in chess.SQUARES else 0.0], dtype=np.float32)
        square_features.append(square_feature)
        square_to_node[square] = square_node_idx
        square_node_idx += 1
    
    # Create piece to square edges
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            piece_node = piece_to_node[(piece.piece_type, piece.color)]
            square_node = square_to_node[square]
            piece_to_square_edges.append([piece_node, square_node])
    
    # Create square to square edges for adjacent squares
    for square1 in chess.SQUARES:
        for square2 in chess.SQUARES:
            if square1 != square2:
                # Check if squares are adjacent (including diagonally)
                file1, rank1 = chess.square_file(square1), chess.square_rank(square1)
                file2, rank2 = chess.square_file(square2), chess.square_rank(square2)
                if abs(file1 - file2) <= 1 and abs(rank1 - rank2) <= 1:
                    square_to_square_edges.append([
                        square_to_node[square1],
                        square_to_node[square2]
                    ])
    
    # Convert lists to numpy arrays before creating tensors
    piece_features = np.array(piece_features, dtype=np.float32)
    square_features = np.array(square_features, dtype=np.float32)
    
    # Convert to tensors (now using numpy arrays)
    piece_features = torch.from_numpy(piece_features)
    square_features = torch.from_numpy(square_features)
    
    # Create edge indices
    if piece_to_square_edges:
        piece_to_square_edge_index = torch.tensor(piece_to_square_edges, dtype=torch.long).t().contiguous()
    else:
        piece_to_square_edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    if square_to_square_edges:
        square_to_square_edge_index = torch.tensor(square_to_square_edges, dtype=torch.long).t().contiguous()
    else:
        square_to_square_edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    # Create heterogeneous graph data
    data = HeteroData()
    
    # Add node features and batch information
    data['piece'].x = piece_features
    data['piece'].batch = torch.zeros(len(piece_features), dtype=torch.long)  # All pieces in batch 0
    
    data['square'].x = square_features
    data['square'].batch = torch.zeros(len(square_features), dtype=torch.long)  # All squares in batch 0
    
    # Add edge indices
    data['piece', 'to', 'square'].edge_index = piece_to_square_edge_index
    data['square', 'to', 'square'].edge_index = square_to_square_edge_index
    
    return data 