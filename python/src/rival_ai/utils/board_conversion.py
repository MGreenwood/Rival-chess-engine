import torch
from torch_geometric.data import HeteroData
import chess
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Import PAG engine for ultra-dense features
try:
    import rival_ai_engine
    PAG_ENGINE_AVAILABLE = True
    logger.info("✅ Rust PAG engine available for ultra-dense features")
except ImportError:
    PAG_ENGINE_AVAILABLE = False
    logger.warning("⚠️ Rust PAG engine not available, falling back to basic features")

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

def extract_dense_pag_features(fen: str) -> Optional[Dict]:
    """Extract ultra-dense PAG features using Rust engine.
    
    Args:
        fen: FEN string of the chess position
        
    Returns:
        Dictionary containing PAG data or None if extraction fails
    """
    if not PAG_ENGINE_AVAILABLE:
        return None
        
    try:
        # Create Rust PAG engine instance
        engine = rival_ai_engine.PyPAGEngine()
        
        # Extract ultra-dense PAG features
        pag_data = engine.fen_to_dense_pag(fen)
        
        return pag_data
        
    except Exception as e:
        logger.warning(f"Failed to extract PAG features: {e}")
        return None

def board_to_hetero_data(board: chess.Board, use_dense_pag: bool = True) -> HeteroData:
    """Convert a chess board to heterogeneous graph data using ultra-dense PAG features.
    
    Args:
        board: Chess board to convert
        use_dense_pag: Whether to use ultra-dense PAG features from Rust engine
        
    Returns:
        HeteroData object containing:
        - piece nodes with ultra-dense features [num_pieces, 308] or basic features [num_pieces, 12]
        - critical_square nodes with dense features [num_critical_squares, 95+] or basic [64, 1]
        - comprehensive edge relationships with 256+ features
        - batch information for global pooling
    """
    fen = board.fen()
    
    # Try to extract ultra-dense PAG features first
    if use_dense_pag and PAG_ENGINE_AVAILABLE:
        pag_data = extract_dense_pag_features(fen)
        
        if pag_data is not None:
            return _pag_data_to_hetero_data(pag_data)
        else:
            logger.warning("PAG extraction failed, falling back to basic features")
    
    # Fallback to basic features
    logger.info("Using basic features (12-dim pieces, 1-dim squares)")
    hetero_data = _board_to_basic_hetero_data(board)
    logger.info(f"🔍 FALLBACK DEBUG: Piece tensor shape: {hetero_data['piece'].x.shape}")
    logger.info(f"🔍 FALLBACK DEBUG: Square tensor shape: {hetero_data['critical_square'].x.shape}")
    return hetero_data

def _pag_data_to_hetero_data(pag_data: Dict) -> HeteroData:
    """Convert ultra-dense PAG data to HeteroData format.
    
    Args:
        pag_data: Dictionary containing PAG nodes and edges from Rust engine
        
    Returns:
        HeteroData with ultra-dense features
    """
    data = HeteroData()
    
    try:
        # Extract node features and types from PAG engine format
        node_features = pag_data.get('node_features', [])
        node_types = pag_data.get('node_types', [])
        node_ids = pag_data.get('node_ids', [])
        
        logger.info(f"🔍 PAG DEBUG: Received {len(node_features)} nodes from Rust engine")
        logger.info(f"🔍 PAG DEBUG: Node types: {node_types}")
        
        if len(node_features) > 0 and len(node_types) > 0:
            # Debug: Show first few nodes and their dimensions
            for i in range(min(5, len(node_features))):
                features = node_features[i]
                node_type = node_types[i]
                if hasattr(features, '__len__'):
                    logger.info(f"🔍 PAG DEBUG: Node {i}: type='{node_type}', features={len(features)} dims")
                else:
                    logger.info(f"🔍 PAG DEBUG: Node {i}: type='{node_type}', features=scalar")
            
            # Separate pieces and critical squares
            piece_features = []
            square_features = []
            piece_indices = []
            square_indices = []
            
            for i, (features, node_type) in enumerate(zip(node_features, node_types)):
                if node_type == 'piece':
                    piece_features.append(features)
                    piece_indices.append(i)
                    logger.debug(f"🔍 PAG DEBUG: Added piece {i} with {len(features)} features")
                elif node_type == 'critical_square':
                    square_features.append(features) 
                    square_indices.append(i)
                    logger.debug(f"🔍 PAG DEBUG: Added square {i} with {len(features)} features")
                else:
                    logger.warning(f"🔍 PAG DEBUG: Unknown node type '{node_type}' for node {i}")
            
            logger.info(f"🔍 PAG DEBUG: Separated into {len(piece_features)} pieces and {len(square_features)} squares")
            
            # Add piece nodes
            if piece_features:
                # FIXED: Convert to numpy array first to avoid slow tensor creation warning
                piece_features_array = np.array(piece_features, dtype=np.float32)
                piece_tensor = torch.from_numpy(piece_features_array)
                data['piece'].x = piece_tensor
                data['piece'].batch = torch.zeros(len(piece_features), dtype=torch.long)
                logger.info(f"🔍 CONVERSION DEBUG: Created piece tensor with shape {piece_tensor.shape}")
                logger.info(f"✅ Loaded {len(piece_features)} pieces with {piece_tensor.shape[1]} ultra-dense features each")
            else:
                # Fallback to basic features if no pieces found
                data['piece'].x = torch.zeros((1, 308), dtype=torch.float32)
                data['piece'].batch = torch.zeros(1, dtype=torch.long)
                logger.warning("⚠️ No pieces found, using zero tensor")
            
            # Add critical square nodes
            if square_features:
                # FIXED: Convert to numpy array first to avoid slow tensor creation warning
                square_features_array = np.array(square_features, dtype=np.float32)
                square_tensor = torch.from_numpy(square_features_array)
                data['critical_square'].x = square_tensor
                data['critical_square'].batch = torch.zeros(len(square_features), dtype=torch.long)
                logger.info(f"🔍 CONVERSION DEBUG: Created square tensor with shape {square_tensor.shape}")
                logger.info(f"✅ Loaded {len(square_features)} critical squares with {square_tensor.shape[1]} dense features each")
            else:
                # Fallback if no critical squares
                data['critical_square'].x = torch.zeros((1, 95), dtype=torch.float32)
                data['critical_square'].batch = torch.zeros(1, dtype=torch.long)
                logger.warning("⚠️ No critical squares found, using zero tensor")
        
        # Extract edges from PAG engine format
        edge_features = pag_data.get('edge_features', [])
        edge_indices = pag_data.get('edge_indices', [])
        edge_types = pag_data.get('edge_types', [])
        
        if len(edge_indices) > 0 and len(edge_features) > 0:
            # Convert edge indices to PyTorch format and add to HeteroData
            # For now, treat all edges as piece-to-piece connections
            # TODO: Distinguish edge types based on source/target node types
            edge_indices_array = np.array(edge_indices, dtype=np.int64)
            edge_index_tensor = torch.from_numpy(edge_indices_array).t().contiguous()
            edge_features_array = np.array(edge_features, dtype=np.float32)
            edge_features_tensor = torch.from_numpy(edge_features_array)
            
            # Add as piece-to-piece edges for now (most common type)
            data['piece', 'to', 'piece'].edge_index = edge_index_tensor
            data['piece', 'to', 'piece'].edge_attr = edge_features_tensor
            logger.info(f"✅ Added {len(edge_indices)} edges with {edge_features_tensor.shape[1]} features each")

        
        logger.info("🚀 Successfully created HeteroData with ultra-dense PAG features!")
        return data
        
    except Exception as e:
        logger.error(f"Failed to convert PAG data to HeteroData: {e}")
        # Create minimal valid HeteroData as fallback
        fallback_data = HeteroData()
        fallback_data['piece'].x = torch.zeros((1, 308), dtype=torch.float32)
        fallback_data['piece'].batch = torch.zeros(1, dtype=torch.long)
        fallback_data['critical_square'].x = torch.zeros((1, 95), dtype=torch.float32)
        fallback_data['critical_square'].batch = torch.zeros(1, dtype=torch.long)
        return fallback_data

def _board_to_basic_hetero_data(board: chess.Board) -> HeteroData:
    """Fallback function for basic features when PAG engine is unavailable.
    FIXED: Creates same dimensions as ultra-dense PAG for model compatibility.
    
    Args:
        board: Chess board to convert
        
    Returns:
        HeteroData with basic features padded to ultra-dense dimensions:
        - piece features: 308-dim (12 real + 296 padding)
        - critical_square features: 95-dim (1 real + 94 padding)
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
            # Create padded piece features compatible with ultra-dense PAG (308-dim)
            piece_feature = np.zeros(308, dtype=np.float32)  # FIXED: 308 dims to match model
            piece_idx = piece.piece_type - 1 + (6 if piece.color else 0)  # 0-5 for white, 6-11 for black
            piece_feature[piece_idx] = 1.0  # Set the basic feature in first 12 positions
            
            # Add piece node
            piece_features.append(piece_feature)
            piece_to_node[(piece.piece_type, piece.color)] = piece_node_idx
            piece_node_idx += 1
    
    # Process squares - FIXED: Create 95-dim critical square features  
    square_node_idx = 0
    for square in chess.SQUARES:
        # Create padded critical square features compatible with ultra-dense PAG (95-dim)
        square_feature = np.zeros(95, dtype=np.float32)  # FIXED: 95 dims to match model
        square_feature[0] = 1.0 if square in chess.SQUARES else 0.0  # Basic feature in first position
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
    
    # Convert lists to numpy arrays before creating tensors - FIXED: Use correct dimensions
    piece_features = np.array(piece_features, dtype=np.float32) if piece_features else np.zeros((0, 308), dtype=np.float32)  # FIXED: 308 dims
    square_features = np.array(square_features, dtype=np.float32)  # Already 95-dim from above
    
    # Convert to tensors
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
    data['piece'].batch = torch.zeros(len(piece_features), dtype=torch.long) if len(piece_features) > 0 else torch.zeros(0, dtype=torch.long)
    
    data['critical_square'].x = square_features  # FIXED: Use 'critical_square' to match model expectation
    data['critical_square'].batch = torch.zeros(len(square_features), dtype=torch.long)
    
    # Add edge indices - FIXED: Use 'critical_square' node name
    data['piece', 'to', 'critical_square'].edge_index = piece_to_square_edge_index
    data['critical_square', 'to', 'critical_square'].edge_index = square_to_square_edge_index
    
    logger.info(f"✅ Created basic HeteroData with model-compatible dimensions:")
    logger.info(f"   Pieces: {len(piece_features)} nodes × 308 features")
    logger.info(f"   Critical squares: {len(square_features)} nodes × 95 features")
    logger.debug(f"   Piece tensor shape: {piece_features.shape}")
    logger.debug(f"   Square tensor shape: {square_features.shape}")
    
    return data 