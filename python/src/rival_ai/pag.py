"""
Positional Adjacency Graph (PAG) implementation for chess positions.
"""

import logging
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from torch_geometric.data import HeteroData
from enum import Enum
from functools import lru_cache
from threading import Lock
import weakref
import chess
from chess import Square, Piece, PieceType, Color

from rival_ai.config import MCTSConfig

logger = logging.getLogger(__name__)

@dataclass
class PAGConfig:
    """Configuration for PAG construction."""
    edge_dims: Dict[str, int] = None  # Will be set to default in __post_init__
    use_advanced_features: bool = True
    use_piece_mobility: bool = True
    use_king_safety: bool = True
    use_pawn_structure: bool = True
    use_piece_coordination: bool = True
    use_center_control: bool = True
    use_material_balance: bool = True
    use_attack_patterns: bool = True

    def __post_init__(self):
        """Set default edge dimensions if not provided."""
        if self.edge_dims is None:
            self.edge_dims = {
                'direct_relation': 12,  # Increased for more features
                'control': 8,  # Added king safety
                'mobility': 10,  # Added piece-specific mobility
                'cooperation': 8,  # Added attack patterns
                'obstruction': 8,  # Added blocking patterns
                'vulnerability': 10,  # Added piece-specific threats
                'pawn_structure': 12,  # Added pawn chain analysis
                'king_safety': 10,  # New edge type
                'center_control': 8,  # New edge type
                'material_tension': 6,  # New edge type
            }

class RayType(Enum):
    ORTHOGONAL = 0
    DIAGONAL = 1
    KNIGHT = 2
    PAWN = 3
    NONE = 4

class MoveType(Enum):
    NORMAL = 0
    CAPTURE = 1
    CASTLE = 2
    EN_PASSANT = 3
    PROMOTION = 4

# Global cache with weak references to prevent memory leaks
_legal_moves_cache = weakref.WeakKeyDictionary()
_control_counts_cache = weakref.WeakKeyDictionary()
_cache_lock = Lock()

class ChessFeatures(Enum):
    """Enhanced chess features for PAG."""
    MOBILITY = 'mobility'
    CONTROL = 'control'
    ATTACK = 'attack'
    DEFENSE = 'defense'
    KING_SAFETY = 'king_safety'
    PAWN_STRUCTURE = 'pawn_structure'
    CENTER_CONTROL = 'center_control'
    MATERIAL_BALANCE = 'material_balance'
    PIECE_COORDINATION = 'piece_coordination'
    ATTACK_PATTERNS = 'attack_patterns'

@dataclass
class PositionalAdjacencyGraph:
    """Represents a chess position as a graph where nodes are pieces and critical squares,
    and edges represent various relationships between them."""
    
    def __init__(self, board: chess.Board, config: Optional[MCTSConfig] = None):
        """Initialize the graph for a given board position.
        
        Args:
            board: A chess.Board object representing the current position
            config: Optional MCTS configuration for edge dimensions
        """
        self.board = board
        self.config = config or MCTSConfig()
        
        # Initialize node dictionaries with enhanced features
        self.piece_nodes: Dict[int, Dict] = {}  # square -> node features
        self.critical_square_nodes: Dict[int, Dict] = {}  # square -> node features
        
        # Initialize edge feature dictionaries with new edge types
        self.direct_edges: Dict[Tuple[int, int], Dict] = {}  # (from_square, to_square) -> edge features
        self.control_edges: Dict[Tuple[int, int], Dict] = {}  # (piece_square, square) -> edge features
        self.mobility_edges: Dict[Tuple[int, int], Dict] = {}  # (piece_square, square) -> edge features
        self.cooperative_edges: Dict[Tuple[int, int], Dict] = {}  # (piece_square, piece_square) -> edge features
        self.obstructive_edges: Dict[Tuple[int, int], Dict] = {}  # (piece_square, piece_square) -> edge features
        self.vulnerability_edges: Dict[Tuple[int, int], Dict] = {}  # (piece_square, piece_square) -> edge features
        self.pawn_structure_edges: Dict[Tuple[int, int], Dict] = {}  # (pawn_square, pawn_square) -> edge features
        self.king_safety_edges: Dict[Tuple[int, int], Dict] = {}  # (king_square, defender_square) -> edge features
        self.center_control_edges: Dict[Tuple[int, int], Dict] = {}  # (piece_square, center_square) -> edge features
        self.material_tension_edges: Dict[Tuple[int, int], Dict] = {}  # (attacker_square, target_square) -> edge features
        
        # Build the graph
        self._build_graph()
    
    def _build_graph(self):
        """Build the positional adjacency graph with enhanced features."""
        # Add piece nodes with enhanced features
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                self.piece_nodes[square] = self._get_piece_features(square, piece)

        # Add critical squares (center, outposts, key squares)
        self._add_critical_squares()

        # Build all edge types
        self._build_direct_edges()
        self._build_control_edges()
        self._build_mobility_edges()
        self._build_cooperative_edges()
        self._build_obstructive_edges()
        self._build_vulnerability_edges()
        self._build_pawn_structure_edges()
        self._build_king_safety_edges()
        self._build_center_control_edges()
        self._build_material_tension_edges()
    
    def _get_piece_features(self, square: int, piece: chess.Piece) -> Dict:
        """Get enhanced piece features."""
        features = {
            'piece_type': piece.piece_type,
            'color': int(piece.color),
            'mobility': self._calculate_mobility(square, piece),
            'control': self._calculate_control(square, piece),
            'attack_value': self._calculate_attack_value(square, piece),
            'defense_value': self._calculate_defense_value(square, piece),
            'development': self._calculate_development(square, piece),
            'center_distance': self._calculate_center_distance(square),
            'king_distance': self._calculate_king_distance(square, piece),
            'pawn_shield': self._calculate_pawn_shield(square, piece),
        }
        return features

    def _calculate_mobility(self, square: int, piece: chess.Piece) -> float:
        """Calculate piece mobility with advanced metrics."""
        if not self.config.use_piece_mobility:
            return 0.0

        legal_moves = set(move.to_square for move in self.board.legal_moves 
                         if move.from_square == square)
        
        # Weight mobility by piece type and position
        weights = {
            chess.PAWN: 0.5,
            chess.KNIGHT: 1.0,
            chess.BISHOP: 1.2,
            chess.ROOK: 1.0,
            chess.QUEEN: 0.8,
            chess.KING: 0.3
        }
        
        base_mobility = len(legal_moves) * weights[piece.piece_type]
        
        # Adjust for center control
        center_moves = sum(1 for sq in legal_moves if self._is_center_square(sq))
        center_bonus = center_moves * 0.2
        
        # Adjust for piece development
        development_bonus = self._calculate_development(square, piece) * 0.1
        
        return base_mobility + center_bonus + development_bonus

    def _calculate_control(self, square: int, piece: chess.Piece) -> float:
        """Calculate square control with advanced metrics."""
        if not self.config.use_center_control:
            return 0.0

        controlled_squares = self._get_controlled_squares(square, piece)
        
        # Weight control by square importance
        total_control = 0.0
        for sq in controlled_squares:
            weight = 1.0
            if self._is_center_square(sq):
                weight = 2.0
            elif self._is_extended_center(sq):
                weight = 1.5
            elif self._is_king_zone(sq, piece.color):
                weight = 1.8
            total_control += weight
            
        return total_control

    def _calculate_attack_value(self, square: int, piece: chess.Piece) -> float:
        """Calculate attacking potential."""
        if not self.config.use_attack_patterns:
            return 0.0

        attacked_pieces = self._get_attacked_pieces(square, piece)
        
        # Weight attacks by target value
        piece_values = {
            chess.PAWN: 1.0,
            chess.KNIGHT: 3.0,
            chess.BISHOP: 3.0,
            chess.ROOK: 5.0,
            chess.QUEEN: 9.0,
            chess.KING: 0.5  # Lower weight for king attacks as they rarely lead to immediate capture
        }
        
        total_attack = sum(piece_values[p.piece_type] for p in attacked_pieces)
        
        # Add bonus for attacking protected pieces
        protected_attacks = sum(1.0 for p in attacked_pieces if self._is_protected(p))
        total_attack += protected_attacks * 0.5
        
        return total_attack

    def _build_king_safety_edges(self):
        """Build edges representing king safety features."""
        if not self.config.use_king_safety:
            return

        for color in [chess.WHITE, chess.BLACK]:
            king_square = self.board.king(color)
            if king_square is None:
                continue

            # Get king zone squares
            king_zone = self._get_king_zone(king_square)
            
            # Analyze pawn shield
            pawn_shield_value = self._analyze_pawn_shield(king_square, color)
            
            # Analyze piece defense
            for square in chess.SQUARES:
                piece = self.board.piece_at(square)
                if piece and piece.color == color:
                    defense_value = self._calculate_king_defense_contribution(square, king_square)
                    if defense_value > 0:
                        self.king_safety_edges[(square, king_square)] = {
                            'defense_value': defense_value,
                            'distance': chess.square_distance(square, king_square),
                            'pawn_shield': float(square in pawn_shield_value),
                            'is_defender': 1.0,
                            'attack_lines': self._count_attack_lines(square, king_square),
                        }

    def _build_center_control_edges(self):
        """Build edges representing center control."""
        if not self.config.use_center_control:
            return

        center_squares = {chess.E4, chess.E5, chess.D4, chess.D5}
        extended_center = {chess.C3, chess.C4, chess.C5, chess.C6,
                         chess.D3, chess.D6,
                         chess.E3, chess.E6,
                         chess.F3, chess.F4, chess.F5, chess.F6}
        
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if not piece:
                continue
                
            # Calculate control over center squares
            controlled_squares = self._get_controlled_squares(square, piece)
            center_control = controlled_squares & center_squares
            extended_control = controlled_squares & extended_center
            
            for target in center_control:
                self.center_control_edges[(square, target)] = {
                    'control_value': 2.0,  # Higher weight for main center
                    'distance': chess.square_distance(square, target),
                    'is_attacked': float(self._is_square_attacked(target, piece.color)),
                    'is_defended': float(self._is_square_defended(target, piece.color)),
                }
                
            for target in extended_control:
                self.center_control_edges[(square, target)] = {
                    'control_value': 1.0,  # Lower weight for extended center
                    'distance': chess.square_distance(square, target),
                    'is_attacked': float(self._is_square_attacked(target, piece.color)),
                    'is_defended': float(self._is_square_defended(target, piece.color)),
                }

    def _build_material_tension_edges(self):
        """Build edges representing material tension/threats."""
        if not self.config.use_material_balance:
            return

        for square in chess.SQUARES:
            attacker = self.board.piece_at(square)
            if not attacker:
                continue
                
            # Find all pieces under attack
            attacked_pieces = self._get_attacked_pieces(square, attacker)
            
            for target_square, target_piece in attacked_pieces:
                # Calculate material tension features
                self.material_tension_edges[(square, target_square)] = {
                    'material_delta': self._calculate_material_delta(attacker, target_piece),
                    'is_protected': float(self._is_protected(target_square)),
                    'exchange_value': self._calculate_exchange_value(square, target_square),
                    'attack_count': self._count_attackers(target_square),
                    'defense_count': self._count_defenders(target_square),
                }

    def _get_king_zone(self, king_square: int) -> Set[int]:
        """Get squares in the king's safety zone."""
        rank = chess.square_rank(king_square)
        file = chess.square_file(king_square)
        
        zone = set()
        for r in range(max(0, rank - 2), min(8, rank + 2)):
            for f in range(max(0, file - 2), min(8, file + 2)):
                zone.add(chess.square(f, r))
        
        return zone

    def _analyze_pawn_shield(self, king_square: int, color: chess.Color) -> Dict[int, float]:
        """Analyze pawn shield structure in front of king."""
        rank = chess.square_rank(king_square)
        file = chess.square_file(king_square)
        
        shield_values = {}
        pawn_ranks = range(rank + 1, min(8, rank + 3)) if color == chess.WHITE else range(max(0, rank - 2), rank)
        
        for r in pawn_ranks:
            for f in range(max(0, file - 1), min(8, file + 2)):
                square = chess.square(f, r)
                piece = self.board.piece_at(square)
                if piece and piece.piece_type == chess.PAWN and piece.color == color:
                    # Weight by distance from king
                    distance = abs(f - file) + abs(r - rank)
                    shield_values[square] = 1.0 / (1.0 + distance)
                    
        return shield_values

    def _calculate_king_defense_contribution(self, piece_square: int, king_square: int) -> float:
        """Calculate how much a piece contributes to king defense."""
        piece = self.board.piece_at(piece_square)
        if not piece:
            return 0.0
            
        # Base defense value by piece type
        defense_values = {
            chess.PAWN: 1.0,
            chess.KNIGHT: 0.8,
            chess.BISHOP: 0.9,
            chess.ROOK: 1.2,
            chess.QUEEN: 1.5,
        }
        
        base_value = defense_values.get(piece.piece_type, 0.0)
        
        # Adjust by distance to king
        distance = chess.square_distance(piece_square, king_square)
        distance_factor = 1.0 / (1.0 + distance)
        
        # Adjust by control of squares around king
        king_zone = self._get_king_zone(king_square)
        controlled_squares = self._get_controlled_squares(piece_square, piece)
        zone_control = len(controlled_squares & king_zone) * 0.2
        
        return base_value * distance_factor * (1.0 + zone_control)

    def _calculate_material_delta(self, attacker: chess.Piece, target: chess.Piece) -> float:
        """Calculate material value change in a capture."""
        piece_values = {
            chess.PAWN: 1.0,
            chess.KNIGHT: 3.0,
            chess.BISHOP: 3.0,
            chess.ROOK: 5.0,
            chess.QUEEN: 9.0,
        }
        
        return piece_values.get(target.piece_type, 0.0) - piece_values.get(attacker.piece_type, 0.0)

    def _calculate_exchange_value(self, attacker_square: int, target_square: int) -> float:
        """Calculate the value of a potential exchange."""
        # Count attackers and defenders
        attackers = self._count_attackers(target_square)
        defenders = self._count_defenders(target_square)
        
        # Get piece values
        attacker = self.board.piece_at(attacker_square)
        target = self.board.piece_at(target_square)
        
        if not attacker or not target:
            return 0.0
            
        # Simple SEE (Static Exchange Evaluation)
        if attackers > defenders:
            return self._calculate_material_delta(attacker, target)
        elif attackers == defenders:
            return 0.0
        else:
            return -self._calculate_material_delta(attacker, target)

    def to_hetero_data(self) -> HeteroData:
        """Convert the PAG to a heterogeneous graph data object."""
        data = HeteroData()
        
        # Add node features
        piece_features = []
        piece_indices = []
        for square, features in self.piece_nodes.items():
            piece_features.append(self._normalize_features(features))
            piece_indices.append(square)
        
        if piece_features:
            data['piece'].x = torch.tensor(piece_features, dtype=torch.float)
            data['piece'].indices = torch.tensor(piece_indices, dtype=torch.long)
        
        # Add edge features for all edge types
        for edge_type, edges in [
            ('direct', self.direct_edges),
            ('control', self.control_edges),
            ('mobility', self.mobility_edges),
            ('cooperation', self.cooperative_edges),
            ('obstruction', self.obstructive_edges),
            ('vulnerability', self.vulnerability_edges),
            ('pawn_structure', self.pawn_structure_edges),
            ('king_safety', self.king_safety_edges),
            ('center_control', self.center_control_edges),
            ('material_tension', self.material_tension_edges),
        ]:
            if not edges:
                continue
                
            edge_indices = []
            edge_features = []
            
            for (src, dst), features in edges.items():
                edge_indices.append([src, dst])
                edge_features.append(self._normalize_features(features))
                
            if edge_indices:
                data[edge_type].edge_index = torch.tensor(edge_indices, dtype=torch.long).t()
                data[edge_type].edge_attr = torch.tensor(edge_features, dtype=torch.float)
        
        return data

    def _normalize_features(self, features: Dict) -> List[float]:
        """Normalize feature values to a fixed range."""
        # Convert categorical features to one-hot
        if 'piece_type' in features:
            piece_type = features['piece_type']
            one_hot = [0.0] * 6  # 6 piece types
            one_hot[piece_type - 1] = 1.0
            features['piece_type'] = one_hot
            
        # Normalize numerical features to [0, 1]
        for key in ['mobility', 'control', 'attack_value', 'defense_value']:
            if key in features:
                features[key] = min(1.0, features[key] / 10.0)
                
        # Convert boolean features to float
        for key in ['is_attacked', 'is_defended', 'is_defender']:
            if key in features:
                features[key] = float(features[key])
                
        # Flatten dictionary to list
        return [v if not isinstance(v, list) else v[0] for v in features.values()]

    def _get_critical_squares(self) -> Set[int]:
        """Get set of critical squares in the position."""
        critical_squares = set()
        
        # Add center squares
        center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
        critical_squares.update(center_squares)
        
        # Add squares around kings
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece is not None and piece.piece_type == chess.KING:
                # Add squares around king
                for offset in [-9, -8, -7, -1, 1, 7, 8, 9]:
                    target = square + offset
                    if 0 <= target < 64 and abs((target % 8) - (square % 8)) <= 1:
                        critical_squares.add(target)
        
        return critical_squares
    
    def _is_piece_pinned(self, square: int) -> bool:
        """Check if a piece is pinned using chess.Board methods."""
        piece = self.board.piece_at(square)
        if piece is None:
            return False
        
        # Use chess.Board's is_pinned method
        return self.board.is_pinned(piece.color, square)
    
    def _is_piece_checking(self, square: int) -> bool:
        """Check if a piece is giving check using chess.Board methods."""
        piece = self.board.piece_at(square)
        if piece is None:
            return False
        
        # Make a copy of the board
        board_copy = self.board.copy()
        # Remove the piece
        board_copy.remove_piece_at(square)
        # Check if the opponent's king is in check
        return board_copy.is_check() != self.board.is_check()
    
    def _is_piece_protected(self, square: int) -> bool:
        """Check if a piece is protected by another piece using chess.Board methods."""
        piece = self.board.piece_at(square)
        if piece is None:
            return False
        
        # Check if any friendly piece can move to this square
        for move in self.board.legal_moves:
            if move.to_square == square:
                from_piece = self.board.piece_at(move.from_square)
                if from_piece is not None and from_piece.color == piece.color:
                    return True
        return False
    
    def _is_piece_mobile(self, square: int) -> bool:
        """Check if a piece has good mobility."""
        piece = self.board.piece_at(square)
        if piece is None:
            return False
        
        # Count legal moves for this piece
        move_count = 0
        for move in self.board.legal_moves:
            if move.from_square == square:
                move_count += 1
        
        # Consider piece type for mobility threshold
        if piece.piece_type == chess.PAWN:
            return move_count > 0
        elif piece.piece_type == chess.KNIGHT:
            return move_count > 2
        elif piece.piece_type == chess.BISHOP:
            return move_count > 3
        elif piece.piece_type == chess.ROOK:
            return move_count > 4
        elif piece.piece_type == chess.QUEEN:
            return move_count > 5
        else:  # King
            return move_count > 2
    
    def _is_piece_valuable(self, square: int) -> bool:
        """Check if a piece is valuable (based on position and role)."""
        piece = self.board.piece_at(square)
        if piece is None:
            return False
        
        # Basic piece values
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0  # King value not included in material count
        }
        
        # Get piece value
        value = piece_values[piece.piece_type]
        
        # Add positional bonus
        if piece.piece_type == chess.PAWN:
            # Pawns are more valuable in center and advanced positions
            rank = square // 8
            file = square % 8
            if 2 <= file <= 5:  # Center files
                value += 0.5
            if (piece.color == chess.WHITE and rank >= 4) or (piece.color == chess.BLACK and rank <= 3):
                value += 0.5
        elif piece.piece_type == chess.KNIGHT:
            # Knights are more valuable in center
            if 2 <= square // 8 <= 5 and 2 <= square % 8 <= 5:
                value += 0.5
        elif piece.piece_type == chess.BISHOP:
            # Bishops are more valuable with open diagonals
            if self._has_open_diagonals(square):
                value += 0.5
        elif piece.piece_type == chess.ROOK:
            # Rooks are more valuable on open files
            if self._is_on_open_file(square):
                value += 0.5
        
        return value >= 3  # Consider pieces with value >= 3 as valuable
    
    def _is_square_attacked(self, square: int, color: bool) -> bool:
        """Check if a square is attacked by a given color using chess.Board methods."""
        # Use chess.Board's is_attacked_by method
        return self.board.is_attacked_by(color, square)
    
    def _is_square_controlled(self, square: int) -> bool:
        """Check if a square is controlled by either side."""
        return self._is_square_attacked(square, chess.WHITE) or self._is_square_attacked(square, chess.BLACK)
    
    def _is_pawn_square(self, square: int) -> bool:
        """Check if a square is a pawn square (in front of pawns)."""
        rank = square // 8
        file = square % 8
        
        # Check for pawns in front
        for color in [chess.WHITE, chess.BLACK]:
            pawn_rank = rank - 1 if color == chess.WHITE else rank + 1
            if 0 <= pawn_rank < 8:
                if self.board.piece_at(chess.square(file, pawn_rank)) == chess.Piece(chess.PAWN, color):
                    return True
        return False
    
    def _is_center_square(self, square: int) -> bool:
        """Check if a square is in the center."""
        rank = square // 8
        file = square % 8
        return 2 <= rank <= 5 and 2 <= file <= 5
    
    def _is_king_square(self, square: int) -> bool:
        """Check if a square is near a king."""
        for king_square in chess.SQUARES:
            piece = self.board.piece_at(king_square)
            if piece is not None and piece.piece_type == chess.KING:
                if abs((square // 8) - (king_square // 8)) <= 1 and abs((square % 8) - (king_square % 8)) <= 1:
                    return True
        return False
    
    def _has_direct_relation(self, square1: int, square2: int) -> bool:
        """Check if two pieces have a direct relation."""
        piece1 = self.board.piece_at(square1)
        piece2 = self.board.piece_at(square2)
        if piece1 is None or piece2 is None:
            return False
        
        # Check if pieces are of the same color
        if piece1.color != piece2.color:
            return False
        
        # Check if pieces can attack each other
        return self._can_attack(square1, square2) or self._can_attack(square2, square1)
    
    def _get_relation_type(self, square1: int, square2: int) -> str:
        """Get the type of relation between two pieces."""
        piece1 = self.board.piece_at(square1)
        piece2 = self.board.piece_at(square2)
        
        if piece1.piece_type == piece2.piece_type:
            return 'same_type'
        elif piece1.color == piece2.color:
            return 'friendly'
        else:
            return 'opposing'
    
    def _get_distance(self, square1: int, square2: int) -> int:
        """Get the Manhattan distance between two squares."""
        rank1, file1 = square1 // 8, square1 % 8
        rank2, file2 = square2 // 8, square2 % 8
        return abs(rank2 - rank1) + abs(file2 - file1)
    
    def _is_path_blocked(self, square1: int, square2: int) -> bool:
        """Check if the path between two squares is blocked using chess.Board methods."""
        # Get the squares between square1 and square2
        try:
            move = chess.Move(square1, square2)
            # If the move is not a straight line or diagonal, it's not blocked
            if not (chess.square_rank(square1) == chess.square_rank(square2) or
                   chess.square_file(square1) == chess.square_file(square2) or
                   abs(chess.square_rank(square2) - chess.square_rank(square1)) ==
                   abs(chess.square_file(square2) - chess.square_file(square1))):
                return False
            
            # Get the squares in between
            between_squares = chess.SquareSet.between(square1, square2)
            # Check if any square in between has a piece
            return any(self.board.piece_at(square) is not None for square in between_squares)
        except ValueError:
            return False
    
    def _are_pieces_cooperative(self, square1: int, square2: int) -> bool:
        """Check if two pieces are cooperating."""
        piece1 = self.board.piece_at(square1)
        piece2 = self.board.piece_at(square2)
        if piece1 is None or piece2 is None or piece1.color != piece2.color:
            return False
        
        # Check if pieces protect each other
        return self._protects(square1, square2) or self._protects(square2, square1)
    
    def _get_cooperation_strength(self, square1: int, square2: int) -> float:
        """Get the strength of cooperation between two pieces."""
        strength = 0.0
        
        # Add bonus for mutual protection
        if self._protects(square1, square2) and self._protects(square2, square1):
            strength += 1.0
        
        # Add bonus for attacking the same square
        common_targets = self._get_common_targets(square1, square2)
        strength += len(common_targets) * 0.5
        
        return min(strength, 1.0)
    
    def _are_pieces_obstructive(self, square1: int, square2: int) -> bool:
        """Check if two pieces are obstructing each other."""
        piece1 = self.board.piece_at(square1)
        piece2 = self.board.piece_at(square2)
        if piece1 is None or piece2 is None or piece1.color != piece2.color:
            return False
        
        # Check if pieces block each other's moves
        return self._blocks(square1, square2) or self._blocks(square2, square1)
    
    def _get_obstruction_severity(self, square1: int, square2: int) -> float:
        """Get the severity of obstruction between two pieces."""
        severity = 0.0
        
        # Add penalty for blocking each other's moves
        if self._blocks(square1, square2):
            severity += 0.5
        if self._blocks(square2, square1):
            severity += 0.5
        
        # Add penalty for restricting mobility
        mobility1 = len(list(self.board.legal_moves))
        mobility2 = len(list(self.board.legal_moves))
        severity += (mobility1 + mobility2) * 0.1
        
        return min(severity, 1.0)
    
    def _are_pieces_vulnerable(self, square1: int, square2: int) -> bool:
        """Check if two pieces are vulnerable to attack."""
        piece1 = self.board.piece_at(square1)
        piece2 = self.board.piece_at(square2)
        if piece1 is None or piece2 is None:
            return False
        
        # Check if pieces are undefended
        return (not self._is_piece_protected(square1) and self._is_square_attacked(square1, not piece1.color)) or \
               (not self._is_piece_protected(square2) and self._is_square_attacked(square2, not piece2.color))
    
    def _get_vulnerability_risk(self, square1: int, square2: int) -> float:
        """Get the risk level of vulnerability between two pieces."""
        risk = 0.0
        
        # Add risk for undefended pieces
        if not self._is_piece_protected(square1):
            risk += 0.5
        if not self._is_piece_protected(square2):
            risk += 0.5
        
        # Add risk for pieces under attack
        piece1 = self.board.piece_at(square1)
        piece2 = self.board.piece_at(square2)
        if piece1 is not None and self._is_square_attacked(square1, not piece1.color):
            risk += 0.5
        if piece2 is not None and self._is_square_attacked(square2, not piece2.color):
            risk += 0.5
        
        return min(risk, 1.0)
    
    def _controls_square(self, square: int, target_square: int) -> bool:
        """Check if a piece controls a square."""
        piece = self.board.piece_at(square)
        if piece is None:
            return False
        
        # Check if piece can move to or attack the target square
        return self._can_move_to(square, target_square) or self._can_attack(square, target_square)
    
    def _get_control_strength(self, square: int, target_square: int) -> float:
        """Get the strength of control over a square."""
        strength = 0.0
        
        # Add strength for direct control
        if self._can_move_to(square, target_square):
            strength += 0.5
        
        # Add strength for attacking
        if self._can_attack(square, target_square):
            strength += 0.5
        
        # Add strength for protection
        if self._protects(square, target_square):
            strength += 0.5
        
        return min(strength, 1.0)
    
    def _can_move_to(self, square: int, target_square: int) -> bool:
        """Check if a piece can move to a square using chess.Board methods."""
        piece = self.board.piece_at(square)
        if piece is None:
            return False
        
        # Create a move and check if it's legal
        try:
            move = chess.Move(square, target_square)
            return move in self.board.legal_moves
        except ValueError:
            return False
    
    def _get_mobility_value(self, square: int, target_square: int) -> float:
        """Get the value of mobility to a square."""
        value = 0.0
        
        # Add value for center control
        if self._is_center_square(target_square):
            value += 0.5
        
        # Add value for attacking
        if self._can_attack(square, target_square):
            value += 0.5
        
        # Add value for protection
        if self._protects(square, target_square):
            value += 0.5
        
        return min(value, 1.0)
    
    def _are_pawns_related(self, square1: int, square2: int) -> bool:
        """Check if two pawns are related in structure."""
        piece1 = self.board.piece_at(square1)
        piece2 = self.board.piece_at(square2)
        if piece1 is None or piece2 is None or piece1.piece_type != chess.PAWN or piece2.piece_type != chess.PAWN:
            return False
        
        # Check if pawns are of the same color
        if piece1.color != piece2.color:
            return False
        
        # Check if pawns are adjacent or in the same file
        rank1, file1 = square1 // 8, square1 % 8
        rank2, file2 = square2 // 8, square2 % 8
        return abs(file2 - file1) <= 1 or file1 == file2
    
    def _get_pawn_relation_type(self, square1: int, square2: int) -> str:
        """Get the type of relation between two pawns."""
        rank1, file1 = square1 // 8, square1 % 8
        rank2, file2 = square2 // 8, square2 % 8
        
        if file1 == file2:
            return 'doubled'
        elif abs(file2 - file1) == 1:
            return 'adjacent'
        else:
            return 'isolated'
    
    def _get_pawn_structure_strength(self, square1: int, square2: int) -> float:
        """Get the strength of pawn structure between two pawns."""
        strength = 0.0
        
        # Add strength for connected pawns
        if self._are_pawns_connected(square1, square2):
            strength += 0.5
        
        # Add strength for advanced pawns
        if self._is_pawn_advanced(square1):
            strength += 0.25
        if self._is_pawn_advanced(square2):
            strength += 0.25
        
        return min(strength, 1.0)
    
    def _are_pawns_connected(self, square1: int, square2: int) -> bool:
        """Check if two pawns are connected."""
        rank1, file1 = square1 // 8, square1 % 8
        rank2, file2 = square2 // 8, square2 % 8
        
        # Pawns are connected if they are adjacent and on the same rank
        return abs(file2 - file1) == 1 and rank1 == rank2
    
    def _is_pawn_advanced(self, square: int) -> bool:
        """Check if a pawn is advanced."""
        piece = self.board.piece_at(square)
        if piece is None or piece.piece_type != chess.PAWN:
            return False
        
        rank = square // 8
        return (piece.color == chess.WHITE and rank >= 4) or (piece.color == chess.BLACK and rank <= 3)
    
    def _has_open_diagonals(self, square: int) -> bool:
        """Check if a piece has open diagonals."""
        piece = self.board.piece_at(square)
        if piece is None or piece.piece_type != chess.BISHOP:
            return False
        
        # Check each diagonal direction
        for dr, df in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
            has_open = False
            current_rank, current_file = square // 8 + dr, square % 8 + df
            while 0 <= current_rank < 8 and 0 <= current_file < 8:
                if self.board.piece_at(chess.square(current_file, current_rank)) is not None:
                    break
                has_open = True
                current_rank += dr
                current_file += df
            if has_open:
                return True
        return False
    
    def _is_on_open_file(self, square: int) -> bool:
        """Check if a piece is on an open file."""
        piece = self.board.piece_at(square)
        if piece is None or piece.piece_type != chess.ROOK:
            return False
        
        file = square % 8
        # Check if there are any pawns on this file
        for rank in range(8):
            if self.board.piece_at(chess.square(file, rank)) == chess.Piece(chess.PAWN, chess.WHITE) or \
               self.board.piece_at(chess.square(file, rank)) == chess.Piece(chess.PAWN, chess.BLACK):
                return False
        return True
    
    def _can_attack(self, square1: int, square2: int) -> bool:
        """Check if a piece can attack another square using chess.Board methods."""
        piece = self.board.piece_at(square1)
        if piece is None:
            return False
        
        # Create a move and check if it's a legal capture
        try:
            move = chess.Move(square1, square2)
            return move in self.board.legal_moves and self.board.is_capture(move)
        except ValueError:
            return False
    
    def _protects(self, square1: int, square2: int) -> bool:
        """Check if a piece protects another square."""
        piece1 = self.board.piece_at(square1)
        piece2 = self.board.piece_at(square2)
        if piece1 is None or piece2 is None or piece1.color != piece2.color:
            return False
        
        # Check if piece1 can move to square2 to protect piece2
        return self._can_move_to(square1, square2)
    
    def _blocks(self, square1: int, square2: int) -> bool:
        """Check if a piece blocks another piece's moves."""
        piece1 = self.board.piece_at(square1)
        piece2 = self.board.piece_at(square2)
        if piece1 is None or piece2 is None or piece1.color != piece2.color:
            return False
        
        # Check if piece1 is in the way of piece2's moves
        for move in self.board.legal_moves:
            if move.from_square == square2:
                if self._is_path_blocked(square2, move.to_square) and \
                   self._is_in_path(square1, square2, move.to_square):
                    return True
        return False
    
    def _is_in_path(self, square: int, start: int, end: int) -> bool:
        """Check if a square is in the path between two squares."""
        rank1, file1 = start // 8, start % 8
        rank2, file2 = end // 8, end % 8
        rank, file = square // 8, square % 8
        
        # Get direction
        dr = 0 if rank2 == rank1 else (rank2 - rank1) // abs(rank2 - rank1)
        df = 0 if file2 == file1 else (file2 - file1) // abs(file2 - file1)
        
        # Check if square is in the path
        current_rank, current_file = rank1 + dr, file1 + df
        while (current_rank, current_file) != (rank2, file2):
            if current_rank == rank and current_file == file:
                return True
            current_rank += dr
            current_file += df
        
        return False
    
    def _get_common_targets(self, square1: int, square2: int) -> Set[int]:
        """Get set of squares that both pieces can attack."""
        targets = set()
        
        # Get all squares that piece1 can attack
        for move in self.board.legal_moves:
            if move.from_square == square1:
                targets.add(move.to_square)
        
        # Get all squares that piece2 can attack
        for move in self.board.legal_moves:
            if move.from_square == square2 and move.to_square in targets:
                targets.add(move.to_square)
        
        return targets
    
    def _convert_to_tensors(self) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Convert the graph data to PyTorch tensors.
        
        Returns:
            Tuple containing:
            - Node features tensor
            - Node types tensor
            - Edge indices dictionary
            - Edge features dictionary
        """
        # Process piece nodes
        piece_features = []
        piece_types = []
        for square, features in self.piece_nodes.items():
            piece_features.append([
                features['type'] / 6.0,  # Normalize piece type
                float(features['color']),  # Color (0 or 1)
                features['value'] / 100.0,  # Normalize value
                features['mobility'],  # Already 0-1
                features['is_pinned'],  # Already 0-1
                features['is_checking'],  # Already 0-1
                features['is_protected'],  # Already 0-1
                features['is_mobile'],  # Already 0-1
                features['is_valuable']  # Already 0-1
            ])
            piece_types.append(0)  # 0 for piece nodes
            
        # Process critical square nodes
        square_features = []
        square_types = []
        for square, features in self.critical_square_nodes.items():
            square_features.append([
                features['type'],  # Critical square type
                features['is_attacked'],  # Already 0-1
                features['is_controlled'],  # Already 0-1
                features['is_pawn_square'],  # Already 0-1
                features['is_center_square'],  # Already 0-1
                features['is_king_square']  # Already 0-1
            ])
            square_types.append(1)  # 1 for critical square nodes
            
        # Combine all node features and types
        node_features = torch.tensor(piece_features + square_features, dtype=torch.float32)
        node_types = torch.tensor(piece_types + square_types, dtype=torch.long)
        
        # Process edge features
        edge_types = {
            'direct_relation': (self.direct_edges, 8),
            'control': (self.control_edges, 6),
            'mobility': (self.mobility_edges, 7),
            'cooperation': (self.cooperative_edges, 5),
            'obstruction': (self.obstructive_edges, 6),
            'vulnerability': (self.vulnerability_edges, 7),
            'pawn_structure': (self.pawn_structure_edges, 8)
        }
        
        edge_indices = {}
        edge_features = {}
        
        for edge_type, (edges, feature_dim) in edge_types.items():
            if not edges:
                # Create empty tensors for this edge type
                edge_indices[edge_type] = torch.zeros((2, 0), dtype=torch.long)
                edge_features[edge_type] = torch.zeros((0, feature_dim), dtype=torch.float32)
                continue
                
            indices = []
            features = []
            
            for (from_square, to_square), edge in edges.items():
                indices.append([from_square, to_square])
                
                if edge_type == 'direct_relation':
                    features.append([
                        float(edge['type'] == 'same_type'),  # Is same type
                        float(edge['type'] == 'friendly'),  # Is friendly
                        float(edge['type'] == 'opposing'),  # Is opposing
                        edge['distance'] / 14.0,  # Normalize distance
                        float(edge['is_blocked']),  # Is blocked
                        0.0, 0.0, 0.0, 0.0  # Padding
                    ])
                elif edge_type == 'control':
                    features.append([
                        float(edge['type'] == 'control'),  # Is control
                        edge['strength'],  # Already 0-1
                        edge['distance'] / 14.0,  # Normalize distance
                        0.0, 0.0, 0.0  # Padding
                    ])
                elif edge_type == 'mobility':
                    features.append([
                        float(edge['type'] == 'mobility'),  # Is mobility
                        edge['value'],  # Already 0-1
                        edge['distance'] / 14.0,  # Normalize distance
                        0.0, 0.0, 0.0, 0.0  # Padding
                    ])
                elif edge_type == 'cooperation':
                    features.append([
                        float(edge['type'] == 'cooperative'),  # Is cooperative
                        edge['strength'],  # Already 0-1
                        edge['distance'] / 14.0,  # Normalize distance
                        0.0, 0.0  # Padding
                    ])
                elif edge_type == 'obstruction':
                    features.append([
                        float(edge['type'] == 'obstructive'),  # Is obstructive
                        edge['severity'],  # Already 0-1
                        edge['distance'] / 14.0,  # Normalize distance
                        0.0  # Padding
                    ])
                elif edge_type == 'vulnerability':
                    features.append([
                        float(edge['type'] == 'vulnerable'),  # Is vulnerable
                        edge['risk'],  # Already 0-1
                        edge['distance'] / 14.0,  # Normalize distance
                        0.0, 0.0  # Padding
                    ])
                elif edge_type == 'pawn_structure':
                    features.append([
                        float(edge['type'] == 'doubled'),  # Is doubled
                        float(edge['type'] == 'adjacent'),  # Is adjacent
                        float(edge['type'] == 'isolated'),  # Is isolated
                        edge['strength'],  # Already 0-1
                        edge['distance'] / 14.0,  # Normalize distance
                        0.0, 0.0, 0.0  # Padding
                    ])
            
            edge_indices[edge_type] = torch.tensor(indices, dtype=torch.long).t()
            edge_features[edge_type] = torch.tensor(features, dtype=torch.float32)
        
        return node_features, node_types, edge_indices, edge_features
    
    def to_device(self, device: torch.device) -> 'PositionalAdjacencyGraph':
        """Move the PAG data to the specified device.
        
        Args:
            device: Device to move the data to
            
        Returns:
            self for method chaining
        """
        self.data = self.data.to(device)
        return self 

    @staticmethod
    def move_to_index(move: chess.Move) -> int:
        """Convert a chess move to a unique index in the range [0, 5311].
        
        The index is calculated as:
        - For non-promotion moves: from_square * 64 + to_square
        - For promotion moves: 4096 + (from_square * 64 + to_square) * 4 + promotion_piece_type - 1
        
        Args:
            move: The chess move to convert
            
        Returns:
            Integer index in range [0, 5311]
        """
        if move.promotion:
            # Promotion moves: 4096 + (from_square * 64 + to_square) * 4 + promotion_piece_type - 1
            base = 4096 + (move.from_square * 64 + move.to_square) * 4
            return base + move.promotion - 1
        else:
            # Regular moves: from_square * 64 + to_square
            return move.from_square * 64 + move.to_square
    
    @staticmethod
    def index_to_move(index: int) -> chess.Move:
        """Convert an index back to a chess move.
        
        Args:
            index: Integer index in range [0, 5311]
            
        Returns:
            Chess move object
        """
        if index < 4096:
            # Regular move
            from_square = index // 64
            to_square = index % 64
            return chess.Move(from_square, to_square)
        else:
            # Promotion move
            index -= 4096
            from_square = (index // 4) // 64
            to_square = (index // 4) % 64
            promotion = (index % 4) + 1  # Convert back to piece type (1=Knight, 2=Bishop, 3=Rook, 4=Queen)
            return chess.Move(from_square, to_square, promotion)
    
    @staticmethod
    def board_to_array(board: chess.Board) -> np.ndarray:
        """Convert a chess board to a numpy array representation.
        
        Args:
            board: The chess board to convert
            
        Returns:
            numpy array of shape (8, 8) with piece values:
            - Positive values for white pieces (1=pawn, 2=knight, 3=bishop, 4=rook, 5=queen, 6=king)
            - Negative values for black pieces (-1=pawn, -2=knight, etc.)
            - 0 for empty squares
        """
        board_array = np.zeros((8, 8), dtype=np.int32)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                rank = chess.square_rank(square)
                file = chess.square_file(square)
                value = piece.piece_type
                if piece.color == chess.BLACK:
                    value = -value
                board_array[rank, file] = value
        return board_array 