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

    def __post_init__(self):
        """Set default edge dimensions if not provided."""
        if self.edge_dims is None:
            self.edge_dims = {
                'direct_relation': 8,
                'control': 6,
                'mobility': 7,
                'cooperation': 5,
                'obstruction': 6,
                'vulnerability': 7,
                'pawn_structure': 8,
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
        self.config = config
        
        # Initialize node dictionaries
        self.piece_nodes: Dict[int, Dict] = {}  # square -> node features
        self.critical_square_nodes: Dict[int, Dict] = {}  # square -> node features
        
        # Initialize edge feature dictionaries
        self.direct_edges: Dict[Tuple[int, int], Dict] = {}  # (from_square, to_square) -> edge features
        self.control_edges: Dict[Tuple[int, int], Dict] = {}  # (piece_square, square) -> edge features
        self.mobility_edges: Dict[Tuple[int, int], Dict] = {}  # (piece_square, square) -> edge features
        self.cooperative_edges: Dict[Tuple[int, int], Dict] = {}  # (piece_square, piece_square) -> edge features
        self.obstructive_edges: Dict[Tuple[int, int], Dict] = {}  # (piece_square, piece_square) -> edge features
        self.vulnerability_edges: Dict[Tuple[int, int], Dict] = {}  # (piece_square, piece_square) -> edge features
        self.pawn_structure_edges: Dict[Tuple[int, int], Dict] = {}  # (pawn_square, pawn_square) -> edge features
        
        # Build the graph
        self._build_graph()
    
    def _build_graph(self):
        """Build the graph representation of the position."""
        # Add piece nodes
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece is not None:
                self.piece_nodes[square] = {
                    'type': piece.piece_type,
                    'color': piece.color,
                    'mobility': self._is_piece_mobile(square),
                    'is_pinned': self._is_piece_pinned(square),
                    'is_checking': self._is_piece_checking(square),
                    'is_protected': self._is_piece_protected(square),
                    'is_mobile': self._is_piece_mobile(square),
                    'is_valuable': piece.piece_type in [chess.QUEEN, chess.ROOK]
                }
        
        # Add critical square nodes
        for square in chess.SQUARES:
            self.critical_square_nodes[square] = {
                'is_attacked': self._is_square_attacked(square, chess.WHITE) or self._is_square_attacked(square, chess.BLACK),
                'is_controlled': self._is_square_controlled(square),
                'is_pawn_square': self._is_pawn_square(square),
                'is_center_square': self._is_center_square(square),
                'is_king_square': self._is_king_square(square)
            }
        
        # Add edges
        for square1 in chess.SQUARES:
            piece1 = self.board.piece_at(square1)
            if piece1 is not None:
                for square2 in chess.SQUARES:
                    piece2 = self.board.piece_at(square2)
                    if piece2 is not None:
                        # Direct relations
                        if self._can_move_to(square1, square2):
                            self.direct_edges[(square1, square2)] = {
                                'type': 'attack' if piece1.color != piece2.color else 'support',
                                'distance': chess.square_distance(square1, square2),
                                'is_capture': piece1.color != piece2.color
                            }
                        
                        # Cooperation
                        if piece1.color == piece2.color and self._are_pieces_cooperative(square1, square2):
                            self.cooperative_edges[(square1, square2)] = {
                                'type': 'protection' if self._is_piece_protected(square2) else 'support',
                                'distance': chess.square_distance(square1, square2)
                            }
                        
                        # Obstruction
                        if piece1.color != piece2.color and self._are_pieces_obstructive(square1, square2):
                            self.obstructive_edges[(square1, square2)] = {
                                'type': 'block' if self._is_path_blocked(square1, square2) else 'threat',
                                'distance': chess.square_distance(square1, square2)
                            }
                        
                        # Vulnerability
                        if piece1.color != piece2.color and self._are_pieces_vulnerable(square1, square2):
                            self.vulnerability_edges[(square1, square2)] = {
                                'type': 'attack' if self._can_attack(square1, square2) else 'threat',
                                'distance': chess.square_distance(square1, square2)
                            }
                    
                    # Control edges
                    if self._controls_square(square1, square2):
                        self.control_edges[(square1, square2)] = {
                            'type': 'control',
                            'distance': chess.square_distance(square1, square2)
                        }
                    
                    # Mobility edges
                    if self._can_move_to(square1, square2):
                        self.mobility_edges[(square1, square2)] = {
                            'type': 'mobility',
                            'distance': chess.square_distance(square1, square2)
                        }
        
        # Add pawn structure edges
        for square1 in chess.SQUARES:
            piece1 = self.board.piece_at(square1)
            if piece1 is not None and piece1.piece_type == chess.PAWN:
                for square2 in chess.SQUARES:
                    piece2 = self.board.piece_at(square2)
                    if piece2 is not None and piece2.piece_type == chess.PAWN and piece1.color == piece2.color:
                        if self._are_pawns_connected(square1, square2):
                            self.pawn_structure_edges[(square1, square2)] = {
                                'type': 'connected',
                                'distance': chess.square_distance(square1, square2)
                            }
                        elif self._are_pawns_related(square1, square2):
                            self.pawn_structure_edges[(square1, square2)] = {
                                'type': 'chain',
                                'distance': chess.square_distance(square1, square2)
                            }
    
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