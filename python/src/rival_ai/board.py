import chess
from dataclasses import dataclass
from typing import Optional, Tuple, List, Iterator
import torch
from torch_geometric.data import HeteroData
import numpy as np

# Use chess package's built-in enums
Color = chess.Color
PieceType = chess.PieceType

# Define our own GameResult enum since chess package uses Outcome
class GameResult:
    """Game result constants matching chess.Outcome.termination values."""
    ONGOING = None
    WHITE_WINS = chess.Termination.CHECKMATE
    BLACK_WINS = chess.Termination.CHECKMATE
    DRAW = chess.Termination.STALEMATE  # This will be one of several draw types

@dataclass
class Move:
    """Chess move representation that wraps chess.Move."""
    move: chess.Move
    
    def __init__(self, from_square: int, to_square: int, promotion: Optional[PieceType] = None):
        """Initialize a move.
        
        Args:
            from_square: Source square index
            to_square: Target square index
            promotion: Optional piece type for promotion
        """
        self.move = chess.Move(from_square, to_square, promotion)
    
    def to_index(self, board: 'Board') -> int:
        """Convert move to neural network output index.
        
        Args:
            board: Current board state
            
        Returns:
            Index in neural network output tensor
        """
        # Calculate base index from from_square and to_square
        base_idx = self.move.from_square * 64 + self.move.to_square
        
        # Add promotion piece offset if this is a promotion move
        if self.move.promotion is not None:
            # Add offset for each promotion piece type
            promotion_offset = {
                PieceType.KNIGHT: 4096,  # 64 * 64
                PieceType.BISHOP: 4096 + 64,
                PieceType.ROOK: 4096 + 128,
                PieceType.QUEEN: 4096 + 192
            }
            base_idx += promotion_offset[self.move.promotion]
        
        return base_idx

class Board:
    """Chess board implementation that wraps chess.Board."""
    
    def __init__(self):
        """Initialize a new chess board."""
        self._board = chess.Board()
    
    @classmethod
    def from_fen(cls, fen: str) -> 'Board':
        """Create a board from a FEN string.
        
        Args:
            fen: FEN string representing a chess position
            
        Returns:
            Board object initialized with the given position
        """
        board = cls()
        board._board = chess.Board(fen)
        return board
    
    def copy(self) -> 'Board':
        """Create a copy of the board.
        
        Returns:
            New board instance with same state
        """
        new_board = Board()
        new_board._board = self._board.copy()
        return new_board
    
    def make_move(self, move: Move) -> None:
        """Make a move on the board.
        
        Args:
            move: Move to make
        """
        self._board.push(move.move)
    
    def get_legal_moves(self) -> List[Move]:
        """Get list of legal moves in current position.
        
        Returns:
            List of legal moves
        """
        return [Move(move.from_square, move.to_square, move.promotion) 
                for move in self._board.legal_moves]
    
    def get_piece(self, file: int, rank: int) -> Optional[Tuple[PieceType, Color]]:
        """Get piece at given square.
        
        Args:
            file: File (0-7)
            rank: Rank (0-7)
            
        Returns:
            Tuple of (piece type, color) if square is occupied, None otherwise
        """
        square = chess.square(file, rank)
        piece = self._board.piece_at(square)
        if piece is None:
            return None
        return (piece.piece_type, piece.color)
    
    @property
    def turn(self) -> Color:
        """Get current player's color.
        
        Returns:
            Color of player to move
        """
        return self._board.turn
    
    @property
    def result(self) -> GameResult:
        """Get current game result.
        
        Returns:
            Current game result
        """
        outcome = self._board.outcome()
        if outcome is None:
            return GameResult.ONGOING
        
        if outcome.winner == chess.WHITE:
            return GameResult.WHITE_WINS
        elif outcome.winner == chess.BLACK:
            return GameResult.BLACK_WINS
        else:
            return GameResult.DRAW
    
    def is_game_over(self) -> bool:
        """Check if the game is over.
        
        Returns:
            True if the game has ended (checkmate, stalemate, or draw), False otherwise
        """
        return self.result != GameResult.ONGOING
    
    @property
    def ep_square(self) -> Optional[int]:
        """Get en passant target square.
        
        Returns:
            En passant target square index or None
        """
        return self._board.ep_square
    
    @property
    def fullmove_number(self) -> int:
        """Get current fullmove number.
        
        Returns:
            Current fullmove number
        """
        return self._board.fullmove_number
    
    @property
    def halfmove_clock(self) -> int:
        """Get halfmove clock.
        
        Returns:
            Current halfmove clock
        """
        return self._board.halfmove_clock
    
    def has_kingside_castling_rights(self, color: Color) -> bool:
        """Check if given color has kingside castling rights.
        
        Args:
            color: Color to check
            
        Returns:
            True if color has kingside castling rights
        """
        return bool(self._board.castling_rights & chess.BB_KINGSIDE & chess.BB_RANKS[color])
    
    def has_queenside_castling_rights(self, color: Color) -> bool:
        """Check if given color has queenside castling rights.
        
        Args:
            color: Color to check
            
        Returns:
            True if color has queenside castling rights
        """
        return bool(self._board.castling_rights & chess.BB_QUEENSIDE & chess.BB_RANKS[color])
    
    def is_checkmate(self) -> bool:
        """Check if current position is checkmate.
        
        Returns:
            True if current position is checkmate
        """
        return self._board.is_checkmate()
    
    def is_stalemate(self) -> bool:
        """Check if current position is stalemate.
        
        Returns:
            True if current position is stalemate
        """
        return self._board.is_stalemate()
    
    def is_insufficient_material(self) -> bool:
        """Check if current position has insufficient material.
        
        Returns:
            True if current position has insufficient material
        """
        return self._board.is_insufficient_material()
    
    def is_fifty_moves(self) -> bool:
        """Check if current position is a draw by fifty-move rule.
        
        Returns:
            True if current position is a draw by fifty-move rule
        """
        return self._board.is_fifty_moves()
    
    def is_repetition(self) -> bool:
        """Check if current position is a draw by repetition.
        
        Returns:
            True if current position is a draw by repetition
        """
        return self._board.is_repetition()
    
    def to_hetero_data(self) -> HeteroData:
        """Convert board state to PyTorch Geometric HeteroData format.
        
        Returns:
            HeteroData object containing the graph representation of the position
        """
        data = HeteroData()
        
        # Get all pieces and their features
        piece_nodes = []
        piece_node_map = {}  # Map square index to node index
        node_idx = 0
        
        for square in chess.SQUARES:
            piece = self._board.piece_at(square)
            if piece is not None:
                # Calculate piece features
                piece_type = piece.piece_type
                color = piece.color
                rank = chess.square_rank(square)
                file = chess.square_file(square)
                
                # Material value
                material_value = {
                    chess.PAWN: 1,
                    chess.KNIGHT: 3,
                    chess.BISHOP: 3,
                    chess.ROOK: 5,
                    chess.QUEEN: 9,
                    chess.KING: 0
                }[piece_type]
                
                # Mobility (number of legal moves)
                mobility = len(list(self._board.legal_moves))
                
                # Control status (1 if square is attacked by white, -1 if by black, 0 if neutral)
                white_control = 1.0 if self._board.is_attacked_by(chess.WHITE, square) else 0.0
                black_control = 1.0 if self._board.is_attacked_by(chess.BLACK, square) else 0.0
                
                # Attack/defense status
                is_attacked = self._board.is_attacked_by(not color, square)
                is_defended = self._board.is_attacked_by(color, square)
                
                # King shield (is piece adjacent to king of same color)
                is_king_shield = False
                if piece_type != chess.KING:
                    for king_square in chess.SQUARES:
                        king = self._board.piece_at(king_square)
                        if king and king.piece_type == chess.KING and king.color == color:
                            king_rank = chess.square_rank(king_square)
                            king_file = chess.square_file(king_square)
                            if abs(rank - king_rank) <= 1 and abs(file - king_file) <= 1:
                                is_king_shield = True
                                break
                
                # Create node features
                node_features = [
                    piece_type / 6.0,  # Normalized piece type
                    1.0 if color else 0.0,  # Color (1 for white, 0 for black)
                    rank / 7.0,  # Normalized rank
                    file / 7.0,  # Normalized file
                    material_value / 9.0,  # Normalized material value
                    mobility / 100.0,  # Normalized mobility
                    white_control,
                    black_control,
                    float(is_attacked),
                    float(is_defended),
                    float(is_king_shield)
                ]
                
                piece_nodes.append(node_features)
                piece_node_map[square] = node_idx
                node_idx += 1
        
        # Add piece nodes to data
        if piece_nodes:
            data['piece'].x = torch.tensor(piece_nodes, dtype=torch.float32)
            data['piece'].batch = torch.zeros(len(piece_nodes), dtype=torch.long)  # Single position in batch
        
        # Add edges between pieces
        piece_edges = []
        piece_edge_attrs = []
        
        for src_square, src_idx in piece_node_map.items():
            for dst_square, dst_idx in piece_node_map.items():
                if src_square == dst_square:
                    continue
                
                # Calculate edge features
                src_rank, src_file = chess.square_rank(src_square), chess.square_file(src_square)
                dst_rank, dst_file = chess.square_rank(dst_square), chess.square_file(dst_square)
                
                # Distance and ray type
                dr, df = abs(dst_rank - src_rank), abs(dst_file - src_file)
                distance = dr + df
                
                # Determine ray type (0: orthogonal, 1: diagonal, 2: knight, 3: none)
                if dr == 0 or df == 0:
                    ray_type = 0  # orthogonal
                elif dr == df:
                    ray_type = 1  # diagonal
                elif (dr == 2 and df == 1) or (dr == 1 and df == 2):
                    ray_type = 2  # knight
                else:
                    ray_type = 3  # none
                
                # Check if path is blocked
                is_blocked = False
                if ray_type in [0, 1]:  # orthogonal or diagonal
                    r_step = 0 if dr == 0 else (dst_rank - src_rank) // dr
                    f_step = 0 if df == 0 else (dst_file - src_file) // df
                    r, f = src_rank + r_step, src_file + f_step
                    while (r, f) != (dst_rank, dst_file):
                        if self._board.piece_at(chess.square(f, r)) is not None:
                            is_blocked = True
                            break
                        r += r_step
                        f += f_step
                
                # Get piece information
                src_piece = self._board.piece_at(src_square)
                dst_piece = self._board.piece_at(dst_square)
                
                # Calculate additional features
                same_color = src_piece.color == dst_piece.color
                can_attack = self._board.is_attacked_by(src_piece.color, dst_square)
                can_be_attacked = self._board.is_attacked_by(dst_piece.color, src_square)
                can_be_attacked_by_opponent = self._board.is_attacked_by(not src_piece.color, src_square)
                
                # Create edge features
                edge_features = [
                    ray_type / 3.0,  # Normalized ray type
                    distance / 14.0,  # Normalized distance
                    float(is_blocked),
                    float(same_color),
                    float(can_attack),
                    float(can_be_attacked),
                    float(can_be_attacked_by_opponent),
                    float(can_be_attacked_by_opponent)  # Duplicated for dimension match
                ]
                
                piece_edges.append([src_idx, dst_idx])
                piece_edge_attrs.append(edge_features)
        
        # Add piece edges to data
        if piece_edges:
            data['piece', 'direct_relation', 'piece'].edge_index = torch.tensor(piece_edges, dtype=torch.long).t()
            data['piece', 'direct_relation', 'piece'].edge_attr = torch.tensor(piece_edge_attrs, dtype=torch.float32)
        
        # Add FEN string for move generation
        data.fen = [self._board.fen()]
        
        return data 