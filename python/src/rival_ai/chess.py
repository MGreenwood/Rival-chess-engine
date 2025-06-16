from enum import Enum
from typing import Optional
import chess

class GameResult(Enum):
    """Possible game results."""
    ONGOING = 0
    WHITE_WINS = "white_wins"
    BLACK_WINS = "black_wins"
    DRAW = "draw"
    REPETITION_DRAW = "repetition_draw"  # New type for repetition draws

class PieceType(Enum):
    """Chess piece types."""
    PAWN = 1
    KNIGHT = 2
    BISHOP = 3
    ROOK = 4
    QUEEN = 5
    KING = 6

class Color(Enum):
    """Chess piece colors."""
    WHITE = 1
    BLACK = -1

class Move:
    """Chess move representation."""
    def __init__(self, from_square: int, to_square: int, promotion: Optional[PieceType] = None):
        """Initialize a move.
        
        Args:
            from_square: Source square index
            to_square: Target square index
            promotion: Optional piece type for promotion
        """
        self.from_square = from_square
        self.to_square = to_square
        self.promotion = promotion
    
    def to_chess_move(self) -> chess.Move:
        """Convert to chess.Move."""
        return chess.Move(self.from_square, self.to_square, self.promotion.value if self.promotion else None)
    
    @classmethod
    def from_chess_move(cls, move: chess.Move) -> 'Move':
        """Create from chess.Move."""
        return cls(move.from_square, move.to_square, 
                  PieceType(move.promotion) if move.promotion else None)
    
    def __eq__(self, other: object) -> bool:
        """Check if two moves are equal."""
        if not isinstance(other, Move):
            return NotImplemented
        return (self.from_square == other.from_square and
                self.to_square == other.to_square and
                self.promotion == other.promotion)
    
    def __hash__(self) -> int:
        """Get hash of move."""
        return hash((self.from_square, self.to_square, self.promotion))
    
    def __str__(self) -> str:
        """Convert move to UCI format."""
        return self.to_chess_move().uci() 