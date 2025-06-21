"""
Position analyzer for detecting tactical and positional characteristics.
"""

import chess
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

@dataclass
class PositionFeatures:
    """Features extracted from a chess position."""
    material_balance: float
    piece_mobility: Dict[chess.PieceType, float]
    pawn_structure_score: float
    king_safety: float
    center_control: float
    piece_coordination: float
    attacking_potential: float
    tactical_opportunities: int
    open_files: Set[int]
    half_open_files: Set[int]
    weak_squares: Set[chess.Square]
    
    @property
    def tactical_score(self) -> float:
        """Calculate overall tactical score."""
        return (
            0.3 * self.attacking_potential +
            0.3 * float(self.tactical_opportunities) +
            0.2 * self.piece_mobility[chess.QUEEN] +
            0.1 * self.piece_mobility[chess.KNIGHT] +
            0.1 * self.piece_mobility[chess.BISHOP]
        )
        
    @property
    def positional_score(self) -> float:
        """Calculate overall positional score."""
        return (
            0.3 * self.pawn_structure_score +
            0.2 * self.king_safety +
            0.2 * self.center_control +
            0.2 * self.piece_coordination +
            0.1 * len(self.weak_squares) / 64
        )

class PositionAnalyzer:
    """Analyzes chess positions for tactical and positional features."""
    
    # Piece values for material evaluation
    PIECE_VALUES = {
        chess.PAWN: 1.0,
        chess.KNIGHT: 3.0,
        chess.BISHOP: 3.25,
        chess.ROOK: 5.0,
        chess.QUEEN: 9.0
    }
    
    # Center squares for control evaluation
    CENTER_SQUARES = {
        chess.E4, chess.E5, chess.D4, chess.D5,  # Inner center
        chess.C3, chess.C4, chess.C5, chess.C6,  # Extended center
        chess.D3, chess.D6, chess.E3, chess.E6,
        chess.F3, chess.F4, chess.F5, chess.F6
    }
    
    def analyze_position(self, board: chess.Board) -> PositionFeatures:
        """Analyze a position for tactical and positional features.
        
        Args:
            board: Chess position to analyze
            
        Returns:
            Extracted position features
        """
        # Material and basic features
        material = self._evaluate_material(board)
        mobility = self._calculate_mobility(board)
        pawns = self._analyze_pawn_structure(board)
        king_safety = self._evaluate_king_safety(board)
        center = self._evaluate_center_control(board)
        
        # Advanced features
        coordination = self._evaluate_piece_coordination(board)
        attacking = self._evaluate_attacking_potential(board)
        tactics = self._count_tactical_opportunities(board)
        
        # File analysis
        open_files, half_open = self._analyze_files(board)
        
        # Weak square analysis
        weak_squares = self._find_weak_squares(board)
        
        return PositionFeatures(
            material_balance=material,
            piece_mobility=mobility,
            pawn_structure_score=pawns,
            king_safety=king_safety,
            center_control=center,
            piece_coordination=coordination,
            attacking_potential=attacking,
            tactical_opportunities=tactics,
            open_files=open_files,
            half_open_files=half_open,
            weak_squares=weak_squares
        )
        
    def _evaluate_material(self, board: chess.Board) -> float:
        """Calculate material balance relative to white."""
        score = 0.0
        
        for piece_type in self.PIECE_VALUES:
            white_pieces = len(board.pieces(piece_type, chess.WHITE))
            black_pieces = len(board.pieces(piece_type, chess.BLACK))
            score += self.PIECE_VALUES[piece_type] * (white_pieces - black_pieces)
            
        return score
        
    def _calculate_mobility(self, board: chess.Board) -> Dict[chess.PieceType, float]:
        """Calculate mobility scores for each piece type."""
        mobility = {piece_type: 0.0 for piece_type in self.PIECE_VALUES}
        
        # Store original turn
        original_turn = board.turn
        
        for color in [chess.WHITE, chess.BLACK]:
            board.turn = color
            multiplier = 1 if color == chess.WHITE else -1
            
            for piece_type in mobility:
                squares = board.pieces(piece_type, color)
                total_moves = 0
                
                for square in squares:
                    moves = len([
                        move for move in board.legal_moves
                        if move.from_square == square
                    ])
                    total_moves += moves
                    
                if squares:
                    avg_mobility = total_moves / len(squares)
                    mobility[piece_type] += multiplier * avg_mobility
                    
        # Restore original turn
        board.turn = original_turn
        
        # Normalize scores
        for piece_type in mobility:
            mobility[piece_type] = (mobility[piece_type] + 8) / 16  # Scale to [0,1]
            
        return mobility
        
    def _analyze_pawn_structure(self, board: chess.Board) -> float:
        """Analyze pawn structure strength."""
        score = 0.0
        
        for color in [chess.WHITE, chess.BLACK]:
            multiplier = 1 if color == chess.WHITE else -1
            pawns = board.pieces(chess.PAWN, color)
            
            # Pawn islands
            files_with_pawns = set(chess.square_file(sq) for sq in pawns)
            islands = len([
                i for i in range(len(files_with_pawns))
                if i-1 not in files_with_pawns
            ])
            score -= multiplier * 0.1 * islands
            
            # Doubled pawns
            for file in range(8):
                file_pawns = len([
                    sq for sq in pawns
                    if chess.square_file(sq) == file
                ])
                if file_pawns > 1:
                    score -= multiplier * 0.2 * (file_pawns - 1)
                    
            # Connected pawns
            for sq in pawns:
                if any(
                    chess.square_file(sq) + d in files_with_pawns
                    for d in [-1, 1]
                ):
                    score += multiplier * 0.1
                    
        return (score + 2) / 4  # Scale to [0,1]
        
    def _evaluate_king_safety(self, board: chess.Board) -> float:
        """Evaluate king safety."""
        score = 0.0
        
        for color in [chess.WHITE, chess.BLACK]:
            multiplier = 1 if color == chess.WHITE else -1
            king_square = board.king(color)
            
            if not king_square:
                continue
                
            # Pawn shield
            shield_squares = self._get_pawn_shield_squares(king_square, color)
            shield_pawns = len([
                sq for sq in shield_squares
                if board.piece_at(sq) == chess.Piece(chess.PAWN, color)
            ])
            score += multiplier * 0.2 * shield_pawns
            
            # Open lines to king
            attacking_pieces = [
                (chess.QUEEN, 0.3),
                (chess.ROOK, 0.2),
                (chess.BISHOP, 0.1)
            ]
            
            for piece_type, weight in attacking_pieces:
                attackers = board.pieces(piece_type, not color)
                for sq in attackers:
                    if board.attacks(sq) & (1 << king_square):
                        score -= multiplier * weight
                        
        return (score + 2) / 4  # Scale to [0,1]
        
    def _evaluate_center_control(self, board: chess.Board) -> float:
        """Evaluate center control."""
        score = 0.0
        
        for color in [chess.WHITE, chess.BLACK]:
            multiplier = 1 if color == chess.WHITE else -1
            
            # Direct center control
            for square in self.CENTER_SQUARES:
                if board.is_attacked_by(color, square):
                    score += multiplier * (
                        0.2 if square in {chess.E4, chess.E5, chess.D4, chess.D5}
                        else 0.1
                    )
                    
            # Pieces in center
            for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP]:
                pieces = board.pieces(piece_type, color)
                center_pieces = len([
                    sq for sq in pieces
                    if sq in self.CENTER_SQUARES
                ])
                score += multiplier * 0.1 * center_pieces
                
        return (score + 3) / 6  # Scale to [0,1]
        
    def _evaluate_piece_coordination(self, board: chess.Board) -> float:
        """Evaluate piece coordination."""
        score = 0.0
        
        for color in [chess.WHITE, chess.BLACK]:
            multiplier = 1 if color == chess.WHITE else -1
            
            # Piece defense
            for piece_type in self.PIECE_VALUES:
                pieces = board.pieces(piece_type, color)
                for square in pieces:
                    if board.is_attacked_by(color, square):
                        score += multiplier * 0.1
                        
            # Minor piece development
            for piece_type in [chess.KNIGHT, chess.BISHOP]:
                pieces = board.pieces(piece_type, color)
                developed = len([
                    sq for sq in pieces
                    if (color == chess.WHITE and chess.square_rank(sq) > 1) or
                    (color == chess.BLACK and chess.square_rank(sq) < 6)
                ])
                score += multiplier * 0.2 * developed
                
        return (score + 3) / 6  # Scale to [0,1]
        
    def _evaluate_attacking_potential(self, board: chess.Board) -> float:
        """Evaluate attacking potential."""
        score = 0.0
        
        for color in [chess.WHITE, chess.BLACK]:
            multiplier = 1 if color == chess.WHITE else -1
            
            # Attack on enemy king
            king_square = board.king(not color)
            if king_square:
                king_attackers = len([
                    move for move in board.legal_moves
                    if move.to_square == king_square
                ])
                score += multiplier * 0.3 * king_attackers
                
            # Pieces pointing at enemy territory
            enemy_half = range(4, 8) if color == chess.WHITE else range(4)
            for piece_type in [chess.QUEEN, chess.ROOK, chess.BISHOP]:
                pieces = board.pieces(piece_type, color)
                for square in pieces:
                    attacks = board.attacks(square)
                    enemy_attacks = len([
                        sq for sq in chess.scan_reversed(attacks)
                        if chess.square_rank(sq) in enemy_half
                    ])
                    score += multiplier * 0.1 * enemy_attacks / 8
                    
        return (score + 3) / 6  # Scale to [0,1]
        
    def _count_tactical_opportunities(self, board: chess.Board) -> int:
        """Count tactical opportunities in position."""
        opportunities = 0
        
        # Material captures
        for move in board.legal_moves:
            if board.is_capture(move):
                captured_piece = board.piece_at(move.to_square)
                if captured_piece:
                    attacker = board.piece_at(move.from_square)
                    if attacker and self.PIECE_VALUES[captured_piece.piece_type] > self.PIECE_VALUES[attacker.piece_type]:
                        opportunities += 1
                        
        # Discovered attacks
        for move in board.legal_moves:
            board.push(move)
            if board.is_check():
                opportunities += 1
            board.pop()
            
        # Fork opportunities
        for move in board.legal_moves:
            board.push(move)
            attacked = 0
            piece_value = 0
            for square in chess.SQUARES:
                if board.is_attacked_by(board.turn, square):
                    piece = board.piece_at(square)
                    if piece and piece.color != board.turn:
                        attacked += 1
                        piece_value += self.PIECE_VALUES[piece.piece_type]
            if attacked >= 2 and piece_value >= 6:
                opportunities += 1
            board.pop()
            
        return opportunities
        
    def _analyze_files(self, board: chess.Board) -> Tuple[Set[int], Set[int]]:
        """Analyze open and half-open files."""
        open_files = set()
        half_open_files = set()
        
        for file in range(8):
            white_pawns = 0
            black_pawns = 0
            
            for rank in range(8):
                square = chess.square(file, rank)
                piece = board.piece_at(square)
                if piece and piece.piece_type == chess.PAWN:
                    if piece.color == chess.WHITE:
                        white_pawns += 1
                    else:
                        black_pawns += 1
                        
            if white_pawns == 0 and black_pawns == 0:
                open_files.add(file)
            elif white_pawns == 0 or black_pawns == 0:
                half_open_files.add(file)
                
        return open_files, half_open_files
        
    def _find_weak_squares(self, board: chess.Board) -> Set[chess.Square]:
        """Find weak squares in the position."""
        weak_squares = set()
        
        for square in chess.SQUARES:
            # Skip squares with pieces
            if board.piece_at(square):
                continue
                
            # Check if square can be defended by pawns
            for color in [chess.WHITE, chess.BLACK]:
                defenders = 0
                pawn_defenders = 0
                
                if board.is_attacked_by(color, square):
                    defenders += 1
                    
                # Check potential pawn defenders
                pawn_squares = self._get_pawn_defender_squares(square, color)
                for pawn_sq in pawn_squares:
                    if (
                        0 <= pawn_sq < 64 and
                        board.piece_at(pawn_sq) == chess.Piece(chess.PAWN, color)
                    ):
                        pawn_defenders += 1
                        
                if defenders == 0 and pawn_defenders == 0:
                    weak_squares.add(square)
                    
        return weak_squares
        
    def _get_pawn_shield_squares(
        self,
        king_square: chess.Square,
        color: chess.Color
    ) -> List[chess.Square]:
        """Get squares for potential pawn shield."""
        rank = chess.square_rank(king_square)
        file = chess.square_file(king_square)
        
        if color == chess.WHITE:
            shield_rank = rank + 1
        else:
            shield_rank = rank - 1
            
        shield_squares = []
        for f in range(max(0, file - 1), min(8, file + 2)):
            square = chess.square(f, shield_rank)
            if 0 <= square < 64:
                shield_squares.append(square)
                
        return shield_squares
        
    def _get_pawn_defender_squares(
        self,
        square: chess.Square,
        color: chess.Color
    ) -> List[chess.Square]:
        """Get squares where pawns could defend a square."""
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        
        if color == chess.WHITE:
            defender_rank = rank - 1
        else:
            defender_rank = rank + 1
            
        defender_squares = []
        for f in [file - 1, file + 1]:
            if 0 <= f < 8:
                square = chess.square(f, defender_rank)
                if 0 <= square < 64:
                    defender_squares.append(square)
                    
        return defender_squares 