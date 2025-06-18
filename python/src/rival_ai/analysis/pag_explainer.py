from typing import List, Dict, Set, Tuple
import chess
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class PAGExplanation:
    """Explanation of a position based on PAG analysis."""
    move_explanation: str  # Why the move was made
    tactical_elements: List[str]  # Tactical elements (attacks, threats, etc.)
    strategic_elements: List[str]  # Strategic elements (development, control, etc.)
    piece_relationships: List[str]  # Key piece relationships
    attention_focus: List[str]  # What the model is focusing on

class PAGExplainer:
    """Generates natural language explanations from PAG data."""
    
    def __init__(self):
        # Define common chess patterns and their explanations
        self.patterns = {
            'attack': {
                'pattern': lambda edge: edge['type'] == 'attack',
                'explanation': lambda edge, board: f"{self._piece_name(board, edge['from'])} attacks {self._piece_name(board, edge['to'])}"
            },
            'defense': {
                'pattern': lambda edge: edge['type'] == 'defense',
                'explanation': lambda edge, board: f"{self._piece_name(board, edge['from'])} defends {self._piece_name(board, edge['to'])}"
            },
            'pin': {
                'pattern': lambda edge: edge['type'] == 'pin',
                'explanation': lambda edge, board: f"{self._piece_name(board, edge['from'])} pins {self._piece_name(board, edge['to'])}"
            },
            'fork': {
                'pattern': lambda edge: edge['type'] == 'fork',
                'explanation': lambda edge, board: f"{self._piece_name(board, edge['from'])} forks {self._piece_name(board, edge['to'])} and {self._piece_name(board, edge['target2'])}"
            },
            'discovered_attack': {
                'pattern': lambda edge: edge['type'] == 'discovered_attack',
                'explanation': lambda edge, board: f"Discovered attack from {self._piece_name(board, edge['from'])} to {self._piece_name(board, edge['to'])}"
            },
            'pawn_chain': {
                'pattern': lambda edge: edge['type'] == 'pawn_chain',
                'explanation': lambda edge, board: f"Pawn chain from {self._square_name(edge['from'])} to {self._square_name(edge['to'])}"
            },
            'piece_coordination': {
                'pattern': lambda edge: edge['type'] == 'coordination',
                'explanation': lambda edge, board: f"{self._piece_name(board, edge['from'])} and {self._piece_name(board, edge['to'])} coordinate to control {self._square_name(edge['target'])}"
            }
        }
        
        # Define strategic concepts
        self.strategic_concepts = {
            'center_control': {
                'pattern': lambda edges, board: any(self._is_center_square(edge['to']) for edge in edges),
                'explanation': lambda edges, board: "Controls central squares"
            },
            'development': {
                'pattern': lambda edges, board: any(self._is_development_move(edge, board) for edge in edges),
                'explanation': lambda edges, board: "Develops pieces toward the center"
            },
            'king_safety': {
                'pattern': lambda edges, board: any(self._is_king_safety_move(edge, board) for edge in edges),
                'explanation': lambda edges, board: "Improves king safety"
            },
            'space_advantage': {
                'pattern': lambda edges, board: any(self._is_space_gaining_move(edge, board) for edge in edges),
                'explanation': lambda edges, board: "Gains space advantage"
            }
        }
    
    def _piece_name(self, board: chess.Board, square: int) -> str:
        """Get descriptive name for a piece."""
        piece = board.piece_at(square)
        if not piece:
            return self._square_name(square)
        
        color = "White" if piece.color == chess.WHITE else "Black"
        piece_type = piece.symbol().upper()
        return f"{color} {piece_type} on {self._square_name(square)}"
    
    def _square_name(self, square: int) -> str:
        """Convert square index to algebraic notation."""
        return chess.square_name(square)
    
    def _is_center_square(self, square: int) -> bool:
        """Check if a square is in the center."""
        file, rank = chess.square_file(square), chess.square_rank(square)
        return 2 <= file <= 5 and 2 <= rank <= 5
    
    def _is_development_move(self, edge: Dict, board: chess.Board) -> bool:
        """Check if a move is a development move."""
        piece = board.piece_at(edge['from'])
        if not piece:
            return False
        
        # Knights and bishops moving from back rank
        if piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
            from_rank = chess.square_rank(edge['from'])
            return from_rank in [0, 7]  # Back rank
        
        return False
    
    def _is_king_safety_move(self, edge: Dict, board: chess.Board) -> bool:
        """Check if a move improves king safety."""
        piece = board.piece_at(edge['from'])
        if not piece or piece.piece_type != chess.KING:
            return False
        
        # Castling moves
        if abs(chess.square_file(edge['from']) - chess.square_file(edge['to'])) == 2:
            return True
        
        # King moving to safer position
        to_rank = chess.square_rank(edge['to'])
        return to_rank in [1, 6]  # Moving up/down one rank
    
    def _is_space_gaining_move(self, edge: Dict, board: chess.Board) -> bool:
        """Check if a move gains space."""
        piece = board.piece_at(edge['from'])
        if not piece:
            return False
        
        # Pawns advancing
        if piece.piece_type == chess.PAWN:
            from_rank = chess.square_rank(edge['from'])
            to_rank = chess.square_rank(edge['to'])
            return abs(to_rank - from_rank) > 0
        
        return False
    
    def explain_position(self, board: chess.Board, pag_edges: List[Dict], attention_weights: Dict[int, float], last_move: chess.Move = None) -> PAGExplanation:
        """Generate explanation for a position based on PAG data."""
        # Analyze tactical elements with more detail
        tactical = []
        for pattern_name, pattern_info in self.patterns.items():
            for edge in pag_edges:
                if pattern_info['pattern'](edge):
                    explanation = pattern_info['explanation'](edge, board)
                    # Add consequence of the tactical element
                    if edge['type'] == 'attack':
                        piece = board.piece_at(edge['to'])
                        if piece:
                            value = self._get_piece_value(piece)
                            explanation += f" (material threat: {value} points)"
                    elif edge['type'] == 'pin':
                        pinned_piece = board.piece_at(edge['to'])
                        if pinned_piece:
                            explanation += f" against {self._piece_name(board, edge['target'])}"
                    tactical.append(explanation)
        
        # Analyze strategic elements with positional context
        strategic = []
        for concept_name, concept_info in self.strategic_concepts.items():
            if concept_info['pattern'](pag_edges, board):
                explanation = concept_info['explanation'](pag_edges, board)
                # Add positional context
                if concept_name == 'center_control':
                    controlled_squares = [edge['to'] for edge in pag_edges if self._is_center_square(edge['to'])]
                    squares_str = ', '.join(chess.square_name(sq) for sq in controlled_squares)
                    explanation += f" ({squares_str})"
                elif concept_name == 'king_safety':
                    king_square = board.king(board.turn)
                    if king_square:
                        attackers = len(list(board.attackers(not board.turn, king_square)))
                        explanation += f" ({attackers} attacking pieces)"
                strategic.append(explanation)
        
        # Analyze piece relationships with more context
        relationships = []
        for edge in pag_edges:
            if edge['type'] in ['coordination', 'pawn_chain']:
                explanation = self.patterns[edge['type']]['explanation'](edge, board)
                # Add control information
                controlled_squares = self._get_controlled_squares(board, edge['from'], edge['to'])
                if controlled_squares:
                    squares_str = ', '.join(chess.square_name(sq) for sq in controlled_squares)
                    explanation += f" controlling {squares_str}"
                relationships.append(explanation)
        
        # Analyze attention focus with piece values
        attention = []
        sorted_attention = sorted(attention_weights.items(), key=lambda x: x[1], reverse=True)
        for square, weight in sorted_attention[:3]:
            piece = board.piece_at(square)
            if piece:
                value = self._get_piece_value(piece)
                attention.append(f"Focuses on {self._piece_name(board, square)} (weight: {weight:.2f}, value: {value})")
        
        # Generate detailed move explanation
        move_explanation = ""
        captured_piece = None
        if last_move:
            # Store the piece at the destination square BEFORE the move is played
            captured_piece = board.piece_at(last_move.to_square)
            move_explanation = self._explain_move(board, last_move, pag_edges, attention_weights)
            # Add material change if it was a capture and both pieces exist
            moving_piece = board.piece_at(last_move.from_square)
            if captured_piece and moving_piece and captured_piece.color != moving_piece.color:
                value = self._get_piece_value(captured_piece)
                move_explanation += f" gaining {value} points in material"
        
        return PAGExplanation(
            move_explanation=move_explanation,
            tactical_elements=tactical,
            strategic_elements=strategic,
            piece_relationships=relationships,
            attention_focus=attention
        )
    
    def _get_piece_value(self, piece: chess.Piece) -> float:
        """Get the standard value of a piece."""
        values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0  # King's value is not relevant for material counting
        }
        return values[piece.piece_type]
    
    def _get_controlled_squares(self, board: chess.Board, from_square: int, to_square: int) -> List[int]:
        """Get squares controlled by the pieces at from_square and to_square."""
        controlled = set()
        for square in [from_square, to_square]:
            piece = board.piece_at(square)
            if piece:
                attackers = board.attackers(piece.color, square)
                controlled.update(attackers)
        return list(controlled)
    
    def _explain_move(self, board: chess.Board, move: chess.Move, pag_edges: List[Dict], attention_weights: Dict[int, float]) -> str:
        """Generate explanation for a specific move."""
        # Get the piece that moved (before the move)
        piece = board.piece_at(move.from_square)
        if not piece:
            return "Invalid move"

        # Check if move was a capture (destination occupied by opponent before move)
        captured_piece = board.piece_at(move.to_square)
        if captured_piece and captured_piece.color != piece.color:
            return f"{self._piece_name(board, move.from_square)} captures {self._piece_name(board, move.to_square)}"

        # Check if move was castling
        if piece.piece_type == chess.KING and abs(move.from_square - move.to_square) == 2:
            side = "kingside" if move.to_square > move.from_square else "queenside"
            return f"Castles {side}"

        # Check if move was a pawn push
        if piece.piece_type == chess.PAWN:
            if abs(move.to_square - move.from_square) == 16:  # Double push
                return f"Advances pawn two squares to {self._square_name(move.to_square)}"
            return f"Advances pawn to {self._square_name(move.to_square)}"

        # Check if move was a check
        board.push(move)
        is_check = board.is_check()
        board.pop()

        # Regular piece move
        move_desc = f"Moves {self._piece_name(board, move.from_square)} from {self._square_name(move.from_square)} to {self._square_name(move.to_square)}"
        if is_check:
            move_desc += " with check"
        return move_desc 