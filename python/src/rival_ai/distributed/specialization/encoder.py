"""
Encoder module for converting chess positions and features to model inputs.
"""

import chess
import torch
import numpy as np
from typing import Dict, List, Tuple
from .position_analyzer import PositionFeatures

class ChessEncoder:
    """Encodes chess positions and features for model input."""
    
    # Board representation planes
    PIECE_PLANES = {
        (chess.PAWN, chess.WHITE): 0,
        (chess.KNIGHT, chess.WHITE): 1,
        (chess.BISHOP, chess.WHITE): 2,
        (chess.ROOK, chess.WHITE): 3,
        (chess.QUEEN, chess.WHITE): 4,
        (chess.KING, chess.WHITE): 5,
        (chess.PAWN, chess.BLACK): 6,
        (chess.KNIGHT, chess.BLACK): 7,
        (chess.BISHOP, chess.BLACK): 8,
        (chess.ROOK, chess.BLACK): 9,
        (chess.QUEEN, chess.BLACK): 10,
        (chess.KING, chess.BLACK): 11
    }
    
    # Additional planes for board state
    AUXILIARY_PLANES = {
        'en_passant': 12,
        'castling_rights': 13,
        'side_to_move': 14,
        'move_count': 15,
        'repetition_count': 16
    }
    
    # Move encoding
    MOVE_DIRECTIONS = [
        # Pawn moves
        (-1, 0), (-2, 0), (-1, -1), (-1, 1),  # White pawn
        (1, 0), (2, 0), (1, -1), (1, 1),      # Black pawn
        # Knight moves
        (-2, -1), (-2, 1), (-1, -2), (-1, 2),
        (1, -2), (1, 2), (2, -1), (2, 1),
        # Bishop moves
        (-1, -1), (-1, 1), (1, -1), (1, 1),
        # Rook moves
        (-1, 0), (1, 0), (0, -1), (0, 1),
        # Queen moves (combination of bishop and rook)
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1)
    ]
    
    def __init__(self):
        """Initialize the chess encoder."""
        # Precompute move indices
        self.move_to_index = {}
        self.index_to_move = []
        self._precompute_move_indices()
        
    def _precompute_move_indices(self):
        """Precompute mappings between moves and indices."""
        index = 0
        
        # Add all possible moves
        for from_square in chess.SQUARES:
            for to_square in chess.SQUARES:
                # Regular moves
                move = chess.Move(from_square, to_square)
                self.move_to_index[move] = index
                self.index_to_move.append(move)
                index += 1
                
                # Promotions
                for promotion in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                    move = chess.Move(from_square, to_square, promotion)
                    self.move_to_index[move] = index
                    self.index_to_move.append(move)
                    index += 1
                    
    def encode_board(self, board: chess.Board) -> torch.Tensor:
        """Encode a chess position as a tensor.
        
        Args:
            board: Chess position to encode
            
        Returns:
            Tensor of shape [N, 8, 8] where N is number of planes
        """
        # Initialize planes
        planes = torch.zeros(len(self.PIECE_PLANES) + len(self.AUXILIARY_PLANES), 8, 8)
        
        # Encode pieces
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                plane_idx = self.PIECE_PLANES[(piece.piece_type, piece.color)]
                rank = chess.square_rank(square)
                file = chess.square_file(square)
                planes[plane_idx, rank, file] = 1
                
        # Encode en passant
        if board.ep_square:
            rank = chess.square_rank(board.ep_square)
            file = chess.square_file(board.ep_square)
            planes[self.AUXILIARY_PLANES['en_passant'], rank, file] = 1
            
        # Encode castling rights
        castling_plane = planes[self.AUXILIARY_PLANES['castling_rights']]
        if board.has_kingside_castling_rights(chess.WHITE):
            castling_plane[0, 4:7] = 1
        if board.has_queenside_castling_rights(chess.WHITE):
            castling_plane[0, 0:4] = 1
        if board.has_kingside_castling_rights(chess.BLACK):
            castling_plane[7, 4:7] = 1
        if board.has_queenside_castling_rights(chess.BLACK):
            castling_plane[7, 0:4] = 1
            
        # Encode side to move
        planes[self.AUXILIARY_PLANES['side_to_move']] = float(board.turn)
        
        # Encode move count (normalized)
        planes[self.AUXILIARY_PLANES['move_count']] = board.fullmove_number / 100.0
        
        # Encode repetition count (if available)
        if hasattr(board, 'is_repetition'):
            repetition_count = sum(1 for _ in range(3) if board.is_repetition(i))
            planes[self.AUXILIARY_PLANES['repetition_count']] = repetition_count / 3.0
            
        return planes
        
    def encode_features(self, features: PositionFeatures) -> torch.Tensor:
        """Encode position features as a tensor.
        
        Args:
            features: Position features to encode
            
        Returns:
            1D tensor of encoded features
        """
        # Initialize feature vector
        feature_vector = []
        
        # Add scalar features
        feature_vector.extend([
            features.material_balance,
            features.pawn_structure_score,
            features.king_safety,
            features.center_control,
            features.piece_coordination,
            features.attacking_potential,
            features.tactical_opportunities / 10.0  # Normalize
        ])
        
        # Add piece mobility (normalized)
        for piece_type in sorted(features.piece_mobility.keys()):
            feature_vector.append(features.piece_mobility[piece_type])
            
        # Add file information (one-hot)
        file_vector = [0] * 8
        for file in features.open_files:
            file_vector[file] = 1
        feature_vector.extend(file_vector)
        
        file_vector = [0] * 8
        for file in features.half_open_files:
            file_vector[file] = 1
        feature_vector.extend(file_vector)
        
        # Add weak square information
        weak_squares = [0] * 64
        for square in features.weak_squares:
            weak_squares[square] = 1
        feature_vector.extend(weak_squares)
        
        return torch.tensor(feature_vector, dtype=torch.float32)
        
    def encode_move(self, move: chess.Move) -> int:
        """Convert a move to its index in the policy vector.
        
        Args:
            move: Chess move to encode
            
        Returns:
            Index in policy vector
        """
        return self.move_to_index.get(move, 0)
        
    def decode_move(self, index: int) -> chess.Move:
        """Convert a policy index back to a chess move.
        
        Args:
            index: Index in policy vector
            
        Returns:
            Corresponding chess move
        """
        if 0 <= index < len(self.index_to_move):
            return self.index_to_move[index]
        return chess.Move.null()
        
    def get_move_probabilities(
        self,
        policy: torch.Tensor,
        board: chess.Board
    ) -> Dict[chess.Move, float]:
        """Convert policy logits to move probabilities.
        
        Args:
            policy: Policy logits from model
            board: Current board position
            
        Returns:
            Dictionary mapping legal moves to probabilities
        """
        # Get legal moves
        legal_moves = list(board.legal_moves)
        
        # Get probabilities for legal moves
        move_probs = {}
        for move in legal_moves:
            move_idx = self.encode_move(move)
            move_probs[move] = policy[move_idx].item()
            
        # Normalize probabilities
        total_prob = sum(move_probs.values())
        if total_prob > 0:
            move_probs = {
                move: prob / total_prob
                for move, prob in move_probs.items()
            }
        else:
            # Fallback to uniform distribution
            uniform_prob = 1.0 / len(legal_moves)
            move_probs = {move: uniform_prob for move in legal_moves}
            
        return move_probs 