"""
Shared types for training components.
"""

from typing import List, NamedTuple, Optional, Dict, Any, Union
import torch
from rival_ai.chess import GameResult
import chess
from dataclasses import dataclass, field
import numpy as np

@dataclass
class GameRecord:
    """Record of a single game."""
    states: List[chess.Board] = field(default_factory=list)  # Board states
    moves: List[chess.Move] = field(default_factory=list)    # Moves made
    policies: List[torch.Tensor] = field(default_factory=list)  # Policy distributions
    values: List[float] = field(default_factory=list)  # Value predictions
    result: Optional[GameResult] = None   # Game result
    num_moves: int = 0       # Number of moves in the game
    
    def add_state(self, state: chess.Board) -> None:
        """Add a board state to the record."""
        self.states.append(state.copy())
    
    def add_move(self, move: chess.Move, policy: Union[torch.Tensor, np.ndarray, Dict], value: float, pag_features: Optional[torch.Tensor] = None):
        """Add a move to the game record.
        
        Args:
            move: The chess move
            policy: Policy distribution (tensor, array, or dict)
            value: Position evaluation
            pag_features: Optional PAG features for the position
        """
        # Convert policy to consistent format
        if isinstance(policy, dict):
            # Convert move dict to tensor
            policy_tensor = torch.zeros(5312, dtype=torch.float32)
            for chess_move, prob in policy.items():
                idx = self._move_to_index(chess_move)
                if 0 <= idx < 5312:
                    policy_tensor[idx] = float(prob)
            policy = policy_tensor
        elif isinstance(policy, np.ndarray):
            policy = torch.from_numpy(policy).float()
        elif not isinstance(policy, torch.Tensor):
            raise ValueError(f"Unsupported policy type: {type(policy)}")
        
        move_data = MoveData(
            move=move,
            policy=policy,
            value=float(value),
            pag_features=pag_features
        )
        self.moves.append(move_data)
    
    def _move_to_index(self, move: chess.Move) -> int:
        """Convert a chess move to an index for policy representation.
        
        Args:
            move: Chess move to convert
            
        Returns:
            Index in the policy vector (0-5311)
        """
        if move.promotion:
            # 🔥 FIXED: Use compact encoding that fits in available slots (4096-5311)
            # The old encoding produced indices beyond 5,311 which broke training!
            
            piece_type = {
                chess.KNIGHT: 0,  # 0-based for compact encoding
                chess.BISHOP: 1, 
                chess.ROOK: 2,
                chess.QUEEN: 3
            }.get(move.promotion, 3)  # Default to queen
            
            # Extract file and rank info
            from_file = move.from_square % 8
            from_rank = move.from_square // 8
            to_file = move.to_square % 8
            to_rank = move.to_square // 8
            
            # Determine promotion direction
            if to_file == from_file:
                direction = 0  # Straight promotion
            elif to_file == from_file - 1:
                direction = 1  # Capture left
            elif to_file == from_file + 1:
                direction = 2  # Capture right
            else:
                return 4096  # Invalid promotion fallback
            
            # Determine side (White or Black promotion)
            if from_rank == 6 and to_rank == 7:  # White promotion
                side_offset = 0
            elif from_rank == 1 and to_rank == 0:  # Black promotion  
                side_offset = 96  # 8 files * 3 directions * 4 pieces = 96
            else:
                return 4096  # Invalid promotion ranks fallback
            
            # Compact index calculation that fits in 1,216 available slots
            # Format: 4096 + side_offset + (file * 12) + (direction * 4) + piece_type
            index = 4096 + side_offset + (from_file * 12) + (direction * 4) + piece_type
            
            # Ensure it's within bounds
            if index < 5312:
                return index
            else:
                return 4096  # Fallback if somehow out of bounds
        else:
            # Regular moves: from_square * 64 + to_square
            return move.from_square * 64 + move.to_square
    
    def add_policy(self, policy: torch.Tensor) -> None:
        """Add a policy distribution to the record."""
        self.policies.append(policy)
    
    def add_value(self, value: float) -> None:
        """Add a value prediction to the record."""
        self.values.append(value)
    
    def set_result(self, result: GameResult) -> None:
        """Set the game result with move-count-aware rewards.
        
        Rewards quick checkmates and strongly penalizes repetitive draws,
        while being neutral about move count for other legitimate outcomes.
        """
        self.result = result
        move_count = len(self.moves)
        
        # Base values for different outcomes
        if result == GameResult.WHITE_WINS:
            # Reward quick checkmates, but don't penalize longer ones
            if move_count <= 20:  # Quick checkmate
                final_value = 2.0  # Bonus for quick checkmate
            elif move_count <= 30:  # Early checkmate
                final_value = 1.5
            else:
                final_value = 1.0  # Normal win value
                
        elif result == GameResult.BLACK_WINS:
            # Same logic for black wins
            if move_count <= 20:
                final_value = -2.0
            elif move_count <= 30:
                final_value = -1.5
            else:
                final_value = -1.0
                
        elif result == GameResult.REPETITION_DRAW:
            # Strong penalties for repetitive draws, especially early ones
            if move_count < 10:  # Very early repetition
                final_value = -3.0
            elif move_count < 20:  # Early repetition
                final_value = -2.5
            elif move_count < 30:  # Middlegame repetition
                final_value = -2.0
            else:  # Late repetition
                final_value = -1.5
                
        else:  # Regular draws (stalemate, insufficient material, etc.)
            # Neutral value for legitimate draws
            if self.states[-1].is_insufficient_material():
                final_value = 0.0  # Completely neutral for insufficient material
            elif self.states[-1].is_stalemate():
                final_value = -0.1  # Small penalty for stalemate
            else:
                final_value = 0.0  # Neutral for other legitimate draws
        
        # Apply the value to all positions, alternating for each player
        self.values = [final_value * (-1) ** (i % 2) for i in range(len(self.values))]
    
    def to_named_tuple(self) -> 'GameRecordTuple':
        """Convert to immutable named tuple format."""
        if self.result is None:
            raise ValueError("Game result must be set before converting to named tuple")
        return GameRecordTuple(
            states=self.states,
            moves=self.moves,
            policies=self.policies,
            values=self.values,
            result=self.result,
            num_moves=self.num_moves
        )

@dataclass
class MoveData:
    """Data for a single move in a game."""
    move: chess.Move
    policy: torch.Tensor  # Policy distribution over all moves
    value: float  # Position evaluation
    pag_features: Optional[torch.Tensor] = None  # PAG features for the position
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = {
            'move': self.move.uci(),
            'policy': self.policy.tolist() if isinstance(self.policy, torch.Tensor) else self.policy,
            'value': float(self.value)
        }
        
        # Include PAG features if available
        if self.pag_features is not None:
            data['pag_features'] = self.pag_features.tolist() if isinstance(self.pag_features, torch.Tensor) else self.pag_features
            
        return data

class GameRecordTuple(NamedTuple):
    """Immutable version of GameRecord."""
    states: List[chess.Board]  # Board states
    moves: List[chess.Move]    # Moves made
    policies: List[torch.Tensor]  # Policy distributions
    values: List[float]  # Value predictions
    result: GameResult   # Game result
    num_moves: int       # Number of moves in the game

def encode_move_to_policy_index(move: chess.Move, board: chess.Board = None) -> int:
    """Convert a chess move to its policy index using the same encoding as training.
    
    This function provides the same move encoding logic used throughout the training
    system, ensuring consistency between UCI tournament data and self-play data.
    
    Args:
        move: Chess move to encode
        board: Chess board (optional, for compatibility)
        
    Returns:
        Index in the policy vector (0-5311)
    """
    if move.promotion:
        # Use the same compact encoding as GameRecord._move_to_index
        piece_type = {
            chess.KNIGHT: 0,
            chess.BISHOP: 1, 
            chess.ROOK: 2,
            chess.QUEEN: 3
        }.get(move.promotion, 3)
        
        from_file = move.from_square % 8
        from_rank = move.from_square // 8
        to_file = move.to_square % 8
        to_rank = move.to_square // 8
        
        if to_file == from_file:
            direction = 0  # Straight promotion
        elif to_file == from_file - 1:
            direction = 1  # Capture left
        elif to_file == from_file + 1:
            direction = 2  # Capture right
        else:
            return 4096  # Invalid promotion fallback
        
        if from_rank == 6 and to_rank == 7:  # White promotion
            side_offset = 0
        elif from_rank == 1 and to_rank == 0:  # Black promotion  
            side_offset = 96
        else:
            return 4096  # Invalid promotion ranks fallback
        
        index = 4096 + side_offset + (from_file * 12) + (direction * 4) + piece_type
        return index if index < 5312 else 4096
    else:
        # Regular moves: from_square * 64 + to_square
        return move.from_square * 64 + move.to_square 