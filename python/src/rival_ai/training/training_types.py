"""
Shared types for training components.
"""

from typing import List, NamedTuple, Optional
import torch
from rival_ai.chess import GameResult
import chess
from dataclasses import dataclass, field

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
    
    def add_move(self, move: chess.Move, policy: Optional[torch.Tensor] = None, value: Optional[float] = None) -> None:
        """Add a move to the record.
        
        Args:
            move: The move made
            policy: Optional policy tensor for the move
            value: Optional value prediction for the move
        """
        self.moves.append(move)
        self.num_moves += 1
        if policy is not None:
            self.policies.append(policy)
        if value is not None:
            self.values.append(value)
    
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

class GameRecordTuple(NamedTuple):
    """Immutable version of GameRecord."""
    states: List[chess.Board]  # Board states
    moves: List[chess.Move]    # Moves made
    policies: List[torch.Tensor]  # Policy distributions
    values: List[float]  # Value predictions
    result: GameResult   # Game result
    num_moves: int       # Number of moves in the game 