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
        """Set the game result."""
        self.result = result
        # Adjust values based on game result
        if result == GameResult.WHITE_WINS:
            final_value = 1.0
        elif result == GameResult.BLACK_WINS:
            final_value = -1.0
        elif result == GameResult.REPETITION_DRAW:
            final_value = -2.0  # Much stronger penalty for repetition draws
        else:  # Regular draw
            final_value = 0.0
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