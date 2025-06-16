"""
Shared type definitions for the RivalAI package.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import torch
import numpy as np
import chess

from .chess import Board, Move, GameResult

@dataclass
class GameRecord:
    """Record of a single game."""
    states: List[chess.Board]  # Board states
    moves: List[chess.Move]    # Moves made
    policies: List[torch.Tensor]  # Policy distributions
    values: List[float]  # Value predictions
    result: GameResult  # Game result
    metadata: Optional[dict] = None  # Additional game metadata

    def __len__(self) -> int:
        """Get number of moves in the game."""
        return len(self.moves)

    def get_training_data(self) -> Tuple[List[chess.Board], List[torch.Tensor], List[float]]:
        """Get training data from the game record.
        
        Returns:
            Tuple of (states, policies, values)
        """
        return self.states, self.policies, self.values

    def to_dict(self) -> dict:
        """Convert game record to dictionary."""
        return {
            'states': [state.fen for state in self.states],
            'moves': [move.uci for move in self.moves],
            'policies': [policy.numpy().tolist() for policy in self.policies],
            'values': self.values,
            'result': self.result.value,
            'metadata': self.metadata or {}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GameRecord':
        """Create a GameRecord from a dictionary.
        
        Args:
            data: Dictionary containing game data
            
        Returns:
            GameRecord object
        """
        return cls(
            states=[chess.Board(fen) for fen in data['states']],
            moves=[chess.Move.from_uci(move) for move in data['moves']],
            policies=[torch.tensor(policy) for policy in data['policies']],
            values=data['values'],
            result=GameResult(data['result']),
            metadata=data.get('metadata')
        ) 