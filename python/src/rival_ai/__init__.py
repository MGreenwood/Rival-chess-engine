"""
RivalAI - A chess AI using deep learning and MCTS.
"""

import chess
from rival_ai.chess import Color, GameResult, Move, PieceType
from rival_ai.models import ChessGNN
from rival_ai.mcts import MCTS, MCTSConfig, MCTSNode
from rival_ai.training import (
    Trainer,
    PolicyValueLoss,
    TrainingVisualizer,
    TrainingMetrics,
    compute_policy_accuracy,
    compute_value_accuracy,
    compute_elo_rating
)

__version__ = "0.1.0"

__all__ = [
    # Core components
    'Color',
    'GameResult',
    'Move',
    'PieceType',
    'ChessGNN',
    'MCTS',
    'MCTSConfig',
    'MCTSNode',
    
    # Training components
    'Trainer',
    'PolicyValueLoss',
    'TrainingVisualizer',
    'TrainingMetrics',
    'compute_policy_accuracy',
    'compute_value_accuracy',
    'compute_elo_rating',
] 