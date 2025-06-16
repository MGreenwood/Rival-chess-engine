"""
Configuration classes for the Rival AI chess engine.
"""

from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class MCTSConfig:
    """Configuration for Monte Carlo Tree Search."""
    num_simulations: int = 800  # Increased from 400 to get better move quality
    temperature: float = 1.0  # Temperature for move selection
    dirichlet_alpha: float = 0.3  # Alpha parameter for Dirichlet noise
    dirichlet_weight: float = 0.25  # Weight of Dirichlet noise
    c_puct: float = 0.5  # Reduced from 1.0 to balance exploration/exploitation
    epsilon: float = 1e-8  # Small constant to prevent division by zero
    batch_size: int = 512  # Batch size for neural network evaluation
    max_table_size: int = 1000000  # Maximum size of the transposition table
    num_parallel_streams: int = 8  # Number of parallel streams
    log_timing: bool = True  # Whether to log timing metrics
    edge_dims: Optional[Dict[str, int]] = None  # Edge dimensions for PAG
    prefetch_factor: int = 4  # Number of batches to prefetch
    num_workers: int = 8  # Number of worker threads for data loading
    pin_memory: bool = True  # Use pinned memory for faster GPU transfer
    use_amp: bool = True  # Use automatic mixed precision
    max_time: float = 5.0  # Maximum time per move in seconds

    def __post_init__(self):
        """Set default edge dimensions if not provided."""
        if self.edge_dims is None:
            self.edge_dims = {
                'direct_relation': 8,
                'control': 6,
                'mobility': 7,
                'cooperation': 5,
                'obstruction': 6,
                'vulnerability': 7,
                'pawn_structure': 8
            }

@dataclass
class SelfPlayConfig:
    """Configuration for self-play generation."""
    num_games: int = 1000  # Increased from 100 to get more training data
    num_simulations: int = 800  # Match MCTSConfig
    max_moves: int = 200  # Increased from 100 to allow proper endgame play
    dirichlet_alpha: float = 0.3
    dirichlet_weight: float = 0.25
    temperature: float = 1.0  # Will be modified during game
    c_puct: float = 0.5  # Match MCTSConfig
    save_dir: str = 'self_play_data'
    device: str = 'cuda'
    batch_size: int = 64
    num_workers: int = 4
    use_tqdm: bool = True
    log_timing: bool = True
    num_parallel_games: int = 10
    prefetch_factor: int = 2
    temperature_decay: float = 0.95  # Decay temperature over time
    min_temperature: float = 0.1  # Minimum temperature to use
    opening_temperature: float = 1.0  # Temperature for first 10 moves
    midgame_temperature: float = 0.5  # Temperature for moves 11-30
    endgame_temperature: float = 0.1  # Temperature for moves 31+

@dataclass
class TrainingConfig:
    """Configuration for model training."""
    num_epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    save_interval: int = 5
    experiment_dir: str = 'experiments'
    checkpoint_dir: str = 'checkpoints'  # Base directory for all model checkpoints
    device: str = 'cuda'
    use_tensorboard: bool = True
    checkpoint_path: Optional[str] = None  # Path to checkpoint for resuming training
    resume_epoch: Optional[int] = None     # Epoch to resume from
    num_workers: int = 4
    pin_memory: bool = True
    use_amp: bool = True
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    max_lr: float = 1e-3
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4

@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.1
    use_amp: bool = True
    use_residual: bool = True
    use_layer_norm: bool = True
    activation: str = 'relu'
    bias: bool = False
    concat: bool = True
    edge_dim: Optional[int] = None
    add_self_loops: bool = False 