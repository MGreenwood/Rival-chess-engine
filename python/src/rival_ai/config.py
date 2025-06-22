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
    max_moves: int = 200  # Reduced from 400 to force more decisive play
    dirichlet_alpha: float = 1.2  # Increased for more noise
    dirichlet_weight: float = 0.6  # Increased for more exploration
    temperature: float = 2.0  # Increased for much more variety
    c_puct: float = 3.0  # Increased to encourage more exploration
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
    opening_temperature: float = 2.5  # Much higher temperature for opening
    midgame_temperature: float = 2.2  # Higher temperature for middlegame
    endgame_temperature: float = 1.8  # Higher temperature for endgame
    
    # Aggressive play settings
    capture_bonus: float = 1.0  # Strong bonus for captures
    check_bonus: float = 0.5  # Bonus for checks
    attack_bonus: float = 0.3  # Bonus for attacking moves
    development_bonus: float = 0.4  # Bonus for piece development
    
    # Game phase detection
    opening_moves: int = 15  # First 15 moves are opening
    midgame_moves: int = 50  # Moves 16-50 are middlegame
    
    # Dynamic move limits
    max_opening_moves: int = 30  # Shorter opening phase
    max_middlegame_moves: int = 100  # Shorter middlegame
    max_endgame_moves: int = 150  # Shorter endgame
    
    # Forced decisive play settings
    force_capture_after_moves: int = 20  # Force captures after 20 moves
    force_attack_after_moves: int = 30  # Force attacks after 30 moves
    force_win_attempt_after_moves: int = 50  # Force win attempts after 50 moves
    
    # Repetition prevention
    repetition_penalty: float = 8.0  # Much stronger penalties
    draw_penalty_scale: float = 6.0  # Much stronger draw penalties
    early_draw_penalty: float = 10.0  # Much stronger early draw penalties

@dataclass
class TrainingConfig:
    """Configuration for model training."""
    num_epochs: int = 100
    batch_size: int = 128  # Increased from 64 for better gradient estimates
    learning_rate: float = 0.0003  # Reduced from 0.001 for more stable training
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
    
    # Loss function configuration - Optimized for tactical learning
    use_improved_loss: bool = True  # Use improved loss function
    use_pag_tactical_loss: bool = True  # Use ultra-dense PAG tactical loss for preventing blunders
    policy_weight: float = 1.0
    value_weight: float = 1.0  # Restored to 1.0 from 0.5 - position evaluation is critical
    entropy_weight: float = 0.01
    l2_weight: float = 1e-4
    
    # PAG Tactical Loss Configuration - CRITICAL for preventing basic blunders
    pag_tactical_config: dict = None  # Will be set in __post_init__
    
    def __post_init__(self):
        """Set default PAG tactical configuration."""
        if self.pag_tactical_config is None:
            self.pag_tactical_config = {
                'vulnerability_weight': 8.0,        # CRITICAL: Heavy penalty for hanging pieces  
                'motif_awareness_weight': 4.0,      # Strong tactical pattern recognition
                'threat_generation_weight': 3.0,    # Reward threat creation
                'material_protection_weight': 6.0,  # Protect valuable pieces strongly
                'pin_exploitation_weight': 3.5,     # Exploit pins/skewers
                'fork_creation_weight': 3.0,        # Create forks
                'discovery_weight': 2.5,            # Discovered attacks
                'defensive_coordination_weight': 3.0, # Coordinate defense
                'tactical_positional_balance': 0.8, # 80% tactical focus (vs 20% positional)
                'endgame_tactical_boost': 2.0,      # Double tactical precision in endgame
                'progressive_difficulty': True,     # Gradually increase complexity
            }

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