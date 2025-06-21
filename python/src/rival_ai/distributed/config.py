"""
Configuration for distributed training components.
"""

from dataclasses import dataclass
from typing import Optional, Dict, List
import torch

@dataclass
class GameCollectorConfig:
    """Configuration for game collection and processing."""
    max_queue_size: int = 10000  # Maximum number of games to queue
    batch_size: int = 64  # Number of games to process in parallel
    num_workers: int = 4  # Number of worker processes
    cache_size: int = 100000  # Number of positions to cache
    min_elo: int = 1500  # Minimum Elo rating for human games
    
@dataclass
class ModelManagerConfig:
    """Configuration for model management."""
    model_dir: str = "models"
    eval_games: int = 100  # Games to play for evaluation
    min_win_rate: float = 0.55  # Required win rate to promote model
    tournament_workers: int = 4
    elo_k_factor: float = 32.0
    
@dataclass
class DistributedTrainerConfig:
    """Configuration for distributed training."""
    batch_size: int = 1024
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    num_workers: int = 8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_queue_size: int = 100000
    checkpoint_interval: int = 1000
    eval_interval: int = 5000
    
    # Replay buffer settings
    buffer_size: int = 1000000
    min_buffer_size: int = 100000
    sample_ratio: float = 0.25  # Ratio of old:new data
    
    # Training specific
    value_loss_weight: float = 1.0
    policy_loss_weight: float = 1.0
    grad_clip: float = 1.0
    
    # Distributed settings
    world_size: int = 1
    rank: int = 0
    dist_backend: str = "nccl"
    dist_url: str = "tcp://localhost:23456"
    
@dataclass
class DistributedConfig:
    """Main configuration for distributed training."""
    game_collector: GameCollectorConfig = GameCollectorConfig()
    model_manager: ModelManagerConfig = ModelManagerConfig()
    trainer: DistributedTrainerConfig = DistributedTrainerConfig()
    
    # Web server settings
    server_host: str = "localhost"
    server_port: int = 8080
    max_concurrent_games: int = 10000
    
    # Logging and monitoring
    log_level: str = "INFO"
    tensorboard_dir: str = "runs"
    prometheus_port: int = 9090 