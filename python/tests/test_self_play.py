"""
Tests for the self-play module.
"""

import os
import pytest
import torch
import numpy as np
from rival_ai.models import GNNModel
from rival_ai.training.self_play import (
    SelfPlay,
    SelfPlayConfig,
    GameRecord,
)
from rival_ai.chess import GameResult
import chess

@pytest.fixture
def model():
    """Create a GNN model for testing."""
    return GNNModel(
        node_features=32,
        edge_features=16,
        hidden_channels=64,
        num_layers=2,
        num_heads=2,
    )

@pytest.fixture
def config(tmp_path):
    """Create a test configuration."""
    return SelfPlayConfig(
        num_games=2,
        num_simulations=10,  # Small number for testing
        temperature=1.0,
        dirichlet_alpha=0.3,
        dirichlet_weight=0.25,
        max_moves=50,  # Small number for testing
        save_dir=str(tmp_path / "test_self_play"),
        device="cpu",
        batch_size=2,
        num_workers=0,
        use_tqdm=False,
    )

def test_self_play_initialization(model, config):
    """Test self-play initialization."""
    self_play = SelfPlay(model, config)
    
    assert self_play.model is model
    assert self_play.config is config
    assert os.path.exists(config.save_dir)
    assert len(self_play.metrics) == 0

def test_play_game(model, config):
    """Test playing a single game."""
    self_play = SelfPlay(model, config)
    game = self_play.play_game()
    
    # Check game record structure
    assert isinstance(game, GameRecord)
    assert len(game.states) > 0
    assert len(game.moves) > 0
    assert len(game.policies) > 0
    assert len(game.values) > 0
    assert isinstance(game.result, GameResult)
    assert game.num_moves == len(game.moves)
    
    # Check state-move alignment
    assert len(game.states) == len(game.moves)
    assert len(game.states) == len(game.policies)
    assert len(game.states) == len(game.values)
    
    # Check board states
    for state in game.states:
        assert isinstance(state, chess.Board)
    
    # Check policy distributions
    for policy in game.policies:
        assert isinstance(policy, torch.Tensor)
        assert policy.ndim == 1
        assert torch.allclose(policy.sum(), torch.tensor(1.0), atol=1e-6)
        assert torch.all(policy >= 0)
    
    # Check values
    for value in game.values:
        assert isinstance(value, float)
        assert -1 <= value <= 1

def test_generate_games(model, config):
    """Test generating multiple games."""
    self_play = SelfPlay(model, config)
    games = self_play.generate_games()
    
    # Check number of games
    assert len(games) == config.num_games
    
    # Check metrics
    assert len(self_play.metrics['num_moves']) == config.num_games
    assert len(self_play.metrics['result']) == config.num_games
    
    # Check game lengths
    for game in games:
        assert 0 < game.num_moves <= config.max_moves

def test_save_and_load_games(model, config):
    """Test saving and loading games."""
    self_play = SelfPlay(model, config)
    games = self_play.generate_games()
    
    # Save games
    epoch = 1
    self_play.save_games(games, epoch)
    
    # Check file exists
    save_path = os.path.join(config.save_dir, f"games_epoch_{epoch}.pt")
    assert os.path.exists(save_path)
    
    # Load dataset
    dataset = torch.load(save_path)
    assert len(dataset) > 0

def test_create_dataloader(model, config):
    """Test creating dataloader from games."""
    self_play = SelfPlay(model, config)
    games = self_play.generate_games()
    
    # Create dataloader
    dataloader = self_play.create_dataloader(games)
    
    # Check dataloader
    assert len(dataloader) > 0
    
    # Check batch
    batch = next(iter(dataloader))
    assert batch.x.size(0) > 0
    assert batch.edge_index.size(1) > 0
    assert batch.policy_target.size(0) == config.batch_size
    assert batch.value_target.size(0) == config.batch_size

def test_metrics_logging(model, config):
    """Test metrics logging."""
    self_play = SelfPlay(model, config)
    
    # Generate some games
    games = self_play.generate_games()
    
    # Check metrics
    assert 'num_moves' in self_play.metrics
    assert 'result' in self_play.metrics
    assert len(self_play.metrics['num_moves']) == len(games)
    assert len(self_play.metrics['result']) == len(games)
    
    # Test reset
    self_play.reset_metrics()
    assert len(self_play.metrics) == 0

def test_game_result_consistency(model, config):
    """Test consistency between game moves and final result."""
    self_play = SelfPlay(model, config)
    game = self_play.play_game()
    
    # Replay the game
    board = chess.Board()
    for move in game.moves:
        board.push(move)
    
    # Check final state matches result
    assert board.is_checkmate() or board.is_stalemate() or board.is_insufficient_material()

def test_policy_value_consistency(model, config):
    """Test consistency between policy and value predictions."""
    self_play = SelfPlay(model, config)
    game = self_play.play_game()
    
    # Check value signs alternate
    for i in range(1, len(game.values)):
        assert game.values[i] * game.values[i-1] <= 0
    
    # Check final value matches game result
    final_value = 1.0 if game.result == GameResult.WHITE_WINS else -1.0 if game.result == GameResult.BLACK_WINS else 0.0
    assert abs(game.values[-1] - final_value) < 1e-6 