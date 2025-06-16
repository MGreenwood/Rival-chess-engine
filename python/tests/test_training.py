"""
Tests for the training components.
"""

import pytest
import torch
import numpy as np
from rival_ai.models import ChessGNN
from rival_ai.training import (
    PolicyValueLoss,
    KLDivergenceLoss,
    MetricTracker,
    TrainingMetrics,
    Trainer,
    TrainerConfig,
    compute_policy_accuracy,
    compute_value_accuracy,
    compute_elo_rating,
)
from rival_ai.training.self_play import SelfPlay, SelfPlayConfig
from rival_ai.chess import GameResult
import chess

@pytest.fixture
def sample_batch():
    """Create a sample batch for testing."""
    batch_size = 4
    num_nodes = 10
    num_edges = 20
    
    # Create random features
    x = torch.randn(batch_size * num_nodes, 32)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, 16)
    batch = torch.repeat_interleave(torch.arange(batch_size), num_nodes)
    
    # Create random targets
    policy_target = torch.randint(0, num_edges, (batch_size,))
    value_target = torch.randn(batch_size, 1)
    
    # Create batch object
    class Batch:
        def __init__(self):
            self.x = x
            self.edge_index = edge_index
            self.edge_attr = edge_attr
            self.batch = batch
            self.policy_target = policy_target
            self.value_target = value_target
            self.num_graphs = batch_size
    
    return Batch()

@pytest.fixture
def model():
    """Create a test model."""
    return ChessGNN(
        hidden_channels=64,  # Smaller for testing
        num_layers=2,
        heads=2,
        dropout=0.1,
    )

def test_policy_value_loss(sample_batch, model):
    """Test policy-value loss computation."""
    # Create loss function
    criterion = PolicyValueLoss()
    
    # Forward pass
    policy_logits, value_pred = model(
        sample_batch.x,
        sample_batch.edge_index,
        sample_batch.edge_attr,
        sample_batch.batch
    )
    
    # Compute loss
    loss, loss_dict = criterion(
        policy_logits,
        value_pred,
        sample_batch.policy_target,
        sample_batch.value_target,
        model
    )
    
    # Check loss components
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # Scalar
    assert loss.item() > 0
    
    # Check loss dictionary
    assert 'total_loss' in loss_dict
    assert 'policy_loss' in loss_dict
    assert 'value_loss' in loss_dict
    assert 'entropy' in loss_dict
    assert 'l2_reg' in loss_dict

def test_kl_divergence_loss():
    """Test KL divergence loss computation."""
    # Create loss function
    criterion = KLDivergenceLoss(temperature=2.0)
    
    # Create random logits
    batch_size = 4
    num_moves = 10
    student_logits = torch.randn(batch_size, num_moves)
    teacher_logits = torch.randn(batch_size, num_moves)
    
    # Compute loss
    loss, loss_dict = criterion(student_logits, teacher_logits)
    
    # Check loss
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # Scalar
    assert loss.item() > 0
    
    # Check loss dictionary
    assert 'kl_div' in loss_dict
    assert 'temperature' in loss_dict
    assert loss_dict['temperature'] == 2.0

def test_metric_tracker():
    """Test metric tracker functionality."""
    tracker = MetricTracker()
    
    # Update metrics
    metrics = {
        'loss': 0.5,
        'accuracy': 0.8,
    }
    tracker.update(metrics, batch_size=2)
    
    # Compute metrics
    computed = tracker.compute()
    assert computed['loss'] == 0.5
    assert computed['accuracy'] == 0.8
    
    # Update again
    tracker.update({'loss': 0.3, 'accuracy': 0.9}, batch_size=2)
    
    # Check running average
    computed = tracker.compute()
    assert computed['loss'] == 0.4  # (0.5 + 0.3) / 2
    assert computed['accuracy'] == 0.85  # (0.8 + 0.9) / 2
    
    # Check history
    assert tracker.get_history('loss') == [0.5, 0.3]
    assert tracker.get_history('accuracy') == [0.8, 0.9]
    
    # Test reset
    tracker.reset()
    assert len(tracker.get_history('loss')) == 0

def test_training_metrics(sample_batch, model):
    """Test training metrics computation."""
    metrics = TrainingMetrics()
    
    # Forward pass
    policy_logits, value_pred = model(
        sample_batch.x,
        sample_batch.edge_index,
        sample_batch.edge_attr,
        sample_batch.batch
    )
    
    # Create loss dictionary
    loss_dict = {
        'total_loss': 0.5,
        'policy_loss': 0.3,
        'value_loss': 0.2,
    }
    
    # Update metrics
    metrics.update(
        policy_logits,
        value_pred,
        sample_batch.policy_target,
        sample_batch.value_target,
        loss_dict,
        batch_size=len(sample_batch)
    )
    
    # Check computed metrics
    computed = metrics.compute()
    assert 'policy_accuracy_top1' in computed
    assert 'policy_accuracy_top3' in computed
    assert 'value_accuracy' in computed
    assert 'total_loss' in computed
    
    # Check best metrics
    best = metrics.get_best_metrics()
    assert 'policy_accuracy_top1' in best
    assert 'total_loss' in best

def test_trainer(sample_batch, model, tmp_path):
    """Test trainer functionality."""
    # Create optimizer and loss
    optimizer = torch.optim.Adam(model.parameters())
    criterion = PolicyValueLoss()
    
    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=torch.device('cpu'),
        save_dir=str(tmp_path),
        log_interval=1,
        save_interval=2,
    )
    
    # Create dummy dataloader
    class DummyLoader:
        def __init__(self, batch):
            self.batch = batch
            self.length = 4
        
        def __iter__(self):
            for _ in range(self.length):
                yield self.batch
        
        def __len__(self):
            return self.length
    
    train_loader = DummyLoader(sample_batch)
    val_loader = DummyLoader(sample_batch)
    
    # Train for one epoch
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=1,
    )
    
    # Check history
    assert 'train_total_loss' in history
    assert 'val_total_loss' in history
    assert len(history['train_total_loss']) == 1
    
    # Check checkpoint saving
    checkpoint_files = list(tmp_path.glob('*.pt'))
    assert len(checkpoint_files) > 0
    
    # Test checkpoint loading
    trainer.load_checkpoint(str(checkpoint_files[0]))
    assert trainer.epoch == 0
    assert trainer.step > 0

def test_accuracy_metrics():
    """Test accuracy metric computation."""
    # Create random predictions and targets
    batch_size = 4
    num_moves = 10
    
    # Policy accuracy
    policy_logits = torch.randn(batch_size, num_moves)
    policy_target = torch.randint(0, num_moves, (batch_size,))
    
    acc1 = compute_policy_accuracy(policy_logits, policy_target, top_k=1)
    acc3 = compute_policy_accuracy(policy_logits, policy_target, top_k=3)
    
    assert 0 <= acc1 <= 1
    assert 0 <= acc3 <= 1
    assert acc3 >= acc1  # Top-3 should be at least as good as top-1
    
    # Value accuracy
    value_pred = torch.randn(batch_size, 1)
    value_target = torch.randn(batch_size, 1)
    
    acc = compute_value_accuracy(value_pred, value_target)
    assert 0 <= acc <= 1

def test_elo_rating():
    """Test Elo rating computation."""
    # Test with no games
    rating, uncertainty = compute_elo_rating(0, 0, 0)
    assert rating == 1500.0
    assert uncertainty == float('inf')
    
    # Test with some games
    rating, uncertainty = compute_elo_rating(10, 5, 5)
    assert 1500.0 < rating < 1600.0  # Should improve with positive record
    assert 0 < uncertainty < 100  # Should have reasonable uncertainty
    
    # Test with negative record
    rating, uncertainty = compute_elo_rating(5, 10, 5)
    assert 1400.0 < rating < 1500.0  # Should decrease with negative record
    assert 0 < uncertainty < 100 