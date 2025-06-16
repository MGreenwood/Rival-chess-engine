"""
Tests for the GNN model.
"""

import pytest
import torch
from torch_geometric.data import HeteroData
from rival_ai.models import ChessGNN
from rival_ai.chess import GameResult
import chess

def test_model_initialization():
    """Test model initialization with default parameters."""
    model = ChessGNN()
    assert model is not None

def test_model_initialization_custom():
    """Test model initialization with custom parameters."""
    model = ChessGNN(
        hidden_channels=128,
        num_layers=3,
        heads=4,
        dropout=0.2,
    )
    assert model is not None

def test_gnn_model_initialization():
    """Test GNN model initialization with different configurations."""
    # Test default configuration
    model = ChessGNN()
    config = model.get_config()
    assert config['node_features'] == 9  # Match Board.to_model_input()
    assert config['edge_features'] == 6  # Match Board.to_model_input()
    assert config['hidden_channels'] == 256
    assert config['num_layers'] == 4
    assert config['num_heads'] == 4
    
    # Test custom configuration
    model = ChessGNN(
        node_features=9,  # Match Board.to_model_input()
        edge_features=6,  # Match Board.to_model_input()
        hidden_channels=512,
        num_layers=6,
        num_heads=8,
        dropout=0.2,
    )
    config = model.get_config()
    assert config['node_features'] == 9
    assert config['edge_features'] == 6
    assert config['hidden_channels'] == 512
    assert config['num_layers'] == 6
    assert config['num_heads'] == 8
    assert config['dropout'] == 0.2

def test_gnn_model_forward():
    """Test GNN model forward pass with different input sizes."""
    model = ChessGNN()
    
    # Test single graph
    num_nodes = 10
    num_edges = 20
    x = torch.randn(num_nodes, 9)  # Match node_features
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, 6)  # Match edge_features
    
    policy_logits, value = model(x, edge_index, edge_attr)
    assert policy_logits.shape == (num_nodes, 1)
    assert value.shape == (1, 1)
    assert torch.all(value >= -1) and torch.all(value <= 1)
    
    # Test batched graphs
    batch_size = 4
    num_nodes = 15
    num_edges = 30
    x = torch.randn(batch_size * num_nodes, 9)  # Match node_features
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, 6)  # Match edge_features
    batch = torch.repeat_interleave(torch.arange(batch_size), num_nodes)
    
    policy_logits, value = model(x, edge_index, edge_attr, batch)
    assert policy_logits.shape == (batch_size * num_nodes, 1)
    assert value.shape == (batch_size, 1)
    assert torch.all(value >= -1) and torch.all(value <= 1)

def test_gnn_model_gradients():
    """Test that gradients flow through the model."""
    model = ChessGNN()
    num_nodes = 10
    num_edges = 20
    x = torch.randn(num_nodes, 32, requires_grad=True)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, 16, requires_grad=True)
    
    policy_logits, value = model(x, edge_index, edge_attr)
    loss = policy_logits.mean() + value.mean()
    loss.backward()
    
    assert x.grad is not None
    assert edge_attr.grad is not None
    assert not torch.isnan(x.grad).any()
    assert not torch.isnan(edge_attr.grad).any()

def test_gnn_model_device():
    """Test model works on different devices."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        model = ChessGNN().to(device)
        
        num_nodes = 10
        num_edges = 20
        x = torch.randn(num_nodes, 32, device=device)
        edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)
        edge_attr = torch.randn(num_edges, 16, device=device)
        
        policy_logits, value = model(x, edge_index, edge_attr)
        assert policy_logits.device == device
        assert value.device == device 