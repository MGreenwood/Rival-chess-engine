"""
Tests for the MCTS module.
"""

import pytest
import numpy as np
import torch
from rival_ai.models import ChessGNN
from rival_ai.mcts import MCTS, MCTSConfig, MCTSNode
from rival_ai.chess import GameResult
import chess
import math

@pytest.fixture
def model():
    """Create a test model."""
    return ChessGNN(
        hidden_channels=64,  # Smaller for testing
        num_layers=2,
        heads=2,
        dropout=0.1,
    )

@pytest.fixture
def config():
    """Create a test configuration."""
    return MCTSConfig(
        num_simulations=10,  # Small number for testing
        temperature=1.0,
        dirichlet_alpha=0.3,
        dirichlet_weight=0.25,
        c_puct=1.0,
    )

def test_mcts_node_initialization(model, config):
    """Test MCTS node initialization."""
    board = chess.Board()
    node = MCTSNode(board, config=config)
    
    # Check basic properties
    assert node.board == board
    assert node.parent is None
    assert node.move is None
    assert node.config == config
    assert node.visit_count == 0
    assert node.value_sum == 0.0
    assert node.prior == 1.0  # Default prior
    assert not node.is_expanded
    assert len(node.children) == 0
    assert node.legal_moves == board.get_legal_moves()
    
    # Check with custom prior
    node = MCTSNode(board, config=config, prior=0.5)
    assert node.prior == 0.5

def test_mcts_node_value(model, config):
    """Test MCTS node value calculation."""
    board = chess.Board()
    node = MCTSNode(board, config=config)
    
    # Check initial value
    assert node.value == 0.0
    
    # Update value
    node.update(1.0)
    assert node.value == 1.0
    
    # Update again
    node.update(-1.0)
    assert node.value == 0.0  # (1.0 + -1.0) / 2

def test_mcts_node_expansion():
    """Test node expansion."""
    board = chess.Board()
    config = MCTSConfig()
    node = MCTSNode(board, config=config)
    
    # Create mock model
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))
            
        def forward(self, x, edge_index, edge_attr, batch):
            # Return non-uniform policy and neutral value
            policy = torch.zeros(32)
            policy[0:3] = torch.tensor([0.6, 0.3, 0.1])  # First 3 moves have non-uniform policy
            value = torch.tensor([0.0])
            return policy, value
            
        def parameters(self):
            return [self.dummy]
    
    model = MockModel()
    
    # Expand node
    node.expand(model, config)
    
    # Check expansion
    assert node.is_expanded
    assert len(node.children) > 0
    assert node._eval_cache is not None
    
    # Check policy and value
    policy, value = node._eval_cache
    assert policy.shape == (32,)
    assert np.all(policy >= 0)
    assert np.isclose(np.sum(policy), 1.0)
    assert isinstance(value, float)

def test_mcts_node_selection():
    """Test node selection."""
    board = chess.Board()
    config = MCTSConfig()
    node = MCTSNode(board, config=config)
    
    # Create mock model
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))
            
        def forward(self, x, edge_index, edge_attr, batch):
            policy = torch.zeros(32)
            policy[0:3] = torch.tensor([0.6, 0.3, 0.1])
            value = torch.tensor([0.0])
            return policy, value
            
        def parameters(self):
            return [self.dummy]
    
    model = MockModel()
    
    # Expand node
    node.expand(model, config)
    
    # Select child
    move, child = node.select_child(config.c_puct)
    assert move is not None
    assert child is not None
    assert move in node.children
    assert child == node.children[move]
    
    # Check UCT calculation
    uct = child.value + config.c_puct * child.prior * math.sqrt(node.visit_count) / (1 + child.visit_count)
    assert uct > 0

def test_mcts_search():
    """Test MCTS search."""
    board = chess.Board()
    config = MCTSConfig(num_simulations=10)
    
    # Create mock model
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))
            
        def forward(self, x, edge_index, edge_attr, batch):
            policy = torch.zeros(32)
            policy[0:3] = torch.tensor([0.6, 0.3, 0.1])
            value = torch.tensor([0.0])
            return policy, value
            
        def parameters(self):
            return [self.dummy]
    
    model = MockModel()
    mcts = MCTS(model, config)
    
    # Run search
    root = mcts.search(board)
    
    # Check root node
    assert root.is_expanded
    assert len(root.children) > 0
    assert root.visit_count == config.num_simulations
    
    # Check child visits
    total_child_visits = sum(child.visit_count for child in root.children.values())
    assert total_child_visits == config.num_simulations

def test_mcts_temperature():
    """Test temperature scaling in policy."""
    board = chess.Board()
    config = MCTSConfig()
    
    # Create mock model
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))
            
        def forward(self, x, edge_index, edge_attr, batch):
            policy = torch.zeros(32)
            policy[0:3] = torch.tensor([0.6, 0.3, 0.1])
            value = torch.tensor([0.0])
            return policy, value
            
        def parameters(self):
            return [self.dummy]
    
    model = MockModel()
    mcts = MCTS(model, config)
    
    # Run search
    root = mcts.search(board, num_simulations=10)
    
    # Set visit counts for first 3 legal moves
    legal_moves = list(root.legal_moves)[:3]
    for i, move in enumerate(legal_moves):
        if move in root.children:
            root.children[move].visit_count = 10 - i
    
    # Test different temperatures
    policy_1 = mcts.get_action_policy(root, temperature=1.0)
    policy_01 = mcts.get_action_policy(root, temperature=0.1)
    policy_0 = mcts.get_action_policy(root, temperature=0.0)
    
    # Check policy properties
    assert policy_1.shape == (32,)
    assert np.all(policy_1 >= 0)
    assert np.isclose(np.sum(policy_1), 1.0)
    
    # Check temperature effects
    assert np.argmax(policy_1) == np.argmax(policy_01)  # Same best move
    assert np.argmax(policy_01) == np.argmax(policy_0)  # Same best move
    assert np.max(policy_0) == 1.0  # Deterministic at T=0

def test_mcts_evaluation(model, config):
    """Test MCTS position evaluation."""
    mcts = MCTS(model, config)
    board = chess.Board()
    
    # Evaluate position
    policy, value = mcts._evaluate(board)
    
    # Check results
    assert policy.shape == (32,)
    assert np.all(policy >= 0)
    assert np.isclose(np.sum(policy), 1.0)
    assert isinstance(value, float)
    
    # Check transposition table
    board_hash = board.get_hash()
    assert board_hash in mcts.transposition_table
    assert mcts.transposition_table[board_hash] == (policy, value)

def test_mcts_transposition_table(model, config):
    """Test MCTS transposition table functionality."""
    mcts = MCTS(model, config)
    board = chess.Board()
    
    # Test empty table
    assert len(mcts.transposition_table) == 0
    assert mcts.current_table_size == 0
    
    # Evaluate position
    policy1, value1 = mcts._evaluate(board)
    
    # Check table size
    assert mcts.current_table_size == 1
    assert len(mcts.transposition_table) == 1
    
    # Evaluate same position again
    policy2, value2 = mcts._evaluate(board)
    
    # Check that table size didn't change
    assert mcts.current_table_size == 1
    assert len(mcts.transposition_table) == 1
    
    # Check that results are the same
    assert np.array_equal(policy1, policy2)
    assert value1 == value2

def test_mcts_batching(model, config):
    """Test MCTS batching functionality."""
    board = chess.Board()
    mcts = MCTS(model, config)
    
    # Test single evaluation
    policy1, value1 = mcts._evaluate(board)
    
    # Test batch evaluation
    boards = [board.copy() for _ in range(3)]
    results = mcts._evaluate_batch(boards)
    
    # Check results
    assert len(results) == len(boards)
    for policy, value in results:
        assert policy.shape == (32,)
        assert np.all(policy >= 0)
        assert np.isclose(np.sum(policy), 1.0)
        assert isinstance(value, float)
    
    # Check that first result matches single evaluation
    assert np.array_equal(policy1, results[0][0])
    assert value1 == results[0][1]

def test_mcts_memory_management(model, config):
    """Test MCTS memory management."""
    mcts = MCTS(model, config)
    board = chess.Board()
    
    # Fill table with dummy entries
    dummy_policy = np.ones(32, dtype=np.float32) / 32
    dummy_value = 0.5
    
    # Add entries until table is almost full
    while mcts.current_table_size < mcts.max_table_size - 1:
        board_hash = f"dummy_{len(mcts.transposition_table)}"
        mcts.transposition_table[board_hash] = (dummy_policy, dummy_value)
        mcts.current_table_size += 1
    
    # Check table size
    assert mcts.current_table_size == mcts.max_table_size - 1
    
    # Add one more entry
    board_hash = "new_entry"
    mcts.transposition_table[board_hash] = (dummy_policy, dummy_value)
    mcts.current_table_size += 1
    
    # Check that table size didn't exceed max
    assert mcts.current_table_size == mcts.max_table_size
    assert len(mcts.transposition_table) == mcts.max_table_size

def test_mcts_search_with_batching():
    """Test MCTS search with batched evaluation."""
    board = chess.Board()
    config = MCTSConfig(num_simulations=10, batch_size=4)
    
    # Create mock model
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))
            
        def forward(self, x, edge_index, edge_attr, batch):
            policy = torch.zeros(32)
            policy[0:3] = torch.tensor([0.6, 0.3, 0.1])
            value = torch.tensor([0.0])
            return policy, value
            
        def parameters(self):
            return [self.dummy]
    
    model = MockModel()
    mcts = MCTS(model, config)
    
    # Run search
    root = mcts.search(board)
    
    # Check search results
    assert root.is_expanded
    assert len(root.children) > 0
    assert root.visit_count == config.num_simulations

def test_mcts_parallel_evaluation(model, config):
    """Test MCTS with multiple parallel evaluations."""
    mcts = MCTS(model, config)
    boards = [chess.Board() for _ in range(10)]
    
    # Start multiple evaluations
    results = mcts._evaluate_batch(boards)
    
    # Check results
    assert len(results) == len(boards)
    for policy, value in results:
        assert policy.shape == (32,)
        assert np.all(policy >= 0)
        assert np.isclose(np.sum(policy), 1.0)
        assert isinstance(value, float)
    
    # Check that all results are in transposition table
    for board in boards:
        assert board.get_hash() in mcts.transposition_table

def test_mcts_terminal_value():
    """Test MCTS terminal value calculation."""
    mcts = MCTS(None, None)  # Model and config not needed for this test
    
    # Test white win
    board = chess.Board()
    board.result = GameResult.WHITE_WINS
    assert mcts._get_terminal_value(board) == 1.0
    
    # Test black win
    board.result = GameResult.BLACK_WINS
    assert mcts._get_terminal_value(board) == -1.0
    
    # Test draw
    board.result = GameResult.DRAW
    assert mcts._get_terminal_value(board) == 0.0 