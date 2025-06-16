"""
Tests for the data processing components.
"""

import os
import json
import pytest
import torch
import tempfile
from rival_ai.data import (
    PAGDataset,
    PAGInMemoryDataset,
    PAGDataLoader,
    PAGInMemoryDataLoader,
    pag_to_tensor,
    tensor_to_pag,
)

@pytest.fixture
def sample_pag_data():
    """Create sample PAG data for testing."""
    return {
        'nodes': [
            {
                'id': 0,
                'piece_type': 1,  # Pawn
                'color': 1,  # White
                'square': [0, 1],
                'material_value': 1.0,
                'mobility_score': 0.5,
                'is_attacked': False,
                'is_defended': True,
                'is_king_shield': False,
            },
            {
                'id': 1,
                'piece_type': 2,  # Knight
                'color': 1,  # White
                'square': [1, 0],
                'material_value': 3.0,
                'mobility_score': 0.8,
                'is_attacked': False,
                'is_defended': True,
                'is_king_shield': False,
            },
        ],
        'edges': [
            {
                'source': 0,
                'target': 1,
                'type': 1,  # Attack
                'strength': 0.5,
                'control_degree': 0.3,
                'safety_score': 0.7,
                'cooperation_type': 0,
                'obstruction_type': 0,
                'vulnerability_type': 0,
                'severity': 0.0,
            },
        ],
        'board_size': [8, 8],
        'move_count': 5,
        'game_phase': 1,
    }

@pytest.fixture
def temp_dataset_dir(sample_pag_data):
    """Create a temporary directory with sample PAG data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create raw data directory
        raw_dir = os.path.join(temp_dir, 'raw')
        os.makedirs(raw_dir)
        
        # Save sample PAG data
        for i in range(3):  # Create 3 sample files
            with open(os.path.join(raw_dir, f'pag_{i}.json'), 'w') as f:
                json.dump(sample_pag_data, f)
        
        # Save metadata
        metadata = {
            'num_positions': 3,
            'board_size': [8, 8],
            'version': '1.0.0',
        }
        with open(os.path.join(temp_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)
        
        yield temp_dir

def test_pag_to_tensor(sample_pag_data):
    """Test PAG to tensor conversion."""
    pag_json = json.dumps(sample_pag_data)
    x, edge_index, edge_attr, metadata = pag_to_tensor(pag_json)
    
    # Check shapes
    assert x.shape == (2, 9)  # 2 nodes, 9 features each
    assert edge_index.shape == (2, 1)  # 1 edge, 2 endpoints
    assert edge_attr.shape == (1, 8)  # 1 edge, 8 features
    
    # Check metadata
    assert metadata['board_size'].tolist() == [8, 8]
    assert metadata['move_count'].item() == 5
    assert metadata['game_phase'].item() == 1

def test_tensor_to_pag(sample_pag_data):
    """Test tensor to PAG conversion."""
    # Convert to tensors
    pag_json = json.dumps(sample_pag_data)
    x, edge_index, edge_attr, metadata = pag_to_tensor(pag_json)
    
    # Convert back to PAG
    new_pag_json = tensor_to_pag(x, edge_index, edge_attr, metadata)
    new_pag_data = json.loads(new_pag_json)
    
    # Check data integrity
    assert len(new_pag_data['nodes']) == len(sample_pag_data['nodes'])
    assert len(new_pag_data['edges']) == len(sample_pag_data['edges'])
    assert new_pag_data['board_size'] == sample_pag_data['board_size']
    assert new_pag_data['move_count'] == sample_pag_data['move_count']
    assert new_pag_data['game_phase'] == sample_pag_data['game_phase']

def test_pag_dataset(temp_dataset_dir):
    """Test PAGDataset functionality."""
    dataset = PAGDataset(temp_dataset_dir)
    
    # Check dataset length
    assert len(dataset) == 3
    
    # Check data loading
    data = dataset[0]
    assert isinstance(data, torch_geometric.data.Data)
    assert data.x.shape[0] == 2  # 2 nodes
    assert data.edge_index.shape[1] == 1  # 1 edge
    assert data.edge_attr.shape[0] == 1  # 1 edge
    
    # Check metadata
    assert data.board_size.tolist() == [8, 8]
    assert data.move_count.item() == 5
    assert data.game_phase.item() == 1

def test_pag_in_memory_dataset(temp_dataset_dir):
    """Test PAGInMemoryDataset functionality."""
    dataset = PAGInMemoryDataset(temp_dataset_dir)
    
    # Check dataset length
    assert len(dataset) == 3
    
    # Check data is loaded into memory
    assert len(dataset.data) == 3
    
    # Check data access
    data = dataset[0]
    assert isinstance(data, torch_geometric.data.Data)
    assert data.x.shape[0] == 2
    assert data.edge_index.shape[1] == 1
    assert data.edge_attr.shape[0] == 1

def test_pag_dataloader(temp_dataset_dir):
    """Test PAGDataLoader functionality."""
    dataset = PAGDataset(temp_dataset_dir)
    dataloader = PAGDataLoader(dataset, batch_size=2, shuffle=False)
    
    # Check batch loading
    batch = next(iter(dataloader))
    assert isinstance(batch, torch_geometric.data.Batch)
    assert batch.num_graphs == 2
    assert batch.x.shape[0] == 4  # 2 graphs * 2 nodes
    assert batch.edge_index.shape[1] == 2  # 2 graphs * 1 edge

def test_pag_in_memory_dataloader(temp_dataset_dir):
    """Test PAGInMemoryDataLoader functionality."""
    dataset = PAGInMemoryDataset(temp_dataset_dir)
    
    # Test invalid num_workers
    with pytest.raises(ValueError):
        PAGInMemoryDataLoader(dataset, num_workers=2)
    
    # Test valid dataloader
    dataloader = PAGInMemoryDataLoader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(dataloader))
    assert isinstance(batch, torch_geometric.data.Batch)
    assert batch.num_graphs == 2
    assert batch.x.shape[0] == 4
    assert batch.edge_index.shape[1] == 2 