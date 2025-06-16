"""
Graph Neural Network model for chess position evaluation using PAG.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GATConv, HeteroConv, global_mean_pool
from torch_geometric.data import HeteroData
from typing import Dict, Optional, Tuple, List, Union
import logging
import chess
from chess import Move, Board
import numpy as np
from torch_geometric.nn import MessagePassing
import math

logger = logging.getLogger(__name__)

class PAGNodeEncoder(nn.Module):
    """Node feature encoder for PAGConv layers."""
    
    def __init__(self, hidden_dim: int = 64):
        """Initialize the encoder.
        
        Args:
            hidden_dim: Dimension of hidden layers
        """
        super().__init__()
        
        # Piece node encoder (11 features -> hidden_dim)
        self.piece_encoder = nn.Sequential(
            nn.Linear(11, hidden_dim),  # Updated from 9 to 11 features
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Critical square node encoder (4 features -> hidden_dim)
        self.critical_square_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, x_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x_dict: Dictionary containing node features for each node type
            
        Returns:
            Tuple of (piece_embeddings, critical_square_embeddings)
        """
        piece_emb = self.piece_encoder(x_dict['piece'])
        critical_square_emb = self.critical_square_encoder(x_dict['critical_square'])
        return piece_emb, critical_square_emb

class PAGEdgeEncoder(nn.Module):
    """Edge encoder for PAGConv layer."""
    
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int = 32):
        """Initialize edge encoder.
        
        Args:
            in_channels: Input feature dimension
            out_channels: Output feature dimension
            hidden_channels: Hidden layer dimension
        """
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )
    
    def forward(self, edge_features: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            edge_features: Edge feature tensor of shape [num_edges, in_channels]
            
        Returns:
            Encoded edge features of shape [num_edges, out_channels]
        """
        return self.encoder(edge_features)

class PAGConv(MessagePassing):
    """Piece-Aware Graph Convolution layer for chess positions."""
    
    def __init__(self, in_channels: Union[int, Dict[str, int]], out_channels: int, edge_dims: Dict[Tuple[str, str, str], int]):
        """Initialize PAGConv layer.
        
        Args:
            in_channels: Either an integer for single node type or a dict mapping node types to their feature dimensions
            out_channels: Output feature dimension
            edge_dims: Dictionary mapping edge types to their feature dimensions
        """
        super().__init__(aggr='mean')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dims = edge_dims
        
        # Node feature transformations for each node type
        if isinstance(in_channels, dict):
            self.node_lins = nn.ModuleDict({
                node_type: nn.Linear(dim, out_channels)
                for node_type, dim in in_channels.items()
            })
        else:
            self.node_lins = nn.ModuleDict({
                'piece': nn.Linear(in_channels, out_channels)
            })
        
        # Edge feature transformations for each edge type
        # Convert edge type tuples to string keys
        self.edge_lins = nn.ModuleDict({
            f"{src}->{rel}->{dst}": nn.Linear(dim, out_channels)
            for (src, rel, dst), dim in edge_dims.items()
        })
        
        # Create mapping from edge type tuples to string keys
        self.edge_type_to_key = {
            (src, rel, dst): f"{src}->{rel}->{dst}"
            for (src, rel, dst) in edge_dims.keys()
        }
        
        # Message transformation
        self.msg_lin = nn.Linear(out_channels * 2 + out_channels, out_channels)  # Added edge features
        
        # Update transformation
        self.update_lin = nn.Linear(out_channels * 2, out_channels)
        
        # Layer normalization
        self.norm = nn.LayerNorm(out_channels)
        
    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        edge_attr_dict: Dict[Tuple[str, str, str], torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the PAGConv layer.
        
        Args:
            x_dict: Dictionary of node features for each node type
            edge_index_dict: Dictionary of edge indices for each edge type
            edge_attr_dict: Dictionary of edge attributes for each edge type
            
        Returns:
            Dictionary of updated node features for each node type
        """
        out_dict = {}
        
        # Process each node type
        for node_type, x in x_dict.items():
            # Initialize output for this node type
            out = torch.zeros(x.size(0), self.out_channels, device=x.device)
            
            # Process each edge type that has this node type as source
            for edge_type, edge_index in edge_index_dict.items():
                src_type, rel_type, dst_type = edge_type
                if src_type != node_type:
                    continue
                    
                # Get edge attributes for this edge type
                edge_attr = edge_attr_dict.get(edge_type)
                if edge_attr is None:
                    continue
                
                # Get source and target node features
                src_x = x_dict[src_type]
                dst_x = x_dict[dst_type]
                
                # Transform edge attributes using string key
                edge_key = self.edge_type_to_key[edge_type]
                edge_emb = self.edge_lins[edge_key](edge_attr)
                
                # Transform source and target node features
                src_emb = self.node_lins[src_type](src_x)
                dst_emb = self.node_lins[dst_type](dst_x)
                
                # Create messages using edge indices
                row, col = edge_index
                src_features = src_emb[row]  # Source node features for each edge
                dst_features = dst_emb[col]  # Target node features for each edge
                edge_features = edge_emb  # Edge features for each edge
                
                # Concatenate features and create messages
                msg = torch.cat([src_features, dst_features, edge_features], dim=-1)
                msg = self.msg_lin(msg)
                
                # Aggregate messages using scatter
                aggr = torch.zeros_like(src_emb)
                aggr.scatter_add_(0, row.unsqueeze(-1).expand(-1, self.out_channels), msg)
                
                # Count number of messages per node
                count = torch.zeros(src_emb.size(0), device=src_emb.device)
                count.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float))
                count = count.unsqueeze(-1).clamp(min=1)  # Avoid division by zero
                
                # Average the aggregated messages
                aggr = aggr / count
                
                # Update node features
                out = out + self.update_lin(torch.cat([src_emb, aggr], dim=-1))
            
            # Apply layer normalization
            out = self.norm(out)
            out_dict[node_type] = out
            
        return out_dict
    
    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        """Message function for the PAGConv layer."""
        return x_j
    
    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Update function for the PAGConv layer."""
        return aggr_out

class ChessGNN(nn.Module):
    """Graph Neural Network for chess position evaluation."""
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_residual: bool = True
    ):
        """Initialize the model.
        
        Args:
            hidden_dim: Hidden dimension size
            num_layers: Number of GNN layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            use_residual: Whether to use residual connections
        """
        super().__init__()
        
        # Input dimensions
        self.piece_dim = 12  # 6 piece types * 2 colors
        self.critical_square_dim = 1  # Binary feature for critical squares
        
        # Model parameters
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_residual = use_residual
        
        # Node type embeddings
        self.piece_embedding = nn.Linear(self.piece_dim, hidden_dim)
        self.square_embedding = nn.Linear(self.critical_square_dim, hidden_dim)
        
        # GNN layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # First layer (piece to square)
        self.convs.append(GATConv(
            in_channels=(hidden_dim, hidden_dim),  # Use explicit dimensions after embedding
            out_channels=hidden_dim // num_heads,
            heads=num_heads,
            dropout=dropout,
            add_self_loops=False,
            edge_dim=None,
            concat=True  # Concatenate attention heads
        ))
        self.norms.append(nn.LayerNorm(hidden_dim))
        
        # Middle layers (square to square)
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // num_heads,
                heads=num_heads,
                dropout=dropout,
                add_self_loops=False,
                edge_dim=None,
                concat=True  # Concatenate attention heads
            ))
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        # Output heads
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 5312)  # Updated to match our move space (64 * 64 + 64 * 64 * 4 for promotions)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, GATConv):
                # Initialize attention parameters
                if hasattr(module, 'lin_src') and module.lin_src is not None:
                    if not isinstance(module.lin_src, nn.UninitializedParameter):
                        nn.init.xavier_uniform_(module.lin_src.weight)
                        if module.lin_src.bias is not None:
                            nn.init.zeros_(module.lin_src.bias)
                if hasattr(module, 'lin_dst') and module.lin_dst is not None:
                    if not isinstance(module.lin_dst, nn.UninitializedParameter):
                        nn.init.xavier_uniform_(module.lin_dst.weight)
                        if module.lin_dst.bias is not None:
                            nn.init.zeros_(module.lin_dst.bias)
                if hasattr(module, 'att_src') and module.att_src is not None:
                    if not isinstance(module.att_src, nn.UninitializedParameter):
                        nn.init.xavier_uniform_(module.att_src)
                if hasattr(module, 'att_dst') and module.att_dst is not None:
                    if not isinstance(module.att_dst, nn.UninitializedParameter):
                        nn.init.xavier_uniform_(module.att_dst)
    
    def forward(self, data: HeteroData) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            data: Heterogeneous graph data containing:
                - piece nodes with features [num_pieces, piece_dim]
                - square nodes with features [num_squares, critical_square_dim]
                - edge_index_dict with 'piece_to_square' and 'square_to_square' edges
                
        Returns:
            Tuple of (policy_logits, value)
        """
        # Get node features and apply embeddings
        piece_x = self.piece_embedding(data['piece'].x)
        square_x = self.square_embedding(data['square'].x)
        
        # Store node features in dictionary
        x_dict = {
            'piece': piece_x,
            'square': square_x
        }
        
        # Process through GNN layers
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            # First layer connects pieces to squares
            if i == 0:
                # Get edge index for piece to square connections
                edge_index = data.edge_index_dict[('piece', 'to', 'square')]
                # Ensure edge indices are contiguous and in the correct range
                edge_index = edge_index.contiguous()
                # Verify edge indices are valid
                assert edge_index[0].max() < piece_x.size(0), f"Invalid source node index: {edge_index[0].max()} >= {piece_x.size(0)}"
                assert edge_index[1].max() < square_x.size(0), f"Invalid target node index: {edge_index[1].max()} >= {square_x.size(0)}"
                # Process piece to square connections
                square_x = conv((piece_x, square_x), edge_index)
                square_x = norm(square_x)
                square_x = F.relu(square_x)
                square_x = F.dropout(square_x, p=self.dropout, training=self.training)
                x_dict['square'] = square_x
            else:
                # Get edge index for square to square connections
                edge_index = data.edge_index_dict[('square', 'to', 'square')]
                # Ensure edge indices are contiguous and in the correct range
                edge_index = edge_index.contiguous()
                # Verify edge indices are valid
                assert edge_index[0].max() < square_x.size(0), f"Invalid source node index: {edge_index[0].max()} >= {square_x.size(0)}"
                assert edge_index[1].max() < square_x.size(0), f"Invalid target node index: {edge_index[1].max()} >= {square_x.size(0)}"
                # Process square to square connections
                square_x = conv(square_x, edge_index)
                square_x = norm(square_x)
                square_x = F.relu(square_x)
                square_x = F.dropout(square_x, p=self.dropout, training=self.training)
                x_dict['square'] = square_x
        
        # Get final square features
        square_features = x_dict['square']
        
        # Global pooling over all squares
        global_features = global_mean_pool(square_features, data['square'].batch)
        
        # Policy and value heads
        policy_logits = self.policy_head(global_features)
        value = self.value_head(global_features)
        
        return policy_logits, value
    
    def get_config(self) -> Dict:
        """Get model configuration.
        
        Returns:
            Dictionary containing model configuration
        """
        return {
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'dropout': self.dropout,
            'use_residual': self.use_residual,
            'piece_dim': self.piece_dim,
            'critical_square_dim': self.critical_square_dim
        } 