use std::collections::HashMap;
use petgraph::graph::{Graph, NodeIndex};
use petgraph::Directed;
use serde::{Serialize, Deserialize};

use super::node::{Node, NodeType, PieceNode, CriticalSquareNode};
use super::edge::Edge;
use super::Coordinate;

/// Represents a Positional Adjacency Graph (PAG)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PAG {
    /// The underlying graph structure
    #[serde(skip)]
    graph: Graph<NodeType, Edge, Directed>,
    
    /// Maps node IDs to their indices in the graph
    #[serde(skip)]
    node_indices: HashMap<u64, NodeIndex>,
    
    /// Maps coordinates to piece node IDs
    piece_positions: HashMap<Coordinate, u64>,
    
    /// Maps coordinates to critical square node IDs
    critical_squares: HashMap<Coordinate, u64>,
    
    /// The size of the board (e.g., 5 for 5x5, 8 for 8x8)
    board_size: u8,
}

impl PAG {
    /// Creates a new empty PAG
    pub fn new(board_size: u8) -> Self {
        Self {
            graph: Graph::new(),
            node_indices: HashMap::new(),
            piece_positions: HashMap::new(),
            critical_squares: HashMap::new(),
            board_size,
        }
    }

    /// Adds a node to the graph
    pub fn add_node(&mut self, node: NodeType) -> NodeIndex {
        let node_id = match &node {
            NodeType::Piece(p) => p.get_id(),
            NodeType::CriticalSquare(c) => c.get_id(),
        };

        let coord = match &node {
            NodeType::Piece(p) => p.get_coordinate(),
            NodeType::CriticalSquare(c) => c.get_coordinate(),
        };

        let idx = self.graph.add_node(node.clone());
        self.node_indices.insert(node_id, idx);

        match node {
            NodeType::Piece(_) => { self.piece_positions.insert(coord, node_id); },
            NodeType::CriticalSquare(_) => { self.critical_squares.insert(coord, node_id); },
        }

        idx
    }

    /// Adds an edge to the graph
    pub fn add_edge(&mut self, edge: Edge) -> Option<()> {
        let source_idx = self.node_indices.get(&edge.source_id())?;
        let target_idx = self.node_indices.get(&edge.target_id())?;
        self.graph.add_edge(*source_idx, *target_idx, edge);
        Some(())
    }

    /// Gets a node by its ID
    pub fn get_node(&self, id: u64) -> Option<&NodeType> {
        let idx = self.node_indices.get(&id)?;
        self.graph.node_weight(*idx)
    }

    /// Gets a mutable reference to a node by its ID
    pub fn get_node_mut(&mut self, id: u64) -> Option<&mut NodeType> {
        let idx = self.node_indices.get(&id)?;
        self.graph.node_weight_mut(*idx)
    }

    /// Gets a piece at a coordinate
    pub fn get_piece_at(&self, coord: Coordinate) -> Option<&PieceNode> {
        let id = self.piece_positions.get(&coord)?;
        match self.get_node(*id)? {
            NodeType::Piece(p) => Some(p),
            _ => None,
        }
    }

    /// Gets a critical square at a coordinate
    pub fn get_critical_square_at(&self, coord: Coordinate) -> Option<&CriticalSquareNode> {
        let id = self.critical_squares.get(&coord)?;
        match self.get_node(*id)? {
            NodeType::CriticalSquare(c) => Some(c),
            _ => None,
        }
    }

    /// Gets all edges connected to a node
    pub fn get_edges_for_node(&self, id: u64) -> Vec<&Edge> {
        let idx = match self.node_indices.get(&id) {
            Some(idx) => *idx,
            None => return Vec::new(),
        };

        self.graph.edges(idx)
            .map(|edge| edge.weight())
            .collect()
    }

    /// Gets the board size
    pub fn board_size(&self) -> u8 {
        self.board_size
    }

    /// Gets the number of nodes in the graph
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Gets the number of edges in the graph
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::node::{PieceType, Color};
    use super::super::edge::{DirectRelationType};

    #[test]
    fn test_pag_creation() {
        let pag = PAG::new(5);
        assert_eq!(pag.board_size(), 5);
        assert_eq!(pag.node_count(), 0);
        assert_eq!(pag.edge_count(), 0);
    }

    #[test]
    fn test_node_addition_and_retrieval() {
        let mut pag = PAG::new(5);
        
        // Create and add a piece node
        let piece = PieceNode::new(
            1,
            PieceType::Knight,
            Color::White,
            Coordinate::new(2, 2),
            3,
        );
        let node_type = NodeType::Piece(piece);
        pag.add_node(node_type.clone());

        // Verify retrieval
        let retrieved = pag.get_node(1).unwrap();
        assert!(matches!(retrieved, NodeType::Piece(_)));
        
        if let NodeType::Piece(p) = retrieved {
            assert_eq!(p.get_id(), 1);
            assert_eq!(p.piece_type(), PieceType::Knight);
        }
    }

    #[test]
    fn test_edge_addition() {
        let mut pag = PAG::new(5);
        
        // Add two pieces
        let piece1 = PieceNode::new(1, PieceType::Queen, Color::White, Coordinate::new(0, 0), 9);
        let piece2 = PieceNode::new(2, PieceType::Pawn, Color::Black, Coordinate::new(1, 1), 1);
        
        pag.add_node(NodeType::Piece(piece1));
        pag.add_node(NodeType::Piece(piece2));

        // Create an attack edge between them
        let edge = Edge::new(
            EdgeType::DirectRelation(DirectRelationType::Attack {
                attacker_type: PieceType::Queen,
                target_type: PieceType::Pawn,
                strength: 1.0,
            }),
            1.0,
            1,
            2,
        );

        // Add the edge
        assert!(pag.add_edge(edge).is_some());
        
        // Verify edge count
        assert_eq!(pag.edge_count(), 1);
        
        // Verify edge retrieval
        let edges = pag.get_edges_for_node(1);
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].source_id(), 1);
        assert_eq!(edges[0].target_id(), 2);
    }
} 