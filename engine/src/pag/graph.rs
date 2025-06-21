use std::collections::HashMap;
use petgraph::graph::{Graph, NodeIndex};
use petgraph::Directed;
use serde::{Serialize, Deserialize, Serializer, Deserializer};


use super::node::{Node, NodeType};
use super::edge::Edge;
use super::Coordinate;

// Custom serialization for the graph structure
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SerializableGraph {
    nodes: Vec<(u64, NodeType)>,  // (id, node)
    edges: Vec<(u64, u64, Edge)>, // (source_id, target_id, edge)
    piece_positions: HashMap<Coordinate, u64>,
    critical_squares: HashMap<Coordinate, u64>,
    board_size: u8,
}

/// Represents a Positional Adjacency Graph (PAG)
#[derive(Debug, Clone)]
pub struct PAG {
    /// The underlying graph structure
    graph: Graph<NodeType, Edge, Directed>,
    
    /// Maps node IDs to their indices in the graph
    node_indices: HashMap<u64, NodeIndex>,
    
    /// Maps coordinates to piece node IDs
    piece_positions: HashMap<Coordinate, u64>,
    
    /// Maps coordinates to critical square node IDs
    critical_squares: HashMap<Coordinate, u64>,
    
    /// The size of the board (e.g., 5 for 5x5, 8 for 8x8)
    board_size: u8,
    
    /// Cache for frequently accessed neighbors
    neighbor_cache: HashMap<u64, Vec<u64>>,
    
    /// Generation counter for cache invalidation
    generation: u64,
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
            neighbor_cache: HashMap::new(),
            generation: 0,
        }
    }

    /// Adds a node to the graph
    pub fn add_node(&mut self, node: NodeType) -> NodeIndex {
        let node_id = match &node {
            NodeType::DensePiece(p) => p.get_id(),
            NodeType::DenseCriticalSquare(c) => c.get_id(),
            NodeType::LegacyPiece(p) => p.get_id(),
            NodeType::LegacyCriticalSquare(c) => c.get_id(),
        };

        let coord = match &node {
            NodeType::DensePiece(p) => p.get_coordinate(),
            NodeType::DenseCriticalSquare(c) => c.get_coordinate(),
            NodeType::LegacyPiece(p) => p.get_coordinate(),
            NodeType::LegacyCriticalSquare(c) => c.get_coordinate(),
        };

        let idx = self.graph.add_node(node);
        self.node_indices.insert(node_id, idx);

        match &self.graph[idx] {
            NodeType::DensePiece(_) | NodeType::LegacyPiece(_) => { 
                self.piece_positions.insert(coord, node_id); 
            },
            NodeType::DenseCriticalSquare(_) | NodeType::LegacyCriticalSquare(_) => { 
                self.critical_squares.insert(coord, node_id); 
            },
        }

        self.invalidate_cache();
        idx
    }

    /// Adds an edge to the graph
    pub fn add_edge(&mut self, edge: Edge) -> Result<(), String> {
        let source_idx = self.node_indices.get(&edge.source_id())
            .ok_or_else(|| format!("Source node {} not found", edge.source_id()))?;
        let target_idx = self.node_indices.get(&edge.target_id())
            .ok_or_else(|| format!("Target node {} not found", edge.target_id()))?;
        
        self.graph.add_edge(*source_idx, *target_idx, edge);
        self.invalidate_cache();
        Ok(())
    }

    /// Batch add multiple edges for better performance
    pub fn add_edges(&mut self, edges: Vec<Edge>) -> Result<(), String> {
        for edge in edges {
            self.add_edge(edge)?;
        }
        Ok(())
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
    pub fn get_piece_at(&self, coord: Coordinate) -> Option<u64> {
        self.piece_positions.get(&coord).copied()
    }

    /// Gets a critical square at a coordinate
    pub fn get_critical_square_at(&self, coord: Coordinate) -> Option<u64> {
        self.critical_squares.get(&coord).copied()
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

    /// Gets neighboring node IDs (cached for performance)
    pub fn get_neighbors(&self, id: u64) -> Vec<u64> {
        if let Some(neighbors) = self.neighbor_cache.get(&id) {
            return neighbors.clone();
        }

        let idx = match self.node_indices.get(&id) {
            Some(idx) => *idx,
            None => return Vec::new(),
        };

        let neighbors: Vec<u64> = self.graph.neighbors(idx)
            .filter_map(|neighbor_idx| {
                self.graph.node_weight(neighbor_idx).and_then(|node| {
                    match node {
                        NodeType::DensePiece(p) => Some(p.get_id()),
                        NodeType::DenseCriticalSquare(c) => Some(c.get_id()),
                        NodeType::LegacyPiece(p) => Some(p.get_id()),
                        NodeType::LegacyCriticalSquare(c) => Some(c.get_id()),
                    }
                })
            })
            .collect();

        // Cache the result (we'll use interior mutability later if needed)
        neighbors
    }

    /// Gets all piece node IDs
    pub fn get_piece_ids(&self) -> Vec<u64> {
        self.piece_positions.values().copied().collect()
    }

    /// Gets all critical square node IDs
    pub fn get_critical_square_ids(&self) -> Vec<u64> {
        self.critical_squares.values().copied().collect()
    }

    /// Removes a node and all its edges
    pub fn remove_node(&mut self, id: u64) -> Result<(), String> {
        let idx = self.node_indices.remove(&id)
            .ok_or_else(|| format!("Node {} not found", id))?;

        // Get the node before removing it to update coordinate mappings
        if let Some(node) = self.graph.node_weight(idx) {
            match node {
                NodeType::DensePiece(p) => {
                    self.piece_positions.remove(&p.get_coordinate());
                },
                NodeType::DenseCriticalSquare(c) => {
                    self.critical_squares.remove(&c.get_coordinate());
                },
                NodeType::LegacyPiece(p) => {
                    self.piece_positions.remove(&p.get_coordinate());
                },
                NodeType::LegacyCriticalSquare(c) => {
                    self.critical_squares.remove(&c.get_coordinate());
                },
            }
        }

        self.graph.remove_node(idx);
        self.invalidate_cache();
        Ok(())
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

    /// Clears the graph
    pub fn clear(&mut self) {
        self.graph.clear();
        self.node_indices.clear();
        self.piece_positions.clear();
        self.critical_squares.clear();
        self.invalidate_cache();
    }

    /// Checks if the graph is empty
    pub fn is_empty(&self) -> bool {
        self.graph.node_count() == 0
    }

    /// Gets statistics about the graph
    pub fn get_stats(&self) -> PAGStats {
        PAGStats {
            node_count: self.node_count(),
            edge_count: self.edge_count(),
            piece_count: self.piece_positions.len(),
            critical_square_count: self.critical_squares.len(),
            board_size: self.board_size,
        }
    }

    /// Invalidates the neighbor cache
    fn invalidate_cache(&mut self) {
        self.neighbor_cache.clear();
        self.generation = self.generation.wrapping_add(1);
    }

    /// Validates the graph structure for consistency
    pub fn validate(&self) -> Result<(), String> {
        // Check that all position mappings point to valid nodes
        for (coord, id) in &self.piece_positions {
            if !self.node_indices.contains_key(id) {
                return Err(format!("Piece position mapping at {:?} points to invalid node {}", coord, id));
            }
        }

        for (coord, id) in &self.critical_squares {
            if !self.node_indices.contains_key(id) {
                return Err(format!("Critical square mapping at {:?} points to invalid node {}", coord, id));
            }
        }

        // Check that all node indices are valid
        for (id, idx) in &self.node_indices {
            if !self.graph.node_weight(*idx).is_some() {
                return Err(format!("Node index mapping for {} points to invalid graph index", id));
            }
        }

        Ok(())
    }
}

/// Statistics about a PAG
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PAGStats {
    pub node_count: usize,
    pub edge_count: usize,
    pub piece_count: usize,
    pub critical_square_count: usize,
    pub board_size: u8,
}

// Custom serialization implementation
impl Serialize for PAG {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let serializable = SerializableGraph {
            nodes: self.node_indices.iter()
                .map(|(id, idx)| (*id, self.graph[*idx].clone()))
                .collect(),
            edges: self.graph.edge_indices()
                .filter_map(|edge_idx| {
                    let (source_idx, target_idx) = self.graph.edge_endpoints(edge_idx)?;
                    let source_node = self.graph.node_weight(source_idx)?;
                    let target_node = self.graph.node_weight(target_idx)?;
                    let edge = self.graph.edge_weight(edge_idx)?;
                    
                    let source_id = match source_node {
                        NodeType::DensePiece(p) => p.get_id(),
                        NodeType::DenseCriticalSquare(c) => c.get_id(),
                        NodeType::LegacyPiece(p) => p.get_id(),
                        NodeType::LegacyCriticalSquare(c) => c.get_id(),
                    };
                    let target_id = match target_node {
                        NodeType::DensePiece(p) => p.get_id(),
                        NodeType::DenseCriticalSquare(c) => c.get_id(),
                        NodeType::LegacyPiece(p) => p.get_id(),
                        NodeType::LegacyCriticalSquare(c) => c.get_id(),
                    };
                    
                    Some((source_id, target_id, edge.clone()))
                })
                .collect(),
            piece_positions: self.piece_positions.clone(),
            critical_squares: self.critical_squares.clone(),
            board_size: self.board_size,
        };
        
        serializable.serialize(serializer)
    }
}

// Custom deserialization implementation
impl<'de> Deserialize<'de> for PAG {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let serializable = SerializableGraph::deserialize(deserializer)?;
        
        let mut pag = PAG::new(serializable.board_size);
        pag.piece_positions = serializable.piece_positions;
        pag.critical_squares = serializable.critical_squares;
        
        // Add nodes
        for (id, node) in serializable.nodes {
            let idx = pag.graph.add_node(node);
            pag.node_indices.insert(id, idx);
        }
        
        // Add edges
        for (_source_id, _target_id, edge) in serializable.edges {
            if let Err(e) = pag.add_edge(edge) {
                return Err(serde::de::Error::custom(format!("Failed to add edge: {}", e)));
            }
        }
        
        Ok(pag)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::node::{PieceType, Color};
    use super::super::edge::{DirectRelationType, EdgeType};

    #[test]
    fn test_pag_creation() {
        let pag = PAG::new(5);
        assert_eq!(pag.board_size(), 5);
        assert_eq!(pag.node_count(), 0);
        assert_eq!(pag.edge_count(), 0);
        assert!(pag.is_empty());
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
        pag.add_node(node_type);

        // Verify retrieval
        let retrieved = pag.get_node(1).unwrap();
        assert!(matches!(retrieved, NodeType::Piece(_)));
        
        if let NodeType::Piece(p) = retrieved {
            assert_eq!(p.get_id(), 1);
            assert_eq!(p.piece_type(), PieceType::Knight);
        }

        // Test coordinate lookup
        let piece_at_coord = pag.get_piece_at(Coordinate::new(2, 2)).unwrap();
        assert_eq!(piece_at_coord.get_id(), 1);
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
        assert!(pag.add_edge(edge).is_ok());
        
        // Verify edge count
        assert_eq!(pag.edge_count(), 1);
        
        // Verify edge retrieval
        let edges = pag.get_edges_for_node(1);
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].source_id(), 1);
        assert_eq!(edges[0].target_id(), 2);
    }

    #[test]
    fn test_batch_operations() {
        let mut pag = PAG::new(8);
        
        // Add multiple nodes
        let piece1 = PieceNode::new(1, PieceType::King, Color::White, Coordinate::new(4, 0), 100);
        let piece2 = PieceNode::new(2, PieceType::Queen, Color::White, Coordinate::new(3, 0), 9);
        let piece3 = PieceNode::new(3, PieceType::Rook, Color::Black, Coordinate::new(0, 7), 5);
        
        pag.add_node(NodeType::Piece(piece1));
        pag.add_node(NodeType::Piece(piece2));
        pag.add_node(NodeType::Piece(piece3));

        // Create multiple edges
        let edges = vec![
            Edge::new(
                EdgeType::DirectRelation(DirectRelationType::Defense {
                    defender_type: PieceType::Queen,
                    protected_type: PieceType::King,
                    strength: 1.0,
                }),
                1.0,
                2,
                1,
            ),
            Edge::new(
                EdgeType::DirectRelation(DirectRelationType::Attack {
                    attacker_type: PieceType::Rook,
                    target_type: PieceType::Queen,
                    strength: 0.8,
                }),
                0.8,
                3,
                2,
            ),
        ];

        // Batch add edges
        assert!(pag.add_edges(edges).is_ok());
        assert_eq!(pag.edge_count(), 2);
    }

    #[test]
    fn test_serialization() {
        let mut pag = PAG::new(5);
        
        // Add some nodes and edges
        let piece = PieceNode::new(1, PieceType::Knight, Color::White, Coordinate::new(2, 2), 3);
        pag.add_node(NodeType::Piece(piece));
        
        // Serialize to JSON
        let json = serde_json::to_string(&pag).unwrap();
        
        // Deserialize back
        let deserialized: PAG = serde_json::from_str(&json).unwrap();
        
        // Verify structure is preserved
        assert_eq!(deserialized.board_size(), pag.board_size());
        assert_eq!(deserialized.node_count(), pag.node_count());
        assert_eq!(deserialized.edge_count(), pag.edge_count());
        
        // Verify specific node is preserved
        let retrieved = deserialized.get_node(1).unwrap();
        assert!(matches!(retrieved, NodeType::Piece(_)));
    }

    #[test]
    fn test_validation() {
        let mut pag = PAG::new(5);
        
        // Add a piece
        let piece = PieceNode::new(1, PieceType::Knight, Color::White, Coordinate::new(2, 2), 3);
        pag.add_node(NodeType::Piece(piece));
        
        // Validation should pass
        assert!(pag.validate().is_ok());
        
        // Test stats
        let stats = pag.get_stats();
        assert_eq!(stats.piece_count, 1);
        assert_eq!(stats.node_count, 1);
        assert_eq!(stats.board_size, 5);
    }

    #[test]
    fn test_node_removal() {
        let mut pag = PAG::new(5);
        
        // Add a piece
        let piece = PieceNode::new(1, PieceType::Knight, Color::White, Coordinate::new(2, 2), 3);
        pag.add_node(NodeType::Piece(piece));
        
        assert_eq!(pag.node_count(), 1);
        
        // Remove the piece
        assert!(pag.remove_node(1).is_ok());
        assert_eq!(pag.node_count(), 0);
        assert!(pag.is_empty());
        
        // Verify coordinate mapping is cleaned up
        assert!(pag.get_piece_at(Coordinate::new(2, 2)).is_none());
    }
} 