use std::str::FromStr;

pub mod node;
pub mod edge;
pub mod graph;
pub mod builder;
pub mod feature_extraction;

pub use node::*;
pub use edge::*;
pub use graph::*;
pub use builder::*;
pub use feature_extraction::*;

/// Represents a coordinate on the chess board
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Coordinate {
    pub rank: u8,
    pub file: u8,
}

impl Coordinate {
    pub fn new(rank: u8, file: u8) -> Self {
        Self { rank, file }
    }
}

/// High-level dense PAG builder
pub struct DensePAGBuilder {
    feature_extractor: FeatureExtractor,
}

impl DensePAGBuilder {
    /// Create a new dense PAG builder
    pub fn new() -> Self {
        Self {
            feature_extractor: FeatureExtractor::new(),
        }
    }
    
    /// Build a complete ultra-dense PAG from a chess board
    pub fn build_from_board(&mut self, board: &chess::Board) -> PAG {
        let mut pag = PAG::new(8); // 8x8 chess board
        
        // Extract ultra-dense features
        let (pieces, critical_squares, edges) = self.feature_extractor.extract_position_features(board);
        
        // Add dense pieces to PAG
        for piece in pieces {
            let node_type = NodeType::DensePiece(piece);
            pag.add_node(node_type);
        }
        
        // Add dense critical squares to PAG
        for square in critical_squares {
            let node_type = NodeType::DenseCriticalSquare(square);
            pag.add_node(node_type);
        }
        
        // Add dense edges to PAG
        for edge in edges {
            let edge_obj = Edge::new_dense(edge);
            let _ = pag.add_edge(edge_obj);
        }
        
        pag
    }
    
    /// Build PAG from FEN string (for Python integration)
    pub fn build_from_fen(&mut self, fen: &str) -> Result<PAG, String> {
        let board = chess::Board::from_str(fen)
            .map_err(|e| format!("Invalid FEN: {}", e))?;
        let result = self.build_from_board(&board);
        Ok(result)
    }
    
    /// Get statistics about the PAG construction
    pub fn get_construction_stats(&self) -> PAGConstructionStats {
        PAGConstructionStats {
            pieces_processed: 0, // Would track during construction
            squares_analyzed: 0,
            edges_created: 0,
            total_features: 0,
            construction_time_ms: 0.0,
        }
    }
}

/// Statistics about PAG construction
#[derive(Debug, Clone)]
pub struct PAGConstructionStats {
    pub pieces_processed: usize,
    pub squares_analyzed: usize,  
    pub edges_created: usize,
    pub total_features: usize,
    pub construction_time_ms: f64,
}

impl Default for DensePAGBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// Convenience function for quick PAG creation
pub fn create_dense_pag_from_fen(fen: &str) -> Result<PAG, String> {
    let mut builder = DensePAGBuilder::new();
    builder.build_from_fen(fen)
}

pub fn create_dense_pag_from_board(board: &chess::Board) -> PAG {
    let mut builder = DensePAGBuilder::new();
    builder.build_from_board(board)
} 