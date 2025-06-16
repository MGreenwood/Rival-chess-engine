pub mod node;
pub mod edge;
pub mod graph;
pub mod builder;

pub use graph::PAG;
pub use node::{Node, NodeType, PieceNode, CriticalSquareNode, PieceType, Color};
pub use edge::{Edge, EdgeType};
pub use builder::PAGBuilder;

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