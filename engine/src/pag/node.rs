use std::fmt;
use serde::{Serialize, Deserialize};
use super::Coordinate;

/// Represents the type of a chess piece
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PieceType {
    Pawn,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
}

/// Represents the color of a chess piece
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Color {
    White,
    Black,
}

/// Represents the type of node in the PAG
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NodeType {
    Piece(PieceNode),
    CriticalSquare(CriticalSquareNode),
}

/// Common traits for all node types
pub trait Node: fmt::Debug {
    fn get_coordinate(&self) -> Coordinate;
    fn get_id(&self) -> u64;
}

/// Represents a piece on the board
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PieceNode {
    id: u64,
    piece_type: PieceType,
    color: Color,
    coordinate: Coordinate,
    material_value: i32,
    mobility_score: f32,
    is_attacked: bool,
    is_defended: bool,
    is_king_shield: bool,
}

impl PieceNode {
    pub fn new(
        id: u64,
        piece_type: PieceType,
        color: Color,
        coordinate: Coordinate,
        material_value: i32,
    ) -> Self {
        Self {
            id,
            piece_type,
            color,
            coordinate,
            material_value,
            mobility_score: 0.0,
            is_attacked: false,
            is_defended: false,
            is_king_shield: false,
        }
    }

    // Getters
    pub fn piece_type(&self) -> PieceType { self.piece_type }
    pub fn color(&self) -> Color { self.color }
    pub fn mobility_score(&self) -> f32 { self.mobility_score }
    pub fn is_attacked(&self) -> bool { self.is_attacked }
    pub fn is_defended(&self) -> bool { self.is_defended }
    pub fn is_king_shield(&self) -> bool { self.is_king_shield }

    // Setters
    pub fn set_mobility_score(&mut self, score: f32) { self.mobility_score = score; }
    pub fn set_attacked(&mut self, attacked: bool) { self.is_attacked = attacked; }
    pub fn set_defended(&mut self, defended: bool) { self.is_defended = defended; }
    pub fn set_king_shield(&mut self, shield: bool) { self.is_king_shield = shield; }
}

impl Node for PieceNode {
    fn get_coordinate(&self) -> Coordinate { self.coordinate }
    fn get_id(&self) -> u64 { self.id }
}

/// Represents a critical square in the position
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CriticalSquareNode {
    id: u64,
    coordinate: Coordinate,
    control_status: ControlStatus,
    importance_score: f32,
    square_type: CriticalSquareType,
}

/// Represents the control status of a critical square
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ControlStatus {
    white_attackers: u8,
    black_attackers: u8,
    white_defenders: u8,
    black_defenders: u8,
}

/// Represents the type of a critical square
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CriticalSquareType {
    Outpost,
    WeakSquare,
    CentralSquare,
    KingVicinity,
}

impl CriticalSquareNode {
    pub fn new(
        id: u64,
        coordinate: Coordinate,
        square_type: CriticalSquareType,
    ) -> Self {
        Self {
            id,
            coordinate,
            control_status: ControlStatus {
                white_attackers: 0,
                black_attackers: 0,
                white_defenders: 0,
                black_defenders: 0,
            },
            importance_score: 0.0,
            square_type,
        }
    }

    // Getters
    pub fn control_status(&self) -> &ControlStatus { &self.control_status }
    pub fn importance_score(&self) -> f32 { self.importance_score }
    pub fn square_type(&self) -> &CriticalSquareType { &self.square_type }

    // Setters
    pub fn set_control_status(&mut self, status: ControlStatus) { self.control_status = status; }
    pub fn set_importance_score(&mut self, score: f32) { self.importance_score = score; }
}

impl Node for CriticalSquareNode {
    fn get_coordinate(&self) -> Coordinate { self.coordinate }
    fn get_id(&self) -> u64 { self.id }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_piece_node_creation() {
        let coord = Coordinate::new(3, 3);
        let node = PieceNode::new(1, PieceType::Knight, Color::White, coord, 3);
        
        assert_eq!(node.get_id(), 1);
        assert_eq!(node.piece_type(), PieceType::Knight);
        assert_eq!(node.color(), Color::White);
        assert_eq!(node.get_coordinate(), coord);
        assert_eq!(node.mobility_score(), 0.0);
    }

    #[test]
    fn test_critical_square_node_creation() {
        let coord = Coordinate::new(4, 4);
        let node = CriticalSquareNode::new(2, coord, CriticalSquareType::Outpost);
        
        assert_eq!(node.get_id(), 2);
        assert_eq!(node.get_coordinate(), coord);
        assert_eq!(node.importance_score(), 0.0);
        assert!(matches!(node.square_type(), CriticalSquareType::Outpost));
    }
} 