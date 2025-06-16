use serde::{Serialize, Deserialize};
use super::node::{PieceType, Color};

/// Represents the type of edge in the PAG
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EdgeType {
    /// Direct attack or defense between pieces
    DirectRelation(DirectRelationType),
    
    /// Control or influence over squares
    Control(ControlType),
    
    /// Potential moves for pieces
    Mobility(MobilityType),
    
    /// Cooperative relationships between friendly pieces
    Cooperation(CooperationType),
    
    /// Obstructive relationships between pieces
    Obstruction(ObstructionType),
    
    /// Vulnerability relationships between pieces
    Vulnerability(VulnerabilityType),
    
    /// Pawn structure relationships
    PawnStructure(PawnStructureType),
}

/// Represents a directed edge in the PAG
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Edge {
    edge_type: EdgeType,
    weight: f32,
    source_id: u64,
    target_id: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DirectRelationType {
    Attack {
        attacker_type: PieceType,
        target_type: PieceType,
        strength: f32,
    },
    Defense {
        defender_type: PieceType,
        protected_type: PieceType,
        strength: f32,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ControlType {
    pub controlling_color: Color,
    pub degree: f32,
    pub is_contested: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MobilityType {
    pub move_type: MoveType,
    pub is_legal: bool,
    pub safety_score: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MoveType {
    Normal,
    Capture,
    EnPassant,
    Castle,
    Promotion(PieceType),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CooperationType {
    MutualDefense { strength: f32 },
    Battery { strength: f32 },
    BishopPair,
    KingShield,
    PawnChain,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ObstructionType {
    BlockingPiece { severity: f32 },
    BlockingPawn { severity: f32 },
    RestrictedMobility { severity: f32 },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum VulnerabilityType {
    Pin { severity: f32 },
    Overload { severity: f32 },
    Undefended,
    WeakSquare { severity: f32 },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PawnStructureType {
    Connected,
    Doubled { weakness: f32 },
    Isolated { weakness: f32 },
    Passed { strength: f32 },
    Backward { weakness: f32 },
}

impl Edge {
    pub fn new(
        edge_type: EdgeType,
        weight: f32,
        source_id: u64,
        target_id: u64,
    ) -> Self {
        Self {
            edge_type,
            weight,
            source_id,
            target_id,
        }
    }

    // Getters
    pub fn edge_type(&self) -> &EdgeType { &self.edge_type }
    pub fn weight(&self) -> f32 { self.weight }
    pub fn source_id(&self) -> u64 { self.source_id }
    pub fn target_id(&self) -> u64 { self.target_id }

    // Setters
    pub fn set_weight(&mut self, weight: f32) { self.weight = weight; }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attack_edge_creation() {
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

        assert_eq!(edge.weight(), 1.0);
        assert_eq!(edge.source_id(), 1);
        assert_eq!(edge.target_id(), 2);

        if let EdgeType::DirectRelation(DirectRelationType::Attack { attacker_type, target_type, strength }) = edge.edge_type() {
            assert_eq!(*attacker_type, PieceType::Queen);
            assert_eq!(*target_type, PieceType::Pawn);
            assert_eq!(*strength, 1.0);
        } else {
            panic!("Expected Attack edge type");
        }
    }

    #[test]
    fn test_pawn_structure_edge_creation() {
        let edge = Edge::new(
            EdgeType::PawnStructure(PawnStructureType::Connected),
            1.0,
            1,
            2,
        );

        assert_eq!(edge.weight(), 1.0);
        if let EdgeType::PawnStructure(PawnStructureType::Connected) = edge.edge_type() {
            // Test passed
        } else {
            panic!("Expected Connected PawnStructure edge type");
        }
    }
} 