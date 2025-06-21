use serde::{Serialize, Deserialize};
use super::node::{PieceType, Color};

/// Dense feature vector for tactical analysis
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TacticalFeatures {
    /// Direct attack potential [8 features]
    pub attack_vectors: [f32; 8],  // Different attack types and strengths
    /// Defensive relationships [6 features]
    pub defense_vectors: [f32; 6], // Protection and support patterns
    /// Tactical motifs [10 features]
    pub motif_vectors: [f32; 10],  // Pins, forks, skewers, discoveries, etc.
    /// Immediate threats [8 features]
    pub threat_vectors: [f32; 8],  // Check threats, capture threats, etc.
}

/// Dense feature vector for positional analysis
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PositionalFeatures {
    /// Square control dynamics [12 features]
    pub control_vectors: [f32; 12], // Control strength, contested squares, etc.
    /// Piece coordination [10 features]
    pub coordination_vectors: [f32; 10], // Harmony, support networks
    /// Mobility and space [8 features]
    pub mobility_vectors: [f32; 8], // Movement potential and restrictions
    /// Structural elements [12 features]
    pub structure_vectors: [f32; 12], // Pawn structure, piece placement
}

/// Dense feature vector for strategic analysis
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StrategicFeatures {
    /// Long-term planning [10 features]
    pub planning_vectors: [f32; 10], // Pawn breaks, piece maneuvering
    /// King safety analysis [8 features]
    pub safety_vectors: [f32; 8], // King exposure, shelter quality
    /// Endgame considerations [10 features]
    pub endgame_vectors: [f32; 10], // Piece activity in endings
    /// Positional themes [8 features]
    pub theme_vectors: [f32; 8], // Weak squares, outposts, etc.
}

/// Dense feature vector for meta-analysis
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetaFeatures {
    /// Temporal context [8 features]
    pub temporal_vectors: [f32; 8], // Game phase, move history influence
    /// Evaluation gradients [6 features]
    pub gradient_vectors: [f32; 6], // How evaluation changes with moves
    /// Pattern confidence [6 features]
    pub confidence_vectors: [f32; 6], // Certainty of pattern recognition
    /// Complexity measures [4 features]
    pub complexity_vectors: [f32; 4], // Position complexity indicators
}

/// Geometric and spatial relationship features
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeometricFeatures {
    /// Spatial relationships [8 features]
    pub spatial_vectors: [f32; 8], // Distance, angles, line relationships
    /// Board geometry [6 features]
    pub geometry_vectors: [f32; 6], // Center distance, edge proximity
    /// Ray analysis [8 features]
    pub ray_vectors: [f32; 8], // Line of sight, blocking patterns
}

/// Comprehensive dense edge representation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DenseEdge {
    /// Source and target node IDs
    pub source_id: u64,
    pub target_id: u64,
    
    /// Dense feature vectors totaling ~200 dimensions
    pub tactical: TacticalFeatures,
    pub positional: PositionalFeatures,
    pub strategic: StrategicFeatures,
    pub meta: MetaFeatures,
    pub geometric: GeometricFeatures,
    
    /// Edge strength and importance
    pub importance_weight: f32,
    pub confidence_score: f32,
}

/// Legacy edge types for backward compatibility
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LegacyEdgeType {
    DirectRelation(DirectRelationType),
    Control(ControlType),
    Mobility(MobilityType),
    Cooperation(CooperationType),
    Obstruction(ObstructionType),
    Vulnerability(VulnerabilityType),
    PawnStructure(PawnStructureType),
}

/// New comprehensive edge type system
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EdgeType {
    /// Dense multi-dimensional edge (primary type)
    Dense(DenseEdge),
    /// Legacy edge type (for compatibility)
    Legacy(LegacyEdgeType),
}

/// Main edge structure supporting both dense and legacy formats
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Edge {
    pub edge_type: EdgeType,
    pub weight: f32,
    pub source_id: u64,
    pub target_id: u64,
}

// Legacy types for backward compatibility
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
pub struct CooperationType {
    pub cooperation_strength: f32,
    pub cooperation_type: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObstructionType {
    pub obstruction_strength: f32,
    pub obstruction_type: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VulnerabilityType {
    pub vulnerability_strength: f32,
    pub vulnerability_type: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PawnStructureType {
    pub structure_strength: f32,
    pub structure_type: String,
}

impl Edge {
    /// Create a new dense edge with comprehensive features
    pub fn new_dense(dense_edge: DenseEdge) -> Self {
        let weight = dense_edge.importance_weight;
        let source_id = dense_edge.source_id;
        let target_id = dense_edge.target_id;
        
        Self {
            edge_type: EdgeType::Dense(dense_edge),
            weight,
            source_id,
            target_id,
        }
    }
    
    /// Create a legacy edge for backward compatibility
    pub fn new_legacy(edge_type: LegacyEdgeType, weight: f32, source_id: u64, target_id: u64) -> Self {
        Self {
            edge_type: EdgeType::Legacy(edge_type),
            weight,
            source_id,
            target_id,
        }
    }
    
    /// Get the source node ID
    pub fn source_id(&self) -> u64 {
        self.source_id
    }
    
    /// Get the target node ID
    pub fn target_id(&self) -> u64 {
        self.target_id
    }
    
    /// Get the edge weight
    pub fn weight(&self) -> f32 {
        self.weight
    }
    
    /// Check if this is a dense edge
    pub fn is_dense(&self) -> bool {
        matches!(self.edge_type, EdgeType::Dense(_))
    }
    
    /// Get dense features if available
    pub fn get_dense_features(&self) -> Option<&DenseEdge> {
        match &self.edge_type {
            EdgeType::Dense(dense) => Some(dense),
            EdgeType::Legacy(_) => None,
        }
    }
    
    /// Convert edge to feature vector (flatten all features)
    pub fn to_feature_vector(&self) -> Vec<f32> {
        match &self.edge_type {
            EdgeType::Dense(dense) => {
                let mut features = Vec::with_capacity(256);
                
                // Tactical features (32 dims)
                features.extend_from_slice(&dense.tactical.attack_vectors);
                features.extend_from_slice(&dense.tactical.defense_vectors);
                features.extend_from_slice(&dense.tactical.motif_vectors);
                features.extend_from_slice(&dense.tactical.threat_vectors);
                
                // Positional features (42 dims)
                features.extend_from_slice(&dense.positional.control_vectors);
                features.extend_from_slice(&dense.positional.coordination_vectors);
                features.extend_from_slice(&dense.positional.mobility_vectors);
                features.extend_from_slice(&dense.positional.structure_vectors);
                
                // Strategic features (36 dims)
                features.extend_from_slice(&dense.strategic.planning_vectors);
                features.extend_from_slice(&dense.strategic.safety_vectors);
                features.extend_from_slice(&dense.strategic.endgame_vectors);
                features.extend_from_slice(&dense.strategic.theme_vectors);
                
                // Meta features (24 dims)
                features.extend_from_slice(&dense.meta.temporal_vectors);
                features.extend_from_slice(&dense.meta.gradient_vectors);
                features.extend_from_slice(&dense.meta.confidence_vectors);
                features.extend_from_slice(&dense.meta.complexity_vectors);
                
                // Geometric features (22 dims)
                features.extend_from_slice(&dense.geometric.spatial_vectors);
                features.extend_from_slice(&dense.geometric.geometry_vectors);
                features.extend_from_slice(&dense.geometric.ray_vectors);
                
                // Add importance and confidence
                features.push(dense.importance_weight);
                features.push(dense.confidence_score);
                
                features
            },
            EdgeType::Legacy(_) => {
                // Return basic feature vector for legacy edges
                vec![self.weight, 0.0, 0.0, 0.0] // Minimal features
            }
        }
    }
    
    /// Get the total feature dimension for this edge type
    pub fn feature_dimension(&self) -> usize {
        match &self.edge_type {
            EdgeType::Dense(_) => 158, // 32+42+36+24+22+2
            EdgeType::Legacy(_) => 4,
        }
    }
}

impl DenseEdge {
    /// Create a new dense edge with default zero features
    pub fn new(source_id: u64, target_id: u64) -> Self {
        Self {
            source_id,
            target_id,
            tactical: TacticalFeatures {
                attack_vectors: [0.0; 8],
                defense_vectors: [0.0; 6],
                motif_vectors: [0.0; 10],
                threat_vectors: [0.0; 8],
            },
            positional: PositionalFeatures {
                control_vectors: [0.0; 12],
                coordination_vectors: [0.0; 10],
                mobility_vectors: [0.0; 8],
                structure_vectors: [0.0; 12],
            },
            strategic: StrategicFeatures {
                planning_vectors: [0.0; 10],
                safety_vectors: [0.0; 8],
                endgame_vectors: [0.0; 10],
                theme_vectors: [0.0; 8],
            },
            meta: MetaFeatures {
                temporal_vectors: [0.0; 8],
                gradient_vectors: [0.0; 6],
                confidence_vectors: [0.0; 6],
                complexity_vectors: [0.0; 4],
            },
            geometric: GeometricFeatures {
                spatial_vectors: [0.0; 8],
                geometry_vectors: [0.0; 6],
                ray_vectors: [0.0; 8],
            },
            importance_weight: 1.0,
            confidence_score: 1.0,
        }
    }
    
    /// Get total feature count (158 dimensions)
    pub fn feature_count() -> usize {
        8 + 6 + 10 + 8 +  // tactical: 32
        12 + 10 + 8 + 12 + // positional: 42
        10 + 8 + 10 + 8 +  // strategic: 36
        8 + 6 + 6 + 4 +    // meta: 24
        8 + 6 + 8 +        // geometric: 22
        2                  // importance + confidence: 2
    }
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