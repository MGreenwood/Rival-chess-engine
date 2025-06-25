use serde::{Serialize, Deserialize};
use super::Coordinate;

/// Chess piece types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PieceType {
    Pawn = 1,
    Knight = 2,
    Bishop = 3,
    Rook = 4,
    Queen = 5,
    King = 6,
}

/// Chess piece colors
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Color {
    White,
    Black,
}

/// Dense tactical features for pieces
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PieceTacticalFeatures {
    /// Attack capabilities [16 features]
    pub attack_patterns: [f32; 16], // Attack reach, power, multi-target potential
    /// Defense capabilities [12 features]
    pub defense_patterns: [f32; 12], // Protection value, defensive coordination
    /// Tactical motif involvement [20 features]
    pub motif_involvement: [f32; 20], // Pin potential, fork potential, skewer potential, etc.
    /// Threat generation [12 features]
    pub threat_generation: [f32; 12], // Immediate threats, discovered threats, etc.
    /// Vulnerability assessment [16 features]
    pub vulnerability_status: [f32; 16], // Hanging, pinned, overloaded, etc.
}

/// Dense positional features for pieces
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PiecePositionalFeatures {
    /// Mobility analysis [20 features]
    pub mobility_metrics: [f32; 20], // Move count, square quality, restrictions
    /// Square control [16 features]
    pub control_influence: [f32; 16], // Control strength, contested control, etc.
    /// Coordination metrics [18 features]
    pub coordination_status: [f32; 18], // Harmony with other pieces, support networks
    /// Activity measures [14 features]
    pub activity_metrics: [f32; 14], // Centralization, aggressive positioning
    /// Structural significance [12 features]
    pub structural_role: [f32; 12], // Pawn structure support, key square occupation
}

/// Dense strategic features for pieces
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PieceStrategicFeatures {
    /// Long-term potential [16 features]
    pub strategic_potential: [f32; 16], // Piece maneuvering, future plans
    /// Endgame considerations [14 features]
    pub endgame_value: [f32; 14], // Activity in endings, king proximity
    /// King safety contribution [12 features]
    pub safety_contribution: [f32; 12], // King defense, shelter quality
    /// Positional themes [18 features]
    pub thematic_elements: [f32; 18], // Outposts, weak squares, color complexes
}

/// Dense meta features for pieces
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PieceMetaFeatures {
    /// Temporal context [10 features]
    pub temporal_factors: [f32; 10], // Development status, game phase relevance
    /// Evaluation impact [8 features]
    pub evaluation_sensitivity: [f32; 8], // How piece affects position value
    /// Pattern recognition [12 features]
    pub pattern_confidence: [f32; 12], // Certainty of various assessments
    /// Complexity contribution [6 features]
    pub complexity_factors: [f32; 6], // How piece adds to position complexity
}

/// Dense geometric and spatial features
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PieceGeometricFeatures {
    /// Board position analysis [16 features]
    pub positional_geometry: [f32; 16], // Center proximity, edge effects, etc.
    /// Spatial relationships [14 features]
    pub spatial_context: [f32; 14], // Distance to key squares, king, etc.
    /// Ray and line analysis [12 features]
    pub ray_analysis: [f32; 12], // Line control, ray intersections
}

/// Comprehensive dense piece representation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DensePiece {
    /// Basic piece information
    pub piece_type: PieceType,
    pub color: Color,
    pub coordinate: Coordinate,
    pub id: u64,
    
    /// Dense feature vectors (total ~300 dimensions)
    pub tactical: PieceTacticalFeatures,
    pub positional: PiecePositionalFeatures,
    pub strategic: PieceStrategicFeatures,
    pub meta: PieceMetaFeatures,
    pub geometric: PieceGeometricFeatures,
    
    /// Summary metrics
    pub overall_value: f32,
    pub activity_score: f32,
    pub safety_score: f32,
    pub importance_weight: f32,
}

/// Dense features for critical squares
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SquareTacticalFeatures {
    /// Control dynamics [12 features]
    pub control_dynamics: [f32; 12], // Who controls, contest strength, etc.
    /// Tactical significance [10 features]
    pub tactical_importance: [f32; 10], // Attack launching, defensive value
    /// Threat vectors [8 features]
    pub threat_vectors: [f32; 8], // Threats through this square
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SquarePositionalFeatures {
    /// Strategic value [14 features]
    pub strategic_value: [f32; 14], // Outpost potential, weak square status
    /// Mobility impact [10 features]
    pub mobility_impact: [f32; 10], // How square affects piece mobility
    /// Structural significance [12 features]
    pub structural_impact: [f32; 12], // Pawn structure considerations
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SquareMetaFeatures {
    /// Game phase relevance [8 features]
    pub phase_relevance: [f32; 8], // Opening, middlegame, endgame importance
    /// Evaluation sensitivity [6 features]
    pub evaluation_impact: [f32; 6], // How square affects evaluation
    /// Pattern involvement [10 features]
    pub pattern_involvement: [f32; 10], // Part of known patterns
}

/// Comprehensive dense critical square representation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DenseCriticalSquare {
    /// Basic square information
    pub coordinate: Coordinate,
    pub id: u64,
    
    /// Dense feature vectors (total ~90 dimensions)
    pub tactical: SquareTacticalFeatures,
    pub positional: SquarePositionalFeatures,
    pub meta: SquareMetaFeatures,
    
    /// Summary metrics
    pub importance_score: f32,
    pub contest_level: f32,
    pub strategic_value: f32,
}

/// Enhanced node type supporting both dense and legacy formats
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NodeType {
    /// Dense piece representation (primary)
    DensePiece(DensePiece),
    /// Dense critical square representation (primary)
    DenseCriticalSquare(DenseCriticalSquare),
    /// Legacy piece representation (compatibility)
    LegacyPiece(LegacyPieceNode),
    /// Legacy critical square representation (compatibility)
    LegacyCriticalSquare(LegacyCriticalSquareNode),
}

/// Legacy piece node for backward compatibility
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LegacyPieceNode {
    id: u64,
    piece_type: PieceType,
    color: Color,
    coordinate: Coordinate,
    material_value: u8,
    mobility_score: f32,
    is_attacked: bool,
    is_defended: bool,
    is_king_shield: bool,
}

/// Legacy critical square node for backward compatibility
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LegacyCriticalSquareNode {
    id: u64,
    coordinate: Coordinate,
    control_value: f32,
    is_contested: bool,
    strategic_importance: f32,
}

/// Trait for common node operations
pub trait Node {
    fn get_id(&self) -> u64;
    fn get_coordinate(&self) -> Coordinate;
    fn to_feature_vector(&self) -> Vec<f32>;
    fn feature_dimension(&self) -> usize;
}

impl Node for DensePiece {
    fn get_id(&self) -> u64 {
        self.id
    }
    
    fn get_coordinate(&self) -> Coordinate {
        self.coordinate
    }
    
    fn to_feature_vector(&self) -> Vec<f32> {
        let mut features = Vec::with_capacity(308);
        
        // Basic piece info (encoded as one-hot + position)
        let mut piece_info = vec![0.0; 8]; // 6 piece types + 2 colors
        piece_info[self.piece_type as usize - 1] = 1.0;
        if self.color == Color::White { piece_info[6] = 1.0; } else { piece_info[7] = 1.0; }
        features.extend_from_slice(&piece_info);
        
        // Position encoding
        features.push(self.coordinate.rank as f32 / 7.0);
        features.push(self.coordinate.file as f32 / 7.0);
        
        // Tactical features (76 dims)
        features.extend_from_slice(&self.tactical.attack_patterns);
        features.extend_from_slice(&self.tactical.defense_patterns);
        features.extend_from_slice(&self.tactical.motif_involvement);
        features.extend_from_slice(&self.tactical.threat_generation);
        features.extend_from_slice(&self.tactical.vulnerability_status);
        
        // Positional features (80 dims)
        features.extend_from_slice(&self.positional.mobility_metrics);
        features.extend_from_slice(&self.positional.control_influence);
        features.extend_from_slice(&self.positional.coordination_status);
        features.extend_from_slice(&self.positional.activity_metrics);
        features.extend_from_slice(&self.positional.structural_role);
        
        // Strategic features (60 dims)
        features.extend_from_slice(&self.strategic.strategic_potential);
        features.extend_from_slice(&self.strategic.endgame_value);
        features.extend_from_slice(&self.strategic.safety_contribution);
        features.extend_from_slice(&self.strategic.thematic_elements);
        
        // Meta features (36 dims)
        features.extend_from_slice(&self.meta.temporal_factors);
        features.extend_from_slice(&self.meta.evaluation_sensitivity);
        features.extend_from_slice(&self.meta.pattern_confidence);
        features.extend_from_slice(&self.meta.complexity_factors);
        
        // Geometric features (42 dims)
        features.extend_from_slice(&self.geometric.positional_geometry);
        features.extend_from_slice(&self.geometric.spatial_context);
        features.extend_from_slice(&self.geometric.ray_analysis);
        
        // Summary metrics (4 dims)
        features.push(self.overall_value);
        features.push(self.activity_score);
        features.push(self.safety_score);
        features.push(self.importance_weight);
        
        features
    }
    
    fn feature_dimension(&self) -> usize {
        10 + 76 + 80 + 60 + 36 + 42 + 4 // 308 total dimensions
    }
}

impl Node for DenseCriticalSquare {
    fn get_id(&self) -> u64 {
        self.id
    }
    
    fn get_coordinate(&self) -> Coordinate {
        self.coordinate
    }
    
    fn to_feature_vector(&self) -> Vec<f32> {
        let mut features = Vec::with_capacity(100);
        
        // Position encoding
        features.push(self.coordinate.rank as f32 / 7.0);
        features.push(self.coordinate.file as f32 / 7.0);
        
        // Tactical features (30 dims)
        features.extend_from_slice(&self.tactical.control_dynamics);
        features.extend_from_slice(&self.tactical.tactical_importance);
        features.extend_from_slice(&self.tactical.threat_vectors);
        
        // Positional features (36 dims)
        features.extend_from_slice(&self.positional.strategic_value);
        features.extend_from_slice(&self.positional.mobility_impact);
        features.extend_from_slice(&self.positional.structural_impact);
        
        // Meta features (24 dims)
        features.extend_from_slice(&self.meta.phase_relevance);
        features.extend_from_slice(&self.meta.evaluation_impact);
        features.extend_from_slice(&self.meta.pattern_involvement);
        
        // Summary metrics (3 dims)
        features.push(self.importance_score);
        features.push(self.contest_level);
        features.push(self.strategic_value);
        
        features
    }
    
    fn feature_dimension(&self) -> usize {
        2 + 30 + 36 + 24 + 3 // 95 total dimensions
    }
}

impl NodeType {
    /// Get the node ID regardless of type
    pub fn get_id(&self) -> u64 {
        match self {
            NodeType::DensePiece(p) => p.get_id(),
            NodeType::DenseCriticalSquare(s) => s.get_id(),
            NodeType::LegacyPiece(p) => p.get_id(),
            NodeType::LegacyCriticalSquare(s) => s.get_id(),
        }
    }
    
    /// Get the coordinate regardless of type
    pub fn get_coordinate(&self) -> Coordinate {
        match self {
            NodeType::DensePiece(p) => p.get_coordinate(),
            NodeType::DenseCriticalSquare(s) => s.get_coordinate(),
            NodeType::LegacyPiece(p) => p.get_coordinate(),
            NodeType::LegacyCriticalSquare(s) => s.get_coordinate(),
        }
    }
    
    /// Convert to feature vector
    pub fn to_feature_vector(&self) -> Vec<f32> {
        match self {
            NodeType::DensePiece(p) => p.to_feature_vector(),
            NodeType::DenseCriticalSquare(s) => s.to_feature_vector(),
            NodeType::LegacyPiece(p) => p.to_feature_vector(),
            NodeType::LegacyCriticalSquare(s) => s.to_feature_vector(),
        }
    }
    
    /// Get feature dimension
    pub fn feature_dimension(&self) -> usize {
        match self {
            NodeType::DensePiece(p) => p.feature_dimension(),
            NodeType::DenseCriticalSquare(s) => s.feature_dimension(),
            NodeType::LegacyPiece(p) => p.feature_dimension(),
            NodeType::LegacyCriticalSquare(s) => s.feature_dimension(),
        }
    }
    
    /// Check if this is a dense node
    pub fn is_dense(&self) -> bool {
        matches!(self, NodeType::DensePiece(_) | NodeType::DenseCriticalSquare(_))
    }
}

// Legacy implementations for backward compatibility
impl Node for LegacyPieceNode {
    fn get_id(&self) -> u64 { self.id }
    fn get_coordinate(&self) -> Coordinate { self.coordinate }
    fn to_feature_vector(&self) -> Vec<f32> {
        vec![
            self.piece_type as u8 as f32 / 6.0,
            if self.color == Color::White { 1.0 } else { 0.0 },
            self.coordinate.rank as f32 / 7.0,
            self.coordinate.file as f32 / 7.0,
            self.material_value as f32 / 9.0,
            self.mobility_score,
            if self.is_attacked { 1.0 } else { 0.0 },
            if self.is_defended { 1.0 } else { 0.0 },
            if self.is_king_shield { 1.0 } else { 0.0 },
        ]
    }
    fn feature_dimension(&self) -> usize { 9 }
}

impl Node for LegacyCriticalSquareNode {
    fn get_id(&self) -> u64 { self.id }
    fn get_coordinate(&self) -> Coordinate { self.coordinate }
    fn to_feature_vector(&self) -> Vec<f32> {
        vec![
            self.coordinate.rank as f32 / 7.0,
            self.coordinate.file as f32 / 7.0,
            self.control_value,
            if self.is_contested { 1.0 } else { 0.0 },
            self.strategic_importance,
        ]
    }
    fn feature_dimension(&self) -> usize { 5 }
}

impl DensePiece {
    /// Create a new dense piece with default zero features
    pub fn new(id: u64, piece_type: PieceType, color: Color, coordinate: Coordinate) -> Self {
        Self {
            piece_type,
            color,
            coordinate,
            id,
            tactical: PieceTacticalFeatures {
                attack_patterns: [0.0; 16],
                defense_patterns: [0.0; 12],
                motif_involvement: [0.0; 20],
                threat_generation: [0.0; 12],
                vulnerability_status: [0.0; 16],
            },
            positional: PiecePositionalFeatures {
                mobility_metrics: [0.0; 20],
                control_influence: [0.0; 16],
                coordination_status: [0.0; 18],
                activity_metrics: [0.0; 14],
                structural_role: [0.0; 12],
            },
            strategic: PieceStrategicFeatures {
                strategic_potential: [0.0; 16],
                endgame_value: [0.0; 14],
                safety_contribution: [0.0; 12],
                thematic_elements: [0.0; 18],
            },
            meta: PieceMetaFeatures {
                temporal_factors: [0.0; 10],
                evaluation_sensitivity: [0.0; 8],
                pattern_confidence: [0.0; 12],
                complexity_factors: [0.0; 6],
            },
            geometric: PieceGeometricFeatures {
                positional_geometry: [0.0; 16],
                spatial_context: [0.0; 14],
                ray_analysis: [0.0; 12],
            },
            overall_value: 0.0,
            activity_score: 0.0,
            safety_score: 0.0,
            importance_weight: 1.0,
        }
    }
    
    /// Get piece type
    pub fn piece_type(&self) -> PieceType {
        self.piece_type
    }
    
    /// Get piece color
    pub fn color(&self) -> Color {
        self.color
    }
    
    /// Get total feature count (308 dimensions)
    pub fn feature_count() -> usize {
        10 + // basic info + position
        76 + // tactical
        80 + // positional  
        60 + // strategic
        36 + // meta
        42 + // geometric
        4    // summary metrics
    }
}

impl DenseCriticalSquare {
    /// Create a new dense critical square with default zero features
    pub fn new(id: u64, coordinate: Coordinate) -> Self {
        Self {
            coordinate,
            id,
            tactical: SquareTacticalFeatures {
                control_dynamics: [0.0; 12],
                tactical_importance: [0.0; 10],
                threat_vectors: [0.0; 8],
            },
            positional: SquarePositionalFeatures {
                strategic_value: [0.0; 14],
                mobility_impact: [0.0; 10],
                structural_impact: [0.0; 12],
            },
            meta: SquareMetaFeatures {
                phase_relevance: [0.0; 8],
                evaluation_impact: [0.0; 6],
                pattern_involvement: [0.0; 10],
            },
            importance_score: 0.0,
            contest_level: 0.0,
            strategic_value: 0.0,
        }
    }
    
    /// Get total feature count (95 dimensions)
    pub fn feature_count() -> usize {
        2 +  // position
        30 + // tactical
        36 + // positional
        24 + // meta
        3    // summary metrics
    }
}

// Type aliases for backward compatibility
pub type PieceNode = LegacyPieceNode;
pub type CriticalSquareNode = LegacyCriticalSquareNode;

impl LegacyPieceNode {
    pub fn new(id: u64, piece_type: PieceType, color: Color, coordinate: Coordinate, material_value: u8) -> Self {
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
    
    pub fn piece_type(&self) -> PieceType { self.piece_type }
    pub fn color(&self) -> Color { self.color }
}

impl LegacyCriticalSquareNode {
    pub fn new(id: u64, coordinate: Coordinate) -> Self {
        Self {
            id,
            coordinate,
            control_value: 0.0,
            is_contested: false,
            strategic_importance: 0.0,
        }
    }
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