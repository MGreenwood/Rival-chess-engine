use chess::{Board, Color, Piece, Square, ChessMove, MoveGen};
use std::collections::HashMap;
use crate::pag::{
    DensePiece, DenseCriticalSquare, DenseEdge,
    PieceTacticalFeatures, PiecePositionalFeatures, PieceStrategicFeatures, 
    PieceMetaFeatures, PieceGeometricFeatures,
    SquareTacticalFeatures, SquarePositionalFeatures, SquareMetaFeatures,
    // Feature structs for ultra-dense PAG (reserved for future implementation)
    Coordinate
};

/// Comprehensive chess position analyzer
pub struct FeatureExtractor {
    /// Game phase analysis
    game_phase: GamePhase,
    /// Move history for temporal analysis (reserved for future temporal analysis)
    #[allow(dead_code)]
    move_history: Vec<ChessMove>,
    /// Evaluation gradients cache (reserved for future caching)
    #[allow(dead_code)]
    eval_cache: HashMap<u64, f32>,
}

/// Game phase classification
#[derive(Debug, Clone, Copy)]
pub enum GamePhase {
    Opening,
    EarlyMiddlegame,
    Middlegame,
    LateMiddlegame,
    Endgame,
}

impl FeatureExtractor {
    /// Create a new feature extractor
    pub fn new() -> Self {
        Self {
            game_phase: GamePhase::Opening,
            move_history: Vec::new(),
            eval_cache: HashMap::new(),
        }
    }
    
    /// Extract all dense features for a position
    pub fn extract_position_features(&mut self, board: &Board) -> (Vec<DensePiece>, Vec<DenseCriticalSquare>, Vec<DenseEdge>) {
        // Analyze game phase
        self.game_phase = self.analyze_game_phase(board);
        
        // Extract pieces with ultra-dense features
        let pieces = self.extract_piece_features(board);
        
        // Extract critical squares with dense features
        let critical_squares = self.extract_critical_square_features(board);
        
        // Extract dense edges between all elements
        let edges = self.extract_edge_features(board, &pieces, &critical_squares);
        
        (pieces, critical_squares, edges)
    }
    
    /// Extract ultra-dense piece features
    fn extract_piece_features(&self, board: &Board) -> Vec<DensePiece> {
        let mut pieces = Vec::new();
        let mut piece_id = 0;
        
        for square in chess::ALL_SQUARES {
            if let Some(piece) = board.piece_on(square) {
                let coord = Coordinate {
                    rank: square.get_rank().to_index() as u8,
                    file: square.get_file().to_index() as u8,
                };
                
                let mut dense_piece = DensePiece::new(piece_id, 
                    self.convert_piece_type(piece), 
                    self.convert_color(board.color_on(square).unwrap()), 
                    coord);
                
                // Extract all feature categories
                dense_piece.tactical = self.extract_piece_tactical_features(board, square, piece);
                dense_piece.positional = self.extract_piece_positional_features(board, square, piece);
                dense_piece.strategic = self.extract_piece_strategic_features(board, square, piece);
                dense_piece.meta = self.extract_piece_meta_features(board, square, piece);
                dense_piece.geometric = self.extract_piece_geometric_features(board, square, piece);
                
                // Calculate summary metrics
                dense_piece.overall_value = self.calculate_piece_value(board, square, piece);
                dense_piece.activity_score = self.calculate_activity_score(board, square, piece);
                dense_piece.safety_score = self.calculate_safety_score(board, square, piece);
                dense_piece.importance_weight = self.calculate_importance_weight(board, square, piece);
                
                pieces.push(dense_piece);
                piece_id += 1;
            }
        }
        
        pieces
    }
    
    /// Extract piece tactical features (76 dimensions)
    fn extract_piece_tactical_features(&self, board: &Board, square: Square, piece: Piece) -> PieceTacticalFeatures {
        // Simplified but comprehensive implementation
        let attack_power = self.calculate_attack_power(board, square, piece);
        let mobility = self.count_attacked_squares(board, square) / 64.0;
        let threats = self.count_attacked_pieces(board, square) / 16.0;
        
        PieceTacticalFeatures {
            // Attack patterns [16 features] - systematic but simpler
            attack_patterns: [
                mobility,
                attack_power,
                threats,
                self.calculate_basic_fork_potential(board, square, piece),
                self.calculate_basic_pin_potential(board, square, piece),
                if self.can_give_check(board, square, piece) { 1.0 } else { 0.0 },
                self.calculate_material_threat(board, square, piece),
                self.calculate_royal_attack(board, square, piece),
                self.get_piece_value_normalized(piece),
                self.calculate_centralization(square),
                self.calculate_development_score(board, square, piece),
                self.calculate_tempo_advantage(board, square, piece),
                self.calculate_initiative_score(board, square, piece),
                self.calculate_pressure_score(board, square, piece),
                self.calculate_breakthrough_potential(board, square, piece),
                self.calculate_conversion_potential(board, square, piece),
            ],
            
            // Defense patterns [12 features]
            defense_patterns: [
                self.count_defended_pieces(board, square) / 16.0,
                self.calculate_defense_power(board, square, piece),
                self.calculate_king_defense(board, square, piece),
                self.calculate_piece_protection(board, square),
                self.calculate_counter_play(board, square, piece),
                self.calculate_blockade_value(board, square, piece),
                self.calculate_interference_value(board, square, piece),
                self.calculate_sacrifice_potential(board, square, piece),
                self.calculate_escape_routes(board, square),
                self.calculate_support_network(board, square, piece),
                self.calculate_coordination_defense(board, square, piece),
                self.calculate_positional_defense(board, square, piece),
            ],
            
            // Motif involvement [20 features] - comprehensive tactical patterns
            motif_involvement: [
                self.calculate_pin_score(board, square, piece),
                self.calculate_skewer_score(board, square, piece),
                self.calculate_fork_score(board, square, piece),
                self.calculate_discovery_score(board, square, piece),
                self.calculate_deflection_score(board, square, piece),
                self.calculate_decoy_score(board, square, piece),
                self.calculate_clearance_score(board, square, piece),
                self.calculate_interference_score(board, square, piece),
                self.calculate_overload_score(board, square, piece),
                self.calculate_zugzwang_score(board, square, piece),
                self.calculate_battery_score(board, square, piece),
                self.calculate_xray_score(board, square, piece),
                self.calculate_windmill_score(board, square, piece),
                self.calculate_mating_net_score(board, square, piece),
                self.calculate_back_rank_score(board, square, piece),
                self.calculate_promotion_score(board, square, piece),
                self.calculate_en_passant_score(board, square, piece),
                self.calculate_castling_score(board, square, piece),
                self.calculate_stalemate_score(board, square, piece),
                self.calculate_perpetual_score(board, square, piece),
            ],
            
            // Threat generation [12 features]
            threat_generation: [
                self.calculate_immediate_threat_level(board, square, piece),
                self.calculate_next_move_threats(board, square, piece),
                self.calculate_discovered_threat_potential(board, square, piece),
                self.calculate_tempo_threats(board, square, piece),
                self.calculate_positional_threats(board, square, piece),
                self.calculate_long_term_threats(board, square, piece),
                self.calculate_multi_piece_coordination(board, square, piece),
                self.calculate_combination_potential(board, square, piece),
                self.calculate_sacrifice_threats(board, square, piece),
                self.calculate_endgame_threats(board, square, piece),
                self.calculate_king_hunt_score(board, square, piece),
                self.calculate_pawn_storm_score(board, square, piece),
            ],
            
            // Vulnerability status [16 features]
            vulnerability_status: [
                if self.is_hanging(board, square) { 1.0 } else { 0.0 },
                if self.is_pinned(board, square) { 1.0 } else { 0.0 },
                if self.is_overloaded(board, square) { 1.0 } else { 0.0 },
                self.calculate_attack_defense_ratio(board, square),
                self.count_escape_squares(board, square) / 8.0,
                self.count_defenders(board, square) / 8.0,
                self.count_attackers(board, square) / 8.0,
                self.calculate_net_safety(board, square, piece),
                self.calculate_removal_impact(board, square, piece),
                self.calculate_deflection_risk(board, square, piece),
                self.calculate_overload_risk(board, square, piece),
                self.calculate_fork_risk(board, square, piece),
                self.calculate_pin_risk(board, square, piece),
                self.calculate_skewer_risk(board, square, piece),
                self.calculate_discovery_risk(board, square, piece),
                self.calculate_overall_weakness(board, square, piece),
            ],
        }
    }
    
    /// Extract piece positional features (80 dimensions)
    fn extract_piece_positional_features(&self, board: &Board, square: Square, piece: Piece) -> PiecePositionalFeatures {
        // Comprehensive positional analysis
        PiecePositionalFeatures {
            // Mobility metrics [20 features]
            mobility_metrics: [
                self.legal_move_count(board, square) / 28.0, // Max queen moves
                self.safe_move_count(board, square) / 28.0,
                self.attacking_move_count(board, square) / 28.0,
                self.calculate_mobility_quality(board, square, piece),
                self.calculate_future_mobility(board, square, piece),
                self.calculate_restricted_mobility(board, square, piece),
                self.calculate_improvement_potential(board, square, piece),
                self.calculate_piece_specific_mobility(board, square, piece),
                self.calculate_dynamic_potential(board, square, piece),
                self.calculate_tactical_mobility_score(board, square, piece),
                self.calculate_positional_mobility_score(board, square, piece),
                self.calculate_endgame_mobility_score(board, square, piece),
                self.calculate_king_distance_factor(board, square, piece),
                self.calculate_center_access(board, square, piece),
                self.calculate_enemy_territory_access(board, square, piece),
                self.calculate_key_square_access(board, square, piece),
                self.calculate_file_control(board, square, piece),
                self.calculate_rank_control(board, square, piece),
                self.calculate_diagonal_control(board, square, piece),
                self.calculate_knight_outpost_score(board, square, piece),
            ],
            
            // Control influence [16 features]
            control_influence: [
                self.calculate_square_control_strength(board, square, piece),
                self.calculate_contested_squares(board, square, piece),
                self.calculate_exclusive_control(board, square, piece),
                self.calculate_weak_square_control(board, square, piece),
                self.calculate_key_square_influence(board, square, piece),
                self.calculate_central_control(board, square, piece),
                self.calculate_flank_control(board, square, piece),
                self.calculate_enemy_king_zone_pressure(board, square, piece),
                self.calculate_pawn_break_support(board, square, piece),
                self.calculate_piece_placement_control(board, square, piece),
                self.calculate_route_control(board, square, piece),
                self.calculate_space_advantage(board, square, piece),
                self.calculate_territorial_gain(board, square, piece),
                self.calculate_influence_projection(board, square, piece),
                self.calculate_dominance_factor(board, square, piece),
                self.calculate_control_sustainability(board, square, piece),
            ],
            
            coordination_status: [0.0; 18], // Simplified for now
            activity_metrics: [0.0; 14],     // Simplified for now
            structural_role: [0.0; 12],      // Simplified for now
        }
    }
    
    fn extract_piece_strategic_features(&self, _board: &Board, _square: Square, _piece: Piece) -> PieceStrategicFeatures {
        // Placeholder - would implement comprehensive strategic analysis
        PieceStrategicFeatures {
            strategic_potential: [0.0; 16],
            endgame_value: [0.0; 14],
            safety_contribution: [0.0; 12],
            thematic_elements: [0.0; 18],
        }
    }
    
    fn extract_piece_meta_features(&self, _board: &Board, _square: Square, _piece: Piece) -> PieceMetaFeatures {
        // Placeholder - would implement meta analysis
        PieceMetaFeatures {
            temporal_factors: [0.0; 10],
            evaluation_sensitivity: [0.0; 8],
            pattern_confidence: [0.0; 12],
            complexity_factors: [0.0; 6],
        }
    }
    
    fn extract_piece_geometric_features(&self, _board: &Board, _square: Square, _piece: Piece) -> PieceGeometricFeatures {
        // Placeholder - would implement geometric analysis
        PieceGeometricFeatures {
            positional_geometry: [0.0; 16],
            spatial_context: [0.0; 14],
            ray_analysis: [0.0; 12],
        }
    }
    
    fn extract_critical_square_features(&self, board: &Board) -> Vec<DenseCriticalSquare> {
        // Identify and analyze critical squares
        let mut squares = Vec::new();
        let mut square_id = 1000;
        
        // Center squares are always critical
        let center_squares = [
            Square::D4, Square::D5, Square::E4, Square::E5
        ];
        
        for &square in &center_squares {
            let coord = Coordinate {
                rank: square.get_rank().to_index() as u8,
                file: square.get_file().to_index() as u8,
            };
            
            let mut dense_square = DenseCriticalSquare::new(square_id, coord);
            dense_square.tactical = self.analyze_square_tactical(board, square);
            dense_square.positional = self.analyze_square_positional(board, square);
            dense_square.meta = self.analyze_square_meta(board, square);
            dense_square.importance_score = self.calculate_square_importance_score(board, square);
            dense_square.contest_level = self.calculate_contest_level_score(board, square);
            dense_square.strategic_value = self.calculate_strategic_value_score(board, square);
            
            squares.push(dense_square);
            square_id += 1;
        }
        
        squares
    }
    
    fn extract_edge_features(&self, _board: &Board, pieces: &[DensePiece], _squares: &[DenseCriticalSquare]) -> Vec<DenseEdge> {
        let mut edges = Vec::new();
        
        // Create piece-to-piece edges (simplified)
        for (i, piece1) in pieces.iter().enumerate() {
            for (j, piece2) in pieces.iter().enumerate() {
                if i != j {
                    let mut edge = DenseEdge::new(piece1.id, piece2.id);
                    
                    // Basic edge features
                    edge.importance_weight = 0.5;
                    edge.confidence_score = 0.8;
                    
                    edges.push(edge);
                }
            }
        }
        
        edges
    }
    
    // Utility methods with actual implementations
    fn count_attacked_squares(&self, board: &Board, square: Square) -> f32 {
        let mut count = 0;
        for target_square in chess::ALL_SQUARES {
            if self.piece_attacks_square(board, square, target_square) {
                count += 1;
            }
        }
        count as f32
    }
    
    fn piece_attacks_square(&self, board: &Board, from: Square, to: Square) -> bool {
        if let Some(_piece) = board.piece_on(from) {
            let mut moves = MoveGen::new_legal(board);
            moves.any(|m| m.get_source() == from && m.get_dest() == to)
        } else {
            false
        }
    }
    
    fn calculate_attack_power(&self, _board: &Board, _square: Square, piece: Piece) -> f32 {
        match piece {
            Piece::Pawn => 0.1,
            Piece::Knight => 0.3,
            Piece::Bishop => 0.35,
            Piece::Rook => 0.5,
            Piece::Queen => 0.9,
            Piece::King => 0.1,
        }
    }
    
    fn count_attacked_pieces(&self, board: &Board, square: Square) -> f32 {
        let mut count = 0;
        if let Some(color) = board.color_on(square) {
            let enemy_color = !color;
            
            for target_square in chess::ALL_SQUARES {
                if let Some(_target_piece) = board.piece_on(target_square) {
                    if board.color_on(target_square).unwrap() == enemy_color {
                        if self.piece_attacks_square(board, square, target_square) {
                            count += 1;
                        }
                    }
                }
            }
        }
        count as f32
    }
    
    // Game phase analysis
    fn analyze_game_phase(&self, board: &Board) -> GamePhase {
        let material_count = self.count_total_material(board);
        
        match material_count {
            0..=15 => GamePhase::Endgame,
            16..=25 => GamePhase::LateMiddlegame,
            26..=35 => GamePhase::Middlegame,
            36..=45 => GamePhase::EarlyMiddlegame,
            _ => GamePhase::Opening,
        }
    }
    
    fn count_total_material(&self, board: &Board) -> u32 {
        let mut total = 0;
        for square in chess::ALL_SQUARES {
            if let Some(piece) = board.piece_on(square) {
                total += match piece {
                    Piece::Pawn => 1,
                    Piece::Knight | Piece::Bishop => 3,
                    Piece::Rook => 5,
                    Piece::Queen => 9,
                    Piece::King => 0,
                };
            }
        }
        total
    }
    
    // Utility conversion methods
    fn convert_piece_type(&self, piece: Piece) -> crate::pag::PieceType {
        match piece {
            Piece::Pawn => crate::pag::PieceType::Pawn,
            Piece::Knight => crate::pag::PieceType::Knight,
            Piece::Bishop => crate::pag::PieceType::Bishop,
            Piece::Rook => crate::pag::PieceType::Rook,
            Piece::Queen => crate::pag::PieceType::Queen,
            Piece::King => crate::pag::PieceType::King,
        }
    }
    
    fn convert_color(&self, color: Color) -> crate::pag::Color {
        match color {
            Color::White => crate::pag::Color::White,
            Color::Black => crate::pag::Color::Black,
        }
    }
    
    // Comprehensive feature calculation methods (simplified implementations)
    fn calculate_basic_fork_potential(&self, board: &Board, square: Square, piece: Piece) -> f32 {
        // Simplified version of fork calculation for basic features
        match piece {
            Piece::Knight => self.calculate_knight_fork_potential(board, square, !board.color_on(square).unwrap()) * 0.1,
            Piece::Pawn => {
                let piece_color = board.color_on(square).unwrap();
                self.calculate_pawn_fork_potential(board, square, piece_color, !piece_color) * 0.1
            },
            _ => 0.0
        }
    }
    
    fn calculate_basic_pin_potential(&self, board: &Board, square: Square, piece: Piece) -> f32 {
        // Simplified pin potential calculation
        match piece {
            Piece::Queen | Piece::Rook | Piece::Bishop => {
                let Some(piece_color) = board.color_on(square) else { return 0.0; };
                let enemy_color = !piece_color;
                let mut pin_potential = 0.0;
                
                // Look for potential pins in the immediate vicinity
                for target_square in chess::ALL_SQUARES {
                    if let Some(_target_piece) = board.piece_on(target_square) {
                        if board.color_on(target_square) == Some(enemy_color) {
                            if let Some(pin_value) = self.check_pin_potential(board, square, target_square, piece, enemy_color) {
                                pin_potential += pin_value;
                            }
                        }
                    }
                }
                
                (pin_potential * 0.1).min(1.0)
            },
            _ => 0.0
        }
    }
    
    fn can_give_check(&self, board: &Board, square: Square, piece: Piece) -> bool {
        let Some(piece_color) = board.color_on(square) else { return false; };
        let enemy_color = !piece_color;
        
        // Find enemy king
        let Some(enemy_king_square) = self.find_king(board, enemy_color) else { return false; };
        
        // Check if this piece can attack the enemy king
        match piece {
            Piece::Queen => {
                // Queen can give check on ranks, files, and diagonals
                self.can_attack_along_line(square, enemy_king_square, true, true)
            },
            Piece::Rook => {
                // Rook can give check on ranks and files
                self.can_attack_along_line(square, enemy_king_square, true, false)
            },
            Piece::Bishop => {
                // Bishop can give check on diagonals
                self.can_attack_along_line(square, enemy_king_square, false, true)
            },
            Piece::Knight => {
                // Knight can give check with L-shaped moves
                self.can_knight_attack(square, enemy_king_square)
            },
            Piece::Pawn => {
                // Pawn can give check diagonally
                self.can_pawn_attack(square, enemy_king_square, piece_color)
            },
            Piece::King => {
                // King can give check adjacently (rare but possible)
                self.can_king_attack(square, enemy_king_square)
            }
        }
    }
    
    // Additional helper methods for compilation
    fn calculate_material_threat(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.0 }
    fn calculate_royal_attack(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.0 }
    fn get_piece_value_normalized(&self, piece: Piece) -> f32 {
        match piece {
            Piece::Pawn => 0.1,
            Piece::Knight | Piece::Bishop => 0.3,
            Piece::Rook => 0.5,
            Piece::Queen => 0.9,
            Piece::King => 1.0,
        }
    }
    
    fn calculate_centralization(&self, square: Square) -> f32 {
        let rank = square.get_rank().to_index() as f32;
        let file = square.get_file().to_index() as f32;
        let center_distance = ((rank - 3.5).abs() + (file - 3.5).abs()) / 7.0;
        1.0 - center_distance
    }
    
    // Placeholder implementations for all called methods
    fn calculate_development_score(&self, board: &Board, square: Square, piece: Piece) -> f32 {
        let Some(piece_color) = board.color_on(square) else { return 0.0; };
        let mut development_score;
        
        match piece {
            Piece::Knight => {
                // Knights develop from back rank to center/active squares
                let rank = square.get_rank().to_index();
                let file = square.get_file().to_index();
                
                // Back rank knights are undeveloped
                let back_rank = if piece_color == Color::White { 0 } else { 7 };
                if rank == back_rank {
                    development_score = 0.0; // Undeveloped
                } else {
                    // Developed - bonus for central squares
                    development_score = 0.6;
                    if (3..=4).contains(&rank) && (2..=5).contains(&file) {
                        development_score += 0.3; // Central development bonus
                    }
                }
            },
            Piece::Bishop => {
                // Bishops develop from back rank, prefer long diagonals
                let rank = square.get_rank().to_index();
                let back_rank = if piece_color == Color::White { 0 } else { 7 };
                
                if rank == back_rank {
                    development_score = 0.0; // Undeveloped
                } else {
                    development_score = 0.5;
                    // Bonus for long diagonal control
                    let diagonal_length = self.calculate_diagonal_control_length(board, square);
                    development_score += diagonal_length * 0.1;
                }
            },
            Piece::Queen => {
                // Queen should not develop too early
                let rank = square.get_rank().to_index();
                let back_rank = if piece_color == Color::White { 0 } else { 7 };
                
                match self.game_phase {
                    GamePhase::Opening => {
                        if rank == back_rank {
                            development_score = 0.8; // Good - queen stays home in opening
                        } else {
                            development_score = 0.2; // Bad - early queen development
                        }
                    },
                    _ => {
                        // In middlegame+, active queen is good
                        development_score = 0.6 + self.calculate_activity_score(board, square, piece);
                    }
                }
            },
            Piece::Rook => {
                // Rooks develop to open files and active ranks
                let file_openness = self.calculate_file_openness(board, square);
                development_score = 0.3 + file_openness * 0.4;
                
                // Bonus for 7th/2nd rank
                let rank = square.get_rank().to_index();
                if (piece_color == Color::White && rank == 6) || 
                   (piece_color == Color::Black && rank == 1) {
                    development_score += 0.3; // 7th rank bonus
                }
            },
            Piece::King => {
                // King safety in opening is crucial
                match self.game_phase {
                    GamePhase::Opening | GamePhase::EarlyMiddlegame => {
                        if self.is_king_castled(board, square, piece_color) {
                            development_score = 0.9; // Castled king is well-developed
                        } else {
                            development_score = 0.2; // Uncastled king is dangerous
                        }
                    },
                    GamePhase::Endgame => {
                        // In endgame, active king is good
                        development_score = self.calculate_king_activity_endgame(board, square);
                    },
                    _ => {
                        development_score = 0.5; // Middlegame neutral
                    }
                }
            },
            Piece::Pawn => {
                // Pawn development: central pawns, pawn chains
                development_score = self.calculate_pawn_development_score(board, square, piece_color);
            }
        }
        
        development_score.min(1.0)
    }
    
    fn calculate_tempo_advantage(&self, board: &Board, square: Square, piece: Piece) -> f32 {
        let Some(piece_color) = board.color_on(square) else { return 0.0; };
        let enemy_color = !piece_color;
        let mut tempo_score = 0.0;
        
        // Tempo is about forcing opponent to respond to threats
        match piece {
            Piece::Queen | Piece::Rook => {
                // Check if piece is creating immediate threats
                let immediate_threats = self.count_immediate_threats(board, square, piece);
                tempo_score += immediate_threats * 0.3;
                
                // Check if piece is attacking undefended enemies
                for target_square in chess::ALL_SQUARES {
                    if let Some(target_piece) = board.piece_on(target_square) {
                        if board.color_on(target_square) == Some(enemy_color) {
                            if self.piece_attacks_square(board, square, target_square) && 
                               self.count_defenders(board, target_square) == 0.0 {
                                tempo_score += self.get_piece_value_normalized(target_piece) * 0.4;
                            }
                        }
                    }
                }
            },
            Piece::Knight | Piece::Bishop => {
                // Minor pieces gain tempo through checks and attacks
                if self.can_give_check(board, square, piece) {
                    tempo_score += 0.6; // Check gives tempo
                }
                
                // Count enemy pieces under attack
                let attacks = self.count_attacked_pieces(board, square);
                tempo_score += attacks * 0.2;
            },
            Piece::Pawn => {
                // Pawn advances can gain tempo
                if self.is_pawn_advance_with_tempo(board, square, piece_color) {
                    tempo_score += 0.4;
                }
            },
            _ => {}
        }
        
        tempo_score.min(1.0)
    }
    
    fn calculate_initiative_score(&self, board: &Board, square: Square, piece: Piece) -> f32 {
        let Some(piece_color) = board.color_on(square) else { return 0.0; };
        let mut initiative_score = 0.0;
        
        // Initiative means controlling the game's direction
        
        // 1. Piece activity and mobility
        let mobility_factor = self.legal_move_count(board, square) / 20.0; // Normalize by reasonable max
        initiative_score += mobility_factor * 0.3;
        
        // 2. Central control
        let central_control = self.calculate_central_control(board, square, piece);
        initiative_score += central_control * 0.4;
        
        // 3. Attacking potential
        let attack_potential = self.count_attacked_pieces(board, square) / 8.0; // Normalize
        initiative_score += attack_potential * 0.3;
        
        // 4. King pressure
        if let Some(enemy_king_square) = self.find_king(board, !piece_color) {
            let king_distance = self.calculate_square_distance(square, enemy_king_square);
            let king_pressure = (8.0 - king_distance as f32) / 8.0;
            initiative_score += king_pressure * 0.2;
        }
        
        initiative_score.min(1.0)
    }
    
    fn calculate_pressure_score(&self, board: &Board, square: Square, piece: Piece) -> f32 {
        let Some(piece_color) = board.color_on(square) else { return 0.0; };
        let enemy_color = !piece_color;
        let mut pressure_score = 0.0;
        
        // Pressure is about creating problems for the opponent
        
        // 1. Direct piece attacks
        for target_square in chess::ALL_SQUARES {
            if let Some(target_piece) = board.piece_on(target_square) {
                if board.color_on(target_square) == Some(enemy_color) {
                    if self.piece_attacks_square(board, square, target_square) {
                        let target_value = self.get_piece_value_normalized(target_piece);
                        let our_value = self.get_piece_value_normalized(piece);
                        
                        // More pressure for attacking higher value pieces
                        if target_value >= our_value {
                            pressure_score += target_value * 0.5;
                        } else {
                            pressure_score += target_value * 0.3;
                        }
                    }
                }
            }
        }
        
        // 2. Constraining enemy pieces
        let constrained_pieces = self.count_constrained_enemy_pieces(board, square, piece_color);
        pressure_score += constrained_pieces * 0.2;
        
        // 3. King zone pressure
        if let Some(enemy_king_square) = self.find_king(board, enemy_color) {
            if self.attacks_king_zone(board, square, piece, enemy_king_square) {
                pressure_score += 0.4;
            }
        }
        
        (pressure_score / 3.0).min(1.0)
    }
    
    fn calculate_breakthrough_potential(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.0 }
    fn calculate_conversion_potential(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.0 }
    
    fn count_defended_pieces(&self, _board: &Board, _square: Square) -> f32 { 0.0 }
    fn calculate_defense_power(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.0 }
    fn calculate_king_defense(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.0 }
    fn calculate_piece_protection(&self, _board: &Board, _square: Square) -> f32 { 0.0 }
    fn calculate_counter_play(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.0 }
    fn calculate_blockade_value(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.0 }
    fn calculate_interference_value(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.0 }
    fn calculate_sacrifice_potential(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.0 }
    fn calculate_escape_routes(&self, _board: &Board, _square: Square) -> f32 { 0.0 }
    fn calculate_support_network(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.0 }
    fn calculate_coordination_defense(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.0 }
    fn calculate_positional_defense(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.0 }
    
    // Tactical motif calculations
    fn calculate_pin_score(&self, board: &Board, square: Square, piece: Piece) -> f32 {
        let Some(piece_color) = board.color_on(square) else { return 0.0; };
        let enemy_color = !piece_color;
        let mut pin_score = 0.0;
        
        // Check if this piece can create pins against enemy pieces
        match piece {
            Piece::Queen | Piece::Rook | Piece::Bishop => {
                // Look for enemy pieces that this piece can pin
                for target_square in chess::ALL_SQUARES {
                    if let Some(_target_piece) = board.piece_on(target_square) {
                        if board.color_on(target_square) == Some(enemy_color) {
                            // Check if there's a more valuable piece behind this target
                            if let Some(pinned_value) = self.check_pin_potential(board, square, target_square, piece, enemy_color) {
                                pin_score += pinned_value;
                            }
                        }
                    }
                }
            },
            _ => {}
        }
        
        // Normalize the score
        (pin_score / 10.0).min(1.0)
    }
    
    fn calculate_skewer_score(&self, board: &Board, square: Square, piece: Piece) -> f32 {
        let Some(piece_color) = board.color_on(square) else { return 0.0; };
        let enemy_color = !piece_color;
        let mut skewer_score = 0.0;
        
        // Check if this piece can create skewers
        match piece {
            Piece::Queen | Piece::Rook | Piece::Bishop => {
                // Look for enemy pieces that can be skewered (high value piece in front of lower value)
                for target_square in chess::ALL_SQUARES {
                    if let Some(_front_piece) = board.piece_on(target_square) {
                        if board.color_on(target_square) == Some(enemy_color) {
                            if let Some(skewer_value) = self.check_skewer_potential(board, square, target_square, piece, enemy_color) {
                                skewer_score += skewer_value;
                            }
                        }
                    }
                }
            },
            _ => {}
        }
        
        (skewer_score / 10.0).min(1.0)
    }
    
    fn calculate_fork_score(&self, board: &Board, square: Square, piece: Piece) -> f32 {
        let Some(piece_color) = board.color_on(square) else { return 0.0; };
        let enemy_color = !piece_color;
        let mut fork_score = 0.0;
        
        match piece {
            Piece::Knight => {
                // Knight forks are the most common and dangerous
                fork_score += self.calculate_knight_fork_potential(board, square, enemy_color);
            },
            Piece::Pawn => {
                // Pawn forks are also important
                fork_score += self.calculate_pawn_fork_potential(board, square, piece_color, enemy_color);
            },
            Piece::Queen | Piece::Rook | Piece::Bishop => {
                // Sliding piece forks (less common but still valuable)
                fork_score += self.calculate_sliding_fork_potential(board, square, piece, enemy_color);
            },
            _ => {}
        }
        
        (fork_score / 5.0).min(1.0)
    }
    
    fn calculate_discovery_score(&self, board: &Board, square: Square, _piece: Piece) -> f32 {
        let Some(piece_color) = board.color_on(square) else { return 0.0; };
        let mut discovery_score = 0.0;
        
        // Check if moving this piece would discover an attack from a piece behind it
        for friendly_square in chess::ALL_SQUARES {
            if let Some(friendly_piece) = board.piece_on(friendly_square) {
                if board.color_on(friendly_square) == Some(piece_color) && friendly_square != square {
                    match friendly_piece {
                        Piece::Queen | Piece::Rook | Piece::Bishop => {
                            if self.is_on_line_between(friendly_square, square, self.find_enemy_targets(board, !piece_color)) {
                                discovery_score += self.get_piece_value_normalized(friendly_piece);
                            }
                        },
                        _ => {}
                    }
                }
            }
        }
        
        (discovery_score / 3.0).min(1.0)
    }
    fn calculate_deflection_score(&self, board: &Board, square: Square, _piece: Piece) -> f32 {
        let Some(piece_color) = board.color_on(square) else { return 0.0; };
        let enemy_color = !piece_color;
        let mut deflection_score = 0.0;
        
        // Check if this piece can deflect enemy pieces from important duties
        for target_square in chess::ALL_SQUARES {
            if let Some(_target_piece) = board.piece_on(target_square) {
                if board.color_on(target_square) == Some(enemy_color) {
                    // Check if target is defending something important
                    let defended_value = self.calculate_defended_pieces_value(board, target_square);
                    
                    // If we can attack this defender, deflection is possible
                    if self.piece_attacks_square(board, square, target_square) && defended_value > 0.0 {
                        deflection_score += defended_value * 0.3; // Deflection bonus
                    }
                }
            }
        }
        
        (deflection_score / 5.0).min(1.0)
    }
    
    fn calculate_decoy_score(&self, board: &Board, square: Square, piece: Piece) -> f32 {
        let Some(piece_color) = board.color_on(square) else { return 0.0; };
        let enemy_color = !piece_color;
        let mut decoy_score: f32 = 0.0;
        
        // Look for enemy pieces that could be decoyed to worse squares
        match piece {
            Piece::Queen | Piece::Rook => {
                // These pieces can often decoy enemy pieces onto disadvantageous squares
                let enemy_king_square = self.find_king(board, enemy_color);
                if let Some(king_square) = enemy_king_square {
                    // Check if we can force the king to a worse square
                    if self.can_attack_along_line(square, king_square, true, true) {
                        decoy_score += 0.8; // High decoy potential against king
                    }
                }
            },
            _ => {}
        }
        
        decoy_score.min(1.0)
    }
    
    fn calculate_clearance_score(&self, board: &Board, square: Square, _piece: Piece) -> f32 {
        let Some(piece_color) = board.color_on(square) else { return 0.0; };
        let mut clearance_score = 0.0;
        
        // Check if moving this piece would clear lines for other pieces
        for friendly_square in chess::ALL_SQUARES {
            if let Some(friendly_piece) = board.piece_on(friendly_square) {
                if board.color_on(friendly_square) == Some(piece_color) && friendly_square != square {
                    match friendly_piece {
                        Piece::Queen | Piece::Rook | Piece::Bishop => {
                            // Check if current piece is blocking a potential attack line
                            if self.is_blocking_attack_line(board, square, friendly_square, friendly_piece) {
                                clearance_score += self.get_piece_value_normalized(friendly_piece) * 0.4;
                            }
                        },
                        _ => {}
                    }
                }
            }
        }
        
        (clearance_score / 2.0).min(1.0)
    }
    
    fn calculate_interference_score(&self, board: &Board, square: Square, _piece: Piece) -> f32 {
        let Some(piece_color) = board.color_on(square) else { return 0.0; };
        let enemy_color = !piece_color;
        let mut interference_score: f32 = 0.0;
        
        // Check if this piece can interfere with enemy piece coordination
        for enemy_square in chess::ALL_SQUARES {
            if let Some(enemy_piece) = board.piece_on(enemy_square) {
                if board.color_on(enemy_square) == Some(enemy_color) {
                    match enemy_piece {
                        Piece::Queen | Piece::Rook | Piece::Bishop => {
                            // Check if we can block important enemy lines
                            if self.can_interfere_with_line(board, square, enemy_square, enemy_piece) {
                                interference_score += 0.5;
                            }
                        },
                        _ => {}
                    }
                }
            }
        }
        
        (interference_score / 3.0).min(1.0)
    }
    
    fn calculate_overload_score(&self, board: &Board, square: Square, _piece: Piece) -> f32 {
        // This piece's contribution to overloading enemy pieces
        let Some(piece_color) = board.color_on(square) else { return 0.0; };
        let enemy_color = !piece_color;
        let mut overload_score = 0.0;
        
        // Check how many enemy pieces we're pressuring that are already defending multiple targets
        for target_square in chess::ALL_SQUARES {
            if let Some(_target_piece) = board.piece_on(target_square) {
                if board.color_on(target_square) == Some(enemy_color) {
                    if self.piece_attacks_square(board, square, target_square) {
                        // Check if target is already overloaded
                        if self.is_overloaded(board, target_square) {
                            overload_score += 0.6; // Bonus for pressuring overloaded pieces
                        }
                        
                        // Check if target is defending multiple pieces
                        let defended_count = self.count_pieces_defended_by(board, target_square);
                        if defended_count >= 2 {
                            overload_score += defended_count as f32 * 0.2;
                        }
                    }
                }
            }
        }
        
        (overload_score / 4.0).min(1.0)
    }
    
    fn calculate_zugzwang_score(&self, board: &Board, square: Square, _piece: Piece) -> f32 {
        // Simplified zugzwang detection - mostly relevant in endgames
        match self.game_phase {
            GamePhase::Endgame => {
                let Some(piece_color) = board.color_on(square) else { return 0.0; };
                let enemy_color = !piece_color;
                
                // In endgames, check if enemy king is in a constrained position
                if let Some(enemy_king_square) = self.find_king(board, enemy_color) {
                    let enemy_king_mobility = self.count_king_legal_moves(board, enemy_king_square);
                    
                    // If enemy king has very limited mobility and we're applying pressure
                    if enemy_king_mobility <= 2 && self.piece_attacks_square(board, square, enemy_king_square) {
                        return 0.7; // Potential zugzwang situation
                    }
                }
                
                0.0
            },
            _ => 0.0 // Zugzwang is primarily an endgame concept
        }
    }
    
    fn calculate_battery_score(&self, board: &Board, square: Square, piece: Piece) -> f32 {
        let Some(piece_color) = board.color_on(square) else { return 0.0; };
        let mut battery_score: f32 = 0.0;
        
        match piece {
            Piece::Queen | Piece::Rook => {
                // Look for battery formations with other heavy pieces
                for friendly_square in chess::ALL_SQUARES {
                    if let Some(friendly_piece) = board.piece_on(friendly_square) {
                        if board.color_on(friendly_square) == Some(piece_color) && friendly_square != square {
                            match friendly_piece {
                                Piece::Queen | Piece::Rook => {
                                    if self.forms_battery(board, square, friendly_square, piece, friendly_piece) {
                                        battery_score += 0.8; // Strong battery formation
                                    }
                                },
                                _ => {}
                            }
                        }
                    }
                }
            },
            Piece::Bishop => {
                // Bishop batteries on diagonals
                for friendly_square in chess::ALL_SQUARES {
                    if let Some(friendly_piece) = board.piece_on(friendly_square) {
                        if board.color_on(friendly_square) == Some(piece_color) && friendly_square != square {
                            match friendly_piece {
                                Piece::Queen | Piece::Bishop => {
                                    if self.forms_diagonal_battery(board, square, friendly_square) {
                                        battery_score += 0.6;
                                    }
                                },
                                _ => {}
                            }
                        }
                    }
                }
            },
            _ => {}
        }
        
        battery_score.min(1.0)
    }
    
    fn calculate_xray_score(&self, board: &Board, square: Square, piece: Piece) -> f32 {
        let Some(piece_color) = board.color_on(square) else { return 0.0; };
        let enemy_color = !piece_color;
        let mut xray_score = 0.0;
        
        match piece {
            Piece::Queen | Piece::Rook | Piece::Bishop => {
                // X-ray attacks through enemy pieces to more valuable targets
                for direction in self.get_piece_directions(piece) {
                    let mut current_square = square;
                    let mut pieces_seen = 0;
                    let mut first_piece_value = 0.0;
                    
                    while let Some(next_square) = self.move_in_direction(current_square, direction) {
                        if let Some(target_piece) = board.piece_on(next_square) {
                            pieces_seen += 1;
                            
                            if pieces_seen == 1 {
                                first_piece_value = self.get_piece_value_normalized(target_piece);
                            } else if pieces_seen == 2 {
                                // X-ray target found
                                if board.color_on(next_square) == Some(enemy_color) {
                                    let second_piece_value = self.get_piece_value_normalized(target_piece);
                                    
                                    // X-ray is valuable if second piece is more valuable than first
                                    if second_piece_value > first_piece_value {
                                        xray_score += (second_piece_value - first_piece_value) * 0.5;
                                    }
                                }
                                break;
                            }
                        }
                        current_square = next_square;
                    }
                }
            },
            _ => {}
        }
        
        (xray_score / 2.0).min(1.0)
    }
    
    fn calculate_windmill_score(&self, board: &Board, square: Square, piece: Piece) -> f32 {
        // Windmill is a complex tactical motif - simplified detection
        let Some(piece_color) = board.color_on(square) else { return 0.0; };
        let mut windmill_score: f32 = 0.0;
        
        match piece {
            Piece::Rook | Piece::Bishop => {
                // Look for discovered check patterns that can be repeated
                if let Some(enemy_king_square) = self.find_king(board, !piece_color) {
                    // Check if moving this piece would discover check and allow repeated attacks
                    if self.can_create_discovered_check(board, square, enemy_king_square) {
                        windmill_score += 0.9; // High value for windmill potential
                    }
                }
            },
            _ => {}
        }
        
        windmill_score.min(1.0)
    }
    
    fn calculate_mating_net_score(&self, board: &Board, square: Square, piece: Piece) -> f32 {
        let Some(piece_color) = board.color_on(square) else { return 0.0; };
        let enemy_color = !piece_color;
        let mut mating_net_score = 0.0;
        
        // Check contribution to mating attack
        if let Some(enemy_king_square) = self.find_king(board, enemy_color) {
            let king_mobility = self.count_king_legal_moves(board, enemy_king_square);
            
            // If enemy king is constrained
            if king_mobility <= 3 {
                // Check if this piece contributes to the attack
                if self.contributes_to_king_attack(board, square, piece, enemy_king_square) {
                    mating_net_score += (4.0 - king_mobility as f32) * 0.2;
                }
                
                // Bonus if we're giving check or controlling escape squares
                if self.can_give_check(board, square, piece) {
                    mating_net_score += 0.4;
                }
            }
        }
        
        mating_net_score.min(1.0)
    }
    
    fn calculate_back_rank_score(&self, board: &Board, square: Square, piece: Piece) -> f32 {
        let Some(piece_color) = board.color_on(square) else { return 0.0; };
        let enemy_color = !piece_color;
        let mut back_rank_score: f32 = 0.0;
        
        match piece {
            Piece::Queen | Piece::Rook => {
                // Check for back rank mate threats
                if let Some(enemy_king_square) = self.find_king(board, enemy_color) {
                    let king_rank = enemy_king_square.get_rank().to_index();
                    
                    // Check if enemy king is on back rank (rank 0 or 7)
                    if king_rank == 0 || king_rank == 7 {
                        let piece_rank = square.get_rank().to_index();
                        
                        // If we're on the same rank as the enemy king
                        if piece_rank == king_rank {
                            // Check if king's escape is blocked by its own pawns
                            if self.is_back_rank_vulnerable(board, enemy_king_square, enemy_color) {
                                back_rank_score += 0.9; // High back rank threat
                            }
                        }
                        
                        // Also check if we can get to the back rank
                        if self.can_reach_back_rank(board, square, piece, king_rank) {
                            back_rank_score += 0.5;
                        }
                    }
                }
            },
            _ => {}
        }
        
        back_rank_score.min(1.0)
    }
    fn calculate_promotion_score(&self, board: &Board, square: Square, piece: Piece) -> f32 {
        // Only pawns can promote
        if piece != Piece::Pawn {
            return 0.0;
        }
        
        let Some(piece_color) = board.color_on(square) else { return 0.0; };
        let rank = square.get_rank().to_index();
        let promotion_rank = if piece_color == Color::White { 7 } else { 0 };
        let pre_promotion_rank = if piece_color == Color::White { 6 } else { 1 };
        
        // High score for pawns that can promote immediately
        if rank == pre_promotion_rank {
            // Check if pawn can actually advance to promotion square
            let promotion_square = Square::make_square(
                chess::Rank::from_index(promotion_rank),
                square.get_file()
            );
            
            // If promotion square is empty or can be captured, this is valuable
            if board.piece_on(promotion_square).is_none() {
                return 0.9; // Very high value for free promotion
            } else if let Some(target_piece) = board.piece_on(promotion_square) {
                if board.color_on(promotion_square) != Some(piece_color) {
                    // Can capture and promote - extremely valuable
                    let capture_value = self.get_piece_value_normalized(target_piece);
                    return 0.8 + capture_value; // Promotion + capture value
                }
            }
        }
        
        // Score for pawns advancing toward promotion
        let distance_to_promotion = if piece_color == Color::White {
            7 - rank
        } else {
            rank
        };
        
        match distance_to_promotion {
            1 => 0.9, // About to promote
            2 => 0.6, // Close to promotion  
            3 => 0.3, // Making progress
            4 => 0.1, // Still distant
            _ => 0.0, // Too far away
        }
    }
    fn calculate_en_passant_score(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.0 }
    fn calculate_castling_score(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.0 }
    fn calculate_stalemate_score(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.0 }
    fn calculate_perpetual_score(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.0 }
    
    // Threat calculations
    fn calculate_immediate_threat_level(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.0 }
    fn calculate_next_move_threats(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.0 }
    fn calculate_discovered_threat_potential(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.0 }
    fn calculate_tempo_threats(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.0 }
    fn calculate_positional_threats(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.0 }
    fn calculate_long_term_threats(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.0 }
    fn calculate_multi_piece_coordination(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.0 }
    fn calculate_combination_potential(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.0 }
    fn calculate_sacrifice_threats(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.0 }
    fn calculate_endgame_threats(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.0 }
    fn calculate_king_hunt_score(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.0 }
    fn calculate_pawn_storm_score(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.0 }
    
    // Vulnerability assessments
    fn is_hanging(&self, board: &Board, square: Square) -> bool {
        if board.piece_on(square).is_none() {
            return false;
        }
        
        let attackers = self.count_attackers(board, square);
        let defenders = self.count_defenders(board, square);
        
        // A piece is hanging if it's attacked and not defended enough
        attackers > 0.0 && defenders == 0.0
    }
    
    fn is_pinned(&self, board: &Board, square: Square) -> bool {
        let Some(_piece) = board.piece_on(square) else { return false; };
        let Some(piece_color) = board.color_on(square) else { return false; };
        
        // Find our king
        let king_square = self.find_king(board, piece_color);
        let Some(king_pos) = king_square else { return false; };
        
        // Check if removing this piece would expose the king to attack
        // Simulate removing the piece temporarily by checking if there's a line attack through it
        let king_rank = king_pos.get_rank().to_index() as i8;
        let king_file = king_pos.get_file().to_index() as i8;
        let piece_rank = square.get_rank().to_index() as i8;
        let piece_file = square.get_file().to_index() as i8;
        
        // Check if piece is on same rank, file, or diagonal as king
        let on_same_line = king_rank == piece_rank || king_file == piece_file ||
                          (king_rank - piece_rank).abs() == (king_file - piece_file).abs();
        
        if !on_same_line {
            return false;
        }
        
        // Look for enemy sliding pieces that could attack through this piece
        let enemy_color = !piece_color;
        
        for enemy_square in chess::ALL_SQUARES {
            if let Some(enemy_piece) = board.piece_on(enemy_square) {
                if board.color_on(enemy_square) == Some(enemy_color) {
                    match enemy_piece {
                        Piece::Queen | Piece::Rook if king_rank == piece_rank || king_file == piece_file => {
                            if self.is_on_line_between(enemy_square, square, king_pos) {
                                return true;
                            }
                        },
                        Piece::Queen | Piece::Bishop if (king_rank - piece_rank).abs() == (king_file - piece_file).abs() => {
                            if self.is_on_line_between(enemy_square, square, king_pos) {
                                return true;
                            }
                        },
                        _ => continue,
                    }
                }
            }
        }
        
        false
    }
    
    fn is_overloaded(&self, board: &Board, square: Square) -> bool {
        let Some(_piece) = board.piece_on(square) else { return false; };
        let Some(piece_color) = board.color_on(square) else { return false; };
        
        // Count how many pieces this piece is defending
        let mut defending_count = 0;
        
        for target_square in chess::ALL_SQUARES {
            if let Some(_target_piece) = board.piece_on(target_square) {
                if board.color_on(target_square) == Some(piece_color) && target_square != square {
                    if self.piece_attacks_square(board, square, target_square) {
                        defending_count += 1;
                    }
                }
            }
        }
        
        // A piece is overloaded if it's defending 2+ pieces and is itself attacked
        defending_count >= 2 && self.count_attackers(board, square) > 0.0
    }
    
    fn calculate_attack_defense_ratio(&self, board: &Board, square: Square) -> f32 {
        let attackers = self.count_attackers(board, square);
        let defenders = self.count_defenders(board, square);
        
        if defenders == 0.0 {
            if attackers > 0.0 { 10.0 } else { 1.0 } // Very dangerous if attacked and undefended
        } else {
            attackers / defenders
        }
    }
    
    fn count_escape_squares(&self, board: &Board, square: Square) -> f32 {
        let Some(piece_color) = board.color_on(square) else { return 0.0; };
        let mut escape_count = 0;
        
        // Check adjacent squares for the piece
        for rank_delta in -1..=1 {
            for file_delta in -1..=1 {
                if rank_delta == 0 && file_delta == 0 { continue; }
                
                let current_rank = square.get_rank().to_index() as i8;
                let current_file = square.get_file().to_index() as i8;
                let new_rank = current_rank + rank_delta;
                let new_file = current_file + file_delta;
                
                if new_rank >= 0 && new_rank < 8 && new_file >= 0 && new_file < 8 {
                    let rank = chess::Rank::from_index(new_rank as usize);
                    let file = chess::File::from_index(new_file as usize);
                    let escape_square = Square::make_square(rank, file);
                    
                    // Check if square is empty or contains enemy piece
                    if board.piece_on(escape_square).is_none() || 
                       board.color_on(escape_square) != Some(piece_color) {
                        // Check if this square is not under attack
                        if self.count_attackers_for_color(board, escape_square, !piece_color) == 0.0 {
                            escape_count += 1;
                        }
                    }
                }
            }
        }
        
        escape_count as f32
    }
    
    fn count_defenders(&self, board: &Board, square: Square) -> f32 {
        let Some(piece_color) = board.color_on(square) else { return 0.0; };
        self.count_attackers_for_color(board, square, piece_color)
    }
    
    fn count_attackers(&self, board: &Board, square: Square) -> f32 {
        let Some(piece_color) = board.color_on(square) else { return 0.0; };
        let enemy_color = !piece_color;
        self.count_attackers_for_color(board, square, enemy_color)
    }
    fn calculate_net_safety(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.5 }
    fn calculate_removal_impact(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.0 }
    fn calculate_deflection_risk(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.0 }
    fn calculate_overload_risk(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.0 }
    fn calculate_fork_risk(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.0 }
    fn calculate_pin_risk(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.0 }
    fn calculate_skewer_risk(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.0 }
    fn calculate_discovery_risk(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.0 }
    fn calculate_overall_weakness(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.0 }
    
    // Mobility assessments
    fn legal_move_count(&self, board: &Board, square: Square) -> f32 {
        let moves = MoveGen::new_legal(board);
        moves.filter(|m| m.get_source() == square).count() as f32
    }
    
    fn safe_move_count(&self, board: &Board, square: Square) -> f32 {
        let Some(piece_color) = board.color_on(square) else { return 0.0; };
        let enemy_color = !piece_color;
        let mut safe_count = 0.0;
        
        let moves = MoveGen::new_legal(board);
        for mv in moves {
            if mv.get_source() == square {
                let target_square = mv.get_dest();
                
                // Check if the destination square is safe (not under enemy attack)
                let enemy_attackers = self.count_attackers_for_color(board, target_square, enemy_color);
                let friendly_defenders = self.count_attackers_for_color(board, target_square, piece_color);
                
                // Square is safe if not attacked, or if we have enough defenders
                if enemy_attackers == 0.0 || friendly_defenders >= enemy_attackers {
                    safe_count += 1.0;
                }
            }
        }
        
        safe_count
    }
    
    fn attacking_move_count(&self, board: &Board, square: Square) -> f32 {
        let Some(piece_color) = board.color_on(square) else { return 0.0; };
        let enemy_color = !piece_color;
        let mut attack_count = 0.0;
        
        let moves = MoveGen::new_legal(board);
        for mv in moves {
            if mv.get_source() == square {
                let target_square = mv.get_dest();
                
                // Check if this move attacks an enemy piece
                if let Some(_target_piece) = board.piece_on(target_square) {
                    if board.color_on(target_square) == Some(enemy_color) {
                        attack_count += 1.0;
                    }
                }
                
                // Also count if the move puts pressure on enemy squares
                if board.piece_on(target_square).is_none() {
                    // Check if this square controls important enemy territory
                    let target_rank = target_square.get_rank().to_index();
                    
                    // Bonus for advancing into enemy territory
                    if (piece_color == Color::White && target_rank >= 4) ||
                       (piece_color == Color::Black && target_rank <= 3) {
                        attack_count += 0.5;
                    }
                }
            }
        }
        
        attack_count
    }
    fn calculate_mobility_quality(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.5 }
    fn calculate_future_mobility(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.5 }
    fn calculate_restricted_mobility(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.2 }
    fn calculate_improvement_potential(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.3 }
    fn calculate_piece_specific_mobility(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.4 }
    fn calculate_dynamic_potential(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.3 }
    fn calculate_tactical_mobility_score(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.4 }
    fn calculate_positional_mobility_score(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.4 }
    fn calculate_endgame_mobility_score(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.5 }
    fn calculate_king_distance_factor(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.3 }
    fn calculate_center_access(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.4 }
    fn calculate_enemy_territory_access(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.2 }
    fn calculate_key_square_access(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.3 }
    fn calculate_file_control(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.3 }
    fn calculate_rank_control(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.3 }
    fn calculate_diagonal_control(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.3 }
    fn calculate_knight_outpost_score(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.2 }
    
    // Control calculations  
    fn calculate_square_control_strength(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.4 }
    fn calculate_contested_squares(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.3 }
    fn calculate_exclusive_control(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.2 }
    fn calculate_weak_square_control(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.1 }
    fn calculate_key_square_influence(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.3 }
    fn calculate_central_control(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.4 }
    fn calculate_flank_control(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.2 }
    fn calculate_enemy_king_zone_pressure(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.3 }
    fn calculate_pawn_break_support(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.2 }
    fn calculate_piece_placement_control(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.3 }
    fn calculate_route_control(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.2 }
    fn calculate_space_advantage(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.3 }
    fn calculate_territorial_gain(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.2 }
    fn calculate_influence_projection(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.3 }
    fn calculate_dominance_factor(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.4 }
    fn calculate_control_sustainability(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.3 }
    
    // Summary calculations
    fn calculate_piece_value(&self, _board: &Board, _square: Square, piece: Piece) -> f32 {
        self.get_piece_value_normalized(piece)
    }
    fn calculate_activity_score(&self, _board: &Board, square: Square, _piece: Piece) -> f32 {
        self.calculate_centralization(square) * 0.7 + 0.3
    }
    fn calculate_safety_score(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 { 0.6 }
    fn calculate_importance_weight(&self, _board: &Board, _square: Square, piece: Piece) -> f32 {
        self.get_piece_value_normalized(piece)
    }
    
    // Square analysis methods
    fn analyze_square_tactical(&self, _board: &Board, _square: Square) -> SquareTacticalFeatures {
        SquareTacticalFeatures {
            control_dynamics: [0.3; 12],
            tactical_importance: [0.2; 10],
            threat_vectors: [0.1; 8],
        }
    }
    
    fn analyze_square_positional(&self, _board: &Board, _square: Square) -> SquarePositionalFeatures {
        SquarePositionalFeatures {
            strategic_value: [0.4; 14],
            mobility_impact: [0.3; 10],
            structural_impact: [0.2; 12],
        }
    }
    
    fn analyze_square_meta(&self, _board: &Board, _square: Square) -> SquareMetaFeatures {
        SquareMetaFeatures {
            phase_relevance: [0.5; 8],
            evaluation_impact: [0.3; 6],
            pattern_involvement: [0.2; 10],
        }
    }
    
    fn calculate_square_importance_score(&self, _board: &Board, square: Square) -> f32 {
        self.calculate_centralization(square)
    }
    
    fn calculate_contest_level_score(&self, _board: &Board, _square: Square) -> f32 { 0.4 }
    fn calculate_strategic_value_score(&self, _board: &Board, square: Square) -> f32 {
        self.calculate_centralization(square) * 0.8
    }
    
    // Helper methods for tactical analysis
    fn find_king(&self, board: &Board, color: Color) -> Option<Square> {
        for square in chess::ALL_SQUARES {
            if let Some(piece) = board.piece_on(square) {
                if piece == Piece::King && board.color_on(square) == Some(color) {
                    return Some(square);
                }
            }
        }
        None
    }
    
    fn is_on_line_between(&self, attacker: Square, blocker: Square, target: Square) -> bool {
        let attacker_rank = attacker.get_rank().to_index() as i8;
        let attacker_file = attacker.get_file().to_index() as i8;
        let blocker_rank = blocker.get_rank().to_index() as i8;
        let blocker_file = blocker.get_file().to_index() as i8;
        let target_rank = target.get_rank().to_index() as i8;
        let target_file = target.get_file().to_index() as i8;
        
        // Check if all three squares are on the same line
        let rank_diff1 = attacker_rank - blocker_rank;
        let file_diff1 = attacker_file - blocker_file;
        let rank_diff2 = blocker_rank - target_rank;
        let file_diff2 = blocker_file - target_file;
        
        // Check if directions are consistent (same line)
        if rank_diff1 == 0 && rank_diff2 == 0 && file_diff1.signum() == file_diff2.signum() {
            return true; // Same rank
        }
        if file_diff1 == 0 && file_diff2 == 0 && rank_diff1.signum() == rank_diff2.signum() {
            return true; // Same file
        }
        if rank_diff1.abs() == file_diff1.abs() && rank_diff2.abs() == file_diff2.abs() &&
           rank_diff1.signum() == rank_diff2.signum() && file_diff1.signum() == file_diff2.signum() {
            return true; // Same diagonal
        }
        
        false
    }
    
    fn count_attackers_for_color(&self, board: &Board, square: Square, attacking_color: Color) -> f32 {
        let mut count = 0.0;
        
        for attacker_square in chess::ALL_SQUARES {
            if let Some(_piece) = board.piece_on(attacker_square) {
                if board.color_on(attacker_square) == Some(attacking_color) {
                    if self.piece_attacks_square(board, attacker_square, square) {
                        count += 1.0;
                    }
                }
            }
        }
        
        count
    }
    
    // Helper methods for tactical motif detection
    fn check_pin_potential(&self, board: &Board, attacker_square: Square, target_square: Square, attacker_piece: Piece, enemy_color: Color) -> Option<f32> {
        // Check if attacker can pin the target against a more valuable piece
        let attacker_rank = attacker_square.get_rank().to_index() as i8;
        let attacker_file = attacker_square.get_file().to_index() as i8;
        let target_rank = target_square.get_rank().to_index() as i8;
        let target_file = target_square.get_file().to_index() as i8;
        
        let rank_direction = (target_rank - attacker_rank).signum();
        let file_direction = (target_file - attacker_file).signum();
        
        // Check if this is a valid attack direction for the piece
        let can_attack = match attacker_piece {
            Piece::Rook => rank_direction == 0 || file_direction == 0,
            Piece::Bishop => rank_direction.abs() == file_direction.abs(),
            Piece::Queen => rank_direction == 0 || file_direction == 0 || rank_direction.abs() == file_direction.abs(),
            _ => false,
        };
        
        if !can_attack { return None; }
        
        // Look for a more valuable piece behind the target
        let mut current_rank = target_rank + rank_direction;
        let mut current_file = target_file + file_direction;
        
        while current_rank >= 0 && current_rank < 8 && current_file >= 0 && current_file < 8 {
            let rank = chess::Rank::from_index(current_rank as usize);
            let file = chess::File::from_index(current_file as usize);
            let check_square = Square::make_square(rank, file);
            
            if let Some(piece) = board.piece_on(check_square) {
                if board.color_on(check_square) == Some(enemy_color) {
                    let pinned_value = self.get_piece_value_normalized(piece);
                    let target_value = board.piece_on(target_square)
                        .map(|p| self.get_piece_value_normalized(p))
                        .unwrap_or(0.0);
                    
                    // Pin is valuable if the pinned piece is more valuable than the blocker
                    if pinned_value > target_value {
                        return Some(pinned_value - target_value);
                    }
                }
                break; // Hit a piece, can't pin beyond it
            }
            current_rank += rank_direction;
            current_file += file_direction;
        }
        
        None
    }
    
    fn check_skewer_potential(&self, board: &Board, attacker_square: Square, front_square: Square, attacker_piece: Piece, enemy_color: Color) -> Option<f32> {
        // Similar to pin, but front piece is more valuable than back piece
        let attacker_rank = attacker_square.get_rank().to_index() as i8;
        let attacker_file = attacker_square.get_file().to_index() as i8;
        let front_rank = front_square.get_rank().to_index() as i8;
        let front_file = front_square.get_file().to_index() as i8;
        
        let rank_direction = (front_rank - attacker_rank).signum();
        let file_direction = (front_file - attacker_file).signum();
        
        // Check if this is a valid attack direction
        let can_attack = match attacker_piece {
            Piece::Rook => rank_direction == 0 || file_direction == 0,
            Piece::Bishop => rank_direction.abs() == file_direction.abs(),
            Piece::Queen => rank_direction == 0 || file_direction == 0 || rank_direction.abs() == file_direction.abs(),
            _ => false,
        };
        
        if !can_attack { return None; }
        
        // Look for a less valuable piece behind the front piece
        let mut current_rank = front_rank + rank_direction;
        let mut current_file = front_file + file_direction;
        
        while current_rank >= 0 && current_rank < 8 && current_file >= 0 && current_file < 8 {
            let rank = chess::Rank::from_index(current_rank as usize);
            let file = chess::File::from_index(current_file as usize);
            let check_square = Square::make_square(rank, file);
            
            if let Some(piece) = board.piece_on(check_square) {
                if board.color_on(check_square) == Some(enemy_color) {
                    let back_value = self.get_piece_value_normalized(piece);
                    let front_value = board.piece_on(front_square)
                        .map(|p| self.get_piece_value_normalized(p))
                        .unwrap_or(0.0);
                    
                    // Skewer is valuable if the front piece is more valuable
                    if front_value > back_value {
                        return Some(front_value + back_value * 0.5);
                    }
                }
                break;
            }
            current_rank += rank_direction;
            current_file += file_direction;
        }
        
        None
    }
    
    fn calculate_knight_fork_potential(&self, board: &Board, square: Square, enemy_color: Color) -> f32 {
        let knight_moves = [
            (-2, -1), (-2, 1), (-1, -2), (-1, 2),
            (1, -2), (1, 2), (2, -1), (2, 1),
        ];
        
        let current_rank = square.get_rank().to_index() as i8;
        let current_file = square.get_file().to_index() as i8;
        let mut total_fork_value = 0.0;
        
        for &(rank_delta, file_delta) in &knight_moves {
            let new_rank = current_rank + rank_delta;
            let new_file = current_file + file_delta;
            
            if new_rank >= 0 && new_rank < 8 && new_file >= 0 && new_file < 8 {
                let rank = chess::Rank::from_index(new_rank as usize);
                let file = chess::File::from_index(new_file as usize);
                let _target_square = Square::make_square(rank, file);
                
                // Count how many enemy pieces this knight move would attack
                let mut attack_count = 0;
                let mut total_value = 0.0;
                
                for &(check_rank, check_file) in &knight_moves {
                    let check_new_rank = new_rank + check_rank;
                    let check_new_file = new_file + check_file;
                    
                    if check_new_rank >= 0 && check_new_rank < 8 && check_new_file >= 0 && check_new_file < 8 {
                        let check_rank_obj = chess::Rank::from_index(check_new_rank as usize);
                        let check_file_obj = chess::File::from_index(check_new_file as usize);
                        let check_square = Square::make_square(check_rank_obj, check_file_obj);
                        
                        if let Some(piece) = board.piece_on(check_square) {
                            if board.color_on(check_square) == Some(enemy_color) {
                                attack_count += 1;
                                total_value += self.get_piece_value_normalized(piece);
                            }
                        }
                    }
                }
                
                // Fork is valuable if attacking 2+ pieces
                if attack_count >= 2 {
                    total_fork_value += total_value * attack_count as f32 * 0.5;
                }
            }
        }
        
        total_fork_value
    }
    
    fn calculate_pawn_fork_potential(&self, board: &Board, square: Square, piece_color: Color, enemy_color: Color) -> f32 {
        let current_rank = square.get_rank().to_index() as i8;
        let current_file = square.get_file().to_index() as i8;
        let direction = if piece_color == Color::White { 1 } else { -1 };
        let mut fork_value = 0.0;
        
        // Check diagonal attacks
        for file_delta in [-1, 1] {
            let attack_rank = current_rank + direction;
            let attack_file = current_file + file_delta;
            
            if attack_rank >= 0 && attack_rank < 8 && attack_file >= 0 && attack_file < 8 {
                let rank = chess::Rank::from_index(attack_rank as usize);
                let file = chess::File::from_index(attack_file as usize);
                let attack_square = Square::make_square(rank, file);
                
                if let Some(piece) = board.piece_on(attack_square) {
                    if board.color_on(attack_square) == Some(enemy_color) {
                        fork_value += self.get_piece_value_normalized(piece);
                    }
                }
            }
        }
        
        // Pawn fork is valuable if attacking multiple pieces
        if fork_value > 0.6 { // Attacking pieces worth more than 2 pawns
            fork_value * 2.0
        } else {
            0.0
        }
    }
    
    fn calculate_sliding_fork_potential(&self, board: &Board, square: Square, _piece: Piece, enemy_color: Color) -> f32 {
        // Simplified: check if the piece can attack multiple enemy pieces from current position
        let mut attacked_value = 0.0;
        let mut attack_count = 0;
        
        for target_square in chess::ALL_SQUARES {
            if let Some(target_piece) = board.piece_on(target_square) {
                if board.color_on(target_square) == Some(enemy_color) {
                    if self.piece_attacks_square(board, square, target_square) {
                        attacked_value += self.get_piece_value_normalized(target_piece);
                        attack_count += 1;
                    }
                }
            }
        }
        
        // Fork bonus if attacking multiple pieces
        if attack_count >= 2 {
            attacked_value * 0.5
        } else {
            0.0
        }
    }
    
    fn find_enemy_targets(&self, board: &Board, enemy_color: Color) -> Square {
        // Find the most valuable enemy piece (simplified - just return king square)
        self.find_king(board, enemy_color).unwrap_or(Square::A1)
    }
    
    // Helper methods for piece attack patterns
    fn can_attack_along_line(&self, from: Square, to: Square, ranks_files: bool, diagonals: bool) -> bool {
        let from_rank = from.get_rank().to_index() as i8;
        let from_file = from.get_file().to_index() as i8;
        let to_rank = to.get_rank().to_index() as i8;
        let to_file = to.get_file().to_index() as i8;
        
        let rank_diff = (to_rank - from_rank).abs();
        let file_diff = (to_file - from_file).abs();
        
        // Check if on same rank/file
        if ranks_files && (rank_diff == 0 || file_diff == 0) {
            return true;
        }
        
        // Check if on same diagonal
        if diagonals && rank_diff == file_diff {
            return true;
        }
        
        false
    }
    
    fn can_knight_attack(&self, from: Square, to: Square) -> bool {
        let from_rank = from.get_rank().to_index() as i8;
        let from_file = from.get_file().to_index() as i8;
        let to_rank = to.get_rank().to_index() as i8;
        let to_file = to.get_file().to_index() as i8;
        
        let rank_diff = (to_rank - from_rank).abs();
        let file_diff = (to_file - from_file).abs();
        
        // Knight moves in L-shape
        (rank_diff == 2 && file_diff == 1) || (rank_diff == 1 && file_diff == 2)
    }
    
    fn can_pawn_attack(&self, from: Square, to: Square, piece_color: Color) -> bool {
        let from_rank = from.get_rank().to_index() as i8;
        let from_file = from.get_file().to_index() as i8;
        let to_rank = to.get_rank().to_index() as i8;
        let to_file = to.get_file().to_index() as i8;
        
        let rank_diff = to_rank - from_rank;
        let file_diff = (to_file - from_file).abs();
        
        // Pawn attacks diagonally forward
        let correct_direction = match piece_color {
            Color::White => rank_diff == 1,
            Color::Black => rank_diff == -1,
        };
        
        correct_direction && file_diff == 1
    }
    
    fn can_king_attack(&self, from: Square, to: Square) -> bool {
        let from_rank = from.get_rank().to_index() as i8;
        let from_file = from.get_file().to_index() as i8;
        let to_rank = to.get_rank().to_index() as i8;
        let to_file = to.get_file().to_index() as i8;
        
        let rank_diff = (to_rank - from_rank).abs();
        let file_diff = (to_file - from_file).abs();
        
        // King attacks adjacent squares
        rank_diff <= 1 && file_diff <= 1 && (rank_diff + file_diff > 0)
    }
    
    // Helper methods for advanced tactical analysis
    fn calculate_defended_pieces_value(&self, board: &Board, defender_square: Square) -> f32 {
        let Some(defender_color) = board.color_on(defender_square) else { return 0.0; };
        let mut total_value = 0.0;
        
        for target_square in chess::ALL_SQUARES {
            if let Some(piece) = board.piece_on(target_square) {
                if board.color_on(target_square) == Some(defender_color) && target_square != defender_square {
                    if self.piece_attacks_square(board, defender_square, target_square) {
                        total_value += self.get_piece_value_normalized(piece);
                    }
                }
            }
        }
        
        total_value
    }
    
    fn is_blocking_attack_line(&self, board: &Board, blocking_square: Square, attacker_square: Square, attacker_piece: Piece) -> bool {
        // Check if the blocking piece is on a line between attacker and potential targets
        for target_square in chess::ALL_SQUARES {
            if target_square != blocking_square && target_square != attacker_square {
                if let Some(_target_piece) = board.piece_on(target_square) {
                    // Check if attacker could attack target if blocker moved
                    match attacker_piece {
                        Piece::Queen | Piece::Rook => {
                            if self.can_attack_along_line(attacker_square, target_square, true, false) {
                                if self.is_on_line_between(attacker_square, blocking_square, target_square) {
                                    return true;
                                }
                            }
                        },
                        Piece::Bishop => {
                            if self.can_attack_along_line(attacker_square, target_square, false, true) {
                                if self.is_on_line_between(attacker_square, blocking_square, target_square) {
                                    return true;
                                }
                            }
                        },
                        _ => {}
                    }
                }
            }
        }
        false
    }
    
    fn can_interfere_with_line(&self, board: &Board, interferer_square: Square, enemy_square: Square, enemy_piece: Piece) -> bool {
        // Check if we can interfere with enemy piece's important lines of attack
        let Some(enemy_color) = board.color_on(enemy_square) else { return false; };
        let our_color = !enemy_color;
        
        // Find targets that the enemy piece is attacking
        for target_square in chess::ALL_SQUARES {
            if let Some(_target_piece) = board.piece_on(target_square) {
                if board.color_on(target_square) == Some(our_color) {
                    match enemy_piece {
                        Piece::Queen | Piece::Rook | Piece::Bishop => {
                            if self.piece_attacks_square(board, enemy_square, target_square) {
                                // Check if interferer can block this line
                                if self.can_block_line(interferer_square, enemy_square, target_square) {
                                    return true;
                                }
                            }
                        },
                        _ => {}
                    }
                }
            }
        }
        false
    }
    
    fn can_block_line(&self, blocker_square: Square, attacker_square: Square, target_square: Square) -> bool {
        // Check if blocker can get between attacker and target
        let _blocker_rank = blocker_square.get_rank().to_index() as i8;
        let _blocker_file = blocker_square.get_file().to_index() as i8;
        let _attacker_rank = attacker_square.get_rank().to_index() as i8;
        let _attacker_file = attacker_square.get_file().to_index() as i8;
        let _target_rank = target_square.get_rank().to_index() as i8;
        let _target_file = target_square.get_file().to_index() as i8;
        
        // Check if blocker is on the line between attacker and target
        // This is a simplified check - in a real implementation you'd check legal moves
        self.is_on_line_between(attacker_square, blocker_square, target_square)
    }
    
    fn count_pieces_defended_by(&self, board: &Board, defender_square: Square) -> u32 {
        let Some(defender_color) = board.color_on(defender_square) else { return 0; };
        let mut count = 0;
        
        for target_square in chess::ALL_SQUARES {
            if let Some(_piece) = board.piece_on(target_square) {
                if board.color_on(target_square) == Some(defender_color) && target_square != defender_square {
                    if self.piece_attacks_square(board, defender_square, target_square) {
                        count += 1;
                    }
                }
            }
        }
        
        count
    }
    
    fn count_king_legal_moves(&self, board: &Board, king_square: Square) -> u32 {
        let mut move_count = 0;
        let Some(king_color) = board.color_on(king_square) else { return 0; };
        let enemy_color = !king_color;
        
        // Check all adjacent squares
        for rank_delta in -1..=1 {
            for file_delta in -1..=1 {
                if rank_delta == 0 && file_delta == 0 { continue; }
                
                let current_rank = king_square.get_rank().to_index() as i8;
                let current_file = king_square.get_file().to_index() as i8;
                let new_rank = current_rank + rank_delta;
                let new_file = current_file + file_delta;
                
                if new_rank >= 0 && new_rank < 8 && new_file >= 0 && new_file < 8 {
                    let rank = chess::Rank::from_index(new_rank as usize);
                    let file = chess::File::from_index(new_file as usize);
                    let target_square = Square::make_square(rank, file);
                    
                    // Check if square is empty or has enemy piece
                    let square_available = board.piece_on(target_square).is_none() || 
                                         board.color_on(target_square) == Some(enemy_color);
                    
                    // Check if square is not under attack
                    let square_safe = self.count_attackers_for_color(board, target_square, enemy_color) == 0.0;
                    
                    if square_available && square_safe {
                        move_count += 1;
                    }
                }
            }
        }
        
        move_count
    }
    
    fn forms_battery(&self, _board: &Board, piece1_square: Square, piece2_square: Square, piece1: Piece, piece2: Piece) -> bool {
        // Check if two pieces form a battery (both on same rank/file)
        let rank1 = piece1_square.get_rank().to_index();
        let file1 = piece1_square.get_file().to_index();
        let rank2 = piece2_square.get_rank().to_index();
        let file2 = piece2_square.get_file().to_index();
        
        // Must be on same rank or file
        if rank1 != rank2 && file1 != file2 {
            return false;
        }
        
        // Check if both pieces can attack along the same line
        match (piece1, piece2) {
            (Piece::Queen, _) | (_, Piece::Queen) | 
            (Piece::Rook, Piece::Rook) => {
                // Battery on rank or file
                rank1 == rank2 || file1 == file2
            },
            _ => false
        }
    }
    
    fn forms_diagonal_battery(&self, _board: &Board, piece1_square: Square, piece2_square: Square) -> bool {
        // Check if two pieces form a battery on a diagonal
        let rank1 = piece1_square.get_rank().to_index() as i8;
        let file1 = piece1_square.get_file().to_index() as i8;
        let rank2 = piece2_square.get_rank().to_index() as i8;
        let file2 = piece2_square.get_file().to_index() as i8;
        
        // Must be on same diagonal
        (rank1 - rank2).abs() == (file1 - file2).abs()
    }
    
    fn get_piece_directions(&self, piece: Piece) -> Vec<(i8, i8)> {
        match piece {
            Piece::Queen => vec![
                (0, 1), (0, -1), (1, 0), (-1, 0),    // Rank/file
                (1, 1), (1, -1), (-1, 1), (-1, -1)   // Diagonals
            ],
            Piece::Rook => vec![
                (0, 1), (0, -1), (1, 0), (-1, 0)     // Rank/file only
            ],
            Piece::Bishop => vec![
                (1, 1), (1, -1), (-1, 1), (-1, -1)   // Diagonals only
            ],
            _ => vec![]
        }
    }
    
    fn move_in_direction(&self, square: Square, direction: (i8, i8)) -> Option<Square> {
        let current_rank = square.get_rank().to_index() as i8;
        let current_file = square.get_file().to_index() as i8;
        let new_rank = current_rank + direction.0;
        let new_file = current_file + direction.1;
        
        if new_rank >= 0 && new_rank < 8 && new_file >= 0 && new_file < 8 {
            let rank = chess::Rank::from_index(new_rank as usize);
            let file = chess::File::from_index(new_file as usize);
            Some(Square::make_square(rank, file))
        } else {
            None
        }
    }
    
    fn can_create_discovered_check(&self, board: &Board, piece_square: Square, enemy_king_square: Square) -> bool {
        let Some(piece_color) = board.color_on(piece_square) else { return false; };
        
        // Look for friendly pieces that could give discovered check if this piece moves
        for friendly_square in chess::ALL_SQUARES {
            if let Some(friendly_piece) = board.piece_on(friendly_square) {
                if board.color_on(friendly_square) == Some(piece_color) && friendly_square != piece_square {
                    match friendly_piece {
                        Piece::Queen | Piece::Rook | Piece::Bishop => {
                            // Check if piece is blocking a potential check
                            if self.is_on_line_between(friendly_square, piece_square, enemy_king_square) {
                                return true;
                            }
                        },
                        _ => {}
                    }
                }
            }
        }
        
        false
    }
    
    fn contributes_to_king_attack(&self, board: &Board, piece_square: Square, _piece: Piece, enemy_king_square: Square) -> bool {
        // Check if piece attacks king or controls escape squares
        if self.piece_attacks_square(board, piece_square, enemy_king_square) {
            return true;
        }
        
        // Check if piece controls king's escape squares
        let king_rank = enemy_king_square.get_rank().to_index() as i8;
        let king_file = enemy_king_square.get_file().to_index() as i8;
        
        for rank_delta in -1..=1 {
            for file_delta in -1..=1 {
                if rank_delta == 0 && file_delta == 0 { continue; }
                
                let escape_rank = king_rank + rank_delta;
                let escape_file = king_file + file_delta;
                
                if escape_rank >= 0 && escape_rank < 8 && escape_file >= 0 && escape_file < 8 {
                    let rank = chess::Rank::from_index(escape_rank as usize);
                    let file = chess::File::from_index(escape_file as usize);
                    let escape_square = Square::make_square(rank, file);
                    
                    if self.piece_attacks_square(board, piece_square, escape_square) {
                        return true;
                    }
                }
            }
        }
        
        false
    }
    
    fn is_back_rank_vulnerable(&self, board: &Board, king_square: Square, king_color: Color) -> bool {
        let king_rank = king_square.get_rank().to_index();
        let king_file = king_square.get_file().to_index();
        
        // Check if king is blocked by its own pawns
        let pawn_rank = if king_color == Color::White { 
            king_rank + 1 
        } else { 
            king_rank.saturating_sub(1) 
        };
        
        // Check squares in front of king for blocking pawns
        for file_offset in -1..=1 {
            let check_file = (king_file as i8 + file_offset) as usize;
            if check_file < 8 && pawn_rank < 8 {
                let rank = chess::Rank::from_index(pawn_rank);
                let file = chess::File::from_index(check_file);
                let check_square = Square::make_square(rank, file);
                
                if let Some(piece) = board.piece_on(check_square) {
                    if piece == Piece::Pawn && board.color_on(check_square) == Some(king_color) {
                        return true; // King is blocked by own pawn
                    }
                }
            }
        }
        
        false
    }
    
    fn can_reach_back_rank(&self, board: &Board, piece_square: Square, piece: Piece, target_rank: usize) -> bool {
        let _piece_rank = piece_square.get_rank().to_index();
        let piece_file = piece_square.get_file().to_index();
        
        match piece {
            Piece::Queen | Piece::Rook => {
                // Can reach back rank if on same file or can move to back rank
                piece_file == piece_file || // Always true, but represents potential to move
                self.has_clear_path_to_rank(board, piece_square, target_rank)
            },
            _ => false
        }
    }
    
    fn has_clear_path_to_rank(&self, board: &Board, from_square: Square, target_rank: usize) -> bool {
        let from_rank = from_square.get_rank().to_index();
        let from_file = from_square.get_file().to_index();
        
        if from_rank == target_rank {
            return true;
        }
        
        // Check if there's a clear path along the file to the target rank
        let direction = if target_rank > from_rank { 1 } else { -1 };
        let mut current_rank = from_rank as i8 + direction;
        
        while current_rank != target_rank as i8 {
            if current_rank < 0 || current_rank >= 8 {
                break;
            }
            
            let rank = chess::Rank::from_index(current_rank as usize);
            let file = chess::File::from_index(from_file);
            let check_square = Square::make_square(rank, file);
            
            if board.piece_on(check_square).is_some() {
                return false; // Path is blocked
            }
            
            current_rank += direction;
        }
        
        true
    }
    
    // Additional helper methods for development and tempo analysis
    fn calculate_diagonal_control_length(&self, board: &Board, square: Square) -> f32 {
        let mut total_length = 0.0;
        let directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
        
        for &(rank_dir, file_dir) in &directions {
            let mut length = 0;
            let mut current_square = square;
            
            while let Some(next_square) = self.move_in_direction(current_square, (rank_dir, file_dir)) {
                length += 1;
                if board.piece_on(next_square).is_some() {
                    break; // Blocked
                }
                current_square = next_square;
            }
            
            total_length += length as f32;
        }
        
        total_length / 28.0 // Normalize by maximum possible diagonal control
    }
    
    fn calculate_file_openness(&self, board: &Board, square: Square) -> f32 {
        let file = square.get_file().to_index();
        let mut pawn_count = 0;
        
        // Count pawns on this file
        for rank in 0..8 {
            let rank_obj = chess::Rank::from_index(rank);
            let file_obj = chess::File::from_index(file);
            let check_square = Square::make_square(rank_obj, file_obj);
            
            if let Some(piece) = board.piece_on(check_square) {
                if piece == Piece::Pawn {
                    pawn_count += 1;
                }
            }
        }
        
        match pawn_count {
            0 => 1.0,  // Open file
            1 => 0.6,  // Half-open file
            _ => 0.2   // Closed file
        }
    }
    
    fn is_king_castled(&self, _board: &Board, king_square: Square, king_color: Color) -> bool {
        let expected_rank = if king_color == Color::White { 0 } else { 7 };
        let king_rank = king_square.get_rank().to_index();
        let king_file = king_square.get_file().to_index();
        
        // Check if king is on castled squares (kingside or queenside)
        king_rank == expected_rank && (king_file == 2 || king_file == 6)
    }
    
    fn calculate_king_activity_endgame(&self, board: &Board, king_square: Square) -> f32 {
        // In endgame, active king is important
        let center_distance = self.calculate_distance_to_center(king_square);
        let mobility = self.count_king_legal_moves(board, king_square) as f32;
        
        let centralization = (7.0 - center_distance) / 7.0;
        let mobility_factor = mobility / 8.0; // Max 8 king moves
        
        (centralization * 0.6 + mobility_factor * 0.4).min(1.0)
    }
    
    fn calculate_distance_to_center(&self, square: Square) -> f32 {
        let rank = square.get_rank().to_index() as f32;
        let file = square.get_file().to_index() as f32;
        let center_rank = 3.5;
        let center_file = 3.5;
        
        ((rank - center_rank).abs() + (file - center_file).abs()) / 2.0
    }
    
    fn calculate_pawn_development_score(&self, board: &Board, square: Square, piece_color: Color) -> f32 {
        let rank = square.get_rank().to_index();
        let file = square.get_file().to_index();
        let mut score = 0.0;
        
        // Central pawns get bonus
        if (3..=4).contains(&file) {
            score += 0.4;
        }
        
        // Advanced pawns get bonus
        let advancement = if piece_color == Color::White {
            rank as f32 / 7.0
        } else {
            (7 - rank) as f32 / 7.0
        };
        score += advancement * 0.3;
        
        // Pawn chain bonus
        if self.is_in_pawn_chain(board, square, piece_color) {
            score += 0.2;
        }
        
        score.min(1.0)
    }
    
    fn is_in_pawn_chain(&self, board: &Board, square: Square, piece_color: Color) -> bool {
        // Check if pawn is part of a chain (supported or supporting other pawns)
        let rank = square.get_rank().to_index() as i8;
        let file = square.get_file().to_index() as i8;
        let direction = if piece_color == Color::White { -1 } else { 1 };
        
        // Check for supporting pawns
        for file_offset in [-1, 1] {
            let support_rank = rank + direction;
            let support_file = file + file_offset;
            
            if support_rank >= 0 && support_rank < 8 && support_file >= 0 && support_file < 8 {
                let rank_obj = chess::Rank::from_index(support_rank as usize);
                let file_obj = chess::File::from_index(support_file as usize);
                let support_square = Square::make_square(rank_obj, file_obj);
                
                if let Some(piece) = board.piece_on(support_square) {
                    if piece == Piece::Pawn && board.color_on(support_square) == Some(piece_color) {
                        return true;
                    }
                }
            }
        }
        
        false
    }
    
    fn count_immediate_threats(&self, board: &Board, square: Square, piece: Piece) -> f32 {
        let Some(piece_color) = board.color_on(square) else { return 0.0; };
        let enemy_color = !piece_color;
        let mut threat_count = 0.0;
        
        // Count immediate threats this piece creates
        for target_square in chess::ALL_SQUARES {
            if let Some(target_piece) = board.piece_on(target_square) {
                if board.color_on(target_square) == Some(enemy_color) {
                    if self.piece_attacks_square(board, square, target_square) {
                        let target_value = self.get_piece_value_normalized(target_piece);
                        let our_value = self.get_piece_value_normalized(piece);
                        
                        // Threat is more significant if we can win material
                        if target_value > our_value || self.count_defenders(board, target_square) == 0.0 {
                            threat_count += 1.0;
                        } else {
                            threat_count += 0.5;
                        }
                    }
                }
            }
        }
        
        threat_count
    }
    
    fn is_pawn_advance_with_tempo(&self, board: &Board, square: Square, piece_color: Color) -> bool {
        let rank = square.get_rank().to_index();
        let file = square.get_file().to_index();
        let enemy_color = !piece_color;
        
        // Check if pawn advance attacks enemy pieces
        let attack_rank = if piece_color == Color::White { rank + 1 } else { rank.saturating_sub(1) };
        
        for file_offset in [-1, 1] {
            let attack_file = (file as i8 + file_offset) as usize;
            if attack_file < 8 && attack_rank < 8 {
                let rank_obj = chess::Rank::from_index(attack_rank);
                let file_obj = chess::File::from_index(attack_file);
                let attack_square = Square::make_square(rank_obj, file_obj);
                
                if let Some(_piece) = board.piece_on(attack_square) {
                    if board.color_on(attack_square) == Some(enemy_color) {
                        return true; // Pawn advance attacks enemy piece
                    }
                }
            }
        }
        
        false
    }
    
    fn calculate_square_distance(&self, square1: Square, square2: Square) -> i32 {
        let rank1 = square1.get_rank().to_index() as i32;
        let file1 = square1.get_file().to_index() as i32;
        let rank2 = square2.get_rank().to_index() as i32;
        let file2 = square2.get_file().to_index() as i32;
        
        // Manhattan distance
        (rank1 - rank2).abs() + (file1 - file2).abs()
    }
    
    fn count_constrained_enemy_pieces(&self, board: &Board, square: Square, our_color: Color) -> f32 {
        let enemy_color = !our_color;
        let mut constrained_count = 0.0;
        
        // Check if our piece constrains enemy piece movement
        for enemy_square in chess::ALL_SQUARES {
            if let Some(_enemy_piece) = board.piece_on(enemy_square) {
                if board.color_on(enemy_square) == Some(enemy_color) {
                    let enemy_mobility = self.legal_move_count(board, enemy_square);
                    
                    // If enemy piece attacks our piece, it might be constraining it
                    if self.piece_attacks_square(board, enemy_square, square) {
                        // Enemy piece is somewhat constrained by having to guard against our threats
                        constrained_count += 0.5;
                    }
                    
                    // If enemy has very low mobility, it's constrained
                    if enemy_mobility <= 2.0 {
                        constrained_count += 0.3;
                    }
                }
            }
        }
        
        constrained_count
    }
    
    fn attacks_king_zone(&self, board: &Board, piece_square: Square, _piece: Piece, king_square: Square) -> bool {
        // Check if piece attacks squares around the enemy king
        let king_rank = king_square.get_rank().to_index() as i8;
        let king_file = king_square.get_file().to_index() as i8;
        
        for rank_offset in -1..=1 {
            for file_offset in -1..=1 {
                let zone_rank = king_rank + rank_offset;
                let zone_file = king_file + file_offset;
                
                if zone_rank >= 0 && zone_rank < 8 && zone_file >= 0 && zone_file < 8 {
                    let rank = chess::Rank::from_index(zone_rank as usize);
                    let file = chess::File::from_index(zone_file as usize);
                    let zone_square = Square::make_square(rank, file);
                    
                    if self.piece_attacks_square(board, piece_square, zone_square) {
                        return true;
                    }
                }
            }
        }
        
        false
    }
}

impl Default for FeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
} 