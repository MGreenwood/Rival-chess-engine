use std::collections::HashMap;
use super::{
    PAG,
    node::{Node, NodeType, PieceNode, CriticalSquareNode, PieceType, Color},
    edge::{Edge, LegacyEdgeType, DirectRelationType, ControlType, CooperationType},
    Coordinate,
};

/// Builds a Positional Adjacency Graph (PAG) from a board position
pub struct PAGBuilder {
    next_id: u64,
    board_size: u8,
}

impl PAGBuilder {
    /// Creates a new PAGBuilder
    pub fn new(board_size: u8) -> Self {
        Self {
            next_id: 1,
            board_size,
        }
    }

    /// Gets the next available ID
    fn get_next_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Creates a piece node
    fn create_piece_node(
        &mut self,
        piece_type: PieceType,
        color: Color,
        coordinate: Coordinate,
    ) -> PieceNode {
        let material_value = match piece_type {
            PieceType::Pawn => 1,
            PieceType::Knight | PieceType::Bishop => 3,
            PieceType::Rook => 5,
            PieceType::Queen => 9,
            PieceType::King => 0,  // King's value is not relevant for material counting
        };

        PieceNode::new(
            self.get_next_id(),
            piece_type,
            color,
            coordinate,
            material_value,
        )
    }

    /// Creates a critical square node
    fn create_critical_square_node(
        &mut self,
        coordinate: Coordinate,
    ) -> CriticalSquareNode {
        CriticalSquareNode::new(
            self.get_next_id(),
            coordinate,
        )
    }

    /// Identifies critical squares in the position
    fn identify_critical_squares(&mut self, pieces: &[PieceNode]) -> Vec<CriticalSquareNode> {
        let mut critical_squares = Vec::new();
        let mut square_importance = HashMap::new();

        // First pass: calculate importance scores for all squares
        for piece in pieces {
            // Example criteria (to be expanded):
            // 1. Squares around kings
            if piece.piece_type() == PieceType::King {
                for dx in -1..=1 {
                    for dy in -1..=1 {
                        let new_rank = piece.get_coordinate().rank as i16 + dx;
                        let new_file = piece.get_coordinate().file as i16 + dy;
                        
                        if new_rank >= 0 && new_rank < self.board_size as i16 &&
                           new_file >= 0 && new_file < self.board_size as i16 {
                            let coord = Coordinate::new(new_rank as u8, new_file as u8);
                            *square_importance.entry(coord).or_insert(0.0) += 1.0;
                        }
                    }
                }
            }

            // 2. Central squares (for small boards, just the center; for larger boards, the extended center)
            let center_start = self.board_size / 3;
            let center_end = self.board_size - center_start;
            for rank in center_start..center_end {
                for file in center_start..center_end {
                    let coord = Coordinate::new(rank, file);
                    *square_importance.entry(coord).or_insert(0.0) += 0.5;
                }
            }
        }

        // Second pass: create critical square nodes for important squares
        for (coord, importance) in square_importance {
            if importance >= 0.5 {  // Threshold for considering a square critical
                critical_squares.push(self.create_critical_square_node(coord));
            }
        }

        critical_squares
    }

    /// Creates attack/defense edges between pieces
    fn create_piece_relationship_edges(
        &self,
        pieces: &[PieceNode],
    ) -> Vec<Edge> {
        let mut edges = Vec::new();

        for (i, piece1) in pieces.iter().enumerate() {
            for piece2 in pieces.iter().skip(i + 1) {
                // Skip pieces of the same color for attack relationships
                if piece1.color() != piece2.color() {
                    // Check if piece1 attacks piece2
                    if self.can_attack(piece1, piece2) {
                        edges.push(Edge::new_legacy(
                            LegacyEdgeType::DirectRelation(DirectRelationType::Attack {
                                attacker_type: piece1.piece_type(),
                                target_type: piece2.piece_type(),
                                strength: 1.0,  // To be refined based on piece values and position
                            }),
                            1.0,
                            piece1.get_id(),
                            piece2.get_id(),
                        ));
                    }

                    // Check if piece2 attacks piece1
                    if self.can_attack(piece2, piece1) {
                        edges.push(Edge::new_legacy(
                            LegacyEdgeType::DirectRelation(DirectRelationType::Attack {
                                attacker_type: piece2.piece_type(),
                                target_type: piece1.piece_type(),
                                strength: 1.0,
                            }),
                            1.0,
                            piece2.get_id(),
                            piece1.get_id(),
                        ));
                    }
                } else {
                    // Same color pieces: check for cooperative relationships
                    if let Some(coop_type) = self.identify_cooperation(piece1, piece2) {
                        edges.push(Edge::new_legacy(
                            LegacyEdgeType::Cooperation(coop_type),
                            1.0,
                            piece1.get_id(),
                            piece2.get_id(),
                        ));
                    }
                }
            }
        }

        edges
    }

    /// Checks if one piece can attack another
    fn can_attack(&self, attacker: &PieceNode, target: &PieceNode) -> bool {
        // This is a simplified implementation
        // TODO: Implement proper move generation and attack detection
        let dx = (attacker.get_coordinate().rank as i8 - target.get_coordinate().rank as i8).abs();
        let dy = (attacker.get_coordinate().file as i8 - target.get_coordinate().file as i8).abs();

        match attacker.piece_type() {
            PieceType::Knight => (dx == 2 && dy == 1) || (dx == 1 && dy == 2),
            PieceType::Bishop => dx == dy,
            PieceType::Rook => dx == 0 || dy == 0,
            PieceType::Queen => dx == dy || dx == 0 || dy == 0,
            PieceType::King => dx <= 1 && dy <= 1,
            PieceType::Pawn => {
                let forward = if attacker.color() == Color::White { 1 } else { -1 };
                dx == 1 && (target.get_coordinate().rank as i8 - attacker.get_coordinate().rank as i8) == forward && dy == 1
            }
        }
    }

    /// Identifies cooperative relationships between pieces
    fn identify_cooperation(&self, piece1: &PieceNode, piece2: &PieceNode) -> Option<CooperationType> {
        // Example cooperation types (to be expanded):
        
        // Bishop pair
        if piece1.piece_type() == PieceType::Bishop && piece2.piece_type() == PieceType::Bishop {
            return Some(CooperationType {
                cooperation_strength: 1.0,
                cooperation_type: "BishopPair".to_string(),
            });
        }

        // Battery formation (rook/queen alignment)
        if (piece1.piece_type() == PieceType::Rook || piece1.piece_type() == PieceType::Queen) &&
           (piece2.piece_type() == PieceType::Rook || piece2.piece_type() == PieceType::Queen) {
            if piece1.get_coordinate().rank == piece2.get_coordinate().rank ||
               piece1.get_coordinate().file == piece2.get_coordinate().file {
                return Some(CooperationType {
                    cooperation_strength: 1.0,
                    cooperation_type: "Battery".to_string(),
                });
            }
        }

        // Mutual defense
        if self.can_attack(piece1, piece2) {
            return Some(CooperationType {
                cooperation_strength: 1.0,
                cooperation_type: "MutualDefense".to_string(),
            });
        }

        None
    }

    /// Creates control edges for critical squares
    fn create_control_edges(
        &self,
        pieces: &[PieceNode],
        critical_squares: &[CriticalSquareNode],
    ) -> Vec<Edge> {
        let mut edges = Vec::new();

        for piece in pieces {
            for square in critical_squares {
                if self.controls_square(piece, square.get_coordinate()) {
                    edges.push(Edge::new_legacy(
                        LegacyEdgeType::Control(ControlType {
                            controlling_color: piece.color(),
                            degree: 1.0,  // To be refined based on piece type and distance
                            is_contested: false,  // To be determined by analyzing all pieces
                        }),
                        1.0,
                        piece.get_id(),
                        square.get_id(),
                    ));
                }
            }
        }

        edges
    }

    /// Checks if a piece controls a square
    fn controls_square(&self, piece: &PieceNode, square: Coordinate) -> bool {
        // This is a simplified implementation
        // TODO: Implement proper control detection considering obstacles
        let dx = (piece.get_coordinate().rank as i8 - square.rank as i8).abs();
        let dy = (piece.get_coordinate().file as i8 - square.file as i8).abs();

        match piece.piece_type() {
            PieceType::Knight => (dx == 2 && dy == 1) || (dx == 1 && dy == 2),
            PieceType::Bishop => dx == dy,
            PieceType::Rook => dx == 0 || dy == 0,
            PieceType::Queen => dx == dy || dx == 0 || dy == 0,
            PieceType::King => dx <= 1 && dy <= 1,
            PieceType::Pawn => {
                let forward = if piece.color() == Color::White { 1 } else { -1 };
                dx == 1 && (square.rank as i8 - piece.get_coordinate().rank as i8) == forward && dy == 1
            }
        }
    }

    /// Builds a PAG from a list of pieces
    pub fn build(&mut self, pieces: Vec<PieceNode>) -> PAG {
        let mut pag = PAG::new(self.board_size);

        // Add all piece nodes
        for piece in &pieces {
            pag.add_node(NodeType::LegacyPiece(piece.clone()));
        }

        // Identify and add critical squares
        let critical_squares = self.identify_critical_squares(&pieces);
        for square in &critical_squares {
            pag.add_node(NodeType::LegacyCriticalSquare(square.clone()));
        }

        // Create and add piece relationship edges
        let piece_edges = self.create_piece_relationship_edges(&pieces);
        for edge in piece_edges {
            let _ = pag.add_edge(edge);
        }

        // Create and add control edges
        let control_edges = self.create_control_edges(&pieces, &critical_squares);
        for edge in control_edges {
            let _ = pag.add_edge(edge);
        }

        pag
    }

    /// Builds a PAG for a 5x5 starting position
    pub fn build_5x5_starting_position(&mut self) -> PAG {
        let mut pieces = Vec::new();
        
        // Create white pieces
        pieces.push(self.create_piece_node(PieceType::King, Color::White, Coordinate::new(0, 2)));
        pieces.push(self.create_piece_node(PieceType::Queen, Color::White, Coordinate::new(0, 1)));
        pieces.push(self.create_piece_node(PieceType::Bishop, Color::White, Coordinate::new(0, 0)));
        pieces.push(self.create_piece_node(PieceType::Knight, Color::White, Coordinate::new(0, 3)));
        pieces.push(self.create_piece_node(PieceType::Rook, Color::White, Coordinate::new(0, 4)));
        
        // Create white pawns
        for file in 0..5 {
            pieces.push(self.create_piece_node(PieceType::Pawn, Color::White, Coordinate::new(1, file)));
        }

        // Create black pieces (mirrored)
        pieces.push(self.create_piece_node(PieceType::King, Color::Black, Coordinate::new(4, 2)));
        pieces.push(self.create_piece_node(PieceType::Queen, Color::Black, Coordinate::new(4, 1)));
        pieces.push(self.create_piece_node(PieceType::Bishop, Color::Black, Coordinate::new(4, 0)));
        pieces.push(self.create_piece_node(PieceType::Knight, Color::Black, Coordinate::new(4, 3)));
        pieces.push(self.create_piece_node(PieceType::Rook, Color::Black, Coordinate::new(4, 4)));
        
        // Create black pawns
        for file in 0..5 {
            pieces.push(self.create_piece_node(PieceType::Pawn, Color::Black, Coordinate::new(3, file)));
        }

        self.build(pieces)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_critical_square_identification() {
        let mut builder = PAGBuilder::new(5);
        
        // Create a simple position with two kings
        let white_king = builder.create_piece_node(
            PieceType::King,
            Color::White,
            Coordinate::new(0, 0),
        );
        let black_king = builder.create_piece_node(
            PieceType::King,
            Color::Black,
            Coordinate::new(4, 4),
        );

        let pieces = vec![white_king, black_king];
        let critical_squares = builder.identify_critical_squares(&pieces);

        // Should identify squares around both kings
        assert!(critical_squares.len() > 0);
        
        // Verify that squares adjacent to kings are marked as critical
        let has_white_king_adjacent = critical_squares.iter().any(|square| {
            square.get_coordinate() == Coordinate::new(0, 1) ||
            square.get_coordinate() == Coordinate::new(1, 0) ||
            square.get_coordinate() == Coordinate::new(1, 1)
        });
        
        let has_black_king_adjacent = critical_squares.iter().any(|square| {
            square.get_coordinate() == Coordinate::new(4, 3) ||
            square.get_coordinate() == Coordinate::new(3, 4) ||
            square.get_coordinate() == Coordinate::new(3, 3)
        });

        assert!(has_white_king_adjacent);
        assert!(has_black_king_adjacent);
    }

    #[test]
    fn test_piece_relationship_edges() {
        let mut builder = PAGBuilder::new(5);
        
        // Create a simple position with attacking pieces
        let white_queen = builder.create_piece_node(
            PieceType::Queen,
            Color::White,
            Coordinate::new(0, 0),
        );
        let black_pawn = builder.create_piece_node(
            PieceType::Pawn,
            Color::Black,
            Coordinate::new(1, 1),
        );

        let pieces = vec![white_queen, black_pawn];
        let edges = builder.create_piece_relationship_edges(&pieces);

        // Should create at least one attack edge
        assert!(edges.iter().any(|edge| {
            matches!(edge.edge_type(),
                EdgeType::DirectRelation(DirectRelationType::Attack { 
                    attacker_type: PieceType::Queen,
                    target_type: PieceType::Pawn,
                    ..
                })
            )
        }));
    }

    #[test]
    fn test_5x5_starting_position() {
        let mut builder = PAGBuilder::new(5);
        let pag = builder.build_5x5_starting_position();

        // Verify basic structure
        assert_eq!(pag.board_size(), 5);
        
        // Should have 20 pieces (10 per side)
        let piece_count = (0..builder.next_id)
            .filter(|&id| matches!(pag.get_node(id), Some(NodeType::Piece(_))))
            .count();
        assert_eq!(piece_count, 20);

        // Should have some critical squares (at least the center and king vicinity)
        let critical_square_count = (0..builder.next_id)
            .filter(|&id| matches!(pag.get_node(id), Some(NodeType::CriticalSquare(_))))
            .count();
        assert!(critical_square_count > 0);

        // Should have some edges
        assert!(pag.edge_count() > 0);
    }

    #[test]
    fn test_pag_building() {
        let mut builder = PAGBuilder::new(5);
        
        // Create a simple position
        let pieces = vec![
            builder.create_piece_node(PieceType::King, Color::White, Coordinate::new(0, 0)),
            builder.create_piece_node(PieceType::Queen, Color::White, Coordinate::new(0, 1)),
            builder.create_piece_node(PieceType::King, Color::Black, Coordinate::new(4, 4)),
        ];

        let pag = builder.build(pieces);

        // Verify nodes
        assert_eq!(pag.board_size(), 5);
        
        // Should have 3 pieces plus some critical squares
        let piece_count = (0..builder.next_id)
            .filter(|&id| matches!(pag.get_node(id), Some(NodeType::Piece(_))))
            .count();
        assert_eq!(piece_count, 3);

        // Should have some edges (at least cooperation between white pieces)
        assert!(pag.edge_count() > 0);
    }
} 