use chess::{Piece, Color, Square, Rank, File};
use crate::board::Board;

const PAWN_VALUE: f32 = 1.0;
const KNIGHT_VALUE: f32 = 3.0;
const BISHOP_VALUE: f32 = 3.0;
const ROOK_VALUE: f32 = 5.0;
const QUEEN_VALUE: f32 = 9.0;
const KING_VALUE: f32 = 0.0; // King's value isn't used in material counting

// Piece-square tables for positional evaluation
const PAWN_TABLE: [[f32; 8]; 8] = [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    [0.1, 0.1, 0.2, 0.3, 0.3, 0.2, 0.1, 0.1],
    [0.05, 0.05, 0.1, 0.25, 0.25, 0.1, 0.05, 0.05],
    [0.0, 0.0, 0.0, 0.2, 0.2, 0.0, 0.0, 0.0],
    [0.05, -0.05, -0.1, 0.0, 0.0, -0.1, -0.05, 0.05],
    [0.05, 0.1, 0.1, -0.2, -0.2, 0.1, 0.1, 0.05],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
];

pub fn evaluate_position(board: &Board) -> f32 {
    let mut score = 0.0;

    // Material evaluation
    for rank in 0..8 {
        for file in 0..8 {
            let square = Square::make_square(Rank::from_index(rank), File::from_index(file));
            if let Some(piece) = board.get_piece_at(square) {
                if let Some(color) = board.get_color_at(square) {
                    let piece_value = get_piece_value(piece);
                    let position_value = get_position_value(piece, rank, file, color);
                    
                    let multiplier = match color {
                        Color::White => 1.0,
                        Color::Black => -1.0,
                    };

                    score += (piece_value + position_value) * multiplier;
                }
            }
        }
    }

    score
}

fn get_piece_value(piece: Piece) -> f32 {
    match piece {
        Piece::Pawn => PAWN_VALUE,
        Piece::Knight => KNIGHT_VALUE,
        Piece::Bishop => BISHOP_VALUE,
        Piece::Rook => ROOK_VALUE,
        Piece::Queen => QUEEN_VALUE,
        Piece::King => KING_VALUE,
    }
}

fn get_position_value(piece: Piece, rank: usize, file: usize, color: Color) -> f32 {
    let (rank, file) = match color {
        Color::White => (rank, file),
        Color::Black => (7 - rank, file),
    };

    match piece {
        Piece::Pawn => PAWN_TABLE[rank][file],
        _ => 0.0, // Add more piece-square tables as needed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_position_evaluation() {
        let board = Board::new();
        let score = evaluate_position(&board);
        
        // Initial position should be equal (score close to 0)
        // Due to piece-square tables, the score might not be exactly 0
        assert!((score).abs() < 1.0);
    }

    #[test]
    fn test_piece_values() {
        assert_eq!(get_piece_value(Piece::Pawn), PAWN_VALUE);
        assert_eq!(get_piece_value(Piece::Queen), QUEEN_VALUE);
    }
} 