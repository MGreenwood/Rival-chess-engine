use chess::{Board, ChessMove, MoveGen, Square, Color, BoardStatus};
use std::collections::VecDeque;

pub struct Game {
    board: Board,
    move_history: VecDeque<String>,
    current_turn: Color,
}

impl Game {
    pub fn new() -> Self {
        Self {
            board: Board::default(),
            move_history: VecDeque::new(),
            current_turn: Color::White,
        }
    }

    pub fn board(&self) -> &Board {
        &self.board
    }

    pub fn make_move(&mut self, move_str: &str) -> bool {
        // Parse move string (e.g., "e2e4" or "e7e8q" for promotion)
        if move_str.len() != 4 && move_str.len() != 5 {
            return false;
        }

        let chars: Vec<char> = move_str.chars().collect();
        let from = Square::make_square(
            chess::Rank::from_index((chars[1] as u8 - b'1') as usize),
            chess::File::from_index((chars[0] as u8 - b'a') as usize),
        );
        let to = Square::make_square(
            chess::Rank::from_index((chars[3] as u8 - b'1') as usize),
            chess::File::from_index((chars[2] as u8 - b'a') as usize),
        );

        // Check if the piece being moved belongs to the current player
        if let Some(piece_color) = self.board.color_on(from) {
            if piece_color != self.current_turn {
                return false;
            }
        } else {
            return false; // No piece at source square
        }

        // Handle promotion
        let promotion = if move_str.len() == 5 {
            match chars[4].to_ascii_lowercase() {
                'q' => Some(chess::Piece::Queen),
                'r' => Some(chess::Piece::Rook),
                'b' => Some(chess::Piece::Bishop),
                'n' => Some(chess::Piece::Knight),
                _ => return false, // Invalid promotion piece
            }
        } else {
            None
        };

        let chess_move = ChessMove::new(from, to, promotion);
        
        // Validate move
        if !MoveGen::new_legal(&self.board).any(|m| m == chess_move) {
            return false;
        }

        // Make the move
        self.board = self.board.make_move_new(chess_move);
        self.move_history.push_back(move_str.to_string());
        self.current_turn = !self.current_turn;
        
        true
    }

    pub fn move_history(&self) -> Vec<String> {
        self.move_history.iter().cloned().collect()
    }

    pub fn is_game_over(&self) -> bool {
        self.board.status() != BoardStatus::Ongoing
    }

    pub fn get_game_status(&self) -> String {
        match self.board.status() {
            BoardStatus::Ongoing => {
                if self.board.checkers().popcnt() > 0 {
                    "check".to_string()
                } else {
                    "ongoing".to_string()
                }
            }
            BoardStatus::Checkmate => "checkmate".to_string(),
            BoardStatus::Stalemate => "stalemate".to_string(),
            // The chess crate only has these three status variants
            // Any other status would be a bug in the chess crate
        }
    }

    pub fn current_turn(&self) -> Color {
        self.current_turn
    }
} 