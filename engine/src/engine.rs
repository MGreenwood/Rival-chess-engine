use chess::{Board, ChessMove, MoveGen};
use crate::bridge::python::ModelBridge;
use crate::mcts::{MCTS, MCTSConfig};
use std::cell::RefCell;
use std::str::FromStr;

pub struct Engine {
    board: Board,
    model: Option<ModelBridge>,
    mcts: Option<RefCell<MCTS>>,
}

impl Engine {
    pub fn new() -> Self {
        Self {
            board: Board::default(),
            model: None,
            mcts: None,
        }
    }

    pub fn new_with_model(model: ModelBridge) -> Self {
        let config = MCTSConfig::default();
        let mcts = MCTS::new(model.clone(), config);
        
        Self {
            board: Board::default(),
            model: Some(model),
            mcts: Some(RefCell::new(mcts)),
        }
    }

    pub fn board(&self) -> &Board {
        &self.board
    }

    pub fn make_move(&mut self, mv: ChessMove) {
        let mut new_board = Board::default();
        self.board.make_move(mv, &mut new_board);
        self.board = new_board;
        
        // Ensure en-passant square is properly handled
        // The en-passant square should only be set after a two-square pawn move
        // and should be cleared after any other move
        let piece_type = self.board.piece_on(mv.get_dest());
        let from_rank = mv.get_source().get_rank().to_index();
        let to_rank = mv.get_dest().get_rank().to_index();
        
        // Check if this was a two-square pawn move
        let is_two_square_pawn_move = piece_type == Some(chess::Piece::Pawn) && 
                                    ((from_rank == 1 && to_rank == 3) || // White pawn from rank 2 to 4
                                     (from_rank == 6 && to_rank == 4));  // Black pawn from rank 7 to 5
        
        if !is_two_square_pawn_move {
            // Clear en-passant square for non-two-square pawn moves
            // This is a workaround for the Rust chess library not properly handling en-passant
            // We'll create a new board with the same position but cleared en-passant square
            let fen_string = self.board.to_string();
            let fen_parts: Vec<&str> = fen_string.split(' ').collect();
            if fen_parts.len() >= 4 {
                let mut new_fen = fen_parts[0..3].join(" "); // Board, turn, castling
                new_fen.push_str(" -"); // Clear en-passant square
                if fen_parts.len() > 4 {
                    new_fen.push_str(&format!(" {}", fen_parts[4])); // Halfmove clock
                }
                if fen_parts.len() > 5 {
                    new_fen.push_str(&format!(" {}", fen_parts[5])); // Fullmove number
                }
                
                // Try to create a new board with the corrected FEN
                if let Ok(corrected_board) = Board::from_str(&new_fen) {
                    self.board = corrected_board;
                }
            }
        }
    }

    pub fn get_best_move(&self, board: &Board) -> Option<ChessMove> {
        // First check if the game is already over
        if board.status() != chess::BoardStatus::Ongoing {
            return None;
        }

        // Get all legal moves
        let legal_moves: Vec<ChessMove> = MoveGen::new_legal(board).collect();
        if legal_moves.is_empty() {
            return None;
        }

        // Use MCTS if available, otherwise fall back to simple model prediction
        if let Some(mcts) = &self.mcts {
            match mcts.borrow_mut().get_best_move_with_time(board, std::time::Duration::from_secs(2), None) {
                Ok(best_move) => {
                    println!("MCTS selected move: {} -> {}", best_move.get_source(), best_move.get_dest());
                    Some(best_move)
                },
                Err(e) => {
                    println!("MCTS failed: {}, falling back to simple prediction", e);
                    self.get_best_move_simple(board, &legal_moves)
                }
            }
        } else {
            self.get_best_move_simple(board, &legal_moves)
        }
    }

    fn get_best_move_simple(&self, board: &Board, legal_moves: &[ChessMove]) -> Option<ChessMove> {
        if let Some(model) = &self.model {
            // Convert board to FEN string for prediction
            let board_fen = board.to_string();
            match model.predict_with_board(board_fen) {
                Ok((policy, _value)) => {
                    // Find the move with highest probability among legal moves
                    let mut best_move = legal_moves[0];
                    let mut best_prob = 0.0;
                    
                    for mv in legal_moves {
                        if let Some(idx) = self.move_to_index(mv) {
                            if idx < policy.len() && policy[idx] > best_prob {
                                best_prob = policy[idx];
                                best_move = *mv;
                            }
                        }
                    }
                    Some(best_move)
                },
                Err(e) => {
                    println!("Model prediction failed: {}, using first legal move", e);
                    Some(legal_moves[0])
                }
            }
        } else {
            println!("No model available, using first legal move");
            Some(legal_moves[0])
        }
    }

    // Convert a chess move to policy index (same logic as in MCTS)
    fn move_to_index(&self, mv: &ChessMove) -> Option<usize> {
        let from = mv.get_source().to_index();
        let to = mv.get_dest().to_index();
        let promotion = mv.get_promotion();
        
        if let Some(promotion) = promotion {
            // Promotion moves are encoded after regular moves
            // Formula: 4096 + (from_square * 64 + to_square) * 4 + promotion_piece_type - 1
            let piece_offset = match promotion {
                chess::Piece::Knight => 0,  // promotion = 1, so 1-1 = 0
                chess::Piece::Bishop => 1,  // promotion = 2, so 2-1 = 1
                chess::Piece::Rook => 2,    // promotion = 3, so 3-1 = 2
                chess::Piece::Queen => 3,   // promotion = 4, so 4-1 = 3
                _ => return None,
            };
            let base = 4096 + (from * 64 + to) * 4;
            Some(base + piece_offset)
        } else {
            // Regular moves are encoded as from * 64 + to
            Some(from * 64 + to)
        }
    }

    pub fn is_game_over(&self) -> bool {
        self.board.status() != chess::BoardStatus::Ongoing || 
        MoveGen::new_legal(&self.board).count() == 0
    }

    pub fn is_check(&self) -> bool {
        self.board.checkers().popcnt() > 0
    }

    pub fn is_draw(&self) -> bool {
        matches!(self.board.status(), chess::BoardStatus::Stalemate) ||
        MoveGen::new_legal(&self.board).count() == 0 && !self.is_check()
    }

    pub fn is_mate(&self) -> bool {
        matches!(self.board.status(), chess::BoardStatus::Checkmate) ||
        MoveGen::new_legal(&self.board).count() == 0 && self.is_check()
    }

    pub fn reset(&mut self) {
        self.board = Board::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_creation() {
        let engine = Engine::new();
        assert_eq!(engine.board(), &Board::default());
    }

    #[test]
    fn test_move_to_index() {
        let engine = Engine::new();
        
        // Test regular move
        let mv = ChessMove::new(chess::Square::E2, chess::Square::E4, None);
        let idx = engine.move_to_index(&mv);
        assert!(idx.is_some());
        assert!(idx.unwrap() < 4096); // Regular moves are in first 4096 indices
        
        // Test promotion move
        let mv = ChessMove::new(chess::Square::E7, chess::Square::E8, Some(chess::Piece::Queen));
        let idx = engine.move_to_index(&mv);
        assert!(idx.is_some());
        assert!(idx.unwrap() >= 4096); // Promotion moves start at index 4096
    }
} 