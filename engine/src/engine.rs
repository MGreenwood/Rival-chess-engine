use chess::{Board, ChessMove, MoveGen};
use crate::bridge::python::ModelBridge;

pub struct Engine {
    board: Board,
    model: Option<ModelBridge>,
}

impl Engine {
    pub fn new() -> Self {
        Self {
            board: Board::default(),
            model: None,
        }
    }

    pub fn new_with_model(model: ModelBridge) -> Self {
        Self {
            board: Board::default(),
            model: Some(model),
        }
    }

    pub fn board(&self) -> &Board {
        &self.board
    }

    pub fn make_move(&mut self, mv: ChessMove) {
        let new_board = self.board;
        new_board.make_move(mv, &mut self.board);
    }

    pub fn get_best_move(&self, board: &Board) -> Option<ChessMove> {
        if let Some(model) = &self.model {
            // Try to get model prediction
            match model.predict_with_board(board.to_string()) {
                Ok((policy, _value)) => {
                    // Convert policy to move probabilities
                    let legal_moves: Vec<ChessMove> = MoveGen::new_legal(board).collect();
                    if legal_moves.is_empty() {
                        return None;
                    }

                    // Find the best move based on policy
                    let mut best_move = legal_moves[0];
                    let mut best_score = 0.0f32;

                    for mv in legal_moves {
                        if let Some(move_index) = self.move_to_index(&mv) {
                            if move_index < policy.len() {
                                let score = policy[move_index];
                                if score > best_score {
                                    best_score = score;
                                    best_move = mv;
                                }
                            }
                        }
                    }

                    println!("Model selected move: {} (score: {:.4})", best_move, best_score);
                    Some(best_move)
                }
                Err(e) => {
                    println!("Model prediction failed: {}, using first legal move", e);
                    MoveGen::new_legal(board).next()
                }
            }
        } else {
            println!("No model available, using first legal move");
            MoveGen::new_legal(board).next()
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
        self.board.status() != chess::BoardStatus::Ongoing
    }

    pub fn is_check(&self) -> bool {
        self.board.checkers().popcnt() > 0
    }

    pub fn is_draw(&self) -> bool {
        matches!(self.board.status(), chess::BoardStatus::Stalemate)
    }

    pub fn is_mate(&self) -> bool {
        matches!(self.board.status(), chess::BoardStatus::Checkmate)
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