use chess::{Board, ChessMove, MoveGen};
use crate::bridge::python::ModelBridge;
use crate::mcts::{MCTS, MCTSConfig};
use rand::{thread_rng, Rng};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

const MAX_POLICY_SIZE: usize = 5312;  // 4096 regular moves + 1216 promotion moves

#[derive(Clone)]
pub struct Engine {
    board: Board,
    model: ModelBridge,
    temperature: f32,
    strength: f32,
    mcts: Option<Arc<Mutex<MCTS>>>,
}

impl Engine {
    pub fn new_with_model(model: ModelBridge) -> Self {
        Self {
            board: Board::default(),
            model,
            temperature: 0.8,  // Slightly lower temperature for more focused play
            strength: 1.0,     // Default strength (full strength)
            mcts: None,
        }
    }

    pub fn new_with_mcts(model: ModelBridge) -> Self {
        let mcts_config = MCTSConfig {
            num_simulations: 500,   // Reduced for faster community games  
            temperature: 0.1,       // Low temperature for strong play
            c_puct: 1.25,           // Slightly more exploitative
            ..MCTSConfig::default()
        };
        
        Self {
            board: Board::default(),
            model: model.clone(),
            temperature: 0.1,       // Low temperature for deterministic strong play
            strength: 1.0,
            mcts: Some(Arc::new(Mutex::new(MCTS::new(model, mcts_config)))),
        }
    }

    pub fn set_temperature(&mut self, temp: f32) {
        self.temperature = temp.max(0.01).min(2.0);  // Clamp between 0.01 and 2.0
    }

    pub fn set_strength(&mut self, strength: f32) {
        self.strength = strength.max(0.0).min(1.0);  // Clamp between 0 and 1
    }

    pub fn board(&self) -> Board {
        self.board
    }

    pub fn make_move(&mut self, mv: ChessMove) {
        self.board = self.board.make_move_new(mv);
    }

    fn softmax(values: &[f32], temperature: f32) -> Vec<f32> {
        if values.is_empty() {
            return vec![];
        }

        let max_val = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_values: Vec<f32> = values
            .iter()
            .map(|&x| ((x - max_val) / temperature).exp())
            .collect();
        let sum: f32 = exp_values.iter().sum();
        if sum == 0.0 {
            // If all values are very negative, return uniform distribution
            return vec![1.0 / values.len() as f32; values.len()];
        }
        exp_values.into_iter().map(|x| x / sum).collect()
    }

    pub fn get_best_move(&self, board: Board) -> Option<ChessMove> {
        // Use MCTS if available (for community engine)
        if let Some(mcts_arc) = &self.mcts {
            let mcts_result: Result<MutexGuard<'_, MCTS>, PoisonError<MutexGuard<'_, MCTS>>> = match mcts_arc.lock() {
                Ok(mcts) => Ok(mcts),
                Err(poisoned) => {
                    eprintln!("⚠️ MCTS mutex was poisoned, recovering...");
                    Ok(poisoned.into_inner())
                }
            };

            if let Ok(mut mcts) = mcts_result {
                // Use time-based MCTS for community games (max 10 seconds to prevent blocking training)
                let timeout = std::time::Duration::from_secs(10);
                match mcts.get_best_move_with_time(&board, timeout) {
                    Ok(mv) => {
                        eprintln!("🧠 MCTS selected move: {} (max 10s timeout)", mv);
                        return Some(mv);
                    }
                    Err(e) => {
                        eprintln!("⚠️ MCTS failed, falling back to direct policy: {}", e);
                        // Fall through to direct policy evaluation
                    }
                }
            }
        }

        // Direct policy evaluation (for single player engine or fallback)
        match self.model.predict_with_board(board.to_string()) {
            Ok((policy, _)) => {
                // Validate policy size
                if policy.len() != MAX_POLICY_SIZE {
                    eprintln!("Invalid policy size: {}, expected {}", policy.len(), MAX_POLICY_SIZE);
                    return self.get_random_move(&board);
                }

                let mut moves = Vec::new();
                let mut probs = Vec::new();

                // Collect legal moves and their probabilities
                for mv in MoveGen::new_legal(&board) {
                    if let Some(idx) = self.model.move_to_policy_idx(mv) {
                        if idx < policy.len() {
                            moves.push(mv);
                            probs.push(policy[idx].max(0.0));  // Ensure non-negative probability
                        }
                    }
                }

                if moves.is_empty() {
                    return None;
                }

                // Apply temperature scaling with better focus
                let scaled_probs = Self::softmax(&probs, self.temperature);

                // Apply strength scaling
                let mut rng = thread_rng();
                if rng.gen::<f32>() > self.strength {
                    // Make a random move with probability (1 - strength)
                    return self.get_random_move(&board);
                }

                // Use more deterministic selection for stronger play
                if self.temperature < 0.1 {
                    // Deterministic: pick the best move
                    let best_idx = scaled_probs
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(i, _)| i)
                        .unwrap_or(0);
                    return Some(moves[best_idx]);
                }

                // Sample move based on probabilities
                let mut cumsum = 0.0;
                let r = rng.gen::<f32>();
                for (i, &p) in scaled_probs.iter().enumerate() {
                    cumsum += p;
                    if r <= cumsum {
                        return Some(moves[i]);
                    }
                }

                // Fallback to highest probability move
                let best_idx = scaled_probs
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                Some(moves[best_idx])
            }
            Err(e) => {
                eprintln!("Failed to get model prediction: {}", e);
                self.get_random_move(&board)
            }
        }
    }

    fn get_random_move(&self, board: &Board) -> Option<ChessMove> {
        let mut rng = thread_rng();
        let moves: Vec<ChessMove> = MoveGen::new_legal(board).collect();
        if moves.is_empty() {
            None
        } else {
            let idx = rng.gen_range(0..moves.len());
            Some(moves[idx])
        }
    }

    pub fn is_game_over(&self) -> bool {
        self.board.status() != chess::BoardStatus::Ongoing
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
        let engine = Engine::new_with_model(ModelBridge::new());
        assert_eq!(engine.board(), Board::default());
    }

    #[test]
    fn test_move_to_index() {
        let engine = Engine::new_with_model(ModelBridge::new());
        
        // Test regular move
        let mv = ChessMove::new(chess::Square::E2, chess::Square::E4, None);
        let idx = engine.model.move_to_policy_idx(&mv);
        assert!(idx.is_some());
        assert!(idx.unwrap() < 4096); // Regular moves are in first 4096 indices
        
        // Test promotion move
        let mv = ChessMove::new(chess::Square::E7, chess::Square::E8, Some(chess::Piece::Queen));
        let idx = engine.model.move_to_policy_idx(&mv);
        assert!(idx.is_some());
        assert!(idx.unwrap() >= 4096); // Promotion moves start at index 4096
    }

    #[test]
    fn test_softmax() {
        let engine = Engine::new_with_model(ModelBridge::new());
        let values = vec![1.0, 2.0, 3.0];
        let probs = Engine::softmax(&values, 1.0);
        
        // Check probabilities sum to 1
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        
        // Check ordering is preserved
        for i in 1..probs.len() {
            assert!(probs[i] > probs[i-1]);
        }
    }
} 