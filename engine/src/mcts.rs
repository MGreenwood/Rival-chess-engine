// MCTS for engine/inference: always uses a fresh tree, no node cache, no batch, no locking, no parallel node access.
// This is different from the training MCTS, which may use persistent node cache and parallelism.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use std::cmp::Ordering;
use std::sync::atomic::{AtomicU32, Ordering as AtomicOrdering, AtomicBool};
use rand::prelude::*;
use anyhow::Result;
use chess::{Board, ChessMove, MoveGen, BoardStatus, Color};
use crate::ModelBridge;

// Constants
const DEFAULT_NUM_SIMULATIONS: u32 = 2500;
const DEFAULT_TEMPERATURE: f32 = 1.0;
const DEFAULT_C_PUCT: f32 = 1.0;
const DEFAULT_DIRICHLET_ALPHA: f32 = 0.3;
const DEFAULT_DIRICHLET_WEIGHT: f32 = 0.25;
const DEFAULT_REPETITION_PENALTY: f32 = 0.1;
const DEFAULT_FORWARD_PROGRESS_BONUS: f32 = 0.01;
const DEFAULT_RANDOM_COMPONENT_STD: f32 = 0.1;
const DEFAULT_MIN_PIECES_FOR_REPETITION: u32 = 7;
const DEFAULT_REPETITION_HISTORY_SIZE: u32 = 8;
const DEFAULT_REPETITION_THRESHOLD: u32 = 3;
const DEFAULT_NUM_THREADS: u32 = 1;
const DEFAULT_BATCH_SIZE: u32 = 1;
const VIRTUAL_LOSS: u32 = 3; // Virtual loss for parallel MCTS

#[derive(Debug, Clone)]
pub struct MCTSConfig {
    pub num_simulations: u32,
    pub temperature: f32,
    pub c_puct: f32,
    pub dirichlet_alpha: f32,
    pub dirichlet_weight: f32,
    pub repetition_penalty: f32,
    pub forward_progress_bonus: f32,
    pub random_component_std: f32,
    pub min_pieces_for_repetition: u32,
    pub repetition_history_size: u32,
    pub repetition_threshold: u32,
    pub num_threads: u32,
    pub batch_size: u32,
    pub use_virtual_loss: bool,
}

impl Default for MCTSConfig {
    fn default() -> Self {
        Self {
            num_simulations: DEFAULT_NUM_SIMULATIONS,
            temperature: DEFAULT_TEMPERATURE,
            c_puct: DEFAULT_C_PUCT,
            dirichlet_alpha: DEFAULT_DIRICHLET_ALPHA,
            dirichlet_weight: DEFAULT_DIRICHLET_WEIGHT,
            repetition_penalty: DEFAULT_REPETITION_PENALTY,
            forward_progress_bonus: DEFAULT_FORWARD_PROGRESS_BONUS,
            random_component_std: DEFAULT_RANDOM_COMPONENT_STD,
            min_pieces_for_repetition: DEFAULT_MIN_PIECES_FOR_REPETITION,
            repetition_history_size: DEFAULT_REPETITION_HISTORY_SIZE,
            repetition_threshold: DEFAULT_REPETITION_THRESHOLD,
            num_threads: DEFAULT_NUM_THREADS,
            batch_size: DEFAULT_BATCH_SIZE,
            use_virtual_loss: true,
        }
    }
}

struct Node {
    children: Mutex<HashMap<ChessMove, Arc<Node>>>,
    visit_count: AtomicU32,
    virtual_loss: AtomicU32,
    value_sum: AtomicU32, // Store as integer to avoid floating point atomic issues
    prior: f32,
    is_expanded: AtomicBool,
}

impl Node {
    fn new(prior: f32) -> Self {
        Self {
            children: Mutex::new(HashMap::new()),
            visit_count: AtomicU32::new(0),
            virtual_loss: AtomicU32::new(0),
            value_sum: AtomicU32::new(0),
            prior,
            is_expanded: AtomicBool::new(false),
        }
    }

    fn get_value(&self) -> f32 {
        let visits = self.visit_count.load(AtomicOrdering::Relaxed);
        if visits == 0 {
            return 0.0;
        }
        let value_sum = self.value_sum.load(AtomicOrdering::Relaxed);
        f32::from_bits(value_sum) / visits as f32
    }

    fn add_virtual_loss(&self) {
        self.virtual_loss.fetch_add(VIRTUAL_LOSS, AtomicOrdering::Relaxed);
    }

    fn remove_virtual_loss(&self) {
        self.virtual_loss.fetch_sub(VIRTUAL_LOSS, AtomicOrdering::Relaxed);
    }

    fn get_ucb_score(&self, parent_visit_sqrt: f32, c_puct: f32) -> f32 {
        let visits = self.visit_count.load(AtomicOrdering::Relaxed) as f32;
        let virtual_loss = self.virtual_loss.load(AtomicOrdering::Relaxed) as f32;
        
        if visits + virtual_loss == 0.0 {
            return f32::INFINITY;
        }

        let q_value = -self.get_value();
        let u_value = c_puct * self.prior * parent_visit_sqrt / (1.0 + visits + virtual_loss);
        
        q_value + u_value
    }

    fn select_child(&self, config: &MCTSConfig) -> Option<(ChessMove, Arc<Node>)> {
        if self.children.lock().unwrap().is_empty() {
            return None;
        }

        let parent_visit_sqrt = (self.visit_count.load(AtomicOrdering::Relaxed) as f32).sqrt();

        self.children.lock().unwrap().iter()
            .max_by(|(_, a), (_, b)| {
                let a_score = a.get_ucb_score(parent_visit_sqrt, config.c_puct);
                let b_score = b.get_ucb_score(parent_visit_sqrt, config.c_puct);
                a_score.partial_cmp(&b_score).unwrap_or(Ordering::Equal)
            })
            .map(|(mv, node)| (*mv, Arc::clone(node)))
    }

    fn backup(&self, value: f32) {
        self.visit_count.fetch_add(1, AtomicOrdering::Relaxed);
        // Convert value to integer for atomic operations
        let value_int = (value * 1000.0) as u32;
        self.value_sum.fetch_add(value_int, AtomicOrdering::Relaxed);
    }

    fn get_policy(&self, temperature: f32) -> HashMap<ChessMove, f32> {
        if self.children.lock().unwrap().is_empty() {
            return HashMap::new();
        }

        let mut policy = HashMap::new();
        let visits: Vec<_> = self.children.lock().unwrap().iter()
            .map(|(mv, node)| (*mv, node.visit_count.load(AtomicOrdering::Relaxed) as f32))
            .collect();

        if temperature <= 0.01 {
            // Use deterministic selection
            let best_move = visits.iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
                .map(|(mv, _)| *mv)
                .unwrap();
            policy.insert(best_move, 1.0);
        } else {
            // Use temperature-based selection
            let mut sum = 0.0;
            for (mv, visits) in visits {
                let prob = (visits / temperature).exp();
                policy.insert(mv, prob);
                sum += prob;
            }

            // Normalize probabilities
            if sum > 0.0 {
                for prob in policy.values_mut() {
                    *prob /= sum;
                }
            } else {
                // Fallback to uniform distribution
                let uniform_prob = 1.0 / self.children.lock().unwrap().len() as f32;
                for prob in policy.values_mut() {
                    *prob = uniform_prob;
                }
            }
        }

        policy
    }
}

pub struct MCTS {
    config: MCTSConfig,
    model: ModelBridge,
    rng: Mutex<StdRng>, // Only keep RNG for move selection
}

impl MCTS {
    pub fn new(model: ModelBridge, config: MCTSConfig) -> Self {
        Self {
            config,
            model,
            rng: Mutex::new(StdRng::from_entropy()),
        }
    }

    pub fn get_best_move(&mut self, board: &Board, temperature: Option<f32>) -> Result<ChessMove> {
        let policy = self.search(board)?;
        
        if policy.is_empty() {
            return Err(anyhow::anyhow!("No legal moves available"));
        }
        
        let temp = temperature.unwrap_or(self.config.temperature);
        let mut rng = self.rng.lock().map_err(|e| anyhow::anyhow!("Failed to acquire RNG lock: {}", e))?;
        
        if temp <= 0.01 {
            // Deterministic selection
            let best_move = policy.iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
                .map(|(mv, _)| *mv)
                .unwrap();
            Ok(best_move)
        } else {
            // Temperature-based selection
            let moves: Vec<_> = policy.into_iter().collect();
            let total_prob: f32 = moves.iter().map(|(_, prob)| prob).sum();
            
            if total_prob <= 0.0 {
                return Err(anyhow::anyhow!("Invalid policy probabilities"));
            }
            
            let mut cumsum = 0.0;
            let rand_val: f32 = rng.gen();
            
            for (mv, prob) in &moves {
                cumsum += prob / total_prob;
                if rand_val <= cumsum {
                    return Ok(*mv);
                }
            }
            
            // Fallback to first move
            Ok(moves[0].0)
        }
    }

    pub fn get_best_move_with_time(&mut self, board: &Board, time_limit: Duration) -> Result<ChessMove> {
        let root = Arc::new(Node::new(0.0));
        let start_time = Instant::now();
        let mut simulation_count = 0;
        
        // Run simulations until time limit is reached
        while start_time.elapsed() < time_limit {
            self.run_simulation(board, Arc::clone(&root))?;
            simulation_count += 1;
            
            // Safety check to prevent infinite loops
            if simulation_count > 100_000 {
                break;
            }
        }
        
        println!("MCTS completed {} simulations in {:?}", simulation_count, start_time.elapsed());
        
        // Get policy from root node and select best move
        let temperature = self.config.temperature;
        let policy = root.get_policy(temperature);
        
        if policy.is_empty() {
            return Err(anyhow::anyhow!("No legal moves available"));
        }
        
        // Select best move from policy
        let best_move = policy.iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(mv, _)| *mv)
            .unwrap();
            
        Ok(best_move)
    }
    
    pub fn get_move_probabilities(&mut self, board: &Board) -> Result<Vec<(ChessMove, f32)>> {
        let policy = self.search(board)?;
        Ok(policy.into_iter().collect())
    }
    
    pub fn get_principal_variation(&mut self, board: &Board, max_depth: Option<u32>) -> Result<Vec<ChessMove>> {
        let mut pv = Vec::new();
        let mut current_board = board.clone();
        let depth = max_depth.unwrap_or(self.config.num_simulations);
        
        for _ in 0..depth {
            if current_board.status() != chess::BoardStatus::Ongoing {
                break;
            }
            
            let policy = self.search(&current_board)?;
            if policy.is_empty() {
                break;
            }
            
            let best_move = policy.iter()
                .max_by(|(_, &a), (_, &b)| a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(mv, _)| *mv)
                .unwrap();
            
            pv.push(best_move);
            let mut new_board = Board::default();
            current_board.make_move(best_move, &mut new_board);
            current_board = new_board;
        }
        
        Ok(pv)
    }
    
    pub fn search(&mut self, board: &Board) -> Result<HashMap<ChessMove, f32>> {
        let root = Arc::new(Node::new(0.0));
        
        // Run simulations
        for _ in 0..self.config.num_simulations {
            self.run_simulation(board, Arc::clone(&root))?;
        }
        
        // Get policy from root node
        let temperature = self.config.temperature;
        Ok(root.get_policy(temperature))
    }

    pub fn search_with_time(&mut self, board: &Board, time_limit: Duration) -> Result<HashMap<ChessMove, f32>> {
        let root = Arc::new(Node::new(0.0));
        let start_time = Instant::now();
        let mut simulation_count = 0;
        
        // Run simulations until time limit is reached
        while start_time.elapsed() < time_limit {
            self.run_simulation(board, Arc::clone(&root))?;
            simulation_count += 1;
            
            // Safety check to prevent infinite loops
            if simulation_count > 100_000 {
                break;
            }
        }
        
        println!("MCTS completed {} simulations in {:?}", simulation_count, start_time.elapsed());
        
        // Get policy from root node
        let temperature = self.config.temperature;
        Ok(root.get_policy(temperature))
    }

    fn run_simulation(&self, board: &Board, root: Arc<Node>) -> Result<()> {
        let mut current_board = board.clone();
        let mut current_node = root;
        let mut path = vec![];

        // Selection phase
        while current_node.is_expanded.load(AtomicOrdering::Relaxed) && !self.is_terminal(&current_board) {
            let (selected_move, selected_node) = match current_node.select_child(&self.config) {
                Some((mv, node)) => (mv, node),
                None => break,
            };

            if self.config.use_virtual_loss {
                selected_node.add_virtual_loss();
            }

            path.push((selected_move, Arc::clone(&selected_node)));
            let mut next_board = Board::default();
            current_board.make_move(selected_move, &mut next_board);
            current_board = next_board;
            current_node = selected_node;
        }

        // Expansion and evaluation
        let value = if self.is_terminal(&current_board) {
            self.get_terminal_value(&current_board)
        } else if !current_node.is_expanded.load(AtomicOrdering::Relaxed) {
            let (policy, value) = self.evaluate_position(&current_board)?;
            let moves = self.get_legal_moves_with_probabilities(&current_board, &policy);
            
            // Expand the node using interior mutability
            {
                let mut children = current_node.children.lock().unwrap();
                for (mv, prior) in &moves {
                    children.insert(*mv, Arc::new(Node::new(*prior)));
                }
            }
            current_node.is_expanded.store(true, AtomicOrdering::Relaxed);
            current_node.visit_count.fetch_add(1, AtomicOrdering::Relaxed);
            value
        } else {
            0.0 // Should not happen
        };

        // Backpropagation
        for (_, node) in path.iter().rev() {
            if self.config.use_virtual_loss {
                node.remove_virtual_loss();
            }
            node.backup(value);
        }

        Ok(())
    }

    fn get_terminal_value(&self, board: &Board) -> f32 {
        match board.status() {
            BoardStatus::Checkmate => {
                if board.side_to_move() == Color::White {
                    -1.0 // Black wins
                } else {
                    1.0 // White wins
                }
            }
            _ => 0.0 // Draw
        }
    }

    fn is_terminal(&self, board: &Board) -> bool {
        board.status() != BoardStatus::Ongoing
    }

    fn get_legal_moves_with_probabilities(&self, board: &Board, policy: &Vec<f32>) -> Vec<(ChessMove, f32)> {
        let mut moves = Vec::new();
        let legal_moves: Vec<_> = MoveGen::new_legal(board).collect();
        let policy_size = policy.len();
        let policy_slice = &policy[0..policy_size];

        for mv in legal_moves {
            if let Some(idx) = self.move_to_index(&mv) {
                // Ensure the index is within bounds
                if idx < policy_size {
                    moves.push((mv, policy_slice[idx]));
                } else {
                    // If index is out of bounds, use a small default probability
                    moves.push((mv, 0.001));
                }
            } else {
                // If move_to_index returns None, use a small default probability
                moves.push((mv, 0.001));
            }
        }

        // Normalize probabilities
        let sum: f32 = moves.iter().map(|(_, p)| *p).sum();
        if sum > 0.0 {
            for (_, p) in &mut moves {
                *p /= sum;
            }
        } else {
            let uniform_prob = 1.0 / moves.len() as f32;
            for (_, p) in &mut moves {
                *p = uniform_prob;
            }
        }

        moves
    }

    fn evaluate_position(&self, board: &Board) -> Result<(Vec<f32>, f32)> {
        let _hash = board.get_hash();
        
        // Evaluate with model
        let (policy_vec, value) = self.model.predict_with_board(board.to_string())?;
        let policy: Vec<f32> = policy_vec.into_iter().collect();
        
        Ok((policy, value))
    }
    
    fn move_to_index(&self, mv: &ChessMove) -> Option<usize> {
        let from = mv.get_source().to_index() as usize;
        let to = mv.get_dest().to_index() as usize;
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
            let index = base + piece_offset;
            
            // Ensure the index is within bounds
            if index < 5312 {
                Some(index)
            } else {
                None
            }
        } else {
            // Regular moves are encoded as from * 64 + to
            let index = from * 64 + to;
            
            // Ensure the index is within bounds
            if index < 4096 {
                Some(index)
            } else {
                None
            }
        }
    }
}

#[derive(Debug)]
pub struct NodeInfo {
    pub visit_count: u32,
    pub value: f32,
    pub is_expanded: bool,
    pub children_info: HashMap<ChessMove, ChildInfo>,
}

#[derive(Debug)]
pub struct ChildInfo {
    pub prior: f32,
    pub visit_count: u32,
    pub value: f32,
}

// Implement Send and Sync for MCTS
unsafe impl Send for MCTS {}
unsafe impl Sync for MCTS {}

#[cfg(test)]
mod tests {
    use super::*;
    use chess::Square;
    
    #[test]
    fn test_move_indexing() {
        let mcts = MCTS::new(ModelBridge::new(Board::default(), None), MCTSConfig::default());
        
        // Test regular move
        let mv = ChessMove::new(Square::E2, Square::E4, None);
        let idx = mcts.move_to_index(&mv);
        assert!(idx.is_some());
        assert!(idx.unwrap() < 4096); // Regular moves are in first 4096 indices
        
        // Test promotion move
        let mv = ChessMove::new(Square::E7, Square::E8, Some(chess::Piece::Queen));
        let idx = mcts.move_to_index(&mv);
        assert!(idx.is_some());
        assert!(idx.unwrap() >= 4096); // Promotion moves start at index 4096
    }
    
    #[test]
    fn test_move_selection() {
        let mcts = MCTS::new(ModelBridge::new(Board::default(), None), MCTSConfig::default());
        let board = Board::default();
        
        // Test deterministic selection
        let best_move = mcts.get_best_move(&board, Some(0.0)).unwrap();
        assert!(MoveGen::new_legal(&board).any(|mv| mv == best_move));
        
        // Test temperature-based selection
        let moves: Vec<_> = (0..10)
            .map(|_| mcts.get_best_move(&board, Some(1.0)).unwrap())
            .collect();
        
        // With temperature > 0, we should see some variation in moves
        let unique_moves = moves.iter().collect::<std::collections::HashSet<_>>();
        assert!(unique_moves.len() > 1);
    }
    
    #[test]
    fn test_principal_variation() {
        let mcts = MCTS::new(ModelBridge::new(Board::default(), None), MCTSConfig::default());
        let board = Board::default();
        
        let pv = mcts.get_principal_variation(&board, Some(5)).unwrap();
        assert!(!pv.is_empty());
        assert!(pv.len() <= 5);
        
        // Verify moves are legal
        let mut current_board = board;
        for mv in pv {
            assert!(MoveGen::new_legal(&current_board).any(|m| m == mv));
            current_board.make_move(mv);
        }
    }
} 