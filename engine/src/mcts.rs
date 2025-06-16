use std::collections::HashMap;
use std::time::{Duration, Instant};
use std::hash::{Hash, Hasher};
use rand::prelude::*;
use rand_distr::Dirichlet;
use anyhow::Result;
use parking_lot::RwLock;
use tch::Tensor;
use rayon::prelude::*;
use std::sync::Arc;

use chess::{Board, ChessMove, MoveGen};
use crate::bridge::python::ModelBridge;
use crate::board::PAGBoard;

// Constants
const MOVE_OVERHEAD: Duration = Duration::from_millis(100);
const MAX_LEGAL_MOVES: usize = 256;
const POLICY_SIZE: usize = 5312; // Total number of possible moves including promotions

#[derive(Debug, Clone)]
pub struct MCTSConfig {
    pub num_simulations: u32,
    pub c_puct: f32,
    pub dirichlet_alpha: f32,
    pub dirichlet_weight: f32,
    pub temperature: f32,
    pub batch_size: usize,
    pub max_batch_size: usize,
    pub max_parallel_searches: usize,
    pub max_depth: u32,
    pub max_time: Duration,
    pub cache_size: usize,
}

impl Default for MCTSConfig {
    fn default() -> Self {
        Self {
            num_simulations: 800,
            c_puct: 1.0,
            dirichlet_alpha: 0.3,
            dirichlet_weight: 0.25,
            temperature: 1.0,
            batch_size: 32,
            max_batch_size: 256,
            max_parallel_searches: 8,
            max_depth: 100,
            max_time: Duration::from_secs(1),
            cache_size: 1_000_000,
        }
    }
}

// Create a wrapper type for Board to implement Hash
#[derive(Clone)]
struct BoardHash(Board);

impl Hash for BoardHash {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.get_hash().hash(state);
    }
}

#[derive(Clone)]
struct Node {
    prior: f32,
    visit_count: u32,
    value_sum: f32,
    children: HashMap<ChessMove, Node>,
    is_expanded: bool,
    position_count: u32,  // Track how many times this position has occurred
}

impl Node {
    fn new(prior: f32) -> Self {
        Self {
            prior,
            visit_count: 0,
            value_sum: 0.0,
            children: HashMap::new(),
            is_expanded: false,
            position_count: 1,  // Initialize to 1 since this is the first occurrence
        }
    }

    fn value(&self) -> f32 {
        if self.visit_count == 0 {
            0.0
        } else {
            // Apply repetition penalty
            let repetition_penalty = if self.position_count > 1 {
                -0.1 * (self.position_count - 1) as f32  // Penalize repeated positions
            } else {
                0.0
            };
            self.value_sum / self.visit_count as f32 + repetition_penalty
        }
    }

    fn expand(&mut self, moves: &[(ChessMove, f32)]) {
        self.is_expanded = true;
        for &(mv, prior) in moves {
            self.children.insert(mv, Node::new(prior));
        }
    }

    fn get_policy(&self, temperature: f32) -> HashMap<ChessMove, f32> {
        if self.children.is_empty() {
            return HashMap::new();
        }

        let mut policy = HashMap::new();
        let total_visits: f32 = self.children.values()
            .map(|child| child.visit_count as f32)
            .sum();

        if total_visits == 0.0 {
            let prob = 1.0 / self.children.len() as f32;
            for mv in self.children.keys() {
                policy.insert(*mv, prob);
            }
            return policy;
        }

        // Apply temperature
        if temperature == 0.0 {
            let best_move = self.children.iter()
                .max_by_key(|(_, child)| child.visit_count)
                .map(|(mv, _)| *mv)
                .unwrap();
            policy.insert(best_move, 1.0);
        } else {
            for (mv, child) in &self.children {
                let count = child.visit_count as f32;
                let prob = (count / temperature).exp();
                policy.insert(*mv, prob);
            }
            
            let sum: f32 = policy.values().sum();
            for prob in policy.values_mut() {
                *prob /= sum;
            }
        }

        policy
    }
}

pub struct MCTS {
    config: MCTSConfig,
    model: ModelBridge,
    cache: RwLock<HashMap<u64, (Tensor, f32)>>, // Maps board hash to (policy, value)
    nodes: RwLock<HashMap<u64, Node>>,
}

impl MCTS {
    pub fn new(model: ModelBridge, config: MCTSConfig) -> Self {
        Self {
            config,
            model,
            cache: RwLock::new(HashMap::new()),
            nodes: RwLock::new(HashMap::new()),
        }
    }

    pub fn get_best_move(&mut self, board: &Board, temperature: Option<f32>) -> Result<ChessMove> {
        let policy = self.search(board)?;
        let temp = temperature.unwrap_or(self.config.temperature);
        
        if policy.is_empty() {
            return Err(anyhow::anyhow!("No legal moves available"));
        }
        
        if temp == 0.0 {
            // Deterministic selection - choose highest probability move
            Ok(*policy.iter()
                .max_by(|(_, &a), (_, &b)| a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(mv, _)| mv)
                .unwrap())
        } else {
            // Sample move based on probabilities
            let moves: Vec<_> = policy.keys().collect();
            let probs: Vec<_> = policy.values().map(|&p| (p / temp).exp()).collect();
            let sum: f32 = probs.iter().sum();
            let normalized: Vec<_> = probs.iter().map(|&p| p / sum).collect();
            
            let mut rng = thread_rng();
            let mut cumsum = 0.0;
            let r = rng.gen::<f32>();
            
            for (i, &p) in normalized.iter().enumerate() {
                cumsum += p;
                if r <= cumsum {
                    return Ok(*moves[i]);
                }
            }
            
            // Fallback to highest probability move
            Ok(*moves[0])
        }
    }
    
    pub fn get_move_probabilities(&mut self, board: &Board) -> Result<Vec<(ChessMove, f32)>> {
        let policy = self.search(board)?;
        Ok(policy.into_iter().collect())
    }
    
    pub fn get_principal_variation(&mut self, board: &Board, max_depth: Option<u32>) -> Result<Vec<ChessMove>> {
        let mut pv = Vec::new();
        let mut current_board = board.clone();
        let depth = max_depth.unwrap_or(self.config.max_depth);
        
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
            let mut new_board = current_board.clone();
            current_board.make_move(best_move, &mut new_board);
            current_board = new_board;
        }
        
        Ok(pv)
    }
    
    pub fn get_node_info(&self, board: &Board) -> Result<NodeInfo> {
        let hash = board.get_hash();
        if let Some(node) = self.nodes.read().get(&hash) {
            Ok(NodeInfo {
                visit_count: node.visit_count,
                value: node.value(),
                is_expanded: node.is_expanded,
                children_info: node.children.iter()
                    .map(|(mv, child)| (
                        *mv,
                        ChildInfo {
                            prior: child.prior,
                            visit_count: child.visit_count,
                            value: child.value(),
                        }
                    ))
                    .collect(),
            })
        } else {
            Err(anyhow::anyhow!("Node not found"))
        }
    }
    
    pub fn clear_cache(&mut self) {
        self.cache.write().clear();
        self.nodes.write().clear();
    }
    
    pub fn get_cache_stats(&self) -> CacheStats {
        let cache = self.cache.read();
        let nodes = self.nodes.read();
        
        CacheStats {
            evaluation_cache_size: cache.len(),
            node_cache_size: nodes.len(),
            memory_usage: (cache.len() + nodes.len()) * std::mem::size_of::<Node>(),
        }
    }

    pub fn search(&mut self, board: &Board) -> Result<HashMap<ChessMove, f32>> {
        let start_time = Instant::now();
        let root_hash = board.get_hash();
        
        // Initialize root node if not exists
        if !self.nodes.read().contains_key(&root_hash) {
            let (_policy, _value) = self.evaluate_position(board)?;
            let mut moves = Vec::new();
            
            for mv in MoveGen::new_legal(board) {
                let idx = self.move_to_index(&mv);
                if let Some(idx) = idx {
                    let prior = _policy.double_value(&[idx as i64]);
                    moves.push((mv, prior as f32));
                }
            }
            
            // Create root node
            let mut root = Node::new(0.0); // Prior doesn't matter for root
            root.expand(&moves);
            
            // Apply Dirichlet noise to root node
            if self.config.dirichlet_weight > 0.0 {
                let noise = self.generate_dirichlet_noise(moves.len());
                let mut moves_mut: Vec<(ChessMove, f32)> = moves.clone();
                for (j, (_, prior)) in moves_mut.iter_mut().enumerate() {
                    *prior = (1.0 - self.config.dirichlet_weight) * *prior + 
                            self.config.dirichlet_weight * noise[j];
                }
                root.expand(&moves_mut);
            }
            
            self.nodes.write().insert(root_hash, root);
        }
        
        // Run simulations
        for _ in 0..self.config.num_simulations {
            if start_time.elapsed() > self.config.max_time {
                break;
            }
            
            let mut path = Vec::new();
            let mut current_board = board.clone();
            let mut current_hash = root_hash;
            
            // Selection phase
            while let Some(node) = self.nodes.read().get(&current_hash) {
                if !node.is_expanded {
                    break;
                }
                
                // Find best child according to PUCT formula
                let mut best_score = f32::NEG_INFINITY;
                let mut best_move = None;
                let parent_visits = (node.visit_count as f32).sqrt();
                
                for (mv, child) in &node.children {
                    let q_value = -child.value(); // Negamax
                    let u_value = self.config.c_puct * child.prior * parent_visits / 
                                (1.0 + child.visit_count as f32);
                    let puct = q_value + u_value;
                    
                    if puct > best_score {
                        best_score = puct;
                        best_move = Some(*mv);
                    }
                }
                
                if let Some(mv) = best_move {
                    path.push((current_hash, mv));
                    let mut new_board = current_board.clone();
                    current_board.make_move(mv, &mut new_board);
                    current_board = new_board;
                    current_hash = current_board.get_hash();
                } else {
                    break;
                }
            }
            
            // Expansion phase
            let mut value = if current_board.status() == chess::BoardStatus::Ongoing {
                let (_policy, node_value) = self.evaluate_position(&current_board)?;
                let mut moves = Vec::new();
                
                for mv in MoveGen::new_legal(&current_board) {
                    let idx = self.move_to_index(&mv);
                    if let Some(idx) = idx {
                        let prior = _policy.double_value(&[idx as i64]);
                        moves.push((mv, prior as f32));
                    }
                }
                
                let mut root = Node::new(0.0);
                let moves: Vec<(ChessMove, f32)> = moves.iter().map(|(mv, prior)| (*mv, *prior)).collect();
                root.expand(&moves);
                
                // Update position count
                if let Some(existing_node) = self.nodes.read().get(&current_hash) {
                    root.position_count = existing_node.position_count + 1;
                }
                
                self.nodes.write().insert(current_hash, root);
                
                node_value
            } else {
                // Game is over
                match current_board.status() {
                    chess::BoardStatus::Checkmate => -1.0,
                    _ => 0.0, // Draw
                }
            };
            
            // Backpropagation phase
            for (hash, _) in path.iter().rev() {
                if let Some(node) = self.nodes.write().get_mut(hash) {
                    node.visit_count += 1;
                    node.value_sum += value;
                    value = -value; // Negamax
                }
            }
        }
        
        // Return normalized visit counts as policy
        if let Some(root) = self.nodes.read().get(&root_hash) {
            Ok(root.get_policy(self.config.temperature))
        } else {
            Ok(HashMap::new())
        }
    }
    
    fn evaluate_position(&self, board: &Board) -> Result<(Tensor, f32)> {
        let hash = board.get_hash();
        let cached_result = self.cache.read().get(&hash).map(|(policy, value)| (policy.shallow_clone(), *value));
        
        // Check for repetition
        let position_count = if let Some(node) = self.nodes.read().get(&hash) {
            node.position_count
        } else {
            1
        };

        if let Some((policy, mut value)) = cached_result {
            // Apply repetition penalty to cached value
            if position_count > 1 {
                value -= 0.1 * (position_count - 1) as f32;
            }
            return Ok((policy, value));
        }
        
        let _pag = PAGBoard::from_board(board);
        let (policy_vec, mut value) = self.model.predict_with_board(board.to_string())?;
        
        // Apply repetition penalty
        if position_count > 1 {
            value -= 0.1 * (position_count - 1) as f32;
        }
        
        // Add forward progress bonus in opening
        if board.fullmove_number() <= 10 {
            let mut forward_progress = 0.0;
            for rank in 0..8 {
                for file in 0..8 {
                    let square = chess::Square::make_square(
                        chess::Rank::from_index(rank),
                        chess::File::from_index(file)
                    );
                    if let Some(piece) = board.piece_on(square) {
                        if piece == chess::Piece::Pawn {
                            let progress = match board.color_on(square) {
                                Some(chess::Color::White) => rank as f32 / 7.0,  // Progress towards rank 8
                                Some(chess::Color::Black) => (7 - rank) as f32 / 7.0,  // Progress towards rank 1
                                None => 0.0,
                            };
                            forward_progress += progress * 0.05;  // Small bonus for pawn advancement
                        }
                    }
                }
            }
            value += forward_progress;
        }
        
        // Convert policy vector to tensor
        let policy = Tensor::from_slice(&policy_vec);
        
        self.cache.write().insert(hash, (policy.shallow_clone(), value));
        
        if self.cache.read().len() > self.config.cache_size {
            let keys: Vec<_> = self.cache.read().keys().copied().collect();
            let to_remove = keys.len() - self.config.cache_size;
            for key in keys.iter().take(to_remove) {
                self.cache.write().remove(key);
            }
        }
        
        Ok((policy, value))
    }
    
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
    
    fn generate_dirichlet_noise(&self, size: usize) -> Vec<f32> {
        let alpha = vec![self.config.dirichlet_alpha; size];
        let dirichlet = Dirichlet::new(&alpha).unwrap();
        let mut rng = thread_rng();
        dirichlet.sample(&mut rng)
    }

    pub fn search_batch(&mut self, boards: &[Board]) -> Result<Vec<HashMap<ChessMove, f32>>> {
        let mut results = Vec::with_capacity(boards.len());
        let mut batch = Vec::new();
        let mut batch_indices = Vec::new();
        
        // Process each board
        for (i, board) in boards.iter().enumerate() {
            let hash = board.get_hash();
            
            // Check cache first
            let cached_policy = self.cache.read().get(&hash).map(|(policy, _)| policy.shallow_clone());
            if let Some(_policy) = cached_policy {
                // If in cache, we can skip evaluation
                if let Some(root) = self.nodes.read().get(&hash) {
                    results.push(root.get_policy(self.config.temperature));
                } else {
                    results.push(HashMap::new());
                }
            } else {
                // Add to batch for evaluation
                batch.push(board.clone());
                batch_indices.push(i);
                
                // Process batch if full
                if batch.len() >= self.config.batch_size {
                    self.process_batch(&batch, &mut results, &batch_indices)?;
                    batch.clear();
                    batch_indices.clear();
                }
            }
        }
        
        // Process remaining boards
        if !batch.is_empty() {
            self.process_batch(&batch, &mut results, &batch_indices)?;
        }
        
        Ok(results)
    }
    
    fn process_batch(
        &mut self,
        batch: &[Board],
        results: &mut Vec<HashMap<ChessMove, f32>>,
        batch_indices: &[usize]
    ) -> Result<()> {
        // Process each board sequentially for now
        for (i, board) in batch.iter().enumerate() {
            let (policy, value) = self.model.predict_with_board(board.to_string())?;
            let policy_tensor = Tensor::from_slice(&policy);
            
            let hash = board.get_hash();
            self.cache.write().insert(hash, (policy_tensor.shallow_clone(), value));
            
            if !self.nodes.read().contains_key(&hash) {
                let mut moves = Vec::new();
                
                for mv in MoveGen::new_legal(board) {
                    let idx = self.move_to_index(&mv);
                    if let Some(idx) = idx {
                        let prior = if idx < policy.len() {
                            policy[idx]
                        } else {
                            0.0
                        };
                        moves.push((mv, prior));
                    }
                }
                
                let mut root = Node::new(0.0);
                let mut final_moves = moves.clone();
                
                if self.config.dirichlet_weight > 0.0 {
                    let noise = self.generate_dirichlet_noise(moves.len());
                    for (j, (_, prior)) in final_moves.iter_mut().enumerate() {
                        *prior = (1.0 - self.config.dirichlet_weight) * *prior + 
                                self.config.dirichlet_weight * noise[j];
                    }
                }
                
                root.expand(&final_moves);
                
                self.nodes.write().insert(hash, root);
            }
            
            let policy_result = self.search(board)?;
            
            let result_idx = batch_indices[i];
            while results.len() <= result_idx {
                results.push(HashMap::new());
            }
            results[result_idx] = policy_result;
        }
        
        Ok(())
    }
    
    pub fn evaluate_batch(&self, boards: &[Board]) -> Result<Vec<(Tensor, f32)>> {
        let mut results = Vec::with_capacity(boards.len());
        let mut batch = Vec::new();
        let mut batch_indices = Vec::new();
        
        // Process each board
        for (i, board) in boards.iter().enumerate() {
            let hash = board.get_hash();
            
            // Check cache first
            let cached_result = self.cache.read().get(&hash).map(|(policy, value)| (policy.shallow_clone(), *value));
            if let Some((policy, value)) = cached_result {
                results.push((policy, value));
            } else {
                // Add to batch for evaluation
                batch.push(board.clone());
                batch_indices.push(i);
                
                // Process batch if full
                if batch.len() >= self.config.batch_size {
                    let boards_vec: Vec<String> = batch.iter().map(|b| b.to_string()).collect();
                    let (policies, values) = self.model.predict_batch(boards_vec)?;
                    
                    // Cache and store results
                    for ((board, policy), value) in batch.iter().zip(policies).zip(values) {
                        let hash = board.get_hash();
                        let policy_tensor = Tensor::from_slice(&policy);
                        self.cache.write().insert(hash, (policy_tensor.shallow_clone(), value));
                        results.push((policy_tensor, value));
                    }
                    
                    batch.clear();
                    batch_indices.clear();
                }
            }
        }
        
        // Process remaining boards
        if !batch.is_empty() {
            let boards_vec: Vec<String> = batch.iter().map(|b| b.to_string()).collect();
            let (policies, values) = self.model.predict_batch(boards_vec)?;
            
            // Cache and store results
            for ((board, policy), value) in batch.iter().zip(policies).zip(values) {
                let hash = board.get_hash();
                let policy_tensor = Tensor::from_slice(&policy);
                self.cache.write().insert(hash, (policy_tensor.shallow_clone(), value));
                results.push((policy_tensor, value));
            }
        }
        
        Ok(results)
    }

    pub fn search_parallel(&self, board: &Board) -> Result<HashMap<ChessMove, f32>> {
        let start_time = Arc::new(Instant::now());
        let root_hash = board.get_hash();
        
        // Initialize root node if not exists
        if !self.nodes.read().contains_key(&root_hash) {
            let (policy_vec, value) = self.model.predict()?;
            let policy = Tensor::from_slice(&policy_vec);
            let mut moves = Vec::new();
            
            for mv in MoveGen::new_legal(board) {
                let idx = self.move_to_index(&mv);
                if let Some(idx) = idx {
                    let prior = policy_vec[idx];
                    moves.push((mv, prior));
                }
            }
            
            let mut root = Node::new(0.0);
            let mut final_moves = moves.clone();
            
            if self.config.dirichlet_weight > 0.0 {
                let noise = self.generate_dirichlet_noise(moves.len());
                for (i, (_, prior)) in final_moves.iter_mut().enumerate() {
                    *prior = (1.0 - self.config.dirichlet_weight) * *prior + 
                            self.config.dirichlet_weight * noise[i];
                }
            }
            
            root.expand(&final_moves);
            
            self.nodes.write().insert(root_hash, root);
            self.cache.write().insert(root_hash, (policy, value));
        }
        
        // Create thread-safe simulation counter
        let simulation_count = Arc::new(parking_lot::Mutex::new(0u32));
        let max_time = self.config.max_time;
        
        // Run parallel simulations
        (0..self.config.num_simulations).into_par_iter().for_each(|_| {
            // Check termination conditions
            if start_time.elapsed() > max_time {
                return;
            }
            
            let mut path = Vec::new();
            let mut current_board = board.clone();
            let mut current_hash = root_hash;
            
            // Selection phase
            while let Some(node) = self.nodes.read().get(&current_hash) {
                if !node.is_expanded {
                    break;
                }
                
                let mut best_score = f32::NEG_INFINITY;
                let mut best_move = None;
                let parent_visits = (node.visit_count as f32).sqrt();
                
                for (mv, child) in &node.children {
                    let q_value = -child.value();
                    let u_value = self.config.c_puct * child.prior * parent_visits / 
                                (1.0 + child.visit_count as f32);
                    let puct = q_value + u_value;
                    
                    if puct > best_score {
                        best_score = puct;
                        best_move = Some(*mv);
                    }
                }
                
                if let Some(mv) = best_move {
                    path.push((current_hash, mv));
                    let mut new_board = current_board.clone();
                    current_board.make_move(mv, &mut new_board);
                    current_board = new_board;
                    current_hash = current_board.get_hash();
                } else {
                    break;
                }
            }
            
            // Expansion phase
            let mut value = if current_board.status() == chess::BoardStatus::Ongoing {
                if let Ok((policy_vec, node_value)) = self.model.predict() {
                    let policy = Tensor::from_slice(&policy_vec);
                    let mut moves = Vec::new();
                    
                    for mv in MoveGen::new_legal(&current_board) {
                        let idx = self.move_to_index(&mv);
                        if let Some(idx) = idx {
                            let prior = policy_vec[idx];
                            moves.push((mv, prior));
                        }
                    }
                    
                    let mut root = Node::new(0.0);
                    let moves: Vec<(ChessMove, f32)> = moves.iter().map(|(mv, prior)| (*mv, *prior)).collect();
                    root.expand(&moves);
                    
                    let hash = current_board.get_hash();
                    self.nodes.write().insert(hash, root);
                    self.cache.write().insert(hash, (policy.shallow_clone(), node_value));
                    
                    node_value
                } else {
                    0.0 // Fallback value if evaluation fails
                }
            } else {
                match current_board.status() {
                    chess::BoardStatus::Checkmate => -1.0,
                    _ => 0.0,
                }
            };
            
            // Backpropagation phase
            for (hash, _) in path.iter().rev() {
                if let Some(node) = self.nodes.write().get_mut(hash) {
                    node.visit_count += 1;
                    node.value_sum += value;
                    value = -value;
                }
            }
            
            // Increment simulation count
            *simulation_count.lock() += 1;
        });
        
        // Return normalized visit counts as policy
        if let Some(root) = self.nodes.read().get(&root_hash) {
            Ok(root.get_policy(self.config.temperature))
        } else {
            Ok(HashMap::new())
        }
    }
    
    pub fn search_batch_parallel(&mut self, boards: &[Board]) -> Result<Vec<HashMap<ChessMove, f32>>> {
        let chunk_size = 8;
        let results: Vec<_> = boards
            .chunks(chunk_size)
            .map(|chunk| {
                let mut local_results = Vec::with_capacity(chunk.len());
                for board in chunk {
                    if let Ok(moves) = self.search(board) {
                        local_results.push(moves);
                    } else {
                        local_results.push(HashMap::new());
                    }
                }
                local_results
            })
            .flatten()
            .collect();
        Ok(results)
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

#[derive(Debug)]
pub struct CacheStats {
    pub evaluation_cache_size: usize,
    pub node_cache_size: usize,
    pub memory_usage: usize,
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
    fn test_dirichlet_noise() {
        let mcts = MCTS::new(ModelBridge::new(Board::default(), None), MCTSConfig::default());
        let noise = mcts.generate_dirichlet_noise(10);
        
        assert_eq!(noise.len(), 10);
        let sum: f32 = noise.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(noise.iter().all(|&x| x >= 0.0 && x <= 1.0));
    }
    
    #[test]
    fn test_batch_evaluation() {
        let mcts = MCTS::new(ModelBridge::new(Board::default(), None), MCTSConfig::default());
        let boards = vec![Board::default(); 3];
        
        let results = mcts.evaluate_batch(&boards).unwrap();
        assert_eq!(results.len(), 3);
        
        // Check that results are cached
        let cached_results = mcts.evaluate_batch(&boards).unwrap();
        assert_eq!(cached_results.len(), 3);
        
        // Verify cache hit rate
        assert_eq!(mcts.cache.read().len(), 1); // Only one unique position
    }
    
    #[test]
    fn test_batch_search() {
        let mcts = MCTS::new(ModelBridge::new(Board::default(), None), MCTSConfig::default());
        let boards = vec![Board::default(); 3];
        
        let results = mcts.search_batch(&boards).unwrap();
        assert_eq!(results.len(), 3);
        
        // Each result should be a valid policy
        for policy in results {
            assert!(!policy.is_empty());
            let sum: f32 = policy.values().sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
    }
    
    #[test]
    fn test_parallel_search() {
        let mcts = MCTS::new(ModelBridge::new(Board::default(), None), MCTSConfig::default());
        let board = Board::default();
        
        let policy = mcts.search_parallel(&board).unwrap();
        assert!(!policy.is_empty());
        
        let sum: f32 = policy.values().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_parallel_batch_search() {
        let mcts = MCTS::new(ModelBridge::new(Board::default(), None), MCTSConfig::default());
        let boards = vec![Board::default(); 3];
        
        let results = mcts.search_batch_parallel(&boards).unwrap();
        assert_eq!(results.len(), 3);
        
        for policy in results {
            assert!(!policy.is_empty());
            let sum: f32 = policy.values().sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
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
    
    #[test]
    fn test_cache_management() {
        let mut mcts = MCTS::new(ModelBridge::new(Board::default(), None), MCTSConfig::default());
        let board = Board::default();
        
        // Fill cache
        mcts.search(&board).unwrap();
        let stats = mcts.get_cache_stats();
        assert!(stats.evaluation_cache_size > 0);
        assert!(stats.node_cache_size > 0);
        
        // Clear cache
        mcts.clear_cache();
        let stats = mcts.get_cache_stats();
        assert_eq!(stats.evaluation_cache_size, 0);
        assert_eq!(stats.node_cache_size, 0);
    }
} 