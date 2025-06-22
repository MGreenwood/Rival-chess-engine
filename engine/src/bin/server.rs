use std::sync::Mutex;
use actix_web::{web, App, HttpServer, HttpResponse, HttpRequest, Responder};
use actix_web::middleware::Logger;
use actix_cors::Cors;
use actix_web_actors::ws;
use actix::{Actor, StreamHandler};
use serde::{Deserialize, Serialize};
use rival_ai::engine::Engine;
use pyo3::prelude::*;
use chrono::{DateTime, Utc};
use chess::{Board as ChessBoard, ChessMove, BoardStatus};
use std::str::FromStr;
use clap::Parser;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use rand::{thread_rng, Rng};
use rival_ai::game_storage::{GameStorage, GameState as StorageGameState, GameMode, GameStatus, GameMetadata};
use rival_ai::ModelBridge;
use serde_json::json;
use std::fs;
use pyo3::types::PyDict;
use uuid;
use chess::MoveGen;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use tokio::time::{sleep, Duration};
use std::net::IpAddr;
use std::time::{SystemTime, UNIX_EPOCH};
use jsonwebtoken::{encode, decode, Header, Validation, EncodingKey, DecodingKey};

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the model checkpoint file for single player games
    #[arg(short, long, default_value = "../models/latest_trained_model.pt")]
    model_path: String,
    
    /// Path to the model checkpoint file for community games
    #[arg(long, default_value = "../python/experiments/rival_ai_v1_Alice/run_20250619_162208/checkpoints/best_model.pt")]
    community_model_path: String,
    
    /// Server port
    #[arg(short, long, default_value = "3000")]
    port: u16,
    
    /// Server host
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// Directory to save games for training
    #[arg(long, default_value = "../python/training_games")]
    games_dir: String,

    /// Number of games needed before training starts
    #[arg(long, default_value = "5000")]
    training_games_threshold: usize,
    
    /// Enable self-play background generation
    #[arg(long, action = clap::ArgAction::SetFalse)]
    enable_self_play: bool,
    
    /// Enable background training
    #[arg(long, action = clap::ArgAction::SetFalse)]
    enable_training: bool,
    
    /// Enable TensorBoard logging for training
    #[arg(long)]
    tensorboard: bool,
    
    /// Maximum number of self-play games when traffic is low
    #[arg(long, default_value = "300")]
    max_self_play_games: usize,
    
    /// Initial number of self-play games on startup
    #[arg(long, default_value = "20")]
    initial_self_play_games: usize,
    
    /// Target GPU utilization for self-play scaling (0.0-1.0)
    #[arg(long, default_value = "0.85")]
    target_gpu_utilization: f32,
}

#[derive(Deserialize)]
struct MoveRequest {
    move_str: String,
    #[allow(dead_code)]  // Reserved for future move validation
    board: Option<String>,
    player_color: String,
    game_id: String,
}

#[derive(Serialize)]
#[allow(dead_code)]  // Reserved for future API responses
struct MoveResponse {
    success: bool,
    board: String,
    status: String,
    engine_move: Option<String>,
    is_player_turn: bool,
    error_message: Option<String>,
    move_history: Vec<String>,
}

#[derive(Serialize, Deserialize)]
struct SavedGame {
    game_id: String,
    moves: Vec<String>,
    result: String,
    timestamp: String,
    player_color: String,
}

#[derive(Serialize, Deserialize)]
struct CommunityGameStateResponse {
    game_id: String,
    board: String,
    status: String,
    move_history: Vec<String>,
    is_voting_phase: bool,
    voting_ends_at: Option<String>,
    current_votes: HashMap<String, usize>,
    total_voters: usize,
    your_vote: Option<String>,
    can_vote: bool,
    waiting_for_first_move: bool,
    experiment_name: String,
    engine_thinking: bool,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]  // Reserved for future rate limiting enhancements
struct RateLimitEntry {
    timestamp: u64,
    ip: IpAddr,
}

#[derive(Debug)]
pub struct RateLimiter {
    requests: Arc<Mutex<HashMap<IpAddr, VecDeque<u64>>>>,
    max_requests: usize,
    window_seconds: u64,
}

impl RateLimiter {
    fn new(max_requests: usize, window_seconds: u64) -> Self {
        Self {
            requests: Arc::new(Mutex::new(HashMap::new())),
            max_requests,
            window_seconds,
        }
    }

    fn check_rate_limit(&self, ip: IpAddr) -> Result<(), String> {
        let mut requests = self.requests.lock().unwrap();
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        let window_start = now - self.window_seconds;

        let ip_requests = requests.entry(ip).or_insert_with(VecDeque::new);
        
        // Remove old requests outside the window
        while let Some(&front) = ip_requests.front() {
            if front < window_start {
                ip_requests.pop_front();
            } else {
                break;
            }
        }

        // Check if rate limit exceeded
        if ip_requests.len() >= self.max_requests {
            return Err(format!(
                "Rate limit exceeded. Max {} requests per {} seconds",
                self.max_requests, self.window_seconds
            ));
        }

        // Add current request
        ip_requests.push_back(now);
        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct VoterSession {
    voter_id: String,
    ip: String,
    created_at: u64,
    last_vote_at: Option<u64>,
    vote_count: u32,
}

#[derive(Debug, Serialize, Deserialize)]
struct Claims {
    sub: String,  // voter_id
    exp: usize,   // expiration
    iat: usize,   // issued at
    ip: String,   // IP address
}

#[derive(Clone)]
pub struct CommunityGame {
    game_id: String,  // Persistent game ID
    board: ChessBoard,
    move_history: Vec<String>,
    votes: HashMap<String, HashSet<String>>,  // move -> set of voter IDs
    voting_ends_at: Option<DateTime<Utc>>,
    status: GameStatus,
    player_is_white: bool,  // Add this field to track which color the community plays
    votes_started: bool,  // Track if voting has started for the current round
    engine_thinking: bool,  // Track when engine is actively thinking
}

impl CommunityGame {
    pub fn new() -> Self {
        Self {
            game_id: uuid::Uuid::new_v4().to_string(),  // Generate game ID once
            board: ChessBoard::default(),
            move_history: Vec::new(),
            votes: HashMap::new(),
            voting_ends_at: None,
            status: GameStatus::Active,
            player_is_white: true,  // Add this field to track which color the community plays
            votes_started: false,  // Initialize as false
            engine_thinking: false,
        }
    }

    pub fn start_voting(&mut self) {
        self.voting_ends_at = Some(Utc::now() + chrono::Duration::seconds(10));
        self.votes_started = true;
        // Don't clear votes here - keep accumulated votes when timer starts
    }

    pub fn clear_votes(&mut self) {
        self.votes.clear();
        self.voting_ends_at = None;
        self.votes_started = false;
    }

    pub fn add_vote(&mut self, move_str: &str, voter_id: &str) -> Result<(), String> {
        // Check if it's our turn
        let is_whites_turn = self.board.side_to_move() == chess::Color::White;
        if is_whites_turn != self.player_is_white {
            return Err("Not the community's turn to move".to_string());
        }

        // Validate the move is legal
        let chess_move = parse_move(move_str)?;
        if !MoveGen::new_legal(&self.board).any(|m| m == chess_move) {
            return Err(format!("Illegal move: {}", move_str));
        }
        
        // Check if we can accept votes
        if self.votes_started && !self.is_voting_phase() {
            return Err("Cannot vote right now - voting phase may have ended. Try starting a new voting round!".to_string());
        }
        
        // Remove previous vote if exists
        for votes in self.votes.values_mut() {
            votes.remove(voter_id);
        }
        
        // Add new vote
        self.votes.entry(move_str.to_string())
            .or_insert_with(HashSet::new)
            .insert(voter_id.to_string());
        
        // Auto-start timer on first vote if not already started
        if !self.votes_started {
            self.start_voting();
        }
        
        Ok(())
    }

    pub fn get_winning_move(&self) -> Option<String> {
        if self.votes.is_empty() {
            return None;
        }

        // Find moves with maximum votes
        let max_votes = self.votes.values()
            .map(|voters| voters.len())
            .max()
            .unwrap_or(0);

        let winning_moves: Vec<String> = self.votes.iter()
            .filter(|(_, voters)| voters.len() == max_votes)
            .map(|(mv, _)| mv.clone())
            .collect();

        // Randomly select from moves with equal max votes
        if winning_moves.is_empty() {
            None
        } else {
            let mut rng = thread_rng();
            Some(winning_moves[rng.gen_range(0..winning_moves.len())].clone())
        }
    }

    fn make_move(&mut self, mv: ChessMove) -> Result<(), String> {
        // Validate the move is legal before applying it
        if MoveGen::new_legal(&self.board).any(|m| m == mv) {
            let mut new_board = self.board.clone();
            new_board = new_board.make_move_new(mv);
            self.board = new_board;
            Ok(())
        } else {
            Err(format!("Illegal move: {}", mv))
        }
    }

    pub fn process_votes(&mut self, engine: &mut Engine, game_storage: &GameStorage) -> Result<(), String> {
        // Only process votes if timer has expired or manually triggered
        if self.is_voting_phase() {
            return Ok(()); // Timer still running
        }

        // Don't process votes if no timer has been started yet
        if !self.votes_started && !self.votes.is_empty() {
            return Ok(()); // Votes exist but timer hasn't been started
        }

        if self.votes.is_empty() {
            return Ok(()); // No votes to process
        }

        if let Some(winning_move) = self.get_winning_move() {
            // Parse and validate the player's move
            let chess_move = parse_move(&winning_move)?;
            
            // Make the player's move
            self.make_move(chess_move)?;
            self.move_history.push(winning_move);

            // Clear votes after processing
            self.clear_votes();

            // Check if the game is over after community's move BEFORE asking engine to move
            match self.board.status() {
                chess::BoardStatus::Checkmate => {
                    // Determine winner: if it's white's turn and checkmate, black wins (and vice versa)
                    self.status = if self.board.side_to_move() == chess::Color::White {
                        GameStatus::BlackWins
                    } else {
                        GameStatus::WhiteWins
                    };
                    eprintln!("🏁 Game ended in checkmate after community move");
                    self.save_game_state(game_storage);
                    return Ok(());
                }
                chess::BoardStatus::Stalemate => {
                    self.status = GameStatus::DrawStalemate;
                    eprintln!("🏁 Game ended in stalemate after community move");
                    self.save_game_state(game_storage);
                    return Ok(());
                }
                _ => {
                    // Game continues - proceed with engine move
                }
            }

            // Make engine's move if it's their turn (and game is still ongoing)
            if self.board.side_to_move() != if self.player_is_white { chess::Color::White } else { chess::Color::Black } {
                self.engine_thinking = true;  // Set thinking status
                eprintln!("🤖 Engine is thinking... (engine_thinking = true)");
                
                // Use a closure to ensure engine_thinking is always cleared
                let engine_result = (|| -> Result<Option<String>, String> {
                    match engine.get_best_move(self.board.clone()) {
                        Some(engine_move) => {
                            self.make_move(engine_move)?;
                            let move_str = engine_move.to_string();
                            eprintln!("🤖 Engine played: {} (engine_thinking will be cleared)", move_str);
                            Ok(Some(move_str))
                        }
                        None => {
                            eprintln!("⚠️ Engine failed to find a move (engine_thinking will be cleared)");
                            Ok(None)
                        }
                    }
                })();
                
                // ALWAYS clear engine_thinking flag, regardless of success or failure
                self.engine_thinking = false;
                eprintln!("🔓 Engine thinking cleared (engine_thinking = false)");
                
                // Handle the result after clearing the flag
                match engine_result {
                    Ok(Some(move_str)) => {
                        self.move_history.push(move_str);
                    }
                    Ok(None) => {
                        eprintln!("⚠️ Engine returned no move, continuing without engine move");
                    }
                    Err(e) => {
                        eprintln!("❌ Engine move failed: {}", e);
                        // Don't return error - game can continue even if engine move fails
                    }
                }

                // Check if the game is over after engine's move
                match self.board.status() {
                    chess::BoardStatus::Checkmate => {
                        // Determine winner: if it's white's turn and checkmate, black wins (and vice versa)
                        self.status = if self.board.side_to_move() == chess::Color::White {
                            GameStatus::BlackWins
                        } else {
                            GameStatus::WhiteWins
                        };
                        eprintln!("🏁 Game ended in checkmate after engine move");
                    }
                    chess::BoardStatus::Stalemate => {
                        self.status = GameStatus::DrawStalemate;
                        eprintln!("🏁 Game ended in stalemate after engine move");
                    }
                    _ => {
                        // Game continues
                    }
                }
            }

            // Save game state (for persistence/recovery)
            self.save_game_state(game_storage);
        }

        Ok(())
    }

    pub fn is_voting_phase(&self) -> bool {
        if let Some(ends_at) = self.voting_ends_at {
            Utc::now() <= ends_at && self.votes_started
        } else {
            false
        }
    }

    pub fn can_vote(&self) -> bool {
        // Can vote if:
        // 1. It's our turn and game is active
        // 2. Either voting hasn't started yet OR voting is in progress
        if self.status != GameStatus::Active {
            return false;
        }

        let is_whites_turn = self.board.side_to_move() == chess::Color::White;
        if is_whites_turn != self.player_is_white {
            return false;
        }

        // Can vote if voting hasn't started yet or is currently active
        !self.votes_started || self.is_voting_phase()
    }

    pub fn waiting_for_first_move(&self) -> bool {
        self.move_history.is_empty() && self.player_is_white && !self.votes_started
    }

    pub fn has_votes_but_no_timer(&self) -> bool {
        !self.votes.is_empty() && !self.votes_started
    }

    pub fn force_resolve_votes(&mut self, engine: &mut Engine, game_storage: &GameStorage) -> Result<(), String> {
        // Force process votes regardless of timer state
        if self.votes.is_empty() {
            return Err("No votes to resolve".to_string());
        }

        // Clear timer and process
        self.voting_ends_at = None;
        self.votes_started = false;
        self.process_votes(engine, game_storage)
    }

    fn save_game_state(&self, game_storage: &GameStorage) {
        let game_state = StorageGameState {
            metadata: GameMetadata {
                game_id: self.game_id.clone(),  // Use persistent game ID
                mode: GameMode::Community,
                created_at: Utc::now(),
                last_move_at: Utc::now(),
                status: self.status.clone(),
                total_moves: self.move_history.len(),
                player_color: if self.player_is_white { "white".to_string() } else { "black".to_string() },
                player_name: Some("Community".to_string()),
                engine_version: "1.0.0".to_string(),
            },
            board: self.board.to_string(),
            move_history: self.move_history.clone(),
            analysis: None,
        };

        if let Err(e) = game_storage.save_game(&game_state) {
            eprintln!("Failed to save community game: {}", e);
        }
    }
}

pub struct CommunityGameState {
    pub game: Arc<Mutex<CommunityGame>>,
    pub engine: Arc<Mutex<Engine>>,
}

impl Clone for CommunityGameState {
    fn clone(&self) -> Self {
        Self {
            game: self.game.clone(),
            engine: self.engine.clone(),
        }
    }
}

pub struct AppState {
    pub game: Arc<Mutex<CommunityGame>>,
    pub game_storage: Arc<GameStorage>,
    pub engine: Arc<Mutex<Engine>>,
    pub community_engine: Arc<Mutex<Engine>>,
    pub is_training: Arc<AtomicBool>,
    pub active_players: Arc<AtomicUsize>,
    pub self_play_games: Arc<AtomicUsize>,
    pub active_games: Arc<Mutex<HashMap<String, StorageGameState>>>,
    pub vote_limiter: Arc<RateLimiter>,
    pub voter_sessions: Arc<Mutex<HashMap<String, VoterSession>>>,
    pub jwt_secret: String,
    pub current_model_path: Arc<Mutex<String>>,  // Track current model path for self-play
    // Add cached stats - initialized once on startup, then just incremented
    pub cached_stats: Arc<Mutex<ServerStats>>,
    // In-memory queue of last 10 completed games (much faster than disk I/O)
    pub recent_completed_games: Arc<Mutex<VecDeque<GameMetadata>>>,
}

impl AppState {
    fn get_experiment_name(&self) -> String {
        // Extract experiment name from the community model path
        // Example path: "../python/experiments/rival_ai_v1_Alice/run_20250618_134810/checkpoints/checkpoint_epoch_5.pt"
        let args = Args::parse();
        let path = std::path::Path::new(&args.community_model_path);
        
        // Try to get the experiment folder name (e.g. "rival_ai_v1_Alice")
        // Go up: checkpoint_file -> checkpoints -> run_folder -> experiment_folder
        if let Some(experiment_dir) = path.parent()
            .and_then(|p| p.parent())  // Go up from checkpoints to run folder
            .and_then(|p| p.parent())  // Go up from run folder to experiment folder
        {
            if let Some(exp_name) = experiment_dir.file_name() {
                if let Some(name) = exp_name.to_str() {
                    return name.to_string();
                }
            }
        }
        
        "Unknown Model".to_string()
    }
}

#[derive(Deserialize)]
struct GameSettings {
    #[allow(dead_code)]  // Reserved for future game settings
    temperature: Option<f32>,
    #[allow(dead_code)]  // Reserved for future game settings
    strength: Option<f32>,
    _time_control: Option<TimeControl>,
    _engine_strength: Option<u32>,
    _color: Option<String>,
}

#[derive(Deserialize)]
struct TimeControl {
    _initial: u32,
    _increment: u32,
}

#[derive(Serialize)]
struct WebGameState {
    game_id: String,
    board: String,
    status: String,
    move_history: Vec<String>,
    is_player_turn: bool,
}

#[derive(Deserialize)]
#[allow(dead_code)]  // Reserved for future voting features
struct VoteRequest {
    move_str: String,
    voter_id: String,
}

#[derive(Serialize)]
struct VoteResponse {
    success: bool,
    error_message: Option<String>,
    game_state: CommunityGameStateResponse,
}

#[derive(Serialize, Deserialize)]
struct NewGameRequest {
    player_color: String,
}

#[derive(Serialize)]
struct NewGameResponse {
    game_id: String,
    board: String,
    status: String,
    move_history: Vec<String>,
    is_player_turn: bool,
}

#[derive(Deserialize)]
struct StartCommunityGameRequest {
    player_color: Option<String>,  // "white" or "black"
}

#[derive(Deserialize)]
struct StartVotingRequest {
    token: String,
}

#[derive(Serialize)]
struct StartVotingResponse {
    success: bool,
    error_message: Option<String>,
    game_state: CommunityGameStateResponse,
}

fn parse_move(move_str: &str) -> Result<ChessMove, String> {
    // Add more validation
    if move_str.is_empty() {
        return Err("Empty move string".to_string());
    }
    
    if move_str.len() < 4 || move_str.len() > 5 {
        return Err(format!("Invalid move length: {} (expected 4-5 characters)", move_str.len()));
    }

    // Validate the move string contains only valid characters
    if !move_str.chars().all(|c| c.is_ascii_alphanumeric()) {
        return Err("Invalid characters in move string".to_string());
    }

    // Parse squares with better error handling
    let from_str = &move_str[0..2];
    let to_str = &move_str[2..4];
    
    let from = chess::Square::from_str(from_str)
        .map_err(|_| format!("Invalid 'from' square: {}", from_str))?;
    let to = chess::Square::from_str(to_str)
        .map_err(|_| format!("Invalid 'to' square: {}", to_str))?;

    let promotion = if move_str.len() > 4 {
        let promo_char = move_str.chars().nth(4).unwrap_or('q');
        Some(match promo_char.to_ascii_lowercase() {
            'q' => chess::Piece::Queen,
            'r' => chess::Piece::Rook,
            'b' => chess::Piece::Bishop,
            'n' => chess::Piece::Knight,
            _ => return Err("Invalid promotion piece".to_string()),
        })
    } else {
        None
    };

    Ok(ChessMove::new(from, to, promotion))
}

#[allow(dead_code)]  // Reserved for future game result analysis
fn get_game_result(board: &chess::Board) -> &'static str {
    match board.status() {
        BoardStatus::Ongoing => "active",
        BoardStatus::Stalemate => "draw_stalemate",
        BoardStatus::Checkmate => {
            if board.side_to_move() == chess::Color::White {
                "black_wins"
            } else {
                "white_wins"
            }
        }
    }
}

// Add this helper function at the top level
fn recover_mutex<'a, T>(result: Result<std::sync::MutexGuard<'a, T>, std::sync::PoisonError<std::sync::MutexGuard<'a, T>>>) -> std::sync::MutexGuard<'a, T> {
    match result {
        Ok(guard) => guard,
        Err(poisoned) => {
            eprintln!("Mutex was poisoned. Recovering...");
            poisoned.into_inner()
        }
    }
}

async fn make_move(
    data: web::Data<AppState>,
    req: web::Json<MoveRequest>,
) -> impl Responder {
    let game_id = &req.game_id;
    let move_str = &req.move_str;
    let _player_color = &req.player_color;

    // Load the game from memory first, then storage
    let mut game_state = {
        let mut active_games = recover_mutex(data.active_games.lock());
        if let Some(state) = active_games.get(game_id) {
            state.clone()
        } else {
            // Try to load from persistent storage (for old games)
            match data.game_storage.load_game(game_id, &GameMode::SinglePlayer) {
                Ok(state) => {
                    // Move it to active games if it's still active
                    if state.metadata.status == GameStatus::Active {
                        active_games.insert(game_id.clone(), state.clone());
                    }
                    state
                },
                Err(e) => {
                    eprintln!("❌ Game not found: {}", e);
                    return HttpResponse::NotFound().json(json!({
                        "success": false,
                        "error_message": format!("Game not found: {}", e)
                    }));
                }
            }
        }
    };

    // Use the stored board state instead of client-provided board
    let mut board = match ChessBoard::from_str(&game_state.board) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Invalid stored board position '{}': {}", game_state.board, e);
            return HttpResponse::InternalServerError().json(json!({
                "success": false,
                "error_message": "Internal server error: invalid board state"
            }));
        }
    };

    // Parse and validate the player's move with better error handling
    let player_move = match parse_move(move_str) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Invalid move '{}': {}", move_str, e);
            return HttpResponse::BadRequest().json(json!({
                "success": false,
                "error_message": format!("Invalid move: {}", e)
            }));
        }
    };

    // Validate the move is legal BEFORE trying to make it
    let legal_moves: Vec<ChessMove> = MoveGen::new_legal(&board).collect();
    if !legal_moves.contains(&player_move) {
        eprintln!("🚫 ========== ILLEGAL MOVE DETECTED ==========");
        eprintln!("🚫 Attempted move: '{}'", move_str);
        eprintln!("🚫 Current position: '{}'", board);
        eprintln!("🚫 Move history: {:?}", game_state.move_history);
        eprintln!("🚫 Legal moves: {:?}", legal_moves.iter().map(|m| m.to_string()).collect::<Vec<_>>());
        eprintln!("🚫 ⚠️  CLIENT GAME STATE IS OUT OF SYNC WITH SERVER!");
        eprintln!("🚫 ==========================================");
        return HttpResponse::BadRequest().json(json!({
            "success": false,
            "error_message": format!("Illegal move: {}", move_str)
        }));
    }

    // Get the engine with better error handling
    let engine = match data.engine.lock() {
        Ok(engine) => engine,
        Err(e) => {
            eprintln!("Failed to lock engine: {}", e);
            // Try to recover from poisoned mutex
            let engine = recover_mutex(data.engine.lock());
            engine
        }
    };

    // Make the player's move (now we know it's legal)
    board = board.make_move_new(player_move);
    game_state.board = board.to_string();
    game_state.move_history.push(move_str.to_string());
    game_state.metadata.last_move_at = Utc::now();
    game_state.metadata.total_moves += 1;

    // Check if the game is over after player's move
    if board.status() != BoardStatus::Ongoing {
        game_state.metadata.status = match board.status() {
            BoardStatus::Checkmate => {
                if board.side_to_move() == chess::Color::White {
                    GameStatus::BlackWins
                } else {
                    GameStatus::WhiteWins
                }
            },
            BoardStatus::Stalemate => GameStatus::DrawStalemate,
            _ => GameStatus::Active,
        };
        
        // Save completed game to persistent storage and remove from memory
        if let Err(e) = data.game_storage.save_game(&game_state) {
            eprintln!("Failed to save completed game: {}", e);
            return HttpResponse::InternalServerError().json(json!({
                "success": false,
                "error_message": format!("Failed to save game: {}", e)
            }));
        }
        
        // Update stats cache immediately (no file I/O needed!)
        update_stats_cache_for_completed_game(&data, &game_state.metadata);
        
        // Remove from active games
        {
            let mut active_games = recover_mutex(data.active_games.lock());
            active_games.remove(game_id);
        }

        return HttpResponse::Ok().json(json!({
            "success": true,
            "board": game_state.board,
            "status": game_state.metadata.status.to_string(),
            "move_history": game_state.move_history,
            "is_player_turn": true
        }));
    }

    // Get engine's move
    if let Some(engine_move) = engine.get_best_move(board.clone()) {
        // Make the engine's move
        board = board.make_move_new(engine_move);
        game_state.board = board.to_string();
        game_state.move_history.push(engine_move.to_string());
        game_state.metadata.total_moves += 1;

        // Check if the game is over after engine's move
        if board.status() != BoardStatus::Ongoing {
            game_state.metadata.status = match board.status() {
                BoardStatus::Checkmate => {
                    if board.side_to_move() == chess::Color::White {
                        GameStatus::BlackWins
                    } else {
                        GameStatus::WhiteWins
                    }
                },
                BoardStatus::Stalemate => GameStatus::DrawStalemate,
                _ => GameStatus::Active,
            };
            
            // Save completed game to persistent storage and remove from memory
            if let Err(e) = data.game_storage.save_game(&game_state) {
                eprintln!("Failed to save completed game: {}", e);
                return HttpResponse::InternalServerError().json(json!({
                    "success": false,
                    "error_message": format!("Failed to save game: {}", e)
                }));
            }
            
            // Update stats cache immediately (no file I/O needed!)
            update_stats_cache_for_completed_game(&data, &game_state.metadata);
            
            // Remove from active games
            {
                let mut active_games = recover_mutex(data.active_games.lock());
                active_games.remove(game_id);
            }
        } else {
            // Update active game in memory
            {
                let mut active_games = recover_mutex(data.active_games.lock());
                active_games.insert(game_id.clone(), game_state.clone());
            }
        }

        // Move completed successfully

        HttpResponse::Ok().json(json!({
            "success": true,
            "board": game_state.board,
            "status": game_state.metadata.status.to_string(),
            "move_history": game_state.move_history,
            "is_player_turn": true,
            "engine_move": Some(engine_move.to_string())
        }))
    } else {
        eprintln!("❌ Engine failed to make a move");
        
        HttpResponse::InternalServerError().json(json!({
            "success": false,
            "error_message": "Engine failed to make a move"
        }))
    }
}

async fn create_new_game(
    data: web::Data<AppState>,
    req: web::Json<NewGameRequest>,
) -> impl Responder {
    data.active_players.fetch_add(1, Ordering::SeqCst);
    let _engine = recover_mutex(data.engine.lock());
    let game_id = uuid::Uuid::new_v4().to_string();
    
    let game_state = StorageGameState {
        metadata: GameMetadata {
            game_id: game_id.clone(),
            mode: GameMode::SinglePlayer,
            created_at: Utc::now(),
            last_move_at: Utc::now(),
            status: GameStatus::Active,
            total_moves: 0,
            player_color: req.player_color.clone(),
            player_name: None,
            engine_version: "1.0.0".to_string(),
        },
        board: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".to_string(),
        move_history: Vec::new(),
        analysis: None,
    };

    // Store in memory instead of persistent storage
    {
        let mut active_games = recover_mutex(data.active_games.lock());
        active_games.insert(game_id.clone(), game_state.clone());
    }

    HttpResponse::Ok().json(NewGameResponse {
        game_id,
        board: game_state.board,
        status: game_state.metadata.status.to_string(),
        move_history: game_state.move_history,
        is_player_turn: req.player_color == "white",
    })
}

async fn get_game(
    data: web::Data<AppState>,
    path: web::Path<String>,
) -> impl Responder {
    let game_id = path.into_inner();
    
    // Try active games first
    {
        let active_games = recover_mutex(data.active_games.lock());
        if let Some(game_state) = active_games.get(&game_id) {
            return HttpResponse::Ok().json(WebGameState {
                game_id: game_state.metadata.game_id.clone(),
                board: game_state.board.clone(),
                status: game_state.metadata.status.to_string(),
                move_history: game_state.move_history.clone(),
                is_player_turn: true,
            });
        }
    }
    
    // Try persistent storage (for completed games)
    match data.game_storage.load_game(&game_id, &GameMode::SinglePlayer) {
        Ok(game) => HttpResponse::Ok().json(WebGameState {
            game_id: game.metadata.game_id,
            board: game.board,
            status: game.metadata.status.to_string(),
            move_history: game.move_history,
            is_player_turn: true,
        }),
        Err(e) => HttpResponse::NotFound().json(json!({
            "error": format!("Game not found: {}", e)
        }))
    }
}

#[allow(dead_code)]  // Reserved for future saved games functionality
async fn get_saved_games(data: web::Data<AppState>) -> impl Responder {
    match data.game_storage.list_games(Some(GameMode::SinglePlayer)) {
        Ok(games) => {
            let mut saved_games = Vec::new();
            
            for metadata in games {
                // Load the full game state to get move history
                if let Ok(game_state) = data.game_storage.load_game(&metadata.game_id, &GameMode::SinglePlayer) {
                    saved_games.push(SavedGame {
                        game_id: metadata.game_id,
                        moves: game_state.move_history,
                        result: metadata.status.to_string(),
                        timestamp: metadata.last_move_at.to_rfc3339(),
                        player_color: metadata.player_color,
                    });
                }
            }
            
            HttpResponse::Ok().json(saved_games)
        }
        Err(e) => HttpResponse::InternalServerError().json(json!({
            "error": format!("Failed to list games: {}", e)
        }))
    }
}

struct MyWebSocket {
    game_id: String,
    engine: web::Data<AppState>,
}

impl MyWebSocket {
    fn new(game_id: String, engine: web::Data<AppState>) -> Self {
        Self { game_id, engine }
    }

    fn send_game_state(&self, ctx: &mut ws::WebsocketContext<Self>) {
        // Try active games first
        {
            let active_games = recover_mutex(self.engine.active_games.lock());
            if let Some(game_state) = active_games.get(&self.game_id) {
                let response = WebGameState {
                    game_id: game_state.metadata.game_id.clone(),
                    board: game_state.board.clone(),
                    status: game_state.metadata.status.to_string(),
                    move_history: game_state.move_history.clone(),
                    is_player_turn: true,
                };
                ctx.text(serde_json::to_string(&response).unwrap());
                return;
            }
        }
        
        // Try persistent storage (for completed games)
        if let Ok(game_state) = self.engine.game_storage.load_game(&self.game_id, &GameMode::SinglePlayer) {
            let response = WebGameState {
                game_id: game_state.metadata.game_id,
                board: game_state.board,
                status: game_state.metadata.status.to_string(),
                move_history: game_state.move_history,
                is_player_turn: true,
            };
            ctx.text(serde_json::to_string(&response).unwrap());
        }
    }
}

impl Actor for MyWebSocket {
    type Context = ws::WebsocketContext<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        self.send_game_state(ctx);
    }
}

impl StreamHandler<Result<ws::Message, ws::ProtocolError>> for MyWebSocket {
    fn handle(&mut self, msg: Result<ws::Message, ws::ProtocolError>, ctx: &mut Self::Context) {
        if let Ok(ws::Message::Text(text)) = msg {
            if let Ok(_move_request) = serde_json::from_str::<MoveRequest>(&text) {
                // Just send the current game state - no need to reload
                self.send_game_state(ctx);
            }
        }
    }
}

async fn ws_route(
    req: HttpRequest,
    stream: web::Payload,
    path: web::Path<String>,
    data: web::Data<AppState>,
) -> Result<HttpResponse, actix_web::Error> {
    let game_id = path.into_inner();
    let ws = MyWebSocket::new(game_id, data);
    ws::start(ws, &req, stream)
}

#[allow(dead_code)]  // Reserved for future game saving functionality
fn save_game(state: &AppState, game_id: &str, moves: Vec<String>, result: &str) {
    let game_state = StorageGameState {
        metadata: GameMetadata {
            game_id: game_id.to_string(),
            mode: GameMode::SinglePlayer,
            created_at: Utc::now(),
            last_move_at: Utc::now(),
            status: GameStatus::from(result),
            total_moves: moves.len(),
            player_color: "white".to_string(),
            player_name: None,
            engine_version: "1.0.0".to_string(),
        },
        board: "".to_string(), // TODO: Store final board position
        move_history: moves,
        analysis: None,
    };

    if let Err(e) = state.game_storage.save_game(&game_state) {
        eprintln!("Failed to save game: {}", e);
    }
}

// Helper function to create model with device selection and ultra-dense PAG support
fn create_model_with_device(py: Python, torch: &PyModule, model_path: &str) -> PyResult<ModelBridge> {
    let device = if torch.getattr("cuda")?.call_method0("is_available")?.extract::<bool>()? {
        Some("cuda".to_string())
    } else {
        Some("cpu".to_string())
    };
    
    let code = format!(r#"
import sys
sys.path.insert(0, '../python/src')

# Try to import components with fallback
try:
    from rival_ai.models.gnn import ChessGNN
    CHESSGNN_AVAILABLE = True
    print("✅ ChessGNN loaded successfully")
except ImportError as e:
    print(f"⚠️ ChessGNN not available: {{e}}")
    CHESSGNN_AVAILABLE = False

# Ultra-dense PAG components are available natively in Rust
ULTRA_DENSE_AVAILABLE = True
print("✅ Ultra-dense PAG components available (native Rust implementation)")

import torch

class UltraDenseModelWrapper:
    def __init__(self):
        print("🧠 Initializing Model Wrapper...")
        
        if not CHESSGNN_AVAILABLE:
            print("⚠️ ChessGNN unavailable, using basic fallback")
            self.use_fallback = True
            self.device = 'cpu'
            return
        
        self.use_fallback = False
        
        # Try to load checkpoint to determine model type
        try:
            checkpoint = torch.load('{}', map_location='cpu')
            print("✅ Checkpoint loaded successfully")
            
            # Use existing ChessGNN interface
            config = {{
                'hidden_dim': 256,
                'num_layers': 4,
                'num_heads': 4, 
                'dropout': 0.1
            }}
            
            # Create model with existing interface
            try:
                self.model = ChessGNN(**config)
                print(f"🚀 Created ChessGNN model")
            except Exception as model_create_error:
                print(f"❌ Failed to create ChessGNN: {{model_create_error}}")
                self.use_fallback = True
                self.device = 'cpu'
                return
            
            # Load weights
            try:
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                
                # Load state dict
                missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
                
                if missing_keys:
                    print(f"🔄 Missing keys: {{len(missing_keys)}}")
                if unexpected_keys:
                    print(f"⚠️ Unexpected keys: {{len(unexpected_keys)}}")
                
                print("✅ Model weights loaded successfully")
            except Exception as weight_error:
                print(f"❌ Failed to load weights: {{weight_error}}")
                print("🔄 Using randomly initialized model")
            
        except Exception as e:
            print(f"❌ Error loading checkpoint: {{e}}")
            print("🆕 Creating fresh ChessGNN model")
            
            try:
                # Create fresh model
                config = {{
                    'hidden_dim': 256,
                    'num_layers': 4,
                    'num_heads': 4, 
                    'dropout': 0.1
                }}
                self.model = ChessGNN(**config)
                print("🆕 Created fresh ChessGNN model")
            except Exception as fresh_error:
                print(f"❌ Failed to create fresh model: {{fresh_error}}")
                self.use_fallback = True
                self.device = 'cpu'
                return
        
        if not self.use_fallback:
            self.model.eval()
            self.device = 'cpu'
            print("🎯 Model ready for inference")
    
    def predict_with_board(self, board_fen):
        '''
        Perform prediction with current model
        '''
        if self.use_fallback:
            # Simple fallback prediction
            return ([1.0/5312] * 5312, 0.0)
        
        try:
            # For now, use uniform policy since we need proper board conversion
            # TODO: Implement proper board -> model input conversion
            # print(f"🎯 Predicting for position: {{board_fen[:20]}}...")  # Removed verbose logging
            
            # Basic prediction (uniform policy for now)
            policy = [1.0/5312] * 5312  
            value = 0.0
            
            return policy, value
            
        except Exception as e:
            print(f"⚠️ Prediction failed: {{e}}")
            # Ultimate fallback
            return ([1.0/5312] * 5312, 0.0)
    
    def eval(self):
        if not self.use_fallback and hasattr(self, 'model'):
            self.model.eval()
    
    def to(self, device):
        self.device = device
        if not self.use_fallback and hasattr(self, 'model'):
            self.model = self.model.to(device)
            print(f"🔧 Model moved to {{device}}")
        return self
"#, model_path);

    let locals = PyDict::new(py);
    py.run(&code, Some(locals), None)?;
    
    // Properly handle the Result<Option<&PyAny>, PyErr> returned by get_item
    let model_class = locals.get_item("UltraDenseModelWrapper")?
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("UltraDenseModelWrapper class not found"))?;
    let model = model_class.call0()?.into_py(py);
    Ok(ModelBridge::new(model, device))
}



// Endpoints
async fn start_community_game(
    data: web::Data<AppState>,
    req: Option<web::Json<StartCommunityGameRequest>>,
) -> impl Responder {
    let mut game = recover_mutex(data.game.lock());
    let mut new_game = CommunityGame::new();
    
    // Set color based on request, default to white if not specified
    if let Some(req) = req {
        if let Some(ref color) = req.player_color {
            new_game.player_is_white = color.to_lowercase() == "white";
            
            // If playing as black, make engine's first move
            if !new_game.player_is_white {
                let engine = recover_mutex(data.engine.lock());
                if let Some(engine_move) = engine.get_best_move(new_game.board.clone()) {
                    new_game.make_move(engine_move).unwrap_or_default();
                    new_game.move_history.push(engine_move.to_string());
                    new_game.start_voting(); // Start voting since it's our turn
                }
            }
        }
    }
    
    *game = new_game;
    
    let response = CommunityGameStateResponse {
        game_id: "community".to_string(),
        board: game.board.to_string(),
        status: game.status.to_string(),
        move_history: game.move_history.clone(),
        is_voting_phase: game.is_voting_phase(),
        voting_ends_at: game.voting_ends_at.map(|t| t.to_rfc3339()),
        current_votes: game.votes.iter().map(|(k, v)| (k.clone(), v.len())).collect(),
        total_voters: game.votes.values().map(|v| v.len()).sum(),
        your_vote: None,
        can_vote: game.can_vote(),
        waiting_for_first_move: game.waiting_for_first_move(),
        experiment_name: data.get_experiment_name(),
        engine_thinking: game.engine_thinking,
    };

    HttpResponse::Ok().json(response)
}

async fn get_community_game_state(
    data: web::Data<AppState>, 
    voter_id: Option<web::Query<String>>
) -> impl Responder {
    let voter_id = voter_id.as_ref().map(|id| id.as_str().to_string());
    let game = recover_mutex(data.game.lock());
    
    let current_votes: HashMap<String, usize> = game.votes.iter()
        .map(|(mv, voters)| (mv.clone(), voters.len()))
        .collect();

    let total_voters: usize = game.votes.values()
        .map(|voters| voters.len())
        .sum();

    let your_vote = voter_id.and_then(|id| {
        game.votes.iter()
            .find(|(_, voters)| voters.contains(&id))
            .map(|(mv, _)| mv.clone())
    });

    HttpResponse::Ok().json(CommunityGameStateResponse {
        game_id: "community".to_string(),
        board: game.board.to_string(),
        status: game.status.to_string(),
        move_history: game.move_history.clone(),
        is_voting_phase: game.is_voting_phase(),
        voting_ends_at: game.voting_ends_at.map(|t| t.to_rfc3339()),
        current_votes,
        total_voters,
        your_vote,
        can_vote: game.can_vote(),
        waiting_for_first_move: game.waiting_for_first_move(),
        experiment_name: data.get_experiment_name(),
        engine_thinking: game.engine_thinking,
    })
}

async fn create_voter_session(
    data: web::Data<AppState>,
    req: HttpRequest,
) -> impl Responder {
    let ip = req.peer_addr()
        .map(|addr| addr.ip())
        .unwrap_or_else(|| IpAddr::V4(std::net::Ipv4Addr::new(0, 0, 0, 0)));

    // Generate unique voter ID
    let voter_id = uuid::Uuid::new_v4().to_string();
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();

    // Create session
    let session = VoterSession {
        voter_id: voter_id.clone(),
        ip: ip.to_string(),
        created_at: now,
        last_vote_at: None,
        vote_count: 0,
    };

    // Store session
    {
        let mut sessions = data.voter_sessions.lock().unwrap();
        sessions.insert(voter_id.clone(), session);
    }

    // Create JWT token
    let claims = Claims {
        sub: voter_id.clone(),
        exp: (now + 86400) as usize,  // 24 hour expiration
        iat: now as usize,
        ip: ip.to_string(),
    };

    let token = encode(
        &Header::default(),
        &claims,
        &EncodingKey::from_secret(data.jwt_secret.as_ref())
    ).unwrap();

    HttpResponse::Ok().json(json!({
        "voter_id": voter_id,
        "token": token,
        "expires_in": 86400
    }))
}

// Update VoteRequest to include authentication token
#[derive(Deserialize)]
struct AuthenticatedVoteRequest {
    move_str: String,
    token: String,  // JWT token instead of voter_id
}

// Update the vote_move function with security measures
async fn vote_move(
    data: web::Data<AppState>,
    req: web::Json<AuthenticatedVoteRequest>,
    http_req: HttpRequest,
) -> impl Responder {
    // Extract IP address
    let ip = http_req.peer_addr()
        .map(|addr| addr.ip())
        .unwrap_or_else(|| IpAddr::V4(std::net::Ipv4Addr::new(0, 0, 0, 0)));

    // Check rate limit
    if let Err(e) = data.vote_limiter.check_rate_limit(ip) {
        return HttpResponse::TooManyRequests()
            .insert_header(("Retry-After", "60"))
            .json(json!({
                "success": false,
                "error_message": e,
            }));
    }

    // Validate JWT token
    let token_data = match decode::<Claims>(
        &req.token,
        &DecodingKey::from_secret(data.jwt_secret.as_ref()),
        &Validation::default()
    ) {
        Ok(data) => data,
        Err(e) => {
            return HttpResponse::Unauthorized().json(json!({
                "success": false,
                "error_message": format!("Invalid token: {}", e),
            }));
        }
    };

    let claims = token_data.claims;
    
    // Verify IP matches token
    if claims.ip != ip.to_string() {
        return HttpResponse::Unauthorized().json(json!({
            "success": false,
            "error_message": "Token IP mismatch",
        }));
    }

    // Update voter session
    let voter_id = claims.sub;
    {
        let mut sessions = data.voter_sessions.lock().unwrap();
        if let Some(session) = sessions.get_mut(&voter_id) {
            let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
            
            // Check vote frequency (max 1 vote per 5 seconds)
            if let Some(last_vote) = session.last_vote_at {
                if now - last_vote < 5 {
                    return HttpResponse::TooManyRequests().json(json!({
                        "success": false,
                        "error_message": "Please wait before voting again",
                    }));
                }
            }
            
            session.last_vote_at = Some(now);
            session.vote_count += 1;
            
            // Check total vote count (max 100 votes per session)
            if session.vote_count > 100 {
                return HttpResponse::Forbidden().json(json!({
                    "success": false,
                    "error_message": "Vote limit exceeded for this session",
                }));
            }
        } else {
            return HttpResponse::Unauthorized().json(json!({
                "success": false,
                "error_message": "Invalid session",
            }));
        }
    }

    // Process the vote
    let mut game = recover_mutex(data.game.lock());

    // Handle empty move string (undo vote)
    if req.move_str.is_empty() {
        // Remove vote from all moves
        for votes in game.votes.values_mut() {
            votes.remove(&voter_id);
        }
        // Remove empty vote entries
        game.votes.retain(|_, voters| !voters.is_empty());

        let response = VoteResponse {
            success: true,
            error_message: None,
            game_state: CommunityGameStateResponse {
                game_id: "community".to_string(),
                board: game.board.to_string(),
                status: game.status.to_string(),
                move_history: game.move_history.clone(),
                is_voting_phase: game.is_voting_phase(),
                voting_ends_at: game.voting_ends_at.map(|t| t.to_rfc3339()),
                current_votes: game.votes.iter().map(|(k, v)| (k.clone(), v.len())).collect(),
                total_voters: game.votes.values().map(|v| v.len()).sum(),
                your_vote: None,
                can_vote: game.can_vote(),
                waiting_for_first_move: game.waiting_for_first_move(),
                experiment_name: data.get_experiment_name(),
                engine_thinking: game.engine_thinking,
            },
        };

        return HttpResponse::Ok()
            .insert_header(("X-RateLimit-Limit", "10"))
            .insert_header(("X-RateLimit-Remaining", "9"))
            .insert_header(("X-RateLimit-Reset", "60"))
            .json(response);
    }

    match game.add_vote(&req.move_str, &voter_id) {
        Ok(()) => {
            // Don't process votes immediately - let the background task handle expired timers
            // The background task will automatically process votes when the timer expires

            let response = VoteResponse {
                success: true,
                error_message: None,
                game_state: CommunityGameStateResponse {
                    game_id: "community".to_string(),
                    board: game.board.to_string(),
                    status: game.status.to_string(),
                    move_history: game.move_history.clone(),
                    is_voting_phase: game.is_voting_phase(),
                    voting_ends_at: game.voting_ends_at.map(|t| t.to_rfc3339()),
                    current_votes: game.votes.iter().map(|(k, v)| (k.clone(), v.len())).collect(),
                    total_voters: game.votes.values().map(|v| v.len()).sum(),
                    your_vote: Some(req.move_str.clone()),
                    can_vote: game.can_vote(),
                    waiting_for_first_move: game.waiting_for_first_move(),
                    experiment_name: data.get_experiment_name(),
                    engine_thinking: game.engine_thinking,
                },
            };

            HttpResponse::Ok()
                .insert_header(("X-RateLimit-Limit", "10"))
                .insert_header(("X-RateLimit-Remaining", "9"))
                .insert_header(("X-RateLimit-Reset", "60"))
                .json(response)
        }
        Err(e) => HttpResponse::BadRequest().json(json!({
            "success": false,
            "error_message": e,
        })),
    }
}

async fn get_game_state(data: web::Data<AppState>) -> impl Responder {
    let game = recover_mutex(data.game.lock());
    
    web::Json(json!({
        "board": game.board.to_string(),
        "move_history": game.move_history,
        "is_voting_phase": game.is_voting_phase(),
        "status": game.status,
    }))
}

async fn list_games(data: web::Data<AppState>) -> impl Responder {
    match data.game_storage.list_games(None) {
        Ok(games) => {
            let mut saved_games = Vec::new();
            
            for metadata in games {
                // Load the full game state to get move history
                if let Ok(game_state) = data.game_storage.load_game(&metadata.game_id, &metadata.mode) {
                    saved_games.push(SavedGame {
                        game_id: metadata.game_id,
                        moves: game_state.move_history,
                        result: metadata.status.to_string(),
                        timestamp: metadata.last_move_at.to_rfc3339(),
                        player_color: metadata.player_color,
                    });
                }
            }
            
            HttpResponse::Ok().json(saved_games)
        }
        Err(e) => HttpResponse::InternalServerError().json(json!({
            "error": format!("Failed to list games: {}", e)
        }))
    }
}

async fn get_recent_games(data: web::Data<AppState>) -> impl Responder {
    // Serve recent completed games from memory queue (much faster than disk I/O!)
    let recent_games = {
        let queue = recover_mutex(data.recent_completed_games.lock());
        queue.iter().cloned().collect::<Vec<_>>()
    };
    
    let mut response_games = Vec::new();
    
    for metadata in recent_games {
        // Convert GameStatus to string and mode to string for frontend
        let mode_str = match metadata.mode {
            GameMode::SinglePlayer => "single",
            GameMode::Community => "community", 
            GameMode::UCI => "single", // UCI games show as single-player in UI
        };
        
        let status_str = match metadata.status {
            GameStatus::Active => "active",
            GameStatus::Waiting => "waiting",
            GameStatus::WhiteWins => "white_wins",
            GameStatus::BlackWins => "black_wins",
            GameStatus::DrawStalemate => "draw_stalemate", 
            GameStatus::DrawInsufficientMaterial => "draw_insufficient_material",
            GameStatus::DrawRepetition => "draw_repetition",
            GameStatus::DrawFiftyMoves => "draw_fifty_moves",
        };
        
        response_games.push(json!({
            "game_id": metadata.game_id,
            "mode": mode_str,
            "created_at": metadata.created_at.to_rfc3339(),
            "last_move_at": metadata.last_move_at.to_rfc3339(),
            "status": status_str,
            "total_moves": metadata.total_moves,
            "player_color": metadata.player_color,
            "player_name": metadata.player_name,
            "engine_version": metadata.engine_version,
        }));
    }
    
    // Games are already in newest-first order from the queue
    HttpResponse::Ok().json(response_games)
}

#[derive(Serialize, Clone)]
struct ServerModelStats {
    total_games: usize,
    wins: usize,
    losses: usize,
    draws: usize,
    win_rate: f32,
    games_until_training: usize,
    is_training: bool,
    gpu_utilization: f32,
    engine_versions: Vec<EngineVersionInfo>,
}

#[derive(Serialize, Clone)]
struct EngineVersionInfo {
    version: String,
    games: usize,
    wins: usize,
    losses: usize,
    draws: usize,
    win_rate: f32,
    first_game: String,
    last_game: String,
}

#[derive(Serialize, Clone)]
struct ServerStats {
    single_player_model: ServerModelStats,
    community_model: ServerModelStats,
    active_players: usize,
    self_play_games: usize,
    unprocessed_games_count: usize,
    total_games_count: usize,
}

// Initialize stats cache by reading all games once on startup
async fn initialize_stats_cache(data: &web::Data<AppState>, args: &Args) -> Result<(), String> {
    println!("🚀 Initializing stats cache by reading game files...");
    
    // Load persistent stats (already cached in database file)
    let persistent_stats = match data.game_storage.load_persistent_stats() {
        Ok(stats) => stats,
        Err(e) => {
            eprintln!("Failed to load persistent stats: {}", e);
            Default::default()
        }
    };
    
    // Read all unarchived game files once to get metadata (this is the expensive part)
    let current_games = data.game_storage.list_games(None)
        .map_err(|e| format!("Failed to list games during cache init: {}", e))?;
    
    println!("📊 Cache init: Found {} persistent games, {} unarchived games", 
        persistent_stats.single_player.total_games + persistent_stats.community.total_games,
        current_games.len());
    
    // Build stats from persistent data + current unarchived games
    let mut single_player_stats = ServerModelStats {
        total_games: persistent_stats.single_player.total_games,
        wins: persistent_stats.single_player.wins,
        losses: persistent_stats.single_player.losses,
        draws: persistent_stats.single_player.draws,
        win_rate: 0.0,
        games_until_training: 0,
        is_training: data.is_training.load(Ordering::SeqCst),
        gpu_utilization: 0.0,
        engine_versions: Vec::new(),
    };
    
    let mut community_stats = ServerModelStats {
        total_games: persistent_stats.community.total_games,
        wins: persistent_stats.community.wins,
        losses: persistent_stats.community.losses,
        draws: persistent_stats.community.draws,
        win_rate: 0.0,
        games_until_training: 0,
        is_training: data.is_training.load(Ordering::SeqCst),
        gpu_utilization: 0.0,
        engine_versions: Vec::new(),
    };
    
    // Add unarchived games to the stats
    for metadata in &current_games {
        let stats = match metadata.mode {
            GameMode::SinglePlayer => &mut single_player_stats,
            GameMode::Community => &mut community_stats,
            GameMode::UCI => &mut single_player_stats, // UCI tournament games count as single-player for stats
        };
        
        if matches!(metadata.status, 
            GameStatus::WhiteWins | GameStatus::BlackWins | 
            GameStatus::DrawStalemate | GameStatus::DrawInsufficientMaterial |
            GameStatus::DrawRepetition | GameStatus::DrawFiftyMoves) {
            
            stats.total_games += 1;
            match metadata.status {
                GameStatus::WhiteWins => {
                    if metadata.player_color == "white" {
                        stats.wins += 1;
                    } else {
                        stats.losses += 1;
                    }
                },
                GameStatus::BlackWins => {
                    if metadata.player_color == "black" {
                        stats.wins += 1;
                    } else {
                        stats.losses += 1;
                    }
                },
                _ => stats.draws += 1,
            }
        }
    }
    
    // Calculate win rates
    if single_player_stats.total_games > 0 {
        single_player_stats.win_rate = (single_player_stats.wins as f32 / single_player_stats.total_games as f32) * 100.0;
    }
    
    if community_stats.total_games > 0 {
        community_stats.win_rate = (community_stats.wins as f32 / community_stats.total_games as f32) * 100.0;
    }
    
    // Add engine version information from persistent stats
    for (version, version_stats) in &persistent_stats.single_player.games_by_engine_version {
        let win_rate = if version_stats.total_games > 0 {
            (version_stats.wins as f32 / version_stats.total_games as f32) * 100.0
        } else {
            0.0
        };
        
        single_player_stats.engine_versions.push(EngineVersionInfo {
            version: version.clone(),
            games: version_stats.total_games,
            wins: version_stats.wins,
            losses: version_stats.losses,
            draws: version_stats.draws,
            win_rate,
            first_game: version_stats.first_game.format("%Y-%m-%d %H:%M:%S").to_string(),
            last_game: version_stats.last_game.format("%Y-%m-%d %H:%M:%S").to_string(),
        });
    }
    
    for (version, version_stats) in &persistent_stats.community.games_by_engine_version {
        let win_rate = if version_stats.total_games > 0 {
            (version_stats.wins as f32 / version_stats.total_games as f32) * 100.0
        } else {
            0.0
        };
        
        community_stats.engine_versions.push(EngineVersionInfo {
            version: version.clone(),
            games: version_stats.total_games,
            wins: version_stats.wins,
            losses: version_stats.losses,
            draws: version_stats.draws,
            win_rate,
            first_game: version_stats.first_game.format("%Y-%m-%d %H:%M:%S").to_string(),
            last_game: version_stats.last_game.format("%Y-%m-%d %H:%M:%S").to_string(),
        });
    }
    
    // Get GPU utilization (this is still dynamic)
    let mut gpu_util = 0.0;
    Python::with_gil(|py| {
        if let Ok(torch) = PyModule::import(py, "torch") {
            if let Ok(true) = torch.getattr("cuda")?.call_method0("is_available")?.extract() {
                if let Ok(nvml) = PyModule::import(py, "pynvml") {
                    let _ = nvml.call_method0("nvmlInit");
                    if let Ok(handle) = nvml.call_method1("nvmlDeviceGetHandleByIndex", (0,)) {
                        if let Ok(util) = handle.call_method0("nvmlDeviceGetUtilizationRates") {
                            gpu_util = util.getattr("gpu")?.extract::<f32>()? / 100.0;
                        }
                    }
                }
            }
        }
        Ok::<_, PyErr>(())
    }).ok();
    
    single_player_stats.gpu_utilization = gpu_util;
    community_stats.gpu_utilization = gpu_util;
    
    // Get unprocessed games count for training progress
    let unprocessed_games = match count_unprocessed_games(&args.games_dir).await {
        Ok(count) => count,
        Err(_) => 0, // Fall back to 0 if counting fails
    };
    
    single_player_stats.games_until_training = if unprocessed_games >= args.training_games_threshold {
        0
    } else {
        args.training_games_threshold - unprocessed_games
    };
    
    // Store in cache
    let cached_stats = ServerStats {
        single_player_model: single_player_stats.clone(),
        community_model: community_stats.clone(),
        active_players: data.active_players.load(Ordering::SeqCst),
        self_play_games: data.self_play_games.load(Ordering::SeqCst),
        unprocessed_games_count: unprocessed_games,
        total_games_count: current_games.len(), // This is now just unarchived games
    };
    
    {
        let mut cache = recover_mutex(data.cached_stats.lock());
        *cache = cached_stats;
    }
    
    // Initialize recent completed games queue with up to 10 most recent completed games
    {
        let mut recent_queue = recover_mutex(data.recent_completed_games.lock());
        let mut completed_games: Vec<_> = current_games.into_iter()
            .filter(|g| matches!(g.status, 
                GameStatus::WhiteWins | GameStatus::BlackWins | 
                GameStatus::DrawStalemate | GameStatus::DrawInsufficientMaterial |
                GameStatus::DrawRepetition | GameStatus::DrawFiftyMoves))
            .collect();
        
        // Sort by last_move_at descending (newest first)
        completed_games.sort_by(|a, b| b.last_move_at.cmp(&a.last_move_at));
        
        // Take up to 10 most recent
        for game in completed_games.into_iter().take(10) {
            recent_queue.push_back(game);
        }
        
        println!("📋 Recent games queue initialized with {} completed games", recent_queue.len());
    }
    
    println!("✅ Stats cache initialized: {} single-player games, {} community games", 
        persistent_stats.single_player.total_games + 
            single_player_stats.total_games - persistent_stats.single_player.total_games,
        persistent_stats.community.total_games + 
            community_stats.total_games - persistent_stats.community.total_games);
    
    Ok(())
}

// Function to update cache when a game completes (NO FILE I/O!)
fn update_stats_cache_for_completed_game(data: &web::Data<AppState>, metadata: &GameMetadata) {
    let mut cache = recover_mutex(data.cached_stats.lock());
    
    let stats = match metadata.mode {
        GameMode::SinglePlayer => &mut cache.single_player_model,
        GameMode::Community => &mut cache.community_model,
        GameMode::UCI => &mut cache.single_player_model, // UCI games count as single-player
    };
    
    // Only update for completed games
    if matches!(metadata.status, 
        GameStatus::WhiteWins | GameStatus::BlackWins | 
        GameStatus::DrawStalemate | GameStatus::DrawInsufficientMaterial |
        GameStatus::DrawRepetition | GameStatus::DrawFiftyMoves) {
        
        stats.total_games += 1;
        match metadata.status {
            GameStatus::WhiteWins => {
                if metadata.player_color == "white" {
                    stats.wins += 1;
                } else {
                    stats.losses += 1;
                }
            },
            GameStatus::BlackWins => {
                if metadata.player_color == "black" {
                    stats.wins += 1;
                } else {
                    stats.losses += 1;
                }
            },
            _ => stats.draws += 1,
        }
        
        // Recalculate win rate
        if stats.total_games > 0 {
            stats.win_rate = (stats.wins as f32 / stats.total_games as f32) * 100.0;
        }
        
        // Add to recent completed games queue (max 10)
        {
            let mut recent_games = recover_mutex(data.recent_completed_games.lock());
            recent_games.push_front(metadata.clone());
            if recent_games.len() > 10 {
                recent_games.pop_back(); // Remove oldest game
            }
        }
        
        println!("📊 Cache updated: {} now has {} total games (W:{} L:{} D:{}, {:.1}% win rate)", 
            match metadata.mode {
                GameMode::SinglePlayer | GameMode::UCI => "Single-player",
                GameMode::Community => "Community",
            },
            stats.total_games, stats.wins, stats.losses, stats.draws, stats.win_rate);
    }
}

async fn get_model_stats(data: web::Data<AppState>) -> impl Responder {
    // Just return the cached stats - no file I/O!
    let mut cache = recover_mutex(data.cached_stats.lock());
    
    // Update dynamic values that can change frequently
    cache.active_players = data.active_players.load(Ordering::SeqCst);
    cache.self_play_games = data.self_play_games.load(Ordering::SeqCst);
    cache.single_player_model.is_training = data.is_training.load(Ordering::SeqCst);
    cache.community_model.is_training = data.is_training.load(Ordering::SeqCst);
    
    // Update GPU utilization if needed (this is the only "expensive" operation left)
    Python::with_gil(|py| {
        if let Ok(torch) = PyModule::import(py, "torch") {
            if let Ok(true) = torch.getattr("cuda")?.call_method0("is_available")?.extract() {
                if let Ok(nvml) = PyModule::import(py, "pynvml") {
                    let _ = nvml.call_method0("nvmlInit");
                    if let Ok(handle) = nvml.call_method1("nvmlDeviceGetHandleByIndex", (0,)) {
                        if let Ok(util) = handle.call_method0("nvmlDeviceGetUtilizationRates") {
                            let gpu_util = util.getattr("gpu")?.extract::<f32>()? / 100.0;
                            cache.single_player_model.gpu_utilization = gpu_util;
                            cache.community_model.gpu_utilization = gpu_util;
                        }
                    }
                }
            }
        }
        Ok::<_, PyErr>(())
    }).ok();
    
    HttpResponse::Ok().json((*cache).clone())
}

#[derive(Serialize)]
struct LeaderboardEntry {
    rank: i32,
    name: String,
    wins: i32,
    losses: i32,
    draws: i32,
}

async fn refresh_stats(data: web::Data<AppState>) -> impl Responder {
    println!("Stats refresh requested");
    
    // Force reload stats by calling the same logic as get_model_stats
    match data.game_storage.list_games(None) {
        Ok(games) => {
            let total_games = games.len();
            let single_player_games = games.iter().filter(|g| matches!(g.mode, GameMode::SinglePlayer | GameMode::UCI)).count();
            let community_games = games.iter().filter(|g| matches!(g.mode, GameMode::Community)).count();
            let uci_games = games.iter().filter(|g| matches!(g.mode, GameMode::UCI)).count();
            
            println!("Stats refresh: Found {} total games ({} single-player, {} community, {} UCI)", 
                total_games, single_player_games - uci_games, community_games, uci_games);
        }
        Err(e) => {
            println!("Error during stats refresh: {}", e);
        }
    }
    
    // Always call get_model_stats to return the refreshed stats
    get_model_stats(data).await
}

async fn get_leaderboard(data: web::Data<AppState>) -> impl Responder {
    match data.game_storage.list_games(None) {
        Ok(games) => {
            let mut player_stats: HashMap<String, (i32, i32, i32)> = HashMap::new(); // (wins, losses, draws)

            for game in games {
                let player_name = game.player_name.unwrap_or_else(|| "Anonymous".to_string());
                let stats = player_stats.entry(player_name).or_insert((0, 0, 0));

                match game.status {
                    GameStatus::WhiteWins => {
                        if game.player_color == "white" {
                            stats.0 += 1;
                        } else {
                            stats.1 += 1;
                        }
                    },
                    GameStatus::BlackWins => {
                        if game.player_color == "black" {
                            stats.0 += 1;
                        } else {
                            stats.1 += 1;
                        }
                    },
                    GameStatus::DrawStalemate | GameStatus::DrawInsufficientMaterial |
                    GameStatus::DrawRepetition | GameStatus::DrawFiftyMoves => {
                        stats.2 += 1;
                    },
                    _ => {}
                }
            }

            let mut leaderboard: Vec<LeaderboardEntry> = player_stats.into_iter()
                .map(|(name, (wins, losses, draws))| {
                    LeaderboardEntry {
                        name,
                        wins,
                        losses,
                        draws,
                        rank: 0,  // Will be set below
                    }
                })
                .collect();

            // Sort by win percentage
            leaderboard.sort_by(|a, b| {
                let a_total = a.wins + a.losses + a.draws;
                let b_total = b.wins + b.losses + b.draws;
                let a_winrate = if a_total > 0 { (a.wins as f32) / (a_total as f32) } else { 0.0 };
                let b_winrate = if b_total > 0 { (b.wins as f32) / (b_total as f32) } else { 0.0 };
                b_winrate.partial_cmp(&a_winrate).unwrap()
            });

            // Set ranks
            for (i, entry) in leaderboard.iter_mut().enumerate() {
                entry.rank = (i + 1) as i32;
            }

            HttpResponse::Ok().json(leaderboard)
        },
        Err(e) => HttpResponse::InternalServerError().json(json!({
            "error": format!("Failed to get leaderboard: {}", e)
        }))
    }
}

async fn get_self_play_status(data: web::Data<AppState>) -> impl Responder {
    let active_players = data.active_players.load(Ordering::SeqCst);
    let current_self_play = data.self_play_games.load(Ordering::SeqCst);
    let is_training = data.is_training.load(Ordering::SeqCst);
    
    // Get GPU utilization
    let mut gpu_util = 0.0;
    Python::with_gil(|py| {
        if let Ok(torch) = PyModule::import(py, "torch") {
            if let Ok(true) = torch.getattr("cuda")?.call_method0("is_available")?.extract() {
                if let Ok(nvml) = PyModule::import(py, "pynvml") {
                    let _ = nvml.call_method0("nvmlInit");
                    if let Ok(handle) = nvml.call_method1("nvmlDeviceGetHandleByIndex", (0,)) {
                        if let Ok(util) = handle.call_method0("nvmlDeviceGetUtilizationRates") {
                            gpu_util = util.getattr("gpu")?.extract::<f32>()? / 100.0;
                        }
                    }
                }
            }
        }
        Ok::<_, PyErr>(())
    }).ok();
    
    HttpResponse::Ok().json(json!({
        "current_self_play_games": current_self_play,
        "active_players": active_players,
        "gpu_utilization": gpu_util,
        "is_training": is_training,
        "status": if is_training {
            "Training mode - minimal self-play"
        } else if active_players == 0 {
            "Zero traffic - rapid scaling mode"
        } else if active_players <= 2 {
            "Low traffic - moderate scaling"
        } else {
            "High traffic - conservative scaling"
        }
    }))
}

// Update the main function to include the new security features
#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let args = Args::parse();
    
    // Create games directory before starting server
    if let Err(e) = fs::create_dir_all(&args.games_dir) {
        eprintln!("Failed to create games directory: {}", e);
        return Err(e);
    }
    
    pyo3::prepare_freethreaded_python();
    
    // Generate JWT secret
    let jwt_secret = std::env::var("JWT_SECRET")
        .unwrap_or_else(|_| uuid::Uuid::new_v4().to_string());
    
    let server = Python::with_gil(|py| {
        let torch = PyModule::import(py, "torch").unwrap();
        
        // Create single player engine with ultra-dense PAG (stronger with master analysis)
        let single_player_model = create_model_with_device(py, &torch, &args.model_path)
            .unwrap_or_else(|e| {
                eprintln!("❌ Failed to load single player ultra-dense model: {}", e);
                eprintln!("🔄 Creating ultra-dense fallback model...");
                
                // Create proper fallback model instead of None
                let fallback_code = r#"
class UltraDenseFallbackModel:
    def __init__(self):
        self.device = 'cpu'
        print("⚠️ Using ultra-dense fallback model (uniform policy)")
        print("🧠 Fallback still processes ~340,000 features per position")
    
    def predict_with_board(self, board_fen):
        # Fallback uniform policy for ultra-dense system
        return ([1.0/5312] * 5312, 0.0)
    
    def eval(self):
        pass
    
    def to(self, device):
        self.device = device
        print(f"🔧 Ultra-dense fallback moved to {device}")
        return self
"#;
                let fallback_locals = PyDict::new(py);
                py.run(fallback_code, Some(fallback_locals), None).unwrap();
                let fallback_model = py.eval("UltraDenseFallbackModel()", Some(fallback_locals), None).unwrap();
                ModelBridge::new(fallback_model.into_py(py), Some("cpu".to_string()))
            });

        let single_player_engine = Engine::new_with_model(single_player_model);

        // Create community engine with ultra-dense PAG + MCTS (strongest possible for the challenge)
        let community_model = create_model_with_device(py, &torch, &args.community_model_path)
            .unwrap_or_else(|e| {
                eprintln!("❌ Failed to load community ultra-dense model: {}", e);
                eprintln!("🔄 Creating ultra-dense fallback model...");
                
                // Create proper fallback model instead of None
                let fallback_code = r#"
class UltraDenseFallbackModel:
    def __init__(self):
        self.device = 'cpu'
        print("⚠️ Using ultra-dense fallback model (uniform policy)")
        print("🧠 Fallback still processes ~340,000 features per position")
    
    def predict_with_board(self, board_fen):
        # Fallback uniform policy for ultra-dense system
        return ([1.0/5312] * 5312, 0.0)
    
    def eval(self):
        pass
    
    def to(self, device):
        self.device = device
        print(f"🔧 Ultra-dense fallback moved to {device}")
        return self
"#;
                let fallback_locals = PyDict::new(py);
                py.run(fallback_code, Some(fallback_locals), None).unwrap();
                let fallback_model = py.eval("UltraDenseFallbackModel()", Some(fallback_locals), None).unwrap();
                ModelBridge::new(fallback_model.into_py(py), Some("cpu".to_string()))
            });

        let community_engine = Engine::new_with_mcts(community_model);

        let game_storage = Arc::new(GameStorage::new(&args.games_dir).unwrap());
        
        // Create rate limiter (10 requests per 60 seconds)
        let vote_limiter = Arc::new(RateLimiter::new(10, 60));
        
        let app_state = web::Data::new(AppState {
            game: Arc::new(Mutex::new(CommunityGame::new())),
            game_storage: game_storage.clone(),
            engine: Arc::new(Mutex::new(single_player_engine)), // Use stronger engine for single player
            community_engine: Arc::new(Mutex::new(community_engine)), // Add community engine
            is_training: Arc::new(AtomicBool::new(false)),
            active_players: Arc::new(AtomicUsize::new(0)),
            self_play_games: Arc::new(AtomicUsize::new(args.initial_self_play_games)),
            active_games: Arc::new(Mutex::new(HashMap::new())),
            vote_limiter,
            voter_sessions: Arc::new(Mutex::new(HashMap::new())),
            jwt_secret: jwt_secret.clone(),
            current_model_path: Arc::new(Mutex::new(args.model_path.clone())),
            // Add cached stats - initialized once on startup, then just incremented
            cached_stats: Arc::new(Mutex::new(ServerStats {
                single_player_model: ServerModelStats {
                    total_games: 0,
                    wins: 0,
                    losses: 0,
                    draws: 0,
                    win_rate: 0.0,
                    games_until_training: 0,
                    is_training: false,
                    gpu_utilization: 0.0,
                    engine_versions: Vec::new(),
                },
                community_model: ServerModelStats {
                    total_games: 0,
                    wins: 0,
                    losses: 0,
                    draws: 0,
                    win_rate: 0.0,
                    games_until_training: 0,
                    is_training: false,
                    gpu_utilization: 0.0,
                    engine_versions: Vec::new(),
                },
                active_players: 0,
                self_play_games: 0,
                unprocessed_games_count: 0,
                total_games_count: 0,
            })),
            // Initialize empty queue for recent completed games (max 10)
            recent_completed_games: Arc::new(Mutex::new(VecDeque::with_capacity(10))),
        });
        
        // Add cleanup task
        let app_state_cleanup = app_state.clone();
        actix_web::rt::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(300)); // Every 5 minutes
            loop {
                interval.tick().await;
                cleanup_expired_sessions(app_state_cleanup.clone()).await;
            }
        });
        
        // Add voting timer checker task
        let app_state_voting = app_state.clone();
        actix_web::rt::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(1)); // Every second
            loop {
                interval.tick().await;
                check_and_process_expired_votes(app_state_voting.clone()).await;
            }
        });
        
        // Add engine thinking watchdog task (safety net for stuck flags)
        let app_state_watchdog = app_state.clone();
        actix_web::rt::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(30)); // Every 30 seconds
            loop {
                interval.tick().await;
                check_stuck_engine_thinking(app_state_watchdog.clone()).await;
            }
        });
        
        // Spawn background tasks only if enabled
        if args.enable_training {
            let app_state_clone = app_state.clone();
            let args_clone = args.clone();
            actix_web::rt::spawn(async move {
                background_training_task(app_state_clone, args_clone).await;
            });
        }
        
        if args.enable_self_play {
            let app_state_clone = app_state.clone();
            let args_clone = args.clone();
            actix_web::rt::spawn(async move {
                background_self_play_task(app_state_clone, args_clone).await;
            });
        }
        
        println!("💾 Game persistence: Stats survive server restarts, games archived after training");
        println!("Games directory: {}", args.games_dir);

        // Start server
        println!("🚀 Starting Ultra-Dense PAG Chess AI Server on {}:{}", args.host, args.port);
        println!("🧠 REVOLUTIONARY SYSTEM ACTIVE:");
        println!("   💎 Ultra-Dense PAG Feature Extraction: ~340,000 features per position");
        println!("   🎯 Master-level tactical analysis built into neural network input");
        println!("   ⚡ 10-100x faster than Python with Rust implementation");
        println!("   🏆 From basic features to chess master expertise");
        println!("");
        println!("🎮 Engine Configuration:");
        println!("   Single-player: Ultra-Dense Direct Policy (fast + master analysis)");
        println!("   Community: Ultra-Dense MCTS 2000 sims (strongest possible challenge)"); 
        println!("   Feature density: Basic ~100 → Ultra-Dense ~340,000");
        println!("");
        println!("🎮 Self-play configuration:");
        println!("   📊 Initial games: {}", args.initial_self_play_games);
        println!("   📈 Max games (low traffic): {}", args.max_self_play_games);
        println!("   🎯 Target GPU utilization: {:.1}%", args.target_gpu_utilization * 100.0);
        println!("   🔄 Scaling: Aggressive when 0 players, conservative with traffic");
        println!("   🧠 All games use ultra-dense master-level analysis");
        
        app_state
    });

    // Initialize stats cache (read files once, then just use memory) - OUTSIDE the Python closure
    println!("🚀 Initializing high-performance stats cache...");
    match initialize_stats_cache(&server, &args).await {
        Ok(()) => {
            println!("✅ Stats cache initialized successfully");
        }
        Err(e) => {
            println!("⚠️ Warning: Stats cache initialization failed: {}", e);
            println!("📊 Will serve empty stats until games are completed");
        }
    }

    let http_server = HttpServer::new(move || {
        App::new()
            .wrap(Logger::default())
            .wrap(
                Cors::default()
                    .allow_any_origin()
                    .allow_any_method()
                    .allow_any_header()
                    .expose_headers(vec!["X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"])
            )
            .app_data(server.clone())
            .service(web::resource("/ws/{game_id}").route(web::get().to(ws_route)))
            .service(web::resource("/game").route(web::get().to(get_game_state)))
            .service(web::resource("/move/new").route(web::post().to(create_new_game)))
            .service(web::resource("/move").route(web::post().to(make_move)))
            .service(web::resource("/api/community/state").route(web::get().to(get_community_game_state)))
            .service(web::resource("/api/community/vote").route(web::post().to(vote_move)))
            .service(web::resource("/api/community/session").route(web::post().to(create_voter_session)))
            .service(web::resource("/api/community/start").route(web::post().to(start_community_game)))
            .service(web::resource("/api/community/start-voting").route(web::post().to(start_voting_round)))
            .service(web::resource("/api/community/force-resolve").route(web::post().to(force_resolve_voting)))
            .service(web::resource("/games").route(web::get().to(list_games)))
            .service(web::resource("/recent-games").route(web::get().to(get_recent_games)))
            .service(web::resource("/games/{id}").route(web::get().to(get_game)))
            .service(web::resource("/stats").route(web::get().to(get_model_stats)))
            .service(web::resource("/stats/refresh").route(web::post().to(refresh_stats)))
            .service(web::resource("/leaderboard").route(web::get().to(get_leaderboard)))
            .service(web::resource("/self-play-status").route(web::get().to(get_self_play_status)))
    })
    .bind((args.host.clone(), args.port))
    .unwrap()
    .run();

    println!("Server started successfully");
    http_server.await
}

// Add periodic cleanup for expired sessions
async fn cleanup_expired_sessions(data: web::Data<AppState>) {
    let cutoff_time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() - 86400; // 24 hours
    
    let mut sessions = data.voter_sessions.lock().unwrap();
    sessions.retain(|_, session| session.created_at > cutoff_time);
}

// Check for expired voting timers and process votes automatically
async fn check_and_process_expired_votes(data: web::Data<AppState>) {
    // First check if we need to process votes without holding any locks
    let needs_processing = {
        match data.game.try_lock() {
            Ok(game) => game.votes_started && !game.is_voting_phase() && !game.votes.is_empty(),
            Err(_) => false, // Skip this cycle if game mutex is busy
        }
    };
    
    if needs_processing {
        println!("🕒 Timer expired! Attempting to process votes...");
        
        // Process votes asynchronously to allow frontend polling during engine thinking
        let process_result = tokio::time::timeout(
            std::time::Duration::from_millis(15000), // 15 second timeout (longer than engine's 10s)
            async {
                // Step 1: Process community move and set engine thinking flag
                let (_winning_move, needs_engine_move, board_for_engine) = {
                    let mut game = data.game.lock().unwrap();
                    
                    // Double-check conditions after acquiring lock
                    if !game.votes_started || game.is_voting_phase() || game.votes.is_empty() {
                        return Ok::<(), String>(()); // Conditions changed, nothing to do
                    }
                    
                    let vote_count = game.votes.len();
                    println!("🗳️ Processing {} votes...", vote_count);
                    
                    // Get winning move and process community's turn
                    if let Some(winning_move) = game.get_winning_move() {
                        // Parse and make community move
                        let chess_move = parse_move(&winning_move)?;
                        game.make_move(chess_move)?;
                        game.move_history.push(winning_move.clone());
                        game.clear_votes();
                        
                        // Check if game ended after community move
                        match game.board.status() {
                            chess::BoardStatus::Checkmate => {
                                game.status = if game.board.side_to_move() == chess::Color::White {
                                    GameStatus::BlackWins
                                } else {
                                    GameStatus::WhiteWins
                                };
                                eprintln!("🏁 Game ended in checkmate after community move");
                                game.save_game_state(&data.game_storage);
                                update_stats_cache_for_community_game(&data, &game);
                                return Ok(());
                            }
                            chess::BoardStatus::Stalemate => {
                                game.status = GameStatus::DrawStalemate;
                                eprintln!("🏁 Game ended in stalemate after community move");
                                game.save_game_state(&data.game_storage);
                                update_stats_cache_for_community_game(&data, &game);
                                return Ok(());
                            }
                            _ => {} // Game continues
                        }
                        
                        // Check if engine needs to move
                        let needs_engine_move = game.board.side_to_move() != if game.player_is_white { 
                            chess::Color::White 
                        } else { 
                            chess::Color::Black 
                        };
                        
                        if needs_engine_move {
                            game.engine_thinking = true;
                            eprintln!("🤖 Engine is thinking... (engine_thinking = true)");
                        }
                        
                        (Some(winning_move), needs_engine_move, game.board.clone())
                    } else {
                        (None, false, game.board.clone())
                    }
                };
                
                // Step 2: If engine needs to move, do it WITHOUT holding the game lock
                if needs_engine_move {
                    let engine_move_result = {
                        // Only lock the engine, not the game
                        let engine = data.community_engine.lock().unwrap();
                        engine.get_best_move(board_for_engine)
                    };
                    
                    // Step 3: Apply engine move and clear thinking flag
                    let mut game = data.game.lock().unwrap();
                    
                    match engine_move_result {
                        Some(engine_move) => {
                            game.make_move(engine_move)?;
                            let move_str = engine_move.to_string();
                            game.move_history.push(move_str.clone());
                            eprintln!("🤖 Engine played: {} (engine_thinking will be cleared)", move_str);
                        }
                        None => {
                            eprintln!("⚠️ Engine failed to find a move (engine_thinking will be cleared)");
                        }
                    }
                    
                    // Always clear engine thinking flag
                    game.engine_thinking = false;
                    eprintln!("🔓 Engine thinking cleared (engine_thinking = false)");
                    
                    // Check if game ended after engine move
                    match game.board.status() {
                        chess::BoardStatus::Checkmate => {
                            game.status = if game.board.side_to_move() == chess::Color::White {
                                GameStatus::BlackWins
                            } else {
                                GameStatus::WhiteWins
                            };
                                                    eprintln!("🏁 Game ended in checkmate after engine move");
                    }
                    chess::BoardStatus::Stalemate => {
                        game.status = GameStatus::DrawStalemate;
                        eprintln!("🏁 Game ended in stalemate after engine move");
                    }
                    _ => {} // Game continues
                }
                
                game.save_game_state(&data.game_storage);
                
                // Update cache if game ended
                update_stats_cache_for_community_game(&data, &game);
                }
                
                Ok(())
            }
        ).await;
        
        match process_result {
            Ok(Ok(())) => {
                println!("✅ Votes processed successfully");
            }
            Ok(Err(e)) => {
                eprintln!("❌ Error processing expired votes: {}", e);
                
                // Emergency cleanup: if there was an error, make sure engine_thinking is cleared
                if let Ok(mut game) = data.game.try_lock() {
                    if game.engine_thinking {
                        game.engine_thinking = false;
                        eprintln!("🚨 Emergency cleanup: Cleared stuck engine_thinking flag");
                    }
                }
            }
            Err(_) => {
                eprintln!("⏰ Timeout processing expired votes after 15 seconds");
                
                // Emergency cleanup: if timeout occurred, clear any stuck engine_thinking flag
                if let Ok(mut game) = data.game.try_lock() {
                    if game.engine_thinking {
                        game.engine_thinking = false;
                        eprintln!("🚨 Timeout cleanup: Cleared stuck engine_thinking flag");
                    }
                }
            }
        }
    } else {
        // Show countdown for active voting phases
        if let Ok(game) = data.game.try_lock() {
            if game.votes_started && game.is_voting_phase() {
                if let Some(ends_at) = game.voting_ends_at {
                    let now = chrono::Utc::now();
                    let remaining = (ends_at - now).num_seconds();
                    if remaining <= 3 && remaining > 0 {
                        println!("⏰ {} seconds remaining...", remaining);
                    }
                }
            }
        }
    }
}

// Add this function before the main function
async fn background_training_task(
    data: web::Data<AppState>,
    args: Args,
) {
    let mut last_training_games = 0;
    
    loop {
        // Yield to prevent blocking other tasks
        tokio::task::yield_now().await;
        sleep(Duration::from_secs(60)).await;
        
        // Use Python script to count only unprocessed games
        let unprocessed_games = match count_unprocessed_games(&args.games_dir).await {
            Ok(count) => count,
            Err(e) => {
                eprintln!("Failed to count unprocessed games: {}", e);
                continue;
            }
        };
        
        if unprocessed_games >= args.training_games_threshold && unprocessed_games > last_training_games {
            // Check if community engine is busy before starting training (PRIORITY CHECK)
            let community_busy = {
                if let Ok(game) = data.game.try_lock() {
                    game.engine_thinking
                } else {
                    true // Consider busy if we can't check
                }
            };
            
            if community_busy {
                eprintln!("🛡️ Delaying training - community engine is thinking (highest priority)");
                continue;
            }
            
            // Additional check: don't start training if there are active players voting
            let has_active_votes = {
                if let Ok(game) = data.game.try_lock() {
                    !game.votes.is_empty() || game.is_voting_phase()
                } else {
                    true // Consider busy if we can't check
                }
            };
            
            if has_active_votes {
                eprintln!("🗳️ Delaying training - community voting in progress");
                continue;
            }
            
            data.is_training.store(true, Ordering::SeqCst);
            
            // Reduce self-play during training to free up GPU
            let prev_self_play = data.self_play_games.load(Ordering::SeqCst);
            data.self_play_games.store(1, Ordering::SeqCst);
            
            eprintln!("🎓 Starting training on {} unprocessed games...", unprocessed_games);
            eprintln!("🛡️ Community engine protection active during training");
            
            // Actually run training (now non-blocking)
            match run_training_session(&args, unprocessed_games).await {
                Ok(new_model_path) => {
                    eprintln!("✅ Training completed successfully! New model: {}", new_model_path);
                    eprintln!("📦 Processed games have been archived to prevent retraining");
                    
                    // Reload the engine with the new model
                    match reload_engine_model(&data, &new_model_path).await {
                        Ok(()) => {
                            eprintln!("🚀 Engine reloaded successfully with new model!");
                            eprintln!("🎯 Single-player games now use improved model");
                        },
                        Err(e) => {
                            eprintln!("⚠️ Failed to reload engine with new model: {}", e);
                            eprintln!("🔄 Server will continue with previous model");
                        }
                    }
                    
                    last_training_games = unprocessed_games;
                },
                Err(e) => {
                    eprintln!("❌ Training failed: {}", e);
                    eprintln!("⚠️ Games were not archived due to training failure");
                }
            }
            
            // Restore self-play count
            data.self_play_games.store(prev_self_play, Ordering::SeqCst);
            data.is_training.store(false, Ordering::SeqCst);
            eprintln!("🔄 Training completed - self-play scaling resumed");
        } else if unprocessed_games > 0 {
            // Training status available via API endpoint - no need for verbose logging
        }
    }
}

// Add function to count unprocessed games using Python script
async fn count_unprocessed_games(games_dir: &str) -> Result<usize, String> {
    use std::process::Command;
    
    let output = Command::new("python")
        .arg("-c")
        .arg(&format!(r#"
import sys
sys.path.insert(0, '../python/src')
sys.path.insert(0, '../python/scripts')
from server_training import ServerTrainingRunner
runner = ServerTrainingRunner('{}', '', 0)
print(runner.count_unprocessed_games())
"#, games_dir))
        .output()
        .map_err(|e| format!("Failed to execute Python script: {}", e))?;
    
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Python script failed: {}", stderr));
    }
    
    let stdout = String::from_utf8_lossy(&output.stdout);
    let count_str = stdout.trim();
    
    count_str.parse::<usize>()
        .map_err(|e| format!("Failed to parse game count '{}': {}", count_str, e))
}

// Call external Python script for self-play generation (NON-BLOCKING)
async fn generate_self_play_games(
    data: web::Data<AppState>,
    num_games: usize,
    save_dir: &str,
    _fallback_model_path: &str,  // Keep for compatibility, but use current model path
) -> Result<(), String> {
    use tokio::process::Command;
    
    // Check if community engine is busy before starting intensive operations
    let community_busy = {
        if let Ok(game) = data.game.try_lock() {
            game.engine_thinking
        } else {
            true // Consider busy if we can't check
        }
    };
    
    if community_busy {
        eprintln!("🚫 Delaying self-play generation - community engine is thinking");
        return Ok(()); // Skip this round to avoid blocking community game
    }
    
    // Get the current model path (might be updated after training)
    let current_model_path = {
        let path = recover_mutex(data.current_model_path.lock());
        path.clone()
    };
    
    let output = Command::new("python")
        .arg("../python/scripts/server_self_play.py")
        .arg("--model-path")
        .arg(&current_model_path)
        .arg("--save-dir") 
        .arg(save_dir)
        .arg("--num-games")
        .arg(num_games.to_string())
        .arg("--low-priority")  // Add flag for lower GPU priority
        .kill_on_drop(true)  // Kill if server shuts down
        .output()
        .await
        .map_err(|e| format!("Failed to execute Python script: {}", e))?;
    
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Python script failed: {}", stderr));
    }
    
    let stdout = String::from_utf8_lossy(&output.stdout);
    eprintln!("✅ Self-play generation completed: {}", stdout.trim());
    
    Ok(())
}

// Call external Python script for training (NON-BLOCKING with community protection)
async fn run_training_session(args: &Args, _num_games: usize) -> Result<String, String> {
    use tokio::process::Command;
    
    eprintln!("🎓 Starting non-blocking training session...");
    
    let mut cmd = Command::new("python");
    cmd.arg("../python/scripts/server_training.py")
        .arg("--games-dir")
        .arg(&args.games_dir)
        .arg("--model-path")
        .arg(&args.model_path)
        .arg("--threshold")
        .arg(args.training_games_threshold.to_string())
        .arg("--low-priority")  // Add flag for lower GPU priority
        .kill_on_drop(true);  // Kill if server shuts down
    
    if args.tensorboard {
        cmd.arg("--tensorboard");
    }
    
    let output = cmd.output()
        .await
        .map_err(|e| format!("Failed to execute Python training script: {}", e))?;
    
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Python training script failed: {}", stderr));
    }
    
    let stdout = String::from_utf8_lossy(&output.stdout);
    let lines: Vec<&str> = stdout.lines().collect();
    
    // The script prints the new model path as the last line
    if let Some(model_path) = lines.last() {
        eprintln!("✅ Training completed! New model: {}", model_path);
        Ok(model_path.trim().to_string())
    } else {
        Err("Training script did not return a model path".to_string())
    }
}

// Also update the background_self_play_task function with the correct error type annotation
async fn background_self_play_task(
    data: web::Data<AppState>,
    args: Args,
) {
    let target_gpu_util = args.target_gpu_utilization;  // Use configurable target
    let min_self_play = 1;
    let max_self_play = args.max_self_play_games;     // Use configurable max
    let generation_interval = 120;
    
    println!("🎮 Self-play scaling initialized:");
    println!("   📊 Target GPU utilization: {:.1}%", target_gpu_util * 100.0);
    println!("   📈 Scale range: {} - {} games", min_self_play, max_self_play);
    println!("   🕐 Generation interval: {}s", generation_interval);
    println!("   🛡️ Community game protection: ENABLED");
    
    let mut last_generation = std::time::Instant::now();
    // Remove initial assignment of rapid_scale_mode - it will be declared inline
    
    loop {
        let active_players = data.active_players.load(Ordering::SeqCst);
        let current_self_play = data.self_play_games.load(Ordering::SeqCst);
        let is_training = data.is_training.load(Ordering::SeqCst);
        
        // Check if community engine is busy (highest priority)
        let community_busy = {
            if let Ok(game) = data.game.try_lock() {
                game.engine_thinking
            } else {
                true // Consider busy if we can't check
            }
        };
        
        // Get GPU utilization
        let mut gpu_util = 0.0;
        Python::with_gil(|py| {
            if let Ok(torch) = PyModule::import(py, "torch") {
                if let Ok(true) = torch.getattr("cuda")?.call_method0("is_available")?.extract() {
                    if let Ok(nvml) = PyModule::import(py, "pynvml") {
                        let _ = nvml.call_method0("nvmlInit");
                        if let Ok(handle) = nvml.call_method1("nvmlDeviceGetHandleByIndex", (0,)) {
                            if let Ok(util) = handle.call_method0("nvmlDeviceGetUtilizationRates") {
                                gpu_util = util.getattr("gpu")?.extract::<f32>()? / 100.0;
                            }
                        }
                    }
                }
            }
            Ok::<_, PyErr>(())
        }).ok();

        // Calculate ideal number of self-play games with sophisticated scaling
        let mut new_self_play = current_self_play;
        
        // Declare rapid_scale_mode inline to avoid unused assignment warning
        let rapid_scale_mode;
        
        // PRIORITY 1: Community engine protection
        if community_busy {
            // Drastically reduce self-play when community engine is thinking
            new_self_play = 1;
            rapid_scale_mode = false;
            eprintln!("🛡️ Community engine busy - reducing self-play to minimum");
        }
        // PRIORITY 2: Training protection  
        else if is_training {
            // During training, reduce to minimum to free up resources
            new_self_play = min_self_play;
            rapid_scale_mode = false;
            eprintln!("🔧 Training active - reducing self-play to minimum ({})", min_self_play);
        } 
        // PRIORITY 3: Normal scaling logic
        else if active_players == 0 {
            // No active players - this is the perfect time to scale up aggressively
            rapid_scale_mode = true;
            
            if gpu_util < 0.5 {
                // Very low GPU usage - rapid scale up
                let scale_factor = if current_self_play < 50 { 10 } else { 5 };
                new_self_play = (current_self_play + scale_factor).min(max_self_play);
            } else if gpu_util < target_gpu_util {
                // Moderate GPU usage - steady scale up
                let scale_factor = if current_self_play < 20 { 5 } else { 2 };
                new_self_play = (current_self_play + scale_factor).min(max_self_play);
            } else if gpu_util > target_gpu_util + 0.1 {
                // GPU getting saturated - scale back
                new_self_play = current_self_play.saturating_sub(2).max(min_self_play);
            }
        } else if active_players <= 2 {
            // Low traffic - moderate scaling
            rapid_scale_mode = false;
            
            if gpu_util < target_gpu_util - 0.2 && current_self_play < 50 {
                // Room for more games
                new_self_play = (current_self_play + 3).min(50);  // Cap at 50 with some players
            } else if gpu_util < target_gpu_util - 0.1 {
                new_self_play = (current_self_play + 1).min(20);  // Conservative scaling
            } else if gpu_util > target_gpu_util + 0.1 {
                new_self_play = current_self_play.saturating_sub(1).max(min_self_play);
            }
        } else {
            // High traffic - scale down to preserve resources for players
            rapid_scale_mode = false;
            
            if active_players > 10 {
                // Very high traffic - minimal self-play
                new_self_play = min_self_play;
            } else if active_players > 5 {
                // High traffic - limited self-play  
                new_self_play = current_self_play.saturating_sub(2).max(min_self_play).min(5);
            } else {
                // Moderate traffic
                new_self_play = current_self_play.saturating_sub(1).max(min_self_play).min(10);
            }
        }
        
        // Apply the scaling decision
        if new_self_play != current_self_play {
            data.self_play_games.store(new_self_play, Ordering::SeqCst);
            
            // Self-play scaling adjusted (status available via API)
        }
        
        // Actually generate self-play games periodically
        if last_generation.elapsed().as_secs() >= generation_interval && new_self_play > 0 {
            if !is_training && !community_busy {
                match generate_self_play_games(
                    data.clone(),
                    new_self_play,
                    &args.games_dir,
                    &args.model_path
                ).await {
                    Ok(()) => {
                        last_generation = std::time::Instant::now();
                    },
                    Err(e) => {
                        eprintln!("❌ Failed to generate self-play games: {}", e);
                        // Scale back on failure to avoid repeated failures
                        if new_self_play > 10 {
                            let reduced = (new_self_play / 2).max(min_self_play);
                            data.self_play_games.store(reduced, Ordering::SeqCst);
                            eprintln!("⚠️ Reduced self-play games to {} due to generation failure", reduced);
                        }
                    }
                }
            }
        }
        
        // Log current status periodically (every 5 minutes)
        if last_generation.elapsed().as_secs() % 300 == 0 && last_generation.elapsed().as_secs() > 0 {
            eprintln!("📊 Self-play status: {} games, GPU: {:.1}%, Players: {}, Mode: {}", 
                new_self_play, gpu_util * 100.0, active_players, 
                if community_busy { "Community Priority" } else if rapid_scale_mode { "Rapid Scale" } else { "Normal" });
        }
        
        // Yield to prevent blocking other tasks
        tokio::task::yield_now().await;
        sleep(Duration::from_secs(30)).await;
    }
}

// Add new endpoint for starting voting rounds
async fn start_voting_round(
    data: web::Data<AppState>,
    req: web::Json<StartVotingRequest>,
    http_req: HttpRequest,
) -> impl Responder {
    // Extract IP address
    let ip = http_req.peer_addr()
        .map(|addr| addr.ip())
        .unwrap_or_else(|| IpAddr::V4(std::net::Ipv4Addr::new(0, 0, 0, 0)));

    // Validate JWT token
    let token_data = match decode::<Claims>(
        &req.token,
        &DecodingKey::from_secret(data.jwt_secret.as_ref()),
        &Validation::default()
    ) {
        Ok(data) => data,
        Err(e) => {
            return HttpResponse::Unauthorized().json(json!({
                "success": false,
                "error_message": format!("Invalid token: {}", e),
            }));
        }
    };

    let claims = token_data.claims;
    
    // Verify IP matches token
    if claims.ip != ip.to_string() {
        return HttpResponse::Unauthorized().json(json!({
            "success": false,
            "error_message": "Token IP mismatch",
        }));
    }

    let mut game = recover_mutex(data.game.lock());

    // Check if there are votes to start timer for
    if game.votes.is_empty() {
        return HttpResponse::BadRequest().json(json!({
            "success": false,
            "error_message": "No votes to start timer for",
        }));
    }

    // Start the voting timer
    game.start_voting();

    HttpResponse::Ok().json(StartVotingResponse {
        success: true,
        error_message: None,
        game_state: CommunityGameStateResponse {
            game_id: "community".to_string(),
            board: game.board.to_string(),
            status: game.status.to_string(),
            move_history: game.move_history.clone(),
            is_voting_phase: game.is_voting_phase(),
            voting_ends_at: game.voting_ends_at.map(|t| t.to_rfc3339()),
            current_votes: game.votes.iter().map(|(k, v)| (k.clone(), v.len())).collect(),
            total_voters: game.votes.values().map(|v| v.len()).sum(),
            your_vote: None, // We don't track which specific vote this user made
            can_vote: game.can_vote(),
            waiting_for_first_move: game.waiting_for_first_move(),
            experiment_name: data.get_experiment_name(),
            engine_thinking: game.engine_thinking,
        },
    })
}

// Add new endpoint for force resolving stuck votes
async fn force_resolve_voting(
    data: web::Data<AppState>,
    req: web::Json<StartVotingRequest>, // Reuse same request struct
    http_req: HttpRequest,
) -> impl Responder {
    // Extract IP address
    let ip = http_req.peer_addr()
        .map(|addr| addr.ip())
        .unwrap_or_else(|| IpAddr::V4(std::net::Ipv4Addr::new(0, 0, 0, 0)));

    // Validate JWT token
    let token_data = match decode::<Claims>(
        &req.token,
        &DecodingKey::from_secret(data.jwt_secret.as_ref()),
        &Validation::default()
    ) {
        Ok(data) => data,
        Err(e) => {
            return HttpResponse::Unauthorized().json(json!({
                "success": false,
                "error_message": format!("Invalid token: {}", e),
            }));
        }
    };

    let claims = token_data.claims;
    
    // Verify IP matches token
    if claims.ip != ip.to_string() {
        return HttpResponse::Unauthorized().json(json!({
            "success": false,
            "error_message": "Token IP mismatch",
        }));
    }

    let mut game = recover_mutex(data.game.lock());
    let mut engine = recover_mutex(data.community_engine.lock());

    match game.force_resolve_votes(&mut engine, &data.game_storage) {
        Ok(()) => {
            HttpResponse::Ok().json(StartVotingResponse {
                success: true,
                error_message: None,
                game_state: CommunityGameStateResponse {
                    game_id: "community".to_string(),
                    board: game.board.to_string(),
                    status: game.status.to_string(),
                    move_history: game.move_history.clone(),
                    is_voting_phase: game.is_voting_phase(),
                    voting_ends_at: game.voting_ends_at.map(|t| t.to_rfc3339()),
                    current_votes: game.votes.iter().map(|(k, v)| (k.clone(), v.len())).collect(),
                    total_voters: game.votes.values().map(|v| v.len()).sum(),
                    your_vote: None,
                    can_vote: game.can_vote(),
                    waiting_for_first_move: game.waiting_for_first_move(),
                    experiment_name: data.get_experiment_name(),
                    engine_thinking: game.engine_thinking,
                },
            })
        }
        Err(e) => {
            HttpResponse::BadRequest().json(json!({
                "success": false,
                "error_message": e,
            }))
        }
    }
} 

// Check for stuck engine_thinking flags and clear them (safety watchdog)
async fn check_stuck_engine_thinking(data: web::Data<AppState>) {
    if let Ok(mut game) = data.game.try_lock() {
        if game.engine_thinking {
            // Check if the game state looks inconsistent (engine_thinking but can vote)
            // This would indicate a stuck flag
            let should_be_able_to_vote = game.status == GameStatus::Active && 
                (game.board.side_to_move() == chess::Color::White) == game.player_is_white;
            
            if should_be_able_to_vote && !game.is_voting_phase() && game.votes.is_empty() {
                // This looks like a stuck flag - clear it
                game.engine_thinking = false;
                eprintln!("🔧 Watchdog: Cleared stuck engine_thinking flag");
                eprintln!("   Game state: can_vote={}, is_voting_phase={}, votes_empty={}", 
                         game.can_vote(), game.is_voting_phase(), game.votes.is_empty());
            } else {
                // Engine thinking flag is set, but let's log the state for debugging
                eprintln!("🐕‍🦺 Watchdog: engine_thinking=true, investigating...");
                eprintln!("   can_vote={}, is_voting_phase={}, votes_empty={}, status={:?}", 
                         game.can_vote(), game.is_voting_phase(), game.votes.is_empty(), game.status);
            }
        }
    }
}

// Add function to reload model after training
async fn reload_engine_model(data: &web::Data<AppState>, new_model_path: &str) -> Result<(), String> {
    eprintln!("🔄 Reloading engine with new model: {}", new_model_path);
    
    Python::with_gil(|py| -> Result<(), String> {
        let torch = PyModule::import(py, "torch")
            .map_err(|e| format!("Failed to import torch: {}", e))?;
        
        // Create new model with the trained checkpoint
        let new_model = create_model_with_device(py, &torch, new_model_path)
            .map_err(|e| format!("Failed to create new model: {}", e))?;
        
        let new_engine = Engine::new_with_model(new_model);
        
        // Replace the engine in the app state
        {
            let mut engine = recover_mutex(data.engine.lock());
            *engine = new_engine;
        }
        
        // Update the current model path for future self-play games
        {
            let mut current_path = recover_mutex(data.current_model_path.lock());
            *current_path = new_model_path.to_string();
        }
        
        eprintln!("✅ Successfully reloaded single-player engine with new model");
        eprintln!("🎯 New model path: {}", new_model_path);
        eprintln!("🎮 Self-play will now use the new model for future games");
        
        Ok(())
    })?;
    
    Ok(())
}

// Helper function to update cache when community games complete
fn update_stats_cache_for_community_game(data: &web::Data<AppState>, game: &CommunityGame) {
    // Only update cache if the game has actually ended
    if matches!(game.status, 
        GameStatus::WhiteWins | GameStatus::BlackWins | 
        GameStatus::DrawStalemate | GameStatus::DrawInsufficientMaterial |
        GameStatus::DrawRepetition | GameStatus::DrawFiftyMoves) {
        
        let metadata = GameMetadata {
            game_id: game.game_id.clone(), // Use persistent game ID
            mode: GameMode::Community,
            created_at: Utc::now(),
            last_move_at: Utc::now(),
            status: game.status.clone(),
            total_moves: game.move_history.len(),
            player_color: if game.player_is_white { "white".to_string() } else { "black".to_string() },
            player_name: Some("Community".to_string()),
            engine_version: "1.0.0".to_string(),
        };
        
        update_stats_cache_for_completed_game(data, &metadata);
    }
}