use std::path::Path;
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
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use rand::{thread_rng, Rng};
use rival_ai::game_storage::{GameStorage, GameState as StorageGameState, GameMode, GameStatus, GameMetadata};
use rival_ai::ModelBridge;
use serde_json::json;
use std::fs;
use pyo3::types::PyDict;
use uuid;

#[derive(Parser, Debug)]
#[command(name = "rival-ai-server")]
#[command(about = "RivalAI Chess Engine Server")]
struct Args {
    /// Path to the model checkpoint file
    #[arg(short, long, default_value = "../python/experiments/rival_ai_v1_Alice/run_20250617_221622/checkpoints/best_model.pt")]
    model_path: String,
    
    /// Server port
    #[arg(short, long, default_value = "3000")]
    port: u16,
    
    /// Server host
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Directory to save games for training
    #[arg(long, default_value = "../python/training_games")]
    games_dir: String,
}

#[derive(Deserialize)]
struct MoveRequest {
    move_str: String,
    board: String,
    player_color: String,
}

#[derive(Serialize)]
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
}

#[derive(Serialize, Deserialize)]
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

#[derive(Clone)]
pub struct CommunityGame {
    board: ChessBoard,
    move_history: Vec<String>,
    votes: HashMap<String, HashSet<String>>,  // move -> set of voter IDs
    voting_ends_at: Option<DateTime<Utc>>,
    status: GameStatus,
}

impl CommunityGame {
    pub fn new() -> Self {
        Self {
            board: ChessBoard::default(),
            move_history: Vec::new(),
            votes: HashMap::new(),
            voting_ends_at: None,
            status: GameStatus::Waiting,
        }
    }

    pub fn start_voting(&mut self) {
        self.voting_ends_at = Some(Utc::now() + chrono::Duration::seconds(10));
        self.votes.clear();
        if self.status == GameStatus::Waiting {
            self.status = GameStatus::Active;
        }
    }

    pub fn add_vote(&mut self, move_str: &str, voter_id: &str) -> Result<(), String> {
        // Start voting phase if this is the first vote and game hasn't started
        if self.status == GameStatus::Waiting {
            self.start_voting();
        } else if !self.is_voting_phase() {
            return Err("No active voting phase".to_string());
        }
        
        // Remove previous vote if exists
        for votes in self.votes.values_mut() {
            votes.remove(voter_id);
        }
        
        // Add new vote
        self.votes.entry(move_str.to_string())
            .or_insert_with(HashSet::new)
            .insert(voter_id.to_string());
        
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

    fn make_move(&mut self, mv: ChessMove) {
        let mut new_board = self.board.clone();
        new_board = new_board.make_move_new(mv);
        self.board = new_board;
    }

    pub fn process_votes(&mut self, engine: &mut Engine) -> Result<(), String> {
        if !self.is_voting_phase() {
            return Ok(());
        }

        if let Some(winning_move) = self.get_winning_move() {
            // Make the player's move
            let chess_move = parse_move(&winning_move)?;
            self.make_move(chess_move);
            self.move_history.push(winning_move);

            // Make engine's move
            if let Some(engine_move) = engine.get_best_move(self.board.clone()) {
                self.make_move(engine_move);
                self.move_history.push(engine_move.to_string());
            }

            // Check game status
            if self.board.status() != BoardStatus::Ongoing {
                self.status = match self.board.status() {
                    BoardStatus::Checkmate => {
                        if self.board.side_to_move() == chess::Color::White {
                            GameStatus::BlackWins
                        } else {
                            GameStatus::WhiteWins
                        }
                    },
                    BoardStatus::Stalemate => GameStatus::DrawStalemate,
                    _ => GameStatus::Active,
                };
            }

            // Start next voting phase
            self.start_voting();
        }

        Ok(())
    }

    pub fn is_voting_phase(&self) -> bool {
        if let Some(ends_at) = self.voting_ends_at {
            Utc::now() <= ends_at
        } else {
            false
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
}

#[derive(Deserialize)]
struct GameSettings {
    temperature: Option<f32>,
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

fn parse_move(move_str: &str) -> Result<ChessMove, String> {
    if move_str.len() < 4 {
        return Err("Move string too short".to_string());
    }

    let from = chess::Square::from_str(&move_str[0..2])
        .map_err(|_| format!("Invalid 'from' square: {}", &move_str[0..2]))?;
    let to = chess::Square::from_str(&move_str[2..4])
        .map_err(|_| format!("Invalid 'to' square: {}", &move_str[2..4]))?;

    let promotion = if move_str.len() > 4 {
        Some(match move_str.chars().nth(4).unwrap() {
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

async fn make_move(
    data: web::Data<AppState>,
    path: web::Path<String>,
    req: web::Json<MoveRequest>,
) -> impl Responder {
    let game_id = path.into_inner();
    let mut engine = data.engine.lock().unwrap();

    // Load the game
    let mut game_state = match data.game_storage.load_game(&game_id, &GameMode::SinglePlayer) {
        Ok(state) => state,
        Err(e) => return HttpResponse::NotFound().json(json!({
            "error": format!("Game not found: {}", e)
        })),
    };

    // Parse the current board position
    let mut board = match ChessBoard::from_str(&req.0.board) {
        Ok(b) => b,
        Err(e) => return HttpResponse::BadRequest().json(json!({
            "error": format!("Invalid board position: {}", e)
        })),
    };

    // Parse and validate the player's move
    let player_move = match parse_move(&req.0.move_str) {
        Ok(m) => m,
        Err(e) => return HttpResponse::BadRequest().json(json!({
            "error": format!("Invalid move: {}", e)
        })),
    };

    // Make the player's move
    board = board.make_move_new(player_move);
    game_state.board = board.to_string();
    game_state.move_history.push(req.0.move_str.clone());
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
        
        // Save the game state
        if let Err(e) = data.game_storage.save_game(&game_state) {
            return HttpResponse::InternalServerError().json(json!({
                "error": format!("Failed to save game: {}", e)
            }));
        }

        return HttpResponse::Ok().json(MoveResponse {
            success: true,
            board: game_state.board,
            status: game_state.metadata.status.to_string(),
            engine_move: None,
            is_player_turn: true,
            error_message: None,
            move_history: game_state.move_history,
        });
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
        }

        // Save the updated game state
        if let Err(e) = data.game_storage.save_game(&game_state) {
            return HttpResponse::InternalServerError().json(json!({
                "error": format!("Failed to save game: {}", e)
            }));
        }

        HttpResponse::Ok().json(MoveResponse {
            success: true,
            board: game_state.board,
            status: game_state.metadata.status.to_string(),
            engine_move: Some(engine_move.to_string()),
            is_player_turn: true,
            error_message: None,
            move_history: game_state.move_history,
        })
    } else {
        return HttpResponse::InternalServerError().json(json!({
            "error": "Engine failed to make a move"
        }));
    }
}

async fn create_new_game(
    data: web::Data<AppState>,
    req: web::Json<NewGameRequest>,
) -> impl Responder {
    let mut engine = data.engine.lock().unwrap();
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

    // Store the game
    if let Err(e) = data.game_storage.save_game(&game_state) {
        return HttpResponse::InternalServerError().json(json!({
            "error": format!("Failed to save game: {}", e)
        }));
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

async fn get_saved_games(data: web::Data<AppState>) -> impl Responder {
    match data.game_storage.list_games(Some(GameMode::SinglePlayer)) {
        Ok(games) => {
            let saved_games: Vec<SavedGame> = games.into_iter()
                .map(|game| SavedGame {
                    game_id: game.game_id,
                    moves: vec![], // TODO: Load moves from game state
                    result: game.status.to_string(),
                    timestamp: game.last_move_at.to_rfc3339(),
                    player_color: game.player_color,
                })
                .collect();
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
                if let Ok(_game_state) = self.engine.game_storage.load_game(&self.game_id, &GameMode::SinglePlayer) {
                    let _engine = self.engine.engine.lock().unwrap();
                    self.send_game_state(ctx);
                }
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

// Helper function to create model with device selection
fn create_model_with_device(py: Python, torch: &PyModule, model_path: &str) -> PyResult<ModelBridge> {
    let device = if torch.getattr("cuda")?.call_method0("is_available")?.extract::<bool>()? {
        Some("cuda".to_string())
    } else {
        Some("cpu".to_string())
    };
    
    let code = format!(r#"
import torch
import chess
from rival_ai.models import ChessGNN
from rival_ai.pag import PositionalAdjacencyGraph
import numpy as np

class ModelWrapper:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ChessGNN(hidden_dim=256, num_layers=4, num_heads=4, dropout=0.1)
        
        try:
            checkpoint = torch.load('{}', map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"Loaded model from {model_path}")
        except Exception as e:
            print(f"Failed to load model from {model_path}: {{e}}")
            print("Using randomly initialized model")
            
        self.model.to(self.device)
        self.model.eval()
"#, model_path.replace("\\", "\\\\"));

    let locals = PyDict::new(py);
    locals.set_item("torch", torch)?;
    py.run(&code, Some(locals), None)?;
    let model = py.eval("ModelWrapper()", Some(locals), None)?;
    
    if device == Some("cuda".to_string()) {
        model.call_method1("to", ("cuda",))?;
    }
    
    Ok(ModelBridge::new(model.into_py(py), device))
}

// Helper function to create fallback engine
fn create_fallback_engine(py: Python) -> Engine {
    let code = r#"
class FallbackModel:
    def __init__(self):
        self.device = 'cpu'
    
    def predict_with_board(self, board_fen):
        return ([0.0] * 5312, 0.0)
    
    def eval(self):
        pass
    
    def to(self, device):
        self.device = device
        return self
"#;
    let locals = PyDict::new(py);
    py.run(code, Some(locals), None).unwrap();
    let model = py.eval("FallbackModel()", Some(locals), None).unwrap();
    Engine::new_with_model(ModelBridge::new(model.into_py(py), Some("cpu".to_string())))
}

// Endpoints
async fn start_community_game(data: web::Data<AppState>) -> impl Responder {
    let mut game = data.game.lock().unwrap();
    *game = CommunityGame::new();
    
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
    };

    HttpResponse::Ok().json(response)
}

async fn get_community_game_state(
    data: web::Data<AppState>, 
    voter_id: Option<web::Query<String>>
) -> impl Responder {
    let voter_id = voter_id.as_ref().map(|id| id.as_str().to_string());
    let game = data.game.lock().unwrap();
    
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
    })
}

async fn vote_move(
    data: web::Data<AppState>,
    req: web::Json<VoteRequest>,
) -> impl Responder {
    let mut game = data.game.lock().unwrap();
    let mut engine = data.engine.lock().unwrap();

    match game.add_vote(&req.move_str, &req.voter_id) {
        Ok(()) => {
            if let Err(e) = game.process_votes(&mut engine) {
                return HttpResponse::BadRequest().json(json!({
                    "success": false,
                    "error_message": e,
                }));
            }

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
                },
            };

            HttpResponse::Ok().json(response)
        }
        Err(e) => HttpResponse::BadRequest().json(json!({
            "success": false,
            "error_message": e,
        })),
    }
}

async fn get_game_state(data: web::Data<AppState>) -> impl Responder {
    let game = data.game.lock().unwrap();
    let _engine = data.engine.lock().unwrap();
    
    web::Json(json!({
        "board": game.board.to_string(),
        "move_history": game.move_history,
        "is_voting_phase": game.is_voting_phase(),
        "status": game.status,
    }))
}

async fn submit_vote(
    data: web::Data<AppState>,
    req: web::Json<VoteRequest>,
) -> impl Responder {
    let mut game = data.game.lock().unwrap();
    let mut engine = data.engine.lock().unwrap();

    match game.add_vote(&req.move_str, &req.voter_id) {
        Ok(_) => {
            if let Err(e) = game.process_votes(&mut engine) {
                return HttpResponse::BadRequest().json(json!({
                    "error": format!("Failed to process votes: {}", e)
                }));
            }
            
            HttpResponse::Ok().json(json!({
                "board": game.board.to_string(),
                "move_history": game.move_history.clone(),
                "is_voting_phase": game.is_voting_phase(),
                "status": game.status,
            }))
        }
        Err(e) => HttpResponse::BadRequest().json(json!({
            "error": e
        }))
    }
}

async fn list_games(data: web::Data<AppState>) -> impl Responder {
    match data.game_storage.list_games(None) {
        Ok(games) => {
            let saved_games: Vec<SavedGame> = games.into_iter()
                .map(|game| SavedGame {
                    game_id: game.game_id,
                    moves: Vec::new(), // Empty vector since move_history is not in metadata
                    result: game.status.to_string(),
                    timestamp: game.last_move_at.to_rfc3339(),
                    player_color: game.player_color,
                })
                .collect();
            HttpResponse::Ok().json(saved_games)
        }
        Err(e) => HttpResponse::InternalServerError().json(json!({
            "error": format!("Failed to list games: {}", e)
        }))
    }
}

async fn get_model_stats(data: web::Data<AppState>) -> impl Responder {
    match data.game_storage.list_games(None) {
        Ok(games) => {
            let mut wins = 0;
            let mut losses = 0;
            let mut draws = 0;

            for game in games {
                match game.status {
                    GameStatus::WhiteWins => {
                        if game.player_color == "white" {
                            wins += 1;
                        } else {
                            losses += 1;
                        }
                    },
                    GameStatus::BlackWins => {
                        if game.player_color == "black" {
                            wins += 1;
                        } else {
                            losses += 1;
                        }
                    },
                    GameStatus::DrawStalemate | GameStatus::DrawInsufficientMaterial |
                    GameStatus::DrawRepetition | GameStatus::DrawFiftyMoves => {
                        draws += 1;
                    },
                    _ => {}
                }
            }

            HttpResponse::Ok().json(ModelStatsResponse {
                wins,
                losses,
                draws,
            })
        },
        Err(e) => {
            eprintln!("Failed to get model stats: {}", e);
            HttpResponse::InternalServerError().json(json!({
                "error": format!("Failed to get model stats: {}", e)
            }))
        }
    }
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

#[derive(Serialize)]
struct ModelStatsResponse {
    wins: i32,
    losses: i32,
    draws: i32,
}

#[derive(Serialize)]
struct LeaderboardEntry {
    rank: i32,
    name: String,
    wins: i32,
    losses: i32,
    draws: i32,
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

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::init_from_env(env_logger::Env::new().default_filter_or("info"));

    let args = Args::parse();
    let games_dir = Path::new(&args.games_dir);
    if !games_dir.exists() {
        fs::create_dir_all(games_dir)?;
    }

    // Initialize Python
    pyo3::prepare_freethreaded_python();

    // Initialize engine
    let engine = Python::with_gil(|py| {
        match py.import("torch") {
            Ok(torch) => {
                match create_model_with_device(py, torch, &args.model_path) {
                    Ok(model) => Engine::new_with_model(model),
                    Err(e) => {
                        eprintln!("Failed to create model: {}", e);
                        create_fallback_engine(py)
                    }
                }
            }
            Err(e) => {
                eprintln!("Failed to import torch: {}", e);
                create_fallback_engine(py)
            }
        }
    });

    let game_storage = Arc::new(GameStorage::new(games_dir)?);
    let host = args.host.clone();
    let port = args.port;

    HttpServer::new(move || {
        let cors = Cors::default()
            .allow_any_origin()
            .allow_any_method()
            .allow_any_header();

        App::new()
            .wrap(cors)
            .wrap(Logger::default())
            .app_data(web::Data::new(AppState {
                game: Arc::new(Mutex::new(CommunityGame::new())),
                game_storage: game_storage.clone(),
                engine: Arc::new(Mutex::new(engine.clone())),
            }))
            .service(web::resource("/ws").route(web::get().to(ws_route)))
            .service(web::resource("/game").route(web::get().to(get_game_state)))
            .service(web::resource("/move/new").route(web::post().to(create_new_game)))
            .service(web::resource("/move/{id}").route(web::post().to(make_move)))
            .service(web::resource("/api/community/state").route(web::get().to(get_community_game_state)))
            .service(web::resource("/api/community/vote").route(web::post().to(vote_move)))
            .service(web::resource("/api/community/start").route(web::post().to(start_community_game)))
            .service(web::resource("/games").route(web::get().to(list_games)))
            .service(web::resource("/games/{id}").route(web::get().to(get_game)))
            .service(web::resource("/stats").route(web::get().to(get_model_stats)))
            .service(web::resource("/leaderboard").route(web::get().to(get_leaderboard)))
    })
    .bind((args.host, port))?
    .run()
    .await?;

    println!("Server running at http://{}:{}", host, port);
    Ok(())
} 