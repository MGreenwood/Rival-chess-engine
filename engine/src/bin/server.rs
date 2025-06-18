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
use pyo3::types::PyDict;
use chrono::Utc;
use chess::{ChessMove, BoardStatus, MoveGen};
use std::str::FromStr;
use uuid::Uuid;
use std::fs;
use clap::Parser;
use std::collections::HashMap;

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

struct AppState {
    engine: Mutex<Engine>,
    games_dir: String,
    move_history: Mutex<HashMap<String, Vec<String>>>,  // Track moves per game
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
struct GameState {
    game_id: String,
    board: String,
    status: String,
    move_history: Vec<String>,
    is_player_turn: bool,
}

// Helper function to parse chess moves
fn parse_move(move_str: &str) -> Result<ChessMove, String> {
    if move_str.len() < 4 || move_str.len() > 5 {
        return Err("Invalid move format".to_string());
    }

    let chars: Vec<char> = move_str.chars().collect();
    if chars.len() < 4 {
        return Err("Invalid move format".to_string());
    }

    let from = chess::Square::from_str(&move_str[0..2])
        .map_err(|_| "Invalid source square".to_string())?;

    let to = chess::Square::from_str(&move_str[2..4])
        .map_err(|_| "Invalid target square".to_string())?;

    let promotion = if move_str.len() == 5 {
        match chars[4].to_ascii_lowercase() {
            'q' => Some(chess::Piece::Queen),
            'r' => Some(chess::Piece::Rook),
            'b' => Some(chess::Piece::Bishop),
            'n' => Some(chess::Piece::Knight),
            _ => return Err("Invalid promotion piece".to_string()),
        }
    } else {
        None
    };

    Ok(ChessMove::new(from, to, promotion))
}

// Helper function to check if a move requires promotion
fn requires_promotion(board: &chess::Board, from: chess::Square, to: chess::Square) -> bool {
    // Get the piece at the source square
    if let Some(piece) = board.piece_on(from) {
        // Check if it's a pawn
        if piece == chess::Piece::Pawn {
            // Check if the move is to the last rank
            let last_rank = if board.side_to_move() == chess::Color::White { 7 } else { 0 };
            to.get_rank() as u8 == last_rank
        } else {
            false
        }
    } else {
        false
    }
}

// Helper function to get game result string
fn get_game_result(board: &chess::Board) -> &'static str {
    if board.status() == BoardStatus::Checkmate {
        if board.side_to_move() == chess::Color::White {
            "black_wins"
        } else {
            "white_wins"
        }
    } else if board.status() == BoardStatus::Stalemate {
        "draw_stalemate"
    } else if MoveGen::new_legal(board).count() == 0 && !board.checkers().popcnt() > 0 {
        "draw_insufficient"
    } else if board.status() == BoardStatus::Stalemate {
        "draw_repetition"
    } else if board.status() == BoardStatus::Stalemate {
        "draw_fifty_moves"
    } else {
        "in_progress"
    }
}

async fn make_move(
    data: web::Data<AppState>,
    game_id: web::Path<String>,
    req: web::Json<MoveRequest>,
) -> impl Responder {
    let mut engine = data.engine.lock().unwrap();
    let board = engine.board();
    let game_id = game_id.into_inner();
    
    println!("Received move request: '{}'", req.move_str);

    // Check if it's the player's turn
    if !engine.is_player_turn() {
        return HttpResponse::Ok().json(MoveResponse {
            success: false,
            board: board.to_string(),
            status: "invalid_move".to_string(),
            engine_move: None,
            is_player_turn: false,
            error_message: Some("Not your turn".to_string()),
            move_history: Vec::new(),
        });
    }
    
    // First parse the move without checking legality
    let chess_move = match parse_move(&req.move_str) {
        Ok(mv) => mv,
        Err(e) => {
            return HttpResponse::Ok().json(MoveResponse {
                success: false,
                board: board.to_string(),
                status: "invalid_move".to_string(),
                engine_move: None,
                is_player_turn: true,
                error_message: Some(e),
                move_history: Vec::new(),
            });
        }
    };

    // Check if promotion is required but not provided
    if requires_promotion(&board, chess_move.get_source(), chess_move.get_dest()) && chess_move.get_promotion().is_none() {
        return HttpResponse::Ok().json(MoveResponse {
            success: false,
            board: board.to_string(),
            status: "invalid_move".to_string(),
            engine_move: None,
            is_player_turn: true,
            error_message: Some("Promotion piece required".to_string()),
            move_history: Vec::new(),
        });
    }

    // Check if the move is legal
    let legal_moves = MoveGen::new_legal(&board);
    if !legal_moves.any(|m| m == chess_move) {
        // Get piece at source square
        let piece = board.piece_on(chess_move.get_source());
        let error_msg = if let Some(piece) = piece {
            match piece {
                chess::Piece::Pawn => {
                    // Check if trying to move forward without capture
                    let src_file = chess_move.get_source().get_file();
                    let dst_file = chess_move.get_dest().get_file();
                    if src_file == dst_file && board.piece_on(chess_move.get_dest()).is_some() {
                        "Pawn cannot move forward when blocked by another piece".to_string()
                    } else if src_file != dst_file && board.piece_on(chess_move.get_dest()).is_none() {
                        "Pawn can only move diagonally when capturing".to_string()
                    } else {
                        format!("Illegal pawn move: {}", req.move_str)
                    }
                },
                _ => format!("Illegal move: {}", req.move_str)
            }
        } else {
            format!("No piece at square {}", &req.move_str[0..2])
        };

        println!("Illegal move: {} on board {}", req.move_str, board.to_string());
        return HttpResponse::Ok().json(MoveResponse {
            success: false,
            board: board.to_string(),
            status: "invalid_move".to_string(),
            engine_move: None,
            is_player_turn: true,
            error_message: Some(error_msg),
            move_history: Vec::new(),
        });
    }

    // Make the move
    if let Err(e) = engine.make_move(&chess_move) {
        return HttpResponse::Ok().json(MoveResponse {
            success: false,
            board: board.to_string(),
            status: "invalid_move".to_string(),
            engine_move: None,
            is_player_turn: true,
            error_message: Some(format!("Failed to make move: {}", e)),
            move_history: Vec::new(),
        });
    }

    // Update move history
    let mut move_history = data.move_history.lock().unwrap();
    let game_moves = move_history.entry(game_id.clone()).or_insert_with(Vec::new);
    game_moves.push(req.move_str.clone());

    // Get updated board state
    let updated_board = engine.board();
    let status = if updated_board.status() == BoardStatus::Ongoing {
        "active".to_string()
    } else {
        get_game_result(&updated_board).to_string()
    };

    HttpResponse::Ok().json(MoveResponse {
        success: true,
        board: updated_board.to_string(),
        status,
        engine_move: None,
        is_player_turn: false,
        error_message: None,
        move_history: game_moves.clone(),
    })
}

async fn engine_move(
    data: web::Data<AppState>,
    game_id: web::Path<String>,
) -> impl Responder {
    let mut engine = data.engine.lock().unwrap();
    let board = engine.board();
    let game_id = game_id.into_inner();
    
    if engine.is_game_over() {
        let result = get_game_result(&board);
        // Get move history before saving
        let moves = data.move_history.lock().unwrap().remove(&game_id).unwrap_or_default();
        save_game(&data, &game_id, moves.clone(), result);
        return HttpResponse::Ok().json(MoveResponse {
            success: true,
            board: board.to_string(),
            status: result.to_string(),
            engine_move: None,
            is_player_turn: true,
            error_message: None,
            move_history: moves,
        });
    }
    
    // Get engine's move
    let engine_move = if let Some(mv) = engine.get_best_move(board) {
        let move_str = format!("{}{}", mv.get_source(), mv.get_dest());
        let promotion = mv.get_promotion().map(|p| match p {
            chess::Piece::Queen => 'q',
            chess::Piece::Rook => 'r',
            chess::Piece::Bishop => 'b',
            chess::Piece::Knight => 'n',
            _ => unreachable!(),
        });
        let move_str = if let Some(p) = promotion {
            format!("{}{}", move_str, p)
        } else {
            move_str
        };
        engine.make_move(mv);
        
        // Add engine's move to history
        if let Some(history) = data.move_history.lock().unwrap().get_mut(&game_id) {
            history.push(move_str.clone());
        }
        
        Some(move_str)
    } else {
        None
    };

    // Check if game is over after engine's move
    let result = get_game_result(&engine.board());
    if result != "in_progress" {
        // Get move history before saving
        let moves = data.move_history.lock().unwrap().remove(&game_id).unwrap_or_default();
        save_game(&data, &game_id, moves.clone(), result);
        return HttpResponse::Ok().json(MoveResponse {
            success: true,
            board: engine.board().to_string(),
            status: result.to_string(),
            engine_move,
            is_player_turn: true,
            error_message: None,
            move_history: moves,
        });
    }

    // Get current move history
    let current_moves = data.move_history.lock().unwrap().get(&game_id).cloned().unwrap_or_default();

    HttpResponse::Ok().json(MoveResponse {
        success: true,
        board: engine.board().to_string(),
        status: if result != "in_progress" { result.to_string() } else { "active".to_string() },
        engine_move,
        is_player_turn: true,
        error_message: None,
        move_history: current_moves,
    })
}

async fn create_game(
    data: web::Data<AppState>,
    settings: Option<web::Json<GameSettings>>,
) -> impl Responder {
    let mut engine = data.engine.lock().unwrap();
    // Reset the engine to starting position
    engine.reset();
    
    // Apply settings if provided
    if let Some(settings) = settings {
        if let Some(temp) = settings.temperature {
            engine.set_temperature(temp);
        }
        if let Some(strength) = settings.strength {
            engine.set_strength(strength);
        }
    }
    
    let game_id = Uuid::new_v4().to_string();
    let board = engine.board().to_string();
    
    // Initialize empty move history for this game
    data.move_history.lock().unwrap().insert(game_id.clone(), Vec::new());
    
    let state = GameState {
        game_id: game_id.clone(),
        board,
        status: "active".to_string(),
        move_history: Vec::new(),
        is_player_turn: true,
    };

    // Drop the engine lock before returning
    drop(engine);

    HttpResponse::Ok().json(state)
}

async fn reset_game(
    data: web::Data<AppState>,
) -> impl Responder {
    let mut engine = data.engine.lock().unwrap();
    // Reset the engine to starting position
    engine.reset();
    
    HttpResponse::Ok().json(serde_json::json!({
        "success": true,
        "message": "Game reset to starting position"
    }))
}

async fn get_game(
    data: web::Data<AppState>,
    game_id: web::Path<String>,
) -> impl Responder {
    let engine = data.engine.lock().unwrap();
    let board = engine.board();
    
    let state = GameState {
        game_id: game_id.into_inner(),
        board: board.to_string(),
        status: if engine.is_game_over() { "game_over".to_string() } else { "active".to_string() },
        move_history: Vec::new(), // TODO: Track move history
        is_player_turn: true,
    };
    
    HttpResponse::Ok().json(state)
}

// New endpoint to get saved games
async fn get_saved_games(data: web::Data<AppState>) -> impl Responder {
    let games_dir = Path::new(&data.games_dir);
    let mut saved_games = Vec::new();

    if games_dir.exists() {
        if let Ok(entries) = fs::read_dir(games_dir) {
            for entry in entries {
                if let Ok(entry) = entry {
                    if let Some(ext) = entry.path().extension() {
                        if ext == "json" {
                            if let Ok(content) = fs::read_to_string(entry.path()) {
                                if let Ok(game) = serde_json::from_str(&content) {
                                    saved_games.push(game);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Sort games by timestamp, newest first
    saved_games.sort_by(|a: &SavedGame, b: &SavedGame| {
        b.timestamp.cmp(&a.timestamp)
    });

    HttpResponse::Ok().json(saved_games)
}

struct MyWebSocket {
    _game_id: Option<String>,
    engine: web::Data<AppState>,
}

impl MyWebSocket {
    fn new(game_id: String, engine: web::Data<AppState>) -> Self {
        Self {
            _game_id: Some(game_id),
            engine,
        }
    }

    fn send_game_state(&self, ctx: &mut ws::WebsocketContext<Self>) {
        if let Ok(engine) = self.engine.engine.lock() {
            let board = engine.board();
            let status = match board.status() {
                BoardStatus::Checkmate => "checkmate",
                BoardStatus::Stalemate => "stalemate",
                _ => "active",
            };
            let message = serde_json::json!({
                "type": "status",
                "payload": {
                    "board": board.to_string(),
                    "status": status,
                }
            });
            ctx.text(message.to_string());
        }
    }
}

impl Actor for MyWebSocket {
    type Context = ws::WebsocketContext<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        // Reset the engine to starting position for new games
        if let Ok(mut engine) = self.engine.engine.lock() {
            engine.reset();
        }
        
        // Send initial game state
        self.send_game_state(ctx);
    }

    fn stopped(&mut self, _ctx: &mut Self::Context) {
        // Clean up the game when the connection is closed
        if let Some(ref game_id) = self._game_id {
            // Remove the game's move history
            if let Ok(mut move_history) = self.engine.move_history.lock() {
                move_history.remove(game_id);
            }
            println!("Cleaned up abandoned game: {}", game_id);
        }
    }
}

impl StreamHandler<Result<ws::Message, ws::ProtocolError>> for MyWebSocket {
    fn handle(&mut self, msg: Result<ws::Message, ws::ProtocolError>, ctx: &mut Self::Context) {
        match msg {
            Ok(ws::Message::Ping(msg)) => {
                ctx.pong(&msg);
                self.send_game_state(ctx);
            },
            Ok(ws::Message::Text(text)) => {
                if let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) {
                    if let Some(command) = value.get("command") {
                        match command.as_str() {
                            Some("refresh") => self.send_game_state(ctx),
                            _ => (),
                        }
                    }
                }
            },
            Ok(ws::Message::Close(reason)) => {
                // Clean up the game before closing
                if let Some(ref game_id) = self._game_id {
                    if let Ok(mut move_history) = self.engine.move_history.lock() {
                        move_history.remove(game_id);
                    }
                    println!("Cleaned up game on close: {}", game_id);
                }
                ctx.close(reason);
            },
            _ => (),
        }
    }
}

async fn ws_route(
    req: HttpRequest,
    stream: web::Payload,
    path: web::Path<String>,
    data: web::Data<AppState>,
) -> Result<HttpResponse, actix_web::Error> {
    ws::start(MyWebSocket::new(path.into_inner(), data), &req, stream)
}

// Function to save a completed game
fn save_game(state: &AppState, game_id: &str, moves: Vec<String>, result: &str) {
    // Only save games with decisive results (white_wins or black_wins)
    let training_result = match result {
        "white_wins" => "win",
        "black_wins" => "loss",
        _ => return, // Don't save draws or incomplete games
    };

    let games_dir = Path::new(&state.games_dir);
    if !games_dir.exists() {
        if let Err(e) = fs::create_dir_all(games_dir) {
            eprintln!("Failed to create games directory: {}", e);
            return;
        }
    }

    // Only save games with a minimum number of moves (to avoid very short games)
    if moves.len() < 10 {
        println!("Game {} not saved: too few moves ({})", game_id, moves.len());
        return;
    }

    let game = SavedGame {
        game_id: game_id.to_string(),
        moves: moves.clone(),
        result: training_result.to_string(),
        timestamp: Utc::now().to_rfc3339(),
        player_color: "white".to_string(),
    };

    let filename = format!("game_{}.json", game_id);
    let file_path = games_dir.join(filename);

    if let Ok(json) = serde_json::to_string_pretty(&game) {
        if let Err(e) = fs::write(&file_path, json) {
            eprintln!("Failed to save game {}: {}", game_id, e);
        } else {
            println!("Saved decisive game {} to {} with {} moves and result: {}", 
                game_id, file_path.display(), moves.len(), training_result);
        }
    } else {
        eprintln!("Failed to serialize game {}", game_id);
    }
}

// Helper function to create model with device selection
fn create_model_with_device(py: Python, torch: &PyModule, checkpoint_path: &str) -> PyResult<rival_ai::ModelBridge> {
    // Check if CUDA is available and select device
    let device_str = if torch.getattr("cuda")?.getattr("is_available")?.call0()?.extract::<bool>()? {
        "cuda"
    } else {
        "cpu"
    };
    
    // Import model wrapper
    let model_wrapper = py.import("rival_ai.models.gnn")?;
    let model_class = model_wrapper.getattr("ChessGNN")?;
    let model = model_class.call0()?;
    
    // Move model to device
    let device = torch.getattr("device")?.call1((device_str,))?;
    let model = model.call_method1("to", (device,))?;
    
    // Load checkpoint
    let checkpoint = torch.getattr("load")?.call1((checkpoint_path,))?;
    let model_state_dict = checkpoint.get_item("model_state_dict")?;
    model.call_method1("load_state_dict", (model_state_dict,))?;
    
    // Set to evaluation mode
    model.call_method0("eval")?;
    
    Ok(rival_ai::ModelBridge::new(model.into(), Some(device_str.to_string())))
}

// Helper function to create fallback engine
fn create_fallback_engine(py: Python) -> Engine {
    // Create a simple fallback model that returns random moves
    let code = r#"
import random
import chess

class FallbackModel:
    def __init__(self):
        self.name = "Fallback Random Model"
    
    def predict_with_board(self, board_fen):
        board = chess.Board(board_fen)
        legal_moves = list(board.legal_moves)
        if legal_moves:
            move = random.choice(legal_moves)
            # Create a dummy policy vector (all zeros except for the chosen move)
            policy = [0.0] * 5312  # Total number of possible moves
            # Set a high probability for the chosen move (simplified)
            policy[0] = 1.0
            return (policy, 0.0)  # Return policy and value
        return ([0.0] * 5312, 0.0)
    
    def eval(self):
        pass

FallbackModel()
"#;
    
    let locals = PyDict::new(py);
    py.run(code, None, Some(locals)).unwrap();
    let fallback_model = locals.get_item("FallbackModel").unwrap().call0().unwrap();
    
    // Create ModelBridge from the fallback model
    let model_bridge = rival_ai::ModelBridge::new(fallback_model.into(), Some("cpu".to_string()));
    
    // Create Engine with the fallback model
    Engine::new_with_model(model_bridge)
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let args = Args::parse();
    
    // Create games directory if it doesn't exist
    if let Err(e) = fs::create_dir_all(&args.games_dir) {
        eprintln!("Failed to create games directory: {}", e);
        return Ok(());
    }
    
    // Initialize Python
    pyo3::prepare_freethreaded_python();
    
    let engine = Python::with_gil(|py| {
        // Add Python package directory to sys.path
        let python_path = Path::new("../python/src").canonicalize().unwrap();
        py.import("sys").unwrap()
            .getattr("path").unwrap()
            .call_method1("append", (python_path.to_str().unwrap(),)).unwrap();
        
        // Import torch
        let torch = py.import("torch").expect("Failed to import torch");
        
        // Try to create model with CUDA first
        match create_model_with_device(py, torch, &args.model_path) {
            Ok(model) => {
                println!("Successfully created model with CUDA support");
                Engine::new_with_model(model)
            }
            Err(e) => {
                eprintln!("Failed to create model with CUDA: {}", e);
                eprintln!("Falling back to CPU model...");
                create_fallback_engine(py)
            }
        }
    });
    
    let app_state = web::Data::new(AppState {
        engine: Mutex::new(engine),
        games_dir: args.games_dir,
        move_history: Mutex::new(HashMap::new()),
    });
    
    println!("Starting server on {}:{}", args.host, args.port);
    
    HttpServer::new(move || {
        App::new()
            .wrap(Cors::permissive())
            .app_data(app_state.clone())
            .wrap(Logger::default())
            .service(
                web::scope("/api")
                    .route("/game", web::post().to(create_game))
                    .route("/game/{game_id}/move", web::post().to(make_move))
                    .route("/game/{game_id}/engine_move", web::post().to(engine_move))
                    .route("/game/{game_id}", web::get().to(get_game))
                    .route("/games", web::get().to(get_saved_games))
                    .route("/reset", web::post().to(reset_game))
            )
            .route("/ws/{game_id}", web::get().to(ws_route))
    })
    .bind((args.host, args.port))?
    .run()
    .await
} 