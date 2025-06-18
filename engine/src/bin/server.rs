use actix_web::{web, App, HttpResponse, HttpServer, Responder, middleware, HttpRequest};
use actix_cors::Cors;
use serde::{Deserialize, Serialize};
use chess::{ChessMove, BoardStatus, MoveGen};
use std::sync::Mutex;
use uuid::Uuid;
use serde_json;
use actix_web_actors::ws;
use actix::{Actor, StreamHandler, ActorContext};
use rival_ai::Engine;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::PyErr;
use clap::Parser;

#[derive(Parser)]
#[command(name = "rival-ai-server")]
#[command(about = "RivalAI Chess Engine Server")]
struct Args {
    /// Path to the model checkpoint file
    #[arg(short, long, default_value = "experiments/rival_ai_v1_Alice/run_20250616_154501/checkpoints/best_model.pt")]
    checkpoint: String,
    
    /// Server port
    #[arg(short, long, default_value = "3000")]
    port: u16,
    
    /// Server host
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
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
}

struct AppState {
    engine: Mutex<Engine>,
}

#[derive(Deserialize)]
struct GameSettings {
    // Accept but ignore for now
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

    let from = chess::Square::make_square(
        chess::Rank::from_index((chars[1] as u8 - b'1') as usize),
        chess::File::from_index((chars[0] as u8 - b'a') as usize),
    );

    let to = chess::Square::make_square(
        chess::Rank::from_index((chars[3] as u8 - b'1') as usize),
        chess::File::from_index((chars[2] as u8 - b'a') as usize),
    );

    let promotion = if chars.len() == 5 {
        match chars[4] {
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

async fn make_move(
    data: web::Data<AppState>,
    _game_id: web::Path<String>,
    req: web::Json<MoveRequest>,
) -> impl Responder {
    let mut engine = data.engine.lock().unwrap();
    let board = engine.board();
    
    println!("Received move request: '{}'", req.move_str);
    println!("Current board: {}", board);
    
    match parse_move(&req.move_str) {
        Ok(chess_move) => {
            println!("Parsed move: {} -> {} (promotion: {:?})", 
                chess_move.get_source(), 
                chess_move.get_dest(), 
                chess_move.get_promotion()
            );
            
            // Check if move is legal
            let legal_moves: Vec<ChessMove> = MoveGen::new_legal(&board).collect();
            println!("Legal moves count: {}", legal_moves.len());
            
            // Print all legal moves for debugging
            for (i, mv) in legal_moves.iter().enumerate() {
                println!("Legal move {}: {} -> {} (promotion: {:?})", 
                    i, mv.get_source(), mv.get_dest(), mv.get_promotion());
            }
            
            // Print board in a readable format
            println!("Current board position:");
            let board_str = board.to_string();
            let lines: Vec<&str> = board_str.split(' ').next().unwrap_or("").split('/').collect();
            for (rank, line) in lines.iter().enumerate() {
                println!("Rank {}: {}", 8 - rank, line);
            }
            
            // Check specific squares
            println!("Piece on f8: {:?}", board.piece_on(chess::Square::F8));
            println!("Piece on h8: {:?}", board.piece_on(chess::Square::H8));
            println!("Piece on g7: {:?}", board.piece_on(chess::Square::G7));
            
            if !board.legal(chess_move) {
                println!("Move {} is not legal!", chess_move);
                return HttpResponse::BadRequest().json(MoveResponse {
                    success: false,
                    board: board.to_string(),
                    status: "invalid_move".to_string(),
                    engine_move: None,
                    is_player_turn: true,
                    error_message: Some("Illegal move".to_string()),
                });
            }

            println!("Move is legal, applying...");
            engine.make_move(chess_move);
            
            // Respond immediately, do not make engine move yet
            HttpResponse::Ok().json(MoveResponse {
                success: true,
                board: engine.board().to_string(),
                status: if engine.is_game_over() { "game_over".to_string() } else { "active".to_string() },
                engine_move: None,
                is_player_turn: false, // Now it's the engine's turn
                error_message: None,
            })
        }
        Err(e) => {
            println!("Failed to parse move '{}': {}", req.move_str, e);
            HttpResponse::BadRequest().json(MoveResponse {
                success: false,
                board: board.to_string(),
                status: "invalid_move".to_string(),
                engine_move: None,
                is_player_turn: true,
                error_message: Some("Invalid move format".to_string()),
            })
        }
    }
}

// New endpoint: engine makes its move
async fn engine_move(
    data: web::Data<AppState>,
    _game_id: web::Path<String>,
) -> impl Responder {
    let mut engine = data.engine.lock().unwrap();
    let board = engine.board();
    if engine.is_game_over() {
        return HttpResponse::Ok().json(MoveResponse {
            success: true,
            board: board.to_string(),
            status: "game_over".to_string(),
            engine_move: None,
            is_player_turn: true,
            error_message: None,
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
        Some(move_str)
    } else {
        None
    };
    HttpResponse::Ok().json(MoveResponse {
        success: true,
        board: engine.board().to_string(),
        status: if engine.is_game_over() { "game_over".to_string() } else { "active".to_string() },
        engine_move,
        is_player_turn: true, // Now it's the player's turn
        error_message: None,
    })
}

async fn create_game(
    _settings: web::Json<serde_json::Value>,
) -> impl Responder {
    let game_id = Uuid::new_v4().to_string();
    let board = chess::Board::default().to_string();
    let state = GameState {
        game_id,
        board,
        status: "active".to_string(),
        move_history: Vec::new(),
        is_player_turn: true,
    };
    HttpResponse::Ok().json(state)
}

async fn reset_game(
    data: web::Data<AppState>,
) -> impl Responder {
    let mut engine = data.engine.lock().unwrap();
    // Reset the engine to starting position
    *engine = Engine::new();
    
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

struct MyWebSocket {
    _game_id: String,
    engine: web::Data<AppState>,
}

impl MyWebSocket {
    fn new(game_id: String, engine: web::Data<AppState>) -> Self {
        Self { _game_id: game_id, engine }
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
                ctx.close(reason);
                ctx.stop();
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

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    dotenv::dotenv().ok();
    
    // Parse command line arguments
    let args = Args::parse();
    
    // Initialize Python interpreter
    pyo3::prepare_freethreaded_python();
    
    println!("Using checkpoint: {}", args.checkpoint);
    
    // Set up Python environment
    let python_src_path = "../python/src";
    let venv_path = "venvEngine/Lib/site-packages";
    std::env::set_var("PYTHONPATH", format!("{};{}", python_src_path, venv_path));
    
    // Initialize Python with proper model loading
    let engine = Python::with_gil(|py| {
        // Add paths to Python sys.path
        let sys = py.import("sys").unwrap();
        let sys_path = sys.getattr("path").unwrap();
        sys_path.call_method1("insert", (0, python_src_path)).unwrap();
        sys_path.call_method1("insert", (0, venv_path)).unwrap();
        
        println!("Python path:");
        for path_result in sys_path.iter().unwrap() {
            match path_result {
                Ok(path) => {
                    match path.str() {
                        Ok(path_str) => println!("  {}", path_str),
                        Err(_) => println!("  <invalid path>"),
                    }
                },
                Err(_) => println!("  <error reading path>"),
            }
        }
        
        // Try to import torch with error handling
        println!("Attempting to import PyTorch...");
        let torch_result = py.import("torch");
        match torch_result {
            Ok(torch) => {
                println!("PyTorch imported successfully");
                
                // Try to create the model with error handling
                match create_model_safely(py, torch, &args.checkpoint) {
                    Ok(model_bridge) => {
                        println!("Model created successfully");
                        rival_ai::Engine::new_with_model(model_bridge)
                    },
                    Err(e) => {
                        println!("Model creation failed: {}. Using fallback model.", e);
                        create_fallback_engine(py)
                    }
                }
            },
            Err(e) => {
                println!("Failed to import PyTorch: {}. Using fallback model.", e);
                create_fallback_engine(py)
            }
        }
    });
    
    // Helper function to create model safely
    fn create_model_safely(py: Python, torch: &PyModule, checkpoint_path: &str) -> PyResult<rival_ai::ModelBridge> {
        // Try to create model with automatic device selection
        println!("Creating model with automatic device selection");
        
        // Try to create model directly
        match create_model_with_device(py, torch, checkpoint_path) {
            Ok(model_bridge) => {
                println!("Model created successfully");
                return Ok(model_bridge);
            },
            Err(e) => {
                println!("Model creation failed: {}. This is unexpected.", e);
                return Err(e);
            }
        }
    }
    
    // Helper function to create model with automatic device selection
    fn create_model_with_device(py: Python, torch: &PyModule, checkpoint_path: &str) -> PyResult<rival_ai::ModelBridge> {
        println!("Creating model with automatic device selection");
        
        // Check if CUDA is available and select device
        let device_str = if torch.getattr("cuda")?.getattr("is_available")?.call0()?.extract::<bool>()? {
            "cuda"
        } else {
            "cpu"
        };
        let device = torch.getattr("device")?.call1((device_str,))?;
        println!("Selected device: {}", device_str);
        
        // First, ensure chess module is available
        println!("Importing chess module...");
        let chess_result = py.import("chess");
        match chess_result {
            Ok(_chess) => {
                println!("Chess module imported successfully");
            },
            Err(e) => {
                println!("Failed to import chess module: {}. This is required for the model.", e);
                return Err(PyErr::new::<pyo3::exceptions::PyImportError, _>("Chess module not available"));
            }
        }
        
        // Import model wrapper
        println!("Importing rival_ai.models.gnn...");
        let model_wrapper_result = py.import("rival_ai.models.gnn");
        match model_wrapper_result {
            Ok(model_wrapper) => {
                println!("rival_ai.models.gnn imported successfully");
                
                // Get the ChessGNN class
                let model_class_result = model_wrapper.getattr("ChessGNN");
                match model_class_result {
                    Ok(model_class) => {
                        println!("ChessGNN class found");
                        
                        // Create model instance
                        let model_result = model_class.call0();
                        match model_result {
                            Ok(model) => {
                                println!("Model instance created successfully");
                                
                                // Move model to the correct device after creation
                                let to_device_result = model.call_method1("to", (device,));
                                match to_device_result {
                                    Ok(model) => {
                                        println!("Model moved to device successfully");
                                        
                                        // Load checkpoint
                                        let checkpoint_result = torch.getattr("load")?.call1((checkpoint_path,));
                                        match checkpoint_result {
                                            Ok(checkpoint) => {
                                                println!("Checkpoint loaded successfully");
                                                
                                                // Load state dict
                                                let model_state_dict = checkpoint.get_item("model_state_dict")?;
                                                let load_result = model.call_method1("load_state_dict", (model_state_dict,));
                                                match load_result {
                                                    Ok(_) => {
                                                        println!("State dict loaded successfully");
                                                        
                                                        // Set to evaluation mode
                                                        let eval_result = model.call_method0("eval");
                                                        match eval_result {
                                                            Ok(_) => {
                                                                println!("Model set to evaluation mode");
                                                                
                                                                // Create ModelBridge from the Python model
                                                                Ok(rival_ai::ModelBridge::new(model.into(), Some(device_str.to_string())))
                                                            },
                                                            Err(e) => {
                                                                println!("Failed to set model to evaluation mode: {}", e);
                                                                Err(e)
                                                            }
                                                        }
                                                    },
                                                    Err(e) => {
                                                        println!("Failed to load state dict: {}", e);
                                                        Err(e)
                                                    }
                                                }
                                            },
                                            Err(e) => {
                                                println!("Failed to load checkpoint: {}", e);
                                                Err(e)
                                            }
                                        }
                                    },
                                    Err(e) => {
                                        println!("Failed to move model to device: {}", e);
                                        Err(e)
                                    }
                                }
                            },
                            Err(e) => {
                                println!("Failed to create model instance: {}", e);
                                Err(e)
                            }
                        }
                    },
                    Err(e) => {
                        println!("Failed to get ChessGNN class: {}", e);
                        Err(e)
                    }
                }
            },
            Err(e) => {
                println!("Failed to import rival_ai.models.gnn: {}", e);
                Err(e)
            }
        }
    }
    
    // Helper function to create fallback engine
    fn create_fallback_engine(py: Python) -> rival_ai::Engine {
        println!("Creating fallback engine with random move generator");
        
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
        rival_ai::Engine::new_with_model(model_bridge)
    }
    
    let app_state = web::Data::new(AppState {
        engine: Mutex::new(engine),
    });

    println!("Starting RivalAI server on http://{}:{}", args.host, args.port);
    HttpServer::new(move || {
        let cors = Cors::default()
            .allow_any_origin()
            .allow_any_method()
            .allow_any_header()
            .max_age(3600);

        App::new()
            .wrap(cors)
            .wrap(middleware::Logger::default())
            .app_data(app_state.clone())
            .route("/api/games/{game_id}/moves", web::post().to(make_move))
            .route("/api/games/{game_id}/engine-move", web::post().to(engine_move))
            .route("/api/games", web::post().to(create_game))
            .route("/api/games/{game_id}", web::get().to(get_game))
            .route("/api/games/{game_id}/reset", web::post().to(reset_game))
            .route("/ws/{game_id}", web::get().to(ws_route))
    })
    .bind(format!("{}:{}", args.host, args.port))?
    .run()
    .await
} 