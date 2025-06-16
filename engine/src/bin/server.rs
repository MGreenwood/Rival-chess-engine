use actix_web::{web, App, HttpResponse, HttpServer, Responder, middleware, HttpRequest};
use actix_cors::Cors;
use serde::{Deserialize, Serialize};
use chess::{ChessMove, BoardStatus, MoveGen};
use std::sync::Mutex;
use uuid::Uuid;
use serde_json;
use actix_web_actors::ws;
use actix::{Actor, StreamHandler, ActorContext};
use rival_ai::{Engine, bridge::python::ModelBridge};
use pyo3::prelude::*;

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
    timeControl: Option<TimeControl>,
    engineStrength: Option<u32>,
    color: Option<String>,
}

#[derive(Deserialize)]
struct TimeControl {
    initial: u32,
    increment: u32,
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
            
            // Get engine's response move
            let engine_move = if let Some(mv) = engine.get_best_move(engine.board()) {
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
                is_player_turn: true,
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

async fn create_game(
    _settings: web::Json<serde_json::Value>,
) -> impl Responder {
    let game_id = Uuid::new_v4().to_string();
    let board = chess::Board::default().to_string();
    let state = GameState {
        game_id,
        board,
        status: "active".to_string(),
        move_history: vec![],
        is_player_turn: true,
    };
    HttpResponse::Ok().json(state)
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
    
    // Initialize Python interpreter
    pyo3::prepare_freethreaded_python();
    
    // Create engine with proper model loading
    let engine = Python::with_gil(|py| {
        // Try to set up the Python environment
        let setup_result = py.run(r#"
import sys
import os

# Try to add the Python source path
try:
    sys.path.append('../python/src')
    print(f"Added Python source path: {os.path.abspath('../python/src')}")
except Exception as e:
    print(f"Could not add Python source path: {e}")

# Try to add the engine's venv path
try:
    venv_path = os.path.join(os.getcwd(), 'venvEngine', 'Lib', 'site-packages')
    if os.path.exists(venv_path):
        sys.path.insert(0, venv_path)
        print(f"Added venv path: {venv_path}")
    else:
        print(f"Venv path not found: {venv_path}")
except Exception as e:
    print(f"Could not add venv path: {e}")

print("Python path:")
for p in sys.path[:10]:  # Show first 10 paths
    print(f"  {p}")
"#, None, None);
        
        if let Err(e) = setup_result {
            println!("Warning: Python setup failed: {}", e);
        }
        
        // Try to create the model wrapper with error handling
        let model_code = r#"
import sys

class ModelWrapper:
    def __init__(self, checkpoint_path=None):
        self.has_torch = False
        self.has_model = False
        
        try:
            import torch
            import chess
            from rival_ai.models import ChessGNN
            from rival_ai.pag import PositionalAdjacencyGraph
            import numpy as np
            
            self.has_torch = True
            print("Successfully imported PyTorch and dependencies")
            
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = ChessGNN(hidden_dim=256, num_layers=4, num_heads=4, dropout=0.1)
            
            if checkpoint_path:
                try:
                    checkpoint = torch.load(checkpoint_path, map_location=self.device)
                    if 'model_state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        self.model.load_state_dict(checkpoint)
                    print(f"Loaded model from {checkpoint_path}")
                    self.has_model = True
                except Exception as e:
                    print(f"Failed to load model from {checkpoint_path}: {e}")
                    print("Using randomly initialized model")
            else:
                print("Using randomly initialized model")
                
            self.model.to(self.device)
            self.model.eval()
            
        except ImportError as e:
            print(f"Could not import required modules: {e}")
            print("Falling back to random policy generation")
            # Import basic modules for fallback
            try:
                import random
                import numpy as np
                self.np = np
                self.random = random
            except ImportError:
                print("Could not import numpy/random, using pure Python fallback")
                self.np = None
                self.random = None
        except Exception as e:
            print(f"Error initializing model: {e}")
            print("Using fallback mode")
    
    def _generate_random_policy(self):
        """Generate a random policy vector."""
        if self.np is not None:
            policy = self.np.random.random(5312).astype(self.np.float32)
            policy = policy / policy.sum()  # Normalize
            return policy.tolist()
        else:
            # Pure Python fallback
            import random
            policy = [random.random() for _ in range(5312)]
            total = sum(policy)
            policy = [p / total for p in policy]  # Normalize
            return policy
    
    def _generate_random_value(self):
        """Generate a random value between -1 and 1."""
        if self.random is not None:
            return self.random.random() * 2 - 1
        else:
            import random
            return random.random() * 2 - 1
    
    def predict(self):
        """Predict policy and value for current position."""
        policy = self._generate_random_policy()
        value = self._generate_random_value()
        return policy, float(value)
    
    def predict_with_board(self, fen_string):
        """Predict policy and value for a given board position."""
        try:
            if self.has_torch:
                import chess
                board = chess.Board(fen_string)
                # TODO: Implement proper PAG conversion and model inference
                # For now, return random values even with torch available
                
            policy = self._generate_random_policy()
            value = self._generate_random_value()
            return policy, float(value)
            
        except Exception as e:
            print(f"Error in predict_with_board: {e}")
            # Return default random policy
            policy = self._generate_random_policy()
            value = 0.0
            return policy, float(value)
    
    def predict_batch(self, fen_strings):
        """Predict policies and values for a batch of positions."""
        policies = []
        values = []
        for fen in fen_strings:
            policy, value = self.predict_with_board(fen)
            policies.append(policy)
            values.append(value)
        return policies, values
"#;
        
        // Execute the model wrapper code
        match py.run(model_code, None, None) {
            Ok(_) => println!("Model wrapper code executed successfully"),
            Err(e) => {
                println!("Error executing model wrapper code: {}", e);
                return Engine::new(); // Fallback to engine without model
            }
        }
        
        // Create the model wrapper instance
        let model_wrapper_result = py.eval("ModelWrapper('checkpoints/rival_ai/checkpoint_20250615_042203_epoch_159.pt')", None, None);
        
        match model_wrapper_result {
            Ok(model_wrapper) => {
                let bridge = ModelBridge::new(model_wrapper.to_object(py), Some("cuda".to_string()));
                println!("Created model bridge with proper model loading");
                Engine::new_with_model(bridge)
            }
            Err(e) => {
                println!("Error creating model wrapper: {}", e);
                println!("Falling back to engine without neural network model");
                Engine::new()
            }
        }
    });
    
    let app_state = web::Data::new(AppState {
        engine: Mutex::new(engine),
    });

    println!("Starting RivalAI server on http://127.0.0.1:3000");
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
            .route("/api/games", web::post().to(create_game))
            .route("/api/games/{game_id}/moves", web::post().to(make_move))
            .route("/ws/{game_id}", web::get().to(ws_route))
    })
    .bind("127.0.0.1:3000")?
    .run()
    .await
} 