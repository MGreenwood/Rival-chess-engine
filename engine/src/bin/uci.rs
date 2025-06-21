use rival_ai::{
    engine::Engine,
    bridge::python::ModelBridge,
    game_storage::{GameStorage, GameState as StorageGameState, GameMode, GameStatus, GameMetadata},
};
use std::io::{self, BufRead, BufReader, Write};
use std::collections::HashMap;
use std::env;
use chess::{Board, ChessMove, Color, BoardStatus};
use std::str::FromStr;
use chrono::Utc;
use pyo3::prelude::*;
use uuid::Uuid;

struct UciEngine {
    engine: Engine,
    board: Board,
    game_storage: GameStorage,
    current_game: Option<StorageGameState>,
    options: HashMap<String, String>,
    debug: bool,
}

impl UciEngine {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        // Initialize Python and create model bridge
        pyo3::prepare_freethreaded_python();
        
        let model_bridge = Python::with_gil(|py| -> PyResult<ModelBridge> {
            // Import necessary modules
            let sys = py.import("sys")?;
            let path = sys.getattr("path")?;
            path.call_method1("append", ("../python/src",))?;
            
            // Model loading code
            let model_code = r#"
import torch
import chess
from rival_ai.models import ChessGNN
from rival_ai.pag import PositionalAdjacencyGraph
import numpy as np

class UCIModelWrapper:
    def __init__(self, checkpoint_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ChessGNN(hidden_dim=256, num_layers=4, num_heads=4, dropout=0.1)
        
        if checkpoint_path:
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                print(f"UCI: Loaded model from {checkpoint_path}")
            except Exception as e:
                print(f"UCI: Failed to load model from {checkpoint_path}: {e}")
                print("UCI: Using randomly initialized model")
        else:
            print("UCI: Using randomly initialized model")
            
        self.model.to(self.device)
        self.model.eval()
    
    def predict_with_board(self, fen_string):
        try:
            board = chess.Board(fen_string)
            # For now, return random values
            # TODO: Implement proper PAG conversion and model inference
            policy = np.random.random(5312).astype(np.float32)
            policy = policy / policy.sum()  # Normalize
            value = np.random.random() * 2 - 1  # Random value between -1 and 1
            return policy.tolist(), float(value)
        except Exception as e:
            print(f"UCI: Error in predict_with_board: {e}")
            # Return default random policy
            policy = np.random.random(5312).astype(np.float32)
            policy = policy / policy.sum()
            value = 0.0
            return policy.tolist(), float(value)
"#;
            
            py.run(model_code, None, None)?;
            
            // Get model path from command line args or use default
            let args: Vec<String> = env::args().collect();
            let model_path = if args.len() > 1 {
                args[1].clone()
            } else {
                "../python/experiments/rival_ai_v1_Alice/run_20250617_221622/checkpoints/best_model.pt".to_string()
            };
            
            let model_wrapper = py.eval(&format!("UCIModelWrapper('{}')", model_path), None, None)?;
            Ok(ModelBridge::new(model_wrapper.to_object(py), Some("cuda".to_string())))
        })?;

        let engine = Engine::new_with_model(model_bridge);
        let board = Board::default();
        
        // Initialize game storage in UCI training games directory
        let games_dir = "../python/training_games/uci_matches";
        std::fs::create_dir_all(games_dir).unwrap_or_default();
        let game_storage = GameStorage::new(games_dir)?;
        
        let mut options = HashMap::new();
        options.insert("Hash".to_string(), "64".to_string());
        options.insert("PAG_Mode".to_string(), "true".to_string());
        options.insert("MCTS_Simulations".to_string(), "1000".to_string());
        options.insert("Training_Mode".to_string(), "true".to_string());
        options.insert("Collect_Data".to_string(), "true".to_string());
        
        Ok(UciEngine {
            engine,
            board,
            game_storage,
            current_game: None,
            options,
            debug: false,
        })
    }

    fn send_id(&self) {
        println!("id name RivalAI v1.0");
        println!("id author RivalAI Development Team");
    }

    fn send_options(&self) {
        println!("option name Hash type spin default 64 min 1 max 1024");
        println!("option name PAG_Mode type check default true");
        println!("option name MCTS_Simulations type spin default 1000 min 100 max 10000");
        println!("option name Training_Mode type check default true");
        println!("option name Collect_Data type check default true");
        println!("option name Neural_Model_Path type string default ../python/experiments/rival_ai_v1_Alice/run_20250617_221622/checkpoints/best_model.pt");
        println!("option name Clear Hash type button");
    }

    fn handle_setoption(&mut self, line: &str) {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 4 && parts[1] == "name" {
            let name = parts[2];
            if parts.len() >= 6 && parts[3] == "value" {
                let value = parts[4..].join(" ");
                self.options.insert(name.to_string(), value);
                if self.debug {
                    println!("info string Set option {} to {}", name, self.options[name]);
                }
            }
        }
    }

    fn handle_position(&mut self, line: &str) {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 2 {
            return;
        }

        // Parse position
        if parts[1] == "startpos" {
            self.board = Board::default();
            self.start_new_game();
        } else if parts[1] == "fen" && parts.len() >= 8 {
            let fen = parts[2..8].join(" ");
            if let Ok(board) = Board::from_str(&fen) {
                self.board = board;
                self.start_new_game();
            }
        }

        // Apply moves if present
        if let Some(moves_idx) = parts.iter().position(|&s| s == "moves") {
            for move_str in &parts[moves_idx + 1..] {
                if let Ok(chess_move) = ChessMove::from_str(move_str) {
                    if self.board.legal(chess_move) {
                        self.board = self.board.make_move_new(chess_move);
                        self.record_move(move_str);
                    }
                }
            }
        }
    }

    fn start_new_game(&mut self) {
        let game_id = Uuid::new_v4().to_string();
        
        self.current_game = Some(StorageGameState {
            metadata: GameMetadata {
                game_id,
                mode: GameMode::UCI, // We'll need to add this variant
                created_at: Utc::now(),
                last_move_at: Utc::now(),
                status: GameStatus::Active,
                total_moves: 0,
                player_color: "".to_string(), // Will be determined by who goes first
                player_name: Some("UCI_Opponent".to_string()),
                engine_version: "RivalAI_v1.0_UCI".to_string(),
            },
            board: self.board.to_string(),
            move_history: Vec::new(),
            analysis: None,
        });

        if self.debug {
            println!("info string Started new UCI game");
        }
    }

    fn record_move(&mut self, move_str: &str) {
        if let Some(ref mut game) = self.current_game {
            game.move_history.push(move_str.to_string());
            game.metadata.total_moves += 1;
            game.metadata.last_move_at = Utc::now();
            game.board = self.board.to_string();
        }
    }

    fn handle_go(&mut self, line: &str) {
        // Parse go command parameters
        let parts: Vec<&str> = line.split_whitespace().collect();
        
        // Parse time controls and search parameters
        let mut movetime_ms: Option<i32> = None;
        let mut depth: Option<i32> = None;
        let mut nodes: Option<i32> = None;
        let mut wtime_ms: Option<i32> = None;
        let mut btime_ms: Option<i32> = None;
        let mut winc_ms: Option<i32> = None;
        let mut binc_ms: Option<i32> = None;
        let mut movestogo: Option<i32> = None;
        let mut infinite = false;
        
        let mut i = 1; // Skip "go"
        while i < parts.len() {
            match parts[i] {
                "movetime" => {
                    if i + 1 < parts.len() {
                        movetime_ms = parts[i + 1].parse().ok();
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
                "depth" => {
                    if i + 1 < parts.len() {
                        depth = parts[i + 1].parse().ok();
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
                "nodes" => {
                    if i + 1 < parts.len() {
                        nodes = parts[i + 1].parse().ok();
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
                "wtime" => {
                    if i + 1 < parts.len() {
                        wtime_ms = parts[i + 1].parse().ok();
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
                "btime" => {
                    if i + 1 < parts.len() {
                        btime_ms = parts[i + 1].parse().ok();
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
                "winc" => {
                    if i + 1 < parts.len() {
                        winc_ms = parts[i + 1].parse().ok();
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
                "binc" => {
                    if i + 1 < parts.len() {
                        binc_ms = parts[i + 1].parse().ok();
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
                "movestogo" => {
                    if i + 1 < parts.len() {
                        movestogo = parts[i + 1].parse().ok();
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
                "infinite" => {
                    infinite = true;
                    i += 1;
                }
                _ => {
                    i += 1;
                }
            }
        }
        
        // Calculate thinking time based on parameters
        let thinking_time_ms = if let Some(mt) = movetime_ms {
            // Fixed time per move
            mt
        } else if let Some(wt) = wtime_ms {
            // Calculate time based on remaining clock time
            let our_time = if self.board.side_to_move() == chess::Color::White {
                wt
            } else {
                btime_ms.unwrap_or(wt)
            };
            
            let increment = if self.board.side_to_move() == chess::Color::White {
                winc_ms.unwrap_or(0)
            } else {
                binc_ms.unwrap_or(0)
            };
            
            // Simple time management: use 1/40th of remaining time + increment
            let moves_left = movestogo.unwrap_or(40);
            (our_time / moves_left as i32) + increment
        } else if infinite {
            // Search until stopped - use a reasonable default
            30000 // 30 seconds
        } else {
            // Default fallback
            5000 // 5 seconds
        };
        
        if self.debug {
            println!("info string Thinking for {}ms", thinking_time_ms);
        }
        
        // TODO: Actually use the thinking time to limit search
        // For now, we just get the best move immediately
        // In the future, this should integrate with MCTS time limits
        
        // Get best move from engine
        if let Some(best_move) = self.engine.get_best_move(self.board) {
            self.record_move(&best_move.to_string());
            self.board = self.board.make_move_new(best_move);
            
            // Check if game is over
            self.check_game_end();
            
            println!("bestmove {}", best_move);
        } else {
            println!("bestmove 0000");
        }
    }

    fn check_game_end(&mut self) {
        if let Some(ref mut game) = self.current_game {
            match self.board.status() {
                BoardStatus::Checkmate => {
                    game.metadata.status = if self.board.side_to_move() == Color::White {
                        GameStatus::BlackWins
                    } else {
                        GameStatus::WhiteWins
                    };
                    self.save_completed_game();
                }
                BoardStatus::Stalemate => {
                    game.metadata.status = GameStatus::DrawStalemate;
                    self.save_completed_game();
                }
                _ => {
                    // Game continues
                }
            }
        }
    }

    fn save_completed_game(&mut self) {
        if let Some(game) = self.current_game.take() {
            if self.options.get("Collect_Data").map_or(true, |v| v == "true") {
                if let Err(e) = self.game_storage.save_game(&game) {
                    if self.debug {
                        println!("info string Failed to save UCI game: {}", e);
                    }
                } else if self.debug {
                    println!("info string Saved UCI game with {} moves", game.metadata.total_moves);
                }
            }
        }
    }

    fn run(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let stdin = io::stdin();
        let mut stdout = io::stdout();
        
        for line in stdin.lock().lines() {
            let line = line?;
            let trimmed = line.trim();
            
            if self.debug {
                println!("info string Received: {}", trimmed);
            }
            
            match trimmed {
                "uci" => {
                    self.send_id();
                    self.send_options();
                    println!("uciok");
                    stdout.flush()?;
                }
                "isready" => {
                    println!("readyok");
                    stdout.flush()?;
                }
                "ucinewgame" => {
                    self.start_new_game();
                }
                "quit" => {
                    // Save any ongoing game
                    if self.current_game.is_some() {
                        self.save_completed_game();
                    }
                    break;
                }
                _ if trimmed.starts_with("debug ") => {
                    self.debug = trimmed.ends_with(" on");
                }
                _ if trimmed.starts_with("setoption ") => {
                    self.handle_setoption(trimmed);
                }
                _ if trimmed.starts_with("position ") => {
                    self.handle_position(trimmed);
                }
                _ if trimmed.starts_with("go ") => {
                    self.handle_go(trimmed);
                    stdout.flush()?;
                }
                _ => {
                    // Ignore unknown commands as per UCI spec
                }
            }
        }
        
        Ok(())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut engine = UciEngine::new()?;
    engine.run()
} 