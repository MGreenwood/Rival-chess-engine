use rival_ai::{
    engine::Engine,
    bridge::python::ModelBridge,
};
use std::io::{self, Write};
use std::env;
use pyo3::prelude::*;

fn parse_move(move_str: &str) -> Option<chess::ChessMove> {
    if move_str.len() != 4 && move_str.len() != 5 {
        return None;
    }

    let chars: Vec<char> = move_str.chars().collect();
    let from_file = chars[0] as u8 - b'a';
    let from_rank = chars[1] as u8 - b'1';
    let to_file = chars[2] as u8 - b'a';
    let to_rank = chars[3] as u8 - b'1';

    if from_file > 7 || from_rank > 7 || to_file > 7 || to_rank > 7 {
        return None;
    }

    let from = chess::Square::make_square(
        chess::Rank::from_index(from_rank as usize),
        chess::File::from_index(from_file as usize),
    );
    let to = chess::Square::make_square(
        chess::Rank::from_index(to_rank as usize),
        chess::File::from_index(to_file as usize),
    );

    // Handle promotion
    let promotion = if move_str.len() == 5 {
        match chars[4].to_ascii_lowercase() {
            'q' => Some(chess::Piece::Queen),
            'r' => Some(chess::Piece::Rook),
            'b' => Some(chess::Piece::Bishop),
            'n' => Some(chess::Piece::Knight),
            _ => return None,
        }
    } else {
        None
    };

    Some(chess::ChessMove::new(from, to, promotion))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize Python interpreter
    pyo3::prepare_freethreaded_python();
    
    let args: Vec<String> = env::args().collect();
    
    // Create engine with model
    let engine = Python::with_gil(|py| {
        // Import necessary modules
        let sys = py.import("sys").unwrap();
        let path = sys.getattr("path").unwrap();
        path.call_method1("append", ("../python/src",)).unwrap();
        
        // Import the model wrapper
        let model_code = r#"
import torch
import chess
from rival_ai.models import ChessGNN
from rival_ai.pag import PositionalAdjacencyGraph
import numpy as np

class ModelWrapper:
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
                print(f"Loaded model from {checkpoint_path}")
            except Exception as e:
                print(f"Failed to load model from {checkpoint_path}: {e}")
                print("Using randomly initialized model")
        else:
            print("Using randomly initialized model")
            
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self):
        # Return random policy and value for default case
        policy = np.random.random(5312).astype(np.float32)
        policy = policy / policy.sum()  # Normalize
        value = np.random.random() * 2 - 1  # Random value between -1 and 1
        return policy.tolist(), float(value)
    
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
            print(f"Error in predict_with_board: {e}")
            # Return default random policy
            policy = np.random.random(5312).astype(np.float32)
            policy = policy / policy.sum()
            value = 0.0
            return policy.tolist(), float(value)
    
    def predict_batch(self, fen_strings):
        policies = []
        values = []
        for fen in fen_strings:
            policy, value = self.predict_with_board(fen)
            policies.append(policy)
            values.append(value)
        return policies, values
"#;
        
        // Execute the model wrapper code
        py.run(model_code, None, None).unwrap();
        
        // Create the model wrapper instance
        let model_path = if args.len() > 1 {
            args[1].clone()
        } else {
            "checkpoints/rival_ai/checkpoint_20250615_042203_epoch_159.pt".to_string()
        };
        
        let model_wrapper = py.eval(&format!("ModelWrapper('{}')", model_path), None, None).unwrap();
        let bridge = ModelBridge::new(model_wrapper.to_object(py), Some("cuda".to_string()));
        Engine::new_with_model(bridge)
    });

    println!("RivalAI Chess Engine");
    println!("Enter moves in algebraic notation (e.g., e2e4, Nf3, O-O)");
    println!("Type 'quit' to exit");

    let mut engine = engine;
    
    loop {
        // Display current board
        println!("\nCurrent position:");
        println!("{}", engine.board());
        
        if engine.is_game_over() {
            println!("Game over!");
            break;
        }
        
        // Get player move
        print!("Your move: ");
        io::stdout().flush()?;
        
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();
        
        if input == "quit" {
            break;
        }
        
        // Parse and make player move
        match parse_move(input) {
            Some(player_move) => {
                if engine.board().legal(player_move) {
                    engine.make_move(player_move);
                    println!("Player move: {}", input);
                    
                    if engine.is_game_over() {
                        println!("\nFinal position:");
                        println!("{}", engine.board());
                        println!("Game over!");
                        break;
                    }
                    
                    // Get engine move
                    if let Some(engine_move) = engine.get_best_move(engine.board()) {
                        engine.make_move(engine_move);
                        println!("Engine move: {}{}", engine_move.get_source(), engine_move.get_dest());
                    } else {
                        println!("Engine couldn't find a move");
                        break;
                    }
                } else {
                    println!("Illegal move: {}", input);
                }
            }
            None => {
                println!("Invalid move format: {}", input);
            }
        }
    }
    
    Ok(())
} 