#![allow(non_local_definitions)]
use pyo3::prelude::*;
use chess::{Board as ChessBoard, ChessMove};
use std::str::FromStr;

pub mod bridge;
pub mod engine;
pub mod game_storage;
pub mod mcts;
pub mod board;
pub mod game;
pub mod evaluation;
pub mod search;
pub mod pag;
pub mod python_bridge;

pub use engine::Engine;
pub use bridge::python::ModelBridge;
pub use game_storage::{GameStorage, GameState, GameMode, GameStatus, GameMetadata};
pub use mcts::{MCTS, MCTSConfig};
pub use board::Board as RivalBoard;
pub use game::Game;
pub use evaluation::evaluate_position;
pub use pag::{PAG, DensePAGBuilder, create_dense_pag_from_fen, create_dense_pag_from_board, FeatureExtractor};
pub use python_bridge::{PyPAGEngine, PyDensePAG, PyPAGStats};

/// Python Engine wrapper for compatibility
#[pyclass]
pub struct PyEngine {
    engine: Engine,
}

#[pymethods]
impl PyEngine {
    #[new]
    fn new() -> Self {
        // Create a dummy ModelBridge for basic functionality
        let model = ModelBridge::new(pyo3::Python::with_gil(|py| py.None()), None);
        let engine = Engine::new_with_model(model);
        PyEngine { engine }
    }

    fn get_best_move(&mut self, fen: &str) -> PyResult<String> {
        let board = ChessBoard::from_str(fen).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Invalid FEN string: {}", e))
        })?;
        
        match self.engine.get_best_move(board) {
            Some(chess_move) => Ok(chess_move.to_string()),
            None => Err(pyo3::exceptions::PyValueError::new_err("No legal moves available")),
        }
    }

    fn evaluate_position(&self, fen: &str) -> PyResult<f64> {
        // Basic evaluation - just count material for now
        let board = ChessBoard::from_str(fen).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Invalid FEN string: {}", e))
        })?;
        
        // Simple material count evaluation
        let mut score = 0.0;
        for square in chess::ALL_SQUARES {
            if let Some(piece) = board.piece_on(square) {
                let piece_value = match piece {
                    chess::Piece::Pawn => 1.0,
                    chess::Piece::Knight | chess::Piece::Bishop => 3.0,
                    chess::Piece::Rook => 5.0,
                    chess::Piece::Queen => 9.0,
                    chess::Piece::King => 0.0,
                };
                
                let multiplier = match board.color_on(square).unwrap() {
                    chess::Color::White => 1.0,
                    chess::Color::Black => -1.0,
                };
                
                score += piece_value * multiplier;
            }
        }
        
        Ok(score)
    }

    #[pyo3(text_signature = "($self, fen)")]
    fn get_best_move_from_fen(&self, fen: &str) -> PyResult<Option<String>> {
        let board = ChessBoard::from_str(fen).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Invalid FEN string: {}", e))
        })?;
        
        Ok(self.engine.get_best_move(board).map(|mv| {
            format!("{}{}", mv.get_source(), mv.get_dest())
        }))
    }

    #[pyo3(text_signature = "($self, move_str)")]
    fn make_move(&mut self, move_str: &str) -> PyResult<()> {
        let board = self.engine.board();
        if let Ok(mv) = ChessMove::from_san(&board, move_str) {
            self.engine.make_move(mv);
            Ok(())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid move"))
        }
    }

    #[pyo3(text_signature = "($self)")]
    fn get_board(&self) -> String {
        self.engine.board().to_string()
    }

    #[pyo3(text_signature = "($self)")]
    fn is_game_over(&self) -> bool {
        self.engine.is_game_over()
    }
}

/// Ultra-Dense PAG Python Module
#[pymodule]
fn rival_ai_engine(_py: Python, m: &PyModule) -> PyResult<()> {
    // Add legacy engine for compatibility
    m.add_class::<PyEngine>()?;
    
    // Add ultra-dense PAG system
    m.add_class::<python_bridge::PyPAGEngine>()?;
    m.add_class::<python_bridge::PyDensePAG>()?;
    m.add_class::<python_bridge::PyPAGStats>()?;
    
    Ok(())
}