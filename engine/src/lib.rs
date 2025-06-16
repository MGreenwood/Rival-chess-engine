use pyo3::prelude::*;
use pyo3::types::PyDict;
use chess::{Board, ChessMove, Color, Piece};
use std::str::FromStr;

// Make modules public
pub mod engine;
pub mod mcts;
pub mod pag;
pub mod game;
pub mod bridge;
pub mod board;
pub mod evaluation;
pub mod search;

// Re-export commonly used types
pub use engine::Engine;
pub use game::Game;
pub use board::PAGBoard;
// Import ModelBridge only once
pub use crate::bridge::python::ModelBridge;

// Create a wrapper type for ChessMove to implement IntoPy
#[derive(Clone)]
struct PyChessMove(ChessMove);

impl From<ChessMove> for PyChessMove {
    fn from(mv: ChessMove) -> Self {
        PyChessMove(mv)
    }
}

impl IntoPy<PyObject> for PyChessMove {
    fn into_py(self, py: Python<'_>) -> PyObject {
        let from_square = self.0.get_source().to_string();
        let to_square = self.0.get_dest().to_string();
        let promotion = self.0.get_promotion().map(|p| p.to_string(Color::White));
        
        let dict = PyDict::new(py);
        dict.set_item("from", from_square).unwrap();
        dict.set_item("to", to_square).unwrap();
        if let Some(p) = promotion {
            dict.set_item("promotion", p).unwrap();
        }
        dict.into_py(py)
    }
}

// Helper function to parse a move string into a ChessMove
fn parse_move(move_str: &str) -> PyResult<ChessMove> {
    if move_str.len() != 4 && move_str.len() != 5 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Move must be 4 characters long (or 5 for promotion)",
        ));
    }

    let chars: Vec<char> = move_str.chars().collect();
    let from_file = chars[0] as u8 - b'a';
    let from_rank = chars[1] as u8 - b'1';
    let to_file = chars[2] as u8 - b'a';
    let to_rank = chars[3] as u8 - b'1';

    if from_file > 7 || from_rank > 7 || to_file > 7 || to_rank > 7 {
        return Err(pyo3::exceptions::PyValueError::new_err("Invalid square coordinates"));
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
            'q' => Some(Piece::Queen),
            'r' => Some(Piece::Rook),
            'b' => Some(Piece::Bishop),
            'n' => Some(Piece::Knight),
            _ => return Err(pyo3::exceptions::PyValueError::new_err("Invalid promotion piece")),
        }
    } else {
        None
    };

    Ok(ChessMove::new(from, to, promotion))
}

#[pyclass]
pub struct PyEngine {
    engine: Engine,
}

#[pymethods]
impl PyEngine {
    #[new]
    pub fn new() -> Self {
        Self {
            engine: Engine::new(),
        }
    }

    pub fn make_move(&mut self, move_str: &str) -> PyResult<bool> {
        let chess_move = parse_move(move_str)?;
        let board = self.engine.board();
        if !board.legal(chess_move) {
            return Ok(false);
        }
        self.engine.make_move(chess_move);
        Ok(true)
    }

    pub fn get_best_move(&self, fen: &str) -> PyResult<Option<PyObject>> {
        let board = Board::from_str(fen).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Invalid FEN string: {}", e))
        })?;
        Python::with_gil(|py| {
            Ok(self.engine.get_best_move(&board).map(|mv| PyChessMove::from(mv).into_py(py)))
        })
    }

    pub fn get_fen(&self) -> String {
        self.engine.board().to_string()
    }

    pub fn is_game_over(&self) -> bool {
        self.engine.is_game_over()
    }

    pub fn is_check(&self) -> bool {
        self.engine.is_check()
    }

    pub fn is_draw(&self) -> bool {
        self.engine.is_draw()
    }

    pub fn is_mate(&self) -> bool {
        self.engine.is_mate()
    }
}

#[pyclass]
pub struct RivalAI {
    engine: Engine,
    max_depth: u32,
    max_nodes: u32,
    model: Option<ModelBridge>,
}

#[pymethods]
impl RivalAI {
    #[new]
    pub fn new(config: Option<&PyDict>) -> PyResult<Self> {
        let max_depth = config
            .and_then(|d| d.get_item("max_depth"))
            .and_then(|v| v.extract::<u32>().ok())
            .unwrap_or(4);
        
        let max_nodes = config
            .and_then(|d| d.get_item("max_nodes"))
            .and_then(|v| v.extract::<u32>().ok())
            .unwrap_or(10000);
        
        let model_path = config
            .and_then(|d| d.get_item("model_path"))
            .and_then(|v| v.extract::<String>().ok());
        
        let use_python_model = config
            .and_then(|d| d.get_item("use_python_model"))
            .and_then(|v| v.extract::<bool>().ok())
            .unwrap_or(false);

        let model = if use_python_model {
            model_path.map(|path| {
                let py = Python::with_gil(|py| {
                    // Import the Python model module
                    let model = PyModule::import(py, "rival_ai.model").unwrap();
                    model.getattr("load_model").unwrap().call1((path,)).unwrap().into_py(py)
                });
                ModelBridge::new(py, None)
            })
        } else {
            None
        };

        Ok(Self {
            engine: Engine::new(),
            max_depth,
            max_nodes,
            model,
        })
    }

    pub fn make_move(&mut self, move_str: &str) -> PyResult<bool> {
        let chess_move = parse_move(move_str)?;
        let board = self.engine.board();
        if !board.legal(chess_move) {
            return Ok(false);
        }
        self.engine.make_move(chess_move);
        Ok(true)
    }

    pub fn get_best_move(&self, fen: &str) -> PyResult<Option<PyObject>> {
        let board = Board::from_str(fen).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Invalid FEN string: {}", e))
        })?;
        Python::with_gil(|py| {
            Ok(self.engine.get_best_move(&board).map(|mv| PyChessMove::from(mv).into_py(py)))
        })
    }

    pub fn get_fen(&self) -> String {
        self.engine.board().to_string()
    }

    pub fn is_game_over(&self) -> bool {
        self.engine.is_game_over()
    }

    pub fn is_check(&self) -> bool {
        self.engine.is_check()
    }

    pub fn is_draw(&self) -> bool {
        self.engine.is_draw()
    }

    pub fn is_mate(&self) -> bool {
        self.engine.is_mate()
    }
}

#[pymodule]
fn rival_ai(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyEngine>()?;
    m.add_class::<RivalAI>()?;
    m.add_class::<ModelBridge>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::IntoPyDict;

    #[test]
    fn test_rival_ai_creation() {
        Python::with_gil(|py| {
            let config = PyDict::new(py);
            config.set_item("max_depth", 4).unwrap();
            config.set_item("max_nodes", 1000).unwrap();
            config.set_item("model_path", "model.pt").unwrap();
            config.set_item("use_python_model", true).unwrap();

            let rival_ai = RivalAI::new(Some(&config)).unwrap();
            assert!(!rival_ai.is_game_over());
        });
    }

    #[test]
    fn test_move_parsing() {
        Python::with_gil(|py| {
            let config = PyDict::new(py);
            config.set_item("max_depth", 4).unwrap();
            config.set_item("max_nodes", 1000).unwrap();
            config.set_item("model_path", "model.pt").unwrap();
            config.set_item("use_python_model", true).unwrap();

            let rival_ai = RivalAI::new(Some(&config)).unwrap();
            let mv_str = rival_ai.parse_move_internal("e2e4").unwrap().to_string();
            assert_eq!(mv_str, "e2e4");
        });
    }
} 