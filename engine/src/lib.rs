use pyo3::prelude::*;
use chess::{Board, ChessMove};
use std::str::FromStr;

pub mod bridge;
pub mod engine;

pub use engine::Engine;
pub use bridge::python::ModelBridge;

#[pyclass]
pub struct PyEngine {
    engine: Engine,
}

#[pymethods]
impl PyEngine {
    #[new]
    fn new() -> Self {
        Python::with_gil(|py| {
            // Create a simple fallback model
            let code = r#"
class FallbackModel:
    def predict_with_board(self, board_fen):
        return ([0.0] * 5312, 0.0)
    def eval(self):
        pass
FallbackModel()
"#;
            let locals = py.eval(code, None, None).unwrap().extract().unwrap();
            let model = ModelBridge::new(locals, Some("cpu".to_string()));
            Self {
                engine: Engine::new_with_model(model),
            }
        })
    }

    #[pyo3(text_signature = "($self, fen)")]
    fn get_best_move(&self, fen: &str) -> PyResult<Option<String>> {
        let board = Board::from_str(fen).map_err(|e| {
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

#[pymodule]
fn rival_ai(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyEngine>()?;
    Ok(())
}