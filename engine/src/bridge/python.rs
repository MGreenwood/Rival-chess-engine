use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use pyo3::exceptions::PyValueError;
use chess::ChessMove;

#[allow(dead_code)]
pub struct ModelBridge {
    model: PyObject,
    device: Option<String>,
}

impl ModelBridge {
    pub fn new(model: PyObject, device: Option<String>) -> Self {
        Self { model, device }
    }
    
    pub fn predict(&self) -> PyResult<(Vec<f32>, f32)> {
        Python::with_gil(|py| {
            let result = self.model.call_method0(py, "predict")
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let (policy, value) = result.extract::<(Vec<f32>, f32)>(py)?;
            Ok((policy, value))
        })
    }
    
    pub fn predict_with_board(&self, board_fen: String) -> PyResult<(Vec<f32>, f32)> {
        Python::with_gil(|py| {
            let args = PyTuple::new(py, &[board_fen]);
            let result = self.model.call_method1(py, "predict_with_board", args)?;
            let tuple: &PyTuple = result.downcast(py)?;
            
            // Extract policy and value from tuple
            let policy_obj = tuple.get_item(0)?;
            let value_obj = tuple.get_item(1)?;
            
            let policy: Vec<f32> = policy_obj.extract()?;
            let value: f32 = value_obj.extract()?;
            
            Ok((policy, value))
        })
    }
    
    pub fn predict_batch(&self, boards: Vec<String>) -> PyResult<(Vec<Vec<f32>>, Vec<f32>)> {
        Python::with_gil(|py| {
            let board_list = PyList::new(py, boards);
            let args = PyTuple::new(py, &[board_list]);
            let result = self.model.call_method1(py, "predict_batch", args)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let (policies, values) = result.extract::<(Vec<Vec<f32>>, Vec<f32>)>(py)?;
            Ok((policies, values))
        })
    }

    pub fn move_to_policy_idx(&self, mv: ChessMove) -> Option<usize> {
        let from = mv.get_source().to_index();
        let to = mv.get_dest().to_index();
        let promotion = mv.get_promotion();
        
        if let Some(promotion) = promotion {
            // Promotion moves are encoded after regular moves
            // Formula: 4096 + (from_square * 64 + to_square) * 4 + promotion_piece_type - 1
            let piece_offset = match promotion {
                chess::Piece::Knight => 0,  // promotion = 1, so 1-1 = 0
                chess::Piece::Bishop => 1,  // promotion = 2, so 2-1 = 1
                chess::Piece::Rook => 2,    // promotion = 3, so 3-1 = 2
                chess::Piece::Queen => 3,   // promotion = 4, so 4-1 = 3
                _ => return None,
            };
            let base = 4096 + (from * 64 + to) * 4;
            Some(base + piece_offset)
        } else {
            // Regular moves are encoded as from * 64 + to
            Some(from * 64 + to)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_bridge_creation() {
        let bridge = ModelBridge::new(PyObject::new(Python::with_gil(|py| py.None())), None);
        assert!(bridge.device == None);
    }
} 