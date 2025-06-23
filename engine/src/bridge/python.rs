use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use pyo3::exceptions::PyValueError;
use chess::ChessMove;

#[allow(dead_code)]
#[derive(Clone)]
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
            // 🔥 FIXED PROMOTION ENCODING 🔥
            // The old formula (from * 64 + to) * 4 produces indices up to 16,380
            // But we only have 1216 promotion slots (4096-5311)
            // Use compact encoding that fits in available space
            
            let promotion_piece_type = match promotion {
                chess::Piece::Knight => 0,  // 0-based for compact encoding
                chess::Piece::Bishop => 1,
                chess::Piece::Rook => 2,
                chess::Piece::Queen => 3,
                _ => return None,
            };
            
            // Extract file and rank information
            let from_file = from % 8;
            let from_rank = from / 8;
            let to_file = to % 8;
            let to_rank = to / 8;
            
            // Determine promotion direction
            let direction = if to_file == from_file {
                0  // Straight promotion
            } else if to_file == from_file.wrapping_sub(1) {
                1  // Capture left
            } else if to_file == from_file + 1 {
                2  // Capture right
            } else {
                return None;  // Invalid promotion
            };
            
            // Determine side (White or Black promotion)
            let side_offset = if from_rank == 6 && to_rank == 7 {
                0    // White promotion (rank 7 to 8)
            } else if from_rank == 1 && to_rank == 0 {
                96   // Black promotion (rank 2 to 1) - 8 files * 3 directions * 4 pieces = 96
            } else {
                return None;  // Invalid promotion ranks
            };
            
            // Compact index calculation
            // Each file gets 12 indices (3 directions * 4 pieces)
            // Format: 4096 + side_offset + (file * 12) + (direction * 4) + piece_type
            let index = 4096 + side_offset + (from_file * 12) + (direction * 4) + promotion_piece_type;
            
            // Ensure the index is within bounds (should always be true with this encoding)
            if index < 5312 {
                Some(index)
            } else {
                None  // Safety check
            }
        } else {
            // Regular moves are encoded as from * 64 + to
            let index = from * 64 + to;
            
            // Ensure the index is within bounds
            if index < 4096 {
                Some(index)
            } else {
                None
            }
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