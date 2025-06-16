use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use pyo3::exceptions::PyValueError;

#[pyclass]
#[derive(Clone)]
pub struct ModelBridge {
    model: PyObject,
    device: String,
}

#[pymethods]
impl ModelBridge {
    #[new]
    pub fn new(model: PyObject, device: Option<String>) -> Self {
        Self {
            model,
            device: device.unwrap_or_else(|| String::from("cuda")),
        }
    }
    
    pub fn predict(&self) -> PyResult<(Vec<f32>, f32)> {
        Python::with_gil(|py| {
            let result = self.model.call_method0(py, "predict")
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let (policy, value) = result.extract::<(Vec<f32>, f32)>(py)?;
            Ok((policy, value))
        })
    }
    
    pub fn predict_with_board(&self, board: String) -> PyResult<(Vec<f32>, f32)> {
        Python::with_gil(|py| {
            let args = PyTuple::new(py, &[board]);
            let result = self.model.call_method1(py, "predict_with_board", args)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let (policy, value) = result.extract::<(Vec<f32>, f32)>(py)?;
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_bridge_creation() {
        let bridge = ModelBridge::new(PyObject::new(Python::with_gil(|py| py.None())), None);
        assert!(bridge.device == "cuda");
    }
} 