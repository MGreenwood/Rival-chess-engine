use pyo3::prelude::*;
use pyo3::types::PyDict;
use rival_ai_engine::python_bridge::PyPAGEngine;
use serde_json;

// For now, we'll just re-export the PyPAGEngine from the main engine crate

/// A Python module implemented in Rust.
#[pymodule]
fn rival_ai_py(_py: Python, m: &PyModule) -> PyResult<()> {
    // Re-export the PyPAGEngine from the main engine crate
    m.add_class::<PyPAGEngine>()?;
    Ok(())
}
