#![allow(non_local_definitions)]
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::exceptions::PyValueError;
use numpy::PyArray2;

use crate::pag::{DensePAGBuilder, PAG, create_dense_pag_from_fen, NodeType};

/// Python wrapper for the ultra-dense PAG system
#[pyclass]
pub struct PyDensePAG {
    #[allow(dead_code)]  // Reserved for future ultra-dense PAG functionality
    pag: PAG,
    #[allow(dead_code)]  // Reserved for future ultra-dense PAG functionality
    builder: DensePAGBuilder,
}

/// Python wrapper for PAG construction statistics
#[pyclass]
#[derive(Clone)]
pub struct PyPAGStats {
    #[pyo3(get)]
    pub node_count: usize,
    #[pyo3(get)]
    pub edge_count: usize,
    #[pyo3(get)]
    pub piece_count: usize,
    #[pyo3(get)]
    pub critical_square_count: usize,
    #[pyo3(get)]
    pub total_feature_dimensions: usize,
}

/// High-level Python interface for ultra-dense PAG
#[pyclass]
pub struct PyPAGEngine {
    builder: DensePAGBuilder,
}

#[pymethods]
impl PyPAGEngine {
    #[new]
    fn new() -> Self {
        Self {
            builder: DensePAGBuilder::new(),
        }
    }
    
    /// Create ultra-dense PAG from FEN string
    /// Returns: Dictionary with node features, edge features, and metadata
    fn fen_to_dense_pag(&mut self, fen: &str) -> PyResult<PyObject> {
        let pag = self.builder.build_from_fen(fen)
            .map_err(|e| PyValueError::new_err(format!("PAG construction failed: {}", e)))?;
        
        Python::with_gil(|py| {
            let result = PyDict::new(py);
            
            // Extract node features
            let (node_features, node_types, node_ids) = self.extract_node_features(&pag, py)?;
            result.set_item("node_features", node_features)?;
            result.set_item("node_types", node_types)?;
            result.set_item("node_ids", node_ids)?;
            
            // Extract edge features
            let (edge_features, edge_indices, edge_types) = self.extract_edge_features(&pag, py)?;
            result.set_item("edge_features", edge_features)?;
            result.set_item("edge_indices", edge_indices)?;
            result.set_item("edge_types", edge_types)?;
            
            // Add metadata
            let stats = pag.get_stats();
            let meta = PyDict::new(py);
            meta.set_item("node_count", stats.node_count)?;
            meta.set_item("edge_count", stats.edge_count)?;
            meta.set_item("piece_count", stats.piece_count)?;
            meta.set_item("critical_square_count", stats.critical_square_count)?;
            meta.set_item("board_size", stats.board_size)?;
            result.set_item("metadata", meta)?;
            
            Ok(result.into())
        })
    }
    
    /// Batch process multiple FENs for training efficiency
    /// Returns: List of PAG dictionaries
    fn batch_fen_to_dense_pag(&mut self, fens: Vec<&str>) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let results = PyList::empty(py);
            
            for fen in fens {
                match self.fen_to_dense_pag(fen) {
                    Ok(pag_dict) => results.append(pag_dict)?,
                    Err(e) => {
                        // Log error but continue processing other FENs
                        eprintln!("Failed to process FEN {}: {}", fen, e);
                        continue;
                    }
                }
            }
            
            Ok(results.into())
        })
    }
    
    /// Get feature dimensions for model architecture planning
    fn get_feature_dimensions(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let dims = PyDict::new(py);
            dims.set_item("piece_features", 308)?; // From DensePiece::feature_count()
            dims.set_item("critical_square_features", 95)?; // From DenseCriticalSquare::feature_count()
            dims.set_item("edge_features", 158)?; // From DenseEdge::feature_count()
            dims.set_item("max_pieces", 32)?; // Max pieces on board
            dims.set_item("max_critical_squares", 64)?; // Max critical squares
            dims.set_item("max_edges", 2048)?; // Rough estimate for max edges
            Ok(dims.into())
        })
    }
    
    /// Convert PAG to PyTorch Geometric format
    fn pag_to_hetero_data(&mut self, fen: &str) -> PyResult<PyObject> {
        let pag = self.builder.build_from_fen(fen)
            .map_err(|e| PyValueError::new_err(format!("PAG construction failed: {}", e)))?;
        
        Python::with_gil(|py| {
            // This would create a format compatible with PyTorch Geometric
            let hetero_data = PyDict::new(py);
            
            // Piece nodes
            let piece_data = PyDict::new(py);
            let (piece_features, piece_types, piece_ids) = self.extract_piece_node_features(&pag, py)?;
            piece_data.set_item("x", piece_features)?;
            piece_data.set_item("node_ids", piece_ids)?;
            hetero_data.set_item("piece", piece_data)?;
            
            // Critical square nodes  
            let square_data = PyDict::new(py);
            let (square_features, square_types, square_ids) = self.extract_square_node_features(&pag, py)?;
            square_data.set_item("x", square_features)?;
            square_data.set_item("node_ids", square_ids)?;
            hetero_data.set_item("critical_square", square_data)?;
            
            // Edge types
            let edge_types = vec!["piece_to_piece", "piece_to_square", "square_to_square"];
            for edge_type in edge_types {
                let (edge_features, edge_indices) = self.extract_typed_edges(&pag, edge_type, py)?;
                let edge_data = PyDict::new(py);
                edge_data.set_item("edge_index", edge_indices)?;
                edge_data.set_item("edge_attr", edge_features)?;
                hetero_data.set_item(edge_type, edge_data)?;
            }
            
            Ok(hetero_data.into())
        })
    }
    
    /// Performance benchmark for speed testing
    fn benchmark_construction(&mut self, fen: &str, iterations: usize) -> PyResult<f64> {
        let start = std::time::Instant::now();
        
        for _ in 0..iterations {
            let _ = self.builder.build_from_fen(fen)
                .map_err(|e| PyValueError::new_err(format!("Benchmark failed: {}", e)))?;
        }
        
        let duration = start.elapsed();
        Ok(duration.as_secs_f64() / iterations as f64)
    }
}

impl PyPAGEngine {
    /// Extract node features as numpy arrays
    fn extract_node_features(&self, pag: &PAG, py: Python) -> PyResult<(PyObject, PyObject, PyObject)> {
        let mut all_features = Vec::new();
        let mut node_types = Vec::new();
        let mut node_ids = Vec::new();
        
        // Get all piece IDs and square IDs
        let piece_ids = pag.get_piece_ids();
        let square_ids = pag.get_critical_square_ids();
        
        // Process piece nodes FIRST (308 dimensions)
        for &piece_id in &piece_ids {
            if let Some(node) = pag.get_node(piece_id) {
                if let NodeType::DensePiece(_) = node {
                    let features = node.to_feature_vector();
                    assert_eq!(features.len(), 308, "Piece features should be 308 dimensions");
                    all_features.push(features);
                    node_types.push("piece".to_string());
                    node_ids.push(piece_id);
                }
            }
        }
        
        // Process critical square nodes SECOND (95 dimensions)
        for &square_id in &square_ids {
            if let Some(node) = pag.get_node(square_id) {
                if let NodeType::DenseCriticalSquare(_) = node {
                    let features = node.to_feature_vector();
                    assert_eq!(features.len(), 95, "Square features should be 95 dimensions");
                    all_features.push(features);
                    node_types.push("critical_square".to_string());
                    node_ids.push(square_id);
                }
            }
        }
        
        // FIXED: Don't pad to max dimensions - keep original dimensions
        // Convert each feature vector directly to maintain correct dimensions
        let features_list = PyList::empty(py);
        for features in all_features {
            let feature_array = numpy::PyArray1::from_vec(py, features);
            features_list.append(feature_array)?;
        }
        
        let types_list = PyList::new(py, &node_types);
        let ids_list = PyList::new(py, &node_ids);
        
        Ok((features_list.into(), types_list.into(), ids_list.into()))
    }
    
    /// Extract edge features as numpy arrays
    fn extract_edge_features(&self, pag: &PAG, py: Python) -> PyResult<(PyObject, PyObject, PyObject)> {
        let mut all_edge_features = Vec::new();
        let mut edge_indices = Vec::new();
        let mut edge_types = Vec::new();
        
        // Get all piece and square IDs for edge iteration
        let piece_ids = pag.get_piece_ids();
        let square_ids = pag.get_critical_square_ids();
        let mut all_ids = piece_ids.clone();
        all_ids.extend(square_ids);
        
        // Extract edges for each node
        for &node_id in &all_ids {
            let edges = pag.get_edges_for_node(node_id);
            for edge in edges {
                if edge.is_dense() {
                    let features = edge.to_feature_vector();
                    all_edge_features.push(features);
                    edge_indices.push([edge.source_id() as i64, edge.target_id() as i64]);
                    edge_types.push("dense".to_string());
                }
            }
        }
        
        // Convert to numpy arrays
        let max_edge_features = all_edge_features.iter().map(|f| f.len()).max().unwrap_or(0);
        let mut edge_feature_matrix = vec![vec![0.0f32; max_edge_features]; all_edge_features.len()];
        
        for (i, features) in all_edge_features.iter().enumerate() {
            for (j, &value) in features.iter().enumerate() {
                if j < max_edge_features {
                    edge_feature_matrix[i][j] = value;
                }
            }
        }
        
        let edge_features_array = PyArray2::from_vec2(py, &edge_feature_matrix)?.to_owned();
        let edge_indices_vec2: Vec<Vec<i64>> = edge_indices.into_iter()
            .map(|arr| vec![arr[0], arr[1]])
            .collect();
        let edge_indices_array = PyArray2::from_vec2(py, &edge_indices_vec2)?.to_owned();
        let edge_types_list = PyList::new(py, &edge_types);
        
        Ok((edge_features_array.into(), edge_indices_array.into(), edge_types_list.into()))
    }
    
    /// Extract piece node features specifically
    fn extract_piece_node_features(&self, pag: &PAG, py: Python) -> PyResult<(PyObject, PyObject, PyObject)> {
        let mut piece_features = Vec::new();
        let mut piece_types = Vec::new();
        let mut piece_ids = Vec::new();
        
        for &piece_id in &pag.get_piece_ids() {
            if let Some(node) = pag.get_node(piece_id) {
                if let NodeType::DensePiece(_) = node {
                    let features = node.to_feature_vector();
                    piece_features.push(features);
                    piece_types.push("piece".to_string());
                    piece_ids.push(piece_id);
                }
            }
        }
        
        let max_features = piece_features.iter().map(|f| f.len()).max().unwrap_or(308);
        let mut feature_matrix = vec![vec![0.0f32; max_features]; piece_features.len()];
        
        for (i, features) in piece_features.iter().enumerate() {
            for (j, &value) in features.iter().enumerate() {
                if j < max_features {
                    feature_matrix[i][j] = value;
                }
            }
        }
        
        let features_array = PyArray2::from_vec2(py, &feature_matrix)?.to_owned();
        let types_list = PyList::new(py, &piece_types);
        let ids_list = PyList::new(py, &piece_ids);
        
        Ok((features_array.into(), types_list.into(), ids_list.into()))
    }
    
    /// Extract critical square node features specifically
    fn extract_square_node_features(&self, pag: &PAG, py: Python) -> PyResult<(PyObject, PyObject, PyObject)> {
        let mut square_features = Vec::new();
        let mut square_types = Vec::new();
        let mut square_ids = Vec::new();
        
        for &square_id in &pag.get_critical_square_ids() {
            if let Some(node) = pag.get_node(square_id) {
                if let NodeType::DenseCriticalSquare(_) = node {
                    let features = node.to_feature_vector();
                    square_features.push(features);
                    square_types.push("critical_square".to_string());
                    square_ids.push(square_id);
                }
            }
        }
        
        let max_features = square_features.iter().map(|f| f.len()).max().unwrap_or(95);
        let mut feature_matrix = vec![vec![0.0f32; max_features]; square_features.len()];
        
        for (i, features) in square_features.iter().enumerate() {
            for (j, &value) in features.iter().enumerate() {
                if j < max_features {
                    feature_matrix[i][j] = value;
                }
            }
        }
        
        let features_array = PyArray2::from_vec2(py, &feature_matrix)?.to_owned();
        let types_list = PyList::new(py, &square_types);
        let ids_list = PyList::new(py, &square_ids);
        
        Ok((features_array.into(), types_list.into(), ids_list.into()))
    }
    
    /// Extract specific edge types
    fn extract_typed_edges(&self, _pag: &PAG, _edge_type: &str, py: Python) -> PyResult<(PyObject, PyObject)> {
        // For now, return empty arrays - would need more sophisticated edge typing
        let empty_features: Vec<Vec<f32>> = Vec::new();
        let empty_indices: Vec<Vec<i64>> = Vec::new();
        
        let features_array = PyArray2::from_vec2(py, &empty_features)?.to_owned();
        let indices_array = PyArray2::from_vec2(py, &empty_indices)?.to_owned();
        
        Ok((features_array.into(), indices_array.into()))
    }
}

/// Convenience functions for Python module
#[pyfunction]
fn create_pag_from_fen(fen: &str) -> PyResult<PyDensePAG> {
    let pag = create_dense_pag_from_fen(fen)
        .map_err(|e| PyValueError::new_err(format!("PAG creation failed: {}", e)))?;
    
    Ok(PyDensePAG {
        pag,
        builder: DensePAGBuilder::new(),
    })
}

#[pyfunction]
fn get_pag_feature_info() -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let info = PyDict::new(py);
        info.set_item("piece_feature_count", 308)?;
        info.set_item("critical_square_feature_count", 95)?;
        info.set_item("edge_feature_count", 158)?;
        info.set_item("total_possible_features", 308 * 32 + 95 * 64 + 158 * 2048)?; // Rough estimate
        info.set_item("description", "Ultra-dense PAG with comprehensive chess analysis")?;
        Ok(info.into())
    })
}

/// Python module definition
#[pymodule]
fn rival_ai_pag(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyPAGEngine>()?;
    m.add_class::<PyDensePAG>()?;
    m.add_class::<PyPAGStats>()?;
    m.add_function(wrap_pyfunction!(create_pag_from_fen, m)?)?;
    m.add_function(wrap_pyfunction!(get_pag_feature_info, m)?)?;
    Ok(())
} 