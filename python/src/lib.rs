use pyo3::prelude::*;
use rival_ai::{Board, PositionalAdjacencyGraph, pag::{PAG, NodeType, EdgeType}};
use serde_json;

#[pyclass]
struct PyPAG {
    pag: PAG,
}

#[pymethods]
impl PyPAG {
    #[new]
    fn new(board: &Board) -> PyResult<Self> {
        let pag = PAG::from_board(board);
        Ok(Self { pag })
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.pag)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    fn get_node_count(&self) -> usize {
        self.pag.node_count()
    }

    fn get_edge_count(&self) -> usize {
        self.pag.edge_count()
    }

    fn get_piece_nodes(&self) -> PyResult<Vec<PyObject>> {
        let mut nodes = Vec::new();
        for (coord, id) in &self.pag.piece_positions {
            if let Some(NodeType::Piece(piece)) = self.pag.get_node(*id) {
                let node_dict = Python::with_gil(|py| {
                    let dict = PyDict::new(py);
                    dict.set_item("id", piece.get_id())?;
                    dict.set_item("type", format!("{:?}", piece.get_type()))?;
                    dict.set_item("color", format!("{:?}", piece.get_color()))?;
                    dict.set_item("coordinate", (coord.0, coord.1))?;
                    dict.set_item("material_value", piece.get_material_value())?;
                    dict.set_item("mobility_score", piece.get_mobility_score())?;
                    dict.set_item("is_attacked", piece.is_attacked())?;
                    dict.set_item("is_defended", piece.is_defended())?;
                    dict.set_item("is_king_shield", piece.is_king_shield())?;
                    Ok(dict.into_py(py))
                })?;
                nodes.push(node_dict);
            }
        }
        Ok(nodes)
    }

    fn get_critical_square_nodes(&self) -> PyResult<Vec<PyObject>> {
        let mut nodes = Vec::new();
        for (coord, id) in &self.pag.critical_squares {
            if let Some(NodeType::CriticalSquare(square)) = self.pag.get_node(*id) {
                let node_dict = Python::with_gil(|py| {
                    let dict = PyDict::new(py);
                    dict.set_item("id", square.get_id())?;
                    dict.set_item("type", format!("{:?}", square.get_type()))?;
                    dict.set_item("coordinate", (coord.0, coord.1))?;
                    dict.set_item("importance_score", square.get_importance_score())?;
                    dict.set_item("control_status", square.get_control_status())?;
                    Ok(dict.into_py(py))
                })?;
                nodes.push(node_dict);
            }
        }
        Ok(nodes)
    }

    fn get_edges_for_node(&self, node_id: u64) -> PyResult<Vec<PyObject>> {
        let mut edges = Vec::new();
        for edge in self.pag.get_edges_for_node(node_id) {
            let edge_dict = Python::with_gil(|py| {
                let dict = PyDict::new(py);
                dict.set_item("source", edge.source_id())?;
                dict.set_item("target", edge.target_id())?;
                dict.set_item("type", format!("{:?}", edge.get_type()))?;
                dict.set_item("weight", edge.get_weight())?;
                dict.set_item("features", format!("{:?}", edge.get_features()))?;
                Ok(dict.into_py(py))
            })?;
            edges.push(edge_dict);
        }
        Ok(edges)
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn rival_ai_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyPAG>()?;
    Ok(())
}
