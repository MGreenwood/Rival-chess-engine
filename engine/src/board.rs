use chess::{Board as ChessBoard, ChessMove, Square, Piece, Color, BoardStatus, BitBoard, MoveGen};
use std::str::FromStr;
use std::fmt;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::exceptions::PyValueError;
use crate::pag::{PAG, node::{PieceType, Color as PAGColor}};
use anyhow::Result;

#[derive(Clone)]
pub struct Board {
    inner: ChessBoard,
}

impl Board {
    pub fn new() -> Self {
        Self {
            inner: ChessBoard::default(),
        }
    }

    pub fn from_fen(fen: &str) -> Result<Self, String> {
        ChessBoard::from_str(fen)
            .map(|board| Self { inner: board })
            .map_err(|e| e.to_string())
    }

    pub fn legal(&self, mv: ChessMove) -> bool {
        self.inner.legal(mv)
    }

    pub fn make_move(&self, mv: ChessMove) -> Self {
        Self {
            inner: self.inner.make_move_new(mv),
        }
    }

    pub fn get_legal_moves(&self) -> MoveGen {
        MoveGen::new_legal(&self.inner)
    }

    pub fn get_piece_at(&self, square: Square) -> Option<Piece> {
        self.inner.piece_on(square)
    }

    pub fn get_hash(&self) -> u64 {
        self.inner.get_hash()
    }

    pub fn is_game_over(&self) -> bool {
        self.inner.status() != BoardStatus::Ongoing
    }

    pub fn is_check(&self) -> bool {
        self.inner.checkers().popcnt() > 0
    }

    pub fn is_draw(&self) -> bool {
        match self.inner.status() {
            BoardStatus::Stalemate => true,
            _ => false,
        }
    }

    pub fn is_mate(&self) -> bool {
        self.inner.status() == BoardStatus::Checkmate
    }

    pub fn get_color_at(&self, square: Square) -> Option<Color> {
        self.inner.color_on(square)
    }

    pub fn pinned(&self, square: Square) -> bool {
        let pinned = self.inner.pinned();
        let square_bb = BitBoard::from_square(square);
        (pinned & square_bb) != BitBoard(0)
    }

    pub fn checkers(&self) -> usize {
        self.inner.checkers().popcnt() as usize
    }

    pub fn to_fen(&self) -> String {
        self.inner.to_string()
    }

    pub fn status(&self) -> BoardStatus {
        self.inner.status()
    }

    pub fn side_to_move(&self) -> Color {
        self.inner.side_to_move()
    }

    pub fn as_chess_board(&self) -> &ChessBoard {
        &self.inner
    }

    pub fn from_chess_board(board: ChessBoard) -> Self {
        Self { inner: board }
    }
}

impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.inner)
    }
}

impl From<ChessBoard> for Board {
    fn from(board: ChessBoard) -> Self {
        Self { inner: board }
    }
}

impl AsRef<ChessBoard> for Board {
    fn as_ref(&self) -> &ChessBoard {
        &self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_board_creation() {
        let board = Board::new();
        assert!(!board.is_game_over());
    }

    #[test]
    fn test_fen_parsing() {
        let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        let board = Board::from_fen(fen).unwrap();
        assert_eq!(board.to_fen(), fen);
    }

    #[test]
    fn test_make_move() {
        let mut board = Board::new();
        let mv = ChessMove::new(
            Square::make_square(chess::Rank::Second, chess::File::E),
            Square::make_square(chess::Rank::Fourth, chess::File::E),
            None,
        );
        assert!(board.legal(mv));
        board = board.make_move(mv);
        assert_eq!(
            board.to_fen(),
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        );
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PAGBoard {
    board: ChessBoard,
    pag: PAG,
}

#[pymethods]
impl PAGBoard {
    #[new]
    pub fn new() -> Self {
        let board = ChessBoard::default();
        let pag = PAG::new(8); // Standard chess board size
        Self { board, pag }
    }

    pub fn from_fen(&mut self, fen: String) -> PyResult<()> {
        match ChessBoard::from_str(&fen) {
            Ok(board) => {
                self.board = board;
                Ok(())
            }
            Err(e) => Err(PyValueError::new_err(format!("Invalid FEN: {}", e))),
        }
    }

    pub fn to_py(&self, py: Python) -> PyObject {
        // Convert to Python dictionary
        let dict = PyDict::new(py);
        dict.set_item("fen", self.board.to_string()).unwrap();
        dict.into_py(py)
    }
}

impl PAGBoard {
    pub fn from_board(board: &ChessBoard) -> Self {
        let pag = PAG::new(8);
        Self { board: board.clone(), pag }
    }

    pub fn get_board(&self) -> &ChessBoard {
        &self.board
    }

    pub fn get_pag(&self) -> &PAG {
        &self.pag
    }

    fn _chess_to_pag_color(color: Color) -> PAGColor {
        match color {
            Color::White => PAGColor::White,
            Color::Black => PAGColor::Black,
        }
    }

    fn _chess_to_pag_piece_type(piece: Piece) -> PieceType {
        match piece {
            Piece::Pawn => PieceType::Pawn,
            Piece::Knight => PieceType::Knight,
            Piece::Bishop => PieceType::Bishop,
            Piece::Rook => PieceType::Rook,
            Piece::Queen => PieceType::Queen,
            Piece::King => PieceType::King,
        }
    }
} 