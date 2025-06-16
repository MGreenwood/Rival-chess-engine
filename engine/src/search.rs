use std::time::Duration;
use chess::{ChessMove, Board, MoveGen, Piece, Color, BoardStatus};

pub struct SearchInfo {
    pub nodes_searched: u64,
    pub time_taken: Duration,
    pub best_move: Option<ChessMove>,
    pub score: f32,
}

pub fn search(board: &Board, max_depth: u32, _max_time: Duration) -> SearchInfo {
    let mut info = SearchInfo {
        nodes_searched: 0,
        time_taken: Duration::from_secs(0),
        best_move: None,
        score: 0.0,
    };

    let moves: Vec<ChessMove> = MoveGen::new_legal(board).collect();
    if moves.is_empty() {
        return info;
    }

    let mut best_score = f32::NEG_INFINITY;
    let mut best_move = moves[0];

    for &mv in &moves {
        let next_board = *board;
        let mut temp_board = Board::default();
        next_board.make_move(mv, &mut temp_board);
        let score = -alpha_beta(&temp_board, max_depth - 1, f32::NEG_INFINITY, f32::INFINITY, &mut info);
        
        if score > best_score {
            best_score = score;
            best_move = mv;
        }
    }

    info.best_move = Some(best_move);
    info.score = best_score;
    info
}

fn alpha_beta(
    board: &Board,
    depth: u32,
    mut alpha: f32,
    beta: f32,
    info: &mut SearchInfo,
) -> f32 {
    info.nodes_searched += 1;

    if depth == 0 || matches!(board.status(), BoardStatus::Checkmate | BoardStatus::Stalemate) {
        return evaluate_position(board);
    }

    let moves: Vec<ChessMove> = MoveGen::new_legal(board).collect();
    if moves.is_empty() {
        return evaluate_position(board);
    }

    for &mv in &moves {
        let next_board = *board;
        let mut temp_board = Board::default();
        next_board.make_move(mv, &mut temp_board);
        let score = -alpha_beta(&temp_board, depth - 1, -beta, -alpha, info);
        
        if score >= beta {
            return beta;
        }
        alpha = alpha.max(score);
    }

    alpha
}

fn evaluate_position(board: &Board) -> f32 {
    // Simple material evaluation for now
    let mut score = 0.0;
    for square in chess::ALL_SQUARES {
        if let Some(piece) = board.piece_on(square) {
            let value = match piece {
                Piece::Pawn => 1.0,
                Piece::Knight => 3.0,
                Piece::Bishop => 3.2,
                Piece::Rook => 5.0,
                Piece::Queen => 9.0,
                Piece::King => 0.0,
            };
            if board.color_on(square).unwrap() == Color::White {
                score += value;
            } else {
                score -= value;
            }
        }
    }
    score
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alpha_beta_search() {
        let board = Board::new();
        
        let search_info = search(
            &board,
            4,
            Duration::from_secs(1),
        );
        
        assert!(search_info.best_move.is_some());
    }
} 