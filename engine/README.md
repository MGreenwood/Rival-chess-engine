# RivalAI Chess Engine

A chess engine implementation using Positional Adjacency Graphs (PAG) for move generation and position evaluation.

## Features

- Modern board representation using Positional Adjacency Graphs
- Monte Carlo Tree Search (MCTS) for move selection
- Alpha-beta pruning search implementation
- Position evaluation with piece-square tables
- Efficient move generation using graph traversal

## Building

```bash
cargo build --release
```

## Testing

```bash
cargo test
```

## Project Structure

- `src/board.rs` - Chess board representation and move generation
- `src/pag.rs` - Positional Adjacency Graph implementation
- `src/engine.rs` - Main engine logic with MCTS
- `src/evaluation.rs` - Position evaluation functions
- `src/search.rs` - Alpha-beta search implementation

## Usage

```rust
use rival_ai::{Board, Engine};

fn main() {
    let board = Board::new();
    let engine = Engine::new(4, 1000); // depth 4, 1 second per move
    
    if let Some((from, to)) = engine.find_best_move(&board) {
        println!("Best move: {:?} -> {:?}", from, to);
    }
}
```

## License

MIT License 