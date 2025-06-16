use rival_ai::{Board, PositionalAdjacencyGraph};
use std::time::Instant;

fn main() {
    let iterations = 1000;
    let board = Board::new();
    
    println!("Running PAG construction benchmark...");
    println!("Iterations: {}", iterations);
    
    let start = Instant::now();
    
    for _ in 0..iterations {
        let _pag = PositionalAdjacencyGraph::from_board(&board);
    }
    
    let elapsed = start.elapsed();
    let avg_time = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
    
    println!("\nResults:");
    println!("Total time: {:.2?}", elapsed);
    println!("Average time per position: {:.3} ms", avg_time);
    println!("Positions per second: {:.0}", 1000.0 / avg_time);
} 