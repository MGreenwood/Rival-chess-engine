use rival_ai::{Board, Engine};
use std::time::Duration;

fn main() {
    let mut board = Board::new();
    let engine = Engine::new(4, 1000); // depth 4, 1 second per move
    
    println!("Starting a new game!");
    println!("\nInitial position:");
    println!("{}", board);
    
    // Play 5 moves
    for move_number in 1..=5 {
        println!("\nMove {}", move_number);
        
        if let Some((from, to)) = engine.find_best_move(&board) {
            let (from_rank, from_file) = from;
            let (to_rank, to_file) = to;
            
            println!(
                "Engine moves from ({}, {}) to ({}, {})",
                from_rank + 1, (b'a' + from_file as u8) as char,
                to_rank + 1, (b'a' + to_file as u8) as char
            );
            
            board.make_move(from, to);
            println!("\nBoard after move {}:", move_number);
            println!("{}", board);
        } else {
            println!("No legal moves available!");
            break;
        }
        
        // Add a small delay between moves for better visualization
        std::thread::sleep(Duration::from_millis(500));
    }
    
    println!("\nGame finished!");
} 