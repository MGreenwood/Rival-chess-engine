"""
Generate self-play games for training.
"""

import os
import json
import time
import chess
import argparse
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from rival_ai.mcts import MCTS
from rival_ai.config import MCTSConfig
import numpy as np

def generate_game(config: MCTSConfig, experiment_name: str, run_dir: str, game_num: int) -> dict:
    """Generate a single self-play game."""
    board = chess.Board()
    positions = []
    start_time = time.time()
    
    # Create MCTS instance
    mcts = MCTS(config)
    
    while not board.is_game_over():
        # Get action probabilities
        policy, value = mcts.get_action_policy(board)
        
        # Sample move from policy dictionary
        moves = list(policy.keys())
        probs = list(policy.values())
        selected_move = np.random.choice(moves, p=probs)
        
        # Record position
        positions.append({
            'fen': board.fen(),
            'move': selected_move.uci(),
            'value': float(value)
        })
        
        # Make move
        board.make_move(selected_move)
    
    # Create experiment directory structure
    experiment_dir = Path('experiments') / experiment_name
    data_dir = Path(run_dir) / 'self_play_data'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Save game in run directory
    game_file = data_dir / f'positions_game_{game_num}.json'
    with open(game_file, 'w') as f:
        json.dump(positions, f)
    
    return {
        'num_positions': len(positions),
        'time_taken': time.time() - start_time
    }

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Generate self-play games for training')
    parser.add_argument('--num-games', type=int, default=100, help='Number of games to generate')
    parser.add_argument('--num-simulations', type=int, default=50, help='Number of MCTS simulations per move')
    parser.add_argument('--experiment-name', type=str, required=True, help='Name of the experiment')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for move selection')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint')
    args = parser.parse_args()
    
    # Create config with timing logs disabled
    config = MCTSConfig(
        num_simulations=args.num_simulations,
        temperature=args.temperature,
        log_timing=False  # Disable timing logs
    )
    
    # Create experiment directory and timestamped run directory
    experiment_dir = Path('experiments') / args.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = experiment_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate games
    total_positions = 0
    total_time = 0
    
    print(f"\nGenerating {args.num_games} games for experiment {args.experiment_name}")
    print(f"Run directory: {run_dir}")
    for game_num in tqdm(range(1, args.num_games + 1), desc="Games"):
        result = generate_game(config, args.experiment_name, str(run_dir), game_num)
        total_positions += result['num_positions']
        total_time += result['time_taken']
    
    # Print summary
    print(f"\nGenerated {total_positions} positions in {total_time:.1f} seconds")
    print(f"Average speed: {total_positions/total_time:.1f} positions/second")
    print(f"Games saved in: {run_dir}/self_play_data/")

if __name__ == '__main__':
    main() 