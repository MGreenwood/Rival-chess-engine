import argparse
import os

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train the chess AI model')
    
    # Experiment settings
    parser.add_argument('--experiment-name', type=str, required=True,
                      help='Name of the experiment')
    parser.add_argument('--num-epochs', type=int, default=100,
                      help='Number of training epochs')
    parser.add_argument('--save-interval', type=int, default=5,
                      help='Save model every N epochs')
    
    # Training settings
    parser.add_argument('--batch-size', type=int, default=64,
                      help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                      help='Weight decay for regularization')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                      help='Gradient clipping value')
    
    # Model settings
    parser.add_argument('--hidden-dim', type=int, default=256,
                      help='Hidden dimension size')
    parser.add_argument('--num-layers', type=int, default=4,
                      help='Number of GNN layers')
    parser.add_argument('--num-heads', type=int, default=4,
                      help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                      help='Dropout rate')
    
    # Self-play settings
    parser.add_argument('--num-games', type=int, default=1000,
                      help='Number of self-play games per epoch')
    parser.add_argument('--num-simulations', type=int, default=400,
                      help='Number of MCTS simulations per move')
    parser.add_argument('--num-parallel-games', type=int, default=10,
                      help='Number of games to generate in parallel')
    
    # Hardware settings
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use (cuda/cpu)')
    
    return parser.parse_args()

def main():
    # ... existing code ...
    
    # Create self-play config
    self_play_config = SelfPlayConfig(
        num_games=args.num_games,
        num_simulations=args.num_simulations,
        max_moves=100,
        dirichlet_alpha=0.3,
        dirichlet_weight=0.25,
        temperature=1.0,
        c_puct=1.0,
        save_dir=os.path.join(experiment_dir, 'self_play_data'),
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_tqdm=True,
        log_timing=True,
        num_parallel_games=args.num_parallel_games,  # Use command line argument
        prefetch_factor=2
    )
# ... existing code ... 