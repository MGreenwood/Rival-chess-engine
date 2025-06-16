import argparse
import logging
import sys
from pathlib import Path
import chess
import torch
from rival_ai.models.gnn import ChessGNN
from rival_ai.mcts import MCTS, MCTSConfig
from rival_ai.utils.board_conversion import board_to_hetero_data
from rival_ai.training.trainer import Trainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Play against RivalAI')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--num-simulations', type=int, default=100,
                      help='Number of MCTS simulations per move')
    parser.add_argument('--temperature', type=float, default=0.1,
                      help='Temperature for move selection (lower = more deterministic)')
    return parser.parse_args()

def load_model(checkpoint_path):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    model = ChessGNN()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def get_move_from_user(board):
    """Get a move from the user."""
    while True:
        try:
            move_str = input("Enter your move (e.g., 'e2e4'): ").strip()
            if move_str.lower() == 'quit':
                sys.exit(0)
            move = chess.Move.from_uci(move_str)
            if move in board.legal_moves:
                return move
            print("Illegal move. Try again.")
        except ValueError:
            print("Invalid move format. Use format 'e2e4'.")
        except KeyboardInterrupt:
            sys.exit(0)

def get_model_move(board, model, mcts, temperature):
    """Get a move from the model using MCTS."""
    with torch.no_grad():
        # Convert board to HeteroData
        data = board_to_hetero_data(board)
        data = data.to(model.device)
        
        # Run MCTS
        root = mcts.search(board, model)
        
        # Select move based on visit counts and temperature
        if temperature == 0:
            # Deterministic: choose move with most visits
            move = max(root.children.items(), key=lambda x: x[1].visit_count)[0]
        else:
            # Probabilistic: sample based on visit counts
            moves = list(root.children.keys())
            visit_counts = torch.tensor([root.children[m].visit_count for m in moves])
            probs = (visit_counts ** (1/temperature)).float()
            probs = probs / probs.sum()
            move_idx = torch.multinomial(probs, 1).item()
            move = moves[move_idx]
        
        return move

def main():
    args = parse_args()
    
    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize MCTS
    mcts_config = MCTSConfig(
        num_simulations=args.num_simulations,
        c_puct=1.0,
        dirichlet_alpha=0.3,
        dirichlet_weight=0.25
    )
    mcts = MCTS(mcts_config)
    
    # Initialize board
    board = chess.Board()
    
    # Game loop
    print("\nWelcome to RivalAI Chess!")
    print("You are playing as White. Enter moves in UCI format (e.g., 'e2e4').")
    print("Type 'quit' to exit.\n")
    
    while not board.is_game_over():
        print("\n" + str(board))
        print(f"\nFEN: {board.fen()}")
        
        if board.turn == chess.WHITE:
            # Human's turn
            move = get_move_from_user(board)
        else:
            # Model's turn
            print("\nRivalAI is thinking...")
            move = get_model_move(board, model, mcts, args.temperature)
            print(f"RivalAI plays: {move.uci()}")
        
        board.push(move)
        
        # Check game over conditions
        if board.is_checkmate():
            winner = "White" if board.turn == chess.BLACK else "RivalAI"
            print(f"\nCheckmate! {winner} wins!")
        elif board.is_stalemate():
            print("\nStalemate! The game is a draw.")
        elif board.is_insufficient_material():
            print("\nInsufficient material! The game is a draw.")
        elif board.is_fifty_moves():
            print("\nFifty-move rule! The game is a draw.")
        elif board.is_repetition():
            print("\nThreefold repetition! The game is a draw.")

if __name__ == '__main__':
    main() 