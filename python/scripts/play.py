import argparse
import logging
import sys
from pathlib import Path
import chess
import torch
import numpy as np

# Add the src directory to the path
script_dir = Path(__file__).parent
python_src = script_dir.parent / 'src'
sys.path.insert(0, str(python_src))

from rival_ai.models import ChessGNN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Play against RivalAI')
    parser.add_argument('--checkpoint', type=str, default='../../models/latest_trained_model.pt',
                      help='Path to model checkpoint')
    parser.add_argument('--fen', type=str, default=None,
                      help='Start from specific FEN position (for testing scenarios)')
    parser.add_argument('--temperature', type=float, default=0.1,
                      help='Temperature for move selection (lower = more deterministic)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to use (cuda/cpu)')
    parser.add_argument('--debug-promotions', action='store_true',
                      help='Show detailed debug info for promotion moves')
    return parser.parse_args()

def load_unified_model(checkpoint_path, device):
    """Load the unified model with the same interface as the server."""
    logger.info(f"Loading unified model from {checkpoint_path}")
    
    try:
        # Create the exact same model wrapper as the server uses
        # This provides the predict_with_board interface that connects to Rust PAG
        
        class UltraDenseModelWrapper:
            def __init__(self):
                print("🧠 Initializing Model Wrapper (same as server)...")
                self.use_fallback = False
                
                try:
                    # Load model the same way as server
                    config = {
                        'hidden_dim': 256,
                        'num_layers': 4,
                        'num_heads': 4, 
                        'dropout': 0.1
                    }
                    
                    self.model = ChessGNN(**config)
                    
                    # Load weights
                    checkpoint = torch.load(checkpoint_path, map_location=device)
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    else:
                        state_dict = checkpoint
                    
                    missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
                    
                    if missing_keys:
                        print(f"🔄 Missing keys: {len(missing_keys)}")
                    if unexpected_keys:
                        print(f"⚠️ Unexpected keys: {len(unexpected_keys)}")
                    
                    self.model.eval()
                    self.device = device
                    print("✅ Model loaded successfully")
                    
                except Exception as e:
                    print(f"❌ Failed to load model: {e}")
                    print("🔄 Using fallback uniform policy")
                    self.use_fallback = True
                    self.device = 'cpu'
            
            def predict_with_board(self, board_fen):
                """
                Predict using the same interface as the server.
                The server calls this method and the Rust bridge handles
                converting the board FEN to ultra-dense PAG features.
                
                For this Python script, we'll use the existing board_to_model_input
                until the full Rust PAG bridge is available.
                """
                if self.use_fallback:
                    # Simple fallback prediction
                    return ([1.0/5312] * 5312, 0.0)
                
                try:
                    # Convert FEN to board and then to model input
                    # (In the server, this happens in Rust with ultra-dense PAG)
                    board = chess.Board(board_fen)
                    board_tensor = board_to_model_input(board)
                    
                    # Debug the tensor format
                    print(f"🔍 Model input debug:")
                    if hasattr(board_tensor, 'x'):
                        print(f"   Hetero data - x shape: {board_tensor.x.shape if hasattr(board_tensor.x, 'shape') else type(board_tensor.x)}")
                        print(f"   Hetero data - edge_index shape: {board_tensor.edge_index.shape if hasattr(board_tensor, 'edge_index') and hasattr(board_tensor.edge_index, 'shape') else 'No edge_index'}")
                    else:
                        print(f"   Tensor type: {type(board_tensor)}")
                        if hasattr(board_tensor, 'shape'):
                            print(f"   Tensor shape: {board_tensor.shape}")
                        if hasattr(board_tensor, 'dtype'):
                            print(f"   Tensor dtype: {board_tensor.dtype}")
                    
                    # Run model inference
                    with torch.no_grad():
                        policy_logits, value = self.model(board_tensor)
                        
                        # Debug model outputs
                        print(f"🔍 Model output debug:")
                        print(f"   Policy logits shape: {policy_logits.shape}")
                        print(f"   Policy logits range: [{policy_logits.min().item():.6f}, {policy_logits.max().item():.6f}]")
                        print(f"   Policy logits mean: {policy_logits.mean().item():.6f}")
                        print(f"   Value: {value.item() if hasattr(value, 'item') else value}")
                        
                        # Convert to probabilities and extract values
                        policy = torch.softmax(policy_logits, dim=-1).squeeze().cpu().numpy()
                        value = value.item()
                        
                        # Debug final policy
                        print(f"   Policy probs shape: {policy.shape}")
                        print(f"   Policy probs range: [{policy.min():.6f}, {policy.max():.6f}]")
                        print(f"   Policy probs sum: {policy.sum():.6f}")
                        print(f"   Non-zero entries: {(policy > 1e-10).sum()}/{len(policy)}")
                    
                    # Check for promotion positions (reduced debug)
                    has_promotion = (
                        (board.turn == chess.WHITE and any(board.piece_at(sq) == chess.Piece(chess.PAWN, chess.WHITE) 
                                                          and chess.square_rank(sq) == 6 for sq in chess.SQUARES)) or
                        (board.turn == chess.BLACK and any(board.piece_at(sq) == chess.Piece(chess.PAWN, chess.BLACK) 
                                                          and chess.square_rank(sq) == 1 for sq in chess.SQUARES))
                    )
                    
                    if has_promotion:
                        # Just show basic promotion stats
                        promotion_probs = policy[4096:5312]
                        max_promotion_prob = promotion_probs.max()
                        avg_promotion_prob = promotion_probs.mean()
                        print(f"🔥 Promotion detected: max={max_promotion_prob:.6f}, avg={avg_promotion_prob:.6f}")
                    
                    return policy.tolist(), value
                    
                except Exception as e:
                    print(f"⚠️ Prediction failed: {e}")
                    print(f"   Board FEN: {board_fen}")
                    import traceback
                    traceback.print_exc()
                    # Ultimate fallback
                    return ([1.0/5312] * 5312, 0.0)
            
            def eval(self):
                if not self.use_fallback and hasattr(self, 'model'):
                    self.model.eval()
            
            def to(self, device):
                self.device = device
                if not self.use_fallback and hasattr(self, 'model'):
                    self.model = self.model.to(device)
                    print(f"🔧 Model moved to {device}")
                return self
        
        # Create the wrapper (same as server creates)
        wrapper = UltraDenseModelWrapper()
        
        logger.info("✅ Created unified model wrapper with Rust PAG interface")
        return wrapper
        
    except Exception as e:
        logger.error(f"❌ Failed to load unified model: {e}")
        return None

def move_to_policy_idx(mv):
    """Convert chess move to policy index with CORRECTED promotion encoding."""
    from_sq = mv.from_square
    to_sq = mv.to_square
    promotion = mv.promotion
    
    # Debug: Print square indices for promotion moves
    if promotion:
        print(f"🔍 DEBUG move_to_policy_idx: {mv.uci()}")
        print(f"   from_square: {chess.square_name(from_sq)} = {from_sq}")
        print(f"   to_square: {chess.square_name(to_sq)} = {to_sq}")
        print(f"   promotion: {chess.piece_name(promotion)}")
    
    if promotion:
        # 🔥 FIXED PROMOTION ENCODING (matches the corrected Rust bridge) 🔥
        # Use the same compact encoding that fits in the 1216 available slots
        
        piece_type = {
            chess.KNIGHT: 0,  # 0-based for compact encoding
            chess.BISHOP: 1, 
            chess.ROOK: 2,
            chess.QUEEN: 3
        }.get(promotion, 3)  # Default to queen
        
        # Extract file and rank info
        from_file = from_sq % 8
        from_rank = from_sq // 8
        to_file = to_sq % 8
        to_rank = to_sq // 8
        
        # Determine promotion direction
        if to_file == from_file:
            direction = 0  # Straight promotion
        elif to_file == from_file - 1:
            direction = 1  # Capture left
        elif to_file == from_file + 1:
            direction = 2  # Capture right
        else:
            # Invalid promotion - shouldn't happen
            print(f"   ❌ INVALID PROMOTION: file change {from_file} -> {to_file}")
            return 4096  # fallback
        
        # Determine side (White or Black promotion)
        if from_rank == 6 and to_rank == 7:  # White promotion
            side_offset = 0
        elif from_rank == 1 and to_rank == 0:  # Black promotion  
            side_offset = 96
        else:
            print(f"   ❌ INVALID PROMOTION RANKS: {from_rank} -> {to_rank}")
            return 4096  # fallback
        
        # Compact index calculation (same as fixed Rust bridge)
        # Each file gets 12 indices (3 directions * 4 pieces)
        # Format: 4096 + side_offset + (file * 12) + (direction * 4) + piece_type
        index = 4096 + side_offset + (from_file * 12) + (direction * 4) + piece_type
        
        print(f"   from_file: {from_file}, to_file: {to_file}")
        print(f"   from_rank: {from_rank}, to_rank: {to_rank}")
        print(f"   direction: {direction} ({'straight' if direction == 0 else 'capture_left' if direction == 1 else 'capture_right'})")
        print(f"   piece_type: {piece_type}")
        print(f"   side_offset: {side_offset} ({'white' if side_offset == 0 else 'black'})")
        print(f"   index: 4096 + {side_offset} + ({from_file} * 12) + ({direction} * 4) + {piece_type}")
        print(f"         = 4096 + {side_offset} + {from_file * 12} + {direction * 4} + {piece_type}")
        print(f"         = {index}")
        print(f"   ✅ Index {index} is {'VALID' if index < 5312 else 'OUT OF BOUNDS'} (max 5311)")
        
        return index
    else:
        # Regular moves: from * 64 + to
        return from_sq * 64 + to_sq

def get_move_from_user(board):
    """Get a move from the user."""
    while True:
        try:
            print(f"\nLegal moves: {[move.uci() for move in board.legal_moves]}")
            move_str = input("Enter your move (e.g., 'e2e4' or 'e7e8q' for promotion): ").strip()
            if move_str.lower() == 'quit':
                sys.exit(0)
            if move_str.lower() == 'help':
                print("Move format: 'e2e4' for normal moves, 'e7e8q' for pawn promotion to queen")
                print("Promotion pieces: q=queen, r=rook, b=bishop, n=knight")
                continue
            
            move = chess.Move.from_uci(move_str)
            if move in board.legal_moves:
                return move
            print("❌ Illegal move. Try again.")
        except ValueError:
            print("❌ Invalid move format. Use 'e2e4' or 'e7e8q' (type 'help' for more info)")
        except KeyboardInterrupt:
            sys.exit(0)

def get_model_move(board, model, temperature, debug_promotions=False):
    """Get a move from the model using direct policy prediction."""
    with torch.no_grad():
        # Get model prediction using the same method as the server
        try:
            # Use the model's predict_with_board method (same as server)
            # This automatically uses the Rust PAG implementation via the bridge
            board_fen = board.fen()
            policy, value = model.predict_with_board(board_fen)
            
        except Exception as e:
            logger.error(f"Model prediction failed: {e}")
            # Fallback to random legal move
            import random
            return random.choice(list(board.legal_moves))
        
        # Collect legal moves and their probabilities
        moves = []
        probs = []
        promotion_moves = []
        promotion_probs = []
        
        for move in board.legal_moves:
            try:
                idx = move_to_policy_idx(move)
                if idx < len(policy):
                    prob = max(0.0, float(policy[idx]))
                    moves.append(move)
                    probs.append(prob)
                    
                    # Track promotions for debugging
                    if move.promotion:
                        promotion_moves.append(move)
                        promotion_probs.append(prob)
            except:
                continue
        
        if not moves:
            # Fallback to random legal move
            import random
            return random.choice(list(board.legal_moves))
        
        # Debug promotion moves if requested (reduced verbosity)
        if debug_promotions and promotion_moves:
            max_promo_prob = max(promotion_probs) if promotion_probs else 0.0
            print(f"🔥 {len(promotion_moves)} promotions available (max prob: {max_promo_prob:.6f})")
            if max_promo_prob < 0.01:
                print("   ⚠️ Model shows low promotion understanding")
        
        # Apply temperature and select move
        if temperature == 0.0:
            # Deterministic: choose highest probability
            best_idx = max(range(len(probs)), key=lambda i: probs[i])
            selected_move = moves[best_idx]
        else:
            # Probabilistic sampling with temperature
            import numpy as np
            probs_array = np.array(probs)
            if probs_array.sum() > 0:
                probs_array = probs_array / probs_array.sum()  # Normalize
                probs_array = np.power(probs_array, 1.0 / temperature)
                probs_array = probs_array / probs_array.sum()  # Re-normalize
                choice_idx = np.random.choice(len(moves), p=probs_array)
                selected_move = moves[choice_idx]
            else:
                selected_move = moves[0]
        
        return selected_move

def board_to_model_input(board):
    """Convert board to model input format (same as training system)."""
    try:
        # Try to use the same board conversion as the training system
        from rival_ai.utils.board_conversion import board_to_hetero_data
        print("✅ Using board_to_hetero_data (proper conversion)")
        return board_to_hetero_data(board)
    except ImportError as e:
        print(f"⚠️ Could not import board_to_hetero_data: {e}")
        print("🚨 USING FALLBACK - This will cause promotion issues!")
        # Fallback: just return the FEN string for now
        return board.fen()

def main():
    args = parse_args()
    
    # Load unified model
    model = load_unified_model(args.checkpoint, args.device)
    if model is None:
        logger.error("Failed to load model")
        return
    
    # Initialize board
    if args.fen:
        try:
            board = chess.Board(args.fen)
            print(f"🎯 Starting from position: {args.fen}")
        except ValueError:
            print(f"❌ Invalid FEN: {args.fen}")
            return
    else:
        board = chess.Board()
        print("🎮 Starting new game from initial position")
    
    # Game loop
    print("\n🎯 RivalAI Chess - Unified Model Test!")
    print("You can play as either color. Enter moves in UCI format (e.g., 'e2e4', 'e7e8q').")
    print("Type 'quit' to exit, 'help' for move format help.\n")
    
    if args.debug_promotions:
        print("🔥 PROMOTION DEBUG MODE: Will show detailed info when promotion moves are available\n")
    
    # Quick promotion test scenarios
    print("🔥 Quick test positions:")
    print("  Promotion test: --fen '8/P7/8/8/8/8/8/8 w - - 0 1'")
    print("  Near promotion: --fen 'rnbqkbnr/pppppppP/8/8/8/8/PPPPPP1P/RNBQKBNR w KQkq - 0 1'")
    print("  Endgame: --fen '8/2P5/8/8/8/8/5ppp/6k1 w - - 0 1'\n")
    
    while not board.is_game_over():
        print("\n" + str(board))
        print(f"FEN: {board.fen()}")
        
        # Check for promotions available
        promotion_moves = [move for move in board.legal_moves if move.promotion]
        if promotion_moves:
            print(f"🔥 {len(promotion_moves)} PROMOTION MOVES AVAILABLE!")
        
        color_name = "White" if board.turn == chess.WHITE else "Black"
        print(f"\n{color_name} to move:")
        
        if board.turn == chess.WHITE:
            # Human's turn (you can change this to test AI vs AI)
            move = get_move_from_user(board)
        else:
            # Model's turn
            print("🤖 RivalAI is thinking...")
            move = get_model_move(board, model, args.temperature, args.debug_promotions)
            print(f"🤖 RivalAI plays: {move.uci()}")
            
            if move.promotion:
                piece_name = chess.piece_name(move.promotion)
                print(f"   ⭐ PROMOTED TO {piece_name.upper()}!")
        
        board.push(move)
        
        # Check game over conditions
        if board.is_checkmate():
            winner = "White" if board.turn == chess.BLACK else "Black"
            print(f"\n🏁 Checkmate! {winner} wins!")
        elif board.is_stalemate():
            print("\n🏁 Stalemate! The game is a draw.")
        elif board.is_insufficient_material():
            print("\n🏁 Insufficient material! The game is a draw.")
        elif board.is_fifty_moves():
            print("\n🏁 Fifty-move rule! The game is a draw.")
        elif board.is_repetition():
            print("\n🏁 Threefold repetition! The game is a draw.")

if __name__ == '__main__':
    main() 