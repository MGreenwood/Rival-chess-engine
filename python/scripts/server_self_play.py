#!/usr/bin/env python3
"""
Server Self-Play Generation Script
Called by the Rust server to generate self-play games.
Updated to use Ultra-Dense PAG Feature Extraction System.
"""

import sys
import os
import argparse
from pathlib import Path
import logging
from datetime import datetime
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the src directory to the path
script_dir = Path(__file__).parent
python_src = script_dir.parent / 'src'
sys.path.insert(0, str(python_src))

# Import required modules
from rival_ai.models import ChessGNN
from rival_ai.training.self_play import SelfPlay, SelfPlayConfig
from rival_ai.utils.board_conversion import board_to_hetero_data
from rival_ai.unified_storage import get_unified_storage, GameSource, UnifiedGameData

# Try to import PAG engine for validation (AFTER logger is defined)
try:
    import rival_ai_engine as engine
    PAG_ENGINE_AVAILABLE = True
    logger.info("✅ PAG engine module available")
except ImportError as e:
    PAG_ENGINE_AVAILABLE = False
    logger.warning(f"⚠️ PAG engine not available: {e}")
    logger.info("🔄 Will use Python-only PAG fallback mode")

class UnifiedSelfPlayRunner:
    def __init__(self, model_path, save_dir, device='cuda', num_simulations=600):
        self.model_path = model_path
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.num_simulations = num_simulations
        
        # Initialize unified storage
        self.storage = get_unified_storage()
        
        logger.info(f"🎯 Unified Self-Play System Initializing...")
        logger.info(f"   Device: {device}")
        logger.info(f"   Output: Unified batched storage")
        
        # Load model with existing ChessGNN interface (will be upgraded later)
        self.model = ChessGNN(
            hidden_dim=256,  # Use existing interface for now
            num_layers=4,    # Use existing interface for now
            num_heads=4,     # Use existing interface for now
            dropout=0.1      # Use existing interface for now
        )
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                # Load existing model weights
                state_dict = checkpoint['model_state_dict']
                # Check if this has ultra-dense features (future compatibility)
                if any('dense_pag' in key for key in state_dict.keys()):
                    logger.info(f"✅ Detected ultra-dense PAG model (future feature)")
                    # For now, just load what we can
                    self.model.load_state_dict(state_dict, strict=False)
                else:
                    logger.info(f"✅ Loading existing model weights")
                    self.model.load_state_dict(state_dict)
            else:
                self.model.load_state_dict(checkpoint)
                logger.info(f"✅ Loaded checkpoint directly")
                
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"🎯 Model ready for self-play (ultra-dense features coming soon)")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
            
        # Create self-play config optimized for current system
        config = SelfPlayConfig(
            num_games=0,  # Will be set dynamically
            num_simulations=self.num_simulations,  # Configurable speed vs quality
            max_moves=150,
            device=self.device,
            batch_size=16,  # Smaller batch for memory efficiency
            num_workers=1,  # Reduced for stability
            save_dir=str(self.save_dir),
            use_tqdm=False,
            c_puct=2.5,
            dirichlet_alpha=0.8,
            dirichlet_weight=0.4,
            temperature=1.5,
            opening_temperature=1.8,
            midgame_temperature=1.5,
            endgame_temperature=1.2,
            repetition_penalty=5.0,
            random_move_probability=0.08,  # Reduced for more focused play
            forward_progress_bonus=0.5,
            use_dense_pag=PAG_ENGINE_AVAILABLE,  # Enable PAG if engine is available
            pag_fallback_to_python=True,  # Fallback to Python PAG if Rust fails
        )
        
        # Create self-play instance
        self.self_play = SelfPlay(self.model, config)
        
        logger.info(f"🚀 Self-Play configuration:")
        logger.info(f"   Simulations per move: {config.num_simulations}")
        logger.info(f"   Batch size (memory-optimized): {config.batch_size}")
        logger.info(f"   Ultra-dense feature extraction: PREPARING")
        
    def generate_games(self, num_games):
        logger.info(f"🎮 Starting generation of {num_games} self-play games")
        logger.info(f"📦 Output: Unified storage format")
        
        try:
            # Generate games but don't save them as PKL files
            games = self.self_play.generate_games(num_games=num_games, save_games=False)
            logger.info(f"✅ Successfully generated {len(games)} games")
            
            # Convert to unified format and store
            unified_games = []
            for i, game_record in enumerate(games):
                unified_game = self._convert_to_unified_format(game_record, f"selfplay_{i}")
                if unified_game:
                    unified_games.append(unified_game)
            
            # Store in unified storage
            if unified_games:
                self.storage.store_multiple_games(unified_games)
                logger.info(f"📦 Stored {len(unified_games)} games in unified format")
            
            logger.info(f"🎯 Games ready for unified training")
            return len(games)
        except Exception as e:
            logger.error(f"❌ Error generating games: {e}")
            raise
    
    def _convert_to_unified_format(self, game_record, game_id):
        """Convert a GameRecord to UnifiedGameData format"""
        try:
            positions = []
            
            # Extract positions from game record
            if hasattr(game_record, 'states') and hasattr(game_record, 'moves'):
                states = game_record.states
                moves = game_record.moves
                values = getattr(game_record, 'values', [0.0] * len(moves))
                policies = getattr(game_record, 'policies', [None] * len(moves))
                
                for i in range(min(len(states), len(moves))):
                    try:
                        state = states[i]
                        move = moves[i]
                        value = values[i] if i < len(values) else 0.0
                        policy = policies[i] if i < len(policies) else None
                        
                        # Convert to consistent format
                        if hasattr(state, 'fen'):
                            fen = state.fen()
                        else:
                            fen = str(state)
                        
                        if hasattr(move, 'uci'):
                            move_str = move.uci()
                        else:
                            move_str = str(move)
                        
                        # Convert value to float
                        if hasattr(value, 'item'):
                            value_float = float(value.item())
                        else:
                            value_float = float(value)
                        
                        positions.append({
                            'fen': fen,
                            'move': move_str,
                            'value': value_float,
                            'policy': policy.tolist() if hasattr(policy, 'tolist') else None
                        })
                    except Exception as e:
                        logger.warning(f"Failed to convert position {i}: {e}")
                        continue
            
            # Determine result
            result = "draw"
            if hasattr(game_record, 'result'):
                result_obj = game_record.result
                result_str = str(result_obj).lower()
                if 'white' in result_str and 'win' in result_str:
                    result = "white_wins"
                elif 'black' in result_str and 'win' in result_str:
                    result = "black_wins"
            
            return UnifiedGameData(
                game_id=game_id,
                source=GameSource.SELF_PLAY,
                positions=positions,
                result=result,
                metadata={'generated_by': 'server_self_play'},
                timestamp=datetime.now().isoformat()
            )
        except Exception as e:
            logger.error(f"Failed to convert game record: {e}")
            return None

def main():
    parser = argparse.ArgumentParser(description='Unified Server Self-Play Generator')
    parser.add_argument('--model-path', required=True, help='Path to model checkpoint')
    parser.add_argument('--save-dir', required=True, help='Directory to save games (used for unified storage base)')
    parser.add_argument('--num-games', type=int, required=True, help='Number of games to generate')
    parser.add_argument('--num-simulations', type=int, default=600, help='MCTS simulations per move (lower = faster)')
    parser.add_argument('--device', default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--low-priority', action='store_true', help='Run with lower GPU priority to not block community engine')
    
    args = parser.parse_args()
    
    # Set lower GPU priority if requested
    if args.low_priority:
        logger.info("🛡️ Running in low-priority mode to protect community engine")
        import os
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
        try:
            import psutil
            p = psutil.Process()
            p.nice(10)  # Lower CPU priority
        except:
            pass
    
    runner = UnifiedSelfPlayRunner(args.model_path, args.save_dir, args.device, args.num_simulations)
    games_generated = runner.generate_games(args.num_games)
    
    print(f"Generated {games_generated} games (unified storage)")  # Output for Rust server to capture
    
if __name__ == '__main__':
    main() 