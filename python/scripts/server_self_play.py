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
from rival_ai.unified_storage import get_unified_storage, initialize_unified_storage, GameSource, UnifiedGameData

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
        
        # Initialize unified storage with the path provided by the server
        # Use the --save-dir argument instead of calculating our own path
        logger.info(f"🗂️ Using training games directory from server: {self.save_dir.absolute()}")
        logger.info(f"🔍 Directory exists: {self.save_dir.exists()}")
        
        self.storage = initialize_unified_storage(str(self.save_dir), batch_size=1000)  # Use consistent batch size with training system
        
        # Debug the storage initialization
        logger.info(f"📦 Storage initialized:")
        logger.info(f"   📁 Base dir: {self.storage.base_dir}")
        logger.info(f"   📁 Unified dir: {self.storage.unified_dir}")
        logger.info(f"   📊 Batch size: {self.storage.batch_size}")
        logger.info(f"   🔢 Current batch: {len(self.storage._current_batch)} games")
        logger.info(f"   🎯 Next batch number: {self.storage._batch_number}")
        
        # Check existing batch files
        existing_batches = list(self.storage.unified_dir.glob("batch_*.json.gz"))
        logger.info(f"🔍 Existing batch files: {len(existing_batches)}")
        
        logger.info(f"🎯 Unified Self-Play System Initializing...")
        logger.info(f"   Device: {device}")
        logger.info(f"   Output: Unified batched storage")
        
        # Load model with ULTRA-DENSE PAG support - FIXED: Match training script configuration
        self.model = ChessGNN(
            hidden_dim=256,
            num_layers=10,   # FIXED: Match training script's 10 layers
            num_heads=4,
            dropout=0.1,
            use_ultra_dense_pag=True,  # Enable ultra-dense PAG features
            piece_dim=350,  # Ultra-dense piece features from Rust PAG
            critical_square_dim=95  # Ultra-dense critical square features
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
            logger.info(f"🎯 Model ready for self-play with ULTRA-DENSE PAG support!")
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
        logger.info(f"   Model layers: 10 (deep architecture)")
        logger.info(f"   Ultra-dense PAG features: ENABLED")
        
    def generate_games(self, num_games, training_threshold=5000):
        logger.info(f"🎮 Starting generation of {num_games} self-play games")
        logger.info(f"📦 Output: Unified storage format")
        
        try:
            # Count existing games before generation
            total_games_before = self._count_total_games()
            
            # Generate games but don't save them as PKL files
            games = self.self_play.generate_games(num_games=num_games, save_games=False)
            logger.info(f"✅ Successfully generated {len(games)} games")
            
            # Convert to unified format and store
            unified_games = []
            for i, game_record in enumerate(games):
                unified_game = self._convert_to_unified_format(game_record, f"selfplay_{i}")
                if unified_game:
                    unified_games.append(unified_game)
            
            # Store in unified storage with guaranteed persistence
            if unified_games:
                try:
                    logger.info(f"💾 Storing {len(unified_games)} games with guaranteed persistence...")
                    logger.info(f"🗂️ Storage directory: {self.storage.unified_dir}")
                    logger.info(f"📊 Current batch size before: {len(self.storage._current_batch)}")
                    
                    # Store all games at once (efficient)
                    self.storage.store_multiple_games(unified_games)
                    logger.info(f"📊 Current batch size after: {len(self.storage._current_batch)}")
                    
                    # GUARANTEED SAVE: Force save current batch regardless of size
                    if self.storage._current_batch:
                        logger.info(f"💾 Force saving {len(self.storage._current_batch)} games to disk NOW...")
                        self.storage.force_save_current_batch()
                        logger.info(f"✅ Batch saved! Current batch size: {len(self.storage._current_batch)}")
                    
                    logger.info(f"📦 Successfully persisted all {len(unified_games)} games to disk")
                    
                    # Verify batch files on disk  
                    batch_files = list(self.storage.unified_dir.glob("batch_*.json.gz"))
                    logger.info(f"🔍 Batch files on disk: {len(batch_files)} files")
                    if batch_files:
                        latest_batch = max(batch_files, key=lambda x: x.stat().st_mtime)
                        logger.info(f"   📄 Latest: {latest_batch.name}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to store games in unified storage: {e}")
                    import traceback
                    logger.error(f"💥 Traceback: {traceback.format_exc()}")
                    raise
            
            # Count total games after generation
            total_games_after = self._count_total_games()
            
            # Calculate actual games generated (should match len(unified_games))
            games_generated_this_round = len(unified_games)
            
            # Debug logging
            logger.info(f"🔍 Debug: before={total_games_before}, generated={games_generated_this_round}, after={total_games_after}")
            
            logger.info(f"🎯 Games ready for unified training")
            return games_generated_this_round, total_games_after, training_threshold
        except Exception as e:
            logger.error(f"❌ Error generating games: {e}")
            raise
    
    def _count_total_games(self):
        """Count total games in unified storage"""
        try:
            return self.storage.get_total_games()
        except Exception as e:
            logger.warning(f"Could not count total games: {e}")
            return 0
    
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
                        
                        # Handle MoveData objects (which contain embedded move, policy, and value)
                        if hasattr(move, 'move') and hasattr(move, 'policy'):
                            # This is a MoveData object - extract the embedded data
                            move_str = move.move.uci() if hasattr(move.move, 'uci') else str(move.move)
                            # Use the policy from MoveData if available
                            if move.policy is not None:
                                policy = move.policy
                            # Use the value from MoveData if available and current value is default
                            if hasattr(move, 'value') and value == 0.0:
                                value = move.value
                        elif hasattr(move, 'uci'):
                            move_str = move.uci()
                        else:
                            move_str = str(move)
                        
                        # Convert value to float
                        if hasattr(value, 'item'):
                            value_float = float(value.item())
                        else:
                            value_float = float(value)
                        
                        # Convert policy to list if it's a tensor
                        policy_list = None
                        if policy is not None:
                            if hasattr(policy, 'tolist'):
                                policy_list = policy.tolist()
                            elif hasattr(policy, 'cpu'):
                                policy_list = policy.cpu().tolist()
                            elif isinstance(policy, (list, tuple)):
                                policy_list = list(policy)
                            else:
                                logger.warning(f"Unknown policy type: {type(policy)}")
                        
                        positions.append({
                            'fen': fen,
                            'move': move_str,
                            'value': value_float,
                            'policy': policy_list
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
    parser.add_argument('--training-threshold', type=int, default=5000, help='Number of games needed before training starts')
    
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
    
    # Log what the server requested
    logger.info(f"🎯 Server requested: {args.num_games} games (target scaling number)")
    
    games_generated, total_games, training_threshold = runner.generate_games(args.num_games, args.training_threshold)
    
    # Log the difference between server request and actual generation
    logger.info(f"📊 Requested: {args.num_games} games, Generated: {games_generated} games, Total: {total_games}")
    
    # Calculate progress toward training
    progress_pct = min(100, (total_games / training_threshold) * 100)
    games_needed = max(0, training_threshold - total_games)
    
    if total_games >= training_threshold:
        status = "READY FOR TRAINING"
    else:
        status = f"{games_needed} more needed"
    
    # Show the actual number of games generated this round (not the server's target request)
    print(f"Generated {games_generated} games ({total_games}/{training_threshold} total, {progress_pct:.1f}% - {status})")  # Output for Rust server to capture
    
if __name__ == '__main__':
    main() 