#!/usr/bin/env python3
"""
Convert UCI Tournament Games to RivalAI Training Format

This script converts games collected from UCI matches into the format
expected by RivalAI's training pipeline, enabling UCI games to contribute
to model improvement.
"""

import os
import sys
import json
import pickle
import chess
import chess.pgn
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import torch
import numpy as np

# Add RivalAI modules to path
sys.path.append('../python/src')

try:
    from rival_ai.training.training_types import GameRecord
    from rival_ai.chess import GameResult
    from rival_ai.utils.board_conversion import board_to_hetero_data
    from rival_ai.unified_storage import get_unified_storage, GameSource, UnifiedGameData
except ImportError as e:
    print(f"❌ Failed to import RivalAI modules: {e}")
    print("Make sure you're running this from the scripts/ directory")
    sys.exit(1)

class UCIGameConverter:
    """Converts UCI tournament games to RivalAI training format."""
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.storage = get_unified_storage()  # Use unified storage
        
    def convert_tournament_results(self, tournament_dir: str, output_dir: str, cleanup_pgn: bool = True):
        """Convert all games from a tournament results directory."""
        tournament_path = Path(tournament_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        if not tournament_path.exists():
            print(f"❌ Tournament directory not found: {tournament_path}")
            return
        
        # Load tournament results
        results_file = tournament_path / "tournament_results.json"
        if results_file.exists():
            with open(results_file) as f:
                tournament_data = json.load(f)
            self.print_tournament_summary(tournament_data)
        else:
            tournament_data = None
        
        # Convert PGN games
        pgn_dir = tournament_path / "pgn"
        if pgn_dir.exists():
            unified_games = self.convert_pgn_directory_to_unified(pgn_dir)
            
            if unified_games:
                # Store in unified storage
                self.storage.store_multiple_games(unified_games)
                
                print(f"✅ Converted {len(unified_games)} games to unified format")
                print(f"📦 Stored in unified storage system")
                
                # Clean up PGN files after successful conversion
                if cleanup_pgn:
                    self.cleanup_pgn_files(pgn_dir)
                else:
                    print(f"📁 PGN files preserved in: {pgn_dir}")
                
                return len(unified_games)
            else:
                print("❌ No games found to convert")
                return 0
        else:
            print(f"❌ No PGN directory found: {pgn_dir}")
            return 0
    
    def convert_pgn_directory_to_unified(self, pgn_dir: Path) -> List[UnifiedGameData]:
        """Convert all PGN files in a directory to UnifiedGameData."""
        unified_games = []
        
        pgn_files = list(pgn_dir.glob("*.pgn"))
        print(f"🔄 Converting {len(pgn_files)} PGN files to unified format...")
        
        for pgn_file in pgn_files:
            try:
                games = self.convert_pgn_file_to_unified(pgn_file)
                unified_games.extend(games)
                print(f"  ✓ {pgn_file.name}: {len(games)} games")
            except Exception as e:
                print(f"  ❌ {pgn_file.name}: {e}")
        
        return unified_games
    
    def convert_pgn_directory(self, pgn_dir: Path) -> List[GameRecord]:
        """Convert all PGN files in a directory to GameRecords."""
        game_records = []
        
        pgn_files = list(pgn_dir.glob("*.pgn"))
        print(f"🔄 Converting {len(pgn_files)} PGN files...")
        
        for pgn_file in pgn_files:
            try:
                records = self.convert_pgn_file(pgn_file)
                game_records.extend(records)
                print(f"  ✓ {pgn_file.name}: {len(records)} games")
            except Exception as e:
                print(f"  ❌ {pgn_file.name}: {e}")
        
        return game_records
    
    def cleanup_pgn_files(self, pgn_dir: Path):
        """Delete all PGN files after successful conversion."""
        pgn_files = list(pgn_dir.glob("*.pgn"))
        deleted_count = 0
        
        print(f"🧹 Cleaning up {len(pgn_files)} PGN files...")
        
        for pgn_file in pgn_files:
            try:
                pgn_file.unlink()  # Delete the file
                deleted_count += 1
                print(f"  🗑️ Deleted: {pgn_file.name}")
            except Exception as e:
                print(f"  ⚠️ Failed to delete {pgn_file.name}: {e}")
        
        print(f"✅ Cleaned up {deleted_count}/{len(pgn_files)} PGN files")
        
        # Remove the pgn directory if it's empty
        try:
            if not any(pgn_dir.iterdir()):
                pgn_dir.rmdir()
                print(f"📁 Removed empty PGN directory: {pgn_dir}")
        except Exception:
            pass  # Directory not empty or other issue, that's fine
    
    def convert_pgn_file_to_unified(self, pgn_file: Path) -> List[UnifiedGameData]:
        """Convert a single PGN file to UnifiedGameData."""
        unified_games = []
        
        with open(pgn_file) as f:
            game_index = 0
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                
                try:
                    unified_game = self.convert_pgn_game_to_unified(game, f"{pgn_file.stem}_{game_index}")
                    if unified_game:
                        unified_games.append(unified_game)
                        game_index += 1
                except Exception as e:
                    print(f"    ⚠️ Failed to convert game: {e}")
        
        return unified_games
    
    def convert_pgn_file(self, pgn_file: Path) -> List[GameRecord]:
        """Convert a single PGN file to GameRecords."""
        game_records = []
        
        with open(pgn_file) as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                
                try:
                    record = self.convert_game_to_record(game)
                    if record:
                        game_records.append(record)
                except Exception as e:
                    print(f"    ⚠️ Failed to convert game: {e}")
        
        return game_records
    
    def convert_pgn_game_to_unified(self, game: chess.pgn.Game, game_id: str) -> Optional[UnifiedGameData]:
        """Convert a chess.pgn.Game to UnifiedGameData format."""
        try:
            # Extract game information
            headers = game.headers
            white_player = headers.get("White", "Unknown")
            black_player = headers.get("Black", "Unknown")
            result = headers.get("Result", "*")
            
            # Determine if RivalAI was playing and what color
            rival_ai_white = white_player == "RivalAI"
            rival_ai_black = black_player == "RivalAI"
            
            if not (rival_ai_white or rival_ai_black):
                # Skip games where RivalAI wasn't playing
                return None
            
            # Convert result
            if result == "1-0":  # White wins
                game_result = "white_wins"
            elif result == "0-1":  # Black wins
                game_result = "black_wins"
            else:  # Draw or unknown
                game_result = "draw"
            
            # Process moves
            positions = []
            board = chess.Board()
            node = game
            
            while node.variations:
                node = node.variations[0]
                move = node.move
                
                if move is None:
                    break
                
                # Record position
                fen = board.fen()
                move_uci = move.uci()
                
                # Estimate value based on game result
                value = self.estimate_position_value(game_result, len(positions), rival_ai_white)
                
                positions.append({
                    'fen': fen,
                    'move': move_uci,
                    'value': value,
                    'policy': None  # No policy data from PGN
                })
                
                board.push(move)
            
            return UnifiedGameData(
                game_id=game_id,
                source=GameSource.UCI_TOURNAMENT,
                positions=positions,
                result=game_result,
                metadata={
                    'white_player': white_player,
                    'black_player': black_player,
                    'tournament': 'UCI',
                    'rival_ai_color': 'white' if rival_ai_white else 'black'
                },
                timestamp=datetime.now().isoformat()
            )
        except Exception as e:
            print(f"Failed to convert PGN game: {e}")
            return None
    
    def estimate_position_value(self, game_result: str, move_number: int, rival_ai_white: bool) -> float:
        """Estimate position value based on final game result."""
        total_moves_estimate = 40  # Typical game length
        progress = min(move_number / total_moves_estimate, 1.0)
        
        # Determine target value based on result
        if game_result == "white_wins":
            target = 1.0 if rival_ai_white else -1.0
        elif game_result == "black_wins":
            target = -1.0 if rival_ai_white else 1.0
        else:
            target = 0.0
        
        # Interpolate from 0 (uncertain at start) to target (known at end)
        return target * progress
    
    def convert_game_to_record(self, game: chess.pgn.Game) -> Optional[GameRecord]:
        """Convert a chess.pgn.Game to a GameRecord."""
        # Extract game information
        headers = game.headers
        white_player = headers.get("White", "Unknown")
        black_player = headers.get("Black", "Unknown")
        result = headers.get("Result", "*")
        
        # Determine if RivalAI was playing and what color
        rival_ai_white = white_player == "RivalAI"
        rival_ai_black = black_player == "RivalAI"
        
        if not (rival_ai_white or rival_ai_black):
            # Skip games where RivalAI wasn't playing
            return None
        
        # Convert result to GameResult
        game_result = self.parse_game_result(result, rival_ai_white)
        
        # Create GameRecord
        record = GameRecord()
        
        # Process moves
        board = chess.Board()
        node = game
        
        while node.variations:
            node = node.variations[0]
            move = node.move
            
            if move is None:
                break
            
            # Record current position state
            record.add_state(board)
            
            # For training purposes, we need policy and value estimates
            # Since we don't have the actual model predictions from the game,
            # we'll create dummy data that can be filtered out later
            policy = torch.zeros(5312, dtype=torch.float32)  # Dummy policy
            value = self.estimate_value_from_result(game_result, len(record.moves), rival_ai_white, rival_ai_black)
            
            record.add_move(move, policy=policy, value=value)
            board.push(move)
        
        # Set final result
        record.set_result(game_result)
        
        return record
    
    def parse_game_result(self, result: str, rival_ai_white: bool) -> GameResult:
        """Parse PGN result into GameResult from RivalAI's perspective."""
        if result == "1-0":  # White wins
            return GameResult.WHITE_WINS if rival_ai_white else GameResult.BLACK_WINS
        elif result == "0-1":  # Black wins
            return GameResult.BLACK_WINS if rival_ai_white else GameResult.WHITE_WINS
        else:  # Draw or unknown
            return GameResult.DRAW
    
    def estimate_value_from_result(self, game_result: GameResult, move_number: int, 
                                 rival_ai_white: bool, rival_ai_black: bool) -> float:
        """Estimate position value based on final game result."""
        # Simple linear interpolation from game start to final result
        total_moves_estimate = 40  # Typical game length
        progress = min(move_number / total_moves_estimate, 1.0)
        
        # Determine target value based on result
        if game_result == GameResult.WHITE_WINS:
            target = 1.0 if rival_ai_white else -1.0
        elif game_result == GameResult.BLACK_WINS:
            target = -1.0 if rival_ai_white else 1.0
        else:
            target = 0.0
        
        # Interpolate from 0 (uncertain at start) to target (known at end)
        return target * progress
    
    def print_tournament_summary(self, tournament_data: Dict[str, Any]):
        """Print a summary of the tournament results."""
        print("\n" + "=" * 50)
        print("🏆 TOURNAMENT SUMMARY")
        print("=" * 50)
        
        info = tournament_data.get('tournament_info', {})
        overall = tournament_data.get('overall_stats', {})
        
        print(f"📅 Date: {info.get('timestamp', 'Unknown')}")
        print(f"🎮 Total Games: {info.get('total_games', 0)}")
        print(f"⏱️  Total Time: {info.get('total_time', 0):.1f} seconds")
        print(f"📊 Win Rate: {overall.get('win_rate', 0):.1f}%")
        print(f"🎯 Score: {overall.get('score', 0):.1f}/{overall.get('total_games', 0)}")
        
        print("\nPer-opponent results:")
        results = tournament_data.get('results', {})
        for opponent, stats in results.items():
            wins = stats.get('wins', 0)
            losses = stats.get('losses', 0)
            draws = stats.get('draws', 0)
            total = wins + losses + draws
            win_rate = (wins / total * 100) if total > 0 else 0
            
            print(f"  vs {opponent:15s}: {wins:2}W-{losses:2}L-{draws:2}D ({win_rate:5.1f}%)")
        
        print("=" * 50)

def main():
    parser = argparse.ArgumentParser(description="Convert UCI tournament games to training format")
    parser.add_argument("tournament_dir", help="Tournament results directory")
    parser.add_argument("--output", default="converted_training_games", 
                      help="Output directory for converted games")
    parser.add_argument("--device", default="auto",
                      help="Device for tensor operations (cuda/cpu/auto)")
    parser.add_argument("--keep-pgn", action="store_true",
                      help="Keep PGN files after conversion (default: delete them)")
    
    args = parser.parse_args()
    
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    converter = UCIGameConverter(device=device)
    
    print(f"🔄 Converting UCI games from: {args.tournament_dir}")
    print(f"📱 Using device: {device}")
    
    num_converted = converter.convert_tournament_results(
        args.tournament_dir, 
        args.output,
        cleanup_pgn=not args.keep_pgn
    )
    
    if num_converted > 0:
        print(f"\n✅ Successfully converted {num_converted} games!")
        print("📦 Games stored in unified storage format")
        print("🚀 Will be automatically included in next training session.")
        
        # Suggest next steps
        print("\n📋 Next Steps:")
        print("1. Games are now ready for unified training")
        print("2. Server will auto-train when enough games accumulate")
        print("3. Monitor training progress via server stats")
        print("4. Run more UCI tournaments to collect additional data")
    else:
        print("\n❌ No games were converted. Check the tournament directory.")

if __name__ == "__main__":
    main() 