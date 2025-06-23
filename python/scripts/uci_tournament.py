#!/usr/bin/env python3
"""
UCI Tournament Script for RivalAI

This script orchestrates matches between RivalAI and other UCI engines,
automatically collecting training data and tracking performance.
"""

import os
import sys
import subprocess
import time
import json
import threading
import queue
import signal
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import chess
import chess.engine
import argparse
from dataclasses import dataclass

# Add src to path for unified storage import
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from rival_ai.unified_storage import get_unified_storage, UnifiedGameData, GameSource

@dataclass
class EngineConfig:
    name: str
    path: str
    options: Dict[str, str] = None
    time_limit: float = 5.0  # seconds per move

@dataclass
class TournamentConfig:
    rival_ai_path: str
    opponent_engines: List[EngineConfig]
    games_per_opponent: int = 10
    save_pgn: bool = False  # Disabled - unified storage is sufficient
    save_training_data: bool = True
    output_dir: str = "results/tournaments"

class UCITournament:
    def __init__(self, config: TournamentConfig):
        self.config = config
        self.results = {}
        self.games_played = 0
        self.total_games = len(config.opponent_engines) * config.games_per_opponent * 2  # Both colors
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize unified storage for automatic training data collection
        self.unified_storage = get_unified_storage()
        print(f"🎯 Unified storage initialized - all games will automatically become training data")
        
        # Initialize results tracking
        for engine in config.opponent_engines:
            self.results[engine.name] = {
                'wins': 0,
                'losses': 0,
                'draws': 0,
                'games': [],
                'total_time': 0.0
            }

    def run_tournament(self):
        """Run the complete tournament."""
        print(f"🏆 Starting UCI Tournament: RivalAI vs {len(self.config.opponent_engines)} engines")
        print(f"📊 Total games to play: {self.total_games}")
        print(f"💾 Results will be saved to: {self.output_dir}")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            for engine_config in self.config.opponent_engines:
                print(f"\n🎯 Playing against {engine_config.name}")
                self.play_against_engine(engine_config)
                
        except KeyboardInterrupt:
            print("\n⚠️ Tournament interrupted by user")
        except Exception as e:
            print(f"\n❌ Tournament error: {e}")
        finally:
            # Save final results
            total_time = time.time() - start_time
            self.save_tournament_results(total_time)
            self.print_final_results()

    def play_against_engine(self, opponent_config: EngineConfig):
        """Play a series of games against one engine."""
        # Start engines once and reuse them
        print(f"  🚀 Starting engines (this takes a moment)...")
        rival_ai = self.start_rival_ai()
        opponent = self.start_opponent_engine(opponent_config)
        
        if not rival_ai or not opponent:
            print(f"  ❌ Failed to start engines")
            return
            
        try:
            for game_num in range(self.config.games_per_opponent):
                # Play as white
                print(f"  Game {game_num * 2 + 1}/{self.config.games_per_opponent * 2}: RivalAI (White) vs {opponent_config.name} (Black)")
                result = self.play_single_game_fast(rival_ai, opponent, opponent_config, rival_ai_white=True)
                self.record_result(opponent_config.name, result, True)
                
                # Play as black  
                print(f"  Game {game_num * 2 + 2}/{self.config.games_per_opponent * 2}: {opponent_config.name} (White) vs RivalAI (Black)")
                result = self.play_single_game_fast(rival_ai, opponent, opponent_config, rival_ai_white=False)
                self.record_result(opponent_config.name, result, False)
        finally:
            # Clean up engines at the end
            self.stop_engine(rival_ai)
            self.stop_engine(opponent)

    def play_single_game(self, opponent_config: EngineConfig, rival_ai_white: bool) -> Tuple[str, List[str], float]:
        """
        Play a single game between RivalAI and opponent engine.
        
        Returns:
            Tuple of (result, moves, duration)
            result: "1-0", "0-1", or "1/2-1/2"
        """
        game_start = time.time()
        moves = []
        board = chess.Board()
        
        # Start engines
        rival_ai = self.start_rival_ai()
        opponent = self.start_opponent_engine(opponent_config)
        
        try:
            while not board.is_game_over():
                if (board.turn == chess.WHITE and rival_ai_white) or \
                   (board.turn == chess.BLACK and not rival_ai_white):
                    # RivalAI's turn
                    move = self.get_rival_ai_move(rival_ai, board, opponent_config.time_limit)
                else:
                    # Opponent's turn  
                    move = self.get_opponent_move(opponent, board, opponent_config.time_limit)
                
                if move is None:
                    print(f"    ⚠️ Engine failed to provide move, game aborted")
                    return "1/2-1/2", moves, time.time() - game_start
                
                board.push(move)
                moves.append(move.uci())
                
                # Prevent extremely long games
                if len(moves) > 200:
                    print(f"    ⏰ Game reached move limit, declaring draw")
                    return "1/2-1/2", moves, time.time() - game_start
                    
        except Exception as e:
            print(f"    ❌ Game error: {e}")
            return "1/2-1/2", moves, time.time() - game_start
        finally:
            # Clean up engines
            self.stop_engine(rival_ai)
            self.stop_engine(opponent)
        
        # Determine result
        result = board.result()
        duration = time.time() - game_start
        
        print(f"    ✅ Game completed: {result} ({len(moves)} moves, {duration:.1f}s)")
        
        # Save PGN if requested
        if self.config.save_pgn:
            self.save_game_pgn(board, moves, opponent_config.name, rival_ai_white, result)
        
        return result, moves, duration

    def play_single_game_fast(self, rival_ai, opponent, opponent_config: EngineConfig, rival_ai_white: bool) -> Tuple[str, List[str], float]:
        """
        Play a single game using pre-started engines (FAST VERSION).
        
        Returns:
            Tuple of (result, moves, duration)
            result: "1-0", "0-1", or "1/2-1/2"
        """
        game_start = time.time()
        moves = []
        board = chess.Board()
        
        try:
            # Reset engines for new game
            if hasattr(rival_ai, 'stdin'):
                rival_ai.stdin.write("ucinewgame\n")
                rival_ai.stdin.flush()
            if hasattr(opponent, 'configure'):
                pass  # python-chess engines don't need ucinewgame
            
            while not board.is_game_over():
                if (board.turn == chess.WHITE and rival_ai_white) or \
                   (board.turn == chess.BLACK and not rival_ai_white):
                    # RivalAI's turn
                    move = self.get_rival_ai_move(rival_ai, board, opponent_config.time_limit)
                else:
                    # Opponent's turn  
                    move = self.get_opponent_move(opponent, board, opponent_config.time_limit)
                
                if move is None:
                    print(f"    ⚠️ Engine failed to provide move, game aborted")
                    return "1/2-1/2", moves, time.time() - game_start
                
                board.push(move)
                moves.append(move.uci())
                
                # Prevent extremely long games
                if len(moves) > 200:
                    print(f"    ⏰ Game reached move limit, declaring draw")
                    return "1/2-1/2", moves, time.time() - game_start
                    
        except Exception as e:
            print(f"    ❌ Game error: {e}")
            return "1/2-1/2", moves, time.time() - game_start
        
        # Determine result
        result = board.result()
        duration = time.time() - game_start
        
        print(f"    ✅ Game completed: {result} ({len(moves)} moves, {duration:.1f}s)")
        
        # Save PGN if requested
        if self.config.save_pgn:
            self.save_game_pgn(board, moves, opponent_config.name, rival_ai_white, result)
        
        return result, moves, duration

    def start_rival_ai(self) -> subprocess.Popen:
        """Start RivalAI UCI engine."""
        try:
            cmd = [self.config.rival_ai_path]
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            # Initialize UCI
            process.stdin.write("uci\n")
            process.stdin.flush()
            
            # Wait for uciok (suppress verbose logging for speed)
            while True:
                line = process.stdout.readline().strip()
                if line == "uciok":
                    break
                elif "id name" in line:
                    pass  # Skip logging for speed
            
            # Set ready
            process.stdin.write("isready\n")
            process.stdin.flush()
            
            # Wait for readyok
            while process.stdout.readline().strip() != "readyok":
                pass
                
            return process
            
        except Exception as e:
            print(f"    ❌ Failed to start RivalAI: {e}")
            return None

    def start_opponent_engine(self, config: EngineConfig) -> chess.engine.SimpleEngine:
        """Start opponent UCI engine using python-chess."""
        try:
            engine = chess.engine.SimpleEngine.popen_uci(config.path)
            
            # Configure engine options
            if config.options:
                for option, value in config.options.items():
                    try:
                        engine.configure({option: value})
                    except:
                        pass  # Ignore failed option settings
                        
            return engine
            
        except Exception as e:
            print(f"    ❌ Failed to start {config.name}: {e}")
            return None

    def get_rival_ai_move(self, engine: subprocess.Popen, board: chess.Board, time_limit: float = 5.0) -> Optional[chess.Move]:
        """Get move from RivalAI engine."""
        try:
            # Send position
            position_cmd = f"position fen {board.fen()}\n"
            engine.stdin.write(position_cmd)
            engine.stdin.flush()
            
            # Send go command with proper time limit
            time_ms = int(time_limit * 1000)  # Convert seconds to milliseconds
            engine.stdin.write(f"go movetime {time_ms}\n")
            engine.stdin.flush()
            
            # Read response
            while True:
                line = engine.stdout.readline().strip()
                if line.startswith("bestmove"):
                    move_str = line.split()[1]
                    if move_str == "0000":
                        return None
                    return chess.Move.from_uci(move_str)
                    
        except Exception as e:
            print(f"    ❌ RivalAI move error: {e}")
            return None

    def get_opponent_move(self, engine: chess.engine.SimpleEngine, board: chess.Board, time_limit: float) -> Optional[chess.Move]:
        """Get move from opponent engine."""
        try:
            result = engine.play(board, chess.engine.Limit(time=time_limit))
            return result.move
        except Exception as e:
            print(f"    ❌ Opponent move error: {e}")
            return None

    def stop_engine(self, engine):
        """Safely stop an engine."""
        try:
            if hasattr(engine, 'quit'):  # python-chess engine
                engine.quit()
            elif hasattr(engine, 'stdin'):  # subprocess
                engine.stdin.write("quit\n")
                engine.stdin.flush()
                engine.terminate()
                engine.wait(timeout=5)
        except:
            pass

    def record_result(self, opponent_name: str, result_data: Tuple[str, List[str], float], rival_ai_white: bool):
        """Record game result."""
        result, moves, duration = result_data
        
        # Determine RivalAI result
        if result == "1-0":
            rival_result = "win" if rival_ai_white else "loss"
        elif result == "0-1":
            rival_result = "loss" if rival_ai_white else "win"
        else:
            rival_result = "draw"
        
        # Convert to unified storage format automatically
        try:
            self.store_game_in_unified_storage(moves, result, opponent_name, rival_ai_white)
        except Exception as e:
            print(f"    ⚠️ Failed to store game in unified storage: {e}")
        
        # Update stats
        stats = self.results[opponent_name]
        if rival_result == "win":
            stats['wins'] += 1
        elif rival_result == "loss":
            stats['losses'] += 1
        else:
            stats['draws'] += 1
            
        stats['total_time'] += duration
        stats['games'].append({
            'result': result,
            'rival_ai_white': rival_ai_white,
            'moves': len(moves),
            'duration': duration,
            'timestamp': datetime.now().isoformat()
        })
        
        self.games_played += 1
        
        # Print progress
        wins = stats['wins']
        losses = stats['losses'] 
        draws = stats['draws']
        total = wins + losses + draws
        win_rate = (wins / total * 100) if total > 0 else 0
        
        print(f"    📊 vs {opponent_name}: {wins}W-{losses}L-{draws}D ({win_rate:.1f}% win rate)")
        print(f"    🏆 Tournament progress: {self.games_played}/{self.total_games} games")
        
        # Show unified storage status
        total_stored = self.unified_storage.get_total_games()
        ready_for_training = self.unified_storage.get_training_ready_count()
        print(f"    💾 Unified storage: {total_stored} total games, {ready_for_training} ready for training")

    def store_game_in_unified_storage(self, moves: List[str], result: str, opponent_name: str, rival_ai_white: bool):
        """Convert UCI game to unified storage format and store automatically."""
        # Create positions by replaying the game
        positions = []
        board = chess.Board()
        
        for i, move_uci in enumerate(moves):
            try:
                # Store position before the move
                move = chess.Move.from_uci(move_uci)
                
                # Basic position data (no policy/value data from UCI games)
                position_data = {
                    'fen': board.fen(),
                    'move': move_uci,
                    'value': 0.0,  # No value data available from UCI games
                    'policy': None,  # No policy data available from UCI games
                    'move_number': i + 1,
                    'player': 'rival_ai' if (board.turn == chess.WHITE and rival_ai_white) or (board.turn == chess.BLACK and not rival_ai_white) else opponent_name
                }
                positions.append(position_data)
                
                # Make the move
                board.push(move)
                
            except Exception as e:
                print(f"    ⚠️ Error processing move {i}: {e}")
                break
        
        # Convert result to unified format
        if result == "1-0":
            unified_result = "white_wins"
        elif result == "0-1":
            unified_result = "black_wins"
        else:
            unified_result = "draw"
        
        # Create unified game data
        game_id = f"uci_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.games_played}"
        unified_game = UnifiedGameData(
            game_id=game_id,
            source=GameSource.UCI_TOURNAMENT,
            positions=positions,
            result=unified_result,
            metadata={
                'opponent': opponent_name,
                'rival_ai_white': rival_ai_white,
                'total_moves': len(moves),
                'tournament_game': True,
                'time_control': self.config.opponent_engines[0].time_limit if self.config.opponent_engines else 5.0
            },
            timestamp=datetime.now().isoformat()
        )
        
        # Store in unified storage
        self.unified_storage.store_game(unified_game)

    def save_game_pgn(self, board: chess.Board, moves: List[str], opponent: str, rival_ai_white: bool, result: str):
        """Save game in PGN format."""
        pgn_dir = self.output_dir / "pgn"
        pgn_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_RivalAI_vs_{opponent}.pgn"
        
        white_player = "RivalAI" if rival_ai_white else opponent
        black_player = opponent if rival_ai_white else "RivalAI"
        
        pgn_content = f'''[Event "UCI Tournament"]
[Date "{datetime.now().strftime('%Y.%m.%d')}"]
[White "{white_player}"]
[Black "{black_player}"]
[Result "{result}"]

'''
        
        # Add moves
        board_copy = chess.Board()
        move_pairs = []
        for i, move_uci in enumerate(moves):
            move = chess.Move.from_uci(move_uci)
            san = board_copy.san(move)
            board_copy.push(move)
            
            if i % 2 == 0:
                move_pairs.append(f"{i//2 + 1}. {san}")
            else:
                move_pairs[-1] += f" {san}"
        
        pgn_content += " ".join(move_pairs) + f" {result}\n"
        
        with open(pgn_dir / filename, 'w') as f:
            f.write(pgn_content)

    def save_tournament_results(self, total_time: float):
        """Save tournament results to JSON."""
        results_file = self.output_dir / "tournament_results.json"
        
        summary = {
            'tournament_info': {
                'total_games': self.games_played,
                'total_time': total_time,
                'timestamp': datetime.now().isoformat(),
                'rival_ai_path': self.config.rival_ai_path
            },
            'results': self.results,
            'overall_stats': self.calculate_overall_stats()
        }
        
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n💾 Results saved to {results_file}")

    def calculate_overall_stats(self) -> Dict:
        """Calculate overall tournament statistics."""
        total_wins = sum(r['wins'] for r in self.results.values())
        total_losses = sum(r['losses'] for r in self.results.values())
        total_draws = sum(r['draws'] for r in self.results.values())
        total_games = total_wins + total_losses + total_draws
        
        return {
            'total_wins': total_wins,
            'total_losses': total_losses,
            'total_draws': total_draws,
            'total_games': total_games,
            'win_rate': (total_wins / total_games * 100) if total_games > 0 else 0,
            'score': total_wins + total_draws * 0.5
        }

    def print_final_results(self):
        """Print final tournament results."""
        print("\n" + "=" * 60)
        print("🏆 TOURNAMENT RESULTS")
        print("=" * 60)
        
        overall = self.calculate_overall_stats()
        print(f"📊 Overall: {overall['total_wins']}W-{overall['total_losses']}L-{overall['total_draws']}D")
        print(f"📈 Win Rate: {overall['win_rate']:.1f}%")
        print(f"🎯 Score: {overall['score']:.1f}/{overall['total_games']}")
        print()
        
        # Per-opponent results
        for opponent, stats in self.results.items():
            total = stats['wins'] + stats['losses'] + stats['draws']
            win_rate = (stats['wins'] / total * 100) if total > 0 else 0
            avg_time = stats['total_time'] / total if total > 0 else 0
            
            print(f"vs {opponent:20s}: {stats['wins']:2}W-{stats['losses']:2}L-{stats['draws']:2}D "
                  f"({win_rate:5.1f}%) avg {avg_time:.1f}s/game")
        
        print(f"\n📁 Detailed results and PGN files saved in: {self.output_dir}")
        
        # Show final unified storage status
        total_games = self.unified_storage.get_total_games()
        ready_for_training = self.unified_storage.get_training_ready_count()
        print(f"\n🎯 TRAINING DATA STATUS:")
        print(f"📦 Total games in unified storage: {total_games}")
        print(f"🎓 Games ready for training: {ready_for_training}")
        
        if ready_for_training >= 1000:
            print(f"✅ Sufficient training data available!")
            print(f"🚀 Start server with --enable-training to begin automatic training")
        else:
            print(f"📈 Need {1000 - ready_for_training} more games for training threshold")
            print(f"🔄 Run more tournaments or start server for self-play generation")

def main():
    parser = argparse.ArgumentParser(description="Run UCI tournament for RivalAI")
    parser.add_argument("--rival-ai", default="../engine/target/release/uci", 
                      help="Path to RivalAI UCI binary")
    parser.add_argument("--engines", nargs="+", 
                      help="Paths to opponent UCI engines")
    parser.add_argument("--games", type=int, default=5,
                      help="Games per opponent (each color)")
    parser.add_argument("--time", type=float, default=5.0,
                      help="Time limit per move (seconds)")
    parser.add_argument("--output", default="results/tournaments",
                      help="Output directory for tournament results")
    
    args = parser.parse_args()
    
    if not args.engines:
        # Default engines for testing
        default_engines = [
            EngineConfig("Stockfish", "stockfish", {"Hash": "64", "Threads": "1"}, args.time),
        ]
        
        # Check if engines exist
        available_engines = []
        for engine in default_engines:
            if subprocess.run(["which", engine.path], capture_output=True).returncode == 0:
                available_engines.append(engine)
            else:
                print(f"⚠️ Engine not found: {engine.path}")
        
        if not available_engines:
            print("❌ No engines found. Please install engines or specify with --engines")
            print("   Example: sudo apt install stockfish")
            sys.exit(1)
        
        opponent_engines = available_engines
    else:
        opponent_engines = [
            EngineConfig(f"Engine_{i}", path, time_limit=args.time) 
            for i, path in enumerate(args.engines)
        ]
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output}/{timestamp}_vs_{'_'.join(engine.name for engine in opponent_engines)}"
    
    config = TournamentConfig(
        rival_ai_path=args.rival_ai,
        opponent_engines=opponent_engines,
        games_per_opponent=args.games,
        output_dir=output_dir
    )
    
    tournament = UCITournament(config)
    tournament.run_tournament()

if __name__ == "__main__":
    main() 