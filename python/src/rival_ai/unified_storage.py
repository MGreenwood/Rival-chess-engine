#!/usr/bin/env python3
"""
Unified Game Storage System
Handles all game types (self-play, tournament, single-player, community) in a unified format.
"""

import json
import gzip
import pickle
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
import logging
import threading
from dataclasses import dataclass, asdict
import chess
from enum import Enum

logger = logging.getLogger(__name__)

class GameSource(Enum):
    SELF_PLAY = "self_play"
    SINGLE_PLAYER = "single_player" 
    COMMUNITY = "community"
    UCI_TOURNAMENT = "uci_tournament"

@dataclass
class UnifiedGameData:
    """Standardized game data format for training"""
    game_id: str
    source: GameSource
    positions: List[Dict[str, Any]]  # FEN, move, policy, value, etc.
    result: str  # "white_wins", "black_wins", "draw"
    metadata: Dict[str, Any]
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['source'] = self.source.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UnifiedGameData':
        data['source'] = GameSource(data['source'])
        return cls(**data)

class UnifiedGameStorage:
    """Unified storage system for all game types"""
    
    def __init__(self, base_dir: str = "training_games", batch_size: int = 1000):
        self.base_dir = Path(base_dir)
        self.batch_size = batch_size
        self._lock = threading.Lock()
        
        # Directory structure
        self.unified_dir = self.base_dir / "unified"
        self.training_dir = self.base_dir / "training" 
        self.archives_dir = self.base_dir / "archives"
        
        # Create directories
        for dir_path in [self.unified_dir, self.training_dir, self.archives_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Track current batch
        self._current_batch: List[UnifiedGameData] = []
        self._batch_number = self._get_next_batch_number()
        
        logger.info(f"🎯 Unified Game Storage initialized")
        logger.info(f"   Base directory: {self.base_dir}")
        logger.info(f"   Batch size: {batch_size} games")
        logger.info(f"   Next batch: {self._batch_number}")
    
    def _get_next_batch_number(self) -> int:
        """Get the next batch number"""
        existing_batches = list(self.unified_dir.glob("batch_*.json.gz"))
        logger.debug(f"Looking for batches in: {self.unified_dir}")
        logger.debug(f"Found {len(existing_batches)} existing batches: {[b.name for b in existing_batches]}")
        
        if not existing_batches:
            logger.debug("No existing batches found, starting from 1")
            return 1
        
        batch_numbers = []
        for batch_file in existing_batches:
            try:
                # Remove .json.gz extensions and parse number
                name_without_extensions = batch_file.name.replace('.json.gz', '')
                num_str = name_without_extensions.split('_')[1]
                batch_numbers.append(int(num_str))
            except (IndexError, ValueError):
                logger.warning(f"Failed to parse batch number from: {batch_file.name}")
                continue
        
        next_batch = max(batch_numbers, default=0) + 1
        logger.debug(f"Batch numbers found: {batch_numbers}, next batch: {next_batch}")
        return next_batch
    
    def store_game(self, game_data: UnifiedGameData) -> None:
        """Store a game in the unified system"""
        with self._lock:
            self._current_batch.append(game_data)
            
            # If batch is full, save it
            if len(self._current_batch) >= self.batch_size:
                self._save_current_batch()
    
    def store_multiple_games(self, games: List[UnifiedGameData]) -> None:
        """Store multiple games efficiently"""
        with self._lock:
            self._current_batch.extend(games)
            
            # Save complete batches
            while len(self._current_batch) >= self.batch_size:
                batch_to_save = self._current_batch[:self.batch_size]
                self._current_batch = self._current_batch[self.batch_size:]
                self._save_batch(batch_to_save)
    
    def _save_current_batch(self) -> None:
        """Save the current batch and start a new one"""
        if not self._current_batch:
            return
        
        self._save_batch(self._current_batch)
        self._current_batch = []
    
    def _save_batch(self, batch: List[UnifiedGameData]) -> None:
        """Save a batch of games to disk"""
        if not batch:
            return
        
        batch_file = self.unified_dir / f"batch_{self._batch_number:06d}.json.gz"
        
        # Convert to dictionaries
        batch_data = {
            "batch_number": self._batch_number,
            "game_count": len(batch),
            "created_at": datetime.now().isoformat(),
            "games": [game.to_dict() for game in batch]
        }
        
        # Save compressed
        with gzip.open(batch_file, 'wt', encoding='utf-8') as f:
            json.dump(batch_data, f, separators=(',', ':'))
        
        logger.info(f"💾 Saved batch {self._batch_number}: {len(batch)} games → {batch_file}")
        self._batch_number += 1
    
    def force_save_current_batch(self) -> None:
        """Force save the current batch even if not full"""
        with self._lock:
            if self._current_batch:
                self._save_current_batch()
    
    def get_total_games(self) -> int:
        """Get total number of stored games"""
        total = 0
        
        # Count games in batches
        for batch_file in self.unified_dir.glob("batch_*.json.gz"):
            try:
                with gzip.open(batch_file, 'rt', encoding='utf-8') as f:
                    batch_data = json.load(f)
                    total += batch_data.get("game_count", 0)
            except Exception as e:
                logger.warning(f"Could not read batch {batch_file}: {e}")
        
        # Add current batch
        total += len(self._current_batch)
        
        return total
    
    def get_training_ready_count(self) -> int:
        """Get number of games ready for training (in complete batches)"""
        total_games = 0
        
        # Count actual games in batch files
        for batch_file in self.unified_dir.glob("batch_*.json.gz"):
            try:
                with gzip.open(batch_file, 'rt', encoding='utf-8') as f:
                    batch_data = json.load(f)
                    total_games += batch_data.get("game_count", 0)
            except Exception as e:
                logger.warning(f"Could not read batch {batch_file}: {e}")
        
        return total_games
    
    def prepare_training_data(self, max_batches: Optional[int] = None) -> List[Path]:
        """Prepare training data by collecting batch files"""
        batch_files = sorted(self.unified_dir.glob("batch_*.json.gz"))
        
        if max_batches:
            batch_files = batch_files[:max_batches]
        
        return batch_files
    
    def archive_used_batches(self, batch_files: List[Path]) -> Path:
        """Archive batches that were used for training"""
        if not batch_files:
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = self.archives_dir / f"training_batch_{timestamp}.zip"
        
        import zipfile
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for batch_file in batch_files:
                zipf.write(batch_file, batch_file.name)
                batch_file.unlink()  # Delete original after archiving
        
        logger.info(f"📦 Archived {len(batch_files)} batches to {archive_path}")
        return archive_path
    
    def convert_legacy_game(self, game_path: Path, source: GameSource) -> Optional[UnifiedGameData]:
        """Convert legacy game files to unified format"""
        try:
            if game_path.suffix == '.json':
                return self._convert_json_game(game_path, source)
            elif game_path.suffix == '.pkl':
                return self._convert_pkl_games(game_path, source)
            else:
                logger.warning(f"Unknown file type: {game_path}")
                return None
        except Exception as e:
            logger.error(f"Failed to convert {game_path}: {e}")
            return None
    
    def _convert_json_game(self, game_path: Path, source: GameSource) -> Optional[UnifiedGameData]:
        """Convert JSON game file to unified format"""
        with open(game_path, 'r') as f:
            data = json.load(f)
        
        # Extract positions from move history
        positions = []
        if 'move_history' in data:
            board = chess.Board()
            for move_str in data['move_history']:
                try:
                    move = chess.Move.from_uci(move_str)
                    if move in board.legal_moves:
                        positions.append({
                            'fen': board.fen(),
                            'move': move_str,
                            'value': 0.0,  # No value data in JSON games
                            'policy': None  # No policy data in JSON games
                        })
                        board.push(move)
                except:
                    break
        
        # Determine result
        result = "draw"
        if 'metadata' in data and 'status' in data['metadata']:
            status = data['metadata']['status']
            if 'white_wins' in status.lower():
                result = "white_wins"
            elif 'black_wins' in status.lower():
                result = "black_wins"
        
        return UnifiedGameData(
            game_id=data.get('metadata', {}).get('game_id', game_path.stem),
            source=source,
            positions=positions,
            result=result,
            metadata=data.get('metadata', {}),
            timestamp=datetime.now().isoformat()
        )
    
    def _convert_pkl_games(self, pkl_path: Path, source: GameSource) -> List[UnifiedGameData]:
        """Convert PKL file containing multiple games to unified format"""
        with open(pkl_path, 'rb') as f:
            games_data = pickle.load(f)
        
        if not isinstance(games_data, list):
            games_data = [games_data]
        
        unified_games = []
        for i, game_record in enumerate(games_data):
            try:
                unified_game = self._convert_game_record(game_record, source, f"{pkl_path.stem}_{i}")
                if unified_game:
                    unified_games.append(unified_game)
            except Exception as e:
                logger.warning(f"Failed to convert game {i} from {pkl_path}: {e}")
        
        return unified_games
    
    def _convert_game_record(self, game_record: Any, source: GameSource, game_id: str) -> Optional[UnifiedGameData]:
        """Convert a GameRecord object to unified format"""
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
            source=source,
            positions=positions,
            result=result,
            metadata={'original_format': 'game_record'},
            timestamp=datetime.now().isoformat()
        )

# Global instance
_unified_storage: Optional[UnifiedGameStorage] = None

def get_unified_storage() -> UnifiedGameStorage:
    """Get the global unified storage instance"""
    global _unified_storage
    if _unified_storage is None:
        _unified_storage = UnifiedGameStorage()
    return _unified_storage

def initialize_unified_storage(base_dir: str = "training_games", batch_size: int = 1000) -> UnifiedGameStorage:
    """Initialize the global unified storage"""
    global _unified_storage
    _unified_storage = UnifiedGameStorage(base_dir, batch_size)
    return _unified_storage 