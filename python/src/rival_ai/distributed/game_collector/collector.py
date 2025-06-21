"""
Game collector for distributed training.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor
import chess
import torch
from torch.multiprocessing import Queue, Process
import numpy as np

from rival_ai.distributed.config import GameCollectorConfig
from rival_ai.pag import PAG
from rival_ai.utils.board_conversion import board_to_hetero_data

logger = logging.getLogger(__name__)

@dataclass
class GameMetadata:
    """Metadata for a collected game."""
    game_id: str
    white_elo: Optional[int] = None
    black_elo: Optional[int] = None
    source: str = "unknown"  # "human", "engine", "self_play"
    model_version: Optional[str] = None
    timestamp: float = 0.0

@dataclass
class CollectedGame:
    """A game collected for training."""
    moves: List[chess.Move]
    result: str  # "1-0", "0-1", "1/2-1/2"
    metadata: GameMetadata
    positions: List[str]  # FEN strings
    pag_cache_keys: Optional[List[str]] = None

class GameCollector:
    """Collects and processes games for training."""
    
    def __init__(self, config: GameCollectorConfig):
        """Initialize the game collector.
        
        Args:
            config: Configuration for game collection
        """
        self.config = config
        self.game_queue = Queue(maxsize=config.max_queue_size)
        self.pag_cache = {}  # Cache of processed PAGs
        self.workers = []
        self.running = False
        self.executor = ThreadPoolExecutor(max_workers=config.num_workers)
        
    async def start(self):
        """Start the game collector workers."""
        self.running = True
        for _ in range(self.config.num_workers):
            worker = Process(target=self._process_games_worker)
            worker.start()
            self.workers.append(worker)
            
    async def stop(self):
        """Stop the game collector."""
        self.running = False
        for worker in self.workers:
            worker.terminate()
        self.executor.shutdown()
        
    async def add_game(self, game: CollectedGame):
        """Add a game to the processing queue.
        
        Args:
            game: The game to process
        """
        if not self.running:
            raise RuntimeError("Game collector is not running")
            
        try:
            # Basic validation
            if game.metadata.source == "human" and (
                game.metadata.white_elo is None or 
                game.metadata.black_elo is None or
                min(game.metadata.white_elo, game.metadata.black_elo) < self.config.min_elo
            ):
                logger.debug(f"Skipping low-rated game {game.metadata.game_id}")
                return
                
            # Add to queue
            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.game_queue.put,
                game
            )
            
        except Exception as e:
            logger.error(f"Error adding game {game.metadata.game_id}: {e}")
            
    def _process_games_worker(self):
        """Worker process for processing games."""
        while self.running:
            try:
                # Get batch of games
                games = []
                for _ in range(self.config.batch_size):
                    if not self.game_queue.empty():
                        games.append(self.game_queue.get())
                    else:
                        break
                        
                if not games:
                    continue
                    
                # Process games in batch
                self._process_game_batch(games)
                
            except Exception as e:
                logger.error(f"Error in game processing worker: {e}")
                
    def _process_game_batch(self, games: List[CollectedGame]):
        """Process a batch of games.
        
        Args:
            games: List of games to process
        """
        try:
            for game in games:
                # Convert positions to PAGs
                board = chess.Board()
                pags = []
                
                for move in game.moves:
                    # Get current position
                    fen = board.fen()
                    
                    # Check cache
                    if fen in self.pag_cache:
                        pag = self.pag_cache[fen]
                    else:
                        # Convert to PAG
                        data = board_to_hetero_data(board)
                        pag = PAG.from_hetero_data(data)
                        
                        # Cache if space available
                        if len(self.pag_cache) < self.config.cache_size:
                            self.pag_cache[fen] = pag
                            
                    pags.append(pag)
                    
                    # Make move
                    board.push(move)
                    
                # Store PAG cache keys
                game.pag_cache_keys = list(self.pag_cache.keys())
                
                # TODO: Send processed game to training queue
                
        except Exception as e:
            logger.error(f"Error processing game batch: {e}")
            
    def get_stats(self) -> Dict[str, Union[int, float]]:
        """Get collector statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            "queue_size": self.game_queue.qsize(),
            "cache_size": len(self.pag_cache),
            "num_workers": len(self.workers),
        } 