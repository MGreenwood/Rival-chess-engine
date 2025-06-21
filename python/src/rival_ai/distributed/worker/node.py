"""
Worker node for running games on a server.
"""

import asyncio
import logging
import uuid
from typing import Dict, List, Optional
import torch
import psutil
import GPUtil
from dataclasses import dataclass

from rival_ai.models.gnn import ChessGNN
from rival_ai.mcts import MCTS, MCTSConfig
from rival_ai.distributed.game_collector.collector import CollectedGame, GameMetadata
from rival_ai.distributed.model_manager.manager import ModelVersion

logger = logging.getLogger(__name__)

@dataclass
class GameSession:
    """Active game session on worker."""
    game_id: str
    model: ChessGNN
    mcts: MCTS
    player_ids: List[str]
    start_time: float
    model_version: str
    priority: int = 0

class WorkerNode:
    """Worker node for running games."""
    
    def __init__(self, node_id: Optional[str] = None):
        """Initialize the worker node.
        
        Args:
            node_id: Optional node ID, will be generated if not provided
        """
        self.node_id = node_id or f"worker_{uuid.uuid4().hex[:8]}"
        self.active_games: Dict[str, GameSession] = {}
        self.model_cache: Dict[str, ChessGNN] = {}
        self.running = False
        
        # Monitor resources
        self.gpu_devices = GPUtil.getGPUs()
        self.cpu_count = psutil.cpu_count()
        self.total_memory = psutil.virtual_memory().total
        
    @property
    def gpu_memory(self) -> Optional[int]:
        """Get available GPU memory in MB."""
        if not self.gpu_devices:
            return None
        return sum(gpu.memoryFree for gpu in self.gpu_devices)
        
    @property
    def capacity(self) -> int:
        """Calculate game capacity based on resources."""
        if self.gpu_devices:
            # GPU-based capacity
            memory_per_game = 500  # Estimated MB per game
            return max(1, self.gpu_memory // memory_per_game)
        else:
            # CPU-based capacity
            return max(1, self.cpu_count // 2)  # 2 cores per game
            
    async def start(self):
        """Start the worker node."""
        logger.info(f"Starting worker node {self.node_id}")
        self.running = True
        
        # Start monitoring task
        asyncio.create_task(self._monitor_resources())
        
    async def stop(self):
        """Stop the worker node."""
        logger.info(f"Stopping worker node {self.node_id}")
        self.running = False
        
        # Stop all games
        game_ids = list(self.active_games.keys())
        for game_id in game_ids:
            await self.stop_game(game_id)
            
    async def start_game(
        self,
        game_id: str,
        player_ids: List[str],
        model_version: str,
        priority: int = 0
    ) -> bool:
        """Start a new game.
        
        Args:
            game_id: Unique game identifier
            player_ids: List of player IDs
            model_version: Version of model to use
            priority: Game priority
            
        Returns:
            Whether game was started successfully
        """
        if len(self.active_games) >= self.capacity:
            logger.warning(f"Cannot start game {game_id}, at capacity")
            return False
            
        try:
            # Get or load model
            if model_version not in self.model_cache:
                model = await self._load_model(model_version)
                self.model_cache[model_version] = model
            else:
                model = self.model_cache[model_version]
                
            # Create MCTS
            config = MCTSConfig(
                num_simulations=800,
                temperature=0.1,
                dirichlet_alpha=0.3,
                dirichlet_weight=0.25
            )
            mcts = MCTS(model, config)
            
            # Create session
            session = GameSession(
                game_id=game_id,
                model=model,
                mcts=mcts,
                player_ids=player_ids,
                start_time=time.time(),
                model_version=model_version,
                priority=priority
            )
            
            self.active_games[game_id] = session
            logger.info(f"Started game {game_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting game {game_id}: {e}")
            return False
            
    async def stop_game(self, game_id: str, collect_data: bool = True) -> Optional[CollectedGame]:
        """Stop a game and optionally collect its data.
        
        Args:
            game_id: Game to stop
            collect_data: Whether to collect game data
            
        Returns:
            Collected game data if collect_data is True
        """
        if game_id not in self.active_games:
            return None
            
        session = self.active_games[game_id]
        
        try:
            # TODO: Collect game data if requested
            if collect_data:
                game_data = CollectedGame(
                    moves=[],  # TODO: Get moves from game
                    result="",  # TODO: Get result
                    metadata=GameMetadata(
                        game_id=game_id,
                        model_version=session.model_version,
                        timestamp=session.start_time
                    ),
                    positions=[]  # TODO: Get positions
                )
            else:
                game_data = None
                
            # Clean up
            del self.active_games[game_id]
            logger.info(f"Stopped game {game_id}")
            
            return game_data
            
        except Exception as e:
            logger.error(f"Error stopping game {game_id}: {e}")
            return None
            
    async def _load_model(self, version: str) -> ChessGNN:
        """Load a model version.
        
        Args:
            version: Model version to load
            
        Returns:
            Loaded model
        """
        # TODO: Implement model loading from storage
        model = ChessGNN()
        model.eval()
        return model
        
    async def _monitor_resources(self):
        """Monitor system resources."""
        while self.running:
            try:
                # Update GPU memory
                if self.gpu_devices:
                    self.gpu_devices = GPUtil.getGPUs()
                    
                # Log resource usage
                memory = psutil.virtual_memory()
                cpu_percent = psutil.cpu_percent()
                
                logger.debug(
                    f"Resource usage - CPU: {cpu_percent}%, "
                    f"Memory: {memory.percent}%, "
                    f"Games: {len(self.active_games)}/{self.capacity}"
                )
                
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Error monitoring resources: {e}")
                await asyncio.sleep(1)
                
    def get_stats(self) -> Dict:
        """Get worker statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            "node_id": self.node_id,
            "active_games": len(self.active_games),
            "capacity": self.capacity,
            "gpu_memory": self.gpu_memory,
            "cpu_count": self.cpu_count,
            "memory_percent": psutil.virtual_memory().percent
        } 