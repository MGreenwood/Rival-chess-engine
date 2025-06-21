"""
Load balancer for distributing games across worker nodes.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
import time
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class WorkerNode:
    """Information about a worker node."""
    node_id: str
    capacity: int  # Maximum number of games
    current_load: int = 0
    last_heartbeat: float = field(default_factory=time.time)
    games: Set[str] = field(default_factory=set)
    gpu_memory: Optional[int] = None  # GPU memory in MB
    cpu_cores: Optional[int] = None
    status: str = "active"  # active, draining, offline
    
    @property
    def load_percentage(self) -> float:
        """Calculate load percentage."""
        return (self.current_load / self.capacity) * 100 if self.capacity > 0 else 100
        
    @property
    def is_available(self) -> bool:
        """Check if worker is available for new games."""
        return (
            self.status == "active" and
            self.current_load < self.capacity and
            time.time() - self.last_heartbeat < 30  # Consider node dead after 30s
        )

@dataclass
class GameAllocation:
    """Information about a game allocation."""
    game_id: str
    node_id: str
    start_time: float
    player_ids: List[str]
    model_version: str
    priority: int = 0  # Higher number = higher priority

class LoadBalancer:
    """Manages game distribution across worker nodes."""
    
    def __init__(self):
        """Initialize the load balancer."""
        self.workers: Dict[str, WorkerNode] = {}
        self.allocations: Dict[str, GameAllocation] = {}
        self.rebalance_lock = asyncio.Lock()
        
    async def register_worker(
        self,
        node_id: str,
        capacity: int,
        gpu_memory: Optional[int] = None,
        cpu_cores: Optional[int] = None
    ) -> WorkerNode:
        """Register a new worker node.
        
        Args:
            node_id: Unique identifier for the worker
            capacity: Maximum number of concurrent games
            gpu_memory: Available GPU memory in MB
            cpu_cores: Number of CPU cores
            
        Returns:
            The registered worker node
        """
        worker = WorkerNode(
            node_id=node_id,
            capacity=capacity,
            gpu_memory=gpu_memory,
            cpu_cores=cpu_cores
        )
        self.workers[node_id] = worker
        logger.info(f"Registered worker {node_id} with capacity {capacity}")
        return worker
        
    async def unregister_worker(self, node_id: str, drain: bool = True):
        """Unregister a worker node.
        
        Args:
            node_id: Worker to unregister
            drain: Whether to wait for games to complete
        """
        if node_id not in self.workers:
            return
            
        worker = self.workers[node_id]
        if drain:
            worker.status = "draining"
            while worker.current_load > 0:
                await asyncio.sleep(1)
                
        # Move games to other workers
        games = list(worker.games)
        for game_id in games:
            await self.reallocate_game(game_id)
            
        del self.workers[node_id]
        logger.info(f"Unregistered worker {node_id}")
        
    async def allocate_game(
        self,
        game_id: str,
        player_ids: List[str],
        model_version: str,
        priority: int = 0
    ) -> Optional[str]:
        """Allocate a game to a worker node.
        
        Args:
            game_id: Unique identifier for the game
            player_ids: List of player IDs
            model_version: Version of the model to use
            priority: Game priority (higher = more important)
            
        Returns:
            ID of the allocated worker, or None if no worker available
        """
        async with self.rebalance_lock:
            # Find best worker
            best_worker = None
            min_load = float('inf')
            
            for worker in self.workers.values():
                if not worker.is_available:
                    continue
                    
                # Consider load and capacity
                load = worker.load_percentage
                if load < min_load:
                    best_worker = worker
                    min_load = load
                    
            if not best_worker:
                logger.warning(f"No worker available for game {game_id}")
                return None
                
            # Allocate game
            allocation = GameAllocation(
                game_id=game_id,
                node_id=best_worker.node_id,
                start_time=time.time(),
                player_ids=player_ids,
                model_version=model_version,
                priority=priority
            )
            
            self.allocations[game_id] = allocation
            best_worker.games.add(game_id)
            best_worker.current_load += 1
            
            logger.info(
                f"Allocated game {game_id} to worker {best_worker.node_id} "
                f"(load: {best_worker.load_percentage:.1f}%)"
            )
            
            return best_worker.node_id
            
    async def deallocate_game(self, game_id: str):
        """Deallocate a game from its worker.
        
        Args:
            game_id: Game to deallocate
        """
        if game_id not in self.allocations:
            return
            
        allocation = self.allocations[game_id]
        if allocation.node_id in self.workers:
            worker = self.workers[allocation.node_id]
            worker.games.remove(game_id)
            worker.current_load -= 1
            
        del self.allocations[game_id]
        logger.info(f"Deallocated game {game_id}")
        
    async def reallocate_game(self, game_id: str) -> Optional[str]:
        """Move a game to a different worker.
        
        Args:
            game_id: Game to reallocate
            
        Returns:
            ID of the new worker, or None if no worker available
        """
        if game_id not in self.allocations:
            return None
            
        allocation = self.allocations[game_id]
        await self.deallocate_game(game_id)
        
        return await self.allocate_game(
            game_id,
            allocation.player_ids,
            allocation.model_version,
            allocation.priority
        )
        
    async def heartbeat(self, node_id: str):
        """Update worker heartbeat timestamp.
        
        Args:
            node_id: Worker to update
        """
        if node_id in self.workers:
            self.workers[node_id].last_heartbeat = time.time()
            
    async def rebalance(self):
        """Rebalance games across workers."""
        async with self.rebalance_lock:
            # Calculate average load
            active_workers = [w for w in self.workers.values() if w.is_available]
            if not active_workers:
                return
                
            avg_load = np.mean([w.load_percentage for w in active_workers])
            
            # Find overloaded and underloaded workers
            threshold = 10  # Allow 10% deviation from average
            overloaded = [w for w in active_workers if w.load_percentage > avg_load + threshold]
            underloaded = [w for w in active_workers if w.load_percentage < avg_load - threshold]
            
            if not (overloaded and underloaded):
                return
                
            # Move games from overloaded to underloaded workers
            for over_worker in overloaded:
                games = sorted(
                    over_worker.games,
                    key=lambda g: self.allocations[g].priority,
                    reverse=True
                )
                
                # Move games until load is balanced
                while (
                    over_worker.load_percentage > avg_load + threshold and
                    games and underloaded
                ):
                    game_id = games.pop()
                    new_node = await self.reallocate_game(game_id)
                    if new_node:
                        logger.info(
                            f"Rebalanced game {game_id} from {over_worker.node_id} "
                            f"to {new_node}"
                        )
                        
    def get_stats(self) -> Dict:
        """Get load balancer statistics.
        
        Returns:
            Dictionary of statistics
        """
        active_workers = [w for w in self.workers.values() if w.is_available]
        return {
            "total_workers": len(self.workers),
            "active_workers": len(active_workers),
            "total_games": len(self.allocations),
            "avg_load": np.mean([w.load_percentage for w in active_workers]) if active_workers else 0,
            "max_load": max([w.load_percentage for w in active_workers]) if active_workers else 0,
        } 