"""
Orchestrator for distributed training system.
"""

import asyncio
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
import torch

from rival_ai.distributed.config import DistributedConfig
from rival_ai.distributed.game_collector.collector import GameCollector, CollectedGame
from rival_ai.distributed.model_manager.manager import ModelManager
from rival_ai.distributed.training.trainer import DistributedTrainer

logger = logging.getLogger(__name__)

@dataclass
class SystemStats:
    """Statistics for the distributed system."""
    games_collected: int = 0
    games_processed: int = 0
    active_players: int = 0
    training_steps: int = 0
    current_elo: float = 1500.0
    model_versions: int = 0

class Orchestrator:
    """Coordinates the distributed training system."""
    
    def __init__(self, config: DistributedConfig):
        """Initialize the orchestrator.
        
        Args:
            config: System configuration
        """
        self.config = config
        self.stats = SystemStats()
        
        # Create components
        self.collector = GameCollector(config.game_collector)
        self.model_manager = ModelManager(config.model_manager)
        self.trainer = DistributedTrainer(config.trainer)
        
        # Game processing queues
        self.game_queue = asyncio.Queue(maxsize=config.game_collector.max_queue_size)
        self.training_queue = asyncio.Queue(maxsize=config.trainer.max_queue_size)
        
        # Component states
        self.running = False
        self.tasks = []
        
    async def start(self):
        """Start the distributed system."""
        logger.info("Starting distributed training system")
        self.running = True
        
        # Start components
        await self.collector.start()
        await self.trainer.start()
        
        # Start processing tasks
        self.tasks.extend([
            asyncio.create_task(self._process_games()),
            asyncio.create_task(self._evaluate_models()),
            asyncio.create_task(self._monitor_system())
        ])
        
    async def stop(self):
        """Stop the distributed system."""
        logger.info("Stopping distributed training system")
        self.running = False
        
        # Stop components
        await self.collector.stop()
        await self.trainer.stop()
        
        # Cancel tasks
        for task in self.tasks:
            task.cancel()
            
        await asyncio.gather(*self.tasks, return_exceptions=True)
        
    async def add_game(self, game: CollectedGame):
        """Add a game to the system.
        
        Args:
            game: Game to add
        """
        await self.game_queue.put(game)
        self.stats.games_collected += 1
        
    async def _process_games(self):
        """Process incoming games."""
        while self.running:
            try:
                # Get batch of games
                games = []
                while len(games) < self.config.game_collector.batch_size:
                    if self.game_queue.empty():
                        break
                    games.append(await self.game_queue.get())
                    
                if not games:
                    await asyncio.sleep(0.1)
                    continue
                    
                # Process games
                await self.collector.add_games(games)
                await self.trainer.add_games(games)
                
                self.stats.games_processed += len(games)
                
            except Exception as e:
                logger.error(f"Error processing games: {e}")
                await asyncio.sleep(1)
                
    async def _evaluate_models(self):
        """Periodically evaluate and update models."""
        while self.running:
            try:
                # Check if we should evaluate
                if self.trainer.steps % self.config.trainer.eval_interval == 0:
                    # Save current model as candidate
                    model = self.trainer.model
                    version = await self.model_manager.add_model(model)
                    
                    # Evaluate candidate
                    promoted, metrics = await self.model_manager.evaluate_candidate(
                        version.version_id
                    )
                    
                    if promoted:
                        logger.info(
                            f"Model {version.version_id} promoted with "
                            f"win rate {metrics['win_rate']:.2f}"
                        )
                        self.stats.current_elo = version.elo
                        self.stats.model_versions += 1
                        
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Error evaluating models: {e}")
                await asyncio.sleep(1)
                
    async def _monitor_system(self):
        """Monitor system health and performance."""
        while self.running:
            try:
                # Gather component stats
                collector_stats = self.collector.get_stats()
                trainer_stats = self.trainer.get_stats()
                
                # Log statistics
                logger.info(
                    f"System stats: "
                    f"Games collected: {self.stats.games_collected}, "
                    f"Games processed: {self.stats.games_processed}, "
                    f"Training steps: {trainer_stats['steps']}, "
                    f"Current Elo: {self.stats.current_elo:.1f}, "
                    f"Model versions: {self.stats.model_versions}"
                )
                
                # TODO: Add metrics to monitoring system
                
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Error monitoring system: {e}")
                await asyncio.sleep(1)
                
    def get_stats(self) -> Dict:
        """Get system statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            "games_collected": self.stats.games_collected,
            "games_processed": self.stats.games_processed,
            "active_players": self.stats.active_players,
            "training_steps": self.trainer.steps,
            "current_elo": self.stats.current_elo,
            "model_versions": self.stats.model_versions,
            "collector": self.collector.get_stats(),
            "trainer": self.trainer.get_stats()
        } 