"""
Worker service for managing communication with load balancer.
"""

import asyncio
import logging
import aiohttp
from typing import Dict, List, Optional
import json
import time

from rival_ai.distributed.worker.node import WorkerNode
from rival_ai.distributed.game_collector.collector import CollectedGame

logger = logging.getLogger(__name__)

class WorkerService:
    """Service for managing worker node and load balancer communication."""
    
    def __init__(
        self,
        load_balancer_url: str,
        node: Optional[WorkerNode] = None,
        heartbeat_interval: int = 10
    ):
        """Initialize the worker service.
        
        Args:
            load_balancer_url: URL of the load balancer
            node: Optional worker node, will be created if not provided
            heartbeat_interval: Seconds between heartbeats
        """
        self.load_balancer_url = load_balancer_url
        self.node = node or WorkerNode()
        self.heartbeat_interval = heartbeat_interval
        self.session: Optional[aiohttp.ClientSession] = None
        self.running = False
        self.tasks = []
        
    async def start(self):
        """Start the worker service."""
        logger.info(f"Starting worker service for node {self.node.node_id}")
        self.running = True
        
        # Create HTTP session
        self.session = aiohttp.ClientSession()
        
        # Start node
        await self.node.start()
        
        # Register with load balancer
        await self._register()
        
        # Start tasks
        self.tasks.extend([
            asyncio.create_task(self._heartbeat_loop()),
            asyncio.create_task(self._handle_commands())
        ])
        
    async def stop(self):
        """Stop the worker service."""
        logger.info(f"Stopping worker service for node {self.node.node_id}")
        self.running = False
        
        # Stop tasks
        for task in self.tasks:
            task.cancel()
            
        await asyncio.gather(*self.tasks, return_exceptions=True)
        
        # Unregister from load balancer
        await self._unregister()
        
        # Stop node
        await self.node.stop()
        
        # Close session
        if self.session:
            await self.session.close()
            
    async def _register(self):
        """Register with load balancer."""
        if not self.session:
            return
            
        try:
            async with self.session.post(
                f"{self.load_balancer_url}/register",
                json={
                    "node_id": self.node.node_id,
                    "capacity": self.node.capacity,
                    "gpu_memory": self.node.gpu_memory,
                    "cpu_cores": self.node.cpu_count
                }
            ) as response:
                if response.status == 200:
                    logger.info(f"Registered with load balancer")
                else:
                    logger.error(f"Failed to register: {response.status}")
                    
        except Exception as e:
            logger.error(f"Error registering with load balancer: {e}")
            
    async def _unregister(self):
        """Unregister from load balancer."""
        if not self.session:
            return
            
        try:
            async with self.session.post(
                f"{self.load_balancer_url}/unregister",
                json={"node_id": self.node.node_id}
            ) as response:
                if response.status == 200:
                    logger.info("Unregistered from load balancer")
                else:
                    logger.error(f"Failed to unregister: {response.status}")
                    
        except Exception as e:
            logger.error(f"Error unregistering from load balancer: {e}")
            
    async def _heartbeat_loop(self):
        """Send periodic heartbeats to load balancer."""
        if not self.session:
            return
            
        while self.running:
            try:
                async with self.session.post(
                    f"{self.load_balancer_url}/heartbeat",
                    json={
                        "node_id": self.node.node_id,
                        "stats": self.node.get_stats()
                    }
                ) as response:
                    if response.status != 200:
                        logger.warning(f"Heartbeat failed: {response.status}")
                        
            except Exception as e:
                logger.error(f"Error sending heartbeat: {e}")
                
            await asyncio.sleep(self.heartbeat_interval)
            
    async def _handle_commands(self):
        """Handle commands from load balancer."""
        if not self.session:
            return
            
        while self.running:
            try:
                async with self.session.get(
                    f"{self.load_balancer_url}/commands/{self.node.node_id}"
                ) as response:
                    if response.status == 200:
                        commands = await response.json()
                        for command in commands:
                            await self._process_command(command)
                    elif response.status != 404:  # 404 means no commands
                        logger.warning(f"Failed to get commands: {response.status}")
                        
            except Exception as e:
                logger.error(f"Error handling commands: {e}")
                
            await asyncio.sleep(1)
            
    async def _process_command(self, command: Dict):
        """Process a command from the load balancer.
        
        Args:
            command: Command to process
        """
        try:
            cmd_type = command.get("type")
            if cmd_type == "start_game":
                success = await self.node.start_game(
                    game_id=command["game_id"],
                    player_ids=command["player_ids"],
                    model_version=command["model_version"],
                    priority=command.get("priority", 0)
                )
                
                await self._send_response(command["id"], {"success": success})
                
            elif cmd_type == "stop_game":
                game_data = await self.node.stop_game(
                    game_id=command["game_id"],
                    collect_data=command.get("collect_data", True)
                )
                
                await self._send_response(command["id"], {"game_data": game_data})
                
            else:
                logger.warning(f"Unknown command type: {cmd_type}")
                
        except Exception as e:
            logger.error(f"Error processing command: {e}")
            await self._send_response(command["id"], {"error": str(e)})
            
    async def _send_response(self, command_id: str, response: Dict):
        """Send command response to load balancer.
        
        Args:
            command_id: ID of the command
            response: Response data
        """
        if not self.session:
            return
            
        try:
            async with self.session.post(
                f"{self.load_balancer_url}/response/{command_id}",
                json=response
            ) as resp:
                if resp.status != 200:
                    logger.warning(f"Failed to send response: {resp.status}")
                    
        except Exception as e:
            logger.error(f"Error sending response: {e}")
            
    def get_stats(self) -> Dict:
        """Get service statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            "node": self.node.get_stats(),
            "running": self.running,
            "tasks": len(self.tasks)
        } 