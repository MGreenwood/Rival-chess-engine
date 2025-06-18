"""
Distributed trainer for online learning.
"""

import os
import logging
import asyncio
from typing import Dict, List, Optional, Tuple
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import numpy as np

from rival_ai.distributed.config import DistributedTrainerConfig
from rival_ai.models.gnn import ChessGNN
from rival_ai.training.loss import PolicyValueLoss
from rival_ai.distributed.game_collector.collector import CollectedGame

logger = logging.getLogger(__name__)

class ReplayBuffer:
    """Replay buffer for training data."""
    
    def __init__(self, max_size: int):
        """Initialize replay buffer.
        
        Args:
            max_size: Maximum number of positions to store
        """
        self.max_size = max_size
        self.buffer = []
        self.position = 0
        
    def add(self, games: List[CollectedGame]):
        """Add games to the buffer.
        
        Args:
            games: List of games to add
        """
        for game in games:
            # Convert game to training examples
            examples = self._game_to_examples(game)
            
            # Add to buffer
            if len(self.buffer) < self.max_size:
                self.buffer.extend(examples)
            else:
                # Replace old examples
                for example in examples:
                    self.buffer[self.position] = example
                    self.position = (self.position + 1) % self.max_size
                    
    def sample(self, batch_size: int) -> List[Dict]:
        """Sample a batch of training examples.
        
        Args:
            batch_size: Number of examples to sample
            
        Returns:
            List of training examples
        """
        if len(self.buffer) < batch_size:
            return self.buffer
            
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
        
    def _game_to_examples(self, game: CollectedGame) -> List[Dict]:
        """Convert a game to training examples.
        
        Args:
            game: Game to convert
            
        Returns:
            List of training examples
        """
        examples = []
        result = self._parse_result(game.result)
        
        for position, move in zip(game.positions, game.moves):
            example = {
                "position": position,
                "move": move,
                "result": result,
                "metadata": game.metadata
            }
            examples.append(example)
            
        return examples
        
    def _parse_result(self, result: str) -> float:
        """Parse game result string to value target.
        
        Args:
            result: Game result string ("1-0", "0-1", or "1/2-1/2")
            
        Returns:
            Value target (-1 to 1)
        """
        if result == "1-0":
            return 1.0
        elif result == "0-1":
            return -1.0
        else:
            return 0.0
            
class DistributedTrainer:
    """Distributed trainer for online learning."""
    
    def __init__(self, config: DistributedTrainerConfig):
        """Initialize the trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize distributed training
        if config.world_size > 1:
            dist.init_process_group(
                backend=config.dist_backend,
                init_method=config.dist_url,
                world_size=config.world_size,
                rank=config.rank
            )
            
        # Create model and move to device
        self.model = ChessGNN()
        self.model.to(self.device)
        
        if config.world_size > 1:
            self.model = DDP(self.model)
            
        # Create optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Create loss function
        self.criterion = PolicyValueLoss(
            value_weight=config.value_loss_weight,
            policy_weight=config.policy_loss_weight
        )
        
        # Create replay buffer
        self.replay_buffer = ReplayBuffer(config.buffer_size)
        
        # Training state
        self.steps = 0
        self.running = False
        
    async def start(self):
        """Start the trainer."""
        self.running = True
        
        # Start training loop
        while self.running:
            try:
                # Wait for sufficient data
                if len(self.replay_buffer.buffer) < self.config.min_buffer_size:
                    await asyncio.sleep(1)
                    continue
                    
                # Train on batch
                await self._train_step()
                
                # Save checkpoint if needed
                if self.steps % self.config.checkpoint_interval == 0:
                    self._save_checkpoint()
                    
            except Exception as e:
                logger.error(f"Error in training loop: {e}")
                await asyncio.sleep(1)
                
    async def stop(self):
        """Stop the trainer."""
        self.running = False
        if self.config.world_size > 1:
            dist.destroy_process_group()
            
    async def add_games(self, games: List[CollectedGame]):
        """Add games to the replay buffer.
        
        Args:
            games: List of games to add
        """
        self.replay_buffer.add(games)
        
    async def _train_step(self):
        """Perform a single training step."""
        self.model.train()
        
        # Sample batch
        batch_size = self.config.batch_size
        if self.config.world_size > 1:
            batch_size //= self.config.world_size
            
        # Mix old and new data
        old_size = int(batch_size * self.config.sample_ratio)
        new_size = batch_size - old_size
        
        old_batch = self.replay_buffer.sample(old_size)
        new_batch = self.replay_buffer.sample(new_size)
        batch = old_batch + new_batch
        
        # Convert batch to tensors
        position_batch = torch.stack([example["position"] for example in batch]).to(self.device)
        policy_batch = torch.stack([example["move"] for example in batch]).to(self.device)
        value_batch = torch.tensor([example["result"] for example in batch]).to(self.device)
        
        # Forward pass
        policy_out, value_out = self.model(position_batch)
        
        # Calculate loss
        loss = self.criterion(policy_out, value_out, policy_batch, value_batch)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients
        if self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.grad_clip
            )
            
        # Update weights
        self.optimizer.step()
        
        # Update step counter
        self.steps += 1
        
    def _save_checkpoint(self):
        """Save a training checkpoint."""
        if self.config.rank == 0:  # Only save on main process
            checkpoint = {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "steps": self.steps,
                "config": self.config
            }
            
            path = f"checkpoints/step_{self.steps}.pt"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(checkpoint, path)
            
    def get_stats(self) -> Dict[str, float]:
        """Get training statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            "buffer_size": len(self.replay_buffer.buffer),
            "steps": self.steps,
            "learning_rate": self.optimizer.param_groups[0]["lr"]
        } 