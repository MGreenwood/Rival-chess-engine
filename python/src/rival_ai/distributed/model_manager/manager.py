"""
Model manager for distributed training.
"""

import os
import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import torch
import chess
from concurrent.futures import ThreadPoolExecutor

from rival_ai.distributed.config import ModelManagerConfig
from rival_ai.models.gnn import ChessGNN
from rival_ai.mcts import MCTS, MCTSConfig

logger = logging.getLogger(__name__)

@dataclass
class ModelVersion:
    """Information about a model version."""
    version_id: str
    path: str
    timestamp: float
    elo: float = 1500.0
    games_played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    
    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        if self.games_played == 0:
            return 0.0
        return (self.wins + 0.5 * self.draws) / self.games_played

class ModelManager:
    """Manages model versions and evaluation."""
    
    def __init__(self, config: ModelManagerConfig):
        """Initialize the model manager.
        
        Args:
            config: Configuration for model management
        """
        self.config = config
        self.models: Dict[str, ModelVersion] = {}
        self.active_model: Optional[ModelVersion] = None
        self.candidate_model: Optional[ModelVersion] = None
        self.executor = ThreadPoolExecutor(max_workers=config.tournament_workers)
        
        # Create model directory if it doesn't exist
        os.makedirs(config.model_dir, exist_ok=True)
        
        # Load existing models
        self._load_existing_models()
        
    def _load_existing_models(self):
        """Load existing models from disk."""
        try:
            metadata_path = os.path.join(self.config.model_dir, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                    
                for version_data in metadata["versions"]:
                    version = ModelVersion(**version_data)
                    self.models[version.version_id] = version
                    
                # Set active model
                if metadata.get("active_model"):
                    self.active_model = self.models[metadata["active_model"]]
                    
        except Exception as e:
            logger.error(f"Error loading existing models: {e}")
            
    def _save_metadata(self):
        """Save model metadata to disk."""
        try:
            metadata = {
                "versions": [vars(v) for v in self.models.values()],
                "active_model": self.active_model.version_id if self.active_model else None
            }
            
            metadata_path = os.path.join(self.config.model_dir, "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
            
    async def add_model(self, model: ChessGNN, version_id: Optional[str] = None) -> ModelVersion:
        """Add a new model version.
        
        Args:
            model: The model to add
            version_id: Optional version ID, will be generated if not provided
            
        Returns:
            The created model version
        """
        # Generate version ID if not provided
        if version_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            version_id = f"model_{timestamp}"
            
        # Save model
        path = os.path.join(self.config.model_dir, f"{version_id}.pt")
        torch.save(model.state_dict(), path)
        
        # Create version
        version = ModelVersion(
            version_id=version_id,
            path=path,
            timestamp=datetime.now().timestamp()
        )
        
        self.models[version_id] = version
        self._save_metadata()
        
        return version
        
    async def evaluate_candidate(self, candidate_id: str) -> Tuple[bool, Dict[str, float]]:
        """Evaluate a candidate model against the active model.
        
        Args:
            candidate_id: Version ID of the candidate model
            
        Returns:
            Tuple of (promoted, metrics)
        """
        if candidate_id not in self.models:
            raise ValueError(f"Unknown model version: {candidate_id}")
            
        if not self.active_model:
            # If no active model, promote candidate immediately
            self.active_model = self.models[candidate_id]
            self._save_metadata()
            return True, {"win_rate": 1.0}
            
        # Load models
        candidate = self.models[candidate_id]
        active = self.active_model
        
        # Run tournament
        results = await self._run_tournament(candidate, active)
        
        # Update Elo ratings
        self._update_elo_ratings(candidate, active, results)
        
        # Check if candidate should be promoted
        win_rate = (results["wins"] + 0.5 * results["draws"]) / self.config.eval_games
        promoted = win_rate >= self.config.min_win_rate
        
        if promoted:
            self.active_model = candidate
            self._save_metadata()
            
        metrics = {
            "win_rate": win_rate,
            "elo_diff": candidate.elo - active.elo,
            "games_played": self.config.eval_games,
        }
        
        return promoted, metrics
        
    async def _run_tournament(self, candidate: ModelVersion, active: ModelVersion) -> Dict[str, int]:
        """Run a tournament between two models.
        
        Args:
            candidate: Candidate model version
            active: Active model version
            
        Returns:
            Dictionary with tournament results
        """
        results = {"wins": 0, "draws": 0, "losses": 0}
        
        # Create MCTS instances
        candidate_mcts = self._create_mcts(candidate)
        active_mcts = self._create_mcts(active)
        
        # Play games in parallel
        tasks = []
        for game_idx in range(self.config.eval_games):
            # Alternate colors
            if game_idx % 2 == 0:
                white_mcts, black_mcts = candidate_mcts, active_mcts
            else:
                white_mcts, black_mcts = active_mcts, candidate_mcts
                
            task = asyncio.create_task(
                self._play_evaluation_game(white_mcts, black_mcts)
            )
            tasks.append(task)
            
        # Wait for all games to complete
        game_results = await asyncio.gather(*tasks)
        
        # Process results
        for result in game_results:
            if result == "1-0":
                results["wins" if game_idx % 2 == 0 else "losses"] += 1
            elif result == "0-1":
                results["losses" if game_idx % 2 == 0 else "wins"] += 1
            else:
                results["draws"] += 1
                
        return results
        
    def _create_mcts(self, version: ModelVersion) -> MCTS:
        """Create an MCTS instance for a model version.
        
        Args:
            version: Model version to create MCTS for
            
        Returns:
            MCTS instance
        """
        # Load model
        model = ChessGNN()  # Use your default architecture
        model.load_state_dict(torch.load(version.path))
        model.eval()
        
        # Create MCTS
        config = MCTSConfig(
            num_simulations=800,  # Use reasonable defaults for evaluation
            temperature=0.1,  # Lower temperature for stronger play
            dirichlet_alpha=0.3,
            dirichlet_weight=0.25
        )
        
        return MCTS(model, config)
        
    async def _play_evaluation_game(self, white_mcts: MCTS, black_mcts: MCTS) -> str:
        """Play a single evaluation game.
        
        Args:
            white_mcts: MCTS for white
            black_mcts: MCTS for black
            
        Returns:
            Game result ("1-0", "0-1", or "1/2-1/2")
        """
        board = chess.Board()
        
        while not board.is_game_over():
            mcts = white_mcts if board.turn == chess.WHITE else black_mcts
            move = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                mcts.get_best_move,
                board,
                0.0  # Temperature
            )
            board.push(move)
            
        return board.result()
        
    def _update_elo_ratings(self, candidate: ModelVersion, active: ModelVersion, results: Dict[str, int]):
        """Update Elo ratings based on tournament results.
        
        Args:
            candidate: Candidate model version
            active: Active model version
            results: Tournament results
        """
        games_played = sum(results.values())
        score = (results["wins"] + 0.5 * results["draws"]) / games_played
        
        # Expected score based on Elo difference
        elo_diff = candidate.elo - active.elo
        expected = 1 / (1 + 10 ** (-elo_diff / 400))
        
        # Update ratings
        rating_change = self.config.elo_k_factor * (score - expected)
        candidate.elo += rating_change
        active.elo -= rating_change
        
        # Update game statistics
        candidate.games_played += games_played
        candidate.wins += results["wins"]
        candidate.draws += results["draws"]
        candidate.losses += results["losses"]
        
        active.games_played += games_played
        active.wins += results["losses"]
        active.draws += results["draws"]
        active.losses += results["wins"]
        
        self._save_metadata() 