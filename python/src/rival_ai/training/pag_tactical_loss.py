#!/usr/bin/env python3
"""
PAG-Aware Tactical Loss Function

Leverages the ultra-dense PAG features from Rust (~340k features) to create
loss functions specifically targeting tactical understanding and preventing
basic blunders like hanging pieces.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np

class PAGTacticalLoss(nn.Module):
    """
    Advanced loss function that leverages PAG features for tactical learning.
    
    Key Features:
    - Vulnerability punishment (hanging pieces penalty)
    - Tactical motif awareness (pins, forks, skewers)
    - Threat generation rewards
    - Material balance with PAG context
    """
    
    def __init__(
        self,
        # Core tactical weights
        vulnerability_weight: float = 5.0,      # Strong penalty for hanging pieces
        motif_awareness_weight: float = 3.0,    # Reward tactical motifs
        threat_generation_weight: float = 2.0,  # Reward creating threats
        material_protection_weight: float = 4.0, # Protect valuable pieces
        
        # Advanced tactical weights
        pin_exploitation_weight: float = 2.5,   # Exploit pins/skewers
        fork_creation_weight: float = 2.0,      # Create forks
        discovery_weight: float = 1.5,          # Discovered attacks
        defensive_coordination_weight: float = 2.0, # Coordinate defense
        
        # Positional-tactical balance
        tactical_positional_balance: float = 0.7,  # 70% tactical, 30% positional
        endgame_tactical_boost: float = 1.5,    # Boost tactical precision in endgame
        
        # Learning progression
        progressive_difficulty: bool = True,    # Gradually increase tactical complexity
        current_epoch: int = 0,
    ):
        super().__init__()
        
        # Store weights
        self.vulnerability_weight = vulnerability_weight
        self.motif_awareness_weight = motif_awareness_weight
        self.threat_generation_weight = threat_generation_weight
        self.material_protection_weight = material_protection_weight
        
        self.pin_exploitation_weight = pin_exploitation_weight
        self.fork_creation_weight = fork_creation_weight
        self.discovery_weight = discovery_weight
        self.defensive_coordination_weight = defensive_coordination_weight
        
        self.tactical_positional_balance = tactical_positional_balance
        self.endgame_tactical_boost = endgame_tactical_boost
        
        self.progressive_difficulty = progressive_difficulty
        self.current_epoch = current_epoch
        
        # PAG feature indices (based on Rust implementation)
        self.pag_indices = {
            # Piece tactical features (starting after basic piece info)
            'vulnerability_status': slice(18, 34),    # 16 features: hanging, pinned, etc.
            'motif_involvement': slice(46, 66),       # 20 features: pins, forks, skewers
            'threat_generation': slice(66, 78),       # 12 features: immediate threats
            'attack_patterns': slice(2, 18),          # 16 features: attack capabilities
            'defense_patterns': slice(34, 46),        # 12 features: defense patterns
            
            # Edge tactical features (for piece relationships)
            'attack_vectors': slice(0, 8),            # 8 features per edge
            'motif_vectors': slice(14, 24),           # 10 features per edge
            'threat_vectors': slice(24, 32),          # 8 features per edge
        }
        
    def forward(
        self, 
        policy_logits: torch.Tensor,
        value_pred: torch.Tensor,
        target_policy: torch.Tensor,
        target_value: torch.Tensor,
        pag_features: torch.Tensor,  # PAG features from Rust
        move_history: Optional[List[str]] = None,
        game_phase: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute PAG-aware tactical loss.
        
        Args:
            policy_logits: Model policy predictions [batch, moves]  
            value_pred: Model value predictions [batch, 1]
            target_policy: Target policy from MCTS [batch, moves]
            target_value: Target value from games [batch, 1]
            pag_features: Ultra-dense PAG features [batch, pieces, features]
            move_history: Optional move history for context
            game_phase: Optional game phase for adaptive weighting
            
        Returns:
            Dictionary of loss components
        """
        batch_size = policy_logits.size(0)
        device = policy_logits.device
        
        # Base policy and value losses
        policy_loss = F.cross_entropy(policy_logits, target_policy.argmax(dim=1))
        value_loss = F.mse_loss(value_pred, target_value)
        
        # Extract PAG tactical features
        tactical_losses = self._compute_tactical_losses(pag_features, target_value, device)
        
        # Game phase adaptive weighting
        phase_multiplier = self._get_phase_multiplier(game_phase)
        
        # Progressive difficulty scaling
        difficulty_multiplier = self._get_difficulty_multiplier()
        
        # Combine losses with tactical emphasis
        total_tactical_loss = (
            tactical_losses['vulnerability_loss'] * self.vulnerability_weight +
            tactical_losses['motif_awareness_loss'] * self.motif_awareness_weight +
            tactical_losses['threat_generation_loss'] * self.threat_generation_weight +
            tactical_losses['material_protection_loss'] * self.material_protection_weight +
            tactical_losses['pin_exploitation_loss'] * self.pin_exploitation_weight +
            tactical_losses['fork_creation_loss'] * self.fork_creation_weight +
            tactical_losses['discovery_loss'] * self.discovery_weight +
            tactical_losses['defensive_coordination_loss'] * self.defensive_coordination_weight
        )
        
        # Apply phase and difficulty multipliers
        total_tactical_loss *= phase_multiplier * difficulty_multiplier
        
        # Balance tactical vs positional learning
        total_loss = (
            policy_loss * (1 - self.tactical_positional_balance) +
            value_loss * 0.5 +
            total_tactical_loss * self.tactical_positional_balance
        )
        
        return {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'tactical_loss': total_tactical_loss,
            **tactical_losses,
            'phase_multiplier': torch.tensor(phase_multiplier, device=device),
            'difficulty_multiplier': torch.tensor(difficulty_multiplier, device=device),
        }
    
    def _compute_tactical_losses(
        self, 
        pag_features: torch.Tensor, 
        target_value: torch.Tensor,
        device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """Extract and penalize specific tactical weaknesses from PAG features."""
        
        batch_size, num_pieces, feature_dim = pag_features.shape
        
        # Extract vulnerability features (16 dims per piece)
        vulnerability_features = pag_features[:, :, self.pag_indices['vulnerability_status']]
        
        # Vulnerability Loss: Heavily penalize hanging pieces
        # Hanging pieces have high vulnerability scores
        hanging_penalty = torch.mean(vulnerability_features ** 2)  # Squared penalty
        
        # Weight by piece value (queens hurt more than pawns)
        piece_values = self._extract_piece_values(pag_features)  # [batch, pieces]
        weighted_vulnerability = torch.sum(
            vulnerability_features * piece_values.unsqueeze(-1), dim=[1, 2]
        )
        vulnerability_loss = torch.mean(weighted_vulnerability)
        
        # Motif Awareness Loss: Reward recognizing tactical patterns
        motif_features = pag_features[:, :, self.pag_indices['motif_involvement']]
        
        # Reward when model recognizes tactical motifs (pins, forks, skewers)
        # High motif involvement should correlate with better moves
        motif_quality = torch.mean(motif_features, dim=2)  # [batch, pieces]
        motif_awareness_loss = -torch.mean(motif_quality * target_value.expand_as(motif_quality))
        
        # Threat Generation Loss: Reward creating threats
        threat_features = pag_features[:, :, self.pag_indices['threat_generation']]
        threat_score = torch.mean(threat_features)
        threat_generation_loss = -threat_score  # Negative because we want to maximize threats
        
        # Material Protection Loss: Protect high-value pieces
        attack_patterns = pag_features[:, :, self.pag_indices['attack_patterns']]
        defense_patterns = pag_features[:, :, self.pag_indices['defense_patterns']]
        
        # Balance attack vs defense for valuable pieces
        piece_importance = piece_values / 9.0  # Normalize by queen value
        defense_deficit = torch.clamp(
            attack_patterns.mean(dim=2) - defense_patterns.mean(dim=2), 
            min=0
        )  # Only penalize when defense < attack
        material_protection_loss = torch.mean(defense_deficit * piece_importance)
        
        # Advanced tactical losses
        pin_exploitation_loss = self._compute_pin_loss(motif_features)
        fork_creation_loss = self._compute_fork_loss(motif_features)
        discovery_loss = self._compute_discovery_loss(motif_features)
        defensive_coordination_loss = self._compute_coordination_loss(defense_patterns)
        
        return {
            'vulnerability_loss': vulnerability_loss,
            'motif_awareness_loss': motif_awareness_loss,
            'threat_generation_loss': threat_generation_loss,
            'material_protection_loss': material_protection_loss,
            'pin_exploitation_loss': pin_exploitation_loss,
            'fork_creation_loss': fork_creation_loss,
            'discovery_loss': discovery_loss,
            'defensive_coordination_loss': defensive_coordination_loss,
        }
    
    def _extract_piece_values(self, pag_features: torch.Tensor) -> torch.Tensor:
        """Extract piece values from PAG features (one-hot piece type encoding)."""
        # Piece type is encoded in first 6 features (one-hot)
        piece_type_features = pag_features[:, :, :6]  # [batch, pieces, 6]
        
        # Standard piece values: [pawn, knight, bishop, rook, queen, king]
        piece_values = torch.tensor([1.0, 3.0, 3.0, 5.0, 9.0, 0.0], device=pag_features.device)
        
        # Compute actual piece values for each piece
        values = torch.sum(piece_type_features * piece_values, dim=2)  # [batch, pieces]
        
        return values
    
    def _compute_pin_loss(self, motif_features: torch.Tensor) -> torch.Tensor:
        """Specific loss for pin recognition and exploitation."""
        # Assume first few motif features relate to pins
        pin_features = motif_features[:, :, :3]  # Pin detection, exploitation, creation
        pin_score = torch.mean(pin_features)
        return -pin_score  # Reward pin awareness
    
    def _compute_fork_loss(self, motif_features: torch.Tensor) -> torch.Tensor:
        """Specific loss for fork recognition and creation."""
        # Next few motif features relate to forks
        fork_features = motif_features[:, :, 3:6]  # Fork potential, creation, double attacks
        fork_score = torch.mean(fork_features)
        return -fork_score  # Reward fork creation
    
    def _compute_discovery_loss(self, motif_features: torch.Tensor) -> torch.Tensor:
        """Specific loss for discovered attack recognition."""
        # Discovery-related motif features
        discovery_features = motif_features[:, :, 6:9]  # Discovered attacks, batteries
        discovery_score = torch.mean(discovery_features)
        return -discovery_score  # Reward discovery awareness
    
    def _compute_coordination_loss(self, defense_patterns: torch.Tensor) -> torch.Tensor:
        """Loss for defensive coordination between pieces."""
        # Reward when pieces defend each other effectively
        coordination_score = torch.mean(defense_patterns)
        
        # Penalize when coordination is poor (each piece fends for itself)
        coordination_variance = torch.var(defense_patterns, dim=2).mean()
        
        return coordination_variance - coordination_score
    
    def _get_phase_multiplier(self, game_phase: Optional[str]) -> float:
        """Adaptive weighting based on game phase."""
        if game_phase == "endgame":
            return self.endgame_tactical_boost
        elif game_phase == "middlegame":
            return 1.2  # Slightly boost middlegame tactics
        else:
            return 1.0  # Standard weighting for opening
    
    def _get_difficulty_multiplier(self) -> float:
        """Progressive difficulty scaling during training."""
        if not self.progressive_difficulty:
            return 1.0
        
        # Gradually increase tactical emphasis over epochs
        # Start with 50% tactical focus, grow to 100% over 50 epochs
        base_multiplier = 0.5
        growth_rate = 0.01  # 1% increase per epoch
        max_epochs = 50
        
        progress = min(self.current_epoch / max_epochs, 1.0)
        return base_multiplier + (0.5 * progress)
    
    def update_epoch(self, epoch: int):
        """Update current epoch for progressive difficulty."""
        self.current_epoch = epoch


class PAGTacticalMetrics:
    """Helper class to compute tactical metrics from PAG features."""
    
    @staticmethod
    def compute_hanging_pieces_count(pag_features: torch.Tensor) -> torch.Tensor:
        """Count hanging pieces per position."""
        vulnerability_features = pag_features[:, :, 18:34]  # Vulnerability slice
        
        # A piece is "hanging" if vulnerability score > threshold
        hanging_threshold = 0.7
        hanging_mask = vulnerability_features.max(dim=2)[0] > hanging_threshold
        
        return hanging_mask.sum(dim=1).float()  # Count per batch
    
    @staticmethod
    def compute_tactical_motif_score(pag_features: torch.Tensor) -> torch.Tensor:
        """Compute overall tactical motif awareness score."""
        motif_features = pag_features[:, :, 46:66]  # Motif involvement slice
        
        # Average motif involvement across all pieces
        return motif_features.mean(dim=[1, 2])
    
    @staticmethod
    def compute_material_safety_score(pag_features: torch.Tensor) -> torch.Tensor:
        """Compute how well material is protected."""
        # Extract piece values and defense patterns
        piece_values = PAGTacticalLoss._extract_piece_values(None, pag_features)
        defense_patterns = pag_features[:, :, 34:46]  # Defense patterns slice
        
        # Weight defense by piece value
        weighted_defense = defense_patterns.mean(dim=2) * piece_values
        
        return weighted_defense.sum(dim=1)  # Total weighted defense per position 