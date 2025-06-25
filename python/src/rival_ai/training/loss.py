"""
Loss functions for training the chess model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Union, List
from dataclasses import dataclass
import chess

@dataclass
class LossConfig:
    """Configuration for loss functions."""
    policy_weight: float = 1.0
    value_weight: float = 1.0
    entropy_weight: float = 0.01
    mse_weight: float = 1.0
    kl_weight: float = 0.1
    pag_weight: float = 2.0  # 🎯 Increased from 0.2 to 2.0 - MUCH stronger PAG learning!
    graph_weight: float = 1.0  # 🎯 Increased from 0.2 to 1.0 - stronger graph structure learning!
    king_safety_weight: float = 1.5  # 🎯 Increased from 0.15 to 1.5 - CRITICAL for safety!
    center_control_weight: float = 1.0  # 🎯 Increased from 0.15 to 1.0 - important for positional play!
    material_balance_weight: float = 5.0  # 🎯 CRITICAL: Increased from 2.0 to 5.0 for VERY strong material learning!
    piece_coordination_weight: float = 1.0  # 🎯 Increased from 0.1 to 1.0 - coordination is key!
    attack_pattern_weight: float = 1.5  # 🎯 Increased from 0.1 to 1.5 - tactical patterns are crucial!

class PolicyValueLoss(nn.Module):
    """Combined policy and value loss for chess model training.
    
    This loss function combines:
    1. Policy loss (cross-entropy)
    2. Value loss (MSE)
    3. Entropy regularization
    4. KL divergence (if teacher model provided)
    5. Dynamic repetition penalty
    6. Forward progress reward
    7. Draw penalty
    """
    
    def __init__(
        self,
        policy_weight: float = 1.0,
        value_weight: float = 1.0,
        entropy_weight: float = 0.01,
        repetition_penalty_weight: float = 2.0,  # Increased from 1.0 to 2.0 - stronger anti-repetition
        forward_progress_weight: float = 0.5,  # Increased from 0.2 - encourage active play
        draw_penalty_weight: float = 0.8,  # Increased from 0.3 - stronger draw penalty
        teacher_model: Optional[nn.Module] = None,
        temperature: float = 1.0,
        l2_weight: float = 1e-4
    ):
        """Initialize loss function.
        
        Args:
            policy_weight: Weight for policy loss
            value_weight: Weight for value loss
            entropy_weight: Weight for entropy regularization
            repetition_penalty_weight: Weight for repetition penalty
            forward_progress_weight: Weight for forward progress reward
            draw_penalty_weight: Weight for draw penalty
            teacher_model: Optional teacher model for distillation
            temperature: Temperature for soft targets
            l2_weight: Weight for L2 regularization
        """
        super().__init__()
        self.policy_weight = policy_weight
        self.value_weight = value_weight
        self.entropy_weight = entropy_weight
        self.repetition_penalty_weight = repetition_penalty_weight
        self.forward_progress_weight = forward_progress_weight
        self.draw_penalty_weight = draw_penalty_weight
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.l2_weight = l2_weight
        
        # Freeze teacher model if provided
        if teacher_model is not None:
            for param in teacher_model.parameters():
                param.requires_grad = False

    def _compute_draw_penalty(self, move_count: int, num_pieces: int) -> torch.Tensor:
        """Compute penalty for draws based on game phase."""
        if move_count < 20:  # Opening
            return torch.tensor(0.3, device=self.device)
        elif move_count < 40:  # Middlegame
            return torch.tensor(0.2, device=self.device)
        elif num_pieces <= 6:  # Endgame with few pieces
            return torch.tensor(0.0, device=self.device)  # No penalty in endgame
        else:
            return torch.tensor(0.1, device=self.device)  # Small penalty in other endgames

    def _compute_repetition_penalty(
        self,
        policy_pred: torch.Tensor,
        board_states: List[chess.Board],
        move_count: int
    ) -> torch.Tensor:
        """Compute penalty for moves that lead to repeated positions with dynamic scaling."""
        penalty = torch.tensor(0.0, device=policy_pred.device)
        
        # Only apply penalty in opening/middlegame
        if move_count < 40:  # Extended from 30 to 40
            # Get current position
            current_board = board_states[-1]
            current_fen = current_board.fen()
            num_pieces = len(current_board.piece_map())
            
            # Check last 10 positions for repetition
            for prev_board in board_states[-11:-1]:  # Skip current position
                if prev_board.fen() == current_fen:
                    # Find moves that would lead to this repeated position
                    for move in current_board.legal_moves:
                        move_idx = self._move_to_index(move)
                        if move_idx is not None:
                            # Scale penalty based on game phase and piece count
                            phase_scale = 1.0
                            if move_count < 10:  # Very early game
                                phase_scale = 3.0  # Much stronger penalty in very early game
                            elif move_count < 20:  # Opening
                                phase_scale = 2.5  # Increased from 1.5
                            elif move_count < 40:  # Middlegame
                                phase_scale = 2.0  # Increased from 1.2
                            
                            # Make piece scaling more aggressive
                            piece_scale = min(1.0, (num_pieces / 12.0) ** 2)  # Quadratic scaling
                            
                            # Add exponential scaling based on how many times position is repeated
                            repetition_count = sum(1 for b in board_states[-11:-1] if b.fen() == current_fen)
                            repetition_scale = 2.0 ** (repetition_count - 1)  # Exponential scaling
                            
                            penalty += (
                                policy_pred[move_idx] * 
                                self.repetition_penalty_weight * 
                                phase_scale * 
                                piece_scale *
                                repetition_scale
                            )
        
        return penalty

    def _compute_forward_progress_reward(
        self,
        policy_pred: torch.Tensor,
        board: chess.Board,
        move_count: int
    ) -> torch.Tensor:
        """Compute reward for pawn advancement and piece development."""
        reward = torch.tensor(0.0, device=policy_pred.device)
        
        # Pawn advancement reward (stronger in opening)
        if move_count <= 20:  # Extended from 10 to 20
            for rank in range(8):
                for file in range(8):
                    square = chess.square(file, rank)
                    piece = board.piece_at(square)
                    if piece and piece.piece_type == chess.PAWN:
                        # Calculate progress towards promotion
                        progress = rank / 7.0 if piece.color == chess.WHITE else (7 - rank) / 7.0
                        # Scale reward based on game phase
                        phase_scale = 1.5 if move_count <= 10 else 1.0
                        reward += progress * self.forward_progress_weight * phase_scale
        
        # Piece development reward (only in opening)
        if move_count <= 15:
            for piece_type in [chess.KNIGHT, chess.BISHOP]:
                for square in board.pieces(piece_type, board.turn):
                    # Reward for pieces that have moved from their starting squares
                    if piece_type == chess.KNIGHT and square in [chess.B1, chess.G1, chess.B8, chess.G8]:
                        reward += 0.05
                    elif piece_type == chess.BISHOP and square in [chess.C1, chess.F1, chess.C8, chess.F8]:
                        reward += 0.05
        
        return reward

    def _move_to_index(self, move: chess.Move) -> Optional[int]:
        """Convert a chess move to policy index."""
        from_sq = move.from_square
        to_sq = move.to_square
        promotion = move.promotion
        
        if promotion:
            # Promotion moves are encoded after regular moves
            piece_offset = promotion - 1  # promotion = 1-4, offset = 0-3
            base = 4096 + (from_sq * 64 + to_sq) * 4
            return base + piece_offset
        else:
            # Regular moves are encoded as from * 64 + to
            return from_sq * 64 + to_sq

    def forward(
        self,
        policy_pred: torch.Tensor,
        value_pred: torch.Tensor,
        policy_target: torch.Tensor,
        value_target: torch.Tensor,
        board_states: List[chess.Board],
        move_count: int,
        model: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the combined loss with enhanced anti-draw measures."""
        # Policy loss (cross-entropy)
        policy_loss = F.cross_entropy(policy_pred, policy_target)
        
        # Value loss (MSE)
        value_loss = F.mse_loss(value_pred, value_target)
        
        # Entropy regularization
        entropy = 0.0
        if model is not None:
            with torch.no_grad():
                policy_probs = F.softmax(policy_pred, dim=-1)
                entropy = -torch.sum(policy_probs * torch.log(policy_probs + 1e-10))
        
        # Dynamic repetition penalty
        repetition_penalty = self._compute_repetition_penalty(
            policy_pred,
            board_states,
            move_count
        )
        
        # Forward progress reward
        forward_progress_reward = self._compute_forward_progress_reward(
            policy_pred,
            board_states[-1],
            move_count
        )
        
        # Draw penalty
        draw_penalty = 0.0
        if board_states[-1].is_game_over() and not board_states[-1].is_checkmate():
            num_pieces = len(board_states[-1].piece_map())
            draw_penalty = self._compute_draw_penalty(move_count, num_pieces)
            draw_penalty *= self.draw_penalty_weight
        
        # KL divergence loss (if teacher model provided)
        kl_loss = 0.0
        if self.teacher_model is not None and model is not None:
            with torch.no_grad():
                teacher_policy, teacher_value = self.teacher_model(model)
                teacher_probs = F.softmax(teacher_policy / self.temperature, dim=-1)
            
            student_log_probs = F.log_softmax(policy_pred / self.temperature, dim=-1)
            kl_loss = F.kl_div(
                student_log_probs,
                teacher_probs,
                reduction='batchmean'
            ) * (self.temperature ** 2)
        
        # Combine losses
        total_loss = (
            self.policy_weight * policy_loss +
            self.value_weight * value_loss -
            self.entropy_weight * entropy +
            repetition_penalty -
            forward_progress_reward +
            draw_penalty +
            kl_loss
        )
        
        # Return loss components for monitoring
        loss_dict = {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'repetition_penalty': repetition_penalty.item(),
            'forward_progress_reward': forward_progress_reward.item(),
            'draw_penalty': draw_penalty.item() if isinstance(draw_penalty, torch.Tensor) else draw_penalty,
            'kl_loss': kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss,
        }
        
        return total_loss, loss_dict

class PAGPolicyValueLoss(PolicyValueLoss):
    """Enhanced policy and value loss with PAG features.
    
    This loss function combines multiple components:
    1. Policy prediction loss (cross entropy)
    2. Value prediction loss (MSE)
    3. PAG feature prediction losses:
        - King safety features
        - Center control features
        - Material balance/tension
        - Piece coordination
        - Attack patterns
    4. Graph structure preservation loss
    5. Auxiliary task losses
    """
    
    def __init__(
        self,
        config: Optional[LossConfig] = None,
        teacher_model: Optional[nn.Module] = None,
        temperature: float = 1.0,
    ):
        """Initialize enhanced PAG loss function.
        
        Args:
            config: Loss configuration
            teacher_model: Optional teacher model for distillation
            temperature: Temperature for soft targets
        """
        super().__init__(config.policy_weight, config.value_weight, config.entropy_weight, 0.0, 0.0, teacher_model, temperature, 0.0)
        self.config = config or LossConfig()
        
    def forward(
        self,
        policy_pred: torch.Tensor,
        value_pred: torch.Tensor,
        pag_pred: Dict[str, torch.Tensor],
        policy_target: torch.Tensor,
        value_target: torch.Tensor,
        pag_target: Dict[str, torch.Tensor],
        model: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the enhanced combined loss.
        
        Args:
            policy_pred: Predicted policy logits [batch_size, num_moves]
            value_pred: Predicted values [batch_size, 1]
            pag_pred: Predicted PAG features
            policy_target: Target policy probabilities [batch_size, num_moves]
            value_target: Target values [batch_size, 1]
            pag_target: Target PAG features
            model: Optional model for entropy computation
            
        Returns:
            Tuple of (total_loss, loss_components)
        """
        components = {}
        
        # 1. Policy loss with optional KL divergence from teacher
        if self.teacher_model is not None:
            with torch.no_grad():
                teacher_policy, _ = self.teacher_model(pag_pred['board_state'])
                teacher_policy = F.softmax(teacher_policy / self.temperature, dim=-1)
            
            # KL divergence loss
            policy_loss = F.kl_div(
                F.log_softmax(policy_pred / self.temperature, dim=-1),
                teacher_policy,
                reduction='batchmean'
            ) * (self.temperature ** 2)
        else:
            # Standard cross entropy loss
            policy_loss = F.cross_entropy(policy_pred, policy_target)
        
        components['policy_loss'] = policy_loss.item()
        
        # 2. Value loss with Huber loss for robustness
        value_loss = F.huber_loss(value_pred, value_target, reduction='mean', delta=1.0)
        components['value_loss'] = value_loss.item()
        
        # 3. PAG feature losses
        pag_losses = {}
        
        # 3.1 King safety features
        if 'king_safety' in pag_pred and 'king_safety' in pag_target:
            king_safety_loss = self._compute_king_safety_loss(
                pag_pred['king_safety'],
                pag_target['king_safety']
            )
            pag_losses['king_safety'] = king_safety_loss
            
        # 3.2 Center control features
        if 'center_control' in pag_pred and 'center_control' in pag_target:
            center_control_loss = self._compute_center_control_loss(
                pag_pred['center_control'],
                pag_target['center_control']
            )
            pag_losses['center_control'] = center_control_loss
            
        # 3.3 Material balance/tension features
        if 'material_tension' in pag_pred and 'material_tension' in pag_target:
            material_loss = self._compute_material_balance_loss(
                pag_pred['material_tension'],
                pag_target['material_tension']
            )
            pag_losses['material_balance'] = material_loss
            
        # 3.4 Piece coordination features
        if 'coordination' in pag_pred and 'coordination' in pag_target:
            coordination_loss = self._compute_coordination_loss(
                pag_pred['coordination'],
                pag_target['coordination']
            )
            pag_losses['coordination'] = coordination_loss
            
        # 3.5 Attack pattern features
        if 'attack_patterns' in pag_pred and 'attack_patterns' in pag_target:
            attack_loss = self._compute_attack_pattern_loss(
                pag_pred['attack_patterns'],
                pag_target['attack_patterns']
            )
            pag_losses['attack_patterns'] = attack_loss
        
        # Combine PAG losses with weights
        total_pag_loss = (
            self.config.king_safety_weight * pag_losses.get('king_safety', 0.0) +
            self.config.center_control_weight * pag_losses.get('center_control', 0.0) +
            self.config.material_balance_weight * pag_losses.get('material_balance', 0.0) +
            self.config.piece_coordination_weight * pag_losses.get('coordination', 0.0) +
            self.config.attack_pattern_weight * pag_losses.get('attack_patterns', 0.0)
        )
        
        components.update({f'pag_{k}_loss': v.item() for k, v in pag_losses.items()})
        
        # 4. Graph structure preservation loss
        graph_loss = self._compute_graph_structure_loss(pag_pred, pag_target)
        components['graph_loss'] = graph_loss.item()
        
        # 5. Optional entropy regularization
        entropy_loss = 0.0
        if model is not None and self.config.entropy_weight > 0:
            entropy_loss = -self.config.entropy_weight * self._compute_entropy(policy_pred)
            components['entropy_loss'] = entropy_loss.item()
        
        # Combine all losses
        total_loss = (
            self.config.policy_weight * policy_loss +
            self.config.value_weight * value_loss +
            self.config.pag_weight * total_pag_loss +
            self.config.graph_weight * graph_loss +
            entropy_loss
        )
        
        components['total_loss'] = total_loss.item()
        return total_loss, components
        
    def _compute_king_safety_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute specialized loss for king safety features."""
        # Weight loss higher for critical defensive features
        weights = torch.ones_like(pred)
        weights[:, :3] = 2.0  # Higher weight for immediate threat features
        
        return F.mse_loss(pred * weights, target * weights)
        
    def _compute_center_control_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute specialized loss for center control features."""
        # Use smooth L1 loss for robustness
        return F.smooth_l1_loss(pred, target)
        
    def _compute_material_balance_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute specialized loss for material balance features."""
        # Use weighted MSE with higher weights for major pieces
        weights = torch.ones_like(pred)
        weights[:, -2:] = 2.0  # Higher weight for queen and rook features
        
        return F.mse_loss(pred * weights, target * weights)
        
    def _compute_coordination_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute specialized loss for piece coordination features."""
        # Combine MSE and cosine similarity loss
        mse_loss = F.mse_loss(pred, target)
        cos_loss = 1 - F.cosine_similarity(pred, target).mean()
        
        return 0.7 * mse_loss + 0.3 * cos_loss
        
    def _compute_attack_pattern_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute specialized loss for attack pattern features."""
        # Use focal loss for imbalanced attack patterns
        gamma = 2.0
        alpha = 0.25
        
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        p_t = torch.exp(-ce_loss)
        loss = alpha * (1 - p_t) ** gamma * ce_loss
        
        return loss.mean()
        
    def _compute_graph_structure_loss(
        self,
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute graph structure preservation loss."""
        if 'edge_index' not in pred or 'edge_index' not in target:
            return torch.tensor(0.0, device=pred[list(pred.keys())[0]].device)
            
        # Convert edge indices to adjacency matrices
        pred_adj = torch.sparse_coo_tensor(
            pred['edge_index'],
            torch.ones(pred['edge_index'].size(1)),
            size=(pred['edge_index'].max() + 1,) * 2
        ).to_dense()
        
        target_adj = torch.sparse_coo_tensor(
            target['edge_index'],
            torch.ones(target['edge_index'].size(1)),
            size=(target['edge_index'].max() + 1,) * 2
        ).to_dense()
        
        # Compute structure similarity loss
        mse_loss = F.mse_loss(pred_adj, target_adj)
        
        # Add graph Laplacian loss for connectivity preservation
        pred_laplacian = self._compute_laplacian(pred_adj)
        target_laplacian = self._compute_laplacian(target_adj)
        laplacian_loss = F.mse_loss(pred_laplacian, target_laplacian)
        
        return 0.7 * mse_loss + 0.3 * laplacian_loss
        
    def _compute_laplacian(self, adj: torch.Tensor) -> torch.Tensor:
        """Compute normalized graph Laplacian."""
        degree = adj.sum(dim=1)
        degree_matrix = torch.diag(degree)
        laplacian = degree_matrix - adj
        
        # Normalize
        degree_sqrt_inv = torch.diag(1.0 / torch.sqrt(degree + 1e-8))
        normalized_laplacian = degree_sqrt_inv @ laplacian @ degree_sqrt_inv
        
        return normalized_laplacian
        
    def _compute_entropy(self, policy: torch.Tensor) -> torch.Tensor:
        """Compute policy entropy for regularization."""
        log_policy = F.log_softmax(policy, dim=-1)
        entropy = -(torch.exp(log_policy) * log_policy).sum(dim=-1).mean()
        return entropy

class PAGFeatureLoss(nn.Module):
    """
    Loss function for predicting PAG features.
    
    This loss function focuses on the accuracy of predicting various PAG features:
    - Piece relationships (attacks, defenses, etc.)
    - Square control
    - Mobility
    - Pawn structure
    """
    
    def __init__(
        self,
        feature_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.feature_weights = feature_weights or {
            'control': 1.0,
            'direct_relation': 1.0,
            'mobility': 1.0,
            'cooperation': 0.8,
            'obstruction': 0.8,
            'vulnerability': 0.8,
            'pawn_structure': 1.0,
        }
        
        # Loss functions for different feature types
        self.control_loss = nn.MSELoss()
        self.relationship_loss = nn.BCEWithLogitsLoss()
        self.mobility_loss = nn.MSELoss()
    
    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the PAG feature loss.
        
        Args:
            pred: Dictionary of predicted PAG features
            target: Dictionary of target PAG features
            
        Returns:
            Tuple of:
            - Total loss
            - Dictionary of individual feature losses
        """
        losses = {}
        
        # Control loss
        if 'control' in pred and 'control' in target:
            losses['control'] = self.control_loss(
                pred['control'],
                target['control']
            )
        
        # Relationship losses
        relationship_types = [
            'direct_relation',
            'cooperation',
            'obstruction',
            'vulnerability',
            'pawn_structure',
        ]
        for rel_type in relationship_types:
            if rel_type in pred and rel_type in target:
                losses[rel_type] = self.relationship_loss(
                    pred[rel_type],
                    target[rel_type]
                )
        
        # Mobility loss
        if 'mobility' in pred and 'mobility' in target:
            losses['mobility'] = self.mobility_loss(
                pred['mobility'],
                target['mobility']
            )
        
        # Weight and combine losses
        total_loss = sum(
            self.feature_weights.get(k, 1.0) * v
            for k, v in losses.items()
        )
        
        # Return total loss and individual components
        loss_components = {
            'total': total_loss.item(),
            **{k: v.item() for k, v in losses.items()}
        }
        
        return total_loss, loss_components 

class UltraDensePAGLoss(nn.Module):
    """
    Ultra-dense PAG loss that directly uses 308-dimensional piece features 
    and 95-dimensional critical square features to teach chess principles.
    """
    
    def __init__(self, config: Optional[LossConfig] = None):
        super().__init__()
        self.config = config or LossConfig()
        
        # Feature dimension mappings for 308-dimensional piece features
        self.tactical_dims = slice(0, 76)      # Tactical features (76 dims)
        self.positional_dims = slice(76, 156)  # Positional features (80 dims) 
        self.strategic_dims = slice(156, 216)  # Strategic features (60 dims)
        self.meta_dims = slice(216, 252)       # Meta features (36 dims)
        self.geometric_dims = slice(252, 308)  # Geometric features (56 dims)
        
        # Vulnerability features within tactical (dimensions 60-75)
        self.vulnerability_offset = 60
        self.vulnerability_size = 16
        
        # Activity features within positional (dimensions 52-65)
        self.activity_offset = 52
        self.activity_size = 14
        
        # King safety features within strategic (dimensions 12-23)
        self.king_safety_offset = 12
        self.king_safety_size = 12
    
    def forward(
        self,
        policy_pred: torch.Tensor,
        value_pred: torch.Tensor,
        pag_features: Dict[str, torch.Tensor],
        policy_target: torch.Tensor,
        value_target: torch.Tensor,
        board_states: List,
        model: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute ultra-dense PAG loss with heavy emphasis on chess principles.
        """
        components = {}
        
        # Standard policy and value losses
        policy_loss = F.cross_entropy(policy_pred, policy_target)
        value_loss = F.mse_loss(value_pred, value_target)
        
        components['policy_loss'] = policy_loss.item()
        components['value_loss'] = value_loss.item()
        
        # Extract PAG features if available
        if 'piece_features' in pag_features:
            piece_features = pag_features['piece_features']  # [batch_size, num_pieces, 308]
            
            # 1. HANGING PIECE PENALTY (CRITICAL!)
            hanging_penalty = self._compute_hanging_piece_penalty(piece_features)
            components['hanging_penalty'] = hanging_penalty.item()
            
            # 2. PIECE ACTIVITY BONUS
            activity_bonus = self._compute_activity_bonus(piece_features)
            components['activity_bonus'] = activity_bonus.item()
            
            # 3. KING SAFETY BONUS/PENALTY
            king_safety_loss = self._compute_king_safety_loss(piece_features)
            components['king_safety_loss'] = king_safety_loss.item()
            
            # 4. COORDINATION BONUS
            coordination_bonus = self._compute_coordination_bonus(piece_features)
            components['coordination_bonus'] = coordination_bonus.item()
            
            # 5. TACTICAL PATTERN BONUS
            tactical_bonus = self._compute_tactical_pattern_bonus(piece_features)
            components['tactical_bonus'] = tactical_bonus.item()
        else:
            hanging_penalty = torch.tensor(0.0, device=policy_pred.device)
            activity_bonus = torch.tensor(0.0, device=policy_pred.device)
            king_safety_loss = torch.tensor(0.0, device=policy_pred.device)
            coordination_bonus = torch.tensor(0.0, device=policy_pred.device)
            tactical_bonus = torch.tensor(0.0, device=policy_pred.device)
        
        # Combine all losses with VERY strong PAG weights
        total_loss = (
            self.config.policy_weight * policy_loss +
            self.config.value_weight * value_loss +
            self.config.material_balance_weight * hanging_penalty +  # MASSIVE penalty for hanging pieces
            self.config.piece_coordination_weight * (-coordination_bonus) +  # Bonus for coordination
            self.config.king_safety_weight * king_safety_loss +  # Strong king safety emphasis
            self.config.attack_pattern_weight * (-tactical_bonus) +  # Bonus for tactical patterns
            (-activity_bonus * 0.5)  # Bonus for piece activity
        )
        
        components['total_loss'] = total_loss.item()
        return total_loss, components
    
    def _compute_hanging_piece_penalty(self, piece_features: torch.Tensor) -> torch.Tensor:
        """
        Compute MASSIVE penalty for hanging pieces using vulnerability features.
        piece_features: [batch_size, num_pieces, 308]
        """
        # Extract vulnerability features (tactical dimensions 60-75)
        tactical_features = piece_features[:, :, self.tactical_dims]  # [batch_size, num_pieces, 76]
        vulnerability_start = self.vulnerability_offset
        vulnerability_end = vulnerability_start + self.vulnerability_size
        vulnerability_features = tactical_features[:, :, vulnerability_start:vulnerability_end]  # [batch_size, num_pieces, 16]
        
        # First feature is hanging indicator
        hanging_scores = vulnerability_features[:, :, 0]  # [batch_size, num_pieces]
        
        # Apply sigmoid to get probability of hanging
        hanging_probs = torch.sigmoid(hanging_scores)
        
        # MASSIVE penalty for hanging pieces (scaled by estimated piece value)
        # Use feature magnitude as proxy for piece value
        piece_values = torch.norm(piece_features, dim=2)  # [batch_size, num_pieces]
        piece_values = torch.sigmoid(piece_values)  # Normalize to [0,1]
        
        # Penalty is hanging_probability * piece_value * massive_multiplier
        hanging_penalties = hanging_probs * piece_values * 20.0  # MASSIVE 20x multiplier!
        
        # Sum across pieces and batch
        total_hanging_penalty = torch.mean(torch.sum(hanging_penalties, dim=1))
        
        return total_hanging_penalty
    
    def _compute_activity_bonus(self, piece_features: torch.Tensor) -> torch.Tensor:
        """
        Compute bonus for active pieces using positional activity features.
        """
        # Extract positional features
        positional_features = piece_features[:, :, self.positional_dims]  # [batch_size, num_pieces, 80]
        
        # Extract activity metrics (dimensions 52-65 in positional)
        activity_start = self.activity_offset - 76  # Offset within positional features
        activity_end = activity_start + self.activity_size
        activity_features = positional_features[:, :, activity_start:activity_end]  # [batch_size, num_pieces, 14]
        
        # Average activity score per piece
        piece_activity = torch.mean(activity_features, dim=2)  # [batch_size, num_pieces]
        
        # Apply sigmoid to normalize
        piece_activity = torch.sigmoid(piece_activity)
        
        # Bonus for high activity
        activity_bonus = torch.mean(torch.sum(piece_activity, dim=1))
        
        return activity_bonus
    
    def _compute_king_safety_loss(self, piece_features: torch.Tensor) -> torch.Tensor:
        """
        Compute king safety loss using strategic features.
        """
        # Extract strategic features
        strategic_features = piece_features[:, :, self.strategic_dims]  # [batch_size, num_pieces, 60]
        
        # Extract king safety contribution (dimensions 12-23 in strategic)
        safety_start = self.king_safety_offset - 156  # Offset within strategic features
        safety_end = safety_start + self.king_safety_size
        safety_features = strategic_features[:, :, safety_start:safety_end]  # [batch_size, num_pieces, 12]
        
        # Average safety contribution per piece
        piece_safety = torch.mean(safety_features, dim=2)  # [batch_size, num_pieces]
        
        # We want high safety scores, so penalty is negative safety
        safety_penalty = -torch.mean(torch.sum(piece_safety, dim=1))
        
        return safety_penalty
    
    def _compute_coordination_bonus(self, piece_features: torch.Tensor) -> torch.Tensor:
        """
        Compute bonus for piece coordination using positional coordination features.
        """
        # Extract positional features
        positional_features = piece_features[:, :, self.positional_dims]  # [batch_size, num_pieces, 80]
        
        # Coordination features are dimensions 18-35 in positional (18 features)
        coordination_features = positional_features[:, :, 18:36]  # [batch_size, num_pieces, 18]
        
        # Average coordination score per piece
        piece_coordination = torch.mean(coordination_features, dim=2)  # [batch_size, num_pieces]
        
        # Apply sigmoid
        piece_coordination = torch.sigmoid(piece_coordination)
        
        # Bonus for high coordination
        coordination_bonus = torch.mean(torch.sum(piece_coordination, dim=1))
        
        return coordination_bonus
    
    def _compute_tactical_pattern_bonus(self, piece_features: torch.Tensor) -> torch.Tensor:
        """
        Compute bonus for good tactical patterns using tactical features.
        """
        # Extract tactical features
        tactical_features = piece_features[:, :, self.tactical_dims]  # [batch_size, num_pieces, 76]
        
        # Motif involvement features (dimensions 20-39 in tactical)
        motif_features = tactical_features[:, :, 20:40]  # [batch_size, num_pieces, 20]
        
        # Threat generation features (dimensions 40-51 in tactical)
        threat_features = tactical_features[:, :, 40:52]  # [batch_size, num_pieces, 12]
        
        # Combine motif and threat features
        tactical_combined = torch.cat([motif_features, threat_features], dim=2)  # [batch_size, num_pieces, 32]
        
        # Average tactical score per piece
        piece_tactical = torch.mean(tactical_combined, dim=2)  # [batch_size, num_pieces]
        
        # Apply sigmoid
        piece_tactical = torch.sigmoid(piece_tactical)
        
        # Bonus for high tactical involvement
        tactical_bonus = torch.mean(torch.sum(piece_tactical, dim=1))
        
        return tactical_bonus 