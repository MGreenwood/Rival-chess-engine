"""
Monte Carlo Tree Search implementation for chess.
"""

import math
import logging
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import torch
import numpy as np
import chess
import torch_geometric.data
import torch.nn.functional as F

from rival_ai.models import ChessGNN
from rival_ai.chess import Color, GameResult, Move, PieceType
from rival_ai.utils.board_conversion import board_to_hetero_data

logger = logging.getLogger(__name__)

@dataclass
class MCTSConfig:
    """Configuration for MCTS."""
    
    num_simulations: int = 800
    c_puct: float = 1.0
    dirichlet_alpha: float = 0.3
    dirichlet_weight: float = 0.25
    temperature: float = 1.0
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_batch_size: int = 32
    use_amp: bool = True
    num_parallel_streams: int = 8
    max_table_size: int = 1_000_000
    log_timing: bool = True
    batch_size: int = 512
    max_time: float = 10.0  # Added for time-based termination

class MCTSNode:
    """Node in the MCTS search tree."""
    
    def __init__(self, prior: float = 0.0):
        """Initialize a new MCTS node.
        
        Args:
            prior: Prior probability for this node (float)
        """
        self.children = {}  # Maps moves to child nodes
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = float(prior)  # Store prior as a float
        self.is_expanded = False
        self.parent = None  # Add parent reference for backpropagation

    def expand(self, move_priors: Dict[int, float], value: float):
        """Expand the node with children for all legal moves.
        
        Args:
            move_priors: Dictionary mapping move indices to prior probabilities
            value: Value estimate for this position
        """
        self.is_expanded = True
        self.value_sum = value
        self.visit_count = 1
        
        # Create children for all legal moves
        for move_idx, prior in move_priors.items():
            self.children[move_idx] = MCTSNode(prior=float(prior))

    def get_policy(self) -> Dict[int, float]:
        """Convert visit counts to a policy distribution.
        
        Returns:
            Dict[int, float]: Dictionary mapping move indices to probabilities
        """
        if not self.children:
            return {}
            
        # Get total visits
        total_visits = sum(child.visit_count for child in self.children.values())
        
        if total_visits == 0:
            return {move: 1.0 / len(self.children) for move in self.children}
            
        # Convert visit counts to probabilities
        policy = {}
        for move, child in self.children.items():
            policy[move] = child.visit_count / total_visits
            
        return policy
        
    @property
    def value(self) -> float:
        """Get the average value of this node.
        
        Returns:
            float: Average value (-1 to 1)
        """
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

class MCTS:
    """Monte Carlo Tree Search implementation."""
    
    def __init__(self, model: ChessGNN, config: MCTSConfig):
        """Initialize MCTS.
        
        Args:
            model: Neural network model for position evaluation
            config: MCTS configuration
        """
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
        self.model.eval()  # Set model to evaluation mode
        
        # Initialize root node
        self.root = None
        self.nodes = {}  # Maps board hash to node
        
        # Initialize caches
        self.position_cache = {}  # Cache for position evaluations
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Initialize CUDA streams for parallel processing
        self.streams = [torch.cuda.Stream() for _ in range(config.num_parallel_streams)]
        
        # Initialize scaler for mixed precision
        if config.use_amp and torch.cuda.is_available() and hasattr(torch.amp, 'GradScaler'):
            self.scaler = torch.amp.GradScaler()
        else:
            self.scaler = None
            
        # Initialize random state for consistent hashing
        self._init_hash_keys()
        
        self.position_history = []  # Track positions for repetition detection
        self.max_history = 20  # Track more positions
    
    def _init_hash_keys(self):
        """Initialize random keys for Zobrist hashing."""
        import random
        random.seed(42)  # For reproducibility
        
        # Generate random keys for each piece type and square
        self.piece_keys = {}
        for piece_type in chess.PIECE_TYPES:
            for color in [chess.WHITE, chess.BLACK]:
                for square in chess.SQUARES:
                    self.piece_keys[(piece_type, color, square)] = random.getrandbits(64)
        
        # Additional keys for game state
        self.side_to_move_key = random.getrandbits(64)
        self.castling_keys = {
            chess.BB_A1: random.getrandbits(64),  # White queenside
            chess.BB_H1: random.getrandbits(64),  # White kingside
            chess.BB_A8: random.getrandbits(64),  # Black queenside
            chess.BB_H8: random.getrandbits(64)   # Black kingside
        }
        self.en_passant_keys = {square: random.getrandbits(64) for square in chess.SQUARES}
    
    def _get_position_hash(self, board: chess.Board) -> int:
        """Generate a unique hash for a board position using Zobrist hashing.
        
        Args:
            board: Chess board to hash
            
        Returns:
            int: 64-bit hash of the position
        """
        # Start with side to move
        h = self.side_to_move_key if board.turn == chess.WHITE else 0
        
        # Add piece positions
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                h ^= self.piece_keys[(piece.piece_type, piece.color, square)]
        
        # Add castling rights
        if board.has_castling_rights(chess.WHITE):
            if board.has_kingside_castling_rights(chess.WHITE):
                h ^= self.castling_keys[chess.BB_H1]  # White kingside
            if board.has_queenside_castling_rights(chess.WHITE):
                h ^= self.castling_keys[chess.BB_A1]  # White queenside
        if board.has_castling_rights(chess.BLACK):
            if board.has_kingside_castling_rights(chess.BLACK):
                h ^= self.castling_keys[chess.BB_H8]  # Black kingside
            if board.has_queenside_castling_rights(chess.BLACK):
                h ^= self.castling_keys[chess.BB_A8]  # Black queenside
        
        # Add en passant square
        if board.ep_square is not None:
            h ^= self.en_passant_keys[board.ep_square]
            
        return h
    
    def _move_to_move_idx(self, move: chess.Move) -> int:
        """Convert a chess move to a unique index in the range [0, 5311].
        
        The index is calculated as:
        - For non-promotion moves: from_square * 64 + to_square
        - For promotion moves: 4096 + (promotion_piece_type - 2) * 64 + (from_square * 8 + to_square % 8)
        
        Args:
            move: The chess move to convert
            
        Returns:
            Integer index in range [0, 5311]
        """
        # Calculate base index for the move
        from_square = move.from_square
        to_square = move.to_square
        
        if move.promotion:
            # For promotion moves, we only consider moves to the last rank
            # Each promotion type (knight=0, bishop=1, rook=2, queen=3) gets 8 slots
            # We use from_square * 8 + to_square % 8 to get a unique index for each promotion move
            promotion_base = 4096  # Start of promotion moves
            promotion_type = move.promotion - 2  # Convert to 0-3 range
            promotion_offset = promotion_type * 64  # Each type gets 64 slots
            move_offset = (from_square * 8) + (to_square % 8)  # Unique index for each promotion move
            return promotion_base + promotion_offset + move_offset
        else:
            # For regular moves, just use the base index
            return from_square * 64 + to_square
    
    def _move_idx_to_move(self, move_idx: int, board: chess.Board) -> Optional[chess.Move]:
        """Convert a move index back to a chess move.
        
        Args:
            move_idx: Index in range [0, 5311]
            board: Current board position
            
        Returns:
            Optional[chess.Move]: The corresponding move if legal, None otherwise
        """
        if not (0 <= move_idx < 5312):
            return None
            
        if move_idx < 4096:
            # Regular move
            from_square = move_idx // 64
            to_square = move_idx % 64
            move = chess.Move(from_square, to_square)
        else:
            # Promotion move
            base_idx = (move_idx - 4096) // 4
            promotion_type = (move_idx - 4096) % 4 + 2  # Convert back to piece type (2-5)
            from_square = base_idx // 64
            to_square = base_idx % 64
            move = chess.Move(from_square, to_square, promotion=promotion_type)
            
        # Verify the move is legal
        if move in board.legal_moves:
            return move
        return None
    
    def _batch_evaluate(self, nodes: List[Tuple[chess.Board, MCTSNode]]) -> Tuple[Dict[int, float], float]:
        """Evaluate a batch of nodes using the model."""
        if not nodes:
            return {}, 0.0
        
        # Convert boards to graph data
        batched_data = []
        for board, _ in nodes:
            # Check cache first
            fen = board.fen()
            if fen in self.position_cache:
                batched_data.append(self.position_cache[fen])
                continue
            
            # Convert board to graph data
            data = board_to_hetero_data(board)
            data = data.to(self.device)
            
            # Cache the result
            self.position_cache[fen] = data
            batched_data.append(data)
            
            # Maintain cache size
            if len(self.position_cache) > self.config.max_table_size:
                # Remove oldest entries
                keys_to_remove = list(self.position_cache.keys())[:-self.config.max_table_size]
                for key in keys_to_remove:
                    del self.position_cache[key]
        
        # Evaluate batch
        with torch.no_grad(), torch.amp.autocast(device_type='cuda', enabled=self.config.use_amp):  # Add device_type
            policies, values = self.model(batched_data[0])  # Process one at a time for now
        
        return policies, values.item()
    
    def _get_position_key(self, board):
        """Get a unique key for the current position."""
        return board.fen()

    def _would_cause_repetition(self, board, move):
        """Check if a move would cause a repetition."""
        # Make a copy of the board and try the move
        test_board = board.copy()
        test_board.push(move)
        new_pos = self._get_position_key(test_board)
        
        # Count how many times this position has occurred recently
        recent_count = self.position_history.count(new_pos)
        return recent_count > 0

    def _apply_repetition_penalty(self, policy: torch.Tensor, board: chess.Board, move_count: int) -> torch.Tensor:
        """Apply penalty to moves that would cause repetition.
        
        Args:
            policy: Policy tensor
            board: Current board position
            move_count: Current move count
            
        Returns:
            Modified policy tensor
        """
        if move_count < 30:  # Only apply in opening/middlegame
            num_pieces = len(board.piece_map())
            if num_pieces >= 12:  # Only apply when enough pieces are on board
                for move in board.legal_moves:
                    if self._would_cause_repetition(board, move):
                        move_idx = self._move_to_move_idx(move)
                        if move_idx is not None:
                            # Scale penalty based on game phase
                            if move_count < 10:
                                policy[move_idx] *= 0.1  # Very early repetition
                            elif move_count < 20:
                                policy[move_idx] *= 0.2  # Early repetition
                            elif move_count < 30:
                                policy[move_idx] *= 0.3  # Middlegame repetition
                            else:
                                policy[move_idx] *= 0.4  # Late repetition
        return policy

    def search(self, board: chess.Board, root: Optional[MCTSNode] = None, epoch: Optional[int] = None, game_count: Optional[int] = None) -> None:
        """Perform MCTS search from the given position.
        
        Args:
            board: Chess board to search from
            root: Optional root node to start search from (for reuse)
            epoch: Current training epoch (for logging)
            game_count: Current game number being played (for logging)
        """
        # Initialize timing
        start_time = time.time()
        
        # Create or reuse root node
        if root is None:
            root = MCTSNode()
            self.root = root
        else:
            self.root = root
            
        # Initialize search board
        search_board = board.copy()
        move_count = search_board.fullmove_number
        
        # Update position history
        self.position_history.append(self._get_position_key(board))
        if len(self.position_history) > self.max_history:
            self.position_history.pop(0)
        
        # Perform simulations
        for sim in range(self.config.num_simulations):
            # Selection
            node = root
            search_board = board.copy()
            current_move_count = move_count
            
            # Selection phase
            while node.is_expanded and not search_board.is_game_over():
                # Get UCB scores for all children
                ucb_scores = {}
                for move, child in node.children.items():
                    # Calculate UCB score
                    if child.visit_count == 0:
                        ucb_scores[move] = float('inf')
                    else:
                        # UCB = Q + c_puct * P * sqrt(N) / (1 + n)
                        q_value = float(-child.value_sum / child.visit_count)
                        prior = child.prior
                        parent_visits = float(node.visit_count)
                        child_visits = float(child.visit_count)
                        
                        # Calculate UCB score with explicit float arithmetic
                        exploration_term = self.config.c_puct * prior * math.sqrt(parent_visits) / (1.0 + child_visits)
                        ucb_score = q_value + exploration_term
                        ucb_scores[move] = ucb_score
                
                # Select move with highest UCB score
                if not ucb_scores:
                    break
                    
                move = max(ucb_scores.items(), key=lambda x: x[1])[0]
                
                # Make move
                search_board.push(move)
                current_move_count += 1
                node = node.children[move]
            
            # Expansion
            if not search_board.is_game_over() and not node.is_expanded:
                # Get policy and value from neural network
                policy = self._get_policy(search_board)
                legal_mask = self._get_legal_move_mask(search_board)
                
                # Apply Dirichlet noise to root node
                if node == root:
                    policy = self._apply_dirichlet_noise(policy, legal_mask)
                
                # Apply repetition penalty
                policy = self._apply_repetition_penalty(policy, search_board, current_move_count)
                
                # Mask illegal moves and convert to probabilities
                policy = policy.masked_fill(~legal_mask, float('-inf'))
                policy = F.softmax(policy, dim=0)
                
                # Expand node
                node.is_expanded = True
                
                # Create children for all legal moves
                for move in search_board.legal_moves:
                    move_idx = self._move_to_move_idx(move)
                    if move_idx is not None:
                        prior_value = float(policy[move_idx].item())
                        node.children[move] = MCTSNode(prior=prior_value)
            
            # Evaluation
            if search_board.is_game_over():
                # Game is over, use actual result with move-count-aware rewards
                if search_board.is_checkmate():
                    # Reward quick checkmates, penalize quick checkmate losses
                    if current_move_count <= 20:
                        value = -2.0 if search_board.turn else 2.0  # Double value for quick checkmate
                    elif current_move_count <= 30:
                        value = -1.5 if search_board.turn else 1.5  # 50% bonus for early checkmate
                    else:
                        value = -1.0 if search_board.turn else 1.0  # Normal checkmate value
                elif search_board.is_stalemate():
                    value = -0.1  # Small penalty for stalemate
                elif search_board.is_insufficient_material():
                    value = 0.0  # Neutral for insufficient material
                elif search_board.is_repetition():
                    # Strong penalties for repetition draws
                    if current_move_count < 10:
                        value = -3.0  # Very early repetition
                    elif current_move_count < 20:
                        value = -2.5  # Early repetition
                    elif current_move_count < 30:
                        value = -2.0  # Middlegame repetition
                    else:
                        value = -1.5  # Late repetition
                else:
                    value = 0.0  # Neutral for other legitimate draws
            else:
                # Get value from neural network
                value = self._get_value(search_board)
            
            # Backpropagation
            while node is not None:
                node.visit_count += 1
                node.value_sum += value
                value = -value  # Negamax
                node = node.parent
                
            # Check if we've exceeded time limit
            if time.time() - start_time > self.config.max_time:
                break
        
        # Record timing
        end_time = time.time()
        total_time = end_time - start_time
        
        # Log timing information
        if self.config.log_timing:
            logger.debug(f"MCTS search completed in {total_time:.3f}s "
                        f"({self.config.num_simulations} simulations, "
                        f"{self.config.num_simulations/total_time:.1f} sims/s)")
            logger.debug(f"Cache hit rate: {self._get_cache_hit_rate():.1%}")
        
        # Log timing metrics with epoch and game information
        if self.config.log_timing:
            epoch_info = f"Epoch {epoch + 1}" if epoch is not None else "No epoch"
            game_info = f"Game {game_count}" if game_count is not None else "No game"
            logger.info(f"{epoch_info} - {game_info} - Cache size: {len(self.position_cache)}")
    
    def get_action_policy(self, board: chess.Board, epoch: Optional[int] = None, game_count: Optional[int] = None) -> Tuple[Dict[int, float], float]:
        """Get action probabilities and value for a board position.
        
        Args:
            board: Current board position
            epoch: Current training epoch (for logging)
            game_count: Current game number being played (for logging)
            
        Returns:
            Tuple of (policy, value) where policy is a dictionary mapping move indices to probabilities
        """
        # Create root node
        root = MCTSNode()
        
        # Run MCTS search
        self.search(board, root, epoch, game_count)
        
        # Get policy and value from root
        policy = root.get_policy()
        value = root.value
        
        return policy, value
    
    def get_best_move(self, temperature: float = 0.0) -> chess.Move:
        """Get best move based on visit counts and temperature.
        
        Args:
            temperature: Temperature for move selection (0.0 for deterministic)
            
        Returns:
            Best move
        """
        if not self.children:
            raise ValueError("No children available for move selection")
        
        if temperature == 0.0:
            # Deterministic: choose move with most visits
            return max(self.children.items(), key=lambda x: x[1].visit_count)[0]
        
        # Probabilistic: sample based on visit counts
        moves = list(self.children.keys())
        visit_counts = torch.tensor([self.children[m].visit_count for m in moves], device=self.device)
        
        # Apply temperature
        probs = (visit_counts ** (1.0 / temperature)).float()
        probs = probs / (probs.sum() + 1e-8)  # Add small epsilon to prevent division by zero
        
        # Sample move
        try:
            move_idx = torch.multinomial(probs, 1).item()
            return moves[move_idx]
        except RuntimeError as e:
            logger.error(f"Error sampling move: {e}")
            logger.error(f"Visit counts: {visit_counts}")
            logger.error(f"Probabilities: {probs}")
            # Fallback to uniform distribution
            probs = torch.ones_like(probs) / len(probs)
            move_idx = torch.multinomial(probs, 1).item()
            return moves[move_idx]

    def _get_cache_hit_rate(self) -> float:
        """Calculate the cache hit rate.
        
        Returns:
            float: Cache hit rate between 0 and 1
        """
        total_requests = self.cache_hits + self.cache_misses
        if total_requests == 0:
            return 0.0
        return self.cache_hits / total_requests
        
    def _get_cached_evaluation(self, board: chess.Board) -> Optional[Tuple[torch.Tensor, float]]:
        """Get cached evaluation for a position if available.
        
        Args:
            board: Chess board to evaluate
            
        Returns:
            Optional[Tuple[torch.Tensor, float]]: Cached (policy, value) if available, None otherwise
        """
        board_hash = self._get_position_hash(board)
        if board_hash in self.position_cache:
            self.cache_hits += 1
            return self.position_cache[board_hash]
        self.cache_misses += 1
        return None
        
    def _cache_evaluation(self, board: chess.Board, policy: torch.Tensor, value: float) -> None:
        """Cache evaluation for a position.
        
        Args:
            board: Chess board that was evaluated
            policy: Policy tensor from evaluation
            value: Value from evaluation
        """
        board_hash = self._get_position_hash(board)
        self.position_cache[board_hash] = (policy, value)
        
        # Limit cache size
        if len(self.position_cache) > 1000000:  # 1 million positions
            # Remove 20% of oldest entries
            num_to_remove = len(self.position_cache) // 5
            keys_to_remove = list(self.position_cache.keys())[:num_to_remove]
            for key in keys_to_remove:
                del self.position_cache[key]
                
    def _get_value(self, board: chess.Board) -> float:
        """Evaluate a board position using the neural network.
        
        Args:
            board: Chess board to evaluate
            
        Returns:
            float: Value estimate from the neural network (-1 to 1)
        """
        # Check cache first
        cached = self._get_cached_evaluation(board)
        if cached is not None:
            _, value = cached
            return value
            
        # Convert board to graph data
        data = board_to_hetero_data(board)
        data = data.to(self.device)
        
        # Get model prediction
        with torch.no_grad():
            policy, value = self.model(data)
            value = value.item()  # Convert to Python float
            
            # Adjust value based on side to move
            if not board.turn:  # If black's turn, negate the value
                value = -value
                
            # Cache the result
            self._cache_evaluation(board, policy, value)
                
        return value
        
    def _get_policy(self, board: chess.Board) -> torch.Tensor:
        """Get policy predictions for all legal moves.
        
        Args:
            board: Chess board to evaluate
            
        Returns:
            torch.Tensor: Policy logits for all legal moves
        """
        # Check cache first
        cached = self._get_cached_evaluation(board)
        if cached is not None:
            policy, _ = cached
            return policy
            
        # Convert board to graph data
        data = board_to_hetero_data(board)
        data = data.to(self.device)
        
        # Get model prediction
        with torch.no_grad():
            policy, value = self.model(data)
            policy = policy.squeeze(0)  # Remove batch dimension
            
            # Cache the result
            self._cache_evaluation(board, policy, value.item())
            
        return policy
        
    def _get_legal_move_mask(self, board: chess.Board) -> torch.Tensor:
        """Create a mask for legal moves.
        
        Args:
            board: Chess board to get legal moves for
            
        Returns:
            torch.Tensor: Boolean mask of shape [5312] where True indicates a legal move
        """
        move_mask = torch.zeros(5312, dtype=torch.bool, device=self.device)
        
        for move in board.legal_moves:
            move_idx = self._move_to_move_idx(move)  # Use the same indexing function
            if 0 <= move_idx < 5312:  # Validate index bounds
                move_mask[move_idx] = True
            else:
                logger.warning(f"Invalid move index {move_idx} for move {move}")
                
        return move_mask
        
    def _apply_dirichlet_noise(self, policy: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
        """Apply Dirichlet noise to the policy for exploration.
        
        Args:
            policy: Policy logits
            legal_mask: Boolean mask of legal moves
            
        Returns:
            torch.Tensor: Policy with Dirichlet noise applied
        """
        # Ensure policy and legal_mask have the same shape
        if policy.dim() == 2 and legal_mask.dim() == 1:
            # Policy has batch dimension [1, 5312], legal_mask is [5312]
            # Remove batch dimension from policy or add to legal_mask
            policy = policy.squeeze(0)  # Remove batch dimension
        elif policy.dim() == 1 and legal_mask.dim() == 2:
            # Policy is [5312], legal_mask has batch dimension [1, 5312]
            legal_mask = legal_mask.squeeze(0)  # Remove batch dimension
        
        # Create Dirichlet noise
        noise = torch.distributions.Dirichlet(
            torch.full((legal_mask.sum(),), self.config.dirichlet_alpha, device=self.device)
        ).sample()
        
        # Apply noise only to legal moves
        policy = policy.clone()
        policy[legal_mask] = (1 - self.config.dirichlet_weight) * policy[legal_mask] + \
                            self.config.dirichlet_weight * noise
            
        return policy 