#!/usr/bin/env python3
"""
Test script to verify aggressive play improvements.
"""

import torch
import numpy as np
import logging
from pathlib import Path
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rival_ai.training.self_play import SelfPlayConfig
from rival_ai.models import ChessGNN
from rival_ai.chess import GameResult
import chess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_aggressive_config():
    """Test the new aggressive configuration."""
    config = SelfPlayConfig()
    
    logger.info("Testing aggressive self-play configuration:")
    logger.info(f"Max moves: {config.max_moves}")
    logger.info(f"Opening temperature: {config.opening_temperature}")
    logger.info(f"Midgame temperature: {config.midgame_temperature}")
    logger.info(f"Endgame temperature: {config.endgame_temperature}")
    logger.info(f"Capture bonus: {config.capture_bonus}")
    logger.info(f"Check bonus: {config.check_bonus}")
    logger.info(f"Attack bonus: {config.attack_bonus}")
    logger.info(f"Development bonus: {config.development_bonus}")
    logger.info(f"Force capture after moves: {config.force_capture_after_moves}")
    logger.info(f"Force attack after moves: {config.force_attack_after_moves}")
    logger.info(f"Force win attempt after moves: {config.force_win_attempt_after_moves}")
    
    return config

def test_aggressive_play_bonus():
    """Test the aggressive play bonus function."""
    # Create a simple board position
    board = chess.Board()
    board.push_san("e4")
    board.push_san("e5")
    board.push_san("Nf3")
    board.push_san("Nc6")
    
    # Create a dummy policy
    policy = torch.zeros(5312, dtype=torch.float32)
    
    # Set some base probabilities for legal moves
    legal_moves = list(board.legal_moves)
    for move in legal_moves:
        # Find move index (simplified)
        from_sq = move.from_square
        to_sq = move.to_square
        move_idx = from_sq * 64 + to_sq
        if move_idx < 5312:
            policy[move_idx] = 1.0
    
    # Create config
    config = SelfPlayConfig()
    
    # Test the aggressive play bonus function directly
    def _apply_aggressive_play_bonus(board, policy, move_count):
        """Test version of aggressive play bonus."""
        policy = policy.clone()
        
        # Get legal moves
        legal_moves = list(board.legal_moves)
        
        for move in legal_moves:
            # Simplified move index calculation
            from_sq = move.from_square
            to_sq = move.to_square
            move_idx = from_sq * 64 + to_sq
            if move_idx < 5312:
                bonus = 0.0
                
                # Capture bonus
                if board.is_capture(move):
                    bonus += config.capture_bonus
                    
                    # Extra bonus for capturing with less valuable pieces
                    captured_piece = board.piece_at(move.to_square)
                    moving_piece = board.piece_at(move.from_square)
                    if captured_piece and moving_piece:
                        piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, 
                                      chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
                        captured_value = piece_values.get(captured_piece.piece_type, 0)
                        moving_value = piece_values.get(moving_piece.piece_type, 0)
                        if moving_value < captured_value:
                            bonus += config.capture_bonus * 0.5  # Extra bonus for good captures
                
                # Check bonus
                if board.gives_check(move):
                    bonus += config.check_bonus
                
                # Development bonus (in opening)
                if move_count <= config.opening_moves:
                    if _is_development_move(board, move):
                        bonus += config.development_bonus
                
                # Apply bonus
                policy[move_idx] += bonus
        
        return policy
    
    def _is_development_move(board, move):
        """Check if a move develops a piece in the opening."""
        piece = board.piece_at(move.from_square)
        if not piece:
            return False
        
        # Consider knight and bishop moves as development
        if piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
            # Check if moving from starting square
            if piece.color == chess.WHITE:
                starting_squares = {
                    chess.KNIGHT: [chess.B1, chess.G1],
                    chess.BISHOP: [chess.C1, chess.F1]
                }
            else:
                starting_squares = {
                    chess.KNIGHT: [chess.B8, chess.G8],
                    chess.BISHOP: [chess.C8, chess.F8]
                }
            
            if move.from_square in starting_squares.get(piece.piece_type, []):
                return True
        
        return False
    
    # Apply aggressive play bonus
    enhanced_policy = _apply_aggressive_play_bonus(board, policy, 5)
    
    logger.info(f"Original policy sum: {policy.sum():.4f}")
    logger.info(f"Enhanced policy sum: {enhanced_policy.sum():.4f}")
    logger.info(f"Policy enhancement: {enhanced_policy.sum() - policy.sum():.4f}")
    
    # Check if captures got bonuses
    capture_moves = [move for move in legal_moves if board.is_capture(move)]
    if capture_moves:
        logger.info(f"Found {len(capture_moves)} capture moves")
        for move in capture_moves:
            from_sq = move.from_square
            to_sq = move.to_square
            move_idx = from_sq * 64 + to_sq
            if move_idx < 5312:
                original_prob = policy[move_idx]
                enhanced_prob = enhanced_policy[move_idx]
                logger.info(f"Capture {move}: {original_prob:.4f} -> {enhanced_prob:.4f} (+{enhanced_prob - original_prob:.4f})")

def test_dynamic_temperature():
    """Test dynamic temperature calculation."""
    config = SelfPlayConfig()
    
    def _get_dynamic_temperature(move_count):
        """Test version of dynamic temperature."""
        if move_count <= config.opening_moves:
            return config.opening_temperature
        elif move_count <= config.midgame_moves:
            return config.midgame_temperature
        else:
            return config.endgame_temperature
    
    logger.info("Testing dynamic temperature:")
    for move_count in [5, 20, 60, 100]:
        temp = _get_dynamic_temperature(move_count)
        phase = "opening" if move_count <= config.opening_moves else "midgame" if move_count <= config.midgame_moves else "endgame"
        logger.info(f"Move {move_count} ({phase}): temperature = {temp:.2f}")

def test_force_decisive_play():
    """Test the force decisive play function."""
    # Create a board with many pieces
    board = chess.Board()
    
    # Create a dummy policy
    policy = torch.ones(5312, dtype=torch.float32) * 0.1
    
    config = SelfPlayConfig()
    
    def _force_decisive_play(board, policy, move_count):
        """Test version of force decisive play."""
        policy = policy.clone()
        
        # Force captures after certain moves
        if move_count >= config.force_capture_after_moves:
            legal_moves = list(board.legal_moves)
            capture_moves = [move for move in legal_moves if board.is_capture(move)]
            
            if capture_moves:
                # Boost capture moves significantly
                for move in capture_moves:
                    from_sq = move.from_square
                    to_sq = move.to_square
                    move_idx = from_sq * 64 + to_sq
                    if move_idx < 5312:
                        policy[move_idx] *= 3.0  # Triple the probability
        
        return policy
    
    # Test at different move counts
    for move_count in [10, 25, 55]:
        enhanced_policy = _force_decisive_play(board, policy, move_count)
        
        # Check if policy was modified
        policy_change = (enhanced_policy - policy).abs().sum()
        logger.info(f"Move {move_count}: policy change = {policy_change:.4f}")
        
        if policy_change > 0:
            logger.info(f"  Policy was modified at move {move_count}")

def test_move_index_conversion():
    """Test move index conversion functions."""
    logger.info("Testing move index conversion:")
    
    # Test some basic moves
    test_moves = [
        ("e2e4", 12 * 64 + 28),  # e2 to e4
        ("g1f3", 6 * 64 + 21),   # g1 to f3
        ("e7e5", 52 * 64 + 36),  # e7 to e5
    ]
    
    for move_str, expected_idx in test_moves:
        from_sq = chess.parse_square(move_str[:2])
        to_sq = chess.parse_square(move_str[2:4])
        move_idx = from_sq * 64 + to_sq
        logger.info(f"Move {move_str}: {from_sq}*64 + {to_sq} = {move_idx} (expected: {expected_idx})")

def test_board_analysis():
    """Test board analysis functions."""
    logger.info("Testing board analysis:")
    
    # Create a position with potential captures
    board = chess.Board()
    board.push_san("e4")
    board.push_san("e5")
    board.push_san("Nf3")
    board.push_san("Nc6")
    board.push_san("Bc4")
    board.push_san("Nf6")
    
    legal_moves = list(board.legal_moves)
    capture_moves = [move for move in legal_moves if board.is_capture(move)]
    check_moves = [move for move in legal_moves if board.gives_check(move)]
    
    logger.info(f"Position after 6 moves:")
    logger.info(f"Legal moves: {len(legal_moves)}")
    logger.info(f"Capture moves: {len(capture_moves)}")
    logger.info(f"Check moves: {len(check_moves)}")
    
    if capture_moves:
        logger.info("Capture moves found:")
        for move in capture_moves[:3]:  # Show first 3
            logger.info(f"  {move}")
    
    if check_moves:
        logger.info("Check moves found:")
        for move in check_moves[:3]:  # Show first 3
            logger.info(f"  {move}")

def main():
    """Main function to run all tests."""
    logger.info("Testing aggressive play improvements...")
    
    # Test configuration
    test_aggressive_config()
    print()
    
    # Test board analysis
    test_board_analysis()
    print()
    
    # Test move index conversion
    test_move_index_conversion()
    print()
    
    # Test aggressive play bonus
    test_aggressive_play_bonus()
    print()
    
    # Test dynamic temperature
    test_dynamic_temperature()
    print()
    
    # Test force decisive play
    test_force_decisive_play()
    print()
    
    logger.info("All tests completed!")

if __name__ == "__main__":
    main() 