#!/usr/bin/env python3
"""
Debug script to analyze high loss issues in the chess model training.
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
from pathlib import Path
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rival_ai.models import ChessGNN
from rival_ai.training.losses import ImprovedPolicyValueLoss, PolicyValueLoss
from rival_ai.data.dataset import ChessDataset, create_dataloader
from rival_ai.utils.board_conversion import board_to_hetero_data
import chess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_batch_loss(model, criterion, batch, device):
    """Analyze loss components for a single batch."""
    model.eval()
    
    with torch.no_grad():
        # Move batch to device
        data = batch['data'].to(device)
        policy_target = batch['policy'].to(device)
        value_target = batch['value'].to(device)
        
        # Forward pass
        policy_pred, value_pred = model(data)
        
        # Analyze predictions
        logger.info(f"Policy prediction shape: {policy_pred.shape}")
        logger.info(f"Value prediction shape: {value_pred.shape}")
        logger.info(f"Policy target shape: {policy_target.shape}")
        logger.info(f"Value target shape: {value_target.shape}")
        
        # Analyze value predictions
        logger.info(f"Value pred range: [{value_pred.min():.4f}, {value_pred.max():.4f}]")
        logger.info(f"Value target range: [{value_target.min():.4f}, {value_target.max():.4f}]")
        
        # Analyze policy predictions
        policy_probs = F.softmax(policy_pred, dim=-1)
        logger.info(f"Policy probs range: [{policy_probs.min():.4f}, {policy_probs.max():.4f}]")
        logger.info(f"Policy probs sum: {policy_probs.sum(dim=-1).mean():.4f}")
        
        # Analyze policy targets
        if policy_target.dim() == 2:
            logger.info(f"Policy target sum: {policy_target.sum(dim=-1).mean():.4f}")
            logger.info(f"Policy target max: {policy_target.max(dim=-1)[0].mean():.4f}")
        else:
            logger.info(f"Policy target (class indices) range: [{policy_target.min()}, {policy_target.max()}]")
        
        # Calculate individual loss components
        if isinstance(criterion, ImprovedPolicyValueLoss):
            # Use improved loss
            loss, components = criterion(policy_pred, value_pred, policy_target, value_target, model)
        else:
            # Use standard loss
            loss, components = criterion(policy_pred, value_pred, policy_target, value_target, model)
        
        logger.info(f"Total loss: {loss.item():.4f}")
        for name, value in components.items():
            logger.info(f"  {name}: {value:.4f}")
        
        return loss.item(), components

def analyze_dataset(dataset_path, model_path=None, device='cuda'):
    """Analyze a dataset to understand the loss distribution."""
    logger.info(f"Analyzing dataset: {dataset_path}")
    
    # Load dataset
    try:
        dataset = ChessDataset.from_pickle(dataset_path)
        logger.info(f"Dataset loaded with {len(dataset)} samples")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return
    
    # Create dataloader
    dataloader = create_dataloader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    # Load model if provided
    model = None
    if model_path and Path(model_path).exists():
        try:
            model = ChessGNN()
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return
    
    # Initialize loss function
    criterion = ImprovedPolicyValueLoss()
    
    # Analyze first few batches
    total_losses = []
    policy_losses = []
    value_losses = []
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 5:  # Analyze first 5 batches
            break
            
        logger.info(f"\n--- Batch {batch_idx + 1} ---")
        
        try:
            loss, components = analyze_batch_loss(model, criterion, batch, device)
            total_losses.append(loss)
            policy_losses.append(components.get('policy_loss', 0))
            value_losses.append(components.get('value_loss', 0))
        except Exception as e:
            logger.error(f"Error analyzing batch {batch_idx}: {e}")
            continue
    
    # Summary statistics
    if total_losses:
        logger.info(f"\n--- Summary Statistics ---")
        logger.info(f"Average total loss: {np.mean(total_losses):.4f} ± {np.std(total_losses):.4f}")
        logger.info(f"Average policy loss: {np.mean(policy_losses):.4f} ± {np.std(policy_losses):.4f}")
        logger.info(f"Average value loss: {np.mean(value_losses):.4f} ± {np.std(value_losses):.4f}")
        logger.info(f"Loss range: [{np.min(total_losses):.4f}, {np.max(total_losses):.4f}]")

def analyze_self_play_data(self_play_dir):
    """Analyze self-play data to understand target quality."""
    logger.info(f"Analyzing self-play data in: {self_play_dir}")
    
    self_play_path = Path(self_play_dir)
    if not self_play_path.exists():
        logger.error(f"Self-play directory does not exist: {self_play_dir}")
        return
    
    # Find pickle files
    pkl_files = list(self_play_path.glob('*.pkl'))
    if not pkl_files:
        logger.error(f"No pickle files found in {self_play_dir}")
        return
    
    # Analyze the most recent file
    latest_file = max(pkl_files, key=lambda x: x.stat().st_mtime)
    logger.info(f"Analyzing latest file: {latest_file}")
    
    try:
        import pickle
        with open(latest_file, 'rb') as f:
            games = pickle.load(f)
        
        logger.info(f"Loaded {len(games)} games")
        
        # Analyze game results
        results = [game.result for game in games if game.result is not None]
        if results:
            from rival_ai.chess import GameResult
            white_wins = sum(1 for r in results if r == GameResult.WHITE_WINS)
            black_wins = sum(1 for r in results if r == GameResult.BLACK_WINS)
            draws = sum(1 for r in results if r == GameResult.DRAW)
            total = len(results)
            
            logger.info(f"Game results:")
            logger.info(f"  White wins: {white_wins}/{total} ({white_wins/total:.1%})")
            logger.info(f"  Black wins: {black_wins}/{total} ({black_wins/total:.1%})")
            logger.info(f"  Draws: {draws}/{total} ({draws/total:.1%})")
        
        # Analyze move counts
        move_counts = [len(game.moves) for game in games]
        if move_counts:
            logger.info(f"Move counts:")
            logger.info(f"  Average: {np.mean(move_counts):.1f} ± {np.std(move_counts):.1f}")
            logger.info(f"  Range: [{np.min(move_counts)}, {np.max(move_counts)}]")
        
        # Analyze policy distributions
        if games and hasattr(games[0], 'policies') and games[0].policies:
            all_policies = []
            for game in games:
                if hasattr(game, 'policies') and game.policies:
                    all_policies.extend(game.policies)
            
            if all_policies:
                policies_tensor = torch.stack(all_policies)
                logger.info(f"Policy analysis:")
                logger.info(f"  Shape: {policies_tensor.shape}")
                logger.info(f"  Range: [{policies_tensor.min():.4f}, {policies_tensor.max():.4f}]")
                logger.info(f"  Mean: {policies_tensor.mean():.4f}")
                logger.info(f"  Std: {policies_tensor.std():.4f}")
                logger.info(f"  Non-zero elements: {(policies_tensor > 0).sum()}/{policies_tensor.numel()} ({(policies_tensor > 0).float().mean():.1%})")
        
        # Analyze value distributions
        if games and hasattr(games[0], 'values') and games[0].values:
            all_values = []
            for game in games:
                if hasattr(game, 'values') and game.values:
                    all_values.extend(game.values)
            
            if all_values:
                values_tensor = torch.tensor(all_values)
                logger.info(f"Value analysis:")
                logger.info(f"  Shape: {values_tensor.shape}")
                logger.info(f"  Range: [{values_tensor.min():.4f}, {values_tensor.max():.4f}]")
                logger.info(f"  Mean: {values_tensor.mean():.4f}")
                logger.info(f"  Std: {values_tensor.std():.4f}")
        
    except Exception as e:
        logger.error(f"Error analyzing self-play data: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function to run loss analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug high loss issues in chess model training")
    parser.add_argument("--dataset", type=str, help="Path to dataset pickle file")
    parser.add_argument("--model", type=str, help="Path to model checkpoint")
    parser.add_argument("--self-play-dir", type=str, help="Path to self-play data directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    if args.dataset:
        analyze_dataset(args.dataset, args.model, args.device)
    
    if args.self_play_dir:
        analyze_self_play_data(args.self_play_dir)
    
    if not args.dataset and not args.self_play_dir:
        logger.error("Please provide either --dataset or --self-play-dir")
        return

if __name__ == "__main__":
    main() 