#!/usr/bin/env python3
"""
Extract training metrics from TensorBoard logs to diagnose training issues.
"""

import os
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd

def extract_metrics(log_dir):
    """Extract metrics from TensorBoard logs."""
    print(f"📊 Analyzing training logs in: {log_dir}")
    
    # Find event files
    event_files = []
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if 'tfevents' in file and file.endswith('.0'):
                event_files.append(os.path.join(root, file))
    
    if not event_files:
        print("❌ No TensorBoard event files found!")
        return
    
    print(f"📂 Found {len(event_files)} event files")
    
    for event_file in event_files:
        print(f"\n🔍 Analyzing: {event_file}")
        
        # Load events
        ea = EventAccumulator(event_file)
        ea.Reload()
        
        # Print available tags
        print("\n📈 Available metrics:")
        tags = ea.Tags()
        for key in ['scalars', 'histograms', 'images']:
            if key in tags and tags[key]:
                print(f"  {key}: {tags[key]}")
        
        # Extract scalar metrics
        if 'scalars' in tags:
            print("\n📊 Training Progress:")
            
            key_metrics = ['total_loss', 'policy_loss', 'value_loss', 'material_balance_loss', 
                          'entropy', 'learning_rate']
            
            for metric in key_metrics:
                if metric in tags['scalars']:
                    events = ea.Scalars(metric)
                    if events:
                        values = [e.value for e in events]
                        steps = [e.step for e in events]
                        print(f"  {metric}:")
                        print(f"    Initial: {values[0]:.6f}")
                        print(f"    Final: {values[-1]:.6f}")
                        print(f"    Change: {values[-1] - values[0]:.6f}")
                        print(f"    Steps: {len(steps)}")
                        
                        # Check for concerning patterns
                        if metric.endswith('_loss'):
                            if abs(values[-1] - values[0]) < 0.001:
                                print(f"    ⚠️ WARNING: {metric} barely changed!")
                            elif values[-1] > values[0]:
                                print(f"    ❌ ERROR: {metric} increased (training degraded)!")
                            else:
                                print(f"    ✅ {metric} decreased (good)")
                        
                        if metric == 'learning_rate':
                            if values[0] == values[-1]:
                                print(f"    ⚠️ WARNING: Learning rate didn't decay")
                            if values[0] > 0.01:
                                print(f"    ⚠️ WARNING: Learning rate might be too high")
        
        # Check for specific chess-related issues
        print("\n🔍 Chess-Specific Analysis:")
        
        # Material balance loss is critical for not hanging pieces
        if 'material_balance_loss' in tags.get('scalars', []):
            events = ea.Scalars('material_balance_loss')
            values = [e.value for e in events]
            if values[-1] > 0.5:
                print("  ❌ CRITICAL: Material balance loss is high - model doesn't value pieces!")
            elif values[-1] > 0.1:
                print("  ⚠️ WARNING: Material balance loss is moderate - some tactical issues expected")
            else:
                print("  ✅ Material balance loss is low - good tactical foundation")
        else:
            print("  ❌ CRITICAL: No material_balance_loss found - this is essential for tactics!")
        
        # Value loss indicates position evaluation accuracy
        if 'value_loss' in tags.get('scalars', []):
            events = ea.Scalars('value_loss')
            values = [e.value for e in events]
            if values[-1] > 1.0:
                print("  ❌ CRITICAL: Value loss is very high - model can't evaluate positions!")
            elif values[-1] > 0.5:
                print("  ⚠️ WARNING: Value loss is high - position evaluation is poor")
            else:
                print("  ✅ Value loss is reasonable")
        
        # Policy loss indicates move prediction accuracy  
        if 'policy_loss' in tags.get('scalars', []):
            events = ea.Scalars('policy_loss')
            values = [e.value for e in events]
            if values[-1] > 8.0:
                print("  ❌ CRITICAL: Policy loss is very high - model makes random moves!")
            elif values[-1] > 5.0:
                print("  ⚠️ WARNING: Policy loss is high - move selection is poor")
            else:
                print("  ✅ Policy loss is reasonable")

if __name__ == "__main__":
    # Analyze the most recent training session
    log_dir = "experiments/ultra_dense_pag_training_1750606370/logs"
    
    if os.path.exists(log_dir):
        extract_metrics(log_dir)
    else:
        print(f"❌ Log directory not found: {log_dir}")
        
    # Also check the earlier session
    log_dir2 = "experiments/ultra_dense_pag_training_1750604853/logs"
    if os.path.exists(log_dir2):
        print(f"\n" + "="*60)
        extract_metrics(log_dir2) 