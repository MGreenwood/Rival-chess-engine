# RivalAI Chess Engine

A novel chess engine implementing the Chess Heterogeneous Encoding State System (CHESS) for position representation, combined with Monte Carlo Tree Search (MCTS) and Graph Neural Networks (GNNs) for evaluation and policy decisions. CHESS transforms chess positions into rich graph structures that capture piece relationships, strategic importance, and positional dynamics.

## Project Structure

```
rival_ai/
├── engine/           # Rust core
│   ├── core/        # Core chess logic
│   ├── pag/         # PAG implementation
│   └── search/      # Search implementation
├── training/        # Python training code
│   ├── models/      # GNN implementations
│   ├── data/        # Data handling
│   └── rl/          # Reinforcement learning
└── bridge/          # Rust-Python bridge
```

## Setup Instructions

### Prerequisites

1. Install Rust:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

2. Install Python 3.8+ and create a virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

3. Install required packages:
```bash
# Install all dependencies from requirements.txt
pip install -r python/requirements.txt
```

### Building the Project

1. Build the Rust engine:
```bash
cd engine
cargo build
```

2. Set up Python environment:
```bash
# Make sure your virtual environment is activated
pip install -r requirements.txt
```

## Training

### Basic Training

To train the model with default settings:

```bash
# Make sure your virtual environment is activated
python python/scripts/train.py --experiment-name rival_ai_v1 --tensorboard
```

### Advanced Training

For more control over the training process, you can customize various parameters:

```bash
python python/scripts/train.py \
    --num-epochs 100 \              # Number of training epochs
    --batch-size 32 \               # Training batch size
    --learning-rate 0.001 \         # Initial learning rate
    --weight-decay 1e-4 \           # L2 regularization
    --grad-clip 1.0 \               # Gradient clipping value
    --patience 10 \                 # Early stopping patience (planned)
    --val-split 0.1 \               # Validation split ratio (planned)
    --warmup-epochs 5 \             # Learning rate warmup epochs (planned)
    --num-games 100 \               # Self-play games per epoch
    --num-simulations 800 \         # MCTS simulations per move
    --temperature 1.0 \             # Move selection temperature
    --experiment-name rival_ai_v1 \ # Experiment name for logging
    --tensorboard \                 # Enable TensorBoard logging
    --log-level INFO                # Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
```

### Training Output

The training process generates:
- Model checkpoints in `experiments/<experiment_name>/checkpoints/`
- Self-play games in `self_play_data/`
- Training logs in `experiments/<experiment_name>/logs/`
- TensorBoard visualizations (if enabled)

Current training metrics include:
- Policy loss
- Value loss
- Total loss
- Entropy
- L2 regularization

### Monitoring Training

You can monitor training progress in several ways:

1. **TensorBoard (Recommended)**
   - Real-time visualization of training metrics
   - Access at `http://localhost:6006` after starting tensorboard
   - Shows:
     - Loss curves (policy, value, total)
     - Entropy
     - L2 regularization
     - Game statistics (planned)

2. **Log Files**
   - Check the experiment logs directory for detailed training logs:
   ```bash
   # On Windows:
   type experiments\rival_ai_v1\logs\training.log
   # On Linux/Mac:
   tail -f experiments/rival_ai_v1/logs/training.log
   ```

3. **Checkpoints**
   - Monitor model improvements in the experiment checkpoints directory:
   ```bash
   # On Windows:
   dir experiments\rival_ai_v1\checkpoints
   # On Linux/Mac:
   ls -l experiments/rival_ai_v1/checkpoints/
   ```
   - Checkpoints are saved after each epoch
   - Metrics are included in checkpoint files

### Debugging and Logging

For detailed debugging of the training process, use the `--log-level DEBUG` option:

```bash
python python/scripts/train.py --experiment-name rival_ai_debug --log-level DEBUG
```

This will show:
- MCTS exploration details:
  - Top moves considered at each position
  - Move probabilities from policy network
  - Position evaluations from value network
  - Visit counts and values for explored moves
- Game state transitions
- Training metrics and statistics
- Model predictions and updates

The debug logs are particularly useful for:
- Understanding how the model learns from random weights
- Verifying MCTS exploration behavior
- Diagnosing training issues
- Monitoring policy and value network predictions

To view training progress with TensorBoard:
```bash
# Make sure your virtual environment is activated
tensorboard --logdir logs
```

Then open your browser to `http://localhost:6006` to view:
- Loss curves (policy loss, value loss, total loss)
- Policy accuracy (top-1 and top-3)
- Value prediction accuracy
- Learning rate schedule
- Game statistics (win rates, draw rates, average game length)

## Current Status

### Implemented Features
- [x] PAG-based position representation
- [x] GNN model with policy and value heads
- [x] MCTS with GNN integration
- [x] Self-play data generation
- [x] Training pipeline with metrics
- [x] Model checkpointing
- [x] TensorBoard integration

### In Progress
- [ ] Early stopping
- [ ] Learning rate scheduling
- [ ] Distributed training
- [ ] Training data augmentation
- [ ] Memory optimization
- [ ] Performance profiling

### Planned Features
- [ ] UCI protocol support
- [ ] Transposition tables
- [ ] Adaptive MCTS parameters
- [ ] Advanced training techniques
- [ ] Endgame tablebases

## Development

See [DESIGN.md](DESIGN.md) for detailed architecture documentation and [MILESTONES.md](MILESTONES.md) for development roadmap.

## License

MIT License - See LICENSE file for details 