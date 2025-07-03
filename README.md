# RivalAI Chess Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/Rust-1.70+-blue.svg)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![Node.js](https://img.shields.io/badge/Node.js-18+-orange.svg)](https://nodejs.org/)

A modern chess engine that combines Graph Neural Networks (GNNs) with Monte Carlo Tree Search (MCTS), featuring a novel position representation system called CHESS (Chess Heterogeneous Encoding State System). The engine uses Positional Adjacency Graphs (PAG) to represent chess positions as rich graph structures, capturing piece relationships and strategic dynamics.

Go see it live NOW at https://rivalchess.xyz

## 🚀 Features

- **High-Performance Engine**: Sub-millisecond move generation with 50,000+ MCTS nodes/second
- **Neural Network Integration**: 4-layer Graph Neural Network with Graph Attention (GAT)
- **Multiple Interfaces**: Web server, UCI protocol, and command-line interfaces
- **Community Gaming**: Real-time collaborative gameplay with voting systems
- **Automated Training**: Background self-play generation and model training
- **Tournament Ready**: Full UCI protocol compliance for competitive play

## 🏗️ Architecture

RivalAI consists of three main components:

### 1. Rust Engine Core (`/engine/`)
High-performance chess engine written in Rust with multiple interfaces:

- **Server Mode** (`server.rs`): WebSocket server with REST API for web interface
- **UCI Mode** (`uci.rs`): Full UCI protocol implementation for tournament play
- **CLI Mode** (`play.rs`): Command-line interface for direct gameplay

**Core Features:**
- Sub-millisecond move generation
- MCTS with neural network integration (50,000+ nodes/second)
- Positional Adjacency Graph (PAG) feature extraction
- Game persistence system supporting multiple game modes
- Automatic training data collection from all games
- Background self-play generation with GPU utilization scaling

### 2. Python ML System (`/python/`)
Complete machine learning pipeline for training and analysis:

**Neural Network Architecture:**
- 4-layer Graph Neural Network with Graph Attention (GAT)
- 256 hidden dimensions, 4 attention heads per layer
- Policy head: 5,312 possible moves (including all promotion combinations)
- Value head: Position evaluation (-1 to +1)

**Training Infrastructure:**
- Unified storage system for all game types
- Automated batch processing and archiving
- Background training with community game protection
- UCI tournament data integration
- Self-play generation with adaptive scaling
- Comprehensive metrics and visualization

### 3. React Web Interface (`/engine/web/`)
Modern web interface with real-time features:

- Interactive chessboard with move validation
- Single-player and community game modes
- Real-time statistics and leaderboards
- Game history browser
- Training progress monitoring
- WebSocket communication (<50ms latency)

## 🎮 Game Modes

### Single Player Mode
Traditional one-on-one gameplay against the AI with:
- Full engine strength with MCTS evaluation
- Move history and game analysis
- Automatic game state persistence
- Player color selection
- Immediate move responses

### Community Mode
Collaborative gameplay where multiple players vote on moves:
- 10-second voting windows with real-time countdown
- Democratic move selection with tie-breaking
- Vote modification during voting periods
- Secure voter authentication with JWT tokens
- Rate limiting and abuse prevention
- Automatic engine response after community moves

### UCI Tournament Mode
Automated competitive play against other engines:
- Full UCI protocol compliance with proper time controls
- Multi-engine tournament support
- Automatic training data collection from all games
- PGN export and game archiving
- Performance tracking and statistics
- Continuous learning from competitive play

## 📊 Technical Specifications

### Performance
- Move computation: <100ms
- MCTS evaluation: 50,000+ nodes/second
- Memory usage: <500MB during gameplay
- WebSocket latency: <50ms
- Training speed: 5,000+ games/hour
- UI responsiveness: 60 FPS

### Position Representation (PAG)
The Positional Adjacency Graph system represents chess positions as:
- **Piece Nodes**: 12 features per piece (type, color, position, mobility)
- **Square Nodes**: 1 feature per critical square (occupancy, control)
- **Edge Relationships**: Attacks, defenses, pins, forks, and strategic connections
- **Global Features**: Game phase, material balance, castling rights

### Neural Network Details
```
Input: Chess Position → PAG Conversion
├── Piece Embeddings (256 dim)
├── Square Embeddings (256 dim)
└── Graph Structure

GNN Layers (4x):
├── Graph Attention Networks (GAT)
├── Multi-head attention (4 heads)
├── Layer normalization
├── ReLU activation
└── Dropout (0.1)

Output Heads:
├── Policy Head → 5,312 move probabilities
└── Value Head → Position evaluation
```

## 🛠️ Installation

### Prerequisites

1. **Rust** (latest stable):
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

2. **Python 3.8+** with virtual environment:
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
pip install -r requirements.txt
```

3. **Node.js** (for web interface):
```bash
# Install Node.js from nodejs.org
cd engine/web
npm install
```

### Building the Project

1. **Build Rust Engine**:
```bash
cd engine
cargo build --release
```

2. **Install Python Package**:
```bash
cd python
pip install -e .
```

3. **Build Web Interface**:
```bash
cd engine/web
npm run build
```

## 🚀 Quick Start

### Web Server Mode
Start the complete web application:
```bash
cd engine
cargo run --bin server -- --tensorboard --enable-training --enable-self-play
```

Features:
- Web interface at `http://localhost:3000`
- Real-time game statistics
- Background training and self-play
- Community game support
- TensorBoard integration at `http://localhost:6006`

### UCI Tournament Mode
Run automated tournaments against other engines:
```bash
cd python
python scripts/uci_tournament.py \
    --rival-ai "../engine/target/release/uci.exe" \
    --engines "path/to/stockfish" "path/to/other/engine" \
    --games 50 \
    --time 5.0 \
    --output "tournament_results"
```

### Training Mode
Start dedicated training from existing game data:
```bash
cd python
python scripts/server_training.py \
    --games-dir "training_games" \
    --model-path "../models/latest_trained_model.pt" \
    --threshold 1000 \
    --tensorboard
```

## 🔧 Development

### Most Used Commands

**Start Web Server:**
```bash
cd engine && cargo run --bin server -- --tensorboard
```

**Run UCI Tournament:**
```bash
cd python && python scripts/uci_tournament.py --rival-ai "../engine/target/release/uci.exe" --engines "path/to/stockfish" --games 20 --time 5.0
```

**Monitor Training:**
```bash
tensorboard --logdir python/experiments --port 6007
```

**Check Training Data:**
```bash
cd python && python check_data.py
```

### Useful Scripts
- `python/scripts/uci_tournament.py` - Automated engine tournaments
- `python/scripts/server_training.py` - Background training system
- `python/scripts/server_self_play.py` - Self-play generation
- `python/check_data.py` - Training data status checker

## 📁 Project Structure

```
RivalAI/
├── engine/                 # Rust core engine
│   ├── src/               # Engine implementation
│   │   ├── bin/          # Binary executables (server, uci, play)
│   │   ├── bridge/       # Python-Rust interface
│   │   ├── pag/          # Position Analysis Graph
│   │   ├── board.rs      # Chess board representation
│   │   ├── mcts.rs       # Monte Carlo Tree Search
│   │   ├── engine.rs     # Main engine logic
│   │   └── game_storage.rs # Game persistence
│   ├── web/              # React web interface
│   │   ├── src/components/ # UI components
│   │   └── public/       # Static assets
│   └── target/           # Compiled binaries
├── python/               # Python ML system
│   ├── src/rival_ai/     # Core Python package
│   │   ├── models/       # Neural network implementations
│   │   ├── training/     # Training infrastructure
│   │   ├── utils/        # Utility functions
│   │   └── unified_storage.py # Unified game storage
│   ├── scripts/          # Training and utility scripts
│   ├── training_games/   # Game storage directory
│   │   ├── unified/      # Batched training data
│   │   ├── archives/     # Processed batches
│   │   ├── single_player/ # Single-player games
│   │   └── community/    # Community games
│   └── experiments/      # Training experiment outputs
├── models/               # Trained model storage
├── cloudflare-tunnel/    # Deployment configuration
└── docs/                 # Documentation
```

## 📈 Current Status

### ✅ Completed Features
- Complete chess engine with web interface
- PAG-based position representation
- GNN model with policy and value heads
- MCTS integration with neural guidance
- Unified storage system for all game types
- Automated training pipeline
- UCI tournament integration
- Community game mode with voting
- Background self-play generation
- Real-time statistics and monitoring
- Game persistence and resumption
- TensorBoard integration

### 🚧 In Development
- Opening book integration
- Endgame tablebase support
- Advanced move explanations
- Distributed training
- Tournament rating calculations

### 📋 Planned Features
- Swiss tournament system
- Advanced analysis tools
- Mobile web interface
- API for external integrations
- Research publication features

## 🤝 Contributing

This is currently a personal research project and is **not accepting external contributions** at this time. The project is being developed as a learning exercise and research platform for chess AI development.

If you have questions about the project or want to discuss the implementation, feel free to:
- Open an issue for bug reports
- Start a discussion about the technical approach
- Fork the project for your own research

See [CONTRIBUTING.md](CONTRIBUTING.md) for more details, [DESIGN.md](DESIGN.md) for detailed architecture documentation, and [MILESTONES.md](MILESTONES.md) for development roadmap.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Chess programming community for inspiration and standards
- Rust and Python ecosystems for excellent tooling
- React and modern web technologies for the interface
- Open source chess engines for benchmarking and comparison


