# RivalAI Chess Engine - Technical Overview

## Project Description
RivalAI is a chess AI system that combines a Rust-based chess engine with a Graph Neural Network (GNN) for position evaluation. The system features a web interface for gameplay and uses both self-play and supervised learning from engine tournaments.

## Architecture Overview

### 1. Rust Engine (/engine/)
**Core Components:**
- `mcts.rs` - Monte Carlo Tree Search implementation
- `board.rs` - Chess board representation and move generation  
- `pag/` - Positional Adjacency Graph (PAG) feature extraction
- `python_bridge.rs` - Integration with Python ML models
- `game_storage.rs` - Game persistence system

**Executables:**
- `server.rs` - Web server with REST API and WebSocket support
- `play.rs` - Command-line chess interface
- `uci.rs` - UCI protocol implementation for engine tournaments

### 2. Python ML System (/python/)
**Neural Network Architecture:**
- `models/gnn.py` - Graph Neural Network using PyTorch Geometric
- 10-layer GNN with 256 hidden dimensions, 4 attention heads
- Approximately 2.3M parameters (not 3.5M as initially estimated)
- Graph Attention Networks (GAT) for processing chess positions

**PAG Feature System:**
- Piece nodes: 308-dimensional features (tactical patterns, mobility, control)
- Critical square nodes: 95-dimensional features (strategic importance)  
- Edge features: 158 dimensions (piece relationships, attacks, defenses)
- Fallback to basic features when Rust PAG engine unavailable

**Training Infrastructure:**
- Self-play game generation using MCTS
- UCI tournament data collection (games vs Stockfish, etc.)
- Expert move supervision (85% weight on played moves)
- Memory-optimized batch processing (resolved previous memory issues)
- Model checkpointing and resumption

### 3. Web Interface (/engine/web/)
- React + TypeScript frontend
- Real-time WebSocket communication
- Interactive chessboard with move validation
- Single-player and community voting modes
- Game statistics and analysis tools

### 4. UCI Tournament System
- Automated games against other engines (Stockfish, etc.)
- Training data generation from competitive play
- Performance benchmarking and statistics
- Tournament management and PGN export

## Game Modes

**Single Player Mode:**
- Play against the neural network engine
- Move analysis and position evaluation
- Game history and resumption

**Community Mode:**
- Collaborative gameplay where users vote on moves
- Real-time voting with time limits
- Democratic move selection with tie-breaking

**UCI Tournament Mode:**
- Automated games against other chess engines
- Data collection for training improvements
- Performance tracking and Elo estimation

## Technical Specifications

**Performance:**
- Move computation: ~100ms on typical hardware
- MCTS simulations: 600 per move (configurable)
- Memory usage: <500MB during normal operation
- Training data generation: ~50,000 games/hour possible

**PAG Features:**
- Piece features: 308 dimensions (confirmed in implementation)
- Square features: 95 dimensions (confirmed in implementation)
- Real-time feature extraction via Rust engine
- Compatibility mode using basic features when needed

**Training:**
- Self-play with MCTS search
- Supervised learning from UCI tournament games
- Expert move supervision (85% probability on played moves)
- Batch optimization prevents memory overflow issues

## Current Status

**Functional:**
- ✅ Complete chess engine with PAG integration
- ✅ Web interface for gameplay
- ✅ UCI tournament system operational
- ✅ Training pipeline with memory optimizations
- ✅ Both dense PAG and fallback modes working

**Performance:**
- Engine plays at estimated 1400-1800 ELO level
- Suitable for educational use and casual play
- Continuous improvement through tournament data
- Good architecture for further development

## Technical Notes

**Training Data:**
- Self-play games provide exploration data
- UCI tournament games provide expert supervision  
- Stockfish games offer high-quality move examples
- Unified storage system for all game types

**Architecture Benefits:**
- Modular design with clear separation of concerns
- Rust performance for compute-heavy operations
- Python flexibility for ML experimentation
- Web interface for accessibility

## Development Areas

**Potential Improvements:**
- Opening book integration
- Endgame tablebase support
- Advanced position analysis
- Distributed training capabilities
- Enhanced PAG feature extraction

**Research Opportunities:**
- Graph neural network architecture optimization
- Alternative MCTS enhancements
- Position evaluation methodology
- Feature engineering for chess-specific patterns

## Conclusion

RivalAI represents a solid implementation of modern chess AI techniques, combining graph neural networks with Monte Carlo Tree Search in a well-architected system. While not groundbreaking in AI research terms, it demonstrates effective integration of multiple technologies and provides a good foundation for chess AI development and learning.

The system achieves its design goals of creating a functional chess AI with modern architecture, web accessibility, and continuous learning capabilities through tournament play.