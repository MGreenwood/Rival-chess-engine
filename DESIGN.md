# RivalAI Chess Engine Design Document

## Overview
RivalAI is a novel chess engine implementing the Chess Heterogeneous Encoding State System (CHESS) for position representation, combined with Monte Carlo Tree Search (MCTS) and Graph Neural Networks (GNNs) for evaluation and policy decisions.

## The CHESS System

The Chess Heterogeneous Encoding State System (CHESS) is a novel approach to chess position representation that transforms traditional board positions into rich, heterogeneous graph structures. This system:

1. **Heterogeneous Encoding**: Captures different types of nodes (pieces) and edges (relationships) with their unique properties
2. **State Representation**: Encodes the complete position state, including:
   - Piece positions and relationships
   - Strategic importance of squares
   - Control and influence patterns
   - Tactical and positional dynamics
3. **System Integration**: Seamlessly connects with:
   - Graph Neural Networks for position evaluation
   - Monte Carlo Tree Search for move selection
   - Training pipeline for model improvement

## Core Architecture

### 1. Position Representation (CHESS)

#### Node Types
1. **Piece Nodes**
   - Properties:
     - Piece type (Pawn, Knight, Bishop, Rook, Queen, King)
     - Color (White, Black)
     - Current square coordinates
     - Unique identifier
     - Material value
     - Mobility score
     - Status flags (attacked, defended, part of king's shield)

2. **Critical Square Nodes**
   - Properties:
     - Coordinates
     - Control status (attackers/defenders count)
     - Strategic importance score
     - Type (outpost, weak square, central square, king vicinity)

#### Edge Types
1. **Direct Attack/Defense**
   - Direction: Directed
   - Properties:
     - Type (Attack/Defense)
     - Strength/value
     - Piece types involved

2. **Control/Influence**
   - Direction: Directed
   - Properties:
     - Type (Control/Influence)
     - Degree of control
     - Potential future control

3. **Mobility Pathway**
   - Direction: Directed
   - Properties:
     - Move type
     - Legality
     - Safety score

4. **Cooperative Relationship**
   - Direction: Undirected
   - Properties:
     - Cooperation type (mutual defense, battery, bishop pair)
     - Strength score

5. **Obstructive Relationship**
   - Direction: Undirected
   - Properties:
     - Obstruction type
     - Severity

6. **Vulnerability Link**
   - Direction: Directed
   - Properties:
     - Vulnerability type (Pin, Overload, Undefended)
     - Severity

7. **Pawn Structure**
   - Direction: Undirected
   - Properties:
     - Structure type (chain, doubled, isolated, passed)
     - Support status

### 2. Search System (MCTS)

#### Components
1. **Selection**
   - UCT formula with PAG-based policy guidance
   - Dynamic exploration factor

2. **Expansion**
   - Legal move generation
   - PAG construction for new positions
   - Node initialization with prior probabilities

3. **Simulation**
   - GNN evaluation instead of random playouts
   - Value prediction for leaf nodes

4. **Backpropagation**
   - Update statistics
   - Update edge visit counts
   - Propagate GNN evaluations

### 3. Neural Network Architecture

#### GNN Structure
1. **Input Layer**
   - Node feature processing
   - Edge feature processing
   - Global graph features

2. **Message Passing Layers**
   - Multiple attention heads
   - Edge feature integration
   - Skip connections

3. **Output Heads**
   - Policy head (move probabilities)
   - Value head (position evaluation)

## Implementation Details

### 1. Rust Engine Core
- Zero-cost abstractions for PAG operations
- Lock-free concurrent MCTS
- SIMD optimizations where applicable
- Efficient memory management for nodes/edges

### 2. Python Training Infrastructure
- PyTorch Geometric for GNN implementation
- Distributed training support (planned)
- Efficient PAG serialization
- Training data augmentation (planned)
- Comprehensive metrics tracking:
  - Policy loss
  - Value loss
  - Total loss
  - Entropy
  - L2 regularization
- Model checkpointing and resumption
- TensorBoard integration for visualization
- Logging and debugging support

### 3. Rust-Python Bridge
- Fast serialization protocol
- Batch processing support
- Asynchronous evaluation

## Training Strategy

### Phase 1: Small Board Training
- 5x5 board size
- Basic piece movement
- Simple PAG structures
- Quick iteration and testing
- Current focus:
  - Training loop optimization
  - Memory usage profiling
  - Policy and value loss balancing
  - Entropy regularization tuning

### Phase 2: Full Board Training
- Transfer learning from 5x5
- Progressive board size increase
- Gradual complexity introduction
- PAG feature enrichment
- Planned improvements:
  - Early stopping
  - Learning rate scheduling
  - Distributed training
  - Data augmentation

### Phase 3: Advanced Training
- Complex position understanding
- Strategic pattern recognition
- Endgame specialization
- Performance optimization

## Performance Considerations

### 1. PAG Construction
- Parallel feature computation
- Incremental updates
- Cache-friendly data structures
- Current metrics:
  - PAG construction: < 0.5ms per position
  - Feature computation: < 0.2ms per position
  - Edge detection: < 0.3ms per position

### 2. MCTS Efficiency
- Batch GNN evaluation
- Parallel tree expansion
- Smart pruning based on PAG properties
- Current focus:
  - Memory optimization during search
  - Adaptive simulation counts
  - Transposition table implementation

### 3. Memory Management
- Object pooling for nodes/edges
- Efficient graph storage
- Smart garbage collection
- Current challenges:
  - High memory usage during training
  - Batch size optimization
  - Gradient accumulation for larger batches

## Future Extensions

### 1. Potential Improvements
- Custom GNN layers for chess
- Advanced PAG features
- Specialized endgame handling

### 2. Research Opportunities
- PAG pattern mining
- Strategic concept learning
- Transfer learning studies 