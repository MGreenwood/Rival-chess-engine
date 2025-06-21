#!/bin/bash

# Setup script for RivalAI UCI Integration
# This script builds the UCI engine and demonstrates how to use it for training data collection

set -e

echo "🏗️  Setting up RivalAI UCI Integration"
echo "====================================="

# Check if we're in the right directory
if [ ! -f "engine/Cargo.toml" ]; then
    echo "❌ Please run this script from the RivalAI root directory"
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p python/training_games/uci_matches
mkdir -p scripts
mkdir -p uci_tournament_results

# Build the UCI engine
echo "🔨 Building UCI engine..."
cd engine
cargo build --release --bin uci
echo "✅ UCI engine built successfully"

# Check if the binary was created
if [ ! -f "target/release/uci" ]; then
    echo "❌ UCI binary not found. Build may have failed."
    exit 1
fi

cd ..

# Make scripts executable
echo "🔧 Setting up scripts..."
chmod +x scripts/uci_tournament.py
chmod +x scripts/convert_uci_games.py

# Install Python dependencies if needed
echo "📦 Checking Python dependencies..."
python3 -c "import chess.engine" 2>/dev/null || {
    echo "Installing python-chess..."
    pip install python-chess
}

echo ""
echo "✅ Setup complete!"
echo ""
echo "🚀 QUICK START GUIDE"
echo "===================="
echo ""
echo "1. Test the UCI engine manually:"
echo "   cd engine"
echo "   echo 'uci' | ./target/release/uci"
echo ""
echo "2. Run a quick tournament against Stockfish:"
echo "   python3 scripts/uci_tournament.py --games 2 --time 1.0"
echo ""
echo "3. Convert tournament results to training data:"
echo "   python3 scripts/convert_uci_games.py uci_tournament_results"
echo ""
echo "4. Check the training data:"
echo "   ls -la python/training_games/uci_matches/"
echo ""
echo "🎯 TRAINING INTEGRATION"
echo "======================"
echo ""
echo "Every UCI match automatically:"
echo "• Saves games in RivalAI's training format"
echo "• Creates PGN files for analysis"
echo "• Tracks win/loss statistics"
echo "• Feeds data into the self-play training pipeline"
echo ""
echo "The model will improve from playing against other engines!"
echo ""
echo "📋 AVAILABLE COMMANDS"
echo "===================="
echo ""
echo "Build UCI engine:"
echo "  cd engine && cargo build --release --bin uci"
echo ""
echo "Test UCI manually:"
echo "  cd engine && ./target/release/uci"
echo "  > uci"
echo "  > isready"
echo "  > position startpos moves e2e4"
echo "  > go movetime 5000"
echo "  > quit"
echo ""
echo "Run tournaments:"
echo "  python3 scripts/uci_tournament.py --help"
echo ""
echo "Convert games:"
echo "  python3 scripts/convert_uci_games.py --help"
echo ""
echo "🔧 ADVANCED CONFIGURATION"
echo "========================="
echo ""
echo "UCI Engine Options:"
echo "• PAG_Mode: Enable/disable Positional Adjacency Graph"
echo "• MCTS_Simulations: Number of MCTS simulations per move"
echo "• Training_Mode: Collect training data from games"
echo "• Collect_Data: Save games for training"
echo ""
echo "Example UCI session with options:"
echo "  setoption name MCTS_Simulations value 2000"
echo "  setoption name PAG_Mode value true"
echo "  setoption name Training_Mode value true"
echo ""
echo "🎮 TESTING AGAINST DIFFERENT ENGINES"
echo "===================================="
echo ""
echo "Install engines for testing:"
echo "  # Stockfish (strong traditional engine)"
echo "  sudo apt install stockfish"
echo ""
echo "  # Leela Chess Zero (neural network engine)"
echo "  # Download from https://lczero.org/"
echo ""
echo "Run against multiple engines:"
echo "  python3 scripts/uci_tournament.py \\"
echo "    --engines stockfish /path/to/lc0 \\"
echo "    --games 10 --time 5.0"
echo ""
echo "📊 MONITORING IMPROVEMENT"
echo "========================"
echo ""
echo "Track RivalAI's progress:"
echo "1. Run tournaments regularly"
echo "2. Convert games to training data"
echo "3. Monitor win rates over time"
echo "4. Retrain model with new data"
echo ""
echo "View game storage:"
echo "  ls -la python/training_games/uci_matches/"
echo ""
echo "Check win rates:"
echo "  cat uci_tournament_results/tournament_results.json | jq '.overall_stats'"
echo ""
echo "🐛 TROUBLESHOOTING"
echo "=================="
echo ""
echo "If UCI engine fails to start:"
echo "• Check model path in engine/src/bin/uci.rs"
echo "• Ensure Python dependencies are installed"
echo "• Try with --verbose flag for debugging"
echo ""
echo "If tournament script fails:"
echo "• Ensure opponent engines are in PATH"
echo "• Check python-chess is installed: pip install python-chess"
echo "• Use absolute paths for engine binaries"
echo ""
echo "For training integration issues:"
echo "• Check that python/training_games/uci_matches/ exists"
echo "• Verify GameMode::UCI was added to game_storage.rs"
echo "• Ensure converted games are in .pkl format"
echo ""
echo "✨ Ready to start playing against other engines!"
echo "   Your model will improve with every match! 🚀" 