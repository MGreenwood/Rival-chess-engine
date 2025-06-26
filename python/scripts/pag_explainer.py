#!/usr/bin/env python3
"""
PAG Chess Position Explainer
===========================

Uses DeepSeek via Ollama to interpret ultra-dense PAG features and explain
what the chess engine is "thinking" about the position in plain English.

Usage:
    python pag_explainer.py "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    python pag_explainer.py --interactive
"""

import argparse
import sys
from pathlib import Path
import chess
import numpy as np
import requests

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    from rival_ai.utils.board_conversion import board_to_hetero_data, PAG_ENGINE_AVAILABLE
    print(f"✅ PAG engine available: {PAG_ENGINE_AVAILABLE}")
except ImportError as e:
    print(f"❌ Failed to import PAG components: {e}")
    sys.exit(1)

class PAGChessExplainer:
    def __init__(self, ollama_url="http://localhost:11434"):
        self.ollama_url = ollama_url
        self.feature_categories = {
            'basic': (0, 10),
            'tactical': (10, 86), 
            'positional': (86, 166),
            'strategic': (166, 226),
            'dynamic': (226, 268),
            'advanced': (268, 308)
        }

    def analyze_piece_features(self, piece_tensor, board):
        """Extract meaningful insights from piece features"""
        insights = []
        
        piece_idx = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece_idx < piece_tensor.shape[0]:
                features = piece_tensor[piece_idx]
                piece_name = f"{chess.piece_name(piece.piece_type)} on {chess.square_name(square)}"
                
                # Analyze each category
                for cat_name, (start, end) in self.feature_categories.items():
                    cat_features = features[start:end]
                    avg_value = cat_features.mean().item()
                    max_value = cat_features.max().item()
                    active_features = (cat_features > 0.1).sum().item()
                    
                    if avg_value > 0.4 or max_value > 0.8:  # High activity threshold
                        insights.append({
                            'piece': piece_name,
                            'color': 'White' if piece.color else 'Black',
                            'category': cat_name,
                            'average': avg_value,
                            'maximum': max_value,
                            'active_count': active_features,
                            'significance': 'high' if avg_value > 0.6 else 'medium'
                        })
                
                piece_idx += 1
        
        return insights

    def call_deepseek(self, analysis_text):
        """Call DeepSeek via Ollama to interpret the PAG analysis"""
        
        prompt = f"""
You are an AI data interpreter with NO prior chess knowledge. You are analyzing output from a chess AI system that extracted 308 mathematical features per piece.

IMPORTANT: Do NOT use any chess knowledge you may have. Only interpret the mathematical feature data provided.

{analysis_text}

The AI system analyzed each piece and assigned activity scores (0.0 to 1.0) across these categories:
- BASIC features (0-10): Fundamental piece properties
- TACTICAL features (10-86): Attack/threat/vulnerability patterns  
- POSITIONAL features (86-166): Control/mobility/coordination
- STRATEGIC features (166-226): Long-term planning elements
- DYNAMIC features (226-268): Tempo/initiative factors
- ADVANCED features (268-308): Complex pattern recognition

Your task: Interpret ONLY what the feature data suggests, without using chess knowledge:

1. **Which pieces have the highest mathematical activity?** (based on the scores)
2. **What categories show the most activity?** (tactical vs positional vs strategic)
3. **What patterns emerge from the data?** (which color has higher scores, which piece types)
4. **What might high scores in each category suggest?** (high tactical = threat patterns, high positional = control patterns)

Focus purely on DATA INTERPRETATION. Explain what the numbers suggest about piece importance and activity patterns, but avoid chess-specific advice or moves.

Present your analysis as: "The data suggests..." or "The feature analysis indicates..."
"""

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": "deepseek-r1:latest",
                    "prompt": prompt,
                    "think": False,  # Disable thinking mode for cleaner responses
                    "stream": False
                },
                timeout=45
            )
            
            if response.ok:
                result = response.json()
                response_text = result.get("response", "No response generated").strip()
                
                # Remove thinking tags if they appear
                if "<think>" in response_text and "</think>" in response_text:
                    # Extract only the part after </think>
                    parts = response_text.split("</think>", 1)
                    if len(parts) > 1:
                        response_text = parts[1].strip()
                
                return response_text
            else:
                return f"❌ Ollama error: {response.status_code}"
                
        except requests.exceptions.ConnectionError:
            return "❌ Cannot connect to Ollama. Make sure it's running: ollama serve"
        except Exception as e:
            return f"❌ Error: {e}"

    def format_insights_for_llm(self, board, piece_insights):
        """Format insights for LLM with proper board context"""
        
        # Create detailed board description
        context = f"""
CHESS POSITION ANALYSIS
FEN: {board.fen()}
Turn: {"White" if board.turn else "Black"} to move

CURRENT BOARD STATE:
{str(board)}

PIECE LOCATIONS:
White pieces: {self._get_piece_locations(board, chess.WHITE)}
Black pieces: {self._get_piece_locations(board, chess.BLACK)}

ULTRA-DENSE PAG FEATURE ANALYSIS:
The AI extracted 308 features per piece, analyzing tactical patterns, positional themes, strategic elements, and dynamic factors.

HIGH ACTIVITY PIECES (based on 308-dimensional feature analysis):
"""
        
        high_pieces = [p for p in piece_insights if p['significance'] == 'high']
        medium_pieces = [p for p in piece_insights if p['significance'] == 'medium']
        
        if high_pieces:
            context += "\nVery High Activity:\n"
            for piece in high_pieces:
                context += f"- {piece['piece']} ({piece['color']}): {piece['category']} features show intense activity (avg: {piece['average']:.3f})\n"
        
        if medium_pieces:
            context += "\nModerate Activity:\n"
            for piece in medium_pieces[:5]:
                context += f"- {piece['piece']} ({piece['color']}): {piece['category']} features show activity (avg: {piece['average']:.3f})\n"
        
        context += "\nPlease analyze this EXACT position and explain what the PAG feature activity means in chess terms."
        
        return context

    def _get_piece_locations(self, board, color):
        """Get readable list of piece locations for a color"""
        pieces = []
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == color:
                pieces.append(f"{chess.piece_name(piece.piece_type)}-{chess.square_name(square)}")
        return ", ".join(pieces)

    def explain_position(self, board_fen):
        """Main method to explain a chess position"""
        print(f"🔍 Analyzing position: {board_fen}")
        
        try:
            board = chess.Board(board_fen)
            
            # Generate ultra-dense PAG data
            print("⚡ Extracting ultra-dense PAG features...")
            data = board_to_hetero_data(board, use_dense_pag=True)
            
            piece_tensor = data['piece'].x
            print(f"✅ Extracted {piece_tensor.shape[0]} pieces with {piece_tensor.shape[1]} features each")
            
            # Analyze the features
            print("🧠 Analyzing piece activity patterns...")
            piece_insights = self.analyze_piece_features(piece_tensor, board)
            
            # Format for LLM
            analysis_text = self.format_insights_for_llm(board, piece_insights)
            
            # Get DeepSeek interpretation
            print("🤖 Calling DeepSeek for chess commentary...")
            explanation = self.call_deepseek(analysis_text)
            
            return {
                'board': board,
                'insights': piece_insights,
                'explanation': explanation
            }
            
        except Exception as e:
            return {
                'error': f"Analysis failed: {e}",
                'explanation': "Could not analyze this position."
            }

def main():
    parser = argparse.ArgumentParser(description='Explain chess position using PAG + DeepSeek')
    parser.add_argument('fen', nargs='?',
                       default='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
                       help='FEN string to analyze')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    
    args = parser.parse_args()
    
    explainer = PAGChessExplainer()
    
    if args.interactive:
        print("🎯 PAG Chess Explainer - Interactive Mode")
        print("Enter FEN positions to get DeepSeek analysis")
        print("Type 'quit' to exit\n")
        
        while True:
            try:
                fen = input("Enter FEN: ").strip()
                if fen.lower() == 'quit':
                    break
                if not fen:
                    fen = args.fen
                
                result = explainer.explain_position(fen)
                
                if 'error' in result:
                    print(f"❌ {result['error']}")
                else:
                    print("\n📋 BOARD:")
                    print(result['board'])
                    print(f"\n🧠 DEEPSEEK ANALYSIS:")
                    print(result['explanation'])
                    print("\n" + "="*60)
                
            except KeyboardInterrupt:
                break
    else:
        result = explainer.explain_position(args.fen)
        
        if 'error' in result:
            print(f"❌ {result['error']}")
        else:
            print("📋 BOARD:")
            print(result['board'])
            print(f"\n🧠 DEEPSEEK ANALYSIS:")
            print(result['explanation'])

if __name__ == "__main__":
    main() 