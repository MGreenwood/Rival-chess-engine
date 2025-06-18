"""
Interactive game viewer for analyzing self-play games.
"""

import sys
import json
import pickle
import chess
import chess.svg
import chess.pgn
import torch
from datetime import datetime
from typing import List, Dict, Tuple
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QListWidget, QLabel, 
                            QSlider, QSpinBox, QSplitter, QFileDialog, QMessageBox,
                            QComboBox, QFrame, QTextEdit, QScrollArea)
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon, QPixmap
import requests

from rival_ai.analysis.pag_explainer import PAGExplainer, PAGExplanation
from rival_ai.pag import PositionalAdjacencyGraph
from rival_ai.models.gnn import ChessGNN
from rival_ai.utils.board_conversion import board_to_hetero_data
from rival_ai.training.training_types import GameRecord

class CustomUnpickler(pickle.Unpickler):
    """Custom unpickler to handle old module paths."""
    def find_class(self, module, name):
        if module == 'rival_ai.training.types':
            # Map old module path to new one
            module = 'rival_ai.training.training_types'
        return super().find_class(module, name)

def load_model(checkpoint_path: str) -> ChessGNN:
    """Load model from checkpoint."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = ChessGNN()
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model

class ExplanationWidget(QWidget):
    """Widget to display position explanations."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(8, 8, 8, 8)

        # LLM controls row
        llm_controls = QHBoxLayout()
        self.personalities = [
            ("Witty Grandmaster", "Respond as a witty, insightful chess grandmaster who enjoys clever remarks."),
            ("Grumpy Coach", "Respond as a grumpy, old-school chess coach who is blunt and critical but knowledgeable."),
            ("Robotic Analyst", "Respond as a precise, emotionless chess engine focused on facts and logic."),
            ("Shakespearean Bard", "Respond in the style of Shakespeare, with poetic and dramatic flair."),
            ("Casual Friend", "Respond as a friendly, casual chess enthusiast offering encouragement.")
        ]
        self.personality_dropdown = QComboBox()
        for name, _ in self.personalities:
            self.personality_dropdown.addItem(name)
        llm_controls.addWidget(QLabel("LLM Personality:"))
        llm_controls.addWidget(self.personality_dropdown)
        self.summarize_btn = QPushButton("Summarize with LLM")
        self.summarize_btn.clicked.connect(self.on_summarize_clicked)
        llm_controls.addWidget(self.summarize_btn)
        llm_controls.addStretch()
        layout.addLayout(llm_controls)

        # LLM summary area (tall, prominent, no max height)
        layout.addWidget(self._bold_label("LLM Summary:"))
        self.llm_summary = QTextEdit()
        self.llm_summary.setReadOnly(True)
        self.llm_summary.setMinimumHeight(140)
        self.llm_summary.setStyleSheet("""
            QTextEdit {
                background-color: #e8f5e9;
                border: 1px solid #8bc34a;
                border-radius: 4px;
                padding: 8px;
                font-style: italic;
                font-size: 15px;
            }
        """)
        layout.addWidget(self.llm_summary)

        # Move explanation
        layout.addWidget(self._bold_label("Move Explanation:"))
        self.move_explanation = QTextEdit()
        self.move_explanation.setReadOnly(True)
        self.move_explanation.setMaximumHeight(40)
        self.move_explanation.setStyleSheet("background-color: #f8f8f8; border: 1px solid #ddd; border-radius: 4px; padding: 4px;")
        layout.addWidget(self.move_explanation)

        # Tactical elements
        layout.addWidget(self._bold_label("Tactical Elements:"))
        self.tactical_text = QTextEdit()
        self.tactical_text.setReadOnly(True)
        self.tactical_text.setMaximumHeight(40)
        self.tactical_text.setStyleSheet("background-color: #f8f8f8; border: 1px solid #ddd; border-radius: 4px; padding: 4px;")
        layout.addWidget(self.tactical_text)

        # Strategic elements
        layout.addWidget(self._bold_label("Strategic Elements:"))
        self.strategic_text = QTextEdit()
        self.strategic_text.setReadOnly(True)
        self.strategic_text.setMaximumHeight(60)
        self.strategic_text.setStyleSheet("background-color: #f8f8f8; border: 1px solid #ddd; border-radius: 4px; padding: 4px;")
        layout.addWidget(self.strategic_text)

        # Piece relationships
        layout.addWidget(self._bold_label("Piece Relationships:"))
        self.relationships_text = QTextEdit()
        self.relationships_text.setReadOnly(True)
        self.relationships_text.setMaximumHeight(40)
        self.relationships_text.setStyleSheet("background-color: #f8f8f8; border: 1px solid #ddd; border-radius: 4px; padding: 4px;")
        layout.addWidget(self.relationships_text)

        # Attention focus
        layout.addWidget(self._bold_label("Model's Focus:"))
        self.attention_text = QTextEdit()
        self.attention_text.setReadOnly(True)
        self.attention_text.setMaximumHeight(40)
        self.attention_text.setStyleSheet("background-color: #f8f8f8; border: 1px solid #ddd; border-radius: 4px; padding: 4px;")
        layout.addWidget(self.attention_text)

        self._last_explanation = None
        self._last_board = None
        self._last_move = None
    
    def _bold_label(self, text):
        label = QLabel(text)
        label.setStyleSheet("font-weight: bold; margin-bottom: 2px;")
        return label

    def update_explanation(self, explanation: PAGExplanation, board=None, last_move=None):
        """Update the explanation display and cache for LLM."""
        self.move_explanation.setText(explanation.move_explanation)
        self.tactical_text.setText("\n".join(explanation.tactical_elements))
        self.strategic_text.setText("\n".join(explanation.strategic_elements))
        self.relationships_text.setText("\n".join(explanation.piece_relationships))
        self.attention_text.setText("\n".join(explanation.attention_focus))
        self.llm_summary.clear()
        self._last_explanation = explanation
        self._last_board = board
        self._last_move = last_move

    def on_summarize_clicked(self):
        """Call Ollama LLM to translate the model's reasoning."""
        if not self._last_explanation or not self._last_board:
            self.llm_summary.setText("No position data available.")
            return
        last_move = self._last_move.uci() if self._last_move else "None"
        explanation = self._last_explanation
        tactical = ", ".join(explanation.tactical_elements) or "None"
        strategic = ", ".join(explanation.strategic_elements) or "None"
        relationships = ", ".join(explanation.piece_relationships) or "None"
        attention = ", ".join(explanation.attention_focus) or "None"
        move_expl = explanation.move_explanation or "None"
        personality_idx = self.personality_dropdown.currentIndex()
        personality_name, personality_desc = self.personalities[personality_idx]
        
        prompt = f"""
[Personality: {personality_name}]
{personality_desc}

You are analyzing a chess position. Your task is to translate the model's internal reasoning into a natural, engaging explanation that matches your personality.

Current Position:
{self._last_board.fen()}

Model's Analysis:
1. Move Information:
   - Last move played: {last_move}
   - Model's explanation: {move_expl}

2. Position Assessment:
   - Tactical elements: {tactical}
   - Strategic elements: {strategic}
   - Key piece relationships: {relationships}
   - Model's attention focus: {attention}

Please provide a natural, engaging explanation that:
1. Describes the move played and its immediate consequences
2. Explains the tactical and strategic elements at play
3. Highlights important piece relationships and interactions
4. Incorporates the model's focus points
5. Maintains your assigned personality throughout

Focus on creating a coherent narrative that ties these elements together. Avoid simply listing facts.
Do not include internal thoughts or explanations of your process.
"""
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "deepseek-r1:latest",
                    "prompt": prompt,
                    "think": False,
                    "stream": False
                },
                timeout=30
            )
            if response.ok:
                summary = response.json().get("response", "(No response)").strip()
                self.llm_summary.setText(summary)
            else:
                self.llm_summary.setText(f"Error: {response.text}")
        except Exception as e:
            self.llm_summary.setText(f"LLM error: {e}")

class GameViewer(QMainWindow):
    def __init__(self, game_file, model_path=None):
        super().__init__()
        self.setWindowTitle("RivalAI Game Viewer")
        self.game_file = game_file
        
        # Initialize PAG explainer
        self.explainer = PAGExplainer()
        
        # Load model if provided
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if model_path:
            try:
                self.model = load_model(model_path)
                print(f"Loaded model from {model_path}")
            except Exception as e:
                print(f"Failed to load model: {e}")
                QMessageBox.warning(self, "Model Load Error", 
                                  f"Failed to load model from {model_path}. Explanations will be limited.")
        
        # Load game
        print(f"Loading game file: {game_file}")
        try:
            # Load pickle file with custom unpickler
            with open(game_file, 'rb') as f:
                unpickler = CustomUnpickler(f)
                games = unpickler.load()
                print(f"Loaded {len(games)} games from pickle file")
                # Convert GameRecord objects to list of positions
                all_positions = []
                for game in games:
                    # Handle both old and new GameRecord formats
                    if hasattr(game, 'states'):  # New format
                        for i, (state, move, value) in enumerate(zip(game.states, game.moves, game.values)):
                            pos = {
                                'fen': state.fen(),
                                'move': move.uci() if i < len(game.moves) else None,
                                'value': float(value)
                            }
                            all_positions.append(pos)
                    else:  # Old format - try to access attributes directly
                        for i in range(len(game.states)):
                            pos = {
                                'fen': game.states[i].fen(),
                                'move': game.moves[i].uci() if i < len(game.moves) else None,
                                'value': float(game.values[i])
                            }
                            all_positions.append(pos)
        except Exception as e:
            print(f"Error loading pickle file: {e}")
            raise
        
        print(f"Total positions: {len(all_positions)}")
        
        # Split into separate games
        initial_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        self.games = []
        current_game = []
        for i, pos in enumerate(all_positions):
            if pos['fen'].strip() == initial_fen and current_game:
                print(f"Detected new game start at FEN: {pos['fen']}")
                self.games.append(current_game)
                current_game = []
            current_game.append(pos)
        if current_game:
            self.games.append(current_game)
            
        print(f"Detected {len(self.games)} separate games")
        for i, game in enumerate(self.games):
            # Get the first few moves to help identify the game
            moves = []
            for pos in game[:3]:  # Look at first 3 moves
                if 'move' in pos:
                    moves.append(pos['move'])
            move_str = ' '.join(moves)
            print(f"Game {i+1}: {len(game)} positions, starts with: {move_str}")
        
        # Start with the first game
        self.current_game_idx = 0
        self.positions = self.games[self.current_game_idx]
        self.current_move = 0
        self.board = chess.Board()
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Add game selector
        game_selector = QWidget()
        game_selector.setStyleSheet("""
            QWidget {
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 5px;
            }
            QLabel {
                font-weight: bold;
                font-size: 14px;
            }
            QComboBox {
                min-width: 300px;
                padding: 5px;
                font-size: 13px;
            }
        """)
        game_selector_layout = QHBoxLayout(game_selector)
        game_selector_layout.addWidget(QLabel("Select Game:"))
        
        self.game_combo = QComboBox()
        for i, game in enumerate(self.games):
            moves = []
            for pos in game[:3]:
                if 'move' in pos:
                    moves.append(pos['move'])
            move_str = ' '.join(moves)
            self.game_combo.addItem(f"Game {i+1} ({len(game)} moves) - {move_str}")
        
        self.game_combo.currentIndexChanged.connect(self.on_game_changed)
        game_selector_layout.addWidget(self.game_combo)
        layout.addWidget(game_selector)
        
        # Add a separator line
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)
        
        # Create main content area with splitter
        content_splitter = QSplitter(Qt.Horizontal)
        
        # Left panel: board and move list
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Create chess board widget
        self.svg_widget = QSvgWidget()
        self.svg_widget.setMinimumSize(400, 400)
        left_layout.addWidget(self.svg_widget)
        
        # Create move list widget
        move_list_layout = QVBoxLayout()
        move_list_layout.addWidget(QLabel("Moves:"))
        self.move_list = QListWidget()
        self.move_list.currentRowChanged.connect(self.on_move_selected)
        move_list_layout.addWidget(self.move_list)
        left_layout.addLayout(move_list_layout)
        
        # Right panel: explanations
        right_panel = QScrollArea()
        right_panel.setWidgetResizable(True)
        self.explanation_widget = ExplanationWidget()
        right_panel.setWidget(self.explanation_widget)
        
        # Add panels to splitter
        content_splitter.addWidget(left_panel)
        content_splitter.addWidget(right_panel)
        content_splitter.setSizes([600, 400])  # Initial split ratio
        
        layout.addWidget(content_splitter)
        
        # Create control panel
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        
        # Playback controls
        playback_layout = QHBoxLayout()
        
        # First move button
        self.first_btn = QPushButton("⏮")
        self.first_btn.clicked.connect(self.go_to_first)
        playback_layout.addWidget(self.first_btn)
        
        # Previous move button
        self.prev_btn = QPushButton("⏪")
        self.prev_btn.clicked.connect(self.previous_move)
        playback_layout.addWidget(self.prev_btn)
        
        # Play/Pause button
        self.play_btn = QPushButton("▶")
        self.play_btn.setCheckable(True)
        self.play_btn.clicked.connect(self.toggle_playback)
        playback_layout.addWidget(self.play_btn)
        
        # Next move button
        self.next_btn = QPushButton("⏩")
        self.next_btn.clicked.connect(self.next_move)
        playback_layout.addWidget(self.next_btn)
        
        # Last move button
        self.last_btn = QPushButton("⏭")
        self.last_btn.clicked.connect(self.go_to_last)
        playback_layout.addWidget(self.last_btn)
        
        # Save PGN button
        self.save_btn = QPushButton("Save PGN")
        self.save_btn.clicked.connect(self.save_pgn)
        playback_layout.addWidget(self.save_btn)
        
        control_layout.addLayout(playback_layout)
        
        # Playback speed control
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Speed:"))
        self.speed_spin = QSpinBox()
        self.speed_spin.setRange(1, 10)
        self.speed_spin.setValue(5)
        self.speed_spin.setSuffix("x")
        speed_layout.addWidget(self.speed_spin)
        control_layout.addLayout(speed_layout)
        
        # Position slider
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("Position:"))
        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.setRange(0, len(self.positions) - 1)
        self.position_slider.valueChanged.connect(self.on_slider_changed)
        slider_layout.addWidget(self.position_slider)
        control_layout.addLayout(slider_layout)
        
        # Add control panel to main layout
        layout.addWidget(control_panel)
        
        # Create timer for playback
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_move)
        
        # Initialize the view
        self.update_board()
        self.populate_move_list()
        
        # Set window size
        self.resize(1200, 800)
    
    def _get_pag_data(self, board: chess.Board) -> Tuple[List[Dict], Dict[int, float]]:
        """Get PAG data for the current position."""
        if not self.model:
            return [], {}
        
        # Create and build PAG for current position
        pag = PositionalAdjacencyGraph(board=board)
        pag._build_graph()  # Build the graph structure
        
        # Convert PAG edges to list of dictionaries
        edges = []
        
        # Add direct relations
        for (from_square, to_square), edge_data in pag.direct_edges.items():
            edge = {
                'from': from_square,
                'to': to_square,
                'type': edge_data['type'],
                'weight': edge_data.get('distance', 1.0) / 14.0  # Normalize distance
            }
            edges.append(edge)
        
        # Add control relations
        for (from_square, to_square), edge_data in pag.control_edges.items():
            edge = {
                'from': from_square,
                'to': to_square,
                'type': 'control',
                'weight': edge_data.get('distance', 1.0) / 14.0  # Normalize distance
            }
            edges.append(edge)
        
        # Add mobility relations
        for (from_square, to_square), edge_data in pag.mobility_edges.items():
            edge = {
                'from': from_square,
                'to': to_square,
                'type': 'mobility',
                'weight': edge_data.get('distance', 1.0) / 14.0  # Normalize distance
            }
            edges.append(edge)
        
        # Add cooperative relations
        for (from_square, to_square), edge_data in pag.cooperative_edges.items():
            edge = {
                'from': from_square,
                'to': to_square,
                'type': edge_data['type'],
                'weight': edge_data.get('distance', 1.0) / 14.0  # Normalize distance
            }
            edges.append(edge)
        
        # Add obstructive relations
        for (from_square, to_square), edge_data in pag.obstructive_edges.items():
            edge = {
                'from': from_square,
                'to': to_square,
                'type': edge_data['type'],
                'weight': edge_data.get('distance', 1.0) / 14.0  # Normalize distance
            }
            edges.append(edge)
        
        # Add vulnerability relations
        for (from_square, to_square), edge_data in pag.vulnerability_edges.items():
            edge = {
                'from': from_square,
                'to': to_square,
                'type': edge_data['type'],
                'weight': edge_data.get('distance', 1.0) / 14.0  # Normalize distance
            }
            edges.append(edge)
        
        # Add pawn structure relations
        for (from_square, to_square), edge_data in pag.pawn_structure_edges.items():
            edge = {
                'from': from_square,
                'to': to_square,
                'type': edge_data['type'],
                'weight': edge_data.get('distance', 1.0) / 14.0  # Normalize distance
            }
            edges.append(edge)
        
        # Get model's attention weights
        with torch.no_grad():
            data = board_to_hetero_data(board)
            data = data.to(self.device)
            _, value = self.model(data)
            # For now, we'll use a simple attention map based on piece positions
            attention_weights = {square: 1.0 for square in chess.SQUARES if board.piece_at(square)}
        
        return edges, attention_weights
    
    def update_board(self):
        """Update the board display and explanations."""
        # Get the last move that was made
        last_move = None
        if self.current_move > 0 and 'move' in self.positions[self.current_move - 1]:
            last_move = chess.Move.from_uci(self.positions[self.current_move - 1]['move'])

        # Generate SVG of current position
        svg = chess.svg.board(
            board=self.board,
            size=400,
            lastmove=last_move,
            check=self.board.king(self.board.turn) if self.board.is_check() else None
        )

        # Update SVG widget
        self.svg_widget.load(bytes(svg, 'utf-8'))

        # Update slider
        self.position_slider.setValue(self.current_move)

        # Update move list selection
        self.move_list.setCurrentRow(self.current_move)

        # Update button states
        self.first_btn.setEnabled(self.current_move > 0)
        self.prev_btn.setEnabled(self.current_move > 0)
        self.next_btn.setEnabled(self.current_move < len(self.positions) - 1)
        self.last_btn.setEnabled(self.current_move < len(self.positions) - 1)

        # Check move legality before calling explainer
        legal = True
        if last_move is not None:
            temp_board = chess.Board(self.positions[self.current_move - 1]['fen'])
            if last_move not in temp_board.legal_moves:
                print(f"Illegal move detected at move {last_move.uci()} in position {temp_board.fen()}")
                legal = False

        # Get PAG data and generate explanation only if legal
        if legal:
            pag_edges, attention_weights = self._get_pag_data(self.board)
            explanation = self.explainer.explain_position(
                self.board, pag_edges, attention_weights, last_move
            )
            self.explanation_widget.update_explanation(explanation, board=self.board, last_move=last_move)
        else:
            self.explanation_widget.move_explanation.setText("Illegal move detected. Explanation unavailable.")
            self.explanation_widget.tactical_text.clear()
            self.explanation_widget.strategic_text.clear()
            self.explanation_widget.relationships_text.clear()
            self.explanation_widget.attention_text.clear()
            self.explanation_widget.llm_summary.clear()
    
    def populate_move_list(self):
        """Populate the move list widget."""
        self.move_list.clear()
        print(f"Populating move list with {len(self.positions)} positions")
        for i, pos in enumerate(self.positions):
            print(f"Position {i}: {pos}")
            try:
                move_text = None
                if 'move' in pos and pos['move']:
                    move = chess.Move.from_uci(pos['move'])
                    move_text = f"{i+1}. {move.uci()}"
                    # Check if this move ends the game
                    board = chess.Board(pos['fen'])
                    if board.is_game_over():
                        outcome = board.outcome()
                        if outcome:
                            if outcome.winner is not None:
                                winner = "White" if outcome.winner == chess.WHITE else "Black"
                                if outcome.termination == chess.Termination.CHECKMATE:
                                    move_text += "# (checkmate)"
                                else:
                                    move_text += f" ({winner} wins)"
                            else:
                                if outcome.termination == chess.Termination.STALEMATE:
                                    move_text += " (= stalemate)"
                                elif outcome.termination == chess.Termination.INSUFFICIENT_MATERIAL:
                                    move_text += " (= insufficient material)"
                                elif outcome.termination == chess.Termination.FIVEFOLD_REPETITION:
                                    move_text += " (= fivefold repetition)"
                                elif outcome.termination == chess.Termination.SEVENTYFIVE_MOVES:
                                    move_text += " (= 75-move rule)"
                                else:
                                    move_text += " (= draw)"
                    if pos.get('value') is not None:
                        move_text += f" (eval: {pos['value']:.2f})"
                    self.move_list.addItem(move_text)
                else:
                    # If no move, just show the position (should only happen for the initial position)
                    move_text = f"{i+1}. [Initial Position]"
                    if pos.get('value') is not None:
                        move_text += f" (eval: {pos['value']:.2f})"
                    self.move_list.addItem(move_text)
            except Exception as e:
                print(f"Unexpected error at position {i}: {e}")
                print(f"Position data: {pos}")
                raise
    
    def on_move_selected(self, row):
        """Handle move selection from the list."""
        if row != self.current_move:
            self.current_move = row
            self.board = chess.Board(self.positions[row]['fen'])
            self.update_board()
    
    def on_slider_changed(self, value):
        """Handle slider value change."""
        if value != self.current_move:
            self.current_move = value
            self.board = chess.Board(self.positions[value]['fen'])
            self.update_board()
    
    def go_to_first(self):
        """Go to the first position."""
        self.current_move = 0
        self.board = chess.Board()
        self.update_board()
    
    def go_to_last(self):
        """Go to the last position."""
        self.current_move = len(self.positions) - 1
        self.board = chess.Board(self.positions[-1]['fen'])
        self.update_board()
    
    def previous_move(self):
        """Go to the previous move."""
        if self.current_move > 0:
            self.current_move -= 1
            self.board = chess.Board(self.positions[self.current_move]['fen'])
            self.update_board()
    
    def next_move(self):
        """Go to the next move."""
        if self.current_move < len(self.positions) - 1:
            self.current_move += 1
            self.board = chess.Board(self.positions[self.current_move]['fen'])
            self.update_board()
        else:
            # Stop playback at the end
            self.play_btn.setChecked(False)
            self.timer.stop()
    
    def toggle_playback(self, checked):
        """Toggle playback on/off."""
        if checked:
            # Start playback
            speed = self.speed_spin.value()
            interval = int(1000 / speed)  # Convert to milliseconds
            self.timer.start(interval)
            self.play_btn.setText("⏸")
        else:
            # Stop playback
            self.timer.stop()
            self.play_btn.setText("▶")
    
    def on_game_changed(self, index):
        """Handle game selection change."""
        self.current_game_idx = index
        self.positions = self.games[index]
        self.current_move = 0
        self.board = chess.Board()
        self.position_slider.setRange(0, len(self.positions) - 1)
        
        # Update window title to show current game
        game_info = self.game_combo.currentText()
        self.setWindowTitle(f"RivalAI Game Viewer - {game_info}")
        
        self.update_board()
        self.populate_move_list()
        
        # Scroll to top of move list
        self.move_list.scrollToTop()
    
    def save_pgn(self):
        """Save the game as a PGN file with evaluation comments."""
        # Create a new game
        game = chess.pgn.Game()
        
        # Set game headers
        game.headers["Event"] = "RivalAI Self-Play"
        game.headers["Site"] = "RivalAI Engine"
        game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
        game.headers["Round"] = "1"
        game.headers["White"] = "RivalAI"
        game.headers["Black"] = "RivalAI"
        game.headers["Result"] = "*"  # Will be updated at the end
        
        # Create the game tree
        node = game
        for pos in self.positions:
            move = chess.Move.from_uci(pos['move'])
            
            # Add evaluation comment
            if pos.get('value') is not None:
                eval_str = f"Eval: {pos['value']:.2f}"
                node = node.add_variation(move, comment=eval_str)
            else:
                node = node.add_variation(move)
        
        # Update result if game is over
        if self.board.is_game_over():
            result = self.board.outcome().result()
            game.headers["Result"] = result
        
        # Get save filename
        default_name = f"rival_ai_game_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pgn"
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save PGN File",
            default_name,
            "PGN Files (*.pgn);;All Files (*.*)"
        )
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    f.write(str(game))
                QMessageBox.information(
                    self,
                    "Success",
                    f"Game saved successfully to:\n{filename}"
                )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to save PGN file:\n{str(e)}"
                )

def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python view_game.py <game_file> [model_path]")
        sys.exit(1)
    
    game_file = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    app = QApplication(sys.argv)
    viewer = GameViewer(game_file, model_path)
    viewer.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main() 