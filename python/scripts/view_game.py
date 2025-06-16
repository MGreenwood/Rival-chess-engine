"""
Interactive game viewer for analyzing self-play games.
"""

import sys
import json
import chess
import chess.svg
import chess.pgn
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QListWidget, QLabel, 
                            QSlider, QSpinBox, QSplitter, QFileDialog, QMessageBox,
                            QComboBox, QFrame)
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon, QPixmap

class GameViewer(QMainWindow):
    def __init__(self, game_file):
        super().__init__()
        self.setWindowTitle("RivalAI Game Viewer")
        self.game_file = game_file
        
        # Load game
        print(f"Loading game file: {game_file}")
        with open(game_file, 'r') as f:
            all_positions = json.load(f)
        print(f"Loaded {len(all_positions)} total positions")
        
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
        
        # Add game selector in a more prominent way
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
            # Get the first few moves to help identify the game
            moves = []
            for pos in game[:3]:  # Look at first 3 moves
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
        
        # Create board and move list layout
        board_list_layout = QHBoxLayout()
        
        # Create chess board widget
        self.svg_widget = QSvgWidget()
        self.svg_widget.setMinimumSize(400, 400)
        board_list_layout.addWidget(self.svg_widget)
        
        # Create move list widget
        move_list_layout = QVBoxLayout()
        move_list_layout.addWidget(QLabel("Moves:"))
        self.move_list = QListWidget()
        self.move_list.currentRowChanged.connect(self.on_move_selected)
        move_list_layout.addWidget(self.move_list)
        board_list_layout.addLayout(move_list_layout)
        
        layout.addLayout(board_list_layout)
        
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
        self.resize(800, 600)
    
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
    
    def update_board(self):
        """Update the board display."""
        # Generate SVG of current position
        svg = chess.svg.board(
            board=self.board,
            size=400,
            lastmove=self.board.peek() if self.board.move_stack else None
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
    
    def populate_move_list(self):
        """Populate the move list widget."""
        self.move_list.clear()
        print(f"Populating move list with {len(self.positions)} positions")
        for i, pos in enumerate(self.positions):
            print(f"Position {i}: {pos}")
            try:
                # Handle the last position which might not have a move
                if i == len(self.positions) - 1 and 'move' not in pos:
                    move_text = f"{i+1}. Game Over"
                    if pos.get('value') is not None:
                        move_text += f" (eval: {pos['value']:.2f})"
                    self.move_list.addItem(move_text)
                    continue

                move = chess.Move.from_uci(pos['move'])
                move_text = f"{i+1}. {move.uci()}"
                if pos.get('value') is not None:
                    move_text += f" (eval: {pos['value']:.2f})"
                self.move_list.addItem(move_text)
            except KeyError as e:
                print(f"Error at position {i}: {e}")
                print(f"Position data: {pos}")
                # For any other positions missing a move, add a placeholder
                move_text = f"{i+1}. [No Move]"
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

def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python view_game.py <game_file>")
        sys.exit(1)
    
    app = QApplication(sys.argv)
    viewer = GameViewer(sys.argv[1])
    viewer.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main() 