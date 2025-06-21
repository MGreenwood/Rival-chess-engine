"""
Opening book manager for handling opening repertoires and move selection.
"""

import chess
import chess.pgn
import io
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
import random
import numpy as np

@dataclass
class BookMove:
    """A move in the opening book."""
    move: chess.Move
    weight: float = 1.0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    
    @property
    def games(self) -> int:
        """Total number of games."""
        return self.wins + self.draws + self.losses
        
    @property
    def score(self) -> float:
        """Scoring percentage."""
        if self.games == 0:
            return 0.0
        return (self.wins + 0.5 * self.draws) / self.games
        
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'uci': self.move.uci(),
            'weight': self.weight,
            'wins': self.wins,
            'draws': self.draws,
            'losses': self.losses
        }
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'BookMove':
        """Create from dictionary."""
        return cls(
            move=chess.Move.from_uci(data['uci']),
            weight=data['weight'],
            wins=data['wins'],
            draws=data['draws'],
            losses=data['losses']
        )

@dataclass
class BookPosition:
    """A position in the opening book."""
    moves: Dict[chess.Move, BookMove] = field(default_factory=dict)
    total_games: int = 0
    
    def add_move(
        self,
        move: chess.Move,
        result: Optional[str] = None,
        weight: float = 1.0
    ):
        """Add a move to this position.
        
        Args:
            move: Chess move
            result: Optional game result
            weight: Move weight
        """
        if move not in self.moves:
            self.moves[move] = BookMove(move=move, weight=weight)
            
        book_move = self.moves[move]
        
        if result == "1-0":
            book_move.wins += 1
        elif result == "0-1":
            book_move.losses += 1
        elif result == "1/2-1/2":
            book_move.draws += 1
            
        self.total_games += 1
        
    def get_move(
        self,
        temperature: float = 1.0,
        randomization: float = 0.1
    ) -> Optional[chess.Move]:
        """Get a move from this position.
        
        Args:
            temperature: Sampling temperature
            randomization: Random move probability
            
        Returns:
            Selected move or None if no moves available
        """
        if not self.moves:
            return None
            
        # Possibly play a random move
        if random.random() < randomization:
            return random.choice(list(self.moves.keys()))
            
        # Calculate move weights
        weights = []
        moves = list(self.moves.keys())
        
        for move in moves:
            book_move = self.moves[move]
            # Combine static weight and performance
            weight = book_move.weight * (1 + book_move.score)
            # Apply temperature
            weight = weight ** (1 / temperature)
            weights.append(weight)
            
        # Normalize weights
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
            
        # Select move
        return np.random.choice(moves, p=weights)
        
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'moves': {
                move.uci(): book_move.to_dict()
                for move, book_move in self.moves.items()
            },
            'total_games': self.total_games
        }
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'BookPosition':
        """Create from dictionary."""
        position = cls(total_games=data['total_games'])
        for uci, move_data in data['moves'].items():
            move = chess.Move.from_uci(uci)
            position.moves[move] = BookMove.from_dict(move_data)
        return position

class OpeningBook:
    """Manages an opening repertoire."""
    
    def __init__(self, name: str):
        """Initialize the opening book.
        
        Args:
            name: Name of the opening book
        """
        self.name = name
        self.positions: Dict[str, BookPosition] = {}
        
    def add_game(
        self,
        moves: List[chess.Move],
        result: Optional[str] = None,
        max_moves: int = 20
    ):
        """Add a game to the book.
        
        Args:
            moves: List of moves in the game
            result: Game result
            max_moves: Maximum number of moves to add
        """
        board = chess.Board()
        
        for i, move in enumerate(moves):
            if i >= max_moves:
                break
                
            # Add position before move
            fen = board.fen().split(' ')[0]  # Only piece placement
            if fen not in self.positions:
                self.positions[fen] = BookPosition()
                
            # Add move
            self.positions[fen].add_move(move, result)
            
            # Make move
            board.push(move)
            
    def add_pgn_game(self, pgn_text: str, max_moves: int = 20):
        """Add a game from PGN text.
        
        Args:
            pgn_text: PGN game text
            max_moves: Maximum number of moves to add
        """
        game = chess.pgn.read_game(io.StringIO(pgn_text))
        if not game:
            return
            
        moves = []
        board = game.board()
        
        for move in game.mainline_moves():
            moves.append(move)
            
        self.add_game(moves, game.headers.get("Result"), max_moves)
        
    def add_pgn_file(
        self,
        pgn_path: str,
        max_games: Optional[int] = None,
        max_moves: int = 20
    ):
        """Add games from a PGN file.
        
        Args:
            pgn_path: Path to PGN file
            max_games: Maximum number of games to add
            max_moves: Maximum moves per game
        """
        games_added = 0
        
        with open(pgn_path) as pgn:
            while True:
                if max_games and games_added >= max_games:
                    break
                    
                game = chess.pgn.read_game(pgn)
                if not game:
                    break
                    
                moves = []
                board = game.board()
                
                for move in game.mainline_moves():
                    moves.append(move)
                    
                self.add_game(moves, game.headers.get("Result"), max_moves)
                games_added += 1
                
    def get_move(
        self,
        board: chess.Board,
        temperature: float = 1.0,
        randomization: float = 0.1
    ) -> Optional[chess.Move]:
        """Get a book move for a position.
        
        Args:
            board: Current position
            temperature: Sampling temperature
            randomization: Random move probability
            
        Returns:
            Selected move or None if position not in book
        """
        fen = board.fen().split(' ')[0]
        position = self.positions.get(fen)
        
        if position:
            return position.get_move(temperature, randomization)
        return None
        
    def save(self, path: str):
        """Save the opening book to a file.
        
        Args:
            path: Path to save to
        """
        data = {
            'name': self.name,
            'positions': {
                fen: position.to_dict()
                for fen, position in self.positions.items()
            }
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
            
    @classmethod
    def load(cls, path: str) -> 'OpeningBook':
        """Load an opening book from a file.
        
        Args:
            path: Path to load from
            
        Returns:
            Loaded opening book
        """
        with open(path) as f:
            data = json.load(f)
            
        book = cls(data['name'])
        for fen, position_data in data['positions'].items():
            book.positions[fen] = BookPosition.from_dict(position_data)
            
        return book
        
class OpeningManager:
    """Manages multiple opening books."""
    
    def __init__(self, books_dir: str = "data/openings"):
        """Initialize the opening manager.
        
        Args:
            books_dir: Directory for opening books
        """
        self.books_dir = books_dir
        self.books: Dict[str, OpeningBook] = {}
        
        # Create directory if needed
        os.makedirs(books_dir, exist_ok=True)
        
        # Load existing books
        self._load_books()
        
    def _load_books(self):
        """Load all opening books from directory."""
        for filename in os.listdir(self.books_dir):
            if filename.endswith('.json'):
                path = os.path.join(self.books_dir, filename)
                book = OpeningBook.load(path)
                self.books[book.name] = book
                
    def create_book(self, name: str) -> OpeningBook:
        """Create a new opening book.
        
        Args:
            name: Name for the book
            
        Returns:
            Created book
        """
        book = OpeningBook(name)
        self.books[name] = book
        return book
        
    def save_book(self, name: str):
        """Save an opening book.
        
        Args:
            name: Name of book to save
        """
        if name not in self.books:
            return
            
        path = os.path.join(self.books_dir, f"{name}.json")
        self.books[name].save(path)
        
    def get_move(
        self,
        board: chess.Board,
        book_weights: Optional[Dict[str, float]] = None,
        temperature: float = 1.0,
        randomization: float = 0.1
    ) -> Optional[chess.Move]:
        """Get a move from multiple opening books.
        
        Args:
            board: Current position
            book_weights: Optional weights for each book
            temperature: Sampling temperature
            randomization: Random move probability
            
        Returns:
            Selected move or None if no book moves available
        """
        if not book_weights:
            # Use uniform weights
            book_weights = {
                name: 1.0 / len(self.books)
                for name in self.books
            }
            
        # Get moves from each book
        moves = []
        weights = []
        
        for name, book in self.books.items():
            move = book.get_move(board, temperature, randomization)
            if move:
                moves.append(move)
                weights.append(book_weights[name])
                
        if not moves:
            return None
            
        # Normalize weights
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
            
        # Select move
        return np.random.choice(moves, p=weights)
        
    def add_pgn_file(
        self,
        book_name: str,
        pgn_path: str,
        max_games: Optional[int] = None,
        max_moves: int = 20
    ):
        """Add games from a PGN file to a book.
        
        Args:
            book_name: Name of book to add to
            pgn_path: Path to PGN file
            max_games: Maximum number of games to add
            max_moves: Maximum moves per game
        """
        if book_name not in self.books:
            self.create_book(book_name)
            
        self.books[book_name].add_pgn_file(pgn_path, max_games, max_moves)
        self.save_book(book_name) 