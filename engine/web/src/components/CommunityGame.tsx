import React, { useState, useEffect, useCallback } from 'react';
import { Chessboard } from 'react-chessboard';
import type { Square } from 'react-chessboard/dist/chessboard/types';
import { Chess } from 'chess.js';
import axios from 'axios';
import { formatDistanceToNowStrict } from 'date-fns';

const API_BASE_URL = 'http://localhost:3000';

interface CommunityGameState {
  game_id: string;
  board: string;
  status: string;
  move_history: string[];
  is_voting_phase: boolean;
  voting_ends_at: string | null;
  current_votes: Record<string, number>;
  total_voters: number;
  your_vote: string | null;
}

interface VoteResponse {
  success: boolean;
  error_message?: string;
  game_state: CommunityGameState;
}

export const CommunityGame: React.FC = () => {
  const [game] = useState(new Chess());
  const [gameState, setGameState] = useState<CommunityGameState | null>(null);
  const [selectedSquare, setSelectedSquare] = useState<Square | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [timeLeft, setTimeLeft] = useState<number>(0);
  const [voterId] = useState(() => `voter-${Math.random().toString(36).substr(2, 9)}`);

  // Update chess.js game state when server state changes
  useEffect(() => {
    if (gameState?.board) {
      game.load(gameState.board);
    }
  }, [gameState?.board, game]);

  // Fetch game state periodically
  useEffect(() => {
    const fetchGameState = async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/api/community/state`);
        setGameState(response.data);
      } catch (error) {
        console.error('Failed to fetch game state:', error);
      }
    };

    fetchGameState();
    const interval = setInterval(fetchGameState, 1000);
    return () => clearInterval(interval);
  }, []);

  // Update countdown timer
  useEffect(() => {
    if (!gameState?.voting_ends_at) {
      setTimeLeft(0);
      return;
    }

    const updateTimer = () => {
      const endTime = new Date(gameState.voting_ends_at!).getTime();
      const now = new Date().getTime();
      const diff = Math.max(0, Math.floor((endTime - now) / 1000));
      setTimeLeft(diff);
    };

    updateTimer();
    const interval = setInterval(updateTimer, 100);
    return () => clearInterval(interval);
  }, [gameState?.voting_ends_at]);

  const handleSquareClick = (square: Square) => {
    if (!gameState?.is_voting_phase) return;

    if (selectedSquare === null) {
      // Check if the clicked square has a piece that can move
      const moves = game.moves({ square, verbose: true });
      if (moves.length > 0) {
        setSelectedSquare(square);
      }
    } else {
      // Try to make the move
      const moveString = `${selectedSquare}${square}`;
      try {
        // Validate move with chess.js
        const move = game.move({
          from: selectedSquare,
          to: square,
          promotion: 'q' // Always promote to queen for simplicity
        });
        
        if (move) {
          // If move is valid, submit the vote
          handleVote(moveString);
          game.undo(); // Undo the move since the server will send the official state
        }
      } catch (error) {
        console.error('Invalid move:', error);
      }
      setSelectedSquare(null);
    }
  };

  const handleVote = async (moveString: string) => {
    try {
      const response = await axios.post<VoteResponse>(`${API_BASE_URL}/api/community/vote`, {
        move_str: moveString,
        voter_id: voterId,
      });

      if (!response.data.success) {
        setErrorMessage(response.data.error_message || 'Failed to vote');
      } else {
        setErrorMessage(null);
      }
      setGameState(response.data.game_state);
    } catch (error) {
      console.error('Failed to vote:', error);
      setErrorMessage('Failed to submit vote');
    }
  };

  const handleStartNewGame = async () => {
    try {
      const response = await axios.post<CommunityGameState>(`${API_BASE_URL}/api/community/start`);
      setGameState(response.data);
      setErrorMessage(null);
    } catch (error) {
      console.error('Failed to start new game:', error);
      setErrorMessage('Failed to start new game');
    }
  };

  const handleUndoVote = async () => {
    if (!gameState?.your_vote) return;
    try {
      const response = await axios.post<VoteResponse>(`${API_BASE_URL}/api/community/vote`, {
        move_str: '',  // Empty move string to clear vote
        voter_id: voterId,
      });
      setGameState(response.data.game_state);
    } catch (error) {
      console.error('Failed to undo vote:', error);
    }
  };

  if (!gameState) {
    return <div>Loading...</div>;
  }

  const sortedMoves = Object.entries(gameState.current_votes)
    .sort(([, a], [, b]) => b - a);

  const getStatusMessage = () => {
    if (!gameState) return null;
    
    if (gameState.status === 'waiting') {
      return (
        <div className="text-lg font-semibold text-gray-600">
          Waiting for first move... Make a move to start the game!
        </div>
      );
    }
    
    if (gameState.is_voting_phase) {
      return (
        <div className="text-blue-600 text-lg font-semibold">
          Voting Phase - {timeLeft}s remaining
        </div>
      );
    }
    
    switch (gameState.status) {
      case 'white_wins':
        return <div className="text-green-600 text-lg font-semibold">Checkmate! Community wins!</div>;
      case 'black_wins':
        return <div className="text-red-600 text-lg font-semibold">Checkmate! Engine wins!</div>;
      case 'draw_stalemate':
        return <div className="text-yellow-600 text-lg font-semibold">Game drawn by stalemate</div>;
      case 'draw_insufficient':
        return <div className="text-yellow-600 text-lg font-semibold">Game drawn by insufficient material</div>;
      case 'draw_repetition':
        return <div className="text-yellow-600 text-lg font-semibold">Game drawn by repetition</div>;
      case 'draw_fifty_moves':
        return <div className="text-yellow-600 text-lg font-semibold">Game drawn by fifty-move rule</div>;
      default:
        return <div className="text-gray-600 text-lg font-semibold">Waiting for next voting phase</div>;
    }
  };

  // Get possible moves for highlighting
  const possibleMoves: { [square: string]: { backgroundColor: string } } = {};
  if (selectedSquare) {
    const moves = game.moves({ square: selectedSquare, verbose: true });
    moves.forEach((move) => {
      possibleMoves[move.to] = { backgroundColor: 'rgba(255, 255, 0, 0.4)' };
    });
  }

  return (
    <div className="flex flex-col items-center p-4">
      <h1 className="text-2xl font-bold mb-4">Community vs AI</h1>
      
      {/* Game Status */}
      <div className="mb-4 text-center">
        {getStatusMessage()}
        {errorMessage && (
          <div className="text-red-500 mt-2">{errorMessage}</div>
        )}
      </div>

      {/* Chessboard */}
      <div className="w-full max-w-[600px] aspect-square mb-4">
        <Chessboard
          position={gameState.board}
          onSquareClick={handleSquareClick}
          customSquareStyles={{
            ...(selectedSquare && {
              [selectedSquare]: { backgroundColor: 'rgba(255, 255, 0, 0.4)' },
            }),
            ...possibleMoves,
            ...(gameState.your_vote && {
              [gameState.your_vote.slice(0, 2)]: { backgroundColor: 'rgba(0, 255, 0, 0.2)' },
              [gameState.your_vote.slice(2, 4)]: { backgroundColor: 'rgba(0, 255, 0, 0.2)' },
            }),
          }}
          boardOrientation="white"
          arePiecesDraggable={false}
        />
      </div>

      {/* Vote Tally */}
      <div className="w-full max-w-[600px] bg-white dark:bg-gray-800 rounded-lg p-4 mb-4">
        <h2 className="text-xl font-semibold mb-2">Current Votes</h2>
        <div className="space-y-2">
          {sortedMoves.map(([move, votes]) => (
            <div
              key={move}
              className="flex items-center justify-between p-2 bg-gray-50 dark:bg-gray-700 rounded"
            >
              <span className="font-mono">{move}</span>
              <div className="flex items-center space-x-2">
                <span className="text-sm">{votes} vote{votes !== 1 ? 's' : ''}</span>
                <div
                  className="h-2 bg-blue-500 rounded"
                  style={{ width: `${(votes / gameState.total_voters) * 100}px` }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Controls */}
      <div className="flex space-x-4">
        <button
          onClick={handleStartNewGame}
          className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors"
          disabled={gameState.is_voting_phase}
        >
          New Game
        </button>
        {gameState.your_vote && gameState.status !== 'waiting' && (
          <button
            onClick={handleUndoVote}
            className="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600 transition-colors"
            disabled={!gameState.is_voting_phase}
          >
            Undo Vote
          </button>
        )}
      </div>
    </div>
  );
}; 