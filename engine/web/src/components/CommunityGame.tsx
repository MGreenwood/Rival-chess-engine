import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Chessboard } from 'react-chessboard';
import type { Square } from 'react-chessboard/dist/chessboard/types';
import { Chess } from 'chess.js';
import axios from 'axios';

// Dynamic API base URL - inline to avoid import issues
const getApiBaseUrl = (): string => {
  const host = window.location.host;
  const protocol = window.location.protocol; // Use current protocol
  
  // If accessing via localhost, use localhost:3000 with current protocol
  if (host.includes('localhost') || host.includes('127.0.0.1')) {
    return `${protocol}//localhost:3000`;
  }
  
  // Otherwise use the current protocol and host (tunnel)
  return `${protocol}//${host}`;
};

const API_BASE_URL = getApiBaseUrl();

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
  experiment_name: string;
  waiting_for_first_move: boolean;
  can_vote: boolean;
  voting_started: boolean;
  engine_thinking: boolean;
}

interface VoteResponse {
  success: boolean;
  error_message?: string;
  game_state: CommunityGameState;
}

interface VoterSession {
  voter_id: string;
  token: string;
  expires_in: number;
}

export const CommunityGame: React.FC = () => {
  const [game] = useState(new Chess());
  const [gameState, setGameState] = useState<CommunityGameState | null>(null);
  const [selectedSquare, setSelectedSquare] = useState<Square | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [timeLeft, setTimeLeft] = useState<number>(0);
  const [voterSession, setVoterSession] = useState<VoterSession | null>(null);
  const [isTimerBroken, setIsTimerBroken] = useState(false);
  const [previewPosition, setPreviewPosition] = useState<string | null>(null);
  const [currentUserVote, setCurrentUserVote] = useState<string | null>(null);
  const [engineThinkingStartTime, setEngineThinkingStartTime] = useState<number | null>(null);

  // Create voter session on component mount
  useEffect(() => {
    const createSession = async () => {
      try {
        const response = await axios.post(`${API_BASE_URL}/api/community/session`);
        setVoterSession(response.data);
      } catch (error) {
        console.error('Failed to create voter session:', error);
        setErrorMessage('Failed to initialize voting session');
      }
    };
    createSession();
  }, []);

  // Track previous board position and move count to detect actual changes
  const previousBoardRef = useRef<string | null>(null);
  const previousMoveCountRef = useRef<number>(0);

  // Track engine thinking duration
  useEffect(() => {
    if (gameState?.engine_thinking && !engineThinkingStartTime) {
      // Engine just started thinking
      setEngineThinkingStartTime(Date.now());
      console.log('🤖 Engine started thinking at:', new Date().toISOString());
    } else if (!gameState?.engine_thinking && engineThinkingStartTime) {
      // Engine stopped thinking
      const duration = Date.now() - engineThinkingStartTime;
      console.log(`🤖 Engine finished thinking after ${duration}ms`);
      setEngineThinkingStartTime(null);
    }
  }, [gameState?.engine_thinking, engineThinkingStartTime]);

  // Update chess.js game state when server state changes
  useEffect(() => {
    if (gameState?.board) {
      game.load(gameState.board);
      
      const currentMoveCount = gameState.move_history?.length || 0;
      
      // Only clear preview if the move count increased (a new move was actually made)
      if (previousMoveCountRef.current < currentMoveCount) {
        // A real move was made - clear preview and user vote
        setPreviewPosition(null);
        setCurrentUserVote(null);
        console.log('New move detected, clearing preview and user vote');
      }
      
      // Update tracked values
      previousBoardRef.current = gameState.board;
      previousMoveCountRef.current = currentMoveCount;
    }
  }, [gameState?.board, gameState?.move_history?.length, game]);

  // Fetch game state periodically
  useEffect(() => {
    const fetchGameState = async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/api/community/state`);
        
        // Log engine thinking status for debugging
        if (response.data.engine_thinking !== gameState?.engine_thinking) {
          console.log('Engine thinking status changed:', response.data.engine_thinking);
        }
        
        setGameState(response.data);
      } catch (error) {
        console.error('Failed to fetch game state:', error);
      }
    };

    fetchGameState();
    const interval = setInterval(fetchGameState, 1000);
    return () => clearInterval(interval);
  }, [gameState?.engine_thinking]);

  // Update countdown timer with broken state detection
  useEffect(() => {
    console.log('Timer effect - voting_ends_at:', gameState?.voting_ends_at, 'is_voting_phase:', gameState?.is_voting_phase);
    
    if (!gameState?.voting_ends_at || !gameState?.is_voting_phase) {
      setTimeLeft(0);
      setIsTimerBroken(false);
      return;
    }

    const updateTimer = () => {
      const endTime = new Date(gameState.voting_ends_at!).getTime();
      const now = new Date().getTime();
      const diff = Math.max(0, Math.floor((endTime - now) / 1000));
      console.log('Timer update - endTime:', endTime, 'now:', now, 'diff:', diff);
      setTimeLeft(diff);
      
      // Detect if timer has been stuck at 0 for too long
      if (diff <= 0 && gameState?.is_voting_phase) {
        console.log('Voting phase should have ended, timer expired');
        setIsTimerBroken(true);
      }
    };

    updateTimer();
    const interval = setInterval(updateTimer, 1000);
    return () => clearInterval(interval);
  }, [gameState?.voting_ends_at, gameState?.is_voting_phase]);

  const handleSquareClick = useCallback((square: Square) => {
    if (!gameState?.can_vote || !voterSession || gameState?.engine_thinking) {
      if (gameState?.engine_thinking) {
        setErrorMessage('Please wait - engine is thinking');
      } else {
        setErrorMessage('Cannot vote at this time');
      }
      return;
    }

    if (selectedSquare === null) {
      // Check if the clicked square has a piece that can move
      const moves = game.moves({ square, verbose: true });
      if (moves.length > 0) {
        setSelectedSquare(square);
      }
    } else {
      // Try to validate the move
      const moveString = `${selectedSquare}${square}`;
      const moves = game.moves({ verbose: true });
      const isValidMove = moves.some(move => 
        move.from === selectedSquare && move.to === square
      );
      
      if (isValidMove) {
        // If move is valid, submit the vote
        handleVote(moveString);
      }
      setSelectedSquare(null);
    }
  }, [game, gameState?.can_vote, gameState?.engine_thinking, voterSession, selectedSquare]);

  const handlePieceDrop = useCallback((sourceSquare: Square, targetSquare: Square) => {
    if (!gameState?.can_vote || !voterSession || gameState?.engine_thinking) {
      if (gameState?.engine_thinking) {
        setErrorMessage('Please wait - engine is thinking');
      } else {
        setErrorMessage('Cannot vote at this time');
      }
      return false;
    }

    // Try to validate the move
    const moveString = `${sourceSquare}${targetSquare}`;
    const moves = game.moves({ verbose: true });
    const isValidMove = moves.some(move => 
      move.from === sourceSquare && move.to === targetSquare
    );
    
    if (isValidMove) {
      // If move is valid, submit the vote
      handleVote(moveString);
      setSelectedSquare(null); // Clear any selected square
      return true;
    }
    
    return false; // Invalid move
  }, [game, gameState?.can_vote, gameState?.engine_thinking, voterSession]);

  const handleVote = async (moveString: string) => {
    if (!voterSession) {
      setErrorMessage('No active voting session');
      return;
    }
    
    if (gameState?.engine_thinking) {
      setErrorMessage('Please wait - engine is thinking');
      return;
    }

    // Create preview position with the voted move
    try {
      const previewGame = new Chess(gameState?.board);
      const result = previewGame.move(moveString);
      if (result) {
        setPreviewPosition(previewGame.fen());
      }
    } catch (previewError) {
      console.warn('Could not create preview position:', previewError);
      // Don't show error to user for preview issues
    }

    try {
      const response = await axios.post<VoteResponse>(`${API_BASE_URL}/api/community/vote`, {
        move_str: moveString,
        token: voterSession.token,
      });

      if (!response.data.success) {
        setErrorMessage(response.data.error_message || 'Failed to vote');
        setPreviewPosition(null); // Clear preview on error
        setCurrentUserVote(null); // Clear user vote on error
      } else {
        setGameState(response.data.game_state);
        setCurrentUserVote(moveString); // Track the user's vote locally
        setErrorMessage(null);
      }
    } catch (error: any) {
      console.error('Failed to vote:', error);
      
      // Only handle actual HTTP errors, not client-side issues
      if (error.response) {
        if (error.response.status === 400) {
          setErrorMessage('Cannot vote right now - voting phase may have ended or it\'s not your turn.');
        } else if (error.response.status === 401) {
          // Token expired or invalid, try to create new session
          const createSession = async () => {
            try {
              const response = await axios.post(`${API_BASE_URL}/api/community/session`);
              setVoterSession(response.data);
              setErrorMessage('Session refreshed - please try voting again!');
            } catch (sessionError) {
              console.error('Failed to create new voter session:', sessionError);
              setErrorMessage('Session expired. Please refresh the page.');
            }
          };
          createSession();
        } else if (error.response.status >= 400) {
          setErrorMessage(error.response?.data?.error_message || 'Failed to submit vote - please try again');
        }
        setPreviewPosition(null); // Clear preview on error
        setCurrentUserVote(null); // Clear user vote on error
      } else {
        // Network error or other non-HTTP issue - don't show error popup
        console.warn('Vote submission failed (non-HTTP error):', error.message);
      }
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
    if (!userVote || !voterSession) return;
    try {
      const response = await axios.post<VoteResponse>(`${API_BASE_URL}/api/community/vote`, {
        move_str: '',  // Empty move string to clear vote
        token: voterSession.token,
      });
      setGameState(response.data.game_state);
      setPreviewPosition(null); // Clear preview when undoing vote
      setCurrentUserVote(null); // Clear user vote when undoing
    } catch (error) {
      console.error('Failed to undo vote:', error);
    }
  };

  const handleForceResolveVoting = async () => {
    if (!voterSession) {
      setErrorMessage('No active voting session');
      return;
    }

    try {
      const response = await axios.post(`${API_BASE_URL}/api/community/force-resolve`, {
        token: voterSession.token,
      });

      if (response.data.success) {
        setGameState(response.data.game_state);
        setErrorMessage(null);
        setIsTimerBroken(false);
      } else {
        setErrorMessage(response.data.error_message || 'Failed to resolve voting');
      }
    } catch (error: any) {
      console.error('Failed to force resolve voting:', error);
      // If the endpoint doesn't exist, suggest starting a new game
      if (error.response?.status === 404) {
        setErrorMessage('Voting appears stuck. Try starting a new game to reset the state.');
      } else {
        setErrorMessage('Failed to resolve voting - please try starting a new game');
      }
    }
  };

  const getStatusMessage = () => {
    if (!gameState) return "Loading...";
    if (!voterSession) return "Initializing voting session...";

    // Engine thinking has highest priority
    if (gameState.engine_thinking) {
      console.log('🤖 Displaying engine thinking status');
      
      // Check if engine has been thinking for too long
      if (engineThinkingStartTime) {
        const thinkingDuration = Date.now() - engineThinkingStartTime;
        const thinkingSeconds = Math.floor(thinkingDuration / 1000);
        
        if (thinkingDuration > 15000) { // 15 seconds
          return `⚠️ Engine is taking unusually long (${thinkingSeconds}s) - this may indicate an issue. Try refreshing if this persists.`;
        } else if (thinkingDuration > 5000) { // 5 seconds
          return `🤖 Engine is thinking... (${thinkingSeconds}s elapsed, max ~10s)`;
        }
      }
      
      return "🤖 Engine is thinking... please wait (may take up to 10 seconds)";
    }

    // Handle broken timer state (should be very rare now with auto-processing)
    if (isTimerBroken) {
      return "⚠️ Voting issue detected - use controls below to fix";
    }

    if (gameState.waiting_for_first_move) {
      return "🎯 Make the first move to begin! Click or drag pieces to vote. Timer starts automatically.";
    }

    if (gameState.is_voting_phase && timeLeft > 0) {
      return `⏰ Voting active - ${timeLeft} seconds remaining! Cast your vote now.`;
    }

    if (gameState.is_voting_phase && timeLeft <= 0) {
      return "⏳ Voting ended - processing results...";
    }

    if (gameState.can_vote && !gameState.is_voting_phase) {
      return "🎯 Ready to vote! Click or drag pieces to vote. First vote starts the 10-second countdown.";
    }

    return "⏸️ Waiting for next turn...";
  };

  // Determine user's vote - prioritize local state for persistence during voting
  const userVote = currentUserVote || gameState?.your_vote;

  // Get possible moves for highlighting
  const customSquareStyles: { [square: string]: { backgroundColor: string } } = {};
  
  // Highlight the last move made (engine or community move)
  if (gameState?.move_history && gameState.move_history.length > 0) {
    const lastMove = gameState.move_history[gameState.move_history.length - 1];
    if (lastMove && lastMove.length >= 4) {
      const from = lastMove.slice(0, 2);
      const to = lastMove.slice(2, 4);
      customSquareStyles[from] = { backgroundColor: 'rgba(173, 216, 230, 0.6)' }; // Light blue
      customSquareStyles[to] = { backgroundColor: 'rgba(173, 216, 230, 0.6)' }; // Light blue
    }
  }
  
  // Highlight selected square (higher priority than last move)
  if (selectedSquare) {
    customSquareStyles[selectedSquare] = { backgroundColor: 'rgba(255, 255, 0, 0.4)' };
    
    // Highlight possible moves
    const moves = game.moves({ square: selectedSquare, verbose: true });
    moves.forEach((move) => {
      customSquareStyles[move.to] = { backgroundColor: 'rgba(255, 255, 0, 0.2)' };
    });
  }

  // Highlight your current vote (highest priority)
  if (userVote) {
    const from = userVote.slice(0, 2);
    const to = userVote.slice(2, 4);
    customSquareStyles[from] = { backgroundColor: 'rgba(0, 255, 0, 0.4)' };
    customSquareStyles[to] = { backgroundColor: 'rgba(0, 255, 0, 0.4)' };
  }

  if (!gameState) {
    return <div>Loading...</div>;
  }

  return (
    <div className="flex flex-col items-center w-full max-w-4xl mx-auto">
      {/* Header */}
      <div className="flex flex-col items-center mb-6">
        <h2 className="text-2xl font-bold">Community vs AI</h2>
        {gameState.experiment_name && (
          <div className="text-lg text-gray-600">
            Playing against: {gameState.experiment_name}
          </div>
        )}
      </div>

      {/* Game Status - Moved to top for visibility */}
      <div className="mb-6 text-center w-full">
        <div className="text-lg font-semibold mb-4">
          {getStatusMessage()}
        </div>
        
        {/* Error Message - Prominently displayed at top */}
        {errorMessage && (
          <div className="mb-4 text-center text-red-500 bg-red-50 border border-red-200 rounded-lg px-4 py-2 mx-auto max-w-2xl">
            {errorMessage}
          </div>
        )}


        
        {/* Your Current Vote - Also prominently displayed */}
        {userVote && (
          <div className="bg-green-100 border border-green-300 text-green-800 px-4 py-2 rounded-lg mb-4 mx-auto max-w-2xl">
            ✅ Your vote: <span className="font-mono font-bold">{userVote}</span>
          </div>
        )}

        {/* Emergency Recovery - Should be rare with auto-processing */}
        {isTimerBroken && (
          <div className="bg-red-100 border border-red-300 text-red-800 px-4 py-3 rounded-lg mb-4 mx-auto max-w-2xl">
            <div className="font-semibold mb-2">⚠️ Voting Issue Detected</div>
            <div className="text-sm mb-3">
              Something went wrong with vote processing. Click below to resolve:
            </div>
            <div className="flex gap-2 justify-center">
              <button
                onClick={handleForceResolveVoting}
                className="px-3 py-1 bg-yellow-600 hover:bg-yellow-700 text-white rounded text-sm"
              >
                Force Process Votes
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Chessboard - Centered */}
      <div className="w-full max-w-[600px] aspect-square mb-6">
        <Chessboard
          position={previewPosition || gameState.board}
          onSquareClick={handleSquareClick}
          onPieceDrop={handlePieceDrop}
          boardWidth={600}
          customSquareStyles={customSquareStyles}
          boardOrientation="white"
          arePiecesDraggable={gameState?.can_vote && !gameState?.engine_thinking}
        />
      </div>

      {/* Vote Tally - Below chessboard */}
      {gameState.current_votes && Object.keys(gameState.current_votes).length > 0 && (
        <div className="mb-4 p-4 bg-gray-800 rounded-lg w-full max-w-2xl">
          <h3 className="text-lg font-semibold mb-2 text-white text-center">
            Current Votes
          </h3>
          <div className="grid grid-cols-2 gap-4">
            {Object.entries(gameState.current_votes)
              .sort(([,a], [,b]) => b - a) // Sort by vote count descending
              .map(([move, count]) => (
              <div key={move} className="flex justify-between text-white">
                <span className="font-mono">{move}</span>
                <span className="ml-4 font-bold">{count} vote{count !== 1 ? 's' : ''}</span>
              </div>
            ))}
          </div>
          {Object.keys(gameState.current_votes).length > 0 && !gameState.is_voting_phase && (
            <div className="mt-2 text-sm text-gray-400 text-center">
              Timer finished - processing votes...
            </div>
          )}
        </div>
      )}

      {/* Controls - At bottom */}
      <div className="flex flex-wrap gap-4 justify-center">
        {userVote && (
          <button
            onClick={handleUndoVote}
            className="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600"
          >
            Undo Vote
          </button>
        )}
        
        {gameState.status !== 'active' && (
          <button
            onClick={handleStartNewGame}
            className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600"
          >
            Start New Game
          </button>
        )}
      </div>
    </div>
  );
}; 