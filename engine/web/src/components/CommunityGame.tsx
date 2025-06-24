import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Chessboard } from 'react-chessboard';
import type { Square } from 'react-chessboard/dist/chessboard/types';
import { Chess } from 'chess.js';
import axios from 'axios';
import DonationButton from './DonationButton';

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
  const [engineThinkingElapsed, setEngineThinkingElapsed] = useState<number>(0);
  const [voteProcessingStartTime, setVoteProcessingStartTime] = useState<number | null>(null);
  const [showProcessingOverlay, setShowProcessingOverlay] = useState<boolean>(false);
  const [lastVoteResults, setLastVoteResults] = useState<Record<string, number> | null>(null);
  const [boardSize, setBoardSize] = useState<number>(600);
  const [isDragging, setIsDragging] = useState(false);
  const [isPlayerInCheck, setIsPlayerInCheck] = useState(false);
  const boardContainerRef = useRef<HTMLDivElement>(null);

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

  // Handle responsive board sizing
  useEffect(() => {
    const updateBoardSize = () => {
      // Calculate available width considering sidebars and padding on large screens
      const isLargeScreen = window.innerWidth >= 1024; // lg breakpoint
      let availableWidth = window.innerWidth;
      
      if (isLargeScreen) {
        // On large screens, account for both sidebars (2 * 320px) plus padding
        availableWidth = window.innerWidth - (2 * 320) - 64; // 320px per sidebar + padding
      } else {
        // On mobile/tablet, just account for padding
        availableWidth = window.innerWidth - 32;
      }
      
      const maxSize = Math.min(600, availableWidth);
      const minSize = 280;
      setBoardSize(Math.max(minSize, maxSize));
    };

    updateBoardSize();
    window.addEventListener('resize', updateBoardSize);
    return () => window.removeEventListener('resize', updateBoardSize);
  }, []);

  // Track previous board position and move count to detect actual changes
  const previousBoardRef = useRef<string | null>(null);
  const previousMoveCountRef = useRef<number>(0);

  // Track engine thinking duration
  useEffect(() => {
    if (gameState?.engine_thinking && !engineThinkingStartTime) {
      // Engine just started thinking
      setEngineThinkingStartTime(Date.now());
      setEngineThinkingElapsed(0);
      console.log('🤖 Engine started thinking at:', new Date().toISOString());
    } else if (!gameState?.engine_thinking && engineThinkingStartTime) {
      // Engine stopped thinking
      const duration = Date.now() - engineThinkingStartTime;
      console.log(`🤖 Engine finished thinking after ${duration}ms`);
      setEngineThinkingStartTime(null);
      setEngineThinkingElapsed(0);
    }
  }, [gameState?.engine_thinking, engineThinkingStartTime]);

  // Update elapsed time for engine thinking overlay
  useEffect(() => {
    if (gameState?.engine_thinking && engineThinkingStartTime) {
      const timer = setInterval(() => {
        setEngineThinkingElapsed(Math.floor((Date.now() - engineThinkingStartTime) / 1000));
      }, 1000);
      return () => clearInterval(timer);
    }
  }, [gameState?.engine_thinking, engineThinkingStartTime]);

  // Add right-click handler to cancel dragging
  useEffect(() => {
    const forceCancelDrag = () => {
      console.log('🔴 FORCE CANCEL DRAG INITIATED');
      
      // Cancel our internal state
      setSelectedSquare(null);
      setIsDragging(false);
      
      // Force reset ALL pieces to their original positions
      const allPieces = document.querySelectorAll('[data-piece]');
      console.log(`🔴 Found ${allPieces.length} pieces to reset`);
      
      allPieces.forEach((piece, index) => {
        const element = piece as HTMLElement;
        
        // Clear all drag-related styles
        element.style.transform = '';
        element.style.transition = '';
        element.style.zIndex = '';
        element.style.position = '';
        element.style.left = '';
        element.style.top = '';
        element.style.pointerEvents = '';
        element.style.cursor = '';
        
        // Remove any drag-related classes
        element.classList.remove('dragging', 'piece-dragging');
        
        console.log(`🔴 Reset piece ${index + 1}: cleared all drag styles`);
      });
      
      // Force re-render by updating the board position if we have game state
      if (gameState?.board) {
        // Create a small state change to force re-render
        const currentBoard = gameState.board;
        setGameState(prev => prev ? { ...prev, board: currentBoard } : null);
        console.log('🔴 Forced board re-render');
      }
    };

    const handleMouseDown = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      const boardContainer = boardContainerRef.current;
      
      console.log('🟡 MouseDown detected - button:', e.button, 'target:', target.tagName);
      
      if (boardContainer && boardContainer.contains(target)) {
        if (e.button === 2) {
          // Right-click
          console.log('🔴 RIGHT-CLICK DETECTED via mousedown');
          e.preventDefault();
          e.stopPropagation();
          e.stopImmediatePropagation();
          forceCancelDrag();
          return false;
        }
        
        // Left-click on piece - start dragging
        if (target.closest('[data-piece]') && e.button === 0) {
          setIsDragging(true);
          console.log('🔵 Started dragging piece');
        }
      }
    };

    const handleContextMenu = (e: MouseEvent) => {
      const boardContainer = boardContainerRef.current;
      const target = e.target as HTMLElement;
      
      console.log('🟡 ContextMenu detected on:', target.tagName);
      
      if (boardContainer && boardContainer.contains(target)) {
        console.log('🔴 RIGHT-CLICK DETECTED via contextmenu');
        e.preventDefault();
        e.stopPropagation();
        e.stopImmediatePropagation();
        forceCancelDrag();
        return false;
      }
    };

    const handleMouseUp = (_e: MouseEvent) => {
      // Clear dragging state on mouse up
      if (isDragging) {
        setIsDragging(false);
        console.log('🔵 Finished dragging');
      }
    };

    // Be more aggressive - add listeners to both document and window
    const options = { capture: true, passive: false };
    
    document.addEventListener('mousedown', handleMouseDown, options);
    document.addEventListener('contextmenu', handleContextMenu, options);
    document.addEventListener('mouseup', handleMouseUp, options);
    
    // Also add to window as backup
    window.addEventListener('mousedown', handleMouseDown, options);
    window.addEventListener('contextmenu', handleContextMenu, options);
    
    console.log('🟡 Event listeners attached');
    
    return () => {
      document.removeEventListener('mousedown', handleMouseDown, options);
      document.removeEventListener('contextmenu', handleContextMenu, options);
      document.removeEventListener('mouseup', handleMouseUp, options);
      
      window.removeEventListener('mousedown', handleMouseDown, options);
      window.removeEventListener('contextmenu', handleContextMenu, options);
      
      console.log('🟡 Event listeners removed');
    };
  }, [isDragging, gameState?.board]);

  // Update chess.js game state when server state changes
  useEffect(() => {
    if (gameState?.board) {
      game.load(gameState.board);
      
      // Check for check status
      setIsPlayerInCheck(game.inCheck());
      
      // Log check status for debugging
      if (game.inCheck()) {
        console.log('🔴 Player is in CHECK!');
      }
      
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

  // Fetch game state periodically - use faster polling during critical moments
  useEffect(() => {
    const fetchGameState = async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/api/community/state`);
        
        // Log engine thinking status for debugging
        if (response.data.engine_thinking !== gameState?.engine_thinking) {
          console.log('🚨 Engine thinking status changed:', response.data.engine_thinking);
          console.log('🚨 Full game state:', response.data);
        }
        
        // Log when engine starts thinking for debugging
        if (response.data.engine_thinking && !gameState?.engine_thinking) {
          console.log('🤖 ENGINE STARTED THINKING - OVERLAY SHOULD APPEAR NOW!');
        }
        
        // Log when engine stops thinking
        if (!response.data.engine_thinking && gameState?.engine_thinking) {
          console.log('🔓 ENGINE STOPPED THINKING - OVERLAY SHOULD DISAPPEAR');
        }
        
        setGameState(response.data);
      } catch (error) {
        console.error('Failed to fetch game state:', error);
      }
    };

    fetchGameState();
    
    // Use faster polling (500ms) to catch engine thinking states more reliably
    // During engine thinking or vote processing, we need quicker updates
    const isActivePhase = gameState?.is_voting_phase || gameState?.engine_thinking || 
                         (gameState?.is_voting_phase === false && timeLeft <= 0);
    const pollInterval = isActivePhase ? 500 : 1000;
    
    const interval = setInterval(fetchGameState, pollInterval);
    return () => clearInterval(interval);
  }, [gameState?.engine_thinking, gameState?.is_voting_phase, timeLeft]);

  // Simplified overlay management - single source of truth
  useEffect(() => {
    // CASE 1: Engine is actively thinking → show overlay
    if (gameState?.engine_thinking) {
      if (!showProcessingOverlay) {
        console.log('🤖 ENGINE THINKING - SHOWING OVERLAY');
        setShowProcessingOverlay(true);
        if (!engineThinkingStartTime) {
          setEngineThinkingStartTime(Date.now());
        }
      }
      return; // Early exit - engine thinking takes priority
    }

    // CASE 2: Voting ended, votes exist, but engine hasn't moved yet → show overlay  
    if (!gameState?.is_voting_phase && 
        gameState?.current_votes && 
        Object.keys(gameState.current_votes).length > 0 && 
        !showProcessingOverlay) {
      console.log('🗳️ VOTES NEED PROCESSING - SHOWING OVERLAY');
      setLastVoteResults(gameState.current_votes);
      setVoteProcessingStartTime(Date.now());
      setShowProcessingOverlay(true);
      return;
    }

    // CASE 3: New voting phase started → hide overlay
    if (gameState?.is_voting_phase && showProcessingOverlay) {
      console.log('🗳️ NEW VOTING STARTED - HIDING OVERLAY');
      setShowProcessingOverlay(false);
      setVoteProcessingStartTime(null);
      setEngineThinkingStartTime(null);
      setEngineThinkingElapsed(0);
      return;
    }

    // CASE 4: Engine stopped thinking and no votes to process → hide overlay
    if (!gameState?.engine_thinking && 
        (!gameState?.current_votes || Object.keys(gameState.current_votes).length === 0) && 
        showProcessingOverlay) {
      console.log('🔓 ENGINE DONE, NO VOTES - HIDING OVERLAY');
      setShowProcessingOverlay(false);
      setVoteProcessingStartTime(null);
      setEngineThinkingStartTime(null);
      setEngineThinkingElapsed(0);
    }
  }, [gameState?.engine_thinking, gameState?.is_voting_phase, gameState?.current_votes]);

  // Hide overlay when new moves are detected (engine finished)
  useEffect(() => {
    const currentMoveCount = gameState?.move_history?.length || 0;
    if (previousMoveCountRef.current < currentMoveCount && showProcessingOverlay) {
      console.log('🎯 NEW MOVE DETECTED - HIDING OVERLAY');
      setShowProcessingOverlay(false);
      setVoteProcessingStartTime(null);
      setEngineThinkingStartTime(null);
      setEngineThinkingElapsed(0);
    }
  }, [gameState?.move_history?.length]);

  // Update countdown timer with improved broken state detection
  useEffect(() => {
    console.log('Timer effect - voting_ends_at:', gameState?.voting_ends_at, 'is_voting_phase:', gameState?.is_voting_phase);
    
    if (!gameState?.voting_ends_at || !gameState?.is_voting_phase) {
      setTimeLeft(0);
      setIsTimerBroken(false);
      setVoteProcessingStartTime(null);
      return;
    }

    const updateTimer = () => {
      const endTime = new Date(gameState.voting_ends_at!).getTime();
      const now = new Date().getTime();
      const diff = Math.max(0, Math.floor((endTime - now) / 1000));
      console.log('Timer update - endTime:', endTime, 'now:', now, 'diff:', diff);
      setTimeLeft(diff);
      
      // Check if we've been waiting too long for vote processing to complete
      if (diff <= 0 && gameState?.is_voting_phase && !gameState?.engine_thinking) {
        if (voteProcessingStartTime) {
          const processingTime = Date.now() - voteProcessingStartTime;
          // Only flag as broken if we've been waiting more than 8 seconds for vote processing
          if (processingTime > 8000) {
            console.log(`Vote processing stuck for ${processingTime}ms - flagging as broken`);
            setIsTimerBroken(true);
          } else {
            console.log(`Vote processing in progress: ${processingTime}ms elapsed (max 8s)`);
          }
        }
      } else if (gameState?.engine_thinking || !gameState?.is_voting_phase) {
        // Clear broken state if engine starts thinking OR voting phase ends (normal progression)
        setIsTimerBroken(false);
      }
    };

    updateTimer();
    const interval = setInterval(updateTimer, 1000);
    return () => clearInterval(interval);
  }, [gameState?.voting_ends_at, gameState?.is_voting_phase]);

  const handleSquareClick = useCallback((square: Square) => {
    // Clear dragging state on click
    setIsDragging(false);
    
    if (!gameState?.can_vote || !voterSession || gameState?.engine_thinking) {
      return; // Silently reject when not able to vote
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
    // Clear dragging state when drop occurs
    setIsDragging(false);
    
    if (!gameState?.can_vote || !voterSession || gameState?.engine_thinking) {
      return false; // Silently reject when not able to vote
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

  // Determine board orientation based on community color
  const getBoardOrientation = (): "white" | "black" => {
    // Extract from experiment name which includes the community's color
    if (gameState?.experiment_name?.includes("plays as Black")) {
      return "black";
    }
    
    // Default to white orientation
    return "white";
  };

  const boardOrientation = getBoardOrientation();

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
    return <div className="text-gray-900 dark:text-gray-100">Loading...</div>;
  }

  return (
    <div className="flex flex-col items-center w-full max-w-3xl mx-auto">
      {/* Header */}
      <div className="flex flex-col items-center mb-3">
        <h2 className="text-xl font-bold text-gray-900 dark:text-gray-100">Community vs AI</h2>
        {gameState.experiment_name && (
          <div className="text-base text-gray-600 dark:text-gray-300">
            Playing against: {gameState.experiment_name}
          </div>
        )}
      </div>

      {/* Game Status - Moved to top for visibility */}
      <div className="mb-3 text-center w-full">
        <div className="text-base font-semibold mb-2 text-gray-900 dark:text-gray-100">
          {getStatusMessage()}
        </div>
        
        {/* Error Message - Prominently displayed at top */}
        {errorMessage && (
          <div className="mb-2 text-center text-red-500 bg-red-50 border border-red-200 rounded-lg px-3 py-2 mx-auto max-w-xl">
            {errorMessage}
          </div>
        )}

        {/* Your Current Vote - Also prominently displayed */}
        {userVote && (
          <div className="bg-green-100 border border-green-300 text-green-800 px-3 py-2 rounded-lg mb-2 mx-auto max-w-xl">
            ✅ Your vote: <span className="font-mono font-bold">{userVote}</span>
          </div>
        )}

        {/* Emergency Recovery - Should be rare with auto-processing */}
        {/* Only show if timer is broken AND engine is not thinking */}
        {isTimerBroken && !gameState?.engine_thinking && (
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

      {/* Chessboard - Centered with Engine Thinking Overlay */}
      <div 
        className={`border-2 rounded-lg p-2 mb-3 ${isPlayerInCheck ? 'border-red-500' : 'border-gray-200 dark:border-gray-600'}`}
        style={{ 
          maxWidth: `${boardSize + 16}px`,
          boxShadow: isPlayerInCheck 
            ? '0 0 0 4px rgba(239, 68, 68, 0.3), 0 2px 8px rgba(0, 0, 0, 0.2)' 
            : undefined,
          transition: 'box-shadow 0.3s ease-in-out'
        }}
      >
        <div ref={boardContainerRef} className="w-full aspect-square relative">

        
        <Chessboard
          position={previewPosition || gameState.board}
          onSquareClick={handleSquareClick}
          onPieceDrop={handlePieceDrop}
          boardWidth={boardSize}
          customSquareStyles={customSquareStyles}
          boardOrientation={boardOrientation}
          arePiecesDraggable={gameState?.can_vote && !gameState?.engine_thinking}
        />
        
                {/* Engine Thinking/Vote Processing Overlay */}
        {(gameState?.engine_thinking || showProcessingOverlay) && (
          <div className="absolute inset-0 bg-black dark:bg-black bg-opacity-60 dark:bg-opacity-70 flex items-center justify-center z-50">
            <div className="bg-white dark:bg-gray-800 rounded-lg px-8 py-6 shadow-2xl text-center border-2 border-blue-500 dark:border-blue-400 max-w-md transition-colors">
              <div className="flex items-center justify-center space-x-3 mb-4">
                <div className="animate-spin rounded-full h-10 w-10 border-b-4 border-blue-500 dark:border-blue-400"></div>
                <div className="text-2xl font-bold text-gray-800 dark:text-gray-100">
                  🤖 Engine is thinking...
                </div>
              </div>
              
              {/* Vote Results */}
              {lastVoteResults && Object.keys(lastVoteResults).length > 0 && (
                <div className="mb-4">
                  <div className="text-lg font-semibold text-gray-700 dark:text-gray-200 mb-3">
                    🗳️ Vote Results
                  </div>
                  {Object.entries(lastVoteResults)
                    .sort(([,a], [,b]) => b - a) // Sort by vote count descending
                    .map(([move, count], index) => {
                      const totalVotes = Object.values(lastVoteResults).reduce((sum, c) => sum + c, 0);
                      const percentage = totalVotes > 0 ? Math.round((count / totalVotes) * 100) : 0;
                      const isWinner = index === 0;
                      
                      return (
                        <div key={move} className={`flex justify-between items-center py-2 px-3 rounded transition-colors ${
                          isWinner ? 'bg-green-100 dark:bg-green-900 border-2 border-green-400 dark:border-green-500' : 'bg-gray-50 dark:bg-gray-700'
                        }`}>
                          <div className="flex items-center space-x-2">
                            {isWinner && <span className="text-green-600 dark:text-green-400 font-bold">👑</span>}
                            <span className={`font-mono ${isWinner ? 'text-xl font-bold text-green-800 dark:text-green-200' : 'text-lg text-gray-800 dark:text-gray-200'}`}>
                              {move}
                            </span>
                            {isWinner && <span className="text-green-600 dark:text-green-400 text-sm font-semibold">WINNER</span>}
                          </div>
                          <div className="text-right">
                            <div className={`${isWinner ? 'text-lg font-bold text-green-800 dark:text-green-200' : 'text-sm font-semibold text-gray-700 dark:text-gray-300'}`}>
                              {count} vote{count !== 1 ? 's' : ''}
                            </div>
                            <div className="text-xs text-gray-500 dark:text-gray-400">
                              {percentage}%
                            </div>
                          </div>
                        </div>
                      );
                    })}
                </div>
              )}
              
              {engineThinkingStartTime && (
                <div className="text-lg text-blue-600 dark:text-blue-400 font-semibold mb-2">
                  {engineThinkingElapsed}s elapsed
                </div>
              )}
              
              <div className="text-sm text-gray-600 dark:text-gray-300 mb-4">
                Running MCTS simulations... (up to 10s)
              </div>
              
              {/* Perfect captive audience moment for donation! */}
              <div className="border-t pt-4 mt-4 border-gray-200 dark:border-gray-600">
                <DonationButton 
                  context="thinking"
                  size="small"
                  variant="ghost"
                  className="text-center scale-90"
                />
              </div>
            </div>
          </div>
        )}
        </div>
      </div>

      {/* Vote Tally - Below chessboard */}
      {gameState.current_votes && Object.keys(gameState.current_votes).length > 0 && (
        <div className="mb-2 p-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg w-full max-w-xl transition-colors">
          <h3 className="text-base font-semibold mb-2 text-gray-900 dark:text-white text-center">
            Current Votes
          </h3>
          <div className="grid grid-cols-2 gap-4">
            {Object.entries(gameState.current_votes)
              .sort(([,a], [,b]) => b - a) // Sort by vote count descending
              .map(([move, count]) => (
              <div key={move} className="flex justify-between text-gray-900 dark:text-white">
                <span className="font-mono">{move}</span>
                <span className="ml-4 font-bold">{count} vote{count !== 1 ? 's' : ''}</span>
              </div>
            ))}
          </div>
          {Object.keys(gameState.current_votes).length > 0 && !gameState.is_voting_phase && (
            <div className="mt-2 text-sm text-gray-500 dark:text-gray-400 text-center">
              Timer finished - processing votes...
            </div>
          )}
        </div>
      )}

      {/* Controls - At bottom - Only show when there are controls */}
      {(userVote || gameState.status !== 'active') && (
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
      )}
    </div>
  );
}; 