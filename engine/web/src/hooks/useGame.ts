import { useState, useEffect, useCallback, useRef } from 'react';
import axios from 'axios';
import type { GameState, GameStatus, MoveRequest, MoveResponse, GameSettings, WebSocketMessage } from '../types/chess';
import useStore from '../store/store';

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

// Dynamic WebSocket URL based on current host
const getWebSocketURL = () => {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const host = window.location.host;
  
  // If accessing via localhost, use localhost:3000 with appropriate WebSocket protocol
  if (host.includes('localhost') || host.includes('127.0.0.1')) {
    return `${protocol}//localhost:3000/ws`;
  }
  
  // Otherwise use the current host with /ws path (tunnel)
  return `${protocol}//${host}/ws`;
};

export function useGame(initialSettings?: GameSettings) {
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number>();

  // Add logging to setGameState
  const setGameStateWithLog = (updater: ((prev: GameState | null) => GameState | null) | GameState | null) => {
    setGameState((prev: GameState | null) => {
      const next = typeof updater === 'function' ? (updater as (prev: GameState | null) => GameState | null)(prev) : updater;
      console.log('[setGameState] prev:', prev, 'next:', next);
      return next;
    });
  };

  // Initialize WebSocket connection
  const connectWebSocket = useCallback((gameId: string) => {
    console.log('[connectWebSocket] Called for gameId:', gameId);
    
    // Update connection status to connecting
    useStore.setState({ connectionStatus: 'connecting' });
    
    // Only close existing connection if it's for a different game
    if (wsRef.current?.readyState === WebSocket.OPEN) {
        const currentGameId = wsRef.current.url.split('/').pop();
        if (currentGameId === gameId) {
            console.log('[connectWebSocket] WebSocket already connected for this game:', gameId);
            return;
        }
        console.log('[connectWebSocket] Closing existing WebSocket for different game:', currentGameId, '->', gameId);
        wsRef.current.close(1000, 'Switching to new game');
    }

    const wsUrl = `${getWebSocketURL()}/${gameId}`;
    console.log('[connectWebSocket] Connecting to WebSocket URL:', wsUrl);
    const socket = new WebSocket(wsUrl);

    socket.onopen = () => {
        console.log('[WebSocket] open for gameId:', gameId);
        setError(null);
        // Update connection status to connected
        useStore.setState({ connectionStatus: 'connected' });
        // Request initial game state
        socket.send(JSON.stringify({ command: 'refresh' }));
    };

    socket.onmessage = (event) => {
        try {
            console.log('[WebSocket] message received for gameId:', gameId, event.data);
            const message = JSON.parse(event.data) as WebSocketMessage;
            if (message.type === 'status' && message.payload.board) {
                // Only update state if this is the current game
                setGameStateWithLog(prev => {
                    if (!prev || prev.game_id !== gameId) {
                        console.log('[WebSocket] Ignoring state update for different game. prev:', prev?.game_id, 'msg:', gameId);
                        return prev;
                    }
                    // Check if the game is over due to no legal moves
                    const isGameOver = message.payload.status === 'checkmate' || 
                                     message.payload.status === 'stalemate' ||
                                     message.payload.status === 'draw' ||
                                     message.payload.error?.includes('No legal moves');
                    const newState: GameState = {
                        ...prev,
                        board: message.payload.board!,
                        status: isGameOver ? (message.payload.status || 'checkmate') : (message.payload.status || 'active'),
                        is_player_turn: true,
                    };
                    console.log('[WebSocket] Updated game state:', newState);
                    return newState;
                });
            } else if (message.type === 'error') {
                console.error('[WebSocket] error message:', message.payload.error);
                // Check if the error indicates a game-ending state
                if (message.payload.error?.includes('No legal moves')) {
                    setGameStateWithLog(prev => prev ? {
                        ...prev,
                        status: 'checkmate',
                        is_player_turn: true,
                    } : prev);
                }
                setError(message.payload.error || 'An error occurred');
            }
        } catch (err) {
            console.error('[WebSocket] Error parsing message:', err, event.data);
        }
    };

    socket.onerror = (error) => {
        console.error('[WebSocket] error for gameId:', gameId, error);
        setError('WebSocket connection error');
        useStore.setState({ connectionStatus: 'disconnected' });
    };

    socket.onclose = (event) => {
        console.log('[WebSocket] closed for gameId:', gameId, 'code:', event.code, 'reason:', event.reason);
        useStore.setState({ connectionStatus: 'disconnected' });
        
        if (reconnectTimeoutRef.current) {
          window.clearTimeout(reconnectTimeoutRef.current);
          reconnectTimeoutRef.current = undefined;
        }
        // Only attempt to reconnect if:
        // 1. The connection was closed abnormally (1006)
        // 2. We're still on the same game
        // 3. The game is still active
        if (event.code === 1006 && 
            gameState?.game_id === gameId && 
            gameState?.status === 'active') {
            useStore.setState({ connectionStatus: 'connecting' });
            reconnectTimeoutRef.current = window.setTimeout(() => {
                console.log('[WebSocket] Attempting reconnection after abnormal close for gameId:', gameId);
                connectWebSocket(gameId);
            }, 3000);
        }
    };

    wsRef.current = socket;
}, [gameState?.game_id]);

// Connect WebSocket when gameId changes
useEffect(() => {
    if (gameState?.game_id) {
        connectWebSocket(gameState.game_id);
    }
    return () => {
        if (wsRef.current) {
            wsRef.current.close(1000, 'Component unmounting');
        }
        if (reconnectTimeoutRef.current) {
            window.clearTimeout(reconnectTimeoutRef.current);
        }
    };
}, [gameState?.game_id, connectWebSocket]);

// Start a new game
const startNewGame = useCallback(async () => {
  try {
    setLoading(true);
    setError(null);
    console.log('Starting new game...');
    // Clear any reconnect timeout before starting a new game
    if (reconnectTimeoutRef.current) {
      window.clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = undefined;
    }
    // Close any existing WebSocket before starting a new game
    if (wsRef.current) {
      wsRef.current.close(1000, 'Starting new game');
      wsRef.current = null;
    }
    const response = await axios.post<GameState>(
      `${API_BASE_URL}/move/new`,
      { player_color: 'white' },
      { headers: { 'Content-Type': 'application/json' } }
    );
    console.log('New game started:', response.data);
    setGameStateWithLog(response.data);
  } catch (err) {
    console.error('Error starting new game:', err);
    setError('Failed to start new game');
  } finally {
    setLoading(false);
  }
}, [initialSettings]);

// Make a move
const makeMove = useCallback(async (move: string) => {
  console.log('makeMove: received move string:', move);
  console.log('Attempting move:', move, 'Game state:', gameState);
  if (!gameState?.game_id || !gameState.is_player_turn) {
    console.log('Cannot make move:', !gameState?.game_id ? 'No game ID' : 'Not player turn');
    throw new Error('Invalid move: not your turn');
  }

  let prevState = gameState;
  try {
    setLoading(true);
    setError(null);
    
    // Send move to backend - server expects game_id in the request body
    const response = await axios.post<MoveResponse>(
      `${API_BASE_URL}/move`,
      { 
        move_str: move,
        game_id: gameState.game_id,
        board: gameState.board,
        player_color: 'white'
      } as MoveRequest
    );
    console.log('Move response:', response.data);

    if (response.data.success) {
      // Update the board with the response
      setGameStateWithLog(prev => {
        const newState = prev ? {
          ...prev,
          board: response.data.board,
          status: response.data.status as GameStatus,
          is_player_turn: response.data.is_player_turn,
          move_history: response.data.move_history || prev.move_history,
        } : null;
        return newState;
      });
    } else {
      throw new Error(response.data.error_message || 'Invalid move');
    }
  } catch (err) {
    console.error('Move error:', err);
    setError('Move failed. Please try again.');
    // Revert to previous state if possible
    setGameStateWithLog(prevState);
    throw err;
  } finally {
    setLoading(false);
  }
}, [gameState]);

// Get current game state
const refreshGameState = useCallback(async () => {
  if (!gameState?.game_id) return;

  try {
    setLoading(true);
    setError(null);
    const response = await axios.get<GameState>(
      `${API_BASE_URL}/games/${gameState.game_id}`
    );
    setGameStateWithLog(response.data);
  } catch (err) {
    setError('Failed to refresh game state');
    console.error('Error refreshing game state:', err);
  } finally {
    setLoading(false);
  }
}, [gameState?.game_id]);

return {
  gameState,
  error,
  loading,
  startNewGame,
  makeMove,
  refreshGameState,
};
} 