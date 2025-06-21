import { useEffect, useRef, useCallback } from 'react';
import { useStore } from '../store/store';

const WS_BASE_URL = 'ws://localhost:3000/ws';

export function useWebSocket(gameId: string | null) {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number>();
  const { currentGame } = useStore();

  const connect = useCallback(() => {
    if (!gameId) return;
    
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      const currentGameId = wsRef.current.url.split('/').pop();
      if (currentGameId === gameId) {
        console.log('WebSocket already connected for this game:', gameId);
        return;
      }
      console.log('Closing existing WebSocket for different game:', currentGameId, '->', gameId);
      wsRef.current.close(1000, 'Switching to new game');
    }

    const wsUrl = `${WS_BASE_URL}/${gameId}`;
    console.log('Connecting to WebSocket URL:', wsUrl);
    const socket = new WebSocket(wsUrl);

    socket.onopen = () => {
      console.log('WebSocket connected for game:', gameId);
      useStore.setState({ connectionStatus: 'connected' });
      // Request initial game state
      socket.send(JSON.stringify({ command: 'refresh' }));
    };

    socket.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        console.log('WebSocket message received:', message);

        if (message.type === 'status' && message.payload.board) {
          useStore.setState(state => {
            if (!state.currentGame || state.currentGame.game_id !== gameId) {
              console.log('Ignoring state update for different game');
              return state;
            }

            const isGameOver = message.payload.status === 'checkmate' ||
                             message.payload.status === 'stalemate' ||
                             message.payload.status === 'draw';

            return {
              ...state,
              currentGame: {
                ...state.currentGame,
                board: message.payload.board,
                status: isGameOver ? message.payload.status : 'active',
                is_player_turn: true
              }
            };
          });
        } else if (message.type === 'analysis') {
          useStore.setState({
            currentAnalysis: message.payload
          });
        } else if (message.type === 'training') {
          useStore.setState({
            trainingMetrics: message.payload
          });
        }
      } catch (err) {
        console.error('Error parsing WebSocket message:', err);
      }
    };

    socket.onerror = (error) => {
      console.error('WebSocket error:', error);
      useStore.setState({ connectionStatus: 'disconnected' });
    };

    socket.onclose = (event) => {
      console.log('WebSocket closed:', event.code, event.reason);
      useStore.setState({ connectionStatus: 'disconnected' });

      // Clear any existing reconnect timeout
      if (reconnectTimeoutRef.current) {
        window.clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = undefined;
      }

      // Attempt to reconnect if:
      // 1. Connection was closed abnormally (1006)
      // 2. We're still on the same game
      // 3. The game is still active
      if (event.code === 1006 &&
          currentGame?.game_id === gameId &&
          currentGame?.status === 'active') {
        useStore.setState({ connectionStatus: 'connecting' });
        reconnectTimeoutRef.current = window.setTimeout(() => {
          console.log('Attempting to reconnect...');
          connect();
        }, 3000);
      }
    };

    wsRef.current = socket;
  }, [gameId, currentGame]);

  // Connect when gameId changes
  useEffect(() => {
    connect();

    return () => {
      if (wsRef.current) {
        wsRef.current.close(1000, 'Component unmounting');
      }
      if (reconnectTimeoutRef.current) {
        window.clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, [gameId, connect]);

  // Return functions to interact with WebSocket
  return {
    sendMessage: (message: any) => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify(message));
      }
    },
    reconnect: connect
  };
} 