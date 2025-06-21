import { useEffect, useRef, useCallback } from 'react';
import useStore from '../store/store';
import type { ConnectionStatus, AnalysisResult, TrainingMetrics } from '../store/types';

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

    const wsUrl = `${getWebSocketURL()}/${gameId}`;
    console.log('Connecting to WebSocket URL:', wsUrl);
    const socket = new WebSocket(wsUrl);

    socket.onopen = () => {
      console.log('WebSocket connected for game:', gameId);
      useStore.setState({ connectionStatus: 'connected' as ConnectionStatus });
      // Request initial game state
      socket.send(JSON.stringify({ command: 'refresh' }));
    };

    socket.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        console.log('WebSocket message received:', message);

        if (message.type === 'status' && message.payload.board) {
          useStore.setState((state) => {
            if (!state.currentGame || state.currentGame.game_id !== gameId) {
              console.log('Ignoring state update for different game');
              return {};
            }

            const isGameOver = message.payload.status === 'checkmate' ||
                             message.payload.status === 'stalemate' ||
                             message.payload.status === 'draw';

            return {
              currentGame: {
                ...state.currentGame,
                board: message.payload.board,
                status: isGameOver ? message.payload.status : 'active',
                is_player_turn: true
              }
            };
          });
        } else if (message.type === 'analysis') {
          useStore.setState({ currentAnalysis: message.payload as AnalysisResult });
        } else if (message.type === 'training') {
          useStore.setState({ trainingMetrics: message.payload as TrainingMetrics });
        }
      } catch (err) {
        console.error('Error parsing WebSocket message:', err);
      }
    };

    socket.onerror = (error) => {
      console.error('WebSocket error:', error);
      useStore.setState({ connectionStatus: 'disconnected' as ConnectionStatus });
    };

    socket.onclose = (event) => {
      console.log('WebSocket closed:', event.code, event.reason);
      useStore.setState({ connectionStatus: 'disconnected' as ConnectionStatus });

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
        useStore.setState({ connectionStatus: 'connecting' as ConnectionStatus });
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