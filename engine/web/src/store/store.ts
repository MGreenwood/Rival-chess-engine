import { create } from 'zustand';
import type { AppState, GameRecord, ModelStats, LeaderboardEntry, UserPreferences, Theme, SavedGameData } from './types';
import axios from 'axios';

const API_BASE_URL = 'http://localhost:3000/api';

// Function to load saved games
const loadSavedGames = async (): Promise<SavedGameData[]> => {
  try {
    const response = await axios.get(`${API_BASE_URL}/games`);
    return response.data;
  } catch (error) {
    console.error('Failed to load saved games:', error);
    return [];
  }
};

const defaultTheme = {
  mode: 'dark' as const,
  colors: {
    primary: '#1a1b26',
    secondary: '#13141f',
    background: '#0d0e14',
    text: '#ffffff'
  }
};

const defaultPreferences: UserPreferences = {
  animationSpeed: 'normal',
  showCoordinates: true,
  pieceStyle: 'standard',
  boardTheme: 'classic',
  soundEnabled: true
};

interface GameSettings {
  temperature?: number;
  strength?: number;
}

export const useStore = create<AppState>((set, get) => ({
  // Initial state
  currentGame: null,
  loading: false,
  gameHistory: [],
  modelStats: {
    totalGames: 0,
    createdAt: new Date().toISOString(),
    currentEpoch: 0,
    nextTrainingAt: null,
    winRate: 0,
    ratingHistory: [],
    recentGames: []
  },
  leaderboard: [],
  preferences: defaultPreferences,
  theme: defaultTheme,
  connectionStatus: 'disconnected',

  // Initialize store with saved games
  init: async () => {
    const savedGames = await loadSavedGames();
    if (savedGames.length > 0) {
      // Find the oldest game by comparing timestamps
      const oldestGame = savedGames.reduce((oldest, current) => {
        return new Date(current.timestamp) < new Date(oldest.timestamp) ? current : oldest;
      }, savedGames[0]);

      // Get the 10 most recent games (sort by timestamp descending)
      const sortedGames = [...savedGames].sort((a, b) => 
        new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
      );
      const recentGames = sortedGames.slice(0, 10).map((game: SavedGameData): GameRecord => ({
        id: game.game_id,
        playerUsername: 'Player',
        playerColor: game.player_color as 'white' | 'black',
        moves: game.moves.length,
        result: game.result as 'win' | 'loss' | 'draw',
        timestamp: game.timestamp
      }));

      set(state => ({
        modelStats: {
          ...state.modelStats,
          totalGames: savedGames.length,
          createdAt: oldestGame.timestamp,
          recentGames,
          winRate: calculateWinRate(recentGames)
        }
      }));
    }
  },

  // UI actions
  uiActions: {
    setTheme: (theme: Theme) => set({ theme }),
    setPreferences: (prefs: Partial<UserPreferences>) => set(state => ({
      preferences: {
        ...state.preferences,
        ...prefs
      }
    }))
  },

  // Game actions
  gameActions: {
    startNewGame: async (settings?: GameSettings) => {
      try {
        set({ loading: true });
        // Extract only the needed settings
        const gameSettings = settings ? {
          temperature: settings.temperature,
          strength: settings.strength
        } : undefined;
        
        const response = await axios.post(`${API_BASE_URL}/game`, gameSettings);
        set({ currentGame: response.data, loading: false });
      } catch (error) {
        console.error('\n Failed to start new game:', error);
        set({ loading: false });
        throw error;
      }
    },

    makeMove: async (move: string) => {
      const { currentGame } = get();
      if (!currentGame?.game_id) return;

      // Don't allow moves if the game is over
      if (currentGame.status !== 'active') {
        console.log('Game is already over:', currentGame.status);
        return;
      }

      try {
        set({ loading: true });
        const response = await axios.post(
          `${API_BASE_URL}/game/${currentGame.game_id}/move`,
          { move_str: move }
        );
        
        // If move failed, show error and return
        if (!response.data.success) {
          console.log('Move error:', response.data.error_message);
          set({ loading: false });
          throw new Error(response.data.error_message || 'Invalid move');
        }

        // Update state with successful move
        set(state => ({
          currentGame: {
            ...state.currentGame!,
            board: response.data.board,
            status: response.data.status,
            is_player_turn: response.data.is_player_turn,
            move_history: [...(state.currentGame?.move_history || []), move]
          },
          loading: response.data.status === 'active' // Only keep loading if game is still active
        }));

        // If game is over after player's move, don't get engine move
        if (response.data.status !== 'active') {
          return;
        }

        // Get engine's move
        const engineResponse = await axios.post(
          `${API_BASE_URL}/game/${currentGame.game_id}/engine_move`
        );

        if (engineResponse.data.success) {
          set(state => ({
            currentGame: {
              ...state.currentGame!,
              board: engineResponse.data.board,
              status: engineResponse.data.status,
              is_player_turn: true,
              move_history: engineResponse.data.engine_move 
                ? [...(state.currentGame?.move_history || []), engineResponse.data.engine_move]
                : state.currentGame?.move_history || []
            },
            loading: false
          }));
        } else {
          set({ loading: false });
          throw new Error('Engine failed to make a move');
        }
      } catch (error: any) {
        console.error('Move error:', error);
        set({ loading: false });
        throw error;
      }
    },

    resetGame: async () => {
      const { currentGame } = get();
      if (!currentGame?.game_id) return;

      try {
        set({ loading: true });
        await axios.post(`${API_BASE_URL}/game/${currentGame.game_id}/reset`);
        set({ currentGame: null, loading: false });
      } catch (error) {
        console.error('Failed to reset game:', error);
        set({ loading: false });
        throw error;
      }
    },

    undoMove: async () => {
      // TODO: Implement undo move functionality
      console.warn('Undo move not implemented yet');
    },

    viewPosition: async (moves: string) => {
      try {
        const response = await axios.post(`${API_BASE_URL}/game/view`, { 
          moves: moves.split(' ').filter(Boolean)  // Convert space-separated moves to array
        });
        if (response.data.success) {
          set(state => ({
            currentGame: {
              ...state.currentGame!,
              board: response.data.board,
              is_player_turn: true,  // Allow moves when at latest position
              current_position: moves // Track current position
            }
          }));
        }
      } catch (error) {
        console.error('Failed to view position:', error);
      }
    }
  },

  // Game history actions
  addGame: (game: GameRecord) => set((state) => ({
    gameHistory: [game, ...state.gameHistory].slice(0, 100), // Keep last 100 games
    modelStats: {
      ...state.modelStats,
      totalGames: state.modelStats.totalGames + 1,
      recentGames: [game, ...state.modelStats.recentGames].slice(0, 10),
      winRate: calculateWinRate([game, ...state.modelStats.recentGames]),
      createdAt: state.modelStats.totalGames === 0 ? game.timestamp : state.modelStats.createdAt
    }
  })),

  updateModelStats: (stats: Partial<ModelStats>) => set((state) => ({
    modelStats: { ...state.modelStats, ...stats }
  })),

  updateLeaderboard: (entries: LeaderboardEntry[]) => set(() => ({
    leaderboard: entries
  }))
}));

function calculateWinRate(games: GameRecord[]): number {
  if (games.length === 0) return 0;
  const wins = games.filter(g => g.result === 'win').length;
  return (wins / games.length) * 100;
}

export default useStore; 