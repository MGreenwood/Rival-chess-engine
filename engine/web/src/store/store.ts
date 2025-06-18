import { create } from 'zustand';
import type { Theme, UserPreferences, GameRecord, ModelStats, LeaderboardEntry, SavedGameData } from './types';
import axios from 'axios';

const API_BASE_URL = 'http://localhost:3000';

export type GameMode = 'single' | 'community';

export interface GameMetadata {
  game_id: string;
  mode: GameMode;
  created_at: string;
  last_move_at: string;
  status: string;
  total_moves: number;
  player_color: string;
  player_name: string | null;
  engine_version: string;
}

export interface GameState {
  metadata: GameMetadata;
  board: string;
  move_history: string[];
  analysis?: Record<string, number>;
}

export interface CommunityGameState {
  metadata: GameMetadata;
  board: string;
  move_history: string[];
  votes: Record<string, string[]>;
  voting_ends_at: string | null;
}

const defaultTheme: Theme = {
  mode: 'dark',
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

interface StoreState {
  currentGame: GameState | null;
  loading: boolean;
  recentGames: GameMetadata[];
  theme: Theme;
  preferences: UserPreferences;
  modelStats: ModelStats | null;
  leaderboard: LeaderboardEntry[];
  gameActions: {
    makeMove: (move: string) => Promise<void>;
    startNewGame: (mode: GameMode) => Promise<void>;
    loadGame: (gameId: string, mode: GameMode) => Promise<void>;
    deleteGame: (gameId: string, mode: GameMode) => Promise<void>;
  };
  uiActions: {
    setTheme: (theme: Theme) => void;
    setPreferences: (prefs: Partial<UserPreferences>) => void;
  };
  init: () => Promise<void>;
}

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

interface GameSettings {
  temperature?: number;
  strength?: number;
}

const useStore = create<StoreState>((set, get) => ({
  currentGame: null,
  loading: false,
  recentGames: [],
  theme: defaultTheme,
  preferences: defaultPreferences,
  modelStats: null,
  leaderboard: [],

  gameActions: {
    makeMove: async (move: string) => {
      const { currentGame } = get();
      if (!currentGame) return;

      try {
        const response = await axios.post(`${API_BASE_URL}/move/${currentGame.metadata.game_id}`, {
          move_str: move,
          board: currentGame.board,
          player_color: currentGame.metadata.player_color,
        });

        if (response.data.success) {
          // Create a new game state from the response
          const newGameState = {
            ...currentGame,
            board: response.data.board,
            move_history: response.data.move_history,
            is_player_turn: response.data.is_player_turn,
            status: response.data.status
          };
          set({ currentGame: newGameState });
          get().init(); // Refresh recent games list
        } else if (response.data.error_message) {
          throw new Error(response.data.error_message);
        }
      } catch (error) {
        console.error('Failed to make move:', error);
        throw error;
      }
    },

    startNewGame: async (mode = 'single') => {
      set({ loading: true });
      try {
        const response = await axios.post(`${API_BASE_URL}/${mode === 'community' ? 'community/start' : 'move/new'}`, {
          player_color: 'white',
        });

        if (response.data) {
          const newGameState: GameState = {
            metadata: {
              game_id: response.data.game_id,
              mode,
              created_at: new Date().toISOString(),
              last_move_at: new Date().toISOString(),
              status: 'active',
              total_moves: 0,
              player_color: 'white',
              player_name: null,
              engine_version: '1.0.0'
            },
            board: response.data.board,
            move_history: response.data.move_history || [],
            is_player_turn: response.data.is_player_turn,
            status: 'active'
          };
          set({ currentGame: newGameState });
          get().init(); // Refresh recent games list
        }
      } catch (error) {
        console.error('Failed to start new game:', error);
      } finally {
        set({ loading: false });
      }
    },

    loadGame: async (gameId: string, mode: GameMode) => {
      set({ loading: true });
      try {
        const response = await axios.get(`${API_BASE_URL}/games/${mode}/${gameId}`);
        set({ currentGame: response.data });
      } catch (error) {
        console.error('Failed to load game:', error);
      } finally {
        set({ loading: false });
      }
    },

    deleteGame: async (gameId: string, mode: GameMode) => {
      try {
        await axios.delete(`${API_BASE_URL}/games/${mode}/${gameId}`);
        get().init(); // Refresh recent games list
      } catch (error) {
        console.error('Failed to delete game:', error);
      }
    },
  },

  uiActions: {
    setTheme: (theme: Theme) => {
      set({ theme });
      localStorage.setItem('chess_theme', JSON.stringify(theme));
    },
    setPreferences: (prefs: Partial<UserPreferences>) => {
      const newPrefs = { ...get().preferences, ...prefs };
      set({ preferences: newPrefs });
      localStorage.setItem('chess_preferences', JSON.stringify(newPrefs));
    }
  },

  init: async () => {
    try {
      // Load theme from localStorage
      const savedTheme = localStorage.getItem('chess_theme');
      if (savedTheme) {
        set({ theme: JSON.parse(savedTheme) });
      }

      // Load preferences from localStorage
      const savedPrefs = localStorage.getItem('chess_preferences');
      if (savedPrefs) {
        set({ preferences: JSON.parse(savedPrefs) });
      }

      // Load model stats
      try {
        const statsResponse = await axios.get(`${API_BASE_URL}/stats`);
        set({ modelStats: statsResponse.data });
      } catch (error) {
        console.error('Failed to load model stats:', error);
        // Set default stats if loading fails
        set({
          modelStats: {
            wins: 0,
            losses: 0,
            draws: 0
          }
        });
      }

      // Load leaderboard
      try {
        const leaderboardResponse = await axios.get(`${API_BASE_URL}/leaderboard`);
        set({ leaderboard: leaderboardResponse.data });
      } catch (error) {
        console.error('Failed to load leaderboard:', error);
        set({ leaderboard: [] });
      }

      // Load recent games
      try {
        const response = await axios.get(`${API_BASE_URL}/games`);
        set({ recentGames: response.data });

        // Load current game if there's an active one
        const activeGame = response.data.find((game: GameMetadata) => game.status === 'active');
        if (activeGame) {
          await get().gameActions.loadGame(activeGame.game_id, activeGame.mode);
        }
      } catch (error) {
        console.error('Failed to load games:', error);
        set({ recentGames: [] });
      }
    } catch (error) {
      console.error('Failed to initialize store:', error);
    }
  },
}));

function calculateWinRate(games: GameRecord[]): number {
  if (games.length === 0) return 0;
  const wins = games.filter(g => g.result === 'win').length;
  return (wins / games.length) * 100;
}

export default useStore; 