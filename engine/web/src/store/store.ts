import { create } from 'zustand';
import type { Theme, UserPreferences, ModelStats, LeaderboardEntry, ConnectionStatus, AnalysisResult, TrainingMetrics } from './types';
import axios from 'axios';
import type { GameState } from '../types/chess';

// Dynamic API base URL - inline to avoid import issues
const getApiBaseUrl = (): string => {
  const host = window.location.host;
  const protocol = window.location.protocol; // Use current protocol
  
  console.log('🌐 getApiBaseUrl debug:', { host, protocol, location: window.location.href });
  
  // If accessing via localhost, use localhost:3000 with current protocol
  if (host.includes('localhost') || host.includes('127.0.0.1')) {
    const apiUrl = `${protocol}//localhost:3000`;
    console.log('🏠 Using localhost API URL:', apiUrl);
    return apiUrl;
  }
  
  // For rivalchess.xyz, use the main domain (likely behind a reverse proxy)
  if (host === 'rivalchess.xyz') {
    // The backend is likely running on the same domain through a reverse proxy
    // or on a different port. Try the main domain first.
    const apiUrl = `${protocol}//rivalchess.xyz`;
    console.log('🏆 Using rivalchess.xyz API URL:', apiUrl);
    console.log('💡 If backend is on different port, update this to use :3000 or :8000');
    return apiUrl;
  }
  
  // Otherwise use the current protocol and host (tunnel)
  const apiUrl = `${protocol}//${host}`;
  console.log('🌍 Using tunnel API URL:', apiUrl);
  return apiUrl;
};

// Alternative API URLs to try if the primary fails
const getApiAlternatives = (): string[] => {
  const host = window.location.host;
  const protocol = window.location.protocol;
  
  if (host === 'rivalchess.xyz') {
    return [
      `${protocol}//rivalchess.xyz`,
      `${protocol}//rivalchess.xyz:3000`,
      `${protocol}//rivalchess.xyz:8000`,
      `${protocol}//rivalchess.xyz/api`
    ];
  }
  
  return [`${protocol}//${host}`];
};

const API_BASE_URL = getApiBaseUrl();

// Cache configuration
const CACHE_DURATION = 5 * 60 * 1000; // 5 minutes
const BACKGROUND_REFRESH_INTERVAL = 2 * 60 * 1000; // 2 minutes

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
  animationSpeed: 300,
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
  currentMode: GameMode;
  leaderboard: LeaderboardEntry[];
  connectionStatus?: ConnectionStatus;
  currentAnalysis?: AnalysisResult;
  analysisHistory: AnalysisResult[];
  trainingMetrics?: TrainingMetrics;
  statsCache: {
    single: { stats: ModelStats | null; timestamp: number; recentGames: GameMetadata[] };
    community: { stats: ModelStats | null; timestamp: number; recentGames: GameMetadata[] };
  };
  refreshInterval: number | null;
  gameActions: {
    makeMove: (move: string) => Promise<void>;
    viewPosition: (moves: string) => Promise<void>;
    startNewGame: (mode: GameMode) => Promise<void>;
    loadGame: (gameId: string, mode: GameMode) => Promise<void>;
    deleteGame: (gameId: string, mode: GameMode) => Promise<void>;
  };
  uiActions: {
    setTheme: (theme: Theme) => void;
    setPreferences: (prefs: Partial<UserPreferences>) => void;
    setCurrentMode: (mode: GameMode) => void;
    loadStatsForMode: (mode: GameMode, forceRefresh?: boolean) => Promise<void>;
    startPeriodicStatsRefresh: () => void;
    stopPeriodicStatsRefresh: () => void;
  };
  init: () => Promise<void>;
}

const useStore = create<StoreState>((set, get) => {
  // Move initPromise inside the store closure
  let initPromise: Promise<void> | null = null;

  return {
    currentGame: null,
    loading: false,
    recentGames: [],
    theme: defaultTheme,
    preferences: defaultPreferences,
    modelStats: null,
    currentMode: 'single',
    leaderboard: [],
    connectionStatus: undefined,
    currentAnalysis: undefined,
    analysisHistory: [],
    trainingMetrics: undefined,
    statsCache: {
      single: { stats: null, timestamp: 0, recentGames: [] },
      community: { stats: null, timestamp: 0, recentGames: [] },
    },
    refreshInterval: null,

    gameActions: {
      makeMove: async (move: string) => {
        const state = get();
        const { currentGame } = state;
        if (!currentGame) {
          throw new Error('No active game');
        }

        // Prevent multiple simultaneous moves
        if (state.loading) {
          throw new Error('Move already in progress');
        }

        // Set loading state to prevent race conditions
        set({ loading: true });

        try {
          // Get fresh state right before making the request
          const freshState = get();
          const freshGame = freshState.currentGame;
          if (!freshGame) {
            throw new Error('No active game');
          }

          console.log('Making move:', move, 'Current board:', freshGame.board);

          const response = await axios.post(`${API_BASE_URL}/move`, {
            move_str: move,
            player_color: 'white',
            game_id: freshGame.metadata?.game_id || freshGame.game_id,
            board: freshGame.board
          });

          if (response.data.success) {
            console.log('Move successful, updating state. New board:', response.data.board);
            const newGameState: GameState = {
              ...freshGame,
              board: response.data.board,
              status: response.data.status,
              move_history: response.data.move_history,
              is_player_turn: response.data.is_player_turn,
              metadata: {
                ...freshGame.metadata!,
                status: response.data.status,
                total_moves: response.data.move_history?.length || 0,
                last_move_at: new Date().toISOString()
              }
            };
            set({ currentGame: newGameState, loading: false });
            
            // If game ended, invalidate cache and refresh stats
            if (response.data.status !== 'active') {
              console.log('Game ended, invalidating cache and refreshing stats...');
              try {
                // Force refresh to get updated stats immediately
                await get().uiActions.loadStatsForMode(get().currentMode, true);
              } catch (refreshError) {
                console.warn('Failed to refresh stats after game end:', refreshError);
              }
            }
          } else if (response.data.error_message) {
            set({ loading: false });
            throw new Error(response.data.error_message);
          }
        } catch (error: any) {
          set({ loading: false });
          // Keep the error throwing but don't log to console
          const errorMessage = error.response?.data?.error_message || error.message || 'Failed to make move';
          throw new Error(errorMessage);
        }
      },

      viewPosition: async (moves: string) => {
        try {
          const currentGame = get().currentGame;
          if (!currentGame) return;

          // Handle empty moves (initial position)
          if (!moves.trim()) {
            set({ 
              currentGame: {
                ...currentGame,
                board: 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
              }
            });
            return;
          }

          // Try to use the backend endpoint first
          try {
            const response = await axios.post(`${API_BASE_URL}/position`, {
              moves: moves,
              game_id: currentGame.metadata?.game_id
            });

            if (response.data && response.data.board) {
              set({
                currentGame: {
                  ...currentGame,
                  board: response.data.board
                }
              });
              return;
            }
          } catch (backendError) {
            // Backend endpoint doesn't exist, fall back to client-side calculation
          }

          // Client-side fallback: reconstruct position from move history
          const moveArray = moves.trim().split(' ').filter(m => m.length > 0);
          
          // For now, if we can't calculate the position, try to find it in the current game
          if (currentGame.move_history) {
            const targetMoveCount = moveArray.length;
            
            // If we're asking for the current position, use the current board
            if (targetMoveCount === currentGame.move_history.length) {
              // This is the current position, don't change the board
              return;
            }
            
            // For historical positions, we'll need to reconstruct or use a default
            // For now, keep the current board position to avoid breaking the UI
            console.log('Position viewing: Client-side position calculation not implemented');
          }
        } catch (error) {
          console.error('Failed to view position:', error);
        }
      },

      startNewGame: async (mode = 'single') => {
        set({ loading: true });
        try {
          console.log('Starting new game...');
          
          const response = await axios.post(`${API_BASE_URL}/${mode === 'community' ? 'community/start' : 'move/new'}`, {
            player_color: 'white',
          });

          if (response.data) {
            console.log('New game response:', response.data);
            
            // Create proper GameState with metadata
            const newGameState: GameState = {
              game_id: response.data.game_id,
              metadata: {
                game_id: response.data.game_id,
                mode: mode,
                created_at: new Date().toISOString(),
                last_move_at: new Date().toISOString(),
                status: response.data.status || 'active',
                total_moves: 0,
                player_color: 'white',
                player_name: null,
                engine_version: '1.0.0'
              },
              board: response.data.board,
              status: response.data.status || 'active',
              move_history: response.data.move_history || [],
              is_player_turn: response.data.is_player_turn !== false // Default to true
            };
            set({ currentGame: newGameState });
            get().init(); // Refresh recent games list
          }
        } catch (error) {
          console.error('Failed to start new game:', error);
          throw error;
        } finally {
          set({ loading: false });
        }
      },

      loadGame: async (gameId: string, mode: GameMode) => {
        set({ loading: true });
        try {
          const response = await axios.get(`${API_BASE_URL}/games/${gameId}`);
          const gameData = response.data;
          
          // Transform server response to proper GameState structure
          const gameState: GameState = {
            game_id: gameData.game_id,
            metadata: {
              game_id: gameData.game_id,
              mode: mode,
              created_at: gameData.created_at || new Date().toISOString(),
              last_move_at: gameData.last_move_at || new Date().toISOString(),
              status: gameData.status,
              total_moves: gameData.move_history?.length || 0,
              player_color: gameData.player_color || 'white',
              player_name: gameData.player_name || null,
              engine_version: gameData.engine_version || '1.0.0'
            },
            board: gameData.board,
            status: gameData.status,
            move_history: gameData.move_history || [],
            is_player_turn: gameData.is_player_turn
          };
          
          set({ currentGame: gameState });
        } catch (error) {
          console.error('Failed to load game:', error);
        } finally {
          set({ loading: false });
        }
      },

      deleteGame: async (gameId: string, mode: GameMode) => {
        try {
          await axios.delete(`${API_BASE_URL}/games/${mode}/${gameId}`);
          // Invalidate cache since game count changed
          const state = get();
          set({
            statsCache: {
              ...state.statsCache,
              [mode]: { ...state.statsCache[mode], timestamp: 0 }
            }
          });
          get().uiActions.loadStatsForMode(mode, true); // Force refresh
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
      },
      setCurrentMode: (mode: GameMode) => {
        const currentMode = get().currentMode;
        if (currentMode === mode) {
          // Already in this mode, use cached data if available
          const cached = get().statsCache[mode];
          const isStale = Date.now() - cached.timestamp > CACHE_DURATION;
          
          if (!isStale && cached.stats) {
            console.log(`📋 Using cached stats for ${mode}`);
            set({
              modelStats: cached.stats,
              recentGames: cached.recentGames
            });
          } else {
            console.log(`🔄 Cache stale for ${mode}, refreshing...`);
            get().uiActions.loadStatsForMode(mode);
          }
          return;
        }
        
        console.log(`🔄 Switching mode from ${currentMode} to ${mode}`);
        
        // Check if we have cached data for the new mode
        const cached = get().statsCache[mode];
        const isStale = Date.now() - cached.timestamp > CACHE_DURATION;
        
        if (!isStale && cached.stats) {
          console.log(`📋 Using cached stats for ${mode}`);
          set({
            currentMode: mode,
            modelStats: cached.stats,
            recentGames: cached.recentGames
          });
        } else {
          console.log(`💾 No valid cache for ${mode}, loading fresh data...`);
          // Clear current stats and games to prevent showing stale data
          set({ 
            currentMode: mode,
            modelStats: null,
            recentGames: []
          });
          // Load fresh data
          get().uiActions.loadStatsForMode(mode);
        }
      },
      loadStatsForMode: async (mode: GameMode, forceRefresh?: boolean) => {
        console.log(`📊 Loading stats for mode: ${mode} (forceRefresh: ${forceRefresh})`);
        
        // Check cache first (unless forcing refresh)
        if (!forceRefresh) {
          const cached = get().statsCache[mode];
          const isStale = Date.now() - cached.timestamp > CACHE_DURATION;
          
          if (!isStale && cached.stats) {
            console.log(`⚡ Using cached stats for ${mode} (age: ${Math.round((Date.now() - cached.timestamp) / 1000)}s)`);
            
            // Only update if we're still in the same mode
            const currentMode = get().currentMode;
            if (currentMode === mode) {
              set({
                modelStats: cached.stats,
                recentGames: cached.recentGames
              });
            }
            return;
          }
        }
        
        try {
          const statsResponse = await axios.get(`${API_BASE_URL}/stats`);
          const statsKey = mode === 'community' ? 'community_model' : 'single_player_model';
          
          console.log(`📈 Fresh stats response for ${mode}:`, {
            hasData: !!statsResponse.data[statsKey],
            totalGames: statsResponse.data[statsKey]?.total_games || 0,
            statsKey
          });
          
          if (statsResponse.data[statsKey]) {
            const stats = statsResponse.data[statsKey];
            
            // Only update if we're still in the same mode (prevent race conditions)
            const currentMode = get().currentMode;
            if (currentMode !== mode) {
              console.log(`⚠️ Mode changed during stats loading, ignoring results for ${mode}`);
              return;
            }
            
            // If we get 0 total games, try refreshing stats once
            if (stats.total_games === 0) {
              console.log('Got 0 total games, attempting stats refresh...');
              try {
                const refreshResponse = await axios.post(`${API_BASE_URL}/stats/refresh`);
                if (refreshResponse.data[statsKey] && refreshResponse.data[statsKey].total_games > 0) {
                  console.log('Stats refresh successful!');
                  const refreshedStats = refreshResponse.data[statsKey];
                  
                  // Cache the refreshed stats
                  const state = get();
                  set({
                    modelStats: refreshedStats,
                    statsCache: {
                      ...state.statsCache,
                      [mode]: { stats: refreshedStats, timestamp: Date.now(), recentGames: state.recentGames }
                    }
                  });
                  
                  // Also refresh recent games when stats are refreshed
                  try {
                    console.log('🔄 Refreshing recent games from:', `${API_BASE_URL}/recent-games`);
                    const recentGamesResponse = await axios.get(`${API_BASE_URL}/recent-games`);
                    console.log('📥 Refresh recent games response:', recentGamesResponse.data?.length || 0, 'games');
                    const allGames = Array.isArray(recentGamesResponse.data) ? recentGamesResponse.data : [];
                    
                    // Update cache with new games
                    const finalState = get();
                    set({ 
                      recentGames: allGames,
                      statsCache: {
                        ...finalState.statsCache,
                        [mode]: { stats: refreshedStats, timestamp: Date.now(), recentGames: allGames }
                      }
                    });
                  } catch (recentGamesError) {
                    console.error('❌ Failed to refresh recent games:', recentGamesError);
                  }
                  return;
                }
              } catch (refreshError) {
                console.warn('Stats refresh failed:', refreshError);
              }
            }
            
            console.log(`✅ Setting stats for ${mode}:`, stats);
            set({ modelStats: stats });
          } else {
            console.log(`❌ No stats found for ${mode}, using defaults`);
            const defaultStats = {
              wins: 0,
              losses: 0,
              draws: 0,
              total_games: 0,
              win_rate: 0.0
            };
            set({ modelStats: defaultStats });
          }
          
          // Always refresh recent games when loading stats for a mode
          try {
            console.log('🔄 Loading recent games from:', `${API_BASE_URL}/recent-games`);
            const recentGamesResponse = await axios.get(`${API_BASE_URL}/recent-games`);
            
            // Check if we're still in the same mode (prevent race conditions)
            const currentMode = get().currentMode;
            if (currentMode !== mode) {
              console.log(`⚠️ Mode changed during recent games loading, ignoring results for ${mode}`);
              return;
            }
            
            console.log('📥 Recent games response:', {
              status: recentGamesResponse.status,
              dataType: typeof recentGamesResponse.data,
              isArray: Array.isArray(recentGamesResponse.data),
              length: recentGamesResponse.data?.length,
              mode: mode
            });
            
            const allGames = Array.isArray(recentGamesResponse.data) ? recentGamesResponse.data : [];
            const currentStats = get().modelStats;
            
            // Update cache with fresh data
            const state = get();
            set({ 
              recentGames: allGames,
              statsCache: {
                ...state.statsCache,
                [mode]: { 
                  stats: currentStats, 
                  timestamp: Date.now(), 
                  recentGames: allGames 
                }
              }
            });
            
            console.log(`✅ Recent games cached for ${mode}:`, {
              totalGames: allGames.length,
              gamesForMode: allGames.filter(g => g.mode === mode).length,
              cacheTimestamp: Date.now()
            });
          } catch (recentGamesError) {
            console.error('❌ Failed to load recent games:', recentGamesError);
            console.error('API URL was:', `${API_BASE_URL}/recent-games`);
            set({ recentGames: [] });
          }
        } catch (error) {
          console.error('Failed to load stats for mode:', mode, error);
          set({
            modelStats: {
              wins: 0,
              losses: 0,
              draws: 0,
              total_games: 0,
              win_rate: 0.0
            }
          });
        }
      },
      startPeriodicStatsRefresh: () => {
        // Clear any existing interval
        const currentInterval = get().refreshInterval;
        if (currentInterval) {
          clearInterval(currentInterval);
        }
        
        console.log('🔄 Starting periodic stats refresh...');
        const intervalId = setInterval(() => {
          const currentMode = get().currentMode;
          console.log(`⏰ Background refresh for ${currentMode}`);
          get().uiActions.loadStatsForMode(currentMode, true);
        }, BACKGROUND_REFRESH_INTERVAL);
        
        set({ refreshInterval: intervalId });
      },
      stopPeriodicStatsRefresh: () => {
        const currentInterval = get().refreshInterval;
        if (currentInterval) {
          console.log('⏹️ Stopping periodic stats refresh');
          clearInterval(currentInterval);
          set({ refreshInterval: null });
        }
      }
    },

    init: async () => {
      // Prevent multiple simultaneous init calls
      if (initPromise) {
        return initPromise;
      }

      initPromise = (async () => {
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

          // Load model stats for the current mode
          await get().uiActions.loadStatsForMode(get().currentMode);

          // Start periodic refresh
          get().uiActions.startPeriodicStatsRefresh();

          // Load recent games
          try {
            console.log('🏁 Init: Loading recent games from:', `${API_BASE_URL}/recent-games`);
            const response = await axios.get(`${API_BASE_URL}/recent-games`);
            console.log('📥 Init recent games response:', {
              type: typeof response.data,
              isArray: Array.isArray(response.data),
              length: response.data?.length,
              keys: response.data ? Object.keys(response.data) : 'null',
              data: response.data
            });
            
            // Check if we got HTML instead of JSON (wrong endpoint)
            if (typeof response.data === 'string' && response.data.includes('<!doctype html>')) {
              console.error('❌ Got HTML instead of JSON - wrong API endpoint!');
              console.error('Expected JSON from:', `${API_BASE_URL}/recent-games`);
              console.error('But got HTML, which means:');
              console.error('  1. The backend API server is not running');
              console.error('  2. The API URL is incorrect');
              console.error('  3. Routing is not set up properly');
              console.error('');
              console.error('🔧 To fix this:');
              console.error('  - Ensure the backend server is running on the expected URL');
              console.error('  - Or update the API URL in getApiBaseUrl()');
              console.error('  - Current API_BASE_URL:', API_BASE_URL);
              set({ recentGames: [] });
              return;
            }
            
            const games = Array.isArray(response.data) ? response.data : [];
            set({ recentGames: games });

            // Load current game if there's an active one
            const activeGame = games.find((game: any) => game.status === 'active');
            if (activeGame) {
              try {
                await get().gameActions.loadGame(activeGame.game_id, activeGame.mode);
              } catch (error) {
                // Don't load corrupted games
              }
            }
          } catch (error) {
            console.error('❌ Init: Failed to load recent games:', error);
            set({ recentGames: [] });
          }
        } catch (error) {
          console.error('Store initialization failed:', error);
        } finally {
          initPromise = null;
        }
      })();

      return initPromise;
    },
  };
});

export default useStore; 