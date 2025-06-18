import type { GameStatus } from '../types/chess';

export interface AnalysisResult {
  fen: string;
  evaluation: number;
  bestMoves: string[];
  depth: number;
  pv: string[];
}

export interface TrainingMetrics {
  loss: number;
  accuracy: number;
  epoch: number;
  totalEpochs: number;
  learningRate: number;
  timestamp: number;
}

export interface ModelInfo {
  id: string;
  name: string;
  version: string;
  createdAt: string;
  metrics: {
    elo: number;
    winRate: number;
    drawRate: number;
  };
}

export interface Theme {
  mode: 'light' | 'dark';
  colors: {
    primary: string;
    secondary: string;
    background: string;
    text: string;
  };
}

export interface UserPreferences {
  animationSpeed: 'fast' | 'normal' | 'slow';
  showCoordinates: boolean;
  pieceStyle: string;
  boardTheme: string;
  soundEnabled: boolean;
}

export type ConnectionStatus = 'connected' | 'disconnected' | 'connecting';

export interface GameActions {
  startNewGame: (settings?: GameSettings) => Promise<void>;
  makeMove: (move: string) => Promise<void>;
  resetGame: () => Promise<void>;
  undoMove: () => Promise<void>;
  viewPosition: (moves: string) => Promise<void>;
}

export interface AnalysisActions {
  analyzePosition: (fen: string) => Promise<void>;
  compareEngines: (fen: string) => Promise<void>;
  clearAnalysis: () => void;
}

export interface TrainingActions {
  startTraining: () => Promise<void>;
  stopTraining: () => Promise<void>;
  loadModel: (modelId: string) => Promise<void>;
}

export interface UIActions {
  setTheme: (theme: Theme) => void;
  setPreferences: (prefs: Partial<UserPreferences>) => void;
}

export interface GameSettings {
  temperature?: number;
  strength?: number;
}

export interface GameRecord {
  id: string;
  result: 'win' | 'loss' | 'draw';
  moves: string[];
  timestamp: string;
}

export interface ModelStats {
  wins: number;
  losses: number;
  draws: number;
}

export interface LeaderboardEntry {
  rank: number;
  name: string;
  elo: number;
  wins: number;
  losses: number;
  draws: number;
}

export interface SavedGameData {
  id: string;
  moves: string[];
  result: string;
  timestamp: string;
}

export type GameMode = 'single' | 'community';

export interface GameMetadata {
  game_id: string;
  mode: GameMode;
  created_at: string;
  last_move_at: string;
  status: GameStatus;
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
  is_player_turn: boolean;
  status: GameStatus;
  game_id?: string;  // For backward compatibility with server responses
}

export interface StoreState {
  currentGame: GameState | null;
  loading: boolean;
  recentGames: GameMetadata[];
  theme: Theme;
  preferences: UserPreferences;
  modelStats: ModelStats | null;
  leaderboard: LeaderboardEntry[];
  gameActions: {
    makeMove: (move: string) => Promise<void>;
    startNewGame: (mode?: GameMode) => Promise<void>;
    loadGame: (gameId: string, mode: GameMode) => Promise<void>;
    deleteGame: (gameId: string, mode: GameMode) => Promise<void>;
  };
  uiActions: {
    setTheme: (theme: Theme) => void;
    setPreferences: (prefs: Partial<UserPreferences>) => void;
  };
  init: () => Promise<void>;
} 