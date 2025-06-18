import type { GameState } from '../types/chess';

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
  theme?: string;
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
  playerUsername: string;
  playerColor: 'white' | 'black';
  moves: number;
  result: 'win' | 'loss' | 'draw';
  timestamp: string;  // ISO string timestamp
}

export interface ModelStats {
  totalGames: number;
  createdAt: string;
  currentEpoch: number;
  nextTrainingAt: string | null;
  winRate: number;
  ratingHistory: number[];
  recentGames: GameRecord[];
}

export interface LeaderboardEntry {
  username: string;
  rating: number;
  wins: number;
  losses: number;
  draws: number;
}

export interface SavedGameData {
  game_id: string;
  moves: string[];
  result: string;
  timestamp: string;
  player_color: string;
}

export interface AppState {
  currentGame: {
    game_id: string;
    board: string;
    status: string;
    move_history: string[];
    is_player_turn: boolean;
  } | null;
  loading: boolean;
  gameHistory: GameRecord[];
  modelStats: ModelStats;
  leaderboard: LeaderboardEntry[];
  preferences: UserPreferences;
  theme: Theme;
  connectionStatus: ConnectionStatus;

  // UI actions
  uiActions: {
    setTheme: (theme: Theme) => void;
    setPreferences: (prefs: Partial<UserPreferences>) => void;
  };

  // Game actions
  gameActions: {
    startNewGame: (settings?: GameSettings) => Promise<void>;
    makeMove: (move: string) => Promise<void>;
    resetGame: () => Promise<void>;
    undoMove: () => Promise<void>;
    viewPosition: (moves: string) => Promise<void>;
  };

  // Game history actions
  addGame: (game: GameRecord) => void;
  updateModelStats: (stats: Partial<ModelStats>) => void;
  updateLeaderboard: (entries: LeaderboardEntry[]) => void;

  // Initialization
  init: () => Promise<void>;
} 