export interface GameState {
  game_id: string;
  board: string;  // FEN string
  status: GameStatus;
  move_history: string[];
  is_player_turn: boolean;
  metadata?: {
    game_id: string;
    mode: 'single' | 'community';
    created_at: string;
    last_move_at: string;
    status: string;
    total_moves: number;
    player_color: string;
    player_name: string | null;
    engine_version: string;
  };
}

export type GameStatus = 
  | 'active'
  | 'check'
  | 'checkmate'
  | 'stalemate'
  | 'draw'
  | 'invalid_move';

export interface MoveRequest {
  move_str: string;
  game_id: string;
  board: string;
  player_color: string;
}

export interface MoveResponse {
  success: boolean;
  board: string;
  status: GameStatus;
  engine_move: string | null;
  is_player_turn: boolean;
  error_message: string | null;
  move_history: string[];
}

export interface GameSettings {
  timeControl?: {
    initial: number;  // seconds
    increment: number;  // seconds per move
  };
  engineStrength?: number;  // 1-20
  color?: 'white' | 'black' | 'random';
}

export interface WebSocketMessage {
  type: 'move' | 'status' | 'error';
  payload: {
    move?: string;
    board?: string;
    status?: GameStatus;
    error?: string;
  };
} 