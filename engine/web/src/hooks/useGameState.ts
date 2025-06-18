import { useMemo } from 'react';
import { useStore } from '../store/store';
import type { GameState } from '../types/chess';

interface GameStateHook {
  game: GameState | null;
  moveHistory: string[];
  isGameOver: boolean;
  isPlayerTurn: boolean;
  currentPosition: string;
  formattedMoves: Array<{
    moveNumber: number;
    move: string;
  }>;
}

export function useGameState(): GameStateHook {
  const currentGame = useStore(state => state.currentGame);

  const formattedMoves = useMemo(() => {
    if (!currentGame?.move_history?.length) return [];
    
    return currentGame.move_history.map((move, index) => ({
      moveNumber: Math.floor(index / 2) + 1,
      move
    }));
  }, [currentGame?.move_history]);

  return {
    game: currentGame,
    moveHistory: currentGame?.move_history ?? [],
    isGameOver: currentGame?.status !== 'active',
    isPlayerTurn: currentGame?.is_player_turn ?? true,
    currentPosition: currentGame?.board ?? 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
    formattedMoves
  };
} 