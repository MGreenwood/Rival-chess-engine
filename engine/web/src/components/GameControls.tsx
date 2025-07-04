import React from 'react';
import useStore from '../store/store';

export const GameControls: React.FC = () => {
  const { gameActions, currentGame } = useStore();
  const { startNewGame } = gameActions;

  // Function to get a user-friendly game result message
  const getGameResultMessage = (status: string) => {
    switch (status) {
      case 'white_wins':
        return 'Checkmate! You won!';
      case 'black_wins':
        return 'Checkmate! The engine wins!';
      case 'draw_stalemate':
        return 'Game drawn by stalemate';
      case 'draw_insufficient':
        return 'Game drawn by insufficient material';
      case 'draw_repetition':
        return 'Game drawn by repetition';
      case 'draw_fifty_moves':
        return 'Game drawn by fifty-move rule';
      default:
        return null;
    }
  };

  const isGameOver = currentGame?.status && currentGame.status !== 'active';
  const gameResultMessage = currentGame ? getGameResultMessage(currentGame.status) : null;

  const handleNewGame = () => {
    startNewGame('single');
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-4">
      {/* Game Result */}
      {isGameOver && (
        <div className="mb-4 space-y-4">
          <div className="p-4 bg-gray-100 dark:bg-gray-700 rounded-lg">
            <p className="text-center text-lg font-semibold text-gray-900 dark:text-white">
              {gameResultMessage}
            </p>
          </div>
          <div className="flex flex-col sm:flex-row gap-3 justify-center">
            <button
              onClick={handleNewGame}
              className="flex items-center justify-center space-x-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
              <span>New Game</span>
            </button>
            <button
              className="flex items-center justify-center space-x-2 px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-lg transition-colors"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
              </svg>
              <span>Analyze Game</span>
            </button>
          </div>
        </div>
      )}

      {/* Controls for Active Game */}
      {!isGameOver && (
        <div className="flex space-x-2">
          <button
            onClick={handleNewGame}
            className="flex items-center space-x-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
            <span>New Game</span>
          </button>
        </div>
      )}
    </div>
  );
}; 