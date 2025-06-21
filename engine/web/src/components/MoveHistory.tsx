import React from 'react';
import useStore from '../store/store';

interface MoveHistoryProps {
  currentMoveIndex: number;
  onMoveSelect: (index: number) => void;
}

export const MoveHistory: React.FC<MoveHistoryProps> = ({ currentMoveIndex, onMoveSelect }) => {
  const { currentGame } = useStore();
  const moves = currentGame?.move_history || [];
  
  // Group moves into pairs (white and black)
  const moveRows = moves.reduce<Array<{ number: number; white: string; black?: string }>>((acc, move, index) => {
    if (index % 2 === 0) {
      acc.push({
        number: Math.floor(index / 2) + 1,
        white: move,
        black: moves[index + 1]
      });
    }
    return acc;
  }, []);

  const canGoBack = currentMoveIndex > -1;
  const canGoForward = currentMoveIndex < moves.length - 1;

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
          Move History
        </h3>
        <div className="flex space-x-2">
          <button
            onClick={() => onMoveSelect(-1)} // Go to start
            disabled={!canGoBack}
            className={`p-1 rounded ${
              canGoBack
                ? 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'
                : 'text-gray-400 dark:text-gray-600 cursor-not-allowed'
            }`}
            title="Go to start"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 19l-7-7 7-7m8 14l-7-7 7-7" />
            </svg>
          </button>
          <button
            onClick={() => canGoBack && onMoveSelect(currentMoveIndex - 1)}
            disabled={!canGoBack}
            className={`p-1 rounded ${
              canGoBack
                ? 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'
                : 'text-gray-400 dark:text-gray-600 cursor-not-allowed'
            }`}
            title="Previous move"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
          </button>
          <button
            onClick={() => canGoForward && onMoveSelect(currentMoveIndex + 1)}
            disabled={!canGoForward}
            className={`p-1 rounded ${
              canGoForward
                ? 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'
                : 'text-gray-400 dark:text-gray-600 cursor-not-allowed'
            }`}
            title="Next move"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
          </button>
          <button
            onClick={() => canGoForward && onMoveSelect(moves.length - 1)} // Go to end
            disabled={!canGoForward}
            className={`p-1 rounded ${
              canGoForward
                ? 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'
                : 'text-gray-400 dark:text-gray-600 cursor-not-allowed'
            }`}
            title="Go to end"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 5l7 7-7 7M5 5l7 7-7 7" />
            </svg>
          </button>
        </div>
      </div>

      <div className="space-y-1 font-mono">
        {moveRows.map(({ number, white, black }, rowIndex) => (
          <div 
            key={number}
            className="flex items-center text-sm space-x-2"
          >
            <span className="w-8 text-gray-500 dark:text-gray-400">{number}.</span>
            <button
              onClick={() => onMoveSelect(rowIndex * 2)}
              className={`min-w-[3rem] px-2 py-0.5 rounded text-left ${
                currentMoveIndex === rowIndex * 2
                  ? 'bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200'
                  : 'hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-900 dark:text-gray-100'
              }`}
            >
              {white}
            </button>
            {black && (
              <button
                onClick={() => onMoveSelect(rowIndex * 2 + 1)}
                className={`min-w-[3rem] px-2 py-0.5 rounded text-left ${
                  currentMoveIndex === rowIndex * 2 + 1
                    ? 'bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200'
                    : 'hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-900 dark:text-gray-100'
                }`}
              >
                {black}
              </button>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}; 