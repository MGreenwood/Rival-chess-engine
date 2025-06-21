import React, { useState, useEffect } from 'react';
import { ChessGame } from '../components/Chessboard';
import { GameControls } from '../components/GameControls';
import ModelStats from '../components/ModelStats';
import useStore from '../store/store';

const TrainModel: React.FC = () => {
  const { preferences, gameActions, currentGame, uiActions } = useStore();
  // Removed loading state since Chessboard component handles move loading
  const [gameOverLogged, setGameOverLogged] = useState(false);

  // Initialize store only once
  useEffect(() => {
    useStore.getState().init();
  }, []); // Empty dependency array - only run once

  // Set mode to single player when this page loads
  useEffect(() => {
    uiActions.setCurrentMode('single');
  }, [uiActions]);

  // Reset game over logged flag when a new game starts
  useEffect(() => {
    if (!currentGame) {
      setGameOverLogged(false);
    } else if (currentGame.status === 'active') {
      // Reset the game over logged flag when game becomes active
      setGameOverLogged(false);
    }
  }, [currentGame]);

  const handleGameOver = (status: string) => {
    // Only log game over once per game
    if (!gameOverLogged && status !== 'active') {
      console.log('Game over:', status);
      setGameOverLogged(true);
    }
  };

  // Calculate animation duration based on preference
  const getAnimationDuration = (): number => {
    // animationSpeed is now a number (milliseconds), so return it directly
    return preferences.animationSpeed;
  };

  // Move history navigation removed

  return (
    <div className="fixed inset-0 pt-14 flex">
      {/* Left sidebar - Model Stats */}
      <div className="w-80 p-4 hidden lg:block overflow-y-auto">
        <ModelStats />
      </div>

      {/* Main game area */}
      <div className="flex-1 flex flex-col items-center justify-start p-4 overflow-y-auto">
        <div className="w-full max-w-[calc(100vh-16rem)]">
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-4 mb-4">
            <div className="relative aspect-square">
              <ChessGame 
                onGameOver={handleGameOver}
                showCoordinates={preferences.showCoordinates}
                animationDuration={getAnimationDuration()}
                boardTheme={preferences.boardTheme}
                initialPosition={currentGame?.board}
                viewOnly={false}
              />
              {currentGame && !currentGame.is_player_turn && currentGame.status === 'active' && (
                <div className="absolute inset-0 bg-chess-darker bg-opacity-50 flex items-center justify-center">
                  <div className="text-white text-sm font-medium px-4 py-2 bg-chess-darker bg-opacity-90 rounded">
                    Engine is thinking...
                  </div>
                </div>
              )}
            </div>
          </div>
          <div className="mb-4">
            <GameControls />
          </div>
        </div>
      </div>

      {/* Mobile view for stats */}
      <div className="fixed bottom-0 left-0 right-0 lg:hidden bg-gray-900">
        <div className="p-2">
          <div className="overflow-y-auto max-h-48">
            <ModelStats />
          </div>
        </div>
      </div>
    </div>
  );
};

export default TrainModel; 