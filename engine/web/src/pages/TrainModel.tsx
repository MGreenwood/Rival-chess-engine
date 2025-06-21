import React, { useState, useEffect } from 'react';
import { ChessGame } from '../components/Chessboard';
import { GameControls } from '../components/GameControls';
import { MoveHistory } from '../components/MoveHistory';
import ModelStats from '../components/ModelStats';
import useStore from '../store/store';

const TrainModel: React.FC = () => {
  const { preferences, gameActions, currentGame, uiActions } = useStore();
  const [currentMoveIndex, setCurrentMoveIndex] = useState(-1);
  const [loading, setLoading] = useState(false);
  const [gameOverLogged, setGameOverLogged] = useState(false);

  // Initialize store only once
  useEffect(() => {
    useStore.getState().init();
  }, []); // Empty dependency array - only run once

  // Set mode to single player when this page loads
  useEffect(() => {
    uiActions.setCurrentMode('single');
  }, [uiActions]);

  // Update currentMoveIndex when a new game starts
  useEffect(() => {
    if (!currentGame) {
      setCurrentMoveIndex(-1);
      setGameOverLogged(false);
    } else if (currentGame.status === 'active') {
      // Reset the game over logged flag when game becomes active
      setGameOverLogged(false);
    }
  }, [currentGame]);

  const handleMove = (move: string) => {
    if (currentGame?.status && currentGame.status !== 'active') {
      return;
    }
    setLoading(true);
    gameActions.makeMove(move).then(() => {
      if (currentGame?.move_history) {
        setCurrentMoveIndex(currentGame.move_history.length - 1);
      }
      setLoading(false);
    }).catch(() => {
      setLoading(false);
    });
  };

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

  // Handle move navigation
  const handleMoveSelect = (index: number) => {
    if (!currentGame?.move_history) return;
    
    // Prevent navigation if the game is not in a viewable state
    if (!currentGame.metadata?.game_id) return;
    
    if (index >= -1 && index < currentGame.move_history.length) {
      setCurrentMoveIndex(index);
      
      if (index === -1) {
        // View starting position
        gameActions.viewPosition('');
      } else {
        // View position after the selected move
        const movesToPlay = currentGame.move_history.slice(0, index + 1);
        gameActions.viewPosition(movesToPlay.join(' '));
      }
    }
  };

  const isViewOnly = () => {
    if (!currentGame?.move_history) return false;
    
    // We're in view-only mode if we're not at the end of the game
    // or if we're specifically viewing a historical position
    return currentMoveIndex !== -1 && currentMoveIndex !== currentGame.move_history.length - 1;
  };

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
                onMove={handleMove}
                onGameOver={handleGameOver}
                showCoordinates={preferences.showCoordinates}
                animationDuration={getAnimationDuration()}
                boardTheme={preferences.boardTheme}
                initialPosition={currentGame?.board}
                viewOnly={isViewOnly()}
              />
              {currentGame && !currentGame.is_player_turn && !loading && !isViewOnly() && currentGame.status === 'active' && (
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

      {/* Right sidebar - Move History */}
      <div className="w-80 p-4 hidden lg:block overflow-y-auto">
        <MoveHistory
          currentMoveIndex={currentMoveIndex}
          onMoveSelect={handleMoveSelect}
        />
      </div>

      {/* Mobile view for sidebars */}
      <div className="fixed bottom-0 left-0 right-0 lg:hidden bg-gray-900">
        <div className="grid grid-cols-2 gap-2 p-2">
          <div className="col-span-1 overflow-y-auto max-h-48">
            <ModelStats />
          </div>
          <div className="col-span-1 overflow-y-auto max-h-48">
            <MoveHistory
              currentMoveIndex={currentMoveIndex}
              onMoveSelect={handleMoveSelect}
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default TrainModel; 