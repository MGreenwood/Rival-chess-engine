import React, { useState, useEffect } from 'react'
import { ChessGame } from './components/Chessboard'
import { SettingsPanel } from './components/SettingsPanel'
import { GameControls } from './components/GameControls'
import { MoveHistory } from './components/MoveHistory'
import ModelStats from './components/ModelStats'
import { CommunityGame } from './components/CommunityGame'
import useStore from './store/store'

const App: React.FC = () => {
  const { preferences, gameActions, currentGame } = useStore();
  const [isSettingsOpen, setIsSettingsOpen] = React.useState(false);
  const [currentMoveIndex, setCurrentMoveIndex] = useState(-1);
  const [gameMode, setGameMode] = useState<'single' | 'community'>('single');
  const [loading, setLoading] = useState(false);

  // Initialize store
  useEffect(() => {
    useStore.getState().init();
  }, []);

  // Update currentMoveIndex when a new game starts
  useEffect(() => {
    if (!currentGame) {
      setCurrentMoveIndex(-1);
    }
  }, [currentGame]);

  const handleMove = (move: string) => {
    // Don't allow moves if the game is over
    if (currentGame?.status && currentGame.status !== 'active') {
      console.log('Game is already over:', currentGame.status);
      return;
    }
    setLoading(true);
    gameActions.makeMove(move).then(() => {
      // After a successful move, update the current move index
      if (currentGame?.move_history) {
        setCurrentMoveIndex(currentGame.move_history.length);
      }
      setLoading(false);
    });
  };

  const handleGameOver = (status: string) => {
    console.log('Game over:', status);
    // You could show a modal or notification here
  };

  // Get user-friendly game status message
  const getGameStatusMessage = () => {
    if (!currentGame) return null;
    switch (currentGame.status) {
      case 'white_wins':
        return 'Checkmate! You won!';
      case 'black_wins':
        return 'Checkmate! Engine wins!';
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

  const gameStatusMessage = getGameStatusMessage();

  // Calculate animation duration based on preference
  const getAnimationDuration = (): number => {
    switch (preferences.animationSpeed) {
      case 'fast':
        return 150;
      case 'slow':
        return 450;
      default:
        return 300;
    }
  };

  // Handle move navigation
  const handleMoveSelect = (index: number) => {
    if (!currentGame?.move_history) return;
    
    // Validate index
    if (index >= -1 && index < currentGame.move_history.length) {
      setCurrentMoveIndex(index);
      
      // Get the position after the selected move
      if (index === -1) {
        // Initial position
        gameActions.viewPosition('');
      } else {
        // Get position after the selected move
        const movesToPlay = currentGame.move_history.slice(0, index + 1);
        gameActions.viewPosition(movesToPlay.join(' '));
      }
    }
  };

  // Determine if the board should be in view-only mode
  const isViewOnly = () => {
    if (!currentGame) return false;
    if (!currentGame.move_history) return false;
    
    // Allow moves only when:
    // 1. We're at the latest position (currentMoveIndex matches the last move)
    // 2. Or we're at the initial position with no moves yet
    const isAtLatestMove = currentMoveIndex === currentGame.move_history.length - 1;
    const isInitialPositionWithNoMoves = currentMoveIndex === -1 && currentGame.move_history.length === 0;
    
    return !(isAtLatestMove || isInitialPositionWithNoMoves);
  };

  // Update currentMoveIndex when a new move is made
  useEffect(() => {
    if (currentGame?.move_history) {
      setCurrentMoveIndex(currentGame.move_history.length - 1);
    }
  }, [currentGame?.move_history?.length]);

  return (
    <div className={`min-h-screen bg-gray-900 ${preferences.theme === 'dark' ? 'dark' : ''}`}>
      {/* Title Bar */}
      <div className="fixed top-0 left-0 right-0 h-14 bg-gray-800 border-b border-gray-700 flex items-center px-4 z-10">
        <div className="flex-1 flex items-center">
          <h1 className="text-xl font-bold text-gray-100">RivalAI Chess</h1>
          <div className="ml-4 flex items-center space-x-4">
            <button
              onClick={() => setGameMode('single')}
              className={`px-3 py-1 rounded-full text-sm font-medium transition-colors ${
                gameMode === 'single'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              Single Player
            </button>
            <button
              onClick={() => setGameMode('community')}
              className={`px-3 py-1 rounded-full text-sm font-medium transition-colors ${
                gameMode === 'community'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              Community vs AI
            </button>
          </div>
        </div>
        <div className="flex items-center space-x-4">
          <button
            onClick={() => setIsSettingsOpen(true)}
            className="p-2 text-gray-400 hover:text-gray-100 rounded-lg hover:bg-gray-700 transition-colors"
            title="Settings"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
          </button>
        </div>
      </div>

      <div className="fixed inset-0 pt-14 flex">
        {/* Left sidebar - Model Stats */}
        <div className="w-80 p-4 hidden lg:block overflow-y-auto">
          <ModelStats />
        </div>

        {/* Main game area */}
        <div className="flex-1 flex flex-col items-center justify-start p-4 overflow-y-auto">
          {gameMode === 'single' ? (
            <div className="w-full max-w-[calc(100vh-16rem)]">
              <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-4 mb-4">
                <div className="relative aspect-square">
                  <ChessGame 
                    onMove={handleMove}
                    onGameOver={handleGameOver}
                    showCoordinates={preferences.showCoordinates}
                    animationDuration={getAnimationDuration()}
                    pieceStyle={preferences.pieceStyle}
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
          ) : (
            <CommunityGame />
          )}
        </div>

        {/* Right sidebar - Move History */}
        <div className="w-80 p-4 hidden lg:block overflow-y-auto">
          {gameMode === 'single' ? (
            <MoveHistory
              currentMoveIndex={currentMoveIndex}
              onMoveSelect={handleMoveSelect}
            />
          ) : (
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-4">
              <h2 className="text-xl font-bold mb-3 text-gray-900 dark:text-white">
                How to Play
              </h2>
              <ul className="space-y-2 text-gray-600 dark:text-gray-300">
                <li>• Click squares to vote for a move</li>
                <li>• Each player gets one vote per turn</li>
                <li>• Voting lasts 10 seconds</li>
                <li>• Most voted move is played</li>
                <li>• Ties are broken randomly</li>
              </ul>
            </div>
          )}
        </div>

        {/* Mobile view for sidebars */}
        <div className="fixed bottom-0 left-0 right-0 lg:hidden bg-gray-900">
          <div className="grid grid-cols-2 gap-2 p-2">
            <div className="col-span-1 overflow-y-auto max-h-48">
              <ModelStats />
            </div>
            <div className="col-span-1 overflow-y-auto max-h-48">
              {gameMode === 'single' ? (
                <MoveHistory
                  currentMoveIndex={currentMoveIndex}
                  onMoveSelect={handleMoveSelect}
                />
              ) : (
                <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-4">
                  <h2 className="text-lg font-bold mb-2 text-gray-900 dark:text-white">
                    How to Play
                  </h2>
                  <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-300">
                    <li>• Click squares to vote</li>
                    <li>• 10 second voting</li>
                    <li>• Most votes wins</li>
                  </ul>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      <SettingsPanel
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
      />
    </div>
  )
}

export default App
