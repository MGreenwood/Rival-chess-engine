import React, { useState, useEffect } from 'react';
import { ChessGame } from '../components/Chessboard';
import { GameControls } from '../components/GameControls';
import ModelStats from '../components/ModelStats';
import useStore from '../store/store';

const TrainModel: React.FC = () => {
  const { preferences, gameActions, currentGame, uiActions } = useStore();
  // Removed loading state since Chessboard component handles move loading
  const [gameOverLogged, setGameOverLogged] = useState(false);
  const [mobileTab, setMobileTab] = useState<'stats' | 'about'>('stats');

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
    <>
    <div className="flex bg-white dark:bg-gray-900 transition-colors">
      {/* Left sidebar - Model Stats */}
      <div className="w-80 p-4 hidden xl:block overflow-y-auto">
        <ModelStats />
      </div>

      {/* Main game area */}
      <div className="flex flex-col items-center justify-start p-3 sm:p-4 flex-1">
        <div className="w-full max-w-[580px]">
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg border-2 border-gray-200 dark:border-gray-600 p-2 mb-2">
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
          <div className="mb-2">
            <GameControls />
          </div>
        </div>
      </div>

      {/* Right sidebar - About PAG Training */}
      <div className="w-[650px] p-4 hidden xl:block overflow-y-auto border-l border-gray-200 dark:border-gray-700">
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg border border-gray-200 dark:border-gray-600 p-4">
          <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-3 flex items-center">
            <span className="w-2 h-2 bg-green-500 rounded-full mr-2 animate-pulse"></span>
            Live AI Training
          </h3>
          
          <div className="space-y-4 text-sm text-gray-700 dark:text-gray-300">
            <div>
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">🧠 Direct Model Output</h4>
              <p className="leading-relaxed">
                This mode uses raw neural network predictions in real-time. Every move you see comes directly 
                from the AI model's policy and value networks, with no search algorithms or external guidance.
              </p>
            </div>

            <div>
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">🕸️ PAG Technology</h4>
              <p className="leading-relaxed mb-2">
                Our <strong>Positional Adjacency Graph (PAG)</strong> system revolutionizes chess AI training:
              </p>
              <ul className="list-disc list-inside space-y-1 text-xs">
                <li><strong>~340,000 features</strong> per position (vs. 64 squares traditional)</li>
                <li><strong>Ultra-dense analysis</strong> of piece relationships and tactical patterns</li>
                <li><strong>Vulnerability detection</strong> prevents hanging pieces</li>
                <li><strong>Motif recognition</strong> for pins, forks, and skewers</li>
              </ul>
            </div>

            <div>
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">⚔️ Tactical Learning</h4>
              <p className="leading-relaxed">
                Our PAG Tactical Loss system weights training heavily toward sound chess:
              </p>
              <ul className="list-disc list-inside space-y-1 text-xs">
                <li><strong>8x penalty</strong> for hanging pieces</li>
                <li><strong>80% tactical focus</strong> vs. 20% positional</li>
                <li><strong>Progressive difficulty</strong> increases over training epochs</li>
              </ul>
            </div>

            <div>
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">🔬 Beyond FEN Data</h4>
              <p className="leading-relaxed">
                Traditional engines parse basic board positions (FEN strings). Our PAG system extracts:
              </p>
              <ul className="list-disc list-inside space-y-1 text-xs">
                <li><strong>Tactical features:</strong> 76 dimensions per piece</li>
                <li><strong>Positional features:</strong> 80 dimensions per piece</li>
                <li><strong>Strategic features:</strong> 60 dimensions per piece</li>
                <li><strong>Edge relationships:</strong> 158 dimensions between pieces</li>
              </ul>
            </div>

            <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-3">
              <h4 className="font-semibold text-blue-900 dark:text-blue-300 mb-1">💡 Training Impact</h4>
              <p className="text-xs text-blue-800 dark:text-blue-400 leading-relaxed">
                This advanced system enables the AI to learn tactical chess fundamentals in just 5-10 epochs, 
                compared to hundreds of epochs with traditional FEN-based training.
              </p>
            </div>

            <div className="text-xs text-gray-500 dark:text-gray-400 pt-2 border-t border-gray-200 dark:border-gray-700">
              <p>Every game you play contributes to the model's tactical understanding through our enhanced training pipeline.</p>
            </div>
          </div>
        </div>
      </div>

    </div>
    
    {/* Mobile view for stats and about */}
    <div className="xl:hidden bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg transition-colors m-2">
      <div className="p-2">
        {/* Mobile tabs */}
        <div className="flex border-b border-gray-200 dark:border-gray-700 mb-2">
          <button 
            onClick={() => setMobileTab('stats')}
            className={`flex-1 py-2 px-3 text-sm font-medium hover:text-gray-900 dark:hover:text-white ${
              mobileTab === 'stats' 
                ? 'border-b-2 border-blue-500 text-blue-600 dark:text-blue-400' 
                : 'text-gray-600 dark:text-gray-300'
            }`}
          >
            Model Stats
          </button>
          <button 
            onClick={() => setMobileTab('about')}
            className={`flex-1 py-2 px-3 text-sm font-medium hover:text-gray-900 dark:hover:text-white ${
              mobileTab === 'about' 
                ? 'border-b-2 border-blue-500 text-blue-600 dark:text-blue-400' 
                : 'text-gray-600 dark:text-gray-300'
            }`}
          >
            About PAG
          </button>
        </div>
        
        {/* Mobile content */}
        <div className="overflow-y-auto max-h-48">
          {mobileTab === 'stats' ? (
            <ModelStats />
          ) : (
            <div className="space-y-3 text-xs text-gray-700 dark:text-gray-300">
              <div>
                <h4 className="font-semibold text-gray-900 dark:text-white mb-1">🧠 Direct AI Output</h4>
                <p className="leading-relaxed">
                  Raw neural network predictions with no search algorithms.
                </p>
              </div>

              <div>
                <h4 className="font-semibold text-gray-900 dark:text-white mb-1">🕸️ PAG Technology</h4>
                <p className="leading-relaxed">
                  <strong>~340,000 features</strong> per position vs. traditional 64 squares.
                  Ultra-dense analysis prevents hanging pieces and recognizes tactical motifs.
                </p>
              </div>

              <div>
                <h4 className="font-semibold text-gray-900 dark:text-white mb-1">⚔️ Tactical Learning</h4>
                <ul className="list-disc list-inside space-y-1">
                  <li><strong>8x penalty</strong> for hanging pieces</li>
                  <li><strong>80% tactical focus</strong> vs. 20% positional</li>
                  <li><strong>Progressive difficulty</strong> scaling</li>
                </ul>
              </div>

              <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded p-2">
                <h4 className="font-semibold text-blue-900 dark:text-blue-300 mb-1">💡 Impact</h4>
                <p className="text-blue-800 dark:text-blue-400 leading-relaxed">
                  Learns tactical fundamentals in 5-10 epochs vs. hundreds with traditional training.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
    </>
  );
};

export default TrainModel; 