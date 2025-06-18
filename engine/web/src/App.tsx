import { useState } from 'react'
import { ChessGame } from './components/Chessboard'

function App() {
  const [lastMove, setLastMove] = useState<string | null>(null)
  const [gameStatus, setGameStatus] = useState<string | null>(null)
  const [currentFen, setCurrentFen] = useState<string>('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')

  const handleMove = (move: string, fen: string) => {
    setLastMove(move)
    setCurrentFen(fen)
  }

  const handleGameOver = (status: string) => {
    setGameStatus(status)
  }

  return (
    <div className="min-h-screen bg-chess-bg text-gray-100 flex">
      {/* Main content */}
      <div className="flex-1 flex flex-col h-screen">
        {/* Top bar */}
        <div className="h-10 bg-chess-darker flex items-center px-4">
          <h1 className="text-lg font-semibold">RivalAI Chess</h1>
        </div>

        {/* Main game area */}
        <div className="flex-1 flex overflow-hidden">
          {/* Center container */}
          <div className="flex-1 flex justify-center">
            {/* Game section - fixed width */}
            <div className="w-[640px] flex flex-col">
              {/* Player info - top */}
              <div className="h-12 bg-chess-dark flex items-center px-3">
                <div className="flex items-center space-x-2">
                  <div className="w-8 h-8 bg-chess-lighter rounded-full flex items-center justify-center">
                    <span className="text-lg">⚫</span>
                  </div>
                  <div>
                    <div className="text-sm font-medium">Black</div>
                    <div className="text-xs text-gray-400">RivalAI Engine</div>
                  </div>
                </div>
              </div>

              {/* Chessboard container */}
              <div className="flex-1 flex items-center justify-center bg-chess-bg">
                <div className="flex-shrink-0">
                  <ChessGame
                    onMove={handleMove}
                    onGameOver={handleGameOver}
                  />
                </div>
              </div>

              {/* Player info - bottom */}
              <div className="h-12 bg-chess-dark flex items-center px-3">
                <div className="flex items-center space-x-2">
                  <div className="w-8 h-8 bg-chess-lighter rounded-full flex items-center justify-center">
                    <span className="text-lg">⚪</span>
                  </div>
                  <div>
                    <div className="text-sm font-medium">White</div>
                    <div className="text-xs text-gray-400">You</div>
                  </div>
                </div>
              </div>

              {/* Controls */}
              <div className="h-12 bg-chess-dark flex items-center justify-center space-x-1 px-4">
                <button className="p-1.5 rounded bg-chess-darker hover:bg-chess-lighter transition-colors">
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 19l-7-7 7-7m8 14l-7-7 7-7" />
                  </svg>
                </button>
                <button className="p-1.5 rounded bg-chess-darker hover:bg-chess-lighter transition-colors">
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                  </svg>
                </button>
                <button className="p-1.5 rounded bg-chess-darker hover:bg-chess-lighter transition-colors">
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </svg>
                </button>
                <button className="p-1.5 rounded bg-chess-darker hover:bg-chess-lighter transition-colors">
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 5l7 7-7 7M5 5l7 7-7 7" />
                  </svg>
                </button>
              </div>

              {/* FEN Display */}
              <div className="bg-chess-dark px-3 py-2 text-xs font-mono">
                <div className="flex items-center space-x-2">
                  <span className="text-gray-400">FEN:</span>
                  <span className="text-gray-300">{currentFen}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Analysis sidebar */}
          <div className="w-80 bg-chess-dark border-l border-chess-border overflow-y-auto">
            <div className="p-4">
              <h2 className="text-base font-medium mb-4">Analysis</h2>
              
              {/* Engine evaluation */}
              <div className="mb-4">
                <div className="text-xs text-gray-400 mb-1">Engine Evaluation</div>
                <div className="font-mono text-base">+0.3</div>
              </div>

              {/* Move history */}
              <div>
                <div className="text-xs text-gray-400 mb-1">Move History</div>
                <div className="bg-chess-darker rounded p-2">
                  {lastMove ? (
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div className="text-gray-400">1.</div>
                      <div>{lastMove}</div>
                    </div>
                  ) : (
                    <div className="text-gray-500 text-xs">No moves yet</div>
                  )}
                </div>
              </div>

              {/* Game status */}
              {gameStatus && (
                <div className="mt-4 p-2 bg-chess-darker rounded">
                  <div className="text-sm font-medium">
                    {gameStatus}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
