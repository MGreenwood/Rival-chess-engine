import { useState } from 'react'
import { ChessGame } from './components/Chessboard'

function App() {
  const [lastMove, setLastMove] = useState<string | null>(null)
  const [gameStatus, setGameStatus] = useState<string | null>(null)

  const handleMove = (move: string) => {
    setLastMove(move)
  }

  const handleGameOver = (status: string) => {
    setGameStatus(status)
  }

  return (
    <div className="min-h-screen bg-gray-100 dark:bg-gray-900 py-6 px-4 flex flex-col items-center">
      <div className="w-full max-w-4xl">
        <div className="bg-white dark:bg-gray-800 shadow-lg rounded-3xl p-6 md:p-10">
          <div className="flex flex-col items-center">
            <ChessGame
              onMove={handleMove}
              onGameOver={handleGameOver}
            />
            
            <div className="w-full mt-8 space-y-4">
              {lastMove && (
                <div className="card">
                  <h3 className="font-semibold mb-2">Last Move</h3>
                  <p className="text-gray-600 dark:text-gray-400">
                    {lastMove}
                  </p>
                </div>
              )}

              {gameStatus && (
                <div className="card">
                  <h3 className="font-semibold mb-2">Game Status</h3>
                  <p className="text-gray-600 dark:text-gray-400">
                    Game Over: {gameStatus}
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>

        <footer className="mt-8 text-center text-gray-600 dark:text-gray-400">
          <p>Built with React, TypeScript, and RivalAI Chess Engine</p>
        </footer>
      </div>
    </div>
  )
}

export default App
