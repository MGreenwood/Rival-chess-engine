import { useState, useEffect, useCallback } from 'react';
import { Chessboard } from 'react-chessboard';
import type { Square, Piece } from 'react-chessboard/dist/chessboard/types';
import { useGame } from '../hooks/useGame';

interface ChessGameProps {
  onMove: (move: string, fen: string) => void;
  onGameOver: (status: string) => void;
}

export function ChessGame({ onMove, onGameOver }: ChessGameProps) {
  const [boardWidth, setBoardWidth] = useState(400);
  const [selectedSquare, setSelectedSquare] = useState<Square | null>(null);
  const [pendingPromotion, setPendingPromotion] = useState<{from: string, to: string} | null>(null);
  const [moveError, setMoveError] = useState<string | null>(null);
  const [lastMoveFrom, setLastMoveFrom] = useState<Square | null>(null);
  const [lastMoveTo, setLastMoveTo] = useState<Square | null>(null);

  const {
    gameState,
    error,
    loading,
    startNewGame,
    makeMove,
  } = useGame();

  // Handle window resize
  useEffect(() => {
    const handleResize = () => {
      // Calculate board size based on viewport height
      const maxHeight = window.innerHeight - 300; // Account for header, footer, and controls
      const maxWidth = window.innerWidth - 400; // Account for sidebar
      const size = Math.min(maxHeight, maxWidth);
      setBoardWidth(Math.max(300, Math.min(600, size))); // Min 300px, max 600px
    };
    
    handleResize(); // Initial size
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Start new game on mount
  useEffect(() => {
    startNewGame();
  }, [startNewGame]);

  // Update parent components
  useEffect(() => {
    if (gameState?.status && gameState.status !== 'active') {
      onGameOver(gameState.status);
    }
  }, [gameState?.status, onGameOver]);

  const onDrop = useCallback((sourceSquare: Square, targetSquare: Square, piece: Piece) => {
    setSelectedSquare(null);
    setMoveError(null);

    // Check if this is a pawn promotion move
    const isPromotion = piece.charAt(1) === 'P' && 
      ((piece.charAt(0) === 'w' && targetSquare.charAt(1) === '8') ||
       (piece.charAt(0) === 'b' && targetSquare.charAt(1) === '1'));

    if (isPromotion) {
      setPendingPromotion({ from: sourceSquare, to: targetSquare });
      return false;
    }

    // Make the move
    const moveString = `${sourceSquare}${targetSquare}`;
    makeMove(moveString)
      .then(() => {
        setLastMoveFrom(sourceSquare);
        setLastMoveTo(targetSquare);
        if (gameState?.board) {
          onMove(moveString, gameState.board);
        }
      })
      .catch((err) => {
        setMoveError(err instanceof Error ? err.message : 'Invalid move');
      });

    // Return true to allow the piece to move visually
    // The actual game state will be updated when the move is confirmed
    return true;
  }, [makeMove, gameState, onMove]);

  const handlePromotion = (promotionPiece: string) => {
    if (!pendingPromotion) return;

    const moveString = `${pendingPromotion.from}${pendingPromotion.to}${promotionPiece}`;
    makeMove(moveString)
      .then(() => {
        setLastMoveFrom(pendingPromotion.from as Square);
        setLastMoveTo(pendingPromotion.to as Square);
        if (gameState?.board) {
          onMove(moveString, gameState.board);
        }
      })
      .catch((err) => {
        setMoveError(err instanceof Error ? err.message : 'Invalid move');
      })
      .finally(() => {
        setPendingPromotion(null);
      });
  };

  const onSquareClick = useCallback((square: Square) => {
    setSelectedSquare(prev => prev === square ? null : square);
  }, []);

  console.log('Rendering chessboard:', { gameState, selectedSquare });
  return (
    <div className="w-full flex flex-col items-center">
      <div className="relative w-full flex justify-center">
        <div style={{ width: boardWidth, maxWidth: '100%' }}>
          <Chessboard
            position={gameState?.board}
            onPieceDrop={onDrop}
            onSquareClick={onSquareClick}
            boardWidth={boardWidth}
            customBoardStyle={{
              borderRadius: '0',
              boxShadow: 'none',
            }}
            customDarkSquareStyle={{ backgroundColor: '#b58863' }}
            customLightSquareStyle={{ backgroundColor: '#f0d9b5' }}
            customSquareStyles={{
              ...(selectedSquare && {
                [selectedSquare]: {
                  backgroundColor: 'rgba(155, 199, 0, 0.41)',
                },
              }),
              ...(lastMoveFrom && {
                [lastMoveFrom]: {
                  backgroundColor: 'rgba(155, 199, 0, 0.41)',
                },
              }),
              ...(lastMoveTo && {
                [lastMoveTo]: {
                  backgroundColor: 'rgba(155, 199, 0, 0.41)',
                },
              }),
            }}
            areArrowsAllowed={true}
            showBoardNotation={true}
            animationDuration={gameState?.is_player_turn ? 0 : 300}
          />
          
          {gameState && !gameState.is_player_turn && !loading && (
            <div className="absolute inset-0 bg-chess-darker bg-opacity-50 flex items-center justify-center">
              <div className="text-white text-sm font-medium px-4 py-2 bg-chess-darker bg-opacity-90 rounded">
                Engine is thinking...
              </div>
            </div>
          )}
        </div>
      </div>
      
      {moveError && (
        <div className="mt-3 text-red-400 text-sm font-medium text-center">
          {moveError}
        </div>
      )}
      
      {error && (
        <div className="mt-3 text-red-400 text-sm font-medium text-center">
          {error}
        </div>
      )}
      
      {pendingPromotion && (
        <div className="fixed inset-0 bg-chess-darker bg-opacity-50 flex items-center justify-center backdrop-blur-sm z-50">
          <div className="bg-chess-dark p-4 rounded shadow-xl">
            <h3 className="text-sm font-medium mb-3 text-center text-gray-200">Choose Promotion</h3>
            <div className="flex gap-2">
              <button onClick={() => handlePromotion('q')} className="px-3 py-1.5 bg-chess-lighter text-white text-sm rounded hover:bg-chess-border transition-colors">Queen</button>
              <button onClick={() => handlePromotion('r')} className="px-3 py-1.5 bg-chess-lighter text-white text-sm rounded hover:bg-chess-border transition-colors">Rook</button>
              <button onClick={() => handlePromotion('b')} className="px-3 py-1.5 bg-chess-lighter text-white text-sm rounded hover:bg-chess-border transition-colors">Bishop</button>
              <button onClick={() => handlePromotion('n')} className="px-3 py-1.5 bg-chess-lighter text-white text-sm rounded hover:bg-chess-border transition-colors">Knight</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
} 