import { useState, useEffect, useCallback, useRef } from 'react';
import { Chessboard } from 'react-chessboard';
import type { Square, Piece } from 'react-chessboard/dist/chessboard/types';
import useStore from '../store/store';

interface ChessGameProps {
  onMove: (move: string) => void;
  onGameOver: (status: string) => void;
  showCoordinates?: boolean;
  animationDuration?: number;
  pieceStyle?: string;
  boardTheme?: string;
  initialPosition?: string;
  viewOnly?: boolean;
}

export function ChessGame({ 
  onMove, 
  onGameOver,
  showCoordinates = true,
  animationDuration = 300,
  pieceStyle = 'standard',
  boardTheme = 'classic',
  initialPosition,
  viewOnly = false
}: ChessGameProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [boardWidth, setBoardWidth] = useState(400);
  const [selectedSquare, setSelectedSquare] = useState<Square | null>(null);
  const [pendingPromotion, setPendingPromotion] = useState<{from: string, to: string} | null>(null);
  const [moveError, setMoveError] = useState<string | null>(null);
  const [lastMoveFrom, setLastMoveFrom] = useState<Square | null>(null);
  const [lastMoveTo, setLastMoveTo] = useState<Square | null>(null);

  const { gameActions, currentGame, loading } = useStore();
  const { makeMove, startNewGame } = gameActions;

  // Add state for controlled board position
  const [boardPosition, setBoardPosition] = useState<string>(initialPosition || currentGame?.board || 'start');

  // Update board width when container size changes
  useEffect(() => {
    if (!containerRef.current) return;

    const updateWidth = () => {
      if (containerRef.current) {
        const containerWidth = containerRef.current.clientWidth;
        // Use container width directly since parent now controls max size
        setBoardWidth(containerWidth);
      }
    };

    // Initial size calculation
    updateWidth();

    // Create resize observer
    const resizeObserver = new ResizeObserver(updateWidth);
    resizeObserver.observe(containerRef.current);

    return () => {
      resizeObserver.disconnect();
    };
  }, []);

  // Start new game on mount
  useEffect(() => {
    startNewGame();
  }, [startNewGame]);

  // Update parent components
  useEffect(() => {
    if (currentGame?.status && currentGame.status !== 'active') {
      onGameOver(currentGame.status);
    }
  }, [currentGame?.status, onGameOver]);

  // Update position when game state changes
  useEffect(() => {
    if (currentGame?.board) {
      console.log('Board state updated:', {
        board: currentGame.board,
        is_player_turn: currentGame.is_player_turn,
        status: currentGame.status
      });
      setBoardPosition(currentGame.board);
    }
  }, [currentGame?.board]);

  // Update position when initialPosition prop changes
  useEffect(() => {
    if (initialPosition) {
      console.log('Initial position set:', initialPosition);
      setBoardPosition(initialPosition);
    }
  }, [initialPosition]);

  const onDrop = useCallback((sourceSquare: Square, targetSquare: Square, piece: Piece) => {
    // If in view-only mode, don't allow moves
    if (viewOnly) {
      return false;
    }

    // If it's not the player's turn, don't allow moves
    if (currentGame && !currentGame.is_player_turn) {
      setMoveError("Not your turn");
      return false;
    }

    console.log('Attempting move:', {
      from: sourceSquare,
      to: targetSquare,
      piece,
      currentBoard: currentGame?.board
    });

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
    
    // We'll return false initially and update the board position after validation
    makeMove(moveString)
      .then(() => {
        setLastMoveFrom(sourceSquare);
        setLastMoveTo(targetSquare);
        if (currentGame?.board) {
          onMove(moveString);
        }
      })
      .catch((err: any) => {
        console.error('Move error:', err);
        const errorMessage = err.response?.data?.error_message || err.message || 'Invalid move';
        setMoveError(errorMessage);
        // Force a board update to the correct position
        if (currentGame?.board) {
          // Small delay to ensure the piece has finished animating back
          setTimeout(() => {
            setBoardPosition(currentGame.board);
          }, 100);
        }
      });

    return false; // Always return false and let the server response update the position
  }, [makeMove, currentGame, onMove, viewOnly]);

  const handlePromotion = (promotionPiece: string) => {
    if (!pendingPromotion) return;

    const moveString = `${pendingPromotion.from}${pendingPromotion.to}${promotionPiece}`;
    makeMove(moveString)
      .then(() => {
        setLastMoveFrom(pendingPromotion.from as Square);
        setLastMoveTo(pendingPromotion.to as Square);
        if (currentGame?.board) {
          onMove(moveString);
        }
      })
      .catch((err: Error) => {
        setMoveError(err.message || 'Invalid move');
      })
      .finally(() => {
        setPendingPromotion(null);
      });
  };

  const onSquareClick = useCallback((square: Square) => {
    if (!viewOnly) {
      setSelectedSquare(prev => prev === square ? null : square);
    }
  }, [viewOnly]);

  console.log('Rendering chessboard:', { currentGame, selectedSquare });
  return (
    <div ref={containerRef} className="w-full h-full">
      {moveError && (
        <div className="absolute top-0 left-0 right-0 z-10 p-2 bg-red-500 text-white text-sm text-center rounded-t-lg">
          {moveError}
        </div>
      )}
      <Chessboard
        position={boardPosition}
        onPieceDrop={onDrop}
        onSquareClick={onSquareClick}
        boardWidth={boardWidth}
        customBoardStyle={{
          borderRadius: '4px',
          boxShadow: '0 2px 8px rgba(0, 0, 0, 0.2)',
        }}
        customDarkSquareStyle={{ 
          backgroundColor: boardTheme === 'classic' ? '#b58863' : '#70a2a3' 
        }}
        customLightSquareStyle={{ 
          backgroundColor: boardTheme === 'classic' ? '#f0d9b5' : '#deeed6' 
        }}
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
        showBoardNotation={showCoordinates}
        animationDuration={currentGame?.is_player_turn ? 0 : animationDuration}
      />
      
      {currentGame && !currentGame.is_player_turn && !loading && !viewOnly && (
        <div className="absolute inset-0 bg-chess-darker bg-opacity-50 flex items-center justify-center">
          <div className="text-white text-sm font-medium px-4 py-2 bg-chess-darker bg-opacity-90 rounded">
            Engine is thinking...
          </div>
        </div>
      )}
      
      {pendingPromotion && !viewOnly && (
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