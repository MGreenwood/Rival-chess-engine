import { useCallback, useEffect, useState } from 'react';
import { Chessboard } from 'react-chessboard';
import type { Square, Piece } from 'react-chessboard/dist/chessboard/types';
import { useGame } from '../hooks/useGame';
import { Chess } from 'chess.js';

interface ChessboardProps {
  onMove?: (move: string) => void;
  onGameOver?: (status: string) => void;
}

export function ChessGame({ onMove, onGameOver }: ChessboardProps) {
  const { gameState, error, loading, startNewGame, makeMove } = useGame();
  const [selectedSquare, setSelectedSquare] = useState<Square | null>(null);
  const [boardWidth, setBoardWidth] = useState(600);
  const [moveError, setMoveError] = useState<string | null>(null);
  const [pendingPromotion, setPendingPromotion] = useState<{from: string, to: string} | null>(null);

  // Handle window resize
  useEffect(() => {
    const handleResize = () => {
      const width = Math.min(600, window.innerWidth - 40); // 40px for padding
      setBoardWidth(width);
    };
    
    handleResize(); // Initial size
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Handle piece movement
  const onDrop = useCallback((sourceSquare: Square, targetSquare: Square, _piece: Piece) => {
    console.log('Piece dropped:', { sourceSquare, targetSquare, gameState });
    if (!gameState?.is_player_turn) {
      console.log('Cannot move: not player turn');
      return false;
    }

    // Use chess.js to get the actual piece type on the source square
    const chess = new Chess(gameState.board);
    const boardPiece = chess.get(sourceSquare);
    const isPawn = boardPiece && boardPiece.type === 'p';
    const isPromotionRank = (boardPiece && boardPiece.color === 'w' && targetSquare[1] === '8') ||
                           (boardPiece && boardPiece.color === 'b' && targetSquare[1] === '1');

    // For debugging
    console.log('Promotion check (chess.js):', {
      isPawn,
      isPromotionRank,
      boardPiece,
      sourceSquare,
      targetSquare
    });

    if (isPawn && isPromotionRank) {
      console.log('Showing promotion dialog');
      setPendingPromotion({ from: sourceSquare, to: targetSquare });
      return false;
    }

    const move = `${sourceSquare}${targetSquare}`;
    console.log('Attempting move:', move);
    makeMove(move).catch((err) => {
      console.log('Move failed:', err);
    });
    return false;
  }, [gameState, makeMove]);

  const handlePromotion = useCallback((piece: 'q' | 'r' | 'b' | 'n') => {
    if (!pendingPromotion) return;
    const { from, to } = pendingPromotion;
    const move = `${from}${to}${piece}`;
    console.log('handlePromotion: move string being sent:', move, 'piece:', piece);
    makeMove(move).catch((err) => {
      console.log('Promotion move failed:', err);
    });
    setPendingPromotion(null);
  }, [pendingPromotion, makeMove]);

  // Handle square selection
  const onSquareClick = useCallback(async (square: Square) => {
    console.log('Square clicked:', { square, selectedSquare, gameState });
    if (!gameState?.is_player_turn) {
      console.log('Cannot select square: not player turn');
      return;
    }

    if (selectedSquare === square) {
      console.log('Deselecting square');
      setSelectedSquare(null);
      return;
    }

    if (selectedSquare) {
      // Check if this is a pawn promotion move
      const piece = gameState?.board ? gameState.board.split(' ')[0].split('/').reverse().join('/') : '';
      // This is a simplified check - we need to determine what piece is on the selected square
      // For now, let's check if it's a pawn move to the promotion rank
      const isPawnPromotion = (selectedSquare[1] === '7' && square[1] === '8') || 
                             (selectedSquare[1] === '2' && square[1] === '1');
      
      if (isPawnPromotion) {
        console.log('Detected pawn promotion via square click');
        setPendingPromotion({ from: selectedSquare, to: square });
        setSelectedSquare(null);
        return;
      }

      const move = `${selectedSquare}${square}`;
      console.log('Attempting move from selection:', move);
      try {
        await makeMove(move);
        onMove?.(move);
        setSelectedSquare(null);
      } catch (err) {
        console.error('Move failed from selection:', err);
        setSelectedSquare(null);
      }
    } else {
      console.log('Selecting square:', square);
      setSelectedSquare(square);
    }
  }, [selectedSquare, gameState?.is_player_turn, makeMove, onMove, gameState]);

  // Update game over status
  useEffect(() => {
    console.log('Game state updated:', gameState);
    if (gameState?.status && ['checkmate', 'stalemate', 'draw'].includes(gameState.status)) {
      console.log('Game over:', gameState.status);
      onGameOver?.(gameState.status);
    }
  }, [gameState?.status, onGameOver, gameState]);

  // Start a new game when component mounts
  useEffect(() => {
    console.log('Starting new game on mount');
    startNewGame();
  }, [startNewGame]);

  if (error) {
    console.error('Rendering error state:', error);
    return (
      <div className="card p-4 text-red-600">
        Error: {error}
      </div>
    );
  }

  if (loading && !gameState) {
    console.log('Rendering loading state');
    return (
      <div className="card p-4">
        Loading...
      </div>
    );
  }

  console.log('Rendering chessboard:', { gameState, selectedSquare });
  return (
    <div className="card p-4 w-full max-w-[600px] mx-auto">
      <div className="mb-4">
        <h2 className="text-xl font-bold mb-2">RivalAI Chess</h2>
        {gameState?.status && (
          <div className={`text-sm ${
            gameState.status === 'check' ? 'text-red-600' :
            gameState.status === 'checkmate' ? 'text-red-600' :
            gameState.status === 'stalemate' ? 'text-yellow-600' :
            'text-gray-600'
          }`}>
            {gameState.status.charAt(0).toUpperCase() + gameState.status.slice(1)}
          </div>
        )}
      </div>
      
      <div className="relative">
        <Chessboard
          position={gameState?.board}
          onPieceDrop={onDrop}
          onSquareClick={onSquareClick}
          boardWidth={boardWidth}
          customBoardStyle={{
            borderRadius: '4px',
            boxShadow: '0 2px 10px rgba(0, 0, 0, 0.5)',
          }}
          customSquareStyles={{
            ...(selectedSquare && {
              [selectedSquare]: {
                backgroundColor: 'rgba(123, 97, 255, 0.4)',
              },
            }),
          }}
          areArrowsAllowed={true}
          showBoardNotation={true}
        />
        
        {gameState && !gameState.is_player_turn && !loading && (
          <div className="absolute inset-0 bg-black bg-opacity-30 flex items-center justify-center pointer-events-none">
            <div className="text-white text-xl font-semibold">
              Engine is thinking...
            </div>
          </div>
        )}
      </div>
      
      {moveError && (
        <div className="text-red-500 font-semibold">
          {moveError}
        </div>
      )}
      
      {pendingPromotion && (
        <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center">
          <div className="bg-white p-4 rounded-lg flex gap-2">
            <button onClick={() => handlePromotion('q')} className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600">Queen</button>
            <button onClick={() => handlePromotion('r')} className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600">Rook</button>
            <button onClick={() => handlePromotion('b')} className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600">Bishop</button>
            <button onClick={() => handlePromotion('n')} className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600">Knight</button>
          </div>
        </div>
      )}
      
      <div className="mt-4 flex flex-col items-center gap-2">
        <button
          onClick={() => {
            console.log('New game button clicked');
            setSelectedSquare(null);
            startNewGame();
          }}
          className="btn btn-primary"
          disabled={loading}
        >
          New Game
        </button>
        
        <div className="text-sm text-gray-600">
          {loading ? "Loading..." : 
           gameState?.is_player_turn ? "Your turn" : "Engine's turn"}
        </div>
      </div>
    </div>
  );
} 