import { useState, useEffect, useCallback, useRef } from 'react';
import { Chessboard } from 'react-chessboard';
import type { Square, Piece } from 'react-chessboard/dist/chessboard/types';
import useStore from '../store/store';
import type { GameState } from '../store/types';

interface ChessGameProps {
  onMove?: (move: string) => void;
  onGameOver: (status: string) => void;
  showCoordinates?: boolean;
  animationDuration?: number;
  boardTheme?: string;
  initialPosition?: string;
  viewOnly?: boolean;
}

export function ChessGame({ 
  onMove, 
  onGameOver,
  showCoordinates = true,
  animationDuration = 300,
  boardTheme = 'classic',
  initialPosition,
  viewOnly = false
}: ChessGameProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [boardWidth, setBoardWidth] = useState(400);
  const [selectedSquare, setSelectedSquare] = useState<Square | null>(null);
  const [moveError, setMoveError] = useState<string | null>(null);
  const [lastMoveFrom, setLastMoveFrom] = useState<Square | null>(null);
  const [lastMoveTo, setLastMoveTo] = useState<Square | null>(null);
  const [isInitializing, setIsInitializing] = useState(true);
  const [lastReportedGameStatus, setLastReportedGameStatus] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);

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
        setBoardWidth(containerWidth);
      }
    };

    updateWidth();
    const resizeObserver = new ResizeObserver(updateWidth);
    resizeObserver.observe(containerRef.current);

    return () => {
      resizeObserver.disconnect();
    };
  }, []);

  // Fix the game initialization to prevent multiple calls
  useEffect(() => {
    const initGame = async () => {
      // Only start a new game if we don't have an active one
      if (!currentGame?.metadata?.game_id) {
        setIsInitializing(true);
        try {
          await startNewGame('single');
        } catch (error) {
          console.error('Failed to initialize game:', error);
        } finally {
          setIsInitializing(false);
        }
      } else {
        setIsInitializing(false);
      }
    };
    
    // Only run once when the component mounts
    initGame();
  }, [startNewGame]); // Remove currentGame from dependencies to prevent loops

  // Update parent components - only trigger when game status actually changes
  useEffect(() => {
    const currentStatus = currentGame?.metadata?.status || currentGame?.status;
    
    // Only report game over if:
    // 1. We have a valid game status
    // 2. The status is not 'active' 
    // 3. We haven't already reported this status
    // 4. We're not in view-only mode (which would be artificial)
    if (currentStatus && 
        currentStatus !== 'active' && 
        currentStatus !== lastReportedGameStatus &&
        !viewOnly) {
      setLastReportedGameStatus(currentStatus);
      onGameOver(currentStatus);
    } else if (currentStatus === 'active' && lastReportedGameStatus !== 'active') {
      // Reset when a new active game starts
      setLastReportedGameStatus('active');
    }
  }, [currentGame?.metadata?.status, currentGame?.status, onGameOver, viewOnly, lastReportedGameStatus]);

  // Update position when game state changes or when viewing specific positions
  useEffect(() => {
    if (viewOnly && initialPosition) {
      // If we're in view-only mode and have an initial position, use that
      setBoardPosition(initialPosition);
    } else if (currentGame?.board) {
      // Otherwise use the current game board
      setBoardPosition(currentGame.board);
    } else if (!isInitializing) {
      // Only set default position if not initializing
      setBoardPosition('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1');
    }
  }, [currentGame?.board, isInitializing, viewOnly, initialPosition]);

  // Simplify move highlighting effect
  useEffect(() => {
    if (currentGame?.move_history && currentGame.move_history.length > 0) {
      const lastMove = currentGame.move_history[currentGame.move_history.length - 1];
      
      if (lastMove && typeof lastMove === 'string' && lastMove.length >= 4) {
        const fromSquare = lastMove.substring(0, 2) as Square;
        const toSquare = lastMove.substring(2, 4) as Square;
        
        const validSquares = /^[a-h][1-8]$/;
        if (validSquares.test(fromSquare) && validSquares.test(toSquare)) {
          setLastMoveFrom(fromSquare);
          setLastMoveTo(toSquare);
        }
      }
    } else {
      setLastMoveFrom(null);
      setLastMoveTo(null);
    }
  }, [currentGame?.move_history?.length]); // Only trigger on history length change

  const onDrop = useCallback((sourceSquare: Square, targetSquare: Square, piece: Piece) => {
    // If in view-only mode, loading, initializing, or no game state, don't allow moves
    if (viewOnly || loading || isInitializing || !currentGame?.metadata?.game_id) {
      if (isInitializing) {
        setMoveError("Game is initializing, please wait...");
      } else if (loading) {
        setMoveError("Processing previous move, please wait...");
      } else if (!currentGame?.metadata?.game_id) {
        setMoveError("Game not properly initialized");
      }
      return false;
    }

    // If it's not the player's turn, don't allow moves
    if (currentGame && !(currentGame as GameState).is_player_turn) {
      setMoveError("Not your turn");
      return false;
    }

    console.log('Attempting move:', {
      from: sourceSquare,
      to: targetSquare,
      piece,
      currentBoard: currentGame.board,
      gameId: currentGame.metadata.game_id
    });

    setSelectedSquare(null);
    setMoveError(null);

    // Let the library handle promotion - just return true to allow the move
    // The library will call onPromotionPieceSelect if it's a promotion
    return true;
  }, [currentGame, viewOnly, loading, isInitializing]);

  // Simplify promotion handling
  const onPromotionPieceSelect = useCallback((piece?: string, promoteFromSquare?: Square, promoteToSquare?: Square) => {
    if (!promoteFromSquare || !promoteToSquare) {
      return false;
    }

    let promotionPiece = 'q';
    if (piece) {
      const cleanPiece = piece.toLowerCase().replace(/[w=]/g, '');
      switch (cleanPiece) {
        case 'queen':
        case 'q': promotionPiece = 'q'; break;
        case 'rook':
        case 'r': promotionPiece = 'r'; break;
        case 'bishop':
        case 'b': promotionPiece = 'b'; break;
        case 'knight':
        case 'n': promotionPiece = 'n'; break;
        default: promotionPiece = 'q';
      }
    }
    
    const moveString = `${promoteFromSquare}${promoteToSquare}${promotionPiece}`;

    console.log('🎯 Component making promotion move:', moveString, {
      gameId: currentGame?.metadata?.game_id || currentGame?.game_id,
      currentBoard: currentGame?.board,
      loading: loading,
      isPlayerTurn: currentGame?.is_player_turn
    });

    makeMove(moveString)
      .then(() => {
        console.log('✅ Promotion move completed successfully:', moveString);
      })
      .catch((error) => {
        console.log('❌ Promotion move failed:', moveString, error.message);
      });

    return true;
  }, [makeMove, currentGame, onMove, loading]);

  // Fix the onPieceDrop to NOT make moves for promotions
  const onPieceDrop = useCallback((sourceSquare: Square, targetSquare: Square, piece: Piece) => {
    // Clear dragging state when drop occurs
    setIsDragging(false);
    
    const result = onDrop(sourceSquare, targetSquare, piece);
    
    if (result) {
      const isPawnMove = piece.toLowerCase().includes('p');
      const isToLastRank = (piece.toLowerCase().includes('w') && targetSquare.endsWith('8')) || 
                           (piece.toLowerCase().includes('b') && targetSquare.endsWith('1'));
      
      if (isPawnMove && isToLastRank) {
        return result;
      }
      
      const moveString = `${sourceSquare}${targetSquare}`;
      
      console.log('🎯 Component making move:', moveString, {
        gameId: currentGame?.metadata?.game_id || currentGame?.game_id,
        currentBoard: currentGame?.board,
        loading: loading,
        isPlayerTurn: currentGame?.is_player_turn
      });
      
      makeMove(moveString)
        .then(() => {
          console.log('✅ Move completed successfully:', moveString);
        })
        .catch((error) => {
          console.log('❌ Move failed:', moveString, error.message);
        });
    }
    
    return result;
  }, [onDrop, makeMove, currentGame, onMove, loading]);

  const onSquareClick = useCallback((square: Square) => {
    if (!viewOnly && !isInitializing && !loading) {
      setSelectedSquare(prev => prev === square ? null : square);
      setIsDragging(false); // Clear dragging state on click
    }
  }, [viewOnly, isInitializing, loading]);

  // Add auto-dismiss for error messages
  useEffect(() => {
    if (moveError) {
      const timer = setTimeout(() => {
        setMoveError(null);
      }, 5000); // Auto-dismiss after 5 seconds

      return () => clearTimeout(timer);
    }
  }, [moveError]);

  // Add right-click handler to cancel dragging
  useEffect(() => {
    const handleMouseDown = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      const boardContainer = containerRef.current;
      
      // Handle right-clicks anywhere when dragging, or on board when not dragging
      if (e.button === 2 && (isDragging || (boardContainer && boardContainer.contains(target)))) {
        e.preventDefault();
        e.stopPropagation();
        
        // Cancel our internal state
        setSelectedSquare(null);
        setIsDragging(false);
        
        // Force cancel any ongoing drag by triggering a mouseup event
        const mouseUpEvent = new MouseEvent('mouseup', {
          bubbles: true,
          cancelable: true,
          button: 0
        });
        
        // Dispatch to any dragged piece or the board itself
        const draggedPiece = document.querySelector('[data-piece][style*="transform"]');
        if (draggedPiece) {
          draggedPiece.dispatchEvent(mouseUpEvent);
          console.log('🔴 FORCED DROP: Dispatched mouseup to dragged piece');
        } else if (boardContainer) {
          boardContainer.dispatchEvent(mouseUpEvent);
          console.log('🔴 FORCED DROP: Dispatched mouseup to board');
        }
        
        console.log('🔴 RIGHT-CLICK DETECTED - canceling drag/selection (isDragging:', isDragging, ')');
        return;
      }
      
      if (boardContainer && boardContainer.contains(target)) {
        console.log('MouseDown on board - button:', e.button, 'target:', target.tagName, 'className:', target.className);
        
        // Left-click on piece - start dragging
        if (target.closest('[data-piece]') && e.button === 0) {
          setIsDragging(true);
          console.log('🔵 Started dragging piece');
        }
      }
    };

    const handleContextMenu = (e: MouseEvent) => {
      // Prevent context menu on chessboard
      const boardContainer = containerRef.current;
      if (boardContainer && boardContainer.contains(e.target as Node)) {
        e.preventDefault();
        e.stopPropagation();
      }
    };

    const handleMouseUp = (e: MouseEvent) => {
      // Clear dragging state on mouse up
      if (isDragging) {
        setIsDragging(false);
        console.log('Finished dragging');
      }
    };

    // Use document-level listeners with capture flag to catch events early
    document.addEventListener('mousedown', handleMouseDown, { capture: true });
    document.addEventListener('contextmenu', handleContextMenu, { capture: true });
    document.addEventListener('mouseup', handleMouseUp);
    
    return () => {
      document.removeEventListener('mousedown', handleMouseDown, { capture: true });
      document.removeEventListener('contextmenu', handleContextMenu, { capture: true });
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging]);

  return (
    <div ref={containerRef} className="w-full h-full relative">
      {/* Move error to bottom-right corner as a toast */}
      {moveError && (
        <div className="absolute bottom-4 right-4 z-20 bg-red-500 text-white text-xs px-3 py-2 rounded-lg shadow-lg max-w-xs">
          {moveError}
          <button 
            onClick={() => setMoveError(null)}
            className="ml-2 text-white hover:text-gray-200"
          >
            ×
          </button>
        </div>
      )}
      
      {/* Loading/initializing message at top but smaller */}
      {(isInitializing || loading) && (
        <div className="absolute top-2 left-1/2 transform -translate-x-1/2 z-10 bg-blue-500 text-white text-xs px-3 py-1 rounded">
          {isInitializing ? 'Initializing game...' : 'Processing move...'}
        </div>
      )}
      
      <Chessboard
        position={boardPosition}
        onPieceDrop={onPieceDrop}
        onPromotionPieceSelect={onPromotionPieceSelect}
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
          backgroundColor: boardTheme === 'classic' ? '#f0d9b5' : '#b1dddf' 
        }}
        showBoardNotation={showCoordinates}
        animationDuration={animationDuration}
        customSquareStyles={{
          ...(selectedSquare ? { [selectedSquare]: { backgroundColor: 'rgba(255, 255, 0, 0.4)' } } : {}),
          ...(lastMoveFrom ? { [lastMoveFrom]: { backgroundColor: 'rgba(255, 255, 0, 0.2)' } } : {}),
          ...(lastMoveTo ? { [lastMoveTo]: { backgroundColor: 'rgba(255, 255, 0, 0.2)' } } : {})
        }}
      />
    </div>
  );
} 