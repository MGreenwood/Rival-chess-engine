import { useState, useEffect, useCallback, useRef } from 'react';
import { Chessboard } from 'react-chessboard';
import type { Square, Piece } from 'react-chessboard/dist/chessboard/types';
import { Chess } from 'chess.js';
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
  skipAutoInit?: boolean; // Skip auto-initialization (during restoration)
}

export function ChessGame({ 
  onMove, 
  onGameOver,
  showCoordinates = true,
  animationDuration = 300,
  boardTheme = 'classic',
  initialPosition,
  viewOnly = false,
  skipAutoInit = false
}: ChessGameProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [boardWidth, setBoardWidth] = useState(400);
  const [selectedSquare, setSelectedSquare] = useState<Square | null>(null);

  const [lastMoveFrom, setLastMoveFrom] = useState<Square | null>(null);
  const [lastMoveTo, setLastMoveTo] = useState<Square | null>(null);
  const [isInitializing, setIsInitializing] = useState(true);
  const [lastReportedGameStatus, setLastReportedGameStatus] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isSyncing, setIsSyncing] = useState(false);
  const [isPlayerInCheck, setIsPlayerInCheck] = useState(false);

  const { gameActions, currentGame, loading } = useStore();
  const { makeMove, startNewGame, syncGameState } = gameActions;

  // Add state for controlled board position
  const [boardPosition, setBoardPosition] = useState<string>(initialPosition || currentGame?.board || 'start');

  // Function to detect if the current player is in check
  const detectCheck = useCallback((fen: string): boolean => {
    try {
      const chess = new Chess(fen);
      return chess.inCheck();
    } catch (error) {
      console.warn('Error detecting check from FEN:', fen, error);
      return false;
    }
  }, []);

  // Check for check status whenever the board position changes
  useEffect(() => {
    if (boardPosition && boardPosition !== 'start') {
      const inCheck = detectCheck(boardPosition);
      setIsPlayerInCheck(inCheck);
      
      // Log check status for debugging
      if (inCheck) {
        console.log('🔴 Player is in CHECK!');
      }
    } else {
      setIsPlayerInCheck(false);
    }
  }, [boardPosition, detectCheck]);

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
      console.log('🎮 ChessGame initialization starting:', {
        skipAutoInit,
        hasCurrentGame: !!currentGame,
        gameId: currentGame?.metadata?.game_id,
        gameStatus: currentGame?.status,
        isPlayerTurn: currentGame?.is_player_turn,
        isInitializing
      });
      
      // Skip auto-initialization if restoration is in progress
      if (skipAutoInit) {
        console.log('⏭️ Skipping auto-init due to restoration in progress');
        setIsInitializing(false);
        return;
      }
      
      // Only start a new game if we don't have an active one
      if (!currentGame?.metadata?.game_id) {
        console.log('🆕 No active game found, starting new game...');
        setIsInitializing(true);
        try {
          await startNewGame('single');
          console.log('✅ New game started successfully');
        } catch (error) {
          console.error('❌ Failed to initialize game:', error);
        } finally {
          setIsInitializing(false);
        }
      } else {
        console.log('✅ Active game found, using existing game:', {
          gameId: currentGame.metadata.game_id,
          status: currentGame.status,
          isPlayerTurn: currentGame.is_player_turn,
          board: currentGame.board?.substring(0, 20) + '...'
        });
        setIsInitializing(false);
      }
    };
    
    // Only run once when the component mounts, or when skipAutoInit changes
    initGame();
  }, [startNewGame, skipAutoInit]); // Add skipAutoInit to dependencies

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
    console.log('🎯 onDrop called with:', {
      sourceSquare,
      targetSquare,
      piece,
      viewOnly,
      loading,
      isInitializing,
      hasGameId: !!currentGame?.metadata?.game_id,
      gameId: currentGame?.metadata?.game_id,
      isPlayerTurn: currentGame?.is_player_turn,
      currentGameStatus: currentGame?.status,
      currentGame: !!currentGame
    });

    // If in view-only mode, loading, initializing, or no game state, don't allow moves
    if (viewOnly || loading || isInitializing || !currentGame?.metadata?.game_id) {
      console.log('❌ Move blocked by initial conditions:', {
        viewOnly,
        loading,
        isInitializing,
        hasGameId: !!currentGame?.metadata?.game_id
      });
      return false; // Silently reject
    }

    // If it's not the player's turn, don't allow moves
    if (currentGame && !(currentGame as GameState).is_player_turn) {
      console.log('❌ Move blocked - not player turn:', {
        isPlayerTurn: currentGame.is_player_turn,
        gameStatus: currentGame.status
      });
      return false; // Silently reject
    }

    console.log('✅ Move allowed! Attempting move:', {
      from: sourceSquare,
      to: targetSquare,
      piece,
      currentBoard: currentGame.board,
      gameId: currentGame.metadata.game_id
    });

    setSelectedSquare(null);

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

    console.log('🎯 Frontend making PROMOTION move:', moveString, {
      gameId: currentGame?.metadata?.game_id || currentGame?.game_id,
      currentBoard: currentGame?.board,
      boardPosition: boardPosition,
      loading: loading,
      isPlayerTurn: currentGame?.is_player_turn,
      moveHistory: currentGame?.move_history,
      moveHistoryLength: currentGame?.move_history?.length || 0
    });

    // Add state validation before making promotion move
    if (currentGame?.board !== boardPosition) {
      console.warn('⚠️ PROMOTION STATE MISMATCH DETECTED!');
      console.warn('📋 currentGame.board:', currentGame?.board);
      console.warn('📋 boardPosition:', boardPosition);
      console.warn('🔄 Forcing board sync before promotion...');
      setBoardPosition(currentGame?.board || boardPosition);
    }

    makeMove(moveString)
      .then(() => {
        console.log('✅ Promotion move completed successfully:', moveString);
      })
              .catch((error) => {
          console.log('❌ Promotion move failed:', moveString, error.message);
          
          // If promotion fails, sync with server to get authoritative state
          if (currentGame?.metadata?.game_id || currentGame?.game_id) {
            const gameId = currentGame.metadata?.game_id || currentGame.game_id;
            if (!error.message.includes('sync issue')) {
              // Only sync if it wasn't already handled by the store
              console.log('🔄 Promotion failed, syncing with server...');
              setIsSyncing(true);
              syncGameState(gameId).finally(() => setIsSyncing(false));
            }
          }
        });

    return true;
  }, [makeMove, currentGame, onMove, loading, boardPosition, syncGameState]);

  // Fix the onPieceDrop to NOT make moves for promotions
  const onPieceDrop = useCallback((sourceSquare: Square, targetSquare: Square, piece: Piece) => {
    console.log('🎯 onPieceDrop called!', {
      sourceSquare,
      targetSquare,
      piece,
      timestamp: new Date().toISOString()
    });
    
    // Clear dragging state when drop occurs
    setIsDragging(false);
    
    const result = onDrop(sourceSquare, targetSquare, piece);
    
    console.log('🎯 onDrop result:', result);
    
    if (result) {
      const isPawnMove = piece.toLowerCase().includes('p');
      const isToLastRank = (piece.toLowerCase().includes('w') && targetSquare.endsWith('8')) || 
                           (piece.toLowerCase().includes('b') && targetSquare.endsWith('1'));
      
      if (isPawnMove && isToLastRank) {
        console.log('🎯 Pawn promotion detected, letting library handle it');
        return result;
      }
      
      const moveString = `${sourceSquare}${targetSquare}`;
      
      console.log('🎯 Frontend making move:', moveString, {
        gameId: currentGame?.metadata?.game_id || currentGame?.game_id,
        currentBoard: currentGame?.board,
        boardPosition: boardPosition,
        loading: loading,
        isPlayerTurn: currentGame?.is_player_turn,
        moveHistory: currentGame?.move_history,
        moveHistoryLength: currentGame?.move_history?.length || 0
      });
      
      // Add state validation before making move
      if (currentGame?.board !== boardPosition) {
        console.warn('⚠️ STATE MISMATCH DETECTED!');
        console.warn('📋 currentGame.board:', currentGame?.board);
        console.warn('📋 boardPosition:', boardPosition);
        console.warn('🔄 Forcing board sync...');
        setBoardPosition(currentGame?.board || boardPosition);
      }
      
      makeMove(moveString)
        .then(() => {
          console.log('✅ Move completed successfully:', moveString);
        })
        .catch((error) => {
          console.log('❌ Move failed:', moveString, error.message);
          
          // If move fails, sync with server to get authoritative state
          if (currentGame?.metadata?.game_id || currentGame?.game_id) {
            const gameId = currentGame.metadata?.game_id || currentGame.game_id;
            if (!error.message.includes('sync issue')) {
              // Only sync if it wasn't already handled by the store
              console.log('🔄 Move failed, syncing with server...');
              setIsSyncing(true);
              syncGameState(gameId).finally(() => setIsSyncing(false));
            }
          }
        });
    } else {
      console.log('❌ onDrop returned false, move was rejected');
    }
    
    return result;
  }, [onDrop, makeMove, currentGame, onMove, loading, boardPosition, syncGameState]);

  const onSquareClick = useCallback((square: Square) => {
    if (!viewOnly && !isInitializing && !loading) {
      setSelectedSquare(prev => prev === square ? null : square);
      setIsDragging(false); // Clear dragging state on click
    }
  }, [viewOnly, isInitializing, loading]);

  // Add right-click handler to cancel dragging
  useEffect(() => {
    const forceCancelDrag = () => {
      console.log('🔴 FORCE CANCEL DRAG INITIATED');
      
      // Cancel our internal state
      setSelectedSquare(null);
      setIsDragging(false);
      
      // Force reset ALL pieces to their original positions
      const allPieces = document.querySelectorAll('[data-piece]');
      console.log(`🔴 Found ${allPieces.length} pieces to reset`);
      
      allPieces.forEach((piece, index) => {
        const element = piece as HTMLElement;
        
        // Clear all drag-related styles
        element.style.transform = '';
        element.style.transition = '';
        element.style.zIndex = '';
        element.style.position = '';
        element.style.left = '';
        element.style.top = '';
        element.style.pointerEvents = '';
        element.style.cursor = '';
        
        // Remove any drag-related classes
        element.classList.remove('dragging', 'piece-dragging');
        
        console.log(`🔴 Reset piece ${index + 1}: cleared all drag styles`);
      });
      
      // Force re-render by updating the board position
      setBoardPosition(currentGame?.board || boardPosition);
      console.log('🔴 Forced board re-render');
    };

    const handleMouseDown = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      const boardContainer = containerRef.current;
      
      console.log('🟡 MouseDown detected - button:', e.button, 'target:', target.tagName);
      
      if (boardContainer && boardContainer.contains(target)) {
        if (e.button === 2) {
          // Right-click
          console.log('🔴 RIGHT-CLICK DETECTED via mousedown');
          e.preventDefault();
          e.stopPropagation();
          e.stopImmediatePropagation();
          forceCancelDrag();
          return false;
        }
        
        // Left-click on piece - start dragging
        if (target.closest('[data-piece]') && e.button === 0) {
          setIsDragging(true);
          console.log('🔵 Started dragging piece');
        }
      }
    };

    const handleContextMenu = (e: MouseEvent) => {
      const boardContainer = containerRef.current;
      const target = e.target as HTMLElement;
      
      console.log('🟡 ContextMenu detected on:', target.tagName);
      
      if (boardContainer && boardContainer.contains(target)) {
        console.log('🔴 RIGHT-CLICK DETECTED via contextmenu');
        e.preventDefault();
        e.stopPropagation();
        e.stopImmediatePropagation();
        forceCancelDrag();
        return false;
      }
    };

    const handleMouseUp = (_e: MouseEvent) => {
      // Clear dragging state on mouse up
      if (isDragging) {
        setIsDragging(false);
        console.log('🔵 Finished dragging');
      }
    };

    // Be more aggressive - add listeners to both document and window
    const options = { capture: true, passive: false };
    
    document.addEventListener('mousedown', handleMouseDown, options);
    document.addEventListener('contextmenu', handleContextMenu, options);
    document.addEventListener('mouseup', handleMouseUp, options);
    
    // Also add to window as backup
    window.addEventListener('mousedown', handleMouseDown, options);
    window.addEventListener('contextmenu', handleContextMenu, options);
    
    console.log('🟡 Event listeners attached');
    
    return () => {
      document.removeEventListener('mousedown', handleMouseDown, options);
      document.removeEventListener('contextmenu', handleContextMenu, options);
      document.removeEventListener('mouseup', handleMouseUp, options);
      
      window.removeEventListener('mousedown', handleMouseDown, options);
      window.removeEventListener('contextmenu', handleContextMenu, options);
      
      console.log('🟡 Event listeners removed');
    };
  }, [isDragging, boardPosition, currentGame?.board]);

  // Prevent file drag icon while preserving chess piece movement
  useEffect(() => {
    const preventFileDragIcon = () => {
      // Only target images and SVGs inside chess pieces to prevent file drag icon
      const images = document.querySelectorAll('[data-piece] img, .piece img, [data-testid*="piece"] img');
      const svgs = document.querySelectorAll('[data-piece] svg, .piece svg, [data-testid*="piece"] svg');
      
      [...images, ...svgs].forEach((element) => {
        const el = element as HTMLElement;
        el.style.setProperty('-webkit-user-drag', 'none');
        el.style.setProperty('user-drag', 'none');
        el.style.pointerEvents = 'none'; // Images/SVGs don't need to handle drag events
      });
    };

    // Apply styling initially and whenever the board position changes  
    const timer = setTimeout(preventFileDragIcon, 100);
    
    // Also add a mutation observer to catch any dynamically added pieces
    const observer = new MutationObserver(preventFileDragIcon);
    if (containerRef.current) {
      observer.observe(containerRef.current, { 
        childList: true, 
        subtree: true 
      });
    }

    return () => {
      clearTimeout(timer);
      observer.disconnect();
    };
  }, [boardPosition]);

  // Add periodic debug logging to track game state
  useEffect(() => {
    if (!viewOnly) {
      const debugInterval = setInterval(() => {
        console.log('🔍 ChessGame State Debug:', {
          timestamp: new Date().toISOString(),
          isInitializing,
          loading,
          viewOnly,
          hasCurrentGame: !!currentGame,
          gameId: currentGame?.metadata?.game_id || currentGame?.game_id,
          gameStatus: currentGame?.status,
          isPlayerTurn: currentGame?.is_player_turn,
          boardPosition: boardPosition?.substring(0, 20) + '...',
          moveHistoryLength: currentGame?.move_history?.length || 0,
          canMove: !viewOnly && !loading && !isInitializing && 
                   !!(currentGame?.metadata?.game_id) && 
                   !!(currentGame?.is_player_turn)
        });
      }, 5000); // Log every 5 seconds

      return () => clearInterval(debugInterval);
    }
  }, [isInitializing, loading, viewOnly, currentGame, boardPosition]);

  return (
    <>
      {/* Manual sync button */}
      {!viewOnly && currentGame?.metadata?.game_id && (
        <div className="flex justify-end mb-2">
          <button
            onClick={() => {
              const gameId = currentGame.metadata?.game_id || currentGame.game_id;
              if (gameId && !isSyncing) {
                console.log('🔄 Manual sync requested');
                setIsSyncing(true);
                syncGameState(gameId).finally(() => setIsSyncing(false));
              }
            }}
            disabled={isSyncing}
            className={`${
              isSyncing 
                ? 'bg-blue-600 cursor-not-allowed' 
                : 'bg-gray-600 hover:bg-gray-700'
            } text-white text-xs px-3 py-1 rounded shadow-lg flex items-center gap-1`}
            title="Sync with server if moves seem out of sync"
          >
            {isSyncing ? '🔄 Syncing...' : '🔄 Sync'}
          </button>
        </div>
      )}
      
    <div 
      ref={containerRef} 
      className="w-full h-full relative"
      style={{
        // Disable default drag behavior on all child elements
        userSelect: 'none',
        WebkitUserSelect: 'none',
        MozUserSelect: 'none',
        msUserSelect: 'none'
      }}
    >

      
      {/* Loading/initializing message at top but smaller */}
      {(isInitializing || loading || isSyncing) && (
        <div className="absolute top-2 left-1/2 transform -translate-x-1/2 z-10 bg-blue-500 text-white text-xs px-3 py-1 rounded">
          {isInitializing ? 'Initializing game...' : 
           isSyncing ? 'Syncing with server...' : 
           'Processing move...'}
        </div>
      )}
      
      <Chessboard
        position={boardPosition}
        onPieceDrop={onPieceDrop}
        onPromotionPieceSelect={onPromotionPieceSelect}
        onSquareClick={onSquareClick}
        boardWidth={boardWidth}
        arePiecesDraggable={!viewOnly && !loading && !isInitializing && !!currentGame?.metadata?.game_id && !!currentGame?.is_player_turn}
        customBoardStyle={{
          borderRadius: '4px',
          boxShadow: isPlayerInCheck 
            ? '0 0 0 4px #ef4444, 0 2px 8px rgba(0, 0, 0, 0.2)' 
            : '0 2px 8px rgba(0, 0, 0, 0.2)',
          transition: 'box-shadow 0.3s ease-in-out',
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
    </>
  );
}