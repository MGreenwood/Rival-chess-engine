import React, { memo, useEffect } from 'react';
import useStore from '../store/store';
import { formatDistanceToNow, isValid } from 'date-fns';

const ModelStats: React.FC = memo(() => {
  const { modelStats, recentGames, currentMode, uiActions } = useStore();
  
  // Refresh stats when mode changes
  useEffect(() => {
    uiActions.loadStatsForMode(currentMode);
  }, [currentMode, uiActions]);
  
  if (!modelStats) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-3 sm:p-4">
        <h2 className="text-xl sm:text-2xl font-bold mb-3 text-gray-900 dark:text-white">
          AI Model Stats
        </h2>
        <p className="text-gray-500 dark:text-gray-400">Loading stats...</p>
      </div>
    );
  }
  
  // The server sends player-perspective stats, but we want to show AI model performance
  const totalGames = (modelStats.total_games || 0);
  const playerWins = (modelStats.wins || 0);      // How many times players won
  const playerLosses = (modelStats.losses || 0);  // How many times players lost
  const playerDraws = (modelStats.draws || 0);
  
  // AI model's performance (inverse of player stats)
  const aiWins = playerLosses;        // AI wins when players lose
  const aiLosses = playerWins;        // AI loses when players win
  const aiDraws = playerDraws;        // Draws are the same
  const aiWinRate = totalGames > 0 ? ((aiWins / totalGames) * 100) : 0;
  
  // Get last 10 completed games for the current mode, sorted by most recent
  // Add date validation to filter out games with invalid dates
  // Ensure recentGames is always an array to prevent "filter is not a function" errors
  const completedGames = (Array.isArray(recentGames) ? recentGames : [])
    .filter(game => {
      // Filter out active games and games with invalid dates
      if (game.status === 'active') return false;
      
      // Filter by current mode (show single-player games for 'single' mode, community games for 'community' mode)
      const gameMode = game.mode;
      if (currentMode === 'single' && gameMode !== 'single') return false;
      if (currentMode === 'community' && gameMode !== 'community') return false;
      
      try {
        const date = new Date(game.last_move_at);
        return isValid(date);
      } catch {
        return false;
      }
    })
    .sort((a, b) => {
      try {
        const dateA = new Date(a.last_move_at);
        const dateB = new Date(b.last_move_at);
        return dateB.getTime() - dateA.getTime();
      } catch {
        return 0;
      }
    })
    .slice(0, 10);

  const formatGameDate = (dateString: string) => {
    try {
      if (!dateString) return 'Unknown';
      
      const date = new Date(dateString);
      if (isNaN(date.getTime())) {
        return 'Invalid date';
      }
      
      return formatDistanceToNow(date, { addSuffix: true });
    } catch (error) {
      console.error('Date formatting error:', error, dateString);
      return 'Unknown';
    }
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-3 sm:p-4">
      {/* AI Model Performance */}
      <div className="mb-4">
        <h2 className="text-xl sm:text-2xl font-bold mb-3 text-gray-900 dark:text-white">
          AI Model Stats
          <span className="text-sm font-normal text-gray-500 dark:text-gray-400 ml-2">
            ({currentMode === 'community' ? 'Community Model' : 'Training Model'})
          </span>
        </h2>
        <div className="grid grid-cols-3 gap-3">
          <div>
            <p className="text-sm text-gray-500 dark:text-gray-400">Total Games</p>
            <p className="text-base sm:text-lg font-semibold text-gray-900 dark:text-white">
              {totalGames.toLocaleString()}
            </p>
          </div>
          <div>
            <p className="text-sm text-gray-500 dark:text-gray-400">AI Win Rate</p>
            <p className="text-base sm:text-lg font-semibold text-gray-900 dark:text-white">
              {aiWinRate.toFixed(1)}%
            </p>
          </div>
          <div>
            <p className="text-sm text-gray-500 dark:text-gray-400">AI Record</p>
            <p className="text-base sm:text-lg font-semibold text-gray-900 dark:text-white">
              {aiWins}W {aiLosses}L {aiDraws}D
            </p>
          </div>
        </div>
        
        {/* Model Strength Indicator */}
        <div className="mt-3 p-2 bg-gray-100 dark:bg-gray-700 rounded">
          <p className="text-sm text-gray-600 dark:text-gray-300">
            <span className="font-medium">Model Strength: </span>
            {aiWinRate > 70 ? '🔥 Very Strong' : 
             aiWinRate > 50 ? '💪 Strong' : 
             aiWinRate > 30 ? '⚖️ Balanced' : 
             aiWinRate > 10 ? '🎯 Learning' : '🤖 Needs Training'}
            {totalGames > 0 && (
              <span className="ml-2 text-xs">
                (vs {totalGames} human player{totalGames !== 1 ? 's' : ''})
              </span>
            )}
          </p>
        </div>
      </div>

      {/* Recent Games (from AI's perspective) */}
      <div>
        <h3 className="text-lg sm:text-xl font-semibold mb-3 text-gray-900 dark:text-white">
          Recent {currentMode === 'community' ? 'Community' : 'Training'} Games
        </h3>
        {completedGames.length > 0 ? (
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {completedGames.map((game, index) => {
              // Show result from AI's perspective
              const getAIResult = (game: any) => {
                switch (game.status) {
                  case 'white_wins':
                    return game.player_color === 'white' 
                      ? { result: 'AI Lost', color: 'text-red-600 dark:text-red-400' }
                      : { result: 'AI Won', color: 'text-green-600 dark:text-green-400' };
                  case 'black_wins':
                    return game.player_color === 'black'
                      ? { result: 'AI Lost', color: 'text-red-600 dark:text-red-400' }
                      : { result: 'AI Won', color: 'text-green-600 dark:text-green-400' };
                  case 'draw_stalemate':
                  case 'draw_repetition':
                  case 'draw_insufficient_material':
                  case 'draw_fifty_moves':
                    return { result: 'Draw', color: 'text-yellow-600 dark:text-yellow-400' };
                  default:
                    return { result: 'Unknown', color: 'text-gray-600 dark:text-gray-400' };
                }
              };

              const { result, color } = getAIResult(game);
              const formattedDate = formatGameDate(game.last_move_at);
              
              return (
                <div
                  key={game.game_id}
                  className="flex items-center justify-between p-2 sm:p-3 bg-gray-50 dark:bg-gray-700 rounded-lg"
                >
                  <div className="flex items-center space-x-3">
                    <div className="text-xs text-gray-500 dark:text-gray-400 w-6">
                      #{completedGames.length - index}
                    </div>
                    <div>
                      <p className={`font-medium ${color}`}>
                        {result}
                      </p>
                      <p className="text-xs text-gray-500 dark:text-gray-400">
                        vs Player • {formattedDate}
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="text-sm font-semibold text-gray-900 dark:text-white">
                      {game.total_moves || 0} moves
                    </p>
                    <p className="text-xs text-gray-500 dark:text-gray-400">
                      {formattedDate}
                    </p>
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <p className="text-gray-500 dark:text-gray-400 text-sm">
            {currentMode === 'community' 
              ? 'No community games played yet. Join the community challenge to start playing!'
              : 'No training games played yet. Start a training game to help improve the AI!'
            }
          </p>
        )}
      </div>
    </div>
  );
});

export default ModelStats; 