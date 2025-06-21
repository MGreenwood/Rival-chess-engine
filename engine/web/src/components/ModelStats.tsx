import React from 'react';
import useStore from '../store/store';
import { formatDistanceToNow, format } from 'date-fns';

const ModelStats: React.FC = () => {
  const { modelStats, leaderboard } = useStore();
  
  // Format creation date
  const creationDate = new Date(modelStats.createdAt);
  const formattedCreationDate = format(creationDate, 'MMM d, yyyy');
  const timeAgo = formatDistanceToNow(creationDate);
  
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-3 sm:p-4">
      {/* Model Info */}
      <div className="mb-4">
        <h2 className="text-xl sm:text-2xl font-bold mb-3 text-gray-900 dark:text-white">
          RivalAI Stats
        </h2>
        <div className="grid grid-cols-2 gap-3">
          <div className="col-span-2">
            <p className="text-sm text-gray-500 dark:text-gray-400">Created</p>
            <p className="text-base sm:text-lg font-semibold text-gray-900 dark:text-white">
              {formattedCreationDate} ({timeAgo} ago)
            </p>
          </div>
          <div>
            <p className="text-sm text-gray-500 dark:text-gray-400">Total Games</p>
            <p className="text-base sm:text-lg font-semibold text-gray-900 dark:text-white">
              {modelStats.totalGames.toLocaleString()}
            </p>
          </div>
          <div>
            <p className="text-sm text-gray-500 dark:text-gray-400">Current Epoch</p>
            <p className="text-base sm:text-lg font-semibold text-gray-900 dark:text-white">
              {modelStats.currentEpoch}
            </p>
          </div>
        </div>
      </div>

      {/* Next Training Countdown */}
      {modelStats.nextTrainingAt && (
        <div className="mb-4 p-3 bg-blue-50 dark:bg-blue-900 rounded-lg">
          <h3 className="text-base sm:text-lg font-semibold mb-1 text-blue-900 dark:text-blue-100">
            Next Training
          </h3>
          <p className="text-sm sm:text-base text-blue-800 dark:text-blue-200">
            Starts in {formatDistanceToNow(new Date(modelStats.nextTrainingAt))}
          </p>
        </div>
      )}

      {/* Recent Games */}
      <div className="mb-4">
        <h3 className="text-lg sm:text-xl font-semibold mb-3 text-gray-900 dark:text-white">
          Recent Games
        </h3>
        <div className="space-y-2">
          {modelStats.recentGames.length === 0 ? (
            <p className="text-sm text-gray-500 dark:text-gray-400">No games played yet</p>
          ) : (
            modelStats.recentGames.map((game) => (
              <div
                key={game.id}
                className="flex items-center justify-between p-2 sm:p-3 bg-gray-50 dark:bg-gray-700 rounded-lg"
              >
                <div className="flex-1">
                  <div className="flex items-center justify-between mb-1">
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                      game.result === 'win'
                        ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                        : game.result === 'loss'
                        ? 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
                        : 'bg-gray-100 text-gray-800 dark:bg-gray-600 dark:text-gray-200'
                    }`}>
                      {game.result.charAt(0).toUpperCase() + game.result.slice(1)}
                    </span>
                    <span className="text-xs text-gray-500 dark:text-gray-400">
                      {formatDistanceToNow(new Date(game.timestamp))} ago
                    </span>
                  </div>
                  <div className="flex items-center space-x-2 text-sm text-gray-600 dark:text-gray-300">
                    <span>{game.moves} moves</span>
                    <span>•</span>
                    <span>Playing as {game.playerColor}</span>
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Leaderboard */}
      <div>
        <h3 className="text-lg sm:text-xl font-semibold mb-3 text-gray-900 dark:text-white">
          Top Players
        </h3>
        <div className="space-y-2">
          {leaderboard.slice(0, 5).map((entry) => (
            <div
              key={entry.username}
              className="flex items-center justify-between p-2 sm:p-3 bg-gray-50 dark:bg-gray-700 rounded-lg"
            >
              <div>
                <p className="font-medium text-gray-900 dark:text-white">
                  {entry.username}
                </p>
                <p className="text-xs sm:text-sm text-gray-500 dark:text-gray-400">
                  {entry.wins}W {entry.losses}L {entry.draws}D
                </p>
              </div>
              <div className="text-base sm:text-lg font-semibold text-gray-900 dark:text-white">
                {entry.rating}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default ModelStats; 