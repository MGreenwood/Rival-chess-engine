import React, { useEffect } from 'react';
import { CommunityGame } from '../components/CommunityGame';
import ModelStats from '../components/ModelStats';
import useStore from '../store/store';

const CommunityChallenge: React.FC = () => {
  const { uiActions } = useStore();

  // Set mode to community when this page loads
  useEffect(() => {
    uiActions.setCurrentMode('community');
  }, [uiActions]);

  return (
    <>
    <div className="flex bg-white dark:bg-gray-900 transition-colors">
      {/* Left sidebar - Model Stats */}
      <div className="w-80 p-4 hidden lg:block overflow-y-auto scrollbar-hide">
        <ModelStats />
      </div>

      {/* Main game area */}
      <div className="flex-1 flex flex-col items-center justify-start p-2 overflow-y-auto scrollbar-hide">
        <CommunityGame />
      </div>

      {/* Right sidebar - How to Play */}
      <div className="w-80 p-4 hidden lg:block overflow-y-auto scrollbar-hide">
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-4">
          <h2 className="text-xl font-bold mb-3 text-gray-900 dark:text-white text-center">
            How to Play
          </h2>
          <ul className="space-y-2 text-gray-600 dark:text-gray-300">
            <li>• Drag pieces or click squares to vote</li>
            <li>• First vote starts 10-second timer</li>
            <li>• Each player gets one vote per turn</li>
            <li>• Most voted move is played</li>
            <li>• Ties are broken randomly</li>
          </ul>
        </div>
      </div>

    </div>
    
    {/* Mobile view for sidebars */}
    <div className="lg:hidden bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg transition-colors m-2">
      <div className="grid grid-cols-2 gap-2 p-2">
        <div className="col-span-1 overflow-y-auto scrollbar-hide max-h-32">
          <ModelStats />
        </div>
        <div className="col-span-1 overflow-y-auto scrollbar-hide max-h-32">
          <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-2">
            <h2 className="text-base font-bold mb-1 text-gray-900 dark:text-white text-center">
              How to Play
            </h2>
            <ul className="space-y-0.5 text-xs text-gray-600 dark:text-gray-300">
              <li>• Drag pieces to vote</li>
              <li>• Auto 10-second timer</li>
              <li>• Most votes wins</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
    </>
  );
};

export default CommunityChallenge; 