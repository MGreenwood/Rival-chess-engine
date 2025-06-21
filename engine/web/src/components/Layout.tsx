import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { SettingsPanel } from './SettingsPanel';

const Layout: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const location = useLocation();

  return (
    <div className="min-h-screen bg-gray-900">
      {/* Title Bar */}
      <div className="fixed top-0 left-0 right-0 h-14 bg-gray-800 border-b border-gray-700 flex items-center px-4 z-10">
        <div className="flex-1 flex items-center">
          <Link to="/" className="text-xl font-bold text-gray-100 hover:text-white transition-colors">
            Rival Chess
          </Link>
          <div className="ml-4 flex items-center space-x-4">
            <Link
              to="/train"
              className={`px-3 py-1 rounded-full text-sm font-medium transition-colors ${
                location.pathname === '/train'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              Train the Model
            </Link>
            <Link
              to="/community"
              className={`px-3 py-1 rounded-full text-sm font-medium transition-colors ${
                location.pathname === '/community'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              Community Challenge
            </Link>
          </div>
        </div>
        <div className="flex items-center space-x-4">
          <button
            onClick={() => setIsSettingsOpen(true)}
            className="p-2 text-gray-400 hover:text-gray-100 rounded-lg hover:bg-gray-700 transition-colors"
            title="Settings"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
          </button>
        </div>
      </div>

      {children}

      <SettingsPanel
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
      />
    </div>
  );
};

export default Layout; 