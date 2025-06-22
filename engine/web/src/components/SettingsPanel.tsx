import React from 'react';
import { Dialog, Transition } from '@headlessui/react';
import { useTheme } from '../hooks/useTheme';

interface SettingsPanelProps {
  isOpen: boolean;
  onClose: () => void;
}

export const SettingsPanel: React.FC<SettingsPanelProps> = ({ isOpen, onClose }) => {
  const {
    preferences,
    setPreferences,
    isDarkMode,
    toggleTheme,
    getThemeColor
  } = useTheme();



  return (
    <Transition appear show={isOpen} as={React.Fragment}>
      <Dialog as="div" className="relative z-50" onClose={onClose}>
        <Transition.Child
          as={React.Fragment}
          enter="ease-out duration-300"
          enterFrom="opacity-0"
          enterTo="opacity-100"
          leave="ease-in duration-200"
          leaveFrom="opacity-100"
          leaveTo="opacity-0"
        >
          <div className="fixed inset-0 bg-black bg-opacity-25" />
        </Transition.Child>

        <div className="fixed inset-0 overflow-y-auto">
          <div className="flex min-h-full items-center justify-center p-4 text-center">
            <Transition.Child
              as={React.Fragment}
              enter="ease-out duration-300"
              enterFrom="opacity-0 scale-95"
              enterTo="opacity-100 scale-100"
              leave="ease-in duration-200"
              leaveFrom="opacity-100 scale-100"
              leaveTo="opacity-0 scale-95"
            >
              <Dialog.Panel 
                className="w-full max-w-md transform overflow-hidden rounded-lg p-6 text-left align-middle shadow-xl transition-all"
                style={{ backgroundColor: getThemeColor('primary') }}
              >
                <Dialog.Title
                  as="h3"
                  className="text-lg font-medium leading-6 mb-4"
                  style={{ color: getThemeColor('text') }}
                >
                  Settings
                </Dialog.Title>

                <div className="space-y-6">
                  {/* Theme */}
                  <div>
                    <label className="block text-sm font-medium mb-2 text-gray-400">
                      Theme
                    </label>
                    <button
                      onClick={toggleTheme}
                      className="flex items-center justify-between w-full p-2 rounded"
                      style={{ backgroundColor: getThemeColor('secondary') }}
                    >
                      <span style={{ color: getThemeColor('text') }}>
                        {isDarkMode ? 'Dark Mode' : 'Light Mode'}
                      </span>
                      <div className="w-5 h-5">
                        {isDarkMode ? (
                          <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
                          </svg>
                        ) : (
                          <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
                          </svg>
                        )}
                      </div>
                    </button>
                  </div>



                  {/* Board Settings */}
                  <div>
                    <label className="block text-sm font-medium mb-2 text-gray-400">
                      Board Settings
                    </label>
                    <div className="space-y-3">
                      {/* Show Coordinates */}
                      <div className="flex items-center justify-between">
                        <span style={{ color: getThemeColor('text') }}>Show Coordinates</span>
                        <button
                          onClick={() => setPreferences({ showCoordinates: !preferences.showCoordinates })}
                          className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                            preferences.showCoordinates ? 'bg-blue-600' : 'bg-gray-600'
                          }`}
                        >
                          <span
                            className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                              preferences.showCoordinates ? 'translate-x-6' : 'translate-x-1'
                            }`}
                          />
                        </button>
                      </div>

                      {/* Sound */}
                      <div className="flex items-center justify-between">
                        <span style={{ color: getThemeColor('text') }}>Sound Effects</span>
                        <button
                          onClick={() => setPreferences({ soundEnabled: !preferences.soundEnabled })}
                          className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                            preferences.soundEnabled ? 'bg-blue-600' : 'bg-gray-600'
                          }`}
                        >
                          <span
                            className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                              preferences.soundEnabled ? 'translate-x-6' : 'translate-x-1'
                            }`}
                          />
                        </button>
                      </div>

                      {/* Animation Speed */}
                      <div>
                        <div className="flex items-center justify-between mb-1">
                          <span style={{ color: getThemeColor('text') }}>Animation Speed</span>
                          <span className="text-sm text-gray-400">{preferences.animationSpeed}ms</span>
                        </div>
                        <input
                          type="range"
                          min="0"
                          max="1000"
                          step="50"
                          value={preferences.animationSpeed}
                          onChange={(e) => setPreferences({ animationSpeed: parseInt(e.target.value) })}
                          className="w-full"
                        />
                      </div>

                      {/* Board Theme */}
                      <div>
                        <div className="flex items-center justify-between mb-2">
                          <span style={{ color: getThemeColor('text') }}>Board Theme</span>
                        </div>
                        <div className="grid grid-cols-2 gap-2">
                          <button
                            onClick={() => setPreferences({ boardTheme: 'classic' })}
                            className={`p-2 rounded text-sm ${
                              preferences.boardTheme === 'classic' ? 'ring-2 ring-blue-500' : ''
                            }`}
                            style={{ backgroundColor: getThemeColor('secondary') }}
                          >
                            <span style={{ color: getThemeColor('text') }}>Classic</span>
                          </button>
                          <button
                            onClick={() => setPreferences({ boardTheme: 'modern' })}
                            className={`p-2 rounded text-sm ${
                              preferences.boardTheme === 'modern' ? 'ring-2 ring-blue-500' : ''
                            }`}
                            style={{ backgroundColor: getThemeColor('secondary') }}
                          >
                            <span style={{ color: getThemeColor('text') }}>Modern</span>
                          </button>
                        </div>
                      </div>

                      {/* Piece Style */}
                      <div>
                        <div className="flex items-center justify-between mb-2">
                          <span style={{ color: getThemeColor('text') }}>Piece Style</span>
                        </div>
                        <div className="grid grid-cols-2 gap-2">
                          <button
                            onClick={() => setPreferences({ pieceStyle: 'standard' })}
                            className={`p-2 rounded text-sm ${
                              preferences.pieceStyle === 'standard' ? 'ring-2 ring-blue-500' : ''
                            }`}
                            style={{ backgroundColor: getThemeColor('secondary') }}
                          >
                            <span style={{ color: getThemeColor('text') }}>Standard</span>
                          </button>
                          <button
                            onClick={() => setPreferences({ pieceStyle: 'modern' })}
                            className={`p-2 rounded text-sm ${
                              preferences.pieceStyle === 'modern' ? 'ring-2 ring-blue-500' : ''
                            }`}
                            style={{ backgroundColor: getThemeColor('secondary') }}
                          >
                            <span style={{ color: getThemeColor('text') }}>Modern</span>
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="mt-6 flex justify-end">
                  <button
                    type="button"
                    className="rounded px-4 py-2 text-sm font-medium hover:bg-opacity-10 hover:bg-white transition-colors"
                    onClick={onClose}
                    style={{ color: getThemeColor('text') }}
                  >
                    Close
                  </button>
                </div>
              </Dialog.Panel>
            </Transition.Child>
          </div>
        </div>
      </Dialog>
    </Transition>
  );
}; 