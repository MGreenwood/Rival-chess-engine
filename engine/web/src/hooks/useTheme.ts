import { useCallback } from 'react';
import { useStore } from '../store/store';
import type { Theme, UserPreferences } from '../store/types';

interface ThemeHook {
  theme: Theme;
  preferences: UserPreferences;
  setTheme: (theme: Theme) => void;
  setPreferences: (prefs: Partial<UserPreferences>) => void;
  toggleTheme: () => void;
  isDarkMode: boolean;
  getThemeColor: (key: keyof Theme['colors']) => string;
}

export function useTheme(): ThemeHook {
  const { theme, preferences, uiActions } = useStore();

  const toggleTheme = useCallback(() => {
    const newMode = theme.mode === 'dark' ? 'light' : 'dark';
    const newColors = newMode === 'dark' ? {
      primary: '#1a1b26',
      secondary: '#13141f',
      background: '#0d0e14',
      text: '#ffffff'
    } : {
      primary: '#ffffff',
      secondary: '#f3f4f6',
      background: '#f9fafb',
      text: '#111827'
    };

    uiActions.setTheme({
      mode: newMode,
      colors: newColors
    });
  }, [theme.mode, uiActions]);

  const getThemeColor = useCallback((key: keyof Theme['colors']) => {
    return theme.colors[key];
  }, [theme.colors]);

  return {
    theme,
    preferences,
    setTheme: uiActions.setTheme,
    setPreferences: uiActions.setPreferences,
    toggleTheme,
    isDarkMode: theme.mode === 'dark',
    getThemeColor
  };
} 