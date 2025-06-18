import { useCallback } from 'react';
import { useStore } from '../store/store';
import type { AnalysisResult } from '../store/types';

interface AnalysisHook {
  currentAnalysis: AnalysisResult | null;
  analysisHistory: AnalysisResult[];
  isAnalyzing: boolean;
  analyzePosition: (fen: string) => Promise<void>;
  clearAnalysis: () => void;
  getFormattedEvaluation: () => string;
}

export function useAnalysis(): AnalysisHook {
  const {
    currentAnalysis,
    analysisHistory,
    analysisActions
  } = useStore();

  const getFormattedEvaluation = useCallback(() => {
    if (!currentAnalysis) return '0.0';

    const evaluation = currentAnalysis.evaluation;
    if (evaluation === 0) return '0.0';

    const isPositive = evaluation > 0;
    const absValue = Math.abs(evaluation);
    
    // Check if it's a mate score
    if (absValue > 100) {
      const mateIn = Math.ceil((1000 - absValue) / 2);
      return `M${mateIn}`;
    }

    // Regular evaluation
    return `${isPositive ? '+' : '-'}${absValue.toFixed(1)}`;
  }, [currentAnalysis]);

  return {
    currentAnalysis,
    analysisHistory,
    isAnalyzing: false, // TODO: Add loading state to store
    analyzePosition: analysisActions.analyzePosition,
    clearAnalysis: analysisActions.clearAnalysis,
    getFormattedEvaluation
  };
} 