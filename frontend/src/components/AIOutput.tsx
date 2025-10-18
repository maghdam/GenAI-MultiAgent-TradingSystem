import { forwardRef, useCallback, useEffect, useImperativeHandle, useMemo, useRef, useState } from 'react';
import type { JSX } from 'react';

import type { AgentSignal } from './SidePanel';
import type { AnalysisResult } from '../types/analysis';

export interface AIOutputHandle {
  runAnalysis: () => void;
  cancelAnalysis: () => void;
  placeTrade: () => void;
}

interface AnalysisOptions {
  fast: boolean;
  max_bars: number;
  max_tokens: number;
  model: string | null;
  options: Record<string, unknown>;
}

interface RunAnalysisPayload extends AnalysisOptions {
  symbol: string;
  timeframe: string;
  indicators: string[];
  strategy: string;
}

interface Notification {
  message: string;
  status: 'success' | 'error';
}

interface AIOutputProps {
  symbol: string;
  timeframe: string;
  strategy: string;
  lotSize: number;
  fastMode: boolean;
  maxBars: number;
  maxTokens: number;
  modelName: string;
  onAnalysisStart?: () => void;
  onAnalysisComplete?: (analysis: AnalysisResult | null) => void;
  onNotify?: (notification: Notification) => void;
  selectedSignal?: AgentSignal | null;
  renderToolbar?: (handlers: {
    runAnalysis: () => void;
    cancelAnalysis: () => void;
    placeTrade: () => void;
    loading: boolean;
    analysis: AnalysisResult | null;
  }) => JSX.Element | null;
  hideToolbar?: boolean;
}

interface AnalysisState {
  analysis: AnalysisResult | null;
  loading: boolean;
  error: string | null;
  cancelled: boolean;
}

const initialState: AnalysisState = {
  analysis: null,
  loading: false,
  error: null,
  cancelled: false,
};

const AIOutput = forwardRef<AIOutputHandle, AIOutputProps>(function AIOutput(
  {
    symbol,
    timeframe,
    strategy,
    lotSize,
    fastMode,
    maxBars,
    maxTokens,
    modelName,
    onAnalysisStart,
    onAnalysisComplete,
    onNotify,
    selectedSignal,
    renderToolbar,
    hideToolbar = false,
  },
  ref,
) {
  const [state, setState] = useState<AnalysisState>(initialState);
  const abortControllerRef = useRef<AbortController | null>(null);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [analysisCounter, setAnalysisCounter] = useState(() => Number(localStorage.getItem('analysisCounter')) || 0);

  const analysisOptions: AnalysisOptions = useMemo(
    () => ({
      fast: fastMode,
      max_bars: maxBars,
      max_tokens: maxTokens,
      model: modelName.trim() ? modelName.trim() : null,
      options: {
        num_predict: maxTokens,
        temperature: fastMode ? 0.2 : 0.4,
        top_k: 30,
        top_p: 0.9,
        num_thread: navigator.hardwareConcurrency ? Math.max(2, Math.min(16, navigator.hardwareConcurrency)) : 8,
      },
    }),
    [fastMode, maxBars, maxTokens, modelName],
  );

  const payload: RunAnalysisPayload = useMemo(
    () => ({
      symbol,
      timeframe,
      indicators: [],
      strategy,
      ...analysisOptions,
    }),
    [analysisOptions, symbol, timeframe, strategy],
  );

  const resetController = () => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
  };

  const handleAnalysisComplete = useCallback(
    (analysis: AnalysisResult | null) => {
      setState(prev => ({ ...prev, analysis, loading: false, cancelled: false }));
      onAnalysisComplete?.(analysis);
      if (analysis) {
        const next = analysisCounter + 1;
        setAnalysisCounter(next);
        localStorage.setItem('analysisCounter', String(next));
      }
    },
    [onAnalysisComplete, analysisCounter],
  );

  const handleError = useCallback(
    (message: string) => {
      setState(prev => ({ ...prev, error: message, loading: false }));
      onNotify?.({ message, status: 'error' });
    },
    [onNotify],
  );

  const runAnalysis = useCallback(async () => {
    if (!symbol || !timeframe || !strategy) {
      handleError('Symbol, timeframe, and strategy are required.');
      return;
    }

    if (abortControllerRef.current) {
      return;
    }

    const controller = new AbortController();
    abortControllerRef.current = controller;
    const abortSignal = controller.signal;
    timeoutRef.current = setTimeout(() => {
      controller.abort();
    }, 90_000);

    setState({ analysis: null, loading: true, error: null, cancelled: false });
    onAnalysisStart?.();

    try {
      const response = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: abortSignal,
      });

      if (!response.ok) {
        const text = await response.text();
        let message = text || `${response.status} ${response.statusText}`;
        try {
          const parsed = JSON.parse(text);
          if (parsed?.detail) {
            message = parsed.detail;
          }
        } catch {
          // ignore JSON parse errors
        }
        handleError(message);
        return;
      }

      const json = await response.json();
      handleAnalysisComplete(json?.analysis ?? null);
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') {
        setState(prev => ({ ...prev, loading: false, cancelled: true }));
        onAnalysisComplete?.(null);
        return;
      }
      handleError(error instanceof Error ? error.message : 'Failed to run analysis.');
    } finally {
      resetController();
    }
  }, [handleAnalysisComplete, handleError, onAnalysisStart, onAnalysisComplete, payload, symbol, timeframe, strategy]);

  const cancelAnalysis = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
  }, []);

  useEffect(() => () => resetController(), []);

  useEffect(() => {
    if (!selectedSignal) {
      return;
    }

    setState(prev => {
      const prior = prev.analysis ?? null;
      const extracted: AnalysisResult = {
        signal: selectedSignal.signal ?? prior?.signal ?? null,
        confidence: selectedSignal.confidence ?? prior?.confidence ?? null,
        rationale: selectedSignal.rationale ?? prior?.rationale ?? null,
        reasons:
          selectedSignal.reasons && selectedSignal.reasons.length > 0
            ? selectedSignal.reasons
            : prior?.reasons,
        entry: prior?.entry ?? null,
        sl:
          selectedSignal.sl ??
          selectedSignal.stop_loss ??
          prior?.sl ??
          prior?.stop_loss ??
          null,
        tp:
          selectedSignal.tp ??
          selectedSignal.take_profit ??
          prior?.tp ??
          prior?.take_profit ??
          null,
        model: prior?.model ?? null,
        generated_at: prior?.generated_at ?? null,
      };

      return { ...prev, analysis: extracted };
    });
  }, [selectedSignal]);

  const placeTrade = useCallback(async () => {
    const analysis = state.analysis;
    if (!analysis || !analysis.signal || !symbol) {
      handleError('No analysis available. Run AI Analysis first.');
      return;
    }

    const signal = (analysis.signal || '').toLowerCase();
    if (signal === 'no_trade' || signal === 'flat') {
      handleError('No trade suggested.');
      return;
    }

    try {
      const payload = {
        symbol,
        action: signal === 'long' ? 'BUY' : 'SELL',
        order_type: 'MARKET',
        volume: lotSize,
        stop_loss: analysis.sl ?? analysis.stop_loss ?? null,
        take_profit: analysis.tp ?? analysis.take_profit ?? null,
        rationale:
          analysis.rationale || (Array.isArray(analysis.reasons) ? analysis.reasons.filter(Boolean).join(' ') : ''),
      };

      const response = await fetch('/api/execute_trade', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || `${response.status} ${response.statusText}`);
      }

      await response.json();
      onNotify?.({ message: 'Trade placed successfully.', status: 'success' });
    } catch (error) {
      handleError(error instanceof Error ? error.message : 'Failed to place trade.');
    }
  }, [state.analysis, symbol, lotSize, handleError, onNotify]);

  useImperativeHandle(
    ref,
    () => ({
      runAnalysis,
      cancelAnalysis,
      placeTrade,
    }),
    [runAnalysis, cancelAnalysis, placeTrade],
  );

  const renderSummary = (analysis: AnalysisResult) => {
    const summary: string[] = [];
    const upperSignal = analysis.signal ? String(analysis.signal).toUpperCase() : null;
    if (upperSignal) {
      let line = `Model suggests ${upperSignal}`;
      if (typeof analysis.confidence === 'number') {
        line += ` with ${(analysis.confidence * 100).toFixed(0)}% confidence.`;
      }
      summary.push(line);
    }

    const targets: string[] = [];
    const formatVal = (value?: number | null) => {
      if (value == null) {
        return null;
      }
      const abs = Math.abs(value);
      const decimals = abs >= 100 ? 2 : abs >= 1 ? 4 : 6;
      return value.toFixed(decimals);
    };

    const entry = formatVal(analysis.entry);
    const tp = formatVal(analysis.tp ?? analysis.take_profit);
    const sl = formatVal(analysis.sl ?? analysis.stop_loss);

    if (entry) targets.push(`entry ${entry}`);
    if (tp) targets.push(`take-profit ${tp}`);
    if (sl) targets.push(`stop-loss ${sl}`);
    if (targets.length) {
      summary.push(`Targets ${targets.join(', ')}.`);
    }

    const reasons = analysis.reasons && analysis.reasons.length > 0 ? analysis.reasons : [];
    const rationale = analysis.rationale ? analysis.rationale.trim() : '';
    if (rationale) {
      reasons.push(rationale);
    }

    return (
      <div id="llmNarrative" className="llm-narrative">
        <div className="title">Decision Summary</div>
        {summary.length ? (
          <div className="summary">{summary.join(' ')}</div>
        ) : (
          <div className="muted summary">No key metrics were provided by the model.</div>
        )}
        <div className="title" style={{ marginTop: '8px' }}>
          Model Rationale
        </div>
        {reasons.length ? (
          <ul>
            {reasons.map((reason, index) => (
              <li key={index}>{reason}</li>
            ))}
          </ul>
        ) : (
          <div className="muted">No rationale supplied by the model.</div>
        )}
        <span className="meta">
          {[analysis.model ? `Model: ${analysis.model}` : null, analysis.generated_at ? `Generated: ${analysis.generated_at}` : null]
            .filter(Boolean)
            .join(' • ')}
        </span>
      </div>
    );
  };

  const renderAnalysis = () => {
    if (state.loading) {
      return <div className="muted">🔄 Running analysis…</div>;
    }
    if (state.cancelled) {
      return <div className="muted">⏹ Analysis cancelled.</div>;
    }
    if (state.error) {
      return <div id="llmError" className="muted" style={{ color: '#ff8a8a' }}>{state.error}</div>;
    }
    if (!state.analysis) {
      return <div className="muted">No analysis yet.</div>;
    }
    return (
      <>
        <pre id="llmOutput" style={{ color: '#9ef', margin: 0 }}>
          {JSON.stringify({ ...state.analysis, rationale: undefined, reasons: undefined }, null, 2)}
        </pre>
        {renderSummary(state.analysis)}
      </>
    );
  };

  return (
    <div className="ai-output">
      {!hideToolbar && (
        renderToolbar?.({
          runAnalysis,
          cancelAnalysis,
          placeTrade,
          loading: state.loading,
          analysis: state.analysis,
        }) ?? (
          <div className="analysis-toolbar">
            <button className="btn primary" type="button" onClick={runAnalysis} disabled={state.loading}>
              🧠 Run AI Analysis
            </button>
            <button
              className="btn warn"
              type="button"
              onClick={cancelAnalysis}
              style={{ display: state.loading ? 'inline-block' : 'none' }}
            >
              ✖ Cancel
            </button>
            <button className="btn success" type="button" onClick={placeTrade} disabled={state.loading}>
              ▶ Place Trade
            </button>
            {state.loading && <span className="muted" style={{ marginLeft: '8px' }}>Analyzing…</span>}
          </div>
        )
      )}
      <div className="llm-output-container">{renderAnalysis()}</div>
    </div>
  );
});

export default AIOutput;