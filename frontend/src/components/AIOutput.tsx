import { forwardRef, useCallback, useEffect, useImperativeHandle, useMemo, useRef, useState } from 'react';
import type { JSX } from 'react';

import { analyzeV2Dashboard, placeV2ManualOrder } from '../services/api';
import type { AgentSignal } from '../types';
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

const initialState: AnalysisState = { analysis: null, loading: false, error: null, cancelled: false };

function toAnalysisResult(payload: {
  signal?: string; confidence?: number; reasons?: string[];
  entry_price?: number | null; stop_loss?: number | null; take_profit?: number | null;
  created_at?: string;
}): AnalysisResult {
  const reasons = Array.isArray(payload.reasons) ? payload.reasons : [];
  return {
    signal: payload.signal ?? null,
    confidence: typeof payload.confidence === 'number' ? payload.confidence : null,
    rationale: reasons.join(' '),
    reasons,
    entry: payload.entry_price ?? null,
    sl: payload.stop_loss ?? null,
    tp: payload.take_profit ?? null,
    model: 'v2-deterministic',
    generated_at: payload.created_at ?? null,
  };
}

function formatPrice(value?: number | null): string {
  if (value == null || !Number.isFinite(value)) return '–';
  const abs = Math.abs(value);
  const d = abs >= 100 ? 2 : abs >= 1 ? 4 : 6;
  return value.toFixed(d);
}

const AIOutput = forwardRef<AIOutputHandle, AIOutputProps>(function AIOutput(
  {
    symbol, timeframe, strategy, lotSize, fastMode, maxBars, maxTokens, modelName,
    onAnalysisStart, onAnalysisComplete, onNotify, selectedSignal, renderToolbar, hideToolbar = false,
  },
  ref,
) {
  const [state, setState] = useState<AnalysisState>(initialState);
  const abortControllerRef = useRef<AbortController | null>(null);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [analysisCounter, setAnalysisCounter] = useState(() => Number(localStorage.getItem('analysisCounter')) || 0);

  const analysisOptions: AnalysisOptions = useMemo(() => ({
    fast: fastMode, max_bars: maxBars, max_tokens: maxTokens,
    model: modelName.trim() ? modelName.trim() : null,
    options: { num_predict: maxTokens, temperature: fastMode ? 0.2 : 0.4, top_k: 30, top_p: 0.9,
      num_thread: navigator.hardwareConcurrency ? Math.max(2, Math.min(16, navigator.hardwareConcurrency)) : 8,
    },
  }), [fastMode, maxBars, maxTokens, modelName]);

  const payload: RunAnalysisPayload = useMemo(() => ({
    symbol, timeframe, indicators: [], strategy, ...analysisOptions,
  }), [analysisOptions, symbol, timeframe, strategy]);

  const resetController = () => {
    if (timeoutRef.current) { clearTimeout(timeoutRef.current); timeoutRef.current = null; }
    if (abortControllerRef.current) { abortControllerRef.current.abort(); abortControllerRef.current = null; }
  };

  const handleAnalysisComplete = useCallback(
    (analysis: AnalysisResult | null) => {
      const cloned: AnalysisResult | null = analysis
        ? { ...analysis, reasons: Array.isArray(analysis.reasons) ? [...analysis.reasons] : [] }
        : null;
      setState((prev) => ({ ...prev, analysis: cloned, loading: false, cancelled: false }));
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
      setState((prev) => ({ ...prev, error: message, loading: false }));
      onNotify?.({ message, status: 'error' });
    },
    [onNotify],
  );

  const runAnalysis = useCallback(async () => {
    if (!symbol || !timeframe || !strategy) { handleError('Symbol, timeframe, and strategy are required.'); return; }
    if (abortControllerRef.current) return;
    const controller = new AbortController();
    abortControllerRef.current = controller;
    const abortSignal = controller.signal;
    timeoutRef.current = setTimeout(() => { controller.abort(); }, 90_000);
    setState({ analysis: null, loading: true, error: null, cancelled: false });
    onAnalysisStart?.();

    try {
      const result = await analyzeV2Dashboard({
        symbol: payload.symbol, timeframe: payload.timeframe, strategy: payload.strategy, num_bars: payload.max_bars,
      }, abortSignal);
      if (abortSignal.aborted) { setState((p) => ({ ...p, loading: false, cancelled: true })); onAnalysisComplete?.(null); return; }
      handleAnalysisComplete(toAnalysisResult(result));
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') {
        setState((p) => ({ ...p, loading: false, cancelled: true })); onAnalysisComplete?.(null); return;
      }
      handleError(error instanceof Error ? error.message : 'Failed to run analysis.');
    } finally { resetController(); }
  }, [handleAnalysisComplete, handleError, onAnalysisStart, onAnalysisComplete, payload, symbol, timeframe, strategy]);

  const cancelAnalysis = useCallback(() => {
    if (abortControllerRef.current) { abortControllerRef.current.abort(); abortControllerRef.current = null; }
  }, []);

  useEffect(() => () => resetController(), []);

  useEffect(() => {
    if (!selectedSignal) return;
    setState((prev) => {
      const prior = prev.analysis ?? null;
      const priorReasons = Array.isArray(prior?.reasons) ? [...(prior!.reasons as string[])] : [];
      const selReasons = Array.isArray(selectedSignal.reasons) ? [...selectedSignal.reasons] : [];
      const extracted: AnalysisResult = {
        signal: selectedSignal.signal ?? prior?.signal ?? null,
        confidence: selectedSignal.confidence ?? prior?.confidence ?? null,
        rationale: selectedSignal.rationale ?? prior?.rationale ?? null,
        reasons: selReasons.length > 0 ? selReasons : priorReasons,
        entry: prior?.entry ?? null,
        sl: selectedSignal.sl ?? prior?.sl ?? null,
        tp: selectedSignal.tp ?? prior?.tp ?? null,
        model: prior?.model ?? null,
        generated_at: prior?.generated_at ?? null,
      };
      return { ...prev, analysis: extracted };
    });
  }, [selectedSignal]);

  const placeTrade = useCallback(async () => {
    const analysis = state.analysis;
    if (!analysis || !analysis.signal || !symbol) { handleError('No analysis available.'); return; }
    const signal = (analysis.signal || '').toLowerCase();
    if (signal === 'no_trade' || signal === 'flat') { handleError('No trade suggested.'); return; }

    try {
      const p = {
        symbol, timeframe, strategy, signal: signal as 'long' | 'short', quantity: lotSize,
        confidence: typeof analysis.confidence === 'number' ? analysis.confidence : 1,
        entry_price: analysis.entry ?? null, stop_loss: analysis.sl ?? null, take_profit: analysis.tp ?? null,
        reasons: Array.isArray(analysis.reasons) ? analysis.reasons.filter(Boolean) : [],
        rationale: analysis.rationale || '',
      };
      const response = await placeV2ManualOrder(p);
      onNotify?.({
        message: response.status === 'executed' ? `Order accepted: ${response.summary}.` : `Order rejected: ${response.summary}.`,
        status: response.status === 'executed' ? 'success' : 'error',
      });
    } catch (error) {
      handleError(error instanceof Error ? error.message : 'Failed to place trade.');
    }
  }, [state.analysis, symbol, timeframe, strategy, lotSize, handleError, onNotify]);

  useImperativeHandle(ref, () => ({ runAnalysis, cancelAnalysis, placeTrade }), [runAnalysis, cancelAnalysis, placeTrade]);

  /* ─── Render ─── */
  const renderAnalysis = () => {
    const a = state.analysis;

    if (state.loading) {
      return (
        <div className="ta-panel__body" style={{ display: 'flex', alignItems: 'center', gap: '12px', padding: '28px' }}>
          <div className="ta-spinner" />
          <span style={{ color: 'var(--ta-text-muted)' }}>Running analysis on {symbol} {timeframe}…</span>
        </div>
      );
    }
    if (state.cancelled) {
      return <div className="ta-panel__empty">Analysis cancelled</div>;
    }
    if (state.error) {
      return <div className="ta-panel__empty" style={{ color: 'var(--ta-bear)' }}>{state.error}</div>;
    }
    if (!a) {
      return <div className="ta-panel__empty">Click ⚡ Analyze to run a strategy analysis</div>;
    }

    const upperSignal = a.signal ? String(a.signal).toUpperCase() : 'NO TRADE';
    const signalClass = a.signal === 'long' ? 'ta-analysis__signal--long' : a.signal === 'short' ? 'ta-analysis__signal--short' : 'ta-analysis__signal--neutral';
    const confPct = typeof a.confidence === 'number' ? Math.round(a.confidence * 100) : null;
    const reasonsBase = Array.isArray(a.reasons) ? a.reasons : [];
    const rationale = a.rationale ? a.rationale.trim() : '';
    const composed = rationale ? [...reasonsBase, rationale] : [...reasonsBase];
    const reasons = Array.from(new Set(composed.filter(Boolean)));

    return (
      <div className="ta-analysis">
        {/* Header */}
        <div className="ta-analysis__header">
          <div className={`ta-analysis__signal ${signalClass}`}>{upperSignal}</div>
          <span className={`ta-pill ta-pill--lg ${a.signal === 'long' ? 'ta-pill--long' : a.signal === 'short' ? 'ta-pill--short' : 'ta-pill--no_trade'}`}>
            {a.signal === 'long' ? '↑ Bullish' : a.signal === 'short' ? '↓ Bearish' : '— Neutral'}
          </span>
          {confPct !== null && (
            <div className="ta-analysis__confidence">
              <div className="ta-analysis__confidence-value">{confPct}%</div>
              <div className="ta-analysis__confidence-label">confidence</div>
            </div>
          )}
        </div>

        {/* Metrics */}
        <div className="ta-analysis__metrics">
          <div className="ta-metric">
            <div className="ta-metric__label">Entry</div>
            <div className="ta-metric__value">{formatPrice(a.entry)}</div>
          </div>
          <div className="ta-metric">
            <div className="ta-metric__label">Stop Loss</div>
            <div className="ta-metric__value" style={{ color: 'var(--ta-bear)' }}>{formatPrice(a.sl)}</div>
          </div>
          <div className="ta-metric">
            <div className="ta-metric__label">Take Profit</div>
            <div className="ta-metric__value" style={{ color: 'var(--ta-bull)' }}>{formatPrice(a.tp)}</div>
          </div>
        </div>

        {/* Confidence bar */}
        {confPct !== null && (
          <div className="ta-confidence" style={{ marginBottom: '16px' }}>
            <div className="ta-confidence__bar">
              <div
                className={`ta-confidence__fill ${a.signal === 'long' ? 'ta-confidence__fill--bull' : a.signal === 'short' ? 'ta-confidence__fill--bear' : ''}`}
                style={{ width: `${confPct}%` }}
              />
            </div>
            <span>{confPct}%</span>
          </div>
        )}

        {/* Reasons */}
        {reasons.length > 0 && (
          <div className="ta-analysis__reasons">
            <div className="ta-analysis__reasons-title">Analysis Rationale</div>
            <ul>
              {reasons.map((r, i) => <li key={i}>{r}</li>)}
            </ul>
          </div>
        )}

        <div className="ta-analysis__meta">
          {[a.model ? `Model: ${a.model}` : null, a.generated_at ? `Generated: ${a.generated_at}` : null]
            .filter(Boolean).join(' · ')}
        </div>
      </div>
    );
  };

  return (
    <div className="ta-panel">
      <div className="ta-panel__header">
        <span className="ta-panel__title">Analysis Output</span>
        {state.analysis?.signal && (
          <span className={`ta-pill ${state.analysis.signal === 'long' ? 'ta-pill--long' : state.analysis.signal === 'short' ? 'ta-pill--short' : 'ta-pill--no_trade'}`}>
            {state.analysis.signal}
          </span>
        )}
      </div>
      {!hideToolbar && (
        renderToolbar?.({ runAnalysis, cancelAnalysis, placeTrade, loading: state.loading, analysis: state.analysis }) ?? (
          <div style={{ display: 'flex', gap: '8px', padding: '12px 18px', borderBottom: '1px solid var(--ta-border-dim)' }}>
            <button className="ta-btn ta-btn--primary ta-btn--sm" type="button" onClick={runAnalysis} disabled={state.loading}>
              ⚡ Analyze
            </button>
            {state.loading && (
              <button className="ta-btn ta-btn--danger ta-btn--sm" type="button" onClick={cancelAnalysis}>✕ Cancel</button>
            )}
            <button className="ta-btn ta-btn--success ta-btn--sm" type="button" onClick={placeTrade} disabled={state.loading}>
              ↗ Place Trade
            </button>
          </div>
        )
      )}
      {renderAnalysis()}
    </div>
  );
});

export default AIOutput;
