import type { AgentSignal } from '../types';
import type { AnalysisResult } from '../types/analysis';
import type { Candle, V2Analysis, V2ManualOrderRequest } from './api';
import { parseBackendUtc } from '../utils/datetime';

export interface ChartSignalMarker {
  id: string;
  time: number;
  position: 'aboveBar' | 'belowBar';
  shape: 'arrowUp' | 'arrowDown';
  color: string;
  text: string;
}

function isActionableSignal(signal: string): signal is 'long' | 'short' {
  return signal === 'long' || signal === 'short';
}

function signalTimestamp(value: string): number | null {
  const parsed = parseBackendUtc(value);
  return parsed ? Math.floor(parsed.getTime() / 1000) : null;
}

function candleTimeForSignal(candles: Candle[], timestamp: number): number | null {
  let matched: number | null = null;
  for (const candle of candles) {
    if (candle.time > timestamp) break;
    matched = candle.time;
  }
  return matched;
}

function markerId(analysis: V2Analysis): string {
  return [
    analysis.symbol,
    analysis.timeframe,
    analysis.strategy,
    analysis.signal,
    analysis.created_at,
  ].join(':');
}

export function buildChartSignalMarkers(
  analyses: V2Analysis[],
  candles: Candle[],
  symbol: string,
  timeframe: string,
  maxMarkers: number = 12,
): ChartSignalMarker[] {
  if (candles.length === 0 || maxMarkers <= 0) return [];

  const normalizedSymbol = symbol.trim().toUpperCase();
  const normalizedTimeframe = timeframe.trim().toUpperCase();

  const markers = analyses
    .filter((analysis) => (
      analysis.symbol.trim().toUpperCase() === normalizedSymbol
      && analysis.timeframe.trim().toUpperCase() === normalizedTimeframe
      && isActionableSignal(analysis.signal)
    ))
    .map((analysis): ChartSignalMarker | null => {
      const timestamp = signalTimestamp(analysis.created_at);
      if (timestamp == null) return null;
      const time = candleTimeForSignal(candles, timestamp);
      if (time == null) return null;

      const strength = Math.round(analysis.confidence * 100);
      const isLong = analysis.signal === 'long';
      return {
        id: markerId(analysis),
        time,
        position: isLong ? 'belowBar' : 'aboveBar',
        shape: isLong ? 'arrowUp' : 'arrowDown',
        color: isLong ? '#10b981' : '#ef4444',
        text: `${isLong ? 'BUY' : 'SELL'} ${strength}%`,
      };
    })
    .filter((marker): marker is ChartSignalMarker => marker !== null)
    .sort((a, b) => a.time - b.time);

  return markers.slice(-maxMarkers);
}

export function selectedSignalToAnalysis(signal: AgentSignal): AnalysisResult {
  return {
    signal: signal.signal,
    confidence: signal.confidence,
    rationale: signal.rationale,
    reasons: Array.isArray(signal.reasons) ? [...signal.reasons] : [],
    entry: signal.entry,
    sl: signal.sl,
    tp: signal.tp,
    model: 'saved-signal',
    generated_at: signal.ts > 0 ? new Date(signal.ts * 1000).toISOString() : null,
  };
}

export function buildSelectedSignalOrder(
  signal: AgentSignal,
  quantity: number,
): V2ManualOrderRequest | null {
  if (!isActionableSignal(signal.signal)) return null;
  if (!Number.isFinite(quantity) || quantity <= 0) return null;

  return {
    symbol: signal.symbol,
    timeframe: signal.timeframe,
    strategy: signal.strategy,
    signal: signal.signal,
    quantity,
    confidence: signal.confidence,
    entry_price: signal.entry,
    stop_loss: signal.sl,
    take_profit: signal.tp,
    reasons: Array.isArray(signal.reasons) ? [...signal.reasons] : [],
    rationale: signal.rationale,
  };
}
