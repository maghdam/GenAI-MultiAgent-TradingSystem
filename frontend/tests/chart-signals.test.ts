import test from 'node:test';
import assert from 'node:assert/strict';

import {
  buildChartSignalMarkers,
  buildSelectedSignalOrder,
  selectedSignalToAnalysis,
} from '../src/services/chartSignals.ts';
import type { AgentSignal } from '../src/types/index.ts';
import type { Candle, V2Analysis } from '../src/services/api.ts';

const candles: Candle[] = [
  { time: Date.UTC(2026, 9, 6, 12, 0) / 1000, open: 100, high: 102, low: 99, close: 101 },
  { time: Date.UTC(2026, 9, 6, 12, 5) / 1000, open: 101, high: 103, low: 100, close: 102 },
  { time: Date.UTC(2026, 9, 6, 12, 10) / 1000, open: 102, high: 104, low: 101, close: 103 },
];

function analysis(overrides: Partial<V2Analysis> = {}): V2Analysis {
  return {
    symbol: 'XAUUSD',
    timeframe: 'M5',
    strategy: 'sma_cross',
    signal: 'long',
    confidence: 0.72,
    entry_price: 102,
    stop_loss: 99,
    take_profit: 108,
    reasons: ['test reason'],
    context: {},
    created_at: '2026-10-06T12:07:30',
    ...overrides,
  };
}

function selectedSignal(overrides: Partial<AgentSignal> = {}): AgentSignal {
  return {
    ts: Date.UTC(2026, 9, 6, 12, 7, 30) / 1000,
    symbol: 'XAUUSD',
    timeframe: 'M5',
    signal: 'long',
    confidence: 0.72,
    rationale: 'exact saved rationale',
    reasons: ['exact reason'],
    entry: 102.25,
    sl: 99.5,
    tp: 108.75,
    strategy: 'sma_cross',
    ...overrides,
  };
}

test('actionable automatic signals become chart markers on their containing candle', () => {
  const markers = buildChartSignalMarkers([
    analysis(),
    analysis({ signal: 'short', confidence: 0.64, created_at: '2026-10-06T12:11:10' }),
    analysis({ signal: 'no_trade', created_at: '2026-10-06T12:12:00' }),
    analysis({ symbol: 'US30', created_at: '2026-10-06T12:08:00' }),
  ], candles, 'XAUUSD', 'M5');

  assert.equal(markers.length, 2);
  assert.deepEqual(
    markers.map(({ time, position, shape, text }) => ({ time, position, shape, text })),
    [
      {
        time: Date.UTC(2026, 9, 6, 12, 5) / 1000,
        position: 'belowBar',
        shape: 'arrowUp',
        text: 'BUY 72%',
      },
      {
        time: Date.UTC(2026, 9, 6, 12, 10) / 1000,
        position: 'aboveBar',
        shape: 'arrowDown',
        text: 'SELL 64%',
      },
    ],
  );
});

test('chart signal markers keep only the newest requested marker count', () => {
  const markers = buildChartSignalMarkers([
    analysis({ created_at: '2026-10-06T12:01:00' }),
    analysis({ created_at: '2026-10-06T12:06:00' }),
    analysis({ created_at: '2026-10-06T12:11:00' }),
  ], candles, 'XAUUSD', 'M5', 2);

  assert.deepEqual(markers.map((marker) => marker.time), [
    Date.UTC(2026, 9, 6, 12, 5) / 1000,
    Date.UTC(2026, 9, 6, 12, 10) / 1000,
  ]);
});

test('selected signal display uses the exact saved entry and protection snapshot', () => {
  const snapshot = selectedSignalToAnalysis(selectedSignal());

  assert.equal(snapshot.entry, 102.25);
  assert.equal(snapshot.sl, 99.5);
  assert.equal(snapshot.tp, 108.75);
  assert.equal(snapshot.confidence, 0.72);
  assert.deepEqual(snapshot.reasons, ['exact reason']);
});

test('trade-from-signal request uses exact signal context and operator-selected quantity', () => {
  const request = buildSelectedSignalOrder(
    selectedSignal({ strategy: 'breakout', timeframe: 'M15', signal: 'short' }),
    0.05,
  );

  assert.deepEqual(request, {
    symbol: 'XAUUSD',
    timeframe: 'M15',
    strategy: 'breakout',
    signal: 'short',
    quantity: 0.05,
    confidence: 0.72,
    entry_price: 102.25,
    stop_loss: 99.5,
    take_profit: 108.75,
    reasons: ['exact reason'],
    rationale: 'exact saved rationale',
  });
});

test('non-actionable signals cannot produce a trade-from-signal request', () => {
  assert.equal(buildSelectedSignalOrder(selectedSignal({ signal: 'no_trade' }), 0.05), null);
  assert.equal(buildSelectedSignalOrder(selectedSignal(), 0), null);
});
