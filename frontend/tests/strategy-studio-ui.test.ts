import test from 'node:test';
import assert from 'node:assert/strict';

import {
  STUDIO_LIFECYCLE_STAGES,
  STUDIO_VALIDATION_OPTIONS,
  lifecycleStageState,
  studioBacktestResultTitle,
} from '../src/services/strategyStudioPresentation.ts';

test('strategy lifecycle stays in the governed progression order', () => {
  assert.deepEqual(
    STUDIO_LIFECYCLE_STAGES,
    ['draft', 'backtested', 'validated', 'paper', 'eligible'],
  );
});

test('strategy validation modes keep development and holdout visually distinct', () => {
  assert.deepEqual(
    STUDIO_VALIDATION_OPTIONS.map((option) => option.value),
    ['development_backtest', 'out_of_sample', 'regime', 'walk_forward'],
  );
});

test('lifecycle stage state distinguishes complete, current, and pending stages', () => {
  assert.equal(lifecycleStageState('validated', 'draft'), 'complete');
  assert.equal(lifecycleStageState('validated', 'validated'), 'current');
  assert.equal(lifecycleStageState('validated', 'paper'), 'pending');
  assert.equal(lifecycleStageState(null, 'draft'), 'pending');
});

test('latest result title uses the completed run rather than edited request controls', () => {
  const snapshot = {
    strategy: 'ema20_50_xauusd_m5_research',
    symbol: 'XAUUSD',
    timeframe: 'M5',
    'Total Return [%]': -6.8671,
    'Fetched Bars': 4089,
    'Selected Bars': 2862,
  };
  const runMeta = { symbol: 'XAUUSD', timeframe: 'M5', numBars: 5000 };
  assert.equal(
    studioBacktestResultTitle(snapshot, runMeta),
    'XAUUSD · M5 · 5000 requested · 4089 fetched · 2862 tested',
  );

  // Live controls changed to a new request after the run; the completed
  // snapshot and runMeta remain unchanged.
  const newControls = { symbol: 'US100', timeframe: 'H1', numBars: 200 };
  assert.equal(newControls.numBars, 200);
  assert.equal(
    studioBacktestResultTitle(snapshot, runMeta),
    'XAUUSD · M5 · 5000 requested · 4089 fetched · 2862 tested',
  );
});

test('legacy result title never invents retrieved or tested data depth', () => {
  assert.equal(
    studioBacktestResultTitle(
      { symbol: 'XAUUSD', timeframe: 'M5', 'Total Return [%]': 5.461 },
      { numBars: 1500 },
    ),
    'XAUUSD · M5 · 1500 requested',
  );
  assert.equal(
    studioBacktestResultTitle({ symbol: 'SPY', timeframe: 'D1', 'Total Return [%]': 1, 'Fetched Bars': 200 }),
    'SPY · D1 · 200 fetched',
  );
  assert.equal(studioBacktestResultTitle(null, null), 'No backtest result yet');
});
