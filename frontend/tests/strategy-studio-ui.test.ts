import test from 'node:test';
import assert from 'node:assert/strict';

import {
  STUDIO_LIFECYCLE_STAGES,
  STUDIO_VALIDATION_OPTIONS,
  lifecycleStageState,
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
