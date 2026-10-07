import test from 'node:test';
import assert from 'node:assert/strict';

import {
  SYSTEM_ACTIVITY_TABS,
  operatorIncidentHistory,
  readinessStats,
} from '../src/services/systemConsole.ts';

test('system readiness summary exposes attention count without duplicating every check', () => {
  const summary = readinessStats([
    { ok: true },
    { ok: true },
    { ok: false },
  ]);

  assert.deepEqual(summary, { passed: 2, total: 3, failed: 1 });
});

test('operator incident history hides routine informational records', () => {
  const rows = operatorIncidentHistory([
    { id: 1, level: 'info', code: 'signal_rejected' },
    { id: 2, level: 'warning', code: 'market_stale' },
    { id: 3, level: 'error', code: 'engine_loop_crash' },
  ]);

  assert.deepEqual(rows.map((row) => row.id), [2, 3]);
});

test('system activity uses one compact tab set for operational history', () => {
  assert.deepEqual(
    SYSTEM_ACTIVITY_TABS.map((tab) => tab.id),
    ['incidents', 'engine', 'trades', 'decisions', 'history'],
  );
});
