import test from 'node:test';
import assert from 'node:assert/strict';

import { loadFrontendRestartSnapshot } from '../src/services/frontendRestart.ts';

test('frontend restart keeps successful read state when the other read fails', async () => {
  const status = { version: 'test-status' };
  const snapshot = await loadFrontendRestartSnapshot({
    getStatus: async () => status as never,
    getStrategies: async () => {
      throw new Error('strategy endpoint temporarily unavailable');
    },
  });

  assert.equal(snapshot.status, status);
  assert.equal(snapshot.strategies, null);
  assert.equal(snapshot.complete, false);
  assert.deepEqual(snapshot.errors, [
    'strategies: strategy endpoint temporarily unavailable',
  ]);
});

test('frontend restart recovers deterministically on the next read without invoking writes', async () => {
  let statusCalls = 0;
  let strategyCalls = 0;
  let writeCalls = 0;

  const readers = {
    getStatus: async () => {
      statusCalls += 1;
      if (statusCalls === 1) {
        throw new Error('status endpoint restarting');
      }
      return { version: 'recovered-status' } as never;
    },
    getStrategies: async () => {
      strategyCalls += 1;
      return [{ key: 'sma_cross' }] as never;
    },
    setConfig: async () => {
      writeCalls += 1;
    },
  };

  const first = await loadFrontendRestartSnapshot(readers);
  const second = await loadFrontendRestartSnapshot(readers);

  assert.equal(first.status, null);
  assert.equal(first.complete, false);
  assert.equal(second.complete, true);
  assert.equal((second.status as { version: string }).version, 'recovered-status');
  assert.deepEqual(second.strategies, [{ key: 'sma_cross' }]);
  assert.equal(statusCalls, 2);
  assert.equal(strategyCalls, 2);
  assert.equal(writeCalls, 0);
});

test('frontend restart captures synchronous read failures instead of aborting the whole snapshot', async () => {
  const snapshot = await loadFrontendRestartSnapshot({
    getStatus: () => {
      throw new Error('synchronous status failure');
    },
    getStrategies: async () => [{ key: 'breakout' }] as never,
  });

  assert.equal(snapshot.status, null);
  assert.deepEqual(snapshot.strategies, [{ key: 'breakout' }]);
  assert.equal(snapshot.complete, false);
  assert.deepEqual(snapshot.errors, ['status: synchronous status failure']);
});
