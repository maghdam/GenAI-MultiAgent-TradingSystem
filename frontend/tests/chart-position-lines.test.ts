import test from 'node:test';
import assert from 'node:assert/strict';

import { buildChartPositionLevels } from '../src/services/chartPositionLines.ts';
import type { V2PaperPosition } from '../src/services/api.ts';

function position(overrides: Partial<V2PaperPosition> = {}): V2PaperPosition {
  return {
    id: 109,
    symbol: 'XAUUSD',
    timeframe: 'M5',
    strategy: 'sma_cross',
    direction: 'long',
    quantity: 0.05,
    status: 'open',
    entry_price: 4131.68,
    current_price: 4135,
    stop_loss: 4120,
    take_profit: 4160,
    opened_at: '2026-10-06T12:00:00',
    realized_pnl: 0,
    unrealized_pnl: 0,
    account_currency: 'CHF',
    cash_per_price_unit_per_lot: 1,
    instrument_spec_source: 'test',
    ...overrides,
  };
}

test('broker-backed chart levels use broker truth instead of tracked local levels', () => {
  const levels = buildChartPositionLevels([
    position({
      broker_position_id: 57783302,
      broker_entry_price: 4131.83,
      broker_stop_loss: 4147.83,
      broker_take_profit: 4160.52,
      broker_sync_status: 'id_match',
      broker_protection_status: 'protected',
    }),
  ], 'XAUUSD');

  assert.deepEqual(
    levels.map(({ kind, price, title, source }) => ({ kind, price, title, source })),
    [
      { kind: 'entry', price: 4131.83, title: 'BUY 0.05 #57783302', source: 'broker' },
      { kind: 'stop_loss', price: 4147.83, title: 'SL #57783302', source: 'broker' },
      { kind: 'take_profit', price: 4160.52, title: 'TP #57783302', source: 'broker' },
    ],
  );
});

test('broker-backed position does not fall back to tracked levels when broker truth is unavailable', () => {
  const levels = buildChartPositionLevels([
    position({
      broker_position_id: 57783302,
      broker_entry_price: null,
      broker_stop_loss: null,
      broker_take_profit: null,
      broker_sync_status: 'unavailable',
      broker_protection_status: 'unavailable',
    }),
  ], 'XAUUSD');

  assert.deepEqual(levels, []);
});

test('pure-paper position uses its tracked entry and protection levels', () => {
  const levels = buildChartPositionLevels([
    position({
      id: 77,
      broker_position_id: null,
      direction: 'short',
      quantity: 0.1,
      entry_price: 4200,
      stop_loss: 4210,
      take_profit: 4180,
    }),
  ], 'xauusd');

  assert.deepEqual(
    levels.map(({ kind, price, title, source }) => ({ kind, price, title, source })),
    [
      { kind: 'entry', price: 4200, title: 'PAPER SELL 0.1 #77', source: 'paper' },
      { kind: 'stop_loss', price: 4210, title: 'SL #77', source: 'paper' },
      { kind: 'take_profit', price: 4180, title: 'TP #77', source: 'paper' },
    ],
  );
});

test('chart levels include only open positions for the selected symbol', () => {
  const levels = buildChartPositionLevels([
    position({ id: 1, broker_position_id: null, symbol: 'XAUUSD', status: 'open' }),
    position({ id: 2, broker_position_id: null, symbol: 'US30', status: 'open' }),
    position({ id: 3, broker_position_id: null, symbol: 'XAUUSD', status: 'closed' }),
  ], 'XAUUSD');

  assert.equal(levels.length, 3);
  assert.ok(levels.every((level) => level.key.startsWith('paper:1:')));
});
