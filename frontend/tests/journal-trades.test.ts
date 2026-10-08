import test from 'node:test';
import assert from 'node:assert/strict';

import type { V2JournalExportRow } from '../src/services/api.ts';
import {
  filterJournalTrades,
  journalTradeLocalDateKey,
  journalTradeSource,
  summarizeJournalTrades,
} from '../src/services/journalTrades.ts';

const row = (
  overrides: Partial<V2JournalExportRow> = {},
): V2JournalExportRow => ({
  row_id: 'local_position:1',
  local_position_id: 1,
  broker_position_id: 5001,
  broker_identity_status: 'resolved',
  broker_identity_detail: '',
  broker_deal_ids: [6001],
  audit_event_ids: [7001],
  symbol: 'XAUUSD',
  timeframe: 'M5',
  strategy: 'sma_cross',
  direction: 'short',
  quantity: 0.01,
  execution_source: 'manual',
  opened_at_utc: '2026-10-07T07:00:00Z',
  closed_at_utc: '2026-10-07T07:25:00Z',
  entry_price: 4130.48,
  exit_price: 4125.25,
  account_currency: 'CHF',
  realized_pnl: 4.36,
  realized_pnl_basis: 'broker_deals',
  broker_deal_count: 1,
  close_reason: 'signal_close',
  ...overrides,
});

test('completed trade filtering uses actual closed trades rather than audit events', () => {
  const first = row();
  const second = row({
    row_id: 'local_position:2',
    local_position_id: 2,
    broker_position_id: null,
    broker_deal_ids: [],
    symbol: 'US30',
    strategy: 'breakout',
    realized_pnl: -1.59,
    realized_pnl_basis: 'paper_estimate',
    execution_source: 'auto',
  });

  assert.equal(journalTradeSource(first), 'broker');
  assert.equal(journalTradeSource(second), 'paper');
  assert.deepEqual(
    filterJournalTrades([first, second], {
      symbol: 'XAUUSD',
      strategy: 'all',
      date: '',
      source: 'broker',
    }).map((item) => item.local_position_id),
    [1],
  );
});

test('completed trade date filter follows the operator local date', () => {
  const trade = row();
  const localDate = journalTradeLocalDateKey(trade.closed_at_utc);

  assert.equal(
    filterJournalTrades([trade], {
      symbol: 'all',
      strategy: 'all',
      date: localDate,
      source: 'all',
    }).length,
    1,
  );
  assert.equal(
    filterJournalTrades([trade], {
      symbol: 'all',
      strategy: 'all',
      date: '1999-01-01',
      source: 'all',
    }).length,
    0,
  );
});

test('completed trade summary counts rows and never mixes account currencies', () => {
  const summary = summarizeJournalTrades([
    row({ realized_pnl: 4.36 }),
    row({ row_id: 'local_position:2', local_position_id: 2, realized_pnl: -20.39 }),
    row({
      row_id: 'local_position:3',
      local_position_id: 3,
      account_currency: 'USD',
      realized_pnl: 2.5,
    }),
  ]);

  assert.equal(summary.count, 3);
  assert.equal(summary.totals.length, 2);
  assert.equal(summary.totals[0]?.currency, 'CHF');
  assert.ok(Math.abs((summary.totals[0]?.realizedPnl ?? 0) - (-16.03)) < 1e-9);
  assert.deepEqual(summary.totals[1], { currency: 'USD', realizedPnl: 2.5 });
});


test('broker account total excludes paper simulation P&L', () => {
  const rows = [
    row({ realized_pnl: -2.44, account_currency: 'CHF' }),
    row({
      row_id: 'local_position:2',
      local_position_id: 2,
      broker_position_id: null,
      broker_deal_ids: [],
      realized_pnl_basis: 'paper_estimate',
      account_currency: 'USD',
      realized_pnl: 434.10,
    }),
  ];

  const allSummary = summarizeJournalTrades(rows);
  const brokerSummary = summarizeJournalTrades(
    rows.filter((item) => journalTradeSource(item) === 'broker'),
  );

  assert.equal(allSummary.count, 2);
  assert.deepEqual(brokerSummary.totals, [{ currency: 'CHF', realizedPnl: -2.44 }]);
});
