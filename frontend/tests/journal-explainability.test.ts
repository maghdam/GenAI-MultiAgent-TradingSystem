import test from 'node:test';
import assert from 'node:assert/strict';

import { describeJournalRealizedResult } from '../src/services/journalExplainability.ts';

test('journal explains broker-backed realized P&L provenance', () => {
  const result = describeJournalRealizedResult({
    realized_pnl: -0.73,
    realized_pnl_source: 'ctrader_deal',
    broker_position_id: 700009,
  });

  assert.deepEqual(result, {
    amount: -0.73,
    source: 'ctrader_deal',
    sourceLabel: 'cTrader broker deals',
    brokerBacked: true,
    brokerPositionId: 700009,
  });
});

test('journal distinguishes partial broker reconciliation from a paper estimate', () => {
  const partial = describeJournalRealizedResult({
    realized_pnl: 4.25,
    realized_pnl_source: 'ctrader_deal_partial',
    broker_position_id: 12345,
  });
  const paper = describeJournalRealizedResult({
    realized_pnl: 1.5,
    realized_pnl_source: 'paper_estimate',
  });

  assert.equal(partial?.sourceLabel, 'cTrader broker deals (partial close)');
  assert.equal(partial?.brokerBacked, true);
  assert.equal(paper?.sourceLabel, 'Paper estimate');
  assert.equal(paper?.brokerBacked, false);
  assert.equal(paper?.brokerPositionId, null);
});

test('journal does not invent a realized result when none is persisted', () => {
  assert.equal(describeJournalRealizedResult({ realized_pnl_source: 'ctrader_deal' }), null);
});
