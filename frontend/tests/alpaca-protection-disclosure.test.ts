import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const tradeSetup = readFileSync(
  new URL('../src/components/TradeSettings.tsx', import.meta.url),
  'utf8',
);

test('Alpaca Paper rows display the missing broker-protection warning', () => {
  assert.match(tradeSetup, /config\.watchlist\.some\(\(row\) => row\.venue === 'alpaca'\)/);
  assert.match(tradeSetup, /role="alert"/);
  assert.match(tradeSetup, /aria-label="Alpaca Paper protection warning"/);
  assert.match(tradeSetup, /do not attach broker-native protective orders/);
  assert.match(tradeSetup, /supervised experiments, not unattended or broker-protected trading/);
});

test('Alpaca Paper enablement tooltip does not imply attached stop-loss orders', () => {
  assert.match(tradeSetup, /Automatic Alpaca Paper entries are blocked pending native protection and bracket reconciliation/);
  assert.match(tradeSetup, /Paper auto-trade/);
  assert.match(tradeSetup, /Automatic Alpaca Paper entries are temporarily blocked until broker-native protection/);
  assert.match(tradeSetup, /Existing strategy closes remain eligible/);
});
