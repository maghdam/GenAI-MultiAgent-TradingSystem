import test from 'node:test';
import assert from 'node:assert/strict';

import {
  CTRADER_LOT_INPUT_MIN,
  CTRADER_LOT_INPUT_STEP,
  formatCTraderLotInput,
  normalizeCTraderLotInput,
} from '../src/services/lotSizeInput.ts';

test('cTrader lot entry uses the normal two-decimal presentation', () => {
  assert.equal(formatCTraderLotInput(0.01), '0.01');
  assert.equal(formatCTraderLotInput(0.1), '0.10');
  assert.equal(formatCTraderLotInput(1), '1.00');
});

test('legacy sub-cent lot values are normalized to the two-decimal UI minimum', () => {
  assert.equal(CTRADER_LOT_INPUT_MIN, 0.01);
  assert.equal(CTRADER_LOT_INPUT_STEP, 0.01);
  assert.equal(normalizeCTraderLotInput(0.0001), 0.01);
  assert.equal(formatCTraderLotInput(0.0001), '0.01');
});

test('lot entry normalization does not encode a broker-specific symbol minimum', () => {
  assert.equal(normalizeCTraderLotInput(0.01), 0.01);
  assert.equal(normalizeCTraderLotInput(0.1), 0.1);
  assert.equal(normalizeCTraderLotInput(1), 1);
});
