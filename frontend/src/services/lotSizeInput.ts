export const CTRADER_LOT_INPUT_MIN = 0.01;
export const CTRADER_LOT_INPUT_STEP = 0.01;

export function normalizeCTraderLotInput(value: number | null | undefined): number {
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric < CTRADER_LOT_INPUT_MIN) {
    return CTRADER_LOT_INPUT_MIN;
  }
  return Math.round((numeric + Number.EPSILON) * 100) / 100;
}

export function formatCTraderLotInput(value: number | null | undefined): string {
  return normalizeCTraderLotInput(value).toFixed(2);
}
