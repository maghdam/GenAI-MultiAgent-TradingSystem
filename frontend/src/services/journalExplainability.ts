export interface JournalRealizedResult {
  amount: number;
  source: string;
  sourceLabel: string;
  brokerBacked: boolean;
  brokerPositionId: number | null;
}

function asFiniteNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function realizedSourceLabel(source: string): string {
  if (source === 'ctrader_deal_partial') return 'cTrader broker deals (partial close)';
  if (source.startsWith('ctrader_deal')) return 'cTrader broker deals';
  if (source === 'paper_estimate') return 'Paper estimate';
  if (!source) return 'Source not recorded';
  return source.replaceAll('_', ' ');
}

export function describeJournalRealizedResult(
  details: Record<string, unknown>,
): JournalRealizedResult | null {
  const amount = asFiniteNumber(details.realized_pnl);
  if (amount === null) return null;

  const source = typeof details.realized_pnl_source === 'string'
    ? details.realized_pnl_source.trim().toLowerCase()
    : '';
  const brokerPositionId = asFiniteNumber(details.broker_position_id);

  return {
    amount,
    source: source || 'unknown',
    sourceLabel: realizedSourceLabel(source),
    brokerBacked: source.startsWith('ctrader_deal'),
    brokerPositionId,
  };
}
