import type { V2JournalExportRow } from './api';
import { parseBackendUtc } from '../utils/datetime.ts';

export type JournalTradeSource = 'all' | 'broker' | 'paper';

export interface JournalTradeFilters {
  symbol: string;
  strategy: string;
  dateFrom: string;
  dateTo: string;
  source: JournalTradeSource;
}

export interface JournalTradePnlTotal {
  currency: string;
  realizedPnl: number;
}

export interface JournalTradeSummary {
  count: number;
  totals: JournalTradePnlTotal[];
}

export function journalTradeSource(row: V2JournalExportRow): Exclude<JournalTradeSource, 'all'> {
  return row.broker_position_id != null || row.realized_pnl_basis === 'broker_deals'
    ? 'broker'
    : 'paper';
}

export function journalTradeLocalDateKey(value: string): string {
  const date = parseBackendUtc(value);
  if (!date) return '';
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  return `${year}-${month}-${day}`;
}

export function filterJournalTrades(
  rows: V2JournalExportRow[],
  filters: JournalTradeFilters,
): V2JournalExportRow[] {
  return rows.filter((row) => {
    if (filters.symbol !== 'all' && row.symbol !== filters.symbol) return false;
    if (filters.strategy !== 'all' && row.strategy !== filters.strategy) return false;
    const localDate = journalTradeLocalDateKey(row.closed_at_utc);
    if (filters.dateFrom && localDate < filters.dateFrom) return false;
    if (filters.dateTo && localDate > filters.dateTo) return false;
    if (filters.source !== 'all' && journalTradeSource(row) !== filters.source) return false;
    return true;
  });
}

export function summarizeJournalTrades(rows: V2JournalExportRow[]): JournalTradeSummary {
  const totals = new Map<string, number>();
  rows.forEach((row) => {
    const currency = (row.account_currency || 'UNKNOWN').toUpperCase();
    totals.set(currency, (totals.get(currency) ?? 0) + row.realized_pnl);
  });

  return {
    count: rows.length,
    totals: [...totals.entries()]
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([currency, realizedPnl]) => ({ currency, realizedPnl })),
  };
}
