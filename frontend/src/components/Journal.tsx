import { Fragment, useEffect, useMemo, useRef, useState } from 'react';
import {
  getV2JournalExport,
  getV2OrderIntents,
  getV2TradeAudit,
  type V2JournalExportRow,
  type V2OrderIntent,
  type V2TradeAudit,
} from '../services/api';
import {
  filterJournalTrades,
  journalTradeSource,
  summarizeJournalTrades,
} from '../services/journalTrades';
import { formatBackendLocalDateTime, parseBackendUtc } from '../utils/datetime';
import { describeJournalRealizedResult } from '../services/journalExplainability';

type JournalCategory = 'trades' | 'all' | 'execution' | 'rejected' | 'protection';
type AuditJournalCategory = Exclude<JournalCategory, 'trades'>;
type JournalSource = 'all' | 'broker' | 'paper';

function formatTimestamp(value: string | null): string {
  return formatBackendLocalDateTime(value, {
    month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit', second: '2-digit',
    hour12: false,
  });
}

function formatPnl(value: unknown): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '–';
  const prefix = value > 0 ? '+' : '';
  return `${prefix}${value.toFixed(2)}`;
}

function formatPrice(value: number | null | undefined): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '–';
  return new Intl.NumberFormat(undefined, {
    minimumFractionDigits: 2,
    maximumFractionDigits: 5,
  }).format(value);
}

function formatQuantity(value: number): string {
  if (!Number.isFinite(value)) return '–';
  return value.toFixed(2);
}

function executionSourceLabel(value: V2JournalExportRow['execution_source']): string {
  if (value === 'manual') return 'Manual confirm';
  if (value === 'auto') return 'Automated';
  return 'Origin unknown';
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function localDateKey(value: string): string {
  const date = parseBackendUtc(value);
  if (!date) return '';
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  return `${year}-${month}-${day}`;
}

function auditSource(trade: V2TradeAudit, intent?: V2OrderIntent): 'broker' | 'paper' {
  const eventType = (trade.event_type || '').toLowerCase();
  if (eventType.startsWith('ctrader_')) return 'broker';

  const details = [trade.details, intent?.details].filter(isRecord);
  for (const detail of details) {
    if (detail.execution_mode === 'ctrader_demo' || detail.execution_mode === 'ctrader_live') return 'broker';
    if (
      isRecord(detail.broker_order)
      || isRecord(detail.broker_close)
      || isRecord(detail.broker_protection)
      || isRecord(detail.protection)
      || typeof detail.broker_position_id === 'number'
    ) {
      return 'broker';
    }
  }
  return 'paper';
}

function matchesCategory(
  trade: V2TradeAudit,
  intent: V2OrderIntent | undefined,
  category: AuditJournalCategory,
): boolean {
  if (category === 'all') return true;

  const eventType = (trade.event_type || '').toLowerCase();
  if (category === 'rejected') {
    return eventType.includes('reject') || intent?.status === 'rejected';
  }

  if (category === 'protection') {
    return (
      eventType.includes('protection')
      || eventType.includes('protective')
      || eventType.includes('unprotected')
    );
  }

  const executionTokens = [
    'position_opened',
    'position_closed',
    'position_updated',
    'signal_update',
    'signal_flip',
    'partial_close',
    'protective_exit',
    'close_reconciled',
    'order_executed',
  ];
  return intent?.status === 'executed' || executionTokens.some((token) => eventType.includes(token));
}

function collectReasons(trade: V2TradeAudit, intent?: V2OrderIntent): string[] {
  const values: string[] = [];
  const add = (value: unknown) => {
    if (typeof value === 'string' && value.trim()) values.push(value.trim());
    if (Array.isArray(value)) {
      value.forEach((item) => {
        if (typeof item === 'string' && item.trim()) values.push(item.trim());
      });
    }
  };

  add(trade.details?.reasons);
  add(intent?.rationale);
  add(intent?.details?.risk_reasons);
  add(intent?.details?.quantity_reasons);
  add(intent?.details?.sizing_reasons);

  return [...new Set(values)];
}

function brokerDetails(trade: V2TradeAudit, intent?: V2OrderIntent): Record<string, unknown> | null {
  const merged: Record<string, unknown> = {};
  const add = (label: string, value: unknown) => {
    if (value !== undefined && value !== null && value !== '') merged[label] = value;
  };

  add('execution_mode', intent?.details?.execution_mode);
  add('broker_position_id', trade.details?.broker_position_id);
  add('broker_order', intent?.details?.broker_order ?? trade.details?.broker_order);
  add('broker_protection', intent?.details?.broker_protection ?? trade.details?.broker_protection);
  add('protection', trade.details?.protection);
  add('broker_close', trade.details?.broker_close ?? intent?.details?.broker_close);
  add('error', trade.details?.error ?? intent?.details?.error);

  return Object.keys(merged).length ? merged : null;
}

function prettyDetails(value: Record<string, unknown>): string {
  return JSON.stringify(value, null, 2);
}

export default function Journal() {
  const [trades, setTrades] = useState<V2JournalExportRow[] | null>(null);
  const [entries, setEntries] = useState<V2TradeAudit[] | null>(null);
  const [intents, setIntents] = useState<V2OrderIntent[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [category, setCategory] = useState<JournalCategory>('trades');
  const [symbol, setSymbol] = useState('all');
  const [strategy, setStrategy] = useState('all');
  const [dateFrom, setDateFrom] = useState('');
  const [dateTo, setDateTo] = useState('');
  const [source, setSource] = useState<JournalSource>('broker');
  const [expandedId, setExpandedId] = useState<number | null>(null);
  const pollTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const load = async () => {
    try {
      setError(null);
      const [completedTrades, audits, orderIntents] = await Promise.all([
        getV2JournalExport({ allTime: true }),
        getV2TradeAudit(100),
        getV2OrderIntents(100),
      ]);
      setTrades(completedTrades.rows);
      setEntries(audits);
      setIntents(orderIntents);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load journal.');
      setTrades([]);
      setEntries([]);
      setIntents([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
    pollTimerRef.current = setInterval(load, 15000);
    return () => { if (pollTimerRef.current) clearInterval(pollTimerRef.current); };
  }, []);

  const intentById = useMemo(() => {
    const index = new Map<number, V2OrderIntent>();
    intents.forEach((intent) => index.set(intent.id, intent));
    return index;
  }, [intents]);

  const symbols = useMemo(
    () => [...new Set([
      ...(trades ?? []).map((trade) => trade.symbol),
      ...(entries ?? []).map((entry) => entry.symbol),
    ].filter(Boolean))].sort(),
    [trades, entries],
  );
  const strategies = useMemo(
    () => [...new Set([
      ...(trades ?? []).map((trade) => trade.strategy),
      ...(entries ?? []).map((entry) => entry.strategy),
    ].filter(Boolean))].sort(),
    [trades, entries],
  );

  const filteredTrades = useMemo(
    () => filterJournalTrades(trades ?? [], { symbol, strategy, dateFrom, dateTo, source }),
    [trades, symbol, strategy, dateFrom, dateTo, source],
  );
  const tradeSummary = useMemo(() => summarizeJournalTrades(filteredTrades), [filteredTrades]);
  const brokerSummary = useMemo(
    () => summarizeJournalTrades(filteredTrades.filter((trade) => journalTradeSource(trade) === 'broker')),
    [filteredTrades],
  );

  const filteredEntries = useMemo(() => {
    if (category === 'trades') return [];
    return (entries ?? []).filter((trade) => {
      const intent = trade.intent_id != null ? intentById.get(trade.intent_id) : undefined;
      if (!matchesCategory(trade, intent, category)) return false;
      if (symbol !== 'all' && trade.symbol !== symbol) return false;
      if (strategy !== 'all' && trade.strategy !== strategy) return false;
      const auditDate = localDateKey(trade.created_at);
      if (dateFrom && auditDate < dateFrom) return false;
      if (dateTo && auditDate > dateTo) return false;
      if (source !== 'all' && auditSource(trade, intent) !== source) return false;
      return true;
    });
  }, [entries, intentById, category, symbol, strategy, dateFrom, dateTo, source]);

  const clearDateRange = () => {
    setDateFrom('');
    setDateTo('');
  };

  const resetFilters = () => {
    setCategory('trades');
    setSymbol('all');
    setStrategy('all');
    clearDateRange();
    setSource('broker');
  };

  const renderTradeBody = () => {
    if (loading) {
      return <tr><td colSpan={9} className="ta-journal__message">Loading completed trades…</td></tr>;
    }
    if (error) {
      return (
        <tr><td colSpan={9} className="ta-journal__message ta-journal__message--error">
          {error}
        </td></tr>
      );
    }
    if (!trades || trades.length === 0) {
      return <tr><td colSpan={9} className="ta-journal__message">No completed trades recorded yet</td></tr>;
    }
    if (filteredTrades.length === 0) {
      return <tr><td colSpan={9} className="ta-journal__message">No completed trades match the active filters</td></tr>;
    }

    return filteredTrades.map((trade) => {
      const pnlClass = trade.realized_pnl >= 0 ? 'ta-cell--good' : 'ta-cell--bad';
      const executionVenue = journalTradeSource(trade) === 'broker' ? 'Broker' : 'Paper';
      return (
        <tr key={trade.row_id} title={trade.broker_identity_detail || undefined}>
          <td>{formatTimestamp(trade.closed_at_utc)}</td>
          <td style={{ fontWeight: 600 }}>{trade.symbol}</td>
          <td>{trade.direction.toUpperCase()}</td>
          <td>{formatQuantity(trade.quantity)}</td>
          <td>{formatPrice(trade.entry_price)}</td>
          <td>{formatPrice(trade.exit_price)}</td>
          <td
            className={pnlClass}
            title={trade.realized_pnl_basis === 'broker_deals' ? 'Authoritative cTrader broker-deal P&L' : 'Local paper estimate'}
          >
            {formatPnl(trade.realized_pnl)} {trade.account_currency}
          </td>
          <td>
            <span>{executionVenue}</span>
            <span className="ta-journal__subtle"> · {executionSourceLabel(trade.execution_source)}</span>
          </td>
          <td className="ta-cell--truncate" title={trade.strategy}>{trade.strategy}</td>
        </tr>
      );
    });
  };

  const renderAuditBody = () => {
    if (loading) {
      return <tr><td colSpan={8} className="ta-journal__message">Loading audit events…</td></tr>;
    }
    if (error) {
      return (
        <tr><td colSpan={8} className="ta-journal__message ta-journal__message--error">
          {error}
        </td></tr>
      );
    }
    if (!entries || entries.length === 0) {
      return <tr><td colSpan={8} className="ta-journal__message">No audit records yet</td></tr>;
    }
    if (filteredEntries.length === 0) {
      return <tr><td colSpan={8} className="ta-journal__message">No audit records match the active filters</td></tr>;
    }

    return filteredEntries.map((trade) => {
      const eventType = (trade.event_type || '').toLowerCase();
      const toneClass =
        eventType.includes('open') || eventType.includes('accepted') ? 'ta-cell--good'
        : eventType.includes('close') || eventType.includes('reject') ? 'ta-cell--bad' : '';
      const realizedPnl = trade.details?.realized_pnl;
      const pnlClass = typeof realizedPnl === 'number'
        ? realizedPnl >= 0 ? 'ta-cell--good' : 'ta-cell--bad'
        : '';

      const closeTime =
        typeof trade.details?.broker_closed_at === 'string'
          ? trade.details.broker_closed_at
          : typeof trade.details?.closed_at === 'string'
            ? trade.details.closed_at
            : trade.created_at;

      const intent = trade.intent_id != null ? intentById.get(trade.intent_id) : undefined;
      const mode = auditSource(trade, intent);
      const reasons = collectReasons(trade, intent);
      const broker = brokerDetails(trade, intent);
      const realizedResult = describeJournalRealizedResult(trade.details);
      const expanded = expandedId === trade.id;

      return (
        <Fragment key={trade.id}>
          <tr
            className={`ta-journal-row ${expanded ? 'ta-journal-row--expanded' : ''}`}
            onClick={() => setExpandedId(expanded ? null : trade.id)}
            title="Click to expand audit details"
          >
            <td>{formatTimestamp(closeTime)}</td>
            <td style={{ fontWeight: 600 }}>{trade.symbol}</td>
            <td>{trade.timeframe}</td>
            <td className={toneClass}>{trade.event_type}</td>
            <td>{trade.strategy}</td>
            <td>{trade.position_id ?? '–'}</td>
            <td className={pnlClass}>{formatPnl(realizedPnl)}</td>
            <td className="ta-cell--truncate">
              <span className="ta-journal__summary">
                <span>{trade.summary}</span>
                <span className="ta-journal__chevron" aria-hidden="true">{expanded ? '▾' : '›'}</span>
              </span>
            </td>
          </tr>
          {expanded && (
            <tr className="ta-journal-detail-row">
              <td colSpan={8}>
                <div className="ta-journal-detail">
                  <div className="ta-journal-detail__meta">
                    <span>audit #{trade.id}</span>
                    <span>{trade.intent_id != null ? `intent #${trade.intent_id}` : 'no linked intent'}</span>
                    <span>source {mode}</span>
                    <span>{intent ? `${intent.intent_type} · ${intent.status}` : 'audit-only event'}</span>
                  </div>

                  <div className="ta-journal-detail__grid">
                    <section>
                      <h4>Intent / risk reasons</h4>
                      {reasons.length > 0 ? (
                        <ul>{reasons.map((reason) => <li key={reason}>{reason}</li>)}</ul>
                      ) : (
                        <div className="ta-journal-detail__empty">No linked risk or intent reasons.</div>
                      )}
                    </section>

                    <section>
                      <h4>Realized result</h4>
                      {realizedResult ? (
                        <ul>
                          <li>Amount: {formatPnl(realizedResult.amount)}</li>
                          <li>Source: {realizedResult.sourceLabel}</li>
                          {realizedResult.brokerPositionId != null && (
                            <li>Broker position: #{realizedResult.brokerPositionId}</li>
                          )}
                        </ul>
                      ) : (
                        <div className="ta-journal-detail__empty">No realized P&amp;L recorded for this event.</div>
                      )}
                    </section>

                    <section>
                      <h4>Broker details</h4>
                      {broker ? (
                        <pre>{prettyDetails(broker)}</pre>
                      ) : (
                        <div className="ta-journal-detail__empty">No broker details for this paper/audit event.</div>
                      )}
                    </section>
                  </div>

                  <details className="ta-journal-detail__raw">
                    <summary>Raw audit details</summary>
                    <pre>{prettyDetails(trade.details)}</pre>
                  </details>
                </div>
              </td>
            </tr>
          )}
        </Fragment>
      );
    });
  };

  const isTradeView = category === 'trades';
  const total = isTradeView ? (trades?.length ?? 0) : (entries?.length ?? 0);
  const visible = loading || error
    ? total
    : isTradeView ? filteredTrades.length : filteredEntries.length;

  return (
    <div className="ta-panel">
      <div className="ta-panel__header">
        <span className="ta-panel__title">Trade Journal</span>
        {(trades || entries) && <span className="ta-panel__count">{visible}/{total}</span>}
      </div>

      <div className="ta-journal-filters">
        <label>
          <span>View</span>
          <select value={category} onChange={(event) => setCategory(event.target.value as JournalCategory)}>
            <option value="trades">Completed trades</option>
            <option value="all">All audit events</option>
            <option value="execution">Execution events</option>
            <option value="rejected">Rejected signals</option>
            <option value="protection">Protection incidents</option>
          </select>
        </label>

        <label>
          <span>Symbol</span>
          <select value={symbol} onChange={(event) => setSymbol(event.target.value)}>
            <option value="all">All symbols</option>
            {symbols.map((value) => <option value={value} key={value}>{value}</option>)}
          </select>
        </label>

        <label>
          <span>Strategy</span>
          <select value={strategy} onChange={(event) => setStrategy(event.target.value)}>
            <option value="all">All strategies</option>
            {strategies.map((value) => <option value={value} key={value}>{value}</option>)}
          </select>
        </label>

        <label>
          <span>From</span>
          <input
            type="date"
            value={dateFrom}
            max={dateTo || undefined}
            onChange={(event) => setDateFrom(event.target.value)}
          />
        </label>

        <label>
          <span>To</span>
          <input
            type="date"
            value={dateTo}
            min={dateFrom || undefined}
            onChange={(event) => setDateTo(event.target.value)}
          />
        </label>

        <label>
          <span>Source</span>
          <select value={source} onChange={(event) => setSource(event.target.value as JournalSource)}>
            <option value="all">Broker + paper records</option>
            <option value="broker">Broker account</option>
            <option value="paper">Paper simulation</option>
          </select>
        </label>

        <button
          type="button"
          onClick={clearDateRange}
          disabled={!dateFrom && !dateTo}
          aria-pressed={!dateFrom && !dateTo}
          title="Clear From/To and show the full recorded history"
        >
          All history
        </button>

        <button type="button" onClick={resetFilters}>Reset</button>
      </div>

      {isTradeView && (
        <div className="ta-journal-totals" aria-label="Filtered completed trade summary">
          <span><strong>{tradeSummary.count}</strong> completed {tradeSummary.count === 1 ? 'trade' : 'trades'}</span>
          {source === 'paper' ? (
            tradeSummary.totals.length === 1 ? (
              <span
                className={tradeSummary.totals[0].realizedPnl >= 0 ? 'ta-cell--good' : 'ta-cell--bad'}
              >
                Paper realized <strong>{formatPnl(tradeSummary.totals[0].realizedPnl)} {tradeSummary.totals[0].currency}</strong>
              </span>
            ) : (
              <span className="ta-journal__subtle">Paper realized unavailable across mixed paper currencies</span>
            )
          ) : brokerSummary.totals.length === 1 ? (
            <span
              className={brokerSummary.totals[0].realizedPnl >= 0 ? 'ta-cell--good' : 'ta-cell--bad'}
            >
              Account realized <strong>{formatPnl(brokerSummary.totals[0].realizedPnl)} {brokerSummary.totals[0].currency}</strong>
            </span>
          ) : brokerSummary.totals.length > 1 ? (
            <span className="ta-journal__subtle">Account realized unavailable across multiple broker account currencies</span>
          ) : (
            <span className="ta-journal__subtle">Account realized –</span>
          )}
          {source === 'all' && (
            <span className="ta-journal__subtle">
              {brokerSummary.count} broker {brokerSummary.count === 1 ? 'trade' : 'trades'} · paper simulation excluded from account realized
            </span>
          )}
          <span className="ta-journal__subtle">All history is the default · From/To narrows the range</span>
        </div>
      )}

      <div className="ta-table-wrap" style={{ maxHeight: '420px', overflowY: 'auto' }}>
        <table className="ta-table">
          <thead>
            {isTradeView ? (
              <tr>
                <th>Close time</th>
                <th>Symbol</th>
                <th>Side</th>
                <th>Qty</th>
                <th>Entry</th>
                <th>Exit</th>
                <th>Realized P&amp;L</th>
                <th>Execution</th>
                <th>Strategy</th>
              </tr>
            ) : (
              <tr>
                <th>Time</th>
                <th>Symbol</th>
                <th>TF</th>
                <th>Event</th>
                <th>Strategy</th>
                <th>Pos</th>
                <th>P&amp;L</th>
                <th>Summary</th>
              </tr>
            )}
          </thead>
          <tbody>{isTradeView ? renderTradeBody() : renderAuditBody()}</tbody>
        </table>
      </div>
    </div>
  );
}
