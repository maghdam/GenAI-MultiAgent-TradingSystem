import { useEffect, useRef, useState } from 'react';
import { getV2TradeAudit, type V2TradeAudit } from '../services/api';
import { formatBackendLocalDateTime } from '../utils/datetime';

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

export default function Journal() {
  const [entries, setEntries] = useState<V2TradeAudit[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const pollTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const load = async () => {
    try {
      setError(null);
      const audits = await getV2TradeAudit(50);
      setEntries(audits);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load journal.');
      setEntries([]);
    } finally { setLoading(false); }
  };

  useEffect(() => {
    load();
    pollTimerRef.current = setInterval(load, 15000);
    return () => { if (pollTimerRef.current) clearInterval(pollTimerRef.current); };
  }, []);

  const renderBody = () => {
    if (loading) {
      return (
        <tr><td colSpan={8} style={{ textAlign: 'center', color: 'var(--ta-text-muted)', padding: '24px' }}>
          Loading journal…
        </td></tr>
      );
    }
    if (error) {
      return (
        <tr><td colSpan={8} style={{ textAlign: 'center', color: 'var(--ta-bear)', padding: '24px' }}>
          {error}
        </td></tr>
      );
    }
    if (!entries || entries.length === 0) {
      return (
        <tr><td colSpan={8} style={{ textAlign: 'center', color: 'var(--ta-text-muted)', padding: '24px' }}>
          No audit records yet
        </td></tr>
      );
    }

    return entries.map((trade) => {
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

      return (
        <tr key={trade.id}>
          <td>{formatTimestamp(closeTime)}</td>
          <td style={{ fontWeight: 600 }}>{trade.symbol}</td>
          <td>{trade.timeframe}</td>
          <td className={toneClass}>{trade.event_type}</td>
          <td>{trade.strategy}</td>
          <td>{trade.position_id ?? '–'}</td>
          <td className={pnlClass}>{formatPnl(realizedPnl)}</td>
          <td className="ta-cell--truncate">{trade.summary}</td>
        </tr>
      );
    });
  };

  return (
    <div className="ta-panel">
      <div className="ta-panel__header">
        <span className="ta-panel__title">Trade Journal</span>
        {entries && <span className="ta-panel__count">{entries.length}</span>}
      </div>
      <div className="ta-table-wrap" style={{ maxHeight: '340px', overflowY: 'auto' }}>
        <table className="ta-table">
          <thead>
            <tr>
              <th>Time</th>
              <th>Symbol</th>
              <th>TF</th>
              <th>Event</th>
              <th>Strategy</th>
              <th>Pos</th>
              <th>P&L</th>
              <th>Summary</th>
            </tr>
          </thead>
          <tbody>{renderBody()}</tbody>
        </table>
      </div>
    </div>
  );
}
