import { useEffect, useRef, useState } from 'react';

interface TradeEntry {
  id: number;
  timestamp: string | null;
  symbol: string;
  direction: string;
  volume: number;
  entry_price: number | null;
  stop_loss: number | null;
  take_profit: number | null;
  rationale: string | null;
  exit_price?: number | null;
  pnl?: number | null;
}

function formatTimestamp(value: string | null): string {
  if (!value) {
    return 'N/A';
  }
  try {
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) {
      return value;
    }
    return date.toLocaleString('en-US', {
      year: '2-digit',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false,
    });
  } catch {
    return value;
  }
}

async function fetchTrades(limit: number): Promise<TradeEntry[]> {
  const response = await fetch(`/api/journal/trades?limit=${limit}`);
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Failed to fetch journal entries (${response.status})`);
  }
  return response.json();
}

export default function Journal() {
  const [entries, setEntries] = useState<TradeEntry[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const pollTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const load = async () => {
    try {
      setError(null);
      const trades = await fetchTrades(50);
      setEntries(trades);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load journal data.');
      setEntries([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
    pollTimerRef.current = setInterval(load, 15000);
    return () => {
      if (pollTimerRef.current) {
        clearInterval(pollTimerRef.current);
      }
    };
  }, []);

  const renderBody = () => {
    if (loading) {
      return (
        <tr>
          <td colSpan={8} className="muted">
            Loading journal…
          </td>
        </tr>
      );
    }

    if (error) {
      return (
        <tr>
          <td colSpan={8} className="muted">
            Error fetching journal data.
          </td>
        </tr>
      );
    }

    if (!entries || entries.length === 0) {
      return (
        <tr>
          <td colSpan={8} className="muted">
            No trades recorded yet.
          </td>
        </tr>
      );
    }

    return entries.map(trade => {
      const directionClass = trade.direction?.toLowerCase() === 'buy' ? 'good' : 'bad';
      const formatPrice = (value: number | null | undefined) =>
        value == null ? 'N/A' : value.toFixed(5);

      return (
        <tr key={trade.id ?? `${trade.symbol}-${trade.timestamp}`}>
          <td className="muted">{formatTimestamp(trade.timestamp)}</td>
          <td>{trade.symbol}</td>
          <td className={directionClass}>{trade.direction?.toUpperCase() ?? '—'}</td>
          <td>{trade.volume?.toFixed(2) ?? '0.00'}</td>
          <td>{formatPrice(trade.entry_price)}</td>
          <td>{formatPrice(trade.stop_loss)}</td>
          <td>{formatPrice(trade.take_profit)}</td>
          <td className="muted rationale">{trade.rationale ?? ''}</td>
        </tr>
      );
    });
  };

  return (
    <div className="journal-table-container">
      <table className="journal-table">
        <thead>
          <tr>
            <th>Timestamp</th>
            <th>Symbol</th>
            <th>Direction</th>
            <th>Volume</th>
            <th>Entry</th>
            <th>SL</th>
            <th>TP</th>
            <th>Rationale</th>
          </tr>
        </thead>
        <tbody>{renderBody()}</tbody>
      </table>
    </div>
  );
}