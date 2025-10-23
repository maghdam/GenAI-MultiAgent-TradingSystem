import { useEffect, useMemo, useRef, useState } from 'react';
import type { AgentSignal } from '../types';

interface AgentTask {
  symbol?: string;
  timeframe?: string;
  state?: string;
  last_signal?: string;
  last_signal_ts?: number;
  last_confidence?: number;
  last_bar_ts?: number;
  last_error_ts?: number;
  last_error?: string | null;
  next_poll_seconds?: number;
  poll_seconds?: number;
  configured_interval_seconds?: number;
  auto_trade?: boolean;
}

interface AgentStatus {
  enabled: boolean;
  running: boolean;
  watchlist: [string, string][];
  interval_sec?: number;
  min_confidence?: number;
  lot_size_lots?: number;
  trading_mode?: string;
  autotrade?: boolean;
  strategy?: string;
  tasks: AgentTask[];
}



interface PositionRow {
  direction: string;
  symbol_name: string;
  volume_lots: number;
  entry_price: number;
}

interface PendingOrderRow {
  side: string;
  symbol: string;
  type: string;
  price: number;
  volume: number;
}

interface SidePanelProps {
  onSignalSelected?: (signal: AgentSignal) => void;
  onAgentStatus?: (status: AgentStatus | null) => void;
}

function formatTimestamp(epochSeconds?: number): string {
  if (!epochSeconds) {
    return '—';
  }
  try {
    return new Date(epochSeconds * 1000).toLocaleTimeString();
  } catch {
    return '—';
  }
}

function formatConfidence(value?: number): string {
  if (value == null || Number.isNaN(value)) {
    return '—';
  }
  return `${Math.round(value * 100)}%`;
}

async function fetchJSON<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `${response.status} ${response.statusText}`);
  }
  return response.json();
}

export default function SidePanel({ onSignalSelected, onAgentStatus }: SidePanelProps) {
  const [signals, setSignals] = useState<AgentSignal[] | null>(null);
  const [positions, setPositions] = useState<PositionRow[] | null>(null);
  const [pendingOrders, setPendingOrders] = useState<PendingOrderRow[] | null>(null);
  const [agentStatus, setAgentStatus] = useState<AgentStatus | null>(null);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const timersRef = useRef<ReturnType<typeof setInterval>[]>([]);

  const recordError = (key: string, message: string) => {
    setErrors(prev => ({ ...prev, [key]: message }));
  };

  const clearError = (key: string) => {
    setErrors(prev => {
      if (!(key in prev)) {
        return prev;
      }
      const next = { ...prev };
      delete next[key];
      return next;
    });
  };

  const loadSignals = async () => {
    try {
      const data = await fetchJSON<AgentSignal[]>('/api/agent/signals?n=10');
      setSignals(Array.isArray(data) ? data : []);
      clearError('signals');
    } catch (error) {
      recordError('signals', error instanceof Error ? error.message : 'Failed to load signals.');
      setSignals([]);
    }
  };

  const loadPositions = async () => {
    try {
      const data = await fetchJSON<PositionRow[]>('/api/open_positions');
      setPositions(Array.isArray(data) ? data : []);
      clearError('positions');
    } catch (error) {
      recordError('positions', error instanceof Error ? error.message : 'Failed to load positions.');
      setPositions([]);
    }
  };

  const loadPendingOrders = async () => {
    try {
      const data = await fetchJSON<PendingOrderRow[]>('/api/pending_orders');
      setPendingOrders(Array.isArray(data) ? data : []);
      clearError('pending');
    } catch (error) {
      recordError('pending', error instanceof Error ? error.message : 'Failed to load pending orders.');
      setPendingOrders([]);
    }
  };

  const loadAgentStatus = async () => {
    try {
      const status = await fetchJSON<AgentStatus>('/api/agent/status');
      setAgentStatus(status);
      onAgentStatus?.(status);
      clearError('status');
    } catch (error) {
      recordError('status', error instanceof Error ? error.message : 'Failed to load agent status.');
      setAgentStatus(null);
      onAgentStatus?.(null);
    }
  };

  useEffect(() => {
    const loaders = [
      loadSignals,
      loadPositions,
      loadPendingOrders,
      loadAgentStatus,
    ];

    loaders.forEach(loader => loader());

    timersRef.current = [
      setInterval(loadSignals, 5000),
      setInterval(loadPositions, 7000),
      setInterval(loadPendingOrders, 9000),
      setInterval(loadAgentStatus, 4000),
    ];

    const handleVisibility = () => {
      if (document.hidden) {
        timersRef.current.forEach(interval => clearInterval(interval));
        timersRef.current = [];
      } else {
        loaders.forEach(loader => loader());
        timersRef.current = [
          setInterval(loadSignals, 5000),
          setInterval(loadPositions, 7000),
          setInterval(loadPendingOrders, 9000),
          setInterval(loadAgentStatus, 4000),
        ];
      }
    };

    document.addEventListener('visibilitychange', handleVisibility);

    return () => {
      document.removeEventListener('visibilitychange', handleVisibility);
      timersRef.current.forEach(interval => clearInterval(interval));
      timersRef.current = [];
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);



  const agentSummary = useMemo(() => {
    if (!agentStatus) {
      return 'Agent: OFF (unavailable) • Watchlist: —';
    }
    const statusLabel = agentStatus.enabled ? 'ON' : 'OFF';
    const running = agentStatus.running ? 'running' : 'idle';
    const watchlist = (agentStatus.watchlist || [])
      .map(pair => pair.join(':').toUpperCase())
      .join(', ')
      || '—';
    return `Agent: ${statusLabel} (${running}) • Watchlist: ${watchlist}`;
  }, [agentStatus]);

  const renderSignals = () => {
    if (!signals) {
      return <div className="muted">Loading…</div>;
    }
    if (signals.length === 0) {
      return <div className="muted">No signals yet.</div>;
    }
    return signals.map((signal, index) => {
      const timestamp = signal.ts ? new Date(signal.ts * 1000).toLocaleTimeString() : '—';
      const lower = (signal.signal || '').toLowerCase();
      const pillClass = lower === 'long' ? 'good' : lower === 'short' ? 'bad' : '';
      const strategyTag = signal.strategy ? ` • ${signal.strategy.toUpperCase()}` : '';
      const reason = signal.rationale
        ? signal.rationale
        : Array.isArray(signal.reasons)
          ? signal.reasons.join('\n')
          : '';

      return (
        <div
          key={`${signal.symbol}-${signal.timeframe}-${index}`}
          className="sig"
          onClick={() => onSignalSelected?.(signal)}
          role="button"
          tabIndex={0}
          onKeyDown={event => {
            if (event.key === 'Enter' || event.key === ' ') {
              onSignalSelected?.(signal);
            }
          }}
        >
          <span className="muted">{timestamp}</span>
          <span className={`pill ${pillClass}`}>{lower || '—'}</span>
          <span className="muted">{formatConfidence(signal.confidence)}</span>
          <div className="muted">{`${signal.symbol ?? ''}:${signal.timeframe ?? ''}${strategyTag}`}</div>
          {reason && (
            <div className="muted" style={{ gridColumn: '1 / -1', fontSize: '12px', opacity: 0.85 }}>
              {reason}
            </div>
          )}
        </div>
      );
    });
  };

  const renderPositions = () => {
    if (!positions) {
      return <div className="muted">Loading…</div>;
    }
    if (positions.length === 0) {
      return <div className="muted">None.</div>;
    }
    return positions.map(position => (
      <div key={`${position.symbol_name}-${position.direction}`} className="row">
        <div className={`pill ${position.direction?.toLowerCase() === 'buy' ? 'good' : 'bad'}`}>
          {position.direction}
        </div>
        <div>{position.symbol_name}</div>
        <div className="muted">vol: {position.volume_lots.toFixed(2)}</div>
        <div className="muted">entry: {position.entry_price.toFixed(5)}</div>
      </div>
    ));
  };

  const renderPendingOrders = () => {
    if (!pendingOrders) {
      return <div className="muted">Loading…</div>;
    }
    if (pendingOrders.length === 0) {
      return <div className="muted">None.</div>;
    }
    return pendingOrders.map(order => (
      <div key={`${order.symbol}-${order.price}-${order.side}`} className="row">
        <div className={`pill ${order.side?.toLowerCase() === 'buy' ? 'good' : 'bad'}`}>{order.side}</div>
        <div>{order.symbol}</div>
        <div className="muted">{order.type}</div>
        <div className="muted">@ {order.price.toFixed(5)}</div>
        <div className="muted">vol: {order.volume.toFixed(2)}</div>
      </div>
    ));
  };

  const renderAgentTasks = () => {
    if (!agentStatus || !agentStatus.tasks || agentStatus.tasks.length === 0) {
      return <div className="muted">No active tasks.</div>;
    }
    return agentStatus.tasks.map(task => {
      const meta: string[] = [];
      if (task.last_signal) {
        meta.push(`Signal: ${task.last_signal}`);
      }
      if (task.last_confidence != null) {
        meta.push(`Conf: ${formatConfidence(task.last_confidence)}`);
      }
      if (task.next_poll_seconds != null) {
        meta.push(`Next poll: ${task.next_poll_seconds}s`);
      }
      if (task.poll_seconds != null) {
        meta.push(`Poll: ${task.poll_seconds}s`);
      }
      if (task.configured_interval_seconds != null) {
        meta.push(`Cfg interval: ${task.configured_interval_seconds}s`);
      }
      if (task.auto_trade) {
        meta.push('Autotrade');
      }
      if (task.last_error) {
        meta.push(`Error: ${task.last_error}`);
      }

      return (
        <div key={`${task.symbol}-${task.timeframe}-${task.state}`} style={{ marginBottom: '10px' }}>
          <div className="row">
            <div className="pill">{(task.symbol || '').toUpperCase()}</div>
            <div className="muted">{(task.timeframe || '').toUpperCase()}</div>
            <div className="muted">State: {task.state ?? 'unknown'}</div>
            <div className="muted">Last bar: {formatTimestamp(task.last_bar_ts)}</div>
            <div className="muted">Last signal: {formatTimestamp(task.last_signal_ts)}</div>
          </div>
          <div className="muted" style={{ fontSize: '12px', margin: '-6px 0 10px 8px' }}>
            {meta.length ? meta.join(' • ') : '—'}
          </div>
        </div>
      );
    });
  };

  return (
    <>
      <div className="box">
        <div style={{ fontWeight: 600, marginBottom: '10px' }}>Navigation</div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '5px' }}>
          <a href="/strategy-studio" className="nav-link" target="_blank" rel="noopener noreferrer">Strategy Studio</a>
        </div>
      </div>
      <div className="box">
        <div style={{ fontWeight: 600 }}>Recent Signals</div>
        <div className="list">
          {errors.signals && <div className="muted">{errors.signals}</div>}
          {renderSignals()}
        </div>
        <div className="muted">Latest 10 signals emitted by the background agent.</div>
      </div>

      <div className="box">
        <div style={{ fontWeight: 600 }}>Open Positions</div>
        <div className="list">
          {errors.positions && <div className="muted">{errors.positions}</div>}
          {renderPositions()}
        </div>
      </div>

      <div className="box">
        <div style={{ fontWeight: 600 }}>Pending Orders</div>
        <div className="list">
          {errors.pending && <div className="muted">{errors.pending}</div>}
          {renderPendingOrders()}
        </div>
      </div>

      <div className="box">
        <div style={{ fontWeight: 600 }}>Agent Tasks</div>
        <div className="list">
          {errors.status && <div className="muted">{errors.status}</div>}
          {renderAgentTasks()}
        </div>
        <div className="muted" style={{ fontSize: '12px', marginTop: '6px' }}>
          {agentSummary}
        </div>
      </div>
    </>
  );
}
