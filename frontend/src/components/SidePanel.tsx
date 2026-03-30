import { useMemo, useState } from 'react';

import { toAgentSignal, type V2Analysis, type V2OrderIntent, type V2PaperPosition, type V2Status } from '../services/api';
import type { AgentSignal } from '../types';

interface SidePanelProps {
  status: V2Status | null;
  onSignalSelected?: (signal: AgentSignal) => void;
}

function formatTimestamp(value?: string | null): string {
  if (!value) return '–';
  try { return new Date(value).toLocaleTimeString(); } catch { return '–'; }
}

function formatConfidence(value?: number): string {
  if (value == null || Number.isNaN(value)) return '–';
  return `${Math.round(value * 100)}%`;
}

function formatPrice(value?: number | null): string {
  if (value == null || !Number.isFinite(value)) return '–';
  return value.toFixed(3);
}

function formatPnl(value?: number | null): string {
  if (value == null || !Number.isFinite(value)) return '–';
  const prefix = value > 0 ? '+' : '';
  return `${prefix}${value.toFixed(2)}`;
}

/* Collapsible section */
function Section({
  title,
  count,
  defaultOpen = true,
  children,
}: {
  title: string;
  count?: number;
  defaultOpen?: boolean;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <>
      <div className="ta-section-header" onClick={() => setOpen((o) => !o)}>
        <span className="ta-section-header__title">{title}</span>
        {count !== undefined && <span className="ta-section-header__count">{count}</span>}
        <span className={`ta-section-header__chevron${open ? ' ta-section-header__chevron--open' : ''}`}>▼</span>
      </div>
      {open && children}
    </>
  );
}

export default function SidePanel({ status, onSignalSelected }: SidePanelProps) {
  const runtimeSummary = useMemo(() => {
    if (!status) return 'Loading…';
    return [
      status.config.enabled ? 'Engine ON' : 'Engine OFF',
      status.runtime.loop_active ? 'Scanning' : 'Idle',
      `${status.runtime.active_watchlist.length} watched`,
      `${status.paper_positions.length} open`,
    ].join(' · ');
  }, [status]);

  /* ─── Signals ─── */
  const renderSignals = (analyses: V2Analysis[]) => {
    if (!analyses.length) return <div className="ta-panel__empty">No analyses yet</div>;
    return analyses.map((a) => {
      const pillClass =
        a.signal === 'long' ? 'ta-pill--long' : a.signal === 'short' ? 'ta-pill--short' : 'ta-pill--no_trade';
      const fillClass = a.signal === 'long' ? 'ta-confidence__fill--bull' : a.signal === 'short' ? 'ta-confidence__fill--bear' : '';
      return (
        <div
          key={`${a.symbol}-${a.timeframe}-${a.created_at}`}
          className="ta-signal"
          onClick={() => onSignalSelected?.(toAgentSignal(a))}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') onSignalSelected?.(toAgentSignal(a)); }}
        >
          <div className="ta-signal__row">
            <span className={`ta-pill ${pillClass}`}>{a.signal}</span>
            <span className="ta-signal__pair">{a.symbol}</span>
            <span className="ta-signal__strategy">{a.strategy}</span>
            <span className="ta-signal__meta" style={{ marginLeft: 'auto' }}>{formatTimestamp(a.created_at)}</span>
          </div>
          <div className="ta-confidence">
            <div className="ta-confidence__bar">
              <div className={`ta-confidence__fill ${fillClass}`} style={{ width: `${Math.round((a.confidence ?? 0) * 100)}%` }} />
            </div>
            <span>{formatConfidence(a.confidence)}</span>
          </div>
          {a.reasons?.length > 0 && (
            <div className="ta-signal__reasons">{a.reasons.join(' · ')}</div>
          )}
        </div>
      );
    });
  };

  /* ─── Positions ─── */
  const renderPositions = (positions: V2PaperPosition[]) => {
    if (!positions.length) return <div className="ta-panel__empty">No open positions</div>;
    return positions.map((p) => {
      const pnl = p.unrealized_pnl || 0;
      return (
        <div key={p.id} className="ta-position">
          <span className={`ta-pill ${p.direction === 'long' ? 'ta-pill--long' : 'ta-pill--short'}`}>
            {p.direction === 'long' ? '↑' : '↓'} {p.direction}
          </span>
          <span className="ta-position__symbol">{p.symbol}</span>
          <span className="ta-position__detail">{p.timeframe}</span>
          <span className="ta-position__detail">qty {p.quantity.toFixed(2)}</span>
          <span className="ta-position__detail">@ {formatPrice(p.entry_price)}</span>
          <span className={`ta-position__pnl ${pnl >= 0 ? 'ta-position__pnl--profit' : 'ta-position__pnl--loss'}`}>
            {formatPnl(pnl)}
          </span>
        </div>
      );
    });
  };

  /* ─── Order Intents ─── */
  const renderIntents = (intents: V2OrderIntent[]) => {
    if (!intents.length) return <div className="ta-panel__empty">No recent intents</div>;
    return intents.slice(0, 6).map((i) => {
      const statusClass =
        i.status === 'rejected' ? 'ta-pill--rejected'
        : i.status === 'executed' ? 'ta-pill--accepted'
        : 'ta-pill--info';
      return (
        <div key={i.id} className="ta-intent">
          <span className={`ta-pill ${statusClass}`}>{i.status}</span>
          <span className="ta-intent__symbol">{i.symbol}</span>
          <span className="ta-intent__detail">{i.intent_type}</span>
          <span className="ta-intent__detail" style={{ marginLeft: 'auto' }}>{i.strategy}</span>
        </div>
      );
    });
  };

  /* ─── Incidents ─── */
  const renderIncidents = (items: V2Status['recent_incidents']) => {
    if (!items.length) return <div className="ta-panel__empty">No incidents</div>;
    return items.slice(0, 5).map((inc) => (
      <div key={inc.id} className={`ta-incident ta-incident--${inc.level}`}>
        <div className="ta-incident__head">
          <span className={`ta-pill ta-pill--${inc.level === 'error' ? 'rejected' : inc.level === 'warning' ? 'warning' : 'info'}`}>
            {inc.level}
          </span>
          <span className="ta-incident__code">{inc.code}</span>
          <span className="ta-incident__time">{formatTimestamp(inc.created_at)}</span>
        </div>
        <div className="ta-incident__message">{inc.message}</div>
      </div>
    ));
  };

  return (
    <>
      {/* Signals panel */}
      <div className="ta-panel" style={{ flex: '1 1 0', minHeight: 0, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        <Section title="Signals" count={status?.recent_analyses.length} defaultOpen>
          <div className="ta-panel__body--scroll">
            {renderSignals(status?.recent_analyses || [])}
          </div>
        </Section>

        <Section title="Positions" count={status?.paper_positions.length} defaultOpen>
          <div className="ta-panel__body--scroll" style={{ maxHeight: '160px' }}>
            {renderPositions(status?.paper_positions || [])}
          </div>
        </Section>

        <Section title="Intents" count={status?.recent_order_intents.length} defaultOpen={false}>
          <div className="ta-panel__body--scroll" style={{ maxHeight: '180px' }}>
            {renderIntents(status?.recent_order_intents || [])}
          </div>
        </Section>

        <Section title="Incidents" count={status?.recent_incidents.length} defaultOpen={false}>
          <div className="ta-panel__body--scroll" style={{ maxHeight: '180px' }}>
            {renderIncidents(status?.recent_incidents || [])}
          </div>
        </Section>

        <div className="ta-runtime">{runtimeSummary}</div>
      </div>
    </>
  );
}
