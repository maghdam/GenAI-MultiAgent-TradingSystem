import { useMemo, useState } from 'react';

import { toAgentSignal, type V2Analysis, type V2OrderIntent, type V2PaperPosition, type V2Status } from '../services/api';
import type { AgentSignal } from '../types';
import { formatBackendLocalTime } from '../utils/datetime';

interface SidePanelProps {
  status: V2Status | null;
  onSignalSelected?: (signal: AgentSignal) => void;
}

function formatTimestamp(value?: string | null): string {
  return formatBackendLocalTime(value);
}

function formatSignalStrength(value?: number): string {
  if (value == null || Number.isNaN(value)) return '–';
  return `${Math.round(value * 100)}%`;
}

function signalStrengthHelp(minimum?: number): string {
  const meaning = 'Deterministic strategy-strength score from the active strategy; it is not a calibrated win probability.';
  if (minimum == null || Number.isNaN(minimum)) return meaning;
  return `${meaning} Current execution threshold: ${formatSignalStrength(minimum)}.`;
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

function intentReason(intent: V2OrderIntent): string {
  if (intent.status === 'rejected' && intent.rationale?.trim()) {
    return intent.rationale.trim();
  }

  const detailCandidates = [
    intent.details?.error,
    intent.details?.protection_error,
    intent.details?.failsafe_close_error,
    intent.details?.reason,
  ];
  for (const candidate of detailCandidates) {
    if (typeof candidate === 'string' && candidate.trim()) return candidate.trim();
  }

  const listCandidates = [
    intent.details?.risk_reasons,
    intent.details?.quantity_reasons,
    intent.details?.sizing_reasons,
  ];
  for (const candidate of listCandidates) {
    if (Array.isArray(candidate)) {
      const reasons = candidate.filter((value): value is string => typeof value === 'string' && value.trim().length > 0);
      if (reasons.length) return reasons.join(' · ');
    }
  }

  return intent.status === 'failed' ? 'Execution failed; inspect the related incident for broker details.' : '';
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
          <div
            className="ta-confidence"
            title={signalStrengthHelp(status?.config.min_confidence)}
            aria-label={`Signal strength ${formatSignalStrength(a.confidence)}`}
          >
            <div className="ta-confidence__bar">
              <div className={`ta-confidence__fill ${fillClass}`} style={{ width: `${Math.round((a.confidence ?? 0) * 100)}%` }} />
            </div>
            <span>Strength {formatSignalStrength(a.confidence)}</span>
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
      const realized = p.realized_pnl || 0;
      const brokerBacked = p.broker_position_id != null;
      return (
        <div key={p.id} className="ta-position">
          <div className="ta-position__row">
            <span className={`ta-pill ${p.direction === 'long' ? 'ta-pill--long' : 'ta-pill--short'}`}>
              {p.direction === 'long' ? '↑' : '↓'} {p.direction}
            </span>
            <span className="ta-position__symbol">{p.symbol}</span>
            <span className="ta-position__detail">{p.timeframe}</span>
            <span className="ta-position__detail">qty {p.quantity.toFixed(2)}</span>
            <span className={`ta-position__pnl ${pnl >= 0 ? 'ta-position__pnl--profit' : 'ta-position__pnl--loss'}`}>
              {formatPnl(pnl)} {p.account_currency}
            </span>
          </div>
          <div className="ta-position__meta">
            <span>local #{p.id}</span>
            <span>{brokerBacked ? `cTrader #${p.broker_position_id}` : 'paper only'}</span>
            {brokerBacked ? (
              <>
                <span>broker entry {formatPrice(p.broker_entry_price)}</span>
                <span>tracked entry {formatPrice(p.entry_price)}</span>
                <span>SL {formatPrice(p.broker_stop_loss)} · TP {formatPrice(p.broker_take_profit)}</span>
                <span>protection {p.broker_protection_status ?? 'unavailable'}</span>
                <span>
                  {p.broker_last_synced_at
                    ? `synced ${formatTimestamp(p.broker_last_synced_at)}`
                    : `broker state ${p.broker_sync_status ?? 'unavailable'}`}
                </span>
              </>
            ) : (
              <span>entry {formatPrice(p.entry_price)}</span>
            )}
            {realized !== 0 && (
              <span>
                realized {formatPnl(realized)} {p.account_currency}
                {p.realized_pnl_source?.startsWith('ctrader_deal') ? ' · broker' : ''}
              </span>
            )}
          </div>
        </div>
      );
    });
  };

  /* ─── Order Intents ─── */
  const renderIntents = (intents: V2OrderIntent[]) => {
    if (!intents.length) return <div className="ta-panel__empty">No recent intents</div>;
    return intents.slice(0, 6).map((i) => {
      const statusClass =
        i.status === 'rejected' || i.status === 'failed' ? 'ta-pill--rejected'
        : i.status === 'executed' ? 'ta-pill--accepted'
        : 'ta-pill--info';
      const reason = intentReason(i);
      return (
        <div key={i.id} className="ta-intent">
          <div className="ta-intent__row">
            <span className={`ta-pill ${statusClass}`}>{i.status}</span>
            <span className="ta-intent__symbol">{i.symbol}</span>
            <span className="ta-intent__detail">{i.intent_type}</span>
            <span className="ta-intent__detail" title={signalStrengthHelp(status?.config.min_confidence)}>
              strength {formatSignalStrength(i.confidence)}
            </span>
            <span className="ta-intent__detail" style={{ marginLeft: 'auto' }}>{formatTimestamp(i.created_at)}</span>
          </div>
          <div className="ta-intent__meta">
            <span>{i.strategy}</span>
            {i.quantity != null && <span>qty {i.quantity}</span>}
          </div>
          {reason && (
            <div className="ta-intent__reason" title={reason}>
              {reason}
            </div>
          )}
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
