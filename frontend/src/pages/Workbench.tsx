import { useEffect, useMemo, useState, type ChangeEvent } from 'react';

import {
  analyzeV2Strategy,
  getV2Symbols,
  getV2Status,
  reconcileV2Engine,
  recoverV2Engine,
  scanV2Engine,
  setV2Config,
  startV2Engine,
  stopV2Engine,
  type V2Analysis,
  type V2Config,
  type V2OrderIntent,
  type V2PaperPosition,
  type V2Status,
  type V2TradeAudit,
  type V2WatchlistItem,
} from '../services/api';

const TIMEFRAMES = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1'];

type FormState = {
  symbol: string;
  timeframe: string;
  strategy: string;
  numBars: number;
};

const DEFAULT_FORM: FormState = {
  symbol: 'XAUUSD',
  timeframe: 'M5',
  strategy: 'sma_cross',
  numBars: 500,
};

const DEFAULT_WATCH_ITEM: V2WatchlistItem = {
  symbol: 'XAUUSD',
  timeframe: 'M5',
  strategy: 'sma_cross',
  enabled: true,
  params: {},
};

function formatTimestamp(value?: string | null) {
  if (!value) {
    return '-';
  }
  return new Date(value).toLocaleString();
}

function formatPrice(value?: number | null) {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return '-';
  }
  return value.toFixed(3);
}

function formatSigned(value?: number | null) {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return '-';
  }
  const prefix = value > 0 ? '+' : '';
  return `${prefix}${value.toFixed(2)}`;
}

function positionSummary(position: V2PaperPosition) {
  return [
    `Entry ${formatPrice(position.entry_price)}`,
    `Mark ${formatPrice(position.current_price)}`,
    `UPnL ${formatSigned(position.unrealized_pnl)}`,
  ].join(' | ');
}

function intentClassName(intent: V2OrderIntent) {
  return `sig-pill ${intent.status === 'rejected' ? 'short' : intent.direction}`;
}

export default function WorkbenchPage() {
  const [status, setStatus] = useState<V2Status | null>(null);
  const [symbols, setSymbols] = useState<string[]>([]);
  const [form, setForm] = useState<FormState>(DEFAULT_FORM);
  const [configDraft, setConfigDraft] = useState<V2Config | null>(null);
  const [analysis, setAnalysis] = useState<V2Analysis | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [scanning, setScanning] = useState(false);
  const [reconciling, setReconciling] = useState(false);
  const [recovering, setRecovering] = useState(false);
  const [switching, setSwitching] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadStatus = async () => {
    const [nextStatus, symbolPayload] = await Promise.all([getV2Status(), getV2Symbols()]);
    setStatus(nextStatus);
    setConfigDraft(nextStatus.config);
    setSymbols(symbolPayload.symbols);
    setForm((current) => ({
      symbol: current.symbol || nextStatus.config.default_symbol || symbolPayload.default || 'XAUUSD',
      timeframe: current.timeframe || nextStatus.config.default_timeframe || 'M5',
      strategy: current.strategy || nextStatus.config.default_strategy || 'sma_cross',
      numBars: current.numBars || 500,
    }));
  };

  useEffect(() => {
    let cancelled = false;
    const run = async () => {
      try {
        setLoading(true);
        setError(null);
        await loadStatus();
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Failed to load the workspace.');
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    };
    run();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const interval = window.setInterval(() => {
      loadStatus().catch(() => undefined);
    }, 5000);
    return () => window.clearInterval(interval);
  }, []);

  const readinessSummary = useMemo(() => {
    if (!status) {
      return { passed: 0, total: 0 };
    }
    return {
      passed: status.readiness.filter((item) => item.ok).length,
      total: status.readiness.length,
    };
  }, [status]);

  const strategyOptions = status?.strategies || [];

  const latestAnalysis = analysis || status?.recent_analyses[0] || null;

  const realizedToday = useMemo(() => {
    return (status?.recent_trade_audits || []).reduce((total, record) => {
      const realized = record.details?.realized_pnl;
      return total + (typeof realized === 'number' ? realized : 0);
    }, 0);
  }, [status]);

  const brokerNotes = (status?.broker.notes || []).filter(Boolean);

  const updateNumberField = <K extends keyof V2Config>(key: K, fallback: number, min?: number, max?: number) =>
    (event: ChangeEvent<HTMLInputElement>) => {
      if (!configDraft) {
        return;
      }
      let nextValue = Number(event.target.value);
      if (!Number.isFinite(nextValue)) {
        nextValue = fallback;
      }
      if (typeof min === 'number') {
        nextValue = Math.max(min, nextValue);
      }
      if (typeof max === 'number') {
        nextValue = Math.min(max, nextValue);
      }
      setConfigDraft({ ...configDraft, [key]: nextValue });
    };

  const handleAnalyze = async () => {
    try {
      setAnalyzing(true);
      setError(null);
      const next = await analyzeV2Strategy({
        symbol: form.symbol,
        timeframe: form.timeframe,
        strategy: form.strategy,
        num_bars: form.numBars,
      });
      setAnalysis(next);
      await loadStatus();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Strategy analysis failed.');
    } finally {
      setAnalyzing(false);
    }
  };

  const handleSaveConfig = async () => {
    if (!configDraft) {
      return;
    }
    try {
      setSaving(true);
      setError(null);
      await setV2Config(configDraft);
      await loadStatus();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save config.');
    } finally {
      setSaving(false);
    }
  };

  const handleEngineToggle = async () => {
    try {
      setSwitching(true);
      setError(null);
      if (status?.config.enabled) {
        await stopV2Engine();
      } else {
        await startV2Engine();
      }
      await loadStatus();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to toggle the engine.');
    } finally {
      setSwitching(false);
    }
  };

  const handleManualScan = async () => {
    try {
      setScanning(true);
      setError(null);
      await scanV2Engine();
      await loadStatus();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Manual scan failed.');
    } finally {
      setScanning(false);
    }
  };

  const handleReconcile = async () => {
    try {
      setReconciling(true);
      setError(null);
      await reconcileV2Engine();
      await loadStatus();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Manual reconciliation failed.');
    } finally {
      setReconciling(false);
    }
  };

  const handleRecover = async () => {
    try {
      setRecovering(true);
      setError(null);
      await recoverV2Engine();
      await loadStatus();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Runtime recovery failed.');
    } finally {
      setRecovering(false);
    }
  };

  const updateWatchItem = (index: number, patch: Partial<V2WatchlistItem>) => {
    if (!configDraft) {
      return;
    }
    const next = configDraft.watchlist.map((item, itemIndex) => {
      if (itemIndex !== index) {
        return item;
      }
      return { ...item, ...patch };
    });
    setConfigDraft({ ...configDraft, watchlist: next });
  };

  const addWatchItem = () => {
    if (!configDraft) {
      return;
    }
    setConfigDraft({ ...configDraft, watchlist: [...configDraft.watchlist, { ...DEFAULT_WATCH_ITEM }] });
  };

  const removeWatchItem = (index: number) => {
    if (!configDraft) {
      return;
    }
    setConfigDraft({ ...configDraft, watchlist: configDraft.watchlist.filter((_, itemIndex) => itemIndex !== index) });
  };

  return (
    <div className="v2-shell">
      <section className="v2-hero">
        <div>
          <p className="v2-kicker">TradeAgent</p>
          <h1>Deterministic paper engine with explicit control surfaces.</h1>
          <p className="v2-lead">
            TradeAgent is now structured around a simpler operator model: deterministic strategies, visible risk decisions,
            persisted paper execution, recovery, reconciliation, and a dedicated audit trail for intent handling.
          </p>
        </div>
        <div className="v2-hero-card">
          <div className="v2-hero-stat">
            <span>Engine</span>
            <strong>{status?.config.enabled ? 'Enabled' : 'Disabled'}</strong>
          </div>
          <div className="v2-hero-stat">
            <span>Loop</span>
            <strong>{status?.runtime.loop_active ? 'Scanning' : 'Idle'}</strong>
          </div>
          <div className="v2-hero-stat">
            <span>Readiness</span>
            <strong>{readinessSummary.passed}/{readinessSummary.total}</strong>
          </div>
          <div className="v2-hero-stat">
            <span>Mode</span>
            <strong>{status?.mode || 'paper_only'}</strong>
          </div>
        </div>
      </section>

      {error && <div className="v2-banner v2-banner-bad">{error}</div>}
      {loading && <div className="v2-banner">Loading workspace...</div>}

      <div className="v2-grid">
        <section className="v2-panel">
          <div className="v2-panel-head">
            <h2>Engine Status</h2>
            <div className="v2-actions-row">
              <button className="btn" type="button" onClick={handleRecover} disabled={recovering}>
                {recovering ? 'Recovering...' : 'Recover'}
              </button>
              <button className="btn" type="button" onClick={handleReconcile} disabled={reconciling}>
                {reconciling ? 'Reconciling...' : 'Reconcile'}
              </button>
              <button className="btn" type="button" onClick={handleManualScan} disabled={scanning}>
                {scanning ? 'Scanning...' : 'Scan Now'}
              </button>
              <button className="btn primary" type="button" onClick={handleEngineToggle} disabled={switching}>
                {switching ? 'Updating...' : status?.config.enabled ? 'Stop Engine' : 'Start Engine'}
              </button>
            </div>
          </div>
          <div className="v2-status-grid">
            <div className="v2-metric-card">
              <span>Broker mode</span>
              <strong>{status?.broker.broker_mode || 'unknown'}</strong>
            </div>
            <div className="v2-metric-card">
              <span>Loop ticks</span>
              <strong>{status?.runtime.tick_count ?? 0}</strong>
            </div>
            <div className="v2-metric-card">
              <span>Paper positions</span>
              <strong>{status?.paper_positions.length ?? 0}</strong>
            </div>
            <div className="v2-metric-card">
              <span>Watchlist active</span>
              <strong>{status?.runtime.active_watchlist.length ?? 0}</strong>
            </div>
            <div className="v2-metric-card">
              <span>Daily trades</span>
              <strong>{status?.recent_order_intents.filter((item) => item.intent_type === 'open').length ?? 0}</strong>
            </div>
            <div className="v2-metric-card">
              <span>Realized audit PnL</span>
              <strong>{formatSigned(realizedToday)}</strong>
            </div>
          </div>
          <div className="v2-checklist">
            {(status?.readiness || []).map((item) => (
              <div key={item.name} className={`v2-check ${item.ok ? 'good' : 'bad'}`}>
                <strong>{item.name.replace(/_/g, ' ')}</strong>
                <span>{item.detail}</span>
              </div>
            ))}
          </div>
          <div className="v2-notes">
            <div>Last cycle: {status?.runtime.last_cycle_summary || '-'}</div>
            <div>Last cycle at: {formatTimestamp(status?.runtime.last_cycle_at)}</div>
            <div>Last reconcile: {status?.runtime.last_reconcile_summary || '-'}</div>
            <div>Reconciled at: {formatTimestamp(status?.runtime.last_reconcile_at)}</div>
            <div>Last error: {status?.runtime.last_error || 'none'}</div>
          </div>
          {!!brokerNotes.length && (
            <div className="v2-feed v2-feed-compact">
              {brokerNotes.map((note) => (
                <article key={note} className="v2-feed-card">
                  <div className="v2-feed-head">
                    <strong>Broker note</strong>
                  </div>
                  <p>{note}</p>
                </article>
              ))}
            </div>
          )}
        </section>

        <section className="v2-panel">
          <div className="v2-panel-head">
            <h2>Strategy Probe</h2>
            <button className="btn primary" type="button" onClick={handleAnalyze} disabled={analyzing}>
              {analyzing ? 'Running...' : 'Run Analysis'}
            </button>
          </div>
          <div className="v2-form-grid">
            <label>
              Symbol
              <select value={form.symbol} onChange={(event) => setForm({ ...form, symbol: event.target.value })}>
                {(symbols.length ? symbols : ['XAUUSD']).map((symbol) => (
                  <option key={symbol} value={symbol}>
                    {symbol}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Timeframe
              <select value={form.timeframe} onChange={(event) => setForm({ ...form, timeframe: event.target.value })}>
                {TIMEFRAMES.map((timeframe) => (
                  <option key={timeframe} value={timeframe}>
                    {timeframe}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Strategy
              <select value={form.strategy} onChange={(event) => setForm({ ...form, strategy: event.target.value })}>
                {strategyOptions.map((strategy) => (
                  <option key={strategy.key} value={strategy.key}>
                    {strategy.label}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Bars
              <input
                type="number"
                min={100}
                max={5000}
                step={100}
                value={form.numBars}
                onChange={(event) => setForm({ ...form, numBars: Math.max(100, Number(event.target.value) || 500) })}
              />
            </label>
          </div>
          <div className="v2-analysis-card">
            <div className="v2-analysis-head">
              <span>Latest analysis</span>
              <strong className={`sig-pill ${latestAnalysis?.signal || 'no_trade'}`}>{latestAnalysis?.signal || 'no_trade'}</strong>
            </div>
            <div className="v2-analysis-stats">
              <span>Confidence: {latestAnalysis ? `${Math.round(latestAnalysis.confidence * 100)}%` : '-'}</span>
              <span>Entry: {formatPrice(latestAnalysis?.entry_price)}</span>
              <span>SL: {formatPrice(latestAnalysis?.stop_loss)}</span>
              <span>TP: {formatPrice(latestAnalysis?.take_profit)}</span>
            </div>
            <ul>
              {(latestAnalysis?.reasons || ['No analysis executed yet.']).map((reason) => (
                <li key={reason}>{reason}</li>
              ))}
            </ul>
          </div>
        </section>

        <section className="v2-panel">
          <div className="v2-panel-head">
            <h2>Operator Guardrails</h2>
            <button className="btn" type="button" onClick={handleSaveConfig} disabled={!configDraft || saving}>
              {saving ? 'Saving...' : 'Save Config'}
            </button>
          </div>
          {configDraft && (
            <>
              <div className="v2-form-grid v2-form-grid-wide">
                <label className="v2-toggle">
                  <input
                    type="checkbox"
                    checked={configDraft.enabled}
                    onChange={(event) => setConfigDraft({ ...configDraft, enabled: event.target.checked })}
                  />
                  Enable engine
                </label>
                <label className="v2-toggle">
                  <input
                    type="checkbox"
                    checked={configDraft.paper_autotrade}
                    onChange={(event) => setConfigDraft({ ...configDraft, paper_autotrade: event.target.checked })}
                  />
                  Paper autotrade
                </label>
                <label className="v2-toggle">
                  <input
                    type="checkbox"
                    checked={configDraft.kill_switch}
                    onChange={(event) => setConfigDraft({ ...configDraft, kill_switch: event.target.checked })}
                  />
                  Kill switch
                </label>
                <label className="v2-toggle">
                  <input
                    type="checkbox"
                    checked={configDraft.allow_live}
                    onChange={(event) => setConfigDraft({ ...configDraft, allow_live: event.target.checked })}
                  />
                  Request live mode
                </label>
                <label className="v2-toggle">
                  <input
                    type="checkbox"
                    checked={configDraft.require_stops}
                    onChange={(event) => setConfigDraft({ ...configDraft, require_stops: event.target.checked })}
                  />
                  Require stops
                </label>
                <label className="v2-toggle">
                  <input
                    type="checkbox"
                    checked={configDraft.session_filter_enabled}
                    onChange={(event) => setConfigDraft({ ...configDraft, session_filter_enabled: event.target.checked })}
                  />
                  Session filter
                </label>
                <label>
                  Scan interval sec
                  <input type="number" min={2} max={300} step={1} value={configDraft.scan_interval_sec} onChange={updateNumberField('scan_interval_sec', 10, 2, 300)} />
                </label>
                <label>
                  Min confidence
                  <input type="number" min={0.1} max={1} step={0.05} value={configDraft.min_confidence} onChange={updateNumberField('min_confidence', 0.6, 0.1, 1)} />
                </label>
                <label>
                  Fallback trade size
                  <input type="number" min={0.1} max={100} step={0.1} value={configDraft.paper_trade_size} onChange={updateNumberField('paper_trade_size', 1, 0.1, 100)} />
                </label>
                <label>
                  Auto risk %
                  <input type="number" min={0.1} max={10} step={0.1} value={configDraft.risk_per_trade_pct} onChange={updateNumberField('risk_per_trade_pct', 0.5, 0.1, 10)} />
                </label>
                <label>
                  Daily loss cap
                  <input type="number" min={0.5} max={10} step={0.5} value={configDraft.daily_loss_limit_pct} onChange={updateNumberField('daily_loss_limit_pct', 2, 0.5, 10)} />
                </label>
                <label>
                  Max daily trades
                  <input type="number" min={1} max={100} step={1} value={configDraft.max_daily_trades} onChange={updateNumberField('max_daily_trades', 12, 1, 100)} />
                </label>
                <label>
                  Max open positions
                  <input type="number" min={1} max={20} step={1} value={configDraft.max_open_positions} onChange={updateNumberField('max_open_positions', 3, 1, 20)} />
                </label>
                <label>
                  Max positions per symbol
                  <input type="number" min={1} max={10} step={1} value={configDraft.max_positions_per_symbol} onChange={updateNumberField('max_positions_per_symbol', 1, 1, 10)} />
                </label>
                <label>
                  Cooldown minutes
                  <input type="number" min={0} max={1440} step={5} value={configDraft.cooldown_minutes} onChange={updateNumberField('cooldown_minutes', 30, 0, 1440)} />
                </label>
                <label>
                  Session start UTC
                  <input type="number" min={0} max={23} step={1} value={configDraft.session_start_hour_utc} onChange={updateNumberField('session_start_hour_utc', 6, 0, 23)} />
                </label>
                <label>
                  Session end UTC
                  <input type="number" min={0} max={23} step={1} value={configDraft.session_end_hour_utc} onChange={updateNumberField('session_end_hour_utc', 21, 0, 23)} />
                </label>
              </div>
              <label className="v2-stack-field">
                Operator note
                <textarea
                  rows={3}
                  value={configDraft.operator_note}
                  onChange={(event) => setConfigDraft({ ...configDraft, operator_note: event.target.value })}
                />
              </label>
            </>
          )}
        </section>

        <section className="v2-panel">
          <div className="v2-panel-head">
            <h2>Watchlist</h2>
            <button className="btn" type="button" onClick={addWatchItem} disabled={!configDraft}>
              Add
            </button>
          </div>
          <div className="v2-watchlist">
            {(configDraft?.watchlist || []).map((item, index) => (
              <div key={`${item.symbol}-${item.timeframe}-${index}`} className="v2-watch-row">
                <label className="v2-toggle">
                  <input
                    type="checkbox"
                    checked={item.enabled}
                    onChange={(event) => updateWatchItem(index, { enabled: event.target.checked })}
                  />
                  Active
                </label>
                <select value={item.symbol} onChange={(event) => updateWatchItem(index, { symbol: event.target.value })}>
                  {(symbols.length ? symbols : ['XAUUSD']).map((symbol) => (
                    <option key={symbol} value={symbol}>
                      {symbol}
                    </option>
                  ))}
                </select>
                <select value={item.timeframe} onChange={(event) => updateWatchItem(index, { timeframe: event.target.value })}>
                  {TIMEFRAMES.map((timeframe) => (
                    <option key={timeframe} value={timeframe}>
                      {timeframe}
                    </option>
                  ))}
                </select>
                <select value={item.strategy} onChange={(event) => updateWatchItem(index, { strategy: event.target.value })}>
                  {strategyOptions.map((strategy) => (
                    <option key={strategy.key} value={strategy.key}>
                      {strategy.label}
                    </option>
                  ))}
                </select>
                <button className="btn" type="button" onClick={() => removeWatchItem(index)}>
                  Remove
                </button>
              </div>
            ))}
            {!configDraft?.watchlist.length && <div className="v2-banner">No watchlist entries configured.</div>}
          </div>
        </section>

        <section className="v2-panel">
          <div className="v2-panel-head">
            <h2>Paper Positions</h2>
          </div>
          <div className="v2-feed">
            {(status?.paper_positions || []).map((position) => (
              <article key={position.id} className="v2-feed-card">
                <div className="v2-feed-head">
                  <strong>{position.symbol} {position.timeframe}</strong>
                  <span className={`sig-pill ${position.direction}`}>{position.direction}</span>
                </div>
                <p>{position.strategy}</p>
                <span>{positionSummary(position)}</span>
                <span>SL {formatPrice(position.stop_loss)} | TP {formatPrice(position.take_profit)}</span>
              </article>
            ))}
            {!status?.paper_positions.length && <div className="v2-banner">No open paper positions.</div>}
          </div>
        </section>

        <section className="v2-panel">
          <div className="v2-panel-head">
            <h2>Order Intents</h2>
          </div>
          <div className="v2-feed">
            {(status?.recent_order_intents || []).map((intent) => (
              <article key={intent.id} className="v2-feed-card">
                <div className="v2-feed-head">
                  <strong>{intent.symbol} {intent.timeframe}</strong>
                  <span className={intentClassName(intent)}>{intent.status}</span>
                </div>
                <p>{intent.strategy} | {intent.intent_type} | confidence {Math.round(intent.confidence * 100)}%</p>
                <span>{intent.rationale || 'No rationale recorded.'}</span>
                <span>{formatTimestamp(intent.created_at)}</span>
              </article>
            ))}
            {!status?.recent_order_intents.length && <div className="v2-banner">No order intents recorded yet.</div>}
          </div>
        </section>

        <section className="v2-panel">
          <div className="v2-panel-head">
            <h2>Trade Audit</h2>
          </div>
          <div className="v2-feed">
            {(status?.recent_trade_audits || []).map((record: V2TradeAudit) => (
              <article key={record.id} className="v2-feed-card">
                <div className="v2-feed-head">
                  <strong>{record.event_type}</strong>
                  <span>{record.symbol} {record.timeframe}</span>
                </div>
                <p>{record.summary}</p>
                <span>{record.strategy}</span>
                <span>{formatTimestamp(record.created_at)}</span>
              </article>
            ))}
            {!status?.recent_trade_audits.length && <div className="v2-banner">No trade audit records yet.</div>}
          </div>
        </section>

        <section className="v2-panel">
          <div className="v2-panel-head">
            <h2>Engine Events</h2>
          </div>
          <div className="v2-feed">
            {(status?.recent_events || []).map((event) => (
              <article key={event.id} className="v2-feed-card">
                <div className="v2-feed-head">
                  <strong>{event.event_type}</strong>
                  <span>{new Date(event.created_at).toLocaleTimeString()}</span>
                </div>
                <p>{event.summary}</p>
              </article>
            ))}
            {!status?.recent_events.length && <div className="v2-banner">No paper events recorded yet.</div>}
          </div>
        </section>

        <section className="v2-panel">
          <div className="v2-panel-head">
            <h2>Incident Log</h2>
          </div>
          <div className="v2-feed">
            {(status?.recent_incidents || []).map((incident) => (
              <article key={incident.id} className={`v2-feed-card incident-${incident.level}`}>
                <div className="v2-feed-head">
                  <strong>{incident.code}</strong>
                  <span>{incident.level}</span>
                </div>
                <p>{incident.message}</p>
                <span>{formatTimestamp(incident.created_at)}</span>
              </article>
            ))}
            {!status?.recent_incidents.length && <div className="v2-banner">No incidents recorded.</div>}
          </div>
        </section>
      </div>
    </div>
  );
}
