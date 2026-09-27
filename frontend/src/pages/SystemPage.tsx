import { useEffect, useMemo, useState, type ChangeEvent } from 'react';

import AppNav from '../components/AppNav';
import {
  getV2Status,
  reconcileV2Engine,
  recoverV2Engine,
  scanV2Engine,
  setV2Config,
  startV2Engine,
  stopV2Engine,
  type V2Config,
  type V2Status,
} from '../services/api';
import { formatBackendLocalDateTime } from '../utils/datetime';

const formatTime = (value?: string | null) => formatBackendLocalDateTime(value);

export default function SystemPage() {
  const [status, setStatus] = useState<V2Status | null>(null);
  const [draft, setDraft] = useState<V2Config | null>(null);
  const [busy, setBusy] = useState<'save' | 'engine' | 'scan' | 'recover' | 'reconcile' | ''>('');
  const [error, setError] = useState('');

  const load = async (syncDraft = false) => {
    const payload = await getV2Status();
    setStatus(payload);
    if (syncDraft || !draft) setDraft(payload.config);
  };

  useEffect(() => {
    load(true).catch((err) => setError(err instanceof Error ? err.message : 'System status unavailable.'));
    const interval = window.setInterval(() => {
      getV2Status().then(setStatus).catch(() => undefined);
    }, 10_000);
    return () => window.clearInterval(interval);
  }, []);

  const readiness = useMemo(() => ({
    passed: status?.readiness.filter((item) => item.ok).length ?? 0,
    total: status?.readiness.length ?? 0,
  }), [status]);

  const updateNumber = <K extends keyof V2Config>(key: K, fallback: number, min: number, max: number) =>
    (event: ChangeEvent<HTMLInputElement>) => {
      if (!draft) return;
      const parsed = Number(event.target.value);
      const value = Math.min(max, Math.max(min, Number.isFinite(parsed) ? parsed : fallback));
      setDraft({ ...draft, [key]: value });
    };

  const run = async (kind: typeof busy, action: () => Promise<unknown>) => {
    setBusy(kind);
    setError('');
    try {
      await action();
      await load(kind === 'save');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'System operation failed.');
    } finally {
      setBusy('');
    }
  };

  const toggleEngine = () => run('engine', () => status?.config.enabled ? stopV2Engine() : startV2Engine());
  const save = () => draft ? run('save', () => setV2Config(draft)) : Promise.resolve();

  return (
    <div className="ta-app">
      <AppNav
        right={(
          <>
            <span className="ta-status"><span className={`ta-status__dot ta-status__dot--${status?.broker.ready ? 'ok' : 'bad'}`} />Broker {status?.broker.ready ? 'ready' : 'not ready'}</span>
            <span className="ta-status"><span className={`ta-status__dot ta-status__dot--${status?.config.enabled ? 'ok' : 'wait'}`} />Engine {status?.config.enabled ? 'enabled' : 'disabled'}</span>
          </>
        )}
      />

      <main className="v2-shell">
        <section className="v2-hero">
          <div>
            <p className="v2-kicker">System</p>
            <h1>Connections, safety controls, recovery, and audit.</h1>
            <p className="v2-lead">This area operates the platform. Strategy research belongs in Build & Test; market decisions and positions belong in Trade.</p>
          </div>
          <div className="v2-hero-card">
            <div className="v2-hero-stat"><span>Readiness</span><strong>{readiness.passed}/{readiness.total}</strong></div>
            <div className="v2-hero-stat"><span>Mode</span><strong>{status?.mode || 'paper_only'}</strong></div>
            <div className="v2-hero-stat"><span>Account</span><strong>{status?.broker.demo_account_confirmed ? 'Demo verified' : 'Not verified'}</strong></div>
            <div className="v2-hero-stat"><span>Market data</span><strong>{status?.broker.market_data_ready ? 'Ready' : 'Waiting'}</strong></div>
            <div className="v2-hero-stat"><span>Kill switch</span><strong>{status?.config.kill_switch ? 'Active' : 'Inactive'}</strong></div>
          </div>
        </section>

        {error && <div className="v2-banner v2-banner-bad">{error}</div>}

        <section className="v2-panel">
          <div className="v2-panel-head">
            <div>
              <h2>Runtime health</h2>
              <span>Last cycle {formatTime(status?.runtime.last_cycle_at)} · {status?.runtime.last_cycle_summary || '--'}</span>
            </div>
            <div className="v2-actions-row">
              <button className="btn" type="button" onClick={() => run('recover', recoverV2Engine)} disabled={busy !== ''}>{busy === 'recover' ? 'Recovering…' : 'Recover'}</button>
              <button className="btn" type="button" onClick={() => run('reconcile', reconcileV2Engine)} disabled={busy !== ''}>{busy === 'reconcile' ? 'Reconciling…' : 'Reconcile'}</button>
              <button className="btn" type="button" onClick={() => run('scan', scanV2Engine)} disabled={busy !== ''}>{busy === 'scan' ? 'Scanning…' : 'Run one scan'}</button>
              <button className={`btn ${status?.config.enabled ? 'danger' : 'primary'}`} type="button" onClick={toggleEngine} disabled={busy !== ''}>{busy === 'engine' ? 'Updating…' : status?.config.enabled ? 'Stop engine' : 'Start engine'}</button>
            </div>
          </div>

          <div className="v2-checklist">
            {(status?.readiness ?? []).map((item) => (
              <div className={`v2-check ${item.ok ? 'good' : 'bad'}`} key={item.name}>
                <strong>{item.name.replaceAll('_', ' ')}</strong>
                <span>{item.detail}</span>
              </div>
            ))}
          </div>
          <div className="v2-notes">
            <div>Broker: {status?.broker.broker_mode || '--'} · symbols {status?.broker.symbols_loaded ?? 0}</div>
            <div>Last reconcile: {status?.runtime.last_reconcile_summary || '--'} at {formatTime(status?.runtime.last_reconcile_at)}</div>
            <div>Last error: {status?.runtime.last_error || 'none'}</div>
          </div>
        </section>

        <section className="v2-panel">
          <div className="v2-panel-head">
            <div>
              <h2>Safety configuration</h2>
              <span>Paper execution and cTrader demo execution are separate. Live-account execution is always blocked.</span>
            </div>
            <button className="btn primary" type="button" onClick={save} disabled={!draft || busy !== ''}>{busy === 'save' ? 'Saving…' : 'Save safety settings'}</button>
          </div>

          {draft && (
            <>
              <div className="v2-form-grid v2-form-grid-wide">
                <label className="v2-toggle"><input type="checkbox" checked={draft.kill_switch} onChange={(event) => setDraft({ ...draft, kill_switch: event.target.checked })} />Kill switch</label>
                <label className="v2-toggle"><input type="checkbox" checked={draft.paper_autotrade} onChange={(event) => setDraft({ ...draft, paper_autotrade: event.target.checked })} />Paper autotrade</label>
                <label className="v2-toggle"><input type="checkbox" checked={draft.demo_autotrade} onChange={(event) => setDraft({ ...draft, demo_autotrade: event.target.checked })} />cTrader auto-trade</label>
                <label className="v2-toggle"><input type="checkbox" checked={draft.require_stops} onChange={(event) => setDraft({ ...draft, require_stops: event.target.checked })} />Require protective stops</label>
                <label className="v2-toggle"><input type="checkbox" checked={draft.session_filter_enabled} onChange={(event) => setDraft({ ...draft, session_filter_enabled: event.target.checked })} />Restrict trading session</label>
                <label>Minimum signal quality<input type="number" min="0" max="1" step="0.05" value={draft.min_confidence} onChange={updateNumber('min_confidence', 0.6, 0, 1)} /></label>
                <label>Risk per trade (%)<input type="number" min="0.01" max="5" step="0.05" value={draft.risk_per_trade_pct} onChange={updateNumber('risk_per_trade_pct', 0.5, 0.01, 5)} /></label>
                <label>Daily loss limit (%)<input type="number" min="0.1" max="20" step="0.1" value={draft.daily_loss_limit_pct} onChange={updateNumber('daily_loss_limit_pct', 2, 0.1, 20)} /></label>
                <label>Maximum daily trades<input type="number" min="1" max="100" step="1" value={draft.max_daily_trades} onChange={updateNumber('max_daily_trades', 12, 1, 100)} /></label>
                <label>Maximum open positions<input type="number" min="1" max="20" step="1" value={draft.max_open_positions} onChange={updateNumber('max_open_positions', 3, 1, 20)} /></label>
                <label>Cooldown (minutes)<input type="number" min="0" max="1440" step="1" value={draft.cooldown_minutes} onChange={updateNumber('cooldown_minutes', 30, 0, 1440)} /></label>
                <label>Session start (UTC)<input type="number" min="0" max="23" step="1" value={draft.session_start_hour_utc} onChange={updateNumber('session_start_hour_utc', 6, 0, 23)} disabled={!draft.session_filter_enabled} /></label>
                <label>Session end (UTC)<input type="number" min="0" max="23" step="1" value={draft.session_end_hour_utc} onChange={updateNumber('session_end_hour_utc', 21, 0, 23)} disabled={!draft.session_filter_enabled} /></label>
              </div>
              <label style={{ display: 'grid', gap: 6, marginTop: 12 }}>Operator note<textarea rows={2} value={draft.operator_note} onChange={(event) => setDraft({ ...draft, operator_note: event.target.value })} /></label>
            </>
          )}
        </section>

        <div className="mi-context-grid">
          <section className="v2-panel">
            <div className="v2-panel-head"><h2>Decision audit</h2></div>
            <div className="v2-feed">
              {(status?.recent_decisions ?? []).slice(0, 10).map((item) => (
                <article className="v2-feed-card" key={item.id}>
                  <div className="v2-feed-head"><strong>{item.outcome} · {item.symbol}</strong><span>{formatTime(item.created_at)}</span></div>
                  <p>{item.summary}</p>
                </article>
              ))}
              {!status?.recent_decisions.length && <div className="v2-empty">No decision records.</div>}
            </div>
          </section>

          <section className="v2-panel">
            <div className="v2-panel-head"><h2>Trade audit</h2></div>
            <div className="v2-feed">
              {(status?.recent_trade_audits ?? []).slice(0, 10).map((item) => (
                <article className="v2-feed-card" key={item.id}>
                  <div className="v2-feed-head"><strong>{item.event_type.replaceAll('_', ' ')}</strong><span>{formatTime(item.created_at)}</span></div>
                  <p>{item.summary}</p>
                </article>
              ))}
              {!status?.recent_trade_audits.length && <div className="v2-empty">No trade audit records.</div>}
            </div>
          </section>

          <section className="v2-panel">
            <div className="v2-panel-head"><h2>Engine events</h2></div>
            <div className="v2-feed">
              {(status?.recent_events ?? []).slice(0, 10).map((item) => (
                <article className="v2-feed-card" key={item.id}>
                  <div className="v2-feed-head"><strong>{item.event_type.replaceAll('_', ' ')}</strong><span>{formatTime(item.created_at)}</span></div>
                  <p>{item.summary}</p>
                </article>
              ))}
              {!status?.recent_events.length && <div className="v2-empty">No engine events.</div>}
            </div>
          </section>

          <section className="v2-panel">
            <div className="v2-panel-head"><h2>Incidents</h2></div>
            <div className="v2-feed">
              {(status?.recent_incidents ?? []).slice(0, 10).map((item) => (
                <article className="v2-feed-card" key={item.id}>
                  <div className="v2-feed-head"><strong>{item.level} · {item.code}</strong><span>{formatTime(item.created_at)}</span></div>
                  <p>{item.message}</p>
                </article>
              ))}
              {!status?.recent_incidents.length && <div className="v2-empty">No incidents.</div>}
            </div>
          </section>
        </div>
      </main>
    </div>
  );
}
