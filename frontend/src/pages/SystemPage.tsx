import { useEffect, useMemo, useState, type ChangeEvent } from 'react';
import { Link } from 'react-router-dom';

import AppNav from '../components/AppNav';
import {
  getV2CTraderAccounts,
  getV2Status,
  reconcileV2Engine,
  recoverV2Engine,
  restartV2Engine,
  scanV2Engine,
  setV2Config,
  setV2LiveTradingArm,
  startV2Engine,
  stopV2Engine,
  type V2Config,
  type V2CTraderAccount,
  type V2Status,
} from '../services/api';
import {
  SYSTEM_ACTIVITY_TABS,
  operatorIncidentHistory,
  readinessStats,
  type SystemActivityTab,
} from '../services/systemConsole';
import { formatBackendLocalDateTime } from '../utils/datetime';

const formatTime = (value?: string | null) => formatBackendLocalDateTime(value);
const pretty = (value: string) => value.replaceAll('_', ' ');

export default function SystemPage() {
  const [status, setStatus] = useState<V2Status | null>(null);
  const [accounts, setAccounts] = useState<V2CTraderAccount[]>([]);
  const [draft, setDraft] = useState<V2Config | null>(null);
  const [busy, setBusy] = useState<'save' | 'live-arm' | 'engine' | 'restart' | 'scan' | 'recover' | 'reconcile' | ''>('');
  const [error, setError] = useState('');
  const [showReadiness, setShowReadiness] = useState(false);
  const [showSafetyEditor, setShowSafetyEditor] = useState(false);
  const [activityTab, setActivityTab] = useState<SystemActivityTab>('incidents');

  const load = async (syncDraft = false) => {
    const [payload, accountRows] = await Promise.all([
      getV2Status(),
      getV2CTraderAccounts(),
    ]);
    setStatus(payload);
    setAccounts(accountRows);
    if (syncDraft || !draft) setDraft(payload.config);
  };

  useEffect(() => {
    load(true).catch((err) => setError(err instanceof Error ? err.message : 'System status unavailable.'));
    const interval = window.setInterval(() => {
      Promise.all([getV2Status(), getV2CTraderAccounts()])
        .then(([payload, accountRows]) => {
          setStatus(payload);
          setAccounts(accountRows);
        })
        .catch(() => undefined);
    }, 5_000);
    return () => window.clearInterval(interval);
  }, []);

  const readiness = useMemo(
    () => readinessStats(status?.readiness ?? []),
    [status],
  );
  const failedReadiness = useMemo(
    () => (status?.readiness ?? []).filter((item) => !item.ok),
    [status],
  );
  const incidentHistory = useMemo(
    () => operatorIncidentHistory(status?.recent_incidents ?? []),
    [status],
  );

  const updateNumber = <K extends keyof V2Config>(key: K, fallback: number, min: number, max: number) =>
    (event: ChangeEvent<HTMLInputElement>) => {
      if (!draft) return;
      const parsed = Number(event.target.value);
      const value = Math.min(max, Math.max(min, Number.isFinite(parsed) ? parsed : fallback));
      setDraft({ ...draft, [key]: value });
    };

  const run = async (kind: typeof busy, action: () => Promise<unknown>): Promise<boolean> => {
    setBusy(kind);
    setError('');
    try {
      await action();
      await load(kind === 'save');
      return true;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'System operation failed.');
      return false;
    } finally {
      setBusy('');
    }
  };

  const toggleEngine = () => {
    void run('engine', () => status?.config.enabled ? stopV2Engine() : startV2Engine());
  };

  const save = async () => {
    if (!draft) return;
    const ok = await run('save', () => setV2Config(draft));
    if (ok) setShowSafetyEditor(false);
  };

  const selectedAccount = accounts.find((account) => account.selected);
  const activeAccount = accounts.find((account) => account.active);
  const accountLabel = (account: V2CTraderAccount) => {
    const broker = account.broker_title || 'cTrader';
    const login = account.trader_login ?? account.account_id;
    return `${broker} · ${account.account_type === 'live' ? 'Live' : 'Demo'} · ${login}`;
  };
  const accountShortLabel = activeAccount
    ? `${activeAccount.account_type === 'live' ? 'Live' : 'Demo'} · ${activeAccount.trader_login ?? activeAccount.account_id}`
    : status?.broker.account_switch_in_progress
      ? 'Switching…'
      : 'Not connected';

  const toggleLiveTradingArm = () => {
    const next = !(status?.live_trading_armed ?? false);
    void run('live-arm', () => setV2LiveTradingArm(next));
  };

  const openSafetyEditor = () => {
    if (status?.config) setDraft(status.config);
    setShowSafetyEditor(true);
  };

  const activityCount = (tab: SystemActivityTab) => {
    if (!status) return 0;
    if (tab === 'incidents') return status.active_incidents.length;
    if (tab === 'engine') return status.recent_events.length;
    if (tab === 'trades') return status.recent_trade_audits.length;
    if (tab === 'decisions') return status.recent_decisions.length;
    return incidentHistory.length;
  };

  const renderActivity = () => {
    if (!status) return <div className="system-empty">Loading activity…</div>;

    if (activityTab === 'incidents') {
      if (!status.active_incidents.length) {
        return <div className="system-empty system-empty--good">✓ No current incidents</div>;
      }
      return status.active_incidents.map((item) => (
        <article className={`system-activity-row system-activity-row--${item.level}`} key={item.code}>
          <div>
            <strong>{pretty(item.code)}</strong>
            <p>{item.message}</p>
          </div>
          <span>{item.level}</span>
        </article>
      ));
    }

    if (activityTab === 'engine') {
      if (!status.recent_events.length) return <div className="system-empty">No engine events.</div>;
      return status.recent_events.slice(0, 20).map((item) => (
        <article className="system-activity-row" key={item.id}>
          <div>
            <strong>{pretty(item.event_type)}</strong>
            <p>{item.summary}</p>
          </div>
          <time>{formatTime(item.created_at)}</time>
        </article>
      ));
    }

    if (activityTab === 'trades') {
      if (!status.recent_trade_audits.length) return <div className="system-empty">No trade audit records.</div>;
      return status.recent_trade_audits.slice(0, 20).map((item) => (
        <article className="system-activity-row" key={item.id}>
          <div>
            <strong>{pretty(item.event_type)}</strong>
            <p>{item.summary}</p>
          </div>
          <time>{formatTime(item.created_at)}</time>
        </article>
      ));
    }

    if (activityTab === 'decisions') {
      if (!status.recent_decisions.length) return <div className="system-empty">No decision records.</div>;
      return status.recent_decisions.slice(0, 20).map((item) => (
        <article className="system-activity-row" key={item.id}>
          <div>
            <strong>{pretty(item.outcome)} · {item.symbol}</strong>
            <p>{item.summary}</p>
          </div>
          <time>{formatTime(item.created_at)}</time>
        </article>
      ));
    }

    if (!incidentHistory.length) {
      return <div className="system-empty system-empty--good">✓ No warning/error incident history in the recent window</div>;
    }
    return incidentHistory.slice(0, 20).map((item) => (
      <article className={`system-activity-row system-activity-row--${item.level}`} key={item.id}>
        <div>
          <strong>{pretty(item.code)}</strong>
          <p>{item.message}</p>
        </div>
        <time>{formatTime(item.created_at)}</time>
      </article>
    ));
  };

  return (
    <div className="ta-app">
      <AppNav
        right={(
          <>
            <span className="ta-status"><span className={`ta-status__dot ta-status__dot--${status?.broker.socket_connected ? 'ok' : 'bad'}`} />Broker {status?.broker.socket_connected ? 'connected' : 'disconnected'}</span>
            <span className="ta-status"><span className={`ta-status__dot ta-status__dot--${status?.runtime.loop_active ? 'ok' : 'wait'}`} />Engine {status?.runtime.loop_active ? 'scanning' : 'stopped'}</span>
          </>
        )}
      />

      <main className="system-shell">
        <header className="system-heading">
          <div>
            <p className="system-kicker">System</p>
            <h1>Operations & safety</h1>
            <p>Platform health, execution controls, recovery, and diagnostics.</p>
          </div>
        </header>

        {error && <div className="v2-banner v2-banner-bad">{error}</div>}

        <section className="system-status-grid" aria-label="System status overview">
          <article className={`system-stat ${readiness.failed ? 'system-stat--warn' : ''}`}>
            <span>Readiness</span>
            <strong>{readiness.passed}/{readiness.total}</strong>
            <small>{readiness.failed ? `${readiness.failed} needs attention` : 'All checks passed'}</small>
          </article>
          <article className="system-stat">
            <span>Active account</span>
            <strong>{accountShortLabel}</strong>
            <small>{status?.broker.account_verified ? 'Verified' : 'Not verified'} · change in Trade</small>
          </article>
          <article className="system-stat">
            <span>Market</span>
            <strong>{status?.broker.market_data_ready ? 'Ready' : 'Waiting'}</strong>
            <small>{status?.broker.symbols_loaded ?? 0} symbols loaded</small>
          </article>
          <article className={`system-stat ${status?.runtime.loop_active ? '' : 'system-stat--muted'}`}>
            <span>Engine</span>
            <strong>{status?.runtime.loop_active ? 'Scanning' : 'Stopped'}</strong>
            <small>{status?.runtime.last_cycle_summary || 'No cycle yet'}</small>
          </article>
          <article className={`system-stat ${status?.config.kill_switch ? 'system-stat--bad' : ''}`}>
            <span>Kill switch</span>
            <strong>{status?.config.kill_switch ? 'ACTIVE' : 'Off'}</strong>
            <small>{status?.config.kill_switch ? 'New trading blocked' : 'Trading permitted by this gate'}</small>
          </article>
          <article className={`system-stat ${status?.active_incidents.length ? 'system-stat--bad' : ''}`}>
            <span>Incidents</span>
            <strong>{status?.active_incidents.length ?? 0}</strong>
            <small>{status?.active_incidents.length ? 'Needs attention' : 'No current incidents'}</small>
          </article>
        </section>

        {selectedAccount && activeAccount && selectedAccount.account_id !== activeAccount.account_id && (
          <div className="system-inline-alert">
            <strong>Account selection is not active.</strong>
            <span>Selected {accountLabel(selectedAccount)} · active {accountLabel(activeAccount)}.</span>
            <Link className="ta-btn ta-btn--sm" to="/">Open Trade</Link>
          </div>
        )}

        <section className="system-main-grid">
          <article className="ta-panel system-card">
            <div className="ta-panel__header system-card__header">
              <div>
                <div className="ta-panel__title">Runtime & readiness</div>
                <p>Last cycle {formatTime(status?.runtime.last_cycle_at)} · {status?.runtime.last_cycle_summary || '—'}</p>
              </div>
              <span className={`system-health-pill ${readiness.failed ? 'warn' : 'good'}`}>
                {readiness.failed ? `${readiness.failed} attention` : 'Healthy'}
              </span>
            </div>
            <div className="ta-panel__body system-card__body">
              <div className={`system-readiness-callout ${readiness.failed ? 'warn' : 'good'}`}>
                <strong>{readiness.failed ? 'Action needed' : 'Execution checks are healthy'}</strong>
                <span>
                  {failedReadiness[0]?.detail
                    || 'Broker, account, market, and risk readiness checks currently pass.'}
                </span>
              </div>

              <div className="system-engine-actions">
                <button
                  className={`ta-btn ${status?.config.enabled ? 'ta-btn--danger' : 'ta-btn--primary'}`}
                  type="button"
                  onClick={toggleEngine}
                  disabled={busy !== ''}
                >
                  {busy === 'engine' ? 'Updating…' : status?.config.enabled ? 'Stop engine' : 'Start engine'}
                </button>
                <button className="ta-btn" type="button" onClick={() => void run('scan', scanV2Engine)} disabled={busy !== ''}>
                  {busy === 'scan' ? 'Scanning…' : 'Run one scan'}
                </button>
                <button className="ta-btn ta-btn--ghost" type="button" onClick={() => setShowReadiness((value) => !value)}>
                  {showReadiness ? 'Hide checks' : 'Show checks'}
                </button>
                <details className="system-more-actions">
                  <summary className="ta-btn ta-btn--ghost">More actions</summary>
                  <div className="system-more-actions__menu">
                    <button className="ta-btn ta-btn--ghost" type="button" onClick={() => void run('reconcile', reconcileV2Engine)} disabled={busy !== ''}>Reconcile</button>
                    <button className="ta-btn ta-btn--ghost" type="button" onClick={() => void run('recover', recoverV2Engine)} disabled={busy !== ''}>Recover runtime</button>
                    <button className="ta-btn ta-btn--ghost" type="button" onClick={() => void run('restart', restartV2Engine)} disabled={busy !== ''}>Restart engine</button>
                  </div>
                </details>
              </div>

              {showReadiness && (
                <div className="system-check-list">
                  {(status?.readiness ?? []).map((item) => (
                    <div className={`system-check-row ${item.ok ? 'good' : 'bad'}`} key={item.name}>
                      <span className="system-check-row__dot" />
                      <div>
                        <strong>{pretty(item.name)}</strong>
                        <small>{item.detail}</small>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              <div className="system-runtime-meta">
                <span>Last reconcile: {status?.runtime.last_reconcile_summary || '—'} · {formatTime(status?.runtime.last_reconcile_at)}</span>
                <span>Last error: {status?.runtime.last_error || 'none'}</span>
              </div>
            </div>
          </article>

          <article className="ta-panel system-card">
            <div className="ta-panel__header system-card__header">
              <div>
                <div className="ta-panel__title">Safety</div>
                <p>Saved execution gates and risk limits.</p>
              </div>
              <button className="ta-btn ta-btn--sm" type="button" onClick={openSafetyEditor}>Edit settings</button>
            </div>
            <div className="ta-panel__body system-card__body">
              <div className="system-safety-grid">
                <div><span>cTrader auto-trade</span><strong>{status?.config.ctrader_autotrade ? 'On' : 'Off'}</strong></div>
                <div><span>Protective stops</span><strong>{status?.config.require_stops ? 'Required' : 'Optional'}</strong></div>
                <div><span>Risk / trade</span><strong>{status?.config.risk_per_trade_pct ?? '—'}%</strong></div>
                <div><span>Daily loss limit</span><strong>{status?.config.daily_loss_limit_pct ?? '—'}%</strong></div>
                <div><span>Signal strength</span><strong>≥ {Math.round((status?.config.min_confidence ?? 0) * 100)}%</strong></div>
                <div><span>Max open positions</span><strong>{status?.config.max_open_positions ?? '—'}</strong></div>
                <div><span>Max daily trades</span><strong>{status?.config.max_daily_trades ?? '—'}</strong></div>
                <div><span>Cooldown</span><strong>{status?.config.cooldown_minutes ?? '—'} min</strong></div>
              </div>

              <div className={`system-live-state ${status?.live_trading_armed ? 'armed' : ''}`}>
                <div>
                  <strong>{status?.live_trading_armed ? 'LIVE TRADING ARMED' : 'Live execution'}</strong>
                  <span>
                    {activeAccount?.account_type === 'live'
                      ? `${accountLabel(activeAccount)} · ${status?.live_trading_armed ? 'real-money entries armed' : 'real-money entries disarmed'}`
                      : 'Demo account active · Live arm not required'}
                  </span>
                </div>
                {activeAccount?.account_type === 'live' && (
                  <button
                    className={`ta-btn ${status?.live_trading_armed ? 'ta-btn--danger' : 'ta-btn--primary'}`}
                    type="button"
                    onClick={toggleLiveTradingArm}
                    disabled={
                      busy !== ''
                      || (
                        !status?.live_trading_armed
                        && (
                          status?.config.enabled
                          || status?.config.ctrader_autotrade !== true
                          || selectedAccount?.account_id !== activeAccount?.account_id
                          || !status?.broker.account_verified
                        )
                      )
                    }
                  >
                    {busy === 'live-arm'
                      ? 'Updating…'
                      : status?.live_trading_armed
                        ? 'Disarm Live Trading'
                        : 'Arm Live Trading'}
                  </button>
                )}
              </div>
            </div>
          </article>
        </section>

        <section className="ta-panel system-activity">
          <div className="ta-panel__header system-activity__header">
            <div>
              <div className="ta-panel__title">Activity & diagnostics</div>
              <p>Current incidents and recent operational history in one place.</p>
            </div>
          </div>
          <div className="system-tabs" role="tablist" aria-label="System activity">
            {SYSTEM_ACTIVITY_TABS.map((tab) => (
              <button
                className={`system-tab ${activityTab === tab.id ? 'active' : ''}`}
                type="button"
                role="tab"
                aria-selected={activityTab === tab.id}
                key={tab.id}
                onClick={() => setActivityTab(tab.id)}
              >
                {tab.label}
                <span>{activityCount(tab.id)}</span>
              </button>
            ))}
          </div>
          <div className="system-activity__feed" role="tabpanel">
            {renderActivity()}
          </div>
        </section>
      </main>

      {showSafetyEditor && draft && (
        <div className="ta-overlay" role="dialog" aria-modal="true" aria-labelledby="system-safety-title">
          <button className="ta-overlay__backdrop" aria-label="Close safety settings" type="button" onClick={() => setShowSafetyEditor(false)} />
          <div className="ta-overlay__panel system-safety-modal">
            <div className="ta-overlay__header">
              <div>
                <div className="ta-overlay__title" id="system-safety-title">Safety settings</div>
                <p className="system-modal-subtitle">Execution controls and risk limits. Changes apply only after Save.</p>
              </div>
              <button className="ta-btn ta-btn--ghost ta-btn--icon" type="button" onClick={() => setShowSafetyEditor(false)} aria-label="Close">×</button>
            </div>
            <div className="ta-overlay__body">
              <div className="system-settings-grid">
                <label className="v2-toggle"><input type="checkbox" checked={draft.kill_switch} onChange={(event) => setDraft({ ...draft, kill_switch: event.target.checked })} />Kill switch</label>
                <label className="v2-toggle"><input type="checkbox" checked={draft.paper_autotrade} onChange={(event) => setDraft({ ...draft, paper_autotrade: event.target.checked })} />Paper auto-trade</label>
                <label className="v2-toggle"><input type="checkbox" checked={draft.ctrader_autotrade} onChange={(event) => setDraft({ ...draft, ctrader_autotrade: event.target.checked })} />cTrader auto-trade</label>
                <label className="v2-toggle"><input type="checkbox" checked={draft.require_stops} onChange={(event) => setDraft({ ...draft, require_stops: event.target.checked })} />Require protective stops</label>
                <label className="v2-toggle system-settings-full"><input type="checkbox" checked={draft.session_filter_enabled} onChange={(event) => setDraft({ ...draft, session_filter_enabled: event.target.checked })} />Restrict trading session</label>

                <label>Minimum signal strength<input className="ta-input" type="number" min="0" max="1" step="0.05" value={draft.min_confidence} onChange={updateNumber('min_confidence', 0.6, 0, 1)} /><small>Strategy strength, not a calibrated win probability.</small></label>
                <label>Risk per trade (%)<input className="ta-input" type="number" min="0.01" max="5" step="0.05" value={draft.risk_per_trade_pct} onChange={updateNumber('risk_per_trade_pct', 0.5, 0.01, 5)} /></label>
                <label>Daily loss limit (%)<input className="ta-input" type="number" min="0.1" max="20" step="0.1" value={draft.daily_loss_limit_pct} onChange={updateNumber('daily_loss_limit_pct', 2, 0.1, 20)} /></label>
                <label>Maximum daily trades<input className="ta-input" type="number" min="1" max="100" step="1" value={draft.max_daily_trades} onChange={updateNumber('max_daily_trades', 12, 1, 100)} /></label>
                <label>Maximum open positions<input className="ta-input" type="number" min="1" max="20" step="1" value={draft.max_open_positions} onChange={updateNumber('max_open_positions', 3, 1, 20)} /></label>
                <label>Maximum positions per symbol<input className="ta-input" type="number" min="1" max="20" step="1" value={draft.max_positions_per_symbol} onChange={updateNumber('max_positions_per_symbol', 1, 1, 20)} /></label>
                <label>Cooldown (minutes)<input className="ta-input" type="number" min="0" max="1440" step="1" value={draft.cooldown_minutes} onChange={updateNumber('cooldown_minutes', 30, 0, 1440)} /></label>
                <label>Session start (UTC)<input className="ta-input" type="number" min="0" max="23" step="1" value={draft.session_start_hour_utc} onChange={updateNumber('session_start_hour_utc', 6, 0, 23)} disabled={!draft.session_filter_enabled} /></label>
                <label>Session end (UTC)<input className="ta-input" type="number" min="0" max="23" step="1" value={draft.session_end_hour_utc} onChange={updateNumber('session_end_hour_utc', 21, 0, 23)} disabled={!draft.session_filter_enabled} /></label>
                <label className="system-settings-full">Operator note<textarea className="ta-input" rows={2} value={draft.operator_note} onChange={(event) => setDraft({ ...draft, operator_note: event.target.value })} /></label>
              </div>
            </div>
            <div className="system-modal-actions">
              <button className="ta-btn" type="button" onClick={() => setShowSafetyEditor(false)} disabled={busy !== ''}>Cancel</button>
              <button className="ta-btn ta-btn--primary" type="button" onClick={() => void save()} disabled={busy !== ''}>{busy === 'save' ? 'Saving…' : 'Save settings'}</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
