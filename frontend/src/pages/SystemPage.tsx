import { useEffect, useMemo, useState, type ChangeEvent } from 'react';

import AppNav from '../components/AppNav';
import {
  getV2CTraderAccounts,
  getV2Status,
  reconcileV2Engine,
  recoverV2Engine,
  restartV2Engine,
  scanV2Engine,
  selectV2CTraderAccount,
  setV2Config,
  setV2LiveTradingArm,
  startV2Engine,
  stopV2Engine,
  type V2Config,
  type V2CTraderAccount,
  type V2Status,
} from '../services/api';
import { formatBackendLocalDateTime } from '../utils/datetime';

const formatTime = (value?: string | null) => formatBackendLocalDateTime(value);

export default function SystemPage() {
  const [status, setStatus] = useState<V2Status | null>(null);
  const [accounts, setAccounts] = useState<V2CTraderAccount[]>([]);
  const [draft, setDraft] = useState<V2Config | null>(null);
  const [busy, setBusy] = useState<'save' | 'account' | 'live-arm' | 'engine' | 'restart' | 'scan' | 'recover' | 'reconcile' | ''>('');
  const [error, setError] = useState('');

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
      await load(kind === 'save' || kind === 'account');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'System operation failed.');
    } finally {
      setBusy('');
    }
  };

  const toggleEngine = () => run('engine', () => status?.config.enabled ? stopV2Engine() : startV2Engine());
  const save = () => draft ? run('save', () => setV2Config(draft)) : Promise.resolve();
  const selectedAccount = accounts.find((account) => account.selected);
  const activeAccount = accounts.find((account) => account.active);
  const accountLabel = (account: V2CTraderAccount) => {
    const broker = account.broker_title || 'cTrader';
    const login = account.trader_login ?? account.account_id;
    return `${broker} · ${account.account_type === 'live' ? 'Live' : 'Demo'} · ${login}`;
  };
  const selectAccount = (event: ChangeEvent<HTMLSelectElement>) => {
    const accountId = Number(event.target.value);
    if (!Number.isInteger(accountId) || accountId <= 0) return;
    void run('account', () => selectV2CTraderAccount(accountId));
  };
  const toggleLiveTradingArm = () => {
    const next = !(status?.live_trading_armed ?? false);
    void run('live-arm', () => setV2LiveTradingArm(next));
  };

  return (
    <div className="ta-app">
      <AppNav
        right={(
          <>
            <span className="ta-status"><span className={`ta-status__dot ta-status__dot--${status?.broker.socket_connected ? 'ok' : 'bad'}`} />Broker {status?.broker.socket_connected ? 'connected' : 'disconnected'}</span>
            <span className="ta-status"><span className={`ta-status__dot ta-status__dot--${status?.runtime.loop_active ? 'ok' : 'wait'}`} />Engine {status?.runtime.loop_active ? 'scanning' : 'not scanning'}</span>
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
            <div className="v2-hero-stat"><span>Account</span><strong>{activeAccount ? accountLabel(activeAccount) : status?.broker.account_switch_in_progress ? 'Switching…' : 'Not connected'}</strong></div>
            <div className="v2-hero-stat"><span>Market data</span><strong>{status?.broker.market_data_ready ? 'Ready' : 'Waiting'}</strong></div>
            <div className="v2-hero-stat"><span>Kill switch</span><strong>{status?.config.kill_switch ? 'Active' : 'Inactive'}</strong></div>
          </div>
        </section>

        {error && <div className="v2-banner v2-banner-bad">{error}</div>}

        <section className="v2-panel">
          <div className="v2-panel-head">
            <div>
              <h2>cTrader account</h2>
              <span>Choose from the accounts authorized by the current cTrader access token.</span>
            </div>
          </div>
          <div className="v2-form-grid">
            <label>
              Selected account
              <select
                value={selectedAccount?.account_id ?? ''}
                onChange={selectAccount}
                disabled={busy !== '' || accounts.length === 0 || status?.broker.account_switch_in_progress}
              >
                {!selectedAccount && <option value="">Select an account</option>}
                {accounts.map((account) => (
                  <option value={account.account_id} key={account.account_id}>
                    {accountLabel(account)}{account.active ? ' · Active' : ''}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <div className="v2-notes">
            <div>Selected: {selectedAccount ? accountLabel(selectedAccount) : 'none'}</div>
            <div>Active transport: {activeAccount ? accountLabel(activeAccount) : 'not authenticated'}</div>
            {status?.broker.account_switch_in_progress && (
              <div>Switching transport and re-authenticating account {status.broker.account_switch_target_id ?? selectedAccount?.account_id ?? '—'}…</div>
            )}
            {status?.broker.account_switch_error && (
              <div>Last account switch error: {status.broker.account_switch_error}</div>
            )}
            {selectedAccount && !selectedAccount.active && !status?.broker.account_switch_in_progress && (
              <div>Selection is saved but not active. Retry the selection or restart TradeAgent to re-establish the broker transport.</div>
            )}
            {!accounts.length && <div>No authorized cTrader accounts are currently available from the connected session.</div>}
          </div>
        </section>

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
              <button className="btn" type="button" onClick={() => run('restart', restartV2Engine)} disabled={busy !== ''}>{busy === 'restart' ? 'Restarting…' : 'Restart engine'}</button>
              <button className={`btn ${status?.config.enabled ? 'danger' : 'primary'}`} type="button" onClick={toggleEngine} disabled={busy !== ''}>{busy === 'engine' ? 'Updating…' : status?.config.enabled ? 'Stop engine' : 'Start engine'}</button>
            </div>
          </div>

          <h3 style={{ margin: '16px 0 10px' }}>Current status</h3>
          <div className="v2-checklist">
            {(status?.status_truth ?? []).map((item) => (
              <div className={`v2-check ${item.ok ? 'good' : 'bad'}`} key={item.name}>
                <strong>{item.name.replaceAll('_', ' ')}</strong>
                <span>{item.detail}</span>
              </div>
            ))}
          </div>
          <h3 style={{ margin: '18px 0 10px' }}>Readiness gates</h3>
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
              <span>Paper execution and cTrader execution are separate controls. Live accounts require an explicit runtime-only Live Trading arm in addition to cTrader auto-trade, the selected active account, and the normal kill-switch/risk gates.</span>
            </div>
            <button className="btn primary" type="button" onClick={save} disabled={!draft || busy !== ''}>{busy === 'save' ? 'Saving…' : 'Save safety settings'}</button>
          </div>

          {draft && (
            <>
              <div className="v2-form-grid v2-form-grid-wide">
                <label className="v2-toggle"><input type="checkbox" checked={draft.kill_switch} onChange={(event) => setDraft({ ...draft, kill_switch: event.target.checked })} />Kill switch</label>
                <label className="v2-toggle"><input type="checkbox" checked={draft.paper_autotrade} onChange={(event) => setDraft({ ...draft, paper_autotrade: event.target.checked })} />Paper autotrade</label>
                <label className="v2-toggle"><input type="checkbox" checked={draft.ctrader_autotrade} onChange={(event) => setDraft({ ...draft, ctrader_autotrade: event.target.checked })} />cTrader auto-trade</label>
                <label className="v2-toggle"><input type="checkbox" checked={draft.require_stops} onChange={(event) => setDraft({ ...draft, require_stops: event.target.checked })} />Require protective stops</label>
                <label className="v2-toggle"><input type="checkbox" checked={draft.session_filter_enabled} onChange={(event) => setDraft({ ...draft, session_filter_enabled: event.target.checked })} />Restrict trading session</label>
                <label>
                  <span>
                    Minimum signal strength{' '}
                    <span
                      title="Deterministic strategy-strength threshold used by the execution gate. It is not a calibrated win probability; signals below this value are rejected."
                      aria-label="Signal strength threshold information"
                    >
                      ⓘ
                    </span>
                  </span>
                  <input
                    type="number"
                    min="0"
                    max="1"
                    step="0.05"
                    value={draft.min_confidence}
                    onChange={updateNumber('min_confidence', 0.6, 0, 1)}
                    title="Signals below this deterministic strategy-strength threshold are rejected."
                  />
                  <small style={{ color: 'var(--ta-text-muted)' }}>Strategy-strength threshold; not a win probability.</small>
                </label>
                <label>Risk per trade (%)<input type="number" min="0.01" max="5" step="0.05" value={draft.risk_per_trade_pct} onChange={updateNumber('risk_per_trade_pct', 0.5, 0.01, 5)} /></label>
                <label>Daily loss limit (%)<input type="number" min="0.1" max="20" step="0.1" value={draft.daily_loss_limit_pct} onChange={updateNumber('daily_loss_limit_pct', 2, 0.1, 20)} /></label>
                <label>Maximum daily trades<input type="number" min="1" max="100" step="1" value={draft.max_daily_trades} onChange={updateNumber('max_daily_trades', 12, 1, 100)} /></label>
                <label>Maximum open positions<input type="number" min="1" max="20" step="1" value={draft.max_open_positions} onChange={updateNumber('max_open_positions', 3, 1, 20)} /></label>
                <label>Maximum positions per symbol<input type="number" min="1" max="20" step="1" value={draft.max_positions_per_symbol} onChange={updateNumber('max_positions_per_symbol', 1, 1, 20)} /></label>
                <label>Cooldown (minutes)<input type="number" min="0" max="1440" step="1" value={draft.cooldown_minutes} onChange={updateNumber('cooldown_minutes', 30, 0, 1440)} /></label>
                <label>Session start (UTC)<input type="number" min="0" max="23" step="1" value={draft.session_start_hour_utc} onChange={updateNumber('session_start_hour_utc', 6, 0, 23)} disabled={!draft.session_filter_enabled} /></label>
                <label>Session end (UTC)<input type="number" min="0" max="23" step="1" value={draft.session_end_hour_utc} onChange={updateNumber('session_end_hour_utc', 21, 0, 23)} disabled={!draft.session_filter_enabled} /></label>
              </div>
              <div className={`v2-banner ${status?.live_trading_armed ? 'v2-banner-bad' : 'v2-banner-good'}`} style={{ marginTop: 14 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 16, flexWrap: 'wrap' }}>
                  <div>
                    <strong>
                      {status?.live_trading_armed
                        ? `LIVE TRADING ARMED · account ${status.live_trading_armed_account_id ?? '—'}`
                        : 'Live Trading disarmed'}
                    </strong>
                    <div style={{ marginTop: 4 }}>
                      {activeAccount?.account_type === 'live'
                        ? `Active Live account: ${accountLabel(activeAccount)}. New real-money entries require this arm in addition to the saved cTrader auto-trade setting.`
                        : 'The active account is Demo, so Live arming is not required. Live arming resets on account changes, engine restart, and backend restart.'}
                    </div>
                  </div>
                  <button
                    className={`btn ${status?.live_trading_armed ? 'danger' : 'primary'}`}
                    type="button"
                    onClick={toggleLiveTradingArm}
                    disabled={
                      busy !== ''
                      || (
                        !status?.live_trading_armed
                        && (
                          status?.config.enabled
                          || status?.config.ctrader_autotrade !== true
                          || activeAccount?.account_type !== 'live'
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
                </div>
                {!status?.live_trading_armed && activeAccount?.account_type === 'live' && (
                  <div style={{ marginTop: 6 }}>
                    To arm: stop the engine, make this Live account both selected and active, save cTrader auto-trade, then arm it explicitly.
                  </div>
                )}
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
            <div className="v2-panel-head"><h2>Current incidents</h2></div>
            <div className="v2-feed">
              {(status?.active_incidents ?? []).map((item) => (
                <article className="v2-feed-card" key={item.code}>
                  <div className="v2-feed-head"><strong>{item.level} · {item.code}</strong><span>current</span></div>
                  <p>{item.message}</p>
                </article>
              ))}
              {!status?.active_incidents.length && <div className="v2-empty">No current incidents.</div>}
            </div>
          </section>

          <section className="v2-panel">
            <div className="v2-panel-head"><h2>Incident history</h2></div>
            <div className="v2-feed">
              {(status?.recent_incidents ?? []).slice(0, 10).map((item) => (
                <article className="v2-feed-card" key={item.id}>
                  <div className="v2-feed-head"><strong>{item.level} · {item.code}</strong><span>{formatTime(item.created_at)}</span></div>
                  <p>{item.message}</p>
                </article>
              ))}
              {!status?.recent_incidents.length && <div className="v2-empty">No incident history.</div>}
            </div>
          </section>
        </div>
      </main>
    </div>
  );
}
