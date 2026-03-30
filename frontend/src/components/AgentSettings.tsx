import { useEffect, useState, type FormEvent } from 'react';

import {
  getV2Status,
  getV2Strategies,
  getV2Symbols,
  setV2Config,
  type V2Config,
  type V2StrategyInfo,
  type V2WatchlistItem,
} from '../services/api';

interface AgentSettingsProps {
  isOpen: boolean;
  onClose: () => void;
}

const TIMEFRAMES = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1'];

const defaultWatchItem = (strategy = 'sma_cross'): V2WatchlistItem => ({
  symbol: 'XAUUSD',
  timeframe: 'M5',
  strategy,
  enabled: true,
  params: {},
});

function Toggle({ label, checked, onChange }: { label: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <div className="ta-settings-toggle-row">
      <span className="ta-settings-label">{label}</span>
      <label className="ta-toggle">
        <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
        <span className="ta-toggle__track" />
      </label>
    </div>
  );
}

export default function AgentSettings({ isOpen, onClose }: AgentSettingsProps) {
  const [config, setConfig] = useState<V2Config | null>(null);
  const [configLoading, setConfigLoading] = useState(true);
  const [symbolsLoading, setSymbolsLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [availableStrategies, setAvailableStrategies] = useState<V2StrategyInfo[]>([]);
  const [availableSymbols, setAvailableSymbols] = useState<string[]>([]);
  const [memoizedOptions, setMemoizedOptions] = useState<string[]>([]);

  useEffect(() => {
    if (!isOpen) return;
    
    // Phase 1: Main Settings (Fast)
    setConfigLoading(true);
    Promise.all([getV2Status(), getV2Strategies()])
      .then(([s, strats]) => {
        setConfig(s.config);
        setAvailableStrategies(strats);
        setConfigLoading(false);
      })
      .catch((err) => {
        console.error('[AgentSettings] Config load failed:', err);
        setError('Failed to load system config.');
        setConfigLoading(false);
      });

    // Phase 2: Symbols (Heavy)
    setSymbolsLoading(true);
    getV2Symbols()
      .then((syms) => {
        const list = Array.isArray(syms.symbols) ? syms.symbols : [];
        setAvailableSymbols(list);
        // Pre-calculate top 200 once to save render cycles
        setMemoizedOptions(list.slice(0, 200));
        setSymbolsLoading(false);
      })
      .catch((err) => {
        console.error('[AgentSettings] Symbols load failed:', err);
        setSymbolsLoading(false);
      });
  }, [isOpen]);

  const updateNum = <K extends keyof V2Config>(key: K, fallback: number, min?: number, max?: number) =>
    (value: string) => {
      if (!config) return;
      let v = Number(value);
      if (!Number.isFinite(v)) v = fallback;
      if (typeof min === 'number') v = Math.max(min, v);
      if (typeof max === 'number') v = Math.min(max, v);
      setConfig({ ...config, [key]: v });
    };

  const handleSave = async (event: FormEvent) => {
    event.preventDefault();
    if (!config) return;
    setError(null);
    setSaving(true);
    try { await setV2Config(config); onClose(); }
    catch (e) { setError(e instanceof Error ? e.message : 'Save failed.'); }
    finally { setSaving(false); }
  };

  const updateWatchItem = (index: number, patch: Partial<V2WatchlistItem>) => {
    if (!config) return;
    const next = config.watchlist.map((item, i) => i !== index ? item : { ...item, ...patch });
    setConfig({ ...config, watchlist: next });
  };

  const addWatchlistItem = () => {
    if (!config) return;
    setConfig({ ...config, watchlist: [...config.watchlist, defaultWatchItem(config.default_strategy)] });
  };

  const removeWatchlistItem = (index: number) => {
    if (!config) return;
    setConfig({ ...config, watchlist: config.watchlist.filter((_, i) => i !== index) });
  };

  if (!isOpen) return null;

  return (
    <div className="ta-overlay">
      <div className="ta-overlay__backdrop" onClick={onClose} />
      <div className="ta-overlay__panel">
        <div className="ta-overlay__header">
          <h2 className="ta-overlay__title">Control Panel</h2>
          <button className="ta-btn ta-btn--ghost ta-btn--icon" type="button" onClick={onClose}>✕</button>
        </div>

        {configLoading && <div style={{ color: 'var(--ta-text-muted)', padding: '20px' }}>Syncing with engine…</div>}
        {error && <div className="ta-error-alert" style={{ margin: '0 20px 20px' }}>{error}</div>}

        {!configLoading && config && (
          <div className="ta-overlay__body">
            <form onSubmit={handleSave}>
              {/* Toggles */}
              <div className="ta-settings-section">Engine Controls</div>
              <Toggle label="Enable engine" checked={config.enabled} onChange={(v) => setConfig({ ...config, enabled: v })} />
              <Toggle label="Paper autotrade" checked={config.paper_autotrade} onChange={(v) => setConfig({ ...config, paper_autotrade: v })} />
              <Toggle label="Kill switch" checked={config.kill_switch} onChange={(v) => setConfig({ ...config, kill_switch: v })} />
              <Toggle label="Require stops" checked={config.require_stops} onChange={(v) => setConfig({ ...config, require_stops: v })} />
              <Toggle label="Session filter" checked={config.session_filter_enabled} onChange={(v) => setConfig({ ...config, session_filter_enabled: v })} />

              {/* Parameters */}
              <div className="ta-settings-section">Parameters</div>
              <div className="ta-settings-grid">
                <div className="ta-settings-field">
                  <label className="ta-settings-label">Default strategy</label>
                  <select className="ta-select" value={config.default_strategy} onChange={(e) => setConfig({ ...config, default_strategy: e.target.value })}>
                    {availableStrategies.map((s) => <option key={s.key} value={s.key}>{s.label}</option>)}
                  </select>
                </div>
                <div className="ta-settings-field">
                  <label className="ta-settings-label">Scan interval (s)</label>
                  <input className="ta-input ta-input--mono" type="number" min="2" max="300" value={config.scan_interval_sec} onChange={(e) => updateNum('scan_interval_sec', 10, 2, 300)(e.target.value)} />
                </div>
                <div className="ta-settings-field">
                  <label className="ta-settings-label">Min confidence</label>
                  <input className="ta-input ta-input--mono" type="number" min="0.1" max="1" step="0.05" value={config.min_confidence} onChange={(e) => updateNum('min_confidence', 0.6, 0.1, 1)(e.target.value)} />
                </div>
                <div className="ta-settings-field">
                  <label className="ta-settings-label">Trade size</label>
                  <input className="ta-input ta-input--mono" type="number" min="0.1" max="100" step="0.1" value={config.paper_trade_size} onChange={(e) => updateNum('paper_trade_size', 1, 0.1, 100)(e.target.value)} />
                </div>
                <div className="ta-settings-field">
                  <label className="ta-settings-label">Risk %</label>
                  <input className="ta-input ta-input--mono" type="number" min="0.1" max="10" step="0.1" value={config.risk_per_trade_pct} onChange={(e) => updateNum('risk_per_trade_pct', 0.5, 0.1, 10)(e.target.value)} />
                </div>
                <div className="ta-settings-field">
                  <label className="ta-settings-label">Daily loss cap %</label>
                  <input className="ta-input ta-input--mono" type="number" min="0.5" max="10" step="0.5" value={config.daily_loss_limit_pct} onChange={(e) => updateNum('daily_loss_limit_pct', 2, 0.5, 10)(e.target.value)} />
                </div>
                <div className="ta-settings-field">
                  <label className="ta-settings-label">Max daily trades</label>
                  <input className="ta-input ta-input--mono" type="number" min="1" max="100" value={config.max_daily_trades} onChange={(e) => updateNum('max_daily_trades', 12, 1, 100)(e.target.value)} />
                </div>
                <div className="ta-settings-field">
                  <label className="ta-settings-label">Max open positions</label>
                  <input className="ta-input ta-input--mono" type="number" min="1" max="20" value={config.max_open_positions} onChange={(e) => updateNum('max_open_positions', 3, 1, 20)(e.target.value)} />
                </div>
                <div className="ta-settings-field">
                  <label className="ta-settings-label">Per-symbol cap</label>
                  <input className="ta-input ta-input--mono" type="number" min="1" max="10" value={config.max_positions_per_symbol} onChange={(e) => updateNum('max_positions_per_symbol', 1, 1, 10)(e.target.value)} />
                </div>
                <div className="ta-settings-field">
                  <label className="ta-settings-label">Cooldown (min)</label>
                  <input className="ta-input ta-input--mono" type="number" min="0" max="1440" step="5" value={config.cooldown_minutes} onChange={(e) => updateNum('cooldown_minutes', 30, 0, 1440)(e.target.value)} />
                </div>
              </div>

              {/* Operator note */}
              <div className="ta-settings-field ta-settings-field--full" style={{ marginTop: '16px' }}>
                <label className="ta-settings-label">Operator note</label>
                <textarea
                  className="ta-input"
                  rows={2}
                  value={config.operator_note}
                  onChange={(e) => setConfig({ ...config, operator_note: e.target.value })}
                  style={{ resize: 'vertical', width: '100%', fontSize: '13px' }}
                />
              </div>

              {/* Watchlist */}
              <div className="ta-settings-section">Watchlist</div>
              {config.watchlist.map((entry, index) => (
                <div key={`${index}-${entry.symbol}`} className="ta-watch-item">
                  <label className="ta-toggle" style={{ width: '36px' }}>
                    <input type="checkbox" checked={entry.enabled} onChange={(e) => updateWatchItem(index, { enabled: e.target.checked })} />
                    <span className="ta-toggle__track" />
                  </label>
                  
                  <select 
                    className="ta-select ta-select--sm" 
                    value={entry.symbol} 
                    onChange={(e) => updateWatchItem(index, { symbol: e.target.value })}
                    disabled={symbolsLoading && !memoizedOptions.length}
                  >
                    {[
                      entry.symbol,
                      ...(memoizedOptions.length ? memoizedOptions : ['XAUUSD']).filter(s => s !== entry.symbol)
                    ].map((s) => (
                      <option key={s} value={s}>{s}</option>
                    ))}
                  </select>

                  <select className="ta-select ta-select--sm" value={entry.timeframe} onChange={(e) => updateWatchItem(index, { timeframe: e.target.value })}>
                    {TIMEFRAMES.map((tf) => <option key={tf} value={tf}>{tf}</option>)}
                  </select>
                  <select className="ta-select ta-select--sm" value={entry.strategy} onChange={(e) => updateWatchItem(index, { strategy: e.target.value })}>
                    {availableStrategies.map((s) => <option key={s.key} value={s.key}>{s.label}</option>)}
                  </select>
                  <button className="ta-btn ta-btn--danger ta-btn--sm ta-btn--icon" type="button" onClick={() => removeWatchlistItem(index)}>✕</button>
                </div>
              ))}
              
              {symbolsLoading && <div style={{ fontSize: '11px', color: 'var(--ta-text-muted)', marginTop: '4px' }}>Updating symbol list…</div>}

              <button className="ta-btn ta-btn--sm" type="button" onClick={addWatchlistItem} style={{ marginTop: '12px' }}>
                + Add Instrument
              </button>

              {/* Actions */}
              <div style={{ display: 'flex', gap: '10px', justifyContent: 'flex-end', marginTop: '24px', paddingTop: '16px', borderTop: '1px solid var(--ta-border-dim)' }}>
                <button className="ta-btn" type="button" onClick={onClose}>Cancel</button>
                <button className="ta-btn ta-btn--primary" type="submit" disabled={saving}>
                  {saving ? 'Saving…' : 'Save Changes'}
                </button>
              </div>
            </form>
          </div>
        )}
      </div>
    </div>
  );
}
