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

const TIMEFRAMES = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1'];

const newWatchItem = (strategy: string, lotSize: number): V2WatchlistItem => ({
  symbol: 'XAUUSD',
  timeframe: 'M5',
  strategy,
  enabled: true,
  trading_enabled: false,
  lot_size: lotSize,
  params: {},
});

export default function TradeSettings({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) {
  const [config, setConfig] = useState<V2Config | null>(null);
  const [strategies, setStrategies] = useState<V2StrategyInfo[]>([]);
  const [symbols, setSymbols] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const [demoConfirmed, setDemoConfirmed] = useState(false);

  useEffect(() => {
    if (!isOpen) return;
    setLoading(true);
    setError('');
    Promise.all([getV2Status(), getV2Strategies(), getV2Symbols()])
      .then(([status, strategyList, symbolPayload]) => {
        setConfig({
          ...status.config,
          watchlist: status.config.watchlist.map((item) => ({
            ...item,
            lot_size: item.lot_size ?? status.config.paper_trade_size,
          })),
        });
        setDemoConfirmed(status.broker.demo_account_confirmed);
        setStrategies(strategyList);
        setSymbols((symbolPayload.symbols || []).slice(0, 500));
      })
      .catch((err) => setError(err instanceof Error ? err.message : 'Trade settings unavailable.'))
      .finally(() => setLoading(false));
  }, [isOpen]);

  const updateWatchItem = (index: number, patch: Partial<V2WatchlistItem>) => {
    if (!config) return;
    setConfig({
      ...config,
      watchlist: config.watchlist.map((item, itemIndex) => itemIndex === index ? { ...item, ...patch } : item),
    });
  };

  const save = async (event: FormEvent) => {
    event.preventDefault();
    if (!config) return;
    setSaving(true);
    setError('');
    try {
      await setV2Config(config);
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Save failed.');
    } finally {
      setSaving(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="ta-overlay">
      <div className="ta-overlay__backdrop" onClick={onClose} />
      <div className="ta-overlay__panel">
        <div className="ta-overlay__header">
          <div>
            <h2 className="ta-overlay__title">Trade setup</h2>
            <div className="mi-muted">Choose per-symbol signal generation, strategy, lot size, and demo-order permission.</div>
          </div>
          <button className="ta-btn ta-btn--ghost ta-btn--icon" type="button" onClick={onClose}>✕</button>
        </div>

        {loading && <div style={{ color: 'var(--ta-text-muted)', padding: 20 }}>Loading trade setup…</div>}
        {error && <div className="ta-error-alert" style={{ margin: '0 20px 20px' }}>{error}</div>}

        {!loading && config && (
          <div className="ta-overlay__body">
            <form onSubmit={save}>
              <div className="ta-settings-section">Default analysis</div>
              <div className="ta-settings-field">
                <label className="ta-settings-label">Default strategy</label>
                <select className="ta-select" value={config.default_strategy} onChange={(event) => setConfig({ ...config, default_strategy: event.target.value })}>
                  {strategies.map((strategy) => <option key={strategy.key} value={strategy.key}>{strategy.label}</option>)}
                </select>
              </div>

              <div className="ta-settings-section">System controls</div>
              <div className="ta-settings-field" style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                <label className="ta-watch-control" style={{ gap: 10, fontSize: 14 }}>
                  <input
                    type="checkbox"
                    checked={config.enabled}
                    onChange={(e) => setConfig({ ...config, enabled: e.target.checked })}
                  />
                  <span>
                    <strong>Engine enabled</strong>
                    <span style={{ display: 'block', fontSize: 12, color: 'var(--ta-text-muted)' }}>
                      Allow the engine to scan watchlist and generate signals.
                    </span>
                  </span>
                </label>

                <label
                  className="ta-watch-control"
                  style={{ gap: 10, fontSize: 14, padding: '8px 10px', borderRadius: 6, background: config.kill_switch ? 'rgba(239,68,68,0.12)' : 'transparent', border: config.kill_switch ? '1px solid rgba(239,68,68,0.35)' : '1px solid transparent' }}
                >
                  <input
                    type="checkbox"
                    checked={config.kill_switch}
                    onChange={(e) => setConfig({ ...config, kill_switch: e.target.checked })}
                  />
                  <span>
                    <strong style={{ color: config.kill_switch ? '#f87171' : undefined }}>
                      {config.kill_switch ? '🔴 Kill switch ON — all orders blocked' : '🟢 Kill switch OFF'}
                    </strong>
                    <span style={{ display: 'block', fontSize: 12, color: 'var(--ta-text-muted)' }}>
                      When ON, no orders are placed regardless of any other setting. Turn OFF to allow trading.
                    </span>
                  </span>
                </label>

                <label
                  className="ta-watch-control"
                  style={{ gap: 10, fontSize: 14, padding: '8px 10px', borderRadius: 6, background: config.demo_autotrade ? 'rgba(34,197,94,0.10)' : 'transparent', border: config.demo_autotrade ? '1px solid rgba(34,197,94,0.30)' : '1px solid transparent' }}
                >
                  <input
                    type="checkbox"
                    checked={config.demo_autotrade}
                    onChange={(e) => setConfig({ ...config, demo_autotrade: e.target.checked })}
                  />
                  <span>
                    <strong>System demo auto-trade</strong>
                    <span style={{ display: 'block', fontSize: 12, color: 'var(--ta-text-muted)' }}>
                      Allow the engine to place real orders on your cTrader demo account automatically.
                      Also requires "Demo orders" enabled per symbol below.
                    </span>
                  </span>
                </label>
              </div>

              <div className="ta-settings-section">Watchlist</div>
              <div className={demoConfirmed ? 'v2-banner v2-banner-good' : 'v2-banner v2-banner-bad'} style={{ marginBottom: 12 }}>
                {demoConfirmed
                  ? 'Connected cTrader account is confirmed as demo. Orders still require the System demo-autotrade switch.'
                  : 'Demo orders are blocked until cTrader confirms the connected account is a demo account.'}
              </div>
              {config.watchlist.map((item, index) => (
                <div className="ta-watch-item" key={`${index}-${item.symbol}-${item.timeframe}`}>
                  <label className="ta-watch-control" title="Generate signals for this row">
                    <input type="checkbox" checked={item.enabled} onChange={(event) => updateWatchItem(index, { enabled: event.target.checked })} />
                    <span>Signals</span>
                  </label>
                  <select className="ta-select ta-select--sm" value={item.symbol} onChange={(event) => updateWatchItem(index, { symbol: event.target.value })}>
                    {[item.symbol, ...symbols.filter((symbol) => symbol !== item.symbol)].map((symbol) => <option key={symbol} value={symbol}>{symbol}</option>)}
                  </select>
                  <select className="ta-select ta-select--sm" value={item.timeframe} onChange={(event) => updateWatchItem(index, { timeframe: event.target.value })}>
                    {TIMEFRAMES.map((timeframe) => <option key={timeframe} value={timeframe}>{timeframe}</option>)}
                  </select>
                  <select className="ta-select ta-select--sm" value={item.strategy} onChange={(event) => updateWatchItem(index, { strategy: event.target.value })}>
                    {strategies.map((strategy) => <option key={strategy.key} value={strategy.key}>{strategy.label}</option>)}
                  </select>
                  <input
                    className="ta-input ta-input--mono ta-input--lot"
                    type="number"
                    min="0.0001"
                    max="100"
                    step="0.01"
                    title="Order size in lots"
                    aria-label={`Lot size for ${item.symbol}`}
                    value={item.lot_size ?? config.paper_trade_size}
                    onChange={(event) => {
                      const value = Number(event.target.value);
                      if (Number.isFinite(value) && value > 0) updateWatchItem(index, { lot_size: value });
                    }}
                  />
                  <label className="ta-watch-control" title="Permit cTrader demo orders for this row">
                    <input type="checkbox" checked={item.trading_enabled} onChange={(event) => updateWatchItem(index, { trading_enabled: event.target.checked })} />
                    <span>Demo orders</span>
                  </label>
                  <button className="ta-btn ta-btn--danger ta-btn--sm ta-btn--icon" type="button" onClick={() => setConfig({ ...config, watchlist: config.watchlist.filter((_, itemIndex) => itemIndex !== index) })}>✕</button>
                </div>
              ))}
              <button className="ta-btn ta-btn--sm" type="button" onClick={() => setConfig({ ...config, watchlist: [...config.watchlist, newWatchItem(config.default_strategy, config.paper_trade_size)] })}>+ Add instrument</button>

              <div style={{ display: 'flex', gap: 10, justifyContent: 'flex-end', marginTop: 24, paddingTop: 16, borderTop: '1px solid var(--ta-border-dim)' }}>
                <button className="ta-btn" type="button" onClick={onClose}>Cancel</button>
                <button className="ta-btn ta-btn--primary" type="submit" disabled={saving}>{saving ? 'Saving…' : 'Save trade setup'}</button>
              </div>
            </form>
          </div>
        )}
      </div>
    </div>
  );
}
