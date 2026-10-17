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
import {
  CTRADER_LOT_INPUT_MIN,
  CTRADER_LOT_INPUT_STEP,
  formatCTraderLotInput,
  normalizeCTraderLotInput,
} from '../services/lotSizeInput';

const TIMEFRAMES = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1'];

const newWatchItem = (strategy: string, lotSize: number): V2WatchlistItem => ({
  symbol: 'XAUUSD',
  timeframe: 'M5',
  strategy,
  enabled: true,
  trading_enabled: false,
  venue: 'ctrader',
  alpaca_asset_class: null,
  alpaca_size_mode: null,
  alpaca_size_value: null,
  lot_size: normalizeCTraderLotInput(lotSize),
  params: {},
});

export default function TradeSettings({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) {
  const [config, setConfig] = useState<V2Config | null>(null);
  const [strategies, setStrategies] = useState<V2StrategyInfo[]>([]);
  const [symbols, setSymbols] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const [accountVerified, setAccountVerified] = useState(false);
  const [accountType, setAccountType] = useState<'demo' | 'live' | 'unknown'>('unknown');
  const [liveTradingArmed, setLiveTradingArmed] = useState(false);

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
            lot_size: normalizeCTraderLotInput(item.lot_size ?? status.config.paper_trade_size),
          })),
        });
        setAccountVerified(status.broker.account_verified);
        setAccountType(status.broker.account_type);
        setLiveTradingArmed(status.live_trading_armed);
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
            <div className="mi-muted">Choose each row's signal source, strategy, broker venue, and automatic-order permission.</div>
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
                  style={{ gap: 10, fontSize: 14, padding: '8px 10px', borderRadius: 6, background: config.ctrader_autotrade ? 'rgba(34,197,94,0.10)' : 'transparent', border: config.ctrader_autotrade ? '1px solid rgba(34,197,94,0.30)' : '1px solid transparent' }}
                >
                  <input
                    type="checkbox"
                    checked={config.ctrader_autotrade}
                    onChange={(e) => setConfig({ ...config, ctrader_autotrade: e.target.checked })}
                  />
                  <span>
                    <strong>cTrader auto-trade</strong>
                    <span style={{ display: 'block', fontSize: 12, color: 'var(--ta-text-muted)' }}>
                      Allow automatic orders only for rows whose venue is cTrader.
                      Live orders also require the separate runtime-only Live Trading arm in System plus per-symbol "Auto-trade" below.
                    </span>
                  </span>
                </label>
              </div>

              <div className="ta-settings-section">Watchlist</div>
              <div className="mi-muted" style={{ marginBottom: 10 }}>
                Broker venue is explicit per row. Alpaca rows are Paper-only and require explicit quantity or notional sizing before auto-trade can be enabled.
              </div>
              <div
                className={
                  accountVerified && accountType === 'demo'
                    ? 'v2-banner v2-banner-good'
                    : 'v2-banner v2-banner-bad'
                }
                style={{ marginBottom: 12 }}
              >
                {accountVerified
                  ? (
                    accountType === 'live'
                      ? (
                        liveTradingArmed
                          ? 'Connected account: LIVE · ARMED. Real-money entries still require cTrader auto-trade, kill switch off, and per-symbol Auto-trade.'
                          : 'Connected account: LIVE · DISARMED. New real-money entries are blocked until Live Trading is explicitly armed in System.'
                      )
                      : 'Connected account: Demo. Automatic execution still requires System auto-trade and per-symbol Auto-trade.'
                  )
                  : 'Automatic cTrader execution is blocked until the selected account is authenticated and execution-ready.'}
              </div>
              {config.watchlist.map((item, index) => (
                <div className="ta-watch-item" key={`${index}-${item.symbol}-${item.timeframe}`}>
                  <label className="ta-watch-control" title="Generate signals for this row">
                    <input type="checkbox" checked={item.enabled} onChange={(event) => updateWatchItem(index, { enabled: event.target.checked })} />
                    <span>Signals</span>
                  </label>
                  <select
                    className="ta-select ta-select--sm"
                    value={item.venue}
                    aria-label={`Broker venue for ${item.symbol}`}
                    onChange={(event) => {
                      const venue = event.target.value as V2WatchlistItem['venue'];
                      updateWatchItem(index, {
                        venue,
                        alpaca_asset_class: venue === 'alpaca' ? (item.alpaca_asset_class ?? 'us_equity') : null,
                        alpaca_size_mode: venue === 'alpaca' ? (item.alpaca_size_mode ?? 'notional') : null,
                        alpaca_size_value: venue === 'alpaca' ? (item.alpaca_size_value ?? 10) : null,
                        trading_enabled: false,
                      });
                    }}
                  >
                    <option value="ctrader">cTrader</option>
                    <option value="alpaca">Alpaca Paper</option>
                  </select>
                  {item.venue === 'alpaca' ? (
                    <input
                      className="ta-input ta-input--mono"
                      value={item.symbol}
                      aria-label="Alpaca symbol"
                      placeholder="SPY or BTC/USD"
                      onChange={(event) => updateWatchItem(index, { symbol: event.target.value.toUpperCase() })}
                    />
                  ) : (
                    <select className="ta-select ta-select--sm" value={item.symbol} onChange={(event) => updateWatchItem(index, { symbol: event.target.value })}>
                      {[item.symbol, ...symbols.filter((symbol) => symbol !== item.symbol)].map((symbol) => <option key={symbol} value={symbol}>{symbol}</option>)}
                    </select>
                  )}
                  {item.venue === 'alpaca' && (
                    <>
                      <select
                        className="ta-select ta-select--sm"
                        value={item.alpaca_asset_class ?? 'us_equity'}
                        aria-label={`Alpaca asset class for ${item.symbol}`}
                        onChange={(event) => updateWatchItem(index, { alpaca_asset_class: event.target.value as 'us_equity' | 'crypto' })}
                      >
                        <option value="us_equity">US stock / ETF</option>
                        <option value="crypto">Crypto</option>
                      </select>
                      <select
                        className="ta-select ta-select--sm"
                        value={item.alpaca_size_mode ?? 'notional'}
                        aria-label={`Alpaca size mode for ${item.symbol}`}
                        onChange={(event) => updateWatchItem(index, { alpaca_size_mode: event.target.value as 'qty' | 'notional' })}
                      >
                        <option value="notional">USD amount</option>
                        <option value="qty">Quantity</option>
                      </select>
                      <input
                        className="ta-input ta-input--mono ta-input--lot"
                        type="number"
                        min="0.000000001"
                        step="any"
                        value={item.alpaca_size_value ?? 10}
                        aria-label={`Alpaca size value for ${item.symbol}`}
                        title={item.alpaca_size_mode === 'qty' ? 'Shares or crypto quantity per entry.' : 'USD notional per long entry.'}
                        onChange={(event) => {
                          const value = Number(event.target.value);
                          updateWatchItem(index, { alpaca_size_value: Number.isFinite(value) && value > 0 ? value : null });
                        }}
                      />
                    </>
                  )}
                  <select className="ta-select ta-select--sm" value={item.timeframe} onChange={(event) => updateWatchItem(index, { timeframe: event.target.value })}>
                    {TIMEFRAMES.map((timeframe) => <option key={timeframe} value={timeframe}>{timeframe}</option>)}
                  </select>
                  <select className="ta-select ta-select--sm" value={item.strategy} onChange={(event) => updateWatchItem(index, { strategy: event.target.value })}>
                    {strategies.map((strategy) => <option key={strategy.key} value={strategy.key}>{strategy.label}</option>)}
                  </select>
                  {item.venue === 'ctrader' && (
                    <input
                      className="ta-input ta-input--mono ta-input--lot"
                      type="number"
                      min={CTRADER_LOT_INPUT_MIN}
                      step={CTRADER_LOT_INPUT_STEP}
                      title="Order size in lots (cTrader-style two-decimal entry; broker limits are validated at execution)"
                      aria-label={`Lot size for ${item.symbol}`}
                      defaultValue={formatCTraderLotInput(item.lot_size ?? config.paper_trade_size)}
                      onChange={(event) => {
                        const value = Number(event.target.value);
                        if (Number.isFinite(value) && value >= CTRADER_LOT_INPUT_MIN) {
                          updateWatchItem(index, { lot_size: value });
                        }
                      }}
                    />
                  )}
                  <label
                    className="ta-watch-control"
                    title={item.venue === 'alpaca'
                      ? 'Permit trusted-strategy automatic orders on Alpaca Paper only.'
                      : 'Permit automatic cTrader order execution for this symbol'}
                  >
                    <input
                      type="checkbox"
                      checked={item.trading_enabled}
                      onChange={(event) => updateWatchItem(index, { trading_enabled: event.target.checked })}
                    />
                    <span>{item.venue === 'alpaca' ? 'Paper auto-trade' : 'Auto-trade'}</span>
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
