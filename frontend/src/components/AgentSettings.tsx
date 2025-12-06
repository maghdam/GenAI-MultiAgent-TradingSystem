import { useEffect, useState, type FormEvent } from 'react';
import { getAgentConfig, setAgentConfig, getAgentStatus, getSymbolLimits, type AgentConfig, type SymbolLimitsMap } from '../services/api';

interface AgentSettingsProps {
  isOpen: boolean;
  onClose: () => void;
}

const initialConfig: AgentConfig = {
  enabled: false,
  watchlist: [],
  interval_sec: 60,
  min_confidence: 0.7,
  trading_mode: 'paper',
  autotrade: false,
  lot_size_lots: 0.01,
  strategy: 'smc',
  order_type: 'MARKET',
  llm_gate_enabled: true,
  llm_gate_threshold: 3,
  risk_mode: 'atr',
  atr_len: 14,
  atr_mult: 1.0,
  rr: 2.0,
  swing_lookback: 10,
  tick_pct: 0.0005,
};

export default function AgentSettings({ isOpen, onClose }: AgentSettingsProps) {
  const [config, setConfig] = useState<AgentConfig>(initialConfig);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [availableStrategies, setAvailableStrategies] = useState<string[]>(['smc','rsi']);
  const [limits, setLimits] = useState<SymbolLimitsMap>({});

  const fmtLots = (v?: number | null) => (typeof v === 'number' && Number.isFinite(v) ? v.toFixed(2) : '—');
  const limsHas = (m: SymbolLimitsMap, sym: string) => m && Object.prototype.hasOwnProperty.call(m, sym);

  useEffect(() => {
    if (!isOpen) return;
    setLoading(true);
    Promise.all([getAgentConfig(), getAgentStatus(), getSymbolLimits() as Promise<SymbolLimitsMap>])
      .then(([cfg, status, lims]) => {
        const fallbackLot = Number.isFinite(cfg.lot_size_lots) ? cfg.lot_size_lots : 0.01;
        const sanitizedWatchlist = (cfg.watchlist || []).map(item => ({
          symbol: item.symbol || '',
          timeframe: item.timeframe || '',
          lot_size: Number.isFinite(item.lot_size) ? item.lot_size : fallbackLot,
          strategy: item.strategy || cfg.strategy || 'smc',
        }));
        setConfig({ ...cfg, watchlist: sanitizedWatchlist });
        const opts = Array.from(new Set([...(status?.available_strategies || []), 'smc', 'rsi', cfg.strategy].filter(Boolean))) as string[];
        setAvailableStrategies(opts);
        if (lims && typeof lims === 'object') setLimits(lims);
        setError(null);
      })
      .catch(err => setError(err.message))
      .finally(() => setLoading(false));
  }, [isOpen]);

  const handleSave = async (event: FormEvent) => {
    event.preventDefault();
    setError(null);
    try {
      await setAgentConfig(config);
      onClose(); // Close on successful save
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save config');
    }
  };

  const handleWatchlistChange = (index: number, field: 'symbol' | 'timeframe' | 'lot_size' | 'strategy', value: string) => {
    const updated = config.watchlist.map((item, i) => {
      if (i !== index) {
        return item;
      }
      if (field === 'symbol') {
        return { ...item, symbol: value };
      }
      if (field === 'timeframe') {
        return { ...item, timeframe: value };
      }
      if (field === 'strategy') {
        return { ...item, strategy: value };
      }
      const parsed = parseFloat(value);
      return { ...item, lot_size: Number.isFinite(parsed) ? parsed : item.lot_size };
    });
    setConfig({ ...config, watchlist: updated });
  };

  const addWatchlistItem = () => {
    const fallbackLot = Number.isFinite(config.lot_size_lots) ? config.lot_size_lots : 0.01;
    setConfig({
      ...config,
      watchlist: [...config.watchlist, { symbol: '', timeframe: '', lot_size: fallbackLot, strategy: config.strategy }],
    });
  };

  const removeWatchlistItem = (index: number) => {
    const newWatchlist = config.watchlist.filter((_, i) => i !== index);
    setConfig({ ...config, watchlist: newWatchlist });
  };

  if (!isOpen) {
    return null;
  }

  return (
    <div className="drawer" style={{ display: isOpen ? 'block' : 'none' }}>
      <h3>Agent Settings</h3>
      {loading && <div>Loading...</div>}
      {error && <div style={{ color: '#ff8a8a' }}>{error}</div>}
      {!loading && (
        <form onSubmit={handleSave}>
          <div className="kv">
            <label>Enabled</label>
            <input
              type="checkbox"
              checked={config.enabled}
              onChange={e => setConfig({ ...config, enabled: e.target.checked })}
            />
            <label>Auto-trade</label>
            <input
              type="checkbox"
              checked={config.autotrade}
              onChange={e => setConfig({ ...config, autotrade: e.target.checked })}
            />
            <label>Interval (sec)</label>
            <input
              type="number"
              min="10"
              value={config.interval_sec}
              onChange={e => setConfig({ ...config, interval_sec: parseInt(e.target.value, 10) })}
            />
            <label>Min Confidence</label>
            <input
              type="number"
              min="0.1"
              max="1.0"
              step="0.05"
              value={config.min_confidence}
              onChange={e => setConfig({ ...config, min_confidence: parseFloat(e.target.value) })}
            />
            <label>Trading Mode</label>
            <select
              value={config.trading_mode}
              onChange={e => setConfig({ ...config, trading_mode: e.target.value })}
            >
              <option value="paper">Paper</option>
              <option value="live">Live</option>
            </select>
            <label>Default Strategy</label>
            <select value={config.strategy} onChange={e => setConfig({ ...config, strategy: e.target.value })}>
              {availableStrategies.map(name => (
                <option key={name} value={name}>{name.toUpperCase()}</option>
              ))}
            </select>
            <div className="muted" style={{ fontSize: '12px' }}>Applied when a watchlist row has no strategy selected.</div>
            <label>Order Type</label>
            <select
              value={config.order_type || 'MARKET'}
              onChange={e => setConfig({ ...config, order_type: e.target.value })}
            >
              <option value="MARKET">MARKET</option>
              <option value="LIMIT">LIMIT</option>
              <option value="STOP">STOP</option>
            </select>
            <label>Gate Weak Votes</label>
            <input
              type="checkbox"
              checked={!!config.llm_gate_enabled}
              onChange={e => setConfig({ ...config, llm_gate_enabled: e.target.checked })}
            />
            <label>Gate Threshold</label>
            <input
              type="number"
              min="1"
              max="5"
              step="1"
              value={config.llm_gate_threshold ?? 3}
              onChange={e => setConfig({ ...config, llm_gate_threshold: parseInt(e.target.value || '3', 10) })}
            />
          </div>

          <h4>Watchlist</h4>
          {config.watchlist.map((entry, index) => (
            <div key={index} className="stack" style={{ marginBottom: '8px' }}>
              <input
                type="text"
                placeholder="Symbol"
                value={entry.symbol}
                onChange={e => handleWatchlistChange(index, 'symbol', e.target.value)}
              />
              <input
                type="text"
                placeholder="Timeframe"
                value={entry.timeframe}
                onChange={e => handleWatchlistChange(index, 'timeframe', e.target.value)}
                style={{ width: '100px' }}
              />
              <select
                value={entry.strategy || config.strategy}
                onChange={e => handleWatchlistChange(index, 'strategy', e.target.value)}
                style={{ width: '120px' }}
              >
                {availableStrategies.map(name => (
                  <option key={name} value={name}>{name.toUpperCase()}</option>
                ))}
              </select>
              <input
                type="number"
                min="0.01"
                step="0.01"
                placeholder="Lot size"
                value={entry.lot_size}
                onChange={e => handleWatchlistChange(index, 'lot_size', e.target.value)}
                style={{ width: '110px' }}
              />
              {(() => {
                const sym = (entry.symbol || '').toUpperCase();
                const lim = limsHas(limits, sym) ? limits[sym] : undefined;
                if (!lim) return <div className="muted" style={{ fontSize: '12px' }}>min/step unknown</div>;
                const tagMin = (lim as any).min_source === 'verified' ? '' : ' (est.)';
                const tagStep = (lim as any).step_source === 'verified' ? '' : ' (est.)';
                const txt = `min ${fmtLots((lim as any).min_lots)}${tagMin} • step ${fmtLots((lim as any).step_lots)}${tagStep}` + (lim.max_lots ? ` • max ${fmtLots(lim.max_lots)}` : '');
                return <div className="muted" style={{ fontSize: '12px' }}>{txt}</div>;
              })()}
              <button type="button" className="btn" onClick={() => removeWatchlistItem(index)}>
                ✖
              </button>
            </div>
          ))}
          <button type="button" className="btn" onClick={addWatchlistItem} style={{ marginTop: '8px' }}>
            Add to Watchlist
          </button>

          <div className="stack" style={{ marginTop: '16px', justifyContent: 'flex-end' }}>
            <div style={{ flex: 1 }} />
            <div className="kv" style={{ alignItems: 'center' }}>
              <label>Risk Mode</label>
              <select
                value={config.risk_mode || 'atr'}
                onChange={e => setConfig({ ...config, risk_mode: e.target.value })}
              >
                <option value="atr">ATR</option>
                <option value="swing">Swing</option>
              </select>
              <label>ATR Len</label>
              <input
                type="number"
                min="5"
                max="100"
                step="1"
                value={config.atr_len ?? 14}
                onChange={e => setConfig({ ...config, atr_len: parseInt(e.target.value || '14', 10) })}
              />
              <label>ATR Mult</label>
              <input
                type="number"
                min="0.5"
                max="5.0"
                step="0.1"
                value={config.atr_mult ?? 1.0}
                onChange={e => setConfig({ ...config, atr_mult: parseFloat(e.target.value || '1.0') })}
              />
              <label>RR</label>
              <input
                type="number"
                min="1.0"
                max="5.0"
                step="0.1"
                value={config.rr ?? 2.0}
                onChange={e => setConfig({ ...config, rr: parseFloat(e.target.value || '2.0') })}
              />
              <label>Swing LB</label>
              <input
                type="number"
                min="5"
                max="50"
                step="1"
                value={config.swing_lookback ?? 10}
                onChange={e => setConfig({ ...config, swing_lookback: parseInt(e.target.value || '10', 10) })}
              />
              <label>Tick %</label>
              <input
                type="number"
                min="0.0001"
                max="0.01"
                step="0.0001"
                value={config.tick_pct ?? 0.0005}
                onChange={e => setConfig({ ...config, tick_pct: parseFloat(e.target.value || '0.0005') })}
              />
            </div>
            <button type="button" className="btn" onClick={onClose}>
              Cancel
            </button>
            <button type="submit" className="btn primary">
              Save
            </button>
          </div>
        </form>
      )}
    </div>
  );
}
