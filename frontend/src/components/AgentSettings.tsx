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

  const handleWatchlistChange = (index: number, field: 'symbol' | 'timeframe' | 'lot_size', value: string) => {
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
      const parsed = parseFloat(value);
      return { ...item, lot_size: Number.isFinite(parsed) ? parsed : item.lot_size };
    });
    setConfig({ ...config, watchlist: updated });
  };

  const addWatchlistItem = () => {
    const fallbackLot = Number.isFinite(config.lot_size_lots) ? config.lot_size_lots : 0.01;
    setConfig({
      ...config,
      watchlist: [...config.watchlist, { symbol: '', timeframe: '', lot_size: fallbackLot }],
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
            <label>Strategy</label>
            <select value={config.strategy} onChange={e => setConfig({ ...config, strategy: e.target.value })}>
              {availableStrategies.map(name => (
                <option key={name} value={name}>{name.toUpperCase()}</option>
              ))}
            </select>
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
