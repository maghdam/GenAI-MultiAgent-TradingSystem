import { useEffect, useState, type FormEvent } from 'react';
import { getAgentConfig, setAgentConfig, getAgentStatus, type AgentConfig } from '../services/api';

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
  lot_size_lots: 0.1,
  strategy: 'smc',
};

export default function AgentSettings({ isOpen, onClose }: AgentSettingsProps) {
  const [config, setConfig] = useState<AgentConfig>(initialConfig);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [availableStrategies, setAvailableStrategies] = useState<string[]>(['smc','rsi']);

  useEffect(() => {
    if (!isOpen) return;
    setLoading(true);
    Promise.all([getAgentConfig(), getAgentStatus()])
      .then(([cfg, status]) => {
        setConfig(cfg);
        const opts = Array.from(new Set([...(status?.available_strategies || []), 'smc', 'rsi', cfg.strategy].filter(Boolean))) as string[];
        setAvailableStrategies(opts);
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

  const handleWatchlistChange = (index: number, value: string, field: 'symbol' | 'tf') => {
    const newWatchlist = [...config.watchlist];
    if (field === 'symbol') {
      newWatchlist[index][0] = value;
    } else {
      newWatchlist[index][1] = value;
    }
    setConfig({ ...config, watchlist: newWatchlist });
  };

  const addWatchlistItem = () => {
    setConfig({ ...config, watchlist: [...config.watchlist, ['', '']] });
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
            <label>Lot Size</label>
            <input
              type="number"
              min="0.01"
              step="0.01"
              value={config.lot_size_lots}
              onChange={e => setConfig({ ...config, lot_size_lots: parseFloat(e.target.value) })}
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
          {config.watchlist.map(([symbol, tf], index) => (
            <div key={index} className="stack" style={{ marginBottom: '8px' }}>
              <input
                type="text"
                placeholder="Symbol"
                value={symbol}
                onChange={e => handleWatchlistChange(index, e.target.value, 'symbol')}
              />
              <input
                type="text"
                placeholder="Timeframe"
                value={tf}
                onChange={e => handleWatchlistChange(index, e.target.value, 'tf')}
                style={{ width: '100px' }}
              />
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
