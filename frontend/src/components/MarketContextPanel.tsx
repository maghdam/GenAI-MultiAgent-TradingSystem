import { useCallback, useEffect, useMemo, useState } from 'react';

import { fetchV2MarketIntelligence, type V2MarketIntelligenceResponse } from '../services/api';

const formatPrice = (value?: number | null) => {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '--';
  return Math.abs(value) >= 1000 ? value.toLocaleString(undefined, { maximumFractionDigits: 2 }) : value.toFixed(4);
};

export default function MarketContextPanel({ symbol }: { symbol: string }) {
  const [data, setData] = useState<V2MarketIntelligenceResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const load = useCallback(async () => {
    try {
      const payload = await fetchV2MarketIntelligence();
      setData(payload);
      setError('');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Market context unavailable.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
    const interval = window.setInterval(load, 60_000);
    return () => window.clearInterval(interval);
  }, [load]);

  const instrument = useMemo(
    () => data?.instruments.find((item) => item.symbol.toUpperCase() === symbol.toUpperCase()) ?? null,
    [data, symbol],
  );
  const nextEvent = data?.macro_events[0] ?? null;

  return (
    <section className="ta-panel" style={{ margin: '12px 14px 0' }}>
      <div className="ta-panel__header">
        <div>
          <div className="ta-panel__title">Market context</div>
          <div className="mi-muted">Read-only evidence for the selected market. It does not create or approve trades.</div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span className={`ta-pill ${data?.regime === 'risk_off' ? 'ta-pill--short' : data?.regime === 'risk_on' ? 'ta-pill--long' : 'ta-pill--info'}`}>
            {data?.regime?.replace('_', ' ') || (loading ? 'loading' : 'unknown')}
          </span>
          <button className="ta-btn ta-btn--sm" type="button" onClick={load} disabled={loading}>Refresh</button>
        </div>
      </div>
      <div className="ta-panel__body" style={{ display: 'grid', gridTemplateColumns: 'minmax(220px, 1.3fr) repeat(3, minmax(140px, 1fr))', gap: 14 }}>
        <div>
          <strong>{data?.headline || 'Waiting for market context.'}</strong>
          <p className="mi-muted" style={{ marginBottom: 0 }}>{data?.summary || error}</p>
        </div>
        <div>
          <div className="mi-muted">{symbol} context</div>
          <strong>{instrument?.bias || 'unavailable'}</strong>
          <div className="mi-muted">{instrument?.direction_note || 'Not in the active context watchlist.'}</div>
        </div>
        <div>
          <div className="mi-muted">Price / sample move</div>
          <strong>{formatPrice(instrument?.price)}</strong>
          <div className="mi-muted">{typeof instrument?.change_pct === 'number' ? `${instrument.change_pct >= 0 ? '+' : ''}${instrument.change_pct.toFixed(2)}%` : '--'}</div>
        </div>
        <div>
          <div className="mi-muted">Next scheduled event</div>
          <strong>{nextEvent?.title || 'None configured'}</strong>
          <div className="mi-muted">{nextEvent ? `${nextEvent.impact} impact · ${nextEvent.source}` : 'Calendar has no upcoming event.'}</div>
        </div>
      </div>
      {error && <div className="v2-banner v2-banner-bad" style={{ margin: '0 14px 12px' }}>{error}</div>}
    </section>
  );
}
