import React from 'react';
import { useNavigate } from 'react-router-dom';
import { CodeDisplay } from '../../components/CodeDisplay';
import { BacktestResult } from '../../components/BacktestResult';
import { BacktestDashboard } from '../../components/BacktestDashboard';

const RESULT_KEY = 'strategyStudio.backtest.lastResult';
const META_KEY = 'strategyStudio.backtest.lastMeta';

type ViewMode = 'auto' | 'raw';

export default function StrategyStudioResultsPage() {
  const navigate = useNavigate();
  const [result, setResult] = React.useState<any>(null);
  const [meta, setMeta] = React.useState<any>(null);
  const [view, setView] = React.useState<ViewMode>('auto');

  React.useEffect(() => {
    const load = () => {
      try {
        const raw = localStorage.getItem(RESULT_KEY);
        setResult(raw ? JSON.parse(raw) : null);
      } catch {
        setResult(null);
      }
      try {
        const rawMeta = localStorage.getItem(META_KEY);
        setMeta(rawMeta ? JSON.parse(rawMeta) : null);
      } catch {
        setMeta(null);
      }
    };

    load();
    const onStorage = (e: StorageEvent) => {
      if (e.key === RESULT_KEY || e.key === META_KEY) load();
    };
    window.addEventListener('storage', onStorage);
    return () => window.removeEventListener('storage', onStorage);
  }, []);

  return (
    <div style={{ padding: 16 }}>
      <div className="box" style={{ marginBottom: 12 }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8, flexWrap: 'wrap' }}>
          <div>
            <div style={{ fontWeight: 700 }}>Backtest Results</div>
            {meta?.symbol && (
              <div className="muted" style={{ fontSize: 12 }}>
                {String(meta.strategy || '').toUpperCase() || 'STRATEGY'} · {String(meta.symbol).toUpperCase()} · {meta.timeframe} · {meta.numBars} bars
              </div>
            )}
          </div>
          <div className="stack">
            <button className="btn" type="button" onClick={() => setView('auto')} disabled={view === 'auto'}>Formatted</button>
            <button className="btn" type="button" onClick={() => setView('raw')} disabled={view === 'raw'}>Raw JSON</button>
            <button className="btn" type="button" onClick={() => navigate('/strategy-studio')}>Back to Studio</button>
          </div>
        </div>
      </div>

      {!result ? (
        <div className="box" style={{ padding: 16 }}>
          <div className="muted">No saved result found. Run a backtest, then click “Open Results”.</div>
        </div>
      ) : (
        renderResult(result, view)
      )}
    </div>
  );
}

function renderResult(lastResult: any, view: ViewMode) {
  if (!lastResult) return <div className="muted">No result yet.</div>;
  if (view === 'raw') return <CodeDisplay code={JSON.stringify(lastResult, null, 2)} />;
  if (lastResult.stdout) return <CodeDisplay code={lastResult.stdout} />;

  // New Backtest Data Shape
  if (lastResult.metrics && (lastResult.equity || lastResult.optimization_results)) {
    return <BacktestDashboard data={lastResult} resizable />;
  }

  // Legacy flat metrics
  if (typeof lastResult === 'object' && lastResult && lastResult['Total Return [%]'] !== undefined) {
    return <BacktestResult metrics={lastResult} />;
  }

  return <CodeDisplay code={JSON.stringify(lastResult, null, 2)} />;
}
