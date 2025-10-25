import React from 'react';
import type { BacktestMetrics } from '../types/backtest';

interface BacktestResultProps {
  metrics: BacktestMetrics;
}

export function BacktestResult({ metrics }: BacktestResultProps) {
  const entries = Object.entries(metrics || {});
  if (entries.length === 0) {
    return <div className="muted">No backtest metrics to show.</div>;
  }

  return (
    <div className="journal-table-container">
      <table className="journal-table">
        <thead>
          <tr>
            <th>Metric</th>
            <th>Value</th>
          </tr>
        </thead>
        <tbody>
          {entries.map(([k, v]) => (
            <tr key={k}>
              <td className="muted">{k}</td>
              <td>{formatValue(v)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function formatValue(v: unknown): string {
  if (v == null) return 'N/A';
  if (typeof v === 'number') {
    const abs = Math.abs(v);
    const decimals = abs >= 100 ? 2 : abs >= 1 ? 4 : 6;
    return v.toFixed(decimals);
  }
  if (typeof v === 'object') return JSON.stringify(v);
  return String(v);
}

