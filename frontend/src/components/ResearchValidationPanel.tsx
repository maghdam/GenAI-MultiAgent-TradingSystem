import { useEffect, useState } from 'react';

import {
  calibrateV2MarketEvents,
  fetchV2ConfluenceReplay,
  fetchV2EventCalibration,
  type V2ConfluenceReplayResponse,
  type V2EventCalibrationResponse,
} from '../services/api';

const pct = (value?: number | null) => typeof value === 'number' ? `${value.toFixed(2)}%` : '--';

export default function ResearchValidationPanel() {
  const [calibration, setCalibration] = useState<V2EventCalibrationResponse | null>(null);
  const [replay, setReplay] = useState<V2ConfluenceReplayResponse | null>(null);
  const [symbol, setSymbol] = useState('US100');
  const [feeBps, setFeeBps] = useState(1);
  const [busy, setBusy] = useState<'calibration' | 'replay' | ''>('');
  const [message, setMessage] = useState('');

  const loadCalibration = async () => {
    try {
      setCalibration(await fetchV2EventCalibration());
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Calibration data unavailable.');
    }
  };

  useEffect(() => {
    loadCalibration();
  }, []);

  const evaluateOutcomes = async () => {
    setBusy('calibration');
    try {
      const result = await calibrateV2MarketEvents();
      setMessage(`${result.outcomes_evaluated} outcomes evaluated; ${result.outcomes_pending} pending.`);
      await loadCalibration();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Outcome evaluation failed.');
    } finally {
      setBusy('');
    }
  };

  const runReplay = async () => {
    setBusy('replay');
    try {
      setReplay(await fetchV2ConfluenceReplay(symbol, feeBps));
      setMessage('');
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Replay failed.');
    } finally {
      setBusy('');
    }
  };

  return (
    <section className="v2-panel" style={{ marginTop: 16 }}>
      <div className="v2-panel-head">
        <div>
          <h2>Evidence validation</h2>
          <p className="mi-muted">Research only. These results cannot alter deployed strategy decisions.</p>
        </div>
      </div>

      {message && <div className="v2-banner">{message}</div>}

      <div className="mi-context-grid">
        <div className="ta-panel">
          <div className="ta-panel__header">
            <div>
              <div className="ta-panel__title">Event outcome calibration</div>
              <div className="mi-muted">A context group remains observational until its sample gate passes.</div>
            </div>
            <button className="ta-btn ta-btn--sm" type="button" onClick={evaluateOutcomes} disabled={busy !== ''}>
              {busy === 'calibration' ? 'Evaluating…' : 'Evaluate outcomes'}
            </button>
          </div>
          <div className="ta-panel__body">
            <div className="mi-calibration-summary">
              <span>Evaluated <strong>{calibration?.evaluated_outcomes ?? 0}</strong></span>
              <span>Pending <strong>{calibration?.pending_outcomes ?? 0}</strong></span>
              <span>Minimum sample <strong>{calibration?.minimum_samples ?? 30}</strong></span>
            </div>
            <div className="mi-calibration-grid">
              {(calibration?.groups ?? []).slice(0, 6).map((group) => (
                <article className="mi-calibration-card" key={`${group.event_type}-${group.horizon}`}>
                  <div className="mi-card__head">
                    <strong>{group.event_type.replaceAll('_', ' ')} · {group.horizon}</strong>
                    <span className={`ta-pill ${group.gate === 'eligible' ? 'ta-pill--long' : group.gate === 'degraded' ? 'ta-pill--short' : 'ta-pill--info'}`}>
                      {group.gate.replaceAll('_', ' ')}
                    </span>
                  </div>
                  <div className="mi-muted">Samples {group.samples} · hit rate {group.hit_rate == null ? '--' : `${Math.round(group.hit_rate * 100)}%`} · return {pct(group.average_return_pct)}</div>
                </article>
              ))}
              {!calibration?.groups.length && <div className="mi-muted">No evaluated event groups yet.</div>}
            </div>
          </div>
        </div>

        <div className="ta-panel">
          <div className="ta-panel__header">
            <div>
              <div className="ta-panel__title">Original vs shadow replay</div>
              <div className="mi-muted">Compare stored decisions after estimated transaction costs.</div>
            </div>
          </div>
          <div className="ta-panel__body">
            <div className="mi-replay-controls">
              <label>Symbol<input className="ta-input ta-input--sm" value={symbol} onChange={(event) => setSymbol(event.target.value.toUpperCase())} /></label>
              <label>Cost / side (bps)<input className="ta-input ta-input--sm" type="number" min="0" max="100" step="0.1" value={feeBps} onChange={(event) => setFeeBps(Number(event.target.value) || 0)} /></label>
              <button className="ta-btn ta-btn--primary ta-btn--sm" type="button" onClick={runReplay} disabled={busy !== ''}>
                {busy === 'replay' ? 'Running…' : 'Run comparison'}
              </button>
            </div>
            {replay ? (
              <div style={{ display: 'grid', gap: 8, marginTop: 12 }}>
                <span className={`ta-pill ${replay.verdict === 'candidate_for_review' ? 'ta-pill--long' : replay.verdict === 'keep_shadow' ? 'ta-pill--short' : 'ta-pill--info'}`}>{replay.verdict.replaceAll('_', ' ')}</span>
                <div className="mi-calibration-summary">
                  <span>Priced <strong>{replay.priced_records}</strong></span>
                  <span>Original expectancy <strong>{pct(replay.original.expectancy_pct)}</strong></span>
                  <span>Shadow expectancy <strong>{pct(replay.shadow.expectancy_pct)}</strong></span>
                </div>
                <div className="mi-muted">{replay.verdict_reason}</div>
              </div>
            ) : <div className="mi-muted" style={{ marginTop: 12 }}>No replay has been run.</div>}
          </div>
        </div>
      </div>
    </section>
  );
}
