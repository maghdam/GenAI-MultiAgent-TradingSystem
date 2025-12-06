import React, { useEffect, useMemo, useState } from 'react';
import { fetchAutoChecklist, fetchNextCalendarEvent, type AutoChecklist, type CalendarEvent } from '../services/api';

type Session = '' | 'London' | 'New York';
type Strength = 'neutral' | 'strong' | 'weak';
type Correlation = 'normal' | 'fear' | 'dollar_crash' | 'weird' | 'unknown';
type Status = 'bullish' | 'bearish' | 'flat';
type Scenario = '' | 'A' | 'B' | 'C' | 'D';
type Structure = '' | 'hh' | 'll' | 'range';
type Volume = '' | 'up' | 'down' | 'flat';
type SlPlan = '' | 'atr' | 'swing';

interface ChecklistState {
  date: string;
  session: Session;
  dxy: Strength;
  redNews: boolean;
  redLockUntil?: number | null;
  newsNotes: string;
  correlation: Correlation;
  components: Record<string, Status>;
  scenario: Scenario;
  structure: Structure;
  volume: Volume;
  slPlan: SlPlan;
  notes: string;
}

const COMPONENTS: Array<{ id: string; label: string; weight: string; sector: string; why: string }> = [
    { id: 'GS', label: 'Goldman Sachs', weight: '~10.5%', sector: 'Finance', why: 'The King. Highest price stock. Moves the index most.' },
    { id: 'CAT', label: 'Caterpillar', weight: '~7.5%', sector: 'Industrial', why: 'The Economy. If CAT is down, "real" growth is weak.' },
    { id: 'MSFT', label: 'Microsoft', weight: '~6.3%', sector: 'Tech', why: 'Sentiment. If Tech is weak, market mood is sour.' },
    { id: 'JPM', label: 'Chase', weight: 'n/a', sector: 'Finance', why: 'Confirmation. Should move with GS.' },
    { id: 'UNH', label: 'UnitedHealth', weight: '~4.2%', sector: 'Healthcare', why: 'Secondary confirmation' },
    { id: 'HD', label: 'Home Depot', weight: '~4.6%', sector: 'Consumer Cyclical', why: 'Secondary confirmation' },
];

const EXTRA_CONF: Array<{ id: string; label: string; sector: string; why: string }> = [
  { id: 'GDX', label: 'VanEck Gold Miners (GDX)', sector: 'Gold Miners ETF', why: 'Miners often lead XAU; divergence can flag fakeouts.' },
  { id: 'GDXJ', label: 'Junior Gold Miners (GDXJ)', sector: 'Junior Miners', why: 'High beta; early warning on gold sentiment.' },
  { id: 'FDX', label: 'FedEx (FDX)', sector: 'Transports', why: 'Dow Theory confirmation; transports should confirm Industrials.' },
];

const today = new Date().toISOString().slice(0, 10);
const storageKey = (date?: string) => `heavyweight-checklist:${date || 'last'}`;

const emptyComponents = COMPONENTS.reduce<Record<string, Status>>((acc, c) => {
  acc[c.id] = 'flat';
  return acc;
}, {});

const baseState: ChecklistState = {
  date: today,
  session: '',
  dxy: 'neutral',
  redNews: false,
  redLockUntil: null,
  newsNotes: '',
  correlation: 'normal',
  components: emptyComponents,
  scenario: '',
  structure: '',
  volume: '',
  slPlan: '',
  notes: '',
};

function loadState(date: string): ChecklistState | null {
  if (typeof window === 'undefined') return null;
  try {
    const raw = localStorage.getItem(storageKey(date)) || localStorage.getItem(storageKey());
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    return { ...baseState, ...parsed, components: { ...emptyComponents, ...(parsed.components || {}) } };
  } catch {
    return null;
  }
}

function saveState(state: ChecklistState) {
  if (typeof window === 'undefined') return;
  const payload = JSON.stringify(state);
  localStorage.setItem(storageKey(state.date), payload);
  localStorage.setItem(storageKey(), payload);
}

const mapDxyBiasToStrength = (bias?: string): Strength => {
  if (bias === 'bullish') return 'strong';
  if (bias === 'bearish') return 'weak';
  return 'neutral';
};

const formatPct = (pct?: number | null): string => {
  if (pct === null || pct === undefined || Number.isNaN(pct)) return '';
  const rounded = Math.abs(pct) >= 1 ? pct.toFixed(2) : pct.toFixed(3);
  return `${pct >= 0 ? '+' : ''}${rounded}%`;
};

const scoreStyle = (score?: number | null) => {
  if (score === null || score === undefined || Number.isNaN(score)) return { label: 'n/a', color: '#9ca3af' };
  if (score >= 0.4) return { label: score.toFixed(2), color: '#4ade80' };
  if (score <= -0.4) return { label: score.toFixed(2), color: '#f87171' };
  return { label: score.toFixed(2), color: '#fbbf24' };
};

const secsSince = (ts?: number) => {
  if (!ts) return null;
  const now = Date.now() / 1000;
  return Math.max(0, now - ts);
};

const pillStyle = (tone: 'neutral' | 'good' | 'warn' | 'bad') => {
  if (tone === 'good') return { borderColor: '#4ade80', color: '#4ade80', background: 'rgba(74,222,128,0.08)' };
  if (tone === 'warn') return { borderColor: '#fbbf24', color: '#fbbf24', background: 'rgba(251,191,36,0.1)' };
  if (tone === 'bad') return { borderColor: '#f87171', color: '#f87171', background: 'rgba(248,113,113,0.1)' };
  return { borderColor: '#9ca3af', color: '#9ca3af', background: 'rgba(156,163,175,0.08)' };
};

const biasTone = (bias?: string): 'neutral' | 'good' | 'warn' | 'bad' => {
  if (bias === 'bullish') return 'good';
  if (bias === 'bearish') return 'bad';
  return 'warn';
};

const guardSeverity = (reasons: string[]): 'bad' | 'warn' | 'neutral' => {
  const lowered = reasons.map(r => r.toLowerCase());
  if (lowered.some(r => r.includes('fear') || r.includes('lockout'))) return 'bad';
  if (lowered.some(r => r.includes('anchor'))) return 'warn';
  return 'neutral';
};

export default function HeavyweightChecklistPage() {
  const [state, setState] = useState<ChecklistState>(baseState);
  const [toast, setToast] = useState<string>('');
  const [auto, setAuto] = useState<AutoChecklist | null>(null);
  const [autoError, setAutoError] = useState<string>('');
  const [autoApply, setAutoApply] = useState<boolean>(true);
  const [guardOverride, setGuardOverride] = useState<boolean>(false);
  const [guardDetailOpen, setGuardDetailOpen] = useState<boolean>(false);
  const [nowTick, setNowTick] = useState<number>(Date.now());
  const [nextEvent, setNextEvent] = useState<CalendarEvent | null>(null);
  const [pinnedAuto, setPinnedAuto] = useState<AutoChecklist | null>(null);
  const activeAuto = pinnedAuto || auto;
  const [biasTf, setBiasTf] = useState<string>('M5');
  const [structureTf, setStructureTf] = useState<string>('H1');

  useEffect(() => {
    const loaded = loadState(today);
    if (loaded) setState(loaded);
  }, []);

  useEffect(() => {
    const id = setInterval(() => setNowTick(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    let mounted = true;
    let timer: any;
    const poll = async () => {
      try {
        const res = await fetchAutoChecklist({ tf: biasTf, structure_tf: structureTf });
        if (!mounted) return;
        setAuto(res);
        setAutoError('');
        if (autoApply && !pinnedAuto) {
          applyAutoSnapshot(res, true);
        }
      } catch (e: any) {
        if (!mounted) return;
        setAutoError(e?.message || 'Auto fetch failed');
      } finally {
        if (mounted) timer = setTimeout(poll, 10000);
      }
    };
    poll();
    return () => { mounted = false; if (timer) clearTimeout(timer); };
  }, [autoApply, pinnedAuto, biasTf, structureTf]);

  useEffect(() => {
    let mounted = true;
    const poll = async () => {
      const evt = await fetchNextCalendarEvent();
      if (!mounted) return;
      setNextEvent(evt);
      setTimeout(poll, 60000);
    };
    poll();
    return () => { mounted = false };
  }, []);

  useEffect(() => {
    if (!nextEvent?.ts) return;
    const msAway = nextEvent.ts * 1000 - Date.now();
    // Auto-flag red news & lockout when within ±15m of high-impact events
    if (nextEvent.impact === 'high' && msAway <= 15 * 60 * 1000 && msAway > -15 * 60 * 1000) {
      setState((prev) => {
        const nextLock = prev.redLockUntil && prev.redLockUntil > Date.now() ? prev.redLockUntil : Date.now() + 15 * 60 * 1000;
        return { ...prev, redNews: true, redLockUntil: nextLock };
      });
    }
  }, [nextEvent]);

  const componentSummary = useMemo(() => {
    const counts = { bullish: 0, bearish: 0, flat: 0 };
    for (const c of Object.values(state.components || {})) {
      counts[c as Status] = (counts[c as Status] || 0) + 1;
    }
    let mood = 'mixed';
    if (counts.bullish >= 4) mood = 'aligned bullish';
    else if (counts.bearish >= 4) mood = 'aligned bearish';
    return { text: `${mood} (${counts.bullish}↑ / ${counts.bearish}↓ / ${counts.flat}·)`, counts };
  }, [state.components]);

  const guardrails = useMemo(() => {
    const reasons: string[] = [];
    if (state.redNews) reasons.push('High-impact news (manual)');
    if (state.redLockUntil && state.redLockUntil > Date.now()) {
      const mins = Math.max(0, Math.round((state.redLockUntil - Date.now()) / 60000));
      reasons.push(`News lockout (${mins}m left)`);
    }
    const scenarioAuto = (activeAuto?.scenario || '').toUpperCase();
    const scenarioState = (state.scenario || '').toUpperCase();
    if (scenarioAuto === 'B' || scenarioState === 'B') reasons.push('Anchor risk: GS dragging');
    if (scenarioAuto === 'D' || scenarioState === 'D') reasons.push('Fear dump: GS+CAT red, Gold pumping');
    if (nextEvent?.ts) {
      const msAway = nextEvent.ts * 1000 - Date.now();
      if (msAway <= 15 * 60 * 1000 && msAway > -15 * 60 * 1000 && nextEvent.impact === 'high') {
        const mins = Math.max(0, Math.round(msAway / 60000));
        reasons.push(`Upcoming red news: ${nextEvent.title || 'Event'} (${mins}m)`);
      }
    }
    return reasons;
  }, [state.redNews, state.redLockUntil, activeAuto?.scenario, nextEvent]);
  const guardActive = guardrails.length > 0 && !guardOverride;
  const guardTone = guardSeverity(guardrails);

  const scenarioHint: Record<Scenario, string> = {
    '': 'Pick the scenario that matches component alignment.',
    A: 'All Clear: Aggressive LONG bias on US30; buy pullbacks.',
    B: 'Anchor: NO TRADE or reduce size; heavyweight is dragging.',
    C: 'Sector War: Likely chop; range tactics only.',
    D: 'Fear Dump: SHORT US30, LONG XAUUSD setups.',
  };

  const summaryText = useMemo(() => {
    const guardActive = guardrails.length > 0 && !guardOverride;
    const actionLine = (() => {
      if (guardActive) return 'Action: BLOCKED by guardrails';
      if (state.scenario === 'A') return 'Action: LONG US30 (trend bias)';
      if (state.scenario === 'B') return 'Action: NO TRADE / reduce size';
      if (state.scenario === 'C') return 'Action: Range tactics only';
      if (state.scenario === 'D') return 'Action: SHORT US30, LONG XAUUSD setups';
      return 'Action: n/a';
    })();
    const whyScenario = (() => {
      const autoScenario = activeAuto?.scenario || '';
      if (autoScenario === 'A') return 'GS/CAT/MSFT aligned bullish';
      if (autoScenario === 'B') return 'US30 bullish but GS red';
      if (autoScenario === 'C') return 'Mixed: tech vs banks';
      if (autoScenario === 'D') return 'GS+CAT red + XAU up';
      return '';
    })();
    const lines = [
      `Date: ${state.date} • Session: ${state.session || 'n/a'}`,
      `Macro: DXY ${state.dxy}${activeAuto?.dxy_bias ? ` (auto: ${activeAuto.dxy_bias}${formatPct(activeAuto.dxy_change_pct) ? ' ' + formatPct(activeAuto.dxy_change_pct) : ''})` : ''}; Red news: ${state.redNews ? 'Yes' : 'No'}${state.newsNotes ? ` (${state.newsNotes})` : ''}`,
      `Correlation: ${state.correlation}`,
      `TFs: bias ${biasTf}, structure ${structureTf}`,
      `Components: ${componentSummary.text}`,
      `Component score: ${activeAuto?.component_score !== null && activeAuto?.component_score !== undefined ? activeAuto.component_score.toFixed(2) : 'n/a'}`,
      `Top movers: ${activeAuto?.top_movers?.map(m => `${m.symbol} ${m.bias} (${formatPct(m.change_pct) || '0%'})`).join(' • ') || 'n/a'}`,
      `Scenario: ${state.scenario || 'n/a'} (${scenarioHint[state.scenario]})${whyScenario ? ` • Why: ${whyScenario}` : ''}`,
      `Structure: ${state.structure || 'n/a'} • Volume: ${state.volume || 'n/a'} • SL plan: ${state.slPlan || 'n/a'}`,
      `Auto hints: ${activeAuto?.structure_hint || 'n/a'} @${activeAuto?.structure_tf || 'H1'} • ${activeAuto?.volume_hint || 'n/a'} @${activeAuto?.volume_tf || 'M5'}${activeAuto?.notes ? ` • ${activeAuto.notes}` : ''}`,
      `Guardrails: ${guardActive ? `BLOCKED (${guardrails.join('; ')})` : (guardrails.length ? `Overridden (${guardrails.join('; ')})` : 'None')}`,
      actionLine,
      `Notes: ${state.notes || '—'}`,
    ];
    return lines.join('\n');
  }, [state, componentSummary, scenarioHint, activeAuto, guardrails, guardOverride, biasTf, structureTf]);

  const setComponentStatus = (id: string, value: Status) => {
    setState((prev) => ({ ...prev, components: { ...prev.components, [id]: value } }));
  };

  const handleSave = () => {
    saveState(state);
    setToast('Saved locally');
    setTimeout(() => setToast(''), 2000);
  };

  const handleLoad = () => {
    const loaded = loadState(state.date);
    if (loaded) {
      setState(loaded);
      setToast('Loaded');
    } else {
      setToast('Nothing saved for that date');
    }
    setTimeout(() => setToast(''), 2000);
  };

  const handleReset = () => {
    setState({ ...baseState, date: state.date });
  };

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(summaryText);
      setToast('Summary copied');
    } catch {
      setToast('Copy failed');
    }
    setTimeout(() => setToast(''), 2000);
  };

  const applyAutoSnapshot = (payload?: AutoChecklist, silent: boolean = false) => {
    const snapshot = payload || activeAuto || auto;
    if (!snapshot) return;
    setState((prev) => {
      const nextComponents = { ...prev.components };
      Object.entries(snapshot.components || {}).forEach(([k, v]) => {
        if (nextComponents.hasOwnProperty(k)) {
          const normalized = v.bias === 'bullish' || v.bias === 'bearish' || v.bias === 'flat' ? v.bias : nextComponents[k];
          nextComponents[k] = (normalized as Status) || nextComponents[k] || 'flat';
        }
      });
      const mappedStructure =
        snapshot.structure_hint === 'bullish' ? 'hh' :
        snapshot.structure_hint === 'bearish' ? 'll' :
        snapshot.structure_hint === 'range' ? 'range' : prev.structure;
      const mappedVolume =
        snapshot.volume_hint === 'rising' ? 'up' :
        snapshot.volume_hint === 'falling' ? 'down' :
        snapshot.volume_hint === 'flat' ? 'flat' : prev.volume;
      const dxyStrength = mapDxyBiasToStrength(snapshot.dxy_bias);
      return {
        ...prev,
        dxy: dxyStrength || prev.dxy,
        correlation: snapshot.correlation === 'unknown' ? prev.correlation : (snapshot.correlation as any),
        scenario: snapshot.scenario || prev.scenario,
        structure: mappedStructure,
        volume: mappedVolume,
        components: nextComponents,
      };
    });
    if (!silent) {
      setToast('Applied auto snapshot');
      setTimeout(() => setToast(''), 1500);
    }
  };

  return (
    <div style={{ padding: '14px', display: 'grid', gap: '14px' }}>
      <div className="box" style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
        <div style={{ fontWeight: 700, fontSize: 18 }}>US30 & XAUUSD Heavyweight Checklist</div>
        <a className="btn" href="/">← Dashboard</a>
        <a className="btn" href="/strategy-studio">Strategy Studio</a>
        <div style={{ flex: 1 }} />
        <label className="chip">
          Date
          <input type="date" value={state.date} onChange={e => setState(s => ({ ...s, date: e.target.value }))} />
        </label>
        <button className="btn" onClick={handleLoad}>Load</button>
        <button className="btn success" onClick={handleSave}>Save</button>
        <button className="btn" onClick={handleReset}>Reset</button>
        <button className="btn primary" onClick={handleCopy} disabled={guardActive} title={guardActive ? 'Guardrails active — override to copy' : undefined}>Copy summary</button>
        {toast && <span className="muted">{toast}</span>}
      </div>

      <div className="box" style={{ display: 'grid', gap: 6 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
          <div style={{ fontWeight: 700, fontSize: 16 }}>Auto snapshot</div>
          <div className="muted">
            {activeAuto ? `Updated: ${new Date(activeAuto.ts * 1000).toLocaleTimeString()}` : autoError || 'Loading…'}
          </div>
          <label className="chip">
            Bias TF
            <select value={biasTf} onChange={e => { setBiasTf(e.target.value); setPinnedAuto(null); setAuto(null); }}>
              {['M1','M5','M15','M30','H1','H4'].map(tf => <option key={tf} value={tf}>{tf}</option>)}
            </select>
          </label>
          <label className="chip">
            Structure TF
            <select value={structureTf} onChange={e => { setStructureTf(e.target.value); setPinnedAuto(null); setAuto(null); }}>
              {['M5','M15','M30','H1','H4','D1'].map(tf => <option key={tf} value={tf}>{tf}</option>)}
            </select>
          </label>
          <div style={{ flex: 1 }} />
          <label className="chip">
            <input type="checkbox" checked={autoApply} onChange={e => setAutoApply(e.target.checked)} />
            Auto-apply
          </label>
          <button className="btn success" onClick={() => applyAutoSnapshot()} disabled={!activeAuto}>Apply auto snapshot</button>
          <button className="btn" onClick={() => setPinnedAuto(activeAuto || null)} disabled={!activeAuto}>{pinnedAuto ? 'Update pinned' : 'Pin snapshot'}</button>
          {pinnedAuto && <button className="btn" onClick={() => setPinnedAuto(null)}>Unpin</button>}
        </div>
        {activeAuto && (
          <div style={{ display: 'grid', gap: 8 }}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 8 }}>
              <span className="chip" style={{ ...pillStyle(scoreStyle(activeAuto.component_score).color === '#4ade80' ? 'good' : scoreStyle(activeAuto.component_score).color === '#f87171' ? 'bad' : 'warn'), fontWeight: 700 }}>
                Score {scoreStyle(activeAuto.component_score).label}
              </span>
              <span className="chip" style={{ ...pillStyle((() => { const age = secsSince(activeAuto.ts); if (age === null) return 'neutral'; if (age < 20) return 'good'; if (age < 60) return 'warn'; return 'bad'; })()), fontWeight: 600 }}>
                Age {(() => { const age = secsSince(activeAuto.ts); if (age === null) return 'n/a'; if (age < 60) return `${Math.round(age)}s`; return `${Math.round(age / 60)}m`; })()}
              </span>
              <span className="chip" style={{ ...pillStyle(activeAuto.scenario === 'A' ? 'good' : activeAuto.scenario === 'D' ? 'bad' : 'warn') }}>
                Scenario {activeAuto.scenario || 'n/a'}
              </span>
              <span className="chip" style={pillStyle(activeAuto.correlation === 'normal' ? 'good' : activeAuto.correlation === 'fear' ? 'bad' : 'warn')}>
                Corr {activeAuto.correlation || 'n/a'}
              </span>
              <span className="chip" style={pillStyle(activeAuto.us30_bias === 'bullish' ? 'good' : activeAuto.us30_bias === 'bearish' ? 'bad' : 'warn')}>US30 {activeAuto.us30_bias}</span>
              <span className="chip" style={pillStyle(activeAuto.xau_bias === 'bullish' ? 'good' : activeAuto.xau_bias === 'bearish' ? 'bad' : 'warn')}>XAU {activeAuto.xau_bias}</span>
              <span className="chip" style={pillStyle(activeAuto.dxy_bias === 'bullish' ? 'bad' : activeAuto.dxy_bias === 'bearish' ? 'good' : 'warn')}>
                DXY {activeAuto.dxy_bias} {formatPct(activeAuto.dxy_change_pct)}
              </span>
              {activeAuto.smc_signal && (
                <span className="chip" style={pillStyle('warn')}>
                  SMC: {activeAuto.smc_signal.signal || 'n/a'} {activeAuto.smc_signal.confidence ? `(${(activeAuto.smc_signal.confidence * 100).toFixed(0)}%)` : ''} {activeAuto.smc_signal.timeframe ? `@${activeAuto.smc_signal.timeframe}` : ''}
                </span>
              )}
              {pinnedAuto && <span className="chip" style={pillStyle('warn')}>Pinned @ {new Date(pinnedAuto.ts * 1000).toLocaleTimeString()}</span>}
            </div>
            <div style={{ display: 'grid', gap: 10, color: '#e5e9f0', fontSize: 14 }}>
              <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', alignItems: 'center' }}>
                <span className="muted">Component score</span>
                <span style={{ color: scoreStyle(activeAuto.component_score).color, fontWeight: 700, fontSize: 15 }}>
                  {activeAuto.component_score !== null && activeAuto.component_score !== undefined ? activeAuto.component_score.toFixed(2) : 'n/a'}
                </span>
              </div>
              {activeAuto.top_movers && activeAuto.top_movers.length > 0 && (
                <div style={{ display: 'grid', gap: 8, background: '#111620', border: '1px solid #1f2433', borderRadius: 10, padding: 10 }}>
                  <div style={{ fontWeight: 700, color: '#cfd8ff' }}>Top movers</div>
                  <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
                    {activeAuto.top_movers.map(m => (
                      <span key={m.symbol} className="chip" style={{ ...pillStyle(biasTone(m.bias)), padding: '6px 10px', fontWeight: 600 }}>
                        {m.symbol} {m.bias} {formatPct(m.change_pct) || '0%'}
                      </span>
                    ))}
                  </div>
                </div>
              )}
              <div style={{ display: 'grid', gap: 8, background: '#111620', border: '1px solid #1f2433', borderRadius: 10, padding: 10 }}>
                <div style={{ fontWeight: 700, color: '#cfd8ff' }}>Structure & Volume</div>
                <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', alignItems: 'center' }}>
                  <span className="muted">Structure</span>
                  <span className="chip" style={{ ...pillStyle(activeAuto.structure_hint === 'bullish' ? 'good' : activeAuto.structure_hint === 'bearish' ? 'bad' : 'warn'), padding: '6px 10px', fontWeight: 600 }}>
                    {activeAuto.structure_hint || 'n/a'} {activeAuto.structure_tf ? `@${activeAuto.structure_tf}` : ''}
                  </span>
                  <span className="muted">Volume</span>
                  <span className="chip" style={{ ...pillStyle(activeAuto.volume_hint === 'rising' ? 'good' : activeAuto.volume_hint === 'falling' ? 'bad' : 'warn'), padding: '6px 10px', fontWeight: 600 }}>
                    {activeAuto.volume_hint || 'n/a'} {activeAuto.volume_tf ? `@${activeAuto.volume_tf}` : ''}
                  </span>
                </div>
              </div>
              {activeAuto.notes && <div className="muted" style={{ fontSize: 14 }}>Note: {activeAuto.notes}</div>}
              {nextEvent?.ts && (
                <div className="muted" style={{ fontSize: 14 }}>
                  Next red event: <strong>{nextEvent.title || 'n/a'}</strong> • {nextEvent.impact || 'unknown'} • {(() => {
                    const msAway = nextEvent.ts ? nextEvent.ts * 1000 - nowTick : null;
                    if (msAway === null) return 'n/a';
                    const mins = Math.round(msAway / 60000);
                    return mins >= 0 ? `${mins}m away` : `${Math.abs(mins)}m ago`;
                  })()} {nextEvent.source ? `(${nextEvent.source})` : ''}
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {guardrails.length > 0 && (
        <div className="box" style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap', borderColor: guardTone === 'bad' ? '#f57f6c' : '#fbbf24', background: guardTone === 'bad' ? 'rgba(245,127,108,0.08)' : 'rgba(251,191,36,0.05)' }}>
          <div style={{ fontWeight: 700, color: guardTone === 'bad' ? '#f57f6c' : '#fbbf24' }}>
            Guardrails {guardActive ? 'BLOCKED' : 'Overridden'}
          </div>
          <button className="btn" onClick={() => setGuardDetailOpen(o => !o)}>
            {guardDetailOpen ? 'Hide reasons' : `View reasons (${guardrails.length})`}
          </button>
          {guardDetailOpen && (
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              {guardrails.map((r) => (
                <span key={r} className="chip" style={{ borderColor: guardTone === 'bad' ? '#f57f6c' : '#fbbf24', color: guardTone === 'bad' ? '#f57f6c' : '#fbbf24' }}>{r}</span>
              ))}
            </div>
          )}
          <div style={{ flex: 1 }} />
          <label className="chip">
            <input
              type="checkbox"
              checked={guardOverride}
              onChange={e => {
                if (!e.target.checked) {
                  setGuardOverride(false);
                  return;
                }
                const ok = window.confirm('Override guardrails for the next 10 minutes? Use only if you accept the risks.');
                if (ok) setGuardOverride(true);
              }}
            />
            Override guardrails
          </label>
          {state.redLockUntil && state.redLockUntil > Date.now() && (
            <span className="muted">News lockout ends in {Math.max(0, Math.round((state.redLockUntil - Date.now()) / 60000))}m</span>
          )}
        </div>
      )}

      <div className="grid" style={{ gridTemplateColumns: '1.1fr 1fr', alignItems: 'start' }}>
        <div className="box" style={{ display: 'grid', gap: 10 }}>
            <h3 style={{ margin: 0 }}>Phase 1: The "Weather Report" (Macro Context)</h3>
            <p className="muted" style={{margin:0}}>Before looking at the US30 chart, check the environment.</p>
            <div className="stack">
                <label className="chip">
                Session
                <select value={state.session} onChange={e => setState(s => ({ ...s, session: e.target.value as Session }))} disabled={guardActive}>
                    <option value="">Select…</option>
                    <option value="London">London</option>
                    <option value="New York">New York</option>
                </select>
                </label>
                <label className="chip">
                DXY (US Dollar Index)
                <select value={state.dxy} onChange={e => setState(s => ({ ...s, dxy: e.target.value as Strength }))} disabled={(autoApply && !!auto?.dxy_bias && auto.dxy_bias !== 'unknown') || guardActive}>
                    <option value="neutral">Neutral</option>
                    <option value="strong">Strong (bad for Gold/Stocks)</option>
                    <option value="weak">Weak (good for Gold/Stocks)</option>
                </select>
                <div className="muted" style={{ marginTop: 4 }}>Auto: {auto?.dxy_bias || 'n/a'} {formatPct(auto?.dxy_change_pct)}</div>
                </label>
                <label className="chip">
                <input type="checkbox" checked={state.redNews} onChange={e => setState(s => ({ ...s, redNews: e.target.checked }))} disabled={guardActive} />
                Red Folder News Today?
                </label>
                <label className="chip">
                  News lockout
                  <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                    <button className="btn" onClick={() => setState(s => ({ ...s, redLockUntil: Date.now() + 15 * 60 * 1000 }))} disabled={guardActive}>Start 15m</button>
                    <button className="btn" onClick={() => setState(s => ({ ...s, redLockUntil: null }))} disabled={guardActive}>Clear</button>
                    <span className="muted">{state.redLockUntil && state.redLockUntil > Date.now() ? `${Math.max(0, Math.round((state.redLockUntil - Date.now()) / 60000))}m left` : 'idle'}</span>
                  </div>
                </label>
                <input
                placeholder="Notes (CPI, NFP, FOMC...)"
                value={state.newsNotes}
                onChange={e => setState(s => ({ ...s, newsNotes: e.target.value }))} disabled={guardActive}
                style={{ minWidth: 220 }}
                />
            </div>
            <div className="stack">
                <label className="chip">
                "Risk" Correlation Check
                <select value={state.correlation} onChange={e => setState(s => ({ ...s, correlation: e.target.value as Correlation }))} disabled={(autoApply && !!auto?.correlation && auto.correlation !== 'unknown') || guardActive}>
                    <option value="normal">Normal: US30🟢 / XAU🔴 (Risk On)</option>
                    <option value="fear">Fear: US30🔴 / XAU🟢 (Risk Off)</option>
                    <option value="dollar_crash">Dollar Crash: US30🟢 / XAU🟢</option>
                    <option value="weird">Warning: Unpredictable Volatility</option>
                </select>
                </label>
            </div>
            {nextEvent?.ts && (
              <div className="muted" style={{ marginTop: 4 }}>
                Next event: <strong>{nextEvent.title || 'n/a'}</strong> • {nextEvent.impact || 'unknown'} • {(() => {
                  const msAway = nextEvent.ts ? nextEvent.ts * 1000 - nowTick : null;
                  if (msAway === null) return 'n/a';
                  const mins = Math.round(msAway / 60000);
                  return mins >= 0 ? `${mins}m away` : `${Math.abs(mins)}m ago`;
                })()} {nextEvent.source ? `(${nextEvent.source})` : ''}
              </div>
            )}
        </div>

        <div className="box" style={{ display: 'grid', gap: 10 }}>
            <h3 style={{ margin: 0 }}>Phase 2: The US30 "Engine Check" (Price-Weighted Analysis)</h3>
            <p className="muted" style={{margin:0}}>Do not trade US30 until you check these specific tickers.</p>
          <div style={{ display: 'grid', gap: 8 }}>
            {COMPONENTS.slice(0, 3).map((c) => ( // Top 3
                <div key={c.id} className="stack" style={{ justifyContent: 'space-between' }}>
                    <div style={{ minWidth: 140 }}>
                        <div style={{ fontWeight: 600 }}>{c.label} ({c.id})</div>
                        <div className="muted">{c.weight} Weight</div>
                        <div className="muted">Auto: {auto?.components?.[c.id]?.bias === 'bullish' ? '🟢' : auto?.components?.[c.id]?.bias === 'bearish' ? '🔴' : '⚪'}{auto?.components?.[c.id]?.bias || 'n/a'} {formatPct(auto?.components?.[c.id]?.change_pct)}</div>
                    </div>
                    <div className="stack">
                        {['bullish', 'flat', 'bearish'].map(v => (
                        <button key={v} className={`btn ${state.components[c.id] === v ? 'primary' : ''}`} onClick={() => setComponentStatus(c.id, v as Status)} disabled={autoApply || guardActive}>
                            {v === 'bullish' ? '🟢 Bullish' : v === 'bearish' ? '🔴 Bearish' : '⚪ Flat'}
                        </button>
                        ))}
                    </div>
                </div>
            ))}
            <h4 style={{margin: '8px 0 0 0'}}>Secondary Confirmation:</h4>
            {COMPONENTS.slice(3).map((c) => (
                 <div key={c.id} className="stack" style={{ justifyContent: 'space-between' }}>
                 <div style={{ minWidth: 140 }}>
                     <div style={{ fontWeight: 600 }}>{c.label} ({c.id})</div>
                     <div className="muted">{c.weight}</div>
                     <div className="muted">Auto: {auto?.components?.[c.id]?.bias === 'bullish' ? '🟢' : auto?.components?.[c.id]?.bias === 'bearish' ? '🔴' : '⚪'}{auto?.components?.[c.id]?.bias || 'n/a'} {formatPct(auto?.components?.[c.id]?.change_pct)}</div>
                 </div>
                 <div className="stack">
                     {['bullish', 'flat', 'bearish'].map(v => (
                     <button key={v} className={`btn ${state.components[c.id] === v ? 'primary' : ''}`} onClick={() => setComponentStatus(c.id, v as Status)} disabled={autoApply || guardActive}>
                         {v === 'bullish' ? '🟢 Bullish' : v === 'bearish' ? '🔴 Bearish' : '⚪ Flat'}
                     </button>
                     ))}
                 </div>
             </div>
            ))}
          </div>
          <div className="muted">Summary: {componentSummary.text}</div>
          <h4 style={{margin: '12px 0 4px 0'}}>Gold / Transports Confirmation</h4>
          <div style={{ display: 'grid', gap: 8 }}>
            {EXTRA_CONF.map((c) => {
              const bias = auto?.components?.[c.id]?.bias;
              const change = auto?.components?.[c.id]?.change_pct;
              return (
                <div key={c.id} className="stack" style={{ justifyContent: 'space-between' }}>
                  <div style={{ minWidth: 160 }}>
                    <div style={{ fontWeight: 600 }}>{c.label} ({c.id})</div>
                    <div className="muted">{c.sector}</div>
                    <div className="muted">Auto: {bias === 'bullish' ? '🟢' : bias === 'bearish' ? '🔴' : '⚪'}{bias || 'n/a'} {formatPct(change)}</div>
                    <div className="muted" style={{ fontSize: 12 }}>{c.why}</div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      <div className="grid" style={{ gridTemplateColumns: '1fr 1fr', alignItems: 'start' }}>
        <div className="box" style={{ display: 'grid', gap: 10 }}>
            <h3 style={{ margin: 0 }}>Phase 3: The "Go / No-Go" Decision Logic</h3>
            <p className="muted" style={{margin:0}}>Based on Phase 2, determine your bias.</p>
            {auto?.scenario && (
              <div className="muted">
                Auto suggests Scenario {auto.scenario} (score {auto.component_score !== null && auto.component_score !== undefined ? auto.component_score.toFixed(2) : 'n/a'})
                {(() => {
                  const sc = auto.scenario;
                  if (sc === 'A') return ' — GS/CAT/MSFT aligned bullish';
                  if (sc === 'B') return ' — US30 bullish but GS red (anchor risk)';
                  if (sc === 'C') return ' — Tech vs Banks divergence (chop)';
                  if (sc === 'D') return ' — GS+CAT red + Gold up (fear dump)';
                  return '';
                })()}
              </div>
            )}
            {auto?.smc_signal && (
              <div className="muted">Latest SMC: {auto.smc_signal.signal || 'n/a'} {auto.smc_signal.confidence ? `(${(auto.smc_signal.confidence * 100).toFixed(0)}%)` : ''} {auto.smc_signal.timeframe ? `@${auto.smc_signal.timeframe}` : ''} {auto.smc_signal.strategy ? `• ${auto.smc_signal.strategy}` : ''}</div>
            )}
            {auto?.scenario === 'B' && <div style={{ color: '#f57f6c', fontWeight: 600 }}>WARNING: Anchor risk (GS dragging) — avoid new longs unless manual override.</div>}
            {auto?.scenario === 'D' && <div style={{ color: '#f57f6c', fontWeight: 600 }}>WARNING: Fear dump — bias SHORT US30, LONG XAUUSD.</div>}
          <div style={{ display: 'grid', gap: 6 }}>
            {(() => {
              const autoScenarioActive = autoApply && !!auto?.scenario;
              return [
              { id: 'A', label: 'SCENARIO A: The "All Clear" (Trend Day)', detail: 'GS 🟢 + CAT 🟢 + MSFT 🟢 → Aggressive LONG on US30. Buy pullbacks.' },
              { id: 'B', label: 'SCENARIO B: The "Anchor" (Reversal Risk)', detail: 'US30 Chart bullish, BUT GS 🔴 → NO TRADE (or reduced size). Breakout will likely fail.' },
              { id: 'C', label: 'SCENARIO C: The "Sector War" (Chop/Range)', detail: 'MSFT (Tech) 🟢 but GS/JPM (Banks) 🔴 → Range trading only.' },
              { id: 'D', label: 'SCENARIO D: The "Fear" Dump', detail: 'GS 🔴 + CAT 🔴 + Gold (XAUUSD) 🟢 → Aggressive SHORT on US30. Look for Longs on XAUUSD.' },
            ].map(opt => {
              const checked = autoScenarioActive ? auto?.scenario === opt.id : state.scenario === opt.id;
              const disabled = autoScenarioActive || guardActive;
              return (
              <label key={opt.id} className="chip" style={{
                justifyContent: 'space-between',
                alignItems: 'start',
                textAlign: 'left',
                borderColor: checked ? '#8ab4ff' : undefined,
                opacity: autoScenarioActive && !checked ? 0.6 : 1,
              }}>
                <span>
                      <input
                        type="radio"
                        name="scenario"
                        checked={!!checked}
                        onChange={() => setState(s => ({ ...s, scenario: opt.id as Scenario }))}
                        disabled={disabled}
                        style={{ marginRight: 8, marginTop: 4 }}
                      />
                </span>
                <span style={{flex: 1}}>
                    <strong>{opt.label}</strong>
                    <div className="muted">{opt.detail}</div>
                </span>
              </label>
              );
            });
            })()}
          </div>
          <div className="muted">{scenarioHint[state.scenario]}</div>
        </div>

        <div className="box" style={{ display: 'grid', gap: 8 }}>
          <h3 style={{ margin: 0 }}>Phase 4: Intraday Execution (Technical)</h3>
          <p className="muted" style={{margin:0}}>Only now do you look at the US30 chart structure.</p>

          <div className="stack">
            <label className="chip">
              1. Market Structure (1HR/15m)
              <select value={state.structure} onChange={e => setState(s => ({ ...s, structure: e.target.value as Structure }))} disabled={(autoApply && !!auto?.structure_hint && auto.structure_hint !== 'unknown') || guardActive}>
                <option value="">Select…</option>
                <option value="hh">Higher Highs (Buy)</option>
                <option value="ll">Lower Lows (Sell)</option>
                <option value="range">Range</option>
              </select>
              <div className="muted" style={{ marginTop: 4 }}>Auto: {auto?.structure_hint || 'n/a'} {auto?.structure_tf ? `@${auto.structure_tf}` : ''} {auto ? `(updated ${new Date(auto.ts * 1000).toLocaleTimeString()})` : ''}</div>
            </label>
            <label className="chip">
              2. Volume Check
              <select value={state.volume} onChange={e => setState(s => ({ ...s, volume: e.target.value as Volume }))} disabled={(autoApply && !!auto?.volume_hint && auto.volume_hint !== 'unknown') || guardActive}>
                <option value="">Select…</option>
                <option value="up">Increasing with "Heavyweights"</option>
                <option value="down">Falling vs move</option>
                <option value="flat">Flat</option>
              </select>
              <div className="muted" style={{ marginTop: 4 }}>Auto: {auto?.volume_hint || 'n/a'} {auto?.volume_tf ? `@${auto.volume_tf}` : ''} {auto ? `(updated ${new Date(auto.ts * 1000).toLocaleTimeString()})` : ''}</div>
            </label>
            <label className="chip">
              3. Stop Loss Plan
              <select value={state.slPlan} onChange={e => setState(s => ({ ...s, slPlan: e.target.value as SlPlan }))} disabled={guardActive}>
                <option value="">Select…</option>
                <option value="atr">Based on ATR</option>
                <option value="swing">Based on Swing Low/High</option>
              </select>
            </label>
          </div>
          <p className="muted">Mental Check: Am I entering in the middle of a range? (Wait for edge).</p>

          <textarea
            placeholder="Notes / entry plan / levels…"
            value={state.notes}
            onChange={e => setState(s => ({ ...s, notes: e.target.value }))}
            style={{ width: '100%', minHeight: 100, background: '#0e1117', color: '#e5e9f0', border: '1px solid #1a2030', borderRadius: 8, padding: 10 }}
            disabled={guardActive}
          />
        </div>
      </div>
        <div className="box" style={{ display: 'grid', gap: 10 }}>
            <h3 style={{margin:0}}>📉 Quick Reference: The "Why" Cheat Sheet</h3>
            <p className="muted" style={{margin:0}}>Keep this visible until you memorize the weights. Live biases included.</p>
            <table style={{width: '100%', borderCollapse: 'collapse'}}>
                <thead>
                    <tr>
                        <th style={{textAlign: 'left', padding: 8, borderBottom: '1px solid #1a2030'}}>Ticker</th>
                        <th style={{textAlign: 'left', padding: 8, borderBottom: '1px solid #1a2030'}}>Sector</th>
                        <th style={{textAlign: 'left', padding: 8, borderBottom: '1px solid #1a2030'}}>Why it matters</th>
                        <th style={{textAlign: 'left', padding: 8, borderBottom: '1px solid #1a2030'}}>Live bias</th>
                    </tr>
                </thead>
                <tbody>
                    {[...COMPONENTS, ...EXTRA_CONF].map(c => {
                      const bias = auto?.components?.[c.id]?.bias;
                      const change = auto?.components?.[c.id]?.change_pct;
                      const tone = biasTone(bias);
                      return (
                        <tr key={c.id}>
                            <td style={{padding: 8, borderBottom: '1px solid #1a2030'}}><strong>{c.label} ({c.id})</strong></td>
                            <td style={{padding: 8, borderBottom: '1px solid #1a2030'}}>{c.sector}</td>
                            <td style={{padding: 8, borderBottom: '1px solid #1a2030'}}>{c.why}</td>
                            <td style={{padding: 8, borderBottom: '1px solid #1a2030'}}>
                              <span className="chip" style={{ ...pillStyle(tone), padding: '4px 8px' }}>
                                {bias === 'bullish' ? '🟢' : bias === 'bearish' ? '🔴' : '⚪'}{bias || 'n/a'} {formatPct(change)}
                              </span>
                            </td>
                        </tr>
                      );
                    })}
                </tbody>
            </table>
        </div>
      <div className="box" style={{ background: '#0e1117', borderColor: '#1a2030', display: 'grid', gap: 6 }}>
        <div style={{ fontWeight: 700, fontSize: 20 }}>Live Summary</div>
        <pre style={{ margin: 0, whiteSpace: 'pre-wrap', color: '#cfd8ff', fontSize: 14, lineHeight: 1.5 }}>{summaryText}</pre>
      </div>
    </div>
  );
}
