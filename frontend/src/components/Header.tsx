import type { ChangeEvent } from 'react';
import type { AgentStatus } from '../services/api';

interface StatusChip {
  status: 'ok' | 'bad' | 'wait';
  label: string;
}

interface HeaderProps {
  strategy: string;
  onStrategyChange: (value: string) => void;
  lotSize: number;
  onLotSizeChange: (value: number) => void;
  fastMode: boolean;
  onFastModeChange: (value: boolean) => void;
  maxBars: number;
  onMaxBarsChange: (value: number) => void;
  maxTokens: number;
  onMaxTokensChange: (value: number) => void;
  modelName: string;
  onModelNameChange: (value: string) => void;
  isAnalyzing: boolean;
  onRunAnalysis: () => void;
  onCancelAnalysis: () => void;
  onPlaceTrade: () => void;
  feedStatus?: StatusChip;
  llmStatus?: StatusChip;
  agentStatus: AgentStatus | null;
  onWatchCurrent?: () => void;
  onToggleAgent?: () => void;
  onOpenAgentSettings?: () => void;
  onReloadStrategies?: () => void;
}

const DEFAULT_FEED_STATUS: StatusChip = { status: 'wait', label: 'cTrader: checking…' };
const DEFAULT_LLM_STATUS: StatusChip = { status: 'wait', label: 'LLM: checking…' };

export default function Header({
  strategy,
  onStrategyChange,
  lotSize,
  onLotSizeChange,
  fastMode,
  onFastModeChange,
  maxBars,
  onMaxBarsChange,
  maxTokens,
  onMaxTokensChange,
  modelName,
  onModelNameChange,
  isAnalyzing,
  onRunAnalysis,
  onCancelAnalysis,
  onPlaceTrade,
  feedStatus = DEFAULT_FEED_STATUS,
  llmStatus = DEFAULT_LLM_STATUS,
  agentStatus,
  onWatchCurrent,
  onToggleAgent,
  onOpenAgentSettings,
  onReloadStrategies,
}: HeaderProps) {
  const handleLotSizeChange = (event: ChangeEvent<HTMLInputElement>) => {
    const nextValue = parseFloat(event.target.value);
    if (Number.isNaN(nextValue)) {
      onLotSizeChange(0);
      return;
    }
    onLotSizeChange(Math.max(0.01, nextValue));
  };

  const handleMaxBarsChange = (event: ChangeEvent<HTMLInputElement>) => {
    const nextValue = parseInt(event.target.value, 10);
    if (Number.isNaN(nextValue)) {
      onMaxBarsChange(50);
      return;
    }
    onMaxBarsChange(Math.min(500, Math.max(50, nextValue)));
  };

  const handleMaxTokensChange = (event: ChangeEvent<HTMLInputElement>) => {
    const nextValue = parseInt(event.target.value, 10);
    if (Number.isNaN(nextValue)) {
      onMaxTokensChange(64);
      return;
    }
    onMaxTokensChange(Math.min(1024, Math.max(64, nextValue)));
  };

  const agentStatusText = agentStatus
    ? `Agent: ${agentStatus.enabled ? (agentStatus.running ? 'ON (running)' : 'ON (idle)') : 'OFF'} • Watchlist: ${agentStatus.watchlist.length}`
    : 'Agent: checking…';

  const strategyOptions = Array.from(
    new Set([
      ...(agentStatus?.available_strategies || []),
      'smc',
      'rsi',
      strategy,
    ])
  );

  return (
    <header className="stack">
      <select title="Strategy" value={strategy} onChange={event => onStrategyChange(event.target.value)}>
        {strategyOptions.map((name) => (
          <option key={name} value={name}>{name.toUpperCase()}</option>
        ))}
      </select>

      <label className="chip">
        Lot size
        <input
          type="number"
          min="0.01"
          step="0.01"
          value={lotSize}
          onChange={handleLotSizeChange}
          style={{ width: '70px' }}
        />
      </label>

      <label className="chip">
        <input type="checkbox" checked={fastMode} onChange={event => onFastModeChange(event.target.checked)} /> Fast
      </label>

      <label className="chip">
        Bars
        <input
          type="number"
          min="50"
          max="500"
          step="50"
          value={maxBars}
          onChange={handleMaxBarsChange}
          style={{ width: '70px' }}
        />
      </label>

      <label className="chip">
        Max tok
        <input
          type="number"
          min="64"
          max="1024"
          step="64"
          value={maxTokens}
          onChange={handleMaxTokensChange}
          style={{ width: '70px' }}
        />
      </label>

      <label className="chip">
        Model
        <input
          list="models"
          placeholder="auto"
          value={modelName}
          onChange={event => onModelNameChange(event.target.value)}
          style={{ width: '140px' }}
        />
        <datalist id="models">
          <option value="llama3.2:3b-instruct-q4_K_M" />
          <option value="phi3:3.8b-mini-instruct-q4_K_M" />
          <option value="mistral:7b-instruct-q4_K_M" />
        </datalist>
      </label>

      <button
        className="btn warn"
        type="button"
        onClick={onCancelAnalysis}
        style={{ display: isAnalyzing ? 'inline-block' : 'none' }}
      >
        ✖ Cancel
      </button>

      <button className="btn primary" type="button" onClick={onRunAnalysis} disabled={isAnalyzing}>
        🧠 Run AI Analysis
      </button>
      <button className="btn success" type="button" onClick={onPlaceTrade} disabled={isAnalyzing}>
        ▶ Place Trade
      </button>
      {isAnalyzing && <span style={{ color: 'var(--accent)', marginLeft: '8px' }}>🔄 Analyzing…</span>}

      <div style={{ flex: 1 }} />

      <span className="chip status">
        <span className={`dot ${feedStatus.status}`}></span>
        {feedStatus.label}
      </span>
      <span className="chip status">
        <span className={`dot ${llmStatus.status}`}></span>
        {llmStatus.label}
      </span>

      <button className="btn" type="button" onClick={() => onWatchCurrent?.()}>
        ➕ Watch current
      </button>
      <button className="btn" type="button" onClick={onToggleAgent} disabled={!agentStatus}>
        {agentStatus?.enabled ? '⏹ Stop Agent' : '▶ Start Agent'}
      </button>
      <button className="btn" type="button" onClick={onOpenAgentSettings}>
        ⚙️ Agent Settings
      </button>
      <button className="btn" type="button" onClick={onReloadStrategies}>
        Reload Strategies
      </button>
      <span className="muted">{agentStatusText}</span>
    </header>
  );
}
