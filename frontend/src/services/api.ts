import type { AgentSignal } from '../types';
import { authFetch } from './http';

export interface Candle {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
}

export interface CandlesResponse {
  candles: Candle[];
  indicators: Record<string, any>;
}

export interface SymbolsResponse {
  symbols: string[];
  default: string | null;
}

export interface V2BrokerStatus {
  connected: boolean;
  socket_connected: boolean;
  account_authorized: boolean;
  auth_error?: string | null;
  last_auth_attempt_at?: string | null;
  symbols_loaded: number;
  open_positions: number;
  pending_orders: number;
  ready: boolean;
  market_data_ready: boolean;
  broker_mode: string;
  account_id?: number | null;
  account_type: 'demo' | 'live' | 'unknown';
  demo_account_confirmed: boolean;
  execution_ready: boolean;
  notes: string[];
}

export interface V2WatchlistItem {
  symbol: string;
  timeframe: string;
  strategy: string;
  enabled: boolean;
  trading_enabled: boolean;
  lot_size?: number | null;
  params: Record<string, unknown>;
}

export interface V2Config {
  enabled: boolean;
  paper_autotrade: boolean;
  demo_autotrade: boolean;
  allow_live: boolean;
  kill_switch: boolean;
  default_symbol: string;
  default_timeframe: string;
  default_strategy: string;
  scan_interval_sec: number;
  min_confidence: number;
  paper_trade_size: number;
  account_currency: string;
  paper_starting_equity_amount: number;
  risk_per_trade_pct: number;
  daily_loss_limit_pct: number;
  max_daily_trades: number;
  max_open_positions: number;
  max_positions_per_symbol: number;
  cooldown_minutes: number;
  session_filter_enabled: boolean;
  session_start_hour_utc: number;
  session_end_hour_utc: number;
  require_stops: boolean;
  operator_note: string;
  watchlist: V2WatchlistItem[];
}

export interface V2ReadinessCheck {
  name: string;
  ok: boolean;
  detail: string;
}

export interface V2StrategyInfo {
  key: string;
  label: string;
  description: string;
  parameters: Record<string, unknown>;
}

export interface V2DashboardAnalysisRequest {
  symbol: string;
  timeframe: string;
  strategy: string;
  num_bars?: number;
}

export interface V2ManualOrderRequest {
  symbol: string;
  timeframe: string;
  strategy: string;
  signal: 'long' | 'short';
  quantity: number;
  confidence?: number;
  entry_price?: number | null;
  stop_loss?: number | null;
  take_profit?: number | null;
  reasons?: string[];
  rationale?: string;
}

export interface V2ManualOrderResponse {
  ok: boolean;
  intent_id: number;
  status: string;
  summary: string;
  position_id?: number | null;
  broker_position_id?: number | null;
  mode: string;
}

export interface V2Incident {
  id: number;
  level: 'info' | 'warning' | 'error';
  code: string;
  message: string;
  details: Record<string, unknown>;
  created_at: string;
}

export interface V2Analysis {
  symbol: string;
  timeframe: string;
  strategy: string;
  signal: 'long' | 'short' | 'no_trade';
  confidence: number;
  entry_price?: number | null;
  stop_loss?: number | null;
  take_profit?: number | null;
  reasons: string[];
  context: Record<string, unknown>;
  created_at: string;
}

export interface V2PaperPosition {
  id: number;
  symbol: string;
  timeframe: string;
  strategy: string;
  direction: 'long' | 'short';
  quantity: number;
  status: 'open' | 'closed';
  entry_price: number;
  current_price?: number | null;
  stop_loss?: number | null;
  take_profit?: number | null;
  opened_at: string;
  closed_at?: string | null;
  exit_price?: number | null;
  realized_pnl: number;
  unrealized_pnl: number;
  close_reason?: string | null;
  account_currency: string;
  cash_per_price_unit_per_lot: number;
  instrument_spec_source: string;
}

export interface V2PaperEvent {
  id: number;
  created_at: string;
  event_type: string;
  summary: string;
  details: Record<string, unknown>;
}

export interface V2OrderIntent {
  id: number;
  created_at: string;
  symbol: string;
  timeframe: string;
  strategy: string;
  direction: 'long' | 'short' | 'no_trade';
  intent_type: 'open' | 'close' | 'update' | 'hold' | 'skip';
  status: 'pending' | 'accepted' | 'rejected' | 'executed' | 'cancelled' | 'failed';
  confidence: number;
  entry_price?: number | null;
  stop_loss?: number | null;
  take_profit?: number | null;
  quantity?: number | null;
  rationale: string;
  details: Record<string, unknown>;
  decision_id?: number | null;
}

export interface V2DecisionRecord {
  id: number;
  created_at: string;
  correlation_id: string;
  decision_type: string;
  symbol: string;
  timeframe: string;
  strategy: string;
  outcome: string;
  summary: string;
  evidence: Record<string, unknown>;
}

export interface V2TradeAudit {
  id: number;
  created_at: string;
  event_type: string;
  symbol: string;
  timeframe: string;
  strategy: string;
  position_id?: number | null;
  intent_id?: number | null;
  summary: string;
  details: Record<string, unknown>;
}

export interface V2Runtime {
  running: boolean;
  loop_active: boolean;
  ollama_ready: boolean;
  last_cycle_at?: string | null;
  last_cycle_summary: string;
  last_reconcile_at?: string | null;
  last_reconcile_summary: string;
  last_error?: string | null;
  tick_count: number;
  active_watchlist: string[];
}

export interface V2MarketIntelligenceDriver {
  label: string;
  detail: string;
  impact: 'bullish' | 'bearish' | 'neutral';
}

export interface V2MarketIntelligenceInstrument {
  symbol: string;
  timeframe: string;
  name: string;
  category: string;
  price?: number | null;
  change?: number | null;
  change_pct?: number | null;
  low_range?: number | null;
  high_range?: number | null;
  bias: 'bullish' | 'bearish' | 'neutral' | 'unknown';
  confidence: number;
  situation: string;
  direction_note: string;
  drivers: V2MarketIntelligenceDriver[];
  support: number[];
  resistance: number[];
  last_updated: string;
  data_status: 'live' | 'cached' | 'unavailable';
  error?: string | null;
}

export interface V2MarketIntelligenceMacroEvent {
  title: string;
  impact: 'high' | 'medium' | 'low' | 'unknown';
  source: string;
  ts?: number | null;
}

export interface V2MarketEvent {
  id: number;
  content_hash: string;
  source: string;
  title: string;
  summary: string;
  url: string;
  published_at?: string | null;
  ingested_at: string;
  symbols: string[];
  event_type: string;
  sentiment: 'bullish' | 'bearish' | 'neutral' | 'mixed';
  sentiment_score: number;
  impact: 'high' | 'medium' | 'low' | 'unknown';
  horizon: 'immediate' | 'intraday' | 'swing' | 'long_term' | 'unknown';
  credibility_score: number;
  classification_version: string;
  raw: Record<string, unknown>;
}

export interface V2EventAlert {
  id: number;
  alert_key: string;
  created_at: string;
  alert_type: string;
  symbol: string;
  severity: 'info' | 'warning' | 'critical';
  summary: string;
  details: Record<string, unknown>;
}

export interface V2EventRefreshResponse {
  ok: boolean;
  configured_sources: number;
  fetched_items: number;
  inserted_events: number;
  duplicate_events: number;
  alerts_created: number;
  errors: string[];
}

export interface V2EventCalibrationGroup {
  event_type: string;
  horizon: '5m' | '30m' | '4h' | '1d';
  samples: number;
  hit_rate?: number | null;
  average_return_pct?: number | null;
  median_return_pct?: number | null;
  average_brier_score?: number | null;
  average_favorable_excursion_pct?: number | null;
  average_adverse_excursion_pct?: number | null;
  gate: 'insufficient_samples' | 'observe' | 'eligible' | 'degraded';
  gate_reason: string;
}

export interface V2EventCalibrationResponse {
  generated_at: string;
  evaluated_outcomes: number;
  pending_outcomes: number;
  unavailable_outcomes: number;
  minimum_samples: number;
  groups: V2EventCalibrationGroup[];
}

export interface V2EventCalibrationRunResponse {
  ok: boolean;
  events_checked: number;
  outcomes_evaluated: number;
  outcomes_pending: number;
  outcomes_unavailable: number;
  errors: string[];
}

export interface V2ConfluenceShadowRecord {
  id: number;
  created_at: string;
  analysis_created_at: string;
  mode: 'shadow';
  symbol: string;
  timeframe: string;
  strategy: string;
  original_signal: 'long' | 'short' | 'no_trade';
  original_confidence: number;
  shadow_signal: 'long' | 'short' | 'no_trade';
  shadow_confidence: number;
  confidence_adjustment: number;
  action: 'confirm' | 'reduce' | 'neutral' | 'context_only' | 'insufficient_evidence';
  target_horizon: '5m' | '30m' | '4h' | '1d';
  event_score: number;
  eligible_event_count: number;
  event_ids: number[];
  original_would_pass: boolean;
  shadow_would_pass: boolean;
  rationale: string;
  evidence: Record<string, unknown>;
  execution_unchanged: boolean;
}

export interface V2ConfluenceReplayMetrics {
  candidate_decisions: number;
  trades: number;
  wins: number;
  losses: number;
  win_rate_pct: number;
  expectancy_pct: number;
  total_return_pct: number;
  max_drawdown_pct: number;
  profit_factor?: number | null;
  trades_per_day: number;
}

export interface V2ConfluenceReplayResponse {
  generated_at: string;
  research_only: true;
  methodology: string;
  total_records: number;
  priced_records: number;
  pending_records: number;
  unavailable_records: number;
  fee_bps_per_side: number;
  original: V2ConfluenceReplayMetrics;
  shadow: V2ConfluenceReplayMetrics;
  deltas: Record<string, number>;
  verdict: 'insufficient_data' | 'keep_shadow' | 'candidate_for_review';
  verdict_reason: string;
  warnings: string[];
}

export interface V2MarketIntelligenceResponse {
  generated_at: string;
  regime: 'risk_on' | 'risk_off' | 'mixed' | 'unknown';
  headline: string;
  summary: string;
  instruments: V2MarketIntelligenceInstrument[];
  macro_events: V2MarketIntelligenceMacroEvent[];
  market_events: V2MarketEvent[];
  event_alerts: V2EventAlert[];
  source_notes: string[];
}

export interface V2Status {
  version: string;
  mode: 'paper_only' | 'demo_enabled' | 'live_enabled';
  broker: V2BrokerStatus;
  config: V2Config;
  runtime: V2Runtime;
  readiness: V2ReadinessCheck[];
  strategies: V2StrategyInfo[];
  recent_incidents: V2Incident[];
  recent_analyses: V2Analysis[];
  paper_positions: V2PaperPosition[];
  recent_events: V2PaperEvent[];
  recent_order_intents: V2OrderIntent[];
  recent_trade_audits: V2TradeAudit[];
  recent_decisions: V2DecisionRecord[];
  recent_confluence_shadows: V2ConfluenceShadowRecord[];
}

export interface V2ModelsResponse {
  provider: string;
  models: string[];
  default?: string | null;
  fallback?: string | null;
  error?: string;
}

export interface V2StudioProviderInfo {
  key: string;
  label: string;
  configured: boolean;
}

export interface V2StudioModelsResponse {
  provider: string;
  providers: V2StudioProviderInfo[];
  models: string[];
  default?: string | null;
  fallback?: string | null;
  error?: string;
}

export interface V2StudioTaskRequest {
  task_type:
    | 'calculate_indicator'
    | 'backtest_strategy'
    | 'save_strategy'
    | 'research_strategy'
    | 'create_strategy'
    | 'backtest'
    | 'optimize'
    | 'chat';
  goal: string;
  params?: Record<string, any>;
}

export interface V2StudioTaskResponse {
  status: 'success' | 'error';
  message?: string;
  result?: any;
}

export type V2StrategyLifecycleStage = 'draft' | 'backtested' | 'validated' | 'paper' | 'eligible' | 'retired';
export type V2StrategyEvidenceType = 'development_backtest' | 'out_of_sample' | 'regime' | 'paper';

export interface V2StrategyLifecycleEvidence {
  id: number;
  lifecycle_id: number;
  evidence_type: V2StrategyEvidenceType;
  passed: boolean;
  summary: string;
  metrics: Record<string, any>;
  context: Record<string, any>;
  created_at: string;
}

export interface V2StrategyLifecycleTransition {
  id: number;
  lifecycle_id: number;
  from_stage: V2StrategyLifecycleStage;
  to_stage: V2StrategyLifecycleStage;
  operator: string;
  reason: string;
  created_at: string;
}

export interface V2StrategyLifecycle {
  id: number;
  strategy: string;
  version: number;
  version_hash: string;
  stage: V2StrategyLifecycleStage;
  hypothesis: string;
  gates: Record<string, any>;
  created_at: string;
  updated_at: string;
  evidence: V2StrategyLifecycleEvidence[];
  transitions: V2StrategyLifecycleTransition[];
  current_source: boolean;
  next_stage?: V2StrategyLifecycleStage | null;
  promotion_ready: boolean;
  blockers: string[];
}

const STATUS_CACHE_TTL_MS = 1_500;
const STRATEGIES_CACHE_TTL_MS = 60_000;
const SYMBOLS_CACHE_TTL_MS = 5 * 60_000;

type CacheEntry<T> = {
  expiresAt: number;
  value: T;
};

let statusCache: CacheEntry<V2Status> | null = null;
let statusInflight: Promise<V2Status> | null = null;
let strategiesCache: CacheEntry<V2StrategyInfo[]> | null = null;
let strategiesInflight: Promise<V2StrategyInfo[]> | null = null;
let symbolsCache: CacheEntry<SymbolsResponse> | null = null;
let symbolsInflight: Promise<SymbolsResponse> | null = null;

const invalidateStatusCache = () => {
  statusCache = null;
  statusInflight = null;
};

const responseErrorMessage = async (response: Response, fallback: string): Promise<string> => {
  const text = await response.text();
  if (!text) return fallback;
  try {
    const payload = JSON.parse(text);
    if (typeof payload?.detail === 'string') return payload.detail;
    if (typeof payload?.message === 'string') return payload.message;
  } catch {
    return text;
  }
  return text;
};

export const toAgentSignal = (analysis: V2Analysis): AgentSignal => ({
  ts: Math.floor(new Date(analysis.created_at).getTime() / 1000),
  symbol: analysis.symbol,
  timeframe: analysis.timeframe,
  signal: analysis.signal,
  confidence: analysis.confidence,
  rationale: analysis.reasons.join(' '),
  reasons: analysis.reasons,
  sl: analysis.stop_loss ?? null,
  tp: analysis.take_profit ?? null,
  entry: analysis.entry_price ?? null,
  strategy: analysis.strategy,
});

export const getV2Status = async (options: { force?: boolean } = {}): Promise<V2Status> => {
  const now = Date.now();
  if (!options.force && statusCache && statusCache.expiresAt > now) {
    return statusCache.value;
  }
  if (!options.force && statusInflight) {
    return statusInflight;
  }

  statusInflight = authFetch('/api/status')
    .then(async (response) => {
      if (!response.ok) {
        throw new Error(`Failed to fetch status: ${response.status} ${response.statusText}`);
      }
      const payload = await response.json() as V2Status;
      statusCache = { value: payload, expiresAt: Date.now() + STATUS_CACHE_TTL_MS };
      return payload;
    })
    .finally(() => {
      statusInflight = null;
    });

  return statusInflight;
};

export const getV2Config = async (): Promise<V2Config> => {
  const response = await authFetch('/api/config');
  if (!response.ok) {
    throw new Error(`Failed to fetch config: ${response.status} ${response.statusText}`);
  }
  return response.json();
};

export const setV2Config = async (config: V2Config): Promise<V2Config> => {
  const response = await authFetch('/api/config', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Failed to save config: ${response.status} ${response.statusText}`);
  }
  invalidateStatusCache();
  return response.json();
};

export const getV2Symbols = async (): Promise<SymbolsResponse> => {
  const now = Date.now();
  if (symbolsCache && symbolsCache.expiresAt > now) {
    return symbolsCache.value;
  }
  if (symbolsInflight) {
    return symbolsInflight;
  }

  symbolsInflight = authFetch('/api/symbols')
    .then(async (response) => {
      if (!response.ok) {
        throw new Error(`Failed to fetch symbols: ${response.status} ${response.statusText}`);
      }
      const payload = await response.json() as SymbolsResponse;
      symbolsCache = { value: payload, expiresAt: Date.now() + SYMBOLS_CACHE_TTL_MS };
      return payload;
    })
    .finally(() => {
      symbolsInflight = null;
    });

  return symbolsInflight;
};

export const getV2Candles = async (
  symbol: string,
  timeframe: string,
  numBars: number = 5000,
  signal?: AbortSignal,
  options: { live?: boolean } = {},
): Promise<CandlesResponse> => {
  const liveQs = options.live ? `&live=1` : '';
  const response = await authFetch(
    `/api/market/candles?symbol=${symbol}&timeframe=${timeframe}&num_bars=${numBars}${liveQs}`,
    { signal },
  );
  if (!response.ok) {
    throw new Error(await responseErrorMessage(response, `Failed to fetch candles: ${response.status} ${response.statusText}`));
  }
  return response.json();
};

export const getV2Strategies = async (): Promise<V2StrategyInfo[]> => {
  const now = Date.now();
  if (strategiesCache && strategiesCache.expiresAt > now) {
    return strategiesCache.value;
  }
  if (strategiesInflight) {
    return strategiesInflight;
  }

  strategiesInflight = authFetch('/api/strategies')
    .then(async (response) => {
      if (!response.ok) {
        throw new Error(`Failed to fetch strategies: ${response.status} ${response.statusText}`);
      }
      const payload = await response.json() as V2StrategyInfo[];
      strategiesCache = { value: payload, expiresAt: Date.now() + STRATEGIES_CACHE_TTL_MS };
      return payload;
    })
    .finally(() => {
      strategiesInflight = null;
    });

  return strategiesInflight;
};

export const getV2Models = async (): Promise<V2ModelsResponse> => {
  const response = await authFetch('/api/models');
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Failed to fetch models: ${response.status} ${response.statusText}`);
  }
  return response.json();
};

export const getV2StudioModels = async (provider?: string): Promise<V2StudioModelsResponse> => {
  const qs = provider ? `?provider=${encodeURIComponent(provider)}` : '';
  const response = await authFetch(`/api/studio/models${qs}`);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Failed to fetch studio models: ${response.status} ${response.statusText}`);
  }
  return response.json();
};

export const listV2StudioStrategyFiles = async (): Promise<string[]> => {
  const response = await authFetch('/api/studio/strategy-files');
  if (!response.ok) {
    throw new Error(`Failed to list strategy files: ${response.status} ${response.statusText}`);
  }
  const data = await response.json();
  const files: string[] = Array.isArray(data?.files) ? data.files : [];
  return files.map((file) => (typeof file === 'string' && file.toLowerCase().endsWith('.py') ? file.slice(0, -3) : file));
};

export const backtestV2SavedStrategy = async (
  strategy: string,
  symbol: string,
  timeframe: string,
  numBars: number,
  feeBps?: number,
  slippageBps?: number,
  validationKind?: Exclude<V2StrategyEvidenceType, 'paper'>,
): Promise<any> => {
  const params = new URLSearchParams({
    strategy,
    symbol,
    timeframe,
    num_bars: String(numBars),
  });
  if (typeof feeBps === 'number' && Number.isFinite(feeBps)) params.set('fee_bps', String(feeBps));
  if (typeof slippageBps === 'number' && Number.isFinite(slippageBps)) params.set('slippage_bps', String(slippageBps));
  if (validationKind) params.set('validation_kind', validationKind);
  const response = await authFetch(`/api/studio/backtest?${params.toString()}`);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Backtest failed: ${response.status} ${response.statusText}`);
  }
  return response.json();
};

export const executeV2StudioTask = async (request: V2StudioTaskRequest): Promise<V2StudioTaskResponse> => {
  const response = await authFetch('/api/studio/tasks', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Failed to execute studio task: ${response.status} ${response.statusText}`);
  }
  return response.json();
};

export const getV2StrategyLifecycle = async (strategy: string): Promise<V2StrategyLifecycle> => {
  const response = await authFetch(`/api/studio/lifecycle/${encodeURIComponent(strategy)}`);
  if (!response.ok) throw new Error(await responseErrorMessage(response, `Failed to load lifecycle: ${response.status}`));
  return response.json();
};

export const updateV2StrategyHypothesis = async (strategy: string, hypothesis: string): Promise<V2StrategyLifecycle> => {
  const response = await authFetch(`/api/studio/lifecycle/${encodeURIComponent(strategy)}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ hypothesis }),
  });
  if (!response.ok) throw new Error(await responseErrorMessage(response, `Failed to update hypothesis: ${response.status}`));
  return response.json();
};

export const promoteV2StrategyLifecycle = async (
  strategy: string,
  operator: string,
  reason = '',
): Promise<V2StrategyLifecycle> => {
  const response = await authFetch(`/api/studio/lifecycle/${encodeURIComponent(strategy)}/promote`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ operator, reason }),
  });
  if (!response.ok) throw new Error(await responseErrorMessage(response, `Promotion failed: ${response.status}`));
  return response.json();
};

export const recordV2PaperEvidence = async (strategy: string): Promise<V2StrategyLifecycle> => {
  const response = await authFetch(`/api/studio/lifecycle/${encodeURIComponent(strategy)}/paper-evidence`, { method: 'POST' });
  if (!response.ok) throw new Error(await responseErrorMessage(response, `Paper evidence failed: ${response.status}`));
  return response.json();
};

export const analyzeV2Strategy = async (
  payload: {
    symbol: string;
    timeframe: string;
    strategy: string;
    num_bars?: number;
    params?: Record<string, unknown>;
  },
  signal?: AbortSignal,
): Promise<V2Analysis> => {  const response = await authFetch('/api/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal,
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Failed to analyze: ${response.status} ${response.statusText}`);
  }
  invalidateStatusCache();
  return response.json();
};

export const analyzeV2Dashboard = async (
  payload: V2DashboardAnalysisRequest,
  signal?: AbortSignal,
): Promise<V2Analysis> => analyzeV2Strategy(payload, signal);

export const placeV2ManualOrder = async (payload: V2ManualOrderRequest): Promise<V2ManualOrderResponse> => {
  const response = await authFetch('/api/orders/manual', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Failed to place manual order: ${response.status} ${response.statusText}`);
  }
  invalidateStatusCache();
  return response.json();
};

export const startV2Engine = async (): Promise<{ ok: boolean; enabled: boolean }> => {
  const response = await authFetch('/api/engine/start', { method: 'POST' });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Failed to start engine: ${response.status} ${response.statusText}`);
  }
  invalidateStatusCache();
  return response.json();
};

export const stopV2Engine = async (): Promise<{ ok: boolean; enabled: boolean }> => {
  const response = await authFetch('/api/engine/stop', { method: 'POST' });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Failed to stop engine: ${response.status} ${response.statusText}`);
  }
  invalidateStatusCache();
  return response.json();
};

export const scanV2Engine = async (): Promise<{ ok: boolean; summary: string }> => {
  const response = await authFetch('/api/engine/scan', { method: 'POST' });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Failed to scan engine: ${response.status} ${response.statusText}`);
  }
  invalidateStatusCache();
  return response.json();
};

export const reconcileV2Engine = async (): Promise<{ ok: boolean; checked: number; closed: number; skipped: number; reason: string }> => {
  const response = await authFetch('/api/engine/reconcile', { method: 'POST' });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Failed to reconcile engine: ${response.status} ${response.statusText}`);
  }
  invalidateStatusCache();
  return response.json();
};

export const recoverV2Engine = async (): Promise<{ ok: boolean; active_watchlist: string[]; enabled: boolean }> => {
  const response = await authFetch('/api/engine/recover', { method: 'POST' });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Failed to recover engine: ${response.status} ${response.statusText}`);
  }
  invalidateStatusCache();
  return response.json();
};

export const getV2OrderIntents = async (limit = 20): Promise<V2OrderIntent[]> => {
  const response = await authFetch(`/api/paper/order-intents?limit=${limit}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch order intents: ${response.status} ${response.statusText}`);
  }
  return response.json();
};

export const getV2TradeAudit = async (limit = 20): Promise<V2TradeAudit[]> => {
  const response = await authFetch(`/api/paper/audit?limit=${limit}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch trade audit: ${response.status} ${response.statusText}`);
  }
  return response.json();
};

export const fetchV2MarketIntelligence = async (): Promise<V2MarketIntelligenceResponse> => {
  const response = await authFetch('/api/market/intelligence');
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Failed to fetch market intelligence: ${response.status} ${response.statusText}`);
  }
  return response.json();
};

export const refreshV2MarketEvents = async (): Promise<V2EventRefreshResponse> => {
  const response = await authFetch('/api/market/events/refresh', { method: 'POST' });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Failed to refresh market events: ${response.status} ${response.statusText}`);
  }
  return response.json();
};

export const fetchV2EventCalibration = async (): Promise<V2EventCalibrationResponse> => {
  const response = await authFetch('/api/market/events/calibration');
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Failed to fetch event calibration: ${response.status} ${response.statusText}`);
  }
  return response.json();
};

export const calibrateV2MarketEvents = async (): Promise<V2EventCalibrationRunResponse> => {
  const response = await authFetch('/api/market/events/calibrate', { method: 'POST' });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Failed to calibrate market events: ${response.status} ${response.statusText}`);
  }
  return response.json();
};

export const fetchV2ConfluenceReplay = async (
  symbol: string = 'US100',
  feeBpsPerSide: number = 1,
): Promise<V2ConfluenceReplayResponse> => {
  const params = new URLSearchParams({
    limit: '1000',
    symbol: symbol.trim().toUpperCase(),
    fee_bps_per_side: String(feeBpsPerSide),
    num_bars: '5000',
  });
  const response = await authFetch(`/api/market/confluence-shadow/replay?${params.toString()}`);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Failed to run confluence replay: ${response.status} ${response.statusText}`);
  }
  return response.json();
};
