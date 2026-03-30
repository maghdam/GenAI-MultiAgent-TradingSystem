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
  notes: string[];
}

export interface V2WatchlistItem {
  symbol: string;
  timeframe: string;
  strategy: string;
  enabled: boolean;
  params: Record<string, unknown>;
}

export interface V2Config {
  enabled: boolean;
  paper_autotrade: boolean;
  allow_live: boolean;
  kill_switch: boolean;
  default_symbol: string;
  default_timeframe: string;
  default_strategy: string;
  scan_interval_sec: number;
  min_confidence: number;
  paper_trade_size: number;
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
  status: 'pending' | 'accepted' | 'rejected' | 'executed' | 'cancelled';
  confidence: number;
  entry_price?: number | null;
  stop_loss?: number | null;
  take_profit?: number | null;
  quantity?: number | null;
  rationale: string;
  details: Record<string, unknown>;
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

export interface V2ChecklistComponent {
  symbol: string;
  bias: 'bullish' | 'bearish' | 'flat' | 'unknown';
  change_pct?: number | null;
}

export interface V2AutoChecklist {
  ts: number;
  us30_bias: 'bullish' | 'bearish' | 'flat' | 'unknown';
  xau_bias: 'bullish' | 'bearish' | 'flat' | 'unknown';
  dxy_bias?: 'bullish' | 'bearish' | 'flat' | 'unknown';
  dxy_change_pct?: number | null;
  correlation: 'normal' | 'fear' | 'dollar_crash' | 'weird' | 'unknown';
  scenario: '' | 'A' | 'B' | 'C' | 'D';
  components: Record<string, V2ChecklistComponent>;
  component_score?: number | null;
  top_movers?: V2ChecklistComponent[];
  structure_hint?: 'bullish' | 'bearish' | 'range' | 'unknown';
  volume_hint?: 'rising' | 'falling' | 'flat' | 'unknown';
  structure_tf?: string | null;
  volume_tf?: string | null;
  smc_signal?: {
    symbol?: string;
    timeframe?: string;
    strategy?: string;
    signal?: string;
    confidence?: number;
    rationale?: string;
  } | null;
  notes?: string | null;
}

export interface V2CalendarEvent {
  ts: number | null;
  title: string | null;
  impact: 'high' | 'medium' | 'low' | 'unknown';
  source?: string | null;
}

export interface V2Status {
  version: string;
  mode: 'paper_only' | 'live_enabled';
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
): Promise<CandlesResponse> => {
  const response = await authFetch(
    `/api/market/candles?symbol=${symbol}&timeframe=${timeframe}&num_bars=${numBars}`,
    { signal },
  );
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Failed to fetch candles: ${response.status} ${response.statusText}`);
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
): Promise<any> => {
  const params = new URLSearchParams({
    strategy,
    symbol,
    timeframe,
    num_bars: String(numBars),
  });
  if (typeof feeBps === 'number' && Number.isFinite(feeBps)) {
    params.set('fee_bps', String(feeBps));
  }
  if (typeof slippageBps === 'number' && Number.isFinite(slippageBps)) {
    params.set('slippage_bps', String(slippageBps));
  }
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

export const analyzeV2Strategy = async (
  payload: {
    symbol: string;
    timeframe: string;
    strategy: string;
    num_bars?: number;
    params?: Record<string, unknown>;
  },
  signal?: AbortSignal,
): Promise<V2Analysis> => {
  const response = await authFetch('/api/analyze', {
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

export const fetchV2AutoChecklist = async (params?: {
  tf?: string;
  structure_tf?: string;
}): Promise<V2AutoChecklist> => {
  const search = new URLSearchParams();
  if (params?.tf) {
    search.set('tf', params.tf);
  }
  if (params?.structure_tf) {
    search.set('structure_tf', params.structure_tf);
  }
  const qs = search.toString();
  const response = await authFetch(`/api/checklist/auto${qs ? `?${qs}` : ''}`);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Failed to fetch auto checklist: ${response.status} ${response.statusText}`);
  }
  return response.json();
};

export const fetchV2NextCalendarEvent = async (): Promise<V2CalendarEvent | null> => {
  try {
    const response = await authFetch('/api/calendar/next');
    if (!response.ok) {
      return null;
    }
    return response.json();
  } catch {
    return null;
  }
};
