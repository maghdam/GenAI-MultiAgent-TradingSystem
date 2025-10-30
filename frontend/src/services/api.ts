import type { AgentSignal } from '../types';

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

export interface SymbolLimit {
  symbol: string;
  min_lots?: number | null;
  step_lots?: number | null;
  max_lots?: number | null;
  min_api?: number | null;
  step_api?: number | null;
  max_api?: number | null;
}

export type SymbolLimitsMap = Record<string, SymbolLimit>;

export interface WatchlistEntry {
  symbol: string;
  timeframe: string;
  lot_size: number;
}

export interface AgentConfig {
  enabled: boolean;
  watchlist: WatchlistEntry[];
  interval_sec: number;
  min_confidence: number;
  trading_mode: string;
  autotrade: boolean;
  lot_size_lots: number;
  strategy: string;
}

export interface AgentStatus extends AgentConfig {
  running: boolean;
  running_pairs: [string, string][];
  tasks: Array<Record<string, any>>;
  available_strategies: string[];
}

export async function getCandles(
  symbol: string,
  timeframe: string,
  numBars: number = 5000
): Promise<CandlesResponse> {
  const response = await fetch(
    `/api/candles?symbol=${symbol}&timeframe=${timeframe}&num_bars=${numBars}`
  );

  if (!response.ok) {
    throw new Error(`Failed to fetch candles: ${response.statusText}`);
  }

  return response.json();
}

export const getSymbols = async (): Promise<SymbolsResponse> => {
  const response = await fetch("/api/symbols");

  if (!response.ok) {
    throw new Error(`Failed to fetch symbols: ${response.statusText}`);
  }

  return response.json();
};

export const getSymbolLimits = async (symbol?: string): Promise<SymbolLimitsMap | SymbolLimit> => {
  const url = symbol ? `/api/symbol_limits?symbol=${encodeURIComponent(symbol)}` : '/api/symbol_limits';
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch symbol limits: ${response.status} ${response.statusText}`);
  }
  return response.json();
};

export const getAgentConfig = async (): Promise<AgentConfig> => {
  const response = await fetch("/api/agent/config");
  if (!response.ok) {
    throw new Error(`Failed to fetch agent config: ${response.statusText}`);
  }
  return response.json();
};

export const setAgentConfig = async (config: AgentConfig): Promise<{ ok: boolean }> => {
  const response = await fetch("/api/agent/config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Failed to set agent config: ${text || response.statusText}`);
  }
  return response.json();
};

export const getAgentStatus = async (): Promise<AgentStatus> => {
  const response = await fetch('/api/agent/status');
  if (!response.ok) {
      throw new Error(`Failed to fetch agent status: ${response.statusText}`);
  }
  return response.json();
};



export const getAgentSignals = async (limit = 50): Promise<AgentSignal[]> => {
  const response = await fetch(`/api/agent/signals?n=${limit}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch agent signals: ${response.statusText}`);
  }
  return response.json();
};

export const addToWatchlist = async (
  symbol: string,
  timeframe: string,
  lotSize?: number
): Promise<{ ok: boolean }> => {
  const params = new URLSearchParams({ symbol, timeframe });
  if (typeof lotSize === 'number' && Number.isFinite(lotSize)) {
    params.set('lot_size', lotSize.toString());
  }
  const response = await fetch(`/api/agent/watchlist/add?${params.toString()}`, {
    method: 'POST',
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Failed to add ${symbol}:${timeframe} to watchlist`);
  }
  return response.json();
};

// --- Strategy Studio Tasks ---

export interface TaskRequest {
  task_type: 'calculate_indicator' | 'backtest_strategy' | 'save_strategy' | 'research_strategy' | 'create_strategy';
  goal: string;
  params?: Record<string, any>;
}

export interface TaskResponse {
  status: 'success' | 'error';
  message?: string;
  result?: any;
}

export const executeTask = async (request: TaskRequest): Promise<TaskResponse> => {
  const response = await fetch('/api/agent/execute_task', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `${response.status} ${response.statusText}`);
  }
  return response.json();
};

// --- Strategies management ---

export const reloadStrategies = async (): Promise<{ ok: boolean; loaded: number; available: string[] } | null> => {
  try {
    const response = await fetch('/api/strategies/reload', { method: 'POST' });
    if (!response.ok) {
      // Fallback to GET if POST is blocked
      const r2 = await fetch('/api/strategies/reload');
      if (!r2.ok) throw new Error(await r2.text());
      return r2.json();
    }
    return response.json();
  } catch {
    return null;
  }
};

export const listStrategies = async (): Promise<{ available: string[]; errors: any[] }> => {
  const response = await fetch('/api/strategies');
  if (!response.ok) {
    throw new Error(`Failed to list strategies: ${response.status} ${response.statusText}`);
  }
  return response.json();
};

export const backtestSavedStrategy = async (
  strategy: string,
  symbol: string,
  timeframe: string,
  numBars: number
): Promise<any> => {
  const params = new URLSearchParams({
    strategy,
    symbol,
    timeframe,
    num_bars: String(numBars),
  });
  const response = await fetch(`/api/strategies/backtest?${params.toString()}`);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Backtest failed: ${response.status} ${response.statusText}`);
  }
  return response.json();
};
