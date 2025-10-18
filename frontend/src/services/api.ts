import { AgentSignal } from '../types';

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

export interface AgentConfig {
  enabled: boolean;
  watchlist: [string, string][];
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
  tasks: Record<string, { status: string; last_run: string; last_error: string | null }>;
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