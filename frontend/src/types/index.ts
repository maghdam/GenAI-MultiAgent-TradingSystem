export interface AgentSignal {
  ts: number;
  symbol: string;
  timeframe: string;
  signal: 'long' | 'short' | 'no_trade' | 'error';
  confidence: number;
  rationale: string;
  reasons: string[];
  sl: number | null;
  tp: number | null;
  entry: number | null;
  strategy: string;
}
