export interface AnalysisResult {
  signal?: string | null;
  confidence?: number | null;
  rationale?: string | null;
  reasons?: string[];
  entry?: number | null;
  sl?: number | null;
  stop_loss?: number | null;
  tp?: number | null;
  take_profit?: number | null;
  model?: string | null;
  generated_at?: string | null;
  [key: string]: unknown;
}
