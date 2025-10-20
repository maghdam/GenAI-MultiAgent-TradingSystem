export interface AnalysisResult {
  signal?: string | null;
  confidence?: number | null;
  rationale?: string | null;
  reasons?: string[];
  entry?: number | null;
  sl?: number | null;
  tp?: number | null;
  model?: string | null;
  generated_at?: string | null;
  [key: string]: unknown;
}
