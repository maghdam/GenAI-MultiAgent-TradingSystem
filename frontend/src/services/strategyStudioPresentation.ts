export const STUDIO_LIFECYCLE_STAGES = [
  'draft',
  'backtested',
  'validated',
  'paper',
  'eligible',
] as const;

export const STUDIO_VALIDATION_OPTIONS = [
  { value: 'development_backtest', label: 'Development 70%' },
  { value: 'out_of_sample', label: 'Holdout 30%' },
  { value: 'regime', label: 'Regime / alternate market' },
  { value: 'walk_forward', label: 'Walk-forward (3 folds)' },
] as const;

export type StudioLifecycleStage = (typeof STUDIO_LIFECYCLE_STAGES)[number];

export function lifecycleStageState(
  currentStage: string | null | undefined,
  stage: StudioLifecycleStage,
): 'complete' | 'current' | 'pending' {
  const currentIndex = STUDIO_LIFECYCLE_STAGES.indexOf(currentStage as StudioLifecycleStage);
  const stageIndex = STUDIO_LIFECYCLE_STAGES.indexOf(stage);
  if (currentIndex < 0) return 'pending';
  if (stageIndex < currentIndex) return 'complete';
  if (stageIndex === currentIndex) return 'current';
  return 'pending';
}

type ResultContext = Record<string, unknown>;

function objectOrEmpty(value: unknown): ResultContext {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? value as ResultContext : {};
}

function positiveBarCount(value: unknown): number | null {
  const number = typeof value === 'number' ? value : Number(value);
  return Number.isSafeInteger(number) && number > 0 ? number : null;
}

/**
 * Display the completed backtest's immutable data provenance, never the live
 * request controls (which the operator can change without running another test).
 */
export function studioBacktestResultTitle(result: unknown, metadata?: unknown): string {
  const report = objectOrEmpty(result);
  if (report['Total Return [%]'] === undefined && report.metrics === undefined) {
    return 'No backtest result yet';
  }
  const meta = objectOrEmpty(metadata);
  const symbol = String(report.symbol || meta.symbol || 'Unknown symbol').toUpperCase();
  const timeframe = String(report.timeframe || meta.timeframe || 'Unknown timeframe').toUpperCase();
  const requested = positiveBarCount(meta.numBars);
  const fetched = positiveBarCount(report['Fetched Bars']);
  const selected = positiveBarCount(report['Selected Bars']);
  const depth = [
    requested === null ? null : `${requested} requested`,
    fetched === null ? null : `${fetched} fetched`,
    selected === null ? null : `${selected} tested`,
  ].filter((part): part is string => part !== null);
  return [symbol, timeframe, ...depth].join(' · ');
}
