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
