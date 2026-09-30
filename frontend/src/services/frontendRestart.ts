import type { V2Status, V2StrategyInfo } from './api';

export interface FrontendRestartReaders {
  getStatus: () => Promise<V2Status>;
  getStrategies: () => Promise<V2StrategyInfo[]>;
}

export interface FrontendRestartSnapshot {
  status: V2Status | null;
  strategies: V2StrategyInfo[] | null;
  errors: string[];
  complete: boolean;
}

const errorMessage = (label: string, error: unknown): string => {
  const detail = error instanceof Error ? error.message : String(error || 'unknown error');
  return `${label}: ${detail}`;
};

export async function loadFrontendRestartSnapshot(
  readers: FrontendRestartReaders,
): Promise<FrontendRestartSnapshot> {
  const [statusResult, strategiesResult] = await Promise.allSettled([
    Promise.resolve().then(() => readers.getStatus()),
    Promise.resolve().then(() => readers.getStrategies()),
  ]);

  const errors: string[] = [];
  const status = statusResult.status === 'fulfilled' ? statusResult.value : null;
  const strategies = strategiesResult.status === 'fulfilled' ? strategiesResult.value : null;

  if (statusResult.status === 'rejected') {
    errors.push(errorMessage('status', statusResult.reason));
  }
  if (strategiesResult.status === 'rejected') {
    errors.push(errorMessage('strategies', strategiesResult.reason));
  }

  return {
    status,
    strategies,
    errors,
    complete: status !== null && strategies !== null,
  };
}
