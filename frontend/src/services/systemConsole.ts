export const SYSTEM_ACTIVITY_TABS = [
  { id: 'incidents', label: 'Incidents' },
  { id: 'engine', label: 'Engine' },
  { id: 'trades', label: 'Trades' },
  { id: 'decisions', label: 'Decisions' },
  { id: 'history', label: 'History' },
] as const;

export type SystemActivityTab = (typeof SYSTEM_ACTIVITY_TABS)[number]['id'];

export const readinessStats = <T extends { ok: boolean }>(checks: T[]) => {
  const passed = checks.filter((item) => item.ok).length;
  return {
    passed,
    total: checks.length,
    failed: checks.length - passed,
  };
};

export const operatorIncidentHistory = <T extends { level: string }>(items: T[]) =>
  items.filter((item) => item.level.toLowerCase() !== 'info');
