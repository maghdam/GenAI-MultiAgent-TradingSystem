export function parseBackendUtc(value?: string | null): Date | null {
  if (!value) return null;
  const text = value.trim();
  if (!text) return null;

  // Backend persistence uses naive ISO strings for UTC timestamps.
  // Browsers interpret a naive ISO string as local time, which shifted the
  // dashboard by the local UTC offset. Only append Z when no zone is present.
  const hasZone = /(?:Z|[+-]\d{2}:?\d{2})$/i.test(text);
  const date = new Date(hasZone ? text : `${text}Z`);
  return Number.isNaN(date.getTime()) ? null : date;
}

export function formatBackendLocalDateTime(
  value?: string | null,
  options?: Intl.DateTimeFormatOptions,
): string {
  const date = parseBackendUtc(value);
  if (!date) return value || '–';
  return date.toLocaleString(undefined, options);
}

export function formatBackendLocalTime(value?: string | null): string {
  const date = parseBackendUtc(value);
  if (!date) return '–';
  return date.toLocaleTimeString();
}
