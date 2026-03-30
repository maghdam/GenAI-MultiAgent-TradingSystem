const API_KEY = import.meta.env.VITE_API_KEY;
const CONFIGURED_API_BASE = (import.meta.env.VITE_API_BASE || '').trim();

const resolveApiBase = (): string => {
  if (CONFIGURED_API_BASE) {
    return CONFIGURED_API_BASE.replace(/\/$/, '');
  }
  if (typeof window !== 'undefined' && window.location.port === '4000') {
    return '';
  }
  if (typeof window !== 'undefined') {
    return '';
  }
  return 'http://127.0.0.1:4000';
};

const toAbsoluteInput = (input: RequestInfo | URL): RequestInfo | URL => {
  const apiBase = resolveApiBase();
  if (!apiBase) {
    return input;
  }
  if (typeof input === 'string') {
    if (/^https?:\/\//i.test(input)) {
      return input;
    }
    if (input.startsWith('/')) {
      return `${apiBase}${input}`;
    }
    return input;
  }
  if (input instanceof URL) {
    if (input.origin === window.location.origin && input.pathname.startsWith('/api/')) {
      return new URL(`${apiBase}${input.pathname}${input.search}`);
    }
    return input;
  }
  return input;
};

export const authFetch = (input: RequestInfo | URL, init: RequestInit = {}) => {
  const headers = new Headers(init.headers || {});
  if (API_KEY) {
    headers.set('x-api-key', API_KEY as string);
  }
  return fetch(toAbsoluteInput(input), { ...init, headers });
};
