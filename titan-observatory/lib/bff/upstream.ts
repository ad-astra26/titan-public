// ── Server-side upstream fetcher ────────────────────────────────
// Mirrors lib/api.ts titanFetch routing rules but runs INSIDE Next.js
// Route Handlers (Node runtime). Nginx prefixes /t2 + /t3 route to VPC
// Titans transparently; T1 uses unprefixed local 7777.

import type { TitanId } from '@/lib/api';

const UPSTREAM_BASE =
  process.env.TITAN_API_URL_INTERNAL ||
  process.env.NEXT_PUBLIC_TITAN_API_URL ||
  'http://127.0.0.1:7777';

const TITAN_PREFIXES: Record<TitanId, string> = {
  T1: '',
  T2: '/t2',
  T3: '/t3',
};

const UPSTREAM_TIMEOUT_MS = 10_000;

/** Fetch an upstream Titan endpoint, unwrap the {status,data} envelope.
 *  Throws on non-2xx or timeout. */
export async function fetchUpstream<T = unknown>(
  titan: TitanId,
  path: string,
): Promise<T> {
  const url = `${UPSTREAM_BASE}${TITAN_PREFIXES[titan]}${path}`;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), UPSTREAM_TIMEOUT_MS);
  try {
    const res = await fetch(url, {
      signal: controller.signal,
      cache: 'no-store',
      headers: { 'Content-Type': 'application/json' },
    });
    if (!res.ok) {
      throw new Error(`upstream ${titan} ${path} → ${res.status} ${res.statusText}`);
    }
    const json = await res.json();
    if (json && typeof json === 'object' && 'data' in json && (json as { status?: string }).status === 'ok') {
      return (json as { data: T }).data;
    }
    return json as T;
  } catch (err: unknown) {
    if (err instanceof DOMException && err.name === 'AbortError') {
      throw new Error(`upstream ${titan} ${path} timeout after ${UPSTREAM_TIMEOUT_MS}ms`);
    }
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

/** Validate `?titan=` query param. Defaults to T1. Rejects unknown values
 *  to prevent path-injection through the upstream prefix table. */
export function parseTitanParam(raw: string | null): TitanId {
  if (raw === 'T2' || raw === 'T3') return raw;
  return 'T1';
}
