// ── Titan API Layer ────────────────────���─────────────────────────
// Unified fetch, multi-Titan routing, and 3-tier caching config.

import { keepPreviousData } from '@tanstack/react-query';

/** API_BASE resolution — split between server-side and browser-side.
 *
 *  Server-side (BFF upstream proxy, SSR queries during page render): returns
 *  the configured env URL (or localhost fallback). Runs inside Next.js,
 *  no browser CORS.
 *
 *  Browser-side (React Query queryFns): returns the EMPTY STRING so every
 *  titanFetch emits a same-origin relative URL (`/v4/...`, `/status`,
 *  `/t2/v4/...`). Same origin = no CORS preflight.
 *
 *    Prod  — nginx on iamtitan.tech proxies those same-origin paths to
 *            FastAPI :7777 (T1) or to T2/T3 backends keyed by /t2 + /t3.
 *    Dev   — Next.js `rewrites` in next.config.mjs forwards those same
 *            paths to http://localhost:7777 (T1) and the VPS LAN host
 *            for T2/T3.
 *
 *  The legacy `NEXT_PUBLIC_TITAN_API_URL=https://iamtitan.tech` config
 *  used to cause every browser fetch from localhost to fail CORS. This
 *  resolver eliminates that class of bug — the env var is preserved
 *  only for server-side paths where CORS doesn't apply. */
function _resolveApiBase(): string {
  if (typeof window === 'undefined') {
    return process.env.NEXT_PUBLIC_TITAN_API_URL || 'http://localhost:7777';
  }
  return '';
}

const API_BASE = _resolveApiBase();

// BFF cache layer is same-origin (Next.js Route Handlers). Resolves to
// http://<observatory-host>/api/v6-cached/... regardless of API_BASE.
const BFF_BASE = process.env.NEXT_PUBLIC_OBS_BFF_BASE || '';

// ── Multi-Titan Routing ─────────────────────────────────────────

export type TitanId = 'T1' | 'T2' | 'T3';

/** Path prefixes for each Titan instance (nginx proxies T2/T3 to VPC) */
const TITAN_PREFIXES: Record<TitanId, string> = {
  T1: '',
  T2: '/t2',
  T3: '/t3',
};

export interface TitanFetchOptions extends RequestInit {
  titan?: TitanId;
}

/** Default fetch timeout (ms). When API event loop is blocked (e.g. during
 *  heavy spirit-worker boot), the TCP connection is accepted but the response
 *  never arrives — without a timeout, fetch() hangs forever and the UI shows
 *  an eternal loading spinner instead of falling back to cached data. */
const FETCH_TIMEOUT_MS = 10_000;

export async function titanFetch<T>(path: string, options?: TitanFetchOptions): Promise<T> {
  const titanId = options?.titan;
  const prefix = titanId ? TITAN_PREFIXES[titanId] : '';
  const url = `${API_BASE}${prefix}${path}`;

  // Strip our custom 'titan' prop before passing to fetch
  const { titan: _titan, ...fetchOptions } = options ?? {};

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);

  try {
    const res = await fetch(url, {
      ...fetchOptions,
      signal: controller.signal,
      headers: {
        'Content-Type': 'application/json',
        ...fetchOptions?.headers,
      },
    });

    if (!res.ok) {
      throw new Error(`Titan API error: ${res.status} ${res.statusText}`);
    }

    const json = await res.json();

    // Unwrap { status: "ok", data: { ... } } envelope from the backend
    if (json && typeof json === 'object' && 'data' in json && json.status === 'ok') {
      return json.data as T;
    }

    return json as T;
  } catch (err: unknown) {
    if (err instanceof DOMException && err.name === 'AbortError') {
      throw new Error(`Titan API timeout after ${FETCH_TIMEOUT_MS}ms: ${path}`);
    }
    throw err;
  } finally {
    clearTimeout(timeout);
  }
}

export async function titanAuthFetch<T>(
  path: string,
  body: Record<string, unknown>,
  signature: string,
  timestamp: string
): Promise<T> {
  return titanFetch<T>(path, {
    method: 'POST',
    headers: {
      'X-Titan-Signature': signature,
      'X-Titan-Timestamp': timestamp,
    },
    body: JSON.stringify(body),
  });
}

// ── 3-Tier Caching Strategy ─────────────────────────────────────
// Consistent refresh intervals for uniform UX across all views.
//
// real-time: Rapidly changing data (neuromods, hormones, clocks)
// active:    Moderate change rate (status, mood, reasoning, chi)
// slow:      Rarely changes (vocabulary, health, history, ARC)

// gcTime intentionally omitted — inherits QueryClient default (24h) so that
// localStorage persistence always has data to hydrate from. Previously each
// tier set gcTime to 30-300s, causing data to be garbage collected before
// persistence could save it — resulting in blank Observatory during restarts.
export const QUERY_TIERS = {
  realtime: { staleTime: 2_000, refetchInterval: 3_000 },
  active:   { staleTime: 8_000, refetchInterval: 10_000 },
  slow:     { staleTime: 25_000, refetchInterval: 30_000 },
} as const;

export type QueryTier = keyof typeof QUERY_TIERS;

/** Build React Query options for a given tier + optional titanId for cache isolation.
 *
 *  Includes `placeholderData: (prev) => prev` (rFP §5.1 Phase 4 / §4.1): when
 *  a user switches T1↔T2↔T3, the OLD titan's data stays visible until the new
 *  titan's data arrives. No loading spinner; no flicker. */
export function tierQueryOptions(tier: QueryTier, queryKey: string[], titanId?: TitanId) {
  const t = QUERY_TIERS[tier];
  return {
    queryKey: titanId ? [...queryKey, titanId] : queryKey,
    staleTime: t.staleTime,
    refetchInterval: t.refetchInterval,
    // gcTime inherited from QueryClient (24h) — don't override here
    retry: 2,
    placeholderData: keepPreviousData,
  };
}

export { API_BASE };

// ── BFF helpers (rFP §5.1 Phase 2/3) ────────────────────────────
// Same-origin fetches against Next.js Route Handlers at /api/v6-cached/*
// and /api/page/*. The Route Handlers proxy through to API_BASE (v6 upstream)
// with per-route TTL + stale-while-revalidate semantics.

/** Resolve a BFF cache slug (e.g. 'narrated-feed') through the BFF cache.
 *  Falls back to direct titanFetch (v6 path) when the feature flag is off. */
export async function bffFetch<T>(
  v4Slug: string,
  options?: { titan?: TitanId; extraQuery?: string },
): Promise<T> {
  const titan = options?.titan ?? 'T1';
  const tail = options?.extraQuery ? `&${options.extraQuery}` : '';
  const url = `${BFF_BASE}/api/v6-cached/${v4Slug}?titan=${titan}${tail}`;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);
  try {
    const res = await fetch(url, {
      signal: controller.signal,
      headers: { 'Content-Type': 'application/json' },
    });
    if (!res.ok) {
      throw new Error(`BFF /v4/${v4Slug} (${titan}) → ${res.status} ${res.statusText}`);
    }
    return (await res.json()) as T;
  } catch (err) {
    if (err instanceof DOMException && err.name === 'AbortError') {
      throw new Error(`BFF /v4/${v4Slug} timeout after ${FETCH_TIMEOUT_MS}ms`);
    }
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

/** Resolve a page-aggregate slug ('feed', 'creative', 'trinity', etc.) and
 *  return the bundled payload. Server-side parallel-fetches all sub-endpoints. */
export async function pageFetch<T>(
  pageSlug: string,
  options?: { titan?: TitanId },
): Promise<T> {
  const titan = options?.titan ?? 'T1';
  const url = `${BFF_BASE}/api/page/${pageSlug}?titan=${titan}`;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);
  try {
    const res = await fetch(url, {
      signal: controller.signal,
      headers: { 'Content-Type': 'application/json' },
    });
    if (!res.ok) {
      throw new Error(`BFF page/${pageSlug} (${titan}) → ${res.status} ${res.statusText}`);
    }
    return (await res.json()) as T;
  } catch (err) {
    if (err instanceof DOMException && err.name === 'AbortError') {
      throw new Error(`BFF page/${pageSlug} timeout after ${FETCH_TIMEOUT_MS}ms`);
    }
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

import { useBFF } from './feature-flags';
import { v6PathFromSlug } from './bff/registry';

/** Picks the right fetch path for a cached endpoint slug based on the feature
 *  flag. Hooks call this so a single env-var flip cleanly routes around BFF.
 *  Query string is preserved through the BFF and folded into the cache key.
 *  Phase E: both paths now emit /v6 — the BFF maps the slug→v6 upstream, and
 *  the direct fallback resolves the slug→v6 path via v6PathFromSlug (the FE
 *  never emits /v4 to the Titan). All v4Fetch callers use cached slugs, so the
 *  slug always resolves; an unmapped slug is a wiring bug (throw). */
export function v4Fetch<T>(
  v4Slug: string,
  options?: { titan?: TitanId; extraQuery?: string },
): Promise<T> {
  if (useBFF(v4Slug)) {
    return bffFetch<T>(v4Slug, options);
  }
  const v6Path = v6PathFromSlug(v4Slug);
  if (!v6Path) {
    throw new Error(`v4Fetch: no v6 mapping for cache slug '${v4Slug}'`);
  }
  const path = options?.extraQuery ? `${v6Path}?${options.extraQuery}` : v6Path;
  return titanFetch<T>(path, { titan: options?.titan });
}
