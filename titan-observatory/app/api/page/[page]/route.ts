// ── Page-Aggregate Route Handler: /api/page/<page> ──────────────
// Collapses N upstream calls into 1 round-trip. Each upstream still goes
// through the same in-memory cache as /api/v6-cached/* — page-aggregate
// re-uses Phase 2 cache for free (N cache hits → 5ms total).
//
// URL: /api/page/<page>?titan=T1
// Example: /api/page/trinity?titan=T2
//   → server-side Promise.all over 6 cached fetches
//   → single JSON response: { trinity, neuromodulators, chi, dreaming,
//                              sphereClocks, metabolism, _meta }

import { NextRequest, NextResponse } from 'next/server';
import { cachedFetch } from '@/lib/bff/cache';
import { fetchUpstream, parseTitanParam } from '@/lib/bff/upstream';
import { TTL_POLICY, PAGE_AGGREGATES, type PageSlug } from '@/lib/bff/registry';

export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';

function isPageSlug(s: string): s is PageSlug {
  return s in PAGE_AGGREGATES;
}

export async function GET(
  req: NextRequest,
  { params }: { params: { page: string } },
) {
  if (!isPageSlug(params.page)) {
    return NextResponse.json(
      { error: `unknown page: ${params.page}` },
      { status: 404 },
    );
  }

  const titan = parseTitanParam(req.nextUrl.searchParams.get('titan'));
  const bindings = PAGE_AGGREGATES[params.page];

  // Parallel fan-out. Each fetch hits the SAME shared in-memory cache used
  // by /api/v6-cached/<endpoint>, so when both layers are queried, the
  // second one returns instantly from the warmed entry.
  const results = await Promise.allSettled(
    bindings.map((b) => {
      const policy = TTL_POLICY[b.endpoint];
      return cachedFetch(
        `${titan}:${b.endpoint}`,
        () => fetchUpstream(titan, b.endpoint),
        policy.ttlMs,
        policy.staleMaxMs,
      );
    }),
  );

  const payload: Record<string, unknown> = {};
  let anyStale = false;
  let oldestFetchedAt = Date.now();
  const errors: Array<{ endpoint: string; message: string }> = [];

  bindings.forEach((b, i) => {
    const r = results[i];
    if (r.status === 'fulfilled') {
      payload[b.resultKey] = r.value.value;
      if (r.value.stale) anyStale = true;
      oldestFetchedAt = Math.min(oldestFetchedAt, r.value.fetchedAt);
    } else {
      payload[b.resultKey] = null;
      errors.push({
        endpoint: b.endpoint,
        message: r.reason instanceof Error ? r.reason.message : String(r.reason),
      });
    }
  });

  payload._meta = {
    titan,
    page: params.page,
    stale: anyStale,
    fetchedAt: oldestFetchedAt,
    errors: errors.length ? errors : undefined,
  };

  // 502 only if EVERY upstream errored — partial degradation still returns 200.
  const status = errors.length === bindings.length ? 502 : 200;
  return NextResponse.json(payload, {
    status,
    headers: {
      'x-cache-stale': anyStale ? '1' : '0',
      'x-cache-fetched-at': String(oldestFetchedAt),
    },
  });
}
