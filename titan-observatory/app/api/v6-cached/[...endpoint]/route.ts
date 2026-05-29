// ── BFF Route Handler: /api/v6-cached/<endpoint> ────────────────
// Single dynamic handler. Reads TTL from registry. Per-Titan cache key.
// Stale-while-revalidate semantics per rFP §2.2-§2.4 (locked 2026-05-14).
//
// Phase E (2026-05-22): upstream migrated to the api/v6 readout roof. The
// cache SLUG callers pass is unchanged (e.g. 'narrated-feed'); it now resolves
// to a /v6 upstream path via SLUG_TO_V6.
//
// URL: /api/v6-cached/<slug>?titan=T1
// Example: /api/v6-cached/narrated-feed?titan=T2
//   → cache key: "T2:/v6/expression/narrated-feed"
//   → upstream:  http://127.0.0.1:7777/t2/v6/expression/narrated-feed

import { NextRequest, NextResponse } from 'next/server';
import { cachedFetch } from '@/lib/bff/cache';
import { fetchUpstream, parseTitanParam } from '@/lib/bff/upstream';
import { TTL_POLICY, v6PathFromSlug } from '@/lib/bff/registry';

export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';

export async function GET(
  req: NextRequest,
  { params }: { params: { endpoint: string[] } },
) {
  const slug = params.endpoint.join('/');
  const v6Path = v6PathFromSlug(slug);
  if (!v6Path) {
    return NextResponse.json(
      { error: `unknown v6 cache slug: ${slug}` },
      { status: 404 },
    );
  }

  const titan = parseTitanParam(req.nextUrl.searchParams.get('titan'));
  const policy = TTL_POLICY[v6Path];

  // Preserve all query params except `titan` (which routes to the upstream
  // prefix) so `?limit=40` and `?limit=10` cache separately and pass through.
  const passthrough = new URLSearchParams(req.nextUrl.searchParams);
  passthrough.delete('titan');
  const querySuffix = passthrough.toString();
  const upstreamPath = querySuffix ? `${v6Path}?${querySuffix}` : v6Path;
  const cacheKey = `${titan}:${upstreamPath}`;

  try {
    const { value, stale, fetchedAt } = await cachedFetch(
      cacheKey,
      () => fetchUpstream(titan, upstreamPath),
      policy.ttlMs,
      policy.staleMaxMs,
    );
    return NextResponse.json(value, {
      headers: {
        'x-cache-stale': stale ? '1' : '0',
        'x-cache-fetched-at': String(fetchedAt),
        'x-cache-ttl-ms': String(policy.ttlMs),
      },
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return NextResponse.json(
      { error: 'upstream_unreachable', detail: message },
      { status: 502 },
    );
  }
}
