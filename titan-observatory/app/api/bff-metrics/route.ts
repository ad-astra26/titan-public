// ── BFF Telemetry Endpoint ──────────────────────────────────────
// Read-only snapshot of in-memory cache counters. Consumed by
// `scripts/arch_map.py observatory-bff` for fleet-wide visibility.
//
// Output: { route: { fresh, stale, miss, error } } per upstream path.

import { NextResponse } from 'next/server';
import { snapshotCounters } from '@/lib/bff/cache';

export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';

export async function GET() {
  const counters = snapshotCounters();
  let totalFresh = 0, totalStale = 0, totalMiss = 0, totalError = 0;
  for (const c of Object.values(counters)) {
    totalFresh += c.fresh;
    totalStale += c.stale;
    totalMiss += c.miss;
    totalError += c.error;
  }
  const totalRequests = totalFresh + totalStale + totalMiss;
  const hitRate = totalRequests > 0 ? (totalFresh + totalStale) / totalRequests : 0;
  return NextResponse.json({
    routes: counters,
    summary: {
      totalRequests,
      totalFresh,
      totalStale,
      totalMiss,
      totalError,
      hitRate,
    },
    capturedAt: Date.now(),
  });
}
