// ── BFF in-memory cache with stale-while-revalidate ────────────
// Per-Next-process Map. No cross-process state (matches rFP §2.3).
// Restart clears it — cache is opportunistic, not persistent.
//
// Concurrency invariant (rFP §2.3): single in-flight fetch per key.
// `entry.fetching` flag prevents thundering herd when N tabs request the
// same key during a stale window — only ONE upstream fetch goes out.
//
// Key isolation (rFP §2.3): caller passes pre-built keys that include
// `titan_id` — T1 data cannot leak into T2 cache by construction.

type Entry<T> = {
  value: T;
  fetchedAt: number;
  fetching: boolean;
};

const cache = new Map<string, Entry<unknown>>();

// ── Telemetry counters (consumed by /api/bff-metrics) ──────────
const counters: Record<string, { fresh: number; stale: number; miss: number; error: number }> = {};
function bump(key: string, kind: 'fresh' | 'stale' | 'miss' | 'error') {
  const route = key.split(':').slice(1).join(':'); // strip titan prefix
  counters[route] ??= { fresh: 0, stale: 0, miss: 0, error: 0 };
  counters[route][kind] += 1;
}

export interface CachedResult<T> {
  value: T;
  stale: boolean;
  fetchedAt: number;
}

/** Fetch with stale-while-revalidate semantics.
 *  - Fresh hit (age < ttlMs): return cached, no background work.
 *  - Stale hit (ttlMs ≤ age < staleMaxMs): return cached + kick background refresh.
 *  - Cold miss / past staleMaxMs: fetch synchronously and return fresh value.
 *  - ttlMs=0 ALWAYS forces synchronous fetch (used for /v4/health). */
export async function cachedFetch<T>(
  key: string,
  fetchFn: () => Promise<T>,
  ttlMs: number,
  staleMaxMs: number = ttlMs * 3,
): Promise<CachedResult<T>> {
  const now = Date.now();

  if (ttlMs === 0) {
    const value = await fetchFn();
    cache.set(key, { value, fetchedAt: now, fetching: false });
    bump(key, 'miss');
    return { value, stale: false, fetchedAt: now };
  }

  const entry = cache.get(key) as Entry<T> | undefined;
  const age = entry ? now - entry.fetchedAt : Infinity;

  if (entry && age < ttlMs) {
    bump(key, 'fresh');
    return { value: entry.value, stale: false, fetchedAt: entry.fetchedAt };
  }

  if (entry && age < staleMaxMs) {
    if (!entry.fetching) {
      entry.fetching = true;
      fetchFn()
        .then((v) => cache.set(key, { value: v, fetchedAt: Date.now(), fetching: false }))
        .catch(() => {
          entry.fetching = false;
          bump(key, 'error');
        });
    }
    bump(key, 'stale');
    return { value: entry.value, stale: true, fetchedAt: entry.fetchedAt };
  }

  try {
    const value = await fetchFn();
    cache.set(key, { value, fetchedAt: now, fetching: false });
    bump(key, 'miss');
    return { value, stale: false, fetchedAt: now };
  } catch (err) {
    bump(key, 'error');
    if (entry) {
      // Past staleMax but upstream failed — better to serve very-stale than 500.
      return { value: entry.value, stale: true, fetchedAt: entry.fetchedAt };
    }
    throw err;
  }
}

/** Telemetry snapshot for /api/bff-metrics + arch_map observatory-bff. */
export function snapshotCounters() {
  return JSON.parse(JSON.stringify(counters)) as typeof counters;
}

/** Test helper — flushes cache + counters. Not used in production. */
export function _resetCacheForTests() {
  cache.clear();
  for (const k of Object.keys(counters)) delete counters[k];
}
