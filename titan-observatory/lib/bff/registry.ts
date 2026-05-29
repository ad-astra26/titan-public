// ── BFF Route Registry ──────────────────────────────────────────
// Single source of truth for:
//   - Per-endpoint TTL + stale-max windows (locked in rFP §5.1 / §2.4)
//   - Page-aggregate composition (which upstream endpoints each page bundles)
//
// Locked by Maker 2026-05-14 — "TTLs intentionally conservative; never higher
// than what's listed here." 60s would kill the spectator feel.
//
// Phase E (2026-05-22): UPSTREAM paths migrated to the api/v6 single readout
// roof. The Observatory now emits ONLY /v6 (+ unversioned /health) to the Titan
// API — /v3,/v4 are hard-deprecated (301/308). The cache SLUGS callers pass to
// the BFF (e.g. 'inner-trinity', 'timechain/status') are unchanged — only the
// upstream path each slug resolves to moved to v6 (SLUG_TO_V6 below). Backend
// lineage for each route is documented at GET /v6/manifest.

/** Upstream readout paths the BFF cache fronts. /v6 roof + unversioned /health. */
export type V6Endpoint =
  | '/health'
  | '/v6/trinity/inner'
  | '/v6/nervous-system'
  | '/v6/nervous-system/neuromodulators'
  | '/v6/nervous-system/chi'
  | '/v6/dreaming'
  | '/v6/expression/narrated-feed'
  | '/v6/expression/creative-works'
  | '/v6/expression/activity-feed'
  | '/v6/trinity/sphere-clocks'
  | '/v6/timechain/status'
  | '/v6/social/persona-profiles'
  | '/v6/social/persona-telemetry'
  | '/v6/metabolism/gate-status'
  | '/v6/expression/mood-narrative'
  | '/v6/social/pressure';

/** Per-endpoint TTL policy. ttlMs=0 means "no cache, always proxy fresh". */
export interface TtlPolicy {
  ttlMs: number;
  staleMaxMs: number;
}

export const TTL_POLICY: Record<V6Endpoint, TtlPolicy> = {
  '/health':                              { ttlMs: 0,      staleMaxMs: 0 },     // §8 Q1: no cache
  '/v6/trinity/inner':                    { ttlMs: 2_000,  staleMaxMs: 8_000 },
  '/v6/nervous-system':                   { ttlMs: 2_000,  staleMaxMs: 8_000 },
  '/v6/nervous-system/neuromodulators':   { ttlMs: 2_000,  staleMaxMs: 8_000 },
  '/v6/nervous-system/chi':               { ttlMs: 3_000,  staleMaxMs: 10_000 },
  '/v6/dreaming':                         { ttlMs: 5_000,  staleMaxMs: 15_000 },
  '/v6/expression/narrated-feed':         { ttlMs: 3_000,  staleMaxMs: 10_000 },
  '/v6/expression/creative-works':        { ttlMs: 3_000,  staleMaxMs: 10_000 },
  '/v6/expression/activity-feed':         { ttlMs: 3_000,  staleMaxMs: 10_000 },
  '/v6/trinity/sphere-clocks':            { ttlMs: 5_000,  staleMaxMs: 15_000 },
  '/v6/timechain/status':                 { ttlMs: 10_000, staleMaxMs: 30_000 },
  '/v6/social/persona-profiles':          { ttlMs: 30_000, staleMaxMs: 120_000 },
  '/v6/social/persona-telemetry':         { ttlMs: 5_000,  staleMaxMs: 15_000 },
  '/v6/metabolism/gate-status':           { ttlMs: 3_000,  staleMaxMs: 10_000 },
  '/v6/expression/mood-narrative':        { ttlMs: 5_000,  staleMaxMs: 20_000 },
  '/v6/social/pressure':                  { ttlMs: 5_000,  staleMaxMs: 20_000 },
};

/** Stable BFF cache slug (used in /api/v6-cached/<slug>) → upstream v6 path.
 *  Slugs are UNCHANGED from the legacy /v4 era so callers need no edits; only
 *  the upstream path each resolves to moved to the v6 roof. */
export const SLUG_TO_V6: Record<string, V6Endpoint> = {
  'health':                  '/health',
  'inner-trinity':           '/v6/trinity/inner',
  'nervous-system':          '/v6/nervous-system',
  'neuromodulators':         '/v6/nervous-system/neuromodulators',
  'chi':                     '/v6/nervous-system/chi',
  'dreaming':                '/v6/dreaming',
  'narrated-feed':           '/v6/expression/narrated-feed',
  'creative-works':          '/v6/expression/creative-works',
  'activity-feed':           '/v6/expression/activity-feed',
  'sphere-clocks':           '/v6/trinity/sphere-clocks',
  'timechain/status':        '/v6/timechain/status',
  'persona-profiles':        '/v6/social/persona-profiles',
  'persona-telemetry':       '/v6/social/persona-telemetry',
  'metabolism/gate-status':  '/v6/metabolism/gate-status',
  'mood-narrative':          '/v6/expression/mood-narrative',
  'social-pressure':         '/v6/social/pressure',
};

/** Resolve a BFF cache slug (e.g. 'inner-trinity', 'timechain/status') to its
 *  upstream v6 path. Returns null for an unknown slug. */
export function v6PathFromSlug(slug: string): V6Endpoint | null {
  return SLUG_TO_V6[slug] ?? null;
}

// ── Page-aggregate composition ──────────────────────────────────
// Each page-aggregate bundles N upstream calls into one round-trip.
// Server-side Promise.all on the BFF cache layer → N cache hits → ONE response.
// Maker priorities per rFP §3.3: feed + creative first (highest user pain),
// then trinity, timechain, persona, metabolism.

export type PageSlug =
  | 'feed'
  | 'creative'
  | 'trinity'
  | 'timechain'
  | 'persona'
  | 'metabolism';

/** Map page → upstream endpoints to bundle + result key in the aggregate response. */
export interface PageAggregateBinding {
  endpoint: V6Endpoint;
  /** Key in the aggregated response object (e.g. `trinity`, `neuromodulators`). */
  resultKey: string;
}

export const PAGE_AGGREGATES: Record<PageSlug, PageAggregateBinding[]> = {
  feed: [
    { endpoint: '/v6/expression/narrated-feed', resultKey: 'narratedFeed' },
    { endpoint: '/v6/expression/activity-feed', resultKey: 'activityFeed' },
    { endpoint: '/v6/social/persona-telemetry', resultKey: 'personaTelemetry' },
  ],
  creative: [
    { endpoint: '/v6/expression/creative-works', resultKey: 'creativeWorks' },
    { endpoint: '/v6/expression/mood-narrative', resultKey: 'moodNarrative' },
  ],
  trinity: [
    { endpoint: '/v6/trinity/inner',                  resultKey: 'trinity' },
    { endpoint: '/v6/nervous-system/neuromodulators', resultKey: 'neuromodulators' },
    { endpoint: '/v6/nervous-system/chi',             resultKey: 'chi' },
    { endpoint: '/v6/dreaming',                       resultKey: 'dreaming' },
    { endpoint: '/v6/trinity/sphere-clocks',          resultKey: 'sphereClocks' },
    { endpoint: '/v6/metabolism/gate-status',         resultKey: 'metabolism' },
  ],
  timechain: [
    { endpoint: '/v6/timechain/status', resultKey: 'timechain' },
  ],
  persona: [
    { endpoint: '/v6/social/persona-profiles',  resultKey: 'profiles' },
    { endpoint: '/v6/social/persona-telemetry', resultKey: 'telemetry' },
    { endpoint: '/v6/social/pressure',          resultKey: 'socialPressure' },
  ],
  metabolism: [
    { endpoint: '/v6/metabolism/gate-status', resultKey: 'metabolism' },
    { endpoint: '/v6/expression/mood-narrative', resultKey: 'moodNarrative' },
  ],
};
