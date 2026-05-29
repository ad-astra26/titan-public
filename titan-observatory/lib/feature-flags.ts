// ── Observatory Feature Flags ───────────────────────────────────
// Per-route opt-in for BFF (v6-cached) layer + page-aggregate layer.
// Default ON — flip individual routes OFF here if regression appears.
// One-line revert pattern (rFP §5.2): no deploy needed when env var changes.
//
// Env var override: NEXT_PUBLIC_OBS_BFF_DISABLE="feed,trinity" turns off
// specific routes; left empty = all on.

const DISABLED = new Set(
  (process.env.NEXT_PUBLIC_OBS_BFF_DISABLE ?? '')
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean),
);

const FORCE_BFF_OFF = process.env.NEXT_PUBLIC_OBS_BFF_OFF === '1';

/** Resolves whether a given route uses the BFF path or the raw /v4/* URL. */
export function useBFF(route: string): boolean {
  if (FORCE_BFF_OFF) return false;
  return !DISABLED.has(route);
}

/** Resolves whether a given page should use /api/page/<slug> aggregation. */
export function usePageAggregate(page: string): boolean {
  if (FORCE_BFF_OFF) return false;
  return !DISABLED.has(`page:${page}`);
}
