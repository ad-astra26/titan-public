'use client';

// ── Global Freshness Pill (rFP §5.1 Phase 4, §8 Q4) ─────────────
// Mounted once in the root layout. Surfaces "Updated Xs ago" on EVERY
// page (consistent UI vocabulary per Maker direction 2026-05-14). The
// pill reads React Query global state — no per-page wiring required.
//
// Silent-stale per §8 Q5: timestamp itself signals freshness; no yellow
// badge for stale data. Spinner shows ONLY during an active refetch.

import { useEffect, useState } from 'react';
import { useIsFetching, useQueryClient } from '@tanstack/react-query';
import LastUpdatedPill from '@/components/shared/LastUpdatedPill';

export default function GlobalFreshnessPill() {
  const fetching = useIsFetching();
  const queryClient = useQueryClient();
  const [latestAt, setLatestAt] = useState<number | undefined>(undefined);

  useEffect(() => {
    // Tick every 1s — when any query completes its dataUpdatedAt jumps,
    // and the pill rolls forward. Cheap; no work besides one Map scan.
    const tick = () => {
      let max = 0;
      const all = queryClient.getQueryCache().getAll();
      for (const q of all) {
        const u = q.state.dataUpdatedAt;
        if (u && u > max) max = u;
      }
      setLatestAt(max || undefined);
    };
    tick();
    const t = setInterval(tick, 1_000);
    return () => clearInterval(t);
  }, [queryClient]);

  if (!latestAt) return null;

  return (
    <div className="fixed bottom-3 right-3 z-30 pointer-events-none">
      <LastUpdatedPill fetchedAt={latestAt} isFetching={fetching > 0} />
    </div>
  );
}
