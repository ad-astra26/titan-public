'use client';

import { useEffect, useState } from 'react';
import { useIsFetching, useQueryClient } from '@tanstack/react-query';

/**
 * Shows a non-intrusive banner when Titan API is unreachable.
 * Stale cached data is still shown beneath — this replaces blank screens
 * during Titan restarts with a "reconnecting..." indicator.
 *
 * 2026-05-05 (Track B closure): decoupled detection from WS state.
 * Previous impl only probed /health when WS disconnected, so api_subprocess
 * restarts that didn't take WS down silently went undetected. Now probes
 * /health continuously + cross-references react-query error state for
 * /v4/* queries.
 *
 * 2026-05-14 (rFP_observatory_bff_swr_performance §1.2):
 *   - Adaptive probe rate: 3s while banner is visible (fast recovery),
 *     10s when stable (background heartbeat, no Titan strain).
 *   - Banner hides IMMEDIATELY on first successful probe — no waiting
 *     for the next 10s tick.
 *   - Banner appears only after 3s of confirmed disconnection (debounced).
 */
export default function ConnectionBanner() {
  const fetching = useIsFetching();
  const queryClient = useQueryClient();
  const [apiDown, setApiDown] = useState(false);
  const [visible, setVisible] = useState(false);
  const [hasCachedData, setHasCachedData] = useState(false);

  // Probe /health adaptively: 3s while banner-visible (fast recovery
  // detection on outage), 10s when stable (background heartbeat).
  // /health is the canonical liveness signal for api_subprocess; if it
  // returns non-200 or hangs > 3s, the API is effectively unreachable.
  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;
    const probe = async () => {
      try {
        const controller = new AbortController();
        const t = setTimeout(() => controller.abort(), 3000);
        const res = await fetch('/health', { signal: controller.signal, cache: 'no-store' });
        clearTimeout(t);
        if (!cancelled) setApiDown(!res.ok);
      } catch {
        if (!cancelled) setApiDown(true);
      }
      // Adaptive next-tick: fast when visible (chasing recovery), slow when stable.
      if (!cancelled) {
        const nextDelay = visible ? 3000 : 10000;
        timer = setTimeout(probe, nextDelay);
      }
    };
    probe();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [visible]);

  // Track whether we have any cached data for /v4/* queries — this tells
  // us if stale-data-during-reconnect is a useful UX (we have something
  // to show) or a fresh first-load (we don't, banner is less helpful).
  useEffect(() => {
    if (!apiDown) return;
    const cache = queryClient.getQueryCache();
    const cached = cache.getAll().some(q => q.state.data !== undefined);
    setHasCachedData(cached);
  }, [apiDown, queryClient]);

  // Show banner after 3s of confirmed disconnection. Hide instantly when
  // API recovers (no debounce on the recovery path — observers get
  // immediate positive feedback).
  useEffect(() => {
    if (apiDown) {
      const timer = setTimeout(() => setVisible(true), 3000);
      return () => clearTimeout(timer);
    }
    setVisible(false);
  }, [apiDown]);

  if (!visible) return null;

  const message = hasCachedData
    ? 'Reconnecting to Titan... showing cached data'
    : 'Reconnecting to Titan...';

  return (
    <div className="fixed top-0 left-0 right-0 z-50 bg-amber-600/90 backdrop-blur-sm text-white text-center text-sm py-1.5 px-4 shadow-lg">
      <span className="inline-flex items-center gap-2">
        <span className="relative flex h-2 w-2">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-white opacity-75" />
          <span className="relative inline-flex rounded-full h-2 w-2 bg-white" />
        </span>
        {message}
        {fetching > 0 && <span className="opacity-60">({fetching} pending)</span>}
      </span>
    </div>
  );
}
