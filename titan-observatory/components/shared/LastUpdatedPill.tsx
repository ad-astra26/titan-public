'use client';

// ── Last-Updated Pill (rFP §5.1 Phase 4, §8 Q4-Q5) ──────────────
// Universal page-freshness indicator. Render on EVERY dashboard page —
// observers learn the vocabulary once, trust the timestamp everywhere.
//
// Maker §8 Q5 (locked 2026-05-14): silent stale — no yellow badge. The
// timestamp itself IS the freshness signal. A separate stale badge
// creates anxiety where the smart cache should be invisible.

import { useEffect, useState } from 'react';

interface LastUpdatedPillProps {
  /** Epoch ms of the most recent successful fetch. */
  fetchedAt: number | undefined;
  /** True if React Query is currently revalidating in the background. */
  isFetching?: boolean;
  /** Optional title context, e.g. "Trinity · T1". */
  context?: string;
  /** Tailwind className override (for pill placement). */
  className?: string;
}

function formatAge(ms: number): string {
  if (ms < 1_000) return 'just now';
  const s = Math.floor(ms / 1_000);
  if (s < 60) return `${s}s ago`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  return `${h}h ago`;
}

export default function LastUpdatedPill({
  fetchedAt,
  isFetching = false,
  context,
  className,
}: LastUpdatedPillProps) {
  const [, force] = useState(0);

  useEffect(() => {
    if (!fetchedAt) return;
    // Tick the age string every second so "3s ago" → "4s ago" smoothly.
    const t = setInterval(() => force((n) => n + 1), 1_000);
    return () => clearInterval(t);
  }, [fetchedAt]);

  if (!fetchedAt) return null;

  const age = formatAge(Date.now() - fetchedAt);

  return (
    <div
      className={
        className ??
        'inline-flex items-center gap-2 px-2 py-0.5 rounded-full bg-slate-900/40 backdrop-blur-sm border border-slate-700/40 text-xs text-slate-400'
      }
      title={`Last updated at ${new Date(fetchedAt).toLocaleString()}`}
      aria-live="polite"
    >
      {context && <span className="text-slate-500">{context}</span>}
      <span>Updated {age}</span>
      {isFetching && (
        <span className="inline-flex h-1.5 w-1.5">
          <span className="animate-ping absolute inline-flex h-1.5 w-1.5 rounded-full bg-emerald-400 opacity-60" />
          <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-emerald-400" />
        </span>
      )}
    </div>
  );
}
