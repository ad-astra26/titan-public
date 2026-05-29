'use client';

import { useSearchParams, useRouter, usePathname } from 'next/navigation';
import { Suspense, useEffect, useState } from 'react';
import type { TitanId } from '@/lib/api';

const TITANS: { id: TitanId; label: string }[] = [
  { id: 'T1', label: 'T1' },
  { id: 'T2', label: 'T2' },
  { id: 'T3', label: 'T3' },
];

/** Read the current titan from URL ?titan= param */
export function useTitanId(): TitanId {
  const searchParams = useSearchParams();
  const raw = searchParams.get('titan');
  if (raw === 'T2' || raw === 'T3') return raw;
  return 'T1';
}

function TitanSelectorInner() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();
  const titanId = useTitanId();

  // Health check dots — ping each titan's /health.
  //
  // State machine per Titan (added 2026-05-14, rFP_observatory_bff_swr_performance §1.1):
  //   unknown  → first probe pending
  //   green    → last probe succeeded (immediate recovery on any success)
  //   checking → 1 consecutive failure (amber dot, NOT red yet)
  //   red      → 2+ consecutive failures (real outage signal)
  //
  // Previously a single 3s probe timeout (transient blip, brief Titan GC pause,
  // nginx upstream hiccup) flipped the dot red instantly — undermined observer
  // confidence. Now we require 2 consecutive failures before red and show an
  // intermediate amber "verifying" state during the first failure.
  type DotState = 'unknown' | 'green' | 'checking' | 'red';
  interface HealthEntry { status: DotState; failures: number }
  const [health, setHealth] = useState<Record<TitanId, HealthEntry>>({
    T1: { status: 'unknown', failures: 0 },
    T2: { status: 'unknown', failures: 0 },
    T3: { status: 'unknown', failures: 0 },
  });

  useEffect(() => {
    // Client-side fetches use a same-origin relative URL — nginx (prod) /
    // Next.js rewrites (dev) handle the /t2 + /t3 prefix routing. Using
    // NEXT_PUBLIC_TITAN_API_URL directly here would trigger CORS against
    // https://iamtitan.tech from localhost dev sessions (caught in browser
    // console: "blocked by CORS policy: No 'Access-Control-Allow-Origin'").
    // Per the lib/api.ts:_resolveApiBase contract — empty string browser-side.
    const API_BASE = '';
    const prefixes: Record<TitanId, string> = { T1: '', T2: '/t2', T3: '/t3' };
    const FAILS_BEFORE_RED = 2;

    const check = async (id: TitanId) => {
      let ok = false;
      try {
        const r = await fetch(`${API_BASE}${prefixes[id]}/health`, {
          signal: AbortSignal.timeout(3000),
        });
        ok = r.ok;
      } catch {
        ok = false;
      }
      setHealth((h) => {
        const prev = h[id];
        if (ok) {
          // Any success → green + reset failure counter (immediate recovery).
          if (prev.status === 'green' && prev.failures === 0) return h;
          return { ...h, [id]: { status: 'green', failures: 0 } };
        }
        const failures = prev.failures + 1;
        const status: DotState = failures >= FAILS_BEFORE_RED ? 'red' : 'checking';
        return { ...h, [id]: { status, failures } };
      });
    };

    TITANS.forEach((t) => check(t.id));
    // Poll every 15s (was 30s) — faster debounce convergence + still polite to Titans.
    const iv = setInterval(() => TITANS.forEach((t) => check(t.id)), 15000);
    return () => clearInterval(iv);
  }, []);

  const handleSelect = (id: TitanId) => {
    const params = new URLSearchParams(searchParams.toString());
    if (id === 'T1') {
      params.delete('titan');
    } else {
      params.set('titan', id);
    }
    const qs = params.toString();
    router.replace(`${pathname}${qs ? '?' + qs : ''}`, { scroll: false });
  };

  return (
    <div className="flex items-center gap-1.5 mb-3">
      <span className="text-[10px] text-titan-metal/40 uppercase tracking-wider mr-1">Instance</span>
      {TITANS.map((t) => {
        const isActive = titanId === t.id;
        const dotState = health[t.id].status;
        // Dot color by state machine. Amber 'checking' is the new intermediate
        // shown after a single probe failure — observers see "verifying" not "down".
        const dotClass =
          dotState === 'green'    ? 'bg-emerald-400' :
          dotState === 'checking' ? 'bg-amber-400 animate-pulse' :
          dotState === 'red'      ? 'bg-red-400' :
                                    'bg-titan-metal/30';  // unknown (pre-first-probe)
        const dotTitle =
          dotState === 'green'    ? `${t.id} healthy` :
          dotState === 'checking' ? `${t.id} verifying (1 missed probe)` :
          dotState === 'red'      ? `${t.id} unreachable (${health[t.id].failures} probes failed)` :
                                    `${t.id} status pending`;
        return (
          <button
            key={t.id}
            onClick={() => handleSelect(t.id)}
            className={`relative px-3 py-1 text-xs font-medium rounded-full border transition-all duration-200 ${
              isActive
                ? 'bg-titan-haze/15 text-titan-haze border-titan-haze/40 shadow-sm shadow-titan-haze/10'
                : 'text-titan-metal/50 border-titan-metal/20 hover:text-titan-metal/80 hover:border-titan-metal/40'
            }`}
          >
            {t.label}
            <span
              title={dotTitle}
              className={`absolute -top-0.5 -right-0.5 w-1.5 h-1.5 rounded-full ${dotClass}`}
            />
          </button>
        );
      })}
    </div>
  );
}

export default function TitanSelector() {
  return (
    <Suspense fallback={<div className="h-7 mb-3" />}>
      <TitanSelectorInner />
    </Suspense>
  );
}
