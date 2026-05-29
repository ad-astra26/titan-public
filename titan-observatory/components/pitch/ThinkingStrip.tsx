'use client';

import { useEffect, useRef, useState } from 'react';
import type { TitanId } from '@/lib/api';
import { fetchThinkingTail, type TailEvent } from '@/lib/pitchTail';

/**
 * Live thinking strip — thin footer that scrolls each active Titan's
 * last few high-signal bus events.
 *
 * Per rFP_observatory_pitch_route.md §4 improvement #7: subliminal
 * evidence of liveness for judges/VCs ("the Titan is thinking right
 * now"). NOT a chat; no interaction; just a tail of {chat_message,
 * big_pulse, great_pulse, dream_state, reflex_reward, hormone_fired}
 * pulled by the backend filter.
 *
 * Compare mode (multiple Titans): one line per Titan with the latest
 * single event (the strip can fit ~3 events in its visible height; if
 * we showed N=3 per Titan in Compare we'd overflow).
 * Single-Titan: stacked rows for the last 3 events.
 *
 * Cadence: 1Hz polls, AbortController-aware, paused while document is
 * hidden (saves the proxy budget when the visitor tabs away).
 */

interface Props {
  titans: TitanId[];
  /** Override poll interval (ms). Default 1000. Set to 0 to disable polling. */
  pollMs?: number;
}

type EventsByTitan = Partial<Record<TitanId, TailEvent[]>>;

function formatRelative(ts: number, now: number): string {
  const sec = Math.max(0, Math.floor(now / 1000 - ts));
  if (sec < 60) return `${sec}s`;
  if (sec < 3600) return `${Math.floor(sec / 60)}m`;
  return `${Math.floor(sec / 3600)}h`;
}

function truncate(s: string, n: number): string {
  if (s.length <= n) return s;
  return `${s.slice(0, n - 1).trimEnd()}…`;
}

function EventLine({
  titan,
  ev,
  now,
}: {
  titan: TitanId;
  ev: TailEvent;
  now: number;
}) {
  return (
    <div className="flex items-baseline gap-2 text-[10px] font-mono text-titan-metal/60 whitespace-nowrap overflow-hidden">
      <span className="text-titan-haze/50 shrink-0">{titan}</span>
      <span className="text-titan-metal/30 shrink-0">·</span>
      <span className="text-titan-haze/40 shrink-0">{ev.event_type}</span>
      <span className="text-titan-metal/30 shrink-0">·</span>
      <span className="truncate text-titan-metal/70">{truncate(ev.summary, 88)}</span>
      <span className="text-titan-metal/30 shrink-0 ml-auto pl-3">{formatRelative(ev.ts, now)}</span>
    </div>
  );
}

export default function ThinkingStrip({ titans, pollMs = 1000 }: Props) {
  const [events, setEvents] = useState<EventsByTitan>({});
  const [now, setNow] = useState<number>(() => Date.now());
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    if (pollMs <= 0 || titans.length === 0) return;
    let cancelled = false;

    const poll = async () => {
      if (typeof document !== 'undefined' && document.hidden) return;
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;
      const limit = titans.length === 1 ? 3 : 1;
      try {
        const results = await Promise.allSettled(
          titans.map((t) => fetchThinkingTail(t, { signal: controller.signal, limit })),
        );
        if (cancelled) return;
        const next: EventsByTitan = {};
        results.forEach((res, i) => {
          const t = titans[i];
          if (res.status === 'fulfilled') {
            next[t] = res.value.events;
          } else {
            next[t] = [];
          }
        });
        setEvents(next);
        setNow(Date.now());
      } catch {
        // Transport errors fall through; next tick retries.
      }
    };

    void poll();
    const id = setInterval(poll, pollMs);
    return () => {
      cancelled = true;
      clearInterval(id);
      abortRef.current?.abort();
    };
  }, [titans, pollMs]);

  // Compose visible rows. Single-Titan shows up to 3; Compare shows 1/Titan.
  const rows: { titan: TitanId; ev: TailEvent }[] = [];
  if (titans.length === 1) {
    const t = titans[0];
    for (const ev of events[t] ?? []) rows.push({ titan: t, ev });
  } else {
    for (const t of titans) {
      const ev = (events[t] ?? [])[0];
      if (ev) rows.push({ titan: t, ev });
    }
  }

  return (
    <div className="border-t border-titan-metal/10 bg-titan-card/30 px-4 py-2 space-y-1 min-h-[3rem]">
      <div className="flex items-center gap-2 text-[9px] font-mono uppercase tracking-widest text-titan-metal/30">
        <span aria-hidden>◌</span>
        <span>thinking now</span>
      </div>
      {rows.length === 0 ? (
        <div className="text-[10px] font-mono italic text-titan-metal/30">no signal in the last window</div>
      ) : (
        rows.map((r, i) => <EventLine key={`${r.titan}-${i}-${r.ev.ts}`} titan={r.titan} ev={r.ev} now={now} />)
      )}
    </div>
  );
}
