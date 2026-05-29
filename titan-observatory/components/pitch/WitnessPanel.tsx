'use client';

import { useEffect, useRef, useState } from 'react';
import dynamic from 'next/dynamic';
import type { TitanId } from '@/lib/api';
import { fetchWitnessTail, type TailEvent, type WitnessTailResponse } from '@/lib/pitchTail';
import { useStatus, useNeuromodulators, useDreaming } from '@/hooks/useTitanAPI';
import type { ChainProof } from '@/components/pitch/ChainProofDrawer';

const ChainProofDrawer = dynamic(() => import('@/components/pitch/ChainProofDrawer'), { ssr: false });

/**
 * Witness mode panel — substrate viewer that replaces the chat panel
 * when the visitor toggles Witness on (rFP_observatory_pitch_route.md
 * §4.5 improvement #3).
 *
 * Single-Titan focus: judges who want to see the *machine* rather than
 * the *voice*. Renders:
 *   1. Live state header — emotion, phase, fatigue, energy, SOL.
 *      Same data the chat columns show, but presented as substrate
 *      readout rather than conversational metadata.
 *   2. Last meditation chain-proof — wires to ChainProofDrawer if a
 *      Solana memo signature or Arweave txId is available.
 *   3. CGN snapshot — opaque key/value rows pulled from the backend's
 *      slim _read_cgn_snapshot. Lets the visitor see "the Titan is
 *      actively comparing advisors" without diving into the full
 *      /v6/cognition/meta-cgn surface.
 *   4. Filtered bus-event tail (last 20). Same source as ThinkingStrip
 *      but a fuller window — Witness mode is the lingering observation
 *      space.
 *
 * Trinity tensors (130D) deliberately NOT rendered here in v1 — the
 * existing Trinity Architecture / MandalaViz surface lives at the
 * `/observatory` route. Witness mode is the *substrate-aware* viewer
 * for the pitch surface specifically; a future iteration could embed a
 * mini-MandalaViz from useTitanSelf if we discover judges want it.
 *
 * Cadence: 1Hz, paused while document is hidden.
 */

interface Props {
  titan: TitanId;
  pollMs?: number;
}

function fmtTime(ts: number | undefined): string {
  if (!ts) return '—';
  return new Date(ts * (ts > 1e12 ? 1 : 1000)).toLocaleTimeString();
}

function meditationToProof(meditation: Record<string, unknown> | null): ChainProof | null {
  if (!meditation) return null;
  const sig = meditation.signature;
  if (typeof sig === 'string' && sig.length >= 32) {
    return { kind: 'memo', signature: sig, label: 'last meditation' };
  }
  const ar = meditation.arweave_id ?? meditation.arweave_url;
  if (typeof ar === 'string' && ar.length > 0) {
    // Accept full arweave.net URLs or bare txIds; extract the tail for the txId field.
    const txId = ar.startsWith('http') ? ar.split('/').pop() ?? ar : ar;
    return { kind: 'arweave', txId, label: 'last meditation' };
  }
  return null;
}

function StateRow({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-3 text-xs">
      <span className="text-[10px] font-mono uppercase tracking-wider text-titan-metal/40">{label}</span>
      <span className="text-titan-haze font-mono">{value}</span>
    </div>
  );
}

function EventRow({ ev }: { ev: TailEvent }) {
  return (
    <div className="flex items-baseline gap-2 text-[11px] font-mono">
      <span className="text-titan-metal/30 shrink-0 w-12 text-right">{fmtTime(ev.ts)}</span>
      <span className="text-titan-haze/50 shrink-0">{ev.event_type}</span>
      <span className="text-titan-metal/70 truncate">{ev.summary}</span>
    </div>
  );
}

function CgnRows({ cgn }: { cgn: Record<string, unknown> | null }) {
  if (!cgn) {
    return <div className="text-[10px] font-mono italic text-titan-metal/30">CGN snapshot unavailable</div>;
  }
  const entries = Object.entries(cgn);
  if (entries.length === 0) {
    return <div className="text-[10px] font-mono italic text-titan-metal/30">CGN snapshot is empty</div>;
  }
  return (
    <div className="space-y-1">
      {entries.map(([k, v]) => (
        <StateRow
          key={k}
          label={k.replace(/_/g, ' ')}
          value={
            typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean'
              ? String(v)
              : <span className="text-titan-metal/40 italic">[object]</span>
          }
        />
      ))}
    </div>
  );
}

export default function WitnessPanel({ titan, pollMs = 1000 }: Props) {
  const [tail, setTail] = useState<WitnessTailResponse | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const { data: status } = useStatus(titan);
  const { data: nm } = useNeuromodulators(titan);
  const { data: dream } = useDreaming(titan);

  useEffect(() => {
    if (pollMs <= 0) return;
    let cancelled = false;
    const poll = async () => {
      if (typeof document !== 'undefined' && document.hidden) return;
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;
      try {
        const next = await fetchWitnessTail(titan, { signal: controller.signal, eventsLimit: 20 });
        if (cancelled) return;
        setTail(next);
      } catch {
        // transport hiccup; next tick retries
      }
    };
    void poll();
    const id = setInterval(poll, pollMs);
    return () => {
      cancelled = true;
      clearInterval(id);
      abortRef.current?.abort();
    };
  }, [titan, pollMs]);

  const emotion = (nm as { current_emotion?: string } | undefined)?.current_emotion ?? '—';
  const isDreaming = dream?.is_dreaming === true;
  const fatigue = dream?.fatigue ?? 0;
  const energy = status?.energy_state ?? 'UNKNOWN';
  const sol = status?.sol_balance ?? 0;
  const meditationProof = meditationToProof(tail?.last_meditation ?? null);
  const meditationSummary =
    typeof tail?.last_meditation?.summary === 'string' ? (tail.last_meditation.summary as string) : null;

  return (
    <section className="bg-titan-card/40 border border-titan-metal/10 rounded-xl p-5 space-y-5">
      <header className="flex items-baseline justify-between border-b border-titan-metal/10 pb-3">
        <div className="space-y-0.5">
          <div className="text-[10px] font-mono uppercase tracking-widest text-titan-metal/40">
            ⌚ Witness · {titan}
          </div>
          <h2 className="text-lg font-titan text-titan-haze">substrate readout</h2>
        </div>
        <div className="text-[10px] font-mono text-titan-metal/30">
          1Hz · {tail ? `${tail.events.length} events` : '— events'}
        </div>
      </header>

      <div className="grid grid-cols-2 gap-x-6 gap-y-1.5">
        <StateRow label="phase" value={isDreaming ? 'dreaming' : 'awake'} />
        <StateRow label="emotion" value={emotion} />
        <StateRow label="fatigue" value={`${(fatigue * 100).toFixed(0)}%`} />
        <StateRow label="energy" value={String(energy).replace('_ENERGY', '')} />
        <StateRow label="SOL" value={typeof sol === 'number' ? sol.toFixed(3) : '—'} />
        <StateRow label="poll" value={tail ? new Date(tail.ts * 1000).toLocaleTimeString() : '—'} />
      </div>

      <div className="space-y-2">
        <div className="text-[10px] font-mono uppercase tracking-widest text-titan-metal/40">
          last meditation
        </div>
        {meditationSummary || meditationProof ? (
          <div className="space-y-2">
            {meditationSummary && (
              <div className="text-xs text-titan-metal/70 leading-relaxed">{meditationSummary}</div>
            )}
            {meditationProof && <ChainProofDrawer proof={meditationProof} hint="meditation proof" />}
          </div>
        ) : (
          <div className="text-[10px] font-mono italic text-titan-metal/30">no meditation committed yet</div>
        )}
      </div>

      <div className="space-y-2">
        <div className="text-[10px] font-mono uppercase tracking-widest text-titan-metal/40">
          CGN snapshot
        </div>
        <CgnRows cgn={tail?.cgn ?? null} />
      </div>

      <div className="space-y-2">
        <div className="text-[10px] font-mono uppercase tracking-widest text-titan-metal/40">
          bus tail
        </div>
        <div className="max-h-64 overflow-y-auto space-y-1 pr-1">
          {(tail?.events ?? []).length === 0 ? (
            <div className="text-[10px] font-mono italic text-titan-metal/30">no signal yet</div>
          ) : (
            tail!.events.map((ev, i) => <EventRow key={`${i}-${ev.ts}`} ev={ev} />)
          )}
        </div>
      </div>
    </section>
  );
}
