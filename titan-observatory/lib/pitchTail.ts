/**
 * Frontend client for the pitch-route tail readouts.
 *   - /v6/pitch/witness-tail   → substrate viewer aux signals (Witness mode)
 *   - /v6/pitch/thinking-tail  → footer event strip for the Pitch page
 *
 * Both endpoints are X-Pitch-Token-gated server-side; same hide-the-token
 * model as pitchChat.ts: the visitor's browser hits a thin Next.js API
 * route which attaches the token from process.env and forwards to the
 * Titan API. The token never lands in a fetchable client chunk.
 *
 * Per rFP_observatory_pitch_route.md §4 (#7 live thinking strip) and
 * §4.5 (#3 Witness mode). Frontend poll cadence is 1Hz — the backend
 * does no server-side caching and the Cache-Control header is no-store,
 * so each poll returns fresh data.
 */

import type { TitanId } from './api';

export interface TailEvent {
  ts: number;
  event_type: string;
  summary: string;
}

export interface ThinkingTailResponse {
  titan_id: string;
  ts: number;
  events: TailEvent[];
}

export interface WitnessTailResponse {
  titan_id: string;
  ts: number;
  events: TailEvent[];
  /** CGN snapshot — shape is intentionally flexible at the backend
   *  (see _read_cgn_snapshot in pitch_chat.py). Rendered as opaque
   *  key/value rows in WitnessPanel for v1. */
  cgn: Record<string, unknown> | null;
  /** Last meditation commit — {ts, signature, arweave_url, summary, …}
   *  when available. Used by WitnessPanel to render a single chain-proof
   *  chevron linking to Solscan / Arweave. */
  last_meditation: Record<string, unknown> | null;
}

/** Poll the witness-tail endpoint for a given Titan. Throws on transport
 *  errors; returns an empty-shape response on 404 (token misconfig). */
export async function fetchWitnessTail(
  titan: TitanId,
  options?: { signal?: AbortSignal; eventsLimit?: number },
): Promise<WitnessTailResponse> {
  const params = new URLSearchParams({ titan });
  if (options?.eventsLimit) params.set('events_limit', String(options.eventsLimit));
  const res = await fetch(`/api/pitch-witness-proxy?${params.toString()}`, {
    method: 'GET',
    cache: 'no-store',
    signal: options?.signal,
  });
  if (res.status === 404) {
    return { titan_id: titan, ts: 0, events: [], cgn: null, last_meditation: null };
  }
  if (!res.ok) {
    throw new Error(`witness_tail HTTP ${res.status}`);
  }
  return (await res.json()) as WitnessTailResponse;
}

/** Poll the thinking-tail endpoint for a given Titan. Same error model
 *  as fetchWitnessTail — empty events on 404, throw on transport. */
export async function fetchThinkingTail(
  titan: TitanId,
  options?: { signal?: AbortSignal; limit?: number },
): Promise<ThinkingTailResponse> {
  const params = new URLSearchParams({ titan });
  if (options?.limit) params.set('limit', String(options.limit));
  const res = await fetch(`/api/pitch-thinking-proxy?${params.toString()}`, {
    method: 'GET',
    cache: 'no-store',
    signal: options?.signal,
  });
  if (res.status === 404) {
    return { titan_id: titan, ts: 0, events: [] };
  }
  if (!res.ok) {
    throw new Error(`thinking_tail HTTP ${res.status}`);
  }
  return (await res.json()) as ThinkingTailResponse;
}
