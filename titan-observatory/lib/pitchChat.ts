/**
 * Frontend client for the /v6/pitch/chat wallet-less endpoint
 * (Phase E single readout roof, D-SPEC-115; legacy /v4/pitch-chat
 * still 308-redirects via v6_deprecation, but we call /v6 directly).
 *
 * Per rFP_observatory_pitch_route.md §5. Compare-mode (rFP §4, default
 * view) fans out across T1+T2+T3 via the same per-Titan nginx prefix
 * routing (`/t2/`, `/t3/`) as the other Observatory API calls.
 *
 * The X-Pitch-Token header is NOT exposed to the client JS bundle.
 * Instead the visitor hits a thin Next.js API route at
 * /api/pitch-chat-proxy which lives in the server bundle and forwards
 * the request with the token attached server-side. This keeps the token
 * out of the publicly-fetchable client chunk.
 */

import type { TitanId } from './api';

const TITAN_PREFIXES: Record<TitanId, string> = { T1: '', T2: '/t2', T3: '/t3' };

export interface PitchChatRequest {
  titan: TitanId;
  thread_id: string;
  message: string;
}

export interface InternalTime {
  epoch: number | null;
  phase: string | null; // "awake" | "dreaming" | "meditating"
  fatigue: number | null;
  emotion: string | null;
}

/** Chain-proof reference shape emitted by /v6/pitch/chat on non-declined
 *  replies. Mirrors the backend `PitchChainProof` (pitch_chat.py) and
 *  fits the frontend `ChainProof` discriminated union for the two kinds
 *  the backend currently emits: `memo` + `timechain_block`. The empty
 *  optionals are kept so a future backend kind can ride this shape
 *  without breaking the wire format. */
export interface PitchChainProof {
  kind: 'memo' | 'timechain_block';
  signature?: string | null;
  height?: number | null;
  merkle?: string | null;
  label?: string | null;
}

export interface PitchChatResponse {
  response: string;
  titan: TitanId;
  thread_id: string;
  internal_time: InternalTime;
  declined: boolean;
  decline_reason: string | null;
  decline_explanation: string | null;
  /** 0–N chain-proof references attached to a successful reply
   *  (rFP §4 #5 Pitch — memory references). Empty on decline. */
  proofs: PitchChainProof[];
}

/** Generate a thread_id stable per Pitch session. Allowed character set
 *  matches the backend regex ([A-Za-z0-9_-]{8,64}). */
export function newThreadId(): string {
  const a = Array.from({ length: 24 }, () => Math.floor(Math.random() * 36).toString(36)).join('');
  return `t-${a}`;
}

/** Send one message to one Titan via the local proxy. Resolves with
 *  either a normal reply or a declined envelope (rate limit, dream
 *  phase, jailbreak rejection, etc.). Throws only on transport errors. */
export async function sendPitchChat(
  req: PitchChatRequest,
  options?: { signal?: AbortSignal },
): Promise<PitchChatResponse> {
  const res = await fetch('/api/pitch-chat-proxy', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
    signal: options?.signal,
  });
  if (!res.ok) {
    // 404 = bad token (or no token configured). Surface so the UI can
    // show a clear "this route is unavailable" rather than a stack trace.
    if (res.status === 404) {
      throw new Error('pitch_route_unavailable');
    }
    throw new Error(`pitch_chat HTTP ${res.status}`);
  }
  return (await res.json()) as PitchChatResponse;
}

/** Per-Titan API base used to compute the upstream URL. The proxy reads
 *  these so the client never has to bake them into its chunk. */
export function pitchChatUpstream(apiBase: string, titan: TitanId): string {
  return `${apiBase}${TITAN_PREFIXES[titan]}/v6/pitch/chat`;
}
