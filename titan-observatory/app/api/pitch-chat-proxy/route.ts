import { NextResponse } from 'next/server';
import { getPitchToken } from '@/lib/pitchToken';
import type { PitchChatRequest } from '@/lib/pitchChat';
import type { TitanId } from '@/lib/api';

/**
 * Server-side proxy for /v6/pitch/chat (Phase E single readout roof,
 * D-SPEC-115; was /v4/pitch-chat — legacy path still 308-redirects via
 * v6_deprecation). The X-Pitch-Token must never land in the public
 * client bundle, so this route runs on the Next.js server, attaches the
 * token from process.env, and forwards to the Titan API (T1 directly;
 * T2/T3 via the same nginx prefix routing as the rest of the
 * Observatory).
 *
 * No auth on this proxy itself — it lives at /api/pitch-chat-proxy
 * and is callable by anyone, BUT the upstream rejects every request
 * without a valid PITCH_TOKEN, so leakage is non-issue. The visitor
 * still has to reach a /v/<token>/* route first to get to the chat UI.
 *
 * Per rFP_observatory_pitch_route.md §5.
 */

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const VALID_TITANS = new Set<TitanId>(['T1', 'T2', 'T3']);

// Per-Titan upstream URLs (server-side only). The proxy resolves to the
// internal address of each Titan's API instance directly, bypassing nginx.
// Going through nginx (https://iamtitan.tech) would force every LLM
// roundtrip — 6 to 90 seconds — to traverse nginx's proxy_read_timeout
// (default 60s) and return 504 Gateway Time-out long before the backend
// replies. The proxy runs on the SAME network as the Titan VPCs, so it
// can reach each Titan over the internal 10.135.0.0/16 VPN.
//
// Defaults match the fleet wiring on T1's host:
//   T1: localhost:7777            (this machine)
//   T2: 10.135.0.6:7777           (shared VPS, T2 instance)
//   T3: 10.135.0.6:7778           (shared VPS, T3 instance — port differs)
// Override any of these with TITAN_T{1,2,3}_INTERNAL_URL env vars.
const TITAN_INTERNAL_URLS: Record<TitanId, string> = {
  T1: process.env.TITAN_T1_INTERNAL_URL || 'http://localhost:7777',
  T2: process.env.TITAN_T2_INTERNAL_URL || 'http://10.135.0.6:7777',
  T3: process.env.TITAN_T3_INTERNAL_URL || 'http://10.135.0.6:7778',
};
const FETCH_TIMEOUT_MS = 195_000; // Match backend bridge (180s) + margin for cold-start cache load.

interface ProxyError {
  error: string;
  detail?: string;
}

function isPitchChatRequest(body: unknown): body is PitchChatRequest {
  if (!body || typeof body !== 'object') return false;
  const b = body as Record<string, unknown>;
  return (
    typeof b.titan === 'string' &&
    typeof b.thread_id === 'string' &&
    typeof b.message === 'string' &&
    VALID_TITANS.has(b.titan as TitanId) &&
    b.thread_id.length >= 8 &&
    b.thread_id.length <= 64 &&
    b.message.length >= 1 &&
    b.message.length <= 500
  );
}

export async function POST(req: Request) {
  const token = getPitchToken();
  if (!token || token.length < 24) {
    const body: ProxyError = { error: 'pitch_route_not_configured' };
    return NextResponse.json(body, { status: 404 });
  }

  let parsed: unknown;
  try {
    parsed = await req.json();
  } catch {
    const body: ProxyError = { error: 'invalid_json' };
    return NextResponse.json(body, { status: 400 });
  }
  if (!isPitchChatRequest(parsed)) {
    const body: ProxyError = { error: 'invalid_request_shape' };
    return NextResponse.json(body, { status: 400 });
  }

  // Each Titan answers on its own internal address — no nginx prefix
  // routing needed. The upstream path is always /v6/pitch/chat because
  // each instance serves its own pitch_chat router (Phase E, D-SPEC-115).
  // We call /v6 directly rather than /v4 → 308 → /v6 to skip the
  // redirect hop on every Compare-mode fan-out (3 requests per turn).
  const upstream = `${TITAN_INTERNAL_URLS[parsed.titan]}/v6/pitch/chat`;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);

  try {
    const upstreamRes = await fetch(upstream, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Pitch-Token': token,
      },
      body: JSON.stringify(parsed),
      signal: controller.signal,
      cache: 'no-store',
    });

    const upstreamText = await upstreamRes.text();

    // Upstream 404 = this Titan instance doesn't have /v6/pitch/chat
    // registered yet (e.g. T3 on the separate Phase C runtime hasn't
    // shipped the Phase E v6 manifest). Surface a friendly decline
    // envelope so the Compare-mode UI can render a "not yet available"
    // card instead of a transport error. Wallet-side /chat keeps its
    // own auth surface and is unaffected.
    if (upstreamRes.status === 404) {
      return NextResponse.json(
        {
          response: '',
          titan: parsed.titan,
          thread_id: parsed.thread_id,
          internal_time: { epoch: null, phase: null, fatigue: null, emotion: null },
          declined: true,
          decline_reason: 'pitch_chat_not_deployed',
          decline_explanation: `${parsed.titan} is running an earlier build that doesn't yet expose the pitch-chat route. Available next session — try T1 in the meantime.`,
        },
        { status: 200, headers: { 'Cache-Control': 'no-store' } },
      );
    }

    // Pass through the upstream content-type (assumed JSON) and status.
    return new NextResponse(upstreamText, {
      status: upstreamRes.status,
      headers: {
        'Content-Type': upstreamRes.headers.get('content-type') || 'application/json',
        'Cache-Control': 'no-store',
      },
    });
  } catch (e: unknown) {
    const aborted = e instanceof DOMException && e.name === 'AbortError';
    const body: ProxyError = {
      error: aborted ? 'upstream_timeout' : 'upstream_error',
      detail: aborted ? `> ${FETCH_TIMEOUT_MS}ms` : (e instanceof Error ? e.message : String(e)),
    };
    return NextResponse.json(body, { status: 502 });
  } finally {
    clearTimeout(timeout);
  }
}
