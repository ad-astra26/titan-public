import { NextResponse } from 'next/server';
import { getPitchToken } from '@/lib/pitchToken';
import type { TitanId } from '@/lib/api';

/**
 * Server-side proxy for /v6/pitch/witness-tail (rFP §4.5 #3).
 *
 * Mirrors pitch-chat-proxy/route.ts: the X-Pitch-Token must never land
 * in the public client bundle, so this route runs server-side, attaches
 * the token from process.env, and forwards to the chosen Titan's
 * internal API address (T1 directly; T2/T3 over the 10.135 VPN, same
 * wiring used by the chat proxy).
 *
 * No auth on this proxy itself — the upstream rejects every request
 * without a valid PITCH_TOKEN, so a public-bundle leak is non-issue.
 * The visitor still has to reach a /v/<TOKEN>/* route to use Witness
 * mode at all.
 */

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const VALID_TITANS = new Set<TitanId>(['T1', 'T2', 'T3']);

const TITAN_INTERNAL_URLS: Record<TitanId, string> = {
  T1: process.env.TITAN_T1_INTERNAL_URL || 'http://localhost:7777',
  T2: process.env.TITAN_T2_INTERNAL_URL || 'http://10.135.0.6:7777',
  T3: process.env.TITAN_T3_INTERNAL_URL || 'http://10.135.0.6:7778',
};
// Tail polls are 1Hz and the upstream is sub-second under normal load.
// Keep the timeout tight so a stuck Titan doesn't queue 60s of polls.
const FETCH_TIMEOUT_MS = 5_000;

export async function GET(req: Request) {
  const token = getPitchToken();
  if (!token || token.length < 24) {
    return NextResponse.json({ error: 'pitch_route_not_configured' }, { status: 404 });
  }

  const url = new URL(req.url);
  const titan = url.searchParams.get('titan');
  if (!titan || !VALID_TITANS.has(titan as TitanId)) {
    return NextResponse.json({ error: 'invalid_titan' }, { status: 400 });
  }
  // Cap events_limit at 50 (matches backend _TAIL_HARD_MAX); reject
  // larger values rather than silently truncating so misuse is loud.
  const rawLimit = url.searchParams.get('events_limit');
  let upstreamQuery = '';
  if (rawLimit) {
    const n = parseInt(rawLimit, 10);
    if (!Number.isFinite(n) || n < 1 || n > 50) {
      return NextResponse.json({ error: 'invalid_events_limit' }, { status: 400 });
    }
    upstreamQuery = `?events_limit=${n}`;
  }

  const upstream =
    `${TITAN_INTERNAL_URLS[titan as TitanId]}/v6/pitch/witness-tail${upstreamQuery}`;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);

  try {
    const upstreamRes = await fetch(upstream, {
      method: 'GET',
      headers: { 'X-Pitch-Token': token },
      signal: controller.signal,
      cache: 'no-store',
    });
    const body = await upstreamRes.text();
    return new NextResponse(body, {
      status: upstreamRes.status,
      headers: {
        'Content-Type': upstreamRes.headers.get('content-type') || 'application/json',
        'Cache-Control': 'no-store',
      },
    });
  } catch (e: unknown) {
    const aborted = e instanceof DOMException && e.name === 'AbortError';
    return NextResponse.json(
      {
        error: aborted ? 'upstream_timeout' : 'upstream_error',
        detail: aborted ? `> ${FETCH_TIMEOUT_MS}ms` : (e instanceof Error ? e.message : String(e)),
      },
      { status: 502 },
    );
  } finally {
    clearTimeout(timeout);
  }
}
