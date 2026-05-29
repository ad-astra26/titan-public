'use client';

import { Component, ReactNode, useCallback, useEffect, useMemo, useState } from 'react';
import { useSignMessage, useWallets } from '@privy-io/react-auth/solana';
import bs58 from 'bs58';
import { titanFetch, type TitanId } from '@/lib/api';
import { useHealth } from '@/hooks/useTitanAPI';

/**
 * Privy hooks (`useWallets`, `useSignMessage`) crash with
 * `TypeError: Cannot read properties of null (reading 'connectors')`
 * when called before the PrivyProvider chunk has finished loading
 * (Providers.tsx lazy-loads it via React.lazy + Suspense, with a naked
 * children fallback). This boundary catches that specific error and
 * remounts after a short delay; by the second attempt Privy is ready.
 *
 * Other errors bubble (so genuine bugs aren't swallowed).
 */
class PrivyReadyBoundary extends Component<
  { children: ReactNode; fallback: ReactNode },
  { error: boolean; attempts: number }
> {
  state = { error: false, attempts: 0 };
  static getDerivedStateFromError(err: Error) {
    const msg = err?.message ?? '';
    if (/connectors|Cannot read properties of null/i.test(msg)) {
      return { error: true };
    }
    throw err;
  }
  componentDidCatch() {
    const next = this.state.attempts + 1;
    const delay = Math.min(150 * Math.pow(2, next), 4000);
    setTimeout(() => this.setState({ error: false, attempts: next }), delay);
  }
  render() {
    return this.state.error ? this.props.fallback : this.props.children;
  }
}

/**
 * Maker pitch-session review UI. Mirrors the auth + signing pattern
 * used by components/chat/MakerPanel.tsx (per-action sign with
 * Privy Solana wallet), targeting the new /v6/pitch/sessions surface
 * (rFP §5.5 improvement #8).
 *
 * The verify_maker_auth dependency on the backend checks:
 *   - signature is Ed25519 over `${ts}:${body}`
 *   - ts within ±60s (anti-replay)
 *   - pubkey matches Titan's stored maker_pubkey
 * For GET endpoints the body is empty → message = `${ts}:`.
 */

interface PitchSessionMeta {
  thread_id: string;
  first_ts: number | null;
  last_ts: number | null;
  message_count: number;
  declined_count: number;
  bytes: number;
}

interface PitchSessionsResponse {
  sessions: PitchSessionMeta[];
  total: number;
}

interface PitchSessionDetailResponse {
  thread_id: string;
  lines: Record<string, unknown>[];
}

const TITANS: TitanId[] = ['T1', 'T2', 'T3'];

function fmtTs(ts: number | null | undefined): string {
  if (!ts) return '—';
  return new Date(ts * 1000).toISOString().replace('T', ' ').slice(0, 19);
}

function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(2)} MB`;
}

/** Build the `${ts}:` header pair that verify_maker_auth expects for a
 *  GET request (empty body). Caller adds the signature; we deliberately
 *  don't stub the Privy wallet type here — wallets[0] flows in with its
 *  full ConnectedStandardSolanaWallet type and is passed to signMessage
 *  inline in each callback. */
function buildAuthTimestamp(): { ts: string; messageBytes: Uint8Array } {
  const ts = (Date.now() / 1000).toFixed(3);
  return { ts, messageBytes: new TextEncoder().encode(`${ts}:`) };
}

/**
 * Mount-deferred wrapper. components/Providers.tsx lazy-loads
 * PrivyProvider via React.lazy + Suspense — during the few hundred ms
 * before Privy mounts, `<PrivyWrapper>` renders children NAKED. Any
 * component that calls `useWallets()` / `useSignMessage()` during that
 * window crashes with `TypeError: Cannot read properties of null
 * (reading 'connectors')` (Privy's Solana context is null).
 *
 * MakerPanel sidesteps this by being conditionally mounted only after
 * `isMaker === true`, which implies the user already authenticated, so
 * Privy is loaded by then. The admin pitch-sessions page mounts on
 * fresh navigation though, so we explicitly wait one render cycle
 * before mounting the inner component that calls the Privy hooks.
 */
function LoadingFrame({ message = 'Initializing wallet provider…' }: { message?: string }) {
  return (
    <main className="max-w-5xl mx-auto px-6 py-12 space-y-4">
      <h1 className="font-titan text-2xl text-titan-haze">Pitch session review</h1>
      <p className="text-sm text-titan-metal/50">{message}</p>
    </main>
  );
}

export default function PitchSessionsClient() {
  const [mounted, setMounted] = useState(false);
  useEffect(() => {
    // Tick after first paint — PrivyProvider has usually had its mount
    // cycle by now, but the lazy-load chunk can still be in flight when
    // Privy's API fetch is slow. PrivyReadyBoundary handles the residual
    // race by catching the `connectors`-null TypeError and remounting.
    setMounted(true);
  }, []);
  if (!mounted) return <LoadingFrame />;
  return (
    <PrivyReadyBoundary fallback={<LoadingFrame message="Waiting for wallet provider…" />}>
      <PitchSessionsInner />
    </PrivyReadyBoundary>
  );
}

function PitchSessionsInner() {
  const { wallets } = useWallets();
  const { signMessage } = useSignMessage();
  const { data: health } = useHealth();

  const [titan, setTitan] = useState<TitanId>('T1');
  const [sessions, setSessions] = useState<PitchSessionMeta[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<PitchSessionDetailResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastFetchTs, setLastFetchTs] = useState<number | null>(null);

  const wallet = wallets[0];
  const walletAddress = wallet?.address ?? null;
  const makerPubkey = health?.maker_pubkey ?? null;
  const isMaker = useMemo(
    () => Boolean(walletAddress && makerPubkey && walletAddress === makerPubkey),
    [walletAddress, makerPubkey],
  );

  const refreshSessions = useCallback(async () => {
    if (!wallet || !isMaker) return;
    setLoading(true);
    setError(null);
    try {
      const { ts, messageBytes } = buildAuthTimestamp();
      const signed = await signMessage({ message: messageBytes, wallet });
      const res = await titanFetch<PitchSessionsResponse>('/v6/pitch/sessions?limit=100', {
        method: 'GET',
        titan,
        headers: {
          'X-Titan-Signature': bs58.encode(signed.signature),
          'X-Titan-Timestamp': ts,
        },
      });
      setSessions(res.sessions);
      setLastFetchTs(Date.now());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [titan, wallet, isMaker, signMessage]);

  const openSession = useCallback(
    async (threadId: string) => {
      if (!wallet || !isMaker) return;
      setSelectedId(threadId);
      setDetail(null);
      setError(null);
      setLoading(true);
      try {
        const { ts, messageBytes } = buildAuthTimestamp();
        const signed = await signMessage({ message: messageBytes, wallet });
        const res = await titanFetch<PitchSessionDetailResponse>(
          `/v6/pitch/sessions/${encodeURIComponent(threadId)}`,
          {
            method: 'GET',
            titan,
            headers: {
              'X-Titan-Signature': bs58.encode(signed.signature),
              'X-Titan-Timestamp': ts,
            },
          },
        );
        setDetail(res);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setLoading(false);
      }
    },
    [titan, wallet, isMaker, signMessage],
  );

  // Reset state when Titan changes; require an explicit Refresh click
  // (each refresh costs one Privy signature pop, no surprise re-signs).
  useEffect(() => {
    setSessions([]);
    setSelectedId(null);
    setDetail(null);
    setError(null);
    setLastFetchTs(null);
  }, [titan]);

  if (!walletAddress) {
    return (
      <main className="max-w-5xl mx-auto px-6 py-12 space-y-4">
        <h1 className="font-titan text-2xl text-titan-haze">Pitch session review</h1>
        <p className="text-sm text-titan-metal/70">
          Connect a Privy wallet matching this Titan&apos;s Maker public key to review recorded sessions.
        </p>
      </main>
    );
  }

  if (!isMaker) {
    return (
      <main className="max-w-5xl mx-auto px-6 py-12 space-y-4">
        <h1 className="font-titan text-2xl text-titan-haze">Pitch session review</h1>
        <p className="text-sm text-titan-metal/70">
          Connected wallet does not match this Titan&apos;s Maker public key. This page is Maker-only.
        </p>
        <p className="text-[10px] font-mono text-titan-metal/40 break-all">
          wallet: {walletAddress}
        </p>
        {makerPubkey && (
          <p className="text-[10px] font-mono text-titan-metal/40 break-all">
            maker:  {makerPubkey}
          </p>
        )}
      </main>
    );
  }

  return (
    <main className="max-w-7xl mx-auto px-6 py-8 space-y-6">
      <header className="flex flex-wrap items-baseline justify-between gap-3">
        <div>
          <h1 className="font-titan text-2xl text-titan-haze">Pitch session review</h1>
          <p className="text-xs text-titan-metal/50 mt-0.5">
            Recordings from <code className="font-mono">data/pitch_sessions/</code> on the selected Titan.
            Each click requires a Maker signature.
          </p>
        </div>
        <div className="flex flex-wrap gap-2 items-center">
          {TITANS.map((t) => (
            <button
              key={t}
              onClick={() => setTitan(t)}
              className={`px-3 py-1.5 rounded-lg border text-xs transition-all ${
                titan === t
                  ? 'bg-titan-haze/15 border-titan-haze text-titan-haze'
                  : 'bg-titan-card/40 border-titan-metal/15 text-titan-metal hover:border-titan-haze/30'
              }`}
            >
              {t}
            </button>
          ))}
          <button
            onClick={refreshSessions}
            disabled={loading}
            className="bg-titan-pulse/20 border border-titan-pulse/40 text-titan-pulse px-4 py-1.5 rounded-lg text-xs disabled:opacity-40 hover:bg-titan-pulse/30"
          >
            {loading ? '…' : sessions.length ? 'Refresh' : 'Load sessions'}
          </button>
        </div>
      </header>

      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg px-3 py-2 text-xs text-red-200">
          {error}
        </div>
      )}

      <div className="flex flex-col lg:flex-row gap-4 min-h-[60vh]">
        <aside className="lg:w-[420px] shrink-0 bg-titan-card/40 border border-titan-metal/10 rounded-xl p-4 overflow-y-auto max-h-[75vh]">
          <div className="flex items-baseline justify-between mb-3">
            <h3 className="text-[10px] font-mono uppercase tracking-wider text-titan-metal/50">
              sessions on {titan}
            </h3>
            <span className="text-[10px] font-mono text-titan-metal/40">
              {sessions.length} total
              {lastFetchTs && ` · ${new Date(lastFetchTs).toLocaleTimeString()}`}
            </span>
          </div>
          {sessions.length === 0 ? (
            <p className="text-xs italic text-titan-metal/40">
              No sessions loaded. Click <span className="text-titan-haze">Load sessions</span> to sign and fetch.
            </p>
          ) : (
            <ul className="space-y-1">
              {sessions.map((s) => (
                <li key={s.thread_id}>
                  <button
                    onClick={() => openSession(s.thread_id)}
                    className={`w-full text-left p-2.5 rounded-md border transition-colors ${
                      selectedId === s.thread_id
                        ? 'bg-titan-haze/10 border-titan-haze/40'
                        : 'bg-titan-bg/40 border-titan-metal/10 hover:border-titan-haze/30'
                    }`}
                  >
                    <div className="font-mono text-[11px] text-titan-haze break-all">{s.thread_id}</div>
                    <div className="mt-1 flex items-baseline gap-3 text-[10px] font-mono text-titan-metal/50">
                      <span>{s.message_count} msgs</span>
                      {s.declined_count > 0 && (
                        <span className="text-amber-300/80">{s.declined_count} declined</span>
                      )}
                      <span>{fmtBytes(s.bytes)}</span>
                      <span className="ml-auto">{fmtTs(s.last_ts)}</span>
                    </div>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </aside>

        <section className="flex-1 bg-titan-card/40 border border-titan-metal/10 rounded-xl p-4 overflow-y-auto max-h-[75vh]">
          {!selectedId ? (
            <p className="text-xs italic text-titan-metal/40">Select a session on the left.</p>
          ) : !detail ? (
            <p className="text-xs italic text-titan-metal/40">{loading ? 'loading…' : 'no data'}</p>
          ) : (
            <>
              <div className="mb-3 border-b border-titan-metal/10 pb-2">
                <div className="text-[10px] font-mono uppercase tracking-wider text-titan-metal/50">
                  thread_id
                </div>
                <div className="font-mono text-sm text-titan-haze break-all">{detail.thread_id}</div>
                <div className="text-[10px] font-mono text-titan-metal/40 mt-1">
                  {detail.lines.length} lines
                </div>
              </div>
              <div className="space-y-1.5">
                {detail.lines.map((line, i) => (
                  <pre
                    key={i}
                    className="text-[10px] font-mono text-titan-metal/80 bg-titan-bg/40 border border-titan-metal/10 rounded-md p-2 overflow-x-auto whitespace-pre-wrap break-words"
                  >
                    {JSON.stringify(line, null, 2)}
                  </pre>
                ))}
              </div>
            </>
          )}
        </section>
      </div>
    </main>
  );
}
