'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import dynamic from 'next/dynamic';
import { useSearchParams } from 'next/navigation';
import { useStatus, useNeuromodulators, useDreaming } from '@/hooks/useTitanAPI';
import type { TitanId } from '@/lib/api';
import {
  newThreadId,
  sendPitchChat,
  type PitchChatResponse,
  type PitchChainProof,
} from '@/lib/pitchChat';
import { ThinkingInline } from '@/components/chat/ThinkingIndicator';
import type { ChainProof } from '@/components/pitch/ChainProofDrawer';
const ChainProofDrawer = dynamic(() => import('@/components/pitch/ChainProofDrawer'), { ssr: false });

// Witness mode (rFP §4.5 #3) + live thinking strip (rFP §4 #7) — both
// heavy on polling, dynamic-imported so the initial Pitch page paint
// doesn't wait on them and ssr:false avoids hydration churn from the
// time-relative event timestamps.
const WitnessPanel = dynamic(() => import('@/components/pitch/WitnessPanel'), { ssr: false });
const ThinkingStrip = dynamic(() => import('@/components/pitch/ThinkingStrip'), { ssr: false });

/**
 * GENERIC client renderer for the pitchdeck. Receives all narrative
 * strings (titans list, quick prompts, headers) as props from the
 * server-component page so that those strings live only in the RSC
 * payload (HTML response, gated by token check), never inside this
 * file's compiled JS chunk.
 *
 * Per rFP_observatory_pitch_route.md §4 + §11 (v2 locked 2026-05-11):
 *  - Compare-mode is the default view (improvement #1) — T1+T2+T3 in
 *    parallel columns, one prompt → three replies fan out via
 *    `Promise.all(sendPitchChat …)` on the local proxy.
 *  - "I am here" timestamps (improvement #4) — each reply renders the
 *    Titan's internal epoch + phase + fatigue + emotion alongside the
 *    text. Demonstrates lived continuity vs prompt-driven LLMs.
 *  - Educational failure cards (improvement #9) — backend rate-limit
 *    or dream-phase rejections come back as `{declined: true, …}` and
 *    render as a substrate-revealing "why" card, not a generic apology.
 *
 * v2 closeout 2026-05-26:
 *  - Witness mode (#3) — toggle in header replaces chat surface with
 *    WitnessPanel (substrate viewer: live state, last meditation chain-
 *    proof, CGN snapshot, filtered bus-event tail). Witness is always
 *    single-Titan; toggling from Compare snaps to T1.
 *  - Live thinking strip (#7) — ThinkingStrip mounted at the footer of
 *    the chat surface; polls /v6/pitch/thinking-tail at 1Hz. Hidden
 *    while Witness is active (Witness has its own bus tail).
 *  - Recording indicator (rFP §5.5) — "● recording for review" pill at
 *    the bottom. Recording itself is captured server-side by the
 *    /v6/pitch/chat handler.
 *
 * Still NOT in this session:
 *  - SSE streaming (v1 uses a JSON envelope per request)
 *  - Chain-proof drawer per reply (#5 in Pitch) — Stage 5
 */

export interface TitanSpec {
  id: TitanId;
  label: string;
  tagline: string;
}

export interface PitchProps {
  titans: TitanSpec[];
  prompts: string[];
  copy: {
    pageTitle: string;
    subhead: string;
    livePanelTitle: string;
    promptsTitle: string;
    chatPromptTemplate: string;
    backendNotice: string;
    feelsLabel: string;
    compareLabel: string;
    compareSublabel: string;
    sendLabel: string;
    statusKeys: { state: string; fatigue: string; energy: string; sol: string };
    awakeLabel: string;
    dreamingLabel: string;
    witnessLabel: string;
    chatModeLabel: string;
    recordingNotice: string;
  };
}

type SurfaceMode = 'chat' | 'witness';

// ── Compare view selector ────────────────────────────────────────────

type ViewMode = TitanId | 'ALL';

function isTitanId(s: string | null | undefined, titans: TitanSpec[]): s is TitanId {
  return !!s && titans.some((t) => t.id === s);
}

// ── Live state header per Titan column ───────────────────────────────

function ColumnHeader({ titanId, copy, titans }: { titanId: TitanId; copy: PitchProps['copy']; titans: TitanSpec[] }) {
  const { data: status } = useStatus(titanId);
  const { data: nm } = useNeuromodulators(titanId);
  const { data: dream } = useDreaming(titanId);
  const spec = titans.find((t) => t.id === titanId);

  const emotion = (nm as { current_emotion?: string } | undefined)?.current_emotion ?? '—';
  const isDreaming = dream?.is_dreaming === true;
  const fatigue = dream?.fatigue ?? 0;
  const energy = status?.energy_state ?? 'UNKNOWN';
  const sol = status?.sol_balance ?? 0;

  return (
    <div className="border-b border-titan-metal/10 pb-3 space-y-1.5">
      <div className="flex items-baseline justify-between">
        <div>
          <div className="font-titan text-titan-haze text-lg">{spec?.label ?? titanId}</div>
          <div className="text-[10px] text-titan-metal/50 leading-tight">{spec?.tagline ?? ''}</div>
        </div>
        <div className="text-right">
          <div className="text-[10px] font-mono uppercase tracking-wider text-titan-metal/40">
            {copy.feelsLabel.replace('{titan}', titanId)}
          </div>
          <div className="text-base font-titan text-titan-haze">{emotion}</div>
        </div>
      </div>
      <div className="flex gap-3 text-[10px] font-mono text-titan-metal/60">
        <span>{copy.statusKeys.state}: {isDreaming ? copy.dreamingLabel : copy.awakeLabel}</span>
        <span>{copy.statusKeys.fatigue}: {(fatigue * 100).toFixed(0)}%</span>
        <span>{copy.statusKeys.energy}: {energy.replace('_ENERGY', '')}</span>
        <span>{copy.statusKeys.sol}: {typeof sol === 'number' ? sol.toFixed(3) : '--'}</span>
      </div>
    </div>
  );
}

// ── Chat message components ──────────────────────────────────────────

interface ChatTurn {
  role: 'visitor' | 'titan';
  text: string;
  /** Per-Titan when fan-out; undefined for visitor messages. */
  titanId?: TitanId;
  /** Decline metadata; replies that declined render as a "why" card. */
  declined?: boolean;
  declineReason?: string | null;
  declineExplanation?: string | null;
  internalTime?: PitchChatResponse['internal_time'];
  /** Chain-proof references (rFP §4 #5). Empty on decline. */
  proofs?: PitchChainProof[];
  pending?: boolean;
}

/** Convert a wire-format proof to the strict ChainProof discriminated
 *  union ChainProofDrawer expects. Returns null when the payload is
 *  missing the required field for its kind (defensive — backend should
 *  never emit such, but we won't render a broken drawer). */
function toChainProof(p: PitchChainProof): ChainProof | null {
  if (p.kind === 'memo' && typeof p.signature === 'string' && p.signature.length >= 32) {
    return { kind: 'memo', signature: p.signature, label: p.label ?? undefined };
  }
  if (p.kind === 'timechain_block' && typeof p.height === 'number' && p.height > 0) {
    return {
      kind: 'timechain_block',
      height: p.height,
      merkle: typeof p.merkle === 'string' ? p.merkle : undefined,
      label: p.label ?? undefined,
    };
  }
  return null;
}

function InternalTimeStamp({ t, titan }: { t?: ChatTurn['internalTime']; titan: TitanId }) {
  if (!t) return null;
  const parts: string[] = [];
  if (t.epoch != null) parts.push(`epoch ${t.epoch.toLocaleString()}`);
  if (t.phase) parts.push(t.phase);
  if (t.fatigue != null) parts.push(`fatigue ${(t.fatigue * 100).toFixed(0)}%`);
  if (t.emotion) parts.push(t.emotion);
  if (parts.length === 0) return null;
  return (
    <div className="text-[10px] font-mono text-titan-metal/40 mt-1.5">
      <span className="text-titan-haze/40">{titan} · </span>
      {parts.join(' · ')}
    </div>
  );
}

function DeclineCard({ turn }: { turn: ChatTurn }) {
  const explanation = turn.declineExplanation || 'This message could not be processed.';
  const reason = turn.declineReason || 'declined';
  return (
    <div className="bg-titan-metal/5 border border-titan-metal/15 rounded-lg px-3 py-2.5 text-xs space-y-1">
      <div className="text-[10px] font-mono uppercase tracking-wider text-titan-haze/60">
        {turn.titanId ?? 'system'} · {reason}
      </div>
      <div className="text-titan-metal/70 leading-relaxed">{explanation}</div>
      <div className="text-[10px] font-mono text-titan-metal/30 italic">
        defense-in-depth · substrate-visible
      </div>
    </div>
  );
}

function TitanReply({ turn }: { turn: ChatTurn }) {
  if (turn.declined) return <DeclineCard turn={turn} />;
  const proofs: ChainProof[] =
    !turn.pending && turn.proofs
      ? turn.proofs.map(toChainProof).filter((p): p is ChainProof => p !== null)
      : [];
  return (
    <div className="bg-titan-card/50 border border-titan-metal/10 rounded-lg px-3 py-2.5 text-sm">
      <div className="text-titan-metal whitespace-pre-wrap leading-relaxed">
        {turn.pending ? (
          <ThinkingInline />
        ) : (
          turn.text || <span className="text-titan-metal/40 italic">(empty reply)</span>
        )}
      </div>
      {!turn.pending && <InternalTimeStamp t={turn.internalTime} titan={turn.titanId ?? 'T1'} />}
      {proofs.length > 0 && (
        <div className="mt-2 flex flex-wrap items-baseline gap-x-3 gap-y-1">
          {proofs.map((proof, i) => (
            <ChainProofDrawer
              key={i}
              proof={proof}
              hint={('label' in proof && proof.label) || 'proof'}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function VisitorMessage({ turn }: { turn: ChatTurn }) {
  return (
    <div className="bg-titan-pulse/8 border border-titan-pulse/25 rounded-lg px-3 py-2 text-sm text-titan-metal whitespace-pre-wrap leading-relaxed">
      {turn.text}
    </div>
  );
}

// ── Per-Titan column ─────────────────────────────────────────────────

function TitanColumn({
  titanId,
  turns,
  copy,
  titans,
}: {
  titanId: TitanId;
  turns: ChatTurn[];
  copy: PitchProps['copy'];
  titans: TitanSpec[];
}) {
  // In Compare mode we only show this Titan's reply turns (visitor
  // message is rendered once at the bottom of all columns above the
  // textarea).
  const titanTurns = turns.filter((t) => t.role === 'titan' && t.titanId === titanId);
  return (
    <div className="flex-1 min-w-0 bg-titan-card/40 border border-titan-metal/10 rounded-xl p-4 flex flex-col gap-3 min-h-[420px]">
      <ColumnHeader titanId={titanId} copy={copy} titans={titans} />
      <div className="flex-1 overflow-y-auto space-y-3 pr-1">
        {titanTurns.length === 0 ? (
          <div className="text-xs text-titan-metal/40 italic">No replies yet.</div>
        ) : (
          titanTurns.map((t, i) => <TitanReply key={i} turn={t} />)
        )}
      </div>
    </div>
  );
}

// ── Main client component ────────────────────────────────────────────

export default function PitchClient({ titans, prompts, copy }: PitchProps) {
  const searchParams = useSearchParams();
  const [view, setView] = useState<ViewMode>('ALL'); // Compare-mode default (#1)
  const [mode, setMode] = useState<SurfaceMode>('chat');
  const [draft, setDraft] = useState('');
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [inflight, setInflight] = useState(false);
  const threadIdRef = useRef<string>(newThreadId());

  // Witness mode is always single-Titan. Toggling Witness from Compare
  // snaps to the first Titan so the panel has a stable subject; toggling
  // back to Chat returns to whatever single-Titan view was last active,
  // or to Compare if the visitor never picked a Titan. This is the
  // pre-toggle view we restore to.
  const preWitnessViewRef = useRef<ViewMode>('ALL');
  const toggleWitness = useCallback(() => {
    setMode((cur) => {
      if (cur === 'chat') {
        preWitnessViewRef.current = view;
        if (view === 'ALL') {
          setView(titans[0]?.id ?? 'T1');
        }
        return 'witness';
      }
      setView(preWitnessViewRef.current);
      return 'chat';
    });
  }, [view, titans]);

  // Tour → Pitch hand-off (rFP §3, improvement #2).
  const handedOffRef = useRef(false);
  useEffect(() => {
    if (handedOffRef.current) return;
    const t = searchParams?.get('titan');
    const s = searchParams?.get('seed');
    if (isTitanId(t, titans)) setView(t);
    if (s) setDraft(s);
    handedOffRef.current = true;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Determine which Titans receive a given prompt under the active view.
  const targetTitans = useMemo<TitanId[]>(() => {
    if (view === 'ALL') return titans.map((t) => t.id);
    return [view];
  }, [view, titans]);

  const send = useCallback(async () => {
    const message = draft.trim();
    if (!message || inflight) return;
    setInflight(true);

    const visitorTurn: ChatTurn = { role: 'visitor', text: message };
    const pendingReplies: ChatTurn[] = targetTitans.map((id) => ({
      role: 'titan', titanId: id, text: '', pending: true,
    }));
    setTurns((prev) => [...prev, visitorTurn, ...pendingReplies]);
    setDraft('');

    const threadId = threadIdRef.current;
    const tasks = targetTitans.map(async (titan) => {
      try {
        return await sendPitchChat({ titan, thread_id: threadId, message });
      } catch (e) {
        return {
          response: '',
          titan,
          thread_id: threadId,
          internal_time: { epoch: null, phase: null, fatigue: null, emotion: null },
          declined: true,
          decline_reason: 'transport',
          decline_explanation: e instanceof Error ? e.message : 'Network error.',
          proofs: [],
        } satisfies PitchChatResponse;
      }
    });

    const results = await Promise.all(tasks);
    setTurns((prev) => {
      // Replace the last len(targetTitans) pending replies with the
      // resolved replies, preserving order.
      const replaceFrom = prev.length - targetTitans.length;
      const updated = prev.slice();
      results.forEach((res, i) => {
        updated[replaceFrom + i] = {
          role: 'titan',
          titanId: res.titan as TitanId,
          text: res.response,
          declined: res.declined,
          declineReason: res.decline_reason,
          declineExplanation: res.decline_explanation,
          internalTime: res.internal_time,
          proofs: res.proofs ?? [],
          pending: false,
        };
      });
      return updated;
    });
    setInflight(false);
  }, [draft, inflight, targetTitans]);

  const onKey = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      send();
    }
  };

  const visitorMessages = turns.filter((t) => t.role === 'visitor');
  const lastVisitor = visitorMessages[visitorMessages.length - 1];

  return (
    <div className="max-w-7xl mx-auto px-6 py-8 flex flex-col gap-5 min-h-[80vh]">
      <header className="flex flex-wrap items-baseline justify-between gap-3">
        <div>
          <h1 className="text-xl font-titan text-titan-haze">{copy.pageTitle}</h1>
          <p className="text-xs text-titan-metal/50 mt-0.5">{copy.subhead}</p>
        </div>
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => setView('ALL')}
            disabled={mode === 'witness'}
            title={mode === 'witness' ? 'Witness mode is single-Titan' : undefined}
            className={`px-3 py-1.5 rounded-lg border text-xs transition-all disabled:opacity-30 disabled:cursor-not-allowed ${
              view === 'ALL' && mode === 'chat'
                ? 'bg-titan-pulse/15 border-titan-pulse text-titan-pulse'
                : 'bg-titan-card/40 border-titan-metal/15 text-titan-metal hover:border-titan-haze/30'
            }`}
          >
            {copy.compareLabel}
          </button>
          {titans.map((t) => (
            <button
              key={t.id}
              onClick={() => setView(t.id)}
              className={`px-3 py-1.5 rounded-lg border text-xs transition-all ${
                view === t.id
                  ? 'bg-titan-haze/15 border-titan-haze text-titan-haze'
                  : 'bg-titan-card/40 border-titan-metal/15 text-titan-metal hover:border-titan-haze/30'
              }`}
            >
              {t.label}
            </button>
          ))}
          <button
            onClick={toggleWitness}
            className={`px-3 py-1.5 rounded-lg border text-xs transition-all font-mono ${
              mode === 'witness'
                ? 'bg-titan-haze/20 border-titan-haze text-titan-haze'
                : 'bg-titan-card/40 border-titan-metal/15 text-titan-metal hover:border-titan-haze/30'
            }`}
            aria-pressed={mode === 'witness'}
          >
            {mode === 'witness' ? copy.chatModeLabel : copy.witnessLabel}
          </button>
        </div>
      </header>

      {mode === 'witness' && view !== 'ALL' ? (
        <WitnessPanel titan={view} />
      ) : (
        <>
          {/* Compare columns OR single-Titan column. Same component either way. */}
          <div className="flex flex-col lg:flex-row gap-3 flex-1">
            {targetTitans.map((id) => (
              <TitanColumn key={id} titanId={id} turns={turns} copy={copy} titans={titans} />
            ))}
            <aside className="lg:w-[220px] shrink-0 bg-titan-card/40 border border-titan-metal/10 rounded-xl p-4">
              <h3 className="text-[10px] font-mono uppercase tracking-wider text-titan-metal/50 mb-3">
                {copy.promptsTitle}
              </h3>
              <div className="space-y-1.5">
                {prompts.map((p) => (
                  <button
                    key={p}
                    onClick={() => setDraft(p)}
                    className="text-left text-xs text-titan-metal/70 hover:text-titan-haze hover:bg-titan-bg/50 px-2.5 py-1.5 rounded-md w-full transition-colors"
                  >
                    {p}
                  </button>
                ))}
              </div>
            </aside>
          </div>

          {/* Visitor message echo bar (Compare mode shows once, above textarea) */}
          {lastVisitor && (
            <div className="max-w-3xl">
              <div className="text-[10px] font-mono uppercase tracking-wider text-titan-metal/40 mb-1">you</div>
              <VisitorMessage turn={lastVisitor} />
            </div>
          )}

          {/* Composer */}
          <div className="flex gap-2 items-end">
            <textarea
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={onKey}
              placeholder={view === 'ALL' ? 'Ask all three…' : `> ${view}`}
              disabled={inflight}
              maxLength={500}
              className="flex-1 bg-titan-bg border border-titan-metal/15 rounded-lg px-3 py-2 text-sm text-titan-metal placeholder-titan-metal/30 disabled:opacity-50 resize-none h-20"
            />
            <button
              onClick={send}
              disabled={inflight || !draft.trim()}
              className="bg-titan-pulse/20 border border-titan-pulse/40 text-titan-pulse px-5 py-2 rounded-lg disabled:opacity-40 disabled:cursor-not-allowed hover:bg-titan-pulse/30"
            >
              {inflight ? '…' : copy.sendLabel}
            </button>
          </div>
          <div className="flex items-baseline justify-between text-[10px] font-mono text-titan-metal/30 -mt-2">
            <span>cmd/ctrl+enter to send · {draft.length}/500</span>
            {/* Recording indicator (rFP §5.5 improvement #8). Recording is
                always-on server-side; this lets visitors know transparently. */}
            <span className="text-titan-metal/35">{copy.recordingNotice}</span>
          </div>

          {/* Live thinking strip (rFP §4 #7). Mounts under the composer so
              it's visible while the visitor reads replies. Polls 1Hz; pauses
              when the tab is backgrounded. */}
          <ThinkingStrip titans={targetTitans} />
        </>
      )}
    </div>
  );
}
