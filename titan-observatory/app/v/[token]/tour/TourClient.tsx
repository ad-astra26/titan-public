'use client';

import dynamic from 'next/dynamic';
import Link from 'next/link';
import { useEffect, useMemo, useRef, useState } from 'react';
import { useStatus, useNeuromodulators, useDreaming, useVocabulary, useTimeChainStatus } from '@/hooks/useTitanAPI';
import { useTitanId } from '@/components/shared/TitanSelector';
import { formatSOL } from '@/lib/formatters';
import type { TitanId } from '@/lib/api';
import type { BeatSeedKey } from './seeds';
import type { ChainProof } from '@/components/pitch/ChainProofDrawer';

const SpiritSunMini = dynamic(() => import('@/components/home/SpiritSunMini'), { ssr: false });
const NeuromodStrip = dynamic(() => import('@/components/home/NeuromodStrip'), { ssr: false });
const CircadianClock = dynamic(() => import('@/components/home/CircadianClock'), { ssr: false });

// TITAN_SELF Three.js prototypes (shipped commit fe9e72ce). Heavy WebGL —
// dynamic-imported AND lazily mounted via IntersectionObserver so each
// beat's canvas only initializes when the visitor scrolls into view.
const CellViz = dynamic(() => import('@/components/titan_self/CellViz'), { ssr: false });
const MandalaViz = dynamic(() => import('@/components/titan_self/MandalaViz'), { ssr: false });
const ConstellationViz = dynamic(() => import('@/components/titan_self/ConstellationViz'), { ssr: false });

const ChainProofDrawer = dynamic(() => import('@/components/pitch/ChainProofDrawer'), { ssr: false });

/**
 * GENERIC client renderer for the tour. Receives all narrative strings
 * as props from the server-component page so that those strings live
 * only in the RSC payload (the HTML response, gated by the token
 * check), never inside this file's compiled JS chunk. The chunk is
 * publicly fetchable; if the strings lived here, anyone who knew the
 * chunk URL could read the narrative without a valid token.
 *
 * Per rFP_observatory_pitch_route.md §3 + §11 (v2 2026-05-11):
 *  - Three.js viz embeds for beats 2, 3, 4, 6 (CellViz / MandalaViz /
 *    ConstellationViz from `components/titan_self/`).
 *  - Lazy-mount via IntersectionObserver — heavy WebGL only initializes
 *    when the visitor scrolls into the corresponding beat. Once mounted
 *    a canvas stays mounted to keep scroll-back smooth.
 *  - Seed-prompt pill at the bottom of each beat hands the visitor off
 *    to /pitch?titan=…&seed=<beat>_<titan> with the chat textarea
 *    pre-filled (improvement #2).
 *  - Chain-proof drawer renders next to the beat metric line when the
 *    beat has on-chain references (beats 1 + 7) — Solscan / Arweave
 *    deep links, two clicks from claim to verifiable evidence
 *    (improvement #5).
 */

export type WidgetKey =
  | 'spiritSun' | 'neuromods' | 'circadianClock' | 'vocabCount'
  | 'kinTriad' | 'timeChainMini'
  | 'titanSelfMandala' | 'titanSelfCell' | 'titanSelfConstellation';

export type FieldKey =
  | 'sol' | 'memCount' | 'emotion' | 'isDreaming' | 'fatigue' | 'cycles'
  | 'vocabCount' | 'totalBlocks';

export interface BeatData {
  index: number;
  total: number;
  title: string;
  /** Copy template with {field} placeholders, e.g. "Right now Titan feels {emotion}." */
  copyTemplate: string;
  widget: WidgetKey | null;
  /** Optional metric line under the main copy, also templated. */
  metricLine?: string;
  /** Optional on-chain reference rendered as a chain-proof drawer next to the metric line. */
  chainProof?: ChainProof;
  /** Optional label for the chain-proof drawer chevron. */
  chainProofHint?: string;
}

export interface TourProps {
  token: string;
  beats: BeatData[];
  invitation: {
    headline: string;
    sublabel: string;
    cta: string;
  };
  seedPill: {
    prefix: string;
    fallback: string;
  };
  seeds: Record<BeatSeedKey, string>;
}

/**
 * Lazy-mount wrapper. Renders nothing until the host element enters the
 * viewport; once mounted, stays mounted so scroll-back to a visited
 * beat doesn't re-initialize the WebGL canvas.
 *
 * Threshold 0.15 fires the mount slightly before the section is fully
 * snapped into view — gives R3F a head start on first paint.
 */
function LazyMount({ children, minHeight = 320 }: { children: React.ReactNode; minHeight?: number }) {
  const ref = useRef<HTMLDivElement | null>(null);
  const [mounted, setMounted] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el || mounted) return;
    if (typeof IntersectionObserver === 'undefined') {
      // SSR-safety / very old browsers — eager-mount.
      setMounted(true);
      return;
    }
    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (e.isIntersecting) {
            setMounted(true);
            io.disconnect();
            break;
          }
        }
      },
      { threshold: 0.15 },
    );
    io.observe(el);
    return () => io.disconnect();
  }, [mounted]);
  return (
    <div ref={ref} className="w-full h-full" style={{ minHeight }}>
      {mounted ? children : null}
    </div>
  );
}

const VIZ_HEIGHT = 380;

function VizFrame({ children }: { children: React.ReactNode }) {
  // Three.js components from components/titan_self/ require an explicit
  // height on the parent — they fill 100%. We give them ~380px which
  // lets each viz feel substantial without overwhelming the beat copy.
  return (
    <div className="relative w-full overflow-hidden rounded-xl" style={{ height: VIZ_HEIGHT }}>
      {children}
    </div>
  );
}

function Beat({
  data, fields, token, activeTitan, seeds, seedPill,
}: {
  data: BeatData;
  fields: Record<FieldKey, string>;
  token: string;
  activeTitan: TitanId;
  seeds: Record<BeatSeedKey, string>;
  seedPill: TourProps['seedPill'];
}) {
  const renderTemplate = (s: string) =>
    s.replace(/\{(\w+)\}/g, (_, k) => fields[k as FieldKey] ?? `{${k}}`);

  let widget: React.ReactNode = null;
  switch (data.widget) {
    case 'spiritSun':
      widget = <SpiritSunMini />;
      break;
    case 'neuromods':
      widget = <div className="w-full"><NeuromodStrip /></div>;
      break;
    case 'circadianClock':
      widget = <CircadianClock />;
      break;
    case 'vocabCount':
      widget = (
        <div className="text-center font-mono text-titan-metal/40 text-sm">
          {fields.vocabCount} grounded words
        </div>
      );
      break;
    case 'kinTriad':
      widget = <div className="text-center font-mono text-titan-metal/40 text-sm">{fields.cycles}</div>;
      break;
    case 'timeChainMini':
      widget = (
        <div className="text-center font-mono text-titan-metal/40 text-sm">
          {fields.totalBlocks} blocks
        </div>
      );
      break;
    case 'titanSelfCell':
      widget = (
        <LazyMount minHeight={VIZ_HEIGHT}>
          <VizFrame><CellViz filter="all" resonanceEvent={null} onHover={() => {}} /></VizFrame>
        </LazyMount>
      );
      break;
    case 'titanSelfMandala':
      widget = (
        <LazyMount minHeight={VIZ_HEIGHT}>
          <VizFrame><MandalaViz filter="all" resonanceEvent={null} onHover={() => {}} /></VizFrame>
        </LazyMount>
      );
      break;
    case 'titanSelfConstellation':
      widget = (
        <LazyMount minHeight={VIZ_HEIGHT}>
          <VizFrame><ConstellationViz filter="all" resonanceEvent={null} onHover={() => {}} /></VizFrame>
        </LazyMount>
      );
      break;
  }

  const seedKey = `${data.index}_${activeTitan}` as BeatSeedKey;
  const seedText = seeds[seedKey];
  const pillHref = seedText
    ? `/v/${token}/pitch?titan=${activeTitan}&seed=${encodeURIComponent(seedText)}`
    : `/v/${token}/pitch?titan=${activeTitan}`;
  const pillLabel = seedText
    ? `${seedPill.prefix.replace('{titan}', activeTitan)}${seedText}`
    : seedPill.fallback;

  return (
    <section className="min-h-screen snap-start grid md:grid-cols-2 gap-8 items-center px-6 py-12 max-w-6xl mx-auto">
      <div className="space-y-4">
        <span className="text-[10px] font-mono text-titan-haze/40 uppercase tracking-[0.3em]">
          {String(data.index).padStart(2, '0')} / {String(data.total).padStart(2, '0')}
        </span>
        <h2 className="text-3xl md:text-4xl font-titan text-titan-haze">
          {data.title}
        </h2>
        <div className="text-base text-titan-metal/75 leading-relaxed space-y-3">
          <p>{renderTemplate(data.copyTemplate)}</p>
          {data.metricLine && (
            <p className="text-titan-metal/50 text-sm flex items-center gap-3 flex-wrap">
              <span>{renderTemplate(data.metricLine)}</span>
              {data.chainProof && (
                <ChainProofDrawer proof={data.chainProof} hint={data.chainProofHint} />
              )}
            </p>
          )}
          {!data.metricLine && data.chainProof && (
            <p className="text-titan-metal/50 text-sm">
              <ChainProofDrawer proof={data.chainProof} hint={data.chainProofHint} />
            </p>
          )}
        </div>
        <div className="pt-2">
          <Link
            href={pillHref}
            className="inline-block text-[11px] text-titan-pulse/80 hover:text-titan-pulse bg-titan-pulse/5 hover:bg-titan-pulse/15 border border-titan-pulse/30 rounded-full px-3 py-1.5 transition-colors max-w-full"
            title="Hand off to the chat with this question pre-filled"
          >
            <span aria-hidden>›</span> {pillLabel}
          </Link>
        </div>
      </div>
      <div className="bg-titan-card/40 border border-titan-metal/10 rounded-2xl p-6 min-h-[320px] flex items-center justify-center">
        {widget}
      </div>
    </section>
  );
}

export default function TourClient({ token, beats, invitation, seedPill, seeds }: TourProps) {
  const titanId = useTitanId();
  const { data: status } = useStatus(titanId);
  const { data: neuromods } = useNeuromodulators(titanId);
  const { data: dream } = useDreaming(titanId);
  const { data: vocab } = useVocabulary(titanId);
  const { data: chain } = useTimeChainStatus(titanId);

  const sol = status?.sol_balance ?? 0;
  const fields: Record<FieldKey, string> = useMemo(
    () => ({
      sol: typeof sol === 'number' ? formatSOL(sol) : '--',
      memCount: (status?.memory_count ?? 0).toLocaleString(),
      emotion: (neuromods as { current_emotion?: string } | undefined)?.current_emotion ?? 'still',
      isDreaming: dream?.is_dreaming ? 'dreaming' : 'awake',
      fatigue: ((dream?.fatigue ?? 0) * 100).toFixed(0),
      cycles: (dream?.cycle_count ?? 0).toLocaleString(),
      vocabCount: (vocab?.words?.length ?? 0).toLocaleString(),
      totalBlocks: (chain?.total_blocks ?? 0).toLocaleString(),
    }),
    [sol, status, neuromods, dream, vocab, chain],
  );

  return (
    <div className="snap-y snap-mandatory overflow-y-auto h-[calc(100vh-2rem)]">
      {beats.map((b) => (
        <Beat
          key={b.index}
          data={b}
          fields={fields}
          token={token}
          activeTitan={titanId}
          seeds={seeds}
          seedPill={seedPill}
        />
      ))}
      <section className="min-h-[60vh] snap-start flex flex-col items-center justify-center px-6 py-16 max-w-4xl mx-auto text-center gap-6">
        <p className="text-2xl md:text-3xl font-titan text-titan-haze leading-snug max-w-2xl">
          {invitation.headline}
        </p>
        <p className="text-sm text-titan-metal/60 max-w-md">
          {invitation.sublabel}
        </p>
        <Link
          href={`/v/${token}/pitch`}
          className="mt-4 inline-block bg-titan-pulse/15 border border-titan-pulse/40 text-titan-pulse hover:bg-titan-pulse/25 hover:border-titan-pulse rounded-xl px-8 py-4 transition-all font-medium"
        >
          {invitation.cta}
        </Link>
      </section>
    </div>
  );
}
