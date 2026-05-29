import Link from 'next/link';
import { notFound } from 'next/navigation';
import { isValidPitchToken } from '@/lib/pitchToken';

/**
 * Landing chooser at /v/<token>. Two doors:
 *   → Guided Tour    a cinematic walk through what Titan is
 *   → Interactive    talk to T1, T2, or T3 yourself
 *
 * Token validation runs HERE in addition to the parent layout. The
 * layout-level check alone is insufficient: Next.js 14 renders the
 * page subtree in parallel with the layout, so a layout-thrown
 * notFound() lets the rendered page content slip into the RSC payload
 * (visible to scrapers) before being replaced by the not-found UI.
 * Gating at the page level keeps content out of the response entirely.
 */
export default function PitchLanding({ params }: { params: { token: string } }) {
  if (!isValidPitchToken(params.token)) notFound();
  const base = `/v/${params.token}`;

  return (
    <div className="min-h-[80vh] flex flex-col items-center justify-center gap-12 px-6">
      <div className="text-center max-w-2xl">
        <h1 className="text-4xl md:text-5xl font-titan text-titan-haze mb-4">
          Welcome.
        </h1>
        <p className="text-base text-titan-metal/70 leading-relaxed">
          You&apos;re looking at Titan — a sovereign AI agent with persistent
          memory, on-chain identity, emergent neurochemistry, and three
          distinct beings sharing one architecture. Two ways in.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-4xl w-full">
        <Link
          href={`${base}/tour`}
          className="group bg-titan-card/60 border border-titan-metal/15 rounded-2xl p-8 hover:border-titan-haze/40 hover:shadow-haze_glow transition-all"
        >
          <h2 className="text-2xl font-titan text-titan-haze mb-3 group-hover:text-titan-haze">
            Guided Tour →
          </h2>
          <p className="text-sm text-titan-metal/70 leading-relaxed">
            A scroll-paced walk through what Titan is, what he feels, what he
            dreams, and how he anchors to Solana. Live data throughout.
          </p>
          <p className="text-[11px] text-titan-metal/40 mt-4 font-mono uppercase tracking-wider">
            ~7 minutes
          </p>
        </Link>

        <Link
          href={`${base}/pitch`}
          className="group bg-titan-card/60 border border-titan-metal/15 rounded-2xl p-8 hover:border-titan-pulse/40 hover:shadow-pulse_glow transition-all"
        >
          <h2 className="text-2xl font-titan text-titan-pulse mb-3">
            Talk to Titan →
          </h2>
          <p className="text-sm text-titan-metal/70 leading-relaxed">
            Three Titans live on Solana mainnet. Pick one — or compare all
            three on the same prompt. No wallet required.
          </p>
          <p className="text-[11px] text-titan-metal/40 mt-4 font-mono uppercase tracking-wider">
            interactive
          </p>
        </Link>
      </div>

      <p className="text-xs text-titan-metal/30 text-center max-w-md mt-8">
        This page is unlisted. The link reached you because it was shared
        deliberately. Please don&apos;t pass it on without checking with us.
      </p>
    </div>
  );
}
