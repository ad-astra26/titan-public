import { notFound } from 'next/navigation';
import type { Metadata } from 'next';
import { isValidPitchToken } from '@/lib/pitchToken';

/**
 * Custom layout for /v/<token>/* — the cinematic, no-public-chrome surface
 * for VC + hackathon presentations. Renders WITHOUT the standard Observatory
 * Header / TabNav / Footer. The MetabolicWrapper + GridBackdrop set by the
 * root layout still surround this layout's children.
 *
 * Token validation happens here so EVERY child page (landing / tour / pitch)
 * is gated. Bad token → 404. We never tell the visitor the route exists.
 *
 * SEO / crawler hygiene: noindex,nofollow on every page in this subtree;
 * /robots.txt also Disallows /v/. See rFP_observatory_pitch_route.md §2.
 */

export const metadata: Metadata = {
  robots: { index: false, follow: false, nocache: true },
};

export default function PitchLayout({
  children,
  params,
}: {
  children: React.ReactNode;
  params: { token: string };
}) {
  if (!isValidPitchToken(params.token)) notFound();

  return (
    <div className="relative min-h-[calc(100vh-2rem)] -mx-4 -my-6 px-4 py-6">
      {/* No Header, no TabNav, no Footer — clean cinematic surface. */}
      <meta name="robots" content="noindex,nofollow" />
      {children}
    </div>
  );
}
