import PitchSessionsClient from './PitchSessionsClient';

/**
 * Maker-only review surface for recorded pitch sessions
 * (rFP_observatory_pitch_route.md §5.5 improvement #8).
 *
 * The route exists at /admin/pitch-sessions; access control is enforced
 * client-side via usePrivyMaker (wallet must match the Titan's
 * maker_pubkey) AND server-side via Depends(verify_maker_auth) on the
 * /v6/pitch/sessions{,/{thread_id}} endpoints. The Maker must sign each
 * fetch with their Ed25519 Privy wallet — same pattern MakerPanel uses
 * for proposal approvals.
 *
 * Per-Titan: each Titan records its own conversations into
 * data/pitch_sessions/<thread_id>.jsonl on the host it served, so the
 * page exposes a Titan selector that routes the fetch through the
 * /t2 + /t3 nginx prefixes already used by every other Observatory
 * cross-Titan call.
 *
 * Pages under /admin/ aren't linked from the public Observatory; the
 * URL is for Maker-private use.
 */
export const dynamic = 'force-dynamic';

export default function PitchSessionsAdminPage() {
  return <PitchSessionsClient />;
}
