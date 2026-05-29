/**
 * Single shared token for the /v/<token>/* pitch + tour routes.
 *
 * Resolution order (server-side only):
 *   1. process.env.PITCH_TOKEN
 *   2. fallback "pitch-disabled-set-PITCH_TOKEN-env" — never matches anything,
 *      so misconfigured deployments fail closed (404 on every /v/ request).
 *
 * Token is checked exactly via constant-time string comparison, with a
 * 24-char minimum length to prevent trivial brute-force or accidental
 * single-char tokens. Bad tokens 404 — we never leak whether the path
 * exists or whether the token was wrong.
 *
 * Per rFP_observatory_pitch_route.md §2.
 */
const FALLBACK = 'pitch-disabled-set-PITCH_TOKEN-env';
const MIN_LEN = 24;

export function getPitchToken(): string {
  return process.env.PITCH_TOKEN || FALLBACK;
}

/** Constant-time-ish equality. Length is leaked but content is not. */
export function isValidPitchToken(candidate: string | undefined): boolean {
  const expected = getPitchToken();
  if (!candidate) return false;
  if (expected.length < MIN_LEN) return false;
  if (candidate.length !== expected.length) return false;
  let diff = 0;
  for (let i = 0; i < expected.length; i++) {
    diff |= expected.charCodeAt(i) ^ candidate.charCodeAt(i);
  }
  return diff === 0;
}
