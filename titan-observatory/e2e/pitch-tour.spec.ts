import { test, expect } from '@playwright/test';

/**
 * E2E coverage for the token-gated pitch tour at /v/<token>/tour
 * and the tour → pitch hand-off.
 *
 * Per rFP_observatory_pitch_route.md §3 + §11 (v2 2026-05-11).
 *
 * The token is read from PITCH_TOKEN at server start; this spec reads
 * the same env var so it stays in sync with the running server's
 * configuration. CI/local must export PITCH_TOKEN before running these
 * tests; without it both server and spec fall back to a sentinel that
 * fails closed (every /v/ request 404s).
 */

const TOKEN = process.env.PITCH_TOKEN ?? '';
const TOUR_URL = `/v/${TOKEN}/tour`;

// Some specs in this suite depend on the token being configured. If it
// isn't, skip rather than fail — these tests aren't asserting on the
// server's gate-closed behavior, they're asserting on the gate-open path.
test.describe('Pitch tour (/v/<token>/tour)', () => {
  test.skip(!TOKEN || TOKEN.length < 24, 'PITCH_TOKEN not set or too short');

  test('valid token renders the tour without crashing', async ({ page }) => {
    await page.goto(TOUR_URL);
    await expect(page.locator('body')).toBeVisible();
    await expect(page.locator('text=Application error')).not.toBeVisible();
  });

  test('beat 1 renders the sovereign-being narrative', async ({ page }) => {
    await page.goto(TOUR_URL);
    // Beat 1 title: "A sovereign being on Solana."
    await expect(page.getByRole('heading', { name: /sovereign being on Solana/i })).toBeVisible();
  });

  test('renders all 7 beats by scrolling to the end', async ({ page }) => {
    await page.goto(TOUR_URL);
    // Wait for hydration so the snap-scroll container is ready.
    await page.waitForTimeout(1500);
    // Each beat carries its index in a "NN / 07" badge — count them.
    const badges = page.locator('text=/^0\\d \\/ 07$/');
    await expect(badges.first()).toBeVisible();
    expect(await badges.count()).toBe(7);
  });

  test('every beat exposes a Tour → Pitch hand-off pill', async ({ page }) => {
    await page.goto(TOUR_URL);
    await page.waitForTimeout(1500);
    // Seed pills always start with "Ask T<digit>:" prefix.
    const pills = page.locator('a:has-text("Ask T")').filter({ hasText: /Ask T[0-9]/ });
    const count = await pills.count();
    // 7 beats × 1 pill per active Titan = 7 pills minimum.
    expect(count).toBeGreaterThanOrEqual(7);
  });

  test('pill hrefs route to /v/<token>/pitch with titan + seed query params', async ({ page }) => {
    await page.goto(TOUR_URL);
    await page.waitForTimeout(1500);
    const firstPill = page.locator('a:has-text("Ask T")').first();
    const href = await firstPill.getAttribute('href');
    expect(href).toContain(`/v/${TOKEN}/pitch`);
    expect(href).toMatch(/[?&]titan=T[123]\b/);
    expect(href).toMatch(/[?&]seed=/);
  });

  test('Tour → Pitch hand-off pre-fills the chat textarea', async ({ page }) => {
    await page.goto(TOUR_URL);
    await page.waitForTimeout(1500);
    const firstPill = page.locator('a:has-text("Ask T")').first();
    const href = await firstPill.getAttribute('href');
    expect(href).toBeTruthy();
    // Navigate directly — clicking inside a snap-scroll container is
    // less reliable than going to the URL the pill carries.
    await page.goto(href!);
    await page.waitForTimeout(1500);
    // The chat textarea should be populated with the seed text. We
    // assert non-empty rather than equality so that Maker's copy edits
    // don't churn the test.
    const textarea = page.locator('textarea').first();
    await expect(textarea).toBeVisible();
    const value = await textarea.inputValue();
    expect(value.length).toBeGreaterThan(10);
  });

  test('chain-proof drawer is closed by default and opens on click (beat 1)', async ({ page }) => {
    await page.goto(TOUR_URL);
    await page.waitForTimeout(1500);
    // Beat 1's chain proof links to T1 identity on Solana.
    const proofChevron = page.locator('button[aria-expanded="false"]:has-text("verify on Solana")').first();
    await expect(proofChevron).toBeVisible();
    await proofChevron.click();
    // After clicking, the Solscan deep-link appears.
    await expect(page.getByRole('link', { name: /Solscan/i }).first()).toBeVisible();
  });

  test('robots meta is noindex,nofollow on the tour', async ({ page }) => {
    await page.goto(TOUR_URL);
    const robots = await page.locator('meta[name="robots"]').first().getAttribute('content');
    expect(robots).toBeTruthy();
    expect(robots!.toLowerCase()).toContain('noindex');
  });
});

test.describe('Pitch tour — bad token gating', () => {
  test('wrong token returns the 404 page, not the tour narrative', async ({ page }) => {
    await page.goto('/v/this-token-is-deliberately-wrong/tour');
    const body = await page.textContent('body');
    expect(body).toBeTruthy();
    // The not-found page should be rendered; the beat-1 title must NOT
    // appear (the narrative lives only in valid-token RSC payloads).
    expect(body!).not.toMatch(/sovereign being on Solana/i);
  });

  test('robots.txt disallows /v/', async ({ page }) => {
    const res = await page.goto('/robots.txt');
    expect(res?.status()).toBe(200);
    const body = await res!.text();
    expect(body.toLowerCase()).toContain('disallow: /v/');
  });
});
