import { test, expect } from '@playwright/test';

// Phase 2 IA + TimeChain promotion 2026-05-10: 7 top-level tabs with Chat last.
const NAV_TABS = [
  { path: '/', label: 'Home' },
  { path: '/trinity', label: 'Self' },
  { path: '/neurology', label: 'Mind' },
  { path: '/expression', label: 'Voice' },
  { path: '/world', label: 'World' },
  { path: '/timechain', label: 'TimeChain' },
  { path: '/chat', label: 'Chat' },
];

// Old nav routes that were folded into the new IA, with their canonical
// new home. We assert the meta-refresh redirect URL because Next.js
// app-router server `redirect()` from a synchronously-rendered server
// component emits HTTP 200 + <meta http-equiv="refresh"> rather than 307.
const REDIRECTS = [
  { from: '/persona',       to: '/expression?tab=persona' },
  { from: '/kin',           to: '/world?tab=society' },
  { from: '/compare',       to: '/world?tab=compare' },
  // /timechain is no longer a redirect — promoted to top-level nav 2026-05-10.
  { from: '/system',        to: '/world?tab=system' },
  { from: '/consciousness', to: '/trinity?tab=unified-spirit' },
  { from: '/dreams',        to: '/neurology?tab=dreaming' },
  { from: '/reflexes',      to: '/neurology?tab=reflexes' },
  { from: '/feed',          to: '/expression?tab=feed' },
  { from: '/creative',      to: '/expression?tab=creative' },
  { from: '/language',      to: '/expression?tab=language' },
  { from: '/neural',        to: '/trinity?tab=memory' },
  { from: '/research',      to: '/world?tab=research' },
  { from: '/soul-mosaic',   to: '/world?tab=soul-mosaic' },
  { from: '/rhythms',       to: '/trinity?tab=rhythms' },
  { from: '/stats',         to: '/world?tab=system' },
  { from: '/timeline',      to: '/trinity?tab=unified-spirit' },
];

test.describe('Phase 2 IA — Top-level navigation', () => {
  for (const route of NAV_TABS) {
    test(`${route.label} tab loads (${route.path})`, async ({ page }) => {
      const response = await page.goto(route.path);
      expect(response?.status()).toBe(200);
      await expect(page.locator('body')).toBeVisible();
    });
  }

  test('all 7 nav tabs visible in header, Chat is last', async ({ page }) => {
    await page.goto('/');
    for (const route of NAV_TABS) {
      const link = page.locator(`nav a[href="${route.path}"]`).first();
      await expect(link).toBeVisible();
    }
    const navLinks = page.locator('nav a').filter({ hasText: /Home|Self|Mind|Voice|World|TimeChain|Chat/ });
    const last = navLinks.last();
    await expect(last).toHaveText(/Chat/);
  });

  test('click-through across all 7 tabs', async ({ page }) => {
    // Slow path — Trinity contains TitanSELF (R3F + sphere clocks) which
    // is heavy to mount. Wait for each route's body to be visible before
    // moving on, so we never click the next nav link mid-mount.
    await page.goto('/', { waitUntil: 'domcontentloaded' });
    for (const route of NAV_TABS.slice(1)) {
      await page.locator(`nav a[href="${route.path}"]`).first().click();
      await page.waitForURL(`**${route.path}`, { timeout: 30000 });
      await expect(page.locator('body')).toBeVisible();
      expect(page.url()).toContain(route.path);
    }
  });

  test('legacy routes redirect via meta-refresh to canonical sub-tab', async ({ request }) => {
    // Use the request fixture instead of page.goto so Playwright doesn't
    // auto-follow the meta refresh — we want to inspect the original
    // response body and find the meta tag.
    for (const r of REDIRECTS) {
      const resp = await request.get(r.from);
      expect(resp.status()).toBe(200);
      const html = await resp.text();
      const m = html.match(/http-equiv="refresh"[^>]+content="\d+;url=([^"]+)"/);
      expect(m, `no meta refresh for ${r.from}`).toBeTruthy();
      expect(m![1]).toBe(r.to);
    }
  });
});

test.describe('Self · TitanSELF + Trinity sub-tabs (2026-05-10 changes)', () => {
  test('Self route shows 7 sub-tabs in correct order (TitanSELF + Trinity prepended)', async ({ page }) => {
    await page.goto('/trinity');
    const expected = ['TitanSELF', 'Trinity', 'Architecture', 'I-Depth', 'Unified Spirit', 'Rhythms', 'Memory'];
    for (const label of expected) {
      await expect(page.getByRole('button', { name: new RegExp(`^${label}$`) }).first()).toBeVisible();
    }
  });

  test('TitanSELF tab renders three viz prototype switcher (Cell · Mandala · Constellation)', async ({ page }) => {
    await page.goto('/trinity');
    // TitanSELF is the default sub-tab; the three prototype switcher buttons must be visible.
    for (const label of ['Cell', 'Mandala', 'Constellation']) {
      await expect(page.getByRole('button', { name: new RegExp(label) }).first()).toBeVisible();
    }
    // The 162D caption should appear in the architectural footer.
    await expect(page.getByText(/162 dimensions/i).first()).toBeVisible();
  });

  test('Trinity sub-tab renders the moved TrinityRadar (Divine Trinity — Live Tensors)', async ({ page }) => {
    await page.goto('/trinity?tab=trinity');
    await expect(page.getByText('Divine Trinity — Live Tensors')).toBeVisible({ timeout: 10000 });
  });

  test('World page no longer has a TimeChain sub-tab', async ({ page }) => {
    await page.goto('/world');
    // The 5 World sub-tabs that should remain
    for (const label of ['Society', 'Research', 'Soul Mosaic', 'Compare', 'System']) {
      await expect(page.getByRole('button', { name: new RegExp(`^${label}$`) }).first()).toBeVisible();
    }
    // TimeChain must NOT appear among the World sub-tab BUTTONS. (The
    // top-level nav LINK still says "TimeChain" — that's the promoted
    // top-level tab and is correct.)
    const subTabButton = page.getByRole('button', { name: /^TimeChain$/ });
    await expect(subTabButton).toHaveCount(0);
  });
});

test.describe('Phase 3 — token-gated /v/<token>/* pitch route', () => {
  test('bad token renders not-found content (no leak of pitch UI)', async ({ page }) => {
    await page.goto('/v/this-is-clearly-not-the-real-token');
    const body = await page.locator('body').textContent();
    expect(body).not.toContain('Welcome.');
    expect(body).not.toContain('Two ways in.');
  });

  test('robots.txt disallows /v/', async ({ request }) => {
    const resp = await request.get('/robots.txt');
    expect(resp.status()).toBe(200);
    const text = await resp.text();
    expect(text).toContain('Disallow: /v/');
  });
});
