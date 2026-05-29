// ── E2E: comprehensive home-page render audit ──────────────────
// Forensic-grade probe to find regressions where individual home
// components fail to mount. Asserts EVERY top-level component is
// visible after queries hydrate, captures console errors + network
// failures, and prints them when something is missing.

import { test, expect, type ConsoleMessage, type Request } from '@playwright/test';

const FRONTEND = 'http://localhost:3000';

test('home page — every component renders + no console errors + no 4xx/5xx', async ({ page }) => {
  const consoleErrors: string[] = [];
  const failedRequests: Array<{ url: string; status: number }> = [];

  page.on('console', (msg: ConsoleMessage) => {
    if (msg.type() === 'error') consoleErrors.push(msg.text());
  });
  page.on('response', async (res) => {
    if (res.status() >= 400) {
      failedRequests.push({ url: res.url(), status: res.status() });
    }
  });

  await page.goto(`${FRONTEND}/`, { waitUntil: 'networkidle' });
  // Give React Query a real chance — the home page mounts ~20 queries.
  await page.waitForTimeout(4_000);

  // Probe the rendered DOM for the actual content each component owns.
  // If a component fails to mount, its hallmark text/role is missing.
  const probes = [
    { name: 'TitanSelector',         re: /T1|T2|T3/ },
    { name: 'NeuromodStrip',         re: /DA|5HT|NE/ },
    { name: 'DreamingIndicator',     re: /Dream|Asleep|Awake|GABA|sleep/i },
    { name: 'HormonalMini',          re: /Hormones|Hormonal|Programs/i },
    { name: 'ChiLifeForce',          re: /Chi|chi|Life Force/i },
    { name: 'SovereigntyGauge',      re: /Sovereignty|Sovereign/i },
    { name: 'CircadianClock',        re: /Circadian|Clock|UTC/i },
    { name: 'MoodIndicator',         re: /Mood/i },
    { name: 'VaultStatus',           re: /Vault|NFT|chain/i },
    { name: 'AgencyFeed',            re: /Agency|Activity|Recent/i },
    { name: 'NeuralNSMini',          re: /Neural Nervous System|progs|fires/i },
    { name: 'MetricCard:SOL Balance',re: /SOL Balance/ },
    { name: 'MetricCard:Vocabulary', re: /Vocabulary/ },
    { name: 'MetricCard:Neural NS',  re: /Neural NS/ },
    { name: 'MetricCard:Creations',  re: /Creations/ },
    { name: 'GlobalFreshnessPill',   re: /Updated .* ago|just now/ },
  ];

  const missing: string[] = [];
  for (const p of probes) {
    const count = await page.getByText(p.re).count();
    if (count === 0) missing.push(p.name);
  }

  if (missing.length || consoleErrors.length || failedRequests.length) {
    console.log('\n── HOME PAGE AUDIT FAILURES ──');
    if (missing.length) {
      console.log('Missing components:');
      missing.forEach(m => console.log(`  - ${m}`));
    }
    if (consoleErrors.length) {
      console.log('Console errors:');
      consoleErrors.slice(0, 20).forEach(e => console.log(`  - ${e.substring(0, 200)}`));
    }
    if (failedRequests.length) {
      console.log('Failed network requests:');
      failedRequests.slice(0, 20).forEach(r => console.log(`  - ${r.status} ${r.url}`));
    }
  }

  expect(missing, `missing components: ${missing.join(', ')}`).toHaveLength(0);
  expect(consoleErrors.filter(e => !e.includes('favicon')), `console errors: ${consoleErrors.join('\n')}`).toHaveLength(0);
  expect(failedRequests, `4xx/5xx requests: ${JSON.stringify(failedRequests)}`).toHaveLength(0);
});

test('home page — Chi Life Force renders a numeric value (not blank)', async ({ page }) => {
  await page.goto(`${FRONTEND}/`, { waitUntil: 'networkidle' });
  await page.waitForTimeout(4_000);
  // Chi component shows the total chi value somewhere in its DOM.
  // Whatever exact element it is, the DOM should contain a numeric
  // glyph near the "Chi" label.
  const chiSection = page.getByText(/Chi|Life Force/i).first();
  await expect(chiSection).toBeVisible({ timeout: 10_000 });
  // The chi numeric value is non-empty
  const ancestor = chiSection.locator('xpath=ancestor::*[contains(@class,"bg-titan-card") or contains(@class,"rounded")][1]');
  const text = (await ancestor.first().textContent()) ?? '';
  expect(text, `Chi section text content`).not.toBe('');
  // Must contain at least one digit somewhere
  expect(text, `Chi section should show numeric data: "${text.slice(0, 200)}"`).toMatch(/\d/);
});
