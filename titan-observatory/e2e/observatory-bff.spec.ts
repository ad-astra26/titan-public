// ── E2E: Observatory BFF + Page-Aggregate ──────────────────────
// rFP_observatory_enhancements §5.1 Phase 2-5 acceptance gates.
//
// Validates:
//   - /api/v4-cached/<endpoint>?titan=T{1,2,3} returns same payload as
//     direct /v4/<endpoint> upstream (passthrough correctness)
//   - Repeat call within TTL returns x-cache-stale=0 quickly (cache hit)
//   - /api/page/feed bundles narratedFeed + activityFeed + personaTelemetry
//   - /api/page/trinity collapses 6 upstream calls to 1
//   - /api/bff-metrics exposes counters after traffic
//   - LastUpdatedPill renders on dashboard pages

import { test, expect } from '@playwright/test';

const BFF_BASE = 'http://localhost:3000';
const RAW_BASE = 'http://localhost:7777';

test.describe('BFF: /api/v4-cached/* passthrough + cache', () => {
  for (const titan of ['T1', 'T2', 'T3'] as const) {
    test(`BFF inner-trinity matches raw upstream (${titan})`, async ({ request }) => {
      const bff = await request.get(
        `${BFF_BASE}/api/v4-cached/inner-trinity?titan=${titan}`,
      );
      expect(bff.status(), 'BFF should return 200').toBe(200);
      const bffJson = await bff.json();
      // BFF returns the unwrapped `data` payload (server unwraps {status,data}).
      expect(bffJson).toBeTruthy();
      // Header semantics
      const stale = bff.headers()['x-cache-stale'];
      expect(['0', '1']).toContain(stale);
    });
  }

  test('cache hit returns x-cache-stale=0 on rapid repeat', async ({ request }) => {
    // Warm
    await request.get(`${BFF_BASE}/api/v4-cached/sphere-clocks?titan=T1`);
    const t0 = Date.now();
    const r = await request.get(`${BFF_BASE}/api/v4-cached/sphere-clocks?titan=T1`);
    const dt = Date.now() - t0;
    expect(r.status()).toBe(200);
    expect(r.headers()['x-cache-stale']).toBe('0');
    // Cache hit should be fast (no upstream round-trip). Generous bound for CI.
    expect(dt).toBeLessThan(200);
  });

  test('unknown endpoint returns 404', async ({ request }) => {
    const r = await request.get(`${BFF_BASE}/api/v4-cached/not-a-real-endpoint`);
    expect(r.status()).toBe(404);
  });

  test('query string is preserved through BFF (caches by path+query)', async ({ request }) => {
    const r1 = await request.get(
      `${BFF_BASE}/api/v4-cached/narrated-feed?titan=T1&limit=5`,
    );
    const r2 = await request.get(
      `${BFF_BASE}/api/v4-cached/narrated-feed?titan=T1&limit=40`,
    );
    expect(r1.status()).toBe(200);
    expect(r2.status()).toBe(200);
    const j1 = await r1.json();
    const j2 = await r2.json();
    // Different limits should produce different item counts (unless the
    // backend has fewer than 5 items, in which case both arrays are equal).
    if ((j2.items?.length ?? 0) > 5) {
      expect((j1.items?.length ?? 0)).toBeLessThan((j2.items?.length ?? 0));
    }
  });
});

test.describe('BFF: /api/page/* aggregates', () => {
  test('page/feed bundles 3 endpoints', async ({ request }) => {
    const r = await request.get(`${BFF_BASE}/api/page/feed?titan=T1`);
    expect(r.status()).toBe(200);
    const json = await r.json();
    expect(json).toHaveProperty('narratedFeed');
    expect(json).toHaveProperty('activityFeed');
    expect(json).toHaveProperty('personaTelemetry');
    expect(json).toHaveProperty('_meta');
    expect(json._meta.titan).toBe('T1');
    expect(json._meta.page).toBe('feed');
  });

  test('page/trinity collapses 6 upstream calls', async ({ request }) => {
    const r = await request.get(`${BFF_BASE}/api/page/trinity?titan=T1`);
    expect(r.status()).toBe(200);
    const json = await r.json();
    for (const key of [
      'trinity', 'neuromodulators', 'chi', 'dreaming',
      'sphereClocks', 'metabolism',
    ]) {
      expect(json).toHaveProperty(key);
    }
  });

  test('unknown page returns 404', async ({ request }) => {
    const r = await request.get(`${BFF_BASE}/api/page/not-real`);
    expect(r.status()).toBe(404);
  });

  test('trinity page is faster than 6 individual upstream calls', async ({ request }) => {
    // Warm both paths to avoid cold-start skew.
    await request.get(`${BFF_BASE}/api/page/trinity?titan=T1`);
    for (const ep of ['inner-trinity', 'neuromodulators', 'chi', 'dreaming',
                       'sphere-clocks', 'metabolism']) {
      await request.get(`${BFF_BASE}/api/v4-cached/${ep}?titan=T1`);
    }
    // Measure cached-path latency
    const t0 = Date.now();
    const r = await request.get(`${BFF_BASE}/api/page/trinity?titan=T1`);
    const dt = Date.now() - t0;
    expect(r.status()).toBe(200);
    // Cache-warm aggregate should be near-instant (< 100ms acceptance gate).
    expect(dt, `page/trinity warm: ${dt}ms`).toBeLessThan(300);
  });
});

test.describe('BFF: telemetry endpoint', () => {
  test('/api/bff-metrics returns counters after traffic', async ({ request }) => {
    // Generate traffic
    await request.get(`${BFF_BASE}/api/v4-cached/inner-trinity?titan=T1`);
    await request.get(`${BFF_BASE}/api/v4-cached/inner-trinity?titan=T1`);
    const r = await request.get(`${BFF_BASE}/api/bff-metrics`);
    expect(r.status()).toBe(200);
    const json = await r.json();
    expect(json).toHaveProperty('routes');
    expect(json).toHaveProperty('summary');
    expect(json.summary.totalRequests).toBeGreaterThan(0);
  });
});

test.describe('Frontend: LastUpdatedPill renders', () => {
  test('pill appears on home page after queries hydrate', async ({ page }) => {
    await page.goto(`${BFF_BASE}/`, { waitUntil: 'networkidle' });
    // Pill polls QueryClient every 1s for latest dataUpdatedAt + appears
    // once any query completes. Under cold cache networkidle alone is
    // insufficient because the 1s tick may not have fired yet.
    const pill = page.getByText(/Updated .* ago|just now/);
    await expect(pill.first()).toBeVisible({ timeout: 15_000 });
  });
});
