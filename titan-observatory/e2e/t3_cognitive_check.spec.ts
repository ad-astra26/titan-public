// Quick spec: verify cognitive_worker is populating the routes Observatory cards hit.
// Per chunk 8L deploy gate verification (post-Phase C C-S8 4B cutover, 2026-05-05).
import { test, expect } from '@playwright/test';

const API_BASE = process.env.TITAN_T3_API_URL || 'http://10.135.0.6:7778';

test.describe('T3 chunk 8L — cognitive_worker /v4/* population', () => {
  const cognitiveRoutes = [
    { path: '/v6/cognition/reasoning', minBytes: 200, key: 'total_chains' },
    { path: '/v6/cognition/meta-reasoning', minBytes: 300, key: 'total_chains' },
    { path: '/v6/dreaming', minBytes: 200, key: 'cycle_count' },
    { path: '/v6/nervous-system/pi-heartbeat', minBytes: 200, key: 'developmental_age' },
    { path: '/v6/trinity/inner', minBytes: 300, key: 'consciousness' },
  ];

  for (const route of cognitiveRoutes) {
    test(`${route.path} populated`, async ({ request }) => {
      const res = await request.get(`${API_BASE}${route.path}`);
      expect(res.status()).toBe(200);
      const text = await res.text();
      expect(text.length).toBeGreaterThanOrEqual(route.minBytes);
      const json = JSON.parse(text);
      expect(json.status).toBe('ok');
      expect(json.data).toHaveProperty(route.key);
    });
  }

  test('cognitive cache keys all fresh (<10s)', async ({ request }) => {
    const res = await request.get(`${API_BASE}/v6/trinity/cache-staleness`);
    const json = await res.json();
    const ages = json.data.ages;
    const cognitiveKeys = ['reasoning.state', 'meta_reasoning.state',
      'dreaming.state', 'pi_heartbeat.state', 'neuromods.state', 'topology.state'];
    for (const k of cognitiveKeys) {
      expect(ages[k], `${k} not in cache`).toBeDefined();
      expect(ages[k], `${k} stale (${ages[k]}s)`).toBeLessThan(10);
    }
  });
});
