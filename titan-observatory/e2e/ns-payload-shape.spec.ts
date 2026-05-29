// ── E2E: Neural Nervous System payload + frontend rendering ─────
//
// Codified after 2026-05-19 NS-payload-drift incident: Phase B.5 backend
// migration (commit cf6a7793) moved get_nervous_system() from rich
// spirit_supplemental_state.bin to lean titanvm_registers.bin. Frontend
// was not updated; header metadata + per-program tooltips went blank for
// 24h+ across T1+T2+T3 before user noticed.
//
// This test ASSERTS the contract:
//   1. /v6/nervous-system payload shape is {programs, age_seconds, seq}
//   2. Each program has {urgency, fire_count, total_updates, last_loss}
//   3. NervousSystemTab renders an urgency value for every program
//
// If the backend schema changes again, this test breaks loudly — that's
// the point. See SPEC §2.7 / D-SPEC-92.

import { test, expect } from '@playwright/test';

const FRONTEND = 'http://localhost:3000';
const TITAN_API = 'http://localhost:7777';

test.describe('NS payload contract (Phase B.5 lean schema)', () => {
  test('GET /v6/nervous-system returns lean schema with urgency', async ({ request }) => {
    const res = await request.get(`${TITAN_API}/v6/nervous-system`);
    expect(res.status()).toBe(200);
    const env = await res.json();
    expect(env.status, 'unwrap envelope').toBe('ok');
    const data = env.data;

    // Top-level shape
    expect(data, 'top-level keys').toEqual(
      expect.objectContaining({
        programs: expect.any(Object),
        age_seconds: expect.any(Number),
        seq: expect.any(Number),
      }),
    );

    // Per-program contract — every program must carry urgency
    const programs = data.programs as Record<string, Record<string, number>>;
    const names = Object.keys(programs);
    expect(names.length, 'at least one NS program').toBeGreaterThanOrEqual(10);

    for (const name of names) {
      const p = programs[name];
      expect(p, `${name} schema`).toEqual(
        expect.objectContaining({
          urgency: expect.any(Number),
          fire_count: expect.any(Number),
          total_updates: expect.any(Number),
          last_loss: expect.any(Number),
        }),
      );
      // Sanity: urgency is a probability-like value in [0, 1]
      expect(p.urgency).toBeGreaterThanOrEqual(0);
      expect(p.urgency).toBeLessThanOrEqual(1);
    }
  });

  test('BFF /api/v4-cached/nervous-system passes the lean schema through', async ({ request }) => {
    const res = await request.get(`${FRONTEND}/api/v4-cached/nervous-system?titan=T1`);
    expect(res.status()).toBe(200);
    const data = await res.json();
    expect(data).toHaveProperty('programs');
    expect(data).toHaveProperty('age_seconds');
    expect(data).toHaveProperty('seq');
  });
});

test.describe('Frontend renders Phase B.5 lean schema', () => {
  test('NervousSystemTab shows urgency for every program', async ({ page }) => {
    await page.goto(`${FRONTEND}/neurology?tab=nervous-system`, { waitUntil: 'networkidle' });
    // The neurology page mounts NervousSystemTab — wait for urgency cells
    // to populate (1 per program tile, marked with data-testid="ns-urgency").
    // If the backend stops returning urgency, count drops to 0 → test fails.
    await page.waitForSelector('[data-testid="ns-urgency"]', { timeout: 15_000 });
    const urgencyCells = await page.locator('[data-testid="ns-urgency"]').all();
    expect(urgencyCells.length, 'urgency cell per program').toBeGreaterThanOrEqual(10);

    // Every urgency cell renders an integer-percent value (not '--' / blank)
    for (const cell of urgencyCells) {
      const text = (await cell.textContent()) ?? '';
      expect(text, 'urgency renders a percentage').toMatch(/^\d{1,3}%$/);
    }
  });
});
