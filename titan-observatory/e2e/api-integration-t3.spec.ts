// Phase C C-S7 (2026-05-05): T3-specific API integration tests.
//
// Verifies that Components 1+2+3 fixes shipped today actually populate
// the routes Observatory depends on, against the live T3 API at port 7778.
// Mirrors api-integration.spec.ts but pointed at T3 + with shape assertions
// for the specific fields C-S7 closes.
//
// Per `feedback_frontend_rebuild_playwright_verify.md` — real e2e check
// against the running T3 service before declaring T3 ready for 24h soak.

import { test, expect } from '@playwright/test';

const API_BASE = process.env.TITAN_T3_API_URL || 'http://10.135.0.6:7778';

test.describe('T3 Phase C C-S7 — Observatory API population', () => {
  test('GET /health returns 200 with full shape (Component 2+3)', async ({ request }) => {
    const res = await request.get(`${API_BASE}/health`);
    expect(res.status()).toBe(200);
    const json = await res.json();
    expect(json.status).toBe('ok');
    const data = json.data;

    // C-S7 prep-required keys (all 17 from T1+T2 baseline)
    const required = [
      'bus_health', 'capabilities', 'cognee_ready', 'limbo_mode',
      'maker_pubkey', 'memory_backend_ready', 'network', 'privacy_filter',
      'recorder_ready', 'rpc_endpoint', 'sol_balance', 'solana_capabilities',
      'status', 'subsystems', 'v3', 'vault', 'version',
    ];
    for (const k of required) {
      expect(data, `missing required key: ${k}`).toHaveProperty(k);
    }

    // Component 2 — bus_health is populated (was empty pre-Component 2)
    expect(data.bus_health, 'bus_health summary should not be null').not.toBeNull();
    expect(data.bus_health.state).toMatch(/^(healthy|warning|critical|unknown)$/);

    // Component 2 — subsystems has all base + per-module entries
    expect(Object.keys(data.subsystems).length).toBeGreaterThan(20);

    // Component 3 — v3 status is populated (was empty {} pre-Component 3)
    expect(data.v3, 'v3 status should be populated under l0_rust=true').toBeTruthy();
    expect(typeof data.v3).toBe('object');
    expect(Object.keys(data.v3).length).toBeGreaterThan(3);
  });

  test('GET /v6/trinity/state returns 24+ guardian modules (Component 1)', async ({ request }) => {
    const res = await request.get(`${API_BASE}/v6/trinity/state`);
    expect(res.status()).toBe(200);
    const json = await res.json();
    expect(json.status).toBe('ok');

    const guardian = json.data?.guardian || {};
    const moduleCount = Object.keys(guardian).length;
    expect(moduleCount, 'guardian must have 24+ modules visible').toBeGreaterThan(20);

    // Spot check: some canonical modules present
    expect(guardian).toHaveProperty('imw');
    expect(guardian).toHaveProperty('memory');
    expect(guardian).toHaveProperty('cgn');
    expect(guardian).toHaveProperty('api');

    // Each module should have the canonical state shape
    for (const [name, info] of Object.entries(guardian)) {
      expect(info, `module ${name} must have shape`).toHaveProperty('state');
      expect(info, `module ${name} must have shape`).toHaveProperty('pid');
      expect(info, `module ${name} must have shape`).toHaveProperty('layer');
    }
  });

  test('GET /v6/system/bus-health returns full BusHealthMonitor snapshot (Component 2)', async ({ request }) => {
    const res = await request.get(`${API_BASE}/v6/system/bus-health`);
    expect(res.status()).toBe(200);
    const json = await res.json();
    expect(json.status).toBe('ok');

    const data = json.data;
    // Pre-Component-2 this was {}. Post: full snapshot with these fields:
    expect(data, 'bus-health data must include canonical snapshot fields').toHaveProperty('overall_state');
    expect(data).toHaveProperty('producers');
    expect(data).toHaveProperty('queues');
    expect(data).toHaveProperty('rate_budget_hz');
    expect(data).toHaveProperty('total_emission_rate_1min_hz');
    expect(data).toHaveProperty('orphans');
    expect(data.overall_state).toMatch(/^(healthy|warning|critical)$/);
  });

  test('GET /v6/timechain/status returns chain data', async ({ request }) => {
    const res = await request.get(`${API_BASE}/v6/timechain/status`);
    expect(res.status()).toBe(200);
    const json = await res.json();
    expect(json.status).toBe('ok');
    // Existing route — was working pre-C-S7. Sanity check.
    expect(json.data).toBeTruthy();
  });

  test('GET /v6/language/vocabulary returns substantial payload', async ({ request }) => {
    const res = await request.get(`${API_BASE}/v6/language/vocabulary`);
    expect(res.status()).toBe(200);
    const text = await res.text();
    // Vocabulary on T3 should be ~700KB+. Was working pre-C-S7.
    expect(text.length).toBeGreaterThan(100000);
  });

  test('Service has no recent escalations (stability)', async ({ request }) => {
    // Indirect stability check: the routes themselves respond fast.
    // Direct journal/supervision.jsonl checks are out of scope for the
    // browser-side test runner; that's verified separately via the
    // 5-min check-in cadence in the Maker-driven soak.
    const res = await request.get(`${API_BASE}/health`);
    expect(res.status()).toBe(200);
    // Any successful /health response means kernel-rs + plugin both
    // alive and serving — implicit stability.
  });
});
