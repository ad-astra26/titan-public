import { test, expect } from '@playwright/test';

const API_BASE = 'http://localhost:7777';

test.describe('Backend API endpoints', () => {
  test('GET /health returns 200', async ({ request }) => {
    const res = await request.get(`${API_BASE}/health`);
    expect(res.status()).toBe(200);
    const json = await res.json();
    expect(json.status).toBe('ok');
  });

  test('GET /v6/cognition/reasoning returns valid data', async ({ request }) => {
    const res = await request.get(`${API_BASE}/v6/cognition/reasoning`);
    expect(res.status()).toBe(200);
    const json = await res.json();
    expect(json.status).toBe('ok');
    expect(json.data).toHaveProperty('total_chains');
    expect(json.data).toHaveProperty('is_active');
    expect(json.data).toHaveProperty('confidence');
    expect(json.data).toHaveProperty('mind_neuromods');
    expect(typeof json.data.total_chains).toBe('number');
  });

  test('GET /v6/cognition/meta-reasoning returns valid data', async ({ request }) => {
    const res = await request.get(`${API_BASE}/v6/cognition/meta-reasoning`);
    expect(res.status()).toBe(200);
    const json = await res.json();
    expect(json.status).toBe('ok');
    expect(json.data).toHaveProperty('total_chains');
    expect(json.data).toHaveProperty('primitive_counts');
    expect(json.data).toHaveProperty('avg_reward');
    expect(typeof json.data.primitive_counts).toBe('object');
  });

  test('GET /v6/expression/compositions returns valid data', async ({ request }) => {
    const res = await request.get(`${API_BASE}/v6/expression/compositions`);
    expect(res.status()).toBe(200);
    const json = await res.json();
    expect(json.status).toBe('ok');
    expect(json.data).toHaveProperty('total_compositions');
    expect(json.data).toHaveProperty('latest');
    expect(json.data).toHaveProperty('recent');
  });

  test('GET /v6/trinity/inner returns 200', async ({ request }) => {
    const res = await request.get(`${API_BASE}/v6/trinity/inner`);
    expect(res.status()).toBe(200);
  });

  test('GET /v6/nervous-system returns 200', async ({ request }) => {
    const res = await request.get(`${API_BASE}/v6/nervous-system`);
    expect(res.status()).toBe(200);
  });

  test('GET /v6/trinity/sphere-clocks returns 200', async ({ request }) => {
    const res = await request.get(`${API_BASE}/v6/trinity/sphere-clocks`);
    expect(res.status()).toBe(200);
  });

  test('GET /v6/nervous-system/pi-heartbeat returns 200', async ({ request }) => {
    const res = await request.get(`${API_BASE}/v6/nervous-system/pi-heartbeat`);
    expect(res.status()).toBe(200);
  });

  test('GET /status returns 200', async ({ request }) => {
    const res = await request.get(`${API_BASE}/status`);
    expect(res.status()).toBe(200);
  });
});
