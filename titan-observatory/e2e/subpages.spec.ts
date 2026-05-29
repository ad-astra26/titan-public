import { test, expect } from '@playwright/test';

test.describe('Trinity page', () => {
  test('loads and shows trinity content', async ({ page }) => {
    await page.goto('/trinity');
    await page.waitForTimeout(2000);
    const body = await page.textContent('body');
    expect(body?.toLowerCase()).toMatch(/trinity|body|mind|spirit/);
  });
});

test.describe('Neurology page', () => {
  test('loads and shows neurology content', async ({ page }) => {
    await page.goto('/neurology');
    await page.waitForTimeout(2000);
    const body = await page.textContent('body');
    expect(body?.toLowerCase()).toMatch(/neuro|nervous|brain/);
  });
});

test.describe('Kin page', () => {
  test('loads without crash', async ({ page }) => {
    const response = await page.goto('/kin');
    expect(response?.status()).toBe(200);
    await expect(page.locator('body')).toBeVisible();
  });
});

test.describe('System page', () => {
  test('loads and shows system info', async ({ page }) => {
    await page.goto('/system');
    await page.waitForTimeout(2000);
    const body = await page.textContent('body');
    expect(body?.toLowerCase()).toMatch(/system|guardian|health|status/);
  });
});

test.describe('Consciousness page', () => {
  test('loads without crash', async ({ page }) => {
    const response = await page.goto('/consciousness');
    expect(response?.status()).toBe(200);
    await expect(page.locator('body')).toBeVisible();
  });
});

test.describe('Rhythms page', () => {
  test('loads without crash', async ({ page }) => {
    const response = await page.goto('/rhythms');
    expect(response?.status()).toBe(200);
  });
});

test.describe('Timeline page', () => {
  test('loads without crash', async ({ page }) => {
    const response = await page.goto('/timeline');
    expect(response?.status()).toBe(200);
  });
});

test.describe('Research page', () => {
  test('loads without crash', async ({ page }) => {
    const response = await page.goto('/research');
    expect(response?.status()).toBe(200);
  });
});

test.describe('Soul Mosaic page', () => {
  test('loads without crash', async ({ page }) => {
    const response = await page.goto('/soul-mosaic');
    expect(response?.status()).toBe(200);
  });
});

test.describe('Chat page', () => {
  test('loads without crash', async ({ page }) => {
    const response = await page.goto('/chat');
    expect(response?.status()).toBe(200);
  });
});
