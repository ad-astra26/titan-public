import { test, expect } from '@playwright/test';

test.describe('Home page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('renders without crash', async ({ page }) => {
    await expect(page.locator('body')).toBeVisible();
    await expect(page.locator('text=Application error')).not.toBeVisible();
  });

  test('displays Titan content', async ({ page }) => {
    const body = await page.textContent('body');
    expect(body).toBeTruthy();
    expect(body!.toLowerCase()).toContain('titan');
  });

  test('page has interactive content', async ({ page }) => {
    // Wait for page to fully hydrate
    await page.waitForTimeout(5000);
    // Should have links in the navigation
    const navLinks = page.locator('nav a');
    const count = await navLinks.count();
    expect(count).toBeGreaterThanOrEqual(5);
  });
});
