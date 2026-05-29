import { test, expect } from '@playwright/test';

/**
 * Verifies the 2026-05-14 metabolic-UI surgical fix (commit e00c6dcf):
 *  1. No top fixed banner with alarming language
 *  2. No page-wide grayscale/desaturate filter
 *  3. LifeForceBar widget removed from home page
 *  4. Header lacks the energy-indicator dot + label
 *
 * Internal /v4/metabolism endpoints intentionally still publish state — only
 * the public-facing dashboard chrome was changed.
 */
test.describe('Public dashboard — no metabolic-tier alarming UI', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
  });

  test('top banner is gone — no Crisis/Starvation/Low-Power/Conserving alarm strip', async ({ page }) => {
    const banned = [
      /Metabolic Crisis/i,
      /Life Support Mode/i,
      /Energy Starvation/i,
      /Mortal Crisis/i,
      /Low-Power Mode/i,
      /Conserving Energy/i,
    ];
    for (const re of banned) {
      await expect(page.getByText(re)).toHaveCount(0, { timeout: 5000 });
    }
  });

  test('page-wide grayscale/desaturate filter is NOT applied', async ({ page }) => {
    // Wrapper must not carry .low-power / .starvation / .dead-mode classes.
    const wrapper = page.locator('main').first().locator('xpath=..');
    await expect(wrapper).not.toHaveClass(/low-power|starvation|dead-mode/);
    // And the computed filter on the wrapper must not be grayscale / saturate(<1)
    const filter = await wrapper.evaluate((el) => getComputedStyle(el).filter);
    expect(filter).not.toMatch(/grayscale\(\s*1/i);
    expect(filter).not.toMatch(/saturate\(\s*0\.[0-9]+/i);
  });

  test('LifeForceBar widget is removed from home', async ({ page }) => {
    // Heading text was "Metabolic Energy · SOL". The bar widget is gone.
    await expect(page.getByText(/Metabolic Energy/i)).toHaveCount(0);
  });

  test('Header energy-indicator dot + label removed', async ({ page }) => {
    // Header used to show a tiny dot + ENERGY_STATE label like "LOW POWER".
    // After removal: no element in the sticky header should match those labels.
    const header = page.locator('header').first();
    const labels = [/THRIVING/i, /HEALTHY/i, /CONSERVING/i, /SURVIVAL/i, /EMERGENCY/i, /HIBERNATION/i, /LOW POWER/i];
    for (const re of labels) {
      await expect(header.getByText(re)).toHaveCount(0);
    }
  });

  test('SOL balance still appears in bottom metric grid (we did not hide raw data)', async ({ page }) => {
    // The metric card grid retains a "SOL Balance" tile — raw value only,
    // no progress bar, no energy state copy.
    await expect(page.getByText(/SOL Balance/i)).toBeVisible({ timeout: 5000 });
  });

  test('CHI LIFE FORCE remains as the cognitive-energy indicator', async ({ page }) => {
    await expect(page.getByText(/CHI LIFE FORCE/i)).toBeVisible({ timeout: 5000 });
  });

  test('screenshot for visual verification', async ({ page }, testInfo) => {
    await page.waitForTimeout(3000);
    const buf = await page.screenshot({ fullPage: true });
    await testInfo.attach('home-post-fix', { body: buf, contentType: 'image/png' });
  });
});
