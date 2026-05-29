import { test, expect } from '@playwright/test';

test.describe('Expression page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/expression');
  });

  test('renders with tab navigation', async ({ page }) => {
    await expect(page.getByRole('button', { name: 'Feed' })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Creative' })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Reasoning' })).toBeVisible();
  });

  test('Feed tab is default', async ({ page }) => {
    const feedTab = page.getByRole('button', { name: 'Feed' });
    await expect(feedTab).toHaveClass(/titan-haze/);
  });

  test('switching to Creative tab works', async ({ page }) => {
    await page.getByRole('button', { name: 'Creative' }).click();
    await page.waitForURL('**/expression?tab=creative');
  });

  test('switching to Reasoning tab works', async ({ page }) => {
    await page.getByRole('button', { name: 'Reasoning' }).click();
    await page.waitForURL('**/expression?tab=reasoning');
  });
});

test.describe('Reasoning tab content', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/expression?tab=reasoning');
  });

  test('shows narration header', async ({ page }) => {
    await expect(page.getByText('thinking process', { exact: false })).toBeVisible({ timeout: 15000 });
  });

  test('shows metric cards after data loads', async ({ page }) => {
    await expect(page.getByText('Reasoning Chains')).toBeVisible({ timeout: 15000 });
    await expect(page.getByText('Meta Chains')).toBeVisible({ timeout: 15000 });
    await expect(page.getByText('Wisdom Saved')).toBeVisible({ timeout: 15000 });
  });

  test('shows confidence and neuromod panels', async ({ page }) => {
    // These panels render after API data loads
    await expect(page.getByText('Reasoning Confidence')).toBeVisible({ timeout: 20000 });
    await expect(page.getByText('Mind-State During Reasoning')).toBeVisible({ timeout: 10000 });
  });

  test('shows cognitive primitives when meta-reasoning data available', async ({ page }) => {
    // This depends on meta-reasoning API returning primitive_counts
    // Wait for either primitives or the "between chains" indicators
    const primitives = page.getByText('Cognitive Primitives');
    const betweenChains = page.getByText('Between reasoning chains');
    await expect(primitives.or(betweenChains).first()).toBeVisible({ timeout: 15000 });
  });
});
