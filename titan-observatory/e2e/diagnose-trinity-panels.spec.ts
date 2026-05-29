import { test, expect } from '@playwright/test';

/**
 * Diagnostic E2E test: capture which Trinity panels render real data vs
 * loading/empty state after rFP_phase_c_substrate_observable_closure deploy.
 * Reports per-panel state so we can target the actual broken renderers.
 */
test('diagnose Trinity panels after rFP deploy', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (err) => errors.push(`PAGE_ERROR: ${err.message}`));
  page.on('console', (msg) => {
    if (msg.type() === 'error') errors.push(`CONSOLE_ERROR: ${msg.text()}`);
  });

  await page.goto('https://iamtitan.tech/trinity?tab=architecture', { waitUntil: 'domcontentloaded', timeout: 20000 });
  await page.waitForTimeout(8000); // let React Query settle + body cycles tick

  // Screenshot the full page for visual diff
  await page.screenshot({ path: '/tmp/diagnose-trinity-T1.png', fullPage: true });

  // Find SECTIONS by their headings
  const sections = await page.locator('text=/SPHERE CLOCKS|INNER TRINITY|GLOBAL OBSERVABLES|OUTER TRINITY|SPACE TOPOLOGY/i').allTextContents();
  console.log('SECTIONS FOUND:', sections);

  // Probe INNER TRINITY Body 5D values — these were 0.50 default in screenshot
  const innerTrinityText = await page.locator('text=/INNER TRINITY/i').first().locator('xpath=ancestor::*[3]').innerText().catch(() => '');
  console.log('=== INNER TRINITY block ===');
  console.log(innerTrinityText.split('\n').slice(0, 30).join('\n'));
  console.log();

  // Probe OUTER TRINITY for comparison
  const outerTrinityText = await page.locator('text=/OUTER TRINITY/i').first().locator('xpath=ancestor::*[3]').innerText().catch(() => '');
  console.log('=== OUTER TRINITY block ===');
  console.log(outerTrinityText.split('\n').slice(0, 30).join('\n'));
  console.log();

  // Probe GLOBAL OBSERVABLES
  const globalObsText = await page.locator('text=/GLOBAL OBSERVABLES/i').first().locator('xpath=ancestor::*[3]').innerText().catch(() => '');
  console.log('=== GLOBAL OBSERVABLES block ===');
  console.log(globalObsText.split('\n').slice(0, 30).join('\n'));
  console.log();

  // Probe SPACE TOPOLOGY heatmap
  const spaceTopoText = await page.locator('text=/SPACE TOPOLOGY/i').first().locator('xpath=ancestor::*[3]').innerText().catch(() => '');
  console.log('=== SPACE TOPOLOGY block ===');
  console.log(spaceTopoText.split('\n').slice(0, 30).join('\n'));
  console.log();

  console.log('=== JS ERRORS (count: ' + errors.length + ') ===');
  errors.slice(0, 20).forEach(e => console.log(e));

  // Don't fail; we're diagnosing
});
