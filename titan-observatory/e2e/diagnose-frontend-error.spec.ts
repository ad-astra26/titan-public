/**
 * Quick diagnostic — load the home page in Chromium, capture all
 * console messages + page errors + failed requests. Goal: surface
 * the client-side exception the user is seeing as "Application error".
 */
import { test, expect } from '@playwright/test';

test('home page client-side exception capture', async ({ page }) => {
  const consoleMessages: { type: string; text: string }[] = [];
  const pageErrors: string[] = [];
  const failedRequests: { url: string; status: number; method: string }[] = [];

  page.on('console', msg => {
    consoleMessages.push({ type: msg.type(), text: msg.text() });
  });

  page.on('pageerror', err => {
    pageErrors.push(`${err.name}: ${err.message}\n${err.stack || ''}`);
  });

  page.on('requestfailed', req => {
    failedRequests.push({
      url: req.url(),
      status: 0,
      method: req.method(),
    });
  });

  page.on('response', resp => {
    if (resp.status() >= 400) {
      failedRequests.push({
        url: resp.url(),
        status: resp.status(),
        method: resp.request().method(),
      });
    }
  });

  await page.goto('http://localhost:3000/', { waitUntil: 'networkidle', timeout: 30000 });

  // Wait a bit for late client-side code to run
  await page.waitForTimeout(3000);

  // Check whether the "Application error" banner is showing
  const bodyText = await page.locator('body').innerText().catch(() => '');
  const hasAppError = bodyText.includes('Application error');

  console.log('\n========== DIAGNOSTIC REPORT ==========');
  console.log(`App-error banner visible: ${hasAppError}`);
  console.log(`Body text length: ${bodyText.length}`);
  console.log(`Body preview (first 300): ${bodyText.slice(0, 300)}`);

  console.log('\n----- pageerror (client-side JS exceptions) -----');
  if (pageErrors.length === 0) {
    console.log('  (none)');
  } else {
    pageErrors.forEach((e, i) => console.log(`  [${i}] ${e.slice(0, 1500)}`));
  }

  console.log('\n----- failed requests (>=400 or network error) -----');
  if (failedRequests.length === 0) {
    console.log('  (none)');
  } else {
    failedRequests.forEach((r, i) =>
      console.log(`  [${i}] ${r.method} ${r.url} → ${r.status}`));
  }

  console.log('\n----- console errors -----');
  const consoleErrors = consoleMessages.filter(m => m.type === 'error' || m.type === 'warning');
  if (consoleErrors.length === 0) {
    console.log('  (none)');
  } else {
    consoleErrors.slice(0, 20).forEach((m, i) =>
      console.log(`  [${i}] ${m.type}: ${m.text.slice(0, 500)}`));
  }

  console.log('=======================================\n');
});
