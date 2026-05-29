// ── E2E: post-build route smoke (every route, every component) ──
//
// Runs after every `npm run build` via `npm run smoke`.
// Catches the class of bug that the 2026-05-19 NS-payload-drift + the
// 2026-05-19 corrupted-.next-bundle incidents both surfaced: individual
// components silently failing to mount while the build "succeeds."
//
// For each route the suite asserts, in order:
//   1. HTTP 200 + page reaches networkidle within 10s
//   2. No 4xx/5xx network requests (catches missing API endpoints,
//      stale-chunk 400s, broken proxy paths)
//   3. No uncaught console errors (favicon noise filtered)
//   4. Every route-specific hallmark text/role is visible (catches
//      blank-component regressions like NeuralNSMini rendering nothing
//      when payload shape drifts)
//
// Wired by `npm run smoke` (titan-observatory/package.json). Intended to
// be the ONE test the CI / pre-deploy gate must pass. Cheap (~30-60s).

import { test, expect, type ConsoleMessage } from '@playwright/test';

const FRONTEND = 'http://localhost:3000';

/** A route the smoke test must validate.
 *
 *  `hallmarks` are the text/regex patterns that MUST be visible on the
 *  page once queries settle — pick one per top-level component on the
 *  route so a single blank widget fails the route, not the whole suite. */
interface RouteSpec {
  path: string;
  hallmarks: RegExp[];
  /** Routes that legitimately redirect — skip 'no 4xx' check for the
   *  redirect's own intermediate response. */
  expectsRedirect?: boolean;
}

const ROUTES: RouteSpec[] = [
  {
    path: '/',
    hallmarks: [
      /T1|T2|T3/,                      // TitanSelector
      /DA|5HT|NE/,                     // NeuromodStrip
      /Dream|Asleep|Awake|GABA/i,      // DreamingIndicator
      /Neural NS|Program|firing/i,     // HormonalMini (renders "Neural NS … firing")
      /Chi|Life Force/i,               // ChiLifeForce
      /Sovereign/i,                    // SovereigntyGauge
      /Emergent Rhythms|Circadian|UTC/i, // CircadianClock (renders "EMERGENT RHYTHMS")
      // /Mood/i — MoodIndicator removed from home page (import remains
      //   but is unused; see app/page.tsx). Re-add this hallmark if
      //   <MoodIndicator/> is restored to the layout.
      /Vault|chain/i,                  // VaultStatus
      // /Agency|Recent|Activity|Disabled/i — AgencyFeed + ActivityFeed
      //   removed from home 2026-05-14 (see app/page.tsx import comments).
      /Neural Nervous System|progs/i,  // NeuralNSMini
      /SOL Balance/,                   // MetricCard
      /Vocabulary/,                    // MetricCard
      /Neural NS/,                     // MetricCard
      /Creations/,                     // MetricCard
      /Updated .* ago|just now/,       // GlobalFreshnessPill
    ],
  },
  {
    path: '/expression',
    hallmarks: [
      /Voice|Expression/i,             // PageHeader
      /Feed|Creative|Language|Reasoning|Social|Persona/, // SubTabs
    ],
  },
  {
    path: '/trinity',
    hallmarks: [
      /Trinity/i,                      // PageHeader
      /Body|Mind|Spirit/,              // Trinity tabs
    ],
  },
  {
    path: '/neurology',
    hallmarks: [
      /Neurochemistry|Dreams|Nervous System|Reflexes/, // SubTabs
    ],
  },
  {
    path: '/neurology?tab=nervous-system',
    hallmarks: [
      /Total Fires|Total Updates|Avg Urgency|Peak Urgency/, // 4 summary cards
      /Neural Programs/,
      /REFLEX|FOCUS|INTUITION/,        // at least one program row
    ],
  },
  {
    path: '/timechain',
    hallmarks: [
      /TimeChain|Chain/i,
    ],
  },
  {
    path: '/creative',
    hallmarks: [
      /Creative/i,
    ],
  },
  {
    path: '/kin',
    hallmarks: [/Kin/i],
  },
  {
    path: '/persona',
    hallmarks: [/Persona/i],
  },
  {
    path: '/dreams',
    hallmarks: [/Dream/i],
  },
  {
    path: '/rhythms',
    hallmarks: [/Rhythm|Circadian|Schumann/i],
  },
  {
    path: '/language',
    hallmarks: [/Language|Vocab/i],
  },
  {
    path: '/research',
    hallmarks: [/Research/i],
  },
  {
    path: '/stats',
    hallmarks: [/Stats|metric/i],
  },
  {
    path: '/system',
    hallmarks: [/System/i],
  },
  {
    // /timeline is a server-side redirect to /trinity?tab=unified-spirit
    // (Phase 2 IA — folded into Self · Trinity tab). After the 307 the
    // landed page renders Trinity content, not "Timeline".
    path: '/timeline',
    hallmarks: [/Trinity|Unified Spirit/i],
    expectsRedirect: true,
  },
  {
    path: '/soul-mosaic',
    hallmarks: [/Soul|Mosaic|Kin/i],
  },
  {
    path: '/world',
    hallmarks: [/World/i],
  },
  {
    path: '/reflexes',
    hallmarks: [/Reflex/i],
  },
  {
    path: '/compare',
    hallmarks: [/Compare|T1.*T2.*T3/is],
  },
  {
    path: '/chat',
    hallmarks: [/Chat|Send|Message/i],
  },
  {
    // Phase 10 (D-SPEC-PHASE10) — Synthesis Engine metrics panel.
    path: '/synthesis',
    hallmarks: [
      /Synthesis Engine/i,        // PageHeader
      /Sovereignty Ratio/i,       // headline card
      /Skill Library|Groundedness|Retrieval/i, // sub-cards
    ],
  },
];

/** Console errors that are safe to ignore (3rd-party noise that doesn't
 *  represent a real Observatory bug). Keep this list TIGHT — if you find
 *  yourself adding many entries, you're probably hiding real bugs. */
const CONSOLE_IGNORES = [
  /favicon/i,
  /404 \(Not Found\) http.*\/favicon/i,
  // Next.js Link-prefetch RSC fetches abort when Playwright navigates away
  // from a page before its hover-prefetches complete. The fallback path
  // (browser navigation) still works — the error is purely a console-level
  // artifact of fast sequential navigation. Not a user-visible bug.
  /Failed to fetch RSC payload .* Falling back to browser navigation/i,
];

// Single sequential test iterating ALL routes inside ONE browser context.
// This is intentionally resource-frugal: 1 chromium process for the whole
// sweep instead of 1-per-test, so it can run on a host that's also serving
// T1+T2+T3 + the Observatory itself without thrashing.
//
// Trade-off: a single failure stops the suite at that route instead of
// continuing through. The summary at the end collates everything that
// failed, so the user gets the full failure picture from one run.
test('smoke: all routes sequentially', async ({ page }) => {
  // 21 routes × ~6-8s each (goto + 3s settle + hallmark probes) = ~3min worst case.
  // Default 30s per-test timeout is too tight for one-context sequential design.
  test.setTimeout(300_000);
  const consoleErrors: Array<{ route: string; text: string }> = [];
  const failedRequests: Array<{ route: string; url: string; status: number }> = [];
  const missingPerRoute: Record<string, string[]> = {};
  let currentRoute = '<init>';

  page.on('console', (msg: ConsoleMessage) => {
    if (msg.type() === 'error') {
      const text = msg.text();
      if (!CONSOLE_IGNORES.some((re) => re.test(text))) {
        consoleErrors.push({ route: currentRoute, text });
      }
    }
  });
  page.on('response', async (res) => {
    const url = res.url();
    const status = res.status();
    if (status >= 400 && !url.includes('favicon') && url.startsWith(FRONTEND)) {
      failedRequests.push({ route: currentRoute, url, status });
    }
  });

  for (const route of ROUTES) {
    currentRoute = route.path;
    // `domcontentloaded` is much faster than `networkidle` (which waits for
    // 500ms of network silence — many home components poll every 2-3s, so
    // networkidle effectively never settles and burns the timeout).
    const resp = await page.goto(`${FRONTEND}${route.path}`, {
      waitUntil: 'domcontentloaded',
      timeout: 10_000,
    });
    if (!route.expectsRedirect && resp && resp.status() >= 400) {
      missingPerRoute[route.path] = [`HTTP ${resp.status()}`];
      continue;
    }
    // Settle pass — React Query hydrates async, components have a 1-3s
    // polling cadence. 3s settle is enough for hallmarks to render.
    await page.waitForTimeout(3_000);

    const missing: string[] = [];
    for (const re of route.hallmarks) {
      const count = await page.getByText(re).count();
      if (count === 0) missing.push(re.toString());
    }
    if (missing.length) missingPerRoute[route.path] = missing;
  }

  // Summary
  const totalMissing = Object.values(missingPerRoute).reduce(
    (s, m) => s + m.length,
    0,
  );
  if (totalMissing || consoleErrors.length || failedRequests.length) {
    console.log('\n── SMOKE AUDIT FAILURES ──');
    for (const [path, missing] of Object.entries(missingPerRoute)) {
      console.log(`  ${path}:`);
      missing.forEach((m) => console.log(`    missing: ${m}`));
    }
    if (consoleErrors.length) {
      console.log('  Console errors:');
      consoleErrors.slice(0, 20).forEach((e) =>
        console.log(`    [${e.route}] ${e.text.substring(0, 180)}`),
      );
    }
    if (failedRequests.length) {
      console.log('  Failed network requests:');
      failedRequests.slice(0, 25).forEach((r) =>
        console.log(`    [${r.route}] ${r.status} ${r.url}`),
      );
    }
  } else {
    console.log(`\n── SMOKE OK ── ${ROUTES.length} routes probed, all green`);
  }

  expect(
    totalMissing,
    `${totalMissing} missing hallmark(s) across ${Object.keys(missingPerRoute).length} route(s)`,
  ).toBe(0);
  expect(
    consoleErrors,
    `${consoleErrors.length} console error(s)`,
  ).toHaveLength(0);
  expect(
    failedRequests,
    `${failedRequests.length} 4xx/5xx request(s)`,
  ).toHaveLength(0);
});
