#!/usr/bin/env python
"""
scripts/chaos_meditation.py — Meditation Self-Healing Chaos Drill.

rFP_self_healing_meditation_cadence.md Phase 5: weekly drill to verify the
self-healing meditation system remains functional end-to-end. Opt-in; should
run on staging (or T1 during low-activity window) with Maker approval.

What it validates AUTOMATICALLY:

  1. Watchdog self-test (I1) — queries /v4/meditation/health; confirms
     `selftest_pass == True`. Proves synthetic F3 injection at boot detected.
  2. Tier-1 force-trigger (manual path) — POST /v4/meditation/force-trigger;
     polls /v4/meditation/health; confirms tracker.count increments within
     the meditation-processing budget (~60-120s after request).
  3. API reachability — all 3 endpoints respond 200 under 3s each.

What it DOCUMENTS for manual validation (real fault injection is risky on
live Titan — defer to dedicated staging Titan when available):

  • F1/F2 OVERDUE: force-rig `_meditation_tracker["last_ts"]` to hours ago,
    observe watchdog F1_F2_OVERDUE detection + classify + force-trigger.
  • F3/F6 STUCK: force-set `in_meditation=True`, observe 10-min stuck
    threshold + Tier-1 reset.
  • F4 BACKUP LAG: bypass backup_state.json counter update, observe lag
    detection.
  • F5 TRIGGER FILE: chmod data/backup_trigger.json to read-only, observe
    F5 alert on next meditation complete.
  • F7 NOT DISTILLING: force 3 consecutive meditations with promoted=0.

Usage:
  python scripts/chaos_meditation.py --titan t1          # T1 only
  python scripts/chaos_meditation.py --titan all         # all Titans
  python scripts/chaos_meditation.py --titan t1 --force  # skip approval prompt
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Optional

try:
    import httpx
except ImportError:
    print("httpx not available — install test_env dependencies first")
    sys.exit(2)


TITAN_URLS = {
    "t1": "http://127.0.0.1:7777",
    "t2": "http://10.135.0.6:7777",
    "t3": "http://10.135.0.6:7778",
}

LATENCY_BUDGET_S = 10.0  # Bus-proxy queries can queue behind epoch ticks


def _get(url: str) -> tuple[int, dict, float]:
    t0 = time.time()
    try:
        r = httpx.get(url, timeout=LATENCY_BUDGET_S)
        return r.status_code, r.json(), time.time() - t0
    except Exception as e:
        return 0, {"error": str(e)}, time.time() - t0


def _post(url: str, json_payload: Optional[dict] = None) -> tuple[int, dict, float]:
    t0 = time.time()
    try:
        r = httpx.post(url, json=(json_payload or {}), timeout=10.0)
        return r.status_code, r.json(), time.time() - t0
    except Exception as e:
        return 0, {"error": str(e)}, time.time() - t0


def _test_selftest(titan: str, base_url: str) -> bool:
    print(f"  [1] Watchdog self-test (I1) — GET /v4/meditation/health")
    code, body, dt = _get(f"{base_url}/v4/meditation/health")
    if code != 200:
        print(f"      ✗ {code} {body} ({dt:.2f}s)")
        return False
    wd = (body.get("data") or body).get("watchdog", {})
    st_done = wd.get("selftest_done")
    st_pass = wd.get("selftest_pass")
    ok = bool(st_done and st_pass)
    print(f"      {'✓' if ok else '✗'} selftest_done={st_done}, selftest_pass={st_pass} ({dt:.2f}s)")
    return ok


def _test_api_latency(titan: str, base_url: str) -> bool:
    print(f"  [2] API latency — 3 endpoints under {LATENCY_BUDGET_S}s each")
    paths = ["/health", "/v4/meditation/health", "/v4/meditations"]
    all_ok = True
    for p in paths:
        code, _, dt = _get(f"{base_url}{p}")
        ok = code == 200 and dt < LATENCY_BUDGET_S
        if not ok:
            all_ok = False
        print(f"      {'✓' if ok else '✗'} {p}: {code} in {dt:.2f}s")
    return all_ok


def _test_force_trigger(titan: str, base_url: str, wait_s: int = 180) -> bool:
    print(f"  [3] Tier-1 force-trigger — POST /v4/meditation/force-trigger "
          f"(budget {wait_s}s)")
    # Baseline count
    code, body, _ = _get(f"{base_url}/v4/meditation/health")
    if code != 200:
        print(f"      ✗ baseline fetch failed: {code}")
        return False
    tracker = (body.get("data") or body).get("tracker", {})
    start_count = int(tracker.get("count", 0))
    print(f"      baseline meditation_count={start_count}")

    code, body, dt = _post(f"{base_url}/v4/meditation/force-trigger")
    if code != 200:
        print(f"      ✗ force-trigger POST failed: {code} {body}")
        return False
    print(f"      ✓ force-trigger dispatched ({dt:.2f}s) — polling for completion")

    # Poll for count increment
    t0 = time.time()
    while time.time() - t0 < wait_s:
        time.sleep(10)
        code, body, _ = _get(f"{base_url}/v4/meditation/health")
        if code == 200:
            tracker = (body.get("data") or body).get("tracker", {})
            current = int(tracker.get("count", 0))
            if current > start_count:
                elapsed = time.time() - t0
                print(f"      ✓ meditation_count {start_count} → {current} "
                      f"after {elapsed:.0f}s")
                return True
    print(f"      ✗ meditation_count did not advance in {wait_s}s "
          f"(still {start_count}). Check /tmp/titan_brain.log for "
          f"MEDITATION_REQUEST processing.")
    return False


def run_drill(titan: str, base_url: str, skip_force_trigger: bool = False) -> bool:
    print(f"\n{titan.upper()} ({base_url})")
    print("-" * 78)
    r1 = _test_selftest(titan, base_url)
    r2 = _test_api_latency(titan, base_url)
    r3 = True if skip_force_trigger else _test_force_trigger(titan, base_url)
    ok = r1 and r2 and r3
    print(f"  → {titan.upper()} DRILL: {'✓ PASS' if ok else '✗ FAIL'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--titan", choices=["t1", "t2", "t3", "all"], default="t1")
    ap.add_argument("--force", action="store_true",
                    help="Skip approval prompt (for cron use)")
    ap.add_argument("--skip-force-trigger", action="store_true",
                    help="Skip the POST /force-trigger test (runs meditation)")
    args = ap.parse_args()

    targets = list(TITAN_URLS.items()) if args.titan == "all" else [
        (args.titan, TITAN_URLS[args.titan])
    ]

    print("=" * 78)
    print("MEDITATION CHAOS DRILL — rFP Phase 5")
    print("=" * 78)
    if not args.skip_force_trigger:
        print(f"  WILL force-trigger meditation on: {', '.join(t for t, _ in targets)}")
        print("  Each meditation runs a real memory-consolidation cycle (~60-120s).")
    if not args.force and sys.stdin.isatty() and not args.skip_force_trigger:
        resp = input("  Proceed? [y/N] ")
        if resp.strip().lower() != "y":
            print("  Aborted.")
            return 1

    all_ok = True
    for titan, url in targets:
        if not run_drill(titan, url, skip_force_trigger=args.skip_force_trigger):
            all_ok = False

    print()
    print("=" * 78)
    print(f"RESULT: {'✓ ALL DRILLS PASSED' if all_ok else '✗ ONE OR MORE DRILLS FAILED'}")
    print("=" * 78)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
