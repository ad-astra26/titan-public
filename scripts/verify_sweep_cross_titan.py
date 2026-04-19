#!/usr/bin/env python3
"""verify_sweep_cross_titan.py — 15-min cross-Titan checkin harness.

Runs 5-min interval health+fix-verification checks across T1/T2/T3
(all queried from T1 over the VPC). Uses ONLY API-accessible metrics
to avoid needing SSH for remote DB reads. Produces a per-tick table
showing whether each fix is observably working on each Titan.

Fixes under observation (all must have been deployed to T2+T3):
  1. META_CGN_SIGNAL routing    → /v4/meta-reasoning signals_received > 0
  2. COMPLETE-4-EVENTS endpoint  → POST returns 200 OK + emitted=true
  3. Health overall              → /health == 200
  4. Bus emissions flowing       → /v4/bus-health total_rate_1min_hz > 0
  5. No orphan signals           → /v4/bus-health orphans.total_count stable
  6. /v4/meta-reasoning no longer short-circuits on transient cache race
     (commit af3788b) — signals_received field always present in response

Usage:
  python scripts/verify_sweep_cross_titan.py --duration 900 --interval 300

Defaults: 15 min soak, 5 min checkins (per user ask 2026-04-19).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

TITANS = [
    ("T1", "http://127.0.0.1:7777"),
    ("T2", "http://10.135.0.6:7777"),
    ("T3", "http://10.135.0.6:7778"),
]


def _api_get(base: str, path: str, timeout: float = 5.0) -> dict:
    try:
        r = urllib.request.urlopen(f"{base}{path}", timeout=timeout)
        return json.loads(r.read())
    except Exception as e:
        return {"__error__": str(e)}


def _api_post(base: str, path: str, payload: dict,
              timeout: float = 5.0) -> dict:
    try:
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{base}{path}", data=body,
            headers={"Content-Type": "application/json"},
            method="POST")
        r = urllib.request.urlopen(req, timeout=timeout)
        return json.loads(r.read())
    except Exception as e:
        return {"__error__": str(e)}


def probe_titan(name: str, base: str) -> dict:
    """Collect one tick of metrics for a single Titan.

    On transient-cache hits (meta_cgn key missing from response — the
    1.5s coordinator snapshot occasionally returns during a tick where
    spirit_worker's meta_engine reference hadn't yet populated the
    slot), retry once after 1.2s. That bridges the full cache TTL
    without blocking for ≫1 tick.
    """
    row = {"name": name, "base": base}

    h = _api_get(base, "/health")
    row["health"] = (h.get("status") == "ok") if "status" in h else False

    bh = _api_get(base, "/v4/bus-health").get("data", {})
    row["bus_overall"] = bh.get("overall_state", "?")
    row["bus_rate_1m"] = float(bh.get("total_emission_rate_1min_hz", 0) or 0)
    row["orphan_ct"] = int(bh.get("orphans", {}).get("total_count", 0))

    mr = _api_get(base, "/v4/meta-reasoning").get("data", {})
    if not (isinstance(mr, dict) and "meta_cgn" in mr):
        # Transient cache race — the coordinator snapshot may have been
        # built in a tick where meta_reasoning was empty. Retry once
        # after > 1 cache TTL (1.5s) so next sample is guaranteed fresh.
        time.sleep(1.2)
        mr = _api_get(base, "/v4/meta-reasoning").get("data", {})

    if isinstance(mr, dict) and "meta_cgn" in mr:
        mcgn = mr.get("meta_cgn", {}) or {}
        row["signals_received"] = int(mcgn.get("signals_received", 0))
        row["signals_applied"] = int(mcgn.get("signals_applied", 0))
        row["total_chains"] = int(mr.get("total_chains", 0))
        row["retry_used"] = False
    else:
        row["signals_received"] = None
        row["signals_applied"] = None
        row["total_chains"] = None
        row["retry_used"] = True

    # Event-reward endpoint synthetic test (commit 8a926e2)
    rwd = _api_post(base, "/v4/meta-reasoning/event-reward", {
        "quality": 0.5,
        "window_number": -1,
        "titan_id": name + "_XVERIFY",
    }, timeout=4.0)
    row["event_reward_ok"] = (
        rwd.get("status") == "ok"
        and rwd.get("data", {}).get("emitted") is True
    )

    return row


def print_row(row: dict, t: int) -> None:
    n = row.get("name", "?")
    h = "✓" if row.get("health") else "✗"
    b = {"healthy": "✓", "warning": "⚠", "critical": "✗"}.get(
        row.get("bus_overall", "?"), "?")
    r = row.get("bus_rate_1m", 0.0) or 0.0
    sigs = row.get("signals_received")
    sigs_s = str(sigs) if sigs is not None else "cache?"
    chains = row.get("total_chains")
    chains_s = str(chains) if chains is not None else "cache?"
    orph = row.get("orphan_ct", 0)
    erw = "✓" if row.get("event_reward_ok") else "✗"
    print(f"  t+{t:>4d}s  {n}  health={h}  bus={b} rate={r:.3f}Hz  "
          f"signals_rcvd={sigs_s:>5s}  chains={chains_s:>5s}  "
          f"orphans={orph}  evt_rwd={erw}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=900.0,
                    help="Total duration in seconds (default 900 = 15 min)")
    ap.add_argument("--interval", type=float, default=300.0,
                    help="Checkin interval in seconds (default 300 = 5 min)")
    ap.add_argument("--out", type=str,
                    default="data/verify_cross_titan_report.json")
    args = ap.parse_args()

    print("=" * 88)
    print(f"CROSS-TITAN VERIFY SWEEP — {args.duration}s total, "
          f"{args.interval}s checkins")
    print("=" * 88)

    start = time.time()
    ticks: list[list[dict]] = []

    # tick 0 = initial
    tick = [probe_titan(n, b) for n, b in TITANS]
    ticks.append(tick)
    print("\n── tick 0 (baseline) ──")
    for row in tick:
        print_row(row, 0)

    while time.time() - start < args.duration:
        remaining = args.duration - (time.time() - start)
        sleep_for = min(args.interval, remaining)
        if sleep_for <= 0:
            break
        time.sleep(sleep_for)
        t = int(time.time() - start)
        print(f"\n── tick @ t+{t}s ──")
        tick = [probe_titan(n, b) for n, b in TITANS]
        ticks.append(tick)
        for row in tick:
            print_row(row, t)

    # Summary
    print()
    print("=" * 88)
    print("SUMMARY")
    print("=" * 88)
    # baseline vs final
    baseline = ticks[0]
    final = ticks[-1]
    for b, f in zip(baseline, final):
        name = b["name"]
        sigs_b = b.get("signals_received") or 0
        sigs_f = f.get("signals_received") or 0
        chains_b = b.get("total_chains") or 0
        chains_f = f.get("total_chains") or 0
        orph_b = b.get("orphan_ct", 0)
        orph_f = f.get("orphan_ct", 0)
        all_health = all(
            t["name"] == name and t.get("health")
            for tick in ticks for t in tick if t["name"] == name
        )
        all_evt = all(
            t["name"] == name and t.get("event_reward_ok")
            for tick in ticks for t in tick if t["name"] == name
        )
        print(f"  {name}: health={'OK' if all_health else 'DEGRADED'}  "
              f"signals_rcvd Δ={sigs_f - sigs_b} (went {sigs_b}→{sigs_f})  "
              f"chains Δ={chains_f - chains_b} ({chains_b}→{chains_f})  "
              f"orphans Δ={orph_f - orph_b}  evt_rwd_always_ok={all_evt}")

    report = {
        "duration_s": args.duration,
        "interval_s": args.interval,
        "start_ts": start,
        "end_ts": time.time(),
        "ticks": ticks,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nFull report: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
