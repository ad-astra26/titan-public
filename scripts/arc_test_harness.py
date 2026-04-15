#!/usr/bin/env python
"""
scripts/arc_test_harness.py — ARC reward-rebalance validation harness.

Analyses the `episode_diagnostics_{YYYYMMDD}.jsonl` files emitted by
titan_plugin/logic/arc/session.py across T1/T2/T3 and reports:

  • L1 completion rate per game
  • Action entropy distribution (collapse detector)
  • Reward-breakdown averages per term
  • Character-target detection rate (ls20-only G2 signal)
  • Epsilon observed vs epsilon_min floor

Modes:
  --compare <cutoff_utc>   Report BEFORE vs AFTER a UTC timestamp cutoff
                           (e.g. "2026-04-15T14:00:00Z" = when rebalance deployed)
  --last N                 Show only last N episodes per Titan per game
  --titan t1|t2|t3|all     Which Titan(s) to pull from (default all)

Pulls remote diagnostics via ssh for T2/T3; reads local files for T1.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Iterable

LOCAL_DIAG_DIR = "data/arc_agi_3"
REMOTE_T2_DIAG_DIR = "/home/antigravity/projects/titan/data/arc_agi_3"
REMOTE_T3_DIAG_DIR = "/home/antigravity/projects/titan3/data/arc_agi_3"


def _local_entries(days: int) -> list[dict]:
    entries: list[dict] = []
    today = datetime.utcnow().date()
    for offset in range(days):
        date = today - timedelta(days=offset)
        path = os.path.join(LOCAL_DIAG_DIR, f"episode_diagnostics_{date.strftime('%Y%m%d')}.jsonl")
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries


def _remote_entries(remote_dir: str, days: int) -> list[dict]:
    """Fetch diagnostic files from T2/T3 via ssh. Best-effort — silent on ssh fail."""
    entries: list[dict] = []
    today = datetime.utcnow().date()
    for offset in range(days):
        date = today - timedelta(days=offset)
        fname = f"episode_diagnostics_{date.strftime('%Y%m%d')}.jsonl"
        try:
            out = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=5", "root@10.135.0.6",
                 f"cat {remote_dir}/{fname} 2>/dev/null || true"],
                capture_output=True, text=True, timeout=15,
            )
            for line in out.stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        except Exception:
            pass
    return entries


def _summarize(label: str, entries: list[dict]) -> None:
    print(f"\n{label}  (n={len(entries)})")
    print("-" * 78)
    if not entries:
        print("  (no entries)")
        return

    per_game: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        per_game[e.get("game_id", "?")].append(e)

    for game, items in sorted(per_game.items()):
        wins = [e for e in items if e.get("final_state") == "WIN"]
        levels = sum(e.get("levels_completed", 0) for e in items)
        entropies = [e.get("action_entropy_norm", 0) for e in items if e.get("action_entropy_norm") is not None]
        eps = [e.get("epsilon_at_start", 0) for e in items if e.get("epsilon_at_start") is not None]

        # Reward breakdown (new schema after 2026-04-15)
        rb_sums: dict[str, list[float]] = defaultdict(list)
        for e in items:
            rb = e.get("reward_breakdown") or {}
            for k, v in rb.items():
                rb_sums[k].append(float(v))

        # Character-target detection rate
        ct_rates = [
            (e.get("character_target_signal") or {}).get("detection_rate", 0.0)
            for e in items if (e.get("character_target_signal") or {})
        ]

        print(f"  {game:5}  episodes={len(items):4}  wins={len(wins):3}  levels={levels:3}  "
              f"L1_rate={len(wins)/len(items)*100:5.1f}%")
        if entropies:
            print(f"         action_entropy_norm  avg={statistics.mean(entropies):.3f}  "
                  f"min={min(entropies):.3f}  max={max(entropies):.3f}")
        if eps:
            print(f"         epsilon_at_start     avg={statistics.mean(eps):.4f}  "
                  f"min={min(eps):.4f}  max={max(eps):.4f}")
        if rb_sums:
            row = "         reward_breakdown avg: " + "  ".join(
                f"{k}={statistics.mean(v):+.3f}" for k, v in sorted(rb_sums.items())
            )
            print(row)
        if ct_rates and game == "ls20":
            avg_ct = statistics.mean(ct_rates)
            nonzero_ct = sum(1 for r in ct_rates if r > 0)
            print(f"         character_target     avg_det_rate={avg_ct:.3f}  "
                  f"nonzero_episodes={nonzero_ct}/{len(ct_rates)}")


def _filter_by_cutoff(entries: list[dict], cutoff: str, before: bool) -> list[dict]:
    cutoff_dt = datetime.strptime(cutoff, "%Y-%m-%dT%H:%M:%SZ")
    out = []
    for e in entries:
        ts = e.get("ts_utc")
        if not ts:
            continue
        try:
            edt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            continue
        if before and edt < cutoff_dt:
            out.append(e)
        elif not before and edt >= cutoff_dt:
            out.append(e)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--titan", choices=["t1", "t2", "t3", "all"], default="all")
    ap.add_argument("--days", type=int, default=7, help="Days of history to read")
    ap.add_argument("--compare", help="UTC cutoff (YYYY-MM-DDTHH:MM:SSZ) for BEFORE/AFTER split")
    ap.add_argument("--last", type=int, help="Only consider last N entries per Titan")
    args = ap.parse_args()

    sources: dict[str, list[dict]] = {}
    if args.titan in ("t1", "all"):
        sources["T1"] = _local_entries(args.days)
    if args.titan in ("t2", "all"):
        sources["T2"] = _remote_entries(REMOTE_T2_DIAG_DIR, args.days)
    if args.titan in ("t3", "all"):
        sources["T3"] = _remote_entries(REMOTE_T3_DIAG_DIR, args.days)

    for titan, entries in sources.items():
        entries.sort(key=lambda e: e.get("ts_utc", ""))
        if args.last:
            entries = entries[-args.last:]
        if args.compare:
            print(f"\n{'='*78}\n{titan}  — BEFORE/AFTER split at {args.compare}\n{'='*78}")
            _summarize(f"{titan} BEFORE", _filter_by_cutoff(entries, args.compare, before=True))
            _summarize(f"{titan} AFTER ", _filter_by_cutoff(entries, args.compare, before=False))
        else:
            print(f"\n{'='*78}\n{titan}\n{'='*78}")
            _summarize(titan, entries)

    return 0


if __name__ == "__main__":
    sys.exit(main())
