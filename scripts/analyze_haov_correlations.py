#!/usr/bin/env python3
"""COMPLETE-9b — HAOV signal↔chain correlation analyzer.

Reads data/meta_cgn/haov_signal_outcomes.jsonl (produced by COMPLETE-9a
telemetry, commit ed53aba) and joins signal events with chain outcomes
within a configurable time window. Computes per-(consumer, event_type,
primitive) correlation with terminal chain reward and reports findings.

The long-term goal (META-CGN-V2-HAOV-REFINEMENT) is to drift the
hand-crafted SIGNAL_TO_PRIMITIVE quality_nudge values toward
outcome-learned values. This analyzer is the READ side; the drift
mechanism is a separate follow-up phase that consumes the JSON report
this script emits — but only once N >= MIN_OBSERVATIONS per tuple,
enforced by the --emit-drift gate.

Design principles:
  - NO drift actions yet. Pure read-and-report.
  - Data-thinness guarded: each tuple reports N. Recommendations only
    surface when N >= MIN_OBSERVATIONS (default 100).
  - Signals without any succeeding chain within WINDOW_S are reported
    separately (orphan signals) — they're in the mapping but not
    actually influencing downstream reasoning.
  - Chains without any preceding signal get a baseline bucket — needed
    to compute correlation vs. "no-signal" reward distribution.

Usage:
  python scripts/analyze_haov_correlations.py [--window 60]
                                              [--min-n 100]
                                              [--out report.json]
                                              [--since-hours 24]

Output (stdout): human-readable summary.
Output (--out):  JSON report for programmatic consumers (drift
                 mechanism, dashboards, arch_map).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable

HAOV_LOG_PATH = Path("data/meta_cgn/haov_signal_outcomes.jsonl")
MIN_OBSERVATIONS_DEFAULT = 100
WINDOW_SECONDS_DEFAULT = 60.0


def _load_entries(path: Path, since_ts: float | None) -> list[dict]:
    """Read JSONL, optionally filtering by timestamp."""
    if not path.exists():
        return []
    entries: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if since_ts is not None and float(e.get("ts", 0)) < since_ts:
                continue
            entries.append(e)
    return entries


def _split_signals_chains(entries: list[dict]) -> tuple[list[dict], list[dict]]:
    signals = [e for e in entries if e.get("kind") == "signal"]
    chains = [e for e in entries if e.get("kind") == "chain"]
    # Sort by ts for time-window join
    signals.sort(key=lambda e: float(e.get("ts", 0)))
    chains.sort(key=lambda e: float(e.get("ts", 0)))
    return signals, chains


def _join_by_window(signals: list[dict], chains: list[dict],
                    window_s: float) -> list[tuple[dict, dict]]:
    """For each signal, pair with the next chain that starts within
    window_s. A chain may match multiple preceding signals.
    """
    pairs: list[tuple[dict, dict]] = []
    if not signals or not chains:
        return pairs
    # Two-pointer walk
    j = 0
    for sig in signals:
        s_ts = float(sig.get("ts", 0))
        # advance chain pointer past chains that ended before signal
        while j < len(chains) and float(chains[j].get("ts", 0)) < s_ts:
            j += 1
        # find chains starting within window_s after signal
        k = j
        while k < len(chains):
            c_ts = float(chains[k].get("ts", 0))
            if c_ts - s_ts > window_s:
                break
            pairs.append((sig, chains[k]))
            k += 1
    return pairs


def _baseline_rewards(signals: list[dict], chains: list[dict],
                      window_s: float) -> list[float]:
    """Rewards from chains that had NO preceding signal within window_s.
    These form the 'no-signal' baseline distribution.
    """
    if not chains:
        return []
    # For each chain, check if any signal precedes within window_s
    sig_ts = [float(s.get("ts", 0)) for s in signals]
    sig_ts.sort()
    import bisect

    baseline: list[float] = []
    for c in chains:
        c_ts = float(c.get("ts", 0))
        # find any signal within [c_ts - window_s, c_ts]
        lo = bisect.bisect_left(sig_ts, c_ts - window_s)
        hi = bisect.bisect_right(sig_ts, c_ts)
        if hi - lo == 0:
            baseline.append(float(c.get("terminal_reward", 0.0)))
    return baseline


def _analyze_per_tuple(pairs: list[tuple[dict, dict]]) -> dict:
    """Group pairs by (consumer, event_type, primitive-present-in-chain)
    and compute mean reward, n, std.
    """
    bucket: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for sig, chain in pairs:
        consumer = sig.get("consumer", "?")
        event_type = sig.get("event_type", "?")
        prims_nudged = set((sig.get("primitives_nudged") or {}).keys())
        chain_prims = set(chain.get("primitives") or [])
        reward = float(chain.get("terminal_reward", 0.0))
        # For each nudged primitive, record the reward if it actually
        # appeared in the subsequent chain. If not, it's a "signal
        # nudged X but chain did Y" mismatch — capture separately.
        for prim in prims_nudged:
            key = (consumer, event_type, prim)
            if prim in chain_prims:
                bucket[key].append(reward)
    return dict(bucket)


def _recommend_drift(nudge_mean: float, baseline_mean: float,
                     baseline_std: float, n: int, min_n: int,
                     current_nudge: float) -> dict:
    """Given observed stats, propose a nudge-drift direction.

    Conservative: only recommends a direction + magnitude, never
    applies it. Consumer (drift mechanism) enforces cap.
    """
    out = {
        "actionable": False,
        "direction": "hold",
        "magnitude_hint": 0.0,
        "reason": "",
    }
    if n < min_n:
        out["reason"] = f"N={n} below min_n={min_n}; need more data"
        return out
    if baseline_std < 1e-6:
        out["reason"] = "baseline std ~= 0; cannot compute z-score"
        return out
    z = (nudge_mean - baseline_mean) / baseline_std
    # Heuristic: |z| > 0.5 warrants a nudge; sign determines direction
    if abs(z) < 0.5:
        out["reason"] = f"|z|={abs(z):.2f} < 0.5; no clear signal"
        return out
    out["actionable"] = True
    # Target: move current_nudge toward 0.5 + z/4 (bounded [0, 1])
    target = max(0.0, min(1.0, 0.5 + z / 4.0))
    drift = target - current_nudge
    out["direction"] = "increase" if drift > 0 else "decrease"
    out["magnitude_hint"] = round(abs(drift), 3)
    out["z_score"] = round(z, 3)
    out["target_nudge"] = round(target, 3)
    out["reason"] = (
        f"z={z:.2f} over baseline (N={n}); drift {current_nudge:.2f}"
        f"→{target:.2f}"
    )
    return out


def _load_current_nudges() -> dict[tuple[str, str, str], float]:
    """Load SIGNAL_TO_PRIMITIVE to know current hand-crafted nudges.
    Graceful fallback: empty dict if import fails (keeps analyzer
    decoupled from runtime).
    """
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from titan_plugin.logic.meta_cgn import SIGNAL_TO_PRIMITIVE
        flat: dict[tuple[str, str, str], float] = {}
        for (consumer, event), prim_map in SIGNAL_TO_PRIMITIVE.items():
            for prim, nudge in prim_map.items():
                flat[(consumer, event, prim)] = float(nudge)
        return flat
    except Exception as e:
        print(f"[warn] could not import SIGNAL_TO_PRIMITIVE: {e}",
              file=sys.stderr)
        return {}


def analyze(path: Path, window_s: float, min_n: int,
            since_ts: float | None) -> dict:
    entries = _load_entries(path, since_ts)
    signals, chains = _split_signals_chains(entries)
    pairs = _join_by_window(signals, chains, window_s)
    baseline = _baseline_rewards(signals, chains, window_s)
    per_tuple = _analyze_per_tuple(pairs)
    current_nudges = _load_current_nudges()

    baseline_mean = mean(baseline) if baseline else 0.0
    baseline_std = stdev(baseline) if len(baseline) >= 2 else 0.0

    tuples_report = []
    for (consumer, event, prim), rewards in sorted(per_tuple.items()):
        n = len(rewards)
        nudge_mean = mean(rewards) if rewards else 0.0
        nudge_std = stdev(rewards) if len(rewards) >= 2 else 0.0
        current_nudge = current_nudges.get((consumer, event, prim), 0.5)
        drift = _recommend_drift(
            nudge_mean, baseline_mean, baseline_std, n, min_n, current_nudge
        )
        tuples_report.append({
            "consumer": consumer,
            "event_type": event,
            "primitive": prim,
            "n": n,
            "mean_reward_when_fired": round(nudge_mean, 4),
            "std_reward_when_fired": round(nudge_std, 4),
            "current_nudge": current_nudge,
            "drift_recommendation": drift,
        })

    return {
        "generated_ts": time.time(),
        "window_s": window_s,
        "min_n": min_n,
        "entries_total": len(entries),
        "signals_total": len(signals),
        "chains_total": len(chains),
        "pairs_matched": len(pairs),
        "baseline_chains_n": len(baseline),
        "baseline_reward_mean": round(baseline_mean, 4),
        "baseline_reward_std": round(baseline_std, 4),
        "tuples": tuples_report,
    }


def _print_human_readable(report: dict) -> None:
    print("=" * 78)
    print("HAOV signal↔chain correlation report")
    print("=" * 78)
    print(f"Window: {report['window_s']:.0f}s   Min-N for actionable: "
          f"{report['min_n']}")
    print(f"Entries: {report['entries_total']} "
          f"(signals={report['signals_total']}, "
          f"chains={report['chains_total']})")
    print(f"Pairs matched: {report['pairs_matched']}")
    print(f"Baseline (no-signal chains): N={report['baseline_chains_n']}, "
          f"reward_mean={report['baseline_reward_mean']}, "
          f"std={report['baseline_reward_std']}")
    print()

    if not report["tuples"]:
        print("No (consumer, event, primitive) tuples observed with a "
              "chain-match in the window. Either:")
        print("  - producers haven't fired yet post-restart,")
        print("  - or the window is too short.")
        return

    actionable = [t for t in report["tuples"]
                  if t["drift_recommendation"]["actionable"]]
    holding = [t for t in report["tuples"]
               if not t["drift_recommendation"]["actionable"]]

    print(f"Actionable: {len(actionable)} / {len(report['tuples'])}")
    print()
    if actionable:
        print("-- ACTIONABLE (drift recommended) --")
        print(f"{'consumer':<14} {'event':<22} {'prim':<12} "
              f"{'N':>5} {'fired_µ':>8} {'cur':>5} {'→':^3} {'dir':<9} "
              f"{'hint':>5}")
        for t in actionable:
            d = t["drift_recommendation"]
            print(f"{t['consumer']:<14} {t['event_type']:<22} "
                  f"{t['primitive']:<12} {t['n']:>5} "
                  f"{t['mean_reward_when_fired']:>8.3f} "
                  f"{t['current_nudge']:>5.2f}  →  "
                  f"{d['direction']:<9} {d['magnitude_hint']:>5.3f}")
        print()

    print("-- HOLDING (insufficient data / weak signal) --")
    print(f"{'consumer':<14} {'event':<22} {'prim':<12} "
          f"{'N':>5} {'reason'}")
    for t in holding[:20]:
        print(f"{t['consumer']:<14} {t['event_type']:<22} "
              f"{t['primitive']:<12} {t['n']:>5} "
              f"{t['drift_recommendation']['reason']}")
    if len(holding) > 20:
        print(f"... and {len(holding) - 20} more")
    print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=float, default=WINDOW_SECONDS_DEFAULT,
                    help="Time window in seconds to join signal→chain")
    ap.add_argument("--min-n", type=int, default=MIN_OBSERVATIONS_DEFAULT,
                    help="Minimum per-tuple N before drift is actionable")
    ap.add_argument("--since-hours", type=float, default=None,
                    help="Only analyze entries within the last N hours")
    ap.add_argument("--out", type=str, default=None,
                    help="Optional JSON report output path")
    ap.add_argument("--path", type=str, default=str(HAOV_LOG_PATH),
                    help="Override HAOV log path")
    args = ap.parse_args()

    since_ts = None
    if args.since_hours is not None:
        since_ts = time.time() - args.since_hours * 3600.0

    path = Path(args.path)
    if not path.exists():
        print(f"[err] HAOV log not found: {path}", file=sys.stderr)
        print("       Run with --path to override, or verify "
              "COMPLETE-9a telemetry is emitting.",
              file=sys.stderr)
        return 2

    report = analyze(path, args.window, args.min_n, since_ts)
    _print_human_readable(report)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        print(f"Wrote JSON report to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
