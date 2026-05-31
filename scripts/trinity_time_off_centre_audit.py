#!/usr/bin/env python3
"""trinity_time_off_centre_audit.py — P1 §9.8 "does the loop reveal a need?" probe.

Measures whether a 6th Trinity observable — `∫delta·dt` (time-off-centre, the
metabolic-coupling signal proposed in `ARCHITECTURE_trinity.md §9.8`) — would
carry **real, non-redundant signal with a consumer that needs it**, BEFORE it is
added to the proven 5-observable basis (`coherence, magnitude, velocity,
direction, polarity`). This is the empirical test of the §9.8 gate clause
*"add when the loop reveals a need"* (and the `feedback_no_hardcoded_values_
emergence_over_determinism` / `feedback_wire_now_gate_later` discipline).

OBSERVATION-ONLY. No kernel/daemon/SPEC change. Reads already-persisted data:

  - `trinity_journey_gifts`  — per balance-cycle digest (P0.5). Each row is one
        journey from last-balanced-pulse → this-balanced-pulse. `cycle_tick_count`
        is the cleanest ABSOLUTE time-off-centre proxy (longer journey = longer
        spent off the Middle Path before rebalancing). `journey_metadata` BLOB
        carries per-dim peak_excursion + a PER-CYCLE-NORMALISED excursion_integral
        (absolute scale is divided out at pack time and NOT persisted — see
        journey_persistence_worker._pack_journey_metadata — so the stored integral
        is shape-only; this is itself evidence the absolute integral is currently
        unmeasured).
  - `trinity_corrective_events` — the extreme-imbalance tail (P0.6-C). Exposes
        `duration_ticks` (how long the excursion was sustained before the
        PolarityHomeostat fired) + `polarity_at_fire` — a directly-readable
        time-off-centre vs instantaneous-polarity pair for the redundancy test.

THREE GATES (a 6th observable earns its place only if all three pass):

  G1 RICHNESS    — is time-off-centre a variable signal, not a near-constant?
                   measured by cycle_tick_count coefficient-of-variation (CV) and
                   the heavy-tail ratio p95/p50. Low CV / no tail → the 5 existing
                   observables already span the state; a duration integral adds
                   nothing.
  G2 INDEPENDENCE— does duration add information beyond the instantaneous
                   `polarity` observable? measured by |Pearson(duration_ticks,
                   polarity_at_fire)| on the corrective tail. High correlation →
                   redundant with `polarity`; the integral is a rescaling, not a
                   new axis.
  G3 CONSUMER    — does a downstream consumer that NEEDS a continuous time-off-
                   centre signal exist yet? (the §9.8 metabolic loop / URGE-to-
                   rest). Detected by a static scan; if absent, wiring a 6th
                   observable now is premature (wire-now-gate-later at best).

VERDICT = NEED only if G1 ∧ G2 ∧ G3. Otherwise GATE, with the failing gate named.

Usage:
  python scripts/trinity_time_off_centre_audit.py --db data/consciousness.db --titan-id T1
  python scripts/trinity_time_off_centre_audit.py --fleet
  python scripts/trinity_time_off_centre_audit.py --db data/consciousness.db --json out.json
"""
import argparse
import json
import math
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

# Trinity per-layer dim counts (power-of-three law: body 5, mind 15, spirit 45).
_PART_DIMS = {"body": 5, "mind": 15, "spirit": 45}

# G1/G2 thresholds — transparent, documented, not tuned. A 6th observable must
# clear all three to be "needed". These are deliberately permissive (easy to
# pass) so the verdict errs toward NEED only when the data is unambiguous.
CV_RICH = 0.50          # cycle_tick_count CV above this = genuinely variable
TAIL_RICH = 2.0         # p95/p50 above this = heavy tail (sustained excursions)
PEARSON_REDUNDANT = 0.70  # |r| above this = duration ~ polarity (redundant)


def _pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def _pct(sorted_vals, q):
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, int(q * (len(sorted_vals) - 1)))
    return float(sorted_vals[idx])


def _cv(vals):
    n = len(vals)
    if n < 2:
        return 0.0
    m = sum(vals) / n
    if m == 0:
        return 0.0
    var = sum((v - m) ** 2 for v in vals) / n
    return math.sqrt(var) / m


def _decode_peak_integral(blob: bytes, part: str):
    """Decode mean peak_excursion + mean (normalised) excursion_integral from the
    journey_metadata BLOB. Returns (peak_mean, integ_norm_mean) or (None, None).

    Layout (journey_persistence_worker._pack_journey_metadata):
      [0:d] peak  [d:2d] path  [2d:3d] integ_norm  [3d:4d] flips  [4d:] pol/coh
    """
    d = _PART_DIMS.get(part)
    if not d or not blob or len(blob) < 3 * d:
        return None, None
    peak = [b / 255.0 for b in blob[0:d]]
    integ = [b / 255.0 for b in blob[2 * d:3 * d]]
    return (sum(peak) / d, sum(integ) / d)


def _fetch_local(db_path: str, titan_id: str):
    c = sqlite3.connect(db_path)
    c.row_factory = sqlite3.Row
    gifts = list(c.execute(
        "SELECT source_part, side, cycle_tick_count, gift_amplitude, journey_metadata "
        "FROM trinity_journey_gifts WHERE titan_id = ?", (titan_id,)))
    corr = list(c.execute(
        "SELECT source_part, side, duration_ticks, polarity_at_fire, dominant_dim_value, "
        "sigma_multiplier FROM trinity_corrective_events WHERE titan_id = ?", (titan_id,)))
    c.close()
    return gifts, corr


def audit_titan(db_path: str, titan_id: str, consumer_present: bool):
    gifts, corr = _fetch_local(db_path, titan_id)
    by_part = {}
    for r in gifts:
        key = (r["source_part"], r["side"])
        b = by_part.setdefault(key, {"ticks": [], "peak": [], "integ": []})
        tc = r["cycle_tick_count"]
        if tc and tc > 0:
            b["ticks"].append(float(tc))
        pk, ig = _decode_peak_integral(
            bytes(r["journey_metadata"]) if r["journey_metadata"] else b"", r["source_part"])
        if pk is not None:
            b["peak"].append(pk)
            b["integ"].append(ig)

    corr_by_part = {}
    for r in corr:
        key = (r["source_part"], r["side"])
        b = corr_by_part.setdefault(key, {"dur": [], "pol": []})
        if r["duration_ticks"] is not None and r["polarity_at_fire"] is not None:
            b["dur"].append(float(r["duration_ticks"]))
            b["pol"].append(float(r["polarity_at_fire"]))

    rows = []
    for key in sorted(set(by_part) | set(corr_by_part)):
        part, side = key
        g = by_part.get(key, {"ticks": [], "peak": [], "integ": []})
        cb = corr_by_part.get(key, {"dur": [], "pol": []})
        ticks = sorted(g["ticks"])
        n = len(ticks)
        cv = _cv(ticks)
        p50 = _pct(ticks, 0.50)
        p95 = _pct(ticks, 0.95)
        tail = (p95 / p50) if p50 > 0 else 0.0
        pear = _pearson(cb["dur"], cb["pol"])
        rows.append({
            "part": part, "side": side, "n_cycles": n,
            "tick_mean": (sum(ticks) / n) if n else 0.0,
            "tick_cv": cv, "tick_p50": p50, "tick_p95": p95, "tick_max": (ticks[-1] if ticks else 0.0),
            "tail_ratio": tail,
            "peak_mean": (sum(g["peak"]) / len(g["peak"])) if g["peak"] else None,
            "integ_shape_mean": (sum(g["integ"]) / len(g["integ"])) if g["integ"] else None,
            "n_corrective": len(cb["dur"]),
            "dur_mean": (sum(cb["dur"]) / len(cb["dur"])) if cb["dur"] else None,
            "pearson_dur_pol": pear,
        })

    # ── fleet-level gate evaluation ──
    rich_rows = [r for r in rows if r["n_cycles"] >= 30]
    g1_rich = any(r["tick_cv"] >= CV_RICH and r["tail_ratio"] >= TAIL_RICH for r in rich_rows)
    pears = [abs(r["pearson_dur_pol"]) for r in rows if r["pearson_dur_pol"] is not None]
    # INDEPENDENCE passes if duration is NOT strongly correlated with polarity
    g2_independent = bool(pears) and (max(pears) < PEARSON_REDUNDANT)
    g2_evidence = max(pears) if pears else None
    g3_consumer = consumer_present

    if g1_rich and g2_independent and g3_consumer:
        verdict = "NEED"
        reason = "all three gates pass — the integral is variable, independent of polarity, and has a consumer."
    else:
        fails = []
        if not g1_rich:
            fails.append("G1 RICHNESS (time-off-centre is near-constant — 5 observables already span it)")
        if not g2_independent:
            fails.append(f"G2 INDEPENDENCE (duration≈polarity, |r|={g2_evidence:.2f} — redundant with the polarity observable)"
                         if g2_evidence is not None else "G2 INDEPENDENCE (insufficient corrective data to prove independence)")
        if not g3_consumer:
            fails.append("G3 CONSUMER (no metabolic loop / URGE-to-rest consumer wired yet — premature, wire-now-gate-later)")
        verdict = "GATE"
        reason = "; ".join(fails)

    return {
        "titan_id": titan_id, "rows": rows,
        "gates": {"G1_richness": g1_rich, "G2_independence": g2_independent,
                  "G2_max_abs_pearson": g2_evidence, "G3_consumer": g3_consumer},
        "verdict": verdict, "reason": reason,
        "n_gift_cycles": len(gifts), "n_corrective": len(corr),
    }


def _print_titan(res):
    print(f"\n{'='*78}\nTRINITY TIME-OFF-CENTRE AUDIT — {res['titan_id']}  "
          f"({res['n_gift_cycles']} cycles, {res['n_corrective']} corrective)\n{'='*78}")
    hdr = (f"  {'part':10s} {'side':6s} {'cycles':>7s} {'tickμ':>7s} {'tickCV':>7s} "
           f"{'p50':>6s} {'p95':>7s} {'tail':>5s} {'peakμ':>6s} {'corr_n':>6s} {'r(dur,pol)':>11s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in res["rows"]:
        peak = f"{r['peak_mean']:.3f}" if r["peak_mean"] is not None else "  -  "
        pear = f"{r['pearson_dur_pol']:+.3f}" if r["pearson_dur_pol"] is not None else "    -  "
        print(f"  {r['part']:10s} {r['side']:6s} {r['n_cycles']:>7d} {r['tick_mean']:>7.0f} "
              f"{r['tick_cv']:>7.2f} {r['tick_p50']:>6.0f} {r['tick_p95']:>7.0f} "
              f"{r['tail_ratio']:>5.1f} {peak:>6s} {r['n_corrective']:>6d} {pear:>11s}")
    g = res["gates"]
    print(f"\n  GATES:  G1 richness={'PASS' if g['G1_richness'] else 'fail'}  "
          f"G2 independence={'PASS' if g['G2_independence'] else 'fail'}"
          f"(max|r|={g['G2_max_abs_pearson']:.2f})" if g["G2_max_abs_pearson"] is not None else
          f"\n  GATES:  G1 richness={'PASS' if g['G1_richness'] else 'fail'}  G2 independence=n/a")
    print(f"          G3 consumer={'PASS' if g['G3_consumer'] else 'fail'}")
    mark = "🟢" if res["verdict"] == "NEED" else "🔴"
    print(f"\n  {mark} VERDICT: {res['verdict']} — {res['reason']}")


def _scan_consumer(repo_root: Path) -> bool:
    """Static scan: does a metabolic consumer of a continuous trinity time-off-
    centre / chronicity signal exist yet? Greps the metabolism + life-force code.
    """
    pats = ["time_off_cent", "off_centre", "off_center", "chronicity",
            "excursion_integral", "delta_dt", "time_off"]
    targets = list((repo_root / "titan_hcl").rglob("*metabol*.py")) + \
        list((repo_root / "titan_hcl").rglob("*life_force*.py"))
    for f in targets:
        try:
            txt = f.read_text(errors="ignore")
        except OSError:
            continue
        if any(p in txt for p in pats):
            return True
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="data/consciousness.db")
    ap.add_argument("--titan-id", default="T1")
    ap.add_argument("--fleet", action="store_true",
                    help="audit T1 (local) + T2/T3 (consciousness.db over scp from 10.135.0.6)")
    ap.add_argument("--remote-host", default="root@10.135.0.6")
    ap.add_argument("--consumer-present", action="store_true",
                    help="force G3=pass (use when the metabolic consumer has been wired)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    consumer = args.consumer_present or _scan_consumer(repo_root)
    if not args.consumer_present:
        print(f"[consumer scan] metabolic time-off-centre consumer wired: {consumer}")

    results = []
    if args.fleet:
        results.append(audit_titan(args.db, "T1", consumer))
        for tid, rdb in (("T2", "/root/titan2/data/consciousness.db"),
                         ("T3", "/root/titan3/data/consciousness.db")):
            try:
                tmp = tempfile.NamedTemporaryFile(suffix=f"_{tid}.db", delete=False).name
                subprocess.run(["scp", "-q", f"{args.remote_host}:{rdb}", tmp],
                               check=True, timeout=120)
                results.append(audit_titan(tmp, tid, consumer))
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                print(f"  [{tid}] fetch failed: {e}")
    else:
        results.append(audit_titan(args.db, args.titan_id, consumer))

    for res in results:
        _print_titan(res)

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.json}")

    # Exit 0 always (observation tool); the verdict is in the report.
    return 0


if __name__ == "__main__":
    sys.exit(main())
