#!/usr/bin/env python3
"""verify_chain_iql_v2.py — chain_iql v2 + COMPLETE-5 verification harness.

Samples meta-reasoning + chain_iql + META-CGN state across T1/T2/T3 and
measures the acceptance criteria from rFP_cgn_keystone_completion.md.

Usage:
    python3 scripts/verify_chain_iql_v2.py                 # one-shot snapshot
    python3 scripts/verify_chain_iql_v2.py --loop 60 5     # 60 min, 5 min interval
    python3 scripts/verify_chain_iql_v2.py --baseline save.json  # save baseline
    python3 scripts/verify_chain_iql_v2.py --baseline save.json --delta  # delta vs baseline
"""

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


def fetch(url: str, timeout: int = 20) -> dict:
    try:
        r = urllib.request.urlopen(url, timeout=timeout)
        return json.loads(r.read())
    except Exception as e:
        return {"_error": str(e)}


def snapshot_titan(label: str, base_url: str) -> dict:
    mr = fetch(f"{base_url}/v4/meta-reasoning")
    audit = fetch(f"{base_url}/v4/meta-reasoning/audit")
    mr_d = mr.get("data", {})
    au_d = audit.get("data", {})
    ci = mr_d.get("chain_iql", {})
    mc = mr_d.get("meta_cgn", {})
    bw = mc.get("blend_weights_preview", {})
    mono = au_d.get("monoculture", {})
    sss = au_d.get("subsystem_signals_status", {})
    rpp = au_d.get("rewards_per_primitive", {})
    # Compute per-primitive reward spread
    reward_totals = [float(rpp[p].get("avg_total", 0)) for p in rpp]
    spread = (max(reward_totals) - min(reward_totals)) if len(reward_totals) >= 2 else 0.0
    return {
        "label": label,
        "ts": time.time(),
        "error": mr.get("_error") or audit.get("_error"),
        "total_chains": mr_d.get("total_chains"),
        "is_active": mr_d.get("is_active"),
        "chain_iql": {
            "tcount": ci.get("template_count"),
            "tmax": ci.get("template_max"),
            "ucb_c": ci.get("ucb_c"),
            "visit_range": ci.get("visit_range"),
            "visit_sum": ci.get("visit_sum"),
            "lru_evictions": ci.get("lru_evictions"),
            "ext_reward_drops": ci.get("external_reward_late_drops"),
        },
        "meta_cgn": {
            "status": mc.get("status"),
            "grad_status": (mc.get("graduation") or {}).get("status"),
            "grad_progress": (mc.get("graduation") or {}).get("progress"),
            "rolled_back_count": (mc.get("graduation") or {}).get("rolled_back_count"),
            "beta_dispersion_ema": bw.get("beta_dispersion_ema"),
            "w_legacy": bw.get("w_legacy"),
            "w_compound": bw.get("w_compound"),
            "w_grounded": bw.get("w_grounded"),
            "shadow_disagreement_rate": (mc.get("shadow_quality") or {}).get("disagreement_rate"),
        },
        "monoculture": {
            "dominant": mono.get("dominant_primitive"),
            "share_500": mono.get("dominant_share_500"),
        },
        "reward_spread": round(spread, 4),
        "reward_per_primitive": {
            p: round(float(rpp[p].get("avg_total", 0)), 4) for p in rpp
        },
        "signals_live": sss.get("live_count"),
        "signals_total": sss.get("total_signals"),
        "cache_age_s": sss.get("cache_age_seconds"),
    }


def print_snapshot(snaps: list) -> None:
    print("=" * 100)
    print(f"TIMESTAMP  {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    print("=" * 100)
    hdr = f"{'':5s} {'chains':>8s} {'tcount':>6s} {'evic':>5s} {'visits':>11s} {'mono':>20s} {'β_disp':>8s} {'w_grd':>6s} {'MCGN':>12s} {'spread':>7s}"
    print(hdr)
    for s in snaps:
        if s.get("error"):
            print(f"{s['label']:5s} ERROR: {s['error']}")
            continue
        ci = s["chain_iql"]
        mc = s["meta_cgn"]
        mn = s["monoculture"]
        vmin, vmax = (ci.get("visit_range") or [0, 0])[:2] or (0, 0)
        mono_str = f"{mn.get('dominant','?')}:{(mn.get('share_500') or 0)*100:.1f}%"
        print(
            f"{s['label']:5s} "
            f"{(s.get('total_chains') or 0):>8d} "
            f"{(ci.get('tcount') or 0):>6d} "
            f"{(ci.get('lru_evictions') or 0):>5d} "
            f"{vmin:>5d}..{vmax:<4d} "
            f"{mono_str:>20s} "
            f"{(mc.get('beta_dispersion_ema') or 0):>8.4f} "
            f"{(mc.get('w_grounded') or 0):>6.3f} "
            f"{str(mc.get('status'))[:12]:>12s} "
            f"{s.get('reward_spread', 0):>7.3f}"
        )


def acceptance_check(snaps: list, baseline: dict | None) -> None:
    """Evaluate rFP_cgn_keystone_completion acceptance criteria."""
    print()
    print("─" * 100)
    print("ACCEPTANCE CRITERIA (rFP_cgn_keystone_completion.md §5 + rFP_chain_iql_v2.md §5)")
    print("─" * 100)
    for s in snaps:
        if s.get("error"):
            continue
        label = s["label"]
        ci = s["chain_iql"]
        mc = s["meta_cgn"]
        mn = s["monoculture"]
        tcount = ci.get("tcount") or 0
        lru_evic = ci.get("lru_evictions") or 0
        mono_share = (mn.get("share_500") or 0) * 100
        beta = mc.get("beta_dispersion_ema") or 0
        w_grd = mc.get("w_grounded") or 0
        disagreement = mc.get("shadow_disagreement_rate") or 0
        mcgn = mc.get("status", "?")
        rolled_back = mc.get("rolled_back_count") or 0
        spread = s.get("reward_spread", 0)
        # Per-criterion passes/fails
        def check(label: str, pass_cond: bool, detail: str) -> str:
            return f"  [{'✅' if pass_cond else '⏳'}] {label}: {detail}"
        print(f"\n{label}:")
        print(check("tcount grew past 50", tcount > 50, f"{tcount}/500"))
        print(check("lru_evictions starting", lru_evic > 0, f"{lru_evic} (need reg→500 first)"))
        print(check("monoculture < 60%", mono_share < 60, f"{mono_share:.1f}%"))
        print(check("β_dispersion ≥ 0.015 (gate opens)", beta >= 0.015, f"{beta:.4f}"))
        print(check("w_grounded ≥ 0.20", w_grd >= 0.20, f"{w_grd:.3f}"))
        print(check("reward_spread ≥ 0.10", spread >= 0.10, f"{spread:.3f}"))
        if mcgn == "active":
            print(check("META-CGN active (no new rollback)", rolled_back <= 1,
                        f"rolled_back={rolled_back}"))
        else:
            print(check("META-CGN active", False, f"status={mcgn} rolled_back={rolled_back}"))
        # Delta from baseline
        if baseline:
            base_s = next((b for b in baseline.get("snapshots", []) if b["label"] == label), None)
            if base_s:
                d_tcount = tcount - (base_s["chain_iql"].get("tcount") or 0)
                d_mono = mono_share - (base_s["monoculture"].get("share_500") or 0) * 100
                d_spread = spread - base_s.get("reward_spread", 0)
                print(f"  [Δ baseline] tcount +{d_tcount} | mono {d_mono:+.1f}pp | spread {d_spread:+.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", nargs=2, type=int, metavar=("MINUTES", "INTERVAL"),
                    help="Run for MINUTES with snapshots every INTERVAL minutes")
    ap.add_argument("--baseline", type=str, help="Path to baseline JSON")
    ap.add_argument("--save-baseline", type=str, help="Save current as baseline")
    ap.add_argument("--delta", action="store_true", help="Print delta vs baseline")
    args = ap.parse_args()

    baseline = None
    if args.baseline and Path(args.baseline).exists():
        with open(args.baseline) as f:
            baseline = json.load(f)

    def do_snapshot():
        snaps = [snapshot_titan(label, url) for label, url in TITANS]
        print_snapshot(snaps)
        acceptance_check(snaps, baseline)
        return snaps

    if args.loop:
        minutes, interval = args.loop
        deadline = time.time() + minutes * 60
        i = 0
        all_snaps = []
        while time.time() < deadline:
            i += 1
            print(f"\n▬▬▬ CHECK-IN #{i} (t+{int((time.time() - (deadline - minutes * 60)) / 60)} min) ▬▬▬")
            snaps = do_snapshot()
            all_snaps.append({"ts": time.time(), "snapshots": snaps})
            sys.stdout.flush()
            if time.time() + interval * 60 < deadline:
                time.sleep(interval * 60)
        # Final report
        if all_snaps:
            final_path = f"/tmp/chain_iql_v2_verify_{int(time.time())}.json"
            with open(final_path, "w") as f:
                json.dump({"run_start": all_snaps[0]["ts"],
                           "run_end": all_snaps[-1]["ts"],
                           "checkins": all_snaps}, f, indent=2)
            print(f"\n✓ Full log saved: {final_path}")
    else:
        snaps = do_snapshot()
        if args.save_baseline:
            with open(args.save_baseline, "w") as f:
                json.dump({"ts": time.time(), "snapshots": snaps}, f, indent=2)
            print(f"\n✓ Baseline saved: {args.save_baseline}")


if __name__ == "__main__":
    main()
