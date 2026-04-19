#!/usr/bin/env python3
"""verify_ns_fix.py — NS program NN recovery verification harness.

Polls /v4/ns-health on T1/T2/T3 to verify the residual-learning target
formula fix (commit 4fac1f2, 2026-04-19) is pulling NN urgency outputs
off the sigmoid(-10) = 4.5e-05 saturation attractor.

Usage:
    python3 scripts/verify_ns_fix.py                 # one-shot snapshot
    python3 scripts/verify_ns_fix.py --loop 30 5     # 30 min, 5 min interval
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

DEAD_URGENCY = 4.5e-05  # sigmoid(-10) clamp floor — the collapse attractor


def fetch(url: str, timeout: int = 20) -> dict:
    try:
        r = urllib.request.urlopen(url, timeout=timeout)
        return json.loads(r.read())
    except Exception as e:
        return {"_error": str(e)}


def snapshot_titan(label: str, base_url: str) -> dict:
    hp = fetch(f"{base_url}/v4/ns-health")
    hp_d = hp.get("data", {})
    training = hp_d.get("training", {})
    progs = hp_d.get("programs", {})
    obs = hp_d.get("neuromod_reward_observer", {})
    # Aggregate per-program stats
    prog_stats = {}
    any_nonzero = 0
    total_avg_u = 0.0
    max_u_seen = 0.0
    for pid, p in progs.items():
        au = float(p.get("avg_urgency", 0) or 0)
        mu = float(p.get("max_urgency", 0) or 0)
        nzu = float(p.get("pct_nonzero_urgency", 0) or 0)
        fp = float(p.get("fire_pct", 0) or 0)
        nzr = float(p.get("pct_nonzero_reward", 0) or 0)
        loss = float(p.get("last_loss", 0) or 0)
        updates = int(p.get("total_updates", 0) or 0)
        verdict = p.get("verdict", "?")
        prog_stats[pid] = {
            "avg_u": au, "max_u": mu, "nz_u": nzu, "fire%": fp,
            "nz_r": nzr, "loss": loss, "updates": updates, "verdict": verdict,
        }
        total_avg_u += au
        max_u_seen = max(max_u_seen, mu)
        if nzu > 5.0:
            any_nonzero += 1
    return {
        "label": label,
        "ts": time.time(),
        "error": hp.get("_error"),
        "overall": hp_d.get("overall"),
        "counts": hp_d.get("overall_counts", {}),
        "training": {
            "phase": training.get("phase"),
            "sup_w": training.get("supervision_weight"),
            "transitions": training.get("total_transitions"),
            "steps": training.get("total_train_steps"),
        },
        "avg_u_mean": total_avg_u / max(1, len(progs)),
        "max_u_any": max_u_seen,
        "programs_above_5pct_nz": any_nonzero,
        "programs_total": len(progs),
        "observer": {
            "emissions": obs.get("emissions_total"),
            "ticks": obs.get("tick_count"),
        },
        "per_program": prog_stats,
    }


def print_snapshot(snaps: list) -> None:
    print("=" * 110)
    print(f"TIMESTAMP  {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    print("=" * 110)
    hdr = (f"{'':5s} {'overall':>10s} {'OK/LOW/DEAD':>13s} {'phase':>12s} "
           f"{'avg_u_mean':>11s} {'max_u_any':>11s} "
           f"{'nz>5%progs':>11s} {'emissions':>10s}")
    print(hdr)
    for s in snaps:
        if s.get("error"):
            print(f"{s['label']:5s} ERROR: {s['error']}")
            continue
        cnt = s.get("counts", {})
        cnt_str = f"{cnt.get('ok',0)}/{cnt.get('low',0)}/{cnt.get('dead',0)}"
        avg_u = s["avg_u_mean"]
        max_u = s["max_u_any"]
        # Highlight if avg_u has moved off the 4.5e-05 attractor
        off_attractor = avg_u > DEAD_URGENCY * 2
        avg_u_str = f"{avg_u:>11.2e}" + ("*" if off_attractor else " ")
        max_u_str = f"{max_u:>11.2e}"
        print(
            f"{s['label']:5s} "
            f"{s.get('overall','?'):>10s} "
            f"{cnt_str:>13s} "
            f"{str(s['training'].get('phase','?')):>12s} "
            f"{avg_u_str:<11s} "
            f"{max_u_str:>11s} "
            f"{s['programs_above_5pct_nz']:>4d}/{s['programs_total']:<5d} "
            f"{s['observer'].get('emissions',0):>10d}"
        )


def acceptance_check(snaps: list) -> None:
    """Post-fix acceptance criteria."""
    print()
    print("─" * 110)
    print("NS FIX ACCEPTANCE CRITERIA (post-commit 4fac1f2)")
    print("─" * 110)
    for s in snaps:
        if s.get("error"):
            continue
        label = s["label"]
        avg_u = s["avg_u_mean"]
        max_u = s["max_u_any"]
        nz_progs = s["programs_above_5pct_nz"]
        print(f"\n{label}:")
        # Primary: avg_u off the sigmoid(-10) attractor
        print(f"  [{'✅' if avg_u > DEAD_URGENCY * 2 else '⏳'}] avg_u off "
              f"saturation (> {DEAD_URGENCY*2:.1e}): {avg_u:.2e}")
        # Secondary: max_u any program risen
        print(f"  [{'✅' if max_u > 0.01 else '⏳'}] max_u > 0.01 (any "
              f"program showing real output): {max_u:.4f}")
        # Tertiary: at least 1 program healthy (nz > 5%)
        print(f"  [{'✅' if nz_progs > 0 else '⏳'}] ≥1 program nz_urgency>5% "
              f"(escape verdict=DEAD): {nz_progs}/{s['programs_total']}")
        # Full recovery
        print(f"  [{'✅' if nz_progs == s['programs_total'] else '⏳'}] "
              f"ALL programs healthy (nz>5%): {nz_progs}/{s['programs_total']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", nargs=2, type=int, metavar=("MINUTES", "INTERVAL"),
                    help="Run for MINUTES with snapshots every INTERVAL minutes")
    args = ap.parse_args()

    def do_snapshot():
        snaps = [snapshot_titan(label, url) for label, url in TITANS]
        print_snapshot(snaps)
        acceptance_check(snaps)
        return snaps

    if args.loop:
        minutes, interval = args.loop
        deadline = time.time() + minutes * 60
        i = 0
        all_snaps = []
        t0 = time.time()
        while time.time() < deadline:
            i += 1
            print(f"\n▬▬▬ NS CHECK-IN #{i} (t+{int((time.time() - t0) / 60)} min) ▬▬▬")
            snaps = do_snapshot()
            all_snaps.append({"ts": time.time(), "snapshots": snaps})
            sys.stdout.flush()
            if time.time() + interval * 60 < deadline:
                time.sleep(interval * 60)
        if all_snaps:
            final_path = f"/tmp/ns_fix_verify_{int(time.time())}.json"
            with open(final_path, "w") as f:
                json.dump({"run_start": all_snaps[0]["ts"],
                           "run_end": all_snaps[-1]["ts"],
                           "checkins": all_snaps}, f, indent=2)
            print(f"\n✓ Full log saved: {final_path}")
    else:
        do_snapshot()


if __name__ == "__main__":
    main()
