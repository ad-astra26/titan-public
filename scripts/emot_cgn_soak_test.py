"""30-minute EMOT-CGN shadow-mode soak test.

Captures state snapshots from T1/T2/T3 every 5 minutes + final report.
Focus: verify the wire end-to-end — clusters being assigned, primitives
getting V updates, HAOV observations accumulating, no errors in brain log.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from datetime import datetime

TITANS = {
    "T1": "http://localhost:7777",
    "T2": "http://10.135.0.6:7777",
    "T3": "http://10.135.0.6:7778",
}

CHECKPOINT_INTERVAL_S = 300  # 5 minutes
DURATION_S = 50 * 60          # 50 minutes (extended from 30)
OUT_PATH = f"/tmp/emot_cgn_soak_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.md"


def fetch(url: str, timeout: float = 5.0) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}


def snapshot(titan: str, base_url: str) -> dict:
    emot = fetch(f"{base_url}/v4/emot-cgn").get("data", {})
    readiness = fetch(f"{base_url}/v4/emot-cgn/graduation-readiness").get("data", {})
    audit = fetch(f"{base_url}/v4/emot-cgn/audit").get("data", {})
    return {
        "titan": titan,
        "ts": time.time(),
        "emot": emot,
        "readiness": readiness,
        "audit_watchdog": audit.get("watchdog", {}),
        "audit_shadow_tail_n": len(audit.get("shadow_tail", []) or []),
        "audit_haov_by_status": _count_haov_by_status(audit.get("haov", {})),
    }


def _count_haov_by_status(haov: dict) -> dict:
    hyps = (haov or {}).get("hypotheses", {}) or {}
    out = {"nascent": 0, "testing": 0, "confirmed": 0, "falsified": 0}
    for h in hyps.values():
        s = h.get("status", "nascent")
        out[s] = out.get(s, 0) + 1
    return out


def render_snapshot(snaps: list, checkpoint: int, minute: int) -> list[str]:
    lines = [f"\n## Checkpoint #{checkpoint} — t={minute}min ({datetime.utcnow().strftime('%H:%M UTC')})"]
    lines.append("")
    lines.append("| Titan | Status | Updates | Observations | Dominant | Confirmed HAOV | Rolled back |")
    lines.append("|---|---|---|---|---|---|---|")
    for s in snaps:
        e = s["emot"]
        hs = s["audit_haov_by_status"]
        # Dominant emotion + its V
        prims = e.get("primitives", {}) or {}
        clusters = e.get("clusters", {}) or {}
        recent = e.get("recent_assignments") or []
        dominant = recent[-1] if recent else "—"
        dom_V = prims.get(dominant, {}).get("V", "—") if dominant != "—" else "—"
        lines.append(
            f"| {s['titan']} | {e.get('status','?')} | "
            f"{e.get('total_updates',0)} | {e.get('total_observations',0)} | "
            f"{dominant} (V={dom_V}) | {hs.get('confirmed',0)} | "
            f"{e.get('rolled_back_count',0)} |")
    # Primitive V table for each Titan
    lines.append("")
    for s in snaps:
        e = s["emot"]
        prims = e.get("primitives", {}) or {}
        if not prims:
            continue
        lines.append(f"\n**{s['titan']} primitive V(s):**")
        header = "| Primitive | V | Conf | n |\n|---|---|---|---|"
        lines.append(header)
        for p_id, p in prims.items():
            lines.append(f"| {p_id} | {p.get('V',0)} | {p.get('confidence',0)} | {p.get('n_samples',0)} |")
    return lines


def main():
    start = time.time()
    checkpoints = []
    print(f"EMOT-CGN 30-min soak test starting — output → {OUT_PATH}")
    with open(OUT_PATH, "w") as f:
        f.write(f"# EMOT-CGN 30-min Soak Test — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n")
        f.write("Purpose: verify EMOT-CGN shadow-mode wire end-to-end on T1+T2+T3.\n\n")
        f.write("Checkpoints every 5 min. Looking for: primitive V accumulation, "
                "cluster assignments, HAOV observation growth, no errors.\n\n")
        f.write("---\n")
    checkpoint = 0
    while (time.time() - start) < DURATION_S + 5:
        checkpoint += 1
        minute = int((time.time() - start) / 60)
        snaps = [snapshot(titan, url) for titan, url in TITANS.items()]
        checkpoints.append(snaps)
        rendered = render_snapshot(snaps, checkpoint, minute)
        with open(OUT_PATH, "a") as f:
            for line in rendered:
                f.write(line + "\n")
        print(f"checkpoint #{checkpoint} at t={minute}min written")
        # Sleep until next checkpoint
        remaining = DURATION_S - (time.time() - start)
        if remaining <= 10:
            break
        time.sleep(min(CHECKPOINT_INTERVAL_S, remaining))
    # Final summary
    with open(OUT_PATH, "a") as f:
        f.write("\n\n---\n## Final Summary\n")
        final = checkpoints[-1]
        initial = checkpoints[0]
        for i, (init, fin) in enumerate(zip(initial, final)):
            titan = init["titan"]
            init_updates = init["emot"].get("total_updates", 0)
            fin_updates = fin["emot"].get("total_updates", 0)
            init_obs = init["emot"].get("total_observations", 0)
            fin_obs = fin["emot"].get("total_observations", 0)
            f.write(f"\n**{titan}:** updates {init_updates}→{fin_updates} (+{fin_updates-init_updates}), "
                    f"observations {init_obs}→{fin_obs} (+{fin_obs-init_obs})\n")
    print(f"Soak test complete — report at {OUT_PATH}")


if __name__ == "__main__":
    main()
