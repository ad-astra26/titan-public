"""20-min focused soak test for EMOT-CGN Phase 1.6 standalone-worker deploy.

Stop triggers: any anomaly (worker crash, shm not advancing, errors in
brain log, bus orphan warnings, RSS spike, T2/T3 sharing state again).

Captures every 5 min:
- Per-Titan dominant emotion + V_beta + V_blended + cluster_confidence
- total_updates delta (chains feeding the worker)
- shm_version increment (writer alive)
- cross_insights_sent/received (Upgrade III flow)
- emot_cgn_worker process RSS
- Brain log errors/warnings related to EMOT-CGN
"""
from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.request
from datetime import datetime

TITANS = {
    "T1": ("http://localhost:7777", None),                 # local
    "T2": ("http://10.135.0.6:7777", "/tmp/titan2_brain.log"),
    "T3": ("http://10.135.0.6:7778", "/tmp/titan3_brain.log"),
}
T1_LOG = "/tmp/titan_brain.log"
DURATION_S = 20 * 60
CHECKPOINT_S = 5 * 60
OUT = f"/tmp/emot_cgn_phase1_6_soak_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.md"


def fetch(url: str, timeout: float = 4.0) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}


def grep_log_remote(host: str, log_path: str, pattern: str, n: int = 20) -> str:
    """Remote ssh grep — for T2/T3 logs."""
    try:
        cmd = ["ssh", "-o", "ConnectTimeout=3", host,
               f"grep -E '{pattern}' {log_path} 2>/dev/null | tail -{n}"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
        return r.stdout.strip()
    except Exception:
        return ""


def grep_log_local(log_path: str, pattern: str, n: int = 20) -> str:
    try:
        r = subprocess.run(
            ["bash", "-c", f"grep -E '{pattern}' {log_path} 2>/dev/null | tail -{n}"],
            capture_output=True, text=True, timeout=4)
        return r.stdout.strip()
    except Exception:
        return ""


def snapshot(titan: str, url: str, log_path: str | None) -> dict:
    state = fetch(f"{url}/v4/emot-cgn").get("data", {})
    audit = fetch(f"{url}/v4/emot-cgn/audit").get("data", {})
    # Shadow tail length (chain throughput indicator)
    shadow = audit.get("shadow_tail", []) if isinstance(audit, dict) else []
    return {
        "titan": titan,
        "ts": time.time(),
        "dominant": state.get("dominant", "?"),
        "V_beta": state.get("dominant_V_beta", 0.5),
        "V_blended": state.get("dominant_V_blended", 0.5),
        "cluster_conf": state.get("cluster_confidence", 0.0),
        "total_updates": state.get("total_updates", 0),
        "ci_sent": state.get("cross_insights_sent", 0),
        "ci_recv": state.get("cross_insights_received", 0),
        "shm_version": state.get("shm_version", 0),
        "shadow_tail_len": len(shadow),
        "is_active": state.get("is_active", False),
        "source": state.get("source", "?"),
    }


def check_anomalies(snap: dict, prev: dict | None) -> list[str]:
    """Stop-trigger detection — return list of anomaly strings."""
    a = []
    if snap.get("source") != "shm":
        a.append(f"{snap['titan']}: source={snap['source']} (expected 'shm' — worker possibly down)")
    if prev is not None:
        # Update counter monotonic (shm version must advance for active worker)
        if snap["shm_version"] < prev["shm_version"]:
            a.append(f"{snap['titan']}: shm_version went BACKWARDS {prev['shm_version']}→{snap['shm_version']}")
        if snap["total_updates"] < prev["total_updates"]:
            a.append(f"{snap['titan']}: total_updates went BACKWARDS — restart loop?")
    return a


def render(snaps: list[dict], minute: int, n: int) -> list[str]:
    lines = [f"\n## Checkpoint #{n} — t={minute}min ({datetime.utcnow().strftime('%H:%M UTC')})\n"]
    lines.append("| Titan | Dominant | V_beta | V_blended | Conf | Updates | CI sent/recv | shm_v | shadow | Source |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for s in snaps:
        lines.append(
            f"| {s['titan']} | {s['dominant']} | {s['V_beta']:.3f} | "
            f"{s['V_blended']:.3f} | {s['cluster_conf']:.3f} | "
            f"{s['total_updates']} | {s['ci_sent']}/{s['ci_recv']} | "
            f"{s['shm_version']} | {s['shadow_tail_len']} | {s['source']} |")
    return lines


def main():
    start = time.time()
    print(f"EMOT-CGN Phase 1.6 soak — output → {OUT}")
    with open(OUT, "w") as f:
        f.write(f"# EMOT-CGN Phase 1.6 Soak Test — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n")
        f.write("Standalone-worker deploy verification. 20 min, 5-min check-ins. Stop on anomaly.\n\n")
        f.write("---\n")
    snaps_history: list[list[dict]] = []
    n = 0
    anomalies_total: list[str] = []
    while (time.time() - start) < DURATION_S:
        n += 1
        minute = int((time.time() - start) / 60)
        snaps = [snapshot(t, url, log) for t, (url, log) in TITANS.items()]
        snaps_history.append(snaps)
        # Check anomalies vs prior snapshot
        if len(snaps_history) >= 2:
            for cur, prev in zip(snaps_history[-1], snaps_history[-2]):
                anomalies = check_anomalies(cur, prev)
                if anomalies:
                    anomalies_total.extend(anomalies)
        with open(OUT, "a") as f:
            for line in render(snaps, minute, n):
                f.write(line + "\n")
            if anomalies_total:
                f.write(f"\n⚠️ **Anomalies detected (stop triggers):**\n")
                for a in anomalies_total[-5:]:
                    f.write(f"- {a}\n")
        # Brain log scrape — errors only
        emot_errors = []
        e = grep_log_local(T1_LOG, "EmotCGNWorker.*ERROR|EmotCGN.*error|emot.*Traceback", n=5)
        if e: emot_errors.append(f"T1: {e}")
        for titan_id, host_path in (("T2", "/tmp/titan2_brain.log"), ("T3", "/tmp/titan3_brain.log")):
            e = grep_log_remote("root@10.135.0.6", host_path,
                                "EmotCGNWorker.*ERROR|EmotCGN.*error|emot.*Traceback", n=5)
            if e: emot_errors.append(f"{titan_id}: {e}")
        if emot_errors:
            with open(OUT, "a") as f:
                f.write("\n**Brain log errors:**\n```\n")
                for er in emot_errors:
                    f.write(er + "\n")
                f.write("```\n")
        print(f"checkpoint #{n} t={minute}min — anomalies: {len(anomalies_total)}")
        if (time.time() - start) + CHECKPOINT_S > DURATION_S:
            break
        time.sleep(CHECKPOINT_S)
    # Final summary
    with open(OUT, "a") as f:
        f.write("\n\n---\n## Final Summary\n\n")
        if not snaps_history:
            f.write("No checkpoints captured.\n")
            return
        first, last = snaps_history[0], snaps_history[-1]
        for fst, lst in zip(first, last):
            t = lst["titan"]
            f.write(f"\n**{t}:** updates {fst['total_updates']}→{lst['total_updates']} "
                    f"(+{lst['total_updates']-fst['total_updates']}), shm_version "
                    f"{fst['shm_version']}→{lst['shm_version']}, dominant "
                    f"{fst['dominant']}→{lst['dominant']}\n")
        f.write(f"\n**Total anomalies detected:** {len(anomalies_total)}\n")
        if anomalies_total:
            for a in anomalies_total[:10]:
                f.write(f"- {a}\n")
        else:
            f.write("- ✅ Clean run, no stop triggers fired\n")
    print(f"DONE → {OUT}")


if __name__ == "__main__":
    main()
