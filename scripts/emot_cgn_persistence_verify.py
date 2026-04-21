"""Brief persistence-verification soak post-deploy.

Verifies:
  1. Save-on-init files exist immediately (check at t=0)
  2. recent_rewards deque populates after save_cadence chains
  3. neuromod_ema populates (grows from {} to populated dict)
  4. dominant_emotion updates from "FLOW" default to actual state

Duration: 10 minutes, checkpoints every 2 min.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from datetime import datetime
from pathlib import Path

TITANS = {
    "T1": ("http://localhost:7777", Path("data/emot_cgn")),
    "T2": ("http://10.135.0.6:7777", None),  # remote — skip disk inspection
    "T3": ("http://10.135.0.6:7778", None),
}

DURATION_S = 10 * 60
CHECKPOINT_S = 120  # 2 min
OUT = f"/tmp/emot_cgn_persist_verify_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.md"


def fetch_json(url, timeout=4.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}


def read_watchdog(path):
    if not path or not (path / "watchdog_state.json").exists():
        return None
    try:
        return json.loads((path / "watchdog_state.json").read_text())
    except Exception:
        return None


def snapshot_titan(titan, url, disk_path):
    emot = fetch_json(f"{url}/v4/emot-cgn").get("data", {})
    wd = read_watchdog(disk_path)
    shadow = fetch_json(f"{url}/v4/emot-cgn/audit").get("data", {}).get("shadow_tail", [])
    return {
        "titan": titan,
        "updates": emot.get("total_updates", 0),
        "dominant": emot.get("recent_assignments", ["?"])[-1] if emot.get("recent_assignments") else "?",
        "shadow_tail_len": len(shadow),
        "latest_chain": shadow[-1].get("chain") if shadow else None,
        "wd_recent_rewards_len": len(wd.get("recent_rewards", [])) if wd else None,
        "wd_neuromod_ema_keys": list(wd.get("neuromod_ema", {}).keys()) if wd else None,
        "wd_dominant_emotion": wd.get("dominant_emotion") if wd else None,
    }


def main():
    start = time.time()
    print(f"EMOT-CGN persistence verify — writing to {OUT}")
    with open(OUT, "w") as f:
        f.write(f"# EMOT-CGN Post-Deploy Persistence Verification — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n")
        f.write("Checks: save-on-init, recent_rewards population, neuromod_ema growth, dominant_emotion update.\n\n")
    n = 0
    while (time.time() - start) < DURATION_S:
        n += 1
        minute = int((time.time() - start) / 60)
        snaps = [snapshot_titan(t, u, p) for t, (u, p) in TITANS.items()]
        with open(OUT, "a") as f:
            f.write(f"\n## Checkpoint #{n} — t={minute}min ({datetime.utcnow().strftime('%H:%M UTC')})\n\n")
            f.write("| Titan | Updates | Shadow | Dominant (live) | Dominant (disk) | recent_rewards on disk | neuromod_ema keys |\n")
            f.write("|---|---|---|---|---|---|---|\n")
            for s in snaps:
                f.write(
                    f"| {s['titan']} | {s['updates']} | "
                    f"{s['shadow_tail_len']} ({s['latest_chain']}) | "
                    f"{s['dominant']} | "
                    f"{s['wd_dominant_emotion'] or '(remote)'} | "
                    f"{s['wd_recent_rewards_len'] if s['wd_recent_rewards_len'] is not None else '(remote)'} | "
                    f"{s['wd_neuromod_ema_keys'] if s['wd_neuromod_ema_keys'] is not None else '(remote)'} |\n")
        print(f"  checkpoint #{n} at t={minute}min")
        if time.time() - start + CHECKPOINT_S > DURATION_S:
            break
        time.sleep(CHECKPOINT_S)
    print(f"done — {OUT}")


if __name__ == "__main__":
    main()
