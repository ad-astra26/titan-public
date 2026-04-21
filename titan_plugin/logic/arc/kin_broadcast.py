"""
titan_plugin/logic/arc/kin_broadcast.py — cross-Titan ARC goal broadcast.

rFP_arc_training_fix Step C (2026-04-20): when one Titan captures its first WIN
on a game, the resulting goal grid is broadcast to kin Titans via HTTP POST.
Kin receivers write the goal to their own `goal_grids.json` so the G1 dense
reward activates everywhere. One Titan's win bootstraps G1 for all three on
that game.

Design:
  - Best-effort. All HTTP failures swallowed — kin might be restarting or
    temporarily unreachable. Caller continues normally.
  - No retries inside the broadcast call (tight timeout). The next WIN on the
    same game will overwrite anyway; we do not build a queue.
  - Self-skip: broadcaster POSTs to all 3 kin URLs; receiver compares
    `source_titan_id` to its own and drops if match.
  - Kin URLs are hardcoded per VPC network (memory/reference_vpc_network.md):
      T1 = http://10.135.0.3:7777
      T2 = http://10.135.0.6:7777
      T3 = http://10.135.0.6:7778
    If this list changes, update this constant.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
import requests

logger = logging.getLogger(__name__)

KIN_URLS = [
    "http://10.135.0.3:7777",   # T1
    "http://10.135.0.6:7777",   # T2
    "http://10.135.0.6:7778",   # T3
]

BROADCAST_TIMEOUT_SECONDS = 5.0


def broadcast_goal(
    game_id: str,
    grid: np.ndarray,
    source_titan_id: str,
    captured_at_utc: Optional[str] = None,
) -> dict:
    """POST goal grid to all kin Titans. Self-receive is filtered on the
    receiver side (by source_titan_id match). Returns a per-URL result dict."""
    payload = {
        "game_id": game_id,
        "grid": grid.astype(int).tolist(),
        "shape": list(grid.shape),
        "source_titan_id": source_titan_id,
        "captured_at_utc": captured_at_utc or time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    results = {}
    for url in KIN_URLS:
        try:
            r = requests.post(
                f"{url}/v4/arc/goal-ingest",
                json=payload,
                timeout=BROADCAST_TIMEOUT_SECONDS,
            )
            results[url] = {"status": r.status_code, "ok": r.status_code == 200}
        except Exception as e:
            results[url] = {"status": "error", "error": type(e).__name__}
    accepted = sum(1 for r in results.values() if r.get("ok"))
    logger.info(
        "[KinBroadcast] goal(%s) broadcast from %s: %d/%d kin accepted",
        game_id, source_titan_id, accepted, len(KIN_URLS),
    )
    return results
