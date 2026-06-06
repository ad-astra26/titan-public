"""Rolling sovereignty read-out (RFP_synthesis_decision_authority P3).

The ONE sovereignty score `S = 0.7·E + 0.3·V` is recorded per reply by the
synthesis `SovereigntyRatioMeter` and exported (rolling, with its E/V
components) to the observation-only metrics snapshot
(`data/synthesis_metrics_snapshot.json`, INV-Syn-25). The parent-side consumers
— the soul-Chronicle re-source, the meditation on-chain ZK-vault anchor, and the
backup persistence — read the rolling S from that snapshot: a cross-process FILE
read, never a recompute or an RPC (G18). This module is the ONE place that read
+ the basis-points conversion live, so the three consumers can never drift on
the metric (the exact failure mode the RFP's "4 disagreeing sovereignty scores"
came from).
"""
from __future__ import annotations

import json
import os

__all__ = [
    "read_rolling_sovereignty",
    "rolling_sovereignty_bp",
    "SNAPSHOT_REL_PATH",
    "SOVEREIGNTY_BP_SCALE",
]

# Snapshot location, relative to the process CWD (repo root for every worker).
SNAPSHOT_REL_PATH = os.path.join("data", "synthesis_metrics_snapshot.json")

# On-chain scale: S ∈ [0,1] → 0..10000 basis points (the ZK-vault commit takes a
# bounded int bp; this is the SAME wire the legacy composite used — only the
# value SOURCE changes, so the instruction bytes are layout-identical).
SOVEREIGNTY_BP_SCALE = 10000


def read_rolling_sovereignty(data_dir: str = ".") -> dict:
    """Return `{window, replies, s, e, v, trend}` from the synthesis metrics
    snapshot. Prefers the 7d window (a stable "who I've been lately"), falls back
    to 24h then all (the first window with ≥1 reply). Returns a zeroed dict when
    the snapshot is missing/unreadable/empty. Never raises (observation-only)."""
    zero = {"window": "7d", "replies": 0, "s": 0.0, "e": 0.0, "v": 0.0,
            "trend": 0.0}
    path = os.path.join(data_dir, SNAPSHOT_REL_PATH)
    try:
        with open(path, "r", encoding="utf-8") as f:
            snap = json.load(f)
    except Exception:
        return zero
    windows = ((snap.get("sovereignty") or {}).get("windows") or {})
    for w in ("7d", "24h", "all"):
        win = windows.get(w)
        if isinstance(win, dict) and int(win.get("replies", 0) or 0) > 0:
            return {
                "window": w,
                "replies": int(win.get("replies", 0) or 0),
                "s": _f(win.get("sovereignty")),
                "e": _f(win.get("e")),
                "v": _f(win.get("v")),
                "trend": _f(win.get("trend")),
            }
    return zero


def rolling_sovereignty_bp(data_dir: str = ".") -> int:
    """The rolling S as on-chain basis points (0..10000) — the ONE value the
    meditation ZK-vault anchor + backup persistence commit. Clamped defensively
    (the chain instruction takes a bounded int)."""
    s = read_rolling_sovereignty(data_dir).get("s", 0.0)
    bp = int(round(_f(s) * SOVEREIGNTY_BP_SCALE))
    if bp < 0:
        return 0
    if bp > SOVEREIGNTY_BP_SCALE:
        return SOVEREIGNTY_BP_SCALE
    return bp


def _f(x) -> float:
    try:
        return float(x) if x is not None else 0.0
    except (TypeError, ValueError):
        return 0.0
