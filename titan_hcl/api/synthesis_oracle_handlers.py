"""API handlers for `/v6/synthesis/oracles/*` + `/v6/synthesis/proofs/*` (P6.K).

Reads ``data/oracles_snapshot.json`` written by synthesis_worker
(``titan_hcl/synthesis/oracle_snapshot.py:OracleSnapshotExporter``).
The api process NEVER reads ``data/synthesis.duckdb`` directly —
INV-Syn-3 / G21 means synthesis_worker is the sole writer + DuckDB ≥1.5
holds an exclusive lock against it. The JSON snapshot is the
cross-process bridge.

Soft-fail contract (matching P4/P5 patterns): missing / unparseable /
stale snapshot → ``{"ok": true, "router": [], "snapshot":
"missing|stale|corrupt"}``. Frontend renders an empty state; no 500.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

from fastapi import Request

logger = logging.getLogger(__name__)


DEFAULT_SNAPSHOT_PATH = "data/oracles_snapshot.json"
SNAPSHOT_STALENESS_SECONDS = 600   # 10× the 60s tick


_SNAPSHOT_CACHE: dict[str, dict] = {}


def _resolve_snapshot_path() -> str:
    data_dir = os.environ.get("TITAN_DATA_DIR", "data")
    return os.path.join(data_dir, "oracles_snapshot.json")


def _load_snapshot(path: Optional[str] = None) -> Optional[dict]:
    if path is None:
        path = _resolve_snapshot_path()
    if not os.path.exists(path):
        return None
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    cached = _SNAPSHOT_CACHE.get(path)
    if cached is not None and cached.get("mtime") == mtime:
        return cached["data"]
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning(
            "[synthesis_oracle_handlers] snapshot parse failed (%s): %s", path, e,
        )
        return None
    if not isinstance(data, dict):
        return None
    _SNAPSHOT_CACHE[path] = {"mtime": mtime, "data": data}
    return data


def _load_snapshot_with_status(
    path: Optional[str] = None,
) -> tuple[Optional[dict], str]:
    if path is None:
        path = _resolve_snapshot_path()
    if not os.path.exists(path):
        return None, "missing"
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None, "missing"
    age = time.time() - mtime
    payload = _load_snapshot(path)
    if payload is None:
        return None, "corrupt"
    if age > SNAPSHOT_STALENESS_SECONDS:
        return payload, "stale"
    return payload, "ok"


# ─────────────────────────────────────────────────────────────────────────
# Handlers — 5 GET routes per PLAN §P6.K
# ─────────────────────────────────────────────────────────────────────────


async def get_synthesis_oracles_router(request: Request) -> dict:
    """List registered TruthOraclePlugs + cost class. Used by Maker
    Panel to surface "which oracles is this Titan capable of?".
    """
    snap, status = _load_snapshot_with_status()
    if snap is None:
        return {"ok": True, "router": [], "snapshot": status}
    return {
        "ok": True,
        "router": list(snap.get("router", []) or []),
        "snapshot": status,
        "exported_at": snap.get("exported_at"),
    }


async def get_synthesis_oracles_recent(request: Request) -> dict:
    """Last N OracleVerdict TXs (oldest→newest within the ring). Used by
    the Observatory to render a live verdict log."""
    snap, status = _load_snapshot_with_status()
    if snap is None:
        return {"ok": True, "verdicts": [], "snapshot": status}
    return {
        "ok": True,
        "verdicts": list(snap.get("recent_verdicts", []) or []),
        "snapshot": status,
        "exported_at": snap.get("exported_at"),
    }


async def get_synthesis_oracles_coverage(request: Request) -> dict:
    """§A.6 ≥95% coverage gate measurement (INV-Syn-15)."""
    snap, status = _load_snapshot_with_status()
    if snap is None:
        return {"ok": True, "coverage": {}, "snapshot": status}
    return {
        "ok": True,
        "coverage": dict(snap.get("coverage", {}) or {}),
        "snapshot": status,
        "exported_at": snap.get("exported_at"),
    }


async def get_synthesis_oracles_budget(request: Request) -> dict:
    """Per-oracle daily SOL spend + remaining budget (INV-Syn-13)."""
    snap, status = _load_snapshot_with_status()
    if snap is None:
        return {"ok": True, "budget": {"per_oracle": []}, "snapshot": status}
    return {
        "ok": True,
        "budget": dict(snap.get("budget", {}) or {"per_oracle": []}),
        "snapshot": status,
        "exported_at": snap.get("exported_at"),
    }


async def get_synthesis_proofs_recent(request: Request) -> dict:
    """Last N proof commits (Merkle + ZK; strategy + cost + commitment)."""
    snap, status = _load_snapshot_with_status()
    if snap is None:
        return {"ok": True, "proofs": [], "snapshot": status}
    return {
        "ok": True,
        "proofs": list(snap.get("recent_proofs", []) or []),
        "snapshot": status,
        "exported_at": snap.get("exported_at"),
    }


__all__ = (
    "get_synthesis_oracles_router",
    "get_synthesis_oracles_recent",
    "get_synthesis_oracles_coverage",
    "get_synthesis_oracles_budget",
    "get_synthesis_proofs_recent",
    "DEFAULT_SNAPSHOT_PATH",
    "SNAPSHOT_STALENESS_SECONDS",
)
