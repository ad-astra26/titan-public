"""API handlers for `/v6/synthesis/forks/*` (Phase 5 §P5.I).

Backs the Observatory's HypothesisForkPanel (forthcoming frontend
follow-up): list active + recently-graduated + recently-abandoned forks,
fetch per-fork detail, list the tombstone log.

**Read source:** mirrors the Phase 4 FU-1 spine_snapshot pattern —
synthesis_worker exports `data/forks_snapshot.json` every 60s recompute
pass. The api process reads this JSON only; it never opens
`synthesis.duckdb` directly (DuckDB 1.5+ exclusive-lock against the
active sole-writer = synthesis_worker).

Snapshot schema (see `HypothesisForkStore.export_snapshot`):
    {
      "version": 1,
      "exported_at": <wall-clock seconds>,
      "forks": [{fork_id, root_anchor, parent_concept_id, intent, status,
                 created_at, last_touched, use_count, activation,
                 graduated_at, graduated_concept_id, graduated_anchor_tx,
                 abandoned_at, abandoned_tombstone_tx, abandonment_reason}, ...],
      "summary": {open: int, graduated: int, abandoned: int}
    }

Soft-fail contract: missing / unparseable snapshot → `{"ok": true,
"forks": [], "snapshot": "missing|stale|corrupt"}`. Frontend renders
an empty state; no 500 cascade.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_SNAPSHOT_PATH = "data/forks_snapshot.json"
SNAPSHOT_STALENESS_SECONDS = 600  # 10× the 60s tick (mirrors P4 FU-1)


# ── Snapshot cache (mtime-keyed) ──────────────────────────────────


_SNAPSHOT_CACHE: dict[str, dict] = {}


def _resolve_snapshot_path() -> str:
    data_dir = os.environ.get("TITAN_DATA_DIR", "data")
    return os.path.join(data_dir, "forks_snapshot.json")


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
            "[synthesis_fork_handlers] snapshot parse failed (%s): %s",
            path, e,
        )
        return None
    if not isinstance(data, dict):
        return None
    _SNAPSHOT_CACHE[path] = {"mtime": mtime, "data": data}
    return data


def _load_snapshot_with_status(
    path: Optional[str] = None,
) -> tuple[Optional[dict], str]:
    """Return (payload, status). status ∈ {"ok","missing","stale","corrupt"}."""
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


def _reset_cache_for_tests() -> None:
    """Test-only — drop the cached snapshot. Production never calls this."""
    _SNAPSHOT_CACHE.clear()


# ── Public handler functions (wired into v6.py ROUTE_TABLE) ──────


def get_synthesis_forks(
    limit: int = 100, offset: int = 0,
    status: Optional[str] = None,
    since: Optional[float] = None,
) -> dict:
    """GET /v6/synthesis/forks — paginated fork list, optional status
    filter (open|graduated|abandoned) + optional since-ts filter (only
    rows with last_touched >= since).

    Default ordering: last_touched DESC so recent activity surfaces first.
    """
    payload, snap_status = _load_snapshot_with_status()
    if payload is None:
        return {
            "ok": True, "forks": [], "total": 0,
            "summary": {"open": 0, "graduated": 0, "abandoned": 0},
            "snapshot": snap_status,
        }
    try:
        forks = list(payload.get("forks") or [])
        if status is not None:
            forks = [f for f in forks if f.get("status") == status]
        if since is not None:
            try:
                since_f = float(since)
                forks = [
                    f for f in forks
                    if float(f.get("last_touched") or 0.0) >= since_f
                ]
            except (TypeError, ValueError):
                pass
        forks.sort(
            key=lambda f: float(f.get("last_touched") or 0.0),
            reverse=True,
        )
        total = len(forks)
        page = forks[offset: offset + max(0, min(int(limit), 500))]
        return {
            "ok": True, "forks": page, "total": total,
            "limit": int(limit), "offset": int(offset),
            "status": status, "since": since,
            "summary": payload.get("summary", {}),
            "snapshot": snap_status,
            "exported_at": payload.get("exported_at"),
        }
    except Exception as e:
        logger.warning("[handlers] get_synthesis_forks failed: %s", e)
        return {"ok": False, "error": str(e), "forks": [], "total": 0}


def get_synthesis_fork(fork_id: str) -> dict:
    """GET /v6/synthesis/forks/{fork_id} — per-fork detail."""
    if not fork_id:
        return {"ok": False, "error": "empty_fork_id"}
    payload, snap_status = _load_snapshot_with_status()
    if payload is None:
        return {
            "ok": True, "fork_id": fork_id, "fork": None,
            "snapshot": snap_status,
        }
    try:
        all_forks = payload.get("forks") or []
        match = next(
            (f for f in all_forks if f.get("fork_id") == fork_id), None,
        )
        return {
            "ok": True,
            "fork_id": fork_id,
            "fork": match,
            "exists": match is not None,
            "snapshot": snap_status,
            "exported_at": payload.get("exported_at"),
        }
    except Exception as e:
        logger.warning(
            "[handlers] get_synthesis_fork(%s) failed: %s", fork_id, e,
        )
        return {"ok": False, "error": str(e), "fork_id": fork_id}


def get_synthesis_fork_tombstones(
    limit: int = 100, since: Optional[float] = None,
) -> dict:
    """GET /v6/synthesis/forks/tombstones — abandoned-fork log (the
    auditable scar log). Filtered to status='abandoned' rows; ordered by
    abandoned_at DESC. Each entry includes the canonical tombstone_tx
    hash (anchored on the meta fork; rebuildable from chain via INV-2)."""
    payload, snap_status = _load_snapshot_with_status()
    if payload is None:
        return {
            "ok": True, "tombstones": [], "total": 0,
            "snapshot": snap_status,
        }
    try:
        forks = list(payload.get("forks") or [])
        tombstones = [
            f for f in forks if f.get("status") == "abandoned"
        ]
        if since is not None:
            try:
                since_f = float(since)
                tombstones = [
                    t for t in tombstones
                    if float(t.get("abandoned_at") or 0.0) >= since_f
                ]
            except (TypeError, ValueError):
                pass
        tombstones.sort(
            key=lambda t: float(t.get("abandoned_at") or 0.0),
            reverse=True,
        )
        total = len(tombstones)
        page = tombstones[:max(0, min(int(limit), 500))]
        # Surface the canonical fields the audit consumer needs.
        out = [
            {
                "fork_id": t.get("fork_id"),
                "intent": t.get("intent"),
                "root_anchor": t.get("root_anchor"),
                "abandoned_at": t.get("abandoned_at"),
                "abandoned_tombstone_tx": t.get("abandoned_tombstone_tx"),
                "abandonment_reason": t.get("abandonment_reason"),
                "explored_from": t.get("created_at"),
                "explored_to": t.get("last_touched"),
            }
            for t in page
        ]
        return {
            "ok": True, "tombstones": out, "total": total,
            "limit": int(limit), "since": since,
            "snapshot": snap_status,
            "exported_at": payload.get("exported_at"),
        }
    except Exception as e:
        logger.warning(
            "[handlers] get_synthesis_fork_tombstones failed: %s", e,
        )
        return {"ok": False, "error": str(e), "tombstones": [], "total": 0}


def get_synthesis_fork_summary() -> dict:
    """GET /v6/synthesis/forks/summary — small headline-metric handler.
    Used by Observatory landing-page health tiles (forthcoming) — cheap
    JSON read, fast under load."""
    payload, snap_status = _load_snapshot_with_status()
    if payload is None:
        return {
            "ok": True,
            "summary": {"open": 0, "graduated": 0, "abandoned": 0},
            "snapshot": snap_status,
        }
    return {
        "ok": True,
        "summary": payload.get("summary", {}),
        "snapshot": snap_status,
        "exported_at": payload.get("exported_at"),
    }
