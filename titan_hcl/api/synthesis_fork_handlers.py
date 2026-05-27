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


# ── POST handlers — publish SYNTHESIS_FORK_COMMAND bus events ───
#
# api process publishes to bus → synthesis_worker recv loop handles
# (sole writer per INV-Syn-8) → re-exports forks_snapshot.json → caller
# polls GET /v6/synthesis/forks until the new state appears.
#
# Fire-and-forget semantics: HTTP 202 Accepted with the request_id;
# subsequent GET confirms outcome. This keeps the api process fully
# async (no kernel_rpc sync round-trip) and is the same pattern as
# POST /v4/timechain/test-commit. Auth is the existing internal_key
# convention enforced by the route — body shape is op-discriminated.


def _publish_fork_command(request, payload: dict) -> dict:
    """Helper: publish SYNTHESIS_FORK_COMMAND on the kernel bus. `request`
    is the FastAPI Request from the calling endpoint (carries titan_state
    in app.state). Returns the standard `{ok, request_id}` response shape."""
    import uuid
    from titan_hcl import bus
    from titan_hcl.bus import make_msg
    request_id = uuid.uuid4().hex
    full_payload = {"request_id": request_id, **payload}
    try:
        titan_state = request.app.state.titan
    except Exception as e:
        return {"ok": False, "error": f"titan_state_unavailable: {e}"}
    try:
        titan_state.bus.publish(make_msg(
            bus.SYNTHESIS_FORK_COMMAND, "dashboard", "synthesis",
            full_payload,
        ))
    except Exception as e:
        logger.warning(
            "[synthesis_fork_handlers] publish failed: %s", e,
        )
        return {"ok": False, "error": str(e)}
    return {"ok": True, "request_id": request_id, "accepted": True}


async def post_synthesis_forks(request):
    """POST /v6/synthesis/forks — create a hypothesis fork.

    Body:
      {intent: str, root_anchor?: str, parent_concept_id?: str}

    Eventual-consistent: returns {ok, request_id, accepted}; the new
    fork appears in GET /v6/synthesis/forks once synthesis_worker has
    handled the command (typically <100ms via eager-re-export)."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    intent = body.get("intent")
    if not isinstance(intent, str) or not intent.strip():
        return {"ok": False, "error": "intent_required"}
    payload = {
        "op": "create",
        "intent": intent.strip(),
        "root_anchor": body.get("root_anchor"),
        "parent_concept_id": body.get("parent_concept_id"),
    }
    return _publish_fork_command(request, payload)


async def post_synthesis_fork_record_exploration(request, fork_id: str):
    """POST /v6/synthesis/forks/{fork_id}/record-exploration-tx.

    Body: {tx_hash: str}
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    tx_hash = body.get("tx_hash")
    if not isinstance(tx_hash, str) or not tx_hash:
        return {"ok": False, "error": "tx_hash_required"}
    return _publish_fork_command(request, {
        "op": "record_exploration_tx",
        "fork_id": fork_id,
        "tx_hash": tx_hash,
    })


async def post_synthesis_fork_graduate_manual(request, fork_id: str):
    """POST /v6/synthesis/forks/{fork_id}/graduate-manual.

    Body: {concept_name?: str, evidence_ref?: str}

    Triggers graduation via the `manual:maker` synthetic OracleVerdict.
    Per Maker decision 2026-05-27 §P5.E this is the only graduation
    write-path until Phase 6 oracle plugs ship; use_count auto-graduation
    via FORK_READ is in-process only (synthesis_worker SC handler).
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    return _publish_fork_command(request, {
        "op": "graduate_manual",
        "fork_id": fork_id,
        "concept_name": body.get("concept_name"),
        "evidence_ref": body.get("evidence_ref", "manual_trigger"),
    })


async def post_synthesis_fork_abandon(request, fork_id: str):
    """POST /v6/synthesis/forks/{fork_id}/abandon.

    Body: {reason?: str}
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    return _publish_fork_command(request, {
        "op": "abandon",
        "fork_id": fork_id,
        "reason": body.get("reason", "manual_abandon"),
    })


async def post_synthesis_fork_sweep(request):
    """POST /v6/synthesis/forks/sweep — manual ForkGC sweep trigger.

    Body: {dry_run?: bool}  (defaults to synthesis.fork_gc_live config inversion)
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    payload = {"op": "sweep"}
    if "dry_run" in body:
        payload["dry_run"] = bool(body["dry_run"])
    return _publish_fork_command(request, payload)
