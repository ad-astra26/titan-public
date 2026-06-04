"""API handlers for `/v6/synthesis/concepts/*` (Phase 4 §P4.I + FU-1).

Backs the Observatory's ConceptSpinePanel: list spines, get one spine's
full version history, fetch the groundedness heatmap.

**Read source (FU-1):** the handlers read `data/spine_snapshot.json` —
the atomic JSON export synthesis_worker writes after every 60s recompute
pass (mirrors `activation_snapshot.json` + `bundle_snapshot.json`). The
api process does NOT open the Kuzu file directly because Kuzu 0.11's
`read_only=True` flag still acquires the exclusive write lock against
the active synthesis_worker writer — same lock-conflict that motivated
the JSON-snapshot pattern for DuckDB in Phase 1.

Snapshot schema (see `ConceptStore.export_snapshot`):
    {
      "version": 1,
      "exported_at": <wall-clock seconds>,
      "concepts": [{concept_id, version, name, memory_type,
                    groundedness, anchor_tx, created_at}, ...],
      "composition_edges": {
        "from": [[(from_id, from_ver), (to_id, to_ver)], ...],
        "into": [[(from_id, from_ver), (to_id, to_ver)], ...]
      }
    }

Soft-fail contract: missing / unparseable snapshot → `{"ok": true,
"concepts": [], "snapshot": "missing|stale|corrupt"}`. Frontend renders
an empty state; no 500 cascade.

`api/v6.py` mounts these handlers via ROUTE_TABLE entries.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_SNAPSHOT_PATH = "data/spine_snapshot.json"
SNAPSHOT_STALENESS_SECONDS = 600  # 10 min — far above synthesis_worker's 60s cadence


# ── Snapshot cache (mtime-keyed; cheap stat to invalidate) ──────────


_SNAPSHOT_CACHE: dict[str, dict] = {}  # path → {mtime, data}


def _resolve_snapshot_path() -> str:
    """Pick up TITAN_DATA_DIR like the rest of the stack so tests + shadow
    directories work without monkey-patching."""
    data_dir = os.environ.get("TITAN_DATA_DIR", "data")
    return os.path.join(data_dir, "spine_snapshot.json")


def _load_snapshot(path: Optional[str] = None) -> Optional[dict]:
    """Read the spine snapshot JSON with mtime-keyed cache. Returns the
    parsed payload dict or None on:
      - missing file (snapshot_status="missing"),
      - parse error (snapshot_status="corrupt"),
      - staleness (snapshot_status="stale") — synthesis_worker hasn't
        exported in > SNAPSHOT_STALENESS_SECONDS.
    Caller distinguishes via the second return value of `_load_snapshot_status`.
    """
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
            "[synthesis_concept_handlers] snapshot parse failed (%s): %s",
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
    """Return (payload, status). status ∈ {"ok", "missing", "stale",
    "corrupt"}. Status "ok" implies payload is non-None."""
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
    """Test-only: drop the cached snapshot. Production must never call
    this (snapshot cache is intentionally process-lifetime + mtime-keyed)."""
    _SNAPSHOT_CACHE.clear()


# ── Public handler functions (wired into v6.py ROUTE_TABLE) ─────────


def _empty_heatmap() -> dict[str, list[int]]:
    return {
        "declarative": [0] * 10,
        "procedural": [0] * 10,
        "episodic": [0] * 10,
        "meta": [0] * 10,
    }


def _latest_per_concept(concepts: list[dict]) -> dict[str, dict]:
    """Collapse a flat list of (concept_id, version) rows into a
    {concept_id: latest_row} dict (highest version wins)."""
    latest: dict[str, dict] = {}
    for r in concepts:
        cid = r.get("concept_id")
        if not isinstance(cid, str):
            continue
        ver = int(r.get("version", 0) or 0)
        existing = latest.get(cid)
        if existing is None or ver > int(existing.get("version", 0) or 0):
            latest[cid] = r
    return latest


def get_synthesis_concepts(
    limit: int = 50, offset: int = 0,
    memory_type: Optional[str] = None,
) -> dict:
    """GET /v6/synthesis/concepts — paginated list of spines (latest
    version per concept_id), ordered by groundedness DESC. Optional
    `memory_type` filter (declarative|procedural|episodic|meta)."""
    payload, status = _load_snapshot_with_status()
    if payload is None:
        return {"ok": True, "concepts": [], "total": 0,
                "snapshot": status}
    try:
        concepts = payload.get("concepts") or []
        latest = _latest_per_concept(concepts)
        rows = list(latest.values())
        if memory_type is not None:
            rows = [r for r in rows if r.get("memory_type") == memory_type]
        rows.sort(
            key=lambda r: r.get("groundedness", 0.0) or 0.0,
            reverse=True,
        )
        total = len(rows)
        page = rows[offset: offset + max(0, min(int(limit), 500))]
        return {
            "ok": True,
            "concepts": page,
            "total": total,
            "limit": int(limit),
            "offset": int(offset),
            "memory_type": memory_type,
            "snapshot": status,
            "exported_at": payload.get("exported_at"),
        }
    except Exception as e:
        logger.warning("[handlers] get_synthesis_concepts failed: %s", e)
        return {"ok": False, "error": str(e), "concepts": [], "total": 0}


def get_synthesis_concept(concept_id: str) -> dict:
    """GET /v6/synthesis/concepts/<concept_id> — full spine of one
    concept: every version + composition edges (both directions) keyed
    by the latest version."""
    if not concept_id:
        return {"ok": False, "error": "empty_concept_id"}
    payload, status = _load_snapshot_with_status()
    if payload is None:
        return {"ok": True, "concept_id": concept_id, "versions": [],
                "snapshot": status}
    try:
        all_concepts = payload.get("concepts") or []
        versions = [
            r for r in all_concepts if r.get("concept_id") == concept_id
        ]
        versions.sort(key=lambda r: int(r.get("version", 0) or 0))
        if not versions:
            return {"ok": True, "concept_id": concept_id,
                    "versions": [], "exists": False,
                    "snapshot": status}
        latest_version = int(versions[-1].get("version", 1) or 1)
        edges = payload.get("composition_edges") or {}

        def _edges_for(direction: str) -> list[dict]:
            raw = edges.get(direction) or []
            out: list[dict] = []
            for entry in raw:
                if not (isinstance(entry, list) and len(entry) == 2):
                    continue
                a, b = entry
                if not (isinstance(a, list) and len(a) == 2):
                    continue
                if not (isinstance(b, list) and len(b) == 2):
                    continue
                # Filter to edges where the anchor side matches our concept_id
                # at the latest_version (per arch §10 — edges are tracked
                # per-version but UI primarily wants the latest spine).
                anchor_id, anchor_ver = a
                if (anchor_id == concept_id
                        and int(anchor_ver or 0) == latest_version):
                    nb_id, nb_ver = b
                    out.append({
                        "concept_id": nb_id,
                        "version": int(nb_ver or 0),
                    })
            return out

        return {
            "ok": True,
            "concept_id": concept_id,
            "exists": True,
            "latest_version": latest_version,
            "versions": versions,
            "composed_from": _edges_for("from"),
            "composed_into": _edges_for("into"),
            "snapshot": status,
            "exported_at": payload.get("exported_at"),
        }
    except Exception as e:
        logger.warning(
            "[handlers] get_synthesis_concept(%s) failed: %s",
            concept_id, e,
        )
        return {"ok": False, "error": str(e),
                "concept_id": concept_id, "versions": []}


def get_synthesis_concepts_heatmap() -> dict:
    """GET /v6/synthesis/concepts/heatmap — 4×10 grid of concept counts
    bucketed by (memory_type, groundedness_decile). memory_type rows:
    declarative, procedural, episodic, meta. Columns: decile 0..9 where
    column k contains concepts with k/10 ≤ groundedness < (k+1)/10."""
    payload, status = _load_snapshot_with_status()
    if payload is None:
        return {
            "ok": True,
            "heatmap": _empty_heatmap(),
            "snapshot": status,
        }
    try:
        all_concepts = payload.get("concepts") or []
        latest = _latest_per_concept(all_concepts)
        heatmap = _empty_heatmap()
        total = 0
        for r in latest.values():
            mt = r.get("memory_type", "meta")
            if mt not in heatmap:
                mt = "meta"
            g = max(
                0.0,
                min(0.9999, float(r.get("groundedness", 0.0) or 0.0)),
            )
            decile = int(g * 10)
            heatmap[mt][decile] += 1
            total += 1
        return {
            "ok": True,
            "heatmap": heatmap,
            "total": total,
            "snapshot": status,
            "exported_at": payload.get("exported_at"),
        }
    except Exception as e:
        logger.warning("[handlers] heatmap failed: %s", e)
        return {"ok": False, "error": str(e), "heatmap": _empty_heatmap()}


__all__ = (
    "get_synthesis_concepts",
    "get_synthesis_concept",
    "get_synthesis_concepts_heatmap",
    "_load_snapshot",
    "_load_snapshot_with_status",
    "_resolve_snapshot_path",
    "_reset_cache_for_tests",
    "DEFAULT_SNAPSHOT_PATH",
    "SNAPSHOT_STALENESS_SECONDS",
)
