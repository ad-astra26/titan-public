"""API handlers for `/v6/synthesis/engrams/*` (Phase 4 §P4.I + FU-1; §7.G rename).

Backs the Observatory's EngramSpinePanel: list spines, get one spine's
full version history, fetch the groundedness heatmap.

**Read source (FU-1):** the handlers read `data/spine_snapshot.json` —
the atomic JSON export synthesis_worker writes after every 60s recompute
pass (mirrors `activation_snapshot.json` + `bundle_snapshot.json`). The
api process does NOT open the Kuzu file directly because Kuzu 0.11's
`read_only=True` flag still acquires the exclusive write lock against
the active synthesis_worker writer — same lock-conflict that motivated
the JSON-snapshot pattern for DuckDB in Phase 1.

Snapshot schema (see `EngramStore.export_snapshot` — INTERNAL plumbing, keyed
`concepts`/`concept_id`; the internal Kuzu PRIMARY KEY is `concept_id`):
    {
      "version": 1,
      "exported_at": <wall-clock seconds>,
      "concepts": [{concept_id, version, name, memory_type,
                    groundedness, anchor_tx, created_at}, ...],
      "composition_edges": {...}
    }

§7.G external grep-clean: the HTTP surface speaks **engrams** — the response
list key is `engrams` and each row's id field is `engram_id` (mapped here from
the snapshot's internal `concept_id`). The internal Kuzu PK + snapshot schema
keep `concept_id` (renaming the live PK would be a data migration, out of scope).

Soft-fail contract: missing / unparseable snapshot → `{"ok": true,
"engrams": [], "snapshot": "missing|stale|corrupt"}`. Frontend renders
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
            "[synthesis_engram_handlers] snapshot parse failed (%s): %s",
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


def _engram_id(row: dict) -> Optional[str]:
    """The Engram's id from a snapshot row — the internal PK is `concept_id`
    (§7.G keeps the internal PK; the external surface speaks `engram_id`)."""
    v = row.get("concept_id")
    return v if isinstance(v, str) else None


def _relabel(row: dict) -> dict:
    """Map a snapshot row's internal `concept_id` → external `engram_id`
    (§7.G), preserving every other field."""
    return {("engram_id" if k == "concept_id" else k): v for k, v in row.items()}


def _latest_per_engram(rows: list[dict]) -> dict[str, dict]:
    """Collapse a flat list of (concept_id, version) rows into a
    {engram_id: latest_row} dict (highest version wins)."""
    latest: dict[str, dict] = {}
    for r in rows:
        eid = _engram_id(r)
        if eid is None:
            continue
        ver = int(r.get("version", 0) or 0)
        existing = latest.get(eid)
        if existing is None or ver > int(existing.get("version", 0) or 0):
            latest[eid] = r
    return latest


def get_synthesis_engrams(
    limit: int = 50, offset: int = 0,
    memory_type: Optional[str] = None,
) -> dict:
    """GET /v6/synthesis/engrams — paginated list of spines (latest
    version per engram), ordered by groundedness DESC. Optional
    `memory_type` filter (declarative|procedural|episodic|meta)."""
    payload, status = _load_snapshot_with_status()
    if payload is None:
        return {"ok": True, "engrams": [], "total": 0,
                "snapshot": status}
    try:
        rows = payload.get("concepts") or []  # snapshot internal key (unchanged)
        latest = _latest_per_engram(rows)
        out = list(latest.values())
        if memory_type is not None:
            out = [r for r in out if r.get("memory_type") == memory_type]
        out.sort(
            key=lambda r: r.get("groundedness", 0.0) or 0.0,
            reverse=True,
        )
        total = len(out)
        page = out[offset: offset + max(0, min(int(limit), 500))]
        return {
            "ok": True,
            "engrams": [_relabel(r) for r in page],
            "total": total,
            "limit": int(limit),
            "offset": int(offset),
            "memory_type": memory_type,
            "snapshot": status,
            "exported_at": payload.get("exported_at"),
        }
    except Exception as e:
        logger.warning("[handlers] get_synthesis_engrams failed: %s", e)
        return {"ok": False, "error": str(e), "engrams": [], "total": 0}


def get_synthesis_engram(engram_id: str) -> dict:
    """GET /v6/synthesis/engrams/<engram_id> — full spine of one
    engram: every version + composition edges (both directions) keyed
    by the latest version."""
    if not engram_id:
        return {"ok": False, "error": "empty_engram_id"}
    payload, status = _load_snapshot_with_status()
    if payload is None:
        return {"ok": True, "engram_id": engram_id, "versions": [],
                "snapshot": status}
    try:
        all_rows = payload.get("concepts") or []  # snapshot internal key
        versions = [r for r in all_rows if _engram_id(r) == engram_id]
        versions.sort(key=lambda r: int(r.get("version", 0) or 0))
        if not versions:
            return {"ok": True, "engram_id": engram_id,
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
                # Filter to edges where the anchor side matches our engram_id
                # at the latest_version (per arch §10 — edges are tracked
                # per-version but UI primarily wants the latest spine).
                anchor_id, anchor_ver = a
                if (anchor_id == engram_id
                        and int(anchor_ver or 0) == latest_version):
                    nb_id, nb_ver = b
                    out.append({
                        "engram_id": nb_id,
                        "version": int(nb_ver or 0),
                    })
            return out

        return {
            "ok": True,
            "engram_id": engram_id,
            "exists": True,
            "latest_version": latest_version,
            "versions": [_relabel(r) for r in versions],
            "composed_from": _edges_for("from"),
            "composed_into": _edges_for("into"),
            "snapshot": status,
            "exported_at": payload.get("exported_at"),
        }
    except Exception as e:
        logger.warning(
            "[handlers] get_synthesis_engram(%s) failed: %s",
            engram_id, e,
        )
        return {"ok": False, "error": str(e),
                "engram_id": engram_id, "versions": []}


def get_synthesis_engrams_heatmap() -> dict:
    """GET /v6/synthesis/engrams/heatmap — 4×10 grid of engram counts
    bucketed by (memory_type, groundedness_decile). memory_type rows:
    declarative, procedural, episodic, meta. Columns: decile 0..9 where
    column k contains engrams with k/10 ≤ groundedness < (k+1)/10."""
    payload, status = _load_snapshot_with_status()
    if payload is None:
        return {
            "ok": True,
            "heatmap": _empty_heatmap(),
            "snapshot": status,
        }
    try:
        all_rows = payload.get("concepts") or []  # snapshot internal key
        latest = _latest_per_engram(all_rows)
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
    "get_synthesis_engrams",
    "get_synthesis_engram",
    "get_synthesis_engrams_heatmap",
    "_load_snapshot",
    "_load_snapshot_with_status",
    "_resolve_snapshot_path",
    "_reset_cache_for_tests",
    "DEFAULT_SNAPSHOT_PATH",
    "SNAPSHOT_STALENESS_SECONDS",
)
