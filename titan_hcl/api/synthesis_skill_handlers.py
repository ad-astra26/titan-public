"""API handlers for `/v6/synthesis/skills/*` (Phase 8 §P8.H).

Backs the Observatory's skill-library panel: list compiled skills,
inspect a single skill, list recent mining passes, surface the §A.6
scored_by coverage readout.

**Read source:** mirrors P4 spine_snapshot / P5 forks_snapshot / P7
buffers_snapshot — synthesis_worker (sole writer per INV-Syn-19)
exports `data/skills_snapshot.json` atomically after every
persist/utility update. The api process reads this JSON only.

Snapshot schema (see `ProceduralSkillStore._build_snapshot_payload`):

    {
      "version": 1,
      "ts": <wall-clock seconds>,
      "count": int,
      "persists_seen": int,
      "utility_updates": int,
      "verifications_seen": int,
      "rejections_seen": int,
      "soft_retires_seen": int,
      "skills": [
        {skill_id, name, nl_description, success_count, failure_count,
         last_used, created_at, utility_score, verified_at},
        ...
      ]
    }

For recent_mining_passes + coverage: lightweight derivation from the
timechain index (mining pass meta-fork TXs) — read-only sqlite URI
opens so we never block the chain writer.

Soft-fail contract: missing / stale / corrupt snapshot → 200 with
`{"ok": true, "snapshot": "missing|stale|corrupt", ...empty...}`.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from typing import Optional

from fastapi import Request

logger = logging.getLogger(__name__)


DEFAULT_SNAPSHOT_NAME = "skills_snapshot.json"
SNAPSHOT_STALENESS_SECONDS = 600  # 10× the 60s recompute heartbeat cadence
DEFAULT_INDEX_DB_RELATIVE = "timechain/index.db"
RECENT_PASSES_LIMIT = 20
COVERAGE_WINDOW_HOURS = 24


# ── Snapshot cache (mtime-keyed) ──────────────────────────────────


_SNAPSHOT_CACHE: dict[str, dict] = {}


def _resolve_snapshot_path() -> str:
    data_dir = os.environ.get("TITAN_DATA_DIR", "data")
    return os.path.join(data_dir, DEFAULT_SNAPSHOT_NAME)


def _resolve_index_db_path() -> str:
    data_dir = os.environ.get("TITAN_DATA_DIR", "data")
    return os.path.join(data_dir, DEFAULT_INDEX_DB_RELATIVE)


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
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning(
            "[synthesis_skill_handlers] snapshot parse failed (%s): %s", path, e,
        )
        return None
    if not isinstance(data, dict):
        return None
    _SNAPSHOT_CACHE[path] = {"mtime": mtime, "data": data}
    return data


def _load_snapshot_with_status(path: Optional[str] = None) -> tuple[Optional[dict], str]:
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


# ── GET handlers ──────────────────────────────────────────────────


async def get_v6_synthesis_skills_list(request: Request):
    """Return list of compiled skills.

    Response shape:
        {ok, snapshot, ts, count, skills: [...]}
    """
    payload, status = _load_snapshot_with_status()
    if payload is None:
        return {
            "ok": True, "snapshot": status, "ts": 0.0,
            "count": 0, "skills": [],
        }
    skills = payload.get("skills") if isinstance(payload, dict) else None
    if not isinstance(skills, list):
        skills = []
    return {
        "ok": True,
        "snapshot": status,
        "ts": float(payload.get("ts") or 0.0),
        "count": int(payload.get("count") or len(skills)),
        "skills": skills[:100],  # cap response payload
    }


async def get_v6_synthesis_skills_detail(request: Request):
    """Return one skill's full row.

    Query params: skill_id (required)

    Response shape:
        {ok, snapshot, skill: {...} | null}
    """
    skill_id = (request.query_params.get("skill_id") or "").strip()
    if not skill_id:
        return {
            "ok": False, "error": "skill_id query param required",
            "snapshot": "missing", "skill": None,
        }
    payload, status = _load_snapshot_with_status()
    if payload is None:
        return {"ok": True, "snapshot": status, "skill": None}
    skills = payload.get("skills") if isinstance(payload, dict) else None
    if not isinstance(skills, list):
        skills = []
    match = next(
        (s for s in skills if isinstance(s, dict) and s.get("skill_id") == skill_id),
        None,
    )
    return {"ok": True, "snapshot": status, "skill": match}


async def get_v6_synthesis_skills_recent(request: Request):
    """Return the most-recent skill_mining_pass meta-fork TXs.

    Reads the timechain index.db read-only. Soft-fails to empty list on
    any I/O error (the snapshot summary section above already exposes
    the in-memory counters; this route is the audit trail).

    Response shape:
        {ok, source, passes: [{tx_hash, ts, ...summary fields}, ...]}
    """
    path = _resolve_index_db_path()
    passes: list[dict] = []
    if not os.path.exists(path):
        return {"ok": True, "source": "no_index_db", "passes": []}
    try:
        conn = sqlite3.connect(
            f"file:{path}?mode=ro&immutable=0", uri=True, timeout=5.0,
        )
        conn.row_factory = sqlite3.Row
    except Exception as e:
        logger.debug("[synthesis_skill_handlers] index open failed: %s", e)
        return {"ok": True, "source": "index_open_failed", "passes": []}
    try:
        cur = conn.execute(
            "SELECT block_hash, fork_id, block_height, timestamp, thought_type, tags "
            "FROM block_index "
            "WHERE thought_type = 'skill_mining_pass' "
            "ORDER BY timestamp DESC LIMIT ?",
            [RECENT_PASSES_LIMIT],
        )
        for row in cur.fetchall():
            bh = row["block_hash"]
            tx_hash = bh.hex() if isinstance(bh, bytes) else str(bh)
            passes.append({
                "tx_hash": tx_hash,
                "block_height": int(row["block_height"]),
                "ts": float(row["timestamp"]),
                "fork_id": int(row["fork_id"]),
                "thought_type": row["thought_type"],
                "tags_raw": row["tags"],
            })
    except sqlite3.Error as e:
        logger.debug("[synthesis_skill_handlers] passes query failed: %s", e)
    finally:
        try:
            conn.close()
        except Exception:
            pass
    return {"ok": True, "source": "block_index", "passes": passes}


async def get_v6_synthesis_skills_coverage(request: Request):
    """Return the §A.6 scored_by coverage readout.

    `scored_by ∈ {oracle, llm, null}` per INV-Syn-15. Numerator = TXs in
    the last 24h with scored_by IS NOT NULL. Denominator = all tool-call
    TXs in the same window. Reads the timechain index.db read-only.

    Response shape:
        {ok, source, window_hours, denominator, numerator,
         coverage_ratio, scored_by_breakdown}
    """
    path = _resolve_index_db_path()
    out_base = {
        "ok": True,
        "window_hours": COVERAGE_WINDOW_HOURS,
        "denominator": 0,
        "numerator": 0,
        "coverage_ratio": 0.0,
        "scored_by_breakdown": {"oracle": 0, "llm": 0, "null": 0},
    }
    if not os.path.exists(path):
        return {**out_base, "source": "no_index_db"}

    since_ts = time.time() - COVERAGE_WINDOW_HOURS * 3600.0
    try:
        conn = sqlite3.connect(
            f"file:{path}?mode=ro&immutable=0", uri=True, timeout=5.0,
        )
        conn.row_factory = sqlite3.Row
    except Exception:
        return {**out_base, "source": "index_open_failed"}

    try:
        # Coverage is derived from the `tags` blob the chain writer
        # stores per block. write_tool_call sets tags including
        # `scored_by:oracle|llm|none`. We rely on substring match — the
        # tag list is a small JSON array per row so the SQL LIKE is
        # cheap (no full chain_*.bin walk).
        cur = conn.execute(
            "SELECT tags FROM block_index "
            "WHERE thought_type = 'tool_call' AND timestamp > ? "
            "LIMIT 50000",
            [since_ts],
        )
        denom = 0
        oracle_n = 0
        llm_n = 0
        null_n = 0
        for row in cur.fetchall():
            tags_raw = row["tags"] or ""
            denom += 1
            if "scored_by:oracle" in tags_raw:
                oracle_n += 1
            elif "scored_by:llm" in tags_raw:
                llm_n += 1
            else:
                # 'scored_by:none' OR no scored_by tag at all
                null_n += 1
        numerator = oracle_n + llm_n
        ratio = (numerator / denom) if denom else 0.0
        return {
            **out_base,
            "source": "block_index",
            "denominator": denom,
            "numerator": numerator,
            "coverage_ratio": float(ratio),
            "scored_by_breakdown": {
                "oracle": oracle_n,
                "llm": llm_n,
                "null": null_n,
            },
        }
    except sqlite3.Error as e:
        logger.debug("[synthesis_skill_handlers] coverage query failed: %s", e)
        return {**out_base, "source": "coverage_query_failed"}
    finally:
        try:
            conn.close()
        except Exception:
            pass
