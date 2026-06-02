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

import asyncio
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


def _resolve_oracles_snapshot_path() -> str:
    # G11 (AUDIT §5.3): the AUTHORITATIVE coverage source. synthesis_worker
    # builds it via OracleSnapshotExporter → CoverageAnalyzer over the v2-AWARE
    # procedural_tx_reader; the /v6/synthesis/oracles/coverage handler already
    # reads exactly this. Cross-process atomic-JSON read (INV-Syn-8: api never
    # opens synthesis.duckdb).
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
    def _impl() -> dict:
        # G11 (AUDIT §5.3): v2-aware. skill_mining_pass TXs are sealed into
        # meta-fork BATCH blocks whose block_index row carries the batch
        # primary_type + a 'v2_batch' tag, NOT 'skill_mining_pass' — so the old
        # `WHERE thought_type='skill_mining_pass'` query returned []. Walk recent
        # meta-fork blocks, resolve each block's tx_summaries, keep the
        # skill_mining_pass entries (the audit trail of WHEN passes ran; rich
        # counters live in the skills snapshot summary). api reads the chain
        # files read-only — it never opens synthesis.duckdb (INV-Syn-8).
        path = _resolve_index_db_path()
        passes: list[dict] = []
        if not os.path.exists(path):
            return {"ok": True, "source": "no_index_db", "passes": []}
        data_dir = os.environ.get("TITAN_DATA_DIR", "data")
        try:
            from titan_hcl.synthesis.chain_reader import read_block_content_at
            from titan_hcl.logic.timechain_v2 import resolve_batch_summaries
        except Exception as e:
            logger.debug("[synthesis_skill_handlers] chain reader unavailable: %s", e)
            return {"ok": True, "source": "chain_reader_unavailable", "passes": []}
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
                "SELECT bi.fork_id, bi.file_offset, bi.timestamp "
                "FROM block_index bi "
                "JOIN fork_registry fr ON bi.fork_id = fr.fork_id "
                "WHERE fr.fork_name = 'meta' "
                "ORDER BY bi.timestamp DESC LIMIT 500",
            )
            rows = list(cur.fetchall())
        except sqlite3.Error as e:
            logger.debug("[synthesis_skill_handlers] passes query failed: %s", e)
            rows = []
        finally:
            try:
                conn.close()
            except Exception:
                pass
        for r in rows:
            if len(passes) >= RECENT_PASSES_LIMIT:
                break
            try:
                content = read_block_content_at(
                    data_dir, int(r["fork_id"]), int(r["file_offset"]))
            except Exception:
                continue
            if not isinstance(content, dict):
                continue
            try:
                summaries = resolve_batch_summaries(content)
            except Exception:
                summaries = ([content]
                             if content.get("txs_scanned") is not None else [])
            block_ts = float(r["timestamp"])
            for s in summaries or []:
                if not isinstance(s, dict):
                    continue
                if (s.get("type") or s.get("thought_type")) != "skill_mining_pass":
                    continue
                passes.append({
                    "tx_hash": s.get("hash") or s.get("tx_hash") or "",
                    "ts": block_ts,
                    "tags_raw": s.get("tags"),
                })
                if len(passes) >= RECENT_PASSES_LIMIT:
                    break
        return {"ok": True, "source": "chain_v2", "passes": passes}

    # Sync sqlite3 I/O off the event loop (§8.0.ter spirit — never block the loop).
    return await asyncio.to_thread(_impl)


async def get_v6_synthesis_skills_coverage(request: Request):
    """Return the §A.6 scored_by coverage readout.

    `scored_by ∈ {oracle, llm, null}` per INV-Syn-15. Numerator = TXs in
    the last 24h with scored_by IS NOT NULL. Denominator = all tool-call
    TXs in the same window. Reads the timechain index.db read-only.

    Response shape:
        {ok, source, window_hours, denominator, numerator,
         coverage_ratio, scored_by_breakdown}
    """
    # G11 (AUDIT §5.3): read the AUTHORITATIVE coverage block from
    # oracles_snapshot.json (synthesis_worker CoverageAnalyzer over the v2-aware
    # procedural_tx_reader) instead of a hand-rolled block_index LIKE-on-tags
    # query — that query returned ~0 under v2 BATCH blocks (per-TX scored_by
    # lives inside tx_summaries, NOT the batch row's `tags`), silently
    # contradicting the authoritative /v6/synthesis/oracles/coverage (~4.9%).
    base = {
        "ok": True,
        "snapshot": "missing",
        "source": "oracles_snapshot",
        "window_hours": 0.0,
        "denominator": 0,
        "numerator": 0,
        "coverage_ratio": 0.0,
        "scored_by_breakdown": {"oracle": 0, "llm": 0, "null": 0},
        "a6_gate_passes": False,
    }
    payload, status = _load_snapshot_with_status(_resolve_oracles_snapshot_path())
    base["snapshot"] = status
    if payload is None:
        return base
    cov = payload.get("coverage") if isinstance(payload, dict) else None
    if not isinstance(cov, dict) or not cov:
        return base
    total = int(cov.get("total_tool_call_txs") or 0)
    oracle_n = int(cov.get("scored_by_oracle") or 0)
    llm_n = int(cov.get("scored_by_llm") or 0)
    unscored_n = int(cov.get("unscored") or 0)
    numerator = oracle_n + llm_n
    ratio = float(cov.get("coverage_ratio") or 0.0)
    window_hours = float(cov.get("window_seconds") or 0.0) / 3600.0
    return {
        "ok": True,
        "snapshot": status,
        "source": "oracles_snapshot",
        "exported_at": payload.get("exported_at"),
        "window_hours": round(window_hours, 2),
        "denominator": total,
        "numerator": numerator,
        "coverage_ratio": ratio,
        "scored_by_breakdown": {
            "oracle": oracle_n, "llm": llm_n, "null": unscored_n,
        },
        "a6_gate_passes": bool(cov.get("a6_gate_passes", ratio >= 0.95)),
    }
