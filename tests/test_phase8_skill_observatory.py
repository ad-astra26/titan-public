"""Phase 8 — /v6/synthesis/skills/* observatory routes (D-SPEC-PHASE8 §P8.H).

Covers:
- All 4 GET handlers return 200-OK shape on healthy snapshot
- All 4 soft-fail on missing snapshot → snapshot="missing"
- list returns capped skills + count
- detail returns null when skill_id absent / not found
- detail returns full row when found
- recent reads chain index sqlite (no-index-db gracefully empty)
- coverage derives §A.6 readout from block_index tags
- Route table registered in v6.py
- Dashboard re-exports all 4 handlers
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# Set TITAN_DATA_DIR to a tmpdir BEFORE importing the handler module
# so the snapshot path resolves there.
@pytest.fixture()
def isolated_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("TITAN_DATA_DIR", str(tmp_path))
    return tmp_path


def _write_snapshot(data_dir: Path, payload: dict) -> Path:
    p = data_dir / "skills_snapshot.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def _seed_chain_index(data_dir: Path, rows: list[dict]) -> Path:
    """Create a minimal index.db with block_index rows for testing recent + coverage."""
    tc_dir = data_dir / "timechain"
    tc_dir.mkdir(parents=True, exist_ok=True)
    p = tc_dir / "index.db"
    conn = sqlite3.connect(str(p))
    conn.execute(
        "CREATE TABLE block_index ("
        "  block_hash BLOB PRIMARY KEY, fork_id INTEGER, block_height INTEGER,"
        "  timestamp REAL, thought_type TEXT, tags TEXT, file_offset INTEGER"
        ")"
    )
    for r in rows:
        conn.execute(
            "INSERT INTO block_index "
            "(block_hash, fork_id, block_height, timestamp, thought_type, tags, file_offset) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                r["block_hash"].encode() if isinstance(r["block_hash"], str) else r["block_hash"],
                int(r.get("fork_id", 1)),
                int(r.get("block_height", 0)),
                float(r["timestamp"]),
                r["thought_type"],
                r.get("tags", ""),
                int(r.get("file_offset", 0)),
            ],
        )
    conn.commit()
    conn.close()
    return p


def _flush_snapshot_cache():
    """Tests share process; clear the mtime cache between tests."""
    from titan_hcl.api import synthesis_skill_handlers
    synthesis_skill_handlers._SNAPSHOT_CACHE.clear()


# ── list ───────────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_list_missing_snapshot_returns_empty(isolated_data_dir):
    _flush_snapshot_cache()
    from titan_hcl.api.synthesis_skill_handlers import get_v6_synthesis_skills_list
    request = MagicMock()
    resp = await get_v6_synthesis_skills_list(request)
    assert resp["ok"] is True
    assert resp["snapshot"] == "missing"
    assert resp["count"] == 0
    assert resp["skills"] == []


@pytest.mark.anyio
async def test_list_returns_skills(isolated_data_dir):
    _flush_snapshot_cache()
    _write_snapshot(isolated_data_dir, {
        "version": 1, "ts": time.time(), "count": 2,
        "skills": [
            {"skill_id": "s1", "name": "skill 1", "utility_score": 0.7},
            {"skill_id": "s2", "name": "skill 2", "utility_score": 0.4},
        ],
    })
    from titan_hcl.api.synthesis_skill_handlers import get_v6_synthesis_skills_list
    resp = await get_v6_synthesis_skills_list(MagicMock())
    assert resp["ok"] is True
    assert resp["snapshot"] == "ok"
    assert resp["count"] == 2
    assert len(resp["skills"]) == 2


@pytest.mark.anyio
async def test_list_stale_snapshot(isolated_data_dir, monkeypatch):
    _flush_snapshot_cache()
    p = _write_snapshot(isolated_data_dir, {
        "version": 1, "ts": time.time(), "count": 0, "skills": [],
    })
    # Backdate mtime past staleness threshold
    old = time.time() - 700
    os.utime(p, (old, old))
    from titan_hcl.api.synthesis_skill_handlers import get_v6_synthesis_skills_list
    resp = await get_v6_synthesis_skills_list(MagicMock())
    assert resp["snapshot"] == "stale"


@pytest.mark.anyio
async def test_list_corrupt_snapshot(isolated_data_dir):
    _flush_snapshot_cache()
    p = isolated_data_dir / "skills_snapshot.json"
    p.write_text("{not json", encoding="utf-8")
    from titan_hcl.api.synthesis_skill_handlers import get_v6_synthesis_skills_list
    resp = await get_v6_synthesis_skills_list(MagicMock())
    assert resp["snapshot"] == "corrupt"
    assert resp["skills"] == []


# ── detail ─────────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_detail_missing_skill_id_returns_error(isolated_data_dir):
    _flush_snapshot_cache()
    from titan_hcl.api.synthesis_skill_handlers import get_v6_synthesis_skills_detail
    req = MagicMock()
    req.query_params = {}
    resp = await get_v6_synthesis_skills_detail(req)
    assert resp["ok"] is False
    assert "skill_id" in resp.get("error", "")


@pytest.mark.anyio
async def test_detail_returns_skill(isolated_data_dir):
    _flush_snapshot_cache()
    _write_snapshot(isolated_data_dir, {
        "version": 1, "ts": time.time(), "count": 1,
        "skills": [{"skill_id": "skill_X", "name": "X", "utility_score": 0.5}],
    })
    from titan_hcl.api.synthesis_skill_handlers import get_v6_synthesis_skills_detail
    req = MagicMock()
    req.query_params = {"skill_id": "skill_X"}
    resp = await get_v6_synthesis_skills_detail(req)
    assert resp["ok"] is True
    assert resp["skill"]["skill_id"] == "skill_X"


@pytest.mark.anyio
async def test_detail_returns_null_when_not_found(isolated_data_dir):
    _flush_snapshot_cache()
    _write_snapshot(isolated_data_dir, {
        "version": 1, "ts": time.time(), "count": 0, "skills": [],
    })
    from titan_hcl.api.synthesis_skill_handlers import get_v6_synthesis_skills_detail
    req = MagicMock()
    req.query_params = {"skill_id": "missing"}
    resp = await get_v6_synthesis_skills_detail(req)
    assert resp["ok"] is True
    assert resp["skill"] is None


# ── recent ─────────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_recent_no_index_db_returns_empty(isolated_data_dir):
    _flush_snapshot_cache()
    from titan_hcl.api.synthesis_skill_handlers import get_v6_synthesis_skills_recent
    resp = await get_v6_synthesis_skills_recent(MagicMock())
    assert resp["ok"] is True
    assert resp["source"] == "no_index_db"
    assert resp["passes"] == []


@pytest.mark.anyio
async def test_recent_reads_mining_passes_from_chain_index(isolated_data_dir):
    _flush_snapshot_cache()
    _seed_chain_index(isolated_data_dir, [
        {"block_hash": "h_pass_1", "fork_id": 4, "block_height": 100,
         "timestamp": time.time() - 100, "thought_type": "skill_mining_pass", "tags": "[]"},
        {"block_hash": "h_pass_2", "fork_id": 4, "block_height": 101,
         "timestamp": time.time() - 50, "thought_type": "skill_mining_pass", "tags": "[]"},
        {"block_hash": "h_other", "fork_id": 4, "block_height": 102,
         "timestamp": time.time() - 25, "thought_type": "tool_call", "tags": "[]"},
    ])
    from titan_hcl.api.synthesis_skill_handlers import get_v6_synthesis_skills_recent
    resp = await get_v6_synthesis_skills_recent(MagicMock())
    assert resp["ok"] is True
    assert len(resp["passes"]) == 2  # excludes tool_call row
    # Most-recent-first
    assert resp["passes"][0]["block_height"] == 101


# ── coverage ───────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_coverage_no_index_db(isolated_data_dir):
    _flush_snapshot_cache()
    from titan_hcl.api.synthesis_skill_handlers import get_v6_synthesis_skills_coverage
    resp = await get_v6_synthesis_skills_coverage(MagicMock())
    assert resp["ok"] is True
    assert resp["source"] == "no_index_db"
    assert resp["coverage_ratio"] == 0.0


@pytest.mark.anyio
async def test_coverage_computes_ratio(isolated_data_dir):
    _flush_snapshot_cache()
    now = time.time()
    _seed_chain_index(isolated_data_dir, [
        {"block_hash": "h1", "timestamp": now - 100, "thought_type": "tool_call",
         "tags": "tool_call|scored_by:oracle"},
        {"block_hash": "h2", "timestamp": now - 50, "thought_type": "tool_call",
         "tags": "tool_call|scored_by:llm"},
        {"block_hash": "h3", "timestamp": now - 30, "thought_type": "tool_call",
         "tags": "tool_call|scored_by:none"},
    ])
    from titan_hcl.api.synthesis_skill_handlers import get_v6_synthesis_skills_coverage
    resp = await get_v6_synthesis_skills_coverage(MagicMock())
    assert resp["denominator"] == 3
    assert resp["numerator"] == 2
    assert resp["coverage_ratio"] == pytest.approx(2 / 3)
    assert resp["scored_by_breakdown"]["oracle"] == 1
    assert resp["scored_by_breakdown"]["llm"] == 1
    assert resp["scored_by_breakdown"]["null"] == 1


@pytest.mark.anyio
async def test_coverage_excludes_outside_window(isolated_data_dir):
    """Only TXs within COVERAGE_WINDOW_HOURS (default 24) count."""
    _flush_snapshot_cache()
    now = time.time()
    _seed_chain_index(isolated_data_dir, [
        {"block_hash": "in_window", "timestamp": now - 3600, "thought_type": "tool_call",
         "tags": "scored_by:oracle"},
        {"block_hash": "out_of_window", "timestamp": now - 48 * 3600,
         "thought_type": "tool_call", "tags": "scored_by:oracle"},
    ])
    from titan_hcl.api.synthesis_skill_handlers import get_v6_synthesis_skills_coverage
    resp = await get_v6_synthesis_skills_coverage(MagicMock())
    assert resp["denominator"] == 1
    assert resp["numerator"] == 1


# ── Route table + dashboard registration ───────────────────────────────


def test_routes_registered_in_v6():
    from titan_hcl.api import v6 as v6_module
    routes = {r[0] for r in v6_module._T if isinstance(r, tuple)}
    assert "/v6/synthesis/skills" in routes
    assert "/v6/synthesis/skills/detail" in routes
    assert "/v6/synthesis/skills/recent" in routes
    assert "/v6/synthesis/skills/coverage" in routes


def test_handlers_re_exported_by_dashboard():
    from titan_hcl.api import dashboard
    for fname in (
        "get_v6_synthesis_skills_list",
        "get_v6_synthesis_skills_detail",
        "get_v6_synthesis_skills_recent",
        "get_v6_synthesis_skills_coverage",
    ):
        assert hasattr(dashboard, fname), f"dashboard missing {fname}"
