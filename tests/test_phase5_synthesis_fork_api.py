"""Phase 5 — synthesis fork API handlers (P5.I).

Covers `titan_hcl/api/synthesis_fork_handlers.py` against PLAN §P5.I:

- snapshot path resolution honors TITAN_DATA_DIR
- snapshot missing → {ok: true, forks: [], snapshot: "missing"}
- snapshot present + fresh → {ok: true, snapshot: "ok", ...}
- snapshot present but stale → snapshot: "stale", payload still returned
- snapshot corrupt → snapshot: "corrupt", forks: []
- status filter + since filter applied correctly
- per-fork detail returns full row or exists:false
- tombstones endpoint filters to status='abandoned'
- summary endpoint returns the embedded summary dict
- v6.py manifest exposes the routes
"""
from __future__ import annotations

import json
import os
import time

import pytest

from titan_hcl.api import synthesis_fork_handlers as handlers


@pytest.fixture()
def fresh_data_dir(tmp_path, monkeypatch):
    """Per-test TITAN_DATA_DIR so the snapshot cache + file isolate."""
    monkeypatch.setenv("TITAN_DATA_DIR", str(tmp_path))
    handlers._reset_cache_for_tests()
    yield str(tmp_path)
    handlers._reset_cache_for_tests()


def _write_snapshot(data_dir: str, payload: dict, age_seconds: float = 0):
    path = os.path.join(data_dir, "forks_snapshot.json")
    with open(path, "w") as f:
        json.dump(payload, f)
    if age_seconds:
        old = time.time() - age_seconds
        os.utime(path, (old, old))
    return path


# ── Missing / corrupt / stale snapshot ──────────────────────────


def test_missing_snapshot_returns_empty(fresh_data_dir):
    r = handlers.get_synthesis_forks()
    assert r == {
        "ok": True, "forks": [], "total": 0,
        "summary": {"open": 0, "graduated": 0, "abandoned": 0},
        "snapshot": "missing",
    }


def test_corrupt_snapshot_returns_empty(fresh_data_dir):
    path = os.path.join(fresh_data_dir, "forks_snapshot.json")
    with open(path, "w") as f:
        f.write("{not valid json")
    r = handlers.get_synthesis_forks()
    assert r["snapshot"] == "corrupt"
    assert r["forks"] == []


def test_stale_snapshot_returns_data_with_stale_status(fresh_data_dir):
    payload = {
        "version": 1, "exported_at": 0.0,
        "forks": [{
            "fork_id": "f1", "intent": "x", "status": "open",
            "use_count": 0, "activation": 0.0, "created_at": 0.0,
            "last_touched": 0.0, "root_anchor": None,
            "parent_concept_id": None,
        }],
        "summary": {"open": 1, "graduated": 0, "abandoned": 0},
    }
    _write_snapshot(fresh_data_dir, payload, age_seconds=1000)
    r = handlers.get_synthesis_forks()
    assert r["snapshot"] == "stale"
    assert r["total"] == 1


# ── List endpoint ────────────────────────────────────────────────


def test_list_returns_forks_with_status_and_summary(fresh_data_dir):
    payload = {
        "version": 1, "exported_at": time.time(),
        "forks": [
            {"fork_id": "a", "intent": "i_a", "status": "open",
             "last_touched": 100.0, "use_count": 0, "activation": 0.0,
             "created_at": 100.0, "root_anchor": None,
             "parent_concept_id": None},
            {"fork_id": "b", "intent": "i_b", "status": "graduated",
             "last_touched": 200.0, "use_count": 3, "activation": -1.0,
             "created_at": 50.0, "root_anchor": None,
             "parent_concept_id": None,
             "graduated_concept_id": "c1",
             "graduated_anchor_tx": "tx_grad_1"},
        ],
        "summary": {"open": 1, "graduated": 1, "abandoned": 0},
    }
    _write_snapshot(fresh_data_dir, payload)
    r = handlers.get_synthesis_forks()
    assert r["ok"] is True
    assert r["snapshot"] == "ok"
    assert r["total"] == 2
    assert r["summary"]["graduated"] == 1
    # Ordered by last_touched DESC.
    assert r["forks"][0]["fork_id"] == "b"
    assert r["forks"][1]["fork_id"] == "a"


def test_status_filter_returns_only_matching(fresh_data_dir):
    payload = {
        "version": 1, "exported_at": time.time(),
        "forks": [
            {"fork_id": "a", "status": "open",
             "last_touched": 100.0, "use_count": 0, "activation": 0.0,
             "created_at": 100.0, "intent": "x", "root_anchor": None,
             "parent_concept_id": None},
            {"fork_id": "b", "status": "abandoned",
             "last_touched": 200.0, "use_count": 0, "activation": -10.0,
             "created_at": 50.0, "intent": "y", "root_anchor": None,
             "parent_concept_id": None,
             "abandoned_tombstone_tx": "tx_t_1",
             "abandoned_at": 200.0, "abandonment_reason": "below_floor"},
        ],
        "summary": {"open": 1, "graduated": 0, "abandoned": 1},
    }
    _write_snapshot(fresh_data_dir, payload)
    r = handlers.get_synthesis_forks(status="abandoned")
    assert r["total"] == 1
    assert r["forks"][0]["fork_id"] == "b"


def test_since_filter(fresh_data_dir):
    payload = {
        "version": 1, "exported_at": time.time(),
        "forks": [
            {"fork_id": "old", "status": "open", "last_touched": 50.0,
             "use_count": 0, "activation": 0.0, "created_at": 50.0,
             "intent": "x", "root_anchor": None,
             "parent_concept_id": None},
            {"fork_id": "new", "status": "open", "last_touched": 500.0,
             "use_count": 0, "activation": 0.0, "created_at": 500.0,
             "intent": "y", "root_anchor": None,
             "parent_concept_id": None},
        ],
        "summary": {"open": 2, "graduated": 0, "abandoned": 0},
    }
    _write_snapshot(fresh_data_dir, payload)
    r = handlers.get_synthesis_forks(since=100.0)
    assert r["total"] == 1
    assert r["forks"][0]["fork_id"] == "new"


# ── Per-fork detail ─────────────────────────────────────────────


def test_get_fork_returns_match(fresh_data_dir):
    payload = {
        "version": 1, "exported_at": time.time(),
        "forks": [
            {"fork_id": "x", "intent": "i", "status": "open",
             "use_count": 0, "activation": 0.0,
             "created_at": 1.0, "last_touched": 2.0,
             "root_anchor": None, "parent_concept_id": None},
        ],
        "summary": {"open": 1, "graduated": 0, "abandoned": 0},
    }
    _write_snapshot(fresh_data_dir, payload)
    r = handlers.get_synthesis_fork("x")
    assert r["exists"] is True
    assert r["fork"]["fork_id"] == "x"


def test_get_fork_returns_exists_false_when_missing(fresh_data_dir):
    payload = {
        "version": 1, "exported_at": time.time(),
        "forks": [], "summary": {"open": 0, "graduated": 0, "abandoned": 0},
    }
    _write_snapshot(fresh_data_dir, payload)
    r = handlers.get_synthesis_fork("ghost")
    assert r["exists"] is False
    assert r["fork"] is None


def test_get_fork_rejects_empty_id(fresh_data_dir):
    r = handlers.get_synthesis_fork("")
    assert r["ok"] is False


# ── Tombstones endpoint ─────────────────────────────────────────


def test_tombstones_filters_to_abandoned_only(fresh_data_dir):
    payload = {
        "version": 1, "exported_at": time.time(),
        "forks": [
            {"fork_id": "open1", "status": "open",
             "last_touched": 100.0, "created_at": 100.0,
             "intent": "i", "root_anchor": None,
             "parent_concept_id": None, "use_count": 0,
             "activation": 0.0},
            {"fork_id": "grad1", "status": "graduated",
             "last_touched": 200.0, "created_at": 100.0,
             "intent": "g", "root_anchor": None,
             "parent_concept_id": None, "use_count": 3,
             "activation": -1.0,
             "graduated_concept_id": "c", "graduated_anchor_tx": "tx"},
            {"fork_id": "aband1", "status": "abandoned",
             "last_touched": 250.0, "created_at": 100.0,
             "intent": "a", "root_anchor": None,
             "parent_concept_id": None, "use_count": 0,
             "activation": -10.0,
             "abandoned_at": 250.0,
             "abandoned_tombstone_tx": "tx_tomb_a",
             "abandonment_reason": "activation_below_floor"},
        ],
        "summary": {"open": 1, "graduated": 1, "abandoned": 1},
    }
    _write_snapshot(fresh_data_dir, payload)
    r = handlers.get_synthesis_fork_tombstones()
    assert r["total"] == 1
    assert r["tombstones"][0]["fork_id"] == "aband1"
    assert r["tombstones"][0]["abandoned_tombstone_tx"] == "tx_tomb_a"
    assert r["tombstones"][0]["abandonment_reason"] == "activation_below_floor"


def test_tombstones_since_filter(fresh_data_dir):
    payload = {
        "version": 1, "exported_at": time.time(),
        "forks": [
            {"fork_id": "old_tomb", "status": "abandoned",
             "abandoned_at": 100.0, "last_touched": 100.0,
             "created_at": 50.0, "intent": "x", "root_anchor": None,
             "parent_concept_id": None, "use_count": 0,
             "activation": -10.0,
             "abandoned_tombstone_tx": "t1",
             "abandonment_reason": "below_floor"},
            {"fork_id": "new_tomb", "status": "abandoned",
             "abandoned_at": 500.0, "last_touched": 500.0,
             "created_at": 50.0, "intent": "y", "root_anchor": None,
             "parent_concept_id": None, "use_count": 0,
             "activation": -10.0,
             "abandoned_tombstone_tx": "t2",
             "abandonment_reason": "below_floor"},
        ],
        "summary": {"open": 0, "graduated": 0, "abandoned": 2},
    }
    _write_snapshot(fresh_data_dir, payload)
    r = handlers.get_synthesis_fork_tombstones(since=200.0)
    assert r["total"] == 1
    assert r["tombstones"][0]["fork_id"] == "new_tomb"


# ── Summary endpoint ─────────────────────────────────────────────


def test_summary_returns_embedded_dict(fresh_data_dir):
    payload = {
        "version": 1, "exported_at": time.time(),
        "forks": [],
        "summary": {"open": 5, "graduated": 12, "abandoned": 3},
    }
    _write_snapshot(fresh_data_dir, payload)
    r = handlers.get_synthesis_fork_summary()
    assert r["summary"] == {"open": 5, "graduated": 12, "abandoned": 3}
    assert r["snapshot"] == "ok"


# ── v6 ROUTE_TABLE wiring ────────────────────────────────────────


def test_v6_manifest_includes_fork_routes():
    """The Phase 5 routes are in v6.py's _T tuple."""
    from titan_hcl.api import v6
    paths = [row[0] for row in v6._T]
    assert "/v6/synthesis/forks" in paths
    assert "/v6/synthesis/forks/summary" in paths
    assert "/v6/synthesis/forks/tombstones" in paths
    assert "/v6/synthesis/forks/{fork_id}" in paths


def test_dashboard_exposes_v6_aliases():
    from titan_hcl.api import dashboard
    assert callable(dashboard.get_v6_synthesis_forks)
    assert callable(dashboard.get_v6_synthesis_fork)
    assert callable(dashboard.get_v6_synthesis_fork_tombstones)
    assert callable(dashboard.get_v6_synthesis_fork_summary)
