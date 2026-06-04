"""Phase 8.X — P5 fork-snapshot write-through fold-in (D-SPEC-PHASE8).

Closes the P5 cascade flake: "new fork visible in snapshot never appeared
after 6s." The 60s recompute-loop snapshot was the only refresh trigger;
P8.X adds synchronous snapshot_export after every lifecycle mutator.

Covers:
- create_fork → snapshot visible within ms (not seconds)
- record_exploration_tx → snapshot visible within ms
- abandon → snapshot reflects status="abandoned" + tombstone_tx
- snapshot_path=None disables write-through (back-compat for existing callers)
- snapshot_export() raise does NOT propagate to caller (soft-fail)
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock

import duckdb
import pytest


# ── Fixtures ───────────────────────────────────────────────────────────


def _make_writer():
    w = MagicMock()
    w.write_concept_version_with_proof = MagicMock(return_value="cv_tx")
    w.write_tombstone = MagicMock(return_value="tombstone_tx_h")
    return w


def _make_kuzu_graph():
    g = MagicMock()
    g.fork_create_node = MagicMock(return_value=True)
    g.fork_update_status = MagicMock(return_value=True)
    g.fork_record_explores_edge = MagicMock(return_value=True)
    return g


def _make_engram_store():
    cs = MagicMock()
    cs.create_concept = MagicMock(return_value=MagicMock(
        anchor_tx="cv_anchor", version=1, name="x",
        memory_type="declarative", concept_id="abc", groundedness=0.5,
    ))
    cs.bump_version = MagicMock()
    cs.spine_get_latest_concept = MagicMock(return_value={"version": 5, "anchor_tx": "p_anchor"})
    return cs


def _make_activation_store():
    a = MagicMock()
    a.record_access = MagicMock()
    return a


@pytest.fixture()
def store_with_writethrough(tmp_path):
    from titan_hcl.synthesis.hypothesis_fork_store import HypothesisForkStore
    conn = duckdb.connect(":memory:")
    snap = tmp_path / "forks_snapshot.json"
    s = HypothesisForkStore(
        duckdb_conn=conn,
        kuzu_graph=_make_kuzu_graph(),
        engram_store=_make_engram_store(),
        outer_memory_writer=_make_writer(),
        activation_store=_make_activation_store(),
        snapshot_path=str(snap),
    )
    return s, str(snap)


@pytest.fixture()
def store_no_writethrough(tmp_path):
    from titan_hcl.synthesis.hypothesis_fork_store import HypothesisForkStore
    conn = duckdb.connect(":memory:")
    return HypothesisForkStore(
        duckdb_conn=conn,
        kuzu_graph=_make_kuzu_graph(),
        engram_store=_make_engram_store(),
        outer_memory_writer=_make_writer(),
        activation_store=_make_activation_store(),
        # snapshot_path=None (default) — no write-through
    )


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ── create_fork triggers snapshot ──────────────────────────────────────


def test_create_fork_write_through_emits_snapshot_immediately(store_with_writethrough):
    store, snap_path = store_with_writethrough
    assert not os.path.exists(snap_path) or _load(snap_path)["forks"] == []
    t0 = time.time()
    fork_id = store.create_fork(intent="test create fork")
    elapsed = time.time() - t0
    assert os.path.exists(snap_path)
    payload = _load(snap_path)
    fork_ids = [f["fork_id"] for f in payload.get("forks", [])]
    assert fork_id in fork_ids
    # Must be way under any 60s recompute tick — closes the "6s flake"
    assert elapsed < 2.0


def test_record_exploration_tx_emits_snapshot(store_with_writethrough):
    store, snap_path = store_with_writethrough
    fork_id = store.create_fork(intent="test record tx")
    # Clear existing snapshot to confirm record_exploration_tx triggers a fresh write
    os.remove(snap_path)
    store.record_exploration_tx(fork_id, "tx_aaaa")
    assert os.path.exists(snap_path)


def test_abandon_emits_snapshot_with_status(store_with_writethrough):
    store, snap_path = store_with_writethrough
    fork_id = store.create_fork(intent="test abandon")
    store.abandon(fork_id=fork_id, reason="test_reason")
    payload = _load(snap_path)
    matching = [f for f in payload.get("forks", []) if f["fork_id"] == fork_id]
    assert len(matching) == 1
    assert matching[0]["status"] == "abandoned"
    assert matching[0]["abandoned_tombstone_tx"] == "tombstone_tx_h"


# ── No write-through path (back-compat) ────────────────────────────────


def test_no_snapshot_path_no_write_through(store_no_writethrough, tmp_path):
    """Without snapshot_path, the store does NOT write any snapshot on create."""
    fork_id = store_no_writethrough.create_fork(intent="no write-through")
    # No path was supplied, so no file should appear in the cwd or tmp_path
    assert not (tmp_path / "forks_snapshot.json").exists()
    # And the fork was created successfully
    assert store_no_writethrough.get_fork(fork_id) is not None


# ── Snapshot export error is soft-failed ───────────────────────────────


def test_snapshot_export_exception_is_swallowed(tmp_path, monkeypatch):
    """If export_snapshot raises (disk full, permission denied, ...), the
    lifecycle mutator must NOT propagate — the DuckDB transaction is
    already committed."""
    from titan_hcl.synthesis.hypothesis_fork_store import HypothesisForkStore
    conn = duckdb.connect(":memory:")
    store = HypothesisForkStore(
        duckdb_conn=conn,
        kuzu_graph=_make_kuzu_graph(),
        engram_store=_make_engram_store(),
        outer_memory_writer=_make_writer(),
        activation_store=_make_activation_store(),
        snapshot_path=str(tmp_path / "forks_snapshot.json"),
    )
    # Monkey-patch export_snapshot to raise
    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(store, "export_snapshot", boom)
    # Should NOT raise
    fork_id = store.create_fork(intent="snapshot-fails")
    assert store.get_fork(fork_id) is not None
