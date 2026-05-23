"""BridgeRecall tests — watermark-gated cross-process activation read.

D-SPEC-123 / SPEC v1.56.0 §25 INV-Syn-4 / G18. Validates the soft-fail
discipline (NEVER raises to the caller) + the watermark freshness gate.
"""
from __future__ import annotations

import os
import time

import pytest

from titan_hcl.synthesis.bridge_recall import (
    DEFAULT_FRESHNESS_WINDOW_S,
    BridgeRecall,
)


@pytest.fixture
def isolated_shm(monkeypatch, tmp_path):
    shm_dir = tmp_path / "shm"
    monkeypatch.setenv("TITAN_SHM_ROOT", str(shm_dir))
    yield shm_dir


@pytest.fixture
def fresh_db_path(tmp_path):
    """Create synthesis.duckdb with activation_state schema initialized.

    Post-relocation 2026-05-23: activation_state lives in synthesis.duckdb
    (synthesis_worker's R/W territory per G21 / INV-Syn-3), NOT in
    titan_memory.duckdb. BridgeRecall opens this file R/O cross-process.
    """
    from titan_hcl.modules.synthesis_worker import ActivationStore
    db_path = tmp_path / "synthesis.duckdb"
    store = ActivationStore(str(db_path))
    store.close()
    return str(db_path)


# ─────────────────────────────────────────────────────────────────────────
# Watermark reading
# ─────────────────────────────────────────────────────────────────────────

def test_read_watermark_returns_none_when_slot_missing(isolated_shm, fresh_db_path):
    """When synthesis_worker hasn't booted, the SHM slot doesn't exist —
    read_watermark returns None (soft-fail signal)."""
    br = BridgeRecall(titan_id="test", db_path=fresh_db_path)
    try:
        wm = br.read_watermark()
        assert wm is None
    finally:
        br.close()


def test_is_fresh_false_when_no_watermark(isolated_shm, fresh_db_path):
    """is_fresh(now) returns False when slot missing → caller degrades."""
    br = BridgeRecall(titan_id="test", db_path=fresh_db_path)
    try:
        assert br.is_fresh(time.time()) is False
    finally:
        br.close()


def test_read_watermark_after_writer_publish(isolated_shm, fresh_db_path):
    """End-to-end: writer publishes, reader reads the same payload."""
    from titan_hcl.modules.synthesis_worker import SynthStatusWriter
    writer = SynthStatusWriter(titan_id="test")
    try:
        now = time.time()
        writer.publish(
            last_consistent_event_ts=now,
            last_recompute_ts=now,
            items_tracked=42,
            recompute_count_increment=7,
        )
    finally:
        writer.close()

    br = BridgeRecall(titan_id="test", db_path=fresh_db_path)
    try:
        wm = br.read_watermark()
        assert wm is not None
        last_consistent, last_recompute, items, count = wm
        assert last_consistent == pytest.approx(now)
        assert last_recompute == pytest.approx(now)
        assert items == 42
        assert count == 7
    finally:
        br.close()


def test_is_fresh_true_within_window(isolated_shm, fresh_db_path):
    """Watermark within freshness_window_s → is_fresh True."""
    from titan_hcl.modules.synthesis_worker import SynthStatusWriter
    writer = SynthStatusWriter(titan_id="test")
    try:
        now = time.time()
        writer.publish(last_consistent_event_ts=now, last_recompute_ts=now,
                       items_tracked=1, recompute_count_increment=1)
    finally:
        writer.close()

    br = BridgeRecall(titan_id="test", db_path=fresh_db_path)
    try:
        assert br.is_fresh(now + 100) is True   # 100s after publish, well within 300s
    finally:
        br.close()


def test_is_fresh_false_outside_window(isolated_shm, fresh_db_path):
    """Watermark older than freshness_window_s → is_fresh False."""
    from titan_hcl.modules.synthesis_worker import SynthStatusWriter
    writer = SynthStatusWriter(titan_id="test")
    try:
        old = time.time() - 3600   # 1 hour ago
        writer.publish(last_consistent_event_ts=old, last_recompute_ts=old,
                       items_tracked=1, recompute_count_increment=1)
    finally:
        writer.close()

    br = BridgeRecall(titan_id="test", db_path=fresh_db_path,
                     freshness_window_s=300.0)
    try:
        assert br.is_fresh(time.time()) is False
    finally:
        br.close()


# ─────────────────────────────────────────────────────────────────────────
# Activation lookup — soft-fail
# ─────────────────────────────────────────────────────────────────────────

def test_activation_lookup_returns_empty_when_watermark_missing(
    isolated_shm, fresh_db_path,
):
    """No watermark → empty dict, never raises."""
    br = BridgeRecall(titan_id="test", db_path=fresh_db_path)
    try:
        out = br.activation_lookup(["kuzu:1", "kuzu:2"])
        assert out == {}
    finally:
        br.close()


def test_activation_lookup_returns_empty_when_db_missing(
    isolated_shm, tmp_path,
):
    """DB file missing → empty dict, never raises."""
    from titan_hcl.modules.synthesis_worker import SynthStatusWriter
    writer = SynthStatusWriter(titan_id="test")
    try:
        writer.publish(last_consistent_event_ts=time.time(),
                       last_recompute_ts=time.time(),
                       items_tracked=0, recompute_count_increment=1)
    finally:
        writer.close()

    nonexistent = str(tmp_path / "does_not_exist.duckdb")
    br = BridgeRecall(titan_id="test", db_path=nonexistent)
    try:
        out = br.activation_lookup(["kuzu:1"])
        assert out == {}
    finally:
        br.close()


def test_activation_lookup_empty_input_returns_empty(isolated_shm, fresh_db_path):
    """Empty input → empty output, no DB hit."""
    br = BridgeRecall(titan_id="test", db_path=fresh_db_path)
    try:
        assert br.activation_lookup([]) == {}
    finally:
        br.close()


def test_activation_lookup_returns_rows_when_present(isolated_shm, fresh_db_path):
    """Happy path: writer published, items in activation_state → reader
    returns base_level for those items."""
    # Pre-populate activation_state via a separate writer connection.
    import duckdb
    con = duckdb.connect(fresh_db_path)
    now = time.time()
    con.execute(
        "INSERT INTO activation_state (item_id, base_level, last_access, "
        "access_count, first_access, last_recompute) VALUES "
        "(?, ?, ?, ?, ?, ?), (?, ?, ?, ?, ?, ?)",
        ("kuzu:1", 1.5, now, 3, now - 100, now,
         "kuzu:2", -0.7, now, 1, now - 50, now),
    )
    con.close()

    from titan_hcl.modules.synthesis_worker import SynthStatusWriter
    writer = SynthStatusWriter(titan_id="test")
    try:
        writer.publish(last_consistent_event_ts=now, last_recompute_ts=now,
                       items_tracked=2, recompute_count_increment=1)
    finally:
        writer.close()

    br = BridgeRecall(titan_id="test", db_path=fresh_db_path)
    try:
        out = br.activation_lookup(["kuzu:1", "kuzu:2", "kuzu:absent"])
        assert out["kuzu:1"] == pytest.approx(1.5)
        assert out["kuzu:2"] == pytest.approx(-0.7)
        assert "kuzu:absent" not in out
    finally:
        br.close()


def test_activation_lookup_returns_empty_when_watermark_stale(
    isolated_shm, fresh_db_path,
):
    """Even if activation_state has rows, a stale watermark = no trust
    → empty dict (caller degrades to cosine-only)."""
    import duckdb
    con = duckdb.connect(fresh_db_path)
    con.execute(
        "INSERT INTO activation_state (item_id, base_level) VALUES (?, ?)",
        ("kuzu:1", 1.0),
    )
    con.close()

    from titan_hcl.modules.synthesis_worker import SynthStatusWriter
    writer = SynthStatusWriter(titan_id="test")
    try:
        old = time.time() - 3600
        writer.publish(last_consistent_event_ts=old, last_recompute_ts=old,
                       items_tracked=1, recompute_count_increment=1)
    finally:
        writer.close()

    br = BridgeRecall(titan_id="test", db_path=fresh_db_path,
                     freshness_window_s=300.0)
    try:
        out = br.activation_lookup(["kuzu:1"])
        assert out == {}
    finally:
        br.close()


def test_activation_lookup_never_raises(isolated_shm, fresh_db_path):
    """Defensive: malformed item_ids must not raise — just return empty.
    Pin this — the hot retrieval path depends on it."""
    br = BridgeRecall(titan_id="test", db_path=fresh_db_path)
    try:
        # Empty strings, weird chars, etc.
        out = br.activation_lookup(["", "kuzu:", "x;DROP TABLE--"])
        # Either empty (watermark missing → soft-fail) OR returns rows
        # (none of these will match). Critical: no exception.
        assert isinstance(out, dict)
    finally:
        br.close()


# ─────────────────────────────────────────────────────────────────────────
# Singleton accessor
# ─────────────────────────────────────────────────────────────────────────

def test_get_bridge_recall_returns_singleton(isolated_shm, fresh_db_path,
                                              monkeypatch):
    """get_bridge_recall returns the same instance on repeated calls
    within a process."""
    # Reset the module-level singleton for this test.
    import titan_hcl.synthesis.bridge_recall as br_mod
    monkeypatch.setattr(br_mod, "_bridge_recall_singleton", None)

    a = br_mod.get_bridge_recall(titan_id="test")
    b = br_mod.get_bridge_recall(titan_id="test")
    assert a is b
