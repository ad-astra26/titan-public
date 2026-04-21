"""rFP_observatory_writer_service Phase 1+2 integration tests.

Phase 1 = shadow mode (writes to BOTH primary direct AND shadow via writer).
Phase 2 = canary canonical (writes for selected tables go ONLY through writer).

These tests cover the routing:
  - direct path when no writer (default)
  - writer-routed when client present
  - per-table cutover via tables_canonical
  - fall-back to direct on writer error (data loss prevention)
"""

from __future__ import annotations

import sqlite3
import tempfile
from unittest.mock import MagicMock

import pytest

from titan_plugin.utils.observatory_db import ObservatoryDB


@pytest.fixture
def fresh_db():
    """A fresh ObservatoryDB on a temp file with NO writer (direct path)."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db = ObservatoryDB(db_path=f.name)
        yield db


def test_default_no_writer_direct_path(fresh_db):
    """Default constructor: no writer client; writes go direct via _lock."""
    assert fresh_db._writer is None
    fresh_db.record_event("test_event", "smoke", {"k": "v"})
    fresh_db.record_vital_snapshot(sovereignty_pct=0.7)
    events = fresh_db.get_events()
    assert len(events) == 1
    assert events[0]["event_type"] == "test_event"
    assert len(fresh_db.get_vital_history(days=1)) == 1


def test_route_write_uses_writer_when_present():
    """If a writer client is injected, _route_write delegates to it."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        mock_writer = MagicMock()
        db = ObservatoryDB(db_path=f.name, writer_client=mock_writer)
    db.record_event("hello", "world")
    assert mock_writer.write.called, "writer.write must be called when client present"
    call_args = mock_writer.write.call_args
    assert "INSERT INTO event_log" in call_args.args[0]
    assert call_args.kwargs.get("table") == "event_log"


def test_writer_failure_propagates_phase_4_sunset():
    """Phase 4 sunset (2026-04-21): writer is the SOLE write path when enabled.
    Errors propagate to caller — no silent direct-path fallback. Loud failure
    beats silent data divergence (which would create primary↔writer drift).

    Replaces the previous fail-open test (which existed during shadow/canary
    Phases 1-2 as a safety net). Sunset gated by 30-min canonical soak with
    zero errors on 25,107 writes across all 3 Titans.
    """
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        bad_writer = MagicMock()
        bad_writer.write.side_effect = RuntimeError("simulated writer outage")
        db = ObservatoryDB(db_path=f.name, writer_client=bad_writer)
    with pytest.raises(RuntimeError, match="simulated writer outage"):
        db.record_event("after_outage", "should raise, not silently fall back")
    # Confirm the direct path was NOT used (no event recorded directly).
    events = db.get_events()
    assert events == [], (
        "Phase 4 sunset: writer error must NOT silently fall back to direct path. "
        f"Got {len(events)} events recorded directly — fail-open is supposed to be "
        "removed.")


def test_each_record_method_passes_correct_table_name():
    """Each record_* method must call _route_write with the right `table` arg.
    (Per-table canonical cutover relies on this for IMWConfig.is_table_canonical())."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        mock_writer = MagicMock()
        db = ObservatoryDB(db_path=f.name, writer_client=mock_writer)

    expected_tables = {
        "record_vital_snapshot": "vital_snapshots",
        "record_trinity_snapshot": "trinity_snapshots",
        "record_growth_snapshot": "growth_snapshots",
        "record_expressive": "expressive_archive",
        "record_event": "event_log",
        "record_guardian_action": "guardian_log",
        "record_reflex": "reflex_log",
        "record_v4_snapshot": "v4_snapshots",
    }

    # Build minimal kwargs to call each method
    call_kwargs = {
        "record_vital_snapshot": {},
        "record_trinity_snapshot": {
            "body_tensor": [0.5] * 5, "mind_tensor": [0.5] * 5, "spirit_tensor": [0.5] * 5},
        "record_growth_snapshot": {},
        "record_expressive": {"type_": "art"},
        "record_event": {"event_type": "x"},
        "record_guardian_action": {"tier": "1", "action": "warn"},
        "record_reflex": {"reflex_type": "test"},
        "record_v4_snapshot": {},
    }

    for method_name, table in expected_tables.items():
        mock_writer.write.reset_mock()
        getattr(db, method_name)(**call_kwargs[method_name])
        assert mock_writer.write.called, f"{method_name} did not call writer.write"
        assert mock_writer.write.call_args.kwargs.get("table") == table, (
            f"{method_name} passed table={mock_writer.write.call_args.kwargs.get('table')!r}, "
            f"expected {table!r}")


def test_writes_with_writer_do_not_hold_observatory_lock():
    """When the writer is in play we must NOT acquire ObservatoryDB._lock —
    writer service serializes internally; double-locking adds contention."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        mock_writer = MagicMock()
        db = ObservatoryDB(db_path=f.name, writer_client=mock_writer)
    # Acquire the lock externally; if _route_write tried to also acquire it we'd deadlock.
    with db._lock:
        db.record_event("inside-lock", "must not deadlock")
    assert mock_writer.write.called


def test_prune_routes_through_writer_when_enabled():
    """rFP_observatory_writer_service post-deploy fix (2026-04-21):
    prune_old_data() previously held a direct lock on observatory.db AND
    ran VACUUM at the end. On the 1.88 GB observatory.db this blocked
    the writer service for minutes. Fix: when writer is enabled, route
    DELETEs through it AND skip VACUUM (writer-side maintenance).
    """
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        mock_writer = MagicMock()
        db = ObservatoryDB(db_path=f.name, writer_client=mock_writer)
    db.prune_old_data(max_days=90)
    # Should call writer.write exactly 7 times — one DELETE per pruned table
    delete_calls = [c for c in mock_writer.write.call_args_list
                    if "DELETE FROM" in c.args[0]]
    assert len(delete_calls) == 7, (
        f"expected 7 DELETE calls (one per pruned table), got {len(delete_calls)}")
    # Should NOT call writer.write with "VACUUM" in SQL
    vacuum_calls = [c for c in mock_writer.write.call_args_list
                    if "VACUUM" in c.args[0].upper()]
    assert vacuum_calls == [], (
        "VACUUM must not run when writer is enabled — it blocks writer for "
        "minutes on large DBs. Writer-side maintenance handles compaction.")


def test_prune_direct_path_keeps_vacuum_for_no_writer_case():
    """When writer is disabled (default), prune still runs VACUUM directly
    — preserves the original behavior for non-writer deployments."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db = ObservatoryDB(db_path=f.name)
        assert db._writer is None
    # No mock = exercises the real direct path including VACUUM.
    # If VACUUM was broken, this would raise. Just verify it completes.
    db.prune_old_data(max_days=365)


def test_init_auto_construct_writer_when_config_enabled(monkeypatch, tmp_path):
    """When [persistence_observatory].enabled=true, ObservatoryDB.__init__
    auto-constructs a writer client (without needing the caller to pass one)."""
    from titan_plugin.persistence import config as cfg_mod

    fake_cfg = cfg_mod.IMWConfig(
        enabled=True, mode="shadow",
        socket_path=str(tmp_path / "obs.sock"),
        wal_path=str(tmp_path / "obs.wal"),
        journal_dir=str(tmp_path),
        db_path=str(tmp_path / "primary.db"),
        shadow_db_path=str(tmp_path / "shadow.db"),
    )
    monkeypatch.setattr(
        cfg_mod.IMWConfig, "from_titan_config_section",
        classmethod(lambda cls, section_name="persistence": fake_cfg),
    )

    fake_client_class = MagicMock()
    fake_client_instance = MagicMock()
    fake_client_class.return_value = fake_client_instance
    monkeypatch.setattr(
        "titan_plugin.persistence.writer_client.InnerMemoryWriterClient",
        fake_client_class,
    )

    db = ObservatoryDB(db_path=str(tmp_path / "primary.db"))
    assert db._writer is fake_client_instance, (
        "__init__ should have constructed a writer client from "
        "[persistence_observatory] when enabled=true")
    assert fake_client_class.call_args.kwargs.get("caller_name") == "observatory_db"
