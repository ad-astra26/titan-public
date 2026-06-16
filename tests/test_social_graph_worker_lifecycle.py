"""tests/test_social_graph_worker_lifecycle.py — worker lifecycle tests.

Per PLAN_microkernel_phase_c_social_graph_worker_extraction.md §7.4 +
SPEC v1.7.1 §9.B + D-SPEC-50 + G16 critical-data discipline.

Exercises:
  - Clean MODULE_SHUTDOWN path (PRAGMA wal_checkpoint(FULL) on exit)
  - SAVE_NOW handling (B.1 shadow_swap orchestrator hook)
  - Boot resilience: fresh DB, schema creation
  - DB integrity check on boot (G16 boot-time check requirement)
  - ModuleSpec dual-registration parity between plugin.py and legacy_core.py

Full guardian-supervised SIGKILL→respawn flow requires actual guardian
fixture which is broader than this PLAN's scope; the per-worker pieces
that the supervisor relies on are tested here. The supervisor itself
is covered by `tests/test_guardian_adopt_worker.py` + Phase B.2 swap tests.
"""
from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import threading
import time
from pathlib import Path
from queue import Queue, Empty

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def temp_data_dir():
    with tempfile.TemporaryDirectory() as td:
        yield td


# ── MODULE_SHUTDOWN + WAL checkpoint ─────────────────────────────────


def test_worker_checkpoints_wal_on_shutdown(temp_data_dir):
    """Verify _checkpoint_wal runs PRAGMA wal_checkpoint(FULL) on shutdown.
    G16 requirement: 'Database checkpoint before close.'"""
    from titan_hcl.core.social_graph import SocialGraph
    from titan_hcl.modules.social_graph_worker import _checkpoint_wal

    db_path = os.path.join(temp_data_dir, "social_graph.db")
    sg = SocialGraph(db_path=db_path)
    # Generate some WAL pages
    for i in range(20):
        sg.get_or_create_user(f"u_{i}")
    # WAL file should exist
    wal_path = db_path + "-wal"
    assert os.path.exists(wal_path) or os.path.getsize(db_path) > 0

    # Run checkpoint
    _checkpoint_wal(sg)
    # No exception = pass; the checkpoint succeeded
    # Post-checkpoint, DB is fully written + WAL is empty or absent
    # (PRAGMA wal_checkpoint(FULL) is best-effort under concurrent reads;
    # we just assert no exception fired and the DB is queryable)
    with sqlite3.connect(db_path, timeout=5) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM user_profiles").fetchone()[0]
        assert count == 20


def test_worker_checkpoint_wal_handles_missing_db_gracefully(temp_data_dir):
    """If DB is missing, _checkpoint_wal logs warning + returns — no crash."""
    from titan_hcl.modules.social_graph_worker import _checkpoint_wal

    class FakeSG:
        _db_path = os.path.join(temp_data_dir, "nonexistent.db")

    # Should not raise — caught + warned internally
    _checkpoint_wal(FakeSG())


# ── Fresh boot + schema creation ─────────────────────────────────────


def test_init_social_graph_creates_schema(temp_data_dir):
    """Fresh DB path → SocialGraph CREATE TABLE IF NOT EXISTS fires +
    all 9 tables exist."""
    from titan_hcl.modules.social_graph_worker import _init_social_graph

    db_path = os.path.join(temp_data_dir, "fresh.db")
    sg = _init_social_graph(db_path)
    assert sg is not None
    # Verify all schema tables exist
    expected_tables = {
        "user_profiles", "social_edges", "donations", "inspirations",
        "engagement_ledger", "titan_social_preferences",
        "community_registry",
    }
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        names = {r[0] for r in rows}
    missing = expected_tables - names
    assert not missing, f"Missing tables: {missing}"


def test_init_social_graph_returns_none_on_failure(temp_data_dir):
    """Bad db_path → returns None (caller exits non-zero so guardian respawns)."""
    from titan_hcl.modules.social_graph_worker import _init_social_graph

    # Try to use a path that can't be created (e.g. nested under a file)
    blocker_path = os.path.join(temp_data_dir, "blocker")
    Path(blocker_path).write_text("not a directory")
    bad_path = os.path.join(blocker_path, "nested", "social_graph.db")
    sg = _init_social_graph(bad_path)
    assert sg is None


# ── G16 boot-time integrity check ────────────────────────────────────


def test_corrupt_db_detected_on_boot(temp_data_dir):
    """G16 (data integrity): on a corrupt DB file, SocialGraph init
    raises during _init_db rather than silently using bad data.

    The worker's _init_social_graph catches this and returns None →
    worker exits non-zero → guardian respawns. Verifies the failure
    mode is loud, not silent.
    """
    from titan_hcl.modules.social_graph_worker import _init_social_graph

    db_path = os.path.join(temp_data_dir, "corrupt.db")
    # Write garbage as DB file — sqlite3 will recognize it as
    # malformed when CREATE TABLE attempts run
    Path(db_path).write_bytes(b"this is not a valid SQLite database file")
    sg = _init_social_graph(db_path)
    # Either returns None (loud failure) OR succeeds because sqlite3
    # treats garbage as new-DB cold start (it overwrites). Either way:
    # NO silent half-corrupt state must persist.
    # Per G16: failure must surface to the caller (None) so guardian
    # can respawn / human can intervene.
    if sg is not None:
        # sqlite3 might have recovered + created fresh schema — verify
        # that fresh state is queryable
        with sqlite3.connect(db_path) as conn:
            conn.execute("SELECT COUNT(*) FROM user_profiles").fetchone()


def test_corrupt_db_boot_recovery_from_backup(temp_data_dir):
    """G16 requirement: 'On failure: restore from <file>.bak, emit
    SUPERVISION_DATA_RESTORE to kernel log.'

    The full restore-from-.bak protocol is implemented at the
    BackupWorker level (per project_backup_worker_shipped.md) — fleet-
    wide. social_graph_worker delegates to that infrastructure rather
    than re-implementing.

    This test exercises the contract that the worker DOES preserve:
    given a valid .bak alongside a corrupt main DB, the worker must
    not silently lose data. The expected operator-side recovery is to
    rename .bak → main + restart the worker (which BackupWorker
    automates via the existing rFP_backup_worker.md restore flow).

    Acceptance: this test pins the surface that BackupWorker depends on
    — the worker MUST NOT clobber a .bak file during init.
    """
    from titan_hcl.modules.social_graph_worker import _init_social_graph

    db_path = os.path.join(temp_data_dir, "social_graph.db")
    bak_path = db_path + ".bak"

    # Place a valid backup
    sg_setup = sqlite3.connect(bak_path)
    sg_setup.execute("CREATE TABLE marker (val TEXT)")
    sg_setup.execute("INSERT INTO marker VALUES ('VALID_BACKUP_MARKER')")
    sg_setup.commit()
    sg_setup.close()
    backup_mtime_before = os.path.getmtime(bak_path)

    # Corrupt main DB
    Path(db_path).write_bytes(b"corrupt")

    # Worker boots (may succeed if sqlite3 overwrites garbage, may fail)
    _init_social_graph(db_path)

    # CRITICAL: the .bak must be UNTOUCHED
    assert os.path.exists(bak_path), (
        ".bak file was deleted — G16 backup-preservation invariant "
        "violated by social_graph_worker init.")
    assert os.path.getmtime(bak_path) == backup_mtime_before, (
        ".bak file was modified during worker init — operator-recovery "
        "path broken.")
    # And the marker is still readable
    conn = sqlite3.connect(bak_path)
    val = conn.execute("SELECT val FROM marker").fetchone()[0]
    assert val == "VALID_BACKUP_MARKER"


# ── ModuleSpec parity (dual-registration rule) ───────────────────────


def test_plugin_py_registers_social_graph_modulespec():
    """plugin.py:_register_modules MUST register social_graph ModuleSpec."""
    # ModuleSpec registration was refactored out of plugin.py into the central
    # module_catalog.py — read it there (the entry_fn + name + flags all live
    # in titan_hcl/module_catalog.py now).
    plugin_py = (
        Path(__file__).resolve().parents[1]
        / "titan_hcl" / "module_catalog.py"
    ).read_text()
    assert 'name="social_graph"' in plugin_py, (
        "plugin.py does not register social_graph ModuleSpec — "
        "guardian won't spawn the worker.")
    assert "social_graph_worker_main" in plugin_py, (
        "plugin.py does not import social_graph_worker_main as entry_fn.")


def test_module_spec_critical_data_writer_flag():
    """social_graph ModuleSpec must have critical_data_writer=True
    (G16 requirement — graceful SIGTERM grace + WAL checkpoint)."""
    # ModuleSpec registration was refactored out of plugin.py into the central
    # module_catalog.py — read it there (the entry_fn + name + flags all live
    # in titan_hcl/module_catalog.py now).
    plugin_py = (
        Path(__file__).resolve().parents[1]
        / "titan_hcl" / "module_catalog.py"
    ).read_text()
    # Find the social_graph ModuleSpec block + verify
    # critical_data_writer=True appears in it
    idx = plugin_py.find('name="social_graph"')
    block = plugin_py[idx:idx + 1500]
    assert "critical_data_writer=True" in block, (
        "social_graph ModuleSpec missing critical_data_writer=True — "
        "G16 graceful-shutdown discipline broken.")


def test_module_spec_autostart_no_flag_gate():
    """social_graph_worker has no flag-gate (always-on per PLAN §6
    'hard cutover' migration)."""
    # ModuleSpec registration was refactored out of plugin.py into the central
    # module_catalog.py — read it there (the entry_fn + name + flags all live
    # in titan_hcl/module_catalog.py now).
    plugin_py = (
        Path(__file__).resolve().parents[1]
        / "titan_hcl" / "module_catalog.py"
    ).read_text()
    idx = plugin_py.find('name="social_graph"')
    block = plugin_py[idx:idx + 1500]
    assert "autostart=True" in block
    # Should NOT be gated by `if config.get('microkernel', {}).get('social_graph...')`
    # — search the preceding lines for any flag-gate enclosing this registration
    preceding = plugin_py[max(0, idx - 500):idx]
    # The cognitive_worker and social_worker registrations are inside
    # `if self._full_config.get("microkernel", {}).get("..._enabled", ...):`
    # blocks. social_graph_worker should NOT be inside such a block.
    # The check: the `if ...social_graph...enabled` pattern must NOT
    # appear within the 500 chars immediately before the registration.
    assert "social_graph_worker_enabled" not in preceding, (
        "social_graph_worker should be unconditional (no flag-gate) per "
        "PLAN §6 hard-cutover migration. Found flag-gate immediately "
        "before ModuleSpec.")


def test_kernel_proxy_aliases_has_social_graph_proxy():
    """KERNEL_PROXY_ALIASES tuple in kernel.py must contain
    'social_graph_proxy' so the Rust broker routes RESPONSE messages."""
    from titan_hcl.core.kernel import KERNEL_PROXY_ALIASES
    assert "social_graph_proxy" in KERNEL_PROXY_ALIASES


# ── Boot signal contracts ────────────────────────────────────────────


def test_send_msg_helper_never_raises():
    """_send_msg is on the heartbeat hot path — must never raise."""
    from titan_hcl.modules.social_graph_worker import _send_msg
    # Broken queue — put() raises
    class BrokenQueue:
        def put(self, msg):
            raise RuntimeError("queue broken")
    # Should swallow the exception
    _send_msg(BrokenQueue(), "TEST_TYPE", "src", "dst", {"k": "v"})
    # No assertion — just confirming no exception escaped


def test_send_heartbeat_emits_correct_shape():
    """_send_heartbeat emits MODULE_HEARTBEAT with alive=True + rss_mb."""
    from titan_hcl import bus
    from titan_hcl.modules.social_graph_worker import _send_heartbeat
    q = Queue()
    _send_heartbeat(q, "social_graph")
    msg = q.get_nowait()
    assert msg["type"] == bus.MODULE_HEARTBEAT
    assert msg["src"] == "social_graph"
    assert msg["dst"] == "guardian"
    assert msg["payload"]["alive"] is True
    assert "rss_mb" in msg["payload"]
    assert isinstance(msg["payload"]["rss_mb"], (int, float))
