"""rFP_universal_sqlite_writer Phase 3 — end-to-end canonical mode test.

Verifies the close of BUG-TRINITY-SNAPSHOT-DB-LOCKED:
  - ObservatoryDB constructed with canonical-mode config + the writer client
  - real IMWDaemon running against a temp DB on a temp socket
  - 100 record_trinity_snapshot() calls all land via writer (no direct writes)
  - rowcount in the primary DB matches what was written
  - no SQLite "database is locked" errors during the run

If this test passes, the canonical-mode routing matches what we ship to
production for the bug fix.
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from titan_plugin.persistence.config import IMWConfig
from titan_plugin.persistence.writer_client import InnerMemoryWriterClient
from titan_plugin.persistence.writer_service import IMWDaemon
from titan_plugin.utils.observatory_db import ObservatoryDB


def _spawn_daemon(cfg: IMWConfig):
    """Spawn an IMWDaemon in a background asyncio thread."""
    loop = asyncio.new_event_loop()
    daemon = IMWDaemon(cfg)
    ready = threading.Event()
    stop_event_ref: list = [None]

    def _run():
        asyncio.set_event_loop(loop)
        stop = asyncio.Event()
        stop_event_ref[0] = stop

        async def _inner():
            await daemon.start()
            ready.set()
            while not stop.is_set():
                await asyncio.sleep(0.05)
            await daemon.stop()

        loop.run_until_complete(_inner())
        loop.close()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    if not ready.wait(timeout=10.0):
        raise RuntimeError("observatory writer daemon failed to start")
    time.sleep(0.15)

    def _stop():
        def _trigger():
            if stop_event_ref[0] is not None:
                stop_event_ref[0].set()
        loop.call_soon_threadsafe(_trigger)
        t.join(timeout=10.0)

    return daemon, _stop


@pytest.fixture
def canonical_cfg(tmp_path):
    """Observatory IMWConfig in canonical mode with all 14 hot tables listed.

    Mirrors production config.toml [persistence_observatory] post-2026-04-27.
    """
    db = tmp_path / "observatory.db"
    shadow = tmp_path / "observatory_shadow.db"
    cfg = IMWConfig.from_dict({
        "enabled": True,
        "mode": "canonical",
        "fast_path_enabled": False,
        "tables_canonical": [
            "trinity_snapshots", "growth_snapshots", "vital_snapshots",
            "event_log", "expressive_archive", "guardian_log",
            "v4_snapshots", "reflex_log", "neuromod_history",
            "hormonal_history", "expression_history", "dreaming_history",
            "training_history", "clock_history",
        ],
        "socket_path": str(tmp_path / "obs_writer.sock"),
        "wal_path": str(tmp_path / "obs_writer.wal"),
        "journal_dir": str(tmp_path / "journals"),
        "db_path": str(db),
        "shadow_db_path": str(shadow),
        "batch_window_ms": 5,
        "max_batch_size": 100,
    })
    cfg.ensure_runtime_dirs()
    # Pre-create the observatory schema in the primary DB (the daemon won't
    # create it for us — observatory schema is owned by ObservatoryDB._init_db).
    obs_init = ObservatoryDB(db_path=str(db))
    del obs_init
    return cfg


def test_canonical_mode_routes_trinity_snapshots_through_daemon(canonical_cfg):
    """100 trinity-snapshot writes via canonical-mode client all land in primary."""
    daemon, stop = _spawn_daemon(canonical_cfg)
    try:
        client = InnerMemoryWriterClient(canonical_cfg, caller_name="test_obs")
        # Wait for client to actually connect (asyncio thread + transport).
        deadline = time.time() + 5.0
        while not client.ping(timeout=1.0) and time.time() < deadline:
            time.sleep(0.1)
        assert client.ping(timeout=2.0), "client failed to reach daemon"

        # ObservatoryDB with the live writer client wired in.
        db = ObservatoryDB(db_path=canonical_cfg.db_path, writer_client=client)
        assert db._writer is client

        # Fire 100 trinity snapshots.
        for i in range(100):
            db.record_trinity_snapshot(
                body_tensor=[float(i), 0.0, 0.0, 0.0, 0.0],
                mind_tensor=[0.0, float(i), 0.0, 0.0, 0.0],
                spirit_tensor=[0.0, 0.0, float(i), 0.0, 0.0],
                middle_path_loss=0.001 * i,
                body_center_dist=0.01 * i,
                mind_center_dist=0.02 * i,
            )

        # Force the daemon to drain — flush by giving it time + ping.
        client.flush(timeout=10.0)
        time.sleep(0.5)

        # Verify all 100 rows present in primary DB.
        conn = sqlite3.connect(canonical_cfg.db_path, timeout=10)
        try:
            cur = conn.execute("SELECT COUNT(*) FROM trinity_snapshots")
            count = cur.fetchone()[0]
        finally:
            conn.close()
        assert count == 100, (
            f"canonical-mode trinity writes lost rows: expected 100, got {count}"
        )

        # Shadow DB should be EMPTY in canonical mode (no shadow safety net,
        # unlike hybrid). Confirms direct-path bypass + shadow-fire-and-forget
        # are both off.
        if Path(canonical_cfg.shadow_db_path).exists():
            sconn = sqlite3.connect(canonical_cfg.shadow_db_path, timeout=10)
            try:
                tables = sconn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name='trinity_snapshots'"
                ).fetchall()
                if tables:
                    scount = sconn.execute(
                        "SELECT COUNT(*) FROM trinity_snapshots"
                    ).fetchone()[0]
                    assert scount == 0, (
                        f"canonical mode must NOT write to shadow DB, got {scount} rows"
                    )
            finally:
                sconn.close()

        client.close()
    finally:
        stop()


def test_canonical_mode_concurrent_writers_no_lock_errors(canonical_cfg):
    """The bug's actual scenario: N concurrent writers feeding the daemon.

    With canonical-mode + single-writer daemon, 4 threads × 50 writes each
    must all succeed with no lock errors and final rowcount = 200.
    """
    daemon, stop = _spawn_daemon(canonical_cfg)
    try:
        client = InnerMemoryWriterClient(canonical_cfg, caller_name="test_obs_concurrent")
        deadline = time.time() + 5.0
        while not client.ping(timeout=1.0) and time.time() < deadline:
            time.sleep(0.1)
        assert client.ping(timeout=2.0)

        db = ObservatoryDB(db_path=canonical_cfg.db_path, writer_client=client)
        errors: list[Exception] = []
        barrier = threading.Barrier(4)

        def worker(worker_id: int):
            try:
                barrier.wait()
                for i in range(50):
                    db.record_trinity_snapshot(
                        body_tensor=[float(worker_id), float(i), 0.0, 0.0, 0.0],
                        mind_tensor=[0.0, 0.0, 0.0, 0.0, 0.0],
                        spirit_tensor=[0.0, 0.0, 0.0, 0.0, 0.0],
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(w,)) for w in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30.0)

        client.flush(timeout=10.0)
        time.sleep(0.5)

        assert not errors, f"writers hit errors under concurrency: {errors}"

        conn = sqlite3.connect(canonical_cfg.db_path, timeout=10)
        try:
            count = conn.execute(
                "SELECT COUNT(*) FROM trinity_snapshots"
            ).fetchone()[0]
        finally:
            conn.close()
        assert count == 200, (
            f"4 threads × 50 writes = 200 expected, got {count} — "
            "lock contention or batch loss"
        )
        client.close()
    finally:
        stop()
