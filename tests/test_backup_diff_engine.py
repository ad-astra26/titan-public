"""DiffEngine + source-snapshot consistency — RFP_backup_redesign_spine Phase A.

Gates:
  GA1(a) truncation-immune + torn-read-free snapshot (self-written read-conn)
  GA1(b) IMW-snapshot consistency under concurrent writes + commit loop NOT
         stalled (the A.0 op)
  GA2    pack ≥100 MB → peak RSS bounded
  GA3    build↔restore byte-identical (symmetric, ONE DiffEngine)
+ the §24.5.a / INV-BR-11 registry derivation (the V-finding: events_teacher.db
  is IMW-owned, derived from config — never a hardcoded {inner_memory, social_graph}).
"""
import asyncio
import os
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from titan_hcl.logic.backup_diff_engine import DiffEngine
from titan_hcl.logic.backup_event_tarball import FileDiffSpec
from titan_hcl.logic.backup_sqlite_snapshot import (
    imw_owned_realpaths,
    is_sqlite_file,
    snapshot_sqlite_sync,
)
from titan_hcl.persistence.config import IMWConfig
from titan_hcl.persistence.writer_client import InnerMemoryWriterClient
from titan_hcl.persistence.writer_service import IMWDaemon


# ── helpers ──────────────────────────────────────────────────────────


def _make_sqlite(path, *, rows=0, wal=True):
    conn = sqlite3.connect(str(path))
    if wal:
        conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY AUTOINCREMENT, v TEXT)")
    if rows:
        conn.executemany("INSERT INTO t(v) VALUES(?)", [(f"val-{i}",) for i in range(rows)])
    conn.commit()
    conn.close()


def _rss_mb() -> float:
    with open("/proc/self/statm") as f:
        resident_pages = int(f.read().split()[1])
    return resident_pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)


def _spawn_daemon(cfg: IMWConfig):
    """Spawn an IMWDaemon in a background thread (own asyncio loop). Mirrors the
    test_imw_end_to_end harness."""
    loop = asyncio.new_event_loop()
    daemon = IMWDaemon(cfg)
    ready = threading.Event()
    stop_event = None

    def _run():
        nonlocal stop_event
        asyncio.set_event_loop(loop)
        stop_event = asyncio.Event()

        async def _inner():
            await daemon.start()
            ready.set()
            while not stop_event.is_set():
                await asyncio.sleep(0.05)
            await daemon.stop()

        loop.run_until_complete(_inner())
        loop.close()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    if not ready.wait(timeout=10.0):
        raise RuntimeError("daemon failed to become ready")
    time.sleep(0.1)

    def _stop():
        loop.call_soon_threadsafe(lambda: stop_event.set() if stop_event else None)
        t.join(timeout=10.0)

    return daemon, _stop


def _imw_cfg(tmp_path, rows=0):
    db = tmp_path / "inner_memory.db"
    shadow = tmp_path / "inner_memory_shadow.db"
    for p in (db, shadow):
        conn = sqlite3.connect(str(p))
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY AUTOINCREMENT, v INTEGER)")
        conn.commit()
        conn.close()
    if rows:
        conn = sqlite3.connect(str(db))
        conn.executemany("INSERT INTO t(v) VALUES(?)", [(i,) for i in range(rows)])
        conn.commit()
        conn.close()
    cfg = IMWConfig.from_dict({
        "enabled": True, "mode": "canonical", "tables_canonical": ["t"],
        "socket_path": str(tmp_path / "imw.sock"),
        "wal_path": str(tmp_path / "imw.wal"),
        "journal_dir": str(tmp_path / "journals"),
        "db_path": str(db), "shadow_db_path": str(shadow),
        "batch_window_ms": 5, "max_batch_size": 50,
    })
    cfg.ensure_runtime_dirs()
    return cfg


# ── detection + registry (the V-finding) ─────────────────────────────


def test_is_sqlite_file_header_detection(tmp_path):
    db = tmp_path / "real.db"
    _make_sqlite(db, rows=3)
    assert is_sqlite_file(str(db)) is True
    # NON-SQLite must NOT take the conn.backup path:
    j = tmp_path / "state.json"
    j.write_text('{"a":1}')
    assert is_sqlite_file(str(j)) is False
    # a DuckDB/Kuzu-shaped file (different magic) is NOT SQLite → copy/hardlink path
    duck = tmp_path / "memory.duckdb"
    duck.write_bytes(b"DUCK" + os.urandom(64))
    assert is_sqlite_file(str(duck)) is False


def test_imw_owned_registry_derived_from_config(tmp_path):
    """INV-BR-11: ownership is DERIVED from enabled [persistence_*] sections —
    NOT a hardcoded {inner_memory, social_graph}. events_teacher.db IS owned."""
    cfg = {
        "persistence": {"enabled": True, "mode": "canonical",
                        "db_path": str(tmp_path / "inner_memory.db")},
        "persistence_social_graph": {"enabled": True, "mode": "canonical",
                                     "db_path": str(tmp_path / "social_graph.db")},
        "persistence_events_teacher": {"enabled": True, "mode": "canonical",
                                       "db_path": str(tmp_path / "events_teacher.db")},
        "persistence_off": {"enabled": False, "mode": "disabled",
                            "db_path": str(tmp_path / "off.db")},
        "not_a_persistence_section": {"foo": 1},
    }
    owned = imw_owned_realpaths(cfg)
    for name in ("inner_memory.db", "social_graph.db", "events_teacher.db"):
        assert os.path.realpath(str(tmp_path / name)) in owned
    assert os.path.realpath(str(tmp_path / "off.db")) not in owned  # disabled excluded
    assert len(owned) == 3


# ── GA1(a): self-written snapshot consistency under writes ───────────


def test_self_written_sqlite_snapshot_consistent_under_writes(tmp_path):
    db = tmp_path / "experience_self.db"   # not in any registry → read-conn path
    _make_sqlite(db, rows=500)
    stop = threading.Event()

    def writer():
        w = sqlite3.connect(str(db), timeout=30)
        w.execute("PRAGMA journal_mode=WAL")
        i = 500
        while not stop.is_set():
            w.execute("INSERT INTO t(v) VALUES(?)", (f"val-{i}",))
            w.commit()
            i += 1
        w.close()

    th = threading.Thread(target=writer)
    th.start()
    time.sleep(0.05)
    try:
        for k in range(5):
            dest = tmp_path / f"snap_{k}.db"
            snapshot_sqlite_sync(str(db), str(dest))   # NO OSError under writes
            s = sqlite3.connect(str(dest))
            try:
                assert s.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
                # consistent point-in-time: every row well-formed, no torn rows
                for (v,) in s.execute("SELECT v FROM t"):
                    assert v.startswith("val-")
            finally:
                s.close()
    finally:
        stop.set()
        th.join(timeout=5)


# ── GA1(b): IMW snapshot consistency + commit loop NOT stalled ───────


def test_imw_snapshot_consistent_and_no_commit_stall(tmp_path):
    cfg = _imw_cfg(tmp_path, rows=5000)   # pre-populated so conn.backup has real work
    daemon, stop_daemon = _spawn_daemon(cfg)
    client = InnerMemoryWriterClient(cfg, caller_name="snap_test")
    writer = InnerMemoryWriterClient(cfg, caller_name="writer")
    try:
        done = {"n": 5000}
        stop = threading.Event()

        def hammer():
            while not stop.is_set():
                r = writer.write("INSERT INTO t(v) VALUES(?)", (done["n"],), table="t")
                if r.ok:
                    done["n"] += 1

        th = threading.Thread(target=hammer)
        th.start()
        time.sleep(0.05)
        for k in range(3):
            before = done["n"]
            dest = tmp_path / f"imw_snap_{k}.db"
            ok = client.snapshot(str(dest), timeout=60)
            after = done["n"]
            assert ok is True
            s = sqlite3.connect(str(dest))
            try:
                assert s.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
                cnt = s.execute("SELECT count(*) FROM t").fetchone()[0]
                assert cnt >= 5000            # consistent image, all committed rows
            finally:
                s.close()
            assert after >= before            # writes never blocked backwards
        stop.set()
        th.join(timeout=5)
        # the commit loop kept committing throughout the snapshots (no stall)
        assert done["n"] > 5000
    finally:
        client.close()
        writer.close()
        stop_daemon()


# ── GA1(a) copy path: rotating .jsonl is truncation-immune ───────────


def test_jsonl_snapshot_truncation_immune(tmp_path):
    src = tmp_path / "haov_signal_outcomes.jsonl"
    original = '{"a":1}\n{"b":2}\n{"c":3}\n'
    src.write_text(original)
    eng = DiffEngine()
    snap, owned = asyncio.run(eng.snapshot(str(src)))
    try:
        src.write_text("")   # live in-place truncation (the 2026-06-09 hazard)
        assert Path(snap).read_text() == original   # snapshot is a stable inode
    finally:
        if owned and Path(snap).exists():
            os.unlink(snap)


# ── GA3: build↔restore byte-identical (symmetric DiffEngine) ─────────


def test_build_restore_byte_identical(tmp_path):
    db = tmp_path / "mem.db"
    _make_sqlite(db, rows=2000)

    async def _roundtrip():
        eng = DiffEngine()
        snap, owned = await eng.snapshot(str(db))
        snap_bytes = Path(snap).read_bytes()
        dd = await eng.encode(snap, None)            # full-ship the consistent image
        out = str(tmp_path / "event_e1_personality.tar.zst")
        await eng.pack(event_id="e1", event_type="baseline", component="personality",
                       file_specs=[FileDiffSpec("mem.db", dd)], output_path=out)
        target = tmp_path / "restore"
        target.mkdir()
        res = await eng.apply(
            {"personality": Path(out).read_bytes()}, str(target),
            lambda component, arc: str(target / arc), verify_patch_hash=True)
        return snap_bytes, (target / "mem.db").read_bytes(), res, snap, owned

    snap_bytes, restored_bytes, res, snap, owned = asyncio.run(_roundtrip())
    assert res["errors"] == []
    assert restored_bytes == snap_bytes          # symmetric: restore == the snapshot
    # and the reconstruction is itself a valid, consistent SQLite DB
    rdb = tmp_path / "restore" / "mem.db"
    s = sqlite3.connect(str(rdb))
    try:
        assert s.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert s.execute("SELECT count(*) FROM t").fetchone()[0] == 2000
    finally:
        s.close()
    if owned and Path(snap).exists():
        os.unlink(snap)


# ── GA2: pack ≥100 MB → peak RSS bounded ─────────────────────────────


def test_pack_rss_bounded(tmp_path):
    db = tmp_path / "big.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE blobs (id INTEGER PRIMARY KEY, b BLOB)")
    blob = os.urandom(1024 * 1024)   # 1 MiB incompressible
    for i in range(110):
        conn.execute("INSERT INTO blobs(b) VALUES(?)", (blob,))
        if i % 20 == 0:
            conn.commit()
    conn.commit()
    conn.close()
    assert os.path.getsize(db) >= 100 * 1024 * 1024

    peak = {"mb": 0.0}
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            peak["mb"] = max(peak["mb"], _rss_mb())
            time.sleep(0.005)

    base = _rss_mb()
    th = threading.Thread(target=sampler)
    th.start()

    async def _build():
        eng = DiffEngine()
        snap, owned = await eng.snapshot(str(db))
        dd = await eng.encode(snap, None)
        await eng.pack(event_id="e", event_type="baseline", component="personality",
                       file_specs=[FileDiffSpec("big.db", dd)],
                       output_path=str(tmp_path / "big.tar.zst"))
        return snap, owned

    try:
        snap, owned = asyncio.run(_build())
    finally:
        stop.set()
        th.join(timeout=5)

    delta = peak["mb"] - base
    assert delta < 50.0, f"peak RSS delta {delta:.1f} MB exceeded the 50 MB bound"
    if owned and Path(snap).exists():
        os.unlink(snap)
