"""Phase 9 Chunk 9K — fork-state cache regression test.

Verifies that `TimeChain._load_fork_state` skips the linear file scan when
the sqlite cache (`cached_file_mtime` + `cached_file_size` columns added in
Chunk 9K) matches the chain file's current mtime+size. Falls back correctly
when the cache is stale (file mtime advanced) or absent (pre-9K database).

Per RFP §3F.2.5 Chunk 9K acceptance: "TimeChain.__init__ wall time <100ms
in happy path (cached tip); fall-back path test exercises stale-mtime
branch correctly."
"""
from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path

import pytest

from titan_hcl.logic.timechain import (
    FORK_MAIN,
    FORK_DECLARATIVE,
    BlockPayload,
    TimeChain,
)


def _make_payload(idx: int) -> BlockPayload:
    return BlockPayload(
        thought_type="test",
        content={"idx": idx, "text": "9K cache test"},
        source="test_9k",
        tags=[],
        significance=0.5,
        db_ref="",
    )


def _commit_n_blocks(chain: TimeChain, n: int, fork_id: int = FORK_MAIN) -> None:
    for i in range(n):
        chain.commit_block(
            fork_id=fork_id, epoch_id=1, payload=_make_payload(i),
            pot_nonce=0, chi_spent=0.0,
            neuromod_state={"DA": 0.5, "ACh": 0.5, "NE": 0.5},
        )


def _read_cache_columns(db_path: Path) -> dict[int, tuple[float, int]]:
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT fork_id, cached_file_mtime, cached_file_size "
            "FROM fork_registry"
        ).fetchall()
        return {fid: (float(m or 0), int(s or 0)) for fid, m, s in rows}
    finally:
        conn.close()


def test_cache_populated_after_block_append(tmp_path: Path) -> None:
    """Every commit_block stamps cached_file_mtime + cached_file_size."""
    chain = TimeChain(data_dir=tmp_path, titan_id="T_TEST")
    chain.create_genesis({"birth": "test"}, birth_timestamp=time.time())
    _commit_n_blocks(chain, 5, FORK_MAIN)

    cache = _read_cache_columns(tmp_path / "index.db")
    assert FORK_MAIN in cache
    mtime, size = cache[FORK_MAIN]
    assert mtime > 0, "cached_file_mtime should be stamped after appends"
    assert size > 0, "cached_file_size should be stamped after appends"

    # Cache mtime+size match the actual file
    chain_file = tmp_path / "chain_main.bin"
    st = chain_file.stat()
    assert abs(st.st_mtime - mtime) < 0.01
    assert st.st_size == size


def test_cache_hit_skips_linear_scan(tmp_path: Path) -> None:
    """Reopening a chain with valid cache should skip _get_file_tip_height."""
    chain = TimeChain(data_dir=tmp_path, titan_id="T_TEST")
    chain.create_genesis({"birth": "test"}, birth_timestamp=time.time())
    _commit_n_blocks(chain, 10, FORK_MAIN)
    _commit_n_blocks(chain, 5, FORK_DECLARATIVE)
    del chain

    # Reopen — every fork's cache should be valid, no scan needed.
    scan_calls = {"count": 0}
    orig = TimeChain._get_file_tip_height

    def _spy(self, path):
        scan_calls["count"] += 1
        return orig(self, path)

    TimeChain._get_file_tip_height = _spy
    try:
        chain2 = TimeChain(data_dir=tmp_path, titan_id="T_TEST")
        assert chain2.get_fork_tip(FORK_MAIN)[0] == 10, "tip preserved"
        assert chain2.get_fork_tip(FORK_DECLARATIVE)[0] == 5
    finally:
        TimeChain._get_file_tip_height = orig

    assert scan_calls["count"] == 0, (
        f"linear scan was called {scan_calls['count']} times despite valid "
        "cache — 9K cache hit not honored"
    )


def test_cache_miss_falls_back_to_scan(tmp_path: Path) -> None:
    """Tampering with file mtime triggers the cold-recovery scan path."""
    chain = TimeChain(data_dir=tmp_path, titan_id="T_TEST")
    chain.create_genesis({"birth": "test"}, birth_timestamp=time.time())
    _commit_n_blocks(chain, 3, FORK_MAIN)
    del chain

    # Touch the chain file to invalidate cache mtime.
    chain_file = tmp_path / "chain_main.bin"
    new_mtime = time.time() + 100
    os.utime(chain_file, (new_mtime, new_mtime))

    scan_calls = {"count": 0}
    orig = TimeChain._get_file_tip_height

    def _spy(self, path):
        scan_calls["count"] += 1
        return orig(self, path)

    TimeChain._get_file_tip_height = _spy
    try:
        chain2 = TimeChain(data_dir=tmp_path, titan_id="T_TEST")
        assert chain2.get_fork_tip(FORK_MAIN)[0] == 3, "tip recovered via scan"
    finally:
        TimeChain._get_file_tip_height = orig

    assert scan_calls["count"] >= 1, (
        "stale-mtime cache should have triggered fall-back linear scan"
    )

    # After the miss-then-recover, the cache should be re-stamped.
    cache = _read_cache_columns(tmp_path / "index.db")
    assert cache[FORK_MAIN][0] > 0


def test_pre_9k_database_migrates_idempotently(tmp_path: Path) -> None:
    """A fork_registry created without 9K columns gains them on next open."""
    db_path = tmp_path / "index.db"

    # Build a pre-9K schema by hand (no cached_file_* columns).
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript("""
            CREATE TABLE fork_registry (
                fork_id INTEGER PRIMARY KEY,
                fork_name TEXT NOT NULL,
                fork_type TEXT NOT NULL,
                parent_fork INTEGER,
                parent_block INTEGER,
                created_at REAL NOT NULL,
                tip_height INTEGER DEFAULT 0,
                tip_hash BLOB,
                topic TEXT,
                compacted INTEGER DEFAULT 0
            );
            CREATE TABLE block_index (
                block_hash BLOB PRIMARY KEY,
                fork_id INTEGER,
                block_height INTEGER,
                timestamp REAL,
                epoch_id INTEGER,
                thought_type TEXT,
                source TEXT,
                significance REAL,
                chi_spent REAL,
                neuromod_da REAL,
                neuromod_ach REAL,
                neuromod_ne REAL,
                tags TEXT,
                cross_refs TEXT,
                db_ref TEXT,
                compacted INTEGER,
                file_offset INTEGER
            );
            CREATE TABLE checkpoints (
                checkpoint_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                epoch_id INTEGER NOT NULL,
                merkle_root BLOB NOT NULL,
                total_blocks INTEGER NOT NULL,
                fork_tips TEXT NOT NULL
            );
        """)
        conn.commit()
    finally:
        conn.close()

    # Opening a TimeChain against this pre-9K db should add the columns.
    TimeChain(data_dir=tmp_path, titan_id="T_TEST")

    conn = sqlite3.connect(str(db_path))
    try:
        cols = [r[1] for r in conn.execute(
            "PRAGMA table_info(fork_registry)").fetchall()]
    finally:
        conn.close()
    assert "cached_file_mtime" in cols
    assert "cached_file_size" in cols
