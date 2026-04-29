"""Unit tests for timechain_v2 rich-signal stats (Phase 1 sensory wiring).

Covers:
- record_tx_latency / get_tx_latency_stats
- get_block_delta_stats (mocked sqlite reads)
- Edge cases: NaN, negative duration, empty buffer
"""
import sqlite3
import tempfile
import time
from unittest.mock import patch

import pytest

from titan_plugin.logic import timechain_v2 as tcv2


@pytest.fixture(autouse=True)
def _reset_state():
    tcv2._reset_rich_signal_state_for_testing()
    yield
    tcv2._reset_rich_signal_state_for_testing()


# ── TX latency ──────────────────────────────────────────────────

def test_tx_latency_empty_buffer_returns_neutral():
    stats = tcv2.get_tx_latency_stats()
    assert stats["samples"] == 0
    assert stats["normalized"] == 0.5


def test_tx_latency_single_healthy_sample():
    tcv2.record_tx_latency(1.0)  # healthy (<= 2s)
    stats = tcv2.get_tx_latency_stats()
    assert stats["samples"] == 1
    assert stats["median_s"] == 1.0
    assert stats["normalized"] == 0.0


def test_tx_latency_congested_sample():
    tcv2.record_tx_latency(20.0)  # way congested (>= 15s)
    stats = tcv2.get_tx_latency_stats()
    assert stats["normalized"] == 1.0


def test_tx_latency_midrange():
    # 8s = midpoint of [2, 15] range
    tcv2.record_tx_latency(8.0)
    stats = tcv2.get_tx_latency_stats()
    # (8 - 2) / (15 - 2) = 6/13 ≈ 0.462
    assert 0.4 <= stats["normalized"] <= 0.5


def test_tx_latency_median_over_multiple_samples():
    for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
        tcv2.record_tx_latency(v)
    stats = tcv2.get_tx_latency_stats()
    assert stats["median_s"] == 3.0
    assert stats["samples"] == 5


def test_tx_latency_p95_greater_than_median():
    for v in [1.0, 1.0, 1.0, 1.0, 10.0]:
        tcv2.record_tx_latency(v)
    stats = tcv2.get_tx_latency_stats()
    assert stats["p95_s"] >= stats["median_s"]


def test_tx_latency_rejects_nan_and_negative():
    tcv2.record_tx_latency(float("nan"))
    tcv2.record_tx_latency(-1.5)
    tcv2.record_tx_latency("not a number")  # type: ignore
    tcv2.record_tx_latency(2.0)  # valid
    stats = tcv2.get_tx_latency_stats()
    assert stats["samples"] == 1
    assert stats["median_s"] == 2.0


def test_tx_latency_buffer_bounded():
    """Feed 50 samples, verify only last 20 retained."""
    for i in range(50):
        tcv2.record_tx_latency(float(i))
    stats = tcv2.get_tx_latency_stats()
    assert stats["samples"] == 20
    # Last 20 samples are 30..49; median is 39 or 40
    assert 38.0 <= stats["median_s"] <= 41.0


# ── Block delta ─────────────────────────────────────────────────

def _make_index_db(path: str, block_count: int) -> None:
    """Create a minimal block_index table with N rows on the main fork."""
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE block_index (fork_id INTEGER, block_hash BLOB, block_height INTEGER)"
    )
    # Seed main fork (fork_id=0 per FORK_IDS['main'])
    for i in range(block_count):
        conn.execute(
            "INSERT INTO block_index (fork_id, block_hash, block_height) VALUES (?, ?, ?)",
            (0, b"x" * 32, i),
        )
    conn.commit()
    conn.close()


def test_block_delta_empty_db(tmp_path):
    """No index_db → returns neutral."""
    stats = tcv2.get_block_delta_stats(str(tmp_path / "nonexistent.db"))
    assert stats["samples"] == 0
    assert stats["normalized"] == 0.5


def test_block_delta_single_sample_neutral(tmp_path):
    db = str(tmp_path / "index.db")
    _make_index_db(db, 100)
    stats = tcv2.get_block_delta_stats(db)
    assert stats["samples"] == 1
    assert stats["latest_height"] == 100
    assert stats["normalized"] == 0.5  # Can't compute rate from 1 sample


def test_block_delta_two_samples_compute_rate(tmp_path):
    db = str(tmp_path / "index.db")
    _make_index_db(db, 100)

    # First sample at t0 with 100 blocks
    tcv2.get_block_delta_stats(db)

    # Simulate 60s passing — bump ttl timer and add 14 blocks
    tcv2._last_block_sample_ts = time.time() - 100.0
    conn = sqlite3.connect(db)
    for i in range(100, 114):
        conn.execute(
            "INSERT INTO block_index (fork_id, block_hash, block_height) VALUES (?, ?, ?)",
            (0, b"x" * 32, i),
        )
    conn.commit()
    conn.close()

    # Manually adjust timestamp of first buffered sample to appear older
    with tcv2._block_height_lock:
        ts0, h0 = tcv2._block_height_buffer[0]
        tcv2._block_height_buffer[0] = (ts0 - 60.0, h0)

    stats = tcv2.get_block_delta_stats(db)
    assert stats["samples"] == 2
    assert stats["latest_height"] == 114
    # ~14 blocks in ~60s = 14 blocks/min → normalized 1.0
    assert stats["blocks_per_min"] >= 10.0
    assert stats["normalized"] >= 0.7


def test_block_delta_ttl_gates_repeated_sampling(tmp_path):
    db = str(tmp_path / "index.db")
    _make_index_db(db, 50)

    tcv2.get_block_delta_stats(db)  # first read
    # Repeated reads within TTL shouldn't add new samples
    tcv2.get_block_delta_stats(db)
    tcv2.get_block_delta_stats(db)

    with tcv2._block_height_lock:
        count = len(tcv2._block_height_buffer)
    assert count == 1, f"Expected 1 buffered sample (TTL-gated), got {count}"


def test_block_delta_handles_negative_rate_gracefully(tmp_path):
    """If DB shrinks (corruption/rebuild), treat as stalled."""
    db = str(tmp_path / "index.db")
    _make_index_db(db, 200)
    tcv2.get_block_delta_stats(db)

    # Backdate first sample and reduce block count (simulating rebuild)
    with tcv2._block_height_lock:
        tcv2._block_height_buffer.clear()
        tcv2._block_height_buffer.append((time.time() - 60.0, 200))
        tcv2._block_height_buffer.append((time.time(), 150))

    stats = tcv2.get_block_delta_stats(db)
    # Force TTL to prevent re-sampling
    tcv2._last_block_sample_ts = time.time()
    assert stats["blocks_per_min"] >= 0.0  # Clamped non-negative


def test_block_delta_db_read_failure_preserves_buffer(tmp_path):
    """DB read failure returns current buffer stats without corrupting."""
    # Point at a bad path — no sample added
    stats = tcv2.get_block_delta_stats("/nonexistent/path/index.db")
    assert stats["normalized"] == 0.5
    assert stats["samples"] == 0
