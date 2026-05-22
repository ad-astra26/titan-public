"""
Tests for shadow_orchestrator preflight_disk_check.

Closes BUG-SHADOW-SWAP-NO-PRE-FLIGHT-DISK-CHECK-20260504. The preflight
must refuse a swap when free disk < computed real-copy + safety margin
to prevent the disk-full cascade observed on T1 2026-05-04 (orchestrator
audit/rollback both depend on writable disk).

Companion fix shadow_data_dir.py (2026-05-04 PM): mtime-gated
hardlink-break — only real-copy files actively written. Preflight
predictor (`_compute_real_copy_bytes`) mirrors the same predicate so
its estimate matches actual swap consumption.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from titan_hcl.core.shadow_orchestrator import (
    SHADOW_SWAP_DISK_SAFETY_MARGIN_BYTES,
    _compute_real_copy_bytes,
    preflight_disk_check,
)


GB = 1024 ** 3


@pytest.fixture
def fake_data_dir(tmp_path: Path) -> Path:
    """A data dir with active DBs (recent mtime) and immutable archives (old mtime)."""
    d = tmp_path / "data"
    d.mkdir()
    now = time.time()

    # Active DBs (mtime = now → real-copy)
    active_files = [
        ("consciousness.db", 100),
        ("observatory.db", 200),
        ("inner_memory.db", 50),
        ("knowledge_graph.kuzu", 30),
        ("knowledge_graph.kuzu.wal", 10),
        ("memory.duckdb", 25),
        ("memory.db-wal", 15),
        ("memory.db-shm", 5),
    ]
    for name, size in active_files:
        p = d / name
        p.write_bytes(b"x" * size)
        # Force mtime to NOW so it counts as recent
        os.utime(p, (now, now))

    # Immutable archives (mtime = 1 hour ago → SKIP)
    archive_files = [
        ("twin_telemetry_20260501_1200.json", 1_000_000),
        ("twin_telemetry_20260502_1200.json", 2_000_000),
        ("child_dev_telemetry_20260321.json", 500_000),
        ("language_pipeline_20260320_1500.json", 50_000),
        ("developmental_day_20260320_0219.json", 1_500_000),
        ("config.toml", 10_000),
        ("identity.json", 200),
    ]
    one_hour_ago = now - 3600
    for name, size in archive_files:
        p = d / name
        p.write_bytes(b"x" * size)
        os.utime(p, (one_hour_ago, one_hour_ago))

    return d


class TestComputeRealCopyBytes:
    def test_counts_only_recently_modified_files(self, fake_data_dir):
        # Active files only (mtime now): 100+200+50+30+10+25+15+5 = 435
        # Archives (1hr old) skipped.
        assert _compute_real_copy_bytes(fake_data_dir) == 435

    def test_old_archives_skipped_regardless_of_size(self, fake_data_dir):
        # Total archive bytes are 5 MB+ — none counted because mtime old.
        # If they were counted, total would be > 1 MB.
        result = _compute_real_copy_bytes(fake_data_dir)
        assert result < 10_000  # all 8 active files combined are 435 bytes

    def test_recency_threshold_is_tunable(self, fake_data_dir):
        # With a 7200s threshold, the 1-hour-old archives now count too.
        assert _compute_real_copy_bytes(fake_data_dir, recency_threshold_s=7200) > 5_000_000

    def test_returns_zero_for_missing_dir(self, tmp_path):
        assert _compute_real_copy_bytes(tmp_path / "does_not_exist") == 0

    def test_returns_zero_for_empty_dir(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        assert _compute_real_copy_bytes(empty) == 0


class TestPreflightDiskCheck:
    def test_passes_when_disk_has_room(self, fake_data_dir):
        # 100 GB free, fake DBs ~435 bytes — trivially passes
        with patch("shutil.disk_usage") as mock_du:
            mock_du.return_value.free = 100 * GB
            result = preflight_disk_check(fake_data_dir)
        assert result is None

    def test_refuses_when_disk_below_threshold(self, fake_data_dir):
        # Free = 1 GB, required = 435B + 4GB margin → REFUSE
        with patch("shutil.disk_usage") as mock_du:
            mock_du.return_value.free = 1 * GB
            result = preflight_disk_check(fake_data_dir)
        assert result is not None
        assert result["outcome"] == "refused"
        assert result["phase"] == "preflight"
        assert result["failure_reason"] == "insufficient_disk"
        assert "GB free" in result["detail"]

    def test_refusal_includes_structured_byte_counts(self, fake_data_dir):
        with patch("shutil.disk_usage") as mock_du:
            mock_du.return_value.free = 1 * GB
            result = preflight_disk_check(fake_data_dir)
        assert result["real_copy_bytes"] == 435
        assert result["required_bytes"] == 435 + SHADOW_SWAP_DISK_SAFETY_MARGIN_BYTES
        assert result["free_bytes"] == 1 * GB
        assert result["safety_margin_bytes"] == SHADOW_SWAP_DISK_SAFETY_MARGIN_BYTES

    def test_passes_at_exact_threshold(self, fake_data_dir):
        # Free = required exactly → passes (>= comparison)
        with patch("shutil.disk_usage") as mock_du:
            mock_du.return_value.free = 435 + SHADOW_SWAP_DISK_SAFETY_MARGIN_BYTES
            result = preflight_disk_check(fake_data_dir)
        assert result is None

    def test_refuses_just_below_threshold(self, fake_data_dir):
        # Free = required - 1 byte → REFUSE
        with patch("shutil.disk_usage") as mock_du:
            mock_du.return_value.free = (
                435 + SHADOW_SWAP_DISK_SAFETY_MARGIN_BYTES - 1
            )
            result = preflight_disk_check(fake_data_dir)
        assert result is not None
        assert result["failure_reason"] == "insufficient_disk"

    def test_refuses_safely_when_disk_usage_raises(self, fake_data_dir):
        # If we can't even stat the FS, fail safe (refuse, don't proceed)
        with patch("shutil.disk_usage", side_effect=OSError("device gone")):
            result = preflight_disk_check(fake_data_dir)
        assert result is not None
        assert result["outcome"] == "refused"
        assert result["failure_reason"] == "preflight_disk_stat_failed"
        assert "device gone" in result["detail"]

    def test_returns_none_when_no_realcopy_files_and_disk_ok(self, tmp_path):
        # Empty data dir — only safety margin matters
        empty = tmp_path / "empty"
        empty.mkdir()
        with patch("shutil.disk_usage") as mock_du:
            mock_du.return_value.free = SHADOW_SWAP_DISK_SAFETY_MARGIN_BYTES
            result = preflight_disk_check(empty)
        assert result is None
