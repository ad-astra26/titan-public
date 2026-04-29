"""Unit tests for warning_monitor events.jsonl rotation.

Closes BUG-WARNING-MONITOR-PERSISTENCE-UNBOUNDED (2026-04-27, ARCHITECTURAL):
events.jsonl was append-only with no rotation policy, growing to 440MB
on T1 before being capped. _maybe_rotate_events_log keeps it bounded.
"""
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

from titan_plugin.modules.warning_monitor_worker import (
    EVENTS_LOG_KEEP_ARCHIVES,
    EVENTS_LOG_MAX_MB,
    _maybe_rotate_events_log,
)


def _write_n_bytes(path: Path, n: int) -> None:
    """Write a file of approximately n bytes (filled with valid JSONL)."""
    line = json.dumps({"ts": 0, "level": "INFO", "key": "x", "msg": "y"}) + "\n"
    needed = max(1, n // len(line))
    with open(path, "w") as f:
        for _ in range(needed):
            f.write(line)


def test_no_rotate_below_threshold(tmp_path):
    """File stays in place when under max_mb."""
    events = tmp_path / "events.jsonl"
    _write_n_bytes(events, 100 * 1024)  # 100 KB
    size_before = events.stat().st_size
    _maybe_rotate_events_log(str(events), max_mb=EVENTS_LOG_MAX_MB)
    assert events.stat().st_size == size_before, "small file should NOT rotate"
    assert not list(tmp_path.glob("events.jsonl.*.gz"))


def test_rotate_above_threshold(tmp_path):
    """File over threshold gets gzip-archived; live file truncates to 0."""
    events = tmp_path / "events.jsonl"
    # Write 2 MB, threshold 1 MB → must rotate
    _write_n_bytes(events, 2 * 1024 * 1024)
    assert events.stat().st_size > 1024 * 1024
    _maybe_rotate_events_log(str(events), max_mb=1)
    # Live file empty after rotation
    assert events.exists()
    assert events.stat().st_size == 0, "live file should be empty post-rotation"
    # One archive created
    archives = sorted(tmp_path.glob("events.jsonl.*.gz"))
    assert len(archives) == 1, f"expected 1 archive, got {len(archives)}"
    # Archive content matches what we wrote (modulo gzip)
    with gzip.open(archives[0], "rt") as f:
        content = f.read()
    # First and last JSONL lines must be intact JSON
    lines = content.strip().split("\n")
    assert len(lines) > 1
    json.loads(lines[0])
    json.loads(lines[-1])


def test_rotate_missing_file_is_noop(tmp_path):
    """Rotating a non-existent file is a noop (no exception, no creation)."""
    events = tmp_path / "events.jsonl"
    _maybe_rotate_events_log(str(events), max_mb=1)
    assert not events.exists()
    assert not list(tmp_path.glob("events.jsonl*"))


def test_archive_pruning_keeps_only_n(tmp_path):
    """When more than `keep` archives exist, oldest are pruned."""
    events = tmp_path / "events.jsonl"
    # Create 7 fake old archives + 1 oversized live file → after rotate
    # we should have keep=3 most-recent archives.
    for i in range(7):
        archive = tmp_path / f"events.jsonl.2026010{i}_000000.gz"
        with gzip.open(archive, "wb") as f:
            f.write(b"old\n")
    _write_n_bytes(events, 2 * 1024 * 1024)
    _maybe_rotate_events_log(str(events), max_mb=1, keep=3)
    archives = sorted(tmp_path.glob("events.jsonl.*.gz"))
    assert len(archives) == 3, (
        f"expected exactly 3 archives after pruning, got {len(archives)}: "
        f"{[a.name for a in archives]}")
    # Newest (just created) survives
    newest = max(archives, key=lambda p: p.stat().st_mtime)
    # Newest archive should NOT be one of the pre-existing fakes
    fake_names = {f"events.jsonl.2026010{i}_000000.gz" for i in range(7)}
    assert newest.name not in fake_names


def test_default_thresholds_sane():
    """Default max_mb + keep are reasonable (not 0, not absurdly large)."""
    assert 1 <= EVENTS_LOG_MAX_MB <= 500
    assert 1 <= EVENTS_LOG_KEEP_ARCHIVES <= 50
