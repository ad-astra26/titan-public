"""Unit tests for warning_monitor _tail_brain_log streaming + LRU eviction.

Regression tests for Component B leak (2026-04-29):
- Pre-fix: text.splitlines()[:max_lines] materialized ~10k transient
  strings per 1MB chunk on T1's brain log (~107 B/line). Under
  MALLOC_ARENA_MAX=2 this fragmented the worker heap, growing RSS from
  ~50 MB cold to 1+ GB before Guardian killed at 300 MB limit. Crash
  loop every 8-10 min on T1.
- Post-fix: streaming line-iterator never materializes the full chunk;
  bytes-per-iter capped at MAX_TAIL_BYTES (256 KB).
- Plus: AGGREGATED_KEY_CAP + LRU eviction in _persist_state defends
  against unbounded aggregated dict growth.
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict, deque
from pathlib import Path

from titan_plugin.modules.warning_monitor_worker import (
    AGGREGATED_KEY_CAP,
    MAX_TAIL_BYTES,
    _evict_aggregated_lru,
    _persist_state,
    _tail_brain_log,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _write_log_lines(path: Path, n: int, level: str = "WARNING",
                     tag: str = "Test") -> int:
    """Write n parseable WARNING+ log lines, return file size."""
    line = f"00:00:00 [{level}] [{tag}] message body\n"
    with open(path, "ab") as f:
        for _ in range(n):
            f.write(line.encode("utf-8"))
    return path.stat().st_size


def _new_aggregated_entry(last_seen_ts: float, count: int = 1) -> dict:
    """Match the in-worker schema for an aggregated entry."""
    return {
        "count": count,
        "first_seen_ts": last_seen_ts - 1.0,
        "last_seen_ts": last_seen_ts,
        "last_msg": "test",
        "by_level": defaultdict(int, {"WARNING": count}),
        "rate_window": deque(maxlen=60),
    }


# ── _tail_brain_log: basic correctness ──────────────────────────────


def test_tail_returns_none_when_log_missing(tmp_path):
    missing = tmp_path / "no.log"
    assert _tail_brain_log(str(missing), prior_inode=42, prior_offset=0,
                           max_lines=100) is None


def test_tail_first_run_returns_empty_at_eof(tmp_path):
    """prior_inode=None means first sighting; advance to EOF, no parse."""
    log = tmp_path / "log"
    _write_log_lines(log, 10)
    parsed, inode, off = _tail_brain_log(
        str(log), prior_inode=None, prior_offset=0, max_lines=100)
    assert parsed == []
    assert inode > 0
    assert off == log.stat().st_size


def test_tail_parses_only_warning_plus(tmp_path):
    log = tmp_path / "log"
    info_line = "00:00:00 [INFO] [X] noise\n"
    warn_line = "00:00:00 [WARNING] [X] alarm\n"
    err_line = "00:00:00 [ERROR] [Y] bad\n"
    crit_line = "00:00:00 [CRITICAL] [Z] worse\n"
    with open(log, "wb") as f:
        f.write((info_line + warn_line + err_line + crit_line).encode())
    inode = os.stat(log).st_ino
    parsed, _, _ = _tail_brain_log(
        str(log), prior_inode=inode, prior_offset=0, max_lines=100)
    levels = [lvl for _, lvl, _ in parsed]
    assert levels == ["WARNING", "ERROR", "CRITICAL"]


# ── _tail_brain_log: streaming (no materialized list) ───────────────


def test_tail_caps_at_max_lines_and_offset_advances_to_last_consumed(tmp_path):
    """When there are far more lines than max_lines, only max_lines are
    parsed AND offset advances ONLY past the consumed lines (not the
    whole chunk). Next call resumes mid-file."""
    log = tmp_path / "log"
    _write_log_lines(log, 5000)  # ~165 KB at ~33 B/line, well under cap
    inode = os.stat(log).st_ino
    parsed, _, off1 = _tail_brain_log(
        str(log), prior_inode=inode, prior_offset=0, max_lines=200)
    assert len(parsed) == 200
    line_size = len("00:00:00 [WARNING] [Test] message body\n")
    assert off1 == 200 * line_size, (
        f"offset must advance past exactly 200 lines, got {off1} "
        f"(expected {200 * line_size})")

    # Second call from off1 picks up the next 200 lines
    parsed2, _, off2 = _tail_brain_log(
        str(log), prior_inode=inode, prior_offset=off1, max_lines=200)
    assert len(parsed2) == 200
    assert off2 == 400 * line_size


def test_tail_byte_cap_prevents_huge_chunk_reads(tmp_path):
    """Even if max_lines is enormous, we stop near MAX_TAIL_BYTES bytes.

    This is the regression guard: pre-fix the chunk could be 1 MB;
    post-fix MAX_TAIL_BYTES = 256 KB cap. The cap is enforced
    pre-consume, so we may go up to one max-line-length past the cap
    before stopping — that's acceptable. We allow +1 KB of slack in the
    assertion to cover a reasonable single-line size.
    """
    log = tmp_path / "log"
    # 30k lines @ ~38 B = ~1.14 MB total. With max_lines=999999 the byte cap
    # must kick in.
    _write_log_lines(log, 30000)
    inode = os.stat(log).st_ino
    parsed, _, off = _tail_brain_log(
        str(log), prior_inode=inode, prior_offset=0, max_lines=999_999)
    assert off <= MAX_TAIL_BYTES + 1024, (
        f"byte cap violated: offset={off} > MAX_TAIL_BYTES+1KB"
        f"={MAX_TAIL_BYTES + 1024}")
    assert off >= MAX_TAIL_BYTES - 1024, (
        f"byte cap not approached: offset={off} too low")
    assert len(parsed) > 0
    # We expect ~MAX_TAIL_BYTES / 38 ≈ 6900 lines, far less than 30k written.
    assert len(parsed) < 30000


def test_tail_partial_trailing_line_not_consumed(tmp_path):
    """A line missing the trailing \\n must be left for the next call,
    so we don't corrupt offset accounting on torn writes."""
    log = tmp_path / "log"
    full = "00:00:00 [WARNING] [A] complete\n"
    partial = "00:00:00 [WARNING] [B] partial-no-newline"
    with open(log, "wb") as f:
        f.write(full.encode())
        f.write(partial.encode())
    inode = os.stat(log).st_ino
    parsed, _, off = _tail_brain_log(
        str(log), prior_inode=inode, prior_offset=0, max_lines=100)
    # Only the complete line is consumed.
    assert len(parsed) == 1
    assert parsed[0][2].startswith("[A]")
    # Offset is at the boundary between full and partial.
    assert off == len(full.encode())


def test_tail_handles_log_rotation(tmp_path):
    """A new inode means rotation: reset offset to 0 and re-parse from start."""
    log = tmp_path / "log"
    _write_log_lines(log, 5)
    parsed1, inode1, off1 = _tail_brain_log(
        str(log), prior_inode=None, prior_offset=0, max_lines=100)
    # Offset is at EOF on first run; no lines parsed yet.
    assert parsed1 == []
    # Simulate rotation: rename old file out of the way, create new file.
    # `log.unlink(); _write(log)` may reuse the same inode on some
    # filesystems (Linux ext4 frequently does), defeating the inode-change
    # detection in the rotation path. Renaming forces a different inode.
    rotated = tmp_path / "log.rotated"
    log.rename(rotated)
    _write_log_lines(log, 3)
    parsed2, inode2, off2 = _tail_brain_log(
        str(log), prior_inode=inode1, prior_offset=off1, max_lines=100)
    # New inode → reset; all 3 lines parsed.
    assert inode2 != inode1
    assert len(parsed2) == 3


def test_tail_no_new_bytes_returns_empty(tmp_path):
    log = tmp_path / "log"
    _write_log_lines(log, 5)
    inode = os.stat(log).st_ino
    size = log.stat().st_size
    parsed, _, off = _tail_brain_log(
        str(log), prior_inode=inode, prior_offset=size, max_lines=100)
    assert parsed == []
    assert off == size


# ── Memory-pattern guard (proxy for "no transient mega-list") ───────


def test_tail_does_not_materialize_full_chunk_via_splitlines():
    """Source-level regression guard: the streaming implementation must
    not call `.splitlines()` on a chunk-sized buffer. CPython makes
    `bytes` / `str` immutable so we can't monkeypatch their methods —
    instead we read the function source and assert the offending
    pattern is absent.

    This is the *behavioral signature* of the bug we just fixed —
    if a future refactor reintroduces `chunk.decode().splitlines()` we
    catch it here even before observing RSS regressions in production.
    """
    import inspect

    src = inspect.getsource(_tail_brain_log)
    # The pre-fix code used `text.splitlines()[:max_lines]` after a
    # `chunk.decode()`. Either token alone is enough to flag the
    # regression class — the byte-streaming variant uses `for raw in f`
    # iteration and never needs splitlines/.decode() on a chunk.
    assert ".splitlines()" not in src, (
        "_tail_brain_log appears to call .splitlines() — Component B "
        "regression: this materializes a transient list of every line "
        "in the chunk, fragmenting glibc arenas under MALLOC_ARENA_MAX=2.")
    assert "chunk = f.read(" not in src, (
        "_tail_brain_log appears to read a megabuffer chunk — "
        "Component B regression: streaming must use `for raw in f` instead.")


# ── _evict_aggregated_lru ────────────────────────────────────────────


def test_evict_noop_below_cap():
    agg = {f"k{i}": _new_aggregated_entry(float(i)) for i in range(5)}
    assert _evict_aggregated_lru(agg, cap=10) == 0
    assert len(agg) == 5


def test_evict_drops_oldest_by_last_seen_ts():
    agg = {f"k{i}": _new_aggregated_entry(float(i)) for i in range(10)}
    n = _evict_aggregated_lru(agg, cap=4)
    assert n == 6
    assert len(agg) == 4
    # The 4 keys remaining must be the youngest (highest last_seen_ts).
    surviving_ts = sorted(v["last_seen_ts"] for v in agg.values())
    assert surviving_ts == [6.0, 7.0, 8.0, 9.0]


def test_evict_handles_missing_last_seen_ts_field():
    """Defensive: if an entry is missing last_seen_ts (e.g. stale state from
    an older worker version), eviction treats it as oldest, not crashes."""
    agg = {
        "fresh": _new_aggregated_entry(100.0),
        "stale": {"count": 5},  # missing last_seen_ts
    }
    n = _evict_aggregated_lru(agg, cap=1)
    assert n == 1
    # The "stale" entry (defaults to ts=0) should be evicted, "fresh" survives.
    assert "fresh" in agg
    assert "stale" not in agg


# ── _persist_state with eviction ────────────────────────────────────


def test_persist_state_evicts_when_over_cap(tmp_path):
    state_path = tmp_path / "state.json"
    # Build agg with 7 entries; cap at 3 forces eviction of oldest 4.
    agg = {f"k{i}": _new_aggregated_entry(float(i)) for i in range(7)}
    _persist_state(str(state_path), agg, deque(), cap=3)
    # On-disk + in-memory dict both shrunk.
    assert len(agg) == 3
    on_disk = json.loads(state_path.read_text())
    assert len(on_disk["aggregated"]) == 3
    # Survivors are the 3 youngest.
    surviving = sorted(int(k[1:]) for k in on_disk["aggregated"].keys())
    assert surviving == [4, 5, 6]


def test_persist_state_no_eviction_under_cap(tmp_path):
    state_path = tmp_path / "state.json"
    agg = {f"k{i}": _new_aggregated_entry(float(i)) for i in range(5)}
    _persist_state(str(state_path), agg, deque(), cap=AGGREGATED_KEY_CAP)
    assert len(agg) == 5
    on_disk = json.loads(state_path.read_text())
    assert len(on_disk["aggregated"]) == 5


# ── Constants sanity ─────────────────────────────────────────────────


def test_constants_have_sane_values():
    """Guard against accidentally bumping back to 1 MB or removing the cap."""
    assert MAX_TAIL_BYTES <= 512 * 1024, (
        f"MAX_TAIL_BYTES grew to {MAX_TAIL_BYTES} — Component B regression "
        "risk if > 512 KB under MALLOC_ARENA_MAX=2")
    assert MAX_TAIL_BYTES >= 64 * 1024, (
        "MAX_TAIL_BYTES too small — would starve the parser on busy logs")
    assert 100 <= AGGREGATED_KEY_CAP <= 100_000
