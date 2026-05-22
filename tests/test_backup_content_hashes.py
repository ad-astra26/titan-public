"""Tests for SPEC §24.6 skip-if-unchanged content_hash cache.

Per rFP §5.2 test coverage:
  - Skip behavior (cheap mtime path + SHA256 confirm path)
  - Cache invalidation on hash change
  - Skipped_file pointer construction for manifest event
  - Cache corruption / cross-titan rejection → rebuild from scratch
"""

import json
import os
import time
from pathlib import Path

import pytest

from titan_hcl.logic.backup_content_hashes import (
    CONTENT_HASH_SCHEMA_VERSION,
    ContentHashCache,
)


def _write(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


# ── construction + round-trip ────────────────────────────────────────────


def test_fresh_cache_has_empty_files(tmp_path):
    c = ContentHashCache(titan_id="T1", base_dir=str(tmp_path))
    assert c._data["files"] == {}
    assert c._data["schema_version"] == CONTENT_HASH_SCHEMA_VERSION
    assert c._data["titan_id"] == "T1"


def test_save_and_reload_round_trip(tmp_path):
    c1 = ContentHashCache(titan_id="T1", base_dir=str(tmp_path))
    c1.record_upload("data/x.db", sha256="aa" * 32,
                     event_id="ev1", tx_id="ar_tx_1", size_bytes=100)
    c1.save()
    c2 = ContentHashCache.load(titan_id="T1", base_dir=str(tmp_path))
    entry = c2.get_cached_entry("data/x.db")
    assert entry["sha256"] == "aa" * 32
    assert entry["last_upload_event_id"] == "ev1"
    assert entry["last_upload_tx_id"] == "ar_tx_1"
    assert entry["size_bytes"] == 100


def test_load_missing_file_starts_fresh(tmp_path):
    c = ContentHashCache.load(titan_id="T1", base_dir=str(tmp_path))
    assert c._data["files"] == {}


def test_load_corrupt_file_starts_fresh(tmp_path):
    """Cache is rebuildable from disk + manifest — corruption → fresh start
    (NOT a hard error like manifest corruption per §24.3)."""
    p = os.path.join(str(tmp_path), "backup_content_hashes_T1.json")
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        f.write("{not json")
    c = ContentHashCache.load(titan_id="T1", base_dir=str(tmp_path))
    assert c._data["files"] == {}


def test_load_cross_titan_starts_fresh(tmp_path):
    """Wrong titan_id in file → fresh start (don't trust foreign hashes)."""
    p = os.path.join(str(tmp_path), "backup_content_hashes_T1.json")
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump({"titan_id": "T2", "schema_version": 1,
                   "files": {"data/x.db": {"sha256": "bb" * 32}}}, f)
    c = ContentHashCache.load(titan_id="T1", base_dir=str(tmp_path))
    assert c._data["files"] == {}


# ── is_unchanged_since_last_upload semantics ─────────────────────────────


def test_unchanged_when_no_cache_entry_returns_false(tmp_path):
    """First-ever check — no cached entry → must ship (False)."""
    src = tmp_path / "data/x.db"
    _write(str(src), b"first content")
    c = ContentHashCache(titan_id="T1", base_dir=str(tmp_path))
    unchanged, current = c.is_unchanged_since_last_upload(str(src))
    assert unchanged is False
    assert current is not None  # SHA256 returned for the upcoming record


def test_unchanged_when_file_missing_returns_false(tmp_path):
    c = ContentHashCache(titan_id="T1", base_dir=str(tmp_path))
    unchanged, current = c.is_unchanged_since_last_upload(
        str(tmp_path / "data/ghost.db"))
    assert unchanged is False
    assert current is None


def test_cheap_path_unchanged_when_mtime_le_last_upload(tmp_path):
    """Cheap mtime path: file untouched since last upload → True, None."""
    src = tmp_path / "data/x.db"
    _write(str(src), b"content")
    file_sha = "ab" * 32  # don't care about real hash for this path
    c = ContentHashCache(titan_id="T1", base_dir=str(tmp_path))
    # Record an upload AFTER the file's mtime
    future_ts = os.path.getmtime(str(src)) + 100.0
    c.record_upload(str(src), sha256=file_sha, event_id="ev1",
                    tx_id="ar_tx", now=future_ts)
    unchanged, current = c.is_unchanged_since_last_upload(str(src))
    assert unchanged is True
    assert current is None  # cheap path — no recompute


def test_recompute_path_unchanged_when_mtime_moved_but_content_same(tmp_path):
    """File touched (mtime bumped) but content same → recompute confirms
    unchanged → True, recomputed_sha."""
    src = tmp_path / "data/x.db"
    _write(str(src), b"content_xyz")
    c = ContentHashCache(titan_id="T1", base_dir=str(tmp_path))
    real_sha = c.compute_current_hash(str(src))
    past_ts = os.path.getmtime(str(src)) - 10.0
    c.record_upload(str(src), sha256=real_sha, event_id="ev1",
                    tx_id="ar_tx", now=past_ts)
    # Touch the file (bump mtime) without changing content
    new_mtime = past_ts + 100.0
    os.utime(str(src), (new_mtime, new_mtime))
    unchanged, current = c.is_unchanged_since_last_upload(str(src))
    assert unchanged is True
    assert current == real_sha  # recomputed


def test_changed_detected_when_content_changes(tmp_path):
    """File mtime moved AND content changed → False, new_sha."""
    src = tmp_path / "data/x.db"
    _write(str(src), b"version_1")
    c = ContentHashCache(titan_id="T1", base_dir=str(tmp_path))
    v1_sha = c.compute_current_hash(str(src))
    c.record_upload(str(src), sha256=v1_sha, event_id="ev1",
                    tx_id="ar_tx", now=os.path.getmtime(str(src)) - 1.0)
    # Mutate content
    _write(str(src), b"version_2_different")
    # Force mtime to be newer than recorded
    new_mtime = time.time() + 100.0
    os.utime(str(src), (new_mtime, new_mtime))
    unchanged, current = c.is_unchanged_since_last_upload(str(src))
    assert unchanged is False
    assert current is not None
    assert current != v1_sha


# ── skipped_file pointer construction ────────────────────────────────────


def test_make_skipped_file_pointer_returns_none_for_unknown(tmp_path):
    c = ContentHashCache(titan_id="T1", base_dir=str(tmp_path))
    assert c.make_skipped_file_pointer("data/never_uploaded.db") is None


def test_make_skipped_file_pointer_for_known(tmp_path):
    c = ContentHashCache(titan_id="T1", base_dir=str(tmp_path))
    c.record_upload("data/foo.db", sha256="cc" * 32,
                    event_id="ev_42", tx_id="ar_tx_42")
    ptr = c.make_skipped_file_pointer("data/foo.db")
    assert ptr == {
        "path": "data/foo.db",
        "prev_tx_id": "ar_tx_42",
        "prev_event_id": "ev_42",
    }


def test_skipped_pointer_for_local_event_id_only(tmp_path):
    """Local plane records event_id but tx_id may be None
    (local-only Titans have no Arweave tx)."""
    c = ContentHashCache(titan_id="T2", base_dir=str(tmp_path))
    c.record_upload("data/foo.db", sha256="dd" * 32,
                    event_id="ev_local_7", tx_id=None)
    ptr = c.make_skipped_file_pointer("data/foo.db")
    assert ptr["prev_event_id"] == "ev_local_7"
    assert ptr["prev_tx_id"] is None


# ── record_upload validates inputs ───────────────────────────────────────


def test_record_upload_uses_provided_size(tmp_path):
    c = ContentHashCache(titan_id="T1", base_dir=str(tmp_path))
    c.record_upload("data/x.db", sha256="ee" * 32, event_id="ev1",
                    tx_id="ar_tx", size_bytes=12345)
    assert c.get_cached_entry("data/x.db")["size_bytes"] == 12345


def test_record_upload_infers_size_from_disk(tmp_path):
    src = tmp_path / "data/x.db"
    _write(str(src), b"a" * 7777)
    c = ContentHashCache(titan_id="T1", base_dir=str(tmp_path))
    c.record_upload(str(src), sha256="ee" * 32, event_id="ev1",
                    tx_id="ar_tx")  # size_bytes omitted
    assert c.get_cached_entry(str(src))["size_bytes"] == 7777


# ── full flow scenario ──────────────────────────────────────────────────


def test_full_skip_dedup_cycle(tmp_path):
    """End-to-end: upload → cache → next-day-no-change → skip → manifest pointer."""
    src = tmp_path / "data/y.db"
    _write(str(src), b"persistent state v1")

    c = ContentHashCache(titan_id="T1", base_dir=str(tmp_path))

    # Day 1: first upload
    unchanged, current = c.is_unchanged_since_last_upload(str(src))
    assert unchanged is False  # never uploaded
    c.record_upload(str(src), sha256=current, event_id="ev_day1",
                    tx_id="ar_tx_day1")
    c.save()

    # Reload (simulating next-day boot)
    c2 = ContentHashCache.load(titan_id="T1", base_dir=str(tmp_path))

    # Day 2: file untouched → unchanged
    unchanged, current = c2.is_unchanged_since_last_upload(str(src))
    assert unchanged is True

    # Build manifest skip pointer
    ptr = c2.make_skipped_file_pointer(str(src))
    assert ptr is not None
    assert ptr["prev_tx_id"] == "ar_tx_day1"
    assert ptr["prev_event_id"] == "ev_day1"

    # Day 3: file mutated → changed
    _write(str(src), b"persistent state v2 -- MUTATED")
    os.utime(str(src), (time.time() + 1000, time.time() + 1000))
    unchanged, current_v2 = c2.is_unchanged_since_last_upload(str(src))
    assert unchanged is False
    assert current_v2 != c2.get_cached_entry(str(src))["sha256"]

    # Record upload at v2
    c2.record_upload(str(src), sha256=current_v2, event_id="ev_day3",
                     tx_id="ar_tx_day3")
    # Pointer now points to day-3 upload
    ptr_v2 = c2.make_skipped_file_pointer(str(src))
    assert ptr_v2["prev_event_id"] == "ev_day3"
