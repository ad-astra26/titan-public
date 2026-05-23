"""Tests for the full_ship encoder's race-safe snapshot.

Closes the TOCTOU race that caused unified_v2 backups to fall back to the
legacy cascade when a rolling-retention source (e.g.
neuromodulator_snapshot_NNN.json — pruned every 500 evals at
neuromodulator.py:552) vanished between encode and pack
(pack_event_tarball:209 raise ValueError → backup.py:236 catches → legacy).

D-SPEC-123 follow-up, 2026-05-23.
"""
from __future__ import annotations

import hashlib
import os
import tempfile

import pytest

from titan_hcl.logic.diff_encoders import full_ship
from titan_hcl.logic import backup_event_tarball


def test_encode_diff_returns_owned_snapshot_not_source(tmp_path):
    """patch_path must NOT be the source path — it must be a snapshot."""
    source = tmp_path / "rolling_source.json"
    source.write_text('{"x": 1}')
    dd = full_ship.encode_diff(str(source))
    assert dd["patch_path"] != str(source)
    assert dd["patch_owned"] is True
    # Snapshot file exists; sits in the same directory as source.
    assert os.path.exists(dd["patch_path"])
    assert os.path.dirname(dd["patch_path"]) == str(tmp_path)


def test_snapshot_survives_source_deletion(tmp_path):
    """The race condition: source is deleted between encode and pack.
    Snapshot's inode must still be readable + bit-identical."""
    source = tmp_path / "rolling.json"
    original_bytes = b'{"eval": 65000, "DA": 0.5}'
    source.write_bytes(original_bytes)
    expected_sha = hashlib.sha256(original_bytes).hexdigest()

    dd = full_ship.encode_diff(str(source))
    # Simulate NeuromodulatorSystem rolling: source is unlinked between
    # encode and pack.
    source.unlink()
    assert not source.exists()

    # Snapshot must still be readable + bit-identical.
    assert os.path.exists(dd["patch_path"])
    with open(dd["patch_path"], "rb") as f:
        snap_bytes = f.read()
    assert snap_bytes == original_bytes
    assert hashlib.sha256(snap_bytes).hexdigest() == expected_sha
    assert dd["merkle_root"] == expected_sha


def test_snapshot_is_hardlink_when_same_filesystem(tmp_path):
    """On same-fs (the common case), the snapshot is a hardlink — same
    inode, no extra disk usage."""
    source = tmp_path / "src.json"
    source.write_text("hello")
    dd = full_ship.encode_diff(str(source))
    src_stat = os.stat(source)
    snap_stat = os.stat(dd["patch_path"])
    assert src_stat.st_ino == snap_stat.st_ino, (
        "snapshot should be a hardlink (same inode) on same-fs")
    assert snap_stat.st_nlink >= 2, "hardlink should bump nlink"


def test_snapshot_survives_atomic_replace(tmp_path, monkeypatch):
    """The REAL race model: NeuromodulatorSystem._save_state writes a tmp
    file then `os.replace(tmp, path)` — this swaps the inode at the source
    path to a NEW inode. The hardlink snapshot still points at the OLD
    inode, so its bytes are preserved.

    Simulates the source being atomic-replaced with different content
    immediately after snapshot, mirroring the production rotation pattern.
    Size + hash + bytes must reflect the snapshot (original), not the
    rotated source.
    """
    source = tmp_path / "src.json"
    original = b"original content"
    source.write_bytes(original)

    real_snapshot = full_ship._race_safe_snapshot

    def racing_snapshot(p):
        snap, owned = real_snapshot(p)
        # Source is atomic-replaced with new content RIGHT AFTER snapshot
        # (the production pattern: write to tmp, then os.replace).
        tmp = str(p) + ".tmp"
        with open(tmp, "wb") as f:
            f.write(b"rotated_to_different_content")
        os.replace(tmp, p)
        return snap, owned

    monkeypatch.setattr(full_ship, "_race_safe_snapshot", racing_snapshot)

    dd = full_ship.encode_diff(str(source))
    # Size + hash + bytes reflect the SNAPSHOT (original), not the rotated
    # source — because os.replace assigned source a NEW inode while the
    # hardlink keeps the old inode alive.
    assert dd["size_bytes"] == len(original)
    assert dd["patch_size_bytes"] == len(original)
    assert dd["merkle_root"] == hashlib.sha256(original).hexdigest()
    with open(dd["patch_path"], "rb") as f:
        assert f.read() == original
    # Sanity: source path now has the rotated content.
    assert source.read_bytes() == b"rotated_to_different_content"


def test_pack_event_tarball_skips_missing_patch_path_no_raise(tmp_path):
    """Defense-in-depth: if a future encoder regression returns a
    patch_path that vanishes before pack, the tarball builder must SKIP
    that file with a WARN — not raise (which would abort the unified_v2
    event and fall back to legacy)."""
    from titan_hcl.logic.backup_event_tarball import (
        FileDiffSpec, pack_event_tarball,
    )

    good = tmp_path / "good.json"
    good.write_text("good content")
    vanished = tmp_path / "vanished.json"
    vanished.write_text("about to disappear")

    dd_good = full_ship.encode_diff(str(good))
    dd_bad = full_ship.encode_diff(str(vanished))
    # Force the "vanished" snapshot to actually vanish (simulating a
    # future encoder regression that doesn't snapshot).
    os.unlink(dd_bad["patch_path"])
    assert not os.path.exists(dd_bad["patch_path"])

    output_path = str(tmp_path / "event.tar.zst")
    specs = [
        FileDiffSpec(arc_name="good.json", diff_dict=dd_good),
        FileDiffSpec(arc_name="vanished.json", diff_dict=dd_bad),
    ]
    # Must NOT raise.
    info = pack_event_tarball(
        event_id="test-event-1", event_type="incremental",
        component="personality", file_specs=specs,
        output_path=output_path,
    )
    # Tarball produced; the good file is in, vanished is skipped.
    assert os.path.exists(output_path)
    assert info["size_bytes"] > 0
    assert info["file_count"] == 1                  # actually packed
    assert info["files_requested"] == 2             # caller asked for two
    assert info["files_skipped_vanished"] == 1      # patch_path-vanished guard fired
    assert "good.json" in info["packed_arc_names"]
    assert "vanished.json" not in info["packed_arc_names"]


def test_full_ship_roundtrip_unchanged(tmp_path):
    """End-to-end: encode + read back through patch_path → bit-identical."""
    source = tmp_path / "x.json"
    content = b'{"key": "value", "n": 42}'
    source.write_bytes(content)
    dd = full_ship.encode_diff(str(source))
    with open(dd["patch_path"], "rb") as f:
        assert f.read() == content
    assert dd["encoder"] == "full_ship"
    assert dd["diff_mode"] == "full"
