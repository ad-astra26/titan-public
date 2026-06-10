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
    """patch_path must NOT be the source path — it must be a snapshot.

    2026-05-29: the snapshot must also live OUTSIDE the source directory
    (in a .bksnap_scratch dir), so a leaked snapshot can never be
    re-snapshotted by the next event's directory walk — the exponential
    blowup that produced 340,445 orphan hardlinks / 358 GB phantom scope.
    """
    source = tmp_path / "rolling_source.json"
    source.write_text('{"x": 1}')
    dd = full_ship.encode_diff(str(source))
    assert dd["patch_path"] != str(source)
    assert dd["patch_owned"] is True
    assert os.path.exists(dd["patch_path"])
    # Snapshot is OUT of the source dir (regression guard for the orphan blowup).
    assert os.path.dirname(dd["patch_path"]) != str(tmp_path)
    assert full_ship._SCRATCH_DIRNAME in dd["patch_path"]
    assert full_ship.BKSNAP_MARKER in os.path.basename(dd["patch_path"])


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


def test_snapshot_copy_for_json_hardlink_for_big_binary(tmp_path):
    """Truncation-race contract (2026-05-31): fast-changing/small files
    (.json/.jsonl + ≤64MB) are COPY-snapshotted — a SEPARATE inode, immune to
    in-place truncation (a rotating log shrinking via open('w')/ftruncate can't
    corrupt the snapshot). Big binary files stay zero-copy hardlinks."""
    # .json (rotating/truncation-prone) → COPY (different inode), bytes preserved
    j = tmp_path / "rotating.json"
    j.write_text("hello")
    ddj = full_ship.encode_diff(str(j))
    assert os.stat(j).st_ino != os.stat(ddj["patch_path"]).st_ino, (
        ".json must be a COPY (separate inode), not a hardlink — else "
        "in-place truncation corrupts the snapshot")
    with open(ddj["patch_path"], "rb") as f:
        assert f.read() == b"hello"

    # big binary (>64MB, non-json) → hardlink (same inode, zero-copy)
    big = tmp_path / "big.bin"
    with open(big, "wb") as f:
        f.write(b"\0" * (65 * 1024 * 1024))
    ddb = full_ship.encode_diff(str(big))
    assert os.stat(big).st_ino == os.stat(ddb["patch_path"]).st_ino, (
        "big binary should be a zero-copy hardlink (same inode)")


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


def test_sweep_orphan_snapshots_removes_aged_orphans(tmp_path):
    """sweep_orphan_snapshots removes aged .bksnap orphans from the scratch
    dir while leaving canonical sources (and fresh in-flight snapshots)
    untouched. Hardlink removal must not affect the source inode."""
    import time as _t

    data_root = tmp_path / "data"
    src_dir = data_root / "neural_nervous_system"
    src_dir.mkdir(parents=True)
    source = src_dir / "reward_log.jsonl"
    source.write_text("real data")

    # Encode → creates an owned snapshot in data/.bksnap_scratch.
    dd = full_ship.encode_diff(str(source))
    assert os.path.exists(dd["patch_path"])
    scratch = data_root / full_ship._SCRATCH_DIRNAME
    assert scratch.is_dir()

    # Fresh snapshot (age 0) must survive a sweep with a 1h floor.
    assert full_ship.sweep_orphan_snapshots(str(data_root), max_age_s=3600.0) == 0
    assert os.path.exists(dd["patch_path"])

    # Age it past the floor → swept; canonical source untouched.
    old = _t.time() - 7200
    os.utime(dd["patch_path"], (old, old))
    removed = full_ship.sweep_orphan_snapshots(str(data_root), max_age_s=3600.0)
    assert removed == 1
    assert not os.path.exists(dd["patch_path"])
    assert source.exists() and source.read_text() == "real data"


def test_sweep_reaps_in_tree_legacy_bksnap_orphans(tmp_path):
    """The fix (AUDIT_bksnap_legacy_orphans_20260610): stray IN-TREE `.bksnap.`
    files — the pre-2026-05-29 in-source-dir legacy incl. recursive chains — are
    reaped (regardless of age), while the canonical source + the active
    `.bksnap_scratch` + `.orch_drip_*` dirs are left untouched."""
    data_root = tmp_path / "data"
    src_dir = data_root / "reasoning"
    src_dir.mkdir(parents=True)
    source = src_dir / "rfp_alpha_activation.json"
    source.write_text("real data")
    # Legacy in-tree orphans INSIDE the source dir (incl. a recursive chain).
    leg1 = src_dir / "rfp_alpha_activation.json.bksnap.aaaa"
    leg2 = src_dir / "rfp_alpha_activation.json.bksnap.aaaa.bksnap.bbbb"
    leg1.write_text("orphan"); leg2.write_text("orphan")
    # An in-flight scratch snapshot (age 0) + an active drip artifact — survive.
    scratch = data_root / full_ship._SCRATCH_DIRNAME
    scratch.mkdir(parents=True)
    fresh_scratch = scratch / "foo.json.bksnap.fresh"
    fresh_scratch.write_text("in-flight")
    drip = data_root / "backups" / ".orch_drip_T1"
    drip.mkdir(parents=True)
    drip_art = drip / "patch_foo.json.bksnap.live"
    drip_art.write_text("active drip")

    removed = full_ship.sweep_orphan_snapshots(str(data_root), max_age_s=3600.0)
    assert removed == 2                              # the 2 in-tree legacy orphans
    assert not leg1.exists() and not leg2.exists()
    assert source.exists() and source.read_text() == "real data"  # source safe
    assert fresh_scratch.exists()                    # in-flight scratch (age 0) safe
    assert drip_art.exists()                         # active drip dir pruned, not reaped


def test_sweep_reaps_aged_sqlite_snap_in_scratch(tmp_path):
    """The scratch pass also reaps aged `<db>.snap.<rand>` SQLite online-backup
    images (the leaked-baseline-snapshot space leak) — a marker distinct from
    `.bksnap.`."""
    import time as _t

    data_root = tmp_path / "data"
    scratch = data_root / full_ship._SCRATCH_DIRNAME
    scratch.mkdir(parents=True)
    snap = scratch / "consciousness.db.snap.deadbeef"
    snap.write_bytes(b"x" * 1024)
    # Fresh → survives (age-gated).
    assert full_ship.sweep_orphan_snapshots(str(data_root), max_age_s=3600.0) == 0
    assert snap.exists()
    # Aged past the floor → reaped.
    old = _t.time() - 7200
    os.utime(snap, (old, old))
    assert full_ship.sweep_orphan_snapshots(str(data_root), max_age_s=3600.0) == 1
    assert not snap.exists()


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
