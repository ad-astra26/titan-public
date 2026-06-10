"""SPEC §24.5 — Full-ship encoder for FAISS indices + small JSONs.

For files where binary diff doesn't help (small JSONs already tiny; FAISS
indices are dense binary blobs whose xdelta3 patch tends to be ≈ their
original size). Ships the entire current content.

§24 / Phase 5 (2026-05-19) — STREAMING refactor: upload-side returns a
`patch_path` (on-disk pointer) instead of `patch_bytes` (in-memory bytes).
Closes the 67× memory amplification bug where reading inner_memory.db
(1.1 GB) into a single Python bytes object peaked backup_worker RSS at
1.7-2 GB during cascade. Now the bytes never leave disk — tarball builder
streams `patch_path` into the gzipped tar via `tar.add(patch_path)`.

Upload-side diff dict shape (consumed by pack_event_tarball):
    {
      "diff_mode": "full",
      "patch_path": <str>,             # on-disk path to upload-time bytes
      "patch_owned": bool,             # True iff caller must unlink after packing
      "patch_size_bytes": <int>,       # tar info.size + sanity check
      "merkle_root": <str>,            # sha256 over content (streamed)
      "size_bytes": <int>,             # original file size (== patch_size_bytes
                                        # here; same field for downstream
                                        # encoders that produce diffs of
                                        # different size from source)
      "encoder": "full_ship",
    }

Restore-side diff dict (consumed by apply_diff via extract_diff_dict) keeps
the original `patch_bytes` field — restore reads the tarball member into
bytes once, which is fine because restore is rare + happens out-of-band.

Restore: apply_diff writes patch_bytes (or streams from a fd) to output_path.
"""

from __future__ import annotations

import errno
import hashlib
import logging
import os
import shutil
import tempfile
import time
from typing import Optional

logger = logging.getLogger(__name__)


# Marker substring on every backup snapshot path. Recognized by the
# backup-scope skip-lists (titan_hcl/logic/backup.py: _tier_specs_from_paths
# IGNORE_SUFFIXES + _BACKUP_SKIP_PATTERNS) and by sweep_orphan_snapshots().
BKSNAP_MARKER = ".bksnap."
# SQLite online-backup images are named `<db>.snap.<rand>` by
# backup_sqlite_snapshot.snapshot_dest_for — a DISTINCT marker from `.bksnap.`
# (".bksnap." does not contain ".snap." as a substring). The scratch sweep must
# match it too, else leaked baseline DB snapshots (e.g. a 3.3 GB
# consciousness.db.snap.*) accumulate in .bksnap_scratch (AUDIT_bksnap_legacy_
# orphans_20260610). NOT reaped in-tree (SQLite snaps only ever live in scratch).
_SQLITE_SNAP_MARKER = ".snap."
# Dedicated out-of-source-tree scratch dir name (a child of the data root).
# CRITICAL: snapshots must NOT live inside a backed-up source dir, or the
# next event's rglob re-snapshots them (.bksnap.X.bksnap.Y …) and the
# orphan set grows exponentially — the cause of the 340,445-orphan /
# 358 GB-phantom-scope incident (2026-05-29). A child of data/ stays on
# the same filesystem (so os.link still works) but is outside every
# PERSONALITY_PATHS / WEEKLY_EXTRA_PATHS entry (those are data/<subdir>/),
# so rglob never walks into it.
_SCRATCH_DIRNAME = ".bksnap_scratch"


def _scratch_dir_for(current_path: str) -> str:
    """Resolve a same-filesystem scratch dir for `current_path`'s snapshot.

    Anchors on the nearest ancestor named `data` (the Titan data root) and
    returns `<data_root>/.bksnap_scratch`, created if absent. Falls back to
    the source dir's parent if no `data` ancestor exists (keeps same fs).
    """
    p = os.path.abspath(current_path)
    parts = p.split(os.sep)
    if "data" in parts:
        # last 'data' component is the data root for this path
        i = len(parts) - 1 - parts[::-1].index("data")
        data_root = os.sep.join(parts[: i + 1]) or os.sep
    else:
        data_root = os.path.dirname(os.path.dirname(p)) or os.path.dirname(p)
    scratch = os.path.join(data_root, _SCRATCH_DIRNAME)
    os.makedirs(scratch, exist_ok=True)
    return scratch


# Files at/below this size are copy-snapshotted (point-in-time, truncation-immune)
# rather than hardlinked. Covers the rotating .json/.jsonl logs + small state
# files; big binary DBs above this stay zero-copy hardlinks (they're page-updated,
# not truncated-to-zero, and copying multi-GB on a baseline is disk-heavy).
_COPY_SNAPSHOT_MAX_BYTES = 64 * 1024 * 1024  # 64 MB


def _should_copy_snapshot(current_path: str) -> bool:
    """True → use a real copy (separate inode, truncation-immune); False →
    hardlink (zero-copy). Copy the truncation-prone files: rotating text logs
    (.json/.jsonl, rewritten/truncated in place by live workers) and anything
    small enough that a copy is cheap."""
    if current_path.endswith((".json", ".jsonl")):
        return True
    try:
        return os.path.getsize(current_path) <= _COPY_SNAPSHOT_MAX_BYTES
    except OSError:
        return False


def _race_safe_snapshot(current_path: str) -> tuple[str, bool]:
    """Materialize a race-immune pointer to `current_path` bytes.

    Returns (snapshot_path, owned). `owned=True` means caller must unlink
    after pack. The snapshot's inode survives even if the source is
    unlinked / rolled / pruned between encode and pack (TOCTOU race fix
    for files like neuromodulator_snapshot_NNN.json that have rolling
    retention — NeuromodulatorSystem._save_state prunes oldest snapshots
    every 500 evals).

    Mechanism:
      1. os.link() — hardlink, atomic + zero-copy, same inode. Pack-time
         reads still see the bytes even if the source path is unlinked.
      2. On EXDEV (cross-device — temp dir on different filesystem from
         source) or EOPNOTSUPP (filesystem doesn't support hardlinks),
         fall back to a shutil.copy2 streaming copy.
      3. On any other OSError, fall back to returning the source path
         directly (preserving prior behavior — the race-window failure
         was already rare; better to ship the file than degrade backup).

    Snapshots live in `<data_root>/.bksnap_scratch` — OUT of the source
    tree (2026-05-29 fix) so a leaked snapshot can never be re-snapshotted
    by the next event's directory walk.
    """
    src_base = os.path.basename(current_path)
    scratch_dir = _scratch_dir_for(current_path)
    fd, snap_path = tempfile.mkstemp(
        prefix=f"{src_base}{BKSNAP_MARKER}", dir=scratch_dir)
    os.close(fd)
    os.unlink(snap_path)   # mkstemp leaves an empty file; remove before link

    # 2026-05-31 truncation-race fix (Maker): a hardlink SHARES the source inode,
    # so it survives unlink/rename but NOT in-place TRUNCATION — a fast-changing
    # log rotated via open("w")/ftruncate shrinks the shared inode, and pack-time
    # tar then reads fewer bytes than the encode-time size → tarfile
    # "unexpected end of data" (observed packing data/meta_cgn/*.jsonl etc.). A
    # real COPY is a SEPARATE inode: a point-in-time, internally-consistent
    # snapshot immune to truncation. Copy the truncation-prone files (rotating
    # .json/.jsonl + anything small); keep zero-copy hardlinks for big binary DBs
    # (sqlite is page-updated, not truncated-to-zero; a full copy would be
    # disk-heavy on a monthly baseline of multi-GB DBs).
    if _should_copy_snapshot(current_path):
        try:
            shutil.copy2(current_path, snap_path)
            return snap_path, True
        except OSError as ce:
            logger.warning(
                "[full_ship] copy snapshot failed for %s: %s — falling back to "
                "hardlink (truncation-race window reopens)", current_path, ce)
            try:
                os.unlink(snap_path)
            except OSError:
                pass
            fd2, snap_path = tempfile.mkstemp(
                prefix=f"{src_base}{BKSNAP_MARKER}", dir=scratch_dir)
            os.close(fd2)
            os.unlink(snap_path)

    try:
        os.link(current_path, snap_path)
        return snap_path, True
    except OSError as e:
        if e.errno in (errno.EXDEV, errno.EOPNOTSUPP, errno.EPERM):
            # Hardlink unsupported on this fs / cross-device — fall back
            # to a streaming copy. Slower but still race-immune.
            try:
                shutil.copy2(current_path, snap_path)
                return snap_path, True
            except OSError as e2:
                logger.warning(
                    "[full_ship] hardlink+copy snapshot failed for %s: %s; "
                    "falling back to direct source path (race window open)",
                    current_path, e2)
                # best-effort cleanup of empty placeholder
                try:
                    os.unlink(snap_path)
                except OSError:
                    pass
                return current_path, False
        # Source disappeared between getsize/_sha256 and link — caller
        # already validated existence, so this is the race window itself.
        # Re-raise so the upper layer can skip this file (encode_diff
        # caller treats None-return as skip; we want a clear signal).
        try:
            os.unlink(snap_path)
        except OSError:
            pass
        raise


def sweep_orphan_snapshots(data_root: str = "data",
                           max_age_s: float = 3600.0) -> int:
    """Remove leaked backup snapshots — from the out-of-tree scratch dir AND any
    stray in-tree LEGACY `.bksnap.` orphans. Best-effort; never raises.

    TWO passes (called at backup_worker boot → self-cleans fleet-wide):

      1. SCRATCH pass — `<data_root>/.bksnap_scratch`: reap `.bksnap.` (copy/
         hardlink snaps) AND `.snap.` (SQLite online-backup images, named
         `<db>.snap.<rand>`) older than `max_age_s` (so it never races a live
         event's in-flight snapshot). The `.snap.` coverage closes the
         leaked-baseline-DB-snapshot space leak (AUDIT_bksnap_legacy_orphans
         _20260610 — a 3.3 GB consciousness.db.snap.* observed on T3).

      2. IN-TREE LEGACY pass — the docstring's long-promised reap that the body
         never did (the bug): walk `data_root` and remove stray `.bksnap.` files
         left INSIDE the source dirs by the pre-2026-05-29 in-source-dir
         placement (the 340k-orphan-incident residue — 346k inert chained
         hardlinks on T3). The fixed creation path writes ONLY to
         `.bksnap_scratch`, so ANY in-tree `.bksnap.` is legacy → safe to reap
         (no age-gate needed). PRUNES `.bksnap_scratch` (pass 1 owns it) + the
         orchestrator's active `.orch_drip_*` drip dirs (live disk-persisted
         artifacts the BackupOrchestrator owns). Hardlink removal only drops the
         extra name; the canonical source inode is untouched → cannot lose data,
         frees file-count / directory entries.

    Returns the number of orphans removed.
    """
    removed = 0
    now = time.time()
    scratch = os.path.join(data_root, _SCRATCH_DIRNAME)
    # ── Pass 1: scratch dir (age-gated — never race an in-flight snapshot) ──
    try:
        if os.path.isdir(scratch):
            for name in os.listdir(scratch):
                if BKSNAP_MARKER not in name and _SQLITE_SNAP_MARKER not in name:
                    continue
                fp = os.path.join(scratch, name)
                try:
                    if now - os.path.getmtime(fp) < max_age_s:
                        continue
                    os.unlink(fp)
                    removed += 1
                except OSError:
                    pass
    except OSError:
        pass
    # ── Pass 2: in-tree legacy `.bksnap.` orphans (the never-reaped residue) ──
    try:
        for dirpath, dirnames, filenames in os.walk(data_root):
            # Prune the scratch dir (pass 1 owns it) + the orchestrator's active
            # drip dirs (live artifacts — not orphans).
            dirnames[:] = [d for d in dirnames
                           if d != _SCRATCH_DIRNAME
                           and not d.startswith(".orch_drip_")]
            for name in filenames:
                if BKSNAP_MARKER not in name:
                    continue
                try:
                    os.unlink(os.path.join(dirpath, name))
                    removed += 1
                except OSError:
                    pass
    except OSError:
        pass
    return removed


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def encode_diff(current_path: str, baseline_path: Optional[str] = None,
                **opts) -> dict:
    """Encode full content (baseline_path is ignored — full-ship is by design).

    STREAMING (Phase 5, 2026-05-19): returns `patch_path` pointing at a
    race-immune snapshot of the current file. No file content loaded into
    Python memory. The tarball builder streams from this path via
    `tar.add()`.

    TOCTOU race fix (2026-05-23): rolling-retention sources (e.g.
    neuromodulator_snapshot_NNN.json — pruned every 500 evals at
    neuromodulator.py:552) could vanish between encode and pack, causing
    pack_event_tarball:209 to raise ValueError and the entire unified_v2
    pipeline to fall back to the bug-laden legacy cascade. We now hardlink
    the source to a snapshot path at encode time — the inode survives even
    if the source is unlinked, eliminating the race. `patch_owned=True`
    tells pack_event_tarball to unlink the snapshot after packing.

    Order matters: take the snapshot FIRST, then compute size + hash on
    the snapshot (not the source). That way size/hash and the bytes packed
    are guaranteed to be the same atomic snapshot — no race where size
    reflects the source at T1 but bytes reflect a rotated source at T2.
    """
    snap_path, owned = _race_safe_snapshot(current_path)
    # 2026-05-24 orphan cleanup hygiene: if size/hash computation raises
    # (source vanished race window — small but real), unlink the
    # owned snapshot so we don't leak orphan hardlinks. Without this
    # try/except, orphans accumulate fleet-wide (757 orphans found on
    # T1 alone before this fix — hardlinks so no disk cost but
    # filename clutter that scales with backup cadence).
    try:
        size = os.path.getsize(snap_path)
        root = _sha256_file(snap_path)
    except Exception:
        if owned:
            try:
                os.unlink(snap_path)
            except OSError:
                pass
        raise
    return {
        "diff_mode": "full",
        "patch_path": snap_path,
        "patch_owned": owned,
        "patch_size_bytes": size,
        "merkle_root": root,
        "size_bytes": size,
        "encoder": "full_ship",
    }


def apply_diff(baseline_path: Optional[str], diff_dict: dict,
               output_path: str, verify_output: bool = True) -> None:
    """Write the full content + verify.

    Accepts either `patch_bytes` (restore-side path: tarball member extracted
    into bytes via UnpackedEvent.diff_dict_for) or `patch_path` (round-trip
    testing path: encoder output consumed directly without going through
    tarball pack/unpack). Streams from `patch_path` if present so restore of
    large files doesn't materialize them in RAM.
    """
    if diff_dict.get("diff_mode") != "full":
        raise ValueError(
            f"full_ship only supports diff_mode='full', got {diff_dict.get('diff_mode')!r}"
        )
    parent = os.path.dirname(output_path) or "."
    os.makedirs(parent, exist_ok=True)
    patch_path = diff_dict.get("patch_path")
    if patch_path is not None and os.path.exists(patch_path):
        # STREAMING path
        with open(patch_path, "rb") as src, open(output_path, "wb") as dst:
            for chunk in iter(lambda: src.read(1 << 20), b""):
                dst.write(chunk)
    else:
        with open(output_path, "wb") as f:
            f.write(diff_dict["patch_bytes"])

    expected_size = diff_dict["size_bytes"]
    actual_size = os.path.getsize(output_path)
    if actual_size != expected_size:
        msg = (f"full_ship apply size mismatch: expected {expected_size}, "
               f"got {actual_size} at {output_path}")
        if verify_output:
            raise ValueError(msg)
        logger.warning("[full_ship] %s — proceeding: source tarball "
                       "on-chain-arc-verified; post-apply check advisory.", msg)
    actual_root = _sha256_file(output_path)
    if actual_root != diff_dict["merkle_root"]:
        msg = (f"full_ship apply merkle_root mismatch: expected "
               f"{diff_dict['merkle_root']}, got {actual_root} at {output_path}")
        if verify_output:
            raise ValueError(msg)
        logger.warning("[full_ship] %s — proceeding: source tarball "
                       "on-chain-arc-verified; post-apply check advisory.", msg)
