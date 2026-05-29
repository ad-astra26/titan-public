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
from typing import Optional

logger = logging.getLogger(__name__)


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
    """
    # Place the snapshot beside the source so we stay on the same
    # filesystem (avoids EXDEV in practice). Suffix marks it as backup-
    # owned so a sweeper would recognize the intent.
    src_dir = os.path.dirname(current_path) or "."
    src_base = os.path.basename(current_path)
    fd, snap_path = tempfile.mkstemp(
        prefix=f".{src_base}.bksnap.", dir=src_dir)
    os.close(fd)
    os.unlink(snap_path)   # mkstemp leaves an empty file; remove before link

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
               output_path: str) -> None:
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
        raise ValueError(
            f"full_ship apply size mismatch: expected {expected_size}, "
            f"got {actual_size} at {output_path}"
        )
    actual_root = _sha256_file(output_path)
    if actual_root != diff_dict["merkle_root"]:
        raise ValueError(
            f"full_ship apply merkle_root mismatch: expected "
            f"{diff_dict['merkle_root']}, got {actual_root} at {output_path}"
        )
