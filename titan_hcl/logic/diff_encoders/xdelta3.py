"""SPEC §24.5 — xdelta3 vcdiff encoder for SQLite/DuckDB/Kuzu DBs + large JSONs.

xdelta3 (Joshua MacDonald, https://github.com/jmacd/xdelta) produces compact
binary diffs (RFC 3284 vcdiff format). System binary at /usr/bin/xdelta3 is
the canonical invocation per CLAUDE.md environment.

§24 / Phase 5 (2026-05-19) — STREAMING refactor: upload-side returns a
`patch_path` (on-disk vcdiff or source) instead of `patch_bytes`. Closes the
67× memory amplification bug. The xdelta3 binary already writes its output
to a temp file — we now return that path with `patch_owned=True` so the
tarball builder streams from it and the caller unlinks after packing.
The full-ship fallback (when no baseline) returns the source path with
`patch_owned=False`.

Upload-side diff dict shape:
    {
      "diff_mode": "incremental" | "full",
      "patch_path": <str>,             # on-disk vcdiff (incremental) OR source (full)
      "patch_owned": bool,             # True for incremental (temp vcdiff), False for full
      "patch_size_bytes": <int>,       # tar info.size
      "merkle_root": <str>,            # sha256 over uncompressed current content
      "size_bytes": <int>,             # uncompressed current size
      "baseline_merkle_root": <str>,   # sha256 of baseline (anchor for restore validation)
      "encoder": "xdelta3",
    }

Restore-side: apply_diff still consumes `patch_bytes` (rebuilt from the
tarball member at extract time). Restore is rare; RSS there is a separate
concern from the daily-cascade upload-side leak this refactor closes.

Restore: write baseline to scratch + `xdelta3 -d -s baseline patch.vcdiff out`
+ verify sha256(out) == merkle_root. Per SPEC §24.5 + rFP §11 risk table:
xdelta3 patch refusal on source-hash mismatch → restore halts cleanly at
first failure; caller emits BACKUP_MERKLE_MISMATCH.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import tempfile
from typing import Optional

logger = logging.getLogger(__name__)


# Configurable but Maker should not change without SPEC update — these are
# documented contract behavior, not perf tuning knobs.
XDELTA3_BIN = "/usr/bin/xdelta3"
XDELTA3_COMPRESSION_LEVEL = 9  # -9 = max compression, slower; acceptable for backup


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _xdelta3_available() -> bool:
    return os.path.isfile(XDELTA3_BIN) and os.access(XDELTA3_BIN, os.X_OK)


def encode_diff(current_path: str, baseline_path: Optional[str] = None,
                **opts) -> dict:
    """Encode vcdiff patch vs baseline using xdelta3.

    If baseline_path is None / missing / empty → full-ship event.
    Else: spawns xdelta3 -e -s baseline current → patch.vcdiff; reads patch.

    If xdelta3 binary is unavailable on the host, falls back to full-ship
    (with a one-time WARN log) so backups never silently break in
    pathological environments. Production hosts (T1/T2/T3) have xdelta3
    installed per CLAUDE.md.
    """
    # §24.5.a / INV-BR-11 (2026-06-11): encode the SNAPSHOT, never the live source.
    # A large live JSON (≥ SMALL_JSON_THRESHOLD → routed here) previously full-
    # shipped from the LIVE path (patch_owned=False); if it changed between encode
    # and pack, the tar read fewer bytes than the recorded size → OSError
    # "unexpected end of data" (2026-06-10: filter_down_*/neuromodulator_state).
    # full_ship._race_safe_snapshot gives a stable inode (copy for .json/.jsonl +
    # ≤64MB; hardlink for big binary). For the full-ship branches it IS the payload
    # (owned → pack unlinks); for the vcdiff branch it's a transient diff INPUT
    # (unlinked after the patch is written). The shipped bytes + merkle are
    # identical to the old live-path full-ship (a point-in-time copy of the same
    # content). NOTE: when the caller already passed a SQLite online-backup image
    # (consistent + stable), this is a cheap zero-copy hardlink of it.
    from . import full_ship
    snap_path, snap_owned = full_ship._race_safe_snapshot(current_path)
    src_size = os.path.getsize(snap_path)
    src_root = _sha256_file(snap_path)

    def _full_ship_from_snapshot() -> dict:
        return {
            "diff_mode": "full",
            "patch_path": snap_path,
            "patch_owned": snap_owned,
            "patch_size_bytes": src_size,
            "merkle_root": src_root,
            "size_bytes": src_size,
            "baseline_merkle_root": None,
            "encoder": "xdelta3",
        }

    def _drop_snapshot_input() -> None:
        if snap_owned:
            try:
                os.unlink(snap_path)
            except OSError:
                pass

    if not _xdelta3_available():
        logger.warning(
            "[diff_encoders.xdelta3] xdelta3 binary unavailable at %s — "
            "falling back to full-ship for %s",
            XDELTA3_BIN, current_path)
        return _full_ship_from_snapshot()

    if (baseline_path is None
            or not os.path.exists(baseline_path)
            or os.path.getsize(baseline_path) == 0):
        return _full_ship_from_snapshot()

    baseline_root = _sha256_file(baseline_path)

    # Phase 5 (2026-05-19) — keep the temp .vcdiff on disk; caller owns
    # cleanup via patch_owned=True. The old code read it back into memory
    # then unlinked, which forced the entire patch into a Python bytes
    # object (the bug we're fixing).
    with tempfile.NamedTemporaryFile(suffix=".vcdiff", delete=False) as tmp:
        patch_path = tmp.name
    try:
        # xdelta3 -e -f -<level> -s <source> <input> <output>
        # -e encode, -f force overwrite, -9 max compression
        # Diff the SNAPSHOT (snap_path), never the live source.
        cmd = [
            XDELTA3_BIN, "-e", "-f", f"-{XDELTA3_COMPRESSION_LEVEL}",
            "-s", baseline_path, snap_path, patch_path,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=600)
        if result.returncode != 0:
            # On failure, clean up our temp file before propagating.
            try:
                os.unlink(patch_path)
            except OSError:
                pass
            raise RuntimeError(
                f"xdelta3 encode failed (rc={result.returncode}): "
                f"{result.stderr.decode('utf-8', errors='replace')}"
            )
        patch_size = os.path.getsize(patch_path)
    except Exception:
        # Any failure path: drop the temp file + the snapshot input before re-raising.
        try:
            os.unlink(patch_path)
        except OSError:
            pass
        _drop_snapshot_input()
        raise

    # The vcdiff is the payload; the snapshot was only the diff INPUT → drop it.
    _drop_snapshot_input()

    return {
        "diff_mode": "incremental",
        "patch_path": patch_path,
        "patch_owned": True,
        "patch_size_bytes": patch_size,
        "merkle_root": src_root,
        "size_bytes": src_size,
        "baseline_merkle_root": baseline_root,
        "encoder": "xdelta3",
    }


def apply_diff(baseline_path: Optional[str], diff_dict: dict,
               output_path: str, verify_output: bool = True) -> None:
    """Reverse of encode_diff.

    For diff_mode="full": writes diff_dict["patch_bytes"] directly to output_path.
    For diff_mode="incremental": runs `xdelta3 -d -s baseline patch out` after
                                  verifying baseline sha256 matches recorded value.

    Post-write check: sha256(output_path) == diff_dict["merkle_root"].
    Mismatch raises ValueError → caller halts + emits BACKUP_MERKLE_MISMATCH.
    """
    mode = diff_dict["diff_mode"]
    # Phase 5 (2026-05-19) — accept either patch_path (streaming) or
    # patch_bytes (legacy/restore-side via UnpackedEvent.diff_dict_for).
    patch_path = diff_dict.get("patch_path")
    patch = diff_dict.get("patch_bytes")
    expected_root = diff_dict["merkle_root"]
    expected_size = diff_dict["size_bytes"]

    parent = os.path.dirname(output_path) or "."
    os.makedirs(parent, exist_ok=True)

    if mode == "full":
        if patch_path is not None and os.path.exists(patch_path):
            with open(patch_path, "rb") as src, open(output_path, "wb") as dst:
                for chunk in iter(lambda: src.read(1 << 20), b""):
                    dst.write(chunk)
        else:
            with open(output_path, "wb") as f:
                f.write(patch)
    elif mode == "incremental":
        if baseline_path is None or not os.path.exists(baseline_path):
            raise ValueError(
                "xdelta3 diff_mode='incremental' requires baseline_path to exist"
            )
        # Verify baseline hash before applying — rFP §11 risk: xdelta3 patch
        # would silently apply against the wrong source and produce garbage
        # if we didn't anchor here. SPEC §24.7 ZK Vault Merkle commit is the
        # outer trust plane; this is the local plane's anchor.
        expected_baseline_root = diff_dict.get("baseline_merkle_root")
        if expected_baseline_root is not None:
            actual_baseline_root = _sha256_file(baseline_path)
            if actual_baseline_root != expected_baseline_root:
                raise ValueError(
                    f"xdelta3 apply baseline merkle_root mismatch: expected "
                    f"{expected_baseline_root}, got {actual_baseline_root} "
                    f"at {baseline_path}. Restore halts (would produce garbage)."
                )

        if not _xdelta3_available():
            raise RuntimeError(
                f"xdelta3 binary unavailable at {XDELTA3_BIN} — cannot apply "
                f"incremental patch. Install xdelta3 or restore from a full event."
            )

        # If caller gave us patch_path directly, use it as the vcdiff source
        # for xdelta3 (no temp file write needed). Otherwise materialize
        # patch_bytes to a scratch vcdiff first (restore path via tarball
        # extract).
        if patch_path is not None and os.path.exists(patch_path):
            vcdiff_path = patch_path
            owned_vcdiff = False
        else:
            with tempfile.NamedTemporaryFile(
                suffix=".vcdiff", delete=False
            ) as tmp:
                tmp.write(patch)
                vcdiff_path = tmp.name
            owned_vcdiff = True
        try:
            cmd = [XDELTA3_BIN, "-d", "-f", "-s", baseline_path,
                   vcdiff_path, output_path]
            result = subprocess.run(cmd, capture_output=True, timeout=600)
            if result.returncode != 0:
                # Clean up partial output before raising
                if os.path.exists(output_path):
                    os.unlink(output_path)
                raise RuntimeError(
                    f"xdelta3 decode failed (rc={result.returncode}): "
                    f"{result.stderr.decode('utf-8', errors='replace')}"
                )
        finally:
            if owned_vcdiff:
                try:
                    os.unlink(vcdiff_path)
                except OSError:
                    pass
    else:
        raise ValueError(f"Unknown diff_mode {mode!r} for xdelta3")

    # Post-write verification. NOTE: the incremental BASELINE check above stays
    # strict regardless of verify_output — applying a patch against a wrong source
    # produces garbage. Only these POST-APPLY (result vs stale recorded hash)
    # checks are downgraded when the source tarball is on-chain-arc-verified.
    actual_size = os.path.getsize(output_path)
    if actual_size != expected_size:
        msg = (f"xdelta3 apply size mismatch: expected {expected_size}, "
               f"got {actual_size} at {output_path}")
        if verify_output:
            raise ValueError(msg)
        logger.warning("[xdelta3] %s — proceeding: source tarball "
                       "on-chain-arc-verified; post-apply check advisory.", msg)
    actual_root = _sha256_file(output_path)
    if actual_root != expected_root:
        msg = (f"xdelta3 apply merkle_root mismatch: expected {expected_root}, "
               f"got {actual_root} at {output_path}")
        if verify_output:
            raise ValueError(msg)
        logger.warning("[xdelta3] %s — proceeding: source tarball "
                       "on-chain-arc-verified; post-apply check advisory.", msg)
