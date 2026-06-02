"""SPEC §24.5 — TimeChain append-only tail-byte diff encoder.

TimeChain `.bin` chains are fixed-record append-only files: every block is
written as a deterministic byte sequence at a deterministic offset, and the
file only ever grows at the tail. Diff = tail bytes since the previous
event's recorded `prev_offset_bytes`.

§24 / Phase 5 (2026-05-19) — STREAMING refactor: upload-side returns
`patch_path` (on-disk pointer) instead of `patch_bytes`. Closes the 67×
memory amplification bug. For "full" mode patch_path = source file
(patch_owned=False). For "tail" mode we stream the tail bytes into a temp
file (patch_owned=True) so the chain file never gets fully read into RAM.

Restore: concat baseline chain bytes + all tail bytes from incrementals in
chronological order. Per-event merkle_root verifies the post-event state.

Upload-side diff dict shape (consumed by pack_event_tarball):
    {
      "diff_mode": "tail" | "full",
      "patch_path": <str>,             # on-disk path to upload-time bytes
      "patch_owned": bool,             # True iff caller must unlink (temp tail)
      "patch_size_bytes": <int>,       # tar info.size (= tail length OR file size)
      "merkle_root": <str>,            # sha256 over uncompressed current content
      "size_bytes": <int>,             # uncompressed current size
      "prev_offset_bytes": <int>,      # offset at which patch_bytes start
                                       # (== baseline size for incremental;
                                       #     0 for full/baseline event)
      "block_range": [first, last],    # block-number range covered by this event
      "encoder": "timechain_tail",
    }

Restore-side diff dict keeps `patch_bytes` field (rebuilt from tarball
member at extract time).
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from typing import Optional

logger = logging.getLogger(__name__)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _stream_tail_to_temp(current_path: str, start_offset: int) -> tuple[str, int]:
    """Copy bytes [start_offset:] from current_path into a temp file. Streams
    in 1 MiB chunks so tail bytes never fully materialize in RAM.

    Returns (temp_path, byte_count).
    """
    with tempfile.NamedTemporaryFile(suffix=".tail.bin", delete=False) as tmp:
        tail_path = tmp.name
    written = 0
    try:
        with open(current_path, "rb") as src, open(tail_path, "wb") as dst:
            src.seek(start_offset)
            while True:
                chunk = src.read(1 << 20)  # 1 MiB
                if not chunk:
                    break
                dst.write(chunk)
                written += len(chunk)
        return tail_path, written
    except Exception:
        try:
            os.unlink(tail_path)
        except OSError:
            pass
        raise


def encode_diff(current_path: str, baseline_path: Optional[str] = None,
                block_range: Optional[tuple] = None, **opts) -> dict:
    """Encode tail-byte diff vs baseline (streaming).

    Args:
        current_path: Path to the current `.bin` chain file.
        baseline_path: Path to the baseline `.bin` chain (the file as it
            existed at the most recent baseline event). If None or the
            baseline file is empty/missing → produces a FULL event
            (patch_path = current_path).
        block_range: (first_block, last_block) tuple covered by this event.
            Caller passes through from the chain index/header. Stored in
            diff_dict for restore-time §3.3 surgical_repair targeting.

    Returns: diff_dict (shape documented at module level).
    """
    current_size = os.path.getsize(current_path)
    current_root = _sha256_file(current_path)

    if baseline_path is None or not os.path.exists(baseline_path):
        return {
            "diff_mode": "full",
            "patch_path": current_path,
            "patch_owned": False,
            "patch_size_bytes": current_size,
            "merkle_root": current_root,
            "size_bytes": current_size,
            "prev_offset_bytes": 0,
            "block_range": list(block_range) if block_range else None,
            "encoder": "timechain_tail",
        }

    baseline_size = os.path.getsize(baseline_path)
    if baseline_size > current_size:
        # Truncation or rebase — chain went backward. Append-only invariant
        # is violated; surface as full ship.
        return {
            "diff_mode": "full",
            "patch_path": current_path,
            "patch_owned": False,
            "patch_size_bytes": current_size,
            "merkle_root": current_root,
            "size_bytes": current_size,
            "prev_offset_bytes": 0,
            "block_range": list(block_range) if block_range else None,
            "encoder": "timechain_tail",
        }

    # Append-only diff: tail bytes from offset baseline_size onward
    if baseline_size == current_size:
        # Nothing new — empty tail. Write an empty temp file so the
        # tarball builder has a path to add (uniform interface). 0-byte
        # files cost nothing.
        with tempfile.NamedTemporaryFile(
            suffix=".tail.bin", delete=False
        ) as tmp:
            tail_path = tmp.name
        tail_size = 0
    else:
        # STREAMING: copy [baseline_size:] of current_path into a temp file
        # in 1 MiB chunks. Never materializes the tail bytes in RAM.
        tail_path, tail_size = _stream_tail_to_temp(
            current_path, baseline_size
        )

    return {
        "diff_mode": "tail",
        "patch_path": tail_path,
        "patch_owned": True,
        "patch_size_bytes": tail_size,
        "merkle_root": current_root,
        "size_bytes": current_size,
        "prev_offset_bytes": baseline_size,
        "block_range": list(block_range) if block_range else None,
        "encoder": "timechain_tail",
    }


def apply_diff(baseline_path: Optional[str], diff_dict: dict,
               output_path: str, verify_output: bool = True) -> None:
    """Reverse of encode_diff.

    For diff_mode="full": writes diff_dict["patch_bytes"] directly to output_path.
    For diff_mode="tail": copies baseline_path[0:prev_offset_bytes] then appends
                          diff_dict["patch_bytes"].

    After write, post-condition checked: sha256(output) == diff_dict["merkle_root"].
    Mismatch raises ValueError — caller halts restore + emits BACKUP_MERKLE_MISMATCH.
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
    elif mode == "tail":
        if baseline_path is None or not os.path.exists(baseline_path):
            raise ValueError(
                "timechain_tail diff_mode='tail' requires baseline_path to exist; "
                "got missing or None"
            )
        prev_offset = diff_dict["prev_offset_bytes"]
        with open(baseline_path, "rb") as src, open(output_path, "wb") as dst:
            # Copy baseline up to prev_offset
            remaining = prev_offset
            while remaining > 0:
                chunk = src.read(min(1 << 20, remaining))
                if not chunk:
                    break
                dst.write(chunk)
                remaining -= len(chunk)
            # Append tail bytes — stream from patch_path if present, else
            # write patch_bytes directly.
            if patch_path is not None and os.path.exists(patch_path):
                with open(patch_path, "rb") as tail_src:
                    for chunk in iter(lambda: tail_src.read(1 << 20), b""):
                        dst.write(chunk)
            else:
                dst.write(patch if patch is not None else b"")
    else:
        raise ValueError(f"Unknown diff_mode {mode!r} for timechain_tail")

    # Post-write verification. The 'tail' baseline dependency above is load-bearing
    # (a wrong baseline corrupts the append) and stays strict; only these POST-APPLY
    # checks are downgraded when the source tarball is on-chain-arc-verified.
    actual_size = os.path.getsize(output_path)
    if actual_size != expected_size:
        msg = (f"timechain_tail apply size mismatch: expected {expected_size}, "
               f"got {actual_size} at {output_path}")
        if verify_output:
            raise ValueError(msg)
        logger.warning("[timechain_tail] %s — proceeding: source tarball "
                       "on-chain-arc-verified; post-apply check advisory.", msg)
    actual_root = _sha256_file(output_path)
    if actual_root != expected_root:
        msg = (f"timechain_tail apply merkle_root mismatch: expected "
               f"{expected_root}, got {actual_root} at {output_path}")
        if verify_output:
            raise ValueError(msg)
        logger.warning("[timechain_tail] %s — proceeding: source tarball "
                       "on-chain-arc-verified; post-apply check advisory.", msg)
