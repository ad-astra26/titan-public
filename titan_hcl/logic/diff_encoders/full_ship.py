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

import hashlib
import os
from typing import Optional


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def encode_diff(current_path: str, baseline_path: Optional[str] = None,
                **opts) -> dict:
    """Encode full content (baseline_path is ignored — full-ship is by design).

    STREAMING (Phase 5, 2026-05-19): returns `patch_path` pointing at the
    current file. No file content loaded into Python memory. The tarball
    builder streams from this path via `tar.add()`.

    `patch_owned=False` tells the caller NOT to unlink — patch_path IS
    the source file. (xdelta3 + timechain_tail encoders that create temp
    diff files return patch_owned=True.)
    """
    size = os.path.getsize(current_path)
    root = _sha256_file(current_path)
    return {
        "diff_mode": "full",
        "patch_path": current_path,
        "patch_owned": False,
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
