"""SPEC §24.5 — Per-format diff encoders for the sovereign backup pipeline.

Dispatch table:
  - TimeChain `.bin` chains (append-only) → `timechain_tail` (tail bytes since prev_offset)
  - SQLite / DuckDB / Kuzu DBs            → `xdelta3` (vcdiff patch vs baseline)
  - FAISS indices                          → full-ship at WEEKLY cadence only
                                             (skip daily unless content_hash >5%)
  - Small JSONs (< 10 KB)                  → full-ship (already tiny)
  - Large JSONs (≥ 10 KB)                  → `xdelta3`

Used by BOTH planes per SPEC §24.1:
  - LOCAL plane (L5 local_diff_manifest) — already shipped 2026-05-14;
    L5's own diff logic is independent but uses the same encoder library.
  - ARWEAVE plane (backup_unified_manifest) — Phase 3 (this code).

Encoder API (per format submodule):
  encode_diff(current_path, baseline_path, **opts) -> dict
      Compute the diff payload. Returns:
        {"diff_mode": "tail" | "incremental" | "full",
         "patch_bytes": bytes,       # the diff payload to ship
         "merkle_root": str (hex),   # sha256 over uncompressed current content
         "size_bytes": int,          # uncompressed current size
         ... format-specific metadata ...}

  apply_diff(baseline_path, diff_dict, output_path) -> None
      Reverse: given baseline_path + a diff_dict produced by encode_diff,
      reconstruct the current state at output_path. Byte-identical to the
      original current_path that was diffed.

  verify(current_path, expected_merkle_root) -> bool
      Cheap post-restore sanity check before applying further patches in
      a chain walk.

The top-level dispatch picks the right submodule from the file extension
(or explicit format_hint), records the chosen encoder in the diff_dict's
`encoder` field so restore can route correctly even if extension semantics
drift.
"""

from __future__ import annotations

import hashlib
import os
from typing import Optional

from . import full_ship, timechain_tail, xdelta3

# Module dispatch by encoder name
ENCODERS = {
    "timechain_tail": timechain_tail,
    "xdelta3": xdelta3,
    "full_ship": full_ship,
}


# SPEC §24.5 thresholds
SMALL_JSON_THRESHOLD_BYTES = 10 * 1024   # < 10 KB → full-ship


def select_encoder(path: str, format_hint: Optional[str] = None) -> str:
    """Pick encoder name for a given file path per SPEC §24.5.

    Args:
        path: Local file path (used for extension sniffing).
        format_hint: One of "timechain_bin", "db", "faiss", "json", "small_json"
            to force a specific format. Used by BackupWorker when the file
            tier is known upfront (e.g., all TIMECHAIN_PATHS use timechain_tail).

    Returns: encoder name (key in ENCODERS dict).
    """
    if format_hint == "timechain_bin":
        return "timechain_tail"
    if format_hint in ("db", "large_json", "xdelta3"):
        return "xdelta3"
    if format_hint in ("faiss", "small_json", "full_ship"):
        return "full_ship"

    # Extension-based dispatch (used when caller didn't hint)
    lower = path.lower()
    if lower.endswith(".bin") and "/timechain/chain_" in lower:
        return "timechain_tail"
    if lower.endswith((".db", ".duckdb", ".kuzu", ".sqlite")):
        return "xdelta3"
    if lower.endswith(".faiss"):
        return "full_ship"  # weekly cadence — skip daily handled by caller
    if lower.endswith(".json"):
        # Size-based for JSON
        try:
            sz = os.path.getsize(path)
            return "full_ship" if sz < SMALL_JSON_THRESHOLD_BYTES else "xdelta3"
        except OSError:
            return "full_ship"  # safe default — never wedge on missing file
    # Default: full_ship for unknown types (safest)
    return "full_ship"


def file_merkle_root(path: str) -> str:
    """SHA256 of file contents — used as `merkle_root` for SPEC §24.7
    event Merkle commit. NOT a Merkle tree; just a flat content hash
    (tarballs are flat enough that tree-hashing adds no integrity beyond
    the sha256 we already compute). The naming matches SPEC §24.3 field.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def encode_diff(current_path: str, baseline_path: Optional[str] = None,
                format_hint: Optional[str] = None, **opts) -> dict:
    """Top-level dispatch: pick encoder + delegate.

    If baseline_path is None or the encoder is `full_ship`, produces a
    full-ship event (no diff). Otherwise produces an incremental diff
    vs baseline_path.

    Returns the diff_dict that downstream caller embeds in the manifest
    event's personality/timechain/soul subdict.
    """
    encoder_name = select_encoder(current_path, format_hint)
    encoder = ENCODERS[encoder_name]
    result = encoder.encode_diff(current_path, baseline_path, **opts)
    result["encoder"] = encoder_name
    return result


def apply_diff(baseline_path: Optional[str], diff_dict: dict,
               output_path: str) -> None:
    """Top-level dispatch: pick encoder + delegate the apply step."""
    encoder_name = diff_dict.get("encoder")
    if encoder_name not in ENCODERS:
        raise ValueError(
            f"Unknown encoder {encoder_name!r} in diff_dict — supported: "
            f"{list(ENCODERS)}"
        )
    encoder = ENCODERS[encoder_name]
    encoder.apply_diff(baseline_path, diff_dict, output_path)


def verify(current_path: str, expected_merkle_root: str) -> bool:
    """Verify that current_path's content sha256 matches expected_merkle_root."""
    return file_merkle_root(current_path) == expected_merkle_root
