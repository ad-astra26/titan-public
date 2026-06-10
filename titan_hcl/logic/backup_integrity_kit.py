"""IntegrityKit — the ONE integrity helper shared by build and restore.

RFP_backup_redesign_spine Phase A (INV-BRS-2). Build and restore are symmetric,
so sha256 / per-component Merkle / event-Merkle / chain-link verification live
in ONE place used by BOTH paths. This WRAPS the existing, proven crypto — it
reimplements none of it (INV-BRS-8 / INV-BR-8): the §24.7 event-Merkle formula
stays `compute_event_merkle_root`, the per-file hash stays the encoders'
`file_merkle_root`.

Integrity authority (INV-MBR-12): the on-chain `arc` (sha256 over the PLAINTEXT
tarball) + `mrkl` (event Merkle root). `verify_tarball` checks the arc; the
per-file `patch_bytes_sha256` is advisory once the arc matches.
"""
from __future__ import annotations

import logging
from typing import Optional

from . import diff_encoders
from .backup_zk_commit import compute_event_merkle_root

logger = logging.getLogger("titan.backup.integrity_kit")


class IntegrityKit:
    """Stateless integrity helpers (no `self` god-state). Async-friendly: every
    method is pure CPU/IO over file paths or hex strings — the caller decides
    whether to off-load to a thread."""

    @staticmethod
    def sha256_file(path: str) -> str:
        """Streamed sha256 of a file's bytes (hex). Single source of truth =
        the encoders' `file_merkle_root` (so a tarball's IntegrityKit hash and
        its encoder-side per-file hash can never drift)."""
        return diff_encoders.file_merkle_root(path)

    @staticmethod
    def verify_tarball(path: str, expected_sha256: str) -> bool:
        """True iff the on-disk tarball's sha256 == the expected arc (over the
        PLAINTEXT tarball — Mode-B verifies AFTER decrypt, INV-MBR-13). The
        content-layer tamper check of §24.7 restore-time verification."""
        actual = IntegrityKit.sha256_file(path)
        if actual != expected_sha256:
            logger.warning("[integrity_kit] tarball sha mismatch for %s: "
                           "expected %s got %s", path, expected_sha256, actual)
            return False
        return True

    @staticmethod
    def event_merkle(personality_merkle_root: str, timechain_merkle_root: str,
                     soul_merkle_root: Optional[str] = None) -> str:
        """SPEC §24.7 step 1 — wraps `compute_event_merkle_root` verbatim
        (INV-BR-2: ONE event Merkle root per event; formula UNCHANGED)."""
        return compute_event_merkle_root(
            personality_merkle_root, timechain_merkle_root, soul_merkle_root)

    @staticmethod
    def verify_chain_link(event_prev_id: Optional[str],
                          expected_prev_id: Optional[str]) -> bool:
        """§24.17 MANIFEST-plane chain linkage (INV-BR-10): an event's recorded
        `prev_event_id` must equal the actual predecessor's id; a baseline's
        prev is None. A break = an orphaned / unrestorable chain."""
        if event_prev_id != expected_prev_id:
            logger.warning("[integrity_kit] chain-link break: prev=%r expected=%r",
                           event_prev_id, expected_prev_id)
            return False
        return True
