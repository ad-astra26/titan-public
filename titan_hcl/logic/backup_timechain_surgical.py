"""SPEC §24 + rFP §3.3 — Arweave-sourced TimeChain surgical repair.

Extends the existing Tier-1 `ChainIntegrity.surgical_repair` (sources from
local backup tarball) with an Arweave-sourced variant: when local
tarballs are missing or also corrupted, walk the unified manifest backward
to find the most recent event whose timechain block_range covers the
corruption height, fetch THAT specific event's timechain tarball from
Arweave, extract the relevant fork's `.bin` file to a scratch directory,
then delegate the splice mechanics to the existing surgical_repair (which
needs no modification — it just needs a directory of pristine fork files).

Design properties:
  - Single fetch: we identify the ONE covering event up front (rather
    than walking forward from baseline) so we transfer ~1 tarball, not
    the whole chain. Matches rFP §3.3 property: "single tampered fork
    doesn't require dropping + refetching the entire timechain".
  - No changes to ChainIntegrity: the existing surgical_repair signature
    accepts `backup_data_dir: str`; we hand it a scratch dir containing
    just the one extracted fork file.
  - Audit trail: returned RepairResult.detail is augmented with the
    sourced Arweave tx_id + event_id; caller is responsible for emitting
    a BACKUP_TIMECHAIN_SURGICAL_ARWEAVE bus event with that audit data
    (we don't synthesize bus events here to keep this module pure-logic).
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from titan_hcl.logic.backup_event_tarball import unpack_event_tarball
from titan_hcl.logic.backup_restore import (
    HALT_APPLY_FAILED,
    HALT_BROKEN_CHAIN,
    HALT_MANIFEST_EMPTY,
    HALT_TARBALL_FETCH_FAILED,
    HALT_TARBALL_HASH_MISMATCH,
    verify_component_merkle,
)
from titan_hcl.logic.backup_unified_manifest import UnifiedManifest

logger = logging.getLogger(__name__)


ArweaveFetcher = Callable[[str], "Awaitable[bytes]"]


@dataclass
class SurgicalArweaveResult:
    """Outcome of a §3.3 Arweave-sourced surgical repair.

    Carries both the RepairResult (from the delegated ChainIntegrity
    operation) and audit metadata about the Arweave source — caller
    uses the audit data to emit the bus event."""
    status: str  # "success" | "halted" | "no_covering_event"
    fork_id: int
    fork_name: str
    corruption_height: int
    sourced_from_event_id: Optional[str] = None
    sourced_from_tx_id: Optional[str] = None
    sourced_block_range: Optional[list[int]] = None
    tarball_size_bytes: int = 0
    repair_result: Optional[object] = None  # RepairResult from ChainIntegrity
    halt_reason: Optional[str] = None
    errors: list[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


def fork_arc_name(fork_name: str) -> str:
    """Map fork_name (FORK_NAMES value) to its arc_name inside the
    timechain tarball.

    TIMECHAIN_PATHS (Phase 1) uses entries like:
        ("data/timechain/chain_main.bin", "timechain/chain_main.bin")
    so the arc_name inside the tarball for fork "main" is
    "timechain/chain_main.bin".

    Per SPEC §24.4.D, the inventory is fixed; we encode it here so the
    surgical repair caller doesn't have to know the mapping convention.
    """
    return f"timechain/chain_{fork_name}.bin"


def find_covering_event(
    manifest: UnifiedManifest,
    fork_name: str,
    corruption_height: int,
) -> Optional[dict]:
    """Walk manifest events newest → oldest, return the most recent event
    whose timechain tarball contains a block_range that covers
    corruption_height for the given fork.

    Lookup strategy:
      1. Prefer the event-level fast index `event["timechain"]["block_ranges"]
         [fork_name]` if populated (avoids fetching the tarball just to
         read block_range from inside).
      2. Fall back to None — caller must fetch tarballs to inspect
         (slower; future optimization).

    The fast index field `block_ranges` is a dict {fork_name: [first_block,
    last_block]} populated at upload time. (Documented in §24.3 schema.)

    Returns the event dict (with all its fields), or None if no covering
    event was found.

    Raises ValueError on empty manifest.
    """
    if not manifest.events:
        raise ValueError(HALT_MANIFEST_EMPTY)

    for ev in manifest.walk_chain():
        tc = ev.get("timechain")
        if not tc:
            continue
        ranges = tc.get("block_ranges")
        if isinstance(ranges, dict) and fork_name in ranges:
            r = ranges[fork_name]
            if isinstance(r, (list, tuple)) and len(r) == 2:
                first, last = int(r[0]), int(r[1])
                if first <= corruption_height <= last:
                    return ev
        # Single-fork legacy: top-level block_range applies to fork "main"
        legacy = tc.get("block_range")
        if (fork_name == "main"
                and isinstance(legacy, (list, tuple)) and len(legacy) == 2):
            first, last = int(legacy[0]), int(legacy[1])
            if first <= corruption_height <= last:
                return ev

    return None


def extract_fork_file_to_scratch(
    tarball_bytes: bytes,
    fork_name: str,
    scratch_dir: str,
) -> str:
    """Unpack the per-event timechain tarball, find the fork's .bin entry,
    write it to scratch_dir/chain_{fork_name}.bin.

    Returns the absolute path to the extracted file.

    Raises ValueError if the tarball doesn't contain the expected fork file
    OR the contained diff isn't a "full" payload (incremental restores of
    a single block require chain-walk + apply — for surgical repair we
    can only operate when the covering event physically uploaded the
    fork's full bytes; tail-only diffs don't carry the bytes at
    corruption_height in a directly-extractable form).
    """
    arc = fork_arc_name(fork_name)
    os.makedirs(scratch_dir, exist_ok=True)
    output_path = os.path.join(scratch_dir, f"chain_{fork_name}.bin")

    with unpack_event_tarball(tarball_bytes) as unpacked:
        file_meta = next(
            (f for f in unpacked.files if f.get("arc_name") == arc), None
        )
        if file_meta is None:
            raise ValueError(
                f"timechain tarball (event {unpacked.event_id!r}) has no "
                f"member {arc!r} — surgical repair cannot proceed"
            )
        diff_mode = file_meta.get("diff_mode")
        if diff_mode not in ("full", "tail"):
            raise ValueError(
                f"timechain tarball member {arc!r} has diff_mode "
                f"{diff_mode!r}; surgical repair from Arweave requires "
                f"full or tail payload (the covering event's tarball "
                f"must physically contain the bytes at corruption_height)"
            )

        patch_bytes = unpacked.get_patch_bytes(arc)

        if diff_mode == "full":
            # patch_bytes IS the full file
            with open(output_path, "wb") as f:
                f.write(patch_bytes)
        else:
            # diff_mode == "tail": the payload is the bytes appended SINCE
            # the prior baseline. For surgical repair, we need the file
            # AT or BEFORE corruption_height to contain the original
            # pre-tamper bytes. If corruption_height is in the tail
            # portion (block_range first_block..last_block of THIS event),
            # we need to know the prior baseline's bytes too — that's a
            # multi-event walk, which we don't support in the single-fetch
            # surgical path. The find_covering_event logic should already
            # have selected an event where the FULL fork covers the
            # height; if the caller selected a tail-only event by
            # mistake, raise so they pick a different event.
            #
            # In the typical case (baseline events have diff_mode="full"
            # for timechain), surgical_repair_from_arweave finds the
            # latest baseline that covers the corruption height.
            raise ValueError(
                f"surgical repair selected tail-mode event {unpacked.event_id!r} "
                f"for fork {fork_name!r}; need a baseline event with full "
                f"fork bytes. Walk further back to a baseline-covering event."
            )

    return output_path


async def surgical_repair_from_arweave(
    chain_integrity,  # ChainIntegrity instance (duck-typed for testability)
    fork_id: int,
    corruption,  # CorruptionReport
    manifest: UnifiedManifest,
    arweave_fetch: ArweaveFetcher,
    *,
    scratch_dir: Optional[str] = None,
) -> SurgicalArweaveResult:
    """rFP §3.3 — Arweave-sourced Tier-1 surgical repair.

    Orchestration:
      1. Resolve fork_name from fork_id (via timechain.FORK_NAMES).
      2. Walk manifest backward for an event whose timechain.block_ranges
         (or legacy block_range) covers corruption.corruption_height.
      3. Fetch that event's timechain tarball from Arweave.
      4. Verify tarball sha256 matches manifest event.timechain.merkle_root.
      5. Extract chain_{fork_name}.bin to scratch_dir.
      6. Delegate to chain_integrity.surgical_repair(fork_id, corruption,
         scratch_dir).
      7. Return SurgicalArweaveResult with delegated RepairResult + Arweave
         audit metadata.

    Caller is responsible for emitting a BACKUP_TIMECHAIN_SURGICAL_ARWEAVE
    bus event with the audit metadata on success.

    `chain_integrity` is duck-typed (no isinstance check) so tests can
    inject a stub that records the surgical_repair call without running
    the full TimeChain machinery.
    """
    from titan_hcl.logic.timechain import FORK_NAMES
    fork_name = FORK_NAMES.get(fork_id, f"sc_{fork_id}")
    corrupt_h = corruption.corruption_height

    out = SurgicalArweaveResult(
        status="halted", fork_id=fork_id, fork_name=fork_name,
        corruption_height=corrupt_h,
    )

    # Step 2 — find covering event
    try:
        event = find_covering_event(manifest, fork_name, corrupt_h)
    except ValueError as e:
        out.halt_reason = HALT_MANIFEST_EMPTY
        out.errors.append(str(e))
        return out
    if event is None:
        out.status = "no_covering_event"
        out.errors.append(
            f"no manifest event has timechain.block_ranges covering "
            f"fork={fork_name!r} height={corrupt_h}"
        )
        return out

    out.sourced_from_event_id = event["event_id"]
    out.sourced_from_tx_id = event["timechain"]["tx_id"]

    # Step 3 — fetch tarball
    try:
        tarball_bytes = await arweave_fetch(out.sourced_from_tx_id)
    except Exception as e:
        out.halt_reason = HALT_TARBALL_FETCH_FAILED
        out.errors.append(
            f"Arweave fetch failed for {out.sourced_from_tx_id!r}: {e}"
        )
        return out
    out.tarball_size_bytes = len(tarball_bytes)

    # Step 4 — verify tarball hash against manifest
    try:
        verify_component_merkle(event, "timechain", tarball_bytes)
    except ValueError as e:
        out.halt_reason = HALT_TARBALL_HASH_MISMATCH
        out.errors.append(str(e))
        return out

    # Step 5 — extract the fork's .bin file to scratch
    cleanup_scratch = False
    if scratch_dir is None:
        scratch_dir = tempfile.mkdtemp(prefix=f"titan_surgical_{fork_name}_")
        cleanup_scratch = True

    try:
        try:
            extract_fork_file_to_scratch(
                tarball_bytes=tarball_bytes,
                fork_name=fork_name,
                scratch_dir=scratch_dir,
            )
        except ValueError as e:
            out.halt_reason = HALT_APPLY_FAILED
            out.errors.append(str(e))
            return out
        # Record block_range that was used (for audit trail)
        tc = event["timechain"]
        ranges = tc.get("block_ranges") or {}
        if fork_name in ranges:
            out.sourced_block_range = list(ranges[fork_name])
        elif tc.get("block_range") and fork_name == "main":
            out.sourced_block_range = list(tc["block_range"])

        # Step 6 — delegate to existing surgical_repair
        try:
            repair_result = chain_integrity.surgical_repair(
                fork_id=fork_id,
                corruption=corruption,
                backup_data_dir=scratch_dir,
            )
        except Exception as e:
            out.halt_reason = HALT_APPLY_FAILED
            out.errors.append(f"chain_integrity.surgical_repair raised: {e}")
            return out
        out.repair_result = repair_result

        if getattr(repair_result, "success", False):
            out.status = "success"
            # Augment the RepairResult.detail with Arweave source for audit
            extra = (
                f" [sourced_from_arweave_tx={out.sourced_from_tx_id} "
                f"event_id={out.sourced_from_event_id}]"
            )
            try:
                repair_result.detail = (
                    (repair_result.detail or "") + extra
                )
            except (AttributeError, TypeError):
                # Some RepairResult variants may be frozen — non-fatal
                pass
        else:
            out.status = "halted"
            out.halt_reason = HALT_APPLY_FAILED
            detail = getattr(repair_result, "detail", "(no detail)")
            out.errors.append(
                f"chain_integrity.surgical_repair returned success=False: "
                f"{detail}"
            )

        return out
    finally:
        if cleanup_scratch:
            try:
                import shutil
                shutil.rmtree(scratch_dir, ignore_errors=True)
            except Exception:
                pass
