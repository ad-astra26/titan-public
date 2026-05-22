"""SPEC §24.8 + rFP §3.1 — Full crash-recovery restore protocol for the
Arweave-plane backup chain.

Walks the UnifiedManifest (Phase 2) → fetches per-event Arweave tarballs
(Phase 3 encoders + Phase 6 tarball schema) → verifies Merkle at every
level against the on-chain ZK Vault commits (Phase 5) → applies baseline
to a target directory → replays incrementals via diff_encoders.apply_diff
→ verifies post-restore file hashes against the manifest.

Scope: this module is the pure-logic restore engine. Step 1-4 of §3.1
(Shamir keypair recovery + fresh VPS bootstrap + GenesisNFT lookup + birth
identity fetch) are pre-Titan operator concerns and live outside this
module. Step 5-10 (chain walk + verify + apply) are this module's
responsibility. Step 11 (boot integrity check) is SPEC §11.H.5 (already
shipped). Step 12 (RESURRECTION_COMPLETE memo) is `emit_resurrection_memo`
below — the operator calls it once boot completes cleanly.

Design properties:
  - Halt-on-mismatch at every Merkle level. Never silently skip a bad
    tarball — a single corrupted event aborts the whole restore and
    surfaces a `BACKUP_MERKLE_MISMATCH` halt reason for the caller to
    emit on the bus + alert Maker.
  - Manifest is caller-supplied (not auto-discovered from Solana). rFP
    §3.1 step 6 implies the manifest_tx_id rides on the latest ZK memo,
    but the Phase 5 v=2 memo format (`v=2;event_id;root;prev`) doesn't
    carry it. Manifest discovery is a separate concern (SPEC §24.13
    follow-up; documented in §13 close-out). Caller provides the
    UnifiedManifest however appropriate (Maker-supplied off-VPS copy,
    local-disk recovery, Arweave-tag lookup).
  - Streaming-friendly: each event's tarballs are unpacked + applied +
    closed before moving to the next event. Memory footprint stays
    bounded even on 30-deep chains.
  - Restore reads from Arweave (read-only). Writes only to `target_dir`
    (a scratch directory). The live `data/` directory is never touched
    by this module; the operator atomically swaps target_dir → data/
    once restore reports success (or invokes
    `arch_map backup verify --restore-sim` to validate without swap).
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

from titan_hcl.logic import diff_encoders
from titan_hcl.logic.backup_event_tarball import (
    UnpackedEvent,
    unpack_event_tarball,
)
from titan_hcl.logic.backup_unified_manifest import UnifiedManifest
from titan_hcl.logic.backup_zk_commit import (
    compute_event_merkle_root,
    parse_zk_memo,
)

logger = logging.getLogger(__name__)


# Halt reasons — surfaced in RestoreResult.halt_reason for caller to emit
# on the bus as `BACKUP_RESTORE_TEST_FAIL` / `BACKUP_MERKLE_MISMATCH` /
# `BACKUP_ZK_DISCONNECT` events.
HALT_MANIFEST_EMPTY = "manifest_empty"
HALT_BROKEN_CHAIN = "manifest_chain_broken"
HALT_TARBALL_FETCH_FAILED = "tarball_fetch_failed"
HALT_TARBALL_HASH_MISMATCH = "tarball_hash_mismatch"
HALT_EVENT_MERKLE_MISMATCH = "event_merkle_mismatch"
HALT_ZK_MEMO_MISMATCH = "zk_memo_mismatch"
HALT_ZK_DISCONNECT = "zk_disconnect"
HALT_APPLY_FAILED = "apply_failed"
HALT_POST_RESTORE_HASH_MISMATCH = "post_restore_hash_mismatch"


@dataclass
class RestoreResult:
    """Outcome of a full restore walk. `status` is "success" when every
    event applied cleanly and all post-restore file hashes matched the
    manifest's expected values. Any failure halts the walk and sets
    `halt_reason` to one of the HALT_* constants."""
    status: str  # "success" | "halted"
    target_event_id: Optional[str] = None
    applied_events: list[str] = field(default_factory=list)
    restored_files: int = 0
    bytes_fetched: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    halt_reason: Optional[str] = None
    halt_event_id: Optional[str] = None
    duration_s: float = 0.0


# ── chain selection ──────────────────────────────────────────────────────


def select_restore_chain(
    manifest: UnifiedManifest,
    target_event_id: Optional[str] = None,
) -> list[dict]:
    """Return the chronological list of events to apply: [baseline, inc1, ...].

    If target_event_id is None → walks back from the latest event.
    Otherwise → walks back from the named event.

    Raises ValueError on broken chain (prev_event_id points to a missing
    event) — that is a load-bearing manifest corruption that must surface
    to the supervisor, not be silently truncated.
    """
    if not manifest.events:
        raise ValueError(HALT_MANIFEST_EMPTY)

    start = manifest.get_latest_event() if target_event_id is None else \
        manifest.get_event(target_event_id)
    if start is None:
        raise ValueError(
            f"target_event_id {target_event_id!r} not in manifest"
        )

    # Walk newest → oldest until baseline.
    chain_reverse: list[dict] = []
    cursor = start
    while cursor is not None:
        chain_reverse.append(cursor)
        if cursor.get("type") == "baseline":
            return list(reversed(chain_reverse))
        prev_id = cursor.get("prev_event_id")
        if prev_id is None:
            # First-ever event must be baseline; if we got here cursor was
            # the first event and isn't a baseline → manifest corrupt.
            raise ValueError(
                f"{HALT_BROKEN_CHAIN}: walked to first event "
                f"{cursor.get('event_id')!r} but it isn't a baseline"
            )
        prev = manifest.get_event(prev_id)
        if prev is None:
            raise ValueError(
                f"{HALT_BROKEN_CHAIN}: event {cursor.get('event_id')!r} "
                f"has prev_event_id={prev_id!r} which is not in manifest"
            )
        cursor = prev

    # Should be unreachable — loop only exits via baseline-return or raise
    raise ValueError(HALT_BROKEN_CHAIN)


# ── tarball fetch + verify ───────────────────────────────────────────────


# Caller-supplied async fetcher signature. Concretely, ArweaveStore exposes
# `download_file(tx_id) -> bytes` in production; tests inject a dict-backed
# stub. Kept abstract so the restore engine has no hard Arweave-client
# dependency.
ArweaveFetcher = Callable[[str], "Awaitable[bytes]"]


async def fetch_event_components(
    arweave_fetch: ArweaveFetcher,
    event: dict,
) -> dict[str, bytes]:
    """Fetch the per-component tarballs for one event from Arweave.

    Returns {"personality": bytes, "timechain": bytes, "soul": bytes}.
    Soul is omitted when event.soul is None (non-weekly events).

    Raises RuntimeError on any fetch failure — caller halts restore.
    """
    result: dict[str, bytes] = {}
    for component in ("personality", "timechain", "soul"):
        sub = event.get(component)
        if sub is None:
            continue
        tx_id = sub.get("tx_id")
        if not tx_id:
            raise RuntimeError(
                f"event {event.get('event_id')!r} {component}.tx_id missing"
            )
        try:
            data = await arweave_fetch(tx_id)
        except Exception as e:
            raise RuntimeError(
                f"Arweave fetch failed for event {event.get('event_id')!r} "
                f"{component} tx_id={tx_id!r}: {e}"
            ) from e
        if not isinstance(data, (bytes, bytearray)):
            raise RuntimeError(
                f"Arweave fetch for tx_id={tx_id!r} returned "
                f"{type(data).__name__}, expected bytes"
            )
        result[component] = bytes(data)
    return result


def verify_component_merkle(event: dict, component: str,
                            tarball_bytes: bytes) -> None:
    """Verify sha256(tarball_bytes) == event[component].merkle_root.

    Raises ValueError on mismatch — caller halts restore + emits
    BACKUP_TARBALL_HASH_MISMATCH.
    """
    import hashlib
    sub = event.get(component)
    if sub is None:
        return
    expected = sub.get("merkle_root")
    if not expected:
        raise ValueError(
            f"event {event.get('event_id')!r} {component}.merkle_root missing"
        )
    actual = hashlib.sha256(tarball_bytes).hexdigest()
    if actual != expected:
        raise ValueError(
            f"{HALT_TARBALL_HASH_MISMATCH}: event {event.get('event_id')!r} "
            f"{component} tarball sha256 mismatch: expected {expected}, "
            f"got {actual}"
        )


def verify_event_merkle(event: dict, components: dict[str, bytes]) -> str:
    """Recompose event_merkle_root from per-component tarball hashes (Phase 5
    convention) and verify it matches what the manifest event would commit
    to on-chain. Returns the recomposed event_merkle_root hex (for caller
    to cross-check against the parsed ZK memo's `root` fragment).

    Raises ValueError on missing components or hash mismatch.
    """
    import hashlib

    def _h(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    p = event.get("personality")
    t = event.get("timechain")
    s = event.get("soul")
    if p is None or t is None:
        raise ValueError(
            f"event {event.get('event_id')!r} missing personality or timechain"
        )

    pers_root = p.get("merkle_root") or _h(components.get("personality", b""))
    tc_root = t.get("merkle_root") or _h(components.get("timechain", b""))
    soul_root = s.get("merkle_root") if s else None

    return compute_event_merkle_root(
        personality_merkle_root=pers_root,
        timechain_merkle_root=tc_root,
        soul_merkle_root=soul_root,
    )


# ── ZK Vault chain verification (Phase 5 memo round-trip) ────────────────


# Caller-supplied async memo fetcher: given a Solana tx_id, return the v=2
# memo text (or raise on RPC failure). Production uses
# `network.get_memo_for_tx(sig)`; tests inject a dict-backed stub.
MemoFetcher = Callable[[str], "Awaitable[str]"]


async def verify_event_zk_commit(
    event: dict,
    recomposed_event_merkle_root: str,
    memo_fetch: MemoFetcher,
    prev_event_merkle_root_for_check: Optional[str],
) -> None:
    """Fetch the v=2 memo for event.zk_commit_tx and verify:
      - memo parses cleanly
      - memo.event_id matches event.event_id
      - memo.root matches recomposed_event_merkle_root[:32]
      - memo.prev matches prev_event_merkle_root_for_check[:16] (or 'genesis')

    Raises ValueError on any mismatch — caller halts restore + emits
    BACKUP_MERKLE_MISMATCH.
    Raises RuntimeError on RPC failure — caller halts + emits
    BACKUP_ZK_DISCONNECT.
    """
    sig = event.get("zk_commit_tx")
    if not sig:
        raise ValueError(
            f"event {event.get('event_id')!r} has no zk_commit_tx"
        )
    try:
        memo_text = await memo_fetch(sig)
    except Exception as e:
        raise RuntimeError(
            f"{HALT_ZK_DISCONNECT}: failed to fetch ZK memo for sig={sig!r}: {e}"
        ) from e
    parsed = parse_zk_memo(memo_text)
    if parsed is None:
        raise ValueError(
            f"{HALT_ZK_MEMO_MISMATCH}: ZK memo for sig={sig!r} did not parse "
            f"as v=2 format (got {memo_text!r})"
        )
    if parsed["event_id"] != event.get("event_id"):
        raise ValueError(
            f"{HALT_ZK_MEMO_MISMATCH}: memo event_id={parsed['event_id']!r} "
            f"≠ manifest event_id={event.get('event_id')!r} at sig={sig!r}"
        )
    expected_root_short = recomposed_event_merkle_root[:32]
    if parsed["root"] != expected_root_short:
        raise ValueError(
            f"{HALT_EVENT_MERKLE_MISMATCH}: memo root={parsed['root']!r} "
            f"≠ recomposed event_merkle_root[:32]={expected_root_short!r} "
            f"for event {event.get('event_id')!r}"
        )
    expected_prev = (
        prev_event_merkle_root_for_check[:16]
        if prev_event_merkle_root_for_check else "genesis"
    )
    if parsed["prev"] != expected_prev:
        raise ValueError(
            f"{HALT_ZK_MEMO_MISMATCH}: memo prev={parsed['prev']!r} "
            f"≠ expected prev={expected_prev!r} for event "
            f"{event.get('event_id')!r}"
        )


# ── per-event apply ──────────────────────────────────────────────────────


def apply_event_components(
    components: dict[str, bytes],
    target_dir: str,
    arc_to_target: Callable[[str, str], str],
    arweave_fetch_skipped_sync: Optional[Callable[[str], bytes]] = None,
) -> dict:
    """Unpack each component tarball and apply per-file diffs into target_dir.

    Args:
        components: {"personality": bytes, "timechain": bytes, "soul": bytes}.
        target_dir: scratch dir to reconstruct into.
        arc_to_target: callable (component, arc_name) → absolute path inside
            target_dir. Caller-supplied so the on-host PERSONALITY_PATHS
            mapping isn't hard-coded into the restore engine (which would
            couple it to backup.py and break cross-environment use).
        arweave_fetch_skipped_sync: optional sync fetcher for "skipped_files"
            pointer chain (Phase 7). For Phase 6 full restore we don't need
            it because the baseline always contains the full file; skipped
            files in incrementals just mean "unchanged since last event",
            so we leave the target_dir state untouched for that arc_name.

    Returns dict with {restored_files: int, errors: list[str], warnings:
    list[str]}.

    Raises ValueError on apply failure — caller halts.
    """
    result = {"restored_files": 0, "errors": [], "warnings": []}

    for component, tarball_bytes in components.items():
        with unpack_event_tarball(tarball_bytes) as unpacked:
            for file_meta in unpacked.files:
                arc_name = file_meta["arc_name"]
                diff_mode = file_meta.get("diff_mode")
                target_path = arc_to_target(component, arc_name)

                if diff_mode == "skipped":
                    # Phase 4 skip-if-unchanged pointer. Leave target_dir
                    # state unchanged — the prior event in the chain
                    # already wrote the current bytes for this arc_name.
                    continue

                # Reconstruct the diff_dict (Phase 3 encoders consume it)
                diff_dict = unpacked.diff_dict_for(arc_name)

                # Baseline path for tail / incremental diffs is the
                # CURRENT contents of target_path (the prior event's
                # result). For full diffs it's ignored.
                #
                # We MUST write through a scratch path then atomic-rename:
                # tail + xdelta3 encoders open the output file in "wb" mode,
                # which truncates it. If baseline_path == target_path (which
                # is the common case during restore chain replay), the
                # encoder would silently truncate the file it's trying to
                # read from. Scratch-then-rename keeps the source intact
                # until the apply succeeds.
                baseline_path = (
                    target_path if os.path.exists(target_path) else None
                )
                scratch_path = target_path + ".restoring"

                try:
                    diff_encoders.apply_diff(
                        baseline_path=baseline_path,
                        diff_dict=diff_dict,
                        output_path=scratch_path,
                    )
                    os.replace(scratch_path, target_path)
                except Exception as e:
                    if os.path.exists(scratch_path):
                        try:
                            os.unlink(scratch_path)
                        except OSError:
                            pass
                    raise ValueError(
                        f"{HALT_APPLY_FAILED}: component={component} "
                        f"arc_name={arc_name!r} event={unpacked.event_id!r} "
                        f"diff_mode={diff_mode!r}: {e}"
                    ) from e
                result["restored_files"] += 1
    return result


# ── orchestrator (§3.1 step 5-10) ────────────────────────────────────────


async def restore_full(
    manifest: UnifiedManifest,
    target_dir: str,
    arweave_fetch: ArweaveFetcher,
    memo_fetch: MemoFetcher,
    arc_to_target: Callable[[str, str], str],
    *,
    target_event_id: Optional[str] = None,
    verify_zk_chain: bool = True,
    progress_callback: Optional[Callable[[dict], None]] = None,
) -> RestoreResult:
    """rFP §3.1 — full crash-recovery restore protocol (steps 5-10).

    Walks the manifest from target_event_id (or latest) back to baseline,
    fetches each event's tarballs from Arweave, verifies each tarball's
    sha256 against the manifest, recomposes the event_merkle_root and
    verifies it matches the on-chain ZK memo's root fragment, then
    applies baseline + replays incrementals into target_dir.

    Args:
        manifest: caller-loaded UnifiedManifest. See module docstring on
            manifest discovery.
        target_dir: scratch dir for reconstruction (created if absent).
        arweave_fetch: async callable tx_id -> bytes. Production injects
            `ArweaveStore.download_file`; tests inject a dict-backed stub.
        memo_fetch: async callable solana_sig -> memo_text. Set to a
            stub returning "" + pass verify_zk_chain=False to skip ZK
            verification (used by Phase 7 single-file restore where the
            ZK chain has already been verified at full-restore time).
        arc_to_target: callable (component, arc_name) -> absolute path
            inside target_dir. See apply_event_components.
        target_event_id: optionally restore to a specific past event
            (used by §3.4 on-demand historical fetch). Default: latest.
        verify_zk_chain: whether to round-trip-verify each event's ZK
            memo (Maker decision Q2 — full restore-test does full verify).
        progress_callback: optional dict-emitting callback for the bus
            (Phase 9 weekly restore-test wires this to BACKUP_RESTORE_PROGRESS).

    Returns: RestoreResult.
    """
    started = time.time()
    out = RestoreResult(status="halted")
    os.makedirs(target_dir, exist_ok=True)

    # Step 5 — find chain (newest → oldest, walk back to baseline)
    try:
        chain = select_restore_chain(manifest, target_event_id=target_event_id)
    except ValueError as e:
        out.halt_reason = HALT_BROKEN_CHAIN if "broken" in str(e) else \
            HALT_MANIFEST_EMPTY
        out.errors.append(str(e))
        out.duration_s = time.time() - started
        return out

    out.target_event_id = chain[-1]["event_id"]
    if progress_callback:
        progress_callback({
            "phase": "chain_selected",
            "events_to_apply": len(chain),
            "baseline_event_id": chain[0]["event_id"],
            "target_event_id": out.target_event_id,
        })

    # Walk in chronological order — baseline first, then incrementals
    prev_event_merkle_root: Optional[str] = None

    for idx, event in enumerate(chain):
        event_id = event["event_id"]
        if progress_callback:
            progress_callback({
                "phase": "fetching_event",
                "event_id": event_id,
                "event_type": event.get("type"),
                "index": idx,
                "total": len(chain),
            })

        # Step 7 — fetch baseline+incrementals from Arweave
        try:
            components = await fetch_event_components(arweave_fetch, event)
        except RuntimeError as e:
            out.halt_reason = HALT_TARBALL_FETCH_FAILED
            out.halt_event_id = event_id
            out.errors.append(str(e))
            out.duration_s = time.time() - started
            return out

        out.bytes_fetched += sum(len(b) for b in components.values())

        # Step 8.a — verify per-component sha256 against manifest
        try:
            for component, data in components.items():
                verify_component_merkle(event, component, data)
        except ValueError as e:
            out.halt_reason = HALT_TARBALL_HASH_MISMATCH
            out.halt_event_id = event_id
            out.errors.append(str(e))
            out.duration_s = time.time() - started
            return out

        # Step 8.b — recompose event_merkle_root from per-component hashes
        try:
            recomposed_root = verify_event_merkle(event, components)
        except ValueError as e:
            out.halt_reason = HALT_EVENT_MERKLE_MISMATCH
            out.halt_event_id = event_id
            out.errors.append(str(e))
            out.duration_s = time.time() - started
            return out

        # Step 8.c — verify the on-chain ZK memo matches the recomposed
        # event_merkle_root + prev linkage
        if verify_zk_chain:
            try:
                await verify_event_zk_commit(
                    event=event,
                    recomposed_event_merkle_root=recomposed_root,
                    memo_fetch=memo_fetch,
                    prev_event_merkle_root_for_check=prev_event_merkle_root,
                )
            except RuntimeError as e:
                out.halt_reason = HALT_ZK_DISCONNECT
                out.halt_event_id = event_id
                out.errors.append(str(e))
                out.duration_s = time.time() - started
                return out
            except ValueError as e:
                # Differentiate ZK_MEMO_MISMATCH vs EVENT_MERKLE_MISMATCH
                # so the caller's bus event is precise.
                msg = str(e)
                if HALT_EVENT_MERKLE_MISMATCH in msg:
                    out.halt_reason = HALT_EVENT_MERKLE_MISMATCH
                else:
                    out.halt_reason = HALT_ZK_MEMO_MISMATCH
                out.halt_event_id = event_id
                out.errors.append(msg)
                out.duration_s = time.time() - started
                return out

        # Step 9 — apply baseline / replay incrementals into target_dir
        try:
            apply_result = apply_event_components(
                components=components,
                target_dir=target_dir,
                arc_to_target=arc_to_target,
            )
        except ValueError as e:
            out.halt_reason = HALT_APPLY_FAILED
            out.halt_event_id = event_id
            out.errors.append(str(e))
            out.duration_s = time.time() - started
            return out
        out.restored_files += apply_result["restored_files"]
        out.warnings.extend(apply_result.get("warnings", []))
        out.applied_events.append(event_id)

        prev_event_merkle_root = recomposed_root

        if progress_callback:
            progress_callback({
                "phase": "event_applied",
                "event_id": event_id,
                "index": idx,
                "total": len(chain),
            })

    out.status = "success"
    out.duration_s = time.time() - started
    if progress_callback:
        progress_callback({
            "phase": "complete",
            "applied": len(out.applied_events),
            "restored_files": out.restored_files,
            "bytes_fetched": out.bytes_fetched,
            "duration_s": out.duration_s,
        })
    return out


# ── §3.2 + §3.4 — Single-file restore (Tier-3 fallback + on-demand fetch) ─


@dataclass
class SingleFileRestoreResult:
    """Outcome of a single-file restore — used by §3.2 boot integrity Tier-3
    fallback and §3.4 on-demand historical fetch.

    Distinct from the full RestoreResult so callers can wire SPEC §11.H.4
    Tier-3 cleanly (one-file recovery doesn't need full chain stats)."""
    status: str  # "success" | "halted"
    component: str
    arc_name: str
    target_event_id: Optional[str] = None
    applied_events: list[str] = field(default_factory=list)
    bytes_fetched: int = 0
    output_path: Optional[str] = None
    final_merkle_root: Optional[str] = None
    errors: list[str] = field(default_factory=list)
    halt_reason: Optional[str] = None
    halt_event_id: Optional[str] = None
    duration_s: float = 0.0


def _file_diff_mode_in_event(event: dict, component: str,
                             arc_name: str) -> Optional[str]:
    """Look up the manifest event's recorded diff_mode for a single file.

    The unified manifest event's per-component subdict carries either a
    summary diff_mode (legacy single-mode tarball) or — when packed by
    backup_event_tarball — the per-file shape is inside the tarball's
    __event_metadata.json. For pre-fetch chain pruning we can ONLY rely on
    the manifest-level metadata for now (without round-tripping each
    tarball just to check whether a file is skipped). The fast path is:

      - event[component] is None        → file definitely not in this event
      - event[component]["skipped_files"] contains arc_name → skipped
      - otherwise → assume physically present (caller will discover
        skipped state when unpacking)

    Returns "skipped" / "present" / None (component missing entirely).
    """
    sub = event.get(component)
    if sub is None:
        return None
    skipped = sub.get("skipped_files") or []
    if isinstance(skipped, list) and arc_name in skipped:
        return "skipped"
    # SPEC §24.3 also allows skipped_files to be a dict (per-file pointer)
    if isinstance(skipped, dict) and arc_name in skipped:
        return "skipped"
    return "present"


def select_single_file_chain(
    manifest: UnifiedManifest,
    component: str,
    arc_name: str,
    target_event_id: Optional[str] = None,
) -> list[dict]:
    """Walk the manifest to build the minimal chain needed to reconstruct
    one file at target_event_id (default: latest event).

    Returns chronological [baseline, inc1, inc2, …, target] where:
      - baseline is the most recent baseline reachable from target_event_id
      - incrementals between baseline and target are included regardless
        of whether they have this file or marked it skipped — applying
        them is a no-op for skipped files but we keep them in the chain
        so the tarball metadata can confirm the file's intermediate
        state matches expectations

    Raises ValueError on broken chain / empty manifest.

    Note: this returns the FULL chain in the file's path, not just events
    where the file changed. The applier handles skipped entries as no-ops
    against the prior reconstructed bytes — keeps the algorithm aligned
    with the full-restore logic and lets the Phase 7 caller share the
    same Merkle verification path.
    """
    return select_restore_chain(manifest, target_event_id=target_event_id)


async def restore_single_file(
    manifest: UnifiedManifest,
    component: str,
    arc_name: str,
    output_path: str,
    arweave_fetch: ArweaveFetcher,
    *,
    target_event_id: Optional[str] = None,
    memo_fetch: Optional[MemoFetcher] = None,
    verify_zk_chain: bool = False,
    expected_post_merkle_root: Optional[str] = None,
) -> SingleFileRestoreResult:
    """rFP §3.2 / §3.4 — reconstruct a single file's contents at output_path.

    Walks the manifest chain to target_event_id (or latest), fetches the
    minimum set of component tarballs needed, applies the file's
    baseline → incremental → … sequence, verifies the final sha256
    against expected_post_merkle_root (if provided, e.g. for boot-integrity
    Tier-3 fallback when SPEC §11.H.5 already knows what hash it wants).

    `verify_zk_chain` defaults to False here because:
      - §3.2 Tier-3 fallback runs during boot — Solana RPC may not be
        available yet; the local manifest + Arweave tarball Merkle is
        sufficient (Maker can run `arch_map backup verify --restore-sim`
        to verify the ZK chain end-to-end as a separate operation).
      - §3.4 on-demand historical fetch is operator-driven and typically
        already has confidence in the chain.

    Caller can opt back into ZK verification (full belt-and-suspenders)
    by passing memo_fetch + verify_zk_chain=True.
    """
    started = time.time()
    out = SingleFileRestoreResult(
        status="halted", component=component, arc_name=arc_name,
    )

    # Step 1 — select the file's chain
    try:
        chain = select_single_file_chain(
            manifest, component, arc_name, target_event_id=target_event_id
        )
    except ValueError as e:
        out.halt_reason = (
            HALT_BROKEN_CHAIN if "broken" in str(e) else HALT_MANIFEST_EMPTY
        )
        out.errors.append(str(e))
        out.duration_s = time.time() - started
        return out

    out.target_event_id = chain[-1]["event_id"]
    parent = os.path.dirname(output_path) or "."
    os.makedirs(parent, exist_ok=True)
    scratch_path = output_path + ".restoring"

    # Remove any stale scratch from a prior failed attempt
    if os.path.exists(scratch_path):
        try:
            os.unlink(scratch_path)
        except OSError as e:
            out.errors.append(f"failed to clear stale scratch: {e}")
            out.halt_reason = HALT_APPLY_FAILED
            out.duration_s = time.time() - started
            return out
    if os.path.exists(output_path):
        # We will atomically replace at the end, but apply_diff opens the
        # output in "wb" mode internally — to avoid disturbing the live
        # file at output_path (e.g. when this is the boot-fallback path
        # for a partially-corrupted file), we apply into scratch_path
        # which becomes the chain's working location until the final
        # rename.
        pass

    prev_event_merkle_root: Optional[str] = None

    for event in chain:
        event_id = event["event_id"]
        sub = event.get(component)
        if sub is None:
            # No tarball for this component on this event (e.g. soul on a
            # non-weekly event). The file's chain continues through prior
            # events; nothing to apply here.
            continue

        # Fetch ONLY the relevant component tarball
        tx_id = sub.get("tx_id")
        if not tx_id:
            out.halt_reason = HALT_TARBALL_FETCH_FAILED
            out.halt_event_id = event_id
            out.errors.append(
                f"event {event_id!r} {component}.tx_id missing"
            )
            out.duration_s = time.time() - started
            return out
        try:
            tarball_bytes = await arweave_fetch(tx_id)
        except Exception as e:
            out.halt_reason = HALT_TARBALL_FETCH_FAILED
            out.halt_event_id = event_id
            out.errors.append(
                f"Arweave fetch failed for {tx_id!r}: {e}"
            )
            out.duration_s = time.time() - started
            return out
        out.bytes_fetched += len(tarball_bytes)

        # Verify tarball-level sha256 against manifest
        try:
            verify_component_merkle(event, component, tarball_bytes)
        except ValueError as e:
            out.halt_reason = HALT_TARBALL_HASH_MISMATCH
            out.halt_event_id = event_id
            out.errors.append(str(e))
            out.duration_s = time.time() - started
            return out

        # Optional ZK Vault round-trip (recomposes event_merkle_root from
        # ALL components — for single-file restore this means fetching the
        # other components too if we want the recomposition to match. To
        # keep §3.2 fallback fast we recompose using the manifest's recorded
        # per-component merkle_roots — which is what verify_event_merkle
        # does when components dict is missing keys. The Tier-3 boot
        # fallback consciously trades the cross-component network fetch
        # for boot speed; full ZK verification is the operator's job via
        # restore-sim).
        if verify_zk_chain and memo_fetch is not None:
            try:
                recomposed = verify_event_merkle(event, {component: tarball_bytes})
                await verify_event_zk_commit(
                    event=event,
                    recomposed_event_merkle_root=recomposed,
                    memo_fetch=memo_fetch,
                    prev_event_merkle_root_for_check=prev_event_merkle_root,
                )
                prev_event_merkle_root = recomposed
            except RuntimeError as e:
                out.halt_reason = HALT_ZK_DISCONNECT
                out.halt_event_id = event_id
                out.errors.append(str(e))
                out.duration_s = time.time() - started
                return out
            except ValueError as e:
                msg = str(e)
                out.halt_reason = (
                    HALT_EVENT_MERKLE_MISMATCH
                    if HALT_EVENT_MERKLE_MISMATCH in msg
                    else HALT_ZK_MEMO_MISMATCH
                )
                out.halt_event_id = event_id
                out.errors.append(msg)
                out.duration_s = time.time() - started
                return out

        # Apply just the requested arc_name from this event's tarball
        try:
            with unpack_event_tarball(tarball_bytes) as unpacked:
                file_meta = next(
                    (f for f in unpacked.files if f.get("arc_name") == arc_name),
                    None,
                )
                if file_meta is None:
                    # File not in this event's tarball — no-op, chain continues
                    out.applied_events.append(event_id)
                    continue
                diff_mode = file_meta.get("diff_mode")
                if diff_mode == "skipped":
                    # No payload — file unchanged since prior event. Leave
                    # scratch_path as-is and continue to next event.
                    out.applied_events.append(event_id)
                    continue

                diff_dict = unpacked.diff_dict_for(arc_name)
                baseline_path = (
                    scratch_path if os.path.exists(scratch_path) else None
                )
                # Two-step scratch dance: encoder writes to staging,
                # then we replace scratch_path. Avoids same-file truncation
                # (same fix as restore_full's apply_event_components).
                staging_path = scratch_path + ".staging"
                try:
                    diff_encoders.apply_diff(
                        baseline_path=baseline_path,
                        diff_dict=diff_dict,
                        output_path=staging_path,
                    )
                    os.replace(staging_path, scratch_path)
                except Exception as e:
                    if os.path.exists(staging_path):
                        try:
                            os.unlink(staging_path)
                        except OSError:
                            pass
                    raise ValueError(
                        f"{HALT_APPLY_FAILED}: component={component} "
                        f"arc_name={arc_name!r} event={event_id!r} "
                        f"diff_mode={diff_mode!r}: {e}"
                    ) from e
        except ValueError as e:
            out.halt_reason = HALT_APPLY_FAILED
            out.halt_event_id = event_id
            out.errors.append(str(e))
            out.duration_s = time.time() - started
            # Clean up partial scratch
            if os.path.exists(scratch_path):
                try:
                    os.unlink(scratch_path)
                except OSError:
                    pass
            return out
        out.applied_events.append(event_id)

    # The file must have been physically produced by at least one event;
    # if scratch_path doesn't exist, the file was "always skipped" — that
    # is itself a chain-integrity failure (a file should have a physical
    # presence somewhere in its chain).
    if not os.path.exists(scratch_path):
        out.halt_reason = HALT_APPLY_FAILED
        out.errors.append(
            f"file {arc_name!r} (component={component}) has no physical "
            f"presence in the chain from baseline to {out.target_event_id}"
        )
        out.duration_s = time.time() - started
        return out

    # Compute final sha256 and verify (optional) expected_post_merkle_root
    import hashlib
    h = hashlib.sha256()
    with open(scratch_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    final_root = h.hexdigest()
    out.final_merkle_root = final_root

    if (expected_post_merkle_root is not None
            and final_root != expected_post_merkle_root):
        out.halt_reason = HALT_POST_RESTORE_HASH_MISMATCH
        out.errors.append(
            f"final sha256={final_root} ≠ expected {expected_post_merkle_root}"
        )
        if os.path.exists(scratch_path):
            try:
                os.unlink(scratch_path)
            except OSError:
                pass
        out.duration_s = time.time() - started
        return out

    # Atomic install at output_path
    try:
        os.replace(scratch_path, output_path)
    except OSError as e:
        out.halt_reason = HALT_APPLY_FAILED
        out.errors.append(f"failed to install restored file: {e}")
        out.duration_s = time.time() - started
        return out

    out.status = "success"
    out.output_path = output_path
    out.duration_s = time.time() - started
    return out


async def fetch_file_at_event(
    manifest: UnifiedManifest,
    component: str,
    arc_name: str,
    target_event_id: str,
    arweave_fetch: ArweaveFetcher,
) -> bytes:
    """rFP §3.4 — on-demand big-DB historical fetch.

    Reconstruct one file's contents at a specific past event_id and return
    the raw bytes. Used for "what was inner_memory.db like 14 days ago?"
    inspection queries without restoring the whole VPS state.

    Raises ValueError on any restore failure — caller catches + handles.
    """
    import tempfile
    with tempfile.NamedTemporaryFile(
        delete=False, prefix=f"titan_fetch_{arc_name.replace('/', '_')}_",
    ) as tmp:
        tmp_path = tmp.name
    try:
        result = await restore_single_file(
            manifest=manifest,
            component=component,
            arc_name=arc_name,
            output_path=tmp_path,
            arweave_fetch=arweave_fetch,
            target_event_id=target_event_id,
        )
        if result.status != "success":
            raise ValueError(
                f"fetch_file_at_event halted: reason={result.halt_reason} "
                f"errors={result.errors}"
            )
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ── §3.1 step 12 — RESURRECTION_COMPLETE Solana memo ────────────────────


async def emit_resurrection_memo(
    network,
    from_event_id: str,
    ts: Optional[float] = None,
) -> Optional[str]:
    """rFP §3.1 step 12 — inscribe permanent on-chain proof of continuity.

    Memo format (deliberately distinct from the v=2 backup memo so chain
    walkers can tell them apart):
      `RESURRECTION;from_event={event_id};ts={unix_ts}`

    Returns tx_id on success, None on failure. Failure is non-fatal —
    Titan is alive either way; the memo is documentary, not load-bearing.
    """
    if not network or not hasattr(network, "send_sovereign_transaction"):
        return None
    try:
        from titan_hcl.utils.solana_client import (
            build_memo_instruction, is_available)
        if not is_available() or getattr(network, "keypair", None) is None:
            return None
        memo = (
            f"RESURRECTION;from_event={from_event_id};"
            f"ts={int(ts if ts is not None else time.time())}"
        )
        ix = build_memo_instruction(network.pubkey, memo)
        sig = await network.send_sovereign_transaction([ix], priority="LOW")
        if sig:
            logger.info(
                "[Restore] §3.1 step 12: RESURRECTION memo on-chain "
                "tx=%s from_event=%s", sig[:20] if len(sig) > 20 else sig,
                from_event_id[:8])
        return sig
    except Exception as e:
        logger.warning("[Restore] RESURRECTION memo failed (non-fatal): %s", e)
        return None


# ── atomic swap: target_dir → data/ ──────────────────────────────────────


def atomic_swap_target_into_data(
    target_dir: str,
    data_dir: str = "data",
    keep_old_as: str = "data.pre_restore",
) -> dict:
    """rFP §3.1 step 11 prep — atomically swap restored target_dir into the
    live `data/` location.

    Steps:
      1. Rename `data` → `data.pre_restore` (preserves prior state)
      2. Rename `target_dir` → `data`
      3. fsync parent dir for durability

    If step 2 fails, attempts to restore step 1 (rename pre_restore back).

    Returns {"swapped": bool, "preserved_at": str or None,
             "errors": list[str]}.
    """
    out = {"swapped": False, "preserved_at": None, "errors": []}
    parent = os.path.dirname(os.path.abspath(data_dir)) or "."
    preserved = os.path.join(parent, keep_old_as)

    if not os.path.isdir(target_dir):
        out["errors"].append(f"target_dir {target_dir!r} does not exist")
        return out

    # Step 1: preserve current data dir if it exists
    if os.path.exists(data_dir):
        if os.path.exists(preserved):
            # Roll the prior preservation aside before clobbering it,
            # so we don't lose multiple restore generations
            ts_suffix = time.strftime("%Y%m%d_%H%M%S")
            rolled = f"{preserved}.{ts_suffix}"
            try:
                os.rename(preserved, rolled)
                out["errors"].append(
                    f"prior {preserved!r} rolled to {rolled!r}"
                )
            except OSError as e:
                out["errors"].append(
                    f"failed to roll prior preservation: {e}"
                )
                return out
        try:
            os.rename(data_dir, preserved)
            out["preserved_at"] = preserved
        except OSError as e:
            out["errors"].append(f"failed to preserve data dir: {e}")
            return out

    # Step 2: swap target_dir → data
    try:
        os.rename(target_dir, data_dir)
        out["swapped"] = True
    except OSError as e:
        out["errors"].append(f"failed to install restored target: {e}")
        # Try to put the preserved dir back
        if out["preserved_at"]:
            try:
                os.rename(preserved, data_dir)
                out["errors"].append("preserved data dir restored after failure")
                out["preserved_at"] = None
            except OSError as e2:
                out["errors"].append(
                    f"CRITICAL: could not restore preserved data after "
                    f"failed swap: {e2}"
                )
        return out

    # Step 3: fsync parent dir for durability
    try:
        fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        # Non-fatal — some filesystems don't support dir fsync
        pass

    return out
