"""SPEC §24 (Arweave plane) — production upload pipeline.

Orchestrates the producer side of the Phase 2-5 modules into one
meditation-event ship cycle, so the consumer side (Phase 6 restore +
Phase 9 weekly restore-test + Phase 10 verify CLI) has real chain data
to walk on T1 mainnet.

Per meditation:
  1. Load or initialize the UnifiedManifest (per-titan, per data_dir).
  2. Decide event type: baseline (per §24.2 FIRST-WINS of month-boundary
     OR depth_cap) or incremental.
  3. For each tier (personality + timechain, + soul on Sundays):
       a. Resolve current on-disk file paths + arc_names from the
          tier's path tuples (PERSONALITY_PATHS, TIMECHAIN_PATHS,
          WEEKLY_EXTRA_PATHS for soul).
       b. For each file:
            - If baseline event: full-ship payload via diff_encoders.
            - Else (incremental): use content_hash_cache to skip-if-
              unchanged (Phase 4); for changed files, run diff_encoders.
              encode_diff against the baseline-extracted file.
       c. pack_event_tarball() — one tarball per tier.
       d. Upload the tarball to Arweave → tx_id.
       e. Capture per-fork block_ranges (for timechain) from each file's
          diff_dict — needed by Phase 8 surgical repair.
  4. Compute event_merkle_root from per-component tarball sha256s.
  5. commit_event_v3_chain / ChainProvider.commit_memo → zk_commit_tx
     (the v=2 commit_event_merkle_to_zk_vault wrapper was retired — RFP Phase E).
  6. UnifiedManifest.append_event(make_event(...)) + .save().
  7. Emit BACKUP_EVENT_COMPLETE on the bus.

The pipeline is dependency-injected: real Arweave + Solana clients are
wired in from RebirthBackup; tests inject stubs and run the full flow
hermetically.

Opt-in: BackupWorker checks `[backup].unified_v2_enabled` (default
False). When True, this pipeline runs instead of the legacy full-tarball
upload. When False, the legacy path runs unchanged — zero risk to T2/T3
or to T1 rollback.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Iterable, Optional

from titan_hcl.logic import diff_encoders
from titan_hcl.logic.backup_sqlite_snapshot import prepare_sqlite_snapshot
from titan_hcl.logic.backup_event_tarball import (
    FileDiffSpec,
    pack_event_tarball,
    unpack_event_tarball,
)
from titan_hcl.logic.backup_unified_manifest import (
    UnifiedManifest,
    make_event,
    new_event_id,
)
from titan_hcl.logic.backup_zk_commit import compute_event_merkle_root

logger = logging.getLogger(__name__)


# ── Bus event ───────────────────────────────────────────────────────────


EVENT_BACKUP_EVENT_COMPLETE = "BACKUP_EVENT_COMPLETE"
EVENT_BACKUP_EVENT_FAILED = "BACKUP_EVENT_FAILED"
# §24.10 / §24.12 — weekly full-chain restore-test (Phase R4, 2026-06-09).
EVENT_BACKUP_RESTORE_TEST_PASS = "BACKUP_RESTORE_TEST_PASS"
EVENT_BACKUP_RESTORE_TEST_FAIL = "BACKUP_RESTORE_TEST_FAIL"


# ── Injected dependencies ────────────────────────────────────────────────


# Async (tarball_bytes) -> Arweave tx_id. Production wires
# ArweaveStore.upload_bytes(data, tag_app, tag_titan). Tests stub.
ArweaveUploader = Callable[[bytes, dict], "Awaitable[str]"]

# Sovereign v=3 chain committer (chunk 5J-2). Async
#   (event_id, ts_unix, event_type, event_merkle_root, components, prev_sig) -> dict | None
# where event_type = "baseline"|"incremental" (recorded on-chain as typ=B|I) and
# components = ordered [{"tier": "PT"|"TC"|"SL", "tx_id": str, "arc": str}] (arc =
# that component's tarball sha256). Emits ONE v=3 memo per component, commit_state
# co-bundled with the head (PT) memo, prev= threaded event-level (all of an event's
# memos carry prev = prior event's head sig). Returns {"head_sig": str,
# "component_sigs": {tier: sig}} (head_sig = the event's chain anchor) or None on
# ANY chain-write failure (no silent fallback). Production wires
# RebirthBackup.commit_event_v3_chain.
ZkCommitter = Callable[
    [str, int, str, str, list, Optional[str]], "Awaitable[Optional[dict]]"
]


# ── File spec ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TierFileSpec:
    """One source file's pipeline specification.

    Mirrors RebirthBackup.PERSONALITY_PATHS tuple structure but as a
    typed record so the pipeline doesn't depend on tuple layout."""
    source_path: str   # absolute path on host
    arc_name: str      # stable identifier inside tarball
    format_hint: Optional[str] = None  # passed to diff_encoders.select_encoder


# ── Result types ────────────────────────────────────────────────────────


@dataclass
class TierShipResult:
    """One tier's ship outcome."""
    tier: str
    tarball_path: Optional[str] = None
    tarball_size_bytes: int = 0
    tarball_sha256: Optional[str] = None
    tx_id: Optional[str] = None
    files_packed: int = 0
    files_skipped: int = 0
    block_ranges: dict[str, list[int]] = field(default_factory=dict)
    error: Optional[str] = None
    # Mode-B (encrypted data) only: the AES-GCM IV (b64) of the ENCRYPTED tarball
    # uploaded to Arweave. None ⇒ Mode-A (plaintext data). tarball_sha256 (arc) is
    # ALWAYS over the PLAINTEXT tarball, regardless of mode.
    iv_b64: Optional[str] = None


@dataclass
class EventShipResult:
    """Outcome of one full meditation-event ship cycle."""
    status: str  # "shipped" | "failed" | "skipped"
    event_id: Optional[str] = None
    event_type: Optional[str] = None  # baseline | incremental
    baseline_trigger: Optional[str] = None
    event_merkle_root: Optional[str] = None
    zk_commit_tx: Optional[str] = None
    tiers: dict[str, TierShipResult] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    skipped_reason: Optional[str] = None
    duration_s: float = 0.0


# ── per-tier pack logic ──────────────────────────────────────────────────


class MissingDiffBaseError(RuntimeError):
    """A KNOWN file's diff-base was unresolvable at incremental-build time
    (RFP_backup_arweave_sustainability Phase B / INV-BR-9 / INV-BKP-2).

    The graceful path is the pre-check in backup.py (`_precheck_diff_base`),
    which forces a labeled `self_heal` baseline BEFORE building when a known
    file's mirror bytes are missing/drifted. Reaching this raise during an
    "incremental" is the FAIL-CLOSED backstop: we refuse to silently full-ship
    a (possibly multi-hundred-MB) known file mislabeled "incremental" — the
    06-03 ~500MB bug. NEW files (never shipped) are NOT this error — they take
    the legitimate per-file full-ship at `_build_incremental_payload`."""


def _settle_source_snapshot(snap: Optional[str], dd: dict) -> None:
    """Resolve the lifecycle of a SQLite source snapshot relative to the encode
    result (§24.5.a). The consistent image is either:
      • the full-ship PAYLOAD itself (`patch_path == snap`, e.g. a baseline `.db`
        or the no-baseline fallback) → mark `patch_owned` so pack unlinks it; or
      • a transient diff INPUT (xdelta3 read it to make a `.vcdiff`,
        `patch_path != snap`) → unlink it now (the `.vcdiff` is what gets packed).
    """
    if snap is None:
        return
    if dd.get("patch_path") == snap:
        dd["patch_owned"] = True
    else:
        try:
            os.unlink(snap)
        except OSError:
            pass


def _build_full_payload(spec: TierFileSpec) -> dict:
    """For baseline events: full-ship every in-scope file as its raw bytes.

    The encoder dispatch decides which module's encode_diff actually runs;
    diff_encoders.encode_diff with baseline_path=None produces a full
    payload using the appropriate encoder.

    §24.5.a / INV-BR-11: a live SQLite source is first captured as a
    transactionally-consistent online-backup image (`prepare_sqlite_snapshot`);
    the encode + Merkle then run over that image, never the live DB.
    """
    if not os.path.exists(spec.source_path):
        return None
    snap = prepare_sqlite_snapshot(spec.source_path)
    src = snap or spec.source_path
    try:
        dd = diff_encoders.encode_diff(
            current_path=src, baseline_path=None,
            format_hint=spec.format_hint,
        )
    except Exception:
        if snap is not None:
            try:
                os.unlink(snap)
            except OSError:
                pass
        raise
    _settle_source_snapshot(snap, dd)
    return dd


def _build_incremental_payload(spec: TierFileSpec,
                               baseline_file_path: Optional[str]) -> dict:
    """For incremental events: produce a diff against the prior baseline's
    file bytes. baseline_file_path is the on-disk path where the prior
    baseline's reconstructed bytes live (caller is responsible for
    materializing it from the manifest if needed).

    If the file is unchanged (content hash matches), returns a skipped
    pointer record; otherwise produces the encoder's diff.
    """
    if not os.path.exists(spec.source_path):
        return None
    # §24.5.a / INV-BR-11: capture a live SQLite source as a consistent
    # online-backup image FIRST, so the skip-hash, the encode, and the Merkle
    # all read the SAME transactionally-consistent bytes (no TOCTOU between the
    # §24.6 hash and the packed bytes). `src` is the image when SQLite, else the
    # live source (unchanged behaviour for .json/.bin/.faiss).
    snap = prepare_sqlite_snapshot(spec.source_path)
    src = snap or spec.source_path
    try:
        if baseline_file_path is None or not os.path.exists(baseline_file_path):
            # No baseline available — full-ship as fallback (will produce
            # diff_mode="full" inside the encoder)
            dd = diff_encoders.encode_diff(
                current_path=src, baseline_path=None,
                format_hint=spec.format_hint,
            )
            _settle_source_snapshot(snap, dd)
            return dd
        # Cheap content-hash check: skip if unchanged
        cur_hash = diff_encoders.file_merkle_root(src)
        base_hash = diff_encoders.file_merkle_root(baseline_file_path)
        if cur_hash == base_hash:
            # Phase 5 (2026-05-19) — STREAMING refactor: "skipped" diff_dicts
            # carry an empty patch via patch_path pointing at /dev/null-equivalent
            # (we just use a 0-byte temp file so pack_event_tarball treats it
            # uniformly). The metadata schema records diff_mode="skipped" +
            # merkle_root so restore can detect the skip and walk back to the
            # most recent event that DID ship the file.
            import tempfile as _tf
            size = os.path.getsize(src)
            with _tf.NamedTemporaryFile(
                suffix=".skip.bin", delete=False
            ) as _f:
                empty_path = _f.name
            if snap is not None:
                try:
                    os.unlink(snap)  # unchanged → the consistent image isn't needed
                except OSError:
                    pass
            return {
                "diff_mode": "skipped",
                "patch_path": empty_path,
                "patch_owned": True,
                "patch_size_bytes": 0,
                "merkle_root": cur_hash,
                "size_bytes": size,
                "encoder": (spec.format_hint or "full_ship"),
            }
        dd = diff_encoders.encode_diff(
            current_path=src, baseline_path=baseline_file_path,
            format_hint=spec.format_hint,
        )
        _settle_source_snapshot(snap, dd)
        return dd
    except Exception:
        if snap is not None:
            try:
                os.unlink(snap)
            except OSError:
                pass
        raise


def _extract_block_ranges(file_diffs: list[tuple[str, dict]]) -> dict:
    """For timechain tier: gather per-fork block_range from each file's
    diff_dict. Returns {fork_name: [first, last]}. Phase 8 surgical
    repair uses this fast index to avoid fetching tarballs just to
    discover which event covers a given height.

    arc_name convention (Phase 8 backup_timechain_surgical): the file
    "timechain/chain_<fork>.bin" carries the block_range for fork
    `<fork>`.
    """
    ranges: dict[str, list[int]] = {}
    for arc, dd in file_diffs:
        if not arc.startswith("timechain/chain_") or not arc.endswith(".bin"):
            continue
        fork = arc[len("timechain/chain_"):-len(".bin")]
        br = dd.get("block_range")
        if isinstance(br, (list, tuple)) and len(br) == 2:
            ranges[fork] = [int(br[0]), int(br[1])]
    return ranges


def build_tier(
    *,
    tier: str,
    event_id: str,
    event_type: str,
    specs: Iterable[TierFileSpec],
    baseline_resolver: Optional[Callable[[str], Optional[str]]],
    scratch_dir: str,
    titan_id: str,
) -> TierShipResult:
    """Pack one tier's per-file diffs into a tarball — NO upload (Phase 2,
    2026-05-31, pre-stage split). The heavy IO+CPU (reading the big mutable DBs
    to diff against the baseline + zstd pack) lives HERE so the stager can run it
    off the recv loop, ahead of the meditation. Returns a TierShipResult with
    tarball_path/sha/size populated and tx_id=None (upload is a later step).

    baseline_resolver: callable arc_name -> on-disk path of the file's
    prior-baseline reconstructed bytes (or None if not available). For
    baseline events, pass None to short-circuit to full-ship.
    """
    result = TierShipResult(tier=tier)
    file_diffs: list[tuple[str, dict]] = []
    skipped = 0

    specs = list(specs)
    for spec in specs:
        if event_type == "baseline" or baseline_resolver is None:
            dd = _build_full_payload(spec)
        else:
            base_path = baseline_resolver(spec.arc_name) if baseline_resolver \
                else None
            dd = _build_incremental_payload(spec, base_path)
        if dd is None:
            # Source file missing — skip silently (matches legacy behavior
            # of create_personality_archive when a path is absent)
            continue
        file_diffs.append((spec.arc_name, dd))
        if dd.get("diff_mode") == "skipped":
            skipped += 1

    if not file_diffs:
        result.error = f"{tier}: no in-scope files found on disk"
        return result

    # Pack tarball — Phase 5 chunk 5F: zstd-3 (was gzip-9). Extension
    # matches the compression so on-disk inspection (`zstd -dc … | tar tv`)
    # is unambiguous and the 5D verify path can route by suffix.
    os.makedirs(scratch_dir, exist_ok=True)
    out_path = os.path.join(
        scratch_dir, f"event_{event_id}_{tier}.tar.zst"
    )
    pack_specs = [FileDiffSpec(arc, dd) for arc, dd in file_diffs]
    pack_info = pack_event_tarball(
        event_id=event_id, event_type=event_type, component=tier,
        file_specs=pack_specs, output_path=out_path,
    )
    result.tarball_path = out_path
    result.tarball_size_bytes = pack_info["size_bytes"]
    result.tarball_sha256 = pack_info["tarball_sha256"]
    result.files_packed = len(file_diffs)
    result.files_skipped = skipped
    if tier == "timechain":
        result.block_ranges = _extract_block_ranges(file_diffs)
    return result


async def upload_tier(
    result: TierShipResult,
    *,
    arweave_uploader: ArweaveUploader,
    titan_id: str,
    event_id: str,
    event_type: str,
    encryptor: Optional[Callable[[bytes, str], tuple[bytes, str]]] = None,
    path_uploader: Optional[Callable[[str, dict], "Awaitable[str]"]] = None,
) -> TierShipResult:
    """Upload a PRE-BUILT tier tarball (from build_tier) to Arweave + set tx_id.

    Mutates + returns `result`. Safe to call on a staged tarball minutes after
    build_tier produced it (the bytes on disk are immutable once packed).

    `encryptor` (Mode B): given (plaintext_tarball, component) → (ciphertext,
    iv_b64). When provided, the ENCRYPTED bytes are uploaded and `result.iv_b64`
    is set; `tarball_sha256` (the memo arc) stays over the PLAINTEXT tarball so
    integrity verification + the per-backup key derivation (arc[:16]) are
    mode-independent. None ⇒ Mode A (plaintext uploaded).

    `path_uploader` (RFP_backup_redesign_spine Phase B / B-2): a streamed
    PATH→tx_id uploader (e.g. `ChainProvider.put(path)`). On **Mode-A** (no
    encryptor) the plaintext tarball is uploaded **straight from disk** via this
    — NO `f.read()` whole-tarball RAM load. Mode-B still buffers (the one-shot
    AES-GCM encrypt is inherently whole-buffer, INV-MBR-13). None → legacy
    bytes-`arweave_uploader` path (unchanged for existing callers/tests).
    """
    if result.tarball_path is None or not os.path.exists(result.tarball_path):
        result.error = f"{result.tier}: staged tarball missing at upload time"
        return result
    tags = {
        "App-Name": "TitanBackupUnified",
        "Titan-Id": titan_id,
        "Tier": result.tier,
        "Event-Id": event_id,
        "Event-Type": event_type,
    }
    try:
        if encryptor is None and path_uploader is not None:
            # Mode-A streamed: upload the plaintext tarball straight from disk.
            tx_id = await path_uploader(result.tarball_path, tags)
        else:
            with open(result.tarball_path, "rb") as f:
                tarball_bytes = f.read()
            if encryptor is not None:
                try:
                    tarball_bytes, result.iv_b64 = encryptor(tarball_bytes, result.tier)
                except Exception as e:
                    result.error = f"{result.tier}: Mode-B encryption failed: {e}"
                    return result
            tx_id = await arweave_uploader(tarball_bytes, tags)
    except Exception as e:
        result.error = f"Arweave upload failed: {e}"
        return result
    if not tx_id:
        result.error = "Arweave upload returned empty tx_id"
        return result
    result.tx_id = tx_id
    return result


async def ship_tier(
    *,
    tier: str,
    event_id: str,
    event_type: str,
    specs: Iterable[TierFileSpec],
    baseline_resolver: Optional[Callable[[str], Optional[str]]],
    arweave_uploader: ArweaveUploader,
    scratch_dir: str,
    titan_id: str,
    encryptor: Optional[Callable[[bytes, str], tuple[bytes, str]]] = None,
) -> TierShipResult:
    """Pack one tier's per-file diffs into a tarball + upload to Arweave.

    Behavior-identical wrapper over build_tier + upload_tier (kept so existing
    callers/tests are unchanged); the two halves are also used independently by
    the Phase 2 pre-stage path (build ahead, upload on meditation). `encryptor`
    (Mode B) is threaded to upload_tier.
    """
    result = build_tier(
        tier=tier, event_id=event_id, event_type=event_type, specs=specs,
        baseline_resolver=baseline_resolver, scratch_dir=scratch_dir,
        titan_id=titan_id,
    )
    if result.error or result.tarball_path is None:
        return result
    return await upload_tier(
        result, arweave_uploader=arweave_uploader, titan_id=titan_id,
        event_id=event_id, event_type=event_type, encryptor=encryptor,
    )


# ── orchestrator ─────────────────────────────────────────────────────────


async def run_unified_event(
    *,
    titan_id: str,
    manifest: UnifiedManifest,
    personality_specs: Iterable[TierFileSpec],
    timechain_specs: Iterable[TierFileSpec],
    soul_specs: Optional[Iterable[TierFileSpec]] = None,
    baseline_resolver: Optional[Callable[[str, str], Optional[str]]] = None,
    arweave_uploader: ArweaveUploader,
    zk_committer: ZkCommitter,
    scratch_dir: Optional[str] = None,
    cleanup_scratch: bool = True,
    bus_emit: Optional[Callable[[str, dict], None]] = None,
    force_event_type: Optional[str] = None,
    force_trigger: Optional[str] = None,
    encryptor: Optional[Callable[[bytes, str], tuple[bytes, str]]] = None,
) -> EventShipResult:
    """One meditation-event ship cycle.

    Args:
        titan_id: per-titan namespace.
        manifest: caller-loaded UnifiedManifest (we mutate + .save() it).
        personality_specs / timechain_specs: required for every event.
        soul_specs: weekly-only (None on non-Sunday events).
        baseline_resolver: callable (component, arc_name) → file path of
            the prior baseline's reconstructed bytes (or None if not
            materialized). For incremental events only; baseline events
            ignore it.
        arweave_uploader / zk_committer: injected dependencies.
        scratch_dir: where to build tarballs before upload. Defaults to
            a fresh tempfile.mkdtemp.
        cleanup_scratch: True → rmtree after; False → leave for inspection.
        bus_emit: optional callable (event_name, payload) → None. Wired
            to MicroBus.emit by BackupWorker so observability sees
            BACKUP_EVENT_COMPLETE / BACKUP_EVENT_FAILED.

    Returns EventShipResult.
    """
    started = time.time()
    out = EventShipResult(status="failed")
    owns_scratch = False
    if scratch_dir is None:
        scratch_dir = tempfile.mkdtemp(prefix=f"titan_backup_event_{titan_id}_")
        owns_scratch = True

    try:
        # Decide event type via SPEC §24.2 first-wins logic — unless the
        # caller forces it (controlled one-time ops, e.g. the soul-first
        # baseline that must stay incremental on a month-boundary day so it
        # doesn't redundantly re-baseline personality/timechain).
        if force_event_type in ("baseline", "incremental"):
            event_type = force_event_type
            # force_trigger must be a make_event-valid baseline trigger
            # (self_heal for the Phase-B precheck rebase). Default self_heal
            # rather than the old "forced" (which was ∉ the allowed enum →
            # make_event raised — a latent bug; no live caller forced baseline).
            trigger = (force_trigger or "self_heal") if event_type == "baseline" else None
        else:
            should_rebase, trigger = manifest.should_rebase()
            event_type = "baseline" if should_rebase else "incremental"
        out.event_type = event_type
        out.baseline_trigger = trigger if event_type == "baseline" else None

        event_id = new_event_id()
        out.event_id = event_id
        prev_event = manifest.get_latest_event()
        prev_event_id = prev_event["event_id"] if prev_event else None
        prev_zk_short = prev_event.get("zk_memo_prev_short") if prev_event else None

        # If baseline event → ignore baseline_resolver (full-ship)
        resolver_for_tier = (
            (lambda arc: baseline_resolver(tier, arc)) if (
                event_type == "incremental" and baseline_resolver is not None
            ) else None
        )

        # Ship each tier (gather results)
        tier_results: dict[str, TierShipResult] = {}

        # personality + timechain are always shipped
        for tier_name, specs in (
            ("personality", personality_specs),
            ("timechain", timechain_specs),
        ):
            tier = tier_name  # capture for resolver closure
            r = await ship_tier(
                tier=tier_name, event_id=event_id, event_type=event_type,
                specs=specs,
                baseline_resolver=(
                    (lambda arc, _t=tier: baseline_resolver(_t, arc))
                    if (event_type == "incremental" and
                        baseline_resolver is not None)
                    else None
                ),
                arweave_uploader=arweave_uploader,
                scratch_dir=scratch_dir, titan_id=titan_id,
                encryptor=encryptor,
            )
            tier_results[tier_name] = r
            if r.error:
                out.errors.append(f"{tier_name}: {r.error}")

        # soul is weekly-only
        if soul_specs is not None:
            tier = "soul"
            r = await ship_tier(
                tier="soul", event_id=event_id, event_type=event_type,
                specs=soul_specs,
                baseline_resolver=(
                    (lambda arc, _t="soul": baseline_resolver(_t, arc))
                    if (event_type == "incremental" and
                        baseline_resolver is not None)
                    else None
                ),
                arweave_uploader=arweave_uploader,
                scratch_dir=scratch_dir, titan_id=titan_id,
                encryptor=encryptor,
            )
            tier_results["soul"] = r
            if r.error:
                out.errors.append(f"soul: {r.error}")

        out.tiers = tier_results

        # Abort if any required tier failed
        if (tier_results["personality"].tx_id is None
                or tier_results["timechain"].tx_id is None):
            out.status = "failed"
            out.duration_s = time.time() - started
            if bus_emit is not None:
                try:
                    bus_emit(EVENT_BACKUP_EVENT_FAILED, _failed_payload(out))
                except Exception:
                    pass
            return out
        if (soul_specs is not None
                and tier_results.get("soul", TierShipResult(tier="soul")).tx_id
                is None):
            out.status = "failed"
            out.errors.append("soul tier failed on weekly event")
            out.duration_s = time.time() - started
            if bus_emit is not None:
                try:
                    bus_emit(EVENT_BACKUP_EVENT_FAILED, _failed_payload(out))
                except Exception:
                    pass
            return out

        # Compute event_merkle_root from tarball sha256s
        pers_sha = tier_results["personality"].tarball_sha256
        tc_sha = tier_results["timechain"].tarball_sha256
        soul_sha = (
            tier_results["soul"].tarball_sha256
            if soul_specs is not None else None
        )
        event_root = compute_event_merkle_root(
            personality_merkle_root=pers_sha,
            timechain_merkle_root=tc_sha,
            soul_merkle_root=soul_sha,
        )
        out.event_merkle_root = event_root

        # Sovereign v=3 chain commit (chunk 5J-2, Option 1) — emit ONE v=3 memo
        # per uploaded component (PT/TC/SL), each carrying its own arc (component
        # tarball sha256) + url (component Arweave tx_id) + the shared event_root
        # (mrkl). commit_state(event_root) is co-bundled with the head (PT) memo.
        # prev= is event-level: every memo of this event points at the prior
        # event's head sig, threading the chain back to the Day-1 genesis anchor.
        # No silent fallback — any chain-write failure fails the event loudly.
        prev_sig = prev_event.get("zk_commit_tx") if prev_event else None
        # Mode-B carries each component's encrypted-tarball IV (arc stays plaintext).
        components = [
            {"tier": "PT", "tx_id": tier_results["personality"].tx_id, "arc": pers_sha,
             "iv": tier_results["personality"].iv_b64},
            {"tier": "TC", "tx_id": tier_results["timechain"].tx_id, "arc": tc_sha,
             "iv": tier_results["timechain"].iv_b64},
        ]
        if soul_specs is not None and soul_sha:
            components.append(
                {"tier": "SL", "tx_id": tier_results["soul"].tx_id, "arc": soul_sha,
                 "iv": tier_results["soul"].iv_b64}
            )

        try:
            commit_result = await zk_committer(
                event_id, int(started), event_type, event_root, components, prev_sig)
        except Exception as e:
            out.errors.append(f"v=3 chain commit raised: {e}")
            commit_result = None
        head_sig = (commit_result or {}).get("head_sig") if commit_result else None
        if not head_sig:
            out.errors.append("v=3 chain commit returned no head_sig (chain_write_failed)")
            out.status = "failed"
            out.duration_s = time.time() - started
            if bus_emit is not None:
                try:
                    bus_emit(EVENT_BACKUP_EVENT_FAILED, _failed_payload(out))
                except Exception:
                    pass
            return out
        out.zk_commit_tx = head_sig

        # Append event to manifest + save
        personality_sub = {
            "tx_id": tier_results["personality"].tx_id,
            "merkle_root": pers_sha,
            "size_bytes": tier_results["personality"].tarball_size_bytes,
            "diff_mode": event_type,
            "skipped_files": [],  # could be enriched from per-file metadata
            # §24.3 (2026-06-09): Mode-B GCM nonce — public (also on-chain in the
            # v=3 memo), stored locally so an in-loop restore decrypts without a
            # Solana RPC. None on Mode-A (plaintext). (Phase B reconstruct.)
            "iv": tier_results["personality"].iv_b64,
        }
        timechain_sub = {
            "tx_id": tier_results["timechain"].tx_id,
            "merkle_root": tc_sha,
            "size_bytes": tier_results["timechain"].tarball_size_bytes,
            "diff_mode": event_type,
            "block_ranges": tier_results["timechain"].block_ranges,
            "iv": tier_results["timechain"].iv_b64,
        }
        soul_sub = None
        if soul_specs is not None:
            soul_sub = {
                "tx_id": tier_results["soul"].tx_id,
                "merkle_root": soul_sha,
                "size_bytes": tier_results["soul"].tarball_size_bytes,
                "diff_mode": event_type,
                "iv": tier_results["soul"].iv_b64,
            }
        event = make_event(
            event_id=event_id, event_type=event_type,
            prev_event_id=prev_event_id,
            baseline_trigger=trigger if event_type == "baseline" else None,
            personality=personality_sub, timechain=timechain_sub,
            soul=soul_sub, zk_commit_tx=head_sig,
            zk_memo_prev_short=(prev_sig[:16] if prev_sig else "genesis"),
        )
        manifest.append_event(event)
        manifest.save()

        out.status = "shipped"
        out.duration_s = time.time() - started
        logger.info(
            "[BackupPipeline:%s] event SHIPPED: id=%s type=%s "
            "p_tx=%s t_tx=%s s_tx=%s zk=%s duration=%.1fs",
            titan_id, event_id[:8], event_type,
            tier_results["personality"].tx_id[:16] + "...",
            tier_results["timechain"].tx_id[:16] + "...",
            (tier_results["soul"].tx_id[:16] + "..."
             if soul_specs is not None else "-"),
            head_sig[:16] + "...", out.duration_s,
        )
        if bus_emit is not None:
            try:
                bus_emit(EVENT_BACKUP_EVENT_COMPLETE, {
                    "event": EVENT_BACKUP_EVENT_COMPLETE,
                    "titan_id": titan_id,
                    "event_id": event_id,
                    "event_type": event_type,
                    "baseline_trigger": out.baseline_trigger,
                    "personality_tx": tier_results["personality"].tx_id,
                    "timechain_tx": tier_results["timechain"].tx_id,
                    "soul_tx": (tier_results["soul"].tx_id
                                if soul_specs is not None else None),
                    "zk_commit_tx": head_sig,
                    "event_merkle_root": event_root,
                    "duration_s": round(out.duration_s, 3),
                })
            except Exception as e:
                logger.warning(
                    "[BackupPipeline:%s] bus_emit raised: %s", titan_id, e,
                )
        return out
    finally:
        if owns_scratch and cleanup_scratch:
            try:
                import shutil
                shutil.rmtree(scratch_dir, ignore_errors=True)
            except Exception:
                pass


@dataclass
class StagedEvent:
    """A unified event pre-BUILT by build_unified_event, ready for a fast
    ship_staged_event (Phase 2 pre-stage). Tarballs live on disk in scratch_dir
    (NOT cleaned at build); upload + chain-commit happen later on meditation.

    baseline_event_id pins which baseline the incremental diffs were computed
    against — ship_staged_event refuses to ship if the manifest baseline has
    since changed (a rebase shipped in between), forcing a fresh rebuild.
    """
    event_id: str
    event_type: str
    baseline_trigger: Optional[str]
    baseline_event_id: Optional[str]
    prev_event_id: Optional[str]
    soul_present: bool
    tier_results: dict
    scratch_dir: str
    built_at: float
    titan_id: str


def build_unified_event(
    *,
    titan_id: str,
    manifest: UnifiedManifest,
    personality_specs: Iterable[TierFileSpec],
    timechain_specs: Iterable[TierFileSpec],
    soul_specs: Optional[Iterable[TierFileSpec]] = None,
    baseline_resolver: Optional[Callable[[str, str], Optional[str]]] = None,
    scratch_dir: Optional[str] = None,
    force_event_type: Optional[str] = None,
    force_trigger: Optional[str] = None,
) -> StagedEvent:
    """Phase 2 — BUILD a unified event's tarballs WITHOUT uploading.

    The heavy half (read big mutable DBs → diff vs baseline → zstd pack); the
    stager runs it OFF the recv loop, ahead of the meditation. event_type
    (baseline vs incremental) is decided here from the manifest and diffs are
    computed against the current baseline; ship_staged_event later verifies that
    baseline is still current before committing. Returns a StagedEvent whose
    tarballs persist in scratch_dir for ship_staged_event to upload. Pure build:
    NO upload, NO manifest mutation, NO chain write.
    """
    if scratch_dir is None:
        scratch_dir = tempfile.mkdtemp(prefix=f"titan_backup_stage_{titan_id}_")
    else:
        os.makedirs(scratch_dir, exist_ok=True)

    if force_event_type in ("baseline", "incremental"):
        event_type = force_event_type
        trigger = (force_trigger or "self_heal") if event_type == "baseline" else None
    else:
        should_rebase, trigger = manifest.should_rebase()
        event_type = "baseline" if should_rebase else "incremental"
    event_id = new_event_id()
    prev_event = manifest.get_latest_event()
    prev_event_id = prev_event["event_id"] if prev_event else None
    baseline_event_id = manifest.current_baseline_event_id

    tier_results: dict[str, TierShipResult] = {}
    tier_inputs = [("personality", personality_specs),
                   ("timechain", timechain_specs)]
    if soul_specs is not None:
        tier_inputs.append(("soul", soul_specs))
    for tier_name, specs in tier_inputs:
        tier = tier_name  # capture for resolver closure
        tier_results[tier_name] = build_tier(
            tier=tier_name, event_id=event_id, event_type=event_type,
            specs=specs,
            baseline_resolver=(
                (lambda arc, _t=tier: baseline_resolver(_t, arc))
                if (event_type == "incremental" and baseline_resolver is not None)
                else None
            ),
            scratch_dir=scratch_dir, titan_id=titan_id,
        )

    return StagedEvent(
        event_id=event_id, event_type=event_type,
        baseline_trigger=(trigger if event_type == "baseline" else None),
        baseline_event_id=baseline_event_id, prev_event_id=prev_event_id,
        soul_present=(soul_specs is not None), tier_results=tier_results,
        scratch_dir=scratch_dir, built_at=time.time(), titan_id=titan_id,
    )


async def ship_staged_event(
    staged: StagedEvent,
    *,
    manifest: UnifiedManifest,
    arweave_uploader: ArweaveUploader,
    zk_committer: ZkCommitter,
    bus_emit: Optional[Callable[[str, dict], None]] = None,
    cleanup_scratch: bool = True,
    encryptor: Optional[Callable[[bytes, str], tuple[bytes, str]]] = None,
    path_uploader: Optional[Callable[[str, dict], "Awaitable[str]"]] = None,
) -> EventShipResult:
    """Phase 2 — SHIP a pre-built StagedEvent (fast, on meditation).

    Validates the baseline is still current (else returns status="stale_baseline"
    so the caller rebuilds), uploads the staged tarballs, then runs the SAME
    merkle → v=3 ZK chain-commit → manifest-append finalize as run_unified_event
    via the same primitives (compute_event_merkle_root / zk_committer / make_event
    / manifest.append_event) → byte-identical on-chain event. No diff/pack here.
    """
    started = time.time()
    titan_id = staged.titan_id
    event_id = staged.event_id
    event_type = staged.event_type
    out = EventShipResult(
        status="failed", event_id=event_id, event_type=event_type,
        baseline_trigger=staged.baseline_trigger,
    )
    try:
        # Staleness guard: a baseline shipped since build → the staged
        # incrementals diff against the WRONG baseline. Refuse; caller rebuilds.
        current_baseline = manifest.current_baseline_event_id
        if staged.baseline_event_id != current_baseline:
            out.status = "stale_baseline"
            out.skipped_reason = (
                f"staged baseline {staged.baseline_event_id} != current "
                f"{current_baseline} — rebuild required")
            out.errors.append(out.skipped_reason)
            out.duration_s = time.time() - started
            return out

        tier_results = staged.tier_results
        prev_event = manifest.get_latest_event()
        prev_event_id = prev_event["event_id"] if prev_event else None

        # Upload each pre-built tier (the only network step now).
        tier_order = ["personality", "timechain"] + (
            ["soul"] if staged.soul_present else [])
        for tier_name in tier_order:
            r = tier_results.get(tier_name)
            if r is None or r.error or r.tarball_path is None:
                out.status = "failed"
                out.errors.append(
                    f"{tier_name}: not built "
                    f"({r.error if r else 'missing'})")
                out.duration_s = time.time() - started
                if bus_emit is not None:
                    try:
                        bus_emit(EVENT_BACKUP_EVENT_FAILED, _failed_payload(out))
                    except Exception:
                        pass
                return out
            await upload_tier(
                r, arweave_uploader=arweave_uploader, titan_id=titan_id,
                event_id=event_id, event_type=event_type, encryptor=encryptor,
                path_uploader=path_uploader)
            if r.error:
                out.errors.append(f"{tier_name}: {r.error}")
        out.tiers = tier_results

        if (tier_results["personality"].tx_id is None
                or tier_results["timechain"].tx_id is None
                or (staged.soul_present
                    and tier_results["soul"].tx_id is None)):
            out.status = "failed"
            out.errors.append("required tier upload failed")
            out.duration_s = time.time() - started
            if bus_emit is not None:
                try:
                    bus_emit(EVENT_BACKUP_EVENT_FAILED, _failed_payload(out))
                except Exception:
                    pass
            return out

        # ── finalize (identical to run_unified_event) ──
        pers_sha = tier_results["personality"].tarball_sha256
        tc_sha = tier_results["timechain"].tarball_sha256
        soul_sha = (tier_results["soul"].tarball_sha256
                    if staged.soul_present else None)
        event_root = compute_event_merkle_root(
            personality_merkle_root=pers_sha, timechain_merkle_root=tc_sha,
            soul_merkle_root=soul_sha)
        out.event_merkle_root = event_root

        prev_sig = prev_event.get("zk_commit_tx") if prev_event else None
        components = [
            {"tier": "PT", "tx_id": tier_results["personality"].tx_id,
             "arc": pers_sha, "iv": tier_results["personality"].iv_b64},
            {"tier": "TC", "tx_id": tier_results["timechain"].tx_id,
             "arc": tc_sha, "iv": tier_results["timechain"].iv_b64},
        ]
        if staged.soul_present and soul_sha:
            components.append(
                {"tier": "SL", "tx_id": tier_results["soul"].tx_id,
                 "arc": soul_sha, "iv": tier_results["soul"].iv_b64})

        try:
            commit_result = await zk_committer(
                event_id, int(started), event_type, event_root, components,
                prev_sig)
        except Exception as e:
            out.errors.append(f"v=3 chain commit raised: {e}")
            commit_result = None
        head_sig = ((commit_result or {}).get("head_sig")
                    if commit_result else None)
        if not head_sig:
            out.errors.append(
                "v=3 chain commit returned no head_sig (chain_write_failed)")
            out.status = "failed"
            out.duration_s = time.time() - started
            if bus_emit is not None:
                try:
                    bus_emit(EVENT_BACKUP_EVENT_FAILED, _failed_payload(out))
                except Exception:
                    pass
            return out
        out.zk_commit_tx = head_sig

        personality_sub = {
            "tx_id": tier_results["personality"].tx_id,
            "merkle_root": pers_sha,
            "size_bytes": tier_results["personality"].tarball_size_bytes,
            "diff_mode": event_type, "skipped_files": [],
            # §24.3 (2026-06-09): Mode-B GCM nonce, stored locally for in-loop
            # decrypt (None on Mode-A). See run_unified_event finalize.
            "iv": tier_results["personality"].iv_b64,
        }
        timechain_sub = {
            "tx_id": tier_results["timechain"].tx_id, "merkle_root": tc_sha,
            "size_bytes": tier_results["timechain"].tarball_size_bytes,
            "diff_mode": event_type,
            "block_ranges": tier_results["timechain"].block_ranges,
            "iv": tier_results["timechain"].iv_b64,
        }
        soul_sub = None
        if staged.soul_present:
            soul_sub = {
                "tx_id": tier_results["soul"].tx_id, "merkle_root": soul_sha,
                "size_bytes": tier_results["soul"].tarball_size_bytes,
                "diff_mode": event_type,
                "iv": tier_results["soul"].iv_b64,
            }
        event = make_event(
            event_id=event_id, event_type=event_type,
            prev_event_id=prev_event_id,
            baseline_trigger=staged.baseline_trigger,
            personality=personality_sub, timechain=timechain_sub,
            soul=soul_sub, zk_commit_tx=head_sig,
            zk_memo_prev_short=(prev_sig[:16] if prev_sig else "genesis"),
        )
        manifest.append_event(event)
        manifest.save()

        out.status = "shipped"
        out.duration_s = time.time() - started
        logger.info(
            "[BackupPipeline:%s] STAGED event SHIPPED: id=%s type=%s zk=%s "
            "ship_dur=%.1fs (built %.0fs earlier)",
            titan_id, event_id[:8], event_type, head_sig[:16] + "...",
            out.duration_s, max(0.0, started - staged.built_at))
        if bus_emit is not None:
            try:
                bus_emit(EVENT_BACKUP_EVENT_COMPLETE, {
                    "event": EVENT_BACKUP_EVENT_COMPLETE, "titan_id": titan_id,
                    "event_id": event_id, "event_type": event_type,
                    "baseline_trigger": out.baseline_trigger,
                    "personality_tx": tier_results["personality"].tx_id,
                    "timechain_tx": tier_results["timechain"].tx_id,
                    "soul_tx": (tier_results["soul"].tx_id
                                if staged.soul_present else None),
                    "zk_commit_tx": head_sig, "event_merkle_root": event_root,
                    "duration_s": round(out.duration_s, 3),
                })
            except Exception as e:
                logger.warning(
                    "[BackupPipeline:%s] bus_emit raised: %s", titan_id, e)
        return out
    finally:
        if cleanup_scratch:
            try:
                import shutil
                shutil.rmtree(staged.scratch_dir, ignore_errors=True)
            except Exception:
                pass


def _failed_payload(out: EventShipResult) -> dict:
    return {
        "event": EVENT_BACKUP_EVENT_FAILED,
        "event_id": out.event_id,
        "event_type": out.event_type,
        "errors": list(out.errors),
        "duration_s": round(out.duration_s, 3),
    }
