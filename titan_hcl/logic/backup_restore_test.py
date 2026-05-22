"""SPEC §24.12 + rFP §3.5 — Weekly mandatory restore-test.

Closes the recurring complaint from `rFP_backup_worker.md` §10.Q3 — "we
backup and never verify we can restore" — by running a full crash-recovery
walk (Phase 6 restore_full) into a scratch dir every Sunday after the
soul upload, verifying every restored file byte-for-byte against the
manifest's recorded hashes.

Maker decision Q2 (2026-05-15): FULL byte-for-byte hash verification,
not sampled. The bandwidth cost (~150 MB/week) + runtime (~5min) is
accepted in exchange for drift-detection strictness. Codified in SPEC
§24.12 + BACKUP_RESTORE_TEST_CADENCE_DAYS=7.

On FAIL:
  - Emit BACKUP_RESTORE_TEST_FAIL on the bus (per-event metadata for
    Maker to investigate)
  - Set a halt flag the BackupWorker checks before next scheduled
    upload — prevents shipping new events on a broken chain
  - Telegram alert via maker_notify (caller wires the actual transport)

On PASS:
  - Emit BACKUP_RESTORE_TEST_PASS with chain-depth + duration metrics
  - Persist last_pass_ts so observability dashboards can age the
    last-known-good timestamp

This module is pure logic — caller (BackupWorker) supplies:
  - arweave_fetch, memo_fetch (the §3.1 step 5-8 inputs)
  - on_pass / on_fail callbacks (bus emit + telegram + halt-flag write)
  - scratch_dir lifecycle (typically tempfile.mkdtemp)

The verification step is delegated to restore_full (which already
verifies tarball sha256 + event Merkle + ZK memo round-trip). The
"full byte-for-byte" requirement is met because restore_full's
apply_event_components calls each diff_encoder's apply_diff(), which
performs a post-write sha256 check against the encoded merkle_root for
every file in every event.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

from titan_hcl.logic.backup_restore import (
    ArweaveFetcher,
    MemoFetcher,
    RestoreResult,
    restore_full,
)
from titan_hcl.logic.backup_unified_manifest import UnifiedManifest

logger = logging.getLogger(__name__)


# Bus event constants — wired by BackupWorker on emit
EVENT_BACKUP_RESTORE_TEST_PASS = "BACKUP_RESTORE_TEST_PASS"
EVENT_BACKUP_RESTORE_TEST_FAIL = "BACKUP_RESTORE_TEST_FAIL"


@dataclass
class RestoreTestResult:
    """Outcome of one weekly restore-test run.

    `restore_result` is the underlying RestoreResult from restore_full —
    callers can inspect halt_reason / halt_event_id / errors for the
    bus event payload."""
    status: str  # "pass" | "fail" | "skipped"
    ts_unix: float = 0.0
    target_event_id: Optional[str] = None
    chain_depth: int = 0
    bytes_fetched: int = 0
    duration_s: float = 0.0
    restore_result: Optional[RestoreResult] = None
    scratch_dir: Optional[str] = None
    skipped_reason: Optional[str] = None
    notes: list[str] = field(default_factory=list)


async def run_weekly_restore_test(
    *,
    titan_id: str,
    manifest: UnifiedManifest,
    arweave_fetch: ArweaveFetcher,
    memo_fetch: MemoFetcher,
    arc_to_target: Callable[[str, str], str],
    on_pass: Optional[Callable[[RestoreTestResult], None]] = None,
    on_fail: Optional[Callable[[RestoreTestResult], None]] = None,
    scratch_dir: Optional[str] = None,
    cleanup_scratch_on_pass: bool = True,
    cleanup_scratch_on_fail: bool = False,
) -> RestoreTestResult:
    """SPEC §24.12 / rFP §3.5 — one full weekly restore-test cycle.

    Args:
        titan_id: Used for log lines + future per-titan observability.
        manifest: The unified manifest to walk (caller loads it).
        arweave_fetch + memo_fetch: §3.1 inputs.
        arc_to_target: callable (component, arc_name) → absolute scratch
            path. Caller-supplied so the on-host PERSONALITY_PATHS
            mapping isn't hard-coded here.
        on_pass / on_fail: invoked with the RestoreTestResult AFTER the
            scratch dir is decided. Wire to bus emit + telegram +
            halt-flag write (BackupWorker uses these).
        scratch_dir: Optional pre-allocated scratch. Defaults to a
            fresh tempfile.mkdtemp(prefix="titan_restore_test_").
        cleanup_scratch_on_pass: Default True — we don't need the
            reconstructed files after PASS (they were just verified).
        cleanup_scratch_on_fail: Default False — leave the scratch in
            place on FAIL so Maker can inspect partial state.

    Returns RestoreTestResult. Status "skipped" when manifest is empty
    (no backup events yet — nothing to test); not a failure, just no-op.
    """
    started = time.time()
    out = RestoreTestResult(status="fail", ts_unix=started)

    # Skip cleanly if no manifest history
    if not manifest.events:
        out.status = "skipped"
        out.skipped_reason = "manifest_empty_no_events_to_restore"
        out.duration_s = time.time() - started
        logger.info(
            "[RestoreTest:%s] skipped — manifest has no events yet",
            titan_id,
        )
        return out

    # Scratch dir
    owns_scratch = False
    if scratch_dir is None:
        scratch_dir = tempfile.mkdtemp(prefix=f"titan_restore_test_{titan_id}_")
        owns_scratch = True
    else:
        os.makedirs(scratch_dir, exist_ok=True)
    out.scratch_dir = scratch_dir

    logger.info(
        "[RestoreTest:%s] starting — chain has %d events, scratch=%s",
        titan_id, len(manifest.events), scratch_dir,
    )

    try:
        restore_result = await restore_full(
            manifest=manifest,
            target_dir=scratch_dir,
            arweave_fetch=arweave_fetch,
            memo_fetch=memo_fetch,
            arc_to_target=arc_to_target,
        )
        out.restore_result = restore_result
        out.target_event_id = restore_result.target_event_id
        out.chain_depth = len(restore_result.applied_events)
        out.bytes_fetched = restore_result.bytes_fetched
        out.duration_s = time.time() - started

        if restore_result.status == "success":
            out.status = "pass"
            logger.info(
                "[RestoreTest:%s] PASS — depth=%d bytes=%d duration=%.1fs "
                "target_event=%s",
                titan_id, out.chain_depth, out.bytes_fetched,
                out.duration_s, out.target_event_id,
            )
            if on_pass is not None:
                try:
                    on_pass(out)
                except Exception as e:
                    out.notes.append(f"on_pass callback raised: {e}")
                    logger.warning(
                        "[RestoreTest:%s] on_pass callback failed: %s",
                        titan_id, e,
                    )
            if owns_scratch and cleanup_scratch_on_pass:
                _cleanup_scratch(scratch_dir, out)
        else:
            out.status = "fail"
            logger.error(
                "[RestoreTest:%s] FAIL — halt_reason=%s halt_event=%s "
                "errors=%s",
                titan_id, restore_result.halt_reason,
                restore_result.halt_event_id,
                restore_result.errors,
            )
            if on_fail is not None:
                try:
                    on_fail(out)
                except Exception as e:
                    out.notes.append(f"on_fail callback raised: {e}")
                    logger.warning(
                        "[RestoreTest:%s] on_fail callback failed: %s",
                        titan_id, e,
                    )
            if owns_scratch and cleanup_scratch_on_fail:
                _cleanup_scratch(scratch_dir, out)
    except Exception as e:
        out.status = "fail"
        out.notes.append(f"restore_full raised: {e}")
        out.duration_s = time.time() - started
        logger.exception(
            "[RestoreTest:%s] restore_full raised unexpectedly", titan_id,
        )
        if on_fail is not None:
            try:
                on_fail(out)
            except Exception as cb_e:
                out.notes.append(f"on_fail callback raised: {cb_e}")
        if owns_scratch and cleanup_scratch_on_fail:
            _cleanup_scratch(scratch_dir, out)

    return out


def _cleanup_scratch(scratch_dir: str, out: RestoreTestResult) -> None:
    try:
        shutil.rmtree(scratch_dir, ignore_errors=True)
        out.notes.append(f"cleaned up scratch {scratch_dir}")
    except Exception as e:
        out.notes.append(f"scratch cleanup failed: {e}")


def is_due_for_test(
    last_test_ts: Optional[float],
    cadence_days: int = 7,
    now: Optional[float] = None,
) -> bool:
    """True when a weekly restore-test should run. Used by BackupWorker
    to schedule.

    SPEC §24.9 / Maker decision Q2: BACKUP_RESTORE_TEST_CADENCE_DAYS=7.
    The exact day-of-week ("Sunday after soul upload") is BackupWorker's
    business; this helper just answers "has it been ≥ cadence_days?"
    """
    if last_test_ts is None:
        return True
    n = now if now is not None else time.time()
    elapsed = n - last_test_ts
    return elapsed >= (cadence_days * 86400)


def build_pass_bus_payload(result: RestoreTestResult,
                           titan_id: str) -> dict:
    """Canonical PASS bus payload — used by BackupWorker when emitting
    BACKUP_RESTORE_TEST_PASS. Centralized here so the schema is one
    source of truth (observability + telegram parsers see the same shape).
    """
    return {
        "event": EVENT_BACKUP_RESTORE_TEST_PASS,
        "titan_id": titan_id,
        "ts_unix": result.ts_unix,
        "target_event_id": result.target_event_id,
        "chain_depth": result.chain_depth,
        "bytes_fetched": result.bytes_fetched,
        "duration_s": round(result.duration_s, 3),
        "scratch_dir": result.scratch_dir,
    }


def build_fail_bus_payload(result: RestoreTestResult,
                           titan_id: str) -> dict:
    """Canonical FAIL bus payload + telegram message body. Includes
    enough information for Maker to investigate without crawling logs."""
    rr = result.restore_result
    payload = {
        "event": EVENT_BACKUP_RESTORE_TEST_FAIL,
        "titan_id": titan_id,
        "ts_unix": result.ts_unix,
        "chain_depth": result.chain_depth,
        "bytes_fetched": result.bytes_fetched,
        "duration_s": round(result.duration_s, 3),
        "scratch_dir": result.scratch_dir,
        "halt_reason": rr.halt_reason if rr else None,
        "halt_event_id": rr.halt_event_id if rr else None,
        "errors": list(rr.errors) if rr else [],
        "notes": list(result.notes),
    }
    return payload


def telegram_fail_message(payload: dict) -> str:
    """Format the FAIL payload as a Telegram message body. Caller passes
    this through their existing telegram client (e.g. maker_notify)."""
    titan = payload.get("titan_id", "?")
    halt = payload.get("halt_reason", "unknown")
    ev = payload.get("halt_event_id") or "-"
    errors = payload.get("errors") or []
    err_first = errors[0] if errors else "(no error detail)"
    return (
        f"🚨 BACKUP RESTORE TEST FAILED ({titan})\n"
        f"halt_reason: {halt}\n"
        f"halt_event_id: {ev}\n"
        f"first_error: {err_first}\n"
        f"chain_depth: {payload.get('chain_depth', 0)}\n"
        f"duration: {payload.get('duration_s', 0)}s\n"
        f"scratch: {payload.get('scratch_dir', '(cleaned)')}\n"
        f"⚠ scheduled backups should be HALTED until investigated"
    )
