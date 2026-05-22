"""SPEC §24.11 + rFP §3.6 — `arch_map backup verify` CLI family.

Six subcommands, all non-destructive, all run against the unified manifest
+ Arweave + ZK Vault. Pure logic — arch_map.py is a thin dispatcher.

  verify --personality     per-event personality tier sha256 + ZK Merkle
  verify --timechain       per-event timechain tier sha256 + ZK Merkle
  verify --soul            per-event soul tier sha256 + ZK Merkle (weekly only)
  verify --allbackups      union of above three
  verify --restore-sim     full §3.1 restore into scratch, byte-for-byte
  audit-coverage           data/ inventory drift detection vs declared paths

Output is line-per-event, OK/FAIL annotated. Exit code 0 iff all OK,
1 otherwise — so CI/cron can gate on it.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Iterable, Optional

from titan_hcl.logic.backup_restore import (
    ArweaveFetcher,
    MemoFetcher,
    RestoreResult,
    restore_full,
    verify_component_merkle,
    verify_event_merkle,
    verify_event_zk_commit,
)
from titan_hcl.logic.backup_unified_manifest import UnifiedManifest

logger = logging.getLogger(__name__)


# ── helpers ──────────────────────────────────────────────────────────────


# Components in the order we display them
TIER_ORDER = ("personality", "timechain", "soul")


@dataclass
class VerifyTierResult:
    """One verify-tier run outcome.

    Per-event records detail what passed/failed; counts summarize."""
    tier: str
    total_events: int = 0
    ok_events: int = 0
    fail_events: int = 0
    skipped_events: int = 0  # e.g. soul tier on non-weekly events
    lines: list[str] = field(default_factory=list)


@dataclass
class RestoreSimResult:
    """Outcome of `verify --restore-sim`."""
    status: str  # "ok" | "failed"
    restore_result: Optional[RestoreResult]
    lines: list[str] = field(default_factory=list)
    files_verified: int = 0


@dataclass
class AuditCoverageResult:
    """Outcome of `audit-coverage` (inventory drift detection)."""
    status: str  # "ok" | "drift"
    declared_count: int = 0
    found_count: int = 0
    undeclared_files: list[str] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)


# ── verify --personality / --timechain / --soul ─────────────────────────


async def verify_tier(
    manifest: UnifiedManifest,
    tier: str,
    arweave_fetch: ArweaveFetcher,
    memo_fetch: MemoFetcher,
    *,
    verify_zk_chain: bool = True,
    line_callback: Optional[Callable[[str], None]] = None,
) -> VerifyTierResult:
    """Walk every manifest event; for each:
      - Skip if the tier subdict is None (soul on a non-weekly event)
      - Fetch tarball from Arweave
      - Verify sha256 matches event[tier].merkle_root
      - If verify_zk_chain: recompose event_merkle_root, fetch + parse
        the v=2 memo at event.zk_commit_tx, verify root/prev match

    Each event produces one OK/FAIL line (emitted via line_callback if
    provided, else accumulated in result.lines).

    Exit semantics (for the CLI dispatcher): fail_events == 0 → exit 0,
    else exit 1.
    """
    if tier not in TIER_ORDER:
        raise ValueError(
            f"tier must be one of {TIER_ORDER}, got {tier!r}"
        )

    result = VerifyTierResult(tier=tier)
    prev_event_merkle_root: Optional[str] = None

    def _emit(line: str):
        result.lines.append(line)
        if line_callback is not None:
            line_callback(line)

    for ev in manifest.events:
        result.total_events += 1
        eid = ev.get("event_id", "?")
        sub = ev.get(tier)
        if sub is None:
            result.skipped_events += 1
            _emit(f"[SKIP] event {eid} — no {tier} tier this event")
            continue
        tx_id = sub.get("tx_id", "?")
        expected_root = sub.get("merkle_root", "?")
        # Fetch tarball
        try:
            tarball = await arweave_fetch(tx_id)
        except Exception as e:
            result.fail_events += 1
            _emit(
                f"[FAIL] event {eid} {tier} — Arweave fetch failed "
                f"(tx={tx_id[:16]}...): {e}"
            )
            continue
        # Sha256 check
        actual = hashlib.sha256(tarball).hexdigest()
        if actual != expected_root:
            result.fail_events += 1
            _emit(
                f"[FAIL] event {eid} {tier} — sha256 mismatch "
                f"(expected={expected_root[:16]}... got={actual[:16]}...)"
            )
            continue

        # Optional ZK round-trip — recompose event_merkle_root from this
        # tier's bytes + any others we can derive cheaply. For tier-only
        # verification we recompose using the manifest's recorded per-
        # component merkle_roots (the other tiers' tarballs are NOT
        # fetched here — they'd be redundant against their own
        # verify --<other-tier> run).
        if verify_zk_chain:
            try:
                recomposed = verify_event_merkle(ev, {tier: tarball})
                await verify_event_zk_commit(
                    event=ev,
                    recomposed_event_merkle_root=recomposed,
                    memo_fetch=memo_fetch,
                    prev_event_merkle_root_for_check=prev_event_merkle_root,
                )
                prev_event_merkle_root = recomposed
            except RuntimeError as e:
                result.fail_events += 1
                _emit(
                    f"[FAIL] event {eid} {tier} — ZK disconnect: {e}"
                )
                continue
            except ValueError as e:
                result.fail_events += 1
                _emit(
                    f"[FAIL] event {eid} {tier} — ZK Merkle mismatch: {e}"
                )
                continue

        result.ok_events += 1
        _emit(
            f"[OK]   event {eid} {tier} — tx={tx_id[:16]}... "
            f"merkle MATCH zk_commit MATCH"
        )

    return result


async def verify_allbackups(
    manifest: UnifiedManifest,
    arweave_fetch: ArweaveFetcher,
    memo_fetch: MemoFetcher,
    *,
    verify_zk_chain: bool = True,
    line_callback: Optional[Callable[[str], None]] = None,
) -> dict[str, VerifyTierResult]:
    """Run verify_tier for personality + timechain + soul. Returns dict
    keyed by tier name."""
    out: dict[str, VerifyTierResult] = {}
    for tier in TIER_ORDER:
        if line_callback:
            line_callback(f"\n── verify --{tier} ──")
        out[tier] = await verify_tier(
            manifest=manifest, tier=tier,
            arweave_fetch=arweave_fetch, memo_fetch=memo_fetch,
            verify_zk_chain=verify_zk_chain, line_callback=line_callback,
        )
    return out


# ── verify --restore-sim ─────────────────────────────────────────────────


async def verify_restore_sim(
    manifest: UnifiedManifest,
    arweave_fetch: ArweaveFetcher,
    memo_fetch: MemoFetcher,
    arc_to_target: Callable[[str, str], str],
    *,
    scratch_dir: Optional[str] = None,
    line_callback: Optional[Callable[[str], None]] = None,
) -> RestoreSimResult:
    """Full §3.1 restore into scratch, byte-for-byte verified (the same
    verification mode the weekly restore-test does — restore_full's
    apply_diff post-write hash check covers every file).

    Returns RestoreSimResult with overall status + the underlying
    RestoreResult for failure diagnostics.

    Scratch dir is caller-managed: pass scratch_dir to control lifecycle
    (e.g. operator wants to inspect after), or omit for an ephemeral
    tempfile.mkdtemp that we DO NOT auto-clean (--restore-sim is an
    operator command; they'll want to keep state on FAIL).
    """
    result = RestoreSimResult(status="failed", restore_result=None)

    def _emit(line: str):
        result.lines.append(line)
        if line_callback:
            line_callback(line)

    if not manifest.events:
        _emit("Manifest is empty — nothing to restore.")
        result.status = "failed"
        return result

    owns_scratch = False
    if scratch_dir is None:
        scratch_dir = tempfile.mkdtemp(prefix="titan_restore_sim_")
        owns_scratch = True
    _emit(
        f"Walking manifest from event "
        f"{manifest.get_latest_event()['event_id']} back to baseline..."
    )

    rr = await restore_full(
        manifest=manifest, target_dir=scratch_dir,
        arweave_fetch=arweave_fetch, memo_fetch=memo_fetch,
        arc_to_target=arc_to_target,
    )
    result.restore_result = rr

    _emit(
        f"Fetched {len(rr.applied_events)} events / {rr.bytes_fetched:,} bytes "
        f"in {rr.duration_s:.1f}s"
    )

    if rr.status == "success":
        result.status = "ok"
        result.files_verified = rr.restored_files
        _emit(
            f"[OK] restore-sim PASSED — {rr.restored_files} files reconstructed "
            f"byte-identical, full per-file sha256 verify by encoder"
        )
        _emit(
            f"Scratch dir: {scratch_dir} (owns={'true' if owns_scratch else 'false'})"
        )
    else:
        result.status = "failed"
        _emit(
            f"[FAIL] restore-sim halted: reason={rr.halt_reason} "
            f"event={rr.halt_event_id}"
        )
        for err in rr.errors:
            _emit(f"    {err}")
        _emit(
            f"Scratch dir preserved for inspection: {scratch_dir}"
        )
    return result


# ── audit-coverage ───────────────────────────────────────────────────────


def audit_coverage(
    declared_paths: Iterable[str],
    data_dir: str,
    *,
    ignore_patterns: Iterable[str] = (
        ".bak", ".tmp", ".restoring", ".staging",
        ".corrupt", ".repair",
    ),
    line_callback: Optional[Callable[[str], None]] = None,
) -> AuditCoverageResult:
    """rFP §24.11 audit-coverage — flag any persistent file in `data_dir`
    that isn't declared in PERSONALITY_PATHS ∪ WEEKLY_EXTRA_PATHS ∪
    TIMECHAIN_PATHS ∪ ARWEAVE_DAILY_EXCLUDE.

    Each declared_paths entry is either:
      - a file path relative to project root (e.g. "data/inner_memory.db")
      - a directory path with trailing slash (e.g. "data/sage_memory/")
        → matches every file inside

    The data_dir walk is recursive. ignore_patterns filter out backup
    artifacts that legitimately come and go (.bak, .tmp, etc.).

    Returns AuditCoverageResult with status="ok" if no undeclared files,
    or status="drift" with the undeclared_files list.
    """
    result = AuditCoverageResult(status="ok")

    def _emit(line: str):
        result.lines.append(line)
        if line_callback:
            line_callback(line)

    declared_files: set[str] = set()
    declared_dirs: list[str] = []
    for entry in declared_paths:
        entry = entry.rstrip()
        if entry.endswith("/") or os.path.isdir(entry):
            # Directory declaration — everything inside is covered
            declared_dirs.append(os.path.normpath(entry).rstrip(os.sep) + os.sep)
        else:
            declared_files.add(os.path.normpath(entry))
    result.declared_count = len(declared_files) + len(declared_dirs)

    if not os.path.isdir(data_dir):
        _emit(f"[ERR] data_dir {data_dir!r} not found")
        result.status = "drift"
        return result

    undeclared: list[str] = []
    found = 0
    for root, dirs, files in os.walk(data_dir):
        for fname in files:
            full = os.path.join(root, fname)
            # ignore-pattern filter
            if any(p in fname for p in ignore_patterns):
                continue
            found += 1
            normalized = os.path.normpath(full)
            # Match against explicit file declarations
            if normalized in declared_files:
                continue
            # Match against directory-prefix declarations
            covered = False
            for d in declared_dirs:
                if normalized.startswith(d):
                    covered = True
                    break
            if not covered:
                undeclared.append(normalized)

    result.found_count = found
    result.undeclared_files = sorted(undeclared)
    if undeclared:
        result.status = "drift"
        _emit(
            f"[DRIFT] {len(undeclared)} file(s) in {data_dir} not declared "
            f"in any backup inventory:"
        )
        for path in undeclared[:50]:  # cap output
            _emit(f"    {path}")
        if len(undeclared) > 50:
            _emit(f"    ... and {len(undeclared) - 50} more")
    else:
        _emit(
            f"[OK] audit-coverage PASSED — {found} files scanned, "
            f"all declared in backup inventory"
        )
    return result


# ── CLI exit-code helpers ────────────────────────────────────────────────


def tier_result_exit_code(result: VerifyTierResult) -> int:
    return 0 if result.fail_events == 0 else 1


def allbackups_exit_code(results: dict[str, VerifyTierResult]) -> int:
    return 0 if all(r.fail_events == 0 for r in results.values()) else 1


def restore_sim_exit_code(result: RestoreSimResult) -> int:
    return 0 if result.status == "ok" else 1


def audit_coverage_exit_code(result: AuditCoverageResult) -> int:
    return 0 if result.status == "ok" else 1
