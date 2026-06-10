"""
Backup Orchestrator — Guardian-supervised process (the `backup` module).

The SOURCE OF TRUTH for backup AND restore routines (RFP_backup_redesign_spine
Phase D). Owns the RebirthBackup instance + the BackupOrchestrator brain:
  • idle-driven per-file-batch DRIP — plan once → BackupWorker.build_slice per
    idle tick → finalize_pack — with a 3-LAYER readiness guarantee (INV-BRS-3/5):
      1. idle drip (primary)        — load1 < ncores·idle_load_factor AND ship-lock free
      2. deadline force (failsafe-1) — past staged_by_utc → force-complete off-path
      3. synchronous last-resort (failsafe-2) — meditation finds no stage → bounded
         inline build+ship via the SAME BackupWorker primitive (run_in_executor)
  • owns the ONE manifest-truth gate (_todays_backup_already_landed) + single-flight
  • DISK-PERSISTED drip — the partial StagedBuild survives a restart (INV-BRS-7);
    baseline-staleness + missing-artifact checks discard+re-plan when unsound
  • drives the Sunday restore-test under the single-flight lock (fixes the audit
    H-bug: the old stager restore-test was unguarded + on its own asyncio loop)
  • owns the SPEC §11.B.2 MODULE_ERROR cascade from ONE place

Replaces the old over-stuffed stager (modules/backup_worker.py → renamed here,
no-shim INV-BRS-9). The build/ship MECHANICS live in
backup_worker_pipeline.BackupWorker (Phase B); the crypto/manifest are KEPT
verbatim (INV-BRS-8). Promoted from TitanCore._backup_loop per rFP_backup_worker
Phase 1 (2026-04-20); bus-event handoff (was the trigger-file handoff).

Bus consumption:
  MEDITATION_COMPLETE   — primary ship trigger (was: data/backup_trigger.json file)
  BACKUP_TRIGGER_MANUAL — Maker-forced backup via /v4/backup/trigger
  MODULE_SHUTDOWN       — graceful exit (Guardian standard)

Bus emission:
  BACKUP_STARTED        — type, trigger, meditation_count, ts
  BACKUP_SUCCEEDED      — type, arweave_tx, size_mb, hash, duration_s
  BACKUP_FAILED         — type, step, error, duration_s
  BACKUP_HEALTH_ALERT   — severity, issue, diagnostic
  BACKUP_WORKER_READY   — titan_id, arweave_wired, boot_elapsed_s
  MODULE_ERROR          — SPEC §11.B.2 cascade (orchestrator-owned, one place)

See: titan-docs/RFP_backup_redesign_spine.md (§7.D) + AUDIT_backup_subsystem.md
"""

import asyncio
import json
import logging
import os
import sys
import threading
import time
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty
from typing import Optional
from titan_hcl.utils.silent_swallow import swallow_warn
from titan_hcl import bus
from titan_hcl.core.module_error_handler import with_error_envelope
from titan_hcl.errors import Severity as _phase11_sev

logger = logging.getLogger(__name__)


# Module-level heartbeat throttle (mirror memory_worker pattern, ≤1 per 3s)
_last_hb_ts: float = 0.0


# Phase 11 §11.I.5 (Chunk 11N) — module-level readiness sentinel; gates
# SHM-slot heartbeat() (legacy bus heartbeat fires unconditionally for
# the boot window so guardian_HCL's stale-heartbeat detector doesn't
# kill a slow boot — backup_worker boot can be 10-20s on cold dry-run).
from titan_hcl.modules._heartbeat_grace import (
    boot_deadline_from_now, shm_heartbeat_allowed,
)

_WORKER_READY: bool = False
_BOOT_DEADLINE = None  # boot-grace deadline (monotonic); None=no grace


@with_error_envelope(module_name="backup", subsystem="entry", severity=_phase11_sev.FATAL)
def backup_orchestrator_main(recv_queue, send_queue, name: str, config: dict) -> None:
    """Main loop for the Backup Orchestrator subprocess (the `backup` module).

    Boots RebirthBackup + the BackupOrchestrator brain (idle-drip + 3-layer
    readiness + the one gate + disk-persisted drip + restore-test), then runs
    the recv loop (meditation ship / manual trigger / SAVE_NOW / shutdown / probe
    / swap). RFP_backup_redesign_spine Phase D.

    Args:
        recv_queue: bus → worker
        send_queue: worker → bus
        name: Guardian module name ("backup")
        config: full config dict (reads [backup] + [backup.orchestrator])
    """
    global _WORKER_READY, _BOOT_DEADLINE
    _WORKER_READY = False
    _BOOT_DEADLINE = boot_deadline_from_now()

    project_root = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    logger.info("[BackupWorker] Initializing...")
    init_start = time.time()

    # ── Phase 11 §11.I.5 (Chunk 11N) — SHM state-slot writer ──
    # Constructed BEFORE the slow RebirthBackup init + dry-run cascade so
    # the slot publishes state="starting" immediately. Heartbeats keep
    # last_heartbeat fresh during the cold-boot window.
    _state_writer = None
    try:
        from titan_hcl.core.module_state import (
            BootPriority,
            ModuleStateWriter,
        )
        _state_writer = ModuleStateWriter(
            module_name=name,
            layer="L3",
            boot_priority=BootPriority.OPTIONAL_POST_BOOT,
        )
        _state_writer.write_state("starting")
    except Exception as _sw_err:  # noqa: BLE001
        logger.warning(
            "[BackupWorker] Phase 11 ModuleStateWriter init failed: %s", _sw_err)

    # Unwrap config — arrives as the full config dict from v5_core
    full_config = config or {}
    backup_cfg = full_config.get("backup", {}) or {}
    # BUG-BACKUP-WORKER-HARDCODED-TITAN-ID-T1: previous fallback to literal
    # "T1" caused T2/T3 to log themselves as T1 whenever info_banner.titan_id
    # was missing/empty in their config. resolve_titan_id() walks the canonical
    # precedence chain (explicit → data/titan_identity.json → TITAN_ID env →
    # "T1"), matching every other per-Titan resolver in the codebase.
    from titan_hcl.core.state_registry import resolve_titan_id
    explicit_id = (full_config.get("info_banner", {}) or {}).get("titan_id") or None
    titan_id = resolve_titan_id(explicit_id)

    # Per-Titan mode (rFP Phase 3). Default: infer from keypair + arweave_enabled +
    # network. 2026-05-14: added solana_network guard — devnet Titans must NEVER
    # enter mainnet_arweave mode regardless of `backup_arweave_enabled` flag,
    # because Arweave on devnet is just a local file:// cache (`arweave_devnet/`)
    # NOT immutable on-chain storage. Without this guard, devnet T2/T3 with
    # `backup_arweave_enabled=true` inherited from T1's config would attempt
    # the Arweave cascade, fail at S4 (no funded Irys deposit on devnet wallets),
    # and the TimeChain path exits without saving the local tarball — observed
    # as 2-week local-TimeChain silence on T2/T3 since 2026-04-29.
    net_cfg = full_config.get("network", {}) or {}
    solana_network = (net_cfg.get("solana_network") or "").strip().lower()
    is_mainnet = solana_network in ("mainnet", "mainnet-beta")
    mode = backup_cfg.get("mode", "").strip().lower()
    if not mode:
        kp_exists = bool(net_cfg.get("wallet_keypair_path", "")) and os.path.exists(
            net_cfg.get("wallet_keypair_path", "")
        )
        budget_enabled = full_config.get("mainnet_budget", {}).get(
            "backup_arweave_enabled", False
        )
        mode = "mainnet_arweave" if (is_mainnet and kp_exists and budget_enabled) else "local_only"
    elif mode == "mainnet_arweave" and not is_mainnet:
        # Explicit mode=mainnet_arweave but solana_network is devnet — override
        # to local_only with a loud warning. Same rationale as the auto-detect
        # branch above: Arweave on devnet is local-cache, not immutable.
        logger.warning(
            "[BackupWorker] mode=mainnet_arweave configured but solana_network=%s "
            "is not mainnet — forcing local_only. Arweave on devnet is local-cache "
            "only (file://data/arweave_devnet/) and provides no immutability.",
            solana_network or "(unset)"
        )
        mode = "local_only"
    logger.info("[BackupWorker] mode=%s titan_id=%s solana_network=%s",
                mode, titan_id, solana_network or "(unset)")

    # Local snapshot dir (rFP Phase 2 step 3 — always-save, even on failed upload)
    local_dir = Path(backup_cfg.get("local_dir", "data/backups"))
    local_dir.mkdir(parents=True, exist_ok=True)
    local_rolling_days = int(backup_cfg.get("local_rolling_days", 30))
    local_snapshot_always = bool(backup_cfg.get("local_snapshot_always", True))
    upload_verify = bool(backup_cfg.get("upload_verification_enabled", True))
    # tarball_validation_enabled / dry_run_on_boot config retired 2026-06-01
    # with the legacy boot dry-run (the stager validates the live pack).

    # Build ArweaveStore once at boot (rFP BUG-5)
    arweave_store = None
    keypair_path = net_cfg.get("wallet_keypair_path", "") or ""
    net_name = net_cfg.get("solana_network", "devnet")
    if net_name == "mainnet-beta":
        net_name = "mainnet"

    if mode == "mainnet_arweave":
        try:
            if keypair_path and os.path.exists(keypair_path):
                from titan_hcl.utils.arweave_store import ArweaveStore
                arweave_store = ArweaveStore(keypair_path=keypair_path, network=net_name)
                logger.info("[BackupWorker] ArweaveStore wired (network=%s)", net_name)
            else:
                logger.warning(
                    "[BackupWorker] mainnet_arweave mode but keypair missing (%s) — "
                    "falling back to local_only",
                    keypair_path,
                )
                mode = "local_only"
        except Exception as e:
            logger.warning("[BackupWorker] ArweaveStore init failed: %s — local_only", e)
            mode = "local_only"

    # ZK-Vault network client (SPEC §24.7 + D-SPEC-BACKUP-ZK-INPROC-CLIENT,
    # 2026-05-29). The backup subprocess ALREADY holds the Titan identity
    # keypair in-process (ArweaveStore signs every Irys upload with it), so an
    # in-process HybridNetworkClient for the event_merkle_root memo adds NO new
    # key exposure — unlike memory_worker, which keeps the *deployer* keypair in
    # the kernel and delegates via ANCHOR_REQUEST. Without this, network_client
    # stayed None → commit_event_merkle_to_zk_vault always returned None → the
    # unified_v2 pipeline uploaded tarballs then FAILED at the ZK commit on
    # EVERY event ("ZK Vault commit returned None") → no manifest ever
    # finalized + paid-but-orphaned uploads. Wired only in mainnet_arweave mode
    # (T1); local_only T2/T3 do not commit to mainnet. Violates no G18–G22 (a
    # direct Solana call is neither a bus.request, SHM write, nor sync-RPC).
    backup_network = None
    if mode == "mainnet_arweave":
        try:
            from titan_hcl.core.network import HybridNetworkClient
            backup_network = HybridNetworkClient(config=net_cfg)
            logger.info(
                "[BackupWorker] ZK-Vault network client wired (in-process, "
                "identity keypair; SPEC §24.7)")
        except Exception as e:
            logger.warning(
                "[BackupWorker] ZK-Vault network client init failed: %s — "
                "event_merkle_root commits will be skipped (events won't "
                "finalize)", e)
            backup_network = None

    # Build RebirthBackup (rFP BUG-2 correct signature + BUG-5 injection)
    try:
        from titan_hcl.logic.backup import RebirthBackup
        backup = RebirthBackup(
            network_client=backup_network,  # §24.7 in-process ZK-Vault client (mainnet_arweave only)
            config=full_config.get("memory_and_storage", {}),
            titan_id=titan_id,
            arweave_store=arweave_store,
            full_config=full_config,
        )
        logger.info(
            "[BackupWorker] RebirthBackup ready (titan_id=%s, arweave=%s, zk_network=%s)",
            titan_id, "wired" if arweave_store else "none",
            "wired" if backup_network else "none",
        )
    except Exception as e:
        logger.error("[BackupWorker] RebirthBackup init FAILED: %s", e, exc_info=True)
        sys.exit(1)

    # Async event loop for upload / anchor calls
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # rFP Phase 4 — boot catch-up check (RebirthBackup.check_on_boot already exists)
    with suppress(Exception):
        loop.run_until_complete(backup.check_on_boot())

    # 2026-05-29 — sweep leaked backup-snapshot hardlinks. Normally unlinked by
    # pack_event_tarball; an abnormal mid-pack termination (RSS-kill, crash)
    # leaks one into data/.bksnap_scratch. The exponential 340,445-orphan blowup
    # is fixed at the SOURCE (snapshots live out of the backed-up tree). 2026-06-10
    # (AUDIT_bksnap_legacy_orphans): the sweep now ALSO reaps the in-tree legacy
    # residue (its long-promised pass that the body never ran) + aged `.snap.`
    # SQLite images. Run OFF the boot critical path (daemon thread) — the in-tree
    # data/ walk + a large legacy reap must never stall boot / the heartbeat-grace.
    _sweep_root = (str(local_dir.parent) if local_dir.name == "backups"
                   else "data")

    def _boot_orphan_sweep():
        try:
            from titan_hcl.logic.diff_encoders.full_ship import (
                sweep_orphan_snapshots,
            )
            swept = sweep_orphan_snapshots(data_root=_sweep_root)
            if swept:
                logger.info(
                    "[BackupOrchestrator] Swept %d orphan backup snapshot(s) "
                    "(scratch + in-tree legacy)", swept)
        except Exception as _sw_e:  # noqa: BLE001
            logger.warning("[BackupOrchestrator] orphan sweep failed: %s", _sw_e)

    threading.Thread(target=_boot_orphan_sweep, name="backup-orphan-sweep",
                     daemon=True).start()

    boot_elapsed = time.time() - init_start
    logger.info("[BackupWorker] Ready in %.1fs (local_dir=%s, mode=%s)",
                boot_elapsed, local_dir, mode)

    # ── Phase 11 §11.I.2 — slot transition: starting → booted ──
    # (legacy boot-signal bus emit deleted per locked D2 / no-shim policy)
    _WORKER_READY = True
    if _state_writer is not None:
        try:
            _state_writer.write_state("booted")
        except Exception as _swb_err:  # noqa: BLE001
            logger.warning(
                "[BackupWorker] Phase 11 write_state(booted) failed: %s",
                _swb_err)

    # 2026-06-01 — the legacy rFP-I4 boot dry-run was REMOVED. It built a FULL
    # legacy tar.gz via create_personality_archive (gzip/zlib + Python-side tar
    # walks MONOPOLIZE the GIL) on a daemon thread every boot → starved the
    # heartbeat ~53s → shm_pid_dead → a wasteful transient self-restart on
    # every boot. Worse, it validated the RETIRED legacy archive path, not the
    # live unified_v2 pack. The Phase 2 stager (_maybe_build_stage, ~60s post-
    # boot) already validates the LIVE pack — it builds the real unified_v2
    # event with zstd (which RELEASES the GIL, so it does not starve the
    # heartbeat) and now records the result to backup_dry_run_result.json for
    # the dashboard health card. No pre-flight value lost; the every-boot waste
    # + heartbeat starvation are gone.
    _send(send_queue, "BACKUP_WORKER_READY", name, "all", {
        "titan_id": titan_id,
        "mode": mode,
        "arweave_wired": arweave_store is not None,
        "local_dir": str(local_dir),
        "boot_elapsed_s": round(boot_elapsed, 2),
    })
    # rFP I7 — initial telemetry anchor
    _write_i7_telemetry(titan_id, backup, mode, "worker_ready", {
        "arweave_wired": arweave_store is not None,
        "boot_elapsed_s": round(boot_elapsed, 2),
    })

    # Shared state for handlers
    state = {
        "backup": backup,
        "mode": mode,
        "local_dir": local_dir,
        "local_rolling_days": local_rolling_days,
        "local_snapshot_always": local_snapshot_always,
        "upload_verify": upload_verify,
        "keypair_path": keypair_path,
        "net_name": net_name,
        "titan_id": titan_id,
        "loop": loop,
        "send_queue": send_queue,
        "name": name,
        "full_config": full_config,  # Phase 9 — offhost mirror reads [backup.mirror]
        # Phase 1 (2026-05-31) single-flight guard: backup cascades run OFF the
        # recv loop (daemon thread) so the multi-minute diff/upload never starves
        # the bus socket → no BrokenPipe → no shm_pid_dead restart loop.
        "_backup_lock": threading.Lock(),
    }

    last_heartbeat = time.time()

    # ── Main loop ──────────────────────────────────────────────────────
    # ── Microkernel v2 Phase B.1 §6 — readiness/hibernate reporter ──
    from titan_hcl.core.readiness_reporter import trivial_reporter
    def _b1_save_state():
        return []
    _b1_reporter = trivial_reporter(
        worker_name=name, layer="L3", send_queue=send_queue,
        save_state_cb=_b1_save_state,
    )

    # RFP_backup_redesign_spine Phase D: the BackupOrchestrator brain replaces
    # the old one-shot stager — idle-driven per-file-batch drip + 3-layer
    # readiness guarantee (INV-BRS-3/5) + the one manifest-truth gate +
    # disk-persisted resume (INV-BRS-7) + the single-flight-guarded Sunday
    # restore-test. The build/ship mechanics stay in BackupWorker (Phase B).
    orchestrator = BackupOrchestrator(state, full_config, send_queue, name)
    state["orchestrator"] = orchestrator
    orchestrator.start()

    while True:
        try:
            msg = recv_queue.get(timeout=5.0)
        except Empty:
            if time.time() - last_heartbeat > 10.0:
                _send_heartbeat(send_queue, name, state_writer=_state_writer)
                last_heartbeat = time.time()
            # RFP_chain_provider Phase C tail — the recv-loop runway check is
            # DELETED (AUDIT worker bug C1: it ran a 30s `node …balance`
            # subprocess INSIDE `except Empty:`, the exact no-periodic-work-in-
            # except-Empty trap → could stall the recv loop 30s → BrokenPipe →
            # shm_pid_dead). Runway/funding now belongs to the ChainProvider
            # (`balance`/`fund`) driven off-loop; the orchestrator owns the alert.
            continue
        except (KeyboardInterrupt, SystemExit):
            break

        msg_type = msg.get("type", "")

        # ── Phase 11 §11.I.3 — MODULE_PROBE_REQUEST handler ──
        if msg_type == bus.MODULE_PROBE_REQUEST and _state_writer is not None:
            try:
                from titan_hcl.core.probe_dispatcher import (
                    handle_module_probe_request,
                )
                handle_module_probe_request(
                    msg,
                    probe_fn=None,
                    send_queue=send_queue,
                    module_name=name,
                    state_writer=_state_writer,
                )
            except Exception as _probe_err:  # noqa: BLE001
                logger.warning(
                    "[BackupWorker] MODULE_PROBE_REQUEST handler failed: %s",
                    _probe_err)
            continue

        # ── Microkernel v2 Phase B.1 §6 — shadow swap dispatch ────
        if _b1_reporter.handles(msg_type):
            _b1_reporter.handle(msg)
            if _b1_reporter.should_exit():
                break
            continue

        # ── Microkernel v2 Phase B.2.1 — supervision-transfer dispatch ──
        from titan_hcl.core import worker_swap_handler as _swap
        if _swap.maybe_dispatch_swap_msg(msg):
            continue

        if msg_type == bus.MODULE_SHUTDOWN:
            logger.info("[BackupOrchestrator] Shutdown: %s",
                        msg.get("payload", {}).get("reason"))
            # Stop the drip tick thread (the in-progress drip is already on disk
            # after every tick → a restart resumes it; INV-BRS-7).
            with suppress(Exception):
                orchestrator.stop()
            # §11.H.9(2) flush-on-MODULE_SHUTDOWN — durably persist the in-memory
            # tracking state (dedup dates + meditation counters) before exit.
            try:
                backup._save_backup_state()
            except Exception as _sd_err:  # noqa: BLE001
                logger.warning("[BackupOrchestrator] shutdown state flush failed: %s",
                               _sd_err)
            break

        # ── §11.H.9(1) SAVE_NOW → durable write + SAVE_DONE (D-SPEC-146 / INV-HRP-1) ──
        # backup was MISSED in the 2026-06-01 persistence rollout
        # (BUG-MODULE-PERSISTENCE-AUDIT-20260601): with no SAVE_NOW handler, the
        # Guardian's restart-time save_first waited the FULL 30s save_timeout for a
        # SAVE_DONE that never came — EVERY restart. Under restart-module that 30s
        # window raced the dual-guardian shm_pid_dead detection → a 5×/600s flap →
        # module DISABLED (the recurring backup instability, observed live 2026-06-09).
        # Reply SAVE_DONE right after the fast atomic state write so the orchestrator
        # proceeds immediately (no timeout) — backup's manifest/L5 are already atomic
        # on-disk; the only in-memory critical state is the tracking dict below.
        if msg_type == bus.SAVE_NOW:
            _save_rid = (msg.get("payload") or {}).get("request_id")
            _save_t0 = time.time()
            _save_ok, _save_errs = True, 0
            try:
                backup._save_backup_state()
            except Exception as _save_err:  # noqa: BLE001
                _save_ok, _save_errs = False, 1
                logger.warning("[BackupWorker] SAVE_NOW checkpoint failed: %s",
                               _save_err)
            try:
                _send(send_queue, bus.SAVE_DONE, name, "guardian", {
                    "module": name, "request_id": _save_rid,
                    "saved": _save_ok, "errors": _save_errs,
                    "duration_ms": int((time.time() - _save_t0) * 1000),
                })
            except Exception:  # noqa: BLE001
                pass
            continue

        try:
            if msg_type == bus.MEDITATION_COMPLETE:
                _send_heartbeat(send_queue, name, state_writer=_state_writer)
                _dispatch_backup_offloop(state, _handle_meditation, msg)
                last_heartbeat = time.time()
            elif msg_type == bus.BACKUP_TRIGGER_MANUAL:
                _send_heartbeat(send_queue, name, state_writer=_state_writer)
                _dispatch_backup_offloop(state, _handle_manual, msg)
                last_heartbeat = time.time()
            # Ignore other msg types — don't complain (bus delivers many)
        except Exception as e:
            logger.error("[BackupWorker] Handler error (%s): %s", msg_type, e,
                         exc_info=True)

    logger.info("[BackupWorker] Exiting")
    with suppress(Exception):
        loop.close()


# ── Off-loop dispatch (Phase 1, 2026-05-31) ───────────────────────────────

def _dispatch_backup_offloop(state: dict, handler, msg: dict) -> None:
    """Run a backup cascade handler OFF the recv-loop thread.

    The unified_v2 cascade reads the big mutable DBs (inner_memory.db ~1.1GB,
    experience DBs) to diff against the baseline — seconds-to-minutes of IO+CPU.
    Running it inline on the recv loop (the prior `_handle_*` call here) stopped
    the worker reading its bus socket → broker silent-hang-defense BrokenPipe →
    Guardian shm_pid_dead → restart loop → backup never completed → proof_day
    `post_generation_failed`. We spawn it in a daemon thread so the recv loop
    keeps reading + heartbeating throughout.

    Single-flight: a non-blocking lock; a 2nd trigger while one runs is skipped
    (the daily CAS gate inside the cascade already enforces one ship/day). The
    cascade reuses the worker's single asyncio loop (state['loop']) — only ever
    driven by ONE backup thread at a time — so RebirthBackup's lazily-created
    asyncio.Locks stay loop-coherent (a fresh loop per thread would trip the
    cross-loop Future hazard).
    """
    lock = state["_backup_lock"]
    if not lock.acquire(blocking=False):
        logger.info(
            "[BackupWorker] backup already in flight — skipping %s trigger "
            "(single-flight; daily CAS gate enforces one ship/day)",
            msg.get("type", "?"))
        return

    def _runner():
        try:
            asyncio.set_event_loop(state["loop"])
            handler(state, msg)
        except Exception as e:  # noqa: BLE001
            logger.error(
                "[BackupWorker] off-loop backup cascade failed: %s", e,
                exc_info=True)
        finally:
            lock.release()

    threading.Thread(
        target=_runner, name="backup-cascade-offloop", daemon=True).start()


# ── BackupOrchestrator — the brain (RFP_backup_redesign_spine Phase D) ─────

# [backup.orchestrator] config defaults (UTC; §7.D / Q-BRS-2/3 / D-BRS-D).
_DEF_BYTE_BUDGET_MB = 64        # per build_slice tick (per-file-batch drip)
_DEF_IDLE_LOAD_FACTOR = 0.75    # idle ⇔ load1 < ncores·factor (meditation-CPU proxy)
_DEF_STAGED_BY_UTC = "06:00"    # deadline-force wall-clock (before the earliest
                                # plausible 1st meditation — tune per fleet)
_DEF_DRIP_INTERVAL_S = 120.0    # tick cadence WHILE actively dripping
_DEF_POLL_INTERVAL_S = 1200.0   # slow poll when staged/complete (≡ old stager)
_DEF_BOOT_SETTLE_S = 60.0       # let boot settle before the first tick


class BackupOrchestrator:
    """The backup-module brain: idle-driven per-file-batch drip + a 3-layer
    readiness guarantee + the ONE manifest-truth gate + a disk-persisted drip +
    the single-flight-guarded Sunday restore-test. Drives BackupWorker (the build
    mechanics, Phase B) + RebirthBackup (plan/ship/restore wiring); owns NO
    crypto/manifest (INV-BRS-8). Replaces the old one-shot stager."""

    def __init__(self, state: dict, full_config: dict, send_queue,
                 name: str) -> None:
        self.state = state
        self.send_queue = send_queue
        self.name = name
        self.backup = state["backup"]
        self.titan_id = state.get("titan_id", "T1")
        oc = (((full_config or {}).get("backup", {}) or {})
              .get("orchestrator", {}) or {})
        self.byte_budget = (
            int(oc.get("byte_budget_mb", _DEF_BYTE_BUDGET_MB)) * 1024 * 1024)
        self.idle_load_factor = float(
            oc.get("idle_load_factor", _DEF_IDLE_LOAD_FACTOR))
        self.staged_by_utc = str(oc.get("staged_by_utc", _DEF_STAGED_BY_UTC))
        self.drip_interval_s = float(
            oc.get("drip_interval_s", _DEF_DRIP_INTERVAL_S))
        self.poll_interval_s = float(
            oc.get("poll_interval_s", _DEF_POLL_INTERVAL_S))
        self.boot_settle_s = float(oc.get("boot_settle_s", _DEF_BOOT_SETTLE_S))
        # Periodic side-state flush (Maker pattern: "save on timer continuously").
        # Belt-and-suspenders — halt/force-baseline/mirror/dry-run already flush
        # immediately on change; this catches meditation-count drift + guarantees
        # the consolidated readout (INV-BRS-7) is fresh for consumers.
        self._state_flush_interval_s = float(
            oc.get("state_flush_interval_s", 300.0))
        # The STABLE per-Titan drip scratch dir (a dot-prefixed sibling of the
        # baseline mirror, OUTSIDE the backed-up source set). Holds the partial
        # patch artifacts + drip_progress.json so a restart resumes (INV-BRS-7).
        self.drip_dir = os.path.join(
            "data", "backups", f".orch_drip_{self.titan_id}")
        # {worker, staged, resolver, known_arcs, today, weekday} | None
        self._drip: Optional[dict] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ── lifecycle ──────────────────────────────────────────────────────────
    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._tick_loop, name="backup-orchestrator", daemon=True)
        self._thread.start()
        logger.info(
            "[BackupOrchestrator] started — drip byte_budget=%d MB, "
            "idle<%.2f·ncores, staged_by=%s UTC, tick=%.0fs/poll=%.0fs "
            "(drip_dir=%s)",
            self.byte_budget // (1024 * 1024), self.idle_load_factor,
            self.staged_by_utc, self.drip_interval_s, self.poll_interval_s,
            self.drip_dir)

    def stop(self) -> None:
        self._stop.set()

    def _tick_loop(self) -> None:
        # Let boot + dependent modules settle before the first heavy tick.
        if self._stop.wait(self.boot_settle_s):
            return
        last_flush = 0.0
        while not self._stop.is_set():
            try:
                self._drip_tick()
            except Exception as e:  # noqa: BLE001
                logger.warning("[BackupOrchestrator] drip tick failed: %s",
                               e, exc_info=True)
                self._emit_module_error("drip", e)
            try:
                self.run_restore_test_if_due()
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "[BackupOrchestrator] restore-test cycle failed: %s",
                    e, exc_info=True)
                self._emit_module_error("restore_test", e)
            # Periodic side-state flush (INV-BRS-7 readout fresh; the per-change
            # flushes already cover halt/force-baseline/mirror/dry-run).
            now = time.monotonic()
            if now - last_flush >= self._state_flush_interval_s:
                last_flush = now
                try:
                    self.backup._save_backup_state()
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "[BackupOrchestrator] periodic state flush failed: %s", e)
            # Tick fast while actively dripping; slow poll when staged/complete.
            interval = (self.drip_interval_s if self._drip is not None
                        else self.poll_interval_s)
            self._stop.wait(interval)

    # ── idle / deadline signals (Q-BRS-3 / Q-BRS-2) ─────────────────────────
    def _is_idle(self) -> bool:
        """Idle ⇔ load1 < ncores·idle_load_factor (the meditation-CPU proxy —
        there is NO MEDITATION_START reaching backup, module_catalog broadcast
        filter) AND the ship single-flight lock is free (§7.D / Q-BRS-3)."""
        try:
            load1 = os.getloadavg()[0]
            ncores = os.cpu_count() or 1
        except (OSError, AttributeError):
            return True  # can't read load (best-effort) — allow the drip
        if load1 >= ncores * self.idle_load_factor:
            return False
        if self.state["_backup_lock"].locked():
            return False  # a ship/cascade already in flight (single-flight)
        return True

    def _past_staged_by_deadline(self) -> bool:
        """True once the UTC time-of-day has passed `staged_by_utc` (failsafe-1:
        force-complete the drip OFF the meditation path)."""
        try:
            hh, mm = (int(x) for x in self.staged_by_utc.split(":", 1))
        except (ValueError, AttributeError):
            hh, mm = 6, 0
        now = datetime.now(timezone.utc)
        return (now.hour, now.minute) >= (hh, mm)

    # ── the drip tick (the 3-layer readiness — INV-BRS-3/5) ─────────────────
    def _drip_tick(self) -> None:
        """One tick of the readiness machine. Layer-1 (idle drip) + Layer-2
        (deadline force) live here; Layer-3 (synchronous last-resort) is in
        RebirthBackup.on_meditation_complete (bounded inline build+ship)."""
        backup = self.backup
        # §24.12 / INV-BR-4 — a failed weekly restore-test halts scheduled backups.
        if backup._is_backups_halted():
            if self._drip is not None:
                self._discard_drip("backups halted (failed restore-test)")
            return
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        # Day rollover → drop a stale in-progress drip.
        if self._drip is not None and self._drip.get("today") != today:
            self._discard_drip("UTC day rollover")
        # A fresh stage is already parked for today → nothing to build.
        cur = getattr(backup, "_staged_event", None)
        if cur is not None and cur.get("date") == today:
            return
        # Today's backup already LANDED (manifest truth — the ONE gate) → no stage.
        if backup._todays_backup_already_landed():
            return
        # Ensure an active drip (resume the persisted one, else plan fresh).
        if self._drip is None:
            if not self._begin_or_resume_drip(today):
                return  # arweave gate off / manifest unloadable → retry next tick
        drip = self._drip
        if drip is None:
            return
        staged = drip["staged"]
        # LAYER 1+2: advance ONE bounded batch if idle (primary) OR past the
        # staged-by deadline (failsafe-1: drop the idle-gate). ≤1 batch/tick →
        # bounded RSS/CPU (INV-BRS-3).
        if not staged.fully_encoded:
            past_deadline = self._past_staged_by_deadline()
            idle = self._is_idle()
            if idle or past_deadline:
                drip["worker"].build_slice(
                    staged, drip["resolver"], byte_budget=self.byte_budget)
                self._persist_drip(drip)
                if past_deadline and not idle:
                    logger.info(
                        "[BackupOrchestrator] deadline-force drip batch (load "
                        "high, past staged_by=%s UTC) — off the meditation path",
                        self.staged_by_utc)
            return
        # Fully encoded → finalize (ONE streamed pack pass) + park for the ship.
        if not staged.tier_results:
            drip["worker"].finalize_pack(staged)
            self._persist_drip(drip)
        backup.stage_built_event(staged, today)
        _record_stage_dry_run_result(backup, staged)
        logger.info(
            "[BackupOrchestrator] daily event STAGED for %s — id=%s type=%s; "
            "awaiting meditation to ship", today, staged.event_id[:8],
            staged.event_type)
        # Parked; the persisted progress self-clears when the ship rmtree's scratch.
        self._drip = None

    # ── drip plan / resume / persist (INV-BRS-7 — disk-persisted drip) ──────
    def _drip_progress_path(self) -> str:
        from titan_hcl.logic.backup_worker_pipeline import (
            DRIP_PROGRESS_FILENAME)
        return os.path.join(self.drip_dir, DRIP_PROGRESS_FILENAME)

    def _begin_or_resume_drip(self, today: str) -> bool:
        if self._try_resume_drip(today):
            return True
        weekday = datetime.now(timezone.utc).weekday()
        planned = self.backup._plan_staged_build_v2(
            weekday, scratch_dir=self.drip_dir, byte_budget=self.byte_budget)
        if planned is None:
            return False  # arweave gate off / manifest unloadable
        worker, staged, resolver, known_arcs = planned
        self._drip = {"worker": worker, "staged": staged, "resolver": resolver,
                      "known_arcs": list(known_arcs), "today": today,
                      "weekday": weekday}
        self._persist_drip(self._drip)
        logger.info(
            "[BackupOrchestrator] drip PLANNED for %s — id=%s type=%s, "
            "%d files pending", today, staged.event_id[:8], staged.event_type,
            sum(len(v) for v in staged.pending.values()))
        return True

    def _try_resume_drip(self, today: str) -> bool:
        """Reload a disk-persisted drip after a restart (INV-BRS-7). Validates
        freshness — same day, baseline still current, all OWNED artifacts on disk
        — and DISCARDS+re-plans on any mismatch (the 3-layer readiness still
        guarantees the ship). A finalized-but-unshipped reload re-parks via the
        tick's finalize branch (no rebuild)."""
        p = self._drip_progress_path()
        if not os.path.exists(p):
            return False
        try:
            with open(p, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, ValueError):
            self._discard_drip("unreadable persisted drip")
            return False
        if payload.get("today") != today:
            self._discard_drip("persisted drip is for a stale day")
            return False
        try:
            from titan_hcl.logic.backup_worker_pipeline import (
                BackupWorker, StagedBuild)
            staged = StagedBuild.from_dict(payload["staged"])
        except Exception as e:  # noqa: BLE001
            logger.warning("[BackupOrchestrator] drip reload parse failed: %s "
                           "— re-planning", e)
            self._discard_drip("reload parse failed")
            return False
        # Freshness: the baseline the incrementals diff against must still be
        # current, and every OWNED patch artifact must still be on disk.
        try:
            from titan_hcl.logic.backup_unified_manifest import UnifiedManifest
            manifest = UnifiedManifest.load(
                titan_id=self.backup._titan_id, base_dir="data")
            cur_baseline = manifest.current_baseline_event_id
        except Exception:  # noqa: BLE001
            cur_baseline = None
        if staged.baseline_event_id != cur_baseline:
            self._discard_drip("baseline moved since the drip was persisted")
            return False
        # Freshness depends on the drip's PHASE. A FINALIZED drip (finalize_pack
        # ran → `tier_results` populated) has had its loose owned artifacts packed
        # INTO the per-component tarballs and unlinked by pack_event_tarball — so
        # `missing_artifacts()` (which checks those loose paths) would flag them ALL
        # missing and DISCARD a COMPLETED baseline. That is the convergence-killer:
        # a baseline that finishes building but is restarted before the meditation
        # ships it gets thrown away → re-planned as a fresh baseline → re-dripped
        # from scratch → never ships → manifest never gets a baseline → every plan
        # is a full baseline forever (never incremental) + the multi-GB working set
        # re-accumulates each cycle (the disk leak). For a finalized drip we must
        # instead validate the TARBALLS survived; the loose artifacts are GONE BY
        # DESIGN. (2026-06-10 — RFP_backup_redesign_spine convergence fix.)
        if staged.tier_results:
            missing_tar = [r.tarball_path for r in staged.tier_results.values()
                           if r.tarball_path and not os.path.exists(r.tarball_path)]
            if missing_tar:
                self._discard_drip(
                    f"{len(missing_tar)} staged tarball(s) missing on disk")
                return False
        else:
            missing = staged.missing_artifacts()
            if missing:
                self._discard_drip(
                    f"{len(missing)} drip artifact(s) missing on disk")
                return False
        known_arcs = set(payload.get("known_arcs") or [])
        resolver = self.backup._make_diff_base_resolver(
            self.backup._baseline_working_dir(), known_arcs)
        worker = BackupWorker(titan_id=self.backup._titan_id,
                              chain_provider=None, byte_budget=self.byte_budget)
        self._drip = {
            "worker": worker, "staged": staged, "resolver": resolver,
            "known_arcs": list(known_arcs), "today": today,
            "weekday": int(payload.get(
                "weekday", datetime.now(timezone.utc).weekday())),
        }
        logger.info(
            "[BackupOrchestrator] drip RESUMED for %s — id=%s, %d files still "
            "pending (restart-survived: baseline current, artifacts intact)",
            today, staged.event_id[:8],
            sum(len(v) for v in staged.pending.values()))
        return True

    def _persist_drip(self, drip: dict) -> None:
        """Atomically write the drip progress (INV-BRS-7) — after every
        build_slice + finalize, so a restart mid-drip resumes. Best-effort: a
        persist failure only forfeits restart-resume (the drip still completes
        in-process + the failsafes still guarantee the ship)."""
        try:
            os.makedirs(self.drip_dir, exist_ok=True)
            payload = {
                "staged": drip["staged"].to_dict(),
                "known_arcs": list(drip["known_arcs"]),
                "today": drip["today"],
                "weekday": drip["weekday"],
            }
            p = self._drip_progress_path()
            tmp = p + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            os.replace(tmp, p)
        except Exception as e:  # noqa: BLE001
            logger.warning("[BackupOrchestrator] drip persist failed: %s", e)

    def _discard_drip(self, reason: str) -> None:
        """Drop the in-progress drip + its scratch (re-plan next tick)."""
        import shutil
        logger.info(
            "[BackupOrchestrator] discarding drip — %s (re-plan next tick)",
            reason)
        drip = self._drip
        self._drip = None
        try:
            sd = (drip["staged"].scratch_dir if drip else self.drip_dir)
            if sd and os.path.isdir(sd):
                shutil.rmtree(sd, ignore_errors=True)
        except Exception:  # noqa: BLE001
            pass

    # ── Sunday restore-test (D.5 — single-flight-guarded; fixes audit H-bug) ──
    def run_restore_test_if_due(self) -> None:
        """§24.12 (Phase R4) — Sunday, once/day, the weekly FULL-chain
        restore-test. Now UNDER the single-flight lock (the old stager ran it
        UNGUARDED on its own asyncio loop — audit §1.5/H-bug) and emits its
        PASS/FAIL events on the REAL send_queue (the old `_bus_emit` probed
        state["bus"]/backup.bus/backup._bus — none exist → silent no-op, audit
        H-bug). Read-only (scratch dir). Warm-unification (routing through
        RestoreWorker.restore — a manifest→chain §24.12 behaviour change) stays a
        Maker-gated follow-up."""
        backup = self.backup
        if backup._is_backups_halted():
            return  # halted → wait for investigation; never auto-clear
        now = datetime.now(timezone.utc)
        if now.weekday() != 6:  # Sunday only (§24.12)
            return
        today = now.strftime("%Y-%m-%d")
        if getattr(backup, "_last_restore_test_date", None) == today:
            return
        lock = self.state["_backup_lock"]
        if not lock.acquire(blocking=False):
            return  # a ship/cascade in flight — retry next tick (single-flight)
        try:
            backup._last_restore_test_date = today

            def _bus_emit(evt, payload):
                try:
                    _send(self.send_queue, evt, self.name, "all", payload)
                except Exception:
                    pass

            import asyncio as _aio
            _aio.run(backup._run_weekly_restore_test(bus_emit=_bus_emit))
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "[BackupOrchestrator] §24.12 restore-test runner failed: %s",
                e, exc_info=True)
            self._emit_module_error("restore_test", e)
        finally:
            lock.release()

    # ── SPEC §11.B.2 MODULE_ERROR cascade (D.4 — ONE place) ─────────────────
    def _emit_module_error(self, subsystem: str, exc: Exception) -> None:
        """The orchestrator-owned MODULE_ERROR emission (SPEC §11.B.2), the ONE
        place backup's structured errors surface (replacing the scattered
        swallow_warn — audit §1.6). Best-effort double-fault guard — never
        raises into the tick loop."""
        try:
            from titan_hcl.errors import (
                ModuleError, ModuleErrorCode, Severity)
            from titan_hcl.bus import publish_module_error
            err = ModuleError.from_exception(
                exc, module_name=self.name, subsystem=subsystem,
                error_code=ModuleErrorCode.STORAGE_DEGRADED,
                severity=Severity.ERROR)
            publish_module_error(self.send_queue, err)
        except Exception:  # noqa: BLE001
            pass


# ── Meditation-triggered cascade (Phase 2 §5.3) ───────────────────────────

def _handle_meditation(state: dict, msg: dict) -> None:
    """Handle MEDITATION_COMPLETE — run personality/soul/timechain cascade.

    This implements rFP §5.3 — the 10-step failsafe cascade. Delegates to
    RebirthBackup.on_meditation_complete for the dedup + scheduling decisions
    (daily personality / weekly soul / TimeChain) but wraps the whole call
    with per-type cascade observability.

    rFP_meditation_worker_latency Fix #B1 (2026-05-07): no longer gate on
    `payload["success"]`. The success flag historically reflected whether
    plugin.bus.request returned in time, NOT whether memory ops landed. With
    Fix #A eliminating the asyncio deadlock, success is reliable — but as
    defense in depth we run the cascade whenever promoted+pruned signal real
    memory work happened. This closes the 26-day Arweave outage on T1
    mainnet (last upload May 4, prior cutoff Apr 29) caused by every cycle
    timing out at 120s/300s and emitting success=False even though the
    worker had completed all node migrations + FAISS/Kuzu writes.
    """
    payload = msg.get("payload", {}) or {}
    promoted = int(payload.get("promoted", 0) or 0)
    pruned = int(payload.get("pruned", 0) or 0)
    if not payload.get("success", False) and promoted == 0 and pruned == 0:
        logger.debug(
            "[BackupWorker] Meditation produced no memory work "
            "(success=%s, promoted=0, pruned=0) — skipping",
            payload.get("success", False))
        return

    backup = state["backup"]
    loop = state["loop"]
    send_queue = state["send_queue"]
    name = state["name"]
    titan_id = state["titan_id"]

    med_count = payload.get("meditation_count", backup._meditation_count + 1)
    logger.info("[BackupWorker] Processing meditation #%d", med_count)

    _send(send_queue, "BACKUP_STARTED", name, "all", {
        "trigger": "meditation", "meditation_count": med_count,
        "ts": time.time(),
    })

    t0 = time.time()
    try:
        # RebirthBackup runs: ZK epoch snapshot, daily personality,
        # weekly soul, TimeChain, MyDay NFT (per its own cadence logic).
        # The Phase 2 cascade is applied INSIDE the upload paths by the
        # RebirthBackup class — our wrapper handles errors + emission here.
        loop.run_until_complete(backup.on_meditation_complete(payload))
        dur = time.time() - t0
        _send(send_queue, "BACKUP_SUCCEEDED", name, "all", {
            "trigger": "meditation", "meditation_count": med_count,
            "duration_s": round(dur, 2),
            "mode": state["mode"],
        })
        logger.info("[BackupWorker] Backup pass complete in %.1fs", dur)
        # rFP Phase 6 — Maker Telegram alert (at-most-daily via dedup)
        _notify_maker_success(titan_id, backup, dur, state["mode"])
        # rFP I7 — update shared telemetry
        _write_i7_telemetry(titan_id, backup, state["mode"], "backup_succeeded",
                            {"meditation_count": med_count, "duration_s": round(dur, 2)})
        # rFP Phase 9 — off-host mirror (T1 only; no-op when disabled or on T2/T3)
        _run_offhost_mirror(state, med_count)
        # L3+L4 (2026-05-14): monthly-archive consolidation + retention cap.
        # Replaces the T1-only cron `scripts/personality_backup_archive.sh` —
        # now runs on all 3 Titans, after every successful backup, atomic with
        # the backup event itself. Retires divergence between T1 (cron) and
        # T2/T3 (no cron). Indefinite-retention bug also closed by the
        # archive_retention_months cap (default 12 months).
        _run_monthly_archive_consolidation(state)
    except Exception as e:
        dur = time.time() - t0
        logger.error("[BackupWorker] Backup pass failed: %s", e, exc_info=True)
        _send(send_queue, "BACKUP_FAILED", name, "all", {
            "trigger": "meditation", "meditation_count": med_count,
            "error": str(e), "duration_s": round(dur, 2),
            "mode": state["mode"],
        })
        # Phase 6 — critical failure: force-send (bypass dedup)
        with suppress(Exception):
            from titan_hcl.utils.maker_notify import (
                notify_maker, format_backup_failure)
            notify_maker("backup_failure", titan_id,
                         format_backup_failure("backup", str(e)),
                         cooldown_s=3600, force=False)
        _write_i7_telemetry(titan_id, backup, state["mode"], "backup_failed",
                            {"meditation_count": med_count, "error": str(e)[:200]})


def _run_offhost_mirror(state: dict, med_count: int) -> None:
    """rFP_backup_worker Phase 9 — pull T2/T3 local snapshots to T1.

    No-op when [backup.mirror].enabled=false or when no hosts configured
    (which is the expected state on T2/T3 themselves). Emits OFFHOST_MIRROR_COMPLETE
    or OFFHOST_MIRROR_FAILED bus events. Runs retention cleanup after each pull.

    Fire-and-forget in spirit: failures here do NOT fail the backup — they're
    a best-effort mirror. Logged + alerted, not raised.
    """
    try:
        from titan_hcl.logic.offhost_mirror import OffhostMirror
        full_config = state.get("full_config", {}) or {}
        mirror = OffhostMirror(full_config)
        if not mirror.enabled or not mirror.hosts:
            return

        t0 = time.time()
        try:
            result = state["loop"].run_until_complete(mirror.pull_all())
        except Exception as e:
            logger.error("[BackupWorker] Offhost mirror crashed: %s",
                         e, exc_info=True)
            _send(state["send_queue"], "OFFHOST_MIRROR_FAILED",
                  state["name"], "all",
                  {"error": str(e), "duration_s": round(time.time() - t0, 2)})
            return

        removed = {}
        try:
            removed = mirror.cleanup_all()
        except Exception as e:
            logger.warning("[BackupWorker] Mirror cleanup error: %s", e)

        dur = round(time.time() - t0, 2)
        if result.get("ok"):
            logger.info(
                "[BackupWorker] Offhost mirror OK in %.1fs — %d host(s), cleanup %s",
                dur, len(result.get("results", [])), removed)
            _send(state["send_queue"], "OFFHOST_MIRROR_COMPLETE",
                  state["name"], "all",
                  {"duration_s": dur, "hosts": result.get("results", []),
                   "cleanup": removed, "meditation_count": med_count})
            _write_i7_telemetry(state["titan_id"], state["backup"], state["mode"],
                                "offhost_mirror_complete",
                                {"duration_s": dur,
                                 "hosts": [r.get("titan_id") for r in result.get("results", [])
                                            if r.get("ok")],
                                 "meditation_count": med_count})
        else:
            errors = [{"titan_id": r.get("titan_id"), "error": r.get("error")}
                      for r in result.get("results", []) if not r.get("ok")]
            logger.warning("[BackupWorker] Offhost mirror partial fail in %.1fs: %s",
                           dur, errors)
            _send(state["send_queue"], "OFFHOST_MIRROR_FAILED",
                  state["name"], "all",
                  {"duration_s": dur, "errors": errors,
                   "meditation_count": med_count})
            _write_i7_telemetry(state["titan_id"], state["backup"], state["mode"],
                                "offhost_mirror_failed",
                                {"duration_s": dur, "errors": errors,
                                 "meditation_count": med_count})
    except Exception as e:
        logger.error("[BackupWorker] _run_offhost_mirror outer error: %s",
                     e, exc_info=True)


def _run_monthly_archive_consolidation(state: dict) -> None:
    """L3+L4 (2026-05-14) — Python-side monthly archive + retention cap.

    Replaces the T1-only cron `scripts/personality_backup_archive.sh` with
    in-Python logic that runs on ALL Titans after each successful backup.
    Reads cadence + retention from [backup] config:
      monthly_archive_loose_days        (default 7)   — files older than this get archived
      monthly_archive_retention_months  (default 12)  — archives older than this get pruned

    Fire-and-forget: failures here do NOT fail the backup pass.
    """
    try:
        full_config = state.get("full_config", {}) or {}
        backup_cfg = (full_config.get("backup", {}) or {})
        loose_days = int(backup_cfg.get("monthly_archive_loose_days", 7))
        retention_months = int(backup_cfg.get(
            "monthly_archive_retention_months", 12))
        local_dir = backup_cfg.get("local_dir", "data/backups")
        backup = state.get("backup")
        if backup is None:
            return
        result = backup.consolidate_monthly_archives(
            local_dir=local_dir,
            loose_retention_days=loose_days,
            archive_retention_months=retention_months,
        )
        if result.get("archived_count") or result.get("pruned_archives_count"):
            logger.info(
                "[BackupWorker] Monthly archive: archived=%d (%d loose deleted), "
                "pruned=%d (>%dmo)",
                result["archived_count"], result["deleted_loose_count"],
                result["pruned_archives_count"], retention_months)
            _send(state["send_queue"], "BACKUP_ARCHIVE_MAINTENANCE",
                  state["name"], "all", result)
        if result.get("errors"):
            logger.warning(
                "[BackupWorker] Monthly archive: %d error(s) — %s",
                len(result["errors"]), result["errors"][:3])
    except Exception as e:
        logger.error("[BackupWorker] _run_monthly_archive_consolidation error: %s",
                     e, exc_info=True)


def _write_i7_telemetry(titan_id: str, backup, mode: str, event: str,
                          event_payload: Optional[dict] = None) -> None:
    """rFP I7 — shared telemetry schema at data/telemetry/sovereign_ops_{titan_id}_{date}.json.

    Single source of truth that arch_map sovereign-ops + Maker dashboards read.
    Updated on: BACKUP_WORKER_READY, BACKUP_SUCCEEDED, BACKUP_FAILED, BACKUP_HEALTH_ALERT.
    """
    try:
        from datetime import datetime, timezone
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = f"data/telemetry/sovereign_ops_{titan_id}_{date_str}.json"
        os.makedirs("data/telemetry", exist_ok=True)

        # Load existing or init
        state = {}
        if os.path.exists(path):
            try:
                with open(path) as f:
                    state = json.load(f)
            except Exception:
                state = {}

        state.setdefault("titan_id", titan_id)
        state.setdefault("date", date_str)
        state.setdefault("mode", mode)
        state["updated_at"] = time.time()
        state.setdefault("events", [])

        # Bounded event log (keep last 200)
        state["events"].append({
            "ts": time.time(),
            "event": event,
            "payload": event_payload or {},
        })
        state["events"] = state["events"][-200:]

        # Update backup section
        bk = state.setdefault("backup", {})
        for btype in ("personality", "soul_package"):
            rec = backup.get_latest_backup_record(btype)
            if rec and rec.get("uploaded_at"):
                bk[f"last_{btype}_ts"] = rec.get("uploaded_at")
                bk[f"last_{btype}_tx"] = rec.get("arweave_tx")
                bk[f"last_{btype}_size_mb"] = rec.get("size_mb")

        # Atomic write
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, path)
    except Exception as e:
        swallow_warn('[BackupWorker] I7 telemetry write failed', e,
                     key="modules.backup_worker.i7_telemetry_write_failed", throttle=100)


def _notify_maker_success(titan_id: str, backup, duration_s: float, mode: str) -> None:
    """Phase 6 — emit Telegram success alert for completed cascade results.

    Pulls from the backup records written during on_meditation_complete. Only
    fires if a new upload happened (not a dedup skip). At-most-daily per type.
    """
    with suppress(Exception):
        from titan_hcl.utils.maker_notify import (
            notify_maker, format_backup_success)
        # Check both personality + soul for any successful upload this cycle
        now = time.time()
        for btype in ("personality", "soul_package"):
            rec = backup.get_latest_backup_record(btype)
            if not rec:
                continue
            up_at = float(rec.get("uploaded_at", 0))
            if now - up_at > 600:  # >10 min stale — not from this cycle
                continue
            tx = rec.get("arweave_tx")
            size_mb = rec.get("size_mb", 0)
            alert_class = ("backup_success_daily" if btype == "personality"
                           else "backup_success_weekly")
            cooldown = 86400 if btype == "personality" else 7 * 86400
            notify_maker(alert_class, titan_id,
                         format_backup_success(btype.replace("_package", ""),
                                                 size_mb, tx, duration_s),
                         cooldown_s=cooldown)


def _handle_manual(state: dict, msg: dict) -> None:
    """Handle Maker-forced backup via /v4/backup/trigger.

    Payload: {type: "personality"|"soul"|"timechain", rid: <request_id>}
    """
    payload = msg.get("payload", {}) or {}
    rid = msg.get("rid")
    src = msg.get("src", "api")
    backup_type = payload.get("type", "personality").lower()

    backup = state["backup"]
    loop = state["loop"]
    send_queue = state["send_queue"]
    name = state["name"]

    logger.info("[BackupWorker] Manual trigger type=%s src=%s", backup_type, src)

    _send(send_queue, "BACKUP_STARTED", name, "all", {
        "trigger": "manual", "type": backup_type, "ts": time.time(),
    })

    t0 = time.time()
    try:
        # Maker policy 2026-05-23 (D-SPEC-123 follow-up): when unified_v2 owns
        # backups, a Maker-forced trigger runs ONE unified diff/baseline event
        # (atomic personality+timechain, +soul on Sunday) — NOT the legacy
        # per-type full-tarball upload (~50MB), which is the SOL-drain path now
        # retired everywhere unified_v2 is enabled. The `type` param is
        # informational under unified_v2 (events are atomic, not per-type).
        if backup._unified_v2_enabled():
            weekday = datetime.now(timezone.utc).weekday()
            shipped = loop.run_until_complete(
                backup._run_unified_event_v2(weekday=weekday))
            dur = time.time() - t0
            _send(send_queue, "BACKUP_SUCCEEDED", name, "all", {
                "trigger": "manual", "type": "unified_v2",
                "requested_type": backup_type, "shipped": bool(shipped),
                "duration_s": round(dur, 2),
            })
            logger.info(
                "[BackupWorker] Manual unified_v2 event: shipped=%s in %.1fs "
                "(requested type=%s ignored — unified events are atomic)",
                shipped, dur, backup_type)
            if rid:
                _send(send_queue, "RESPONSE", name, src, {
                    "ok": True,
                    "result": {"unified_v2": True, "shipped": bool(shipped),
                               "note": ("event shipped" if shipped else
                                        "clean no-op: nothing changed or arweave gate off")},
                }, rid=rid)
            return
        # Legacy per-type full-tarball Arweave upload (the SOL-drain) is RETIRED
        # (RFP_backup_redesign_spine Phase B / B-1, no-shim). unified_v2 is
        # fleet-wide (the branch above handles a Maker-forced trigger as ONE
        # atomic unified event + returns); a non-unified_v2 install must ENABLE
        # it, never fall back to the per-type ~50 MB drain. The except below
        # turns this into a clean BACKUP_FAILED + RESPONSE.
        raise RuntimeError(
            "manual backup requires [backup].unified_v2_enabled=true — the "
            "legacy per-type Arweave upload is retired (RFP_backup_redesign_spine B-1)")
    except Exception as e:
        dur = time.time() - t0
        logger.error("[BackupWorker] Manual trigger failed: %s", e, exc_info=True)
        _send(send_queue, "BACKUP_FAILED", name, "all", {
            "trigger": "manual", "type": backup_type,
            "error": str(e), "duration_s": round(dur, 2),
        })
        if rid:
            _send(send_queue, "RESPONSE", name, src,
                  {"ok": False, "error": str(e)}, rid=rid)


# ── Phase 2 / I4 helpers ─────────────────────────────────────────────────

def _record_stage_dry_run_result(backup, staged) -> None:
    """Record a freshly-staged unified_v2 event's per-tier pack sizes for the
    dashboard backup-health card — the LIVE-path replacement for the retired
    legacy boot dry-run (a real unified_v2 build: zstd, GIL-friendly).

    Phase D (INV-BRS-7): writes into the CONSOLIDATED orchestrator side-state
    (`backup._last_dry_run`, persisted in _BACKUP_STATE_PATH) instead of the
    standalone data/backup_dry_run_result.json. Best-effort; never raises into
    the drip loop."""
    try:
        steps = {}
        total = 0
        for tier, r in (staged.tier_results or {}).items():
            sz = int(getattr(r, "tarball_size_bytes", 0) or 0)
            total += sz
            steps[tier] = f"OK ({sz/1024/1024:.1f} MB)"
        backup._last_dry_run = {
            "ok": True,
            "ts": time.time(),
            "source": "unified_v2_drip",
            "event_type": getattr(staged, "event_type", "?"),
            "steps": steps,
            "total_mb": round(total / 1024 / 1024, 1),
        }
        backup._save_backup_state()
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "[BackupOrchestrator] stage dry-run result record failed: %s", e)


# ── _check_runway + _node_path DELETED (RFP_backup_redesign_spine Phase D) ──
#    Dead since the RFP_chain_provider Phase C tail removed the recv-loop call
#    (the `node …balance` subprocess inside `except Empty:` — audit worker bug
#    C1). Zero callers remained (verified). Runway/funding now belongs to the
#    ChainProvider (`balance`/`fund`, in-process) driven off-loop; the
#    orchestrator owns the alert (no-shim, INV-BRS-9).


# ── Bus helpers (mirror memory_worker pattern) ────────────────────────────

def _send(send_queue, msg_type: str, src: str, dst: str, payload: dict,
          rid: Optional[str] = None) -> None:
    try:
        send_queue.put_nowait({
            "type": msg_type,
            "src": src,
            "dst": dst,
            "ts": time.time(),
            "rid": rid,
            "payload": payload,
        })
    except Exception:
        with suppress(Exception):
            from titan_hcl.bus import record_send_drop
            record_send_drop(src, dst, msg_type)


def _send_heartbeat(send_queue, name: str,
                    state_writer: Optional[object] = None) -> None:
    """Heartbeat to Guardian (≤1 per 3s, matches memory_worker).

    Phase 11 §11.I.5: also publishes state_writer.heartbeat() on the SHM
    slot once _WORKER_READY is True. SHM writes are best-effort.
    """
    global _last_hb_ts
    now = time.time()
    if now - _last_hb_ts < 3.0:
        return
    _last_hb_ts = now
    try:
        import psutil
        rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        rss_mb = 0
    _send(send_queue, "MODULE_HEARTBEAT", name, "guardian",
          {"rss_mb": round(rss_mb, 1)})
    if state_writer is not None and shm_heartbeat_allowed(_WORKER_READY, _BOOT_DEADLINE):
        try:
            state_writer.heartbeat()
        except Exception:  # noqa: BLE001 — never crash the heartbeat
            pass
