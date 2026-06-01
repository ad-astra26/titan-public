"""
Backup Worker — Guardian-supervised process.

Owns the RebirthBackup instance (daily personality + weekly soul + TimeChain +
ZK epoch + MyDay NFT). Promoted from TitanCore._backup_loop per rFP_backup_worker
Phase 1 (2026-04-20). Replaces the trigger-file handoff with bus-event handoff.

Bus consumption:
  MEDITATION_COMPLETE   — primary trigger (was: data/backup_trigger.json file)
  BACKUP_TRIGGER_MANUAL — Maker-forced backup via /v4/backup/trigger
  MODULE_SHUTDOWN       — graceful exit (Guardian standard)

Bus emission:
  BACKUP_STARTED        — type, trigger, meditation_count, ts
  BACKUP_SUCCEEDED      — type, arweave_tx, size_mb, hash, duration_s
  BACKUP_FAILED         — type, step, error, duration_s
  BACKUP_HEALTH_ALERT   — severity, issue, diagnostic
  BACKUP_DIFF_ALERT     — (Phase I5) size_delta, missing_aux, etc.
  BACKUP_WORKER_READY   — titan_id, arweave_wired, boot_elapsed_s

Phase 2 failsafe cascade (rFP §5.3) wraps each upload in 10 steps:
  1 build → 2 validate → 3 local-ALWAYS → 4 balance → 5 upload →
  6 verify → 7 manifest → 8 anchor → 9 emit → 10 cleanup

See: titan-docs/rFP_backup_worker.md
"""

import asyncio
import json
import logging
import os
import subprocess
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
_WORKER_READY: bool = False


@with_error_envelope(module_name="backup", subsystem="entry", severity=_phase11_sev.FATAL)
def backup_worker_main(recv_queue, send_queue, name: str, config: dict) -> None:
    """Main loop for the Backup Worker subprocess.

    Args:
        recv_queue: bus → worker
        send_queue: worker → bus
        name: Guardian module name ("backup")
        config: full config dict (rFP Phase 3 reads [backup] section from here)
    """
    global _WORKER_READY
    _WORKER_READY = False

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

    # 2026-05-29 — sweep leaked backup-snapshot hardlinks. A snapshot is
    # normally unlinked by pack_event_tarball, but an abnormal mid-pack
    # termination (RSS-kill, crash) leaks one into data/.bksnap_scratch.
    # Bounded best-effort sweep at boot keeps the orphan set from growing.
    # (The exponential blowup that produced 340,445 orphans is fixed at the
    # source — snapshots now live out of the backed-up tree — this is the
    # belt-and-suspenders hygiene pass.)
    with suppress(Exception):
        from titan_hcl.logic.diff_encoders.full_ship import (
            sweep_orphan_snapshots,
        )
        swept = sweep_orphan_snapshots(
            data_root=str(local_dir.parent) if local_dir.name == "backups"
            else "data")
        if swept:
            logger.info("[BackupWorker] Swept %d orphan backup snapshot(s)",
                        swept)

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
        "tarball_validate": tarball_validate,
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
    last_runway_check = 0.0
    runway_check_interval = 3600.0  # once per hour (rFP I6 is separate daily cron; this is in-loop light check)

    # ── Main loop ──────────────────────────────────────────────────────
    # ── Microkernel v2 Phase B.1 §6 — readiness/hibernate reporter ──
    from titan_hcl.core.readiness_reporter import trivial_reporter
    def _b1_save_state():
        return []
    _b1_reporter = trivial_reporter(
        worker_name=name, layer="L3", send_queue=send_queue,
        save_state_cb=_b1_save_state,
    )

    # Phase 2 (2026-05-31): pre-stage the daily event off the recv loop so the
    # meditation ship is fast + never blocks the bus (the diff-build that used to
    # crash the worker now happens here, ahead of time).
    _start_stager(state)

    while True:
        try:
            msg = recv_queue.get(timeout=5.0)
        except Empty:
            if time.time() - last_heartbeat > 10.0:
                _send_heartbeat(send_queue, name, state_writer=_state_writer)
                last_heartbeat = time.time()
            # Periodic in-loop runway check (I6 also writes daily telemetry via cron)
            if (mode == "mainnet_arweave"
                    and time.time() - last_runway_check > runway_check_interval):
                with suppress(Exception):
                    _check_runway(state)
                last_runway_check = time.time()
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
            logger.info("[BackupWorker] Shutdown: %s",
                        msg.get("payload", {}).get("reason"))
            break

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


# ── Phase 2 pre-stage daemon (2026-05-31) ─────────────────────────────────

def _maybe_build_stage(state: dict) -> None:
    """Ensure a fresh pre-built event exists for today (build it if not).

    The heavy diff/pack runs HERE, on the stager's daemon thread, ahead of the
    first meditation — so the meditation ship is fast + never blocks the recv
    loop. The existing full_ship.encode_diff already race-safe-snapshots the live
    DBs, so reading them mid-write is safe. Gate-aware: _build_staged_event_v2
    returns None when backup_arweave is disabled.
    """
    backup = state["backup"]
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cur = getattr(backup, "_staged_event", None)
    if cur is not None and cur.get("date") == today:
        return  # already staged for today
    # Don't pile a heavy build on top of a meditation cascade already running
    # (single-flight with the off-loop ship/inline-build).
    if state["_backup_lock"].locked():
        logger.debug("[BackupWorker] stager: cascade in flight — deferring build")
        return
    weekday = datetime.now(timezone.utc).weekday()
    staged = backup._build_staged_event_v2(weekday)
    if staged is not None:
        backup.stage_built_event(staged, today)
        # LIVE-path validation record (replaces the retired legacy boot
        # dry-run) — feeds the dashboard backup health card.
        _record_stage_dry_run_result(staged)


def _start_stager(state: dict) -> None:
    """Start the Phase 2 background stager thread."""
    poll_s = 1200.0  # 20 min — cheap no-op when a fresh stage exists; catches
                     # the UTC day rollover well before the first meditation.

    def _loop():
        time.sleep(60.0)   # let boot + boot-dry-run settle first
        while True:
            try:
                _maybe_build_stage(state)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "[BackupWorker] stager cycle failed: %s", e, exc_info=True)
            time.sleep(poll_s)

    threading.Thread(target=_loop, name="backup-stager", daemon=True).start()
    logger.info(
        "[BackupWorker] Phase 2 stager started (poll=%.0fs) — pre-builds the "
        "daily event off-loop so meditation ships fast", poll_s)


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
        if backup_type == "personality":
            result = loop.run_until_complete(backup.upload_personality_to_arweave())
        elif backup_type == "soul":
            result = loop.run_until_complete(backup.upload_soul_package_to_arweave())
        elif backup_type == "timechain":
            # TimeChain uses its own path; requires TimeChainBackup wrapper
            from titan_hcl.logic.timechain_backup import TimeChainBackup
            tcb = TimeChainBackup(
                data_dir="data/timechain",
                titan_id=state["titan_id"],
                arweave_store=backup._arweave_store,
            )
            tx_id = loop.run_until_complete(tcb.snapshot_to_arweave())
            result = {"arweave_tx": tx_id} if tx_id else None
        else:
            raise ValueError(f"Unknown backup type: {backup_type}")

        dur = time.time() - t0
        if result:
            _send(send_queue, "BACKUP_SUCCEEDED", name, "all", {
                "trigger": "manual", "type": backup_type,
                "arweave_tx": result.get("arweave_tx"),
                "size_mb": result.get("size_mb"),
                "archive_hash": (result.get("archive_hash") or "")[:16],
                "duration_s": round(dur, 2),
            })
            if rid:
                _send(send_queue, "RESPONSE", name, src,
                      {"ok": True, "result": result}, rid=rid)
        else:
            err = "upload returned None"
            _send(send_queue, "BACKUP_FAILED", name, "all", {
                "trigger": "manual", "type": backup_type, "error": err,
                "duration_s": round(dur, 2),
            })
            if rid:
                _send(send_queue, "RESPONSE", name, src, {"ok": False, "error": err}, rid=rid)
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

def _record_stage_dry_run_result(staged) -> None:
    """Write data/backup_dry_run_result.json from a freshly-built unified_v2
    staged event — the LIVE-path replacement for the retired legacy boot
    dry-run. Records per-tier pack sizes so the dashboard's backup health card
    reflects a real unified_v2 build (zstd, GIL-friendly), not a redundant
    legacy gzip archive. Best-effort; never raises into the stager loop.
    """
    try:
        steps = {}
        total = 0
        for tier, r in (staged.tier_results or {}).items():
            sz = int(getattr(r, "tarball_size_bytes", 0) or 0)
            total += sz
            steps[tier] = f"OK ({sz/1024/1024:.1f} MB)"
        status = {
            "ok": True,
            "ts": time.time(),
            "source": "unified_v2_stager",
            "event_type": getattr(staged, "event_type", "?"),
            "steps": steps,
            "total_mb": round(total / 1024 / 1024, 1),
        }
        with open("data/backup_dry_run_result.json", "w") as f:
            json.dump(status, f, indent=2)
    except Exception as e:  # noqa: BLE001
        logger.warning("[BackupWorker] stage dry-run result write failed: %s", e)


def _check_runway(state: dict) -> None:
    """rFP §5.5 — compute days-of-runway at current spend rate.

    Emits BACKUP_HEALTH_ALERT on tier transitions. Separate from I6 daily
    cron — that one writes telemetry JSON; this is in-loop tier watch.
    """
    if state["mode"] != "mainnet_arweave":
        return
    keypair = state["keypair_path"]
    if not keypair or not os.path.exists(keypair):
        return

    try:
        out = subprocess.check_output(
            ["node", "scripts/irys_upload.js", "balance", keypair,
             "https://api.mainnet-beta.solana.com"],
            env={**os.environ, "NODE_PATH": _node_path()},
            timeout=30,
        )
        data = json.loads(out.decode())
        if data.get("status") != "ok":
            return
        irys_sol = float(data.get("balance_readable", 0))
    except Exception as e:
        logger.debug("[BackupWorker] Irys balance check failed: %s", e)
        return

    # Estimate at ~0.0125 SOL/day (rFP §3 budget + shrink-daily applied)
    # Conservative: daily personality ~35MB + weekly soul amortized ~30MB/day
    # At Irys ~0.0002 SOL/MB → ~0.013 SOL/day. Use 0.015 for safety margin.
    daily_est = 0.015
    days_runway = irys_sol / daily_est if daily_est > 0 else 9999

    if days_runway > 30:
        tier = "green"
    elif days_runway > 7:
        tier = "yellow"
    elif days_runway > 1:
        tier = "orange"
    else:
        tier = "red"

    # Only alert on yellow+ (green is silent)
    if tier != "green":
        _send(state["send_queue"], "BACKUP_HEALTH_ALERT", state["name"], "all", {
            "severity": tier,
            "issue": "irys_runway",
            "irys_sol": round(irys_sol, 6),
            "days_runway": round(days_runway, 1),
            "daily_est_sol": daily_est,
        })
        logger.info("[BackupWorker] Runway tier=%s (%.1f days, %.4f SOL deposit)",
                    tier, days_runway, irys_sol)
        # rFP Phase 6 — Maker Telegram per §5.5 tiers
        # Cooldown escalates with urgency: red=4h, orange=12h, yellow=daily
        cooldowns = {"yellow": 86400, "orange": 43200, "red": 14400}
        with suppress(Exception):
            from titan_hcl.utils.maker_notify import (
                notify_maker, format_runway_alert)
            notify_maker(f"runway_{tier}", state["titan_id"],
                         format_runway_alert(tier, irys_sol, days_runway),
                         cooldown_s=cooldowns.get(tier, 86400))


def _node_path() -> str:
    """Resolve NODE_PATH for the subprocess (for @irys/sdk global install)."""
    try:
        return subprocess.check_output(["npm", "root", "-g"], timeout=10).decode().strip()  # noqa: async-block — backup worker sequential main loop (run_until_complete); not a concurrent/FastAPI loop
    except Exception:
        return "/usr/lib/node_modules"


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
    if state_writer is not None and _WORKER_READY:
        try:
            state_writer.heartbeat()
        except Exception:  # noqa: BLE001 — never crash the heartbeat
            pass
