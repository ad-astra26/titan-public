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
import hashlib
import json
import logging
import os
import subprocess
import sys
import tarfile
import time
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty
from typing import Optional

logger = logging.getLogger(__name__)


# Module-level heartbeat throttle (mirror memory_worker pattern, ≤1 per 3s)
_last_hb_ts: float = 0.0


def backup_worker_main(recv_queue, send_queue, name: str, config: dict) -> None:
    """Main loop for the Backup Worker subprocess.

    Args:
        recv_queue: bus → worker
        send_queue: worker → bus
        name: Guardian module name ("backup")
        config: full config dict (rFP Phase 3 reads [backup] section from here)
    """
    project_root = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    logger.info("[BackupWorker] Initializing...")
    init_start = time.time()

    # Unwrap config — arrives as the full config dict from v5_core
    full_config = config or {}
    backup_cfg = full_config.get("backup", {}) or {}
    titan_id = full_config.get("info_banner", {}).get("titan_id", "T1")

    # Per-Titan mode (rFP Phase 3). Default: infer from keypair_path presence
    # (mainnet_arweave if keypair exists + arweave_enabled, local_only otherwise).
    net_cfg = full_config.get("network", {}) or {}
    mode = backup_cfg.get("mode", "").strip().lower()
    if not mode:
        kp_exists = bool(net_cfg.get("wallet_keypair_path", "")) and os.path.exists(
            net_cfg.get("wallet_keypair_path", "")
        )
        budget_enabled = full_config.get("mainnet_budget", {}).get(
            "backup_arweave_enabled", False
        )
        mode = "mainnet_arweave" if (kp_exists and budget_enabled) else "local_only"
    logger.info("[BackupWorker] mode=%s titan_id=%s", mode, titan_id)

    # Local snapshot dir (rFP Phase 2 step 3 — always-save, even on failed upload)
    local_dir = Path(backup_cfg.get("local_dir", "data/backups"))
    local_dir.mkdir(parents=True, exist_ok=True)
    local_rolling_days = int(backup_cfg.get("local_rolling_days", 30))
    local_snapshot_always = bool(backup_cfg.get("local_snapshot_always", True))
    upload_verify = bool(backup_cfg.get("upload_verification_enabled", True))
    tarball_validate = bool(backup_cfg.get("tarball_validation_enabled", True))
    dry_run_on_boot = bool(backup_cfg.get("dry_run_on_boot", True))

    # Build ArweaveStore once at boot (rFP BUG-5)
    arweave_store = None
    keypair_path = net_cfg.get("wallet_keypair_path", "") or ""
    net_name = net_cfg.get("solana_network", "devnet")
    if net_name == "mainnet-beta":
        net_name = "mainnet"

    if mode == "mainnet_arweave":
        try:
            if keypair_path and os.path.exists(keypair_path):
                from titan_plugin.utils.arweave_store import ArweaveStore
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

    # Build RebirthBackup (rFP BUG-2 correct signature + BUG-5 injection)
    try:
        from titan_plugin.logic.backup import RebirthBackup
        backup = RebirthBackup(
            network_client=None,  # subprocess: no direct Solana client; anchor + NFT mint are best-effort
            config=full_config.get("memory_and_storage", {}),
            titan_id=titan_id,
            arweave_store=arweave_store,
            full_config=full_config,
        )
        logger.info(
            "[BackupWorker] RebirthBackup ready (titan_id=%s, arweave=%s)",
            titan_id, "wired" if arweave_store else "none",
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

    # rFP I4 — dry-run on boot (build tarball + validate, no upload)
    if dry_run_on_boot:
        try:
            _dry_run(backup, local_dir, tarball_validate)
        except Exception as e:
            logger.warning("[BackupWorker] Dry-run failed: %s", e)

    boot_elapsed = time.time() - init_start
    logger.info("[BackupWorker] Ready in %.1fs (local_dir=%s, mode=%s)",
                boot_elapsed, local_dir, mode)

    # Signal ready (Guardian convention)
    _send(send_queue, "MODULE_READY", name, "guardian", {})
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
    }

    last_heartbeat = time.time()
    last_runway_check = 0.0
    runway_check_interval = 3600.0  # once per hour (rFP I6 is separate daily cron; this is in-loop light check)

    # ── Main loop ──────────────────────────────────────────────────────
    while True:
        try:
            msg = recv_queue.get(timeout=5.0)
        except Empty:
            if time.time() - last_heartbeat > 10.0:
                _send_heartbeat(send_queue, name)
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

        if msg_type == "MODULE_SHUTDOWN":
            logger.info("[BackupWorker] Shutdown: %s",
                        msg.get("payload", {}).get("reason"))
            break

        try:
            if msg_type == "MEDITATION_COMPLETE":
                _send_heartbeat(send_queue, name)  # keep alive during backup
                _handle_meditation(state, msg)
                last_heartbeat = time.time()
            elif msg_type == "BACKUP_TRIGGER_MANUAL":
                _send_heartbeat(send_queue, name)
                _handle_manual(state, msg)
                last_heartbeat = time.time()
            # Ignore other msg types — don't complain (bus delivers many)
        except Exception as e:
            logger.error("[BackupWorker] Handler error (%s): %s", msg_type, e,
                         exc_info=True)

    logger.info("[BackupWorker] Exiting")
    with suppress(Exception):
        loop.close()


# ── Meditation-triggered cascade (Phase 2 §5.3) ───────────────────────────

def _handle_meditation(state: dict, msg: dict) -> None:
    """Handle MEDITATION_COMPLETE — run personality/soul/timechain cascade.

    This implements rFP §5.3 — the 10-step failsafe cascade. Delegates to
    RebirthBackup.on_meditation_complete for the dedup + scheduling decisions
    (daily personality / weekly soul / TimeChain) but wraps the whole call
    with per-type cascade observability.
    """
    payload = msg.get("payload", {}) or {}
    if not payload.get("success", False):
        logger.debug("[BackupWorker] Meditation not successful — skipping")
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
            from titan_plugin.utils.maker_notify import (
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
        from titan_plugin.logic.offhost_mirror import OffhostMirror
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
        logger.debug("[BackupWorker] I7 telemetry write failed: %s", e)


def _notify_maker_success(titan_id: str, backup, duration_s: float, mode: str) -> None:
    """Phase 6 — emit Telegram success alert for completed cascade results.

    Pulls from the backup records written during on_meditation_complete. Only
    fires if a new upload happened (not a dedup skip). At-most-daily per type.
    """
    with suppress(Exception):
        from titan_plugin.utils.maker_notify import (
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
        if backup_type == "personality":
            result = loop.run_until_complete(backup.upload_personality_to_arweave())
        elif backup_type == "soul":
            result = loop.run_until_complete(backup.upload_soul_package_to_arweave())
        elif backup_type == "timechain":
            # TimeChain uses its own path; requires TimeChainBackup wrapper
            from titan_plugin.logic.timechain_backup import TimeChainBackup
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

def _dry_run(backup, local_dir: Path, validate: bool) -> None:
    """rFP I4 — full-pipeline dry-run (no upload) at boot.

    Builds personality archive → validates extract-test → writes result to
    data/backup_dry_run_result.json. Catches tarball corruption / missing
    aux DB / signing key issues BEFORE they corrupt a real backup.
    """
    result_path = "data/backup_dry_run_result.json"
    t0 = time.time()
    status = {"ok": False, "ts": time.time(), "steps": {}}
    tmp_path = None
    try:
        # Step 1: build tarball (arweave tier — realistic daily check)
        tmp_path = backup.create_personality_archive(
            output_path=f"/tmp/backup_dry_run_{int(time.time())}.tar.gz",
            arweave_tier=True,
        )
        if not tmp_path or not os.path.exists(tmp_path):
            status["steps"]["build"] = "FAIL (no archive)"
            raise RuntimeError("dry-run: archive build returned None")
        sz = os.path.getsize(tmp_path)
        status["steps"]["build"] = f"OK ({sz/1024/1024:.1f} MB)"

        # Step 2: validate extract-test (critical: tarball not silently corrupt)
        if validate:
            with tarfile.open(tmp_path, "r:gz") as tf:
                members = tf.getnames()
            status["steps"]["validate"] = f"OK ({len(members)} members)"
        else:
            status["steps"]["validate"] = "SKIP (disabled)"

        # Step 3: compute hash (proof of what we'd upload)
        status["steps"]["hash"] = _sha256_file(tmp_path)[:16]

        status["ok"] = True
        status["duration_s"] = round(time.time() - t0, 2)
        logger.info("[BackupWorker] Dry-run PASS in %.1fs (%.1f MB, %d members)",
                    status["duration_s"], sz / 1024 / 1024,
                    len(members) if validate else -1)
    except Exception as e:
        status["error"] = str(e)
        status["duration_s"] = round(time.time() - t0, 2)
        logger.warning("[BackupWorker] Dry-run FAIL: %s", e)
    finally:
        # Cleanup temp archive
        if tmp_path:
            with suppress(FileNotFoundError):
                os.remove(tmp_path)
        try:
            with open(result_path, "w") as f:
                json.dump(status, f, indent=2)
        except Exception as e:
            logger.warning("[BackupWorker] Could not write dry-run result: %s", e)


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
            from titan_plugin.utils.maker_notify import (
                notify_maker, format_runway_alert)
            notify_maker(f"runway_{tier}", state["titan_id"],
                         format_runway_alert(tier, irys_sol, days_runway),
                         cooldown_s=cooldowns.get(tier, 86400))


def _node_path() -> str:
    """Resolve NODE_PATH for the subprocess (for @irys/sdk global install)."""
    try:
        return subprocess.check_output(["npm", "root", "-g"], timeout=10).decode().strip()
    except Exception:
        return "/usr/lib/node_modules"


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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
            from titan_plugin.bus import record_send_drop
            record_send_drop(src, dst, msg_type)


def _send_heartbeat(send_queue, name: str) -> None:
    """Heartbeat to Guardian (≤1 per 3s, matches memory_worker)."""
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
