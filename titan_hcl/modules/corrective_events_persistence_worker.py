"""corrective_events_persistence_worker — L2 subscriber for §G6.6 corrective events.

Per PLAN_trinity_homeostasis_p0 §6.6.6 + SPEC D-SPEC-132 (P0.6-C). Sole
consumer of `EXTREME_IMBALANCE_DETECTED` + `CORRECTIVE_NUDGE` events on
the persistence side — pairs them by (source_part, side, dominant_dim_idx)
and writes one `trinity_corrective_events` row per fire-then-nudge cycle
in `consciousness.db` so the per-Titan PolarityHomeostat trajectory
survives a restart and inherits the §24 Arweave-backed sovereign-backup
chain.

Sister of `journey_persistence_worker` (P0.5). Same best-effort posture:
SQL write failures → warn-and-continue; the Rust source-of-truth is the
B/M daemon's PolarityHomeostat state, not this row.

Bus subscriptions:
  REQUIRED — EXTREME_IMBALANCE_DETECTED, CORRECTIVE_NUDGE,
             MODULE_SHUTDOWN, SAVE_NOW.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from queue import Empty
from typing import Any, Dict, Optional

from titan_hcl import bus
from titan_hcl.core.module_error_handler import with_error_envelope
from titan_hcl.errors import Severity as _phase11_sev

logger = logging.getLogger(__name__)

_HEARTBEAT_INTERVAL_S = 10.0
_POLL_INTERVAL_S = 0.2


# Phase 11 §11.I.3 / §11.I.5 (Chunk 11N) — module-level readiness sentinel
# mirrored to per-process SHM slot via ModuleStateWriter.
from titan_hcl.modules._heartbeat_grace import (
    boot_deadline_from_now, shm_heartbeat_allowed,
)
from titan_hcl.params import get_params

_WORKER_READY: bool = False
_BOOT_DEADLINE = None  # boot-grace deadline (monotonic); None=no grace

_CORRECTIVE_PERSISTENCE_SUBSCRIBE_TOPICS: list[str] = [
    bus.EXTREME_IMBALANCE_DETECTED,
    bus.CORRECTIVE_NUDGE,
    bus.MODULE_SHUTDOWN,
    bus.SAVE_NOW,
]


def _send_msg(send_queue, msg_type: str, src: str, dst: str,
              payload: dict) -> None:
    try:
        send_queue.put({
            "type": msg_type, "src": src, "dst": dst,
            "payload": payload, "ts": time.time(),
        })
    except Exception:
        pass


def _send_heartbeat(send_queue, name: str, extra: Optional[dict] = None,
                    state_writer: Optional[Any] = None) -> None:
    """MODULE_HEARTBEAT + Phase 11 SHM state-slot heartbeat."""
    try:
        import resource
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        rss_mb = 0.0
    payload = {"alive": True, "ts": time.time(), "rss_mb": round(rss_mb, 1)}
    if extra:
        payload.update(extra)
    _send_msg(send_queue, bus.MODULE_HEARTBEAT, name, "guardian", payload)
    if state_writer is not None and shm_heartbeat_allowed(_WORKER_READY, _BOOT_DEADLINE):
        try:
            state_writer.heartbeat()
        except Exception:  # noqa: BLE001 — never crash heartbeat
            pass


def _decode_extreme(msg: dict) -> Optional[Dict[str, Any]]:
    payload = msg.get("payload") or {}
    if not isinstance(payload, dict):
        return None
    side = payload.get("side")
    src = payload.get("src")
    if side not in ("inner", "outer") or src not in ("body", "mind"):
        return None
    try:
        return {
            "src": src,
            "side": side,
            "dominant_dim_idx": int(payload.get("dominant_dim_idx") or 0),
            "dominant_dim_value": float(payload.get("dominant_dim_value") or 0.0),
            "polarity_at_fire": float(payload.get("polarity_at_fire") or 0.0),
            "polarity_sign": float(payload.get("polarity_sign") or 0.0),
            "duration_ticks": int(payload.get("duration_ticks") or 0),
            "sigma_multiplier": float(payload.get("sigma_multiplier") or 0.0),
            "extreme_event_count_lifetime": int(
                payload.get("extreme_event_count_lifetime") or 0
            ),
            "ts": float(payload.get("ts") or time.time()),
        }
    except (TypeError, ValueError):
        return None


def _decode_corrective(msg: dict) -> Optional[Dict[str, Any]]:
    payload = msg.get("payload") or {}
    if not isinstance(payload, dict):
        return None
    side = payload.get("target_side")
    src = payload.get("target_src")
    if side not in ("inner", "outer") or src not in ("body", "mind"):
        return None
    try:
        return {
            "target_src": src,
            "target_side": side,
            "target_dim_idx": int(payload.get("target_dim_idx") or 0),
            "nudge_value": float(payload.get("nudge_value") or 0.0),
            "intensity": float(payload.get("intensity") or 0.0),
            "ts": float(payload.get("ts") or time.time()),
        }
    except (TypeError, ValueError):
        return None


@with_error_envelope(module_name="corrective_events_persistence", subsystem="entry", severity=_phase11_sev.FATAL)
def corrective_events_persistence_worker_main(recv_queue, send_queue, name: str,
                                              config: dict) -> None:
    """Main loop for the corrective_events_persistence_worker subprocess.

    Pairs EXTREME_IMBALANCE_DETECTED + CORRECTIVE_NUDGE events into one
    row of `trinity_corrective_events`. Best-effort delivery — orphan
    fires (no matching nudge within 5s) are persisted with nudge_* NULL.
    """
    project_root = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from titan_hcl.core.worker_bus_bootstrap import setup_worker_bus
    try:
        recv_queue, send_queue, _bus_client = setup_worker_bus(
            name, recv_queue, send_queue,
            topics=list(_CORRECTIVE_PERSISTENCE_SUBSCRIBE_TOPICS),
        )
    except Exception as _err:
        logger.error("[CorrectiveEventsPersistence] setup_worker_bus failed: %s",
                     _err, exc_info=True)
        return

    try:
        from titan_hcl.core.worker_lifecycle import install_parent_death_signal
        install_parent_death_signal()
    except Exception as _err:
        logger.debug("[CorrectiveEventsPersistence] pdeathsig skipped: %s", _err)

    # Phase 11 §11.I.5 (Chunk 11N) — reset module-level readiness sentinel.
    global _WORKER_READY, _BOOT_DEADLINE
    _WORKER_READY = False
    _BOOT_DEADLINE = boot_deadline_from_now()

    from titan_hcl.core.state_registry import resolve_titan_id
    titan_id = (
        (get_params("info_banner") or {}).get("titan_id")
        or resolve_titan_id()
    )

    # ── Phase 11 §11.I.5 / Chunk 11N — SHM state-slot writer (G21) ──
    _state_writer = None
    try:
        from titan_hcl.core.module_state import (
            BootPriority,
            ModuleStateWriter,
        )
        _state_writer = ModuleStateWriter(
            module_name=name,
            layer="L2",
            boot_priority=BootPriority.OPTIONAL_POST_BOOT,
        )
        _state_writer.write_state("starting")
    except Exception as _sw_err:  # noqa: BLE001
        logger.warning(
            "[CorrectiveEventsPersistence] Phase 11 ModuleStateWriter init "
            "failed (continuing — SHM slot disabled): %s", _sw_err)

    db_path = config.get("consciousness_db", "./data/consciousness.db")
    db = None
    try:
        from titan_hcl.logic.consciousness import ConsciousnessDB
        db = ConsciousnessDB(db_path)
        logger.info("[CorrectiveEventsPersistence] Connected to %s", db_path)
    except Exception as _err:
        logger.error("[CorrectiveEventsPersistence] DB connect failed: %s — "
                     "events dropped (warned). Fix + restart to resume.",
                     _err, exc_info=True)

    # ── Phase 11 §11.I.2 — slot transition: starting → booted ─────────
    # Legacy MODULE_READY bus emit retired per locked D2.
    _WORKER_READY = True
    if _state_writer is not None:
        try:
            _state_writer.write_state("booted")
        except Exception as _swb_err:  # noqa: BLE001
            logger.warning(
                "[CorrectiveEventsPersistence] Phase 11 write_state(booted) "
                "failed: %s", _swb_err)
    logger.info("[CorrectiveEventsPersistence] Booted (titan_id=%s, db=%s)",
                titan_id, "connected" if db else "disconnected")

    extremes_persisted = 0
    nudges_paired = 0
    nudges_orphan = 0
    schema_skipped = 0
    db_skipped = 0
    last_heartbeat = time.time()

    while True:
        now = time.time()
        if now - last_heartbeat > _HEARTBEAT_INTERVAL_S:
            _send_heartbeat(send_queue, name, extra={
                "extremes_persisted": extremes_persisted,
                "nudges_paired": nudges_paired,
                "nudges_orphan": nudges_orphan,
                "schema_skipped": schema_skipped,
                "db_skipped": db_skipped,
            }, state_writer=_state_writer)
            last_heartbeat = now

        try:
            msg = recv_queue.get(timeout=_POLL_INTERVAL_S)
        except Empty:
            continue
        except (KeyboardInterrupt, SystemExit):
            break

        try:
            from titan_hcl.core import worker_swap_handler as _swap
            if _swap.maybe_dispatch_swap_msg(msg):
                continue
        except Exception:
            pass

        msg_type = msg.get("type") if isinstance(msg, dict) else None

        # ── Phase 11 §11.I.3 — MODULE_PROBE_REQUEST handler ────────
        if msg_type == bus.MODULE_PROBE_REQUEST:
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
                    "[CorrectiveEventsPersistence] MODULE_PROBE_REQUEST "
                    "handler failed: %s", _probe_err)
            continue

        if msg_type == bus.MODULE_SHUTDOWN:
            logger.info("[CorrectiveEventsPersistence] MODULE_SHUTDOWN — exiting")
            break
        if msg_type == bus.SAVE_NOW:
            continue

        if msg_type == bus.EXTREME_IMBALANCE_DETECTED:
            decoded = _decode_extreme(msg)
            if decoded is None:
                schema_skipped += 1
                continue
            if db is None:
                db_skipped += 1
                continue
            try:
                db.insert_trinity_corrective_event(
                    timestamp=decoded["ts"],
                    titan_id=titan_id,
                    source_part=decoded["src"],
                    side=decoded["side"],
                    dominant_dim_idx=decoded["dominant_dim_idx"],
                    dominant_dim_value=decoded["dominant_dim_value"],
                    polarity_at_fire=decoded["polarity_at_fire"],
                    polarity_sign=decoded["polarity_sign"],
                    duration_ticks=decoded["duration_ticks"],
                    sigma_multiplier=decoded["sigma_multiplier"],
                    lifetime_event_count=decoded["extreme_event_count_lifetime"],
                )
                extremes_persisted += 1
            except Exception as _err:
                logger.warning(
                    "[CorrectiveEventsPersistence] EXTREME insert failed: %s",
                    _err, exc_info=True,
                )
                db_skipped += 1
            continue

        if msg_type == bus.CORRECTIVE_NUDGE:
            decoded = _decode_corrective(msg)
            if decoded is None:
                schema_skipped += 1
                continue
            if db is None:
                db_skipped += 1
                continue
            try:
                paired = db.update_corrective_nudge_fields(
                    source_part=decoded["target_src"],
                    side=decoded["target_side"],
                    dominant_dim_idx=decoded["target_dim_idx"],
                    nudge_value=decoded["nudge_value"],
                    nudge_intensity=decoded["intensity"],
                    nudge_ts=decoded["ts"],
                    match_within_seconds=5.0,
                )
                if paired:
                    nudges_paired += 1
                else:
                    nudges_orphan += 1
            except Exception as _err:
                logger.warning(
                    "[CorrectiveEventsPersistence] NUDGE update failed: %s",
                    _err, exc_info=True,
                )
                db_skipped += 1
            continue

    _send_msg(send_queue, bus.MODULE_SHUTDOWN, name, "guardian", {
        "ts": time.time(),
        "extremes_persisted": extremes_persisted,
        "nudges_paired": nudges_paired,
        "nudges_orphan": nudges_orphan,
    })
    logger.info(
        "[CorrectiveEventsPersistence] Exiting — extremes=%d paired=%d "
        "orphan=%d schema_skipped=%d db_skipped=%d",
        extremes_persisted, nudges_paired, nudges_orphan,
        schema_skipped, db_skipped,
    )
