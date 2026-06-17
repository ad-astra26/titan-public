"""journey_persistence_worker — L2 subscriber for §G5.1 UP-leg balance gifts.

Per PLAN_trinity_homeostasis_p0 §6.5.6 + SPEC D-SPEC-131 (P0.5). Sole
consumer of `BODY_BALANCE_GIFT` + `MIND_BALANCE_GIFT` events on the
persistence side — translates each event into one row of
`trinity_journey_gifts` in `consciousness.db` so the per-cycle journey
history survives a Titan restart and inherits the §24 Arweave-backed
sovereign-backup chain.

The Rust body / mind daemons (titan-{inner,outer}-{body,mind}-rs) publish
gifts on their own sphere clock's balanced rising-edge (sub-1% of
Schumann ticks). Spirit daemons consume the amplitude-only field for
mask-weighted enrichment; THIS worker stores the full digest (per-dim
contribution, journey metadata, snapshot ring) for later analysis.

Bus subscriptions:
  REQUIRED — BODY_BALANCE_GIFT (all sides), MIND_BALANCE_GIFT (all sides),
             MODULE_SHUTDOWN, SAVE_NOW.

Bus publications:
  - MODULE_HEARTBEAT / MODULE_SHUTDOWN (standard per §11; legacy MODULE_READY
    retired per Phase 11 §11.I.2 — SHM slot state=booted is the contract).

Best-effort delivery: gift events are enrichment + journey-history, NOT
load-bearing for any tick. Queue overflow → drop oldest (per PLAN
§6.5.6); SQL failures → warn-and-continue. The Rust source-of-truth is
the spirit daemon's per-dim mask application, not this row.
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
# mirrored to per-process SHM slot via ModuleStateWriter. Set False at
# import; flipped True after DB connect attempt completes (whether or
# not the DB connected — worker stays alive in DB-disconnected mode).
from titan_hcl.modules._heartbeat_grace import (
    boot_deadline_from_now, shm_heartbeat_allowed,
)
from titan_hcl.params import get_params

_WORKER_READY: bool = False
_BOOT_DEADLINE = None  # boot-grace deadline (monotonic); None=no grace

_JOURNEY_PERSISTENCE_SUBSCRIBE_TOPICS: list[str] = [
    bus.BODY_BALANCE_GIFT,
    bus.MIND_BALANCE_GIFT,
    bus.MODULE_SHUTDOWN,
    bus.SAVE_NOW,
]

# Per PLAN §6.5.2 body=5D, mind=15D, snapshot ring 32 slots.
_RING_LEN = 32
_BODY_DIMS = 5
_MIND_DIMS = 15


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
    """MODULE_HEARTBEAT to guardian_HCL + SHM state-slot heartbeat (Phase 11)."""
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


def _u8_quantise_seq(values, *, n: int) -> bytes:
    """u8-quantise a sequence of values in [0,1] to N bytes; out-of-range saturated."""
    if values is None:
        return bytes(n)
    out = bytearray(n)
    for i, v in enumerate(values[:n]):
        try:
            f = float(v)
        except (TypeError, ValueError):
            f = 0.0
        if f < 0.0:
            f = 0.0
        elif f > 1.0:
            f = 1.0
        out[i] = int(round(f * 255.0))
    return bytes(out)


def _pack_journey_metadata(payload: Dict[str, Any], dims: int) -> bytes:
    """Pack the per-cycle metadata BLOB.

    Layout (1 byte per value):
      [0:dims]                peak_excursion (u8, [0,1])
      [dims:2·dims]           path_length    (u8, saturated [0,1])
      [2·dims:3·dims]         excursion_integral (u8, normalised by dividing
                              by max for the layer so it lands in [0,1])
      [3·dims:4·dims]         direction_flips_norm (u8 = flips / cycle_ticks)
      [4·dims]                polarity_max (u8, [0,1])
      [4·dims+1]              polarity_at_balance (u8, [0,1])
      [4·dims+2]              coherence_climb_max (u8, [0,1])  (0 for body payloads)
    """
    peak = payload.get("peak_excursion") or []
    path = payload.get("path_length") or []
    integ = payload.get("excursion_integral") or []
    flips = payload.get("direction_flips") or []
    cycle_ticks = max(1, int(payload.get("cycle_tick_count") or 1))
    pol_max = float(payload.get("polarity_max") or 0.0)
    pol_at = float(payload.get("polarity_at_balance") or 0.0)
    coh_climb = float(payload.get("coherence_climb_max") or 0.0)
    integ_max = max((float(v) for v in integ), default=1.0) or 1.0
    integ_norm = [float(v) / integ_max for v in integ]
    flips_norm = [min(1.0, float(v) / float(cycle_ticks)) for v in flips]
    return b"".join([
        _u8_quantise_seq(peak, n=dims),
        _u8_quantise_seq(path, n=dims),
        _u8_quantise_seq(integ_norm, n=dims),
        _u8_quantise_seq(flips_norm, n=dims),
        _u8_quantise_seq([pol_max, pol_at, coh_climb], n=3),
    ])


def _extract_snapshot_bytes(payload: Dict[str, Any], dims: int) -> bytes:
    """The Rust encoder already u8-quantises the snapshot ring per
    `gift_events::encode_*_balance_gift` (using `u8_quantise_ring`).
    Python bus client surfaces msgpack Binary as `bytes` — pass through
    verbatim when length matches the expected 32·N bytes; otherwise
    return zero-filled (warn at caller).
    """
    raw = payload.get("snapshots")
    expected = _RING_LEN * dims
    if isinstance(raw, (bytes, bytearray, memoryview)):
        b = bytes(raw)
        if len(b) == expected:
            return b
    return bytes(expected)


def _decode_gift_event(msg: dict, *, expected_part: str) -> Optional[Dict[str, Any]]:
    """Validate + extract the gift fields needed for SQL insert.

    `expected_part` is 'body' or 'mind' — fed from the msg_type the worker
    received (BODY_BALANCE_GIFT vs MIND_BALANCE_GIFT). Returns None on
    schema drift so the caller can warn-and-continue.
    """
    payload = msg.get("payload") or {}
    if not isinstance(payload, dict):
        return None
    side = payload.get("side")
    if side not in ("inner", "outer"):
        return None
    src_field = payload.get("src")
    if src_field != expected_part:
        return None
    try:
        amp = float(payload.get("gift_amplitude") or 0.0)
        dur = float(payload.get("cycle_duration_s") or 0.0)
        ticks = int(payload.get("cycle_tick_count") or 0)
    except (TypeError, ValueError):
        return None
    return {
        "side": side,
        "gift_amplitude": amp,
        "cycle_duration_s": dur,
        "cycle_tick_count": ticks,
        "_payload": payload,
    }


@with_error_envelope(module_name="journey_persistence", subsystem="entry", severity=_phase11_sev.FATAL)
def journey_persistence_worker_main(recv_queue, send_queue, name: str,
                                    config: dict) -> None:
    """Main loop for the journey_persistence_worker subprocess.

    Subscribes to BODY_BALANCE_GIFT + MIND_BALANCE_GIFT, decodes the
    Rust-published digest, and writes one `trinity_journey_gifts` row to
    `consciousness.db` per event.
    """
    project_root = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from titan_hcl.core.worker_bus_bootstrap import setup_worker_bus
    try:
        recv_queue, send_queue, _bus_client = setup_worker_bus(
            name, recv_queue, send_queue,
            topics=list(_JOURNEY_PERSISTENCE_SUBSCRIBE_TOPICS),
        )
    except Exception as _err:
        logger.error("[JourneyPersistence] setup_worker_bus failed: %s — exiting",
                     _err, exc_info=True)
        return

    try:
        from titan_hcl.core.worker_lifecycle import install_parent_death_signal
        install_parent_death_signal()
    except Exception as _err:
        logger.debug("[JourneyPersistence] pdeathsig install skipped: %s", _err)

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
            "[JourneyPersistence] Phase 11 ModuleStateWriter init failed "
            "(continuing — SHM slot disabled): %s", _sw_err)

    db_path = config.get("consciousness_db", "./data/consciousness.db")
    db = None
    try:
        from titan_hcl.logic.consciousness import ConsciousnessDB
        db = ConsciousnessDB(db_path)
        logger.info("[JourneyPersistence] Connected to consciousness.db at %s",
                    db_path)
    except Exception as _err:
        logger.error("[JourneyPersistence] DB connect failed: %s — worker stays "
                     "alive but every gift will be dropped (warned). Fix and "
                     "restart to resume persistence.",
                     _err, exc_info=True)

    # ── Phase 11 §11.I.2 — slot transition: starting → booted ─────────
    # Legacy MODULE_READY bus emit retired per locked D2. The worker
    # stays alive even when db is None (warned mode), so the slot still
    # transitions to "booted" — DB-disconnected mode is operational, just
    # degraded (every gift dropped with WARN).
    _WORKER_READY = True
    if _state_writer is not None:
        try:
            _state_writer.write_state("booted")
        except Exception as _swb_err:  # noqa: BLE001
            logger.warning(
                "[JourneyPersistence] Phase 11 write_state(booted) failed: %s",
                _swb_err)
    logger.info("[JourneyPersistence] Booted (titan_id=%s, db=%s)",
                titan_id, "connected" if db else "disconnected")

    gifts_persisted = 0
    gifts_skipped_schema = 0
    gifts_skipped_db = 0
    last_heartbeat = time.time()

    while True:
        now = time.time()
        if now - last_heartbeat > _HEARTBEAT_INTERVAL_S:
            _send_heartbeat(send_queue, name, extra={
                "gifts_persisted": gifts_persisted,
                "gifts_skipped_schema": gifts_skipped_schema,
                "gifts_skipped_db": gifts_skipped_db,
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
                    "[JourneyPersistence] MODULE_PROBE_REQUEST handler "
                    "failed: %s", _probe_err)
            continue

        if msg_type == bus.MODULE_SHUTDOWN:
            logger.info("[JourneyPersistence] MODULE_SHUTDOWN received — exiting")
            break

        if msg_type == bus.SAVE_NOW:
            # No batched state to flush — SQL writes are eager.
            continue

        if msg_type == bus.BODY_BALANCE_GIFT:
            decoded = _decode_gift_event(msg, expected_part="body")
            if decoded is None:
                gifts_skipped_schema += 1
                continue
            if db is None:
                gifts_skipped_db += 1
                continue
            try:
                payload = decoded["_payload"]
                per_dim = _u8_quantise_seq(
                    payload.get("per_dim_contribution") or [],
                    n=_BODY_DIMS,
                )
                meta = _pack_journey_metadata(payload, dims=_BODY_DIMS)
                ring = _extract_snapshot_bytes(payload, dims=_BODY_DIMS)
                db.insert_trinity_journey_gift(
                    timestamp=float(payload.get("ts") or time.time()),
                    titan_id=titan_id,
                    source_part="body",
                    side=decoded["side"],
                    gift_amplitude=decoded["gift_amplitude"],
                    cycle_duration_s=decoded["cycle_duration_s"],
                    cycle_tick_count=decoded["cycle_tick_count"],
                    per_dim_contribution=per_dim,
                    journey_metadata=meta,
                    snapshot_ring=ring,
                )
                gifts_persisted += 1
            except Exception as _err:
                logger.warning(
                    "[JourneyPersistence] BODY_BALANCE_GIFT persist failed: %s",
                    _err, exc_info=True,
                )
                gifts_skipped_db += 1
            continue

        if msg_type == bus.MIND_BALANCE_GIFT:
            decoded = _decode_gift_event(msg, expected_part="mind")
            if decoded is None:
                gifts_skipped_schema += 1
                continue
            if db is None:
                gifts_skipped_db += 1
                continue
            try:
                payload = decoded["_payload"]
                per_dim = _u8_quantise_seq(
                    payload.get("per_dim_contribution") or [],
                    n=_MIND_DIMS,
                )
                meta = _pack_journey_metadata(payload, dims=_MIND_DIMS)
                ring = _extract_snapshot_bytes(payload, dims=_MIND_DIMS)
                db.insert_trinity_journey_gift(
                    timestamp=float(payload.get("ts") or time.time()),
                    titan_id=titan_id,
                    source_part="mind",
                    side=decoded["side"],
                    gift_amplitude=decoded["gift_amplitude"],
                    cycle_duration_s=decoded["cycle_duration_s"],
                    cycle_tick_count=decoded["cycle_tick_count"],
                    per_dim_contribution=per_dim,
                    journey_metadata=meta,
                    snapshot_ring=ring,
                )
                gifts_persisted += 1
            except Exception as _err:
                logger.warning(
                    "[JourneyPersistence] MIND_BALANCE_GIFT persist failed: %s",
                    _err, exc_info=True,
                )
                gifts_skipped_db += 1
            continue

    _send_msg(send_queue, bus.MODULE_SHUTDOWN, name, "guardian", {
        "ts": time.time(),
        "gifts_persisted": gifts_persisted,
        "gifts_skipped_schema": gifts_skipped_schema,
        "gifts_skipped_db": gifts_skipped_db,
    })
    logger.info(
        "[JourneyPersistence] Exiting cleanly — persisted=%d "
        "skipped_schema=%d skipped_db=%d",
        gifts_persisted, gifts_skipped_schema, gifts_skipped_db,
    )
