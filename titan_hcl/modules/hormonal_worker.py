"""
Hormonal Worker — L2 Subprocess (C-S5 §10 D22 + SPEC §7.1 v0.1.4).

Owns one HormonalSystem instance + the hormonal_state.bin shm slot.

Sibling module of:
  - ns_worker.py (titanvm_registers.bin — 11 NS programs × 4 fields)
  - neuromod_worker.py (neuromod_state.bin — 6 neuromods × float32)

All three share the NS_PROGRAMS canonical roster (REFLEX, FOCUS, INTUITION,
IMPULSE, METABOLISM + CREATIVITY, CURIOSITY, EMPATHY, REFLECTION,
INSPIRATION, VIGILANCE) for byte-symmetric layout.

# Slot

`hormonal_state.bin` — 24 (universal SeqLock header) + 176 (11 × 4 × float32
LE) = 200 bytes total. Same byte count as `titanvm_registers.bin`.

Per-hormone fields (axis 1 of the (11, 4) shape):
  [0] level         — current accumulated hormone level
  [1] threshold     — current fire threshold (DNA-anchored, adapts via reward)
  [2] refractory    — current refractory level (post-fire suppression)
  [3] peak_level    — peak level observed since boot

# Bus protocol

  CONSUMES:
    - KERNEL_EPOCH_TICK / EPOCH_TICK — drives slot write cadence (per epoch)
    - HORMONE_STIMULUS — cross-worker stimulus accumulation (per
      rFP_phase_c_impulse_engine_d8_3_migration §2.D; IMPULSE program
      shipped 2026-05-12; other NS programs follow in their own rFPs)
    - MODULE_SHUTDOWN — clean exit
    - SWAP_HANDOFF / ADOPTION_REQUEST — B.2.1 supervision-transfer (via
      worker_swap_handler)
  EMITS:
    - MODULE_READY — once on boot
    - MODULE_HEARTBEAT — periodic (every HEARTBEAT_INTERVAL_S seconds)
    - HORMONAL_READY — once on boot (peers know hormonal slot is live)

Under flag-on (microkernel.shm_hormonal_enabled = true), this worker is
the authoritative writer of hormonal_state.bin AND the sole consumer of
HORMONE_STIMULUS bus events (replaces the legacy in-process
neural_nervous_system._hormonal.get_hormone(name).accumulate(...) call
that lived in spirit_worker). The IMPULSE NS program ships this cross-
worker bridge per rFP_phase_c_impulse_engine_d8_3_migration §2.D
(2026-05-12); REFLEX/FOCUS/METABOLISM + 6 outer programs follow in
their own per-program rFPs.

# Flag-gated startup

When `microkernel.shm_hormonal_enabled = false` (default), this worker is
NOT autostarted by Guardian — the existing spirit_worker hormonal logic
remains authoritative (byte-identical to today). When the flag flips,
this worker's slot writes become the canonical hormonal state surface.

See: titan-docs/PLAN_microkernel_phase_c_s5_inner_trinity.md §0.5 +
     titan-docs/PLAN_microkernel_phase_c_l0_l1_rust.md §10.5 chunk C5-7 +
     titan-docs/specs/SPEC_titan_architecture.md §7.1 (v0.1.4 row).
"""
from __future__ import annotations

import logging
import os
import sys
import time
from queue import Empty
from typing import Optional

import numpy as np

from titan_hcl import bus
from titan_hcl.logic.emot_bundle_protocol import NS_PROGRAMS
from titan_hcl.core.module_error_handler import with_error_envelope
from titan_hcl.errors import Severity as _phase11_sev

logger = logging.getLogger("hormonal")

# Canonical hormone roster — 1:1 with NS_PROGRAMS per
# emot_bundle_protocol.py:164-168. Frozen here for slot-layout determinism;
# any addition of an NS program requires a SPEC PATCH bump (HORMONE_COUNT
# change → HORMONAL_STATE_SCHEMA_VERSION bump per SPEC §3.1 D05).
HORMONE_NAMES: tuple[str, ...] = tuple(NS_PROGRAMS)

HORMONE_COUNT = 11
HORMONE_FIELD_COUNT = 4  # level, threshold, refractory, peak_level
HORMONAL_STATE_PAYLOAD_BYTES = HORMONE_COUNT * HORMONE_FIELD_COUNT * 4  # 176

# Field axis indexes (within the (11, 4) ndarray).
FIELD_LEVEL = 0
FIELD_THRESHOLD = 1
FIELD_REFRACTORY = 2
FIELD_PEAK_LEVEL = 3

# Cadence + lifecycle constants.
HEARTBEAT_INTERVAL_S = 30.0
SLOT_WRITE_MIN_INTERVAL_S = 1.0  # rate-limit slot writes to ≥ 1s apart
POLL_INTERVAL_S = 0.2

# Module name (matches Guardian registry per SPEC §9.B line 982).
MODULE_NAME = "hormonal_module"


def encode_hormonal_state(hormonal_system) -> np.ndarray:
    """Encode a HormonalSystem instance to a (11, 4) float32 numpy array
    suitable for direct write to `hormonal_state.bin` via StateRegistryWriter.

    Row order = HORMONE_NAMES (= NS_PROGRAMS canonical). Missing hormones
    in the system are encoded as zero rows (defensive — should not happen
    in production where the registry is fully populated at boot).

    Field order per row: [level, threshold, refractory, peak_level].
    """
    arr = np.zeros((HORMONE_COUNT, HORMONE_FIELD_COUNT), dtype=np.float32)
    for i, name in enumerate(HORMONE_NAMES):
        hormone = hormonal_system.get_hormone(name) if hormonal_system else None
        if hormone is None:
            continue
        arr[i, FIELD_LEVEL] = float(hormone.level)
        arr[i, FIELD_THRESHOLD] = float(hormone.threshold)
        arr[i, FIELD_REFRACTORY] = float(hormone.refractory)
        arr[i, FIELD_PEAK_LEVEL] = float(getattr(hormone, "peak_level", 0.0))
    return arr


def decode_hormonal_state(arr: np.ndarray) -> dict[str, dict[str, float]]:
    """Reverse of `encode_hormonal_state` — used by tests + observability
    consumers that read the slot directly without instantiating a full
    HormonalSystem.

    Returns: `{hormone_name: {level, threshold, refractory, peak_level}}`.
    """
    if arr.shape != (HORMONE_COUNT, HORMONE_FIELD_COUNT):
        raise ValueError(
            f"hormonal_state shape mismatch: expected "
            f"({HORMONE_COUNT}, {HORMONE_FIELD_COUNT}), got {arr.shape}"
        )
    out: dict[str, dict[str, float]] = {}
    for i, name in enumerate(HORMONE_NAMES):
        out[name] = {
            "level": float(arr[i, FIELD_LEVEL]),
            "threshold": float(arr[i, FIELD_THRESHOLD]),
            "refractory": float(arr[i, FIELD_REFRACTORY]),
            "peak_level": float(arr[i, FIELD_PEAK_LEVEL]),
        }
    return out


def _build_hormonal_system(full_config: dict):
    """Construct a HormonalSystem with the canonical 11-hormone roster.

    Uses default cross-talk + circadian + hormone params (per
    `hormonal_pressure.py` defaults) unless overridden in `full_config`.
    """
    from titan_hcl.logic.hormonal_pressure import HormonalSystem
    hs_cfg = (full_config.get("hormonal_pressure", {}) or {})
    return HormonalSystem(
        program_names=list(HORMONE_NAMES),
        cross_talk=hs_cfg.get("cross_talk"),
        circadian=hs_cfg.get("circadian"),
        hormone_params=hs_cfg.get("hormone_params"),
    )


def _maybe_get_writer(spec, titan_id: str):
    """Lazily construct a StateRegistryWriter for hormonal_state.bin.

    Returns None on flag-off / setup failure; the caller falls back to no-op.
    """
    try:
        from titan_hcl.core.state_registry import (
            StateRegistryWriter,
            ensure_shm_root,
        )
        shm_root = ensure_shm_root(titan_id=titan_id)
        return StateRegistryWriter(spec, shm_root)
    except Exception as e:
        logger.warning(
            "[HormonalWorker] hormonal_state writer init failed: %s — "
            "slot writes disabled", e)
        return None


# Phase 11 §11.I.3 / §11.I.5 (Chunk 11N) — module-level readiness sentinel.
# Flipped True after HormonalSystem init + state load + SHM writer ready.
# Gates SHM-slot heartbeat so titan_hcl's 1Hz poll sees real liveness.
from titan_hcl.modules._heartbeat_grace import (
    boot_deadline_from_now, shm_heartbeat_allowed,
)

_WORKER_READY: bool = False
_BOOT_DEADLINE = None  # boot-grace deadline (monotonic); None=no grace


@with_error_envelope(module_name="hormonal_module", subsystem="entry", severity=_phase11_sev.FATAL)
def hormonal_worker_main(
    recv_queue,
    send_queue,
    name: str,
    config: dict,
) -> None:
    """Main loop for the Hormonal worker subprocess.

    Args:
        recv_queue: bus → worker
        send_queue: worker → bus
        name: Guardian module name (must equal MODULE_NAME = "hormonal_module")
        config: full config dict — used for titan_id, hormonal_pressure config,
            and the microkernel.shm_hormonal_enabled flag.
    """
    # Phase 11 §11.I.5 (Chunk 11N) — readiness flag reset per entry.
    global _WORKER_READY, _BOOT_DEADLINE
    _WORKER_READY = False
    _BOOT_DEADLINE = boot_deadline_from_now()

    project_root = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # ── Phase 11 §11.I.5 / Chunk 11N — SHM state-slot writer (G21 per worker) ──
    # Constructed BEFORE slow HormonalSystem init so the slot publishes
    # state="starting" immediately. In-loop heartbeat calls
    # _state_writer.heartbeat() so guardian_hcl staleness detector survives boot.
    _state_writer = None
    try:
        from titan_hcl.core.module_state import (
            BootPriority,
            ModuleStateWriter,
        )
        _state_writer = ModuleStateWriter(
            module_name="hormonal_module",
            layer="L2",
            boot_priority=BootPriority.OPTIONAL_POST_BOOT,
        )
        _state_writer.write_state("starting")
    except Exception as _sw_err:  # noqa: BLE001
        logger.warning(
            "[HormonalWorker] Phase 11 ModuleStateWriter init failed "
            "(continuing on legacy path): %s", _sw_err)

    full_config = config or {}
    # rFP_trinity_130d_phase2_5_closure §4 (2026-05-08): use canonical
    # resolve_titan_id() instead of hardcoding "T1" fallback. Pre-fix,
    # T2/T3 deployments had no `info_banner.titan_id` in their config
    # (deploy_t2.sh preserves T2's config.toml across pulls but the
    # banner key isn't always populated) → workers wrote SHM to
    # /dev/shm/titan_T1/ on the wrong machine, leaving the canonical
    # /dev/shm/titan_T2/hormonal_state.bin (etc.) empty → cascading
    # PARTIAL classifications on T2's outer_spirit + inner_mind blocks.
    from titan_hcl.core.state_registry import resolve_titan_id
    titan_id = (
        (full_config.get("info_banner", {}) or {}).get("titan_id")
        or resolve_titan_id()
    )

    logger.info("[HormonalWorker] Booting — titan_id=%s, name=%s", titan_id, name)

    # Build HormonalSystem
    try:
        hormonal_system = _build_hormonal_system(full_config)
    except Exception as e:
        logger.error("[HormonalWorker] HormonalSystem init failed: %s — exiting", e)
        return

    # Try to load persisted state if the path exists (preserves cross-restart
    # continuity per directive_memory_preservation.md).
    try:
        state_path = os.path.join(
            project_root, "data", "hormonal_state.json")
        hormonal_system.load(state_path)
    except Exception as e:
        logger.debug("[HormonalWorker] no prior state to load (%s)", e)

    # Lazy slot writer — None if flag-off or shm setup fails.
    flag_on = bool(
        (full_config.get("microkernel") or {}).get("shm_hormonal_enabled", False))
    writer = None
    if flag_on:
        from titan_hcl.core.state_registry import HORMONAL_STATE
        writer = _maybe_get_writer(HORMONAL_STATE, titan_id)

    # Boot signals (per reflex_worker.py:81-93 pattern).
    boot_ts = time.time()
    # Phase 11 §11.I.2 — slot transition: starting → booted (D-SPEC-141 / v1.65.0).
    # MODULE_READY bus emit DELETED per locked D2 (no shim, no dual-publish).
    _WORKER_READY = True
    if _state_writer is not None:
        try:
            _state_writer.write_state("booted")
            logger.info(
                "[HormonalWorker] Phase 11 §11.I.2 — SHM slot state=booted "
                "(awaiting MODULE_PROBE_REQUEST from titan_hcl)")
        except Exception as _swb_err:  # noqa: BLE001
            logger.warning(
                "[HormonalWorker] Phase 11 write_state(booted) failed: %s",
                _swb_err)
    if hasattr(bus, "HORMONAL_READY"):
        try:
            send_queue.put({
                "type": bus.HORMONAL_READY, "src": name, "dst": "all",
                "payload": {
                    "titan_id": titan_id, "ts": boot_ts,
                    "hormone_count": HORMONE_COUNT,
                    "slot_writer_active": writer is not None,
                },
                "ts": boot_ts,
            })
        except Exception:
            pass

    last_heartbeat_ts = 0.0
    last_slot_write_ts = 0.0
    epoch_count = 0
    error_count = 0
    slot_write_count = 0
    # Track hormones we've already logged-info on first stimulus —
    # operational-visibility throttle for HORMONE_STIMULUS consumer.
    _logged_stimulus_hormones: set[str] = set()

    # Periodic-save state. Pre-2026-05-15, hormonal_worker only saved its
    # state to `data/hormonal_state.json` on graceful MODULE_SHUTDOWN.
    # Process kills (any non-graceful exit — restart cascade SIGKILL,
    # OOM, panic) silently lost the accumulated hormone state. Fleet
    # audit 2026-05-15 confirmed: T3's data/hormonal_state.json was last
    # modified 2026-05-06 with all levels=0 / fire_count=0; T1 had no
    # file at all. Effect: every restart loaded a stale-zero state →
    # cognitive_worker pre-extraction had its OWN HormonalSystem in-
    # process that warmed independently of the persisted file → masked
    # the bug. Phase C cross-process expression_worker (§4.B Track 3)
    # reads hormone-levels from hormonal_worker via SHM → exposed the
    # cold-start.
    # Fix: save every _PERIODIC_SAVE_INTERVAL_S seconds + on SAVE_NOW.
    last_save_ts = 0.0
    _PERIODIC_SAVE_INTERVAL_S = 30.0

    while True:
        now = time.time()

        # Periodic save — durability invariant (see comment block above).
        # Best-effort; failures are warning-logged but do NOT crash the
        # worker.
        if now - last_save_ts >= _PERIODIC_SAVE_INTERVAL_S:
            try:
                state_path = os.path.join(
                    project_root, "data", "hormonal_state.json")
                hormonal_system.save(state_path)
            except Exception as e:
                logger.warning(
                    "[HormonalWorker] periodic save failed: %s", e)
            last_save_ts = now

        # Periodic heartbeat (per SPEC §10.B MODULE_HEARTBEAT_INTERVAL_S=10s
        # but reflex_worker uses 30s — we follow reflex_worker's lighter
        # cadence since hormonal updates are themselves epoch-paced).
        if now - last_heartbeat_ts >= HEARTBEAT_INTERVAL_S:
            try:
                send_queue.put({
                    "type": bus.MODULE_HEARTBEAT, "src": name, "dst": "guardian",
                    "payload": {
                        "alive": True, "ts": now,
                        "epoch_count": epoch_count,
                        "slot_write_count": slot_write_count,
                        "error_count": error_count,
                        "slot_writer_active": writer is not None,
                    },
                    "ts": now,
                })
            except Exception:
                pass
            # Phase 11 §11.I.5 — SHM-slot heartbeat sidecar.
            if _state_writer is not None and shm_heartbeat_allowed(_WORKER_READY, _BOOT_DEADLINE):
                try:
                    _state_writer.heartbeat()
                except Exception:  # noqa: BLE001
                    pass
            last_heartbeat_ts = now

        # Pull next bus message (or timeout for heartbeat loop).
        try:
            msg = recv_queue.get(timeout=POLL_INTERVAL_S)
        except Empty:
            continue
        except Exception:
            continue

        msg_type = msg.get("type")

        # ── Phase 11 §11.I.3 — MODULE_PROBE_REQUEST handler ─────────────
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
                    "[HormonalWorker] MODULE_PROBE_REQUEST handler failed: %s",
                    _probe_err)
            continue

        # B.2.1 supervision-transfer dispatch (mirrors reflex_worker:130-132).
        try:
            from titan_hcl.core import worker_swap_handler as _swap
            if _swap.maybe_dispatch_swap_msg(msg):
                continue
        except Exception:
            pass

        if msg_type == bus.MODULE_SHUTDOWN:
            logger.info("[HormonalWorker] Shutdown received — saving state + exiting")
            try:
                state_path = os.path.join(
                    project_root, "data", "hormonal_state.json")
                hormonal_system.save(state_path)
            except Exception as e:
                logger.warning("[HormonalWorker] save on shutdown failed: %s", e)
            return

        # SAVE_NOW — B.1 shadow_swap orchestrator + manual checkpoint
        # trigger. Durability invariant — see periodic-save comment block
        # above. Mirrors social_graph_worker / memory_worker SAVE_NOW
        # handlers.
        if msg_type == bus.SAVE_NOW:
            try:
                state_path = os.path.join(
                    project_root, "data", "hormonal_state.json")
                hormonal_system.save(state_path)
                logger.info(
                    "[HormonalWorker] SAVE_NOW — checkpoint to %s", state_path)
                last_save_ts = now
            except Exception as e:
                logger.warning(
                    "[HormonalWorker] SAVE_NOW save failed: %s", e)
            continue

        # KERNEL_EPOCH_TICK (or legacy EPOCH_TICK) drives the per-epoch slot
        # write. Under flag-on, stimulus accumulation arrives via
        # HORMONE_STIMULUS bus events (per rFP_phase_c_impulse_engine_d8_3_migration
        # §2.D + §3.B.7) — handled below this branch.
        is_epoch_tick = msg_type == getattr(bus, "KERNEL_EPOCH_TICK", "KERNEL_EPOCH_TICK") \
            or msg_type == getattr(bus, "EPOCH_TICK", "EPOCH_TICK")
        if is_epoch_tick:
            epoch_count += 1
            # Rate-limit slot writes — even at high tick rate, slot writes
            # happen at ≥ 1s spacing.
            if writer is not None and (now - last_slot_write_ts) >= SLOT_WRITE_MIN_INTERVAL_S:
                try:
                    arr = encode_hormonal_state(hormonal_system)
                    writer.write(arr)
                    slot_write_count += 1
                    last_slot_write_ts = now
                except Exception as e:
                    error_count += 1
                    logger.warning(
                        "[HormonalWorker] slot write failed (epoch=%d): %s",
                        epoch_count, e)
            continue

        # HORMONE_STIMULUS — cross-worker stimulus accumulation bridge.
        # Per rFP_phase_c_impulse_engine_d8_3_migration §2.D — replaces the
        # legacy in-process spirit_worker:2692-2703 call
        # `neural_nervous_system._hormonal.get_hormone(name).accumulate(...)`.
        # Producer: ns_worker (IMPULSE program first SHIPPED 2026-05-12;
        # REFLEX/FOCUS/METABOLISM + 6 outer follow per-program in TBD rFPs).
        if msg_type == bus.HORMONE_STIMULUS:
            try:
                payload = msg.get("payload", {}) or {}
                hormone_name = payload.get("hormone_name")
                stimulus = float(payload.get("stimulus", 0.0))
                dt = float(payload.get("dt", 0.1))
                if hormone_name and hormonal_system is not None:
                    hormone = hormonal_system.get_hormone(hormone_name)
                    if hormone is not None:
                        hormone.accumulate(stimulus, dt)
                        # Throttled info log on first stimulus per hormone
                        # per Titan-uptime — operational visibility.
                        if hormone_name not in _logged_stimulus_hormones:
                            _logged_stimulus_hormones.add(hormone_name)
                            logger.info(
                                "[HormonalWorker] First HORMONE_STIMULUS "
                                "received: hormone=%s stimulus=%.4f dt=%.3f "
                                "src=%s", hormone_name, stimulus, dt,
                                payload.get("src", "?"))
                    else:
                        logger.warning(
                            "[HormonalWorker] HORMONE_STIMULUS for unknown "
                            "hormone: %s — dropped", hormone_name)
            except Exception as e:
                error_count += 1
                logger.warning(
                    "[HormonalWorker] HORMONE_STIMULUS handling failed: %s", e)
            continue

        # NOTE: HORMONE_CONSUME is NOT handled here. EXPRESSION composite
        # consumption is applied in cognitive_worker against the NNS
        # HormonalSystem (neural_nervous_system._hormonal) — the instance
        # published to nns_hormonal_state.bin and read by expression_worker.
        # This worker owns a SEPARATE hormonal_state.bin instance the
        # expression urge never reads, so consuming here had no effect
        # (2026-06-01 correction).

        # Other messages currently no-op until per-program follow-up rFPs
        # extend HORMONE_STIMULUS to REFLEX/FOCUS/METABOLISM + 6 outer.
