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

# Bus protocol (transitional — full stimulus integration deferred)

  CONSUMES:
    - KERNEL_EPOCH_TICK / EPOCH_TICK — drives slot write cadence (per epoch)
    - MODULE_SHUTDOWN — clean exit
    - SWAP_HANDOFF / ADOPTION_REQUEST — B.2.1 supervision-transfer (via
      worker_swap_handler)
  EMITS:
    - MODULE_READY — once on boot
    - MODULE_HEARTBEAT — periodic (every HEARTBEAT_INTERVAL_S seconds)
    - HORMONAL_READY — once on boot (peers know hormonal slot is live)

Stimulus accumulation (HormonalSystem.accumulate_all) is currently driven
by spirit_worker via NeuralNervousSystem.update — that wiring stays as-is
under flag-off. Under flag-on (microkernel.shm_hormonal_enabled = true),
this worker becomes the authoritative slot writer; the actual cross-worker
stimulus pipeline (ns_module → neuromod_module → hormonal_module) is wired
in a follow-up commit per master plan §10.5 chunk C5-7 + the parallel
ns_worker/neuromod_worker extractions.

# Flag-gated startup

When `microkernel.shm_hormonal_enabled = false` (default), this worker is
NOT autostarted by Guardian — the existing spirit_worker hormonal logic
remains authoritative (byte-identical to today). When the flag flips,
this worker's slot writes become the canonical hormonal state surface.

See: titan-docs/PLAN_microkernel_phase_c_s5_inner_trinity.md §0.5 +
     titan-docs/PLAN_microkernel_phase_c_l0_l1_rust.md §10.5 chunk C5-7 +
     titan-docs/SPEC_titan_architecture.md §7.1 (v0.1.4 row).
"""
from __future__ import annotations

import logging
import os
import sys
import time
from queue import Empty
from typing import Optional

import numpy as np

from titan_plugin import bus
from titan_plugin.logic.emot_bundle_protocol import NS_PROGRAMS

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
    from titan_plugin.logic.hormonal_pressure import HormonalSystem
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
        from titan_plugin.core.state_registry import (
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
    project_root = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    full_config = config or {}
    # rFP_trinity_130d_phase2_5_closure §4 (2026-05-08): use canonical
    # resolve_titan_id() instead of hardcoding "T1" fallback. Pre-fix,
    # T2/T3 deployments had no `info_banner.titan_id` in their config
    # (deploy_t2.sh preserves T2's config.toml across pulls but the
    # banner key isn't always populated) → workers wrote SHM to
    # /dev/shm/titan_T1/ on the wrong machine, leaving the canonical
    # /dev/shm/titan_T2/hormonal_state.bin (etc.) empty → cascading
    # PARTIAL classifications on T2's outer_spirit + inner_mind blocks.
    from titan_plugin.core.state_registry import resolve_titan_id
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
        from titan_plugin.core.state_registry import HORMONAL_STATE
        writer = _maybe_get_writer(HORMONAL_STATE, titan_id)

    # Boot signals (per reflex_worker.py:81-93 pattern).
    boot_ts = time.time()
    try:
        send_queue.put({
            "type": bus.MODULE_READY, "src": name, "dst": "guardian",
            "payload": {"titan_id": titan_id, "ts": boot_ts},
            "ts": boot_ts,
        })
    except Exception:
        pass
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

    while True:
        now = time.time()

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
            last_heartbeat_ts = now

        # Pull next bus message (or timeout for heartbeat loop).
        try:
            msg = recv_queue.get(timeout=POLL_INTERVAL_S)
        except Empty:
            continue
        except Exception:
            continue

        msg_type = msg.get("type")

        # B.2.1 supervision-transfer dispatch (mirrors reflex_worker:130-132).
        try:
            from titan_plugin.core import worker_swap_handler as _swap
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

        # KERNEL_EPOCH_TICK (or legacy EPOCH_TICK) drives the per-epoch slot
        # write. Stimulus accumulation lives in spirit_worker until the full
        # split lands per the parallel ns_worker / neuromod_worker extractions.
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

        # Other messages currently no-op until follow-up stimulus integration.
        # Tracked at master plan §10.5 chunk C5-7 + dependent on parallel
        # ns_worker.py / neuromod_worker.py extractions.
