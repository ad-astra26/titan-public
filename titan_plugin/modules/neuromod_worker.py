"""
Neuromod Worker — L2 Subprocess (C-S5 §10 D22 + SPEC §7.1).

Owns one NeuromodulatorSystem instance + the neuromod_state.bin shm slot.

Sibling module of:
  - ns_worker.py (titanvm_registers.bin — 11 NS programs × 4 fields)
  - hormonal_worker.py (hormonal_state.bin — 11 hormones × 4 fields)

# Slot

`neuromod_state.bin` — 24 (universal SeqLock header) + 24 (6 × float32
LE) = 48 bytes total. Stores per-modulator `level` only (other fields —
tonic_level, phasic_level, sensitivity, setpoint, etc. — stay in the
internal NeuromodulatorSystem state file at data/neuromodulator/).

Per-modulator field (axis 0 of the (6,) shape): `level` ∈ [0, 1].

Canonical order (matches `titan_plugin/logic/neuromodulator.py:33-38`
CLEARANCE_RATES dict + SPEC §7.1 row 574 description):
  [0] DA           — Dopamine (fast → sharp reward signals)
  [1] 5HT          — Serotonin (slow → stable mood baseline)
  [2] NE           — Norepinephrine (moderate → arousal dynamics)
  [3] ACh          — Acetylcholine (fastest → precise attention shifts)
  [4] Endorphin    — Endorphins (slow → sustained flow states)
  [5] GABA         — Gamma-aminobutyric acid (moderate → calming dynamics)

# Bus protocol (transitional — full stimulus integration deferred)

  CONSUMES:
    - KERNEL_EPOCH_TICK / EPOCH_TICK — drives slot write cadence (per epoch)
    - MODULE_SHUTDOWN — clean exit
    - SWAP_HANDOFF / ADOPTION_REQUEST — B.2.1 supervision-transfer
  EMITS:
    - MODULE_READY — once on boot
    - MODULE_HEARTBEAT — periodic
    - NEUROMOD_READY — once on boot (peers know neuromod slot is live)

Stimulus accumulation (NeuromodulatorSystem.evaluate) is currently driven
by spirit_worker via NeuralNervousSystem — that wiring stays as-is under
flag-off. Under flag-on (microkernel.shm_neuromod_enabled = true), this
worker becomes the authoritative slot writer; cross-worker stimulus
pipeline (ns_worker → neuromod_worker → hormonal_worker) is wired in a
follow-up commit per master plan §10.5 + the parallel ns_worker /
hormonal_worker extractions.

# Flag-gated startup

When `microkernel.shm_neuromod_enabled = false` (default), this worker is
NOT autostarted by Guardian — existing spirit_worker neuromod logic
remains authoritative (byte-identical to today). When flag flips, slot
writes here become canonical.

See: titan-docs/PLAN_microkernel_phase_c_s5_inner_trinity.md §0.5 +
     titan-docs/PLAN_microkernel_phase_c_l0_l1_rust.md §10.5 chunk C5-6 +
     titan-docs/SPEC_titan_architecture.md §7.1 row 574.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from queue import Empty

import numpy as np

from titan_plugin import bus

logger = logging.getLogger("neuromod")

# Canonical neuromodulator roster — order MUST match SPEC §7.1 row 574 +
# titan_plugin/logic/neuromodulator.py:33-38. Frozen here for slot-layout
# determinism; any change requires a SPEC PATCH bump (NEUROMOD_COUNT
# change → NEUROMOD_SCHEMA_VERSION bump per SPEC §3.1 D05).
NEUROMOD_NAMES: tuple[str, ...] = ("DA", "5HT", "NE", "ACh", "Endorphin", "GABA")

NEUROMOD_COUNT = 6
NEUROMOD_STATE_PAYLOAD_BYTES = NEUROMOD_COUNT * 4  # 24

# Cadence + lifecycle constants (mirror hormonal_worker.py).
HEARTBEAT_INTERVAL_S = 30.0
SLOT_WRITE_MIN_INTERVAL_S = 1.0
POLL_INTERVAL_S = 0.2

# Module name (matches SPEC §9.B titan_HCL row line 982 + SPEC §7.1).
MODULE_NAME = "neuromod_module"


def encode_neuromod_state(neuromod_system) -> np.ndarray:
    """Encode a NeuromodulatorSystem instance to a (6,) float32 numpy array
    suitable for direct write to `neuromod_state.bin` via StateRegistryWriter.

    Order = NEUROMOD_NAMES canonical (DA, 5HT, NE, ACh, Endorphin, GABA).
    Missing modulators are encoded as 0.0 (defensive — should not happen
    in production where all 6 are constructed at boot).
    """
    arr = np.zeros((NEUROMOD_COUNT,), dtype=np.float32)
    if neuromod_system is None:
        return arr
    modulators = getattr(neuromod_system, "modulators", None) or {}
    for i, name in enumerate(NEUROMOD_NAMES):
        mod = modulators.get(name)
        if mod is None:
            continue
        arr[i] = float(getattr(mod, "level", 0.0))
    return arr


def decode_neuromod_state(arr: np.ndarray) -> dict[str, float]:
    """Reverse of `encode_neuromod_state` — used by tests + observability
    consumers that read the slot directly.

    Returns: `{neuromod_name: level}` for each of the 6 canonical neuromods.
    """
    if arr.shape != (NEUROMOD_COUNT,):
        raise ValueError(
            f"neuromod_state shape mismatch: expected ({NEUROMOD_COUNT},), "
            f"got {arr.shape}"
        )
    return {name: float(arr[i]) for i, name in enumerate(NEUROMOD_NAMES)}


def _build_neuromod_system(full_config: dict):
    """Construct a NeuromodulatorSystem.

    Uses default cross-coupling + clearance rates per
    `neuromodulator.py` module-level constants. The `data_dir` is resolved
    relative to project root (matches existing spirit_worker pattern).
    """
    from titan_plugin.logic.neuromodulator import NeuromodulatorSystem
    project_root = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", ".."))
    nm_cfg = (full_config.get("neuromodulator", {}) or {})
    data_dir = nm_cfg.get("data_dir") or os.path.join(
        project_root, "data", "neuromodulator")
    return NeuromodulatorSystem(data_dir=data_dir)


def _maybe_get_writer(spec, titan_id: str):
    """Lazily construct a StateRegistryWriter for neuromod_state.bin.

    Returns None on flag-off / setup failure.
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
            "[NeuromodWorker] neuromod_state writer init failed: %s — "
            "slot writes disabled", e)
        return None


def neuromod_worker_main(
    recv_queue,
    send_queue,
    name: str,
    config: dict,
) -> None:
    """Main loop for the Neuromod worker subprocess.

    Args:
        recv_queue: bus → worker
        send_queue: worker → bus
        name: Guardian module name (must equal MODULE_NAME = "neuromod_module")
        config: full config dict — used for titan_id, neuromodulator config,
            and the microkernel.shm_neuromod_enabled flag.
    """
    project_root = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    full_config = config or {}
    # rFP_trinity_130d_phase2_5_closure §4 (2026-05-08): use canonical
    # resolve_titan_id() — see hormonal_worker.py for full rationale.
    from titan_plugin.core.state_registry import resolve_titan_id
    titan_id = (
        (full_config.get("info_banner", {}) or {}).get("titan_id")
        or resolve_titan_id()
    )

    logger.info("[NeuromodWorker] Booting — titan_id=%s, name=%s", titan_id, name)

    # Build NeuromodulatorSystem — auto-loads persisted state from data_dir
    # per __init__ behavior at neuromodulator.py:267.
    try:
        neuromod_system = _build_neuromod_system(full_config)
    except Exception as e:
        logger.error("[NeuromodWorker] NeuromodulatorSystem init failed: %s — exiting", e)
        return

    # Lazy slot writer — None if flag-off or shm setup fails.
    flag_on = bool(
        (full_config.get("microkernel") or {}).get("shm_neuromod_enabled", False))
    writer = None
    if flag_on:
        from titan_plugin.core.state_registry import NEUROMOD_STATE
        writer = _maybe_get_writer(NEUROMOD_STATE, titan_id)

    # Boot signals (per reflex_worker.py + hormonal_worker.py pattern).
    boot_ts = time.time()
    try:
        send_queue.put({
            "type": bus.MODULE_READY, "src": name, "dst": "guardian",
            "payload": {"titan_id": titan_id, "ts": boot_ts},
            "ts": boot_ts,
        })
    except Exception:
        pass
    if hasattr(bus, "NEUROMOD_READY"):
        try:
            send_queue.put({
                "type": bus.NEUROMOD_READY, "src": name, "dst": "all",
                "payload": {
                    "titan_id": titan_id, "ts": boot_ts,
                    "neuromod_count": NEUROMOD_COUNT,
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

        # Periodic heartbeat (30s — same cadence as hormonal_worker).
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

        try:
            msg = recv_queue.get(timeout=POLL_INTERVAL_S)
        except Empty:
            continue
        except Exception:
            continue

        msg_type = msg.get("type")

        # B.2.1 supervision-transfer dispatch (mirrors reflex_worker).
        try:
            from titan_plugin.core import worker_swap_handler as _swap
            if _swap.maybe_dispatch_swap_msg(msg):
                continue
        except Exception:
            pass

        if msg_type == bus.MODULE_SHUTDOWN:
            logger.info("[NeuromodWorker] Shutdown received — saving + exiting")
            try:
                # NeuromodulatorSystem auto-saves periodically; force one final
                # save to capture latest state.
                if hasattr(neuromod_system, "_save_state"):
                    neuromod_system._save_state()
            except Exception as e:
                logger.warning("[NeuromodWorker] save on shutdown failed: %s", e)
            return

        # KERNEL_EPOCH_TICK drives the per-epoch slot write. Stimulus
        # evaluation (NeuromodulatorSystem.evaluate) lives in spirit_worker
        # until the full split lands per parallel ns_worker / hormonal_worker
        # extractions.
        is_epoch_tick = msg_type == getattr(bus, "KERNEL_EPOCH_TICK", "KERNEL_EPOCH_TICK") \
            or msg_type == getattr(bus, "EPOCH_TICK", "EPOCH_TICK")
        if is_epoch_tick:
            epoch_count += 1
            if writer is not None and (now - last_slot_write_ts) >= SLOT_WRITE_MIN_INTERVAL_S:
                try:
                    arr = encode_neuromod_state(neuromod_system)
                    writer.write(arr)
                    slot_write_count += 1
                    last_slot_write_ts = now
                except Exception as e:
                    error_count += 1
                    logger.warning(
                        "[NeuromodWorker] slot write failed (epoch=%d): %s",
                        epoch_count, e)
            continue

        # Other messages currently no-op until full stimulus integration.
        # Tracked at master plan §10.5 chunk C5-6 + dependent on parallel
        # ns_worker.py / hormonal_worker.py extractions + spirit_worker shim.
