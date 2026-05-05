"""
NS Worker — L2 Subprocess (C-S5 §10 D22 + SPEC §7.1).

Owns the 11 NeuralReflexNet program references + the titanvm_registers.bin
shm slot.

Sibling module of:
  - hormonal_worker.py (hormonal_state.bin — 11 hormones × 4 fields)
  - neuromod_worker.py (neuromod_state.bin — 6 neuromods × float32)

# Slot

`titanvm_registers.bin` — 24 (universal SeqLock header) + 176 (11 × 4
× float32 LE) = 200 bytes total. Same byte count as hormonal_state.bin
(SPEC v0.1.4 sibling-symmetric layout).

Per-program fields (axis 1 of the (11, 4) shape):
  [0] urgency       — most recent output urgency from this program
                       (tracked by the worker, NOT a NeuralReflexNet attr)
  [1] fire_count    — cumulative fire count (NeuralReflexNet.fire_count)
  [2] total_updates — cumulative training steps (NeuralReflexNet.total_updates)
  [3] last_loss     — most recent training loss (NeuralReflexNet.last_loss)

Program canonical order matches NS_PROGRAMS in
`titan_plugin/logic/emot_bundle_protocol.py:164-168`:
  inner: REFLEX, FOCUS, INTUITION, IMPULSE, METABOLISM
  outer: CREATIVITY, CURIOSITY, EMPATHY, REFLECTION, INSPIRATION, VIGILANCE

# Bus protocol (transitional — full stimulus integration deferred)

  CONSUMES:
    - KERNEL_EPOCH_TICK / EPOCH_TICK — drives slot write cadence (per epoch)
    - MODULE_SHUTDOWN — clean exit
    - SWAP_HANDOFF / ADOPTION_REQUEST — B.2.1 supervision-transfer
  EMITS:
    - MODULE_READY — once on boot
    - MODULE_HEARTBEAT — periodic
    - NS_READY — once on boot (peers know NS slot is live)

NS program evaluation (NeuralNervousSystem.evaluate) is currently driven
by spirit_worker — that wiring stays under flag-off. Under flag-on
(microkernel.shm_ns_enabled = true), this worker becomes the
authoritative slot writer; the cross-worker stimulus pipeline (sensor →
ns_worker → neuromod_worker → hormonal_worker) is wired in a follow-up
commit per master plan §10.5.

# Flag-gated startup

When `microkernel.shm_ns_enabled = false` (default), this worker is NOT
autostarted by Guardian — existing spirit_worker NS logic remains
authoritative. When the flag flips, slot writes here become canonical.

See: titan-docs/PLAN_microkernel_phase_c_s5_inner_trinity.md §0.5 +
     titan-docs/PLAN_microkernel_phase_c_l0_l1_rust.md §10.5 chunk C5-5 +
     titan-docs/SPEC_titan_architecture.md §7.1 row 578.
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

logger = logging.getLogger("ns")

# Canonical NS program roster — must match emot_bundle_protocol.py:164-168.
# Frozen here for slot-layout determinism; any addition requires a SPEC
# PATCH bump (NS_PROGRAM_COUNT change → TITANVM_REGISTERS_SCHEMA_VERSION
# bump per SPEC §3.1 D05).
NS_PROGRAM_NAMES: tuple[str, ...] = tuple(NS_PROGRAMS)

NS_PROGRAM_COUNT = 11
NS_FIELD_COUNT = 4  # urgency, fire_count, total_updates, last_loss
TITANVM_REGISTERS_PAYLOAD_BYTES = NS_PROGRAM_COUNT * NS_FIELD_COUNT * 4  # 176

# Field axis indexes (within the (11, 4) ndarray).
FIELD_URGENCY = 0
FIELD_FIRE_COUNT = 1
FIELD_TOTAL_UPDATES = 2
FIELD_LAST_LOSS = 3

# Cadence + lifecycle constants.
HEARTBEAT_INTERVAL_S = 30.0
SLOT_WRITE_MIN_INTERVAL_S = 1.0
POLL_INTERVAL_S = 0.2

# Module name (matches SPEC §9.B titan_HCL row line 982 + SPEC §7.1).
MODULE_NAME = "ns_module"


def encode_ns_state(
    programs: Optional[dict],
    urgencies: Optional[dict] = None,
) -> np.ndarray:
    """Encode NS program state to a (11, 4) float32 numpy array suitable
    for direct write to `titanvm_registers.bin` via StateRegistryWriter.

    Args:
        programs: dict[str, NeuralReflexNet] — typically
            `NeuralNervousSystem.programs`. None or empty → all-zero output.
        urgencies: dict[str, float] — most-recent urgency per program,
            tracked separately by the worker (NeuralReflexNet does not
            store urgency as an attribute). None → urgency column all 0.0.

    Row order = NS_PROGRAM_NAMES canonical (= NS_PROGRAMS).
    Field order per row: [urgency, fire_count, total_updates, last_loss].
    Counts (fire_count, total_updates) are float32-cast from int (precision
    loss above 2^24 ≈ 16.7M is acceptable; titanvm_registers is read-mostly
    observability state, not counter authority).
    """
    arr = np.zeros((NS_PROGRAM_COUNT, NS_FIELD_COUNT), dtype=np.float32)
    if not programs:
        return arr
    urgencies = urgencies or {}
    for i, name in enumerate(NS_PROGRAM_NAMES):
        prog = programs.get(name)
        if prog is None:
            continue
        arr[i, FIELD_URGENCY] = float(urgencies.get(name, 0.0))
        arr[i, FIELD_FIRE_COUNT] = float(getattr(prog, "fire_count", 0))
        arr[i, FIELD_TOTAL_UPDATES] = float(getattr(prog, "total_updates", 0))
        arr[i, FIELD_LAST_LOSS] = float(getattr(prog, "last_loss", 0.0))
    return arr


def decode_ns_state(arr: np.ndarray) -> dict[str, dict[str, float]]:
    """Reverse of `encode_ns_state` — returns
    `{program_name: {urgency, fire_count, total_updates, last_loss}}`.
    """
    if arr.shape != (NS_PROGRAM_COUNT, NS_FIELD_COUNT):
        raise ValueError(
            f"titanvm_registers shape mismatch: expected "
            f"({NS_PROGRAM_COUNT}, {NS_FIELD_COUNT}), got {arr.shape}"
        )
    out: dict[str, dict[str, float]] = {}
    for i, name in enumerate(NS_PROGRAM_NAMES):
        out[name] = {
            "urgency": float(arr[i, FIELD_URGENCY]),
            "fire_count": float(arr[i, FIELD_FIRE_COUNT]),
            "total_updates": float(arr[i, FIELD_TOTAL_UPDATES]),
            "last_loss": float(arr[i, FIELD_LAST_LOSS]),
        }
    return out


def _build_neural_nervous_system(full_config: dict):
    """Construct a NeuralNervousSystem.

    Uses default config from titan_params.toml's [neural_nervous_system]
    section (same as spirit_worker's existing NN init). Each L2 worker
    that owns NS state instantiates its own copy — no shared singleton.
    """
    from titan_plugin.logic.neural_nervous_system import NeuralNervousSystem
    project_root = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", ".."))
    nn_cfg = (full_config.get("neural_nervous_system", {}) or {})
    data_dir = nn_cfg.get("data_dir") or os.path.join(
        project_root, "data", "neural_nervous_system")
    return NeuralNervousSystem(config=nn_cfg, data_dir=data_dir)


def _maybe_get_writer(spec, titan_id: str):
    """Lazily construct a StateRegistryWriter for titanvm_registers.bin."""
    try:
        from titan_plugin.core.state_registry import (
            StateRegistryWriter,
            ensure_shm_root,
        )
        shm_root = ensure_shm_root(titan_id=titan_id)
        return StateRegistryWriter(spec, shm_root)
    except Exception as e:
        logger.warning(
            "[NSWorker] titanvm_registers writer init failed: %s — "
            "slot writes disabled", e)
        return None


def _maybe_get_titanvm_spec():
    """Build the RegistrySpec for titanvm_registers.bin.

    The shape (11, 4) is NOT yet declared in state_registry.py module-level
    (only neuromod_state, inner_spirit_45d, etc. live there today). We
    construct it inline using the auto-generated TITANVM_REGISTERS_SCHEMA_VERSION
    constant. Future cleanup: promote to module-level RegistrySpec parallel
    to NEUROMOD_STATE / HORMONAL_STATE.
    """
    from titan_plugin.core.state_registry import RegistrySpec
    from titan_plugin._phase_c_constants import TITANVM_REGISTERS_SCHEMA_VERSION
    return RegistrySpec(
        name="titanvm_registers",
        dtype=np.dtype("<f4"),
        shape=(NS_PROGRAM_COUNT, NS_FIELD_COUNT),
        feature_flag="microkernel.shm_ns_enabled",
        schema_version=int(TITANVM_REGISTERS_SCHEMA_VERSION),
    )


def ns_worker_main(
    recv_queue,
    send_queue,
    name: str,
    config: dict,
) -> None:
    """Main loop for the NS worker subprocess.

    Args:
        recv_queue: bus → worker
        send_queue: worker → bus
        name: Guardian module name (must equal MODULE_NAME = "ns_module")
        config: full config dict — used for titan_id, neural_nervous_system
            config, and the microkernel.shm_ns_enabled flag.
    """
    project_root = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    full_config = config or {}
    titan_id = (full_config.get("info_banner", {}) or {}).get("titan_id") or "T1"

    logger.info("[NSWorker] Booting — titan_id=%s, name=%s", titan_id, name)

    try:
        nervous_system = _build_neural_nervous_system(full_config)
    except Exception as e:
        logger.error("[NSWorker] NeuralNervousSystem init failed: %s — exiting", e)
        return

    flag_on = bool(
        (full_config.get("microkernel") or {}).get("shm_ns_enabled", False))
    writer = None
    if flag_on:
        spec = _maybe_get_titanvm_spec()
        writer = _maybe_get_writer(spec, titan_id)

    # Most-recent urgency tracked separately — NeuralReflexNet does not
    # store urgency as an attribute; we update on bus messages (when wired).
    last_urgencies: dict[str, float] = {n: 0.0 for n in NS_PROGRAM_NAMES}

    boot_ts = time.time()
    try:
        send_queue.put({
            "type": bus.MODULE_READY, "src": name, "dst": "guardian",
            "payload": {"titan_id": titan_id, "ts": boot_ts},
            "ts": boot_ts,
        })
    except Exception:
        pass
    if hasattr(bus, "NS_READY"):
        try:
            send_queue.put({
                "type": bus.NS_READY, "src": name, "dst": "all",
                "payload": {
                    "titan_id": titan_id, "ts": boot_ts,
                    "program_count": NS_PROGRAM_COUNT,
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

        try:
            from titan_plugin.core import worker_swap_handler as _swap
            if _swap.maybe_dispatch_swap_msg(msg):
                continue
        except Exception:
            pass

        if msg_type == bus.MODULE_SHUTDOWN:
            logger.info("[NSWorker] Shutdown received — exiting")
            return

        is_epoch_tick = msg_type == getattr(bus, "KERNEL_EPOCH_TICK", "KERNEL_EPOCH_TICK") \
            or msg_type == getattr(bus, "EPOCH_TICK", "EPOCH_TICK")
        if is_epoch_tick:
            epoch_count += 1
            if writer is not None and (now - last_slot_write_ts) >= SLOT_WRITE_MIN_INTERVAL_S:
                try:
                    arr = encode_ns_state(
                        getattr(nervous_system, "programs", None),
                        urgencies=last_urgencies,
                    )
                    writer.write(arr)
                    slot_write_count += 1
                    last_slot_write_ts = now
                except Exception as e:
                    error_count += 1
                    logger.warning(
                        "[NSWorker] slot write failed (epoch=%d): %s",
                        epoch_count, e)
            continue

        # Future: subscribe to NS-urgency bus events to update last_urgencies
        # before slot writes. Tracked at master plan §10.5 chunk C5-5
        # follow-up commit.
