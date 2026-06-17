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
`titan_hcl/logic/emot_bundle_protocol.py:164-168`:
  inner: REFLEX, FOCUS, INTUITION, IMPULSE, METABOLISM
  outer: CREATIVITY, CURIOSITY, EMPATHY, REFLECTION, INSPIRATION, VIGILANCE

# Bus protocol

  CONSUMES:
    - KERNEL_EPOCH_TICK / EPOCH_TICK — drives slot write cadence (per epoch)
    - ACTION_RESULT — IMPULSE outcome recording (per rFP_phase_c_impulse_engine_d8_3_migration §2.E)
    - MODULE_SHUTDOWN — clean exit; triggers ImpulseEngine + IntuitionEngine state persist
    - SWAP_HANDOFF / ADOPTION_REQUEST — B.2.1 supervision-transfer
  EMITS:
    - MODULE_READY — once on boot
    - MODULE_HEARTBEAT — periodic
    - NS_READY — once on boot (peers know NS slot is live)
    - IMPULSE — autonomous-action signal (per rFP §2.C, every spirit-clock tick on Trinity deficit)
    - HORMONE_STIMULUS — cross-worker hormone accumulation bridge (per rFP §2.D, → hormonal_worker)

# NS program ownership (per SPEC §7.1 line 702 + master plan §10 D15/D22/§10.5)

Under flag-on (microkernel.shm_ns_enabled = true), this worker is the
authoritative writer of titanvm_registers.bin AND the authoritative
owner of the IMPULSE + INTUITION NS programs (lifted from spirit_worker
per rFP_phase_c_impulse_engine_d8_3_migration §2.A, 2026-05-12).

Pipeline (IMPULSE NS program):
  1. Read inner_body_5d.bin / inner_mind_15d.bin / inner_spirit_45d.bin
     via StateRegistryReader (per G18-G22, SHM not bus.request).
  2. ImpulseEngine.observe(body, mind, spirit_tensor, intuition_suggestion)
     → fire IMPULSE on Trinity deficit exceeding adaptive threshold.
  3. Publish bus.IMPULSE (consumed by parent's _agency_loop → _handle_impulse
     → bus.request_async("handle_intent", agency_worker) — unchanged).
  4. On Trinity max_deficit > 0.1: publish bus.HORMONE_STIMULUS to
     hormonal_worker (replaces in-process neural_nervous_system._hormonal
     accumulate call from spirit_worker:2692-2703 — HormonalSystem lives
     in hormonal_worker per SPEC §7.1 line 703 + master plan D15).
  5. Subscribe bus.ACTION_RESULT (from agency_worker after handle_intent):
       - impulse_engine.record_outcome(impulse_id, trinity_before, trinity_after)
         (adaptive threshold EMA — in-process)
       - programs["IMPULSE"].record_outcome(reward=delta, program="IMPULSE")
         (discrete reward to IMPULSE NeuralReflexNet — in-process)

Pipeline (INTUITION NS program — co-migrated as IMPULSE dependency):
  ImpulseEngine.observe() reads intuition._last_suggestion in-process.
  IntuitionEngine.suggest(body, mind, spirit) runs on ns_worker's tick
  (no filter_down dependency — current spirit_worker passes filter_down
  but IntuitionEngine.suggest's main body does not use it; verified
  2026-05-12). On ACTION_RESULT: intuition.record_outcome + per master
  plan D15 bridge to INTUITION NeuralReflexNet via in-process
  programs["INTUITION"].record_outcome.

# Flag-gated startup

When microkernel.shm_ns_enabled = false (legacy), this worker is NOT
autostarted; spirit_worker NS logic remains authoritative on that Titan.
Default in plugin.py:1053 is True (autostart). T1+T2 remain on Phase
A+B legacy under flag-off until cascade per rFP §2.H + §3.G.5-G.6.

# Flag-gated startup

When `microkernel.shm_ns_enabled = false` (default), this worker is NOT
autostarted by Guardian — existing spirit_worker NS logic remains
authoritative. When the flag flips, slot writes here become canonical.

See: titan-docs/PLAN_microkernel_phase_c_s5_inner_trinity.md §0.5 +
     titan-docs/PLAN_microkernel_phase_c_l0_l1_rust.md §10.5 chunk C5-5 +
     titan-docs/specs/SPEC_titan_architecture.md §7.1 row 578.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from queue import Empty
from typing import Any, Optional

import numpy as np

from titan_hcl import bus
from titan_hcl.logic.emot_bundle_protocol import NS_PROGRAMS
from titan_hcl.core.module_error_handler import with_error_envelope
from titan_hcl.errors import Severity as _phase11_sev

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

# Peak-hold decay for `last_urgencies` (rFP_dead_dim_wiring_fix §2.C +
# SPEC §7.1 + D-SPEC-68 v1.13.0). NS programs publish per-tick urgencies
# that snap to 0 immediately post-fire (legacy NervousSystem.evaluate
# returns only `urgency > 0` entries); downstream consumers
# (titanvm_registers SHM + emot_cgn substrate cache) need non-zero
# variance to register a live signal. Peak-hold formula applied on each
# `ns_program_urgencies_input.bin` SHM read: new = max(current, prev × decay).
# With decay=0.9 at ~1Hz consciousness-epoch cadence, a transient peak of
# 0.7 decays to ~0.07 after ~22 ticks (~22s) — enough memory to register
# variance, short enough to track real urgency shifts.
URGENCY_PEAK_HOLD_DECAY = 0.9

# Cadence + lifecycle constants.
HEARTBEAT_INTERVAL_S = 30.0
SLOT_WRITE_MIN_INTERVAL_S = 1.0
POLL_INTERVAL_S = 0.2

# Spirit-clock cadence preserved verbatim from spirit_worker (60s default).
# Per rFP_phase_c_impulse_engine_d8_3_migration §3.B.3 — the cadence at which
# ImpulseEngine.observe() ticks. Internal _last_impulse_ts gate enforces.
IMPULSE_TICK_INTERVAL_S = 60.0

# impulse_engine_state.bin publish cadence — SPEC §7.1 row 711 specifies 1 Hz.
IMPULSE_STATE_PUBLISH_INTERVAL_S = 1.0

# IMPULSE deficit threshold for HORMONE_STIMULUS bridge — verbatim from
# spirit_worker.py:2698. Trinity max_deficit above this fires the cross-worker
# stimulus event to hormonal_worker.
HORMONE_STIMULUS_DEFICIT_THRESHOLD = 0.1
HORMONE_STIMULUS_SCALE = 0.5
HORMONE_STIMULUS_DT = 0.1

# Reward magnitude threshold for IMPULSE NeuralReflexNet record_outcome —
# verbatim from spirit_worker.py:8423.
NS_REWARD_THRESHOLD = 0.05

# Persistence cadence — save state every N ticks (in addition to MODULE_SHUTDOWN).
STATE_PERSIST_EVERY_N_TICKS = 100

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
    from titan_hcl.logic.neural_nervous_system import NeuralNervousSystem
    project_root = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", ".."))
    nn_cfg = (get_params("neural_nervous_system") or {})
    data_dir = nn_cfg.get("data_dir") or os.path.join(
        project_root, "data", "neural_nervous_system")
    return NeuralNervousSystem(config=nn_cfg, data_dir=data_dir)


def _maybe_get_writer(spec, titan_id: str):
    """Lazily construct a StateRegistryWriter for titanvm_registers.bin."""
    try:
        from titan_hcl.core.state_registry import (
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
    from titan_hcl.core.state_registry import RegistrySpec
    from titan_hcl._phase_c_constants import TITANVM_REGISTERS_SCHEMA_VERSION
    return RegistrySpec(
        name="titanvm_registers",
        dtype=np.dtype("<f4"),
        shape=(NS_PROGRAM_COUNT, NS_FIELD_COUNT),
        feature_flag="microkernel.shm_ns_enabled",
        schema_version=int(TITANVM_REGISTERS_SCHEMA_VERSION),
    )


# ── IMPULSE + INTUITION NS program engines (rFP §3.B.1-B.4) ──────────

def _init_impulse_engine(config: dict):
    """Initialize Step 7.1 Impulse Engine.

    Lifted verbatim from spirit_worker.py:1078 + spirit_worker.py:_init_impulse_engine
    (see legacy site for cfg dispatch semantics). [impulse] config section
    is plumbed through ImpulseEngine.__init__ which falls back to historical
    hardcoded defaults on missing keys.
    """
    try:
        from titan_hcl.logic.impulse_engine import ImpulseEngine
        return ImpulseEngine(config=config)
    except Exception as e:
        logger.warning("[NSWorker] ImpulseEngine init failed: %s", e)
        return None


def _init_intuition_engine(config: Optional[dict]):
    """Initialize INTUITION engine.

    Lifted verbatim from spirit_worker.py:12442 — same fallback semantics.
    Co-migrated alongside ImpulseEngine per rFP §2.I (mandatory dependency
    closure: ImpulseEngine.observe() reads intuition._last_suggestion
    in-process).
    """
    try:
        from titan_hcl.logic.intuition import IntuitionEngine
        return IntuitionEngine(config=config)
    except Exception as e:
        logger.warning("[NSWorker] IntuitionEngine init failed: %s", e)
        return None


def _build_shm_reader_bank(titan_id: str):
    """Construct ShmReaderBank for Inner Trinity + hormonal SHM reads.

    Per G18-G22 — SHM only, no bus.request for state lookup. The bank
    handles SeqLock validation + age-tagging internally.
    """
    try:
        from titan_hcl.api.shm_reader_bank import ShmReaderBank
        return ShmReaderBank(titan_id=titan_id)
    except Exception as e:
        logger.warning(
            "[NSWorker] ShmReaderBank init failed: %s — IMPULSE pipeline disabled",
            e)
        return None


def _build_impulse_state_writer(titan_id: str):
    """Construct StateRegistryWriter for impulse_engine_state.bin.

    Uses the canonical IMPULSE_ENGINE_STATE_SPEC from spirit_state_specs
    (variable_size=True per SPEC §7.1 — msgpack payload requires
    write_variable() so reader decodes exactly len(encoded) bytes; padding
    after encoded bytes breaks msgpack decode with "extra data" error).

    G21 single-writer: spirit_state_publisher's writer is gated off under
    shm_ns_enabled=true; ns_worker is the sole writer here.
    """
    try:
        from titan_hcl.logic.spirit_state_specs import (
            IMPULSE_ENGINE_STATE_SPEC,
        )
        from titan_hcl.core.state_registry import (
            StateRegistryWriter,
            ensure_shm_root,
        )
        shm_root = ensure_shm_root(titan_id=titan_id)
        return StateRegistryWriter(IMPULSE_ENGINE_STATE_SPEC, shm_root)
    except Exception as e:
        logger.warning(
            "[NSWorker] impulse_engine_state writer init failed: %s — "
            "slot writes disabled", e)
        return None


def _publish_impulse_engine_state(
    writer,
    impulse_engine,
    shm_bank,
) -> None:
    """Write impulse_engine_state.bin payload.

    Schema (preserves spirit_state_publisher._publish_impulse_engine_state
    contract verbatim — consumers unchanged):
      {
        "engine": ImpulseEngine.get_stats() output,
        "hormones": {name: {level, fire_threshold, refractory_until_ts,
                            peak_level, last_fire_ts}, ...},
        "ts": float,
      }

    Under flag-on ns_worker reads `hormonal_state.bin` via ShmReaderBank for
    the per-hormone section (HormonalSystem lives in hormonal_worker per
    SPEC §7.1 line 703; G18 SHM-only state transport).
    """
    if writer is None or impulse_engine is None:
        return
    try:
        import msgpack
    except Exception as e:
        logger.debug("[NSWorker] msgpack unavailable: %s", e)
        return

    engine_stats: dict[str, Any] = {}
    try:
        if hasattr(impulse_engine, "get_stats"):
            engine_stats = dict(impulse_engine.get_stats())
    except Exception as e:
        logger.debug("[NSWorker] ImpulseEngine.get_stats raised: %s", e)

    per_hormone: dict[str, dict[str, float]] = {}
    if shm_bank is not None:
        try:
            horm_snap = shm_bank.read_hormonal()
            if horm_snap and isinstance(horm_snap.get("hormones"), dict):
                # ShmReaderBank surfaces hormones with field names per
                # HORMONE_FIELDS. Map to legacy spirit_state_publisher
                # schema keys (consumers depend on these names).
                for name, fields in horm_snap["hormones"].items():
                    per_hormone[str(name)] = {
                        "level": float(fields.get("level", 0.0)),
                        "fire_threshold": float(fields.get("threshold", 0.0)),
                        "refractory_until_ts": float(
                            fields.get("refractory", 0.0)),
                        "peak_level": float(fields.get("peak_level", 0.0)),
                        # last_fire_ts is not surfaced by hormonal_state.bin
                        # (per SPEC §7.1 line 703: 4 fields = level / threshold
                        # / refractory / peak_level). Default 0.0; consumers
                        # tolerate per legacy publisher.
                        "last_fire_ts": 0.0,
                    }
        except Exception as e:
            logger.debug("[NSWorker] hormonal SHM read failed: %s", e)

    payload = {
        "engine": engine_stats,
        "hormones": per_hormone,
        "ts": time.time(),
    }
    try:
        encoded = msgpack.packb(payload, use_bin_type=True)
        from titan_hcl._phase_c_constants import IMPULSE_ENGINE_STATE_MAX_BYTES
        max_bytes = int(IMPULSE_ENGINE_STATE_MAX_BYTES)
        if len(encoded) > max_bytes:
            logger.warning(
                "[NSWorker] impulse_engine_state payload %d > cap %d — skipping",
                len(encoded), max_bytes)
            return
        # variable_size=True slot: write_variable(bytes) — payload size
        # lands in per-buffer metadata so readers decode exactly len(encoded)
        # bytes (no zero-padding suffix; otherwise msgpack chokes on
        # "extra data" — verified live on T3 2026-05-12).
        writer.write_variable(encoded)
    except Exception as e:
        logger.debug("[NSWorker] impulse_engine_state write failed: %s", e)


# ── State persistence (rFP §2.G + B.10-B.11) ─────────────────────────

def _ns_state_path(full_config: dict) -> str:
    """Resolve `data/ns_worker_state.json` per data_dir config."""
    data_dir = (get_params("memory_and_storage") or {}).get(
        "data_dir", "./data")
    if not data_dir:
        data_dir = "./data"
    return os.path.join(data_dir, "ns_worker_state.json")


def _load_ns_state(path: str, impulse_engine, intuition_engine) -> None:
    """Restore adaptive state from JSON on boot. G16 critical-data."""
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
    except Exception as e:
        logger.warning("[NSWorker] state load failed (%s): %s", path, e)
        return
    if impulse_engine is not None:
        try:
            ie = state.get("impulse_engine", {})
            if "threshold" in ie:
                impulse_engine._threshold = float(ie["threshold"])
            if "outcome_history" in ie and isinstance(
                    ie["outcome_history"], list):
                # ImpulseEngine uses _outcome_history deque (cap 100 per
                # impulse_engine.py:64); ndarray-safe truncation.
                from collections import deque
                impulse_engine._outcome_history = deque(
                    ie["outcome_history"][-100:], maxlen=100)
            logger.info(
                "[NSWorker] ImpulseEngine restored: threshold=%.4f",
                getattr(impulse_engine, "_threshold", 0.0))
        except Exception as e:
            logger.warning("[NSWorker] ImpulseEngine restore failed: %s", e)
    if intuition_engine is not None:
        try:
            ig = state.get("intuition_engine", {})
            if "trust" in ig:
                intuition_engine._trust = float(ig["trust"])
            if "suggestion_count" in ig:
                intuition_engine._suggestion_count = int(ig["suggestion_count"])
            logger.info(
                "[NSWorker] IntuitionEngine restored: trust=%.3f",
                getattr(intuition_engine, "_trust", 0.5))
        except Exception as e:
            logger.warning("[NSWorker] IntuitionEngine restore failed: %s", e)


def _save_ns_state(path: str, impulse_engine, intuition_engine) -> None:
    """Snapshot adaptive state to JSON. G16 critical-data."""
    state: dict[str, Any] = {"ts": time.time()}
    if impulse_engine is not None:
        try:
            state["impulse_engine"] = {
                "threshold": float(getattr(impulse_engine, "_threshold", 0.3)),
                "outcome_history": list(
                    getattr(impulse_engine, "_outcome_history", []))[-100:],
            }
        except Exception:
            pass
    if intuition_engine is not None:
        try:
            state["intuition_engine"] = {
                "trust": float(getattr(intuition_engine, "_trust", 0.5)),
                "suggestion_count": int(
                    getattr(intuition_engine, "_suggestion_count", 0)),
            }
        except Exception:
            pass
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # Atomic write — temp + rename per G16 critical-data discipline.
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, path)
    except Exception as e:
        logger.warning("[NSWorker] state save failed (%s): %s", path, e)


# ── IMPULSE tick pipeline (rFP §3.B.3 + B.6) ─────────────────────────

def _run_impulse_tick(
    send_queue,
    name: str,
    impulse_engine,
    intuition_engine,
    shm_bank,
) -> None:
    """Run one IMPULSE NS program tick.

    Lifted from spirit_loop._run_impulse + spirit_worker.py:2685-2703,
    rewired for cross-worker architecture per rFP §2.B + §2.D.

    Reads (body, mind, spirit_tensor) from SHM (G18). Calls
    ImpulseEngine.observe(). On positive return → publish bus.IMPULSE.
    On Trinity deficit > HORMONE_STIMULUS_DEFICIT_THRESHOLD → publish
    bus.HORMONE_STIMULUS to hormonal_worker.
    """
    if not impulse_engine or shm_bank is None:
        return

    # G18: SHM reads for Inner Trinity tensors.
    body_snap = None
    mind_snap = None
    spirit_snap = None
    try:
        body_snap = shm_bank.read_inner_body_5d()
        mind_snap = shm_bank.read_inner_mind_15d()
        spirit_snap = shm_bank.read_inner_spirit_45d()
    except Exception as e:
        logger.debug("[NSWorker] inner trinity SHM read failed: %s", e)
        return

    body_values = body_snap["values"] if body_snap else [0.5] * 5
    mind_values = mind_snap["values"] if mind_snap else [0.5] * 15
    # spirit_tensor: ImpulseEngine.observe() accepts the unified spirit
    # tensor (45D). Reconstruct from SAT/CHIT/ANANDA groups.
    spirit_tensor: list = []
    if spirit_snap:
        spirit_tensor = (
            list(spirit_snap.get("SAT") or [])
            + list(spirit_snap.get("CHIT") or [])
            + list(spirit_snap.get("ANANDA") or [])
        )
    if not spirit_tensor:
        spirit_tensor = [0.5] * 45

    # IntuitionEngine.suggest() runs first so its _last_suggestion is fresh
    # for ImpulseEngine.observe() to read.
    intuition_suggestion = None
    if intuition_engine is not None:
        try:
            # IntuitionEngine.suggest accepts filter_down=None per its
            # signature (intuition.py:71); main body does not use it
            # (verified 2026-05-12 — see rFP §1.G).
            intuition_engine.suggest(
                body_values, mind_values, spirit_tensor, filter_down=None)
            intuition_suggestion = getattr(
                intuition_engine, "_last_suggestion", None)
        except Exception as e:
            logger.debug("[NSWorker] IntuitionEngine.suggest failed: %s", e)

    # ImpulseEngine.observe() — emit bus.IMPULSE on positive return.
    try:
        impulse = impulse_engine.observe(
            body_values, mind_values, spirit_tensor, intuition_suggestion)
        if impulse:
            now = time.time()
            try:
                send_queue.put({
                    "type": bus.IMPULSE,
                    "src": name,
                    "dst": "all",
                    "payload": impulse,
                    "ts": now,
                })
            except Exception:
                pass
    except Exception as e:
        logger.warning("[NSWorker] ImpulseEngine.observe failed: %s", e)

    # HORMONE_STIMULUS bridge — Trinity deficit → IMPULSE hormone via
    # hormonal_worker. Replaces in-process spirit_worker.py:2692-2703 call.
    try:
        deficits = [abs(v - 0.5) for v in body_values + mind_values]
        max_deficit = max(deficits) if deficits else 0.0
        if max_deficit > HORMONE_STIMULUS_DEFICIT_THRESHOLD:
            now = time.time()
            send_queue.put({
                "type": bus.HORMONE_STIMULUS,
                "src": name,
                "dst": "hormonal_module",
                "payload": {
                    "hormone_name": "IMPULSE",
                    "stimulus": max_deficit * HORMONE_STIMULUS_SCALE,
                    "dt": HORMONE_STIMULUS_DT,
                    "src": f"{name}._run_impulse",
                    "ts": now,
                },
                "ts": now,
            })
    except Exception as e:
        logger.debug("[NSWorker] HORMONE_STIMULUS publish failed: %s", e)


def _handle_action_result(
    msg: dict,
    impulse_engine,
    intuition_engine,
    nervous_system,
) -> None:
    """Consume bus.ACTION_RESULT for IMPULSE outcome learning.

    Lifted from spirit_worker.py:8402-8430 (verbatim semantics).
    Adaptive threshold EMA + IMPULSE NeuralReflexNet discrete reward.

    nervous_system.programs contains NeuralReflexNets including IMPULSE +
    INTUITION (NeuralReflexNets live in ns_worker per master plan §10 D15).
    """
    if not impulse_engine:
        return
    try:
        payload = msg.get("payload", {}) or {}
        impulse_id = payload.get("impulse_id")
        trinity_before = payload.get("trinity_before")
        trinity_after = payload.get("trinity_after")
        if not (impulse_id and trinity_before and trinity_after):
            return
        impulse_engine.record_outcome(
            impulse_id, trinity_before, trinity_after)

        # Discrete reward to IMPULSE NeuralReflexNet (in-process —
        # NeuralReflexNets live in ns_worker per master plan §10 D15).
        if nervous_system is not None:
            try:
                delta = (
                    (float(trinity_after.get("body", 0.5))
                     - float(trinity_before.get("body", 0.5))) * 0.4
                    + (float(trinity_after.get("mind", 0.5))
                       - float(trinity_before.get("mind", 0.5))) * 0.3
                    + (float(trinity_after.get("spirit", 0.5))
                       - float(trinity_before.get("spirit", 0.5))) * 0.3
                )
                if abs(delta) > NS_REWARD_THRESHOLD:
                    try:
                        nervous_system.record_outcome(
                            reward=delta,
                            program="IMPULSE",
                            source="ns_worker.action_result")
                    except TypeError:
                        # Backward-compat: older signature without kwargs.
                        nervous_system.record_outcome(delta)
            except Exception as e:
                logger.debug("[NSWorker] NS record_outcome failed: %s", e)
    except Exception as e:
        logger.warning("[NSWorker] ACTION_RESULT handling failed: %s", e)


# Phase 11 §11.I.3 / §11.I.5 (Chunk 11N) — module-level readiness sentinel.
# Flipped True after NeuralNervousSystem + (optional) IMPULSE pipeline init
# complete. Gates SHM-slot heartbeat so titan_hcl's 1Hz poll sees real
# liveness rather than the boot-time "subscribed-but-not-warm" lie.
from titan_hcl.modules._heartbeat_grace import (
    boot_deadline_from_now, shm_heartbeat_allowed,
)
from titan_hcl.params import get_params

_WORKER_READY: bool = False
_BOOT_DEADLINE = None  # boot-grace deadline (monotonic); None=no grace


@with_error_envelope(module_name="ns_module", subsystem="entry", severity=_phase11_sev.FATAL)
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
    # Phase 11 §11.I.5 (Chunk 11N) — readiness flag reset per entry.
    global _WORKER_READY, _BOOT_DEADLINE
    _WORKER_READY = False
    _BOOT_DEADLINE = boot_deadline_from_now()

    project_root = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # ── Phase 11 §11.I.5 / Chunk 11N — SHM state-slot writer (G21 per worker) ──
    # Constructed BEFORE slow NeuralNervousSystem init so the slot publishes
    # state="starting" immediately. In-loop heartbeat (below) calls
    # _state_writer.heartbeat() each cycle so guardian_hcl's staleness
    # detector doesn't kill mid-boot.
    _state_writer = None
    try:
        from titan_hcl.core.module_state import (
            BootPriority,
            ModuleStateWriter,
        )
        _state_writer = ModuleStateWriter(
            module_name="ns_module",
            layer="L2",
            boot_priority=BootPriority.OPTIONAL_POST_BOOT,
        )
        _state_writer.write_state("starting")
    except Exception as _sw_err:  # noqa: BLE001
        logger.warning(
            "[NSWorker] Phase 11 ModuleStateWriter init failed "
            "(continuing on legacy path): %s", _sw_err)

    full_config = config or {}
    # rFP_trinity_130d_phase2_5_closure §4 (2026-05-08): use canonical
    # resolve_titan_id() — see hormonal_worker.py for full rationale.
    from titan_hcl.core.state_registry import resolve_titan_id
    titan_id = (
        (get_params("info_banner") or {}).get("titan_id")
        or resolve_titan_id()
    )

    logger.info("[NSWorker] Booting — titan_id=%s, name=%s", titan_id, name)

    try:
        nervous_system = _build_neural_nervous_system(full_config)
    except Exception as e:
        logger.error("[NSWorker] NeuralNervousSystem init failed: %s — exiting", e)
        return

    flag_on = bool(
        (get_params("microkernel") or {}).get("shm_ns_enabled", False))
    writer = None
    if flag_on:
        spec = _maybe_get_titanvm_spec()
        writer = _maybe_get_writer(spec, titan_id)

    # IMPULSE + INTUITION NS programs (rFP_phase_c_impulse_engine_d8_3_migration §3.B).
    # Both engines instantiated under flag-on; spirit_worker's legacy paths
    # gated off in parallel per §3.C.
    impulse_engine = None
    intuition_engine = None
    shm_bank = None
    impulse_state_writer = None
    ns_state_path = _ns_state_path(full_config)
    if flag_on:
        impulse_engine = _init_impulse_engine(get_params("impulse"))
        intuition_engine = _init_intuition_engine(full_config)
        shm_bank = _build_shm_reader_bank(titan_id)
        impulse_state_writer = _build_impulse_state_writer(titan_id)
        # G16 critical-data: restore adaptive state on boot.
        _load_ns_state(ns_state_path, impulse_engine, intuition_engine)
        logger.info(
            "[NSWorker] IMPULSE pipeline armed — engine=%s intuition=%s "
            "shm_bank=%s state_writer=%s",
            "yes" if impulse_engine else "no",
            "yes" if intuition_engine else "no",
            "yes" if shm_bank else "no",
            "yes" if impulse_state_writer else "no")

    # Most-recent urgency tracked separately — NeuralReflexNet does not
    # store urgency as an attribute. Source-of-truth (rFP_dead_dim_wiring_fix
    # §2.C, SPEC §7.1 + D-SPEC-68 v1.13.0): cognitive_worker writes
    # `ns_program_urgencies_input.bin` SHM slot per consciousness epoch
    # from `coordinator._last_nervous_signals`. We poll the slot in the
    # slot-write block below and apply peak-hold-decay to preserve
    # transient post-fire urgency peaks across the ~10-tick fire→reset
    # cycle (so the downstream SHM titanvm_registers urgency column +
    # emot_cgn ns_urgencies bundle have nonzero variance for the
    # dead-dim detector).
    last_urgencies: dict[str, float] = {n: 0.0 for n in NS_PROGRAM_NAMES}

    boot_ts = time.time()
    # Phase 11 §11.I.2 — slot transition: starting → booted (D-SPEC-141 / v1.65.0).
    # MODULE_READY bus emit DELETED per locked D2 (no shim, no dual-publish).
    # SHM slot is the contract; titan_hcl's 1Hz poll detects "booted" and
    # dispatches MODULE_PROBE_REQUEST → handler below transitions slot to "running".
    _WORKER_READY = True
    if _state_writer is not None:
        try:
            _state_writer.write_state("booted")
            logger.info(
                "[NSWorker] Phase 11 §11.I.2 — SHM slot state=booted "
                "(awaiting MODULE_PROBE_REQUEST from titan_hcl)")
        except Exception as _swb_err:  # noqa: BLE001
            logger.warning(
                "[NSWorker] Phase 11 write_state(booted) failed: %s", _swb_err)
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
    last_impulse_tick_ts = 0.0
    last_impulse_state_publish_ts = 0.0
    last_state_persist_tick = 0
    epoch_count = 0
    error_count = 0
    slot_write_count = 0
    impulse_tick_count = 0
    impulse_publish_count = 0
    urg_input_read_err_count = 0  # rFP_dead_dim_wiring_fix §2.C — first-fail + every-100th log

    while True:
        now = time.time()

        # IMPULSE tick — spirit-clock 60s cadence preserved verbatim from
        # spirit_worker. Independent of NS epoch cadence; runs even when no
        # message is on the recv_queue.
        if (flag_on and impulse_engine is not None
                and (now - last_impulse_tick_ts) >= IMPULSE_TICK_INTERVAL_S):
            _run_impulse_tick(
                send_queue, name, impulse_engine, intuition_engine, shm_bank)
            last_impulse_tick_ts = now
            impulse_tick_count += 1
            # Periodic persistence after every N ticks (G16 critical-data).
            if (impulse_tick_count - last_state_persist_tick
                    >= STATE_PERSIST_EVERY_N_TICKS):
                _save_ns_state(ns_state_path, impulse_engine, intuition_engine)
                last_state_persist_tick = impulse_tick_count

        # impulse_engine_state.bin publish — 1 Hz per SPEC §7.1 line 711.
        # G21 single-writer: spirit_worker's _publish_impulse_engine_state
        # gated off under flag-on (§3.C.8).
        if (flag_on and impulse_state_writer is not None
                and impulse_engine is not None
                and (now - last_impulse_state_publish_ts)
                >= IMPULSE_STATE_PUBLISH_INTERVAL_S):
            _publish_impulse_engine_state(
                impulse_state_writer, impulse_engine, shm_bank)
            last_impulse_state_publish_ts = now
            impulse_publish_count += 1

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
                        "impulse_tick_count": impulse_tick_count,
                        "impulse_publish_count": impulse_publish_count,
                        "impulse_pipeline_active": (
                            impulse_engine is not None
                            and shm_bank is not None),
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
                    "[NSWorker] MODULE_PROBE_REQUEST handler failed: %s",
                    _probe_err)
            continue

        try:
            from titan_hcl.core import worker_swap_handler as _swap
            if _swap.maybe_dispatch_swap_msg(msg):
                continue
        except Exception:
            pass

        if msg_type == bus.MODULE_SHUTDOWN:
            logger.info("[NSWorker] Shutdown received — persisting state + exit")
            # G16 critical-data — final state snapshot on clean shutdown.
            # AUDIT §C fix (rFP §P2): was gated on `impulse_engine is not None`
            # while intuition state is LOADED unconditionally on boot (line 773).
            # If ImpulseEngine init fails (None) but IntuitionEngine succeeds,
            # intuition._trust/_suggestion_count loaded-but-never-saved → lost on
            # respawn. _save_ns_state already serializes each engine independently
            # (lines 493/502), so fire the flush if EITHER engine is present.
            if flag_on and (impulse_engine is not None
                            or intuition_engine is not None):
                _save_ns_state(
                    ns_state_path, impulse_engine, intuition_engine)
            return

        is_epoch_tick = msg_type == getattr(bus, "KERNEL_EPOCH_TICK", "KERNEL_EPOCH_TICK") \
            or msg_type == getattr(bus, "EPOCH_TICK", "EPOCH_TICK")
        if is_epoch_tick:
            epoch_count += 1
            # Bug C consumer (rFP_dead_dim_wiring_fix §2.C, SPEC §7.1 +
            # D-SPEC-68 v1.13.0): read canonical NS-program urgencies
            # from `ns_program_urgencies_input.bin` SHM slot (written by
            # cognitive_worker from `coordinator._last_nervous_signals`)
            # and update last_urgencies with peak-hold-decay. Closes the
            # load-bearing cross-process wire-up gap from the ns_worker
            # L2 carve-out — without this, last_urgencies stayed at
            # zeros → SHM titanvm_registers urgency column + downstream
            # NS_URGENCIES_UPDATE → emot_cgn `ns_urgencies` substrate
            # cache all read dead values fleet-wide.
            if shm_bank is not None:
                try:
                    _urg_snap = shm_bank.read_ns_program_urgencies_input()
                    if _urg_snap and isinstance(_urg_snap, dict):
                        _urg_per = _urg_snap.get("urgencies_by_program") or {}
                        for _name in NS_PROGRAM_NAMES:
                            if _name in _urg_per:
                                try:
                                    _curr = float(_urg_per[_name])
                                except (TypeError, ValueError):
                                    continue
                                _prev = last_urgencies.get(_name, 0.0)
                                # Peak-hold-with-decay: snap to current
                                # value on any peak, decay otherwise.
                                last_urgencies[_name] = max(
                                    _curr, _prev * URGENCY_PEAK_HOLD_DECAY)
                except Exception as _u_err:
                    if urg_input_read_err_count == 0 or \
                            urg_input_read_err_count % 100 == 0:
                        logger.warning(
                            "[NSWorker] ns_program_urgencies_input read "
                            "error (count=%d): %s",
                            urg_input_read_err_count + 1, _u_err)
                    urg_input_read_err_count += 1

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
                # RFP_meta-reasoning_CGN_FIX.md §8 — direct producer
                # emit for emot_cgn substrate cache (closes DEAD-DIM
                # `ns_urgencies`). Coalesced to the same 1Hz slot-write
                # cadence above; the dict carries per-program urgency
                # values, emot_cgn normalizes ordering to
                # emot_bundle_protocol.NS_PROGRAMS.
                try:
                    send_queue.put_nowait({
                        "type": bus.NS_URGENCIES_UPDATE,
                        "src": name,
                        "dst": "emot_cgn",
                        "payload": {
                            "urgencies_by_program": dict(last_urgencies),
                            "ts": now,
                        },
                    })
                except Exception as _emit_err:
                    logger.debug(
                        "[NSWorker] NS_URGENCIES_UPDATE emit failed: %s",
                        _emit_err)
            continue

        # ACTION_RESULT — IMPULSE outcome learning (rFP §2.E).
        # Adaptive threshold EMA + IMPULSE NeuralReflexNet discrete reward.
        if flag_on and msg_type == bus.ACTION_RESULT:
            _handle_action_result(
                msg, impulse_engine, intuition_engine, nervous_system)
            continue
