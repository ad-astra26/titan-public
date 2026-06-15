"""
Neuromod Worker — L2 Subprocess (C-S5 §10 D22 + SPEC §7.1).

Owns one NeuromodulatorSystem instance + the neuromod_state.bin shm slot.

Sibling module of:
  - ns_worker.py (titanvm_registers.bin — 11 NS programs × 4 fields)
  - hormonal_worker.py (hormonal_state.bin — 11 hormones × 4 fields)

# Slot (v2 layout — bumped 2026-05-15 with §4.Q neuromod_worker.evaluate migration)

`neuromod_state.bin` — 64 (universal SeqLock header) + 3 × (16 buffer-meta
+ 96 payload) = 400 bytes total. **Per-modulator 4 fields** = 6 × 4 × 4 = 96
payload bytes. v1 (6,) layout deprecated.

Layout: `(6, 4)` float32 LE — axis 0 = modulator, axis 1 = field.
  arr[i, 0] = level     ∈ [0, 1]
  arr[i, 1] = gain      ∈ [0.3, 3.0] (output of Modulator.get_gain())
  arr[i, 2] = phasic    ∈ ℝ (event-driven spike)
  arr[i, 3] = tonic     ∈ [0, 1] (EMA baseline)

The 4-field layout exists so cognitive_worker (cross-process consumer) can
reconstruct the 14-key modulation dict via
`compute_modulation_from_state(state)` (titan_hcl.logic.neuromodulator)
without access to the live NeuromodulatorSystem instance. Other state
fields — sensitivity, setpoint, peak_level, trough_level, _activation_history
— stay in the NeuromodulatorSystem state file at data/neuromodulator/.

Canonical order (matches `titan_hcl/logic/neuromodulator.py:33-38`
CLEARANCE_RATES dict + SPEC §7.1 row 574 description):
  [0] DA           — Dopamine (fast → sharp reward signals)
  [1] 5HT          — Serotonin (slow → stable mood baseline)
  [2] NE           — Norepinephrine (moderate → arousal dynamics)
  [3] ACh          — Acetylcholine (fastest → precise attention shifts)
  [4] Endorphin    — Endorphins (slow → sustained flow states)
  [5] GABA         — Gamma-aminobutyric acid (moderate → calming dynamics)

# Bus protocol

  CONSUMES:
    - KERNEL_EPOCH_TICK / EPOCH_TICK — drives evaluate + slot write cadence
    - NEUROMOD_EXTERNAL_NUDGE — 7 producer sites (cognitive_worker × 6 +
      outer_interface_worker × 1) emit; this worker applies via
      `NeuromodulatorSystem.apply_external_nudge(nudge_map, max_delta,
      developmental_age)`
    - MODULE_SHUTDOWN — clean exit
    - SWAP_HANDOFF / ADOPTION_REQUEST — B.2.1 supervision-transfer
  EMITS:
    - MODULE_READY — once on boot
    - MODULE_HEARTBEAT — periodic
    - NEUROMOD_READY — once on boot (peers know neuromod slot is live)
    - NEUROMOD_STATS_UPDATED — 2.5s coalesced; payload feeds
      /v4/inner-trinity.neuromodulators + /status mood

Evaluate driver: per KERNEL_EPOCH_TICK, this worker reads neuromod_inputs.bin
(written by cognitive_worker) → set_chi_health → compute_emergent_inputs →
merge kin overrides → evaluate(inputs, dt=1.0) → apply_movement_excess_clearance
→ write neuromod_state.bin (4-field v2 layout).

See: titan-docs/PLAN_microkernel_phase_c_neuromod_worker_evaluate_migration.md
     titan-docs/specs/SPEC_titan_architecture.md §7.1 + §9.B neuromod_worker block.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from queue import Empty

import numpy as np

from titan_hcl import bus
from titan_hcl.core.module_error_handler import with_error_envelope
from titan_hcl.errors import Severity as _phase11_sev

logger = logging.getLogger("neuromod")

# Canonical neuromodulator roster — order MUST match SPEC §7.1 row 574 +
# titan_hcl/logic/neuromodulator.py:33-38. Frozen here for slot-layout
# determinism; any change requires a SPEC PATCH bump (NEUROMOD_COUNT
# change → NEUROMOD_SCHEMA_VERSION bump per SPEC §3.1 D05).
NEUROMOD_NAMES: tuple[str, ...] = ("DA", "5HT", "NE", "ACh", "Endorphin", "GABA")
NEUROMOD_FIELD_NAMES: tuple[str, ...] = ("level", "gain", "phasic", "tonic")

NEUROMOD_COUNT = 6
NEUROMOD_FIELDS_PER_MOD = 4
NEUROMOD_STATE_PAYLOAD_BYTES = NEUROMOD_COUNT * NEUROMOD_FIELDS_PER_MOD * 4  # 96

# Cadence + lifecycle constants (mirror hormonal_worker.py).
HEARTBEAT_INTERVAL_S = 30.0
SLOT_WRITE_MIN_INTERVAL_S = 1.0
POLL_INTERVAL_S = 0.2
# §4.Q (2026-05-15) NEUROMOD_STATS_UPDATED 2.5s coalesced publish cadence —
# matches /v4/inner-trinity.neuromodulators + /status mood route caching contract
# (replaces dead spirit_loop._publish_coord_subdomains path).
STATS_PUBLISH_INTERVAL_S = 2.5

# Module name (matches SPEC §9.B titan_HCL row line 982 + SPEC §7.1).
MODULE_NAME = "neuromod_module"


def encode_neuromod_state(neuromod_system) -> np.ndarray:
    """Encode a NeuromodulatorSystem instance to a (6, 4) float32 numpy
    array suitable for direct write to `neuromod_state.bin` v2 layout via
    StateRegistryWriter.

    Axis 0 = NEUROMOD_NAMES canonical order (DA, 5HT, NE, ACh, Endorphin, GABA).
    Axis 1 = NEUROMOD_FIELD_NAMES (level, gain, phasic, tonic).

    Missing modulators encode as zeros (defensive — should not happen in
    production where all 6 are constructed at boot).
    """
    arr = np.zeros((NEUROMOD_COUNT, NEUROMOD_FIELDS_PER_MOD), dtype=np.float32)
    if neuromod_system is None:
        return arr
    modulators = getattr(neuromod_system, "modulators", None) or {}
    for i, name in enumerate(NEUROMOD_NAMES):
        mod = modulators.get(name)
        if mod is None:
            continue
        arr[i, 0] = float(getattr(mod, "level", 0.0))
        get_gain = getattr(mod, "get_gain", None)
        arr[i, 1] = float(get_gain()) if callable(get_gain) else 1.0
        arr[i, 2] = float(getattr(mod, "phasic_level", 0.0))
        arr[i, 3] = float(getattr(mod, "tonic_level", 0.0))
    return arr


def decode_neuromod_state(arr: np.ndarray) -> dict[str, dict[str, float]]:
    """Reverse of `encode_neuromod_state` (v2 layout).

    Returns: `{modulator_name: {"level": float, "gain": float,
                                "phasic": float, "tonic": float}}`
    for each of the 6 canonical neuromods. The dict shape matches the input
    that `titan_hcl.logic.neuromodulator.compute_modulation_from_state()`
    expects, so cognitive_worker can pipe straight through:
        state = decode_neuromod_state(reader.read())
        modulation = compute_modulation_from_state(state)
    """
    if arr.shape != (NEUROMOD_COUNT, NEUROMOD_FIELDS_PER_MOD):
        raise ValueError(
            f"neuromod_state shape mismatch: expected "
            f"({NEUROMOD_COUNT}, {NEUROMOD_FIELDS_PER_MOD}), got {arr.shape}"
        )
    return {
        name: {
            NEUROMOD_FIELD_NAMES[j]: float(arr[i, j])
            for j in range(NEUROMOD_FIELDS_PER_MOD)
        }
        for i, name in enumerate(NEUROMOD_NAMES)
    }


def decode_neuromod_levels(arr: np.ndarray) -> dict[str, float]:
    """Levels-only convenience decoder for callers that need `{name: level}`
    (the legacy v1 contract). Accepts both v2 (6, 4) and v1 (6,) layouts so
    consumers upgrading at different rates remain compatible.

    Raises ValueError on any other shape — silent fallback would hide
    schema drift (per `feedback_three_state_health_checks.md`).
    """
    if arr.shape == (NEUROMOD_COUNT, NEUROMOD_FIELDS_PER_MOD):
        return {name: float(arr[i, 0]) for i, name in enumerate(NEUROMOD_NAMES)}
    if arr.shape == (NEUROMOD_COUNT,):
        return {name: float(arr[i]) for i, name in enumerate(NEUROMOD_NAMES)}
    raise ValueError(
        f"neuromod_state shape mismatch: expected "
        f"({NEUROMOD_COUNT}, {NEUROMOD_FIELDS_PER_MOD}) or ({NEUROMOD_COUNT},), "
        f"got {arr.shape}"
    )


def _build_neuromod_system(full_config: dict):
    """Construct a NeuromodulatorSystem.

    Uses default cross-coupling + clearance rates per
    `neuromodulator.py` module-level constants. The `data_dir` is resolved
    relative to project root (matches existing spirit_worker pattern).
    """
    from titan_hcl.logic.neuromodulator import NeuromodulatorSystem
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
        from titan_hcl.core.state_registry import (
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


def _maybe_get_inputs_reader(titan_id: str):
    """Lazily construct a StateRegistryReader for neuromod_inputs.bin (§4.Q).

    Returns None on shm setup failure — evaluate driver runs with default
    inputs in that case (preserves boot-resilience).
    """
    try:
        from titan_hcl.core.state_registry import (
            NEUROMOD_INPUTS,
            StateRegistryReader,
            ensure_shm_root,
        )
        shm_root = ensure_shm_root(titan_id=titan_id)
        return StateRegistryReader(NEUROMOD_INPUTS, shm_root)
    except Exception as e:
        logger.warning(
            "[NeuromodWorker] neuromod_inputs reader init failed: %s — "
            "evaluate will run with empty inputs", e)
        return None


def _drive_evaluate(
    neuromod_system,
    inputs_reader,
) -> tuple[bool, int]:
    """Per-tick evaluate driver (§4.Q chunk Q6).

    Steps:
        1. Read neuromod_inputs.bin via SHM (msgpack-decoded payload).
        2. set_chi_health(payload["chi_health"]).
        3. Set _topology_velocity for movement clearance.
        4. evaluate(inputs, dt) — drives DA/5HT/NE/ACh/Endorphin/GABA dynamics.
        5. apply_movement_excess_clearance per modulator (topology-modulated).

    Returns: (ok, kin_overrides_present)
        ok: True if evaluate ran end-to-end
        kin_overrides_present: count of kin_* keys in inputs (preserved
            dead-code semantics from pre-§4.Q spirit_worker.py:4318-4327)
    """
    if inputs_reader is None or neuromod_system is None:
        return (False, 0)

    try:
        raw = inputs_reader.read_variable()
    except Exception as e:
        logger.warning("[NeuromodWorker] inputs slot read failed: %s", e)
        return (False, 0)
    if not raw:
        return (False, 0)

    try:
        import msgpack
        payload = msgpack.unpackb(raw, raw=False)
    except Exception as e:
        logger.warning("[NeuromodWorker] inputs msgpack decode failed: %s", e)
        return (False, 0)

    inputs = payload.get("inputs") or {}
    if not inputs:
        return (False, 0)

    # chi_health metabolic gate (preserves spirit_worker.py:4329-4332 math).
    chi_health = float(payload.get("chi_health", 1.0))
    try:
        neuromod_system.set_chi_health(chi_health)
    except Exception:
        pass

    # Topology velocity — used by movement clearance below.
    topo_v = float(payload.get("topology_velocity", 0.3))
    neuromod_system._topology_velocity = topo_v

    dt = float(payload.get("dt", 1.0))

    # Preserve dead-code kin overrides — evaluate() never reads them but
    # we attach to inputs dict for downstream debugging + pre-§4.Q parity.
    kin_overrides = payload.get("kin_overrides") or {}
    for k, v in kin_overrides.items():
        inputs[k] = float(v)

    # Drive evaluate — THIS is the call that produces fresh DA/5HT/NE/ACh/
    # Endorphin/GABA levels (off-air fleet-wide since 2026-05-14 16:47 UTC
    # until this chunk lands).
    try:
        neuromod_system.evaluate(inputs, dt=dt)
    except Exception as e:
        logger.warning("[NeuromodWorker] evaluate failed: %s", e)
        return (False, len(kin_overrides))

    # Movement excess clearance — drains DA/5HT/etc. above setpoint when
    # topology velocity is high (preserves spirit_worker.py:4336-4342 math).
    try:
        from titan_hcl.logic.neuromodulator import apply_movement_excess_clearance
        # DNA movement-clearance keys named `movement_<MOD>` (e.g. movement_DA).
        # NeuromodulatorSystem caches dna at _dna_cache for compute_emergent_inputs;
        # if absent fall back to {} which yields 0.0 per get() defaults.
        dna = getattr(neuromod_system, "_dna_cache", {}) or {}
        for mod_name, mod in neuromod_system.modulators.items():
            move_rate = float(dna.get(f"movement_{mod_name}", 0.0) or 0.0)
            if move_rate > 0:
                apply_movement_excess_clearance(mod, topo_v, move_rate)
    except Exception as e:
        logger.debug("[NeuromodWorker] movement clearance skipped: %s", e)

    return (True, len(kin_overrides))


def _build_stats_payload(neuromod_system, titan_id: str) -> dict:
    """Build NEUROMOD_STATS_UPDATED payload (§4.Q chunk Q12).

    Schema matches /v4/inner-trinity.neuromodulators + /status mood
    expected fields — modulators (per-name 4 fields), modulation dict
    (14 keys), current_emotion + confidence, total_evaluations.
    """
    modulators_dict = {}
    for name, mod in neuromod_system.modulators.items():
        modulators_dict[name] = {
            "level": float(mod.level),
            "gain": float(mod.get_gain()),
            "phasic": float(mod.phasic_level),
            "tonic": float(mod.tonic_level),
        }
    return {
        "titan_id": titan_id,
        "modulators": modulators_dict,
        "modulation": neuromod_system.get_modulation(),
        "current_emotion": getattr(neuromod_system, "_current_emotion", "neutral"),
        "emotion_confidence": float(getattr(neuromod_system, "_emotion_confidence", 0.0)),
        "total_evaluations": int(getattr(neuromod_system, "_total_evaluations", 0)),
        "ts": time.time(),
    }


def _apply_external_nudge_payload(neuromod_system, payload: dict) -> bool:
    """Apply an external nudge from NEUROMOD_EXTERNAL_NUDGE bus event (§4.Q chunk Q7).

    Payload schema (two shapes per v1.8.3 D-SPEC-57):
      (a) §4.Q original: {"nudge_map": dict[str, float], "max_delta": float,
                          "developmental_age": float, "source": str}
          → forwards to neuromod_system.apply_external_nudge(...)
      (b) §4.G chi_health bridge: {"chi_health": float, "source":
          "life_force_chi_health", "ts": float}
          → forwards to neuromod_system.set_chi_health(chi_health) —
          closes the §4.Q D-SPEC-54 orphan-nudge tracked item (the
          dead spirit_worker.py:3770 direct call now arrives here via
          life_force_worker per evaluate).

    Returns True if applied (developmental gate at 0.1 may suppress).
    """
    source = str(payload.get("source", ""))

    # §4.G chi_health bridge (v1.8.3 D-SPEC-57).
    if source == "life_force_chi_health" and "chi_health" in payload:
        try:
            neuromod_system.set_chi_health(float(payload["chi_health"]))
            return True
        except Exception as e:
            logger.warning(
                "[NeuromodWorker] set_chi_health (life_force bridge) failed: %s",
                e)
            return False

    # §4.Q original path — nudge_map fan-out.
    nudge_map = payload.get("nudge_map") or {}
    if not nudge_map:
        return False
    max_delta = float(payload.get("max_delta", 0.015))
    dev_age = float(payload.get("developmental_age", 0.0))
    try:
        neuromod_system.apply_external_nudge(
            nudge_map, max_delta=max_delta, developmental_age=dev_age)
        return True
    except Exception as e:
        logger.warning("[NeuromodWorker] apply_external_nudge failed: %s", e)
        return False


# Phase 11 §11.I.3 / §11.I.5 (Chunk 11N) — module-level readiness sentinel.
# Flipped True after NeuromodulatorSystem init + DNA cache load complete.
# Gates SHM-slot heartbeat so titan_hcl's 1Hz poll sees real liveness.
from titan_hcl.modules._heartbeat_grace import (
    boot_deadline_from_now, shm_heartbeat_allowed,
)

_WORKER_READY: bool = False
_BOOT_DEADLINE = None  # boot-grace deadline (monotonic); None=no grace


@with_error_envelope(module_name="neuromod_module", subsystem="entry", severity=_phase11_sev.FATAL)
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
    # Phase 11 §11.I.5 (Chunk 11N) — readiness flag reset per entry.
    global _WORKER_READY, _BOOT_DEADLINE
    _WORKER_READY = False
    _BOOT_DEADLINE = boot_deadline_from_now()

    project_root = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # ── Phase 11 §11.I.5 / Chunk 11N — SHM state-slot writer (G21 per worker) ──
    # Constructed BEFORE slow NeuromodulatorSystem init so the slot publishes
    # state="starting" immediately. In-loop heartbeat calls
    # _state_writer.heartbeat() so guardian_hcl staleness detector survives boot.
    _state_writer = None
    try:
        from titan_hcl.core.module_state import (
            BootPriority,
            ModuleStateWriter,
        )
        _state_writer = ModuleStateWriter(
            module_name="neuromod_module",
            layer="L2",
            boot_priority=BootPriority.OPTIONAL_POST_BOOT,
        )
        _state_writer.write_state("starting")
    except Exception as _sw_err:  # noqa: BLE001
        logger.warning(
            "[NeuromodWorker] Phase 11 ModuleStateWriter init failed "
            "(continuing on legacy path): %s", _sw_err)

    full_config = config or {}
    # rFP_trinity_130d_phase2_5_closure §4 (2026-05-08): use canonical
    # resolve_titan_id() — see hormonal_worker.py for full rationale.
    from titan_hcl.core.state_registry import resolve_titan_id
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

    # RFP_supervision_lifecycle §7.D / Phase D.1 — bus-INDEPENDENT save of the
    # neuromodulator learned state on any shutdown (SIGTERM/control-group/
    # PDEATHSIG). Guarded: not every system build exposes _save_state.
    from titan_hcl.core.worker_shutdown import register_shutdown_save
    register_shutdown_save(
        name,
        lambda: neuromod_system._save_state() if hasattr(neuromod_system, "_save_state") else None,
    )

    # Lazy slot writer — None if flag-off or shm setup fails.
    flag_on = bool(
        (full_config.get("microkernel") or {}).get("shm_neuromod_enabled", False))
    writer = None
    inputs_reader = None
    if flag_on:
        from titan_hcl.core.state_registry import NEUROMOD_STATE
        writer = _maybe_get_writer(NEUROMOD_STATE, titan_id)
        # §4.Q chunk Q6 — pair the inputs_reader with the writer so evaluate
        # always has cognitive_worker's latest state aggregate.
        inputs_reader = _maybe_get_inputs_reader(titan_id)

    # §4.Q chunk Q5 — DNA cache for movement clearance + apply_external_nudge
    # gating (developmental_age). Loaded once at boot from titan_params.toml
    # via the shared loader. NeuromodulatorSystem.evaluate uses this cache for
    # cross-coupling weights (set by compute_emergent_inputs in cognitive_worker
    # builder, but movement clearance reads movement_<MOD> keys here too).
    try:
        import tomllib
        _params_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "titan_params.toml")
        if os.path.exists(_params_path):
            with open(_params_path, "rb") as f:
                _dna_full = tomllib.load(f)
            neuromod_system._dna_cache = _dna_full.get("neuromodulator_dna", {})
            logger.info("[NeuromodWorker] DNA cache loaded (%d keys)",
                        len(neuromod_system._dna_cache))
    except Exception as e:
        logger.warning("[NeuromodWorker] DNA cache load failed: %s — using defaults", e)
        neuromod_system._dna_cache = {}

    # Boot signals (per reflex_worker.py + hormonal_worker.py pattern).
    boot_ts = time.time()
    # Phase 11 §11.I.2 — slot transition: starting → booted (D-SPEC-141 / v1.65.0).
    # MODULE_READY bus emit DELETED per locked D2 (no shim, no dual-publish).
    _WORKER_READY = True
    if _state_writer is not None:
        try:
            _state_writer.write_state("booted")
            logger.info(
                "[NeuromodWorker] Phase 11 §11.I.2 — SHM slot state=booted "
                "(awaiting MODULE_PROBE_REQUEST from titan_hcl)")
        except Exception as _swb_err:  # noqa: BLE001
            logger.warning(
                "[NeuromodWorker] Phase 11 write_state(booted) failed: %s",
                _swb_err)
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
    last_stats_publish_ts = 0.0
    epoch_count = 0
    error_count = 0
    slot_write_count = 0
    evaluate_count = 0
    nudge_count = 0
    nudge_by_source: dict[str, int] = {}

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
                        "evaluate_count": evaluate_count,
                        "slot_write_count": slot_write_count,
                        "nudge_count": nudge_count,
                        "nudge_by_source": dict(nudge_by_source),
                        "error_count": error_count,
                        "slot_writer_active": writer is not None,
                        "inputs_reader_active": inputs_reader is not None,
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

        # §4.Q chunk Q12 — 2.5s coalesced NEUROMOD_STATS_UPDATED publish.
        # Replaces the dead spirit_loop._publish_coord_subdomains path.
        # Cadence matches /v4/inner-trinity.neuromodulators + /status mood
        # route caching contract.
        if now - last_stats_publish_ts >= STATS_PUBLISH_INTERVAL_S:
            try:
                send_queue.put({
                    "type": bus.NEUROMOD_STATS_UPDATED,
                    "src": name,
                    "dst": "all",
                    "payload": _build_stats_payload(neuromod_system, titan_id),
                    "ts": now,
                })
            except Exception as _stats_err:
                logger.debug(
                    "[NeuromodWorker] NEUROMOD_STATS_UPDATED publish failed: %s",
                    _stats_err)
            # RFP_meta-reasoning_CGN_FIX.md §8 — clean 6D flat-vector
            # emit dedicated to emot_cgn substrate cache (closes
            # DEAD-DIM `neuromod_state`). Distinct from the verbose
            # NEUROMOD_STATS_UPDATED above which feeds dashboards with
            # the full (6,4) level/gain/phasic/tonic matrix.
            #
            # rFP_dead_dim_wiring_fix §2.D — read canonical per-modulator
            # `.level` from neuromod_system.modulators directly (same
            # source as _build_stats_payload above). The previous code
            # called `get_modulation()` which returns gain-related
            # modulation FACTORS (sensory_gain etc.) — never DA/5HT/...
            # level keys — so all 6 fallbacks fired and `levels_6d` was
            # [0.5]*6 every emit → bundle nonzero=6 std=0 (dead).
            try:
                _mods = getattr(neuromod_system, "modulators", None) or {}
                _levels = None
                if _mods:
                    # Order locked: DA, 5HT, NE, ACh, Endorphin, GABA —
                    # matches emot_bundle_protocol concat order.
                    _order = ("DA", "5HT", "NE", "ACh", "Endorphin", "GABA")
                    _levels = [
                        float(getattr(_mods.get(_k), "level", 0.5))
                        if _mods.get(_k) is not None else 0.5
                        for _k in _order
                    ]
                # Gate: only emit when at least one modulator level
                # diverges from the default 0.5 — avoids flooding emot_cgn
                # with all-default warmup pulses before NeuromodulatorSystem
                # has integrated any real evidence.
                if _levels and any(abs(v - 0.5) > 1e-6 for v in _levels):
                    send_queue.put({
                        "type": bus.NEUROMOD_LEVELS_UPDATE,
                        "src": name,
                        "dst": "emot_cgn",
                        "payload": {"levels_6d": _levels},
                        "ts": now,
                    })
            except Exception as _nl_err:
                logger.debug(
                    "[NeuromodWorker] NEUROMOD_LEVELS_UPDATE publish failed: %s",
                    _nl_err)
            last_stats_publish_ts = now

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
                    "[NeuromodWorker] MODULE_PROBE_REQUEST handler failed: %s",
                    _probe_err)
            continue

        # B.2.1 supervision-transfer dispatch (mirrors reflex_worker).
        try:
            from titan_hcl.core import worker_swap_handler as _swap
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

        # §4.Q chunk Q7 — NEUROMOD_EXTERNAL_NUDGE subscriber. Producers:
        # cognitive_worker × 6 (MSL ×2, FILTER_DOWN ×2, META eureka,
        # SPIRIT_SELF) + outer_interface_worker × 1 (self-exploration novelty).
        if msg_type == bus.NEUROMOD_EXTERNAL_NUDGE:
            try:
                payload = msg.get("payload") or {}
                # Payload may arrive msgpack-encoded under socket-mode bus
                # (Phase B.2 §C7). Decode best-effort.
                if isinstance(payload, (bytes, bytearray)):
                    try:
                        import msgpack as _msgpack
                        payload = _msgpack.unpackb(payload, raw=False)
                    except Exception:
                        payload = {}
                if _apply_external_nudge_payload(neuromod_system, payload):
                    nudge_count += 1
                    _src = str(payload.get("source", "unknown"))
                    nudge_by_source[_src] = nudge_by_source.get(_src, 0) + 1
            except Exception as e:
                error_count += 1
                logger.warning(
                    "[NeuromodWorker] NEUROMOD_EXTERNAL_NUDGE handling failed: %s", e)
            continue

        # §4.Q chunk Q6 — KERNEL_EPOCH_TICK drives evaluate.
        # Read neuromod_inputs.bin → set_chi_health → evaluate → movement
        # clearance → write neuromod_state.bin (4-field v2 layout).
        is_epoch_tick = msg_type == getattr(bus, "KERNEL_EPOCH_TICK", "KERNEL_EPOCH_TICK") \
            or msg_type == getattr(bus, "EPOCH_TICK", "EPOCH_TICK")
        if is_epoch_tick:
            epoch_count += 1
            ran, _kin_overrides_n = _drive_evaluate(neuromod_system, inputs_reader)
            if ran:
                evaluate_count += 1

            # Write expanded neuromod_state.bin (v2 4-field layout) per tick.
            # SLOT_WRITE_MIN_INTERVAL_S throttling preserved from pre-§4.Q.
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

        # Other messages no-op — bus dispatch fall-through.
