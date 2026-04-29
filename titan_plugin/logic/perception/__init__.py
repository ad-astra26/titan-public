"""
titan_plugin/logic/perception/ — SensoryHub: Extensible Multi-Modal Perception Framework.

Registry pattern for sensory modalities with wiring validation, index remapping,
and unified routing. All heavy computation stays in MediaWorker — this module
is a pure router (~5μs per process() call, zero computation).

Usage in spirit_worker:
    from titan_plugin.logic.perception import SensoryHub
    hub = SensoryHub()
    hub.auto_discover()
    hub.process(msg_type, payload, source, outer_state, body_state, mind_state)
"""
import importlib
import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Optional
from titan_plugin import bus

logger = logging.getLogger(__name__)

# ── Protected dimensions — wiring rules ──────────────────────────────

# Hard Rule R1: outer_mind_15d Willing [10:15] is GROUND_UP exclusive
PROTECTED_WILLING = set(range(10, 15))

# Valid target names for modalities
VALID_TARGETS = {"outer_body", "outer_mind_feeling"}

# Hard rule bounds
MAX_SELF_STRENGTH = 0.05
MAX_EXTERNAL_STRENGTH = 0.035
MAX_CLAMP = 0.08
MAX_FEATURES_PER_MODALITY = 50

# rFP_phase1_sensory_wiring (2026-04-23): creation_nudge was unbounded
# accumulator — every SENSE event pushed outer_body[2] by +0.03 with no
# decay, so it saturated to 1.0 permanently on any creating Titan. Fix:
# exponential mean-revert toward 0.5 with ~5min half-life BEFORE each
# new nudge. Lets sensory activity show up as real signal, but idle
# periods return the dim to baseline.
_CREATION_NUDGE_HALF_LIFE_S = 300.0  # 5 minutes
_CREATION_NUDGE_BASELINE = 0.5
_creation_nudge_last_ts = 0.0
_creation_nudge_lock = threading.Lock()


def _apply_creation_nudge_with_decay(
        current: float, nudge: float, now: Optional[float] = None) -> float:
    """Exponential mean-revert toward baseline (0.5) + new nudge.

    Prevents permanent saturation of outer_body[2] somatosensation from
    accumulating creation_nudges across thousands of perception events.

    Args:
        current: current outer_body[2] value
        nudge: the new event's creation_nudge magnitude
        now: wall-clock time (injectable for testing)
    Returns:
        new outer_body[2] value, clamped [0, 1]
    """
    global _creation_nudge_last_ts
    if now is None:
        now = time.time()

    with _creation_nudge_lock:
        last_ts = _creation_nudge_last_ts
        _creation_nudge_last_ts = now

    # Decay toward baseline based on time since last event.
    # decay_factor: 1.0 at dt=0 (no decay), → 0.0 as dt → inf
    if last_ts > 0 and now > last_ts:
        dt = now - last_ts
        # Use half-life-based exponential: decay_factor = 2^(-dt/half_life)
        decay_factor = math.pow(2.0, -dt / _CREATION_NUDGE_HALF_LIFE_S)
    else:
        decay_factor = 1.0  # First event, no prior timestamp

    # Mean-revert current toward baseline
    decayed = _CREATION_NUDGE_BASELINE + (current - _CREATION_NUDGE_BASELINE) * decay_factor

    # Apply new event nudge
    return max(0.0, min(1.0, decayed + nudge))


def _reset_creation_nudge_for_testing() -> None:
    """Test helper — clears decay timer state."""
    global _creation_nudge_last_ts
    with _creation_nudge_lock:
        _creation_nudge_last_ts = 0.0


# ── Data classes ─────────────────────────────────────────────────────

@dataclass
class TargetSpec:
    """Where a feature group writes in the Trinity."""
    target: str                    # "outer_body" | "outer_mind_feeling"
    index_remap: dict[int, int]    # {target_dim: source_idx} — e.g. {0: 4, 1: 1, 3: 3, 4: 0}
    clamp: float = 0.05           # max absolute delta per event
    strength_multiplier: float = 1.0  # multiplied with base strength


@dataclass
class ModulationSpec:
    """Secondary modulation of a primary target."""
    source_group: str    # feature group name, e.g. "spatial"
    source_idx: int      # index within the feature group
    target: str          # "outer_body"
    target_idx: int      # dimension index in target array
    strength_mult: float # multiplier on base strength (e.g., 0.5)
    clamp: float = 0.03 # max absolute delta


@dataclass
class SensoryModality:
    """Declaration of a sensory modality for SensoryHub registration."""
    name: str                              # "visual", "audio"
    msg_types: set[str]                    # {"SENSE_VISUAL"}
    payload_key: str                       # "features_30d", "features_15d"
    feature_groups: dict[str, int]         # {"physical": 5, "pattern": 5, ...}
    target_map: dict[str, TargetSpec]      # group_name → TargetSpec
    modulations: list[ModulationSpec] = field(default_factory=list)
    self_strength: float = 0.04
    external_strength: float = 0.025
    creation_nudge_self: float = 0.03
    creation_nudge_external: float = 0.015


# ── Validation ───────────────────────────────────────────────────────

class WiringViolation(Exception):
    """Hard wiring rule violated — registration rejected."""
    pass


def validate_modality(modality: SensoryModality,
                      existing: dict[str, SensoryModality]) -> list[str]:
    """
    Validate a modality against wiring rules.

    Returns list of warnings (soft rules).
    Raises WiringViolation for hard rule violations.
    """
    warnings = []

    # R1: No Willing writes
    for group_name, spec in modality.target_map.items():
        if spec.target == "outer_mind_feeling":
            for target_dim in spec.index_remap.keys():
                if target_dim in PROTECTED_WILLING:
                    raise WiringViolation(
                        f"[R1] {modality.name}.{group_name} targets Willing dim {target_dim} "
                        f"— GROUND_UP exclusive")

    # R2: No outer_spirit writes
    for group_name, spec in modality.target_map.items():
        if "outer_spirit" in spec.target:
            raise WiringViolation(
                f"[R2] {modality.name}.{group_name} targets outer_spirit — derived tensor")

    # R3: No inner_body direct writes
    for group_name, spec in modality.target_map.items():
        if "inner" in spec.target or "body_state" in spec.target:
            raise WiringViolation(
                f"[R3] {modality.name}.{group_name} targets inner dimension — boundary crossing")

    for mod in modality.modulations:
        if "outer_spirit" in mod.target or "inner" in mod.target:
            raise WiringViolation(
                f"[R2/R3] {modality.name} modulation targets {mod.target}")

    # R4: Strength bounds
    if modality.self_strength > MAX_SELF_STRENGTH:
        raise WiringViolation(
            f"[R4] {modality.name} self_strength {modality.self_strength} > {MAX_SELF_STRENGTH}")
    if modality.external_strength > MAX_EXTERNAL_STRENGTH:
        raise WiringViolation(
            f"[R4] {modality.name} external_strength {modality.external_strength} > {MAX_EXTERNAL_STRENGTH}")

    # R5: Clamp bounds
    for group_name, spec in modality.target_map.items():
        if spec.clamp > MAX_CLAMP:
            raise WiringViolation(
                f"[R5] {modality.name}.{group_name} clamp {spec.clamp} > {MAX_CLAMP}")
    for mod in modality.modulations:
        if mod.clamp > MAX_CLAMP:
            raise WiringViolation(
                f"[R5] {modality.name} modulation clamp {mod.clamp} > {MAX_CLAMP}")

    # R6: Target validity
    for group_name, spec in modality.target_map.items():
        if spec.target not in VALID_TARGETS:
            raise WiringViolation(
                f"[R6] {modality.name}.{group_name} target '{spec.target}' not in {VALID_TARGETS}")

    # S1: Multi-writer warning
    for group_name, spec in modality.target_map.items():
        for ex_name, ex_mod in existing.items():
            for ex_group, ex_spec in ex_mod.target_map.items():
                if ex_spec.target == spec.target:
                    overlap = set(spec.index_remap.keys()) & set(ex_spec.index_remap.keys())
                    if overlap:
                        warnings.append(
                            f"[S1] {modality.name}.{group_name} and {ex_name}.{ex_group} "
                            f"both write to {spec.target} dims {overlap}")

    # S2: Feature count
    total = sum(modality.feature_groups.values())
    if total > MAX_FEATURES_PER_MODALITY:
        warnings.append(
            f"[S2] {modality.name} has {total}D features (> {MAX_FEATURES_PER_MODALITY}D)")

    return warnings


# ── SensoryHub ───────────────────────────────────────────────────────

# PERSISTENCE_BY_DESIGN: SensoryHub._registry / _msg_type_map are the
# sensory-modality dispatch table built up via explicit register() calls
# at boot. Not state to persist — re-registered every run.
class SensoryHub:
    """
    Extensible multi-modal perception router.

    Registers sensory modalities, validates wiring, and routes pre-computed
    features to Trinity dimensions. ZERO computation — pure arithmetic routing.
    """

    def __init__(self):
        self._registry: dict[str, SensoryModality] = {}
        self._msg_type_map: dict[str, str] = {}  # msg_type → modality name

    def register(self, modality: SensoryModality) -> list[str]:
        """
        Register a sensory modality. Validates wiring rules.

        Returns list of warnings. Raises WiringViolation on hard rule failure.
        """
        warnings = validate_modality(modality, self._registry)

        self._registry[modality.name] = modality
        for mt in modality.msg_types:
            self._msg_type_map[mt] = modality.name

        if warnings:
            for w in warnings:
                logger.warning("[SensoryHub] %s", w)

        logger.info("[SensoryHub] Registered '%s': %dD, targets=%s, msg_types=%s",
                    modality.name, sum(modality.feature_groups.values()),
                    list(modality.target_map.keys()), modality.msg_types)
        return warnings

    def auto_discover(self) -> int:
        """
        Scan perception/ directory for modality declarations.
        Each file should export a MODALITY constant of type SensoryModality.
        Returns count of successfully registered modalities.
        """
        perception_dir = os.path.dirname(__file__)
        count = 0
        for fname in sorted(os.listdir(perception_dir)):
            if fname.startswith("_") or not fname.endswith("_modality.py"):
                continue
            module_name = fname[:-3]  # strip .py
            try:
                mod = importlib.import_module(f"titan_plugin.logic.perception.{module_name}")
                modality = getattr(mod, "MODALITY", None)
                if isinstance(modality, SensoryModality):
                    self.register(modality)
                    count += 1
                else:
                    logger.debug("[SensoryHub] %s has no MODALITY constant", fname)
            except WiringViolation as e:
                logger.error("[SensoryHub] REJECTED %s: %s", fname, e)
            except Exception as e:
                logger.warning("[SensoryHub] Failed to load %s: %s", fname, e)
        return count

    def reload(self) -> int:
        """Atomically reload all modalities. Old registry stays if any error."""
        new_hub = SensoryHub()
        count = new_hub.auto_discover()
        # Atomic swap
        self._registry = new_hub._registry
        self._msg_type_map = new_hub._msg_type_map
        logger.info("[SensoryHub] Reloaded: %d modalities", count)
        return count

    def list_modalities(self) -> list[str]:
        return list(self._registry.keys())

    def process(self, msg_type: str, payload: dict, source: str,
                outer_state: dict, body_state: dict, mind_state: dict) -> bool:
        """
        Route pre-computed features to Trinity dimensions.

        Returns True if a modality was found and processed, False otherwise.
        This is the HOT PATH — must be < 10 microseconds.
        """
        modality_name = self._msg_type_map.get(msg_type)
        if not modality_name:
            return False

        modality = self._registry.get(modality_name)
        if not modality:
            return False

        # Read features from payload
        features = payload.get(modality.payload_key)
        if not features:
            # Try 5D fallback
            features_5d = payload.get("features")
            if features_5d and len(features_5d) >= 5:
                self._apply_5d_fallback(features_5d, msg_type, source, outer_state)
                return True
            return False

        strength = modality.self_strength if source == "self" else modality.external_strength

        # Collect target arrays (read once, modify, write back once)
        targets_modified: dict[str, list] = {}

        def _get_target(target_name: str) -> list | None:
            if target_name in targets_modified:
                return targets_modified[target_name]
            if target_name == "outer_body":
                arr = outer_state.get("outer_body", [0.5] * 5)
            elif target_name == "outer_mind_feeling":
                arr = outer_state.get("outer_mind_15d")
                if arr is None or len(arr) < 10:
                    return None
            else:
                return None
            targets_modified[target_name] = arr
            return arr

        # Apply feature groups via index_remap
        for group_name, target_spec in modality.target_map.items():
            group_values = features.get(group_name, [])
            if not group_values:
                continue

            target_array = _get_target(target_spec.target)
            if target_array is None:
                logger.debug("[SensoryHub] %s: target %s not initialized, skipping %s",
                             modality.name, target_spec.target, group_name)
                continue

            group_strength = strength * target_spec.strength_multiplier
            for target_dim, source_idx in target_spec.index_remap.items():
                if source_idx < len(group_values) and target_dim < len(target_array):
                    delta = (group_values[source_idx] - target_array[target_dim]) * group_strength
                    delta = max(-target_spec.clamp, min(target_spec.clamp, delta))
                    target_array[target_dim] = max(0.0, min(1.0, target_array[target_dim] + delta))

        # Apply modulations
        for mod in modality.modulations:
            source_vals = features.get(mod.source_group, [])
            if not source_vals or mod.source_idx >= len(source_vals):
                continue
            target_array = _get_target(mod.target)
            if target_array is None or mod.target_idx >= len(target_array):
                continue
            mod_strength = strength * mod.strength_mult
            delta = (source_vals[mod.source_idx] - target_array[mod.target_idx]) * mod_strength
            delta = max(-mod.clamp, min(mod.clamp, delta))
            target_array[mod.target_idx] = max(0.0, min(1.0, target_array[mod.target_idx] + delta))

        # Creation nudge (somatosensation) — decay-aware (rFP Phase 1)
        nudge = modality.creation_nudge_self if source == "self" else modality.creation_nudge_external
        if nudge > 0:
            ob = _get_target("outer_body")
            if ob and len(ob) > 2:
                ob[2] = _apply_creation_nudge_with_decay(ob[2], nudge)

        # Write back all modified targets (once per target)
        for target_name, arr in targets_modified.items():
            if target_name == "outer_body":
                outer_state["outer_body"] = arr
            elif target_name == "outer_mind_feeling":
                outer_state["outer_mind_15d"] = arr

        # Unified logging
        ob = outer_state.get("outer_body", [0.5] * 5)
        om = outer_state.get("outer_mind_15d")
        om_str = ""
        if om and len(om) >= 10 and "outer_mind_feeling" in targets_modified:
            om_str = " oMfeel=[%.3f,%.3f,%.3f,%.3f,%.3f]" % (om[5], om[6], om[7], om[8], om[9])
        logger.info("[SENSE_%s] oBody=[%.3f,%.3f,%.3f,%.3f,%.3f]%s (src=%s/%s)",
                    modality.name.upper(),
                    ob[0], ob[1], ob[2], ob[3], ob[4], om_str,
                    source, payload.get("filename", "?"))
        return True

    @staticmethod
    def _apply_5d_fallback(features: list, msg_type: str, source: str,
                           outer_state: dict) -> None:
        """Legacy 5D enrichment path — hardcoded, not declarative."""
        strength = 0.04 if source == "self" else 0.025
        ob = outer_state.get("outer_body", [0.5] * 5)

        # Audio-specific remap: dim[3] ← features[2] (rhythm)
        # Visual remap: dim[3] ← features[3] (spatial_freq)
        remap = {
            0: features[4],  # interoception ← harmony
            1: features[1],  # proprioception ← structure
            3: features[2] if msg_type == bus.SENSE_AUDIO else features[3],  # entropy
            4: features[0],  # thermal ← warmth
        }
        for dim, target in remap.items():
            delta = (target - ob[dim]) * strength
            delta = max(-0.05, min(0.05, delta))
            ob[dim] = max(0.0, min(1.0, ob[dim] + delta))
        ob[2] = min(1.0, ob[2] + 0.03)
        outer_state["outer_body"] = ob

        logger.info("[SENSE_5D_FALLBACK] %s: oBody=[%.3f,%.3f,%.3f,%.3f,%.3f] (src=%s)",
                    msg_type, ob[0], ob[1], ob[2], ob[3], ob[4], source)
