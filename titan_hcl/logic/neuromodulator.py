"""
titan_hcl/logic/neuromodulator.py — Self-Learning Neuromodulator Meta-Layer.

6 neuromodulators that adjust PARAMETERS of existing systems (not signals):
  DA:        Reward prediction error → learning rates + thresholds
  5-HT:     Patience/stability → temporal discount + emotional baseline
  NE:       Unexpected uncertainty → gain + explore/exploit
  ACh:      Expected uncertainty → learning rate + memory encoding
  Endorphin: Intrinsic reward/flow → motivation + discomfort suppression
  GABA:     Global inhibition/calm → all thresholds raised

Self-regulation (3 layers, NO hardcoding):
  1. Autoreceptors (fast): negative feedback prevents runaway levels
  2. Homeostatic density (slow): sensitivity adapts to chronic over/under-activation
  3. Allostatic set-point (very slow): "normal" drifts → personality evolution

Emotions EMERGE as patterns across all 6 modulators + hormones + 130D state.
We don't program joy — joy arises when DA↑ + 5-HT↑ + NE↓ + Endorphin↑.
"""
import json
import logging
import math
import os
import time
from typing import Optional
from titan_hcl.utils.silent_swallow import swallow_warn

logger = logging.getLogger(__name__)

# Clearance rates — determine temporal dynamics of each modulator
# Higher = faster clearance = sharper signal; Lower = smoother/slower
CLEARANCE_RATES = {
    "DA": 0.3,         # Fast → sharp reward signals
    "5HT": 0.05,       # Slow → stable mood baseline
    "NE": 0.2,         # Moderate → arousal dynamics
    "ACh": 0.5,        # Fastest → precise attention shifts
    "Endorphin": 0.08,  # Slow → sustained flow states
    "GABA": 0.15,      # Moderate → calming dynamics
}

# Cross-coupling matrix: [source][target] = weight
# Positive = facilitatory, Negative = inhibitory
# NOTE (2026-03-22): Reduced 5HT↔GABA mutual facilitation and GABA→NE suppression.
# Original values (+0.20, -0.15) created a positive feedback loop → bliss lock.
# Biological brains counteract this via metabolic cost (planned: Chi-governed production).
# Until metabolic regulation is implemented, halved coupling prevents runaway saturation.
COUPLING_MATRIX = {
    "DA":        {"5HT": -0.20, "NE": +0.10, "ACh": +0.10, "Endorphin": +0.15, "GABA": -0.07},
    "5HT":       {"DA": -0.15, "NE": -0.08, "ACh": +0.05, "Endorphin": +0.10, "GABA": +0.10},
    "NE":        {"DA": +0.10, "5HT": -0.10, "ACh": +0.15, "Endorphin": -0.10, "GABA": -0.09},
    "ACh":       {"DA": +0.10, "5HT": +0.05, "NE": +0.15, "Endorphin": +0.05, "GABA": -0.05},
    "Endorphin": {"DA": +0.15, "5HT": +0.10, "NE": -0.08, "ACh": +0.05, "GABA": -0.11},
    "GABA":      {"DA": -0.10, "5HT": +0.10, "NE": -0.08, "ACh": -0.05, "Endorphin": -0.15},
}


# Relative metabolic costs per modulator (normalized: Endorphin=1.0 reference)
# Based on biological synthesis complexity: peptides > monoamines > amino acids
METABOLIC_COSTS = {
    "Endorphin": 1.0,    # Peptide synthesis — most expensive
    "5HT": 0.75,         # Tryptophan synthesis
    "NE": 0.625,         # DA derivative + extra step
    "DA": 0.5,           # Tyrosine hydroxylation
    "ACh": 0.5,          # Choline + acetyl-CoA
    "GABA": 0.375,       # Glutamate conversion — cheapest
}

# Per-modulator allostatic set-point bounds (rFP_meditation_emergent_onset_redesign
# §2.7 / D-SPEC-107). The legacy global clamp [0.3, 0.7] floored ABOVE GABA's true
# resting baseline (~0.15) → GABA's setpoint could never converge → gain stuck
# <1.0 → homeostat read GABA as eternally dysregulated, poisoning any
# homeostatic-balance signal. GABA's floor now sits below its baseline so its
# gain can center at ~1.0 at rest; the rest keep legacy bounds (unchanged).
SETPOINT_BOUNDS: dict[str, tuple[float, float]] = {
    "DA": (0.3, 0.7),
    "5HT": (0.3, 0.7),
    "NE": (0.3, 0.7),
    "ACh": (0.3, 0.7),
    "Endorphin": (0.3, 0.7),
    "GABA": (0.10, 0.7),
}
# Per-modulator initial set-point (fresh boot only — persisted state overrides
# on restart). GABA seeded at its true resting baseline so fresh Titans read
# "balanced" at rest; on existing Titans the allostatic drift converges the
# restored setpoint down into the widened bound over time.
INITIAL_SETPOINTS: dict[str, float] = {
    "GABA": 0.15,
}

# Pressure rate: how much metabolic pressure per unit production
# Calibrated: 6 modulators × avg 0.5 production × 0.6 avg cost × rate × 800 ticks ≈ 0.4
NEUROMOD_PRESSURE_RATE = 0.00045


# PERSISTENCE_BY_DESIGN: Neuromodulator._peak_level / _trough_level are
# dynamic observability metrics that recompute from the live level stream;
# restoring them from stale disk state would be misleading. Only level /
# tonic / sensitivity / setpoint are restored via restore_state.
class Neuromodulator:
    """A single neuromodulator with self-regulating dynamics.

    Three-layer self-regulation:
    1. Autoreceptor: fast negative feedback (prevents runaway)
    2. Homeostatic: slow sensitivity adaptation (tolerance/sensitization)
    3. Allostatic: very slow set-point drift (personality evolution)
    """

    def __init__(
        self,
        name: str,
        clearance_rate: float = 0.1,
        autoreceptor_gain: float = 2.0,
        initial_level: float = 0.5,
        initial_sensitivity: float = 1.0,
        initial_setpoint: float = 0.5,
        homeo_lr: float = 0.002,
        allo_lr: float = 0.0002,
        setpoint_min: float = 0.3,
        setpoint_max: float = 0.7,
    ):
        self.name = name
        self.clearance_rate = clearance_rate
        self.autoreceptor_gain = autoreceptor_gain

        # Core state
        self.level: float = initial_level
        self.tonic_level: float = initial_level  # EMA baseline
        self.phasic_level: float = 0.0  # Event-driven spike

        # Self-regulation
        self.sensitivity: float = initial_sensitivity  # receptor density analogue
        self.setpoint: float = initial_setpoint  # what "normal" means
        self._homeo_lr = homeo_lr
        self._allo_lr = allo_lr
        # Per-modulator allostatic set-point bounds. The legacy global clamp
        # [0.3, 0.7] floored ABOVE GABA's true resting baseline (~0.15), so
        # GABA's setpoint could never converge to where its level actually
        # lives → gain permanently <1.0 → the homeostat read GABA as eternally
        # dysregulated (rFP_meditation_emergent_onset_redesign §2.7). Bounds are
        # now per-modulator so a low-resting modulator (GABA) can center its
        # gain at ~1.0 at its real baseline. Defaults preserve legacy behavior.
        self._setpoint_min = setpoint_min
        self._setpoint_max = setpoint_max
        # Clamp the (possibly default-0.5) initial setpoint into the bounds.
        self.setpoint = max(self._setpoint_min, min(self._setpoint_max, self.setpoint))

        # History for adaptation
        self._activation_history: list[float] = []
        self._history_max = 200

        # Stats
        self._total_updates = 0
        self._peak_level = initial_level
        self._trough_level = initial_level
        self._last_production = 0.0  # For metabolic pressure reporting

    def update(self, input_signal: float, cross_coupling: float = 0.0,
               dt: float = 1.0, chi_health: float = 1.0) -> float:
        """Update neuromodulator level from input + cross-coupling.

        Args:
            input_signal: raw input (0-1) from Titan's state
            cross_coupling: sum of coupling influences from other modulators
            dt: time delta (clamped to prevent explosion)

        Returns:
            Current level after update
        """
        dt = min(dt, 30.0)
        self._total_updates += 1

        # ── Layer 1: Autoreceptor feedback ──
        # Production suppressed by own level (negative feedback)
        effective_input = input_signal + cross_coupling
        effective_input = max(0.0, min(2.0, effective_input))
        production = effective_input * (
            1.0 / (1.0 + self.autoreceptor_gain * self.level)
        )

        # Metabolic gating: production costs Chi energy
        # chi_health ∈ [0.1, 1.0] — low Chi → reduced production (survival baseline)
        production *= max(0.1, chi_health)
        self._last_production = production

        # Clearance proportional to current level
        # During dreaming: boost governed by GABA (set by spirit_worker)
        _dream_boost = getattr(self, '_dream_clearance_boost', 1.0)
        clearance = self.clearance_rate * self.level * _dream_boost

        # Net change — biological clearance separation (2026-03-24)
        # Sensitivity only gates production (receptor-mediated response).
        # Clearance always runs at full rate (physical reuptake/MAO process).
        # When tolerance develops (sensitivity↓), production drops but clearance
        # continues, naturally pulling levels toward equilibrium.
        # OLD: delta = (production - clearance) * dt * self.sensitivity
        delta = production * self.sensitivity * dt - clearance * dt
        self.level += delta
        self.level = max(0.0, min(1.0, self.level))

        # Soft bound resistance: exponential pull-back prevents saturation.
        # Linear 10% was too weak — 5-HT/GABA/Endorphin still saturated at
        # 0.99 over long runs, causing bliss lock (NE suppressed to floor).
        # Exponential: gentle at 0.85, strong at 0.95, overwhelming at 0.99.
        if self.level > 0.85:
            excess = self.level - 0.85
            self.level -= excess * excess * 3.0  # quadratic: 0.85→0%, 0.90→0.75%, 0.95→3%, 0.99→5.9%
            self.level = max(0.5, self.level)  # never slam below midpoint
        elif self.level < 0.15:
            deficit = 0.15 - self.level
            self.level += deficit * deficit * 3.0
            self.level = min(0.5, self.level)

        # Update tonic baseline (slow EMA)
        tonic_tau = 50.0
        alpha = 1.0 / tonic_tau
        self.tonic_level += alpha * (self.level - self.tonic_level)

        # Phasic = deviation from tonic
        self.phasic_level = self.level - self.tonic_level

        # Track peaks/troughs
        if self.level > self._peak_level:
            self._peak_level = self.level
        if self.level < self._trough_level:
            self._trough_level = self.level

        # ── Layer 2: Homeostatic sensitivity adaptation (slow) ──
        self._activation_history.append(self.level)
        if len(self._activation_history) > self._history_max:
            self._activation_history.pop(0)

        if len(self._activation_history) >= 50:
            avg_activation = sum(self._activation_history[-50:]) / 50
            # If chronically over setpoint → reduce sensitivity (tolerance)
            # If chronically under setpoint → increase sensitivity
            homeo_delta = (self.setpoint - avg_activation) * self._homeo_lr
            self.sensitivity += homeo_delta
            self.sensitivity = max(0.1, min(3.0, self.sensitivity))

        # ── Layer 3: Allostatic set-point drift (very slow) ──
        if len(self._activation_history) >= 100:
            long_avg = sum(self._activation_history[-100:]) / 100
            allo_delta = (long_avg - self.setpoint) * self._allo_lr
            self.setpoint += allo_delta
            self.setpoint = max(self._setpoint_min, min(self._setpoint_max, self.setpoint))

        return self.level

    def get_gain(self, base_value: float = 1.0) -> float:
        """Get multiplicative gain factor for modulating a downstream parameter.

        Returns a value in [0.3, 3.0] centered around 1.0 at setpoint.
        """
        deviation = (self.level - self.setpoint) / max(0.01, self.setpoint)
        gain = base_value * (1.0 + deviation * self.sensitivity)
        return max(0.3, min(3.0, gain))

    def get_state(self) -> dict:
        return {
            "level": round(self.level, 4),
            "tonic": round(self.tonic_level, 4),
            "phasic": round(self.phasic_level, 4),
            "sensitivity": round(self.sensitivity, 4),
            "setpoint": round(self.setpoint, 4),
            "peak": round(self._peak_level, 4),
            "trough": round(self._trough_level, 4),
        }

    def restore_state(self, state: dict) -> None:
        self.level = state.get("level", self.level)
        self.tonic_level = state.get("tonic", self.tonic_level)
        self.sensitivity = state.get("sensitivity", self.sensitivity)
        self.setpoint = state.get("setpoint", self.setpoint)


# PERSISTENCE_BY_DESIGN: NeuromodulatorSystem._total_evaluations is a
# cumulative observability counter. Saved for debugging (in get_state) but
# reset on boot is acceptable — the live neuromod levels themselves are
# what actually persist.
class NeuromodulatorSystem:
    """Manages all 6 neuromodulators with cross-coupling and emotion detection."""

    def __init__(self, data_dir: str = "./data/neuromodulator"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        # Initialize all 6 modulators
        self.modulators: dict[str, Neuromodulator] = {}
        for name, clearance in CLEARANCE_RATES.items():
            _smin, _smax = SETPOINT_BOUNDS.get(name, (0.3, 0.7))
            self.modulators[name] = Neuromodulator(
                name=name,
                clearance_rate=clearance,
                initial_setpoint=INITIAL_SETPOINTS.get(name, 0.5),
                setpoint_min=_smin,
                setpoint_max=_smax,
            )

        self._total_evaluations = 0
        self._current_emotion: str = "neutral"
        self._emotion_confidence: float = 0.0
        self._chi_health: float = 1.0
        self._neuromod_pressure: float = 0.0

        # Load persisted state
        self._load_state()

        logger.info(
            "[NeuromodulatorSystem] Initialized: %d modulators (%s)",
            len(self.modulators), ", ".join(self.modulators.keys()))

    def evaluate(self, inputs: dict[str, float], dt: float = 1.0) -> dict:
        """Evaluate all neuromodulators with cross-coupling.

        Args:
            inputs: {modulator_name: input_signal} from Titan's state
            dt: time delta

        Returns:
            {modulator_name: {"level", "gain", "phasic"}}
        """
        self._total_evaluations += 1

        # Compute cross-coupling for each modulator
        cross_couplings = {}
        for target_name in self.modulators:
            coupling_sum = 0.0
            for source_name, source_mod in self.modulators.items():
                if source_name == target_name:
                    continue
                weight = COUPLING_MATRIX.get(source_name, {}).get(target_name, 0.0)
                coupling_sum += weight * source_mod.level
            cross_couplings[target_name] = coupling_sum

        # Update each modulator (with metabolic gating from Chi health)
        results = {}
        for name, mod in self.modulators.items():
            input_signal = inputs.get(name, 0.5)
            mod.update(input_signal, cross_couplings.get(name, 0.0), dt,
                       chi_health=self._chi_health)
            results[name] = {
                "level": mod.level,
                "gain": mod.get_gain(),
                "phasic": mod.phasic_level,
                "tonic": mod.tonic_level,
            }

        # Compute aggregate neurochemical pressure for metabolic system
        self._neuromod_pressure = sum(
            getattr(mod, '_last_production', 0) * METABOLIC_COSTS.get(name, 0.5)
            for name, mod in self.modulators.items()
        ) * NEUROMOD_PRESSURE_RATE

        # Detect emergent emotion from pattern
        self._current_emotion, self._emotion_confidence = self._detect_emotion()

        # Auto-save periodically
        if self._total_evaluations % 100 == 0:
            self._save_state()

        return results

    def apply_external_nudge(
        self,
        nudge_map: dict,
        max_delta: float = 0.015,
        developmental_age: float = 0.0,
    ) -> None:
        """Apply gentle external nudge to modulators (e.g., from visual resonance).

        Each entry in nudge_map is {modulator_name: target_value}.
        The nudge pulls the modulator TOWARD the target, not additive.
        This is self-correcting: overshoot naturally reverses direction.

        Safety:
        - GABA is ALWAYS excluded (hard gate — bliss-lock prevention)
        - Developmental gate: no nudge before age 0.1 (system must stabilize first)
        - Max delta clamped to [-max_delta, +max_delta] per modulator per call
        - Autoreceptor/homeostatic/allostatic layers are NOT modified
        """
        if developmental_age < 0.1:
            return
        if not nudge_map:
            return

        for mod_name, target_value in nudge_map.items():
            if mod_name == "GABA":
                continue  # NEVER nudge GABA
            if mod_name not in self.modulators:
                continue

            mod = self.modulators[mod_name]
            delta = (target_value - mod.level) * max_delta
            delta = max(-max_delta, min(max_delta, delta))
            mod.level = max(0.0, min(1.0, mod.level + delta))

    def set_chi_health(self, chi_health: float) -> None:
        """Set Chi health factor for metabolic gating. Called from spirit_worker."""
        self._chi_health = max(0.1, min(1.0, chi_health))

    def get_modulation(self) -> dict:
        """Get current modulation gains for downstream systems.

        Returns dict of parameter multipliers that other systems should apply.

        Refactored 2026-05-15 (§4.Q): delegates byte-identical math to the
        module-level pure helper `compute_modulation_from_state` so the
        cross-process consumer in cognitive_worker (which reads the 4-field
        neuromod_state.bin SHM slot) gets the same modulation dict without
        access to the live NeuromodulatorSystem instance.
        """
        state = {
            name: {
                "level": mod.level,
                "gain": mod.get_gain(),
                "phasic": mod.phasic_level,
                "tonic": mod.tonic_level,
            }
            for name, mod in self.modulators.items()
        }
        return compute_modulation_from_state(state)

    def _detect_emotion(self) -> tuple[str, float]:
        """Detect emergent emotion from neuromodulator pattern.

        Maps current modulator levels to the closest emotion template.
        Returns (emotion_name, confidence). Thin wrapper over the pure
        module-level helper `detect_emotion_from_levels` so cross-process
        consumers (e.g. social_worker reading neuromod_state.bin via
        ShmReaderBank under D-SPEC-66 PLAN §1.4) can compute the same
        result from the SHM-published level dict without the live
        NeuromodulatorSystem instance.
        """
        levels = {name: mod.level for name, mod in self.modulators.items()}
        return detect_emotion_from_levels(levels)

    def get_stats(self) -> dict:
        return {
            "total_evaluations": self._total_evaluations,
            "current_emotion": self._current_emotion,
            "emotion_confidence": self._emotion_confidence,
            "modulators": {
                name: mod.get_state()
                for name, mod in self.modulators.items()
            },
            "modulation": self.get_modulation(),
        }

    def get_state(self) -> dict:
        """Return ALL mutable state needed for exact hot-reload reconstruction."""
        return {
            "data_dir": self.data_dir,
            "total_evaluations": self._total_evaluations,
            "current_emotion": self._current_emotion,
            "emotion_confidence": self._emotion_confidence,
            "modulators": {
                name: {
                    **mod.get_state(),
                    "activation_history": list(mod._activation_history),
                    "total_updates": mod._total_updates,
                }
                for name, mod in self.modulators.items()
            },
        }

    @classmethod
    def from_state(cls, state: dict) -> "NeuromodulatorSystem":
        """Create a fully reconstructed NeuromodulatorSystem from a state dict."""
        obj = cls.__new__(cls)
        obj.data_dir = state.get("data_dir", "./data/neuromodulator")
        os.makedirs(obj.data_dir, exist_ok=True)

        # Initialize modulators with default clearance rates + per-modulator
        # set-point bounds (rFP_meditation §2.7). restore_state below overrides
        # level/setpoint/sensitivity from persisted state, but the bounds govern
        # subsequent allostatic drift.
        obj.modulators = {}
        for name, clearance in CLEARANCE_RATES.items():
            _smin, _smax = SETPOINT_BOUNDS.get(name, (0.3, 0.7))
            obj.modulators[name] = Neuromodulator(
                name=name,
                clearance_rate=clearance,
                initial_setpoint=INITIAL_SETPOINTS.get(name, 0.5),
                setpoint_min=_smin,
                setpoint_max=_smax,
            )

        # Restore per-modulator state
        mod_states = state.get("modulators", {})
        for name, mod_state in mod_states.items():
            if name in obj.modulators:
                mod = obj.modulators[name]
                mod.restore_state(mod_state)
                # Restore fields beyond what restore_state covers
                mod.phasic_level = mod_state.get("phasic", mod.phasic_level)
                mod._peak_level = mod_state.get("peak", mod._peak_level)
                mod._trough_level = mod_state.get("trough", mod._trough_level)
                mod._total_updates = mod_state.get("total_updates", mod._total_updates)
                if "activation_history" in mod_state:
                    mod._activation_history = list(mod_state["activation_history"])

        obj._total_evaluations = state.get("total_evaluations", 0)
        obj._current_emotion = state.get("current_emotion", "neutral")
        obj._emotion_confidence = state.get("emotion_confidence", 0.0)

        logger.info(
            "[NeuromodulatorSystem] Restored from state: eval=%d, emotion=%s (%.3f)",
            obj._total_evaluations, obj._current_emotion, obj._emotion_confidence)
        return obj

    def _save_state(self) -> None:
        """Save neuromod state with rolling snapshots (keeps last 5).

        Atomic write (tmp→rename) prevents corruption on crash.
        Rolling snapshots allow recovery from bad state without losing history.
        """
        try:
            import time as _time
            state = {
                name: mod.get_state()
                for name, mod in self.modulators.items()
            }
            # Add metadata for validation on load
            state["_meta"] = {
                "epoch": self._total_evaluations,
                "emotion": self._current_emotion,
                "confidence": self._emotion_confidence,
                "timestamp": _time.time(),
            }

            # Atomic write to primary file
            path = os.path.join(self.data_dir, "neuromodulator_state.json")
            tmp_path = path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(state, f, indent=2)
            os.replace(tmp_path, path)

            # Rolling snapshot every 500 evaluations (keeps last 5)
            if self._total_evaluations % 500 == 0:
                snap_path = os.path.join(
                    self.data_dir,
                    f"neuromodulator_snapshot_{self._total_evaluations}.json"
                )
                with open(snap_path, "w") as f:
                    json.dump(state, f, indent=2)
                # Prune old snapshots, keep last 5
                import glob
                snaps = sorted(
                    glob.glob(os.path.join(self.data_dir, "neuromodulator_snapshot_*.json")),
                    key=os.path.getmtime
                )
                for old in snaps[:-5]:
                    os.remove(old)
                logger.info("[NeuromodulatorSystem] Rolling snapshot saved: eval=%d emotion=%s",
                            self._total_evaluations, self._current_emotion)
        except Exception as e:
            swallow_warn('[NeuromodulatorSystem] Save error', e,
                         key="logic.neuromodulator.save_error", throttle=100)

    def _load_state(self) -> None:
        """Load neuromod state from primary file, fall back to latest snapshot."""
        path = os.path.join(self.data_dir, "neuromodulator_state.json")

        # Try primary file first
        loaded = self._try_load_from(path)
        if loaded:
            return

        # Fall back to latest rolling snapshot
        import glob
        snaps = sorted(
            glob.glob(os.path.join(self.data_dir, "neuromodulator_snapshot_*.json")),
            key=os.path.getmtime
        )
        for snap in reversed(snaps):
            if self._try_load_from(snap):
                logger.warning("[NeuromodulatorSystem] Loaded from SNAPSHOT (primary missing): %s",
                               os.path.basename(snap))
                return

        logger.info("[NeuromodulatorSystem] No saved state — starting fresh")

    def _try_load_from(self, path: str) -> bool:
        """Attempt to load state from a specific file. Returns True on success."""
        if not os.path.exists(path):
            return False
        try:
            with open(path) as f:
                state = json.load(f)
            meta = state.pop("_meta", {})
            for name, mod_state in state.items():
                if name in self.modulators:
                    self.modulators[name].restore_state(mod_state)
            meta_str = ""
            if meta:
                meta_str = f" (eval={meta.get('epoch', '?')}, emotion={meta.get('emotion', '?')})"
            logger.info("[NeuromodulatorSystem] Loaded state from %s%s",
                        os.path.basename(path), meta_str)
            return True
        except Exception as e:
            logger.warning("[NeuromodulatorSystem] Failed to load %s: %s", path, e)
            return False


# D-SPEC-66 v1.11.0 PLAN §1.4 — pure module-level helper extracted from
# NeuromodulatorSystem._detect_emotion so cross-process consumers
# (social_worker via ShmReaderBank.read_neuromod) can derive
# current_emotion from level dict without holding the live system
# instance. Byte-identical to the in-process logic (extract-only refactor).
_EMOTION_TEMPLATES: dict[str, dict[str, float]] = {
    "joy":       {"DA": 0.8, "5HT": 0.7, "NE": 0.3, "ACh": 0.5, "Endorphin": 0.8, "GABA": 0.3},
    "peace":     {"DA": 0.5, "5HT": 0.8, "NE": 0.2, "ACh": 0.3, "Endorphin": 0.6, "GABA": 0.7},
    "curiosity": {"DA": 0.6, "5HT": 0.5, "NE": 0.7, "ACh": 0.8, "Endorphin": 0.5, "GABA": 0.3},
    "fear":      {"DA": 0.2, "5HT": 0.2, "NE": 0.9, "ACh": 0.7, "Endorphin": 0.2, "GABA": 0.1},
    "love":      {"DA": 0.7, "5HT": 0.8, "NE": 0.3, "ACh": 0.5, "Endorphin": 0.8, "GABA": 0.5},
    "anger":     {"DA": 0.6, "5HT": 0.1, "NE": 0.9, "ACh": 0.3, "Endorphin": 0.2, "GABA": 0.1},
    "sadness":   {"DA": 0.1, "5HT": 0.3, "NE": 0.3, "ACh": 0.3, "Endorphin": 0.1, "GABA": 0.7},
    "wonder":    {"DA": 0.7, "5HT": 0.6, "NE": 0.6, "ACh": 0.9, "Endorphin": 0.7, "GABA": 0.3},
    "flow":      {"DA": 0.8, "5HT": 0.7, "NE": 0.5, "ACh": 0.7, "Endorphin": 0.9, "GABA": 0.4},
    "calm":      {"DA": 0.4, "5HT": 0.7, "NE": 0.2, "ACh": 0.3, "Endorphin": 0.5, "GABA": 0.8},
}


def detect_emotion_from_levels(
        levels: dict[str, float]) -> tuple[str, float]:
    """Return (best_emotion, confidence) for a 6-modulator level dict.

    Pure helper: takes {"DA": float, "5HT": float, "NE": float, "ACh":
    float, "Endorphin": float, "GABA": float} (any missing keys default
    to 0.5) and returns the closest-matching emotion template by cosine
    similarity over the 6 level vector. Used by NeuromodulatorSystem.
    _detect_emotion (in-process owner) AND by social_worker after
    reading neuromod_state.bin levels via ShmReaderBank — guarantees
    byte-identical emotion derivation across process boundaries
    without requiring SHM schema bump.
    """
    best_emotion = "neutral"
    best_similarity = -1.0
    mag_l = math.sqrt(sum(v * v for v in levels.values()))
    for emotion, template in _EMOTION_TEMPLATES.items():
        dot = sum(levels.get(k, 0.5) * v for k, v in template.items())
        mag_t = math.sqrt(sum(v * v for v in template.values()))
        if mag_l > 1e-10 and mag_t > 1e-10:
            sim = dot / (mag_l * mag_t)
            if sim > best_similarity:
                best_similarity = sim
                best_emotion = emotion
    return best_emotion, round(best_similarity, 3)


def compute_modulation_from_state(state: dict) -> dict:
    """Compute the modulation gains dict from a 4-field neuromod state.

    Pure helper — no access to live `Neuromodulator` instances required.
    Used by `NeuromodulatorSystem.get_modulation()` (in-process owner) AND by
    `cognitive_worker` after reading the 4-field `neuromod_state.bin` SHM slot
    (cross-process consumer). Single source of truth for the modulation math.

    Args:
        state: `{modulator_name: {"level": float, "gain": float,
                                  "phasic": float, "tonic": float}}` for each
            of the 6 canonical modulators (DA, 5HT, NE, ACh, Endorphin, GABA).
            Missing modulators default to gain=1.0 (no modulation).

    Returns:
        14-key modulation dict matching the legacy `get_modulation()` output:
            learning_rate_gain, fire_threshold_gain, working_memory_retention,
            accumulation_rate_gain, refractory_gain, patience_factor,
            sensory_gain, exploration_temperature, filter_down_strength,
            training_frequency_gain, memory_encoding_gain, observation_precision,
            intrinsic_motivation, discomfort_suppression,
            global_threshold_raise, system_energy.

    Added 2026-05-15 (§4.Q neuromod_worker.evaluate migration) — see
    `titan-docs/PLAN_microkernel_phase_c_neuromod_worker_evaluate_migration.md`
    chunk Q1.
    """
    da_gain = state.get("DA", {}).get("gain", 1.0)
    sht_gain = state.get("5HT", {}).get("gain", 1.0)
    ne_gain = state.get("NE", {}).get("gain", 1.0)
    ach_gain = state.get("ACh", {}).get("gain", 1.0)
    end_gain = state.get("Endorphin", {}).get("gain", 1.0)
    gaba_gain = state.get("GABA", {}).get("gain", 1.0)

    return {
        # DA modulates
        "learning_rate_gain": da_gain,
        "fire_threshold_gain": 1.0 / max(0.3, da_gain),
        "working_memory_retention": da_gain,
        # 5-HT modulates
        "accumulation_rate_gain": 1.0 / max(0.3, sht_gain),
        "refractory_gain": sht_gain,
        "patience_factor": sht_gain,
        # NE modulates
        "sensory_gain": ne_gain,
        "exploration_temperature": ne_gain,
        "filter_down_strength": ne_gain,
        # ACh modulates
        "training_frequency_gain": ach_gain,
        "memory_encoding_gain": ach_gain,
        "observation_precision": ach_gain,
        # Endorphin modulates
        "intrinsic_motivation": end_gain,
        "discomfort_suppression": end_gain,
        # GABA modulates
        "global_threshold_raise": gaba_gain,
        "system_energy": 1.0 / max(0.3, gaba_gain),
    }


def compute_emergent_inputs(
    sphere_balance: dict,
    trinity_coherence: dict,
    chi_state: dict,
    consciousness_dynamics: dict,
    pi_state: dict,
    prediction_state: dict,
    ns_state: dict,
    expression_state: dict,
    resonance_state: dict,
    is_dreaming: bool,
    dna: dict,
) -> dict[str, float]:
    """Compute neuromodulator inputs from architectural state + DNA weights.

    DESIGN: Every signal is derived from a different architectural subsystem.
    No two modulators share the same primary source. No circular dependencies.
    Weight coefficients are DNA (birth parameters from titan_params.toml).

    Sources:
      DA       → Prediction Engine + Experience Memory
      5-HT     → Sphere Clock System (balance, body×1 mind×3)
      NE       → Life Force + Consciousness State Vectors
      ACh      → Consciousness Dynamics + Neural NS
      Endorphin → π-Heartbeat + Resonance Detector
      GABA     → Dreaming Engine + Metabolism + Expression

    Validated: 2026-03-23 via scripts/neuromod_simulation.py
    """
    _c = lambda v: max(0.0, min(1.0, v))

    # ── DA: Reward Prediction Error (minimal change — already emergent) ──
    da_input = _c(
        dna.get("da_prediction_surprise", 0.30) * prediction_state.get("surprise", 0.0) +
        dna.get("da_action_outcome", 0.40) * prediction_state.get("action_outcome", 0.5) +
        dna.get("da_success_trend", 0.20) * prediction_state.get("success_rate", 0.5) +
        dna.get("da_pi_curvature_reward", 0.10) * pi_state.get("curvature_delta", 0.0)
    )

    # ── 5-HT: Balance-Derived Stability ──
    # Sphere clock balance: body×1, mind×3 weighted (power-of-3 principle)
    _body_bal = (sphere_balance.get("inner_body", 0.0) + sphere_balance.get("outer_body", 0.0)) / 2
    _mind_bal = (sphere_balance.get("inner_mind", 0.0) + sphere_balance.get("outer_mind", 0.0)) / 2
    _bw = dna.get("balance_body_weight", 1)
    _mw = dna.get("balance_mind_weight", 3)
    _weighted_balance = (_body_bal * _bw + _mind_bal * _mw) / (_bw + _mw)

    sht_input = _c(
        dna.get("sht_sphere_balance", 0.35) * _weighted_balance +
        dna.get("sht_epoch_regularity", 0.20) * pi_state.get("epoch_regularity", 0.0) +
        dna.get("sht_chi_circulation", 0.15) * chi_state.get("circulation", 0.0) +
        dna.get("sht_drift_stability", 0.15) * (1.0 - consciousness_dynamics.get("drift_magnitude", 0.5)) +
        dna.get("sht_epoch_maturity", 0.15) * pi_state.get("epoch_maturity", min(1.0, pi_state.get("developmental_age", 0) / 50.0))
    )

    # ── NE: Chi-Tonic + Trinity Coherence ──
    _avg_coh = (trinity_coherence.get("inner", 0.5) + trinity_coherence.get("outer", 0.5)) / 2
    ne_input = _c(
        dna.get("ne_chi_tonic", 0.10) * chi_state.get("total", 0.5) +
        dna.get("ne_trinity_coherence", 0.15) * _avg_coh +
        dna.get("ne_prediction_surprise", 0.25) * prediction_state.get("surprise", 0.0) +
        dna.get("ne_state_change_rate", 0.20) * consciousness_dynamics.get("drift_delta", 0.0) +
        dna.get("ne_action_uncertainty", 0.15) * (1.0 - prediction_state.get("success_rate", 0.5)) +
        dna.get("ne_system_excitation", 0.15) * consciousness_dynamics.get("density", 0.0)
    )

    # ── ACh: Attention / Learning Demand ──
    ach_input = _c(
        dna.get("ach_state_change_rate", 0.30) * consciousness_dynamics.get("drift_delta", 0.0) +
        dna.get("ach_ns_learning_rate", 0.25) * ns_state.get("transition_delta", 0.0) +
        dna.get("ach_pi_irregularity", 0.20) * (1.0 - pi_state.get("regularity", 0.5)) +
        dna.get("ach_filter_down_activity", 0.25) * ns_state.get("filter_down_writes", 0.0)
    )

    # ── Endorphin: Flow / Intrinsic Reward ──
    endorphin_input = _c(
        dna.get("endorphin_action_alignment", 0.25) * expression_state.get("alignment", 0.5) +
        dna.get("endorphin_pi_flow", 0.25) * pi_state.get("cluster_streak", 0.0) +
        dna.get("endorphin_resonance_harmony", 0.20) * resonance_state.get("resonant_fraction", 0.0) +
        dna.get("endorphin_chi_body_vitality", 0.15) * chi_state.get("body", 0.5) +
        dna.get("endorphin_chi_circulation", 0.15) * chi_state.get("circulation", 0.0)
    )

    # ── GABA: Inhibition / Rest Need (NO circular neuromod deps) ──
    gaba_input = _c(
        dna.get("gaba_dreaming", 0.35) * (1.0 if is_dreaming else 0.0) +
        dna.get("gaba_metabolic_drain", 0.15) * chi_state.get("drain", 0.0) +
        dna.get("gaba_expression_fire_rate", 0.15) * expression_state.get("fire_rate", 0.0) +
        dna.get("gaba_chi_stagnation", 0.20) * (1.0 - chi_state.get("circulation", 0.0)) +
        dna.get("gaba_epoch_saturation", 0.15) * consciousness_dynamics.get("epoch_gap_ratio", 0.0)
    )

    result = {
        "DA": da_input,
        "5HT": sht_input,
        "NE": ne_input,
        "ACh": ach_input,
        "Endorphin": endorphin_input,
        "GABA": gaba_input,
    }

    # ── Circadian dreaming modulation ──
    # During dreaming: suppress production inputs, let clearance dominate.
    # Suppression strength governed by GABA (Titan's own inhibitory tone).
    if is_dreaming:
        _gaba_level = result["GABA"]
        _dream_suppression = max(0.1, 1.0 - _gaba_level * 0.8)
        for key in result:
            if key != "GABA":
                result[key] *= _dream_suppression
        result["GABA"] = min(1.0, result["GABA"] * 1.3)

    return result


def apply_movement_excess_clearance(
    mod: "Neuromodulator",
    topology_velocity: float,
    movement_dna: float,
) -> None:
    """Apply topology-modulated excess clearance after neuromod update.

    Biology: physical activity speeds neurotransmitter turnover, preferentially
    clearing excess above setpoint (Michaelis-Menten-like substrate dependence).

    Only drains EXCESS above setpoint. Never pushes below 80% of setpoint.
    Per-modulator sensitivity defined by DNA (movement_dna param).

    Args:
        mod: The Neuromodulator to apply clearance to
        topology_velocity: normalized state-space movement speed (0-1)
        movement_dna: per-modulator movement sensitivity from DNA
    """
    excess = max(0.0, mod.level - mod.setpoint)
    if excess > 0 and topology_velocity > 0 and movement_dna > 0:
        drain = movement_dna * excess * topology_velocity
        mod.level = max(mod.setpoint * 0.8, mod.level - drain)


# ── Legacy function kept for rollback ──
def compute_inputs_from_titan(
    prediction_surprise: float = 0.0,
    action_outcome: float = 0.5,
    middle_path_stability: float = 0.5,
    pi_regularity: float = 0.5,
    developmental_age: int = 0,
    episodic_growth_rate: float = 0.0,
    action_success_rate: float = 0.5,
    outcome_variance: float = 0.0,
    new_info_rate: float = 0.0,
    action_state_alignment: float = 0.5,
    creative_quality: float = 0.0,
    system_excitation: float = 0.5,
    is_dreaming: bool = False,
    chi_total: float = 0.5,
    chi_body_vitality: float = 0.5,
    chi_circulation: float = 0.5,
) -> dict[str, float]:
    """LEGACY: Pre-emergent neuromod inputs. Kept for rollback.
    Use compute_emergent_inputs() instead.
    """
    da_input = max(0.0, min(1.0,
        0.5 + (action_outcome - 0.5) * 0.4 +
        prediction_surprise * 0.3 +
        (action_success_rate - 0.5) * 0.2
    ))
    dev_norm = min(1.0, developmental_age / 50.0)
    sht_input = max(0.0, min(1.0,
        middle_path_stability * 0.25 + pi_regularity * 0.25 +
        dev_norm * 0.15 + chi_total * 0.15 + (1.0 - system_excitation) * 0.2
    ))
    ne_input = max(0.0, min(1.0,
        episodic_growth_rate * 0.25 + prediction_surprise * 0.3 +
        (1.0 - action_success_rate) * 0.2 + system_excitation * 0.25
    ))
    ach_input = max(0.0, min(1.0,
        outcome_variance * 0.3 + new_info_rate * 0.3 +
        episodic_growth_rate * 0.2 + (1.0 - pi_regularity) * 0.2
    ))
    endorphin_input = max(0.0, min(1.0,
        action_state_alignment * 0.3 + creative_quality * 0.25 +
        chi_body_vitality * 0.2 + (1.0 - abs(system_excitation - 0.5)) * 0.25
    ))
    gaba_input = max(0.0, min(1.0,
        (1.0 - system_excitation) * 0.25 + sht_input * 0.15 +
        (0.5 if is_dreaming else 0.0) + (1.0 - ne_input) * 0.15 +
        (1.0 - chi_circulation) * 0.15
    ))
    result = {"DA": da_input, "5HT": sht_input, "NE": ne_input,
              "ACh": ach_input, "Endorphin": endorphin_input, "GABA": gaba_input}
    if is_dreaming:
        _gaba_level = result["GABA"]
        _dream_suppression = max(0.1, 1.0 - _gaba_level * 0.8)
        for key in result:
            if key != "GABA":
                result[key] *= _dream_suppression
        result["GABA"] = min(1.0, result["GABA"] * 1.3)
    return result
