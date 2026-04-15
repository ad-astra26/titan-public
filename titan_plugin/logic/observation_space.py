"""
titan_plugin/logic/observation_space.py — V5 Centralized Input Builder.

Collects data from all available sources once per coordinator tick,
caches the result, and builds input vectors for any requested feature set.

Tier 1 (30D): Core Observables — 6 body parts × 5 metrics
Tier 2 (25D): Temporal & Spatial — clocks, topology, resonance, spirit, consciousness
Tier 3 (20D): Learned Attention — FilterDown multipliers, Focus nudges
Tier 4 (13D): Meta-State — consciousness vector, distances, impulse pressure
Tier 5 (12D): Neurochemical State — neuromod levels, chi components
Tier 6 (12D): System Dynamics + Reasoning — metabolic drain, sleep/wake drives, experience, reasoning state

Named feature sets:
  "core"           = Tier 1              (30D)
  "standard"       = Tier 1 + 2          (55D)
  "extended"       = Tier 1 + 2 + 3      (75D)
  "full"           = Tier 1-4            (88D)
  "enriched"       = Tier 1+2 + 5+6      (79D)  — inner programs: standard + neurochemical + dynamics + reasoning
  "full_enriched"  = Tier 1-6            (112D) — outer personality programs: all tiers
"""
import logging
import math

import numpy as np

logger = logging.getLogger(__name__)

# Canonical ordering for body parts in Tier 1
BODY_PARTS = [
    "inner_body", "inner_mind", "inner_spirit",
    "outer_body", "outer_mind", "outer_spirit",
]

# Canonical ordering for observables per part
OBS_KEYS = ["coherence", "magnitude", "velocity", "direction", "polarity"]

# Canonical ordering for sphere clocks
CLOCK_NAMES = [
    "inner_body", "inner_mind", "inner_spirit",
    "outer_body", "outer_mind", "outer_spirit",
]

DIMS = {
    "core": 30, "standard": 55, "extended": 75, "full": 88,
    "enriched": 79,        # T1+T2+T5+T6: standard + neurochemical + dynamics + reasoning (inner programs)
    "full_enriched": 112,  # T1+T2+T3+T4+T5+T6: all tiers (outer personality programs)
}

# Canonical neuromodulator ordering for Tier 5
NEUROMOD_NAMES = ["DA", "5-HT", "NE", "ACh", "Endorphin", "GABA"]


class ObservationSpace:
    """
    Centralized input builder for neural nervous system programs.

    Called once per coordinator tick with all available data.
    Programs request input vectors by feature set name.
    """

    def __init__(self):
        self._tier1: np.ndarray = np.zeros(30, dtype=np.float64)
        self._tier2: np.ndarray = np.zeros(25, dtype=np.float64)
        self._tier3: np.ndarray = np.zeros(20, dtype=np.float64)
        self._tier4: np.ndarray = np.zeros(13, dtype=np.float64)
        self._tier5: np.ndarray = np.zeros(12, dtype=np.float64)
        self._tier6: np.ndarray = np.zeros(12, dtype=np.float64)
        self._updated = False
        # Cache raw inputs for hormonal stimulus extraction
        self._observables: dict = {}
        self._topology: dict = {}
        self._fatigue: float = 0.0
        self._readiness: float = 0.0

    def update(
        self,
        observables: dict = None,
        topology: dict = None,
        dreaming: dict = None,
        sphere_clocks: dict = None,
        resonance: dict = None,
        unified_spirit: dict = None,
        consciousness: dict = None,
        state_register_snapshot: dict = None,
        filter_down_mults: dict = None,
        focus_nudges: dict = None,
        impulse_state: dict = None,
        middle_path_loss: float = 0.0,
        # Tier 5: Neurochemical State
        neuromodulator_levels: dict = None,
        neuromodulator_setpoints: dict = None,
        chi_state: dict = None,
        # Tier 6: System Dynamics + Reasoning
        metabolic_drain: float = 0.0,
        sleep_drive: float = 0.0,
        wake_drive: float = 0.0,
        experience_pressure: float = 0.0,
        expression_repetitiveness: float = 0.0,
        teacher_active: bool = False,
        vocabulary_confidence: float = 0.0,
        time_since_dream: float = 0.0,
        # Tier 6 extension: Reasoning state
        reasoning_active: float = 0.0,
        reasoning_chain_length: float = 0.0,
        reasoning_confidence: float = 0.0,
        reasoning_gut_agreement: float = 0.0,
    ) -> None:
        """
        Refresh all tiers from available data. Called once per coordinator tick.

        All parameters are optional — missing data gets zero-filled.
        """
        # Cache raw inputs for hormonal stimulus extraction
        self._observables = observables or {}
        self._topology = topology or {}
        dream = dreaming or {}
        self._fatigue = dream.get("fatigue", 0.0)
        self._readiness = dream.get("readiness", 0.0)

        self._build_tier1(observables or {})
        self._build_tier2(
            topology or {}, dreaming or {}, sphere_clocks or {},
            resonance or {}, unified_spirit or {}, consciousness or {},
            middle_path_loss,
        )
        self._build_tier3(filter_down_mults or {}, focus_nudges or {})
        self._build_tier4(
            consciousness or {}, state_register_snapshot or {},
            impulse_state or {},
        )
        self._build_tier5(
            neuromodulator_levels or {}, neuromodulator_setpoints or {},
            chi_state or {},
        )
        self._build_tier6(
            metabolic_drain, sleep_drive, wake_drive,
            experience_pressure, expression_repetitiveness,
            teacher_active, vocabulary_confidence, time_since_dream,
            reasoning_active, reasoning_chain_length,
            reasoning_confidence, reasoning_gut_agreement,
        )
        self._updated = True

    def build_input(self, feature_set: str = "standard") -> np.ndarray:
        """Build input vector for a given feature set."""
        if feature_set == "core":
            return self._tier1.copy()
        elif feature_set == "standard":
            return np.concatenate([self._tier1, self._tier2])
        elif feature_set == "extended":
            return np.concatenate([self._tier1, self._tier2, self._tier3])
        elif feature_set == "full":
            return np.concatenate([self._tier1, self._tier2, self._tier3, self._tier4])
        elif feature_set == "enriched":
            return np.concatenate([self._tier1, self._tier2, self._tier5, self._tier6])
        elif feature_set == "full_enriched":
            return np.concatenate([self._tier1, self._tier2, self._tier3, self._tier4,
                                   self._tier5, self._tier6])
        # Fallback to standard
        return np.concatenate([self._tier1, self._tier2])

    @staticmethod
    def get_dim(feature_set: str = "standard") -> int:
        """Return expected input dimension for a feature set."""
        return DIMS.get(feature_set, 55)

    def get_feature_names(self, feature_set: str = "standard") -> list[str]:
        """Return ordered feature names for debugging/API."""
        names = list(self._tier1_names())
        if feature_set in ("standard", "extended", "full", "enriched", "full_enriched"):
            names.extend(self._tier2_names())
        if feature_set in ("extended", "full", "full_enriched"):
            names.extend(self._tier3_names())
        if feature_set in ("full", "full_enriched"):
            names.extend(self._tier4_names())
        if feature_set in ("enriched", "full_enriched"):
            names.extend(self._tier5_names())
            names.extend(self._tier6_names())
        return names

    # ── Tier Builders ─────────────────────────────────────────────

    def _build_tier1(self, observables: dict) -> None:
        """Tier 1: Core Observables (30D) — 6 parts × 5 metrics."""
        vec = []
        for part in BODY_PARTS:
            obs = observables.get(part, {})
            for key in OBS_KEYS:
                vec.append(float(obs.get(key, 0.0)))
        self._tier1 = np.array(vec, dtype=np.float64)

    def _build_tier2(self, topology: dict, dreaming: dict,
                     sphere_clocks: dict, resonance: dict,
                     unified_spirit: dict, consciousness: dict,
                     middle_path_loss: float) -> None:
        """Tier 2: Temporal & Spatial (25D)."""
        vec = []

        # Sphere clock phases (6) — normalized to [0, 1]
        for name in CLOCK_NAMES:
            clock = sphere_clocks.get(name, {})
            if isinstance(clock, dict):
                phase = clock.get("phase", 0.0)
                vec.append(phase / (2.0 * math.pi))  # normalize to [0, 1]
            else:
                vec.append(0.0)

        # Sphere clock velocities (6)
        for name in CLOCK_NAMES:
            clock = sphere_clocks.get(name, {})
            if isinstance(clock, dict):
                vec.append(float(clock.get("contraction_velocity",
                           clock.get("velocity", 0.0))))
            else:
                vec.append(0.0)

        # Topology (5): volume, curvature, mean_distance, cluster_count, isolated_count
        vec.append(float(topology.get("volume", 0.0)) / 15.0)  # normalize
        vec.append(float(topology.get("curvature", 0.0)))
        vec.append(float(topology.get("mean_distance", 0.0)))
        clusters = topology.get("clusters", [])
        vec.append(float(len(clusters)) / 6.0)  # normalize
        isolated = topology.get("isolated", [])
        vec.append(float(len(isolated)) / 6.0)  # normalize

        # Resonance (1): resonant pair count [0, 3] → [0, 1]
        if isinstance(resonance, dict):
            pairs = resonance.get("pairs", resonance)
            if isinstance(pairs, dict):
                resonant = sum(1 for p in pairs.values()
                               if isinstance(p, dict) and p.get("is_resonant", False))
            else:
                resonant = 0
        else:
            resonant = 0
        vec.append(float(resonant) / 3.0)

        # Fatigue (1)
        vec.append(float(dreaming.get("fatigue", 0.0)))

        # Unified Spirit (3): velocity, alignment, quality (normalized)
        vec.append(float(unified_spirit.get("velocity", 1.0)))
        vec.append(float(unified_spirit.get("last_alignment",
                   unified_spirit.get("alignment", 0.5))))
        quality = float(unified_spirit.get("cumulative_quality",
                        unified_spirit.get("quality", 0.0)))
        vec.append(min(1.0, quality / 100.0))  # soft normalize

        # Consciousness (2): drift, trajectory
        vec.append(min(1.0, float(consciousness.get("drift_magnitude",
                   consciousness.get("drift", 0.0)))))
        vec.append(min(1.0, float(consciousness.get("trajectory_magnitude",
                   consciousness.get("trajectory", 0.0)))))

        # Middle path loss (1)
        vec.append(float(middle_path_loss))

        self._tier2 = np.array(vec, dtype=np.float64)

    def _build_tier3(self, filter_down_mults: dict, focus_nudges: dict) -> None:
        """Tier 3: Learned Attention (20D) — FilterDown multipliers + Focus nudges."""
        vec = []

        # FilterDown body multipliers (5) — normalize from [0.3, 3.0] to ~[0, 1]
        fd_body = filter_down_mults.get("body", filter_down_mults.get("filter_down_body", [1.0] * 5))
        for v in (fd_body if isinstance(fd_body, list) else [1.0] * 5)[:5]:
            vec.append((float(v) - 0.3) / 2.7)  # map [0.3, 3.0] → [0, 1]

        # FilterDown mind multipliers (5)
        fd_mind = filter_down_mults.get("mind", filter_down_mults.get("filter_down_mind", [1.0] * 5))
        for v in (fd_mind if isinstance(fd_mind, list) else [1.0] * 5)[:5]:
            vec.append((float(v) - 0.3) / 2.7)

        # Focus body nudges (5) — normalize from [-0.5, 0.5] to [0, 1]
        fb = focus_nudges.get("body", focus_nudges.get("focus_body", [0.0] * 5))
        for v in (fb if isinstance(fb, list) else [0.0] * 5)[:5]:
            vec.append(float(v) + 0.5)  # map [-0.5, 0.5] → [0, 1]

        # Focus mind nudges (5)
        fm = focus_nudges.get("mind", focus_nudges.get("focus_mind", [0.0] * 5))
        for v in (fm if isinstance(fm, list) else [0.0] * 5)[:5]:
            vec.append(float(v) + 0.5)

        self._tier3 = np.array(vec, dtype=np.float64)

    def _build_tier4(self, consciousness: dict, state_register: dict,
                     impulse_state: dict) -> None:
        """Tier 4: Meta-State (13D)."""
        vec = []

        # Consciousness state vector (9) — already [0, 1] normalized
        sv = consciousness.get("state_vector", [0.5] * 9)
        if isinstance(sv, list) and len(sv) >= 9:
            vec.extend([float(v) for v in sv[:9]])
        else:
            vec.extend([0.5] * 9)

        # Body/mind center distances (2) — normalize by max (~1.118)
        vec.append(float(state_register.get("body_center_dist", 0.0)) / 1.118)
        vec.append(float(state_register.get("mind_center_dist", 0.0)) / 1.118)

        # Impulse urgency (1)
        vec.append(float(impulse_state.get("urgency",
                   impulse_state.get("last_urgency", 0.0))))

        # Experience buffer fullness (1)
        buf_size = float(state_register.get("experience_buffer_size", 0))
        vec.append(min(1.0, buf_size / 100.0))

        self._tier4 = np.array(vec, dtype=np.float64)

    def _build_tier5(self, levels: dict, setpoints: dict, chi: dict) -> None:
        """Tier 5: Neurochemical State (12D) — neuromod levels + chi components."""
        vec = []

        # Neuromodulator levels (6) — clamped to [0, 1]
        for nm in NEUROMOD_NAMES:
            vec.append(min(1.0, max(0.0, float(levels.get(nm, 0.5)))))

        # Neuromod deviation from homeostasis (1) — mean |level-setpoint|/setpoint
        devs = []
        for nm in NEUROMOD_NAMES:
            lvl = float(levels.get(nm, 0.5))
            sp = max(float(setpoints.get(nm, 0.5)), 0.1)
            devs.append(abs(lvl - sp) / sp)
        vec.append(min(1.0, sum(devs) / max(len(devs), 1)))

        # Chi components (5) — metabolic energy state
        vec.append(min(1.0, max(0.0, float(chi.get("total", 0.5)))))
        vec.append(min(1.0, max(0.0, float(chi.get("circulation", 0.5)))))
        vec.append(min(1.0, max(0.0, float(chi.get("body", 0.5)))))
        vec.append(min(1.0, max(0.0, float(chi.get("mind", 0.5)))))
        vec.append(min(1.0, max(0.0, float(chi.get("spirit", 0.5)))))

        self._tier5 = np.array(vec, dtype=np.float64)

    def _build_tier6(self, metabolic_drain: float, sleep_drive: float,
                     wake_drive: float, experience_pressure: float,
                     expression_repetitiveness: float, teacher_active: bool,
                     vocabulary_confidence: float, time_since_dream: float,
                     reasoning_active: float = 0.0,
                     reasoning_chain_length: float = 0.0,
                     reasoning_confidence: float = 0.0,
                     reasoning_gut_agreement: float = 0.0) -> None:
        """Tier 6: System Dynamics + Reasoning (12D) — drives, pressures, reasoning state."""
        vec = [
            min(1.0, max(0.0, metabolic_drain)),
            min(1.0, max(0.0, sleep_drive)),
            min(1.0, max(0.0, wake_drive)),
            min(1.0, max(0.0, experience_pressure)),
            min(1.0, max(0.0, expression_repetitiveness)),
            1.0 if teacher_active else 0.0,
            min(1.0, max(0.0, vocabulary_confidence)),
            min(1.0, max(0.0, time_since_dream / 3600.0)),  # normalize: 1hr = 1.0
            # Reasoning state (4D extension)
            min(1.0, max(0.0, reasoning_active)),
            min(1.0, max(0.0, reasoning_chain_length)),
            min(1.0, max(0.0, reasoning_confidence)),
            min(1.0, max(0.0, reasoning_gut_agreement)),
        ]
        self._tier6 = np.array(vec, dtype=np.float64)

    # ── Feature Names ─────────────────────────────────────────────

    @staticmethod
    def _tier1_names() -> list[str]:
        names = []
        for part in BODY_PARTS:
            for key in OBS_KEYS:
                names.append(f"{part}.{key}")
        return names

    @staticmethod
    def _tier2_names() -> list[str]:
        names = []
        for c in CLOCK_NAMES:
            names.append(f"clock.{c}.phase")
        for c in CLOCK_NAMES:
            names.append(f"clock.{c}.velocity")
        names.extend([
            "topology.volume", "topology.curvature", "topology.mean_distance",
            "topology.cluster_count", "topology.isolated_count",
            "resonance.resonant_count", "dreaming.fatigue",
            "spirit.velocity", "spirit.alignment", "spirit.quality",
            "consciousness.drift", "consciousness.trajectory",
            "middle_path_loss",
        ])
        return names

    @staticmethod
    def _tier3_names() -> list[str]:
        names = []
        for i in range(5):
            names.append(f"filter_down.body.{i}")
        for i in range(5):
            names.append(f"filter_down.mind.{i}")
        for i in range(5):
            names.append(f"focus.body.{i}")
        for i in range(5):
            names.append(f"focus.mind.{i}")
        return names

    @staticmethod
    def _tier4_names() -> list[str]:
        sv_names = ["mood", "energy", "memory_pressure", "social_entropy",
                     "sovereignty", "learning_velocity", "social_density",
                     "curvature", "density"]
        names = [f"consciousness.{n}" for n in sv_names]
        names.extend([
            "body_center_dist", "mind_center_dist",
            "impulse.urgency", "experience_buffer_fullness",
        ])
        return names

    @staticmethod
    def _tier5_names() -> list[str]:
        return [
            "neuromod.DA", "neuromod.5-HT", "neuromod.NE",
            "neuromod.ACh", "neuromod.Endorphin", "neuromod.GABA",
            "neuromod.deviation",
            "chi.total", "chi.circulation", "chi.body", "chi.mind", "chi.spirit",
        ]

    @staticmethod
    def _tier6_names() -> list[str]:
        return [
            "dynamics.metabolic_drain", "dynamics.sleep_drive",
            "dynamics.wake_drive", "dynamics.experience_pressure",
            "dynamics.expression_repetitiveness", "dynamics.teacher_active",
            "dynamics.vocabulary_confidence", "dynamics.time_since_dream",
            "reasoning.is_active", "reasoning.chain_length",
            "reasoning.confidence", "reasoning.gut_agreement",
        ]
