"""
titan_plugin/logic/hormonal_pressure.py — Hormonal Pressure System.

Models endocrine-like pressure accumulation for neural program firing.
Each program has a "hormone" that builds pressure from stimuli over time.
When pressure exceeds a learned threshold, the program fires and enters
a refractory period. Programs interact via excitatory/inhibitory cross-talk.

Replaces the static fire_threshold model with temporal pressure dynamics:
- Pressure BUILDS over time (even idle — base secretion)
- Stimuli AMPLIFY pressure (NN output + environmental signals)
- Cross-talk: programs excite/inhibit each other
- Circadian: awake/dreaming modulates accumulation rates
- Refractory: post-fire suppression prevents spam
- Threshold adapts via IQL from action outcomes

This is the autonomous nervous system that makes CURIOSITY eventually fire.
"""
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Default Cross-Talk Matrix ────────────────────────────────────────
# (+) = excitatory: this hormone's level increases target's accumulation
# (-) = inhibitory: this hormone's level decreases target's accumulation

DEFAULT_CROSS_TALK = {
    "REFLEX": {
        "excitors": {"VIGILANCE": 0.3, "IMPULSE": 0.2},
        "inhibitors": {"REFLECTION": 0.2, "CREATIVITY": 0.3},
    },
    "FOCUS": {
        "excitors": {"CURIOSITY": 0.1, "IMPULSE": 0.1},
        "inhibitors": {"REFLECTION": 0.1},
    },
    "INTUITION": {
        "excitors": {"FOCUS": 0.2, "REFLECTION": 0.1},
        "inhibitors": {"REFLEX": 0.2},
    },
    "IMPULSE": {
        "excitors": {"REFLEX": 0.2, "CURIOSITY": 0.1},
        "inhibitors": {"REFLECTION": 0.2},
    },
    "VIGILANCE": {
        "excitors": {"REFLEX": 0.3, "FOCUS": 0.2},
        "inhibitors": {"REFLECTION": 0.3, "CREATIVITY": 0.2},
    },
    "CREATIVITY": {
        "excitors": {"INSPIRATION": 0.4, "CURIOSITY": 0.2, "REFLECTION": 0.1},
        "inhibitors": {"REFLEX": 0.3, "VIGILANCE": 0.2},
    },
    "CURIOSITY": {
        "excitors": {"INSPIRATION": 0.3, "CREATIVITY": 0.2},
        "inhibitors": {"REFLEX": 0.5, "VIGILANCE": 0.1},
    },
    "EMPATHY": {
        "excitors": {"CURIOSITY": 0.2, "REFLECTION": 0.2},
        "inhibitors": {"REFLEX": 0.2, "VIGILANCE": 0.1},
    },
    "REFLECTION": {
        "excitors": {"EMPATHY": 0.2, "INSPIRATION": 0.1},
        "inhibitors": {"REFLEX": 0.4, "IMPULSE": 0.3},
    },
    "INSPIRATION": {
        "excitors": {"CURIOSITY": 0.3, "CREATIVITY": 0.4, "REFLECTION": 0.2},
        "inhibitors": {"REFLEX": 0.2, "VIGILANCE": 0.1},
    },
}

# ── Circadian Multipliers ────────────────────────────────────────────
CIRCADIAN = {
    "awake": {
        "REFLEX": 1.0, "FOCUS": 1.0, "INTUITION": 1.0, "IMPULSE": 1.0,
        "VIGILANCE": 1.0, "CREATIVITY": 0.7, "CURIOSITY": 1.2,
        "EMPATHY": 1.0, "REFLECTION": 0.6, "INSPIRATION": 0.8,
    },
    "dreaming": {
        "REFLEX": 0.3, "FOCUS": 0.3, "INTUITION": 0.8, "IMPULSE": 0.2,
        "VIGILANCE": 0.2, "CREATIVITY": 1.3, "CURIOSITY": 0.4,
        "EMPATHY": 0.5, "REFLECTION": 1.5, "INSPIRATION": 1.5,
    },
}

# ── Default Hormone Parameters per Program ───────────────────────────
# Inner programs: high base_rate, fast decay (responsive, autonomic)
# Outer programs: low base_rate, slow decay (builds up, conscious)
DEFAULT_HORMONE_PARAMS = {
    "REFLEX": {
        "base_secretion_rate": 0.008, "stimulus_sensitivity": 1.5,
        "decay_rate": 0.003, "fire_threshold": 0.5,
        "refractory_strength": 0.9, "refractory_decay": 0.015,
    },
    "FOCUS": {
        "base_secretion_rate": 0.006, "stimulus_sensitivity": 1.3,
        "decay_rate": 0.002, "fire_threshold": 0.4,
        "refractory_strength": 0.7, "refractory_decay": 0.012,
    },
    "INTUITION": {
        "base_secretion_rate": 0.004, "stimulus_sensitivity": 1.0,
        "decay_rate": 0.002, "fire_threshold": 0.5,
        "refractory_strength": 0.8, "refractory_decay": 0.010,
    },
    "IMPULSE": {
        "base_secretion_rate": 0.005, "stimulus_sensitivity": 1.2,
        "decay_rate": 0.002, "fire_threshold": 0.5,
        "refractory_strength": 0.8, "refractory_decay": 0.010,
    },
    "VIGILANCE": {
        "base_secretion_rate": 0.006, "stimulus_sensitivity": 1.4,
        "decay_rate": 0.003, "fire_threshold": 0.5,
        "refractory_strength": 0.8, "refractory_decay": 0.012,
    },
    "CREATIVITY": {
        "base_secretion_rate": 0.003, "stimulus_sensitivity": 1.0,
        "decay_rate": 0.001, "fire_threshold": 0.6,
        "refractory_strength": 0.9, "refractory_decay": 0.008,
    },
    "CURIOSITY": {
        "base_secretion_rate": 0.004, "stimulus_sensitivity": 1.2,
        "decay_rate": 0.001, "fire_threshold": 0.5,
        "refractory_strength": 0.7, "refractory_decay": 0.010,
    },
    "EMPATHY": {
        "base_secretion_rate": 0.003, "stimulus_sensitivity": 1.0,
        "decay_rate": 0.001, "fire_threshold": 0.5,
        "refractory_strength": 0.7, "refractory_decay": 0.010,
    },
    "REFLECTION": {
        "base_secretion_rate": 0.002, "stimulus_sensitivity": 0.8,
        "decay_rate": 0.0008, "fire_threshold": 0.6,
        "refractory_strength": 0.9, "refractory_decay": 0.006,
    },
    "INSPIRATION": {
        "base_secretion_rate": 0.002, "stimulus_sensitivity": 0.8,
        "decay_rate": 0.0008, "fire_threshold": 0.7,
        "refractory_strength": 0.9, "refractory_decay": 0.005,
    },
}


class HormonalPressure:
    """Per-program pressure accumulator modeled on endocrine hormones."""

    def __init__(
        self,
        name: str,
        base_secretion_rate: float = 0.005,
        stimulus_sensitivity: float = 1.0,
        decay_rate: float = 0.002,
        fire_threshold: float = 0.5,
        refractory_strength: float = 0.8,
        refractory_decay: float = 0.01,
        dna_sensitivity_bias: float = 1.0,
    ):
        self.name = name
        self.level: float = 0.0
        self.threshold: float = fire_threshold
        self.refractory: float = 0.0

        self.base_rate = base_secretion_rate
        self.sensitivity = stimulus_sensitivity * dna_sensitivity_bias
        self.decay_rate = decay_rate
        self._base_refractory_strength = refractory_strength
        self.refractory_strength = refractory_strength
        self._base_refractory_decay = refractory_decay
        self.refractory_decay = refractory_decay

        # Cross-talk references (set by HormonalSystem)
        self.excitors: dict[str, float] = {}
        self.inhibitors: dict[str, float] = {}

        # Maturity (0.0 = infant, 1.0 = mature) — set by HormonalSystem
        self._maturity: float = 0.0
        # Layer identity — inner programs maintain survival floor at low Chi
        self._is_inner: bool = False  # Set by HormonalSystem from NS program config

        # Stats
        self.fire_count: int = 0
        self.last_fire_ts: float = 0.0
        self.peak_level: float = 0.0

    def accumulate(
        self,
        stimulus: float,
        dt: float,
        other_levels: Optional[dict[str, float]] = None,
        circadian_multiplier: float = 1.0,
        gaba_level: float = 0.35,
        chi_total: float = 0.6,
        accumulation_rate_gain: float = 1.0,
    ) -> None:
        """
        Accumulate pressure from stimulus. Called every evaluation cycle.

        Args:
            stimulus: [0, 1] magnitude of relevant observation
            dt: seconds since last accumulation
            other_levels: current levels of other hormones (for cross-talk)
            circadian_multiplier: awake=1.0, dreaming=varies by program
            gaba_level: GABA neuromodulator level (Governor 1: decay rate)
            chi_total: Chi life force (Governor 2: capacity ceiling)
            accumulation_rate_gain: from neuromod modulation (Governor 3: secretion)
        """
        # Clamp dt to prevent explosion after long pauses
        dt = min(dt, 30.0)

        # Cross-talk modulation — Governor 5: normalize by threshold
        # Prevents runaway positive feedback loop between CURIOSITY↔INSPIRATION↔CREATIVITY
        cross_talk = 1.0
        if other_levels:
            for h_name, weight in self.excitors.items():
                if h_name in other_levels:
                    cross_talk += weight * (other_levels[h_name] / max(0.01, self.threshold))
            for h_name, weight in self.inhibitors.items():
                if h_name in other_levels:
                    cross_talk -= weight * (other_levels[h_name] / max(0.01, self.threshold))
            cross_talk = max(0.1, min(3.0, cross_talk))

        # Accumulate: basal + stimulus-driven, modulated by cross-talk & circadian
        # Governor 3: neuromod-governed accumulation (DA/ACh → action readiness)
        secretion = (self.base_rate + stimulus * self.sensitivity)
        secretion *= cross_talk * circadian_multiplier * dt * accumulation_rate_gain

        # Refractory suppresses accumulation (not eliminates)
        effective = secretion * (1.0 - self.refractory * 0.8)
        self.level += max(0.0, effective)

        # Governor 1: GABA-governed decay (Titan's own inhibitory tone)
        # High GABA = faster decay = natural regulation
        # Same formula used for neuromod clearance during dreaming
        gaba_decay_mult = 1.0 + gaba_level * 3.0
        decay = self.decay_rate * dt * (1.0 + self.level * 0.5) * gaba_decay_mult
        self.level *= max(0.0, 1.0 - decay)
        self.level = max(0.0, self.level)

        # Governor 2: Chi-governed capacity with survival floor
        # Inner programs (REFLEX, FOCUS, INTUITION, IMPULSE, VIGILANCE)
        # maintain 50% capacity even at Chi=0 (survival reflexes)
        # Outer programs floor at 30% (minimal engagement)
        chi_floor = 0.5 if self._is_inner else 0.3
        chi_factor = max(chi_floor, chi_total)
        mature_cap = 3.0  # Chi governs the rest (DNA value)
        effective_cap = (2.0 + self._maturity * (mature_cap - 2.0)) * chi_factor
        max_level = self.threshold * effective_cap
        if self.level > max_level:
            self.level = max_level

        # Track peak
        if self.level > self.peak_level:
            self.peak_level = self.level

        # Refractory decay — faster recovery with maturity
        effective_refractory_decay = self.refractory_decay * (1.0 + self._maturity * 0.5)
        self.refractory *= (1.0 - effective_refractory_decay * dt)
        self.refractory = max(0.0, self.refractory)

    def should_fire(self) -> bool:
        """Check if pressure exceeds threshold and refractory is low enough."""
        return self.level >= self.threshold and self.refractory < 0.15

    def fire(self) -> float:
        """Fire the program. Returns intensity (how much over threshold)."""
        intensity = self.level / max(0.01, self.threshold)
        self.fire_count += 1
        self.last_fire_ts = time.time()

        # Dramatic pressure drop (hormone consumed by receptors)
        self.level *= 0.15

        # Full refractory suppression
        self.refractory = self.refractory_strength

        return intensity

    def adapt_threshold(self, reward: float, lr: float = 0.01) -> None:
        """IQL-based threshold adaptation from action outcomes."""
        if reward > 0:
            # Good outcome: lower threshold slightly (fire more easily)
            self.threshold -= lr * reward
        else:
            # Bad outcome: raise threshold (more cautious)
            self.threshold += lr * abs(reward) * 1.5  # Asymmetric

        # Clamp
        self.threshold = max(0.1, min(2.0, self.threshold))

    def get_state(self) -> dict:
        return {
            "name": self.name,
            "level": self.level,
            "threshold": self.threshold,
            "refractory": self.refractory,
            "fire_count": self.fire_count,
            "last_fire_ts": self.last_fire_ts,
            "peak_level": self.peak_level,
        }

    def restore_state(self, state: dict) -> None:
        self.level = state.get("level", 0.0)
        self.threshold = state.get("threshold", self.threshold)
        self.refractory = state.get("refractory", 0.0)
        self.fire_count = state.get("fire_count", 0)
        self.last_fire_ts = state.get("last_fire_ts", 0.0)
        self.peak_level = state.get("peak_level", 0.0)


class HormonalSystem:
    """Manages all program hormones with cross-talk and circadian modulation."""

    def __init__(
        self,
        program_names: list[str],
        cross_talk: Optional[dict] = None,
        circadian: Optional[dict] = None,
        hormone_params: Optional[dict[str, dict]] = None,
    ):
        self._cross_talk_config = cross_talk or DEFAULT_CROSS_TALK
        self._circadian = circadian or CIRCADIAN
        self._hormones: dict[str, HormonalPressure] = {}
        self._last_accumulate_ts: float = time.time()

        # Initialize hormones
        params = hormone_params or {}
        for name in program_names:
            hp = params.get(name, DEFAULT_HORMONE_PARAMS.get(name, {}))
            hormone = HormonalPressure(name=name, **hp)
            # Wire cross-talk
            ct = self._cross_talk_config.get(name, {})
            hormone.excitors = ct.get("excitors", {})
            hormone.inhibitors = ct.get("inhibitors", {})
            self._hormones[name] = hormone

        # Set layer identity for Chi survival floor
        _INNER_LAYER = {"REFLEX", "FOCUS", "INTUITION", "IMPULSE", "VIGILANCE"}
        for name, hormone in self._hormones.items():
            hormone._is_inner = name in _INNER_LAYER

        # Maturity — computed from Titan's emergent time signals
        self._maturity: float = 0.0

        logger.info("[HormonalSystem] Initialized %d hormones: %s",
                     len(self._hormones), list(self._hormones.keys()))

    def get_hormone(self, name: str) -> Optional[HormonalPressure]:
        return self._hormones.get(name)

    @property
    def maturity(self) -> float:
        return self._maturity

    def update_maturity(
        self,
        great_epochs: int = 0,
        sphere_radius: float = 1.0,
        consciousness_epochs: int = 0,
        total_fires: int = 0,
    ) -> float:
        """
        Compute nervous system maturity from Titan's EMERGENT time signals.

        Not human clock time — Titan's own developmental milestones:
        - GREAT EPOCHs: crystallized growth moments
        - Sphere clock radius: how balanced/contracted (closer to center = mature)
        - Consciousness epochs: depth of self-reflection experience
        - Total hormone fires: lived experience through the nervous system

        Returns maturity [0.0 = infant, 1.0 = fully mature]
        """
        m1 = min(1.0, great_epochs / 10.0)            # 10 GREAT PULSEs
        m2 = max(0.0, 1.0 - sphere_radius / 0.5)      # radius 0.5→0 = mature
        m3 = min(1.0, consciousness_epochs / 5000.0)   # 5000 epochs
        m4 = min(1.0, total_fires / 500.0)             # 500 fires

        self._maturity = round(
            0.3 * m1 + 0.3 * m2 + 0.2 * m3 + 0.2 * m4, 4)

        # Propagate to all hormones
        for hormone in self._hormones.values():
            hormone._maturity = self._maturity

        return self._maturity

    def get_levels(self) -> dict[str, float]:
        return {n: h.level for n, h in self._hormones.items()}

    def accumulate_all(
        self,
        stimuli: dict[str, float],
        dt: float,
        is_dreaming: bool = False,
        gaba_level: float = 0.35,
        chi_total: float = 0.6,
        accumulation_rate_gain: float = 1.0,
    ) -> None:
        """Accumulate all hormones with stimuli, cross-talk, circadian, and self-emergent governors."""
        levels = self.get_levels()
        circadian_mode = "dreaming" if is_dreaming else "awake"
        circadian_mults = self._circadian.get(circadian_mode, {})

        for name, hormone in self._hormones.items():
            stimulus = stimuli.get(name, 0.0)
            circ_mult = circadian_mults.get(name, 1.0)
            hormone.accumulate(stimulus, dt, levels, circ_mult,
                               gaba_level=gaba_level,
                               chi_total=chi_total,
                               accumulation_rate_gain=accumulation_rate_gain)

    def reset_thresholds_to_dna(self) -> None:
        """Reset all thresholds to DNA values. Use after governor changes."""
        for name, hormone in self._hormones.items():
            dna_thresh = DEFAULT_HORMONE_PARAMS.get(name, {}).get("fire_threshold", 0.5)
            old = hormone.threshold
            hormone.threshold = dna_thresh
            logger.info("[Hormonal] Reset %s threshold: %.3f → %.3f", name, old, dna_thresh)

    def get_fire_candidates(self) -> list[str]:
        """Return names of hormones ready to fire."""
        return [n for n, h in self._hormones.items() if h.should_fire()]

    def fire(self, name: str) -> float:
        """Fire a specific hormone. Returns intensity."""
        hormone = self._hormones.get(name)
        if hormone and hormone.should_fire():
            intensity = hormone.fire()
            logger.info("[Hormone] %s FIRED — intensity=%.2f (threshold=%.3f, "
                        "refractory=%.2f)", name, intensity, hormone.threshold,
                        hormone.refractory)
            return intensity
        return 0.0

    def adapt(self, name: str, reward: float) -> None:
        """Adapt a hormone's threshold from action outcome."""
        hormone = self._hormones.get(name)
        if hormone:
            old_threshold = hormone.threshold
            hormone.adapt_threshold(reward)
            if abs(old_threshold - hormone.threshold) > 0.001:
                logger.debug("[Hormone] %s threshold %.3f → %.3f (reward=%.2f)",
                             name, old_threshold, hormone.threshold, reward)

    def get_all_states(self) -> dict:
        return {n: h.get_state() for n, h in self._hormones.items()}

    # Aliases for hot-reload API consistency
    get_state = get_all_states

    def restore_all_states(self, states: dict) -> None:
        for name, state in states.items():
            if name in self._hormones:
                self._hormones[name].restore_state(state)
        logger.info("[HormonalSystem] Restored %d hormone states", len(states))

    restore_state = restore_all_states

    def save(self, path: str) -> None:
        """Atomic write (tmp→rename) to prevent corruption on crash."""
        data = self.get_all_states()
        tmp = path + ".tmp"
        Path(tmp).write_text(json.dumps(data, indent=2))
        os.replace(tmp, path)

    def load(self, path: str) -> None:
        p = Path(path)
        if p.exists():
            data = json.loads(p.read_text())
            self.restore_all_states(data)


# ── Stimulus Extraction ──────────────────────────────────────────────

# Boredom half-life: after this many seconds without exploration,
# CURIOSITY stimulus reaches 0.5 from boredom alone
BOREDOM_HALF_LIFE = 1800.0  # 30 minutes


def extract_stimuli(
    observables: dict,
    topology: Optional[dict] = None,
    dreaming: Optional[dict] = None,
    events: Optional[dict] = None,
) -> dict[str, float]:
    """
    Extract per-hormone stimulus from current Trinity observation state.

    Each stimulus is [0, 1] — the magnitude of the relevant signal
    that feeds this hormone's pressure accumulation.

    Args:
        observables: dict with 6 parts, each having 5 values
                     (coherence, magnitude, velocity, direction, polarity)
        topology: dict with volume, curvature, clusters
        dreaming: dict with fatigue, readiness
        events: dict with time_since_explore, time_since_social, etc.
    """
    obs = observables or {}
    topo = topology or {}
    dream = dreaming or {}
    ev = events or {}

    # Collect all values for aggregate metrics
    all_coherences = []
    all_magnitudes = []
    all_velocities = []
    for part_name, values in obs.items():
        if isinstance(values, dict):
            all_coherences.append(values.get("coherence", 0.5))
            all_magnitudes.append(values.get("magnitude", 0.5))
            all_velocities.append(abs(values.get("velocity", 0.0)))
        elif isinstance(values, (list, tuple)) and len(values) >= 5:
            all_coherences.append(values[0])
            all_magnitudes.append(values[1])
            all_velocities.append(abs(values[2]))

    mean_coherence = sum(all_coherences) / max(1, len(all_coherences)) if all_coherences else 0.5
    mean_velocity = sum(all_velocities) / max(1, len(all_velocities)) if all_velocities else 0.0
    max_velocity = max(all_velocities) if all_velocities else 0.0

    # Deficit: how far from center (0.5) are magnitudes?
    all_deficits = [abs(m - 0.5) for m in all_magnitudes] if all_magnitudes else [0.0]
    mean_deficit = sum(all_deficits) / max(1, len(all_deficits))
    max_deficit = max(all_deficits) if all_deficits else 0.0

    # Topology signals
    volume = topo.get("volume", 0.5)
    curvature = topo.get("curvature", 0.0)

    # Fatigue
    fatigue = dream.get("fatigue", 0.0)

    # Time-based signals
    time_since_explore = ev.get("time_since_explore", 3600.0)
    time_since_social = ev.get("time_since_social", 3600.0)
    time_since_create = ev.get("time_since_create", 3600.0)

    stimuli = {}

    # REFLEX: responds to sudden changes (high velocity = perturbation)
    stimuli["REFLEX"] = min(1.0, max_velocity * 2.0)

    # FOCUS: responds to imbalance (deficits need attention)
    stimuli["FOCUS"] = min(1.0, mean_deficit * 2.0)

    # INTUITION: responds to pattern complexity (curvature + low coherence)
    stimuli["INTUITION"] = min(1.0, curvature * 0.3 + (1.0 - mean_coherence) * 0.5)

    # IMPULSE: responds to sustained deficit + urgency
    stimuli["IMPULSE"] = min(1.0, max_deficit * 1.5 + mean_velocity * 0.5)

    # VIGILANCE: responds to anomalies (high single deficit + velocity)
    stimuli["VIGILANCE"] = min(1.0, max_deficit * 1.2 + max_velocity * 0.8)

    # CREATIVITY: responds to inspiration + calm (post-reflection creative urge)
    boredom_create = min(1.0, time_since_create / BOREDOM_HALF_LIFE)
    stimuli["CREATIVITY"] = min(1.0, mean_coherence * 0.4 + boredom_create * 0.6)

    # CURIOSITY: responds to boredom + novelty deficit (KEY: time-based buildup)
    boredom_explore = min(1.0, time_since_explore / BOREDOM_HALF_LIFE)
    novelty_deficit = 1.0 - min(1.0, volume * 2.0)  # Low topology volume = unexplored
    stimuli["CURIOSITY"] = min(1.0, boredom_explore * 0.6 + novelty_deficit * 0.4)

    # EMPATHY: responds to social signals
    social_hunger = min(1.0, time_since_social / BOREDOM_HALF_LIFE)
    stimuli["EMPATHY"] = min(1.0, social_hunger * 0.7 + mean_coherence * 0.3)

    # KIN LONGING: isolation from kin drives seeking behavior (1h half-life)
    time_since_kin = ev.get("time_since_kin", 7200.0)
    kin_hunger = min(1.0, time_since_kin / 3600.0)
    stimuli["EMPATHY"] = min(1.0, stimuli["EMPATHY"] + kin_hunger * 0.15)

    # REFLECTION: responds to post-action stability + fatigue + kin isolation
    stimuli["REFLECTION"] = min(1.0, mean_coherence * 0.4 + fatigue * 0.6 + kin_hunger * 0.10)

    # INSPIRATION: responds to harmony (high coherence across parts)
    harmony = mean_coherence * (1.0 - mean_deficit)
    stimuli["INSPIRATION"] = min(1.0, harmony * 1.5 + curvature * 0.2)

    return stimuli
