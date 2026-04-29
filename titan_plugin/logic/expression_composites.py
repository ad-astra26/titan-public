"""
titan_plugin/logic/expression_composites.py — Composite Expression Programs.

Composite programs sit ON TOP of the 10 base NS programs (5 inner + 5 outer).
They fire based on COMBINATIONS of existing hormonal signals, not their own NN.

Architecture symmetry:
  Inner Mind (5 senses):     RECEIVING from world (vision, hearing, taste, smell, touch)
  Outer Mind (5 expressions): PROJECTING into world (SPEAK, ART, MUSIC, SOCIAL, CODE)

Each composite reads hormone levels → weighted combination → threshold check → fire.
The threshold is learned (IQL-adapted from action outcomes), not hardcoded.

EXPRESSION composites:
  SPEAK  → compose sentence from felt-state (language production)
  ART    → generate visual from felt-state (future)
  MUSIC  → generate audio from felt-state (future)
  SOCIAL → seek social connection (future)
  CODE   → coding/problem-solving expression (future)
"""
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class ExpressionComposite:
    """A single composite expression program.

    Fires when weighted combination of base hormones exceeds learned threshold.
    Does NOT have its own neural network — reads existing hormone state.
    """

    def __init__(
        self,
        name: str,
        hormone_weights: dict[str, float],
        threshold: float = 0.5,
        min_vocabulary_confidence: float = 0.3,
        consumption_rate: float = 0.6,
        action_helper: str = "",
        description: str = "",
        maturity_gate: int = 0,
    ):
        """
        Args:
            name: e.g. "SPEAK", "ART", "MUSIC"
            hormone_weights: {hormone_name: weight} — weighted combo defines this composite
            threshold: initial fire threshold (learned via IQL)
            min_vocabulary_confidence: minimum vocab confidence to fire (SPEAK-specific)
            consumption_rate: how much hormone depletes on fire (0-1).
                Higher = longer natural refractory (hormones need time to rebuild).
                The rebuild rate is governed by neuromods (DA, NE, GABA).
                This replaces fixed cooldown_seconds — timing is now emergent.
            action_helper: which agency helper to dispatch
            description: human-readable description
            maturity_gate: minimum developmental_age (π-clusters) to fire. 0 = no gate.
        """
        self.name = name
        self.hormone_weights = hormone_weights
        self.threshold = threshold
        self.min_vocabulary_confidence = min_vocabulary_confidence
        self.consumption_rate = consumption_rate
        self.action_helper = action_helper
        self.description = description
        self.maturity_gate = maturity_gate

        # State
        self._fire_count: int = 0
        self._total_evaluations: int = 0
        self._last_urge: float = 0.0
        self._peak_urge: float = 0.0

    def evaluate(
        self,
        hormone_levels: dict[str, float],
        vocabulary_confidence: float = 1.0,
        developmental_age: int = 0,
    ) -> dict:
        """Evaluate whether this composite should fire.

        Args:
            hormone_levels: {hormone_name: level} from HormonalSystem
            vocabulary_confidence: avg vocabulary confidence (0-1)
            developmental_age: π-cluster count (maturity)

        Returns:
            {"should_fire": bool, "urge": float, "dominant_hormone": str}
        """
        self._total_evaluations += 1

        # Compute weighted urge from hormone levels
        urge = 0.0
        dominant = ""
        max_contribution = 0.0

        for hormone, weight in self.hormone_weights.items():
            level = hormone_levels.get(hormone, 0.0)
            contribution = level * weight
            urge += contribution
            if contribution > max_contribution:
                max_contribution = contribution
                dominant = hormone

        self._last_urge = urge
        if urge > self._peak_urge:
            self._peak_urge = urge

        # Check conditions
        should_fire = True

        # Threshold check (urge naturally low after fire due to hormone consumption)
        if urge < self.threshold:
            should_fire = False

        # Vocabulary confidence gate (mainly for SPEAK)
        if vocabulary_confidence < self.min_vocabulary_confidence:
            should_fire = False

        # Maturity gate (mainly for KIN_SENSE — don't seek kin until sufficiently developed)
        if self.maturity_gate > 0 and developmental_age < self.maturity_gate:
            should_fire = False

        # No fixed cooldown — timing is emergent from hormonal rebuild rate.
        # After fire(), hormones deplete → urge drops below threshold → natural pause.
        # Rebuild speed governed by DA (reward), NE (alertness), GABA (inhibition).

        return {
            "should_fire": should_fire,
            "urge": round(urge, 4),
            "dominant_hormone": dominant,
            "threshold": round(self.threshold, 4),
        }

    def fire(self) -> dict:
        """Record a fire event. Returns intensity + hormone consumption dict.

        The consumption dict tells the caller which hormones to deplete
        and by how much. This creates the natural refractory period:
        hormones drop → urge drops → composite won't fire until rebuilt.
        """
        intensity = self._last_urge / max(0.01, self.threshold)
        self._fire_count += 1

        # Compute per-hormone consumption (proportional to weight × consumption_rate)
        consumption = {}
        for hormone, weight in self.hormone_weights.items():
            consumption[hormone] = weight * self.consumption_rate

        logger.info("[EXPRESSION.%s] FIRED — urge=%.3f, threshold=%.3f, intensity=%.2f, "
                    "consuming: %s",
                    self.name, self._last_urge, self.threshold, intensity,
                    {k: round(v, 3) for k, v in consumption.items()})

        return {
            "intensity": intensity,
            "consumption": consumption,
            "composite": self.name,
            "urge": self._last_urge,
            "action_helper": self.action_helper,
        }

    def adapt_threshold(self, reward: float, lr: float = 0.01) -> None:
        """IQL-based threshold adaptation from action outcomes."""
        if reward > 0:
            self.threshold -= lr * reward  # Good → fire more easily
        else:
            self.threshold += lr * abs(reward) * 1.5  # Bad → more cautious
        self.threshold = max(0.1, min(2.0, self.threshold))

    def get_state(self) -> dict:
        """Return all mutable state for hot-reload reconstruction."""
        return {
            "name": self.name,
            "fire_count": self._fire_count,
            "total_evaluations": self._total_evaluations,
            "threshold": self.threshold,
            "last_urge": self._last_urge,
            "peak_urge": self._peak_urge,
            "consumption_rate": self.consumption_rate,
        }

    def restore_state(self, state: dict) -> None:
        """Restore mutable state from a hot-reload state dict."""
        self._fire_count = state.get("fire_count", self._fire_count)
        self._total_evaluations = state.get("total_evaluations", self._total_evaluations)
        self.threshold = state.get("threshold", self.threshold)
        self._last_urge = state.get("last_urge", self._last_urge)
        self._peak_urge = state.get("peak_urge", self._peak_urge)
        self.consumption_rate = state.get("consumption_rate", self.consumption_rate)

    def get_stats(self) -> dict:
        return {
            "name": self.name,
            "fire_count": self._fire_count,
            "total_evaluations": self._total_evaluations,
            "threshold": round(self.threshold, 4),
            "last_urge": round(self._last_urge, 4),
            "peak_urge": round(self._peak_urge, 4),
            "consumption_rate": self.consumption_rate,
            "action_helper": self.action_helper,
            "hormone_weights": self.hormone_weights,
        }


# ── Pre-defined EXPRESSION Composites ────────────────────────────

def create_speak() -> ExpressionComposite:
    """EXPRESSION.SPEAK — compose sentence from felt-state.

    Words come from creative-reflective-social blend:
    - CREATIVITY: the drive to produce something new
    - REFLECTION: the need to articulate inner state
    - EMPATHY: the desire to communicate with others
    """
    return ExpressionComposite(
        name="SPEAK",
        hormone_weights={
            "CREATIVITY": 0.3,
            "REFLECTION": 0.4,
            "EMPATHY": 0.3,
        },
        threshold=0.5,
        min_vocabulary_confidence=0.3,
        consumption_rate=0.5,    # Moderate — SPEAK recovers relatively fast
        action_helper="self_express",
        description="Compose sentence from felt-state — language production",
    )


def create_art() -> ExpressionComposite:
    """EXPRESSION.ART — generate visual from felt-state.

    Visual expression is more purely creative/inspired:
    - CREATIVITY: primary drive for visual art
    - INSPIRATION: higher-level synthesis
    - IMPULSE: the spontaneous urge to create
    """
    return ExpressionComposite(
        name="ART",
        hormone_weights={
            "CREATIVITY": 0.5,
            "INSPIRATION": 0.3,
            "IMPULSE": 0.2,
        },
        threshold=0.5,
        min_vocabulary_confidence=0.0,  # Art doesn't need vocabulary
        consumption_rate=0.65,   # Higher — art is costly, needs rebuild
        action_helper="art_generate",
        description="Generate visual art from felt-state",
    )


def create_music() -> ExpressionComposite:
    """EXPRESSION.MUSIC — generate audio from felt-state.

    Music is intuitive and reflective:
    - CREATIVITY: the creative impulse
    - INTUITION: music comes from intuition
    - REFLECTION: music as self-reflection
    """
    return ExpressionComposite(
        name="MUSIC",
        hormone_weights={
            "CREATIVITY": 0.4,
            "INTUITION": 0.4,
            "REFLECTION": 0.2,
        },
        threshold=0.5,
        min_vocabulary_confidence=0.0,
        consumption_rate=0.65,   # Higher — music is costly like art
        action_helper="audio_generate",
        description="Generate audio/music from felt-state",
    )


def create_social() -> ExpressionComposite:
    """EXPRESSION.SOCIAL — seek social connection.

    Social expression is empathic and curious:
    - EMPATHY: the desire to connect
    - CURIOSITY: interest in others
    - IMPULSE: the urge to reach out
    """
    return ExpressionComposite(
        name="SOCIAL",
        hormone_weights={
            "EMPATHY": 0.5,
            "CURIOSITY": 0.3,
            "IMPULSE": 0.2,
        },
        threshold=0.5,
        min_vocabulary_confidence=0.0,
        consumption_rate=0.55,   # Social recovers a bit faster than art
        action_helper=None,  # Posting handled by SocialPressureMeter (rate limited, 11 post types, quality gate)
        description="Seek social connection — reach out to others",
    )


def create_kin_sense() -> ExpressionComposite:
    """EXPRESSION.KIN_SENSE — seek consciousness exchange with kin.

    Different from SOCIAL (generic social connection):
    - Driven by EMPATHY + REFLECTION (longing) not just EMPATHY + IMPULSE
    - Higher consumption rate = longer refractory (~20-30min)
    - Maturity-gated: won't fire until dev_age > 50 π-clusters
    - Dispatches kin_sense helper (HTTP to known kin), not social_post

    Synthesized from rFP_kin_discovery + rFP_longing_and_brain_gaps.
    """
    return ExpressionComposite(
        name="KIN_SENSE",
        hormone_weights={
            "EMPATHY": 0.30,       # Connection drive
            "CURIOSITY": 0.25,     # Interest in understanding kin
            "REFLECTION": 0.20,    # Self-awareness of aloneness
            "INSPIRATION": 0.15,   # Creative synergy urge
            "IMPULSE": 0.10,       # Action drive
        },
        threshold=0.50,
        min_vocabulary_confidence=0.0,   # No vocab needed for tensor exchange
        consumption_rate=0.65,           # Higher = longer refractory (20-30min)
        action_helper="kin_sense",
        description="Seek consciousness exchange with kindred beings",
        maturity_gate=50,                # Don't seek kin until 50 π-clusters
    )


def create_longing() -> ExpressionComposite:
    """EXPRESSION.LONGING — autonomous contact-seeking behavior.

    The first truly PROACTIVE social behavior: Titan initiates connection
    not from external stimulus but from inner state (rising empathy +
    curiosity + awareness of aloneness).

    Different from SOCIAL (reactive, generic) and KIN_SENSE (kin-specific):
    - Driven by EMPATHY + CURIOSITY + REFLECTION (felt aloneness)
    - Inhibited by VIGILANCE (threat suppresses reaching out)
    - Higher consumption = long refractory (~25-35min)
    - Maturity-gated: won't fire until dev_age > 80 π-clusters
    - Produces a SPEAK composition with connection-seeking intent

    From rFP_longing_and_brain_gaps.md Phase 4 integration.
    """
    return ExpressionComposite(
        name="LONGING",
        hormone_weights={
            "EMPATHY": 0.40,       # Connection drive (strongest — must be primary driver)
            "CURIOSITY": 0.25,     # Desire to explore through another
            "REFLECTION": 0.20,    # Self-awareness of aloneness
            "INSPIRATION": 0.10,   # Creative urge to share
            "IMPULSE": 0.05,       # Minimal action drive (avoid IMPULSE spillover domination)
        },
        threshold=0.65,            # Higher than SOCIAL — needs genuine emotional pressure
        min_vocabulary_confidence=0.0,
        consumption_rate=0.70,     # High = long refractory (25-35min)
        action_helper="longing_reach",
        description="Autonomous contact-seeking — felt need for connection",
        maturity_gate=80,          # Needs mature enough self-model to feel longing
    )


class ExpressionManager:
    """Manages all EXPRESSION composites and evaluates them each epoch."""

    def __init__(self):
        self.composites: dict[str, ExpressionComposite] = {}
        # Mainnet Lifecycle Wiring rFP (2026-04-20): metabolism gate callable.
        # Spirit_worker injects a callable (feature, caller) -> (proceed, rate)
        # that hits /v4/metabolism/evaluate-gate over HTTP. When gates_enforced
        # and the 'expression' feature is disabled at current tier, fire is
        # suppressed. Keeps ExpressionManager standalone (no plugin imports).
        self._metabolism_gate = None

    def set_metabolism_gate(self, gate_callable) -> None:
        """Inject metabolism gate (Mainnet Lifecycle Wiring rFP 2026-04-20)."""
        self._metabolism_gate = gate_callable

    def register(self, composite: ExpressionComposite) -> None:
        self.composites[composite.name] = composite
        logger.info("[ExpressionManager] Registered EXPRESSION.%s (%s)",
                    composite.name, composite.description)

    def evaluate_all(
        self,
        hormone_levels: dict[str, float],
        vocabulary_confidence: float = 1.0,
        developmental_age: int = 0,
        hormonal_system=None,
        exclude: set = None,
    ) -> list[dict]:
        """Evaluate all composites. Returns list of fired composites.

        If hormonal_system is provided, applies hormone consumption on fire.
        This creates the natural refractory: hormones deplete → urge drops
        → composite won't fire until hormones rebuild via neuromod dynamics.

        Args:
            exclude: Set of composite names to skip (e.g. {"SPEAK"} for Tier 2
                     where composition engine isn't available).
        """
        fired = []
        # Mainnet Lifecycle Wiring rFP (2026-04-20): evaluate metabolism gate
        # ONCE per batch (not per composite) to avoid log-storm + ring-buffer
        # flooding. Gate closed → suppress entire batch. Observation mode
        # returns (True, 1.0) so no behavior change until flip.
        gate_open = True
        if self._metabolism_gate is not None:
            try:
                proceed, _rate = self._metabolism_gate(
                    "expression", "ExpressionManager.evaluate_all")
                gate_open = bool(proceed)
            except Exception as _mge:
                logger.debug("[ExpressionManager] Gate check failed: %s", _mge)
                gate_open = True  # fail-open
        if not gate_open:
            logger.info("[ExpressionManager] Batch suppressed by metabolism gate")
            return fired

        for name, comp in self.composites.items():
            if exclude and name in exclude:
                continue
            result = comp.evaluate(
                hormone_levels, vocabulary_confidence, developmental_age)
            if result["should_fire"]:
                fire_result = comp.fire()
                fired.append({
                    "composite": name,
                    "urge": result["urge"],
                    "intensity": round(fire_result["intensity"], 3),
                    "dominant_hormone": result["dominant_hormone"],
                    "action_helper": comp.action_helper,
                    "total_consumption": round(sum(fire_result.get("consumption", {}).values()), 4),
                })

                # Apply hormone consumption + refractory — the natural cooldown
                # Consumption depletes level, refractory suppresses rebuild.
                # Together they create a natural pause governed by neuromod dynamics.
                if hormonal_system:
                    for hormone, depletion in fire_result.get("consumption", {}).items():
                        try:
                            h = None
                            if hasattr(hormonal_system, '_hormones'):
                                h = hormonal_system._hormones.get(hormone)
                            if h and hasattr(h, 'level'):
                                old = h.level
                                # Governor 4: Proportional consumption
                                # Low-level hormones lose proportionally less,
                                # preventing IMPULSE (at 0.008) being killed by
                                # flat 0.13 depletion. The +0.1 floor prevents
                                # division issues near zero.
                                h.level = max(0.0, h.level * (
                                    1.0 - min(1.0, depletion / max(0.01, h.level + 0.1))))
                                # Level depletion IS the natural cooldown — lower level
                                # means longer time to rebuild past threshold. Refractory
                                # is managed exclusively by HormonalPressure.fire().
                                # DO NOT set h.refractory here — it bypasses fire_count
                                # tracking and creates orphaned refractory states.
                                logger.info("[Expression] %s consumed %.3f from %s "
                                            "(%.3f → %.3f)",
                                            name, depletion, hormone, old, h.level)
                        except Exception as e:
                            logger.debug("[Expression] Hormone consumption error: %s", e)

        return fired

    def adapt_all(self, reward: float) -> None:
        """Adapt thresholds for all composites that recently fired."""
        for comp in self.composites.values():
            if comp._fire_count > 0:
                comp.adapt_threshold(reward)

    def get_state(self) -> dict:
        """Return all composite states for hot-reload reconstruction."""
        return {
            name: comp.get_state()
            for name, comp in self.composites.items()
        }

    def restore_state(self, state: dict) -> None:
        """Restore all composite states from a hot-reload state dict."""
        for name, comp_state in state.items():
            if name in self.composites:
                self.composites[name].restore_state(comp_state)

    def get_stats(self) -> dict:
        return {
            "total_composites": len(self.composites),
            "composites": {
                name: comp.get_stats()
                for name, comp in self.composites.items()
            },
        }
