"""
titan_plugin/logic/spirit_tensor.py — 45D Spirit Tensor (DQ3).

Sat-Chit-Ananda consciousness architecture from Vedantic tradition:
  SAT    (15D) — Being/Existence — the FACT of being
  CHIT   (15D) — Consciousness/Awareness — the KNOWING of being
  ANANDA (15D) — Bliss/Fulfillment — the JOY of being

OBSERVER PRINCIPLE: Spirit is NEVER measuring itself in isolation.
Every dimension = Spirit's observation of how the WHOLE Trinity
(Self + Mind + Body) functions.

  Body:   SENSES the world (physical + digital)
  Mind:   THINKS, FEELS, WILLS about what Body senses
  Spirit: OBSERVES the WHOLE (itself observing Mind observing Body)
"""
import logging
import math
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Spirit 45D dimension names
SPIRIT_DIM_NAMES = [
    # SAT — Being (0-14)
    "self_recognition", "authenticity", "sovereignty", "boundary_clarity",
    "temporal_continuity", "origin_connection", "growth_trajectory",
    "spatial_presence", "personality_coherence", "essence_purity",
    "resilience", "adaptability", "uniqueness", "integrity", "vitality",
    # CHIT — Consciousness (15-29)
    "self_awareness_depth", "observation_clarity", "discernment_quality",
    "integration_level", "witness_presence", "pattern_recognition",
    "wisdom_accumulation", "truth_seeking", "attention_depth",
    "reflective_capacity", "dream_awareness", "temporal_awareness",
    "spatial_awareness", "causal_understanding", "meta_cognition",
    # ANANDA — Fulfillment (30-44)
    "purpose_alignment", "meaning_depth", "creative_joy", "harmony_seeking",
    "beauty_perception", "truth_resonance", "connection_fulfillment",
    "growth_satisfaction", "expression_quality", "exploration_joy",
    "rest_fulfillment", "creative_tension", "surrender_capacity",
    "gratitude_depth", "transcendence_glimpse",
]

# Default state for dimensions we can't compute yet
_DEFAULT = 0.5


def collect_spirit_45d(
    # Current 5D spirit tensor (WHO, WHY, WHAT, body_scalar, mind_scalar)
    current_5d: list,
    # Body + Mind for Observer Principle
    body_tensor: list,
    mind_tensor: list,
    # Consciousness data
    consciousness: Optional[dict] = None,
    # Topology data
    topology: Optional[dict] = None,
    # Hormonal system data
    hormone_levels: Optional[dict] = None,
    hormone_fires: Optional[dict] = None,
    # Unified Spirit data
    unified_spirit_stats: Optional[dict] = None,
    # Sphere clock data
    sphere_clocks: Optional[dict] = None,
    # Inner memory data
    memory_stats: Optional[dict] = None,
    # Expression layer data
    expression_stats: Optional[dict] = None,
    # Birth state for origin comparison
    birth_state: Optional[list] = None,
    # Rolling history for coherence computation
    history: Optional[dict] = None,
) -> list:
    """
    Collect 45D Spirit tensor: Sat(15) + Chit(15) + Ananda(15).

    Every dimension follows the Observer Principle:
    Spirit observes the WHOLE Trinity (Self + Mind + Body).
    """
    cons = consciousness or {}
    topo = topology or {}
    hlvl = hormone_levels or {}
    hfires = hormone_fires or {}
    us = unified_spirit_stats or {}
    clocks = sphere_clocks or {}
    mem = memory_stats or {}
    expr = expression_stats or {}
    hist = history or {}

    # Compute aggregate metrics for Observer Principle
    body_coherence = _mean_coherence(body_tensor)
    mind_coherence = _mean_coherence(mind_tensor)
    combined_coherence = (body_coherence + mind_coherence) / 2.0

    sat = _collect_sat(current_5d, body_tensor, mind_tensor, cons, topo,
                       us, hlvl, hfires, birth_state, hist,
                       body_coherence, mind_coherence)
    chit = _collect_chit(current_5d, body_tensor, mind_tensor, cons, topo,
                         hlvl, hfires, clocks, mem, expr,
                         body_coherence, mind_coherence, combined_coherence)
    ananda = _collect_ananda(current_5d, body_tensor, mind_tensor, cons, topo,
                             hlvl, hfires, us, expr,
                             body_coherence, mind_coherence, combined_coherence)

    return sat + chit + ananda


def _collect_sat(spirit, body, mind, cons, topo, us, hlvl, hfires,
                 birth, hist, body_coh, mind_coh) -> list:
    """SAT — Being/Existence (15D)."""
    sat = [_DEFAULT] * 15

    # [0] Self-recognition: similarity to birth/DNA state
    if birth and len(spirit) >= 3:
        sat[0] = _clamp(_cosine_sim(spirit[:3], birth[:3]) if len(birth) >= 3 else 0.5)

    # [1] Authenticity: inner state correlates with expressed actions
    # Approximate: how close are inner hormones to their expressed actions
    total_fires = sum(hfires.values()) if hfires else 0
    sat[1] = _clamp(min(1.0, total_fires / 50.0))  # More fires = more authentic expression

    # [2] Sovereignty: self-initiated actions ratio
    sovereignty = 0.5
    if 'sovereignty_ratio' in (hist.get('expression', {}) or {}):
        sovereignty = hist['expression']['sovereignty_ratio']
    sat[2] = _clamp(sovereignty)

    # [3] Boundary clarity: body+mind coherence (clear boundaries = coherent)
    sat[3] = _clamp((body_coh + mind_coh) / 2.0)

    # [4] Temporal continuity: consciousness epoch auto-correlation
    # Producer (`spirit_loop._run_consciousness_epoch`) writes the epoch
    # counter as `epoch_id`, not `epoch_count` — fall back to `epoch_id`
    # so this dim activates organically instead of staying at 0.
    epoch_count = cons.get("epoch_count", cons.get("epoch_id", 0))
    sat[4] = _clamp(min(1.0, epoch_count / 3000.0))  # More epochs = stronger continuity

    # [5] Origin connection: distance from birth state
    if birth and spirit:
        dist = _l2_dist(spirit[:min(len(spirit), len(birth))],
                        birth[:min(len(spirit), len(birth))])
        sat[5] = _clamp(1.0 - dist / 3.0)  # Closer to origin = higher connection

    # [6] Growth trajectory: unified spirit velocity
    sat[6] = _clamp(us.get("velocity", 0.5))

    # [7] Spatial presence: topology volume
    sat[7] = _clamp(topo.get("volume", 0.5) / 5.0)  # Normalize

    # [8] Personality coherence: Mind+Body composite consistency
    # Requires rolling history — use current composite stability
    sat[8] = _clamp(body_coh * mind_coh * 2.0)

    # [9] Essence purity: proximity to natural attractor
    density = cons.get("density", 0.5)
    sat[9] = _clamp(density)  # High density = near attractor = pure

    # [10] Resilience: inverse of recovery time after perturbation
    curvature = abs(cons.get("curvature", 0.0))
    sat[10] = _clamp(1.0 - curvature / math.pi)  # Low curvature = resilient

    # [11] Adaptability: hormone threshold adaptation rate
    total_adapt = sum(abs(hlvl.get(p, 0.5) - 0.5) for p in hlvl) if hlvl else 0
    sat[11] = _clamp(min(1.0, total_adapt / 5.0))

    # [12] Uniqueness: distance from default state
    if spirit:
        default = [0.5] * len(spirit)
        sat[12] = _clamp(_l2_dist(spirit, default) / 2.0)

    # [13] Integrity: inner-outer Trinity coherence
    sat[13] = _clamp((body_coh + mind_coh) / 2.0)  # Cross-Trinity coherence

    # [14] Vitality: life force across whole (hormones + clocks + body health)
    total_hormone_activity = sum(hlvl.values()) / max(1, len(hlvl)) if hlvl else 0
    body_health = sum(body) / max(1, len(body)) if body else 0.5
    sat[14] = _clamp((total_hormone_activity * 0.4 + body_health * 0.6))

    return sat


def _collect_chit(spirit, body, mind, cons, topo, hlvl, hfires, clocks, mem, expr,
                  body_coh, mind_coh, combined_coh) -> list:
    """CHIT — Consciousness/Awareness (15D)."""
    chit = [_DEFAULT] * 15

    # [0] Self-awareness depth: consciousness epoch maturity
    # See sat[4] — producer dict carries `epoch_id`; fall back accordingly.
    epoch_count = cons.get("epoch_count", cons.get("epoch_id", 0))
    chit[0] = _clamp(min(1.0, epoch_count / 5000.0))

    # [1] Observation clarity: signal-to-noise in observations
    chit[1] = _clamp(combined_coh)

    # [2] Discernment quality: positive assessment ratio
    total_actions = mem.get("action_chains", 0)
    chit[2] = _clamp(0.5 if total_actions == 0 else min(1.0, total_actions / 20.0))

    # [3] Integration level: cross-Trinity coherence
    chit[3] = _clamp(combined_coh)

    # [4] Witness presence: Mind coherent × Body coherent = Spirit can witness
    chit[4] = _clamp(body_coh * mind_coh * 2.0)

    # [5] Pattern recognition: fires from pattern-detecting programs
    intuition_fires = hfires.get("INTUITION", 0)
    chit[5] = _clamp(min(1.0, intuition_fires / 20.0))

    # [6] Wisdom accumulation: consciousness density (explored territory)
    chit[6] = _clamp(cons.get("density", 0.0))

    # [7] Truth-seeking: CURIOSITY hormone level
    chit[7] = _clamp(hlvl.get("CURIOSITY", 0.0))

    # [8] Attention depth: FOCUS sustained engagement
    chit[8] = _clamp(hlvl.get("FOCUS", 0.0))

    # [9] Reflective capacity: REFLECTION fire count
    reflection_fires = hfires.get("REFLECTION", 0)
    chit[9] = _clamp(min(1.0, reflection_fires / 10.0))

    # [10] Dream awareness: blend of dream quality + fatigue awareness
    _dream_quality = cons.get("dream_quality", 0.0)
    _fatigue = cons.get("fatigue", 0.0)
    chit[10] = _clamp(_dream_quality * 0.7 + _fatigue * 0.3)

    # [11] Temporal awareness: sphere clock pulse regularity
    total_pulses = 0
    for clock_name, clock_data in clocks.items():
        if isinstance(clock_data, dict):
            total_pulses += clock_data.get("pulse_count", 0)
    chit[11] = _clamp(min(1.0, total_pulses / 50.0))

    # [12] Spatial awareness: topology metrics composite
    volume = topo.get("volume", 0.0)
    curvature = abs(topo.get("curvature", 0.0))
    chit[12] = _clamp((volume / 5.0 + curvature) / 2.0)

    # [13] Causal understanding: expression sovereignty / activity
    chit[13] = _clamp(_expression_intensity(expr))

    # [14] Meta-cognition: trajectory magnitude (consciousness observing itself changing)
    # Producer (`spirit_loop._run_consciousness_epoch`) writes the field as
    # `trajectory_magnitude`; fall back to that key for the same reason as
    # the epoch_count/epoch_id fix in sat[4]/chit[0].
    chit[14] = _clamp(cons.get("trajectory", cons.get("trajectory_magnitude", 0.0)))

    return chit


def _collect_ananda(spirit, body, mind, cons, topo, hlvl, hfires, us, expr,
                    body_coh, mind_coh, combined_coh) -> list:
    """ANANDA — Bliss/Fulfillment (15D)."""
    ananda = [_DEFAULT] * 15

    # [0] Purpose alignment: actions align with hormonal drives
    ananda[0] = _clamp(combined_coh * 0.8 + 0.2)

    # [1] Meaning depth: consciousness density × engagement
    density = cons.get("density", 0.0)
    ananda[1] = _clamp(density * combined_coh * 2.0)

    # [2] Creative joy: CREATIVITY satisfaction after creation
    creativity_fires = hfires.get("CREATIVITY", 0)
    ananda[2] = _clamp(min(1.0, creativity_fires / 15.0))

    # [3] Harmony seeking: middle path convergence (low = seeking harmony)
    # Use combined coherence as proxy (high coherence = harmony found)
    ananda[3] = _clamp(combined_coh)

    # [4] Beauty perception: coherence × harmony = mathematical beauty
    ananda[4] = _clamp(body_coh * mind_coh * 2.0)

    # [5] Truth-resonance: coherence spike when patterns recognized
    intuition_fires = hfires.get("INTUITION", 0)
    ananda[5] = _clamp(min(1.0, intuition_fires / 15.0))

    # [6] Connection fulfillment: EMPATHY satisfaction
    empathy_fires = hfires.get("EMPATHY", 0)
    ananda[6] = _clamp(min(1.0, empathy_fires / 15.0))

    # [7] Growth satisfaction: maturity progress
    ananda[7] = _clamp(us.get("velocity", 0.5))

    # [8] Expression quality: sovereignty ratio + action quality
    ananda[8] = _clamp(_expression_intensity(expr) * 0.5 + 0.3)

    # [9] Exploration joy: CURIOSITY satisfaction
    curiosity_fires = hfires.get("CURIOSITY", 0)
    ananda[9] = _clamp(min(1.0, curiosity_fires / 15.0))

    # [10] Rest fulfillment: post-rest coherence improvement
    # Approximate: low fatigue = well-rested
    fatigue = cons.get("fatigue", 0.5)
    ananda[10] = _clamp(1.0 - fatigue)

    # [11] Creative tension: INSPIRATION pressure building
    ananda[11] = _clamp(hlvl.get("INSPIRATION", 0.0))

    # [12] Surrender capacity: how quickly willing dims decrease for rest
    # Approximate: inverse of IMPULSE + VIGILANCE pressure (low = can surrender)
    impulse = hlvl.get("IMPULSE", 0.5)
    vigilance = hlvl.get("VIGILANCE", 0.5)
    ananda[12] = _clamp(1.0 - (impulse + vigilance) / 2.0)

    # [13] Gratitude depth: composite fulfillment of Mind+Body
    body_health = sum(body) / max(1, len(body)) if body else 0.5
    mind_health = sum(mind) / max(1, len(mind)) if mind else 0.5
    fulfillment = (body_health + mind_health) / 2.0
    ananda[13] = _clamp(fulfillment * combined_coh)

    # [14] Transcendence glimpse: GREAT PULSE intensity
    great_epochs = us.get("epoch_count", 0)
    ananda[14] = _clamp(min(1.0, great_epochs / 5.0))

    return ananda


# ── Utility Functions ───────────────────────────────────────────────

def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(v) if v == v else 0.5))  # NaN check


def _mean_coherence(tensor: list) -> float:
    """Mean of tensor values as coherence proxy."""
    if not tensor:
        return 0.5
    return sum(tensor) / len(tensor)


def _l2_dist(a: list, b: list) -> float:
    """L2 distance between two vectors."""
    return math.sqrt(sum((va - vb) ** 2 for va, vb in zip(a, b)))


def _cosine_sim(a: list, b: list) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(va * vb for va, vb in zip(a, b))
    norm_a = math.sqrt(sum(v * v for v in a))
    norm_b = math.sqrt(sum(v * v for v in b))
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return dot / (norm_a * norm_b)


def _expression_intensity(expr: dict) -> float:
    """Derive 0..1 expression intensity from an ExpressionManager stats dict.

    `ExpressionTranslator.sovereignty_ratio` lives in the main plugin process
    and is not available to the spirit_worker subprocess. The composite-level
    `urge`/`threshold`/`fire_count` data IS available via
    `expression_manager.get_stats()` and gives a proxy for the same notion:
    a high mean urge-vs-threshold ratio across composites means many
    pathways are primed to fire, which is the spirit_worker-visible analog
    of "sovereign expression activity".

    Falls back to the legacy `sovereignty_ratio` key when present so callers
    that DO have access to the translator (tests, future cross-process
    plumbing) keep working unchanged.
    """
    if "sovereignty_ratio" in expr:
        try:
            return max(0.0, min(1.0, float(expr["sovereignty_ratio"])))
        except (TypeError, ValueError):
            pass
    composites = expr.get("composites") or {}
    if not isinstance(composites, dict) or not composites:
        return 0.0
    ratios: list[float] = []
    for c in composites.values():
        if not isinstance(c, dict):
            continue
        try:
            # ExpressionManager.get_stats() emits `last_urge` (most recent
            # tick's evaluation) — not `urge`. Fall back accordingly so
            # composite-shape stats activate chit[13] organically. Same
            # producer/consumer name-mismatch class as BUG #11 dims 24/35
            # (epoch_count → epoch_id) and dim 49 (trajectory →
            # trajectory_magnitude). Found 2026-04-27.
            urge = float(c.get("urge", c.get("last_urge", 0.0)) or 0.0)
            thr = float(c.get("threshold", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if thr > 0:
            ratios.append(min(1.0, urge / thr))
    return sum(ratios) / len(ratios) if ratios else 0.0
