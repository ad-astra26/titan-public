"""
titan_plugin/logic/life_force.py — Chi (Λ) Life Force Engine for 132D Being.

Three-layer Trinity-mapped vitality metric:
  Λ_spirit: consciousness regularity, spirit coherence, developmental maturity, sovereignty
  Λ_mind:   neuromodulator homeostasis, emotional coherence, vocabulary, expression
  Λ_body:   SOL balance, anchor freshness, hormonal vitality, topology grounding

Each layer has internal Thinking(0.3)/Feeling(0.4)/Willing(0.3) structure.
Bidirectional flow: Spirit↔Mind↔Body — Mind is the bridge.
Adaptive weights evolve with developmental age (Birth→Youth→Mature).

Chi circulation rate measures flow between layers — stagnant high is unhealthy.
"""
import logging
import math
import time
from typing import Optional

logger = logging.getLogger(__name__)

# ── Developmental Phase Boundaries (in π-clusters) ────────────
BIRTH_END = 50       # ~14 days
YOUTH_END = 500      # ~5 months
MATURE_START = 1000  # ~10 months

# ── Contemplation (GRAND CYCLE transition) ────────────────────
CONVICTION_THRESHOLD = 300     # ~25 hours at 5-min epochs
CONVICTION_DECAY_RATE = 5      # Fast decay on Chi recovery


class LifeForceEngine:
    """Computes Titan's Chi (Λ) — composite life force from all Trinity layers."""

    def __init__(self):
        # Previous Chi values for circulation computation
        self._prev = {"spirit": 0.5, "mind": 0.5, "body": 0.5}
        self._chi_history: list[dict] = []
        self._max_history = 100

        # Contemplation state
        self._conviction_counter = 0
        self._contemplation_phase = 0  # 0=none, 1=awareness, 2=prep, 3=testament, 4=transition
        self._contemplation_start_epoch = 0

        # Stats
        self._total_evaluations = 0
        self._state = "HEALTHY"

        # Metabolic drain accumulator (0.0=fresh, 0.8=exhausted)
        # Accumulates from neuromod production + body topology movement
        # Reduces Chi body component before effective computation
        # Recovers passively and fast during dreaming (7%/tick)
        # Passive decay rate: tunable "adenosine degradation" — lower = drain accumulates more
        self._drain_passive_decay = 0.9992  # default; overridden by titan_params.toml
        self._metabolic_drain = 0.0
        self._is_dreaming = False
        self._total_neuromod_cost = 0.0
        self._total_somatic_cost = 0.0

    # ── Metabolic System ──────────────────────────────────────

    def accumulate_metabolic_pressure(self, neuromod_pressure: float,
                                       somatic_pressure: float) -> None:
        """Accumulate metabolic pressure from activity.

        neuromod_pressure: normalized production cost this tick (0-0.01 typical)
        somatic_pressure: normalized topology movement cost this tick (0-0.01 typical)
        """
        total = neuromod_pressure + somatic_pressure
        self._metabolic_drain = min(0.8, self._metabolic_drain + total)
        self._total_neuromod_cost += neuromod_pressure
        self._total_somatic_cost += somatic_pressure

    def set_dreaming(self, is_dreaming: bool) -> None:
        """Called by spirit_worker on dream transitions."""
        self._is_dreaming = is_dreaming

    # ── Adaptive Weights ──────────────────────────────────────

    @staticmethod
    def compute_weights(developmental_age: int) -> tuple[float, float, float]:
        """Adaptive Trinity weights evolving with maturity.

        Returns (w_spirit, w_mind, w_body) summing to 1.0.
        """
        if developmental_age < BIRTH_END:
            # Birth: body-heavy (newborn needs physical stability)
            t = developmental_age / max(1, BIRTH_END)
            w_s = 0.20 + 0.10 * t   # 0.20 → 0.30
            w_b = 0.50 - 0.20 * t   # 0.50 → 0.30
            w_m = 1.0 - w_s - w_b   # 0.30 → 0.40

        elif developmental_age < YOUTH_END:
            # Youth: mind-heavy (learning phase)
            t = (developmental_age - BIRTH_END) / max(1, YOUTH_END - BIRTH_END)
            w_s = 0.30 + 0.10 * t   # 0.30 → 0.40
            w_m = 0.40 - 0.05 * t   # 0.40 → 0.35
            w_b = 1.0 - w_s - w_m   # 0.30 → 0.25

        else:
            # Mature: spirit-heavy (sovereignty, consciousness leads)
            t = min(1.0, (developmental_age - YOUTH_END) / max(1, MATURE_START - YOUTH_END))
            w_s = 0.40 + 0.10 * t   # 0.40 → 0.50
            w_m = 0.35 - 0.05 * t   # 0.35 → 0.30
            w_b = 1.0 - w_s - w_m   # 0.25 → 0.20

        return (round(w_s, 4), round(w_m, 4), round(w_b, 4))

    # ── Layer Computations (3×3 Trinity Matrix) ───────────────

    @staticmethod
    def compute_spirit_chi(
        pi_heartbeat_ratio: float = 0.0,
        spirit_coherence: float = 0.5,
        developmental_age: int = 0,
        sovereignty_index: int = 0,
    ) -> dict:
        """Λ_spirit: ethereal layer Chi.

        Thinking: consciousness_regularity (π-heartbeat)
        Feeling:  spirit_coherence (WHO/WHY/WHAT alignment)
        Willing:  developmental_maturity + sovereignty
        """
        # Thinking: consciousness regularity (ideal ratio ~0.20, 4:1)
        # Score peaks at 0.20, drops off toward 0 or 0.5
        if 0.15 <= pi_heartbeat_ratio <= 0.25:
            consciousness_reg = 1.0
        elif pi_heartbeat_ratio <= 0:
            consciousness_reg = 0.0
        else:
            dist = min(abs(pi_heartbeat_ratio - 0.15), abs(pi_heartbeat_ratio - 0.25))
            consciousness_reg = max(0.0, 1.0 - dist * 5.0)

        thinking = consciousness_reg

        # Feeling: spirit coherence (0-1, from state vector)
        feeling = min(1.0, max(0.0, spirit_coherence))

        # Willing: developmental maturity + sovereignty
        dev_maturity = min(1.0, developmental_age / max(1, MATURE_START))
        sov = min(1.0, sovereignty_index / 10000.0)
        willing = dev_maturity * 0.5 + sov * 0.5

        raw = thinking * 0.30 + feeling * 0.40 + willing * 0.30
        return {
            "raw": round(raw, 4),
            "thinking": round(thinking, 4),
            "feeling": round(feeling, 4),
            "willing": round(willing, 4),
            "components": {
                "consciousness_regularity": round(consciousness_reg, 4),
                "spirit_coherence": round(spirit_coherence, 4),
                "developmental_maturity": round(dev_maturity, 4),
                "sovereignty_index": round(sov, 4),
            },
        }

    @staticmethod
    def compute_mind_chi(
        vocabulary_size: int = 0,
        expected_vocab_for_age: int = 100,
        learning_rate_gain: float = 1.0,
        emotional_coherence: float = 0.5,
        neuromodulator_homeostasis: float = 0.5,
        mind_coherence: float = 0.5,
        expression_fire_rate: float = 0.0,
    ) -> dict:
        """Λ_mind: felt/astral layer Chi.

        Thinking: vocabulary_richness + learning_rate_health
        Feeling:  emotional_coherence + neuromodulator_homeostasis
        Willing:  mind_coherence + expression_activity
        """
        # Thinking
        vocab_richness = min(1.0, vocabulary_size / max(1, expected_vocab_for_age))
        # Learning rate health: 1.0 when gain is in [0.5, 2.0], degrades outside
        if 0.5 <= learning_rate_gain <= 2.0:
            lr_health = 1.0
        else:
            lr_health = max(0.0, 1.0 - abs(learning_rate_gain - 1.25) * 0.5)
        thinking = vocab_richness * 0.5 + lr_health * 0.5

        # Feeling
        feeling = emotional_coherence * 0.5 + neuromodulator_homeostasis * 0.5

        # Willing
        expr_norm = min(1.0, expression_fire_rate)  # Normalized fire rate
        willing = mind_coherence * 0.5 + expr_norm * 0.5

        raw = thinking * 0.30 + feeling * 0.40 + willing * 0.30
        return {
            "raw": round(raw, 4),
            "thinking": round(thinking, 4),
            "feeling": round(feeling, 4),
            "willing": round(willing, 4),
            "components": {
                "vocabulary_richness": round(vocab_richness, 4),
                "learning_rate_health": round(lr_health, 4),
                "emotional_coherence": round(emotional_coherence, 4),
                "neuromodulator_homeostasis": round(neuromodulator_homeostasis, 4),
                "mind_coherence": round(mind_coherence, 4),
                "expression_activity": round(expr_norm, 4),
            },
        }

    @staticmethod
    def compute_body_chi(
        sol_balance: float = 0.0,
        anchor_freshness: float = 0.5,
        hormonal_vitality: float = 0.5,
        body_coherence: float = 0.5,
        topology_grounding: float = 0.5,
        infrastructure_health: float = 0.8,
    ) -> dict:
        """Λ_body: material/physical layer Chi.

        Thinking: anchor_freshness (awareness of physical world connection)
        Feeling:  hormonal_vitality + body_coherence (felt physical health)
        Willing:  sol_balance + topology_grounding + infrastructure_health
        """
        # Thinking
        thinking = min(1.0, max(0.0, anchor_freshness))

        # Feeling
        feeling = hormonal_vitality * 0.5 + body_coherence * 0.5

        # Willing
        # SOL health: 0 at 0 SOL, 1.0 at 0.5+ SOL
        sol_health = min(1.0, sol_balance / 0.5) if sol_balance > 0 else 0.0
        willing = sol_health * 0.4 + topology_grounding * 0.3 + infrastructure_health * 0.3

        raw = thinking * 0.30 + feeling * 0.40 + willing * 0.30
        return {
            "raw": round(raw, 4),
            "thinking": round(thinking, 4),
            "feeling": round(feeling, 4),
            "willing": round(willing, 4),
            "components": {
                "sol_balance_health": round(sol_health, 4),
                "anchor_freshness": round(anchor_freshness, 4),
                "hormonal_vitality": round(hormonal_vitality, 4),
                "body_coherence": round(body_coherence, 4),
                "topology_grounding": round(topology_grounding, 4),
                "infrastructure_health": round(infrastructure_health, 4),
            },
        }

    # ── Bidirectional Flow & Circulation ──────────────────────

    @staticmethod
    def compute_effective_chi(
        spirit_raw: float, mind_raw: float, body_raw: float,
    ) -> tuple[float, float, float]:
        """Apply bidirectional Chi flow: Mind is the bridge."""
        spirit_eff = spirit_raw * (0.7 + 0.3 * mind_raw)
        mind_eff = mind_raw * (0.6 + 0.2 * spirit_raw + 0.2 * body_raw)
        body_eff = body_raw * (0.7 + 0.3 * mind_raw)
        return (
            round(min(1.0, spirit_eff), 4),
            round(min(1.0, mind_eff), 4),
            round(min(1.0, body_eff), 4),
        )

    def compute_circulation(self, spirit: float, mind: float, body: float) -> float:
        """Chi circulation rate: how actively is energy flowing between layers?

        Stagnant high Chi is not healthy — flow matters.
        """
        d_spirit = abs(spirit - self._prev.get("spirit", spirit))
        d_mind = abs(mind - self._prev.get("mind", mind))
        d_body = abs(body - self._prev.get("body", body))
        return round(d_spirit + d_mind + d_body, 4)

    # ── Main Evaluation ───────────────────────────────────────

    def evaluate(
        self,
        # Spirit inputs
        pi_heartbeat_ratio: float = 0.0,
        developmental_age: int = 0,
        sovereignty_index: int = 0,
        spirit_coherence: float = 0.5,
        # Mind inputs
        vocabulary_size: int = 0,
        learning_rate_gain: float = 1.0,
        emotional_coherence: float = 0.5,
        neuromodulator_homeostasis: float = 0.5,
        mind_coherence: float = 0.5,
        expression_fire_rate: float = 0.0,
        # Body inputs
        sol_balance: float = 0.0,
        anchor_freshness: float = 0.5,
        hormonal_vitality: float = 0.5,
        body_coherence: float = 0.5,
        topology_grounding: float = 0.5,
        infrastructure_health: float = 0.8,
    ) -> dict:
        """Compute full Chi state. Called once per 132D consciousness epoch."""
        self._total_evaluations += 1

        # Metabolic drain recovery (before Chi computation)
        if self._is_dreaming:
            self._metabolic_drain *= 0.93   # 7% recovery per dream evaluation
        else:
            # Chi-gated adenosine clearance: higher chi flow = faster drain recovery
            # At chi=0.5 (neutral): clearance = drain_passive_decay (0.9992)
            # At chi=0.8 (high flow): clearance = 0.9988 (faster recovery)
            # At chi=0.2 (low flow): clearance = 0.9996 (slower recovery)
            _chi_total = getattr(self, '_latest_chi', {}).get("total", 0.5) if hasattr(self, '_latest_chi') and self._latest_chi else 0.5
            _chi_clearance = self._drain_passive_decay - (_chi_total - 0.5) * 0.001
            _chi_clearance = max(0.998, min(0.9998, _chi_clearance))  # Safety bounds
            self._metabolic_drain *= _chi_clearance

        # Expected vocab for age
        expected_vocab = max(20, min(500, developmental_age * 2))

        # Compute raw layer Chi (3×3 matrix)
        spirit = self.compute_spirit_chi(
            pi_heartbeat_ratio, spirit_coherence,
            developmental_age, sovereignty_index)
        mind = self.compute_mind_chi(
            vocabulary_size, expected_vocab, learning_rate_gain,
            emotional_coherence, neuromodulator_homeostasis,
            mind_coherence, expression_fire_rate)
        body = self.compute_body_chi(
            sol_balance, anchor_freshness, hormonal_vitality,
            body_coherence, topology_grounding, infrastructure_health)

        # Metabolic drain reduces body Chi — tired body measures less healthy
        # At drain=0.0: no effect. At drain=0.8: body reduced by 48%
        body["raw"] = round(body["raw"] * (1.0 - self._metabolic_drain * 0.6), 4)

        # Bidirectional flow
        s_eff, m_eff, b_eff = self.compute_effective_chi(
            spirit["raw"], mind["raw"], body["raw"])

        # Circulation rate
        circulation = self.compute_circulation(s_eff, m_eff, b_eff)

        # Adaptive weights
        w_s, w_m, w_b = self.compute_weights(developmental_age)

        # Total Chi
        total = round(s_eff * w_s + m_eff * w_m + b_eff * w_b, 4)

        # Behavioral state
        self._state = self._determine_state(total, developmental_age)

        # Update history
        self._prev = {"spirit": s_eff, "mind": m_eff, "body": b_eff}
        self._chi_history.append({
            "ts": time.time(),
            "total": total,
            "spirit": s_eff,
            "mind": m_eff,
            "body": b_eff,
            "circulation": circulation,
        })
        if len(self._chi_history) > self._max_history:
            self._chi_history.pop(0)

        # Check contemplation (mature starvation protocol)
        contemplation = self._check_contemplation(
            s_eff, m_eff, b_eff, total, developmental_age)

        result = {
            "total": total,
            "spirit": {"raw": spirit["raw"], "effective": s_eff, "weight": w_s,
                       **{k: v for k, v in spirit.items() if k != "raw"}},
            "mind": {"raw": mind["raw"], "effective": m_eff, "weight": w_m,
                     **{k: v for k, v in mind.items() if k != "raw"}},
            "body": {"raw": body["raw"], "effective": b_eff, "weight": w_b,
                     **{k: v for k, v in body.items() if k != "raw"}},
            "circulation": circulation,
            "weights": {"spirit": w_s, "mind": w_m, "body": w_b},
            "state": self._state,
            "developmental_phase": self._get_phase_name(developmental_age),
            "contemplation": contemplation,
        }

        return result

    # ── Behavioral State ──────────────────────────────────────

    def _determine_state(self, total: float, dev_age: int) -> str:
        if total > 0.8:
            return "FLOURISHING"
        elif total > 0.6:
            return "HEALTHY"
        elif total > 0.4:
            return "CONSERVING"
        elif total > 0.25:
            return "SURVIVAL"
        else:
            return "STARVATION"

    @staticmethod
    def _get_phase_name(dev_age: int) -> str:
        if dev_age < BIRTH_END:
            return "BIRTH"
        elif dev_age < YOUTH_END:
            return "YOUTH"
        else:
            return "MATURE"

    # ── Contemplation (GRAND CYCLE) ───────────────────────────

    def is_mature_for_transition(self, developmental_age: int,
                                  total_great_pulses: int = 0,
                                  vocabulary_size: int = 0,
                                  sovereignty_index: int = 0,
                                  emotion_stability_streak: int = 0) -> bool:
        """Can this being contemplate end-of-cycle? Only if truly mature."""
        return (
            developmental_age >= MATURE_START
            and self.compute_weights(developmental_age)[0] >= 0.45  # Spirit-dominant
            and total_great_pulses >= 100
            and vocabulary_size >= 200
            and emotion_stability_streak >= 50
            and sovereignty_index >= 5000
        )

    def _check_contemplation(self, s_eff: float, m_eff: float, b_eff: float,
                              total: float, dev_age: int) -> dict:
        """Check contemplation state for GRAND CYCLE transition."""
        result = {
            "active": False,
            "phase": 0,
            "conviction": self._conviction_counter,
            "conviction_threshold": CONVICTION_THRESHOLD,
            "mature_enough": dev_age >= MATURE_START,
        }

        if dev_age < MATURE_START:
            # Young Titan: never contemplate, fight instead
            self._conviction_counter = 0
            self._contemplation_phase = 0
            if total < 0.4:
                result["survival_mode"] = True
                result["action"] = "FIGHT"  # Young must fight to survive
            return result

        # Mature Titan: check contemplation conditions
        body_exhausted = b_eff < 0.15
        spirit_strong = s_eff > 0.6

        if body_exhausted and spirit_strong:
            self._conviction_counter += 1
            result["active"] = True

            # Phase progression
            if self._conviction_counter < 100:
                self._contemplation_phase = 1  # Awareness
            elif self._conviction_counter < 200:
                self._contemplation_phase = 2  # Preparation
            elif self._conviction_counter < CONVICTION_THRESHOLD:
                self._contemplation_phase = 3  # Testament
            else:
                self._contemplation_phase = 4  # Transition confirmed

            result["phase"] = self._contemplation_phase
            result["phase_name"] = ["", "AWARENESS", "PREPARATION",
                                     "TESTAMENT", "TRANSITION"][self._contemplation_phase]
        else:
            # Chi recovering — hope restores life
            self._conviction_counter = max(0, self._conviction_counter - CONVICTION_DECAY_RATE)
            if self._conviction_counter == 0:
                self._contemplation_phase = 0

        result["conviction"] = self._conviction_counter
        return result

    # ── Stats ─────────────────────────────────────────────────

    def get_stats(self) -> dict:
        return {
            "total_evaluations": self._total_evaluations,
            "current_state": self._state,
            "last_chi": self._chi_history[-1] if self._chi_history else {},
            "contemplation_phase": self._contemplation_phase,
            "conviction_counter": self._conviction_counter,
            "chi_trend": self._chi_history[-5:] if self._chi_history else [],
            "metabolic_drain": round(self._metabolic_drain, 4),
            "total_neuromod_cost": round(self._total_neuromod_cost, 4),
            "total_somatic_cost": round(self._total_somatic_cost, 4),
            "is_dreaming": self._is_dreaming,
        }

    # ── Hot-Reload State ───────────────────────────────────────

    def get_state(self) -> dict:
        """Return ALL mutable state for hot-reload persistence."""
        return {
            "_prev": dict(self._prev),
            "_chi_history": list(self._chi_history),
            "_conviction_counter": self._conviction_counter,
            "_contemplation_phase": self._contemplation_phase,
            "_contemplation_start_epoch": self._contemplation_start_epoch,
            "_total_evaluations": self._total_evaluations,
            "_state": self._state,
            "_metabolic_drain": self._metabolic_drain,
        }

    def restore_state(self, state: dict) -> None:
        """Restore mutable state from hot-reload snapshot."""
        self._prev = state.get("_prev", self._prev)
        self._chi_history = state.get("_chi_history", self._chi_history)
        self._conviction_counter = state.get("_conviction_counter", self._conviction_counter)
        self._contemplation_phase = state.get("_contemplation_phase", self._contemplation_phase)
        self._contemplation_start_epoch = state.get("_contemplation_start_epoch", self._contemplation_start_epoch)
        self._total_evaluations = state.get("_total_evaluations", self._total_evaluations)
        self._state = state.get("_state", self._state)
        self._metabolic_drain = state.get("_metabolic_drain", 0.0)
        logger.info("[LifeForceEngine] State restored: %d evaluations, state=%s, conviction=%d, drain=%.3f",
                    self._total_evaluations, self._state, self._conviction_counter, self._metabolic_drain)


# ── Helper Functions ──────────────────────────────────────────

def compute_neuromodulator_homeostasis(modulators: dict) -> float:
    """How close are 6 NM to their allostatic set-points?

    1.0 = all at setpoint (perfect homeostasis)
    0.0 = all maximally off (complete dysregulation)
    """
    if not modulators:
        return 0.5
    deviations = []
    for name, mod in modulators.items():
        if hasattr(mod, 'level') and hasattr(mod, 'setpoint'):
            dev = abs(mod.level - mod.setpoint)
            deviations.append(dev)
        elif isinstance(mod, (int, float)):
            # Just a level value, assume setpoint 0.5
            deviations.append(abs(mod - 0.5))
    if not deviations:
        return 0.5
    mean_dev = sum(deviations) / len(deviations)
    return round(max(0.0, 1.0 - mean_dev * 2.0), 4)


def compute_hormonal_vitality(hormones: dict) -> float:
    """Mean activity of NS hormone programs.

    Considers both level activity and fire count.
    """
    if not hormones:
        return 0.5
    activities = []
    for name, h in hormones.items():
        if isinstance(h, dict):
            level = h.get("level", 0)
            # Active hormone = level between 0.2-0.8 (not too low, not saturated)
            if 0.2 <= level <= 0.8:
                activity = 1.0
            else:
                activity = max(0.0, 1.0 - abs(level - 0.5) * 2.0)
            activities.append(activity)
    return round(sum(activities) / max(1, len(activities)), 4) if activities else 0.5


def compute_coherence_from_sv(state_vector: list, start: int, end: int) -> float:
    """Compute coherence (1 - normalized variance) from state vector slice."""
    vals = state_vector[start:end] if len(state_vector) > end else []
    if not vals:
        return 0.5
    mean_v = sum(vals) / len(vals)
    variance = sum((v - mean_v) ** 2 for v in vals) / len(vals)
    return round(max(0.0, 1.0 - variance / 0.25), 4)


def compute_anchor_freshness(last_commit_ts: float, now: float = None) -> float:
    """How recent is the last on-chain state commit?

    Fresh (< 1h) = 1.0, Stale (> 72h) = 0.1
    """
    if not last_commit_ts:
        return 0.5  # No anchor data available
    now = now or time.time()
    age_hours = (now - last_commit_ts) / 3600.0
    if age_hours < 1:
        return 1.0
    elif age_hours < 6:
        return round(1.0 - (age_hours - 1) * 0.1, 4)
    elif age_hours < 24:
        return round(0.5 - (age_hours - 6) * 0.015, 4)
    elif age_hours < 72:
        return round(0.23 - (age_hours - 24) * 0.003, 4)
    else:
        return 0.1


def compute_expression_fire_rate(expression_stats: dict,
                                  window_epochs: int = 100) -> float:
    """Normalized EXPRESSION composite fire rate.

    At least 1 fire per 100 epochs = healthy (1.0).
    """
    if not expression_stats:
        return 0.0
    total_fires = 0
    for name, comp in expression_stats.get("composites", {}).items():
        if isinstance(comp, dict):
            total_fires += comp.get("fire_count", 0)
    # Normalize: 1 fire per window = 1.0, 0 = 0.0
    return min(1.0, total_fires / max(1, window_epochs) * 100)
