"""
Proof of Thought (PoT) — Cognitive Validation Gate for TimeChain.

Biological parallel: Long-Term Potentiation (LTP) requires ATP for synaptic
remodeling, sustained neural activity, and protein synthesis. Not every sensory
input becomes a memory — energy constraints naturally filter significant
experiences from noise.

PoT is the computational analog. Every thought entering the TimeChain must pass
a multi-criteria validation gate that consumes chi (metabolic energy). Only
thoughts worth the metabolic cost get committed to permanent memory.

Validation criteria:
  1. Energy gate — chi must be available to spend
  2. Cognitive readiness — attention, I-confidence, chi-coherence
  3. Neuromodulator profile — DA+ACh+NE = optimal memory formation
  4. Content quality — novelty, significance, coherence
  5. π-curvature adjustment — surprising moments lower threshold
  6. Fork-specific thresholds — episodic easy, meta hard
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger("ProofOfThought")

# ── Fork-Specific Thresholds ───────────────────────────────────────────
# Different memory types have different admission thresholds —
# episodic memories form more easily than procedural skills
DEFAULT_THRESHOLDS = {
    "main": 0.20,        # heartbeat blocks: low bar
    "declarative": 0.15, # facts: moderate bar
    "procedural": 0.25,  # skills: higher bar (must be proven useful)
    "episodic": 0.10,    # experiences: lowest bar (most memories are episodic)
    "meta": 0.25,        # meta-cognition: high bar (I AM events already self-gated)
    "sidechain": 0.12,   # topic extension: low bar
}

# Minimum threshold floor — never let curvature reduce below this
MIN_THRESHOLD = 0.05

# ── Base Chi Costs ─────────────────────────────────────────────────────
BASE_CHI_COSTS = {
    "declarative": 0.005,
    "procedural": 0.010,   # Skills cost more to encode
    "episodic": 0.003,     # Experiences are cheap
    "meta": 0.015,         # Meta-cognition is expensive
    "main": 0.002,         # Heartbeats are minimal
    "sidechain": 0.004,
}


@dataclass
class ProofOfThought:
    """Validation gate for TimeChain block admission.

    Captures the full cognitive state at the moment of thought formation.
    The PoT score determines whether the thought enters the chain.
    """

    # ── Energy Gate ────────────────────────────────────────────────
    chi_cost: float = 0.0       # metabolic energy to consume (computed)
    chi_available: float = 0.0  # current chi level
    metabolic_drain: float = 0.0  # current drain level (high = harder)

    # ── Cognitive Readiness ────────────────────────────────────────
    attention_level: float = 0.0  # MSL homeostatic attention (>0.3)
    i_confidence: float = 0.0    # self-model confidence (>0.1)
    chi_coherence: float = 0.0   # modality alignment (>0.2)

    # ── Neuromodulator Profile ─────────────────────────────────────
    da_level: float = 0.0       # dopamine (reward relevance)
    ach_level: float = 0.0      # acetylcholine (attention)
    ne_level: float = 0.0       # norepinephrine (arousal/salience)
    serotonin: float = 0.0      # serotonin (confidence/mood)
    gaba_level: float = 0.0     # GABA (stability gate)
    endorphin: float = 0.0      # endorphin (positive valence)

    # ── Content Quality ────────────────────────────────────────────
    novelty: float = 0.0        # how new is this thought? (dedup check)
    significance: float = 0.0   # significance score from source system
    coherence: float = 0.0      # coherence with recent chain tip

    # ── Context ────────────────────────────────────────────────────
    source: str = ""            # provenance
    thought_type: str = ""      # declarative|procedural|episodic|meta
    fork_name: str = ""         # target fork name for threshold lookup
    pi_curvature: float = 1.0   # current pi-curvature (1.0 = normal)

    # ── Computed ───────────────────────────────────────────────────
    pot_score: float = 0.0      # composite validation score
    threshold: float = 0.0      # fork-specific threshold (after curvature)
    valid: bool = False         # did it pass?
    nonce: int = 0              # validation nonce
    rejection_reason: str = ""  # why it failed (if invalid)


class PoTValidator:
    """Validates Proof of Thought for TimeChain admission.

    Configurable thresholds from titan_params.toml [timechain.pot_thresholds].
    """

    def __init__(self, thresholds: dict = None, chi_costs: dict = None):
        self._thresholds = thresholds or dict(DEFAULT_THRESHOLDS)
        self._chi_costs = chi_costs or dict(BASE_CHI_COSTS)

        # Stats
        self._total_submitted = 0
        self._total_admitted = 0
        self._total_rejected = 0
        self._total_chi_spent = 0.0

    # ── Public API ─────────────────────────────────────────────────

    def validate(self, pot: ProofOfThought) -> ProofOfThought:
        """Validate a Proof of Thought.

        Computes the PoT score, applies curvature adjustment,
        and determines if the thought passes the threshold.

        Mutates and returns the same ProofOfThought instance.
        """
        self._total_submitted += 1

        # Compute chi cost if not already set
        if pot.chi_cost <= 0:
            pot.chi_cost = self.compute_chi_cost(
                pot.thought_type, pot.significance, 500)

        # Get threshold for this fork
        base_threshold = self._thresholds.get(
            pot.fork_name, self._thresholds.get("sidechain", 0.12))

        # Apply curvature adjustment
        pot.threshold = self._apply_curvature(base_threshold, pot.pi_curvature)

        # Compute score
        pot.pot_score = self._compute_score(pot)

        # Validate
        if pot.pot_score >= pot.threshold:
            pot.valid = True
            pot.nonce = 1
            pot.rejection_reason = ""
            self._total_admitted += 1
            self._total_chi_spent += pot.chi_cost
        else:
            pot.valid = False
            pot.nonce = 0
            self._total_rejected += 1

        return pot

    def compute_chi_cost(self, thought_type: str, significance: float,
                         payload_size: int = 500,
                         fork_depth: int = 0) -> float:
        """Compute chi cost for committing a thought.

        More significant thoughts cost more energy (deeper encoding).
        Deeper forks (sidechains) cost slightly less.
        """
        base = self._chi_costs.get(thought_type, 0.005)

        # Significance multiplier (1x - 3x)
        sig_mult = 1.0 + (significance * 2.0)

        # Payload size factor (larger thoughts cost slightly more)
        size_factor = 1.0 + (payload_size / 10000) * 0.1

        # Fork depth discount (extending existing topic is cheaper)
        depth_discount = max(0.5, 1.0 - fork_depth * 0.02)

        return round(base * sig_mult * size_factor * depth_discount, 6)

    def create_pot(self, chi_available: float, metabolic_drain: float,
                   attention: float, i_confidence: float,
                   chi_coherence: float, neuromods: dict,
                   novelty: float, significance: float,
                   coherence: float, source: str,
                   thought_type: str, fork_name: str,
                   pi_curvature: float = 1.0) -> ProofOfThought:
        """Convenience: create and validate a PoT in one call."""
        # Biological baseline: even at rest, neuromodulators have tonic levels.
        # When source modules don't have neuromod access, use 0.5 baseline.
        _nm_baseline = 0.5 if not neuromods else 0.0
        pot = ProofOfThought(
            chi_available=chi_available,
            metabolic_drain=metabolic_drain,
            attention_level=attention,
            i_confidence=i_confidence,
            chi_coherence=chi_coherence,
            da_level=neuromods.get("DA", _nm_baseline),
            ach_level=neuromods.get("ACh", _nm_baseline),
            ne_level=neuromods.get("NE", _nm_baseline),
            serotonin=neuromods.get("5HT", _nm_baseline),
            gaba_level=neuromods.get("GABA", 0.2 if not neuromods else 0.0),
            endorphin=neuromods.get("endorphin", 0.3 if not neuromods else 0.0),
            novelty=novelty,
            significance=significance,
            coherence=coherence,
            source=source,
            thought_type=thought_type,
            fork_name=fork_name,
            pi_curvature=pi_curvature,
        )
        return self.validate(pot)

    # ── Scoring ────────────────────────────────────────────────────

    def _compute_score(self, pot: ProofOfThought) -> float:
        """Compute Proof of Thought validation score.

        Score > threshold -> thought admitted to chain.
        Mirrors how biological memory formation depends on multiple factors.
        """
        # Energy gate: must have chi to spend
        if pot.chi_available < pot.chi_cost:
            pot.rejection_reason = "insufficient_chi"
            return 0.0

        # Cognitive readiness (3 gates, all must be minimally met)
        # Using safe denominators — at very low cognitive states, no memory forms
        attn_gate = min(1.0, pot.attention_level / 0.3) if pot.attention_level > 0 else 0.0
        conf_gate = min(1.0, pot.i_confidence / 0.1) if pot.i_confidence > 0 else 0.0
        cohr_gate = min(1.0, pot.chi_coherence / 0.2) if pot.chi_coherence > 0 else 0.0

        readiness = min(attn_gate, conf_gate, cohr_gate)

        if readiness < 0.1:
            pot.rejection_reason = "low_cognitive_readiness"
            return 0.0

        # Neuromodulator formation factor
        # High DA + high ACh + moderate NE = optimal memory formation
        neuromod_factor = (
            0.30 * min(1.0, pot.da_level)     +  # reward relevance
            0.25 * min(1.0, pot.ach_level)     +  # attention
            0.20 * min(1.0, pot.ne_level)      +  # salience
            0.10 * min(1.0, pot.serotonin)     +  # confidence
            0.10 * (1.0 - pot.gaba_level)      +  # low inhibition = more formation
            0.05 * min(1.0, pot.endorphin)        # positive valence
        )

        # Content quality
        content_factor = (
            0.40 * pot.novelty       +   # novel thoughts more valuable
            0.35 * pot.significance  +   # significant events remembered
            0.25 * pot.coherence         # coherent with existing knowledge
        )

        # Metabolic penalty (tired = harder to form memories)
        drain_penalty = max(0.1, 1.0 - (pot.metabolic_drain * 0.5))

        # Composite
        score = readiness * neuromod_factor * content_factor * drain_penalty
        return round(score, 6)

    def _apply_curvature(self, base_threshold: float,
                         curvature: float) -> float:
        """Apply pi-curvature to PoT threshold.

        High curvature (surprising) -> LOWER threshold (easier to commit)
        because surprising events are biologically more memorable.

        Low curvature (routine) -> HIGHER threshold (harder to commit)
        because routine events are naturally forgotten.
        """
        # Curvature typically 0.5 - 3.0, normalized to modifier
        curvature_modifier = 1.0 - min(0.4, (curvature - 1.0) * 0.2)
        # At curvature 1.0 (normal): modifier = 1.0
        # At curvature 3.0 (very surprising): modifier = 0.6 (40% easier)
        # At curvature 0.5 (routine): modifier = 1.1 (10% harder)

        adjusted = base_threshold * curvature_modifier
        return max(MIN_THRESHOLD, round(adjusted, 6))

    # ── Stats ──────────────────────────────────────────────────────

    @property
    def acceptance_rate(self) -> float:
        if self._total_submitted == 0:
            return 0.0
        return self._total_admitted / self._total_submitted

    @property
    def stats(self) -> dict:
        return {
            "submitted": self._total_submitted,
            "admitted": self._total_admitted,
            "rejected": self._total_rejected,
            "acceptance_rate": round(self.acceptance_rate, 3),
            "total_chi_spent": round(self._total_chi_spent, 6),
        }

    def get_neuromod_dict(self, neuromods: dict) -> dict:
        """Extract standard neuromod dict from various input formats.

        Accepts both flat dict {"DA": 0.5, ...} and nested
        {"levels": {"DA": 0.5, ...}} formats.
        """
        if "levels" in neuromods:
            return neuromods["levels"]
        return {
            "DA": float(neuromods.get("DA", neuromods.get("dopamine", 0))),
            "ACh": float(neuromods.get("ACh", neuromods.get("acetylcholine", 0))),
            "NE": float(neuromods.get("NE", neuromods.get("norepinephrine", 0))),
            "5HT": float(neuromods.get("5HT", neuromods.get("serotonin", 0))),
            "GABA": float(neuromods.get("GABA", neuromods.get("gaba", 0))),
            "endorphin": float(neuromods.get("endorphin", 0)),
        }
