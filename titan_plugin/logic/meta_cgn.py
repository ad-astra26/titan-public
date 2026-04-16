"""META-CGN — Meta-Reasoning as CGN Consumer (Layer β Atomic Grounding).

Adds cognitive primitives (FORMULATE, RECALL, HYPOTHESIZE, DELEGATE, SYNTHESIZE,
EVALUATE, BREAK, SPIRIT_SELF, INTROSPECT) as CGN concepts with grounded V(s).
Registers as the 7th CGN consumer via the existing CGNConsumerClient pattern.

Phase 1 scope (shadow mode): observe chain outcomes, accumulate primitive
grounding, log what META-CGN would recommend — but do NOT influence template
selection yet. Graduation to active influence gated on rFP §5.3 criteria.

See: titan-docs/rFP_meta_cgn_v2.md
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

try:
    from scipy.stats import beta as _scipy_beta  # type: ignore
    _HAS_SCIPY_BETA = True
except Exception:  # pragma: no cover — fall back to analytical approximations
    _scipy_beta = None
    _HAS_SCIPY_BETA = False

logger = logging.getLogger("titan.meta_cgn")

# ── P6: Bayesian Beta + composition tunables (rFP §P6 lock-in 2026-04-12) ──
# Prior floor — never collapse below uniform prior
BETA_PARAM_FLOOR = 1.0
# n_eff cap for converted-bootstrap migration from v2 files
MIGRATION_N_EFF_CAP = 200
# Composition tunables — loaded from titan_params.toml [meta_cgn] if present
COMPOSITION_DEFAULTS = {
    "kappa_explore": 0.15,     # F — anti-monoculture bonus weight
    "kappa_ci": 0.5,           # D4 — UCB pessimism/optimism width
    "n_anchor": 100,           # D4 — optimistic→pessimistic switchpoint (samples)
    "decay_gamma": 0.98,       # D2 — decay multiplier applied per cadence
    "decay_cadence_chains": 500,  # D2 — how often to decay
    "decay_skip_n_min": 20,    # D2 — skip primitives below this n
    "domain_obs_threshold": 10,  # I3 — domain needs ≥N obs before using
    "impasse_weight_mul": 2.0,   # impasse α-boost port — 2× observation weight
    "ci_quantile_lo": 0.05,
    "ci_quantile_hi": 0.95,
}


def _beta_mean(a: float, b: float) -> float:
    """Posterior mean = α/(α+β), safe against collapse."""
    tot = max(BETA_PARAM_FLOOR * 2, a + b)
    return float(a / tot)


def _beta_ci_width(a: float, b: float,
                   q_lo: float = 0.05, q_hi: float = 0.95) -> tuple[float, float, float]:
    """Return (lo, hi, width) for Beta(α,β). Uses scipy when available,
    otherwise a normal approximation on the logit scale.
    """
    a = max(BETA_PARAM_FLOOR, float(a))
    b = max(BETA_PARAM_FLOOR, float(b))
    if _HAS_SCIPY_BETA:
        try:
            lo = float(_scipy_beta.ppf(q_lo, a, b))
            hi = float(_scipy_beta.ppf(q_hi, a, b))
        except Exception:
            lo, hi = _normal_approx_beta_ci(a, b, q_lo, q_hi)
    else:
        lo, hi = _normal_approx_beta_ci(a, b, q_lo, q_hi)
    lo = max(0.0, min(1.0, lo))
    hi = max(0.0, min(1.0, hi))
    if hi < lo:
        hi = lo
    return lo, hi, hi - lo


def _normal_approx_beta_ci(a: float, b: float,
                           q_lo: float, q_hi: float) -> tuple[float, float]:
    """Normal approximation fallback — mean ± z·sd, clipped to [0,1].
    z(0.05)=−1.645, z(0.95)=+1.645 (symmetric 90% interval).
    """
    mean = a / max(1e-9, (a + b))
    var = (a * b) / (max(1e-9, (a + b) ** 2) * max(1e-9, (a + b + 1)))
    sd = math.sqrt(max(0.0, var))
    # Use ~1.645 for 90% symmetric if quantiles hint that; else scale by z-ish.
    z_lo = -1.645 if q_lo <= 0.05 + 1e-6 else -1.0
    z_hi = 1.645 if q_hi >= 0.95 - 1e-6 else 1.0
    return mean + z_lo * sd, mean + z_hi * sd


def _posterior_confidence(a: float, b: float, ref_n: float = 500.0) -> float:
    """Saturating confidence in [0,1] from (α−1)+(β−1) effective evidence count.

    Matches pre-P6 semantics (n_samples/500) but uses Beta evidence count.
    """
    n_eff = max(0.0, (a - BETA_PARAM_FLOOR) + (b - BETA_PARAM_FLOOR))
    return float(min(1.0, n_eff / max(1.0, ref_n)))


def _gini(values: list) -> float:
    """Gini coefficient over non-negative values. 0 = uniform, 1 = monoculture."""
    xs = [max(0.0, float(v)) for v in values]
    n = len(xs)
    if n == 0:
        return 0.0
    s = sum(xs)
    if s <= 0:
        return 0.0
    xs_sorted = sorted(xs)
    cum = 0.0
    for i, v in enumerate(xs_sorted, start=1):
        cum += i * v
    return float((2 * cum) / (n * s) - (n + 1) / n)


# ── P8: SOAR-via-CGN full protocol tunables ──────────────────────────

# D8.3 B-hybrid: source affinity table — impasse signal → per-consumer affinity
# Values in [0, 1]. Hand-crafted priors; Maker-tunable without retraining.
SOURCE_AFFINITY: dict[str, dict[str, float]] = {
    "v_flatline": {
        "knowledge": 1.0, "language": 0.5, "social": 0.3,
        "self_model": 0.4, "coding": 0.2, "meta_cgn_grounding": 0.5,
    },
    "conf_flatline": {
        "knowledge": 0.9, "self_model": 0.7, "language": 0.4,
        "social": 0.3, "coding": 0.3, "meta_cgn_grounding": 0.6,
    },
    "haov_stagnant": {
        "self_model": 1.0, "knowledge": 0.7, "language": 0.5,
        "social": 0.3, "coding": 0.4, "meta_cgn_grounding": 0.8,
    },
    "graduation_flatline": {
        "knowledge": 0.9, "coding": 0.6, "self_model": 0.6,
        "language": 0.4, "social": 0.2, "meta_cgn_grounding": 0.7,
    },
}

# D8.2 aggregation window: finalize pending requests after this many seconds
P8_RESPONSE_WINDOW_SECONDS = 2.0

# I-P8.1 dedup: each impasse cycle emits ONE knowledge request


def _extract_keywords(text: str) -> set:
    """Cheap keyword extraction for B-hybrid relevance — lowercase tokens
    with length ≥ 3, stripped of common stopwords. Used for keyword overlap."""
    if not text:
        return set()
    stop = {"the", "and", "for", "with", "that", "this", "from", "into",
            "are", "was", "were", "have", "has", "had", "its", "their",
            "these", "those", "when", "what", "will", "but", "not", "all"}
    tokens = set()
    for tok in str(text).lower().replace("(", " ").replace(")", " ")\
            .replace(",", " ").replace(".", " ").replace("→", " ")\
            .replace("-", " ").replace("_", " ").split():
        t = tok.strip("'\"")
        if len(t) >= 3 and t not in stop:
            tokens.add(t)
    return tokens


# ── P10: Cross-consumer signal flow (Layer 1) ─────────────────────────

# ──────────────────────────────────────────────────────────────────
# EdgeDetector — helper for event-shaped signal emission (rFP v3 § 7)
# ──────────────────────────────────────────────────────────────────
class EdgeDetector:
    """Turns continuous value streams into discrete events.

    Built 2026-04-14 post-incident. Enforces the architectural invariant:
    signals emit ONLY on threshold crossings or first-time observations,
    never per-call. Prevents the 2026-04-14 Phase 2 mistake (emit-every-
    time producers flooded the bus 13x).

    Two modes:
      - observe(key, value, threshold): returns True on first time value
        crosses threshold for that key. Subsequent values stay within or
        above threshold → False. Value drops below then re-crosses →
        True again (per-crossing, not per-key).
      - observe_first_time(key): returns True the first time a key is
        seen, False thereafter. For unique-event detection (e.g. first
        crystallization of a unique wisdom signature).
      - observe_new_max(key, value): returns True if value > previous
        max for this key. For 'new personal best' events.

    State is per-instance; create one EdgeDetector per producer context.
    """

    def __init__(self) -> None:
        # threshold crossing state: key → {'above': bool}
        self._crossed: dict[str, bool] = {}
        # first-time seen set
        self._seen: set[str] = set()
        # new-max tracking: key → float
        self._max: dict[str, float] = {}

    def observe(self, key: str, value: float, threshold: float) -> bool:
        """Fire True on first crossing above threshold per key.
        Falls below → next crossing fires again."""
        above = value >= threshold
        was_above = self._crossed.get(key, False)
        self._crossed[key] = above
        return above and not was_above

    def observe_first_time(self, key: str) -> bool:
        """Fire True exactly once per unique key."""
        if key in self._seen:
            return False
        self._seen.add(key)
        return True

    def observe_new_max(self, key: str, value: float) -> bool:
        """Fire True if value exceeds previous maximum for this key."""
        prev = self._max.get(key, float("-inf"))
        if value > prev:
            self._max[key] = value
            return True
        return False

    def reset(self, key: Optional[str] = None) -> None:
        """Reset state. If key given, only that key; else all."""
        if key is None:
            self._crossed.clear()
            self._seen.clear()
            self._max.clear()
        else:
            self._crossed.pop(key, None)
            self._seen.discard(key)
            self._max.pop(key, None)

    # ── Persistence (2026-04-15) ─────────────────────────────────
    # EdgeDetectors need to survive worker restarts to preserve their
    # "once per lifetime" / "once per threshold crossing" semantics.
    # Without this, every spirit restart resets _crossed/_seen/_max
    # and producers re-emit past events on next observation (bug:
    # T1 sphere_clock emitted 26 times across 4 spirit restarts
    # when the rFP-specified max is 16 per Titan lifetime).
    def to_dict(self) -> dict:
        """Serialize state to dict for JSON persistence."""
        return {
            "crossed": dict(self._crossed),
            "seen": sorted(self._seen),
            "max": dict(self._max),
        }

    def load_dict(self, d: dict) -> None:
        """Restore state from a to_dict()-produced dict. Merges into
        existing state (additive) rather than replacing, so a fresh
        detector primed from disk keeps any already-observed keys.
        """
        if not isinstance(d, dict):
            return
        c = d.get("crossed", {})
        if isinstance(c, dict):
            for k, v in c.items():
                self._crossed[str(k)] = bool(v)
        s = d.get("seen", [])
        if isinstance(s, list):
            for k in s:
                self._seen.add(str(k))
        m = d.get("max", {})
        if isinstance(m, dict):
            for k, v in m.items():
                try:
                    self._max[str(k)] = float(v)
                except (TypeError, ValueError):
                    continue


# D10.1: SIGNAL_TO_PRIMITIVE — hand-crafted Maker-tunable table.
# Each cross-consumer event emits a META_CGN_SIGNAL with an event_type. The
# table maps (consumer, event_type) → {primitive: quality_nudge}.
# quality_nudge is in [0, 1] and applied as a pseudo-observation via Beta.
# quality > 0.5 increases V; quality < 0.5 decreases. Magnitude is modest
# (weight P10_SIGNAL_WEIGHT) so signals bias grounding without dominating
# real chain evidence.
SIGNAL_TO_PRIMITIVE: dict[tuple, dict[str, float]] = {
    # Producer #8 (rFP § 12 row 8 wire 2026-04-15): per-word vocab grounding.
    # MONOCULTURE-AWARE DEVIATION from rFP spec (was FORMULATE 0.55, SYNTHESIZE
    # 0.65, HYPOTHESIZE 0.55, RECALL 0.60). T1+T3 at 77% FORMULATE monoculture;
    # T2 at 77% RECALL monoculture. Applied same rebalance philosophy as P7:
    # FORMULATE 0.20 and RECALL 0.30 BIAS AWAY from dominant primitives on
    # each Titan; SYNTHESIZE/HYPOTHESIZE bumped to reinforce underserved
    # integrative/exploratory primitives. "Word grounded → use it via
    # SYNTHESIZE/HYPOTHESIZE rather than re-formulating or re-recalling."
    ("language", "concept_grounded"):   {"FORMULATE": 0.20, "SYNTHESIZE": 0.70,
                                         "HYPOTHESIZE": 0.65, "RECALL": 0.30},
    # Producer #7 (rFP § 12 row 7 + 2026-04-15 empirical deviation).
    # WEIGHTS DEVIATE FROM rFP SPEC (was FORMULATE 0.50, SYNTHESIZE 0.60,
    # HYPOTHESIZE 0.60). Live primitive distribution at time of ship:
    # T1=77% FORMULATE monoculture, T3=77% FORMULATE monoculture, T2=77%
    # RECALL monoculture. Adding further FORMULATE reinforcement would
    # amplify existing pathology. Instead: FORMULATE=0.20 BIASES AWAY
    # (quality < 0.5 reduces V); SYNTHESIZE bumped 0.60→0.70 to reinforce
    # the underserved integrative primitive (2-4% current). Net aggregate
    # FORMULATE across language producers stays ≤ 2.10 rFP budget.
    # Semantic reading: "vocab expanded → next chain, use the new word via
    # SYNTHESIZE/HYPOTHESIZE rather than formulating yet another one."
    ("language", "vocab_expanded"):     {"FORMULATE": 0.20, "SYNTHESIZE": 0.70,
                                         "HYPOTHESIZE": 0.60},
    # Producer #10 (rFP § 12 row 10 wire 2026-04-15): knowledge concept grounding.
    # MONOCULTURE-AWARE DEVIATION from rFP spec (was HYPOTHESIZE 0.70, DELEGATE 0.65,
    # RECALL 0.60). T2 at 77% RECALL monoculture; RECALL 0.60 reinforcement would
    # amplify pathology. Revised: RECALL 0.30 biases AWAY from T2 dominant; other
    # weights preserved. HYPOTHESIZE 0.70 and DELEGATE 0.65 are healing-aligned
    # (both primitives underserved across 3 Titans).
    ("knowledge", "concept_grounded"):  {"HYPOTHESIZE": 0.70, "DELEGATE": 0.65,
                                         "RECALL": 0.30},
    ("knowledge", "impasse_resolved"):  {"BREAK": 0.75, "HYPOTHESIZE": 0.7},
    # Producer #12 (rFP § 12 row 12 wire 2026-04-15): high-quality persona session.
    # MONOCULTURE-AWARE DEVIATION from rFP spec (was SPIRIT_SELF 0.70, INTROSPECT 0.65,
    # FORMULATE 0.55, SYNTHESIZE 0.55). T1/T3 at 77% FORMULATE monoculture;
    # FORMULATE 0.55 reinforcement amplifies pathology. Revised: FORMULATE 0.20
    # biases AWAY, SYNTHESIZE bumped to 0.65. SPIRIT_SELF + INTROSPECT unchanged
    # (both healing-aligned and underserved at 2-4%).
    ("social",    "session_high_qual"): {"SPIRIT_SELF": 0.70, "INTROSPECT": 0.65,
                                         "FORMULATE": 0.20, "SYNTHESIZE": 0.65},
    # Producer #11 (rFP § 12 row 11 wire 2026-04-15): low-quality persona session.
    # rFP spec preserved: EVALUATE 0.45 (mild anti) + BREAK 0.55 (healing).
    # BREAK is anti-monoculture primitive; EVALUATE < 0.5 gently discourages
    # re-evaluation of bad sessions.
    ("social",    "session_low_qual"):  {"EVALUATE": 0.45, "BREAK": 0.55},
    ("coding",    "problem_solved"):    {"SYNTHESIZE": 0.7, "EVALUATE": 0.7},
    ("coding",    "test_failed"):       {"EVALUATE": 0.4, "BREAK": 0.55},
    ("self_model", "reflection_depth"): {"INTROSPECT": 0.75, "SPIRIT_SELF": 0.65},
    ("self_model", "coherence_gain"):   {"SPIRIT_SELF": 0.7, "EVALUATE": 0.6},
    # Producer #15 (rFP § 12 row 15 wire 2026-04-15): EUREKA-adjacent signal from
    # meta_wisdom worker. INTROSPECT 0.55 added per rFP spec (was missing in
    # earlier Phase 2 mapping — picked up now that producer is being wired).
    ("meta_wisdom", "crystallized"):    {"SYNTHESIZE": 0.75, "HYPOTHESIZE": 0.65,
                                         "INTROSPECT": 0.55},
    # v3 additions (2026-04-14, rFP_meta_cgn_v3) — paired with producers
    # in Phase D rollout. Mappings determined per principle 3
    # (causation / diversification / monoculture-breaking).
    # Each producer activates when a SIGNAL_TO_PRIMITIVE entry is added.
    ("sphere_clock", "balance_held"):   {"SPIRIT_SELF": 0.70, "EVALUATE": 0.60,
                                         "INTROSPECT": 0.55},
    # Producer #2 (rFP § 12 row 2) — MSL ConceptGrounder per-concept first
    # grounding crossing (conf ≥ 0.5). Edge-detected (max 5 emissions per
    # Titan lifetime under normal flow: YOU/YES/NO/WE/THEY).
    ("msl", "concept_grounded"):        {"FORMULATE": 0.70, "INTROSPECT": 0.65,
                                         "SYNTHESIZE": 0.60},
    # Producer #3 (rFP § 12 row 3) — Dream cycle completion. Naturally throttled
    # by END_DREAMING cadence (~3×/day per Titan). Intensity scales with harvest
    # count. Anchors SPIRIT_SELF + integrative primitives (dream = consolidation).
    ("dreaming", "insight_distilled"):  {"SPIRIT_SELF": 0.75, "RECALL": 0.70,
                                         "SYNTHESIZE": 0.70, "INTROSPECT": 0.65},
    # Producer #4 (rFP § 12 row 4) — Meta-reasoning EUREKA insight (DA burst +
    # wisdom crystallization on high-conf primitive). Naturally rate-limited by
    # _eureka_cooldown_steps in meta_reasoning. Queue-pattern emission: logic
    # module is bus-ignorant, spirit_worker drains after each meta tick.
    # Intensity = novelty (first-time pattern recognition signal).
    ("meta_reasoning", "eureka"):       {"HYPOTHESIZE": 0.80, "SYNTHESIZE": 0.75,
                                         "BREAK": 0.70, "INTROSPECT": 0.65},

    # rFP β Phase 3 — NS outer-program → META-CGN coupling (5 outer/personality
    # programs become META-CGN producers via their existing Phase 2c discrete
    # event hooks in spirit_worker). Bridges NS interoceptive layer to META-CGN
    # cognitive layer — the architectural goal of § 4g.
    #
    # All entries are MONOCULTURE-AWARE per same discipline as Phase D producers:
    # FORMULATE 0.20 + RECALL 0.30 BIAS AWAY from T1/T3 FORMULATE / T2 RECALL
    # monoculture; underserved primitives (SYNTHESIZE, HYPOTHESIZE, DELEGATE,
    # INTROSPECT, BREAK, SPIRIT_SELF) get the reinforcement.
    ("inspiration", "fired"):           {"HYPOTHESIZE": 0.70, "SYNTHESIZE": 0.65,
                                         "BREAK": 0.60},
    ("creativity",  "fired"):           {"SYNTHESIZE": 0.70, "FORMULATE": 0.20,
                                         "HYPOTHESIZE": 0.60},
    ("curiosity",   "fired"):           {"HYPOTHESIZE": 0.70, "DELEGATE": 0.65,
                                         "RECALL": 0.30},
    ("empathy",     "fired"):           {"DELEGATE": 0.70, "SPIRIT_SELF": 0.60,
                                         "EVALUATE": 0.55},
    ("reflection",  "fired"):           {"INTROSPECT": 0.75, "SPIRIT_SELF": 0.65,
                                         "EVALUATE": 0.55},
}

# D10.2: pseudo-observation weight (small vs real chain evidence weight=1.0)
P10_SIGNAL_WEIGHT = 0.1

# D10.3: signal-specific decay — faster than grounding decay (γ=0.9 vs 0.98)
P10_SIGNAL_DECAY_GAMMA = 0.9
P10_SIGNAL_DECAY_CADENCE = 500      # chains between decay ticks on signal slice

# Narrative bridge intensity threshold — documented only in v1 (hook stub)
P10_NARRATIVE_TRIGGER_INTENSITY = 0.8


def _rank_hybrid(response: dict, impasse_signal: str,
                 impasse_diagnostic: str,
                 impasse_domain: str = "general") -> float:
    """D8.3 B-hybrid relevance × confidence ranking.

    relevance = 0.5 · source_affinity + 0.3 · keyword_overlap + 0.2 · domain_match
    rank_score = relevance · confidence
    """
    source = str(response.get("source", ""))
    confidence = float(response.get("confidence", 0.0))
    summary = str(response.get("summary", ""))
    resp_domain = str(response.get("domain", ""))
    # Source affinity
    affinity_row = SOURCE_AFFINITY.get(impasse_signal, {})
    source_aff = float(affinity_row.get(source, 0.3))  # unknown sources → 0.3
    # Keyword overlap between diagnostic and summary (Jaccard)
    diag_kws = _extract_keywords(impasse_diagnostic)
    resp_kws = _extract_keywords(summary)
    if diag_kws or resp_kws:
        inter = len(diag_kws & resp_kws)
        union = len(diag_kws | resp_kws)
        kw_overlap = float(inter / union) if union else 0.0
    else:
        kw_overlap = 0.0
    # Domain match bonus
    domain_match = 1.0 if (resp_domain and resp_domain == impasse_domain) else 0.3
    # Combined relevance
    relevance = 0.5 * source_aff + 0.3 * kw_overlap + 0.2 * domain_match
    return float(relevance * confidence)

# Must match META_PRIMITIVES order in meta_reasoning.py
PRIMITIVES = [
    "FORMULATE", "RECALL", "HYPOTHESIZE", "DELEGATE",
    "SYNTHESIZE", "EVALUATE", "BREAK", "SPIRIT_SELF",
    "INTROSPECT",
]
PRIMITIVE_INDEX = {p: i for i, p in enumerate(PRIMITIVES)}
NUM_PRIMITIVES = len(PRIMITIVES)

# CGN consumer config — matches rFP_meta_cgn_v2.md §3.0 and §16 Phase 1
CONSUMER_NAME = "meta"
FEATURE_DIMS = 30
ACTION_DIMS = NUM_PRIMITIVES  # 9 — one per primitive


@dataclass
class PrimitiveConcept:
    """Grounded value state for a single cognitive primitive (rFP §4.1).

    P6 (2026-04-12): `alpha`/`beta` are the Bayesian Beta posterior parameters —
    authoritative source of V, CI, and confidence. Legacy `V`/`confidence`/
    `n_samples`/`variance` fields are kept as derived views (recomputed from
    α,β on every update) so existing readers (API, dashboard, tests) keep
    working without churn.

    `by_domain` (I3): per-domain Beta posteriors — dict[domain] = [alpha,beta,n].
    Populated lazily as domain observations arrive; used in composition when
    n_domain ≥ domain_obs_threshold, else composition falls back to pooled.
    """
    primitive_id: str
    # P6 core: Beta posterior — uniform prior at Beta(1, 1)
    alpha: float = 1.0
    beta: float = 1.0
    # Derived views (kept for backcompat)
    V: float = 0.5                  # grounded value [0, 1], neutral prior
    confidence: float = 0.0         # [0, 1] — grows with samples
    n_samples: int = 0              # total grounding events
    variance: float = 0.25          # running variance of observations
    last_updated_ts: float = 0.0
    last_updated_chain: int = 0
    cross_consumer_signals: dict = field(default_factory=dict)
    # Phase 2: HAOV confirmed rules affecting this primitive
    haov_rules: list = field(default_factory=list)  # list of confirmed hypothesis_ids
    # P6 I3: per-domain Beta posterior — {domain: [alpha, beta, n]}
    by_domain: dict = field(default_factory=dict)

    def recompute_derived(self) -> None:
        """Refresh V/confidence/n_samples/variance from α,β (idempotent)."""
        a = max(BETA_PARAM_FLOOR, float(self.alpha))
        b = max(BETA_PARAM_FLOOR, float(self.beta))
        self.V = _beta_mean(a, b)
        self.confidence = _posterior_confidence(a, b)
        self.n_samples = int((a - BETA_PARAM_FLOOR) + (b - BETA_PARAM_FLOOR))
        # Beta variance
        self.variance = float((a * b) /
                              (max(1e-9, (a + b) ** 2) * max(1e-9, (a + b + 1))))


@dataclass
class HypothesisTest:
    """A single HAOV hypothesis about primitive dynamics (rFP §4.2).

    Evidence accumulates in `observations` (rolling window). When enough data
    exists, the registered test function runs and updates status. Confirmed
    rules are applied as confidence multipliers when composition runs.
    """
    hypothesis_id: str
    description: str
    # Test function key — dispatches to _run_test_<key> at test time.
    test_kind: str
    # Target: specific primitive or None (population-level)
    target_primitive: Optional[str] = None
    min_samples: int = 30
    evidence_window: int = 500
    confirmation_threshold: float = 0.1  # effect size required
    observations: deque = field(default_factory=lambda: deque(maxlen=500))
    status: str = "nascent"          # nascent | testing | confirmed | falsified
    test_count: int = 0
    last_test_ts: float = 0.0
    effect_size: float = 0.0         # observed magnitude if confirmed
    confidence_multiplier: float = 1.0  # applied when composing
    notes: str = ""


# ── Seed hypotheses (5 per Maker approval 2026-04-12) ─────────────────

def _build_seed_hypotheses() -> dict[str, HypothesisTest]:
    """Return the 5 initial HAOV hypotheses seeded at consumer init.

    H1, H4, H5 are monoculture/impasse-related and testable with modest
    samples (30-50). H2, H3 require ≥100 samples per stratum — gated by
    min_samples so they don't fire prematurely.
    """
    return {
        "H1_monoculture": HypothesisTest(
            hypothesis_id="H1_monoculture",
            description="Most-selected primitive has V below population average",
            test_kind="monoculture",
            min_samples=30,
            evidence_window=500,
            confirmation_threshold=0.05,
        ),
        "H2_domain_affinity": HypothesisTest(
            hypothesis_id="H2_domain_affinity",
            description="Primitive V varies by domain (|V_D - V_D'| ≥ 0.1)",
            test_kind="domain_affinity",
            min_samples=100,            # stricter — per-primitive × per-domain
            evidence_window=1000,
            confirmation_threshold=0.10,
        ),
        "H3_position_effect": HypothesisTest(
            hypothesis_id="H3_position_effect",
            description="Primitive V at position 0 differs from position 3+",
            test_kind="position_effect",
            min_samples=100,
            evidence_window=1000,
            confirmation_threshold=0.10,
        ),
        "H4_mono_context_v_drop": HypothesisTest(
            hypothesis_id="H4_mono_context_v_drop",
            description="Dominant primitive V drops ≥0.1 in high-monoculture state",
            test_kind="mono_context_v_drop",
            min_samples=30,
            evidence_window=500,
            confirmation_threshold=0.10,
        ),
        "H5_impasse_primitives": HypothesisTest(
            hypothesis_id="H5_impasse_primitives",
            description="BREAK/HYPOTHESIZE V in impasse > in non-impasse",
            test_kind="impasse_primitives",
            min_samples=20,
            evidence_window=500,
            confirmation_threshold=0.05,
        ),
        # P7: advisor disagreement quality hypothesis
        "H6_advisor_disagreement": HypothesisTest(
            hypothesis_id="H6_advisor_disagreement",
            description="High α-vs-β disagreement correlates with lower "
                        "terminal reward (signals miscalibration)",
            test_kind="advisor_disagreement",
            min_samples=30,
            evidence_window=500,
            confirmation_threshold=0.1,
        ),
    }


class MetaCGNConsumer:
    """META-CGN consumer client — registers primitives as CGN concepts.

    Lifecycle:
      1. __init__ — load primitive_grounding.json, create CGNConsumerClient,
         send CGN_REGISTER once.
      2. encode_state(primitive_id, ctx) — build 30D state vector.
      3. send_transition(...) — per chain conclude, emit CGN_TRANSITION for
         each participating primitive.
      4. update_primitive_V(...) — local EMA update (source of truth for
         primitive grounding; /dev/shm V(s) is the CGN-worker-trained version
         used for composition).
      5. compose_template_score(...) — shadow-mode info signal in Phase 1.

    Failsafe: all public methods are exception-safe. If anything breaks,
    meta-reasoning continues via chain_iql alone (rFP §11).
    """

    def __init__(self, send_queue=None, titan_id: str = "T1",
                 save_dir: str = "data/meta_cgn",
                 module_name: str = "spirit",
                 shm_path: str = "/dev/shm/cgn_live_weights.bin"):
        self._send_queue = send_queue
        self._titan_id = titan_id
        self._save_dir = save_dir
        self._module_name = module_name
        self._shm_path = shm_path

        os.makedirs(save_dir, exist_ok=True)
        self._grounding_path = os.path.join(save_dir, "primitive_grounding.json")
        self._shadow_log_path = os.path.join(save_dir, "shadow_mode_log.jsonl")
        self._haov_path = os.path.join(save_dir, "haov_hypotheses.json")
        # P4+P5 paths
        self._watchdog_path = os.path.join(save_dir, "watchdog_state.json")
        self._failure_log_path = os.path.join(save_dir, "failure_log.jsonl")
        self._disagreements_log_path = os.path.join(save_dir,
                                                    "disagreements.jsonl")

        self._primitives: dict[str, PrimitiveConcept] = {
            p: PrimitiveConcept(primitive_id=p) for p in PRIMITIVES
        }
        # Phase 2 — HAOV hypothesis tracking (initialized BEFORE _load_state
        # so HAOV reload can restore hypothesis status fields).
        self._hypotheses: dict[str, HypothesisTest] = _build_seed_hypotheses()
        self._test_cadence = 50
        self._evidence_since_last_test = 0

        # ── P4: Graduation state machine (linear ramp over 100 chains) ──
        # Status progression: shadow_mode → graduating → active → (maybe) rolled_back
        # Failure side: {status} → disabled_failsafe → shadow_mode (post-cooldown)
        self._status = "shadow_mode"
        self._graduation_progress = 0       # 0..100, scales β contribution
        self._graduation_ts = 0.0            # when first entered graduating
        self._pre_graduation_baseline: dict = {}  # for rollback detector
        self._chains_since_graduation = 0
        self._rolled_back_count = 0
        # Per-chain outcome tracking (success rate + chain count post-grad)
        self._post_grad_outcomes: deque = deque(maxlen=50)  # 1.0 / 0.0 per chain
        # Pre-graduation rolling baseline
        self._pre_grad_outcomes: deque = deque(maxlen=100)

        # ── P5: Failsafe watchdog (severity-weighted, dedup'd) ──
        # Severity per failure kind (I-5)
        self._severity_map = {
            "encoding_error": 1,
            "transition_error": 1,
            "persistence_error": 1,
            "haov_error": 2,
            "composition_error": 3,
            "graduation_error": 3,
        }
        self._failure_window = deque(maxlen=100)  # rolling window of {ts, chain_id, kind, sig, severity}
        self._failure_signatures_in_window: set[str] = set()
        self._consecutive_failures = 0
        self._total_failures = 0
        self._cooldown_remaining = 0          # chains remaining in cooldown
        self._disabled_reason = ""
        self._failsafe_trip_count = 0         # lifetime count of failsafe trips
        self._last_failure_ts = 0.0
        self._severity_trip_threshold = 9      # trip at sum severity ≥ 9

        # ── P5 F8: Cognitive impasse detection ──
        # Rolling windows of primitive V and confidence snapshots
        self._impasse_window_chains = 500
        self._impasse_v_history: deque = deque(maxlen=500)  # list of {p: V}
        self._impasse_conf_history: deque = deque(maxlen=500)
        self._impasse_state = "healthy"        # healthy | v_flatline | conf_flatline | haov_stagnant | graduation_flatline
        self._impasse_detected_ts = 0.0
        self._impasse_alpha_boost_remaining = 0  # chains of 2x α boost
        self._impasse_total_fires = 0
        self._last_graduation_blockers_hash = ""
        self._graduation_blockers_unchanged_chains = 0

        self._load_state()
        self._load_watchdog_state()  # P5 I-9: persistent failsafe state

        self._cgn_client = None
        self._registered = False
        self._total_transitions_sent = 0
        self._total_updates_applied = 0
        self._total_compositions = 0
        self._total_disagreements = 0
        self._start_ts = time.time()
        # ── P6: decay cadence + β-dispersion EMA (I1) ──
        self._chains_since_decay = 0
        self._beta_dispersion_ema = 0.0
        self._beta_dispersion_alpha = 0.01  # EMA weight → ~100-chain window
        self._total_rerank_samples = 0
        self._domain_fallbacks = 0    # I3: count uses of pooled fallback
        self._domain_hits = 0          # I3: count uses of per-domain V
        # ── P7: EUREKA telemetry + advisor-conflict throttle ──
        self._eureka_accelerated_updates = 0   # total Beta updates at weight>1
        self._eureka_trigger_counts: dict = {  # per-primitive trigger tallies
            p: 0 for p in PRIMITIVES}
        self._chain_counter = 0                # monotonic; advances every outcome
        self._conflict_throttle: dict = {}     # {(tmpl, domain): last_chain_emitted}
        self._conflict_throttle_cooldown = 100  # chains
        self._conflict_bus_events_emitted = 0
        self._conflict_sigs_throttled = 0
        # Bounded FIFO: max disagreement observed per chain during rerank →
        # picked up by observe_chain_evidence for H6 observations
        self._pending_disagreement_by_chain: dict = {}
        self._disagreement_cache_max = 50
        # ── P8: SOAR-via-CGN full protocol ──
        # Pending knowledge requests (multi-consumer aggregation window)
        # {request_id: {"start_ts", "impasse_signal", "impasse_diagnostic",
        #               "domain", "responses": [response_dict, ...],
        #               "finalized": bool}}
        self._pending_knowledge_requests: dict = {}
        self._current_impasse_req_signature: Optional[str] = None
        self._knowledge_requests_emitted = 0
        self._knowledge_requests_deduped = 0
        self._knowledge_responses_received = 0
        self._knowledge_requests_finalized = 0
        self._knowledge_requests_empty = 0   # window closed with 0 responses
        # D8.5 source credit tracking
        self._knowledge_provided_by_source: dict = {}  # consumer → count
        self._knowledge_helpful_by_source: dict = {}    # consumer → count (injected)
        # D8.4 responder tallies
        self._knowledge_responses_sent = 0
        # ── P10: Cross-consumer signal flow Layer 1 ──
        self._signals_received = 0
        self._signals_applied = 0
        self._signals_rejected_unknown = 0
        self._signals_by_consumer: dict = {}
        self._narrative_hooks_deferred = 0   # count when intensity ≥ threshold
                                              # but no narrative bridge yet
        self._chains_since_signal_decay = 0

        self._init_cgn_client()

        # ── P5 I-8: Boot self-test before registering ──
        if self._run_boot_selftest():
            self._send_register()
        else:
            logger.warning("[MetaCGN] Boot self-test failed — consumer "
                           "disabled. Manual reset via reset_watchdog() to "
                           "re-enable.")

    # ── Initialization ──────────────────────────────────────────────

    def _init_cgn_client(self) -> None:
        """Lazy-create the CGNConsumerClient. Failsafe — if CGN infra is down,
        consumer still accumulates local grounding (just no /dev/shm reads)."""
        try:
            from titan_plugin.logic.cgn_consumer_client import CGNConsumerClient
            self._cgn_client = CGNConsumerClient(
                consumer_name=CONSUMER_NAME,
                send_queue=self._send_queue,
                module_name=self._module_name,
                shm_path=self._shm_path,
            )
            logger.info("[MetaCGN] CGNConsumerClient initialized (consumer=%s, "
                        "module=%s, shm=%s)",
                        CONSUMER_NAME, self._module_name, self._shm_path)
        except Exception as e:
            self._cgn_client = None
            logger.warning("[MetaCGN] CGNConsumerClient init failed: %s "
                           "(shadow-mode grounding still accumulates locally)",
                           e)

    def _send_register(self) -> None:
        """Register META as 7th CGN consumer with the CGN Worker."""
        if self._send_queue is None:
            logger.info("[MetaCGN] No send_queue — skipping CGN_REGISTER "
                        "(standalone/test mode)")
            return
        try:
            msg = {
                "type": "CGN_REGISTER",
                "src": self._module_name,
                "dst": "cgn",
                "ts": time.time(),
                "payload": {
                    "name": CONSUMER_NAME,
                    "feature_dims": FEATURE_DIMS,
                    "action_dims": ACTION_DIMS,
                    "action_names": list(PRIMITIVES),
                    "reward_source": "meta_chain_terminal_reward",
                    "max_buffer_size": 500,
                    "consolidation_priority": 5,
                },
            }
            self._send_queue.put(msg)
            self._registered = True
            logger.info("[MetaCGN] Sent CGN_REGISTER (primitives=%d, "
                        "feature_dims=%d, action_dims=%d)",
                        NUM_PRIMITIVES, FEATURE_DIMS, ACTION_DIMS)
        except Exception as e:
            logger.warning("[MetaCGN] CGN_REGISTER send failed: %s", e)

    # ── State encoding ─────────────────────────────────────────────

    def encode_state(self, primitive_id: str, ctx: dict) -> np.ndarray:
        """Build the 30D state vector for a given primitive + context.

        Schema (per rFP §3 and Phase 0 design):
          [0:9]   primitive one-hot
          [9:13]  chain position (4)
          [13:17] reward trend (4)
          [17:22] neuromod + chi (5)
          [22:24] meta-state signals (2)
          [24:27] cross-consumer recency (3)
          [27:30] spirit context (3)

        All fields bounded [-1, 1] or [0, 1]. Missing ctx keys fall back to
        safe defaults (typically 0.5 or 0). Failsafe — returns zeros on error.
        """
        try:
            vec = np.zeros(FEATURE_DIMS, dtype=np.float32)
            idx = PRIMITIVE_INDEX.get(primitive_id, -1)
            if 0 <= idx < NUM_PRIMITIVES:
                vec[idx] = 1.0

            # Chain position (4)
            vec[9]  = _clip01(float(ctx.get("chain_len", 0)) / 20.0)
            vec[10] = _clip01(float(ctx.get("chain_step", 0)) / 20.0)
            vec[11] = _clip01(float(ctx.get("chains_since_eureka", 100)) / 100.0)
            vec[12] = _clip01(float(ctx.get("chains_since_impasse", 50)) / 50.0)

            # Reward trend (4)
            vec[13] = _clip01(float(ctx.get("terminal_reward_ema", 0.5)))
            vec[14] = _clip01(float(ctx.get("success_rate_20", 0.5)))
            vec[15] = _clip01(float(ctx.get("confidence_avg_20", 0.5)))
            vec[16] = _clip01(abs(float(ctx.get("gut_conf_gap", 0.0))))

            # Neuromod + chi (5)
            vec[17] = _clip01(float(ctx.get("DA", 0.5)))
            vec[18] = _clip01(float(ctx.get("5HT", 0.5)))
            vec[19] = _clip01(float(ctx.get("NE", 0.5)))
            vec[20] = _clip01(float(ctx.get("GABA", 0.5)))
            vec[21] = _clip01(float(ctx.get("chi_total", 0.5)))

            # Meta-state signals (2)
            vec[22] = 1.0 if ctx.get("is_in_soar_impasse", False) else 0.0
            vec[23] = _clip01(float(ctx.get("monoculture_share", 0.0)))

            # Cross-consumer recency (3)
            vec[24] = _clip01(float(ctx.get("language_grounding_rate", 0.0)))
            vec[25] = _clip01(float(ctx.get("knowledge_acq_rate", 0.0)))
            vec[26] = _clip01(float(ctx.get("social_quality_delta", 0.5)))

            # Spirit context (3)
            vec[27] = _clip01(float(ctx.get("sleep_drive", 0.5)))
            vec[28] = _clip01(float(ctx.get("wake_drive", 0.5)))
            vec[29] = 1.0 if ctx.get("is_dreaming", False) else 0.0

            return vec
        except Exception as e:
            logger.warning("[MetaCGN] encode_state failed (%s) — returning "
                           "zeros", e)
            return np.zeros(FEATURE_DIMS, dtype=np.float32)

    # ── Grounding updates ──────────────────────────────────────────

    def update_primitive_V(self, primitive_id: str, quality: float,
                           chain_id: int = 0, weight: float = 1.0,
                           domain: Optional[str] = None) -> None:
        """P6: Bayesian Beta(α,β) posterior update of primitive grounding.

        Conjugate update: α += w·quality; β += w·(1-quality). Quality is
        treated as continuous evidence (not hard pass/fail). Confidence grows
        via saturating n_eff/500 as before. V is now posterior mean α/(α+β).

        `domain` (I3): when provided, also updates per-domain Beta posterior.
        `weight`: P5 impasse-boost multiplier (2× during impasse recovery).

        Failsafe — exceptions don't propagate; logged to failure watchdog.
        """
        # P5: disabled-state short circuit
        if self._status in ("disabled_failsafe",
                            "disabled_boot_selftest_failed"):
            return
        try:
            if primitive_id not in self._primitives:
                return
            p = self._primitives[primitive_id]
            quality = max(0.0, min(1.0, float(quality)))
            w = max(0.0, float(weight))
            # P5 F8: impasse α-boost ports to Beta as 2× observation weight
            if self._impasse_alpha_boost_remaining > 0:
                w *= COMPOSITION_DEFAULTS["impasse_weight_mul"]
            # P7: telemetry — any weight>1 is an accelerated update (EUREKA/impasse)
            if w > 1.0 + 1e-6:
                self._eureka_accelerated_updates += 1
            # Beta posterior update
            p.alpha = max(BETA_PARAM_FLOOR, p.alpha + w * quality)
            p.beta = max(BETA_PARAM_FLOOR, p.beta + w * (1.0 - quality))
            # I3: per-domain posterior (dynamic enumeration)
            if domain:
                d = str(domain)
                entry = p.by_domain.get(d)
                if entry is None:
                    entry = [BETA_PARAM_FLOOR, BETA_PARAM_FLOOR, 0]
                a_d, b_d, n_d = float(entry[0]), float(entry[1]), int(entry[2])
                a_d = max(BETA_PARAM_FLOOR, a_d + w * quality)
                b_d = max(BETA_PARAM_FLOOR, b_d + w * (1.0 - quality))
                n_d += 1
                p.by_domain[d] = [a_d, b_d, n_d]
            # Refresh derived fields + bookkeeping
            p.recompute_derived()
            p.last_updated_ts = time.time()
            p.last_updated_chain = chain_id
            self._total_updates_applied += 1
        except Exception as e:
            logger.warning("[MetaCGN] update_primitive_V(%s) failed: %s",
                           primitive_id, e)
            self._record_failure("encoding_error", e, chain_id)

    # ── P6 D2: chain-count decay ─────────────────────────────────────

    def _maybe_apply_decay(self) -> None:
        """P6 D2: every `decay_cadence_chains` chains, multiply α,β by γ.
        Floors at BETA_PARAM_FLOOR. Skips primitives with n < decay_skip_n_min.

        Decay re-opens confidence (n_eff shrinks), keeps V unchanged, and lets
        stale grounding fade. Called from record_chain_outcome.
        """
        cadence = int(COMPOSITION_DEFAULTS["decay_cadence_chains"])
        gamma = float(COMPOSITION_DEFAULTS["decay_gamma"])
        skip_n_min = int(COMPOSITION_DEFAULTS["decay_skip_n_min"])
        if cadence <= 0 or gamma >= 1.0 or gamma <= 0.0:
            return
        if getattr(self, "_chains_since_decay", 0) < cadence:
            return
        self._chains_since_decay = 0
        try:
            for p in self._primitives.values():
                n_eff = (p.alpha - BETA_PARAM_FLOOR) + (p.beta - BETA_PARAM_FLOOR)
                if n_eff < skip_n_min:
                    continue
                # Decay the excess above the floor; floor is preserved
                excess_a = max(0.0, p.alpha - BETA_PARAM_FLOOR) * gamma
                excess_b = max(0.0, p.beta - BETA_PARAM_FLOOR) * gamma
                p.alpha = BETA_PARAM_FLOOR + excess_a
                p.beta = BETA_PARAM_FLOOR + excess_b
                # Domain posteriors decay too (same multiplicative factor)
                for dkey, entry in list(p.by_domain.items()):
                    a_d, b_d, n_d = float(entry[0]), float(entry[1]), int(entry[2])
                    n_eff_d = (a_d - BETA_PARAM_FLOOR) + (b_d - BETA_PARAM_FLOOR)
                    if n_eff_d < skip_n_min:
                        continue
                    a_d = BETA_PARAM_FLOOR + max(0.0, a_d - BETA_PARAM_FLOOR) * gamma
                    b_d = BETA_PARAM_FLOOR + max(0.0, b_d - BETA_PARAM_FLOOR) * gamma
                    p.by_domain[dkey] = [a_d, b_d, n_d]
                p.recompute_derived()
            logger.info("[MetaCGN] Applied decay γ=%.3f to α,β posteriors "
                        "(cadence=%d chains)", gamma, cadence)
        except Exception as e:
            logger.warning("[MetaCGN] decay failed: %s", e)

    # ── Phase 2: HAOV hypothesis tracking ──────────────────────────

    def observe_chain_evidence(self, chain_info: dict) -> None:
        """Ingest per-chain evidence for all active hypotheses.

        chain_info schema (emitted by MetaReasoningEngine._conclude_chain):
          {
            "primitives": [primitive_id, ...],       # in-chain order
            "quality": float,                        # normalized terminal_reward
            "domain": str,                           # chain domain
            "monoculture_share": float,              # [0, 1]
            "is_in_soar_impasse": bool,
            "dominant_primitive": str,               # currently-dominant primitive
            "pop_avg_V": float,                      # population V snapshot
            "per_primitive_V": dict,                 # V before this update
            "chain_id": int,
            "ts": float,
          }

        Observation fan-out per hypothesis — kept simple: each hypothesis
        appends a relevant projection to its observations deque.
        """
        if self._status in ("disabled_failsafe",
                            "disabled_boot_selftest_failed"):
            return
        try:
            ts = chain_info.get("ts", time.time())
            chain_id = chain_info.get("chain_id", 0)
            primitives = chain_info.get("primitives", []) or []
            quality = float(chain_info.get("quality", 0.0))
            domain = chain_info.get("domain", "general")
            mono_share = float(chain_info.get("monoculture_share", 0.0))
            in_impasse = bool(chain_info.get("is_in_soar_impasse", False))
            dominant = chain_info.get("dominant_primitive", "")
            pop_avg_V = float(chain_info.get("pop_avg_V", 0.5))
            per_prim_V = chain_info.get("per_primitive_V", {}) or {}

            # H1 — monoculture: dominant primitive V relative to population avg
            if dominant:
                self._hypotheses["H1_monoculture"].observations.append({
                    "ts": ts, "dominant": dominant,
                    "dominant_V": float(per_prim_V.get(dominant, 0.5)),
                    "pop_avg_V": pop_avg_V,
                })

            # H2 — domain affinity: per (primitive, domain) V observation
            for prim in set(primitives):
                self._hypotheses["H2_domain_affinity"].observations.append({
                    "ts": ts, "primitive": prim, "domain": domain,
                    "quality": quality,
                })

            # H3 — position effect: per (primitive, position) V observation
            for pos, prim in enumerate(primitives):
                self._hypotheses["H3_position_effect"].observations.append({
                    "ts": ts, "primitive": prim, "position": pos,
                    "quality": quality,
                })

            # H4 — mono-context V drop: dominant primitive V in high vs low share
            if dominant:
                self._hypotheses["H4_mono_context_v_drop"].observations.append({
                    "ts": ts, "dominant": dominant,
                    "mono_share": mono_share,
                    "dominant_V": float(per_prim_V.get(dominant, 0.5)),
                })

            # H5 — impasse primitives: per primitive × impasse flag
            for prim in set(primitives):
                if prim in ("BREAK", "HYPOTHESIZE"):
                    self._hypotheses["H5_impasse_primitives"].observations.append({
                        "ts": ts, "primitive": prim,
                        "in_impasse": in_impasse, "quality": quality,
                    })

            # P7: EUREKA telemetry — count per-trigger primitive
            if bool(chain_info.get("eureka_fired", False)):
                trig = str(chain_info.get("eureka_trigger", ""))
                if trig in self._eureka_trigger_counts:
                    self._eureka_trigger_counts[trig] += 1

            # P8: stash current chain domain — if an impasse fires during
            # this session, the knowledge request can target this domain
            if domain:
                self._last_impasse_domain = domain

            # P7 H6: advisor-disagreement observation — link the cached
            # max disagreement from template selection to the chain's quality
            cid = chain_info.get("chain_id")
            disagreement_for_h6 = self._pending_disagreement_by_chain.pop(
                cid, None)
            if disagreement_for_h6 is not None:
                self._hypotheses["H6_advisor_disagreement"].observations.append({
                    "ts": ts,
                    "chain_id": cid,
                    "disagreement": float(disagreement_for_h6),
                    "quality": quality,
                })

            self._evidence_since_last_test += 1
            if self._evidence_since_last_test >= self._test_cadence:
                self._evidence_since_last_test = 0
                self._run_due_tests()
        except Exception as e:
            logger.warning("[MetaCGN] observe_chain_evidence failed: %s", e)
            self._record_failure("haov_error", e,
                                 chain_info.get("chain_id", 0))

    def _run_due_tests(self) -> None:
        """Run statistical tests on all hypotheses with enough samples."""
        dispatch = {
            "monoculture": self._test_h1_monoculture,
            "domain_affinity": self._test_h2_domain_affinity,
            "position_effect": self._test_h3_position_effect,
            "mono_context_v_drop": self._test_h4_mono_context_v_drop,
            "impasse_primitives": self._test_h5_impasse_primitives,
            "advisor_disagreement": self._test_h6_advisor_disagreement,
        }
        for h in self._hypotheses.values():
            if h.status in ("confirmed", "falsified"):
                continue  # latched — ignore further tests until reset
            if len(h.observations) < h.min_samples:
                continue
            try:
                fn = dispatch.get(h.test_kind)
                if fn is None:
                    continue
                effect, confirmed = fn(h)
                h.test_count += 1
                h.last_test_ts = time.time()
                h.effect_size = float(effect)
                if confirmed:
                    h.status = "confirmed"
                    h.confidence_multiplier = 1.0 + min(0.5,
                        max(0.0, abs(float(effect))))
                    # Tag primitives affected by this rule
                    self._apply_rule_to_primitives(h)
                    logger.info("[MetaCGN] HAOV confirmed: %s (effect=%.3f, "
                                "multiplier=%.2f)",
                                h.hypothesis_id, effect, h.confidence_multiplier)
            except Exception as e:
                logger.warning("[MetaCGN] Hypothesis test %s failed: %s",
                               h.hypothesis_id, e)

    def _apply_rule_to_primitives(self, h: HypothesisTest) -> None:
        """Mark primitives referenced by a confirmed hypothesis."""
        # Determine which primitives are affected based on hypothesis kind.
        affected: list[str] = []
        if h.test_kind == "monoculture":
            # Affects whichever primitive was most-selected across window
            dom_counts: dict[str, int] = {}
            for obs in h.observations:
                d = obs.get("dominant", "")
                if d:
                    dom_counts[d] = dom_counts.get(d, 0) + 1
            if dom_counts:
                affected = [max(dom_counts, key=dom_counts.get)]
        elif h.test_kind == "impasse_primitives":
            affected = ["BREAK", "HYPOTHESIZE"]
        # Generic fan-out — all others would be per-primitive in evidence
        else:
            prim_set = set()
            for obs in h.observations:
                p = obs.get("primitive", "")
                if p:
                    prim_set.add(p)
            affected = list(prim_set)
        for prim in affected:
            if prim in self._primitives:
                rules = self._primitives[prim].haov_rules
                if h.hypothesis_id not in rules:
                    rules.append(h.hypothesis_id)

    # ── Individual hypothesis tests ────────────────────────────────

    def _test_h1_monoculture(self, h: HypothesisTest) -> tuple[float, bool]:
        """H1: Most-selected primitive has V below population average.

        Computes mean (dominant_V - pop_avg_V) across observations. Confirms
        if dominant_V is meaningfully lower than pop_avg (evidence of
        over-selection despite inferior value).
        """
        diffs = [float(o["dominant_V"]) - float(o["pop_avg_V"])
                 for o in h.observations
                 if "dominant_V" in o and "pop_avg_V" in o]
        if not diffs:
            return 0.0, False
        mean_diff = sum(diffs) / len(diffs)
        # Confirmed if dominant_V is at least threshold BELOW pop avg
        confirmed = mean_diff <= -h.confirmation_threshold
        return abs(mean_diff), confirmed

    def _test_h2_domain_affinity(self, h: HypothesisTest) -> tuple[float, bool]:
        """H2: Primitive V varies by domain (some pair shows |ΔV| ≥ threshold).

        Group observations by (primitive, domain), require ≥20 samples per cell.
        Find the largest |V_D - V_D'| for any primitive with ≥2 domains.
        """
        by_prim_domain: dict[tuple, list] = {}
        for obs in h.observations:
            key = (obs.get("primitive", ""), obs.get("domain", ""))
            by_prim_domain.setdefault(key, []).append(float(obs["quality"]))
        # Group by primitive
        by_prim: dict[str, dict[str, list]] = {}
        for (prim, dom), qs in by_prim_domain.items():
            if len(qs) >= 20:
                by_prim.setdefault(prim, {})[dom] = qs
        best_diff = 0.0
        for prim, doms in by_prim.items():
            if len(doms) < 2:
                continue
            means = {d: sum(q) / len(q) for d, q in doms.items()}
            span = max(means.values()) - min(means.values())
            if span > best_diff:
                best_diff = span
        return best_diff, best_diff >= h.confirmation_threshold

    def _test_h3_position_effect(self, h: HypothesisTest) -> tuple[float, bool]:
        """H3: Primitive V at position 0 differs from position 3+ for any prim."""
        by_pos: dict[tuple, list] = {}
        for obs in h.observations:
            prim = obs.get("primitive", "")
            pos = int(obs.get("position", 0))
            pos_bucket = "early" if pos == 0 else ("late" if pos >= 3 else "mid")
            by_pos.setdefault((prim, pos_bucket), []).append(
                float(obs["quality"]))
        by_prim: dict[str, dict[str, list]] = {}
        for (prim, bucket), qs in by_pos.items():
            if len(qs) >= 20:
                by_prim.setdefault(prim, {})[bucket] = qs
        best_diff = 0.0
        for prim, buckets in by_prim.items():
            if "early" in buckets and "late" in buckets:
                m_early = sum(buckets["early"]) / len(buckets["early"])
                m_late = sum(buckets["late"]) / len(buckets["late"])
                diff = abs(m_early - m_late)
                if diff > best_diff:
                    best_diff = diff
        return best_diff, best_diff >= h.confirmation_threshold

    def _test_h4_mono_context_v_drop(self, h: HypothesisTest
                                     ) -> tuple[float, bool]:
        """H4: Dominant primitive V is lower in high-monoculture states."""
        high_V: list = []
        low_V: list = []
        for obs in h.observations:
            share = float(obs.get("mono_share", 0.0))
            dv = float(obs.get("dominant_V", 0.5))
            if share >= 0.7:
                high_V.append(dv)
            elif share <= 0.3:
                low_V.append(dv)
        if len(high_V) < 10 or len(low_V) < 10:
            return 0.0, False
        m_high = sum(high_V) / len(high_V)
        m_low = sum(low_V) / len(low_V)
        diff = m_low - m_high   # positive means V drops in high-mono
        return diff, diff >= h.confirmation_threshold

    def _test_h5_impasse_primitives(self, h: HypothesisTest
                                    ) -> tuple[float, bool]:
        """H5: BREAK/HYPOTHESIZE V in impasse > non-impasse."""
        by_prim_impasse: dict[tuple, list] = {}
        for obs in h.observations:
            key = (obs.get("primitive", ""), bool(obs.get("in_impasse", False)))
            by_prim_impasse.setdefault(key, []).append(float(obs["quality"]))
        best_diff = 0.0
        for prim in ("BREAK", "HYPOTHESIZE"):
            qs_imp = by_prim_impasse.get((prim, True), [])
            qs_non = by_prim_impasse.get((prim, False), [])
            if len(qs_imp) >= 5 and len(qs_non) >= 10:
                m_imp = sum(qs_imp) / len(qs_imp)
                m_non = sum(qs_non) / len(qs_non)
                diff = m_imp - m_non
                if diff > best_diff:
                    best_diff = diff
        return best_diff, best_diff >= h.confirmation_threshold

    def _test_h6_advisor_disagreement(self, h: HypothesisTest
                                      ) -> tuple[float, bool]:
        """P7 H6: high α-β disagreement correlates with low terminal reward.

        Split observations into high-disagreement (>= 0.3) and low-disagreement
        (< 0.15) groups. Effect = mean_quality(low_disagree) − mean_quality(
        high_disagree). If positive and above threshold, high-disagreement
        chains under-perform → one of the advisors is systematically wrong.
        """
        highs, lows = [], []
        for obs in h.observations:
            d = float(obs.get("disagreement", 0.0))
            q = float(obs.get("quality", 0.0))
            if d >= 0.3:
                highs.append(q)
            elif d < 0.15:
                lows.append(q)
        if len(highs) < 5 or len(lows) < 5:
            return 0.0, False
        m_high = sum(highs) / len(highs)
        m_low = sum(lows) / len(lows)
        effect = m_low - m_high  # positive means high-disagree → worse quality
        return effect, effect >= h.confirmation_threshold

    def get_haov_stats(self) -> dict:
        """HAOV telemetry for audit endpoint + get_stats."""
        by_status: dict[str, int] = {"nascent": 0, "testing": 0,
                                     "confirmed": 0, "falsified": 0}
        details = {}
        for h in self._hypotheses.values():
            by_status[h.status] = by_status.get(h.status, 0) + 1
            details[h.hypothesis_id] = {
                "status": h.status,
                "observations": len(h.observations),
                "test_count": h.test_count,
                "effect_size": round(h.effect_size, 4),
                "confidence_multiplier": round(h.confidence_multiplier, 3),
            }
        return {
            "total": len(self._hypotheses),
            "by_status": by_status,
            "details": details,
        }

    def send_transition(self, primitive_id: str, state_vec: np.ndarray,
                        reward: float, chain_id: int = 0,
                        metadata: Optional[dict] = None) -> None:
        """Emit CGN_TRANSITION to the CGN Worker for this primitive.

        Transition is buffered and trained during dream consolidation. Local
        V(s) updates happen via update_primitive_V — this feeds the standalone
        CGN worker's SharedValueNet + ConsumerActionNet training pipeline.
        """
        if self._send_queue is None:
            return
        if self._status in ("disabled_failsafe",
                            "disabled_boot_selftest_failed"):
            return
        try:
            action_idx = PRIMITIVE_INDEX.get(primitive_id, 0)
            payload = {
                "consumer": CONSUMER_NAME,
                "state": state_vec.tolist() if isinstance(state_vec, np.ndarray)
                         else list(state_vec),
                "action": int(action_idx),
                "action_name": primitive_id,
                "reward": float(reward),
                "timestamp": time.time(),
                "epoch": (metadata or {}).get("epoch", 0),
                "metadata": metadata or {"chain_id": chain_id},
            }
            self._send_queue.put({
                "type": "CGN_TRANSITION",
                "src": self._module_name,
                "dst": "cgn",
                "ts": time.time(),
                "payload": payload,
            })
            self._total_transitions_sent += 1
        except Exception as e:
            logger.warning("[MetaCGN] send_transition(%s) failed: %s",
                           primitive_id, e)
            self._record_failure("transition_error", e, chain_id)

    # ── Shadow-mode composition (rFP §3.2, §5.3) ───────────────────

    def compose_template_score(self, template_id: str, state_ctx: dict,
                               chain_iql_score: float = 0.0,
                               chain_iql_confidence: float = 0.0,
                               template_primitives: Optional[list] = None,
                               ) -> dict:
        """P6: composed V with Beta CI + anti-monoculture + per-domain signal.

        Score contributions per primitive in template:
          • V_effective = per-domain posterior mean when n_domain ≥ threshold,
                         else pooled posterior mean (I3 fallback)
          • UCB shift = ±κ_CI · CI_width
              - n < N_ANCHOR → optimistic (+κ·(CI_hi − V_eff))
              - n ≥ N_ANCHOR → pessimistic (−κ·(V_eff − CI_lo))
          • Anti-monoculture bonus = κ_explore · (n_mean − n_prim) / n_mean
              - under-sampled primitives get +bonus
              - over-sampled primitives get −bonus (breaks FORMULATE lock-in)

        Aggregation: confidence-weighted arithmetic mean with HAOV
        multiplier. Returns rich dict with breakdown for telemetry.
        """
        try:
            self._total_compositions += 1
            cfg = COMPOSITION_DEFAULTS
            kappa_explore = float(cfg["kappa_explore"])
            kappa_ci = float(cfg["kappa_ci"])
            n_anchor = float(cfg["n_anchor"])
            q_lo = float(cfg["ci_quantile_lo"])
            q_hi = float(cfg["ci_quantile_hi"])
            dom_thresh = int(cfg["domain_obs_threshold"])
            domain = str(state_ctx.get("domain", "")) if state_ctx else ""

            # Population n_mean across primitives (for F anti-monoculture bonus)
            n_list = [max(0, p.n_samples) for p in self._primitives.values()]
            n_mean_pop = (sum(n_list) / len(n_list)) if n_list else 1.0
            n_mean_safe = max(1.0, n_mean_pop)

            primitives = template_primitives or []
            per_prim_breakdown = []
            if primitives:
                # Build per-primitive (V_eff, confidence, score, breakdown)
                valid = []
                for p_id in primitives:
                    p = self._primitives.get(p_id)
                    if p is None:
                        continue
                    # I3: per-domain posterior with pooled fallback
                    a_dom, b_dom, n_dom, used_domain = (
                        p.alpha, p.beta, p.n_samples, False)
                    if domain and domain in p.by_domain:
                        a_d, b_d, n_d = p.by_domain[domain]
                        if int(n_d) >= dom_thresh:
                            a_dom, b_dom, n_dom, used_domain = (
                                float(a_d), float(b_d), int(n_d), True)
                    if used_domain:
                        self._domain_hits += 1
                    else:
                        self._domain_fallbacks += 1
                    V_eff = _beta_mean(a_dom, b_dom)
                    lo, hi, ci_width = _beta_ci_width(a_dom, b_dom, q_lo, q_hi)
                    # D4: UCB-style shift — optimistic when low-n, pessimistic when high-n
                    if n_dom < n_anchor:
                        ucb_shift = kappa_ci * (hi - V_eff)
                    else:
                        ucb_shift = -kappa_ci * (V_eff - lo)
                    # F: anti-monoculture exploration bonus — tanh-bounded so
                    # the penalty/bonus never dominates V. Range: [-κ, +κ].
                    novelty = kappa_explore * math.tanh(
                        (n_mean_safe - p.n_samples) / n_mean_safe)
                    # HAOV-confirmed rule multiplier on confidence
                    haov_boost = 1.0
                    for rid in p.haov_rules:
                        h = self._hypotheses.get(rid)
                        if h and h.status == "confirmed":
                            haov_boost *= h.confidence_multiplier
                    conf_raw = _posterior_confidence(a_dom, b_dom)
                    weight = max(0.05, conf_raw * haov_boost)
                    prim_score = V_eff + ucb_shift + novelty
                    prim_score = max(0.0, min(1.0, prim_score))
                    valid.append((prim_score, weight, V_eff, conf_raw,
                                  ucb_shift, novelty, used_domain))
                    per_prim_breakdown.append({
                        "p": p_id, "V_eff": round(V_eff, 4),
                        "ucb": round(ucb_shift, 4),
                        "novelty": round(novelty, 4),
                        "n_used": n_dom, "domain_used": used_domain,
                    })
                if valid:
                    w_total = sum(w for _, w, *_ in valid)
                    if w_total > 1e-6:
                        norm_w = [w / w_total for _, w, *_ in valid]
                    else:
                        norm_w = [1.0 / len(valid)] * len(valid)
                    composed_V = float(sum(
                        s * w for (s, _, *_), w in zip(valid, norm_w)))
                    V_confidence = float(min(c for _, _, _, c, *_ in valid))
                else:
                    composed_V = 0.5
                    V_confidence = 0.0
            else:
                composed_V = 0.5
                V_confidence = 0.0

            # Adaptive λ (rFP §3.3) — trust layer with higher confidence
            Q_conf = float(chain_iql_confidence)
            if Q_conf + V_confidence > 1e-6:
                lam = Q_conf / (Q_conf + V_confidence + 1e-6)
            else:
                lam = 0.5
            # Force low λ if primitives underexplored
            if V_confidence < 0.2 and Q_conf > 0.5:
                lam = min(0.9, max(lam, 0.7))
            lam = max(0.1, min(0.9, lam))

            blended = lam * float(chain_iql_score) + (1 - lam) * composed_V
            disagreement = abs(float(chain_iql_score) - composed_V)

            result = {
                "direct_Q": float(chain_iql_score),
                "composed_V": composed_V,
                "V_confidence": V_confidence,
                "Q_confidence": Q_conf,
                "lambda_used": lam,
                "blended_score": blended,
                "disagreement": disagreement,
                "shadow_only": True,
                # P6 breakdown for dashboard/tests
                "per_primitive": per_prim_breakdown,
            }

            # Log disagreements above threshold (rFP §13 advisor conflict)
            if disagreement > 0.3 and V_confidence > 0.3:
                self._total_disagreements += 1
                self._log_disagreement(template_id, state_ctx, result,
                                       primitives)

            return result
        except Exception as e:
            logger.warning("[MetaCGN] compose_template_score failed: %s", e)
            self._record_failure("composition_error", e)
            return {"shadow_only": True, "error": str(e)}

    def _log_disagreement(self, template_id: str, state_ctx: dict,
                          result: dict, primitives: list) -> None:
        """Append advisor-disagreement record to shadow_mode_log.jsonl.

        P7: also emit META_CGN_ADVISOR_CONFLICT bus event, throttled to one
        fire per (template, domain) signature per 100 chains. Rationale: when
        disagreement rate rises to target 15-25%, unthrottled bus emission
        would spam; throttle keeps the bus signal sparse while JSONL keeps
        full history for Maker review.
        """
        domain = str(state_ctx.get("domain", "")) if state_ctx else ""
        try:
            record = {
                "ts": time.time(),
                "template_id": template_id,
                "primitives": list(primitives or []),
                "direct_Q": result.get("direct_Q"),
                "composed_V": result.get("composed_V"),
                "disagreement": result.get("disagreement"),
                "lambda": result.get("lambda_used"),
                "domain": domain,
                "state_summary": {
                    k: state_ctx.get(k) for k in (
                        "chain_len", "monoculture_share",
                        "is_in_soar_impasse", "confidence_avg_20")
                },
            }
            with open(self._shadow_log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.debug("[MetaCGN] disagreement log failed: %s", e)
        # P7: bus event emission with per-signature throttle
        try:
            sig = (template_id, domain)
            last = self._conflict_throttle.get(sig)
            if last is not None and \
                    (self._chain_counter - last) < \
                    self._conflict_throttle_cooldown:
                self._conflict_sigs_throttled += 1
                return
            self._conflict_throttle[sig] = self._chain_counter
            if self._send_queue is not None:
                self._send_queue.put({
                    "type": "META_CGN_ADVISOR_CONFLICT",
                    "src": self._module_name,
                    "dst": "all",
                    "ts": time.time(),
                    "payload": {
                        "template_id": template_id,
                        "primitives": list(primitives or []),
                        "direct_Q": float(result.get("direct_Q", 0.0)),
                        "composed_V": float(result.get("composed_V", 0.0)),
                        "disagreement": float(result.get("disagreement", 0.0)),
                        "lambda_used": float(result.get("lambda_used", 0.5)),
                        "domain": domain,
                    },
                })
            self._conflict_bus_events_emitted += 1
        except Exception as e:
            logger.debug("[MetaCGN] conflict bus emit failed: %s", e)

    # ── Persistence ─────────────────────────────────────────────────

    def save_state(self) -> None:
        """Persist primitive_grounding.json + haov_hypotheses.json.
        Called periodically + on shutdown."""
        try:
            # Ensure derived fields reflect current α,β before serializing
            for c in self._primitives.values():
                c.recompute_derived()
            data = {
                "version": 3,  # v3: Bayesian Beta(α,β) + by_domain (P6)
                "titan_id": self._titan_id,
                "saved_ts": time.time(),
                "primitives": {
                    p: asdict(c) for p, c in self._primitives.items()
                },
                "stats": self.get_stats_compact(),
            }
            tmp = self._grounding_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, self._grounding_path)
            # Persist HAOV state separately (hypotheses + effect sizes).
            # v2 adds observation deques + impasse V/confidence histories so
            # restart doesn't reset the graduation window or HAOV sample pool.
            haov_data = {
                "version": 2,
                "saved_ts": time.time(),
                "hypotheses": {
                    h_id: {
                        "hypothesis_id": h.hypothesis_id,
                        "description": h.description,
                        "test_kind": h.test_kind,
                        "status": h.status,
                        "test_count": h.test_count,
                        "effect_size": h.effect_size,
                        "confidence_multiplier": h.confidence_multiplier,
                        "observation_count": len(h.observations),
                        "observations": list(h.observations),
                    }
                    for h_id, h in self._hypotheses.items()
                },
                "impasse_v_history": list(self._impasse_v_history),
                "impasse_conf_history": list(self._impasse_conf_history),
                "impasse_state": self._impasse_state,
                "impasse_total_fires": self._impasse_total_fires,
                "graduation_blockers_unchanged_chains":
                    self._graduation_blockers_unchanged_chains,
                "last_graduation_blockers_hash":
                    self._last_graduation_blockers_hash,
            }
            haov_tmp = self._haov_path + ".tmp"
            with open(haov_tmp, "w") as f:
                json.dump(haov_data, f, indent=2)
            os.replace(haov_tmp, self._haov_path)
        except Exception as e:
            logger.warning("[MetaCGN] save_state failed: %s", e)
            self._record_failure("persistence_error", e)

    def _load_state(self) -> None:
        """Load primitive_grounding.json + haov_hypotheses.json if present.
        Safe on first boot. v1/v2 files auto-migrate to v3 Beta schema via
        converted bootstrap (α = V·n_eff + 1, β = (1-V)·n_eff + 1, n_eff ≤ 200).
        """
        try:
            if not os.path.exists(self._grounding_path):
                return
            with open(self._grounding_path) as f:
                data = json.load(f)
            schema_version = int(data.get("version", 1))
            loaded = data.get("primitives", {})
            migrated_from = None
            for p_id, p_data in loaded.items():
                if p_id not in self._primitives:
                    continue
                # Defaults ensure v1/v2 files round-trip cleanly
                p_data.setdefault("haov_rules", [])
                p_data.setdefault("by_domain", {})
                if schema_version < 3 or "alpha" not in p_data:
                    # Converted bootstrap from v2's (V, n_samples) → Beta(α, β)
                    V_old = float(p_data.get("V", 0.5))
                    n_old = int(p_data.get("n_samples", 0))
                    n_eff = min(MIGRATION_N_EFF_CAP, max(0, n_old))
                    a = V_old * n_eff + BETA_PARAM_FLOOR
                    b = (1.0 - V_old) * n_eff + BETA_PARAM_FLOOR
                    p_data["alpha"] = float(a)
                    p_data["beta"] = float(b)
                    migrated_from = schema_version
                # Strip unknown fields (resilience against future additions)
                allowed = {
                    "primitive_id", "alpha", "beta", "V", "confidence",
                    "n_samples", "variance", "last_updated_ts",
                    "last_updated_chain", "cross_consumer_signals",
                    "haov_rules", "by_domain",
                }
                filtered = {k: v for k, v in p_data.items() if k in allowed}
                concept = PrimitiveConcept(**filtered)
                concept.recompute_derived()
                self._primitives[p_id] = concept
            if migrated_from is not None:
                logger.info("[MetaCGN] Migrated %d primitives from schema v%d "
                            "→ v3 (converted bootstrap, n_eff≤%d)",
                            len(loaded), migrated_from, MIGRATION_N_EFF_CAP)
            else:
                logger.info("[MetaCGN] Loaded %d primitives from %s (v%d)",
                            len(loaded), self._grounding_path, schema_version)
        except Exception as e:
            logger.warning("[MetaCGN] _load_state failed: %s — starting fresh",
                           e)
        # HAOV state restore (only restores status + effect — observations
        # rebuild from ongoing evidence)
        try:
            if not os.path.exists(self._haov_path):
                return
            with open(self._haov_path) as f:
                haov_data = json.load(f)
            schema = int(haov_data.get("version", 1))
            obs_restored = 0
            for h_id, h_saved in haov_data.get("hypotheses", {}).items():
                if h_id in self._hypotheses:
                    h = self._hypotheses[h_id]
                    h.status = h_saved.get("status", h.status)
                    h.test_count = h_saved.get("test_count", 0)
                    h.effect_size = h_saved.get("effect_size", 0.0)
                    h.confidence_multiplier = h_saved.get(
                        "confidence_multiplier", 1.0)
                    # v2: restore observation deque (maxlen preserved via
                    # deque ctor, which respects the original maxlen when
                    # re-wrapping). Drops excess if file is larger than cap.
                    if schema >= 2:
                        saved_obs = h_saved.get("observations", [])
                        h.observations = deque(saved_obs,
                                               maxlen=h.observations.maxlen)
                        obs_restored += len(h.observations)
            # v2: restore impasse detection history + latched state
            if schema >= 2:
                v_hist = haov_data.get("impasse_v_history", [])
                c_hist = haov_data.get("impasse_conf_history", [])
                self._impasse_v_history = deque(
                    v_hist, maxlen=self._impasse_window_chains)
                self._impasse_conf_history = deque(
                    c_hist, maxlen=self._impasse_window_chains)
                self._impasse_state = haov_data.get(
                    "impasse_state", self._impasse_state)
                self._impasse_total_fires = int(haov_data.get(
                    "impasse_total_fires", 0))
                self._graduation_blockers_unchanged_chains = int(
                    haov_data.get(
                        "graduation_blockers_unchanged_chains", 0))
                self._last_graduation_blockers_hash = haov_data.get(
                    "last_graduation_blockers_hash", "")
            logger.info(
                "[MetaCGN] Loaded HAOV state from %s "
                "(schema v%d, obs_restored=%d, v_history=%d)",
                self._haov_path, schema, obs_restored,
                len(self._impasse_v_history))
        except Exception as e:
            logger.warning("[MetaCGN] HAOV load failed: %s", e)

    # ── P4+P5: Graduation, rollback, failsafe, impasse ─────────────

    def _run_boot_selftest(self) -> bool:
        """P5 I-8: Verify core methods work before allowing activation.

        Returns True if self-test passes, False otherwise. On failure,
        sets _status = 'disabled_boot_selftest_failed'.

        State-safe: snapshots/restores primitives + hypothesis observations
        so selftest doesn't pollute counters. Skipped if already in a
        disabled_* state (e.g., reloaded from watchdog) — a prior failsafe
        trip is authoritative.
        """
        if self._status.startswith("disabled_"):
            return False
        # Snapshot state for restoration after selftest
        # P6: include α, β, by_domain — boot selftest must be fully side-effect-free
        _r = self._primitives["RECALL"]
        pre_recall = (_r.V, _r.n_samples, _r.variance, _r.confidence,
                      _r.alpha, _r.beta,
                      {k: list(v) for k, v in _r.by_domain.items()})
        # 2026-04-14 fix: snapshot full deque contents (not just lengths).
        # Previous version compared len() pre/post, which is broken when
        # the deque is at maxlen=500 (append evicts oldest, len unchanged →
        # assertion always fails after persisted state restore). Snapshotting
        # full contents also fixes the cleanup loop so test observations
        # don't permanently pollute hypothesis deques.
        pre_obs_snapshot = {h_id: list(h.observations)
                            for h_id, h in self._hypotheses.items()}
        pre_updates = self._total_updates_applied
        try:
            # 1. Encoding works
            vec = self.encode_state("FORMULATE", {"chain_len": 1})
            assert vec.shape == (FEATURE_DIMS,), "encode_state shape wrong"
            # 2. Grounding update works (uses RECALL to not touch FORMULATE)
            self.update_primitive_V("RECALL", 0.5, chain_id=-1)
            assert self._primitives["RECALL"].n_samples == pre_recall[1] + 1
            # 3. HAOV observation ingestion works — verify by domain marker
            #    (deque-maxlen-safe: works whether deque was full or not)
            self.observe_chain_evidence({
                "ts": time.time(), "chain_id": -1,
                "primitives": ["FORMULATE"], "quality": 0.5,
                "domain": "selftest", "monoculture_share": 0.5,
                "is_in_soar_impasse": False,
                "dominant_primitive": "FORMULATE", "pop_avg_V": 0.5,
                "per_primitive_V": {"FORMULATE": 0.5},
            })
            h2_obs = self._hypotheses["H2_domain_affinity"].observations
            assert h2_obs and h2_obs[-1].get("domain") == "selftest", \
                "observe_chain_evidence did not record selftest observation"
            # 4. Stats assemble
            s = self.get_stats()
            assert "status" in s and "haov" in s
            # Restore pre-selftest state — selftest must be side-effect-free.
            # RECALL primitive (P6: also restore α, β, by_domain)
            p = self._primitives["RECALL"]
            (p.V, p.n_samples, p.variance, p.confidence,
             p.alpha, p.beta, _dom_snap) = pre_recall
            p.by_domain = {k: list(v) for k, v in _dom_snap.items()}
            # Restore each hypothesis' observations from full snapshot.
            # Using deque() with original maxlen is critical so semantics
            # stay identical to a fresh load.
            for h_id, h in self._hypotheses.items():
                snap = pre_obs_snapshot.get(h_id, [])
                h.observations = deque(snap, maxlen=h.observations.maxlen)
            self._total_updates_applied = pre_updates
            self._evidence_since_last_test = max(
                0, self._evidence_since_last_test - 1)
            logger.info("[MetaCGN] Boot self-test PASSED (4/4 checks)")
            return True
        except Exception as e:
            self._status = "disabled_boot_selftest_failed"
            import traceback
            logger.error("[MetaCGN] Boot self-test FAILED: %r\n%s",
                         e, traceback.format_exc())
            return False

    # ── P5: Failsafe watchdog ──────────────────────────────────────

    def _record_failure(self, kind: str, exception: Exception,
                        chain_id: int = 0) -> None:
        """Record a failure event. May trip the failsafe (I-5 severity-weighted
        + I-6 signature dedup). Safe to call from any exception handler."""
        try:
            self._total_failures += 1
            self._last_failure_ts = time.time()
            severity = self._severity_map.get(kind, 1)
            exc_msg = str(exception)[:50]
            # I-6: signature hash for dedup
            sig = f"{kind}::{type(exception).__name__}::{exc_msg}"

            # Append to window (rolling, auto-evicts old)
            record = {
                "ts": self._last_failure_ts, "chain_id": chain_id,
                "kind": kind, "signature": sig, "severity": severity,
            }
            self._failure_window.append(record)
            # Rebuild signature set from window (dedup across window)
            self._failure_signatures_in_window = {
                r["signature"] for r in self._failure_window
            }
            # I-7: structured failure log (bounded)
            self._append_failure_log(record, exception)

            # Compute severity sum — ONCE per unique signature in window
            severity_by_sig: dict[str, int] = {}
            for r in self._failure_window:
                severity_by_sig[r["signature"]] = max(
                    severity_by_sig.get(r["signature"], 0), r["severity"])
            severity_sum = sum(severity_by_sig.values())

            if severity_sum >= self._severity_trip_threshold and \
                    self._status != "disabled_failsafe":
                self._trip_failsafe(
                    reason=f"severity_sum={severity_sum} >= "
                           f"{self._severity_trip_threshold}",
                    window_signatures=list(severity_by_sig.keys()),
                )
        except Exception:
            # Recording a failure must never itself fail
            pass

    def _append_failure_log(self, record: dict, exc: Exception) -> None:
        """P5 I-7: append to data/meta_cgn/failure_log.jsonl (FIFO bounded)."""
        try:
            record = {**record, "exception_type": type(exc).__name__,
                      "exception_msg": str(exc)[:200]}
            # Rotate if >1000 lines (rough bound)
            if os.path.exists(self._failure_log_path):
                size = os.path.getsize(self._failure_log_path)
                if size > 500_000:  # ~1000 entries worth
                    # Keep last 500 entries
                    with open(self._failure_log_path) as f:
                        lines = f.readlines()[-500:]
                    with open(self._failure_log_path, "w") as f:
                        f.writelines(lines)
            with open(self._failure_log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            pass

    def _trip_failsafe(self, reason: str,
                       window_signatures: list) -> None:
        """Enter disabled_failsafe state with 1000-chain cooldown."""
        self._status = "disabled_failsafe"
        self._cooldown_remaining = 1000
        self._disabled_reason = reason
        self._failsafe_trip_count += 1
        logger.warning("[MetaCGN] FAILSAFE TRIPPED: %s (signatures=%d, "
                       "cooldown=1000 chains). All operations no-op until "
                       "cooldown expires.", reason, len(window_signatures))
        # Emit bus event
        if self._send_queue is not None:
            try:
                self._send_queue.put({
                    "type": "META_CGN_FAILED",
                    "src": self._module_name,
                    "dst": "all",
                    "ts": time.time(),
                    "payload": {
                        "reason": reason,
                        "trip_count": self._failsafe_trip_count,
                        "window_signatures": window_signatures,
                        "cooldown_chains": self._cooldown_remaining,
                    },
                })
            except Exception:
                pass
        self._save_watchdog_state()

    def _maybe_recover_from_failsafe(self) -> None:
        """Decrement cooldown each chain; re-enable in shadow mode when 0."""
        if self._status != "disabled_failsafe":
            return
        self._cooldown_remaining -= 1
        if self._cooldown_remaining <= 0:
            logger.info("[MetaCGN] Failsafe cooldown expired — re-enabling "
                        "in shadow mode")
            self._status = "shadow_mode"
            self._cooldown_remaining = 0
            self._failure_window.clear()
            self._failure_signatures_in_window.clear()
            self._consecutive_failures = 0
            self._save_watchdog_state()

    def reset_watchdog(self) -> None:
        """Maker manual reset — clear failures + reset to shadow mode."""
        logger.info("[MetaCGN] Watchdog manually reset by Maker")
        self._status = "shadow_mode"
        self._cooldown_remaining = 0
        self._consecutive_failures = 0
        self._failure_window.clear()
        self._failure_signatures_in_window.clear()
        self._disabled_reason = ""
        self._save_watchdog_state()

    def _save_watchdog_state(self) -> None:
        """P5 I-9: persist watchdog state across reboots."""
        try:
            data = {
                "version": 1,
                "saved_ts": time.time(),
                "status": self._status,
                "cooldown_remaining": self._cooldown_remaining,
                "disabled_reason": self._disabled_reason,
                "total_failures": self._total_failures,
                "failsafe_trip_count": self._failsafe_trip_count,
                "last_failure_ts": self._last_failure_ts,
                "window": list(self._failure_window),
                "graduation_progress": self._graduation_progress,
                "graduation_ts": self._graduation_ts,
                "rolled_back_count": self._rolled_back_count,
            }
            tmp = self._watchdog_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, self._watchdog_path)
        except Exception:
            pass

    def _load_watchdog_state(self) -> None:
        """P5 I-9: restore watchdog state — failsafe survives reboots."""
        try:
            if not os.path.exists(self._watchdog_path):
                return
            with open(self._watchdog_path) as f:
                data = json.load(f)
            self._status = data.get("status", "shadow_mode")
            self._cooldown_remaining = int(data.get("cooldown_remaining", 0))
            self._disabled_reason = data.get("disabled_reason", "")
            self._total_failures = int(data.get("total_failures", 0))
            self._failsafe_trip_count = int(data.get("failsafe_trip_count", 0))
            self._last_failure_ts = float(data.get("last_failure_ts", 0.0))
            for r in data.get("window", []):
                self._failure_window.append(r)
            self._failure_signatures_in_window = {
                r["signature"] for r in self._failure_window}
            self._graduation_progress = int(data.get("graduation_progress", 0))
            self._graduation_ts = float(data.get("graduation_ts", 0.0))
            self._rolled_back_count = int(data.get("rolled_back_count", 0))
            logger.info("[MetaCGN] Loaded watchdog state: status=%s, "
                        "cooldown=%d, total_failures=%d",
                        self._status, self._cooldown_remaining,
                        self._total_failures)
        except Exception as e:
            logger.warning("[MetaCGN] Watchdog state load failed: %s", e)

    # ── P4: Graduation state machine ───────────────────────────────

    def evaluate_graduation(self) -> None:
        """Check readiness + advance graduation progress each chain conclude.
        Called from the per-chain observe path. Non-destructive if already
        disabled_failsafe or rolled_back."""
        if self._status in ("disabled_failsafe",
                            "disabled_boot_selftest_failed"):
            self._maybe_recover_from_failsafe()
            return
        try:
            stats = self.get_stats()
            ready = stats.get("ready_to_graduate", False)
            # Transition: shadow_mode + ready → start ramping
            if self._status == "shadow_mode" and ready:
                self._enter_graduating()
            # Progress ramp
            if self._status == "graduating":
                self._graduation_progress = min(100,
                    self._graduation_progress + 1)
                if self._graduation_progress >= 100:
                    self._enter_active()
            # Rollback check for active mode
            if self._status == "active":
                self._chains_since_graduation += 1
                self._check_rollback_conditions()
        except Exception as e:
            self._record_failure("graduation_error", e)

    def _enter_graduating(self) -> None:
        """Shadow → graduating transition. Captures pre-graduation baseline."""
        logger.info("[MetaCGN] GRADUATION ramp started — shadow → graduating "
                    "(linear ramp over 100 chains)")
        self._status = "graduating"
        self._graduation_ts = time.time()
        self._graduation_progress = 0
        # Pre-graduation baseline snapshot for rollback comparison
        self._pre_graduation_baseline = {
            "success_rate_ema": self._compute_recent_success_rate(),
            "primitive_variances": {
                p: self._primitives[p].variance for p in self._primitives},
            "baseline_ts": time.time(),
        }
        # Emit bus event
        if self._send_queue is not None:
            try:
                self._send_queue.put({
                    "type": "META_CGN_GRADUATING",
                    "src": self._module_name,
                    "dst": "all",
                    "ts": time.time(),
                    "payload": {
                        "status": "graduating",
                        "ramp_chains": 100,
                        "baseline": self._pre_graduation_baseline,
                    },
                })
            except Exception:
                pass
        self._save_watchdog_state()

    def _enter_active(self) -> None:
        """Graduating → active transition. β now reranks chain_iql top-K."""
        logger.info("[MetaCGN] GRADUATION complete — graduating → active. "
                    "β will now rerank chain_iql top-K templates.")
        self._status = "active"
        self._chains_since_graduation = 0
        self._post_grad_outcomes.clear()
        # P4 D6: chain_archive record + TimeChain commit
        if self._send_queue is not None:
            try:
                self._send_queue.put({
                    "type": "META_CGN_ACTIVE",
                    "src": self._module_name,
                    "dst": "all",
                    "ts": time.time(),
                    "payload": {"status": "active",
                                "rolled_back_count": self._rolled_back_count},
                })
                # TimeChain: permanent marker of graduation
                self._send_queue.put({
                    "type": "TIMECHAIN_COMMIT",
                    "src": self._module_name,
                    "dst": "timechain",
                    "ts": time.time(),
                    "payload": {
                        "fork": "declarative",
                        "thought_type": "declarative",
                        "source": "meta_cgn_graduation",
                        "content": {
                            "event": "META-CGN graduated from shadow to "
                                     "active mode",
                            "chains_in_shadow": self._total_updates_applied,
                            "primitives_grounded": sum(
                                1 for p in self._primitives.values()
                                if p.confidence > 0.5),
                            "confirmed_hypotheses": sum(
                                1 for h in self._hypotheses.values()
                                if h.status == "confirmed"),
                        },
                    },
                })
            except Exception:
                pass
        self._save_watchdog_state()

    def _compute_recent_success_rate(self) -> float:
        """Rolling success rate proxy — uses pre_grad_outcomes if populated,
        otherwise falls back to primitive V average."""
        if self._pre_grad_outcomes:
            return sum(self._pre_grad_outcomes) / len(self._pre_grad_outcomes)
        Vs = [p.V for p in self._primitives.values() if p.n_samples > 0]
        return sum(Vs) / max(1, len(Vs))

    def _check_rollback_conditions(self) -> None:
        """P4 D3: auto-rollback if active mode is hurting. 3 conditions."""
        if self._chains_since_graduation < 50:
            return  # Give ramp time to settle
        try:
            # Condition 1: success rate drops ≥20% vs baseline
            baseline_rate = self._pre_graduation_baseline.get(
                "success_rate_ema", 0.5)
            current_rate = (sum(self._post_grad_outcomes) /
                            max(1, len(self._post_grad_outcomes)))
            if baseline_rate > 0 and current_rate < baseline_rate * 0.8:
                self._rollback_to_shadow(
                    reason=f"success_rate dropped {baseline_rate:.3f} → "
                           f"{current_rate:.3f} (≥20%)")
                return
            # Condition 2: variance spike
            baseline_vars = self._pre_graduation_baseline.get(
                "primitive_variances", {})
            for p_id, p in self._primitives.items():
                baseline_var = baseline_vars.get(p_id, 0.25)
                if baseline_var > 0.01 and p.variance > baseline_var * 2.0:
                    self._rollback_to_shadow(
                        reason=f"{p_id} variance spike: "
                               f"{baseline_var:.3f} → {p.variance:.3f} (>2×)")
                    return
        except Exception as e:
            self._record_failure("graduation_error", e)

    def _rollback_to_shadow(self, reason: str) -> None:
        """Active → shadow rollback. Preserves grounding state; resets ramp."""
        logger.warning("[MetaCGN] ROLLBACK — active → shadow: %s", reason)
        self._status = "rolled_back"
        self._rolled_back_count += 1
        self._graduation_progress = 0
        # Emit bus event
        if self._send_queue is not None:
            try:
                self._send_queue.put({
                    "type": "META_CGN_ROLLED_BACK",
                    "src": self._module_name,
                    "dst": "all",
                    "ts": time.time(),
                    "payload": {"reason": reason,
                                "count": self._rolled_back_count},
                })
            except Exception:
                pass
        # Return to shadow_mode after brief latched rolled_back state
        self._status = "shadow_mode"
        self._save_watchdog_state()

    def force_graduate(self) -> None:
        """Maker manual graduation override. Skips criteria check."""
        logger.info("[MetaCGN] Maker-forced graduation — bypassing criteria")
        self._enter_graduating()

    def force_shadow(self) -> None:
        """Maker manual rollback to shadow mode."""
        logger.info("[MetaCGN] Maker-forced rollback to shadow mode")
        self._rollback_to_shadow(reason="maker_manual_rollback")

    # ── P4 Improvement I-1: Graduation-blocker diagnostic ──────────

    def get_graduation_readiness(self) -> dict:
        """Detailed blockers view for Maker review via API/CLI."""
        primitives_well_sampled = sum(
            1 for p in self._primitives.values() if p.n_samples >= 50)
        confirmed = sum(1 for h in self._hypotheses.values()
                        if h.status == "confirmed")
        blockers = []
        under_sampled = [
            f"{p_id}(n={p.n_samples})" for p_id, p in self._primitives.items()
            if p.n_samples < 50
        ]
        if primitives_well_sampled < 5:
            blockers.append(
                f"primitives_well_sampled: {primitives_well_sampled}/5 "
                f"(need more samples for: {', '.join(under_sampled[:5])})")
        if self._total_updates_applied < 2000:
            blockers.append(
                f"total_updates: {self._total_updates_applied}/2000")
        if confirmed < 3:
            nascent_names = [h.hypothesis_id for h in self._hypotheses.values()
                             if h.status == "nascent"]
            blockers.append(
                f"confirmed_hypotheses: {confirmed}/3 "
                f"(nascent: {', '.join(nascent_names[:3])})")
        return {
            "status": self._status,
            "ready": len(blockers) == 0,
            "primitives_well_sampled": primitives_well_sampled,
            "confirmed_hypotheses": confirmed,
            "total_updates": self._total_updates_applied,
            "graduation_progress": self._graduation_progress,
            "blockers": blockers,
            "rolled_back_count": self._rolled_back_count,
        }

    # ── P4 Improvement I-2: Shadow-mode quality metric ─────────────

    def shadow_quality_metric(self) -> dict:
        """Disagreement rate health check — target 10-25%."""
        if self._total_compositions == 0:
            return {"disagreement_rate": 0.0, "compositions": 0,
                    "health": "no_data"}
        rate = self._total_disagreements / self._total_compositions
        if rate < 0.05:
            health = "too_low_beta_adds_little"
        elif rate > 0.40:
            health = "too_high_beta_overconfident"
        else:
            health = "healthy"
        return {
            "disagreement_rate": round(rate, 3),
            "compositions": self._total_compositions,
            "disagreements": self._total_disagreements,
            "health": health,
            "target_range": [0.10, 0.25],
        }

    # ── P4 Active-mode composition (called from chain_iql integration) ──

    def rerank_templates(self, candidates: list,
                         state_ctx: dict) -> tuple:
        """P4 D2: top-K reranking entrypoint for meta_reasoning.
        candidates: list of (template_string, chain_iql_Q_value).
        Returns: (best_template, blended_score, composition_info).

        In shadow_mode/graduating: logs disagreements but preserves top-1.
        In active: reranks by blended score (chain_iql Q + β composed V).
        In disabled_failsafe: returns top-1 unchanged (short-circuit)."""
        if self._status in ("disabled_failsafe",
                            "disabled_boot_selftest_failed") or not candidates:
            if candidates:
                return candidates[0][0], candidates[0][1], {"mode": "bypass"}
            return None, 0.0, {"mode": "bypass"}

        try:
            scored = []
            for template, q in candidates:
                primitives = self._template_primitives(template)
                result = self.compose_template_score(
                    template_id=template,
                    state_ctx=state_ctx,
                    chain_iql_score=float(q),
                    chain_iql_confidence=0.6,  # heuristic baseline
                    template_primitives=primitives,
                )
                scored.append((template, q, result))
            # P6 I1: β-dispersion — spread in composed_V across top-K candidates.
            # Leading indicator for β influence (fires before disagreements).
            if len(scored) >= 2:
                composed_vals = [r.get("composed_V", 0.5) for _, _, r in scored]
                dispersion = float(max(composed_vals) - min(composed_vals))
                a = self._beta_dispersion_alpha
                self._beta_dispersion_ema = (
                    (1 - a) * self._beta_dispersion_ema + a * dispersion)
                self._total_rerank_samples += 1
            # P7: stash max disagreement for this chain (state_ctx may carry
            # chain_id). observe_chain_evidence retrieves it for H6 obs.
            chain_id_for_cache = state_ctx.get("chain_id") if state_ctx else None
            if chain_id_for_cache is not None and scored:
                max_disagree = max(
                    abs(r.get("disagreement", 0.0)) for _, _, r in scored)
                self._pending_disagreement_by_chain[chain_id_for_cache] = \
                    float(max_disagree)
                # Bounded FIFO eviction
                if len(self._pending_disagreement_by_chain) > \
                        self._disagreement_cache_max:
                    # Drop oldest entry
                    oldest = next(iter(self._pending_disagreement_by_chain))
                    self._pending_disagreement_by_chain.pop(oldest, None)
            # Decide reranking by status
            if self._status == "shadow_mode":
                return scored[0][0], scored[0][1], {
                    "mode": "shadow", "all_scored": [
                        {"tmpl": t, "Q": q,
                         "composed_V": r.get("composed_V", 0.5)}
                        for t, q, r in scored]}
            if self._status in ("graduating", "active"):
                # Weight β contribution by graduation_progress (0..100 → 0..1)
                ramp = (100 if self._status == "active"
                        else self._graduation_progress) / 100.0
                reranked = []
                for tmpl, q, r in scored:
                    composed = r.get("composed_V", 0.5)
                    lam = r.get("lambda_used", 0.5)
                    # At ramp=0 → pure α; at ramp=1 → λ-weighted blend
                    final = q * (1 - ramp * (1 - lam)) + \
                            composed * ramp * (1 - lam)
                    reranked.append((tmpl, final, q, r))
                reranked.sort(key=lambda t: t[1], reverse=True)
                best = reranked[0]
                # Log when chosen != top-1 from chain_iql
                if best[0] != candidates[0][0]:
                    self._log_active_rerank(
                        chosen=best[0], chain_iql_top=candidates[0][0],
                        chosen_score=best[1], chain_iql_top_q=candidates[0][1],
                        ramp=ramp)
                return best[0], best[1], {
                    "mode": self._status, "ramp": ramp, "β_influenced": True}
            return candidates[0][0], candidates[0][1], {"mode": "pass"}
        except Exception as e:
            self._record_failure("composition_error", e)
            return candidates[0][0], candidates[0][1], {"mode": "error"}

    def _template_primitives(self, template: str) -> list:
        """Extract primitive names from arrow-joined template string."""
        if not template:
            return []
        parts = template.replace("→", "->").split("->")
        return [p.strip() for p in parts if p.strip() in PRIMITIVE_INDEX]

    def _log_active_rerank(self, chosen: str, chain_iql_top: str,
                           chosen_score: float, chain_iql_top_q: float,
                           ramp: float) -> None:
        """Log case where β overrode chain_iql's top pick."""
        try:
            with open(self._disagreements_log_path, "a") as f:
                f.write(json.dumps({
                    "ts": time.time(),
                    "event": "active_rerank_override",
                    "β_chose": chosen,
                    "chain_iql_top": chain_iql_top,
                    "β_score": round(chosen_score, 4),
                    "chain_iql_top_Q": round(chain_iql_top_q, 4),
                    "ramp": round(ramp, 3),
                }) + "\n")
        except Exception:
            pass

    # ── P4 Chain-outcome tracker (feeds rollback detector) ─────────

    def record_chain_outcome(self, chain_succeeded: bool) -> None:
        """Feed post-graduation rollback detector. Called per chain conclude.

        P6: also tick decay cadence counter (applies γ every N chains).
        P7: also tick monotonic chain_counter for conflict throttle cooldown.
        """
        outcome = 1.0 if chain_succeeded else 0.0
        if self._status == "active":
            self._post_grad_outcomes.append(outcome)
        else:
            self._pre_grad_outcomes.append(outcome)
        # P6 D2: tick decay cadence
        self._chains_since_decay = getattr(self, "_chains_since_decay", 0) + 1
        self._maybe_apply_decay()
        # P7: monotonic chain counter — used by conflict throttle cooldown check
        self._chain_counter += 1

    # ── P5 F8: Cognitive impasse detection ─────────────────────────

    def check_impasse(self) -> Optional[str]:
        """Four-signal impasse detector. Returns signal name or None."""
        try:
            # Snapshot per-chain state for flatline detection
            v_snap = {p: self._primitives[p].V for p in self._primitives}
            c_snap = {p: self._primitives[p].confidence
                      for p in self._primitives}
            self._impasse_v_history.append(v_snap)
            self._impasse_conf_history.append(c_snap)

            # Need full window to detect
            if len(self._impasse_v_history) < self._impasse_window_chains:
                return None
            if self._impasse_state != "healthy":
                return self._impasse_state  # latched; auto-resolves elsewhere

            # Signal 1: V-flatline — max |ΔV| across primitives over window
            first_v = self._impasse_v_history[0]
            last_v = self._impasse_v_history[-1]
            max_dv = max(abs(last_v.get(p, 0.5) - first_v.get(p, 0.5))
                         for p in PRIMITIVES)
            if max_dv < 0.02:
                return self._enter_impasse("v_flatline",
                    f"max|ΔV|={max_dv:.4f} over 500 chains")

            # Signal 2: Confidence-flatline
            first_c = self._impasse_conf_history[0]
            last_c = self._impasse_conf_history[-1]
            max_dc = max(abs(last_c.get(p, 0) - first_c.get(p, 0))
                         for p in PRIMITIVES)
            if max_dc < 0.05:
                return self._enter_impasse("conf_flatline",
                    f"max|Δconfidence|={max_dc:.4f} over 500 chains")

            # Signal 3: HAOV stagnation — all nascent + obs count flat
            all_nascent = all(h.status == "nascent"
                              for h in self._hypotheses.values())
            if all_nascent:
                total_obs = sum(len(h.observations)
                                for h in self._hypotheses.values())
                # Use a simple proxy: obs-per-chain rate should exceed 5
                obs_rate = total_obs / max(1, len(self._impasse_v_history))
                if obs_rate < 5:
                    return self._enter_impasse("haov_stagnant",
                        f"all 5 hypotheses nascent, obs_rate={obs_rate:.2f}")

            # Signal 4: Graduation-flatline
            blockers = tuple(self.get_graduation_readiness()["blockers"])
            bh = str(hash(blockers))
            if bh == self._last_graduation_blockers_hash:
                self._graduation_blockers_unchanged_chains += 1
            else:
                self._graduation_blockers_unchanged_chains = 0
                self._last_graduation_blockers_hash = bh
            if self._graduation_blockers_unchanged_chains >= 1000:
                return self._enter_impasse("graduation_flatline",
                    f"blockers unchanged for "
                    f"{self._graduation_blockers_unchanged_chains} chains")
            return None
        except Exception as e:
            self._record_failure("haov_error", e)
            return None

    def _enter_impasse(self, signal: str, diagnostic: str) -> str:
        """P8: Emit META_CGN_IMPASSE + multi-consumer CGN_KNOWLEDGE_REQ.

        Dedup (I-P8.1): one knowledge request per (signal, diagnostic_hash)
        per impasse cycle. Second entry with same signature emits META_CGN_IMPASSE
        but skips the knowledge request.
        """
        self._impasse_state = signal
        self._impasse_detected_ts = time.time()
        self._impasse_total_fires += 1
        # Conservative self-adjustment: 2× learning rate for 100 chains
        self._impasse_alpha_boost_remaining = 100
        # Lower HAOV test min_samples by 50% (temporary)
        for h in self._hypotheses.values():
            if h.status == "nascent":
                h.min_samples = max(10, h.min_samples // 2)
        logger.warning("[MetaCGN] IMPASSE detected: %s — %s. Gentle "
                       "self-adjust active for 100 chains.",
                       signal, diagnostic)
        # Build dedup signature: stable hash of (signal, diagnostic prefix)
        import hashlib
        dedup_key = hashlib.sha1(
            f"{signal}|{diagnostic[:200]}".encode()).hexdigest()[:12]
        dedup_seen = (self._current_impasse_req_signature == dedup_key)
        if not dedup_seen:
            self._current_impasse_req_signature = dedup_key
        # Emit bus events
        if self._send_queue is not None:
            try:
                self._send_queue.put({
                    "type": "META_CGN_IMPASSE",
                    "src": self._module_name,
                    "dst": "all",
                    "ts": time.time(),
                    "payload": {
                        "signal": signal,
                        "diagnostic": diagnostic,
                        "total_fires": self._impasse_total_fires,
                    },
                })
                if dedup_seen:
                    self._knowledge_requests_deduped += 1
                    logger.info("[MetaCGN] Dedup: skipping CGN_KNOWLEDGE_REQ "
                                "for same impasse signature %s", dedup_key)
                else:
                    # Multi-consumer broadcast (D8.1) with request_id (I-P8.2)
                    import uuid
                    request_id = uuid.uuid4().hex[:16]
                    current_domain = self._last_impasse_domain if \
                        hasattr(self, "_last_impasse_domain") else "general"
                    self._pending_knowledge_requests[request_id] = {
                        "start_ts": time.time(),
                        "impasse_signal": signal,
                        "impasse_diagnostic": diagnostic,
                        "domain": current_domain,
                        "responses": [],
                        "finalized": False,
                    }
                    self._knowledge_requests_emitted += 1
                    self._send_queue.put({
                        "type": "CGN_KNOWLEDGE_REQ",
                        "src": self._module_name,
                        "dst": "all",       # D8.1 broadcast
                        "ts": time.time(),
                        "payload": {
                            "request_id": request_id,
                            "consumer": CONSUMER_NAME,
                            "requestor": "meta_reasoning",
                            "query": f"reasoning patterns that improve "
                                     f"primitive grounding diversity ({signal})",
                            "context": {"impasse_signal": signal,
                                        "diagnostic": diagnostic,
                                        "domain": current_domain},
                        },
                    })
            except Exception as e:
                logger.debug("[MetaCGN] impasse emit failed: %s", e)
        return signal

    # ── P8: SOAR-via-CGN handlers ──────────────────────────────────────

    def handle_knowledge_response(self, response_payload: dict) -> Optional[dict]:
        """P8: Accumulate a CGN_KNOWLEDGE_RESP keyed by request_id.

        If this response closes the window (or aggregation condition met),
        returns the ranked winner for injection. Else returns None.
        """
        try:
            rid = str(response_payload.get("request_id", ""))
            if not rid or rid not in self._pending_knowledge_requests:
                return None
            req = self._pending_knowledge_requests[rid]
            if req.get("finalized"):
                return None
            # Track source credit (D8.5)
            src = str(response_payload.get("source", "unknown"))
            self._knowledge_provided_by_source[src] = \
                self._knowledge_provided_by_source.get(src, 0) + 1
            req["responses"].append(response_payload)
            self._knowledge_responses_received += 1
            # Check if window expired — finalize
            if (time.time() - req["start_ts"]) >= P8_RESPONSE_WINDOW_SECONDS:
                return self._finalize_knowledge_request(rid)
            return None
        except Exception as e:
            logger.debug("[MetaCGN] handle_knowledge_response failed: %s", e)
            return None

    def finalize_expired_requests(self) -> list:
        """P8: Called per chain (via record_chain_outcome). Closes any window
        older than P8_RESPONSE_WINDOW_SECONDS. Returns list of winners for
        injection by caller (meta_reasoning engine).
        """
        now = time.time()
        winners = []
        for rid in list(self._pending_knowledge_requests.keys()):
            req = self._pending_knowledge_requests[rid]
            if req.get("finalized"):
                continue
            if (now - req["start_ts"]) >= P8_RESPONSE_WINDOW_SECONDS:
                winner = self._finalize_knowledge_request(rid)
                if winner is not None:
                    winners.append(winner)
        return winners

    def _finalize_knowledge_request(self, request_id: str) -> Optional[dict]:
        """Rank accumulated responses via B-hybrid, return winner for injection."""
        req = self._pending_knowledge_requests.get(request_id)
        if not req or req.get("finalized"):
            return None
        req["finalized"] = True
        self._knowledge_requests_finalized += 1
        responses = req.get("responses", [])
        if not responses:
            self._knowledge_requests_empty += 1
            # Garbage collect empty
            self._pending_knowledge_requests.pop(request_id, None)
            return None
        # D8.3 B-hybrid ranking
        signal = req.get("impasse_signal", "")
        diag = req.get("impasse_diagnostic", "")
        domain = req.get("domain", "general")
        scored = [(_rank_hybrid(r, signal, diag, domain), r) for r in responses]
        scored.sort(key=lambda kv: kv[0], reverse=True)
        best_score, best = scored[0]
        logger.info("[MetaCGN] P8 finalized request %s: %d responses, "
                    "winner source=%s score=%.3f",
                    request_id[:8], len(responses),
                    best.get("source", "?"), best_score)
        # Attach ranking metadata for inject_knowledge's relevance use
        best["_rank_score"] = best_score
        best["_request_id"] = request_id
        # Drop pending entry (caller will inject)
        self._pending_knowledge_requests.pop(request_id, None)
        return best

    def mark_helpful(self, source: str) -> None:
        """D8.5: called when a knowledge response was successfully injected."""
        self._knowledge_helpful_by_source[source] = \
            self._knowledge_helpful_by_source.get(source, 0) + 1

    # ── P9: Q-i reward blending helpers ────────────────────────────────

    def compute_grounded_reward(self, primitives: list,
                                 domain: str = "general",
                                 high_disagreement_this_chain: bool = False,
                                 kappa_ci_reward: float = 0.3
                                 ) -> tuple:
        """P9: compute r_grounded for reward blending.

        Returns (r_grounded, per_primitive_debug).

        E1 (pessimistic CI):  r_grounded = V_eff − κ·avg(CI_width)
        E2 (H6 down-weight):  if H6 confirmed AND this chain had high α-β
                              disagreement, divide by H6 confidence_multiplier
                              (which is > 1 when H6 confirmed)

        Falsafe: if consumer disabled or primitives empty → return (0.5, [])
        to stay neutral in the blend.
        """
        if self._status in ("disabled_failsafe",
                            "disabled_boot_selftest_failed") or not primitives:
            return 0.5, []
        try:
            q_lo = COMPOSITION_DEFAULTS["ci_quantile_lo"]
            q_hi = COMPOSITION_DEFAULTS["ci_quantile_hi"]
            dom_thresh = int(COMPOSITION_DEFAULTS["domain_obs_threshold"])
            rows = []
            conf_sum = 0.0
            conf_weighted_V = 0.0
            ci_widths = []
            for p_id in primitives:
                p = self._primitives.get(p_id)
                if p is None:
                    continue
                a_d, b_d = p.alpha, p.beta
                used_domain = False
                if domain and domain in p.by_domain:
                    a_d_try, b_d_try, n_d = p.by_domain[domain]
                    if int(n_d) >= dom_thresh:
                        a_d, b_d = float(a_d_try), float(b_d_try)
                        used_domain = True
                V_eff = _beta_mean(a_d, b_d)
                lo, hi, ci_w = _beta_ci_width(a_d, b_d, q_lo, q_hi)
                conf = _posterior_confidence(a_d, b_d)
                w = max(0.05, conf)
                conf_sum += w
                conf_weighted_V += V_eff * w
                ci_widths.append(ci_w)
                rows.append({
                    "p": p_id, "V_eff": round(V_eff, 4),
                    "ci_width": round(ci_w, 4),
                    "domain_used": used_domain, "w": round(w, 4),
                })
            if conf_sum <= 1e-6:
                return 0.5, rows
            V_composed = conf_weighted_V / conf_sum
            avg_ci = sum(ci_widths) / max(1, len(ci_widths))
            # E1: pessimistic shift — rewards discount uncertainty
            r_grounded = V_composed - kappa_ci_reward * avg_ci
            # E2: HAOV-aware down-weight on disagreement chains
            h6 = self._hypotheses.get("H6_advisor_disagreement")
            if (h6 is not None and h6.status == "confirmed"
                    and high_disagreement_this_chain
                    and h6.confidence_multiplier > 1.0):
                r_grounded = r_grounded / h6.confidence_multiplier
            r_grounded = max(0.0, min(1.0, r_grounded))
            return float(r_grounded), rows
        except Exception as e:
            logger.debug("[MetaCGN] compute_grounded_reward failed: %s", e)
            return 0.5, []

    def compute_blend_weights(self, current_domain: str = "general"
                              ) -> tuple:
        """P9 D2 + E3 + E4: stage-driven (w_legacy, w_compound, w_grounded).

        Base weights from graduation state (matches rFP §7):
          shadow_mode     →  (0.5,  0.5,  0.0)   # Bootstrap
          graduating      →  (0.4,  0.4,  0.2)   # Calibration (ramp)
          active          →  (0.2,  0.3,  0.5)   # Mature
          rolled_back     →  shadow_mode weights (safe fallback)
          disabled_*      →  (0.5,  0.5,  0.0)   # D4 auto-zero

        E3: β-dispersion EMA secondary gate — if β is still silent (EMA<0.05),
            cap w_grounded at 0.05 regardless of stage.
        E4: per-domain bonus — if current_domain has well-grounded primitives
            (≥ 2 primitives with n_domain ≥ threshold), w_grounded × 1.15.
        Re-normalizes to sum=1 at the end.
        """
        # D4 safety: disabled states force w_grounded = 0
        if self._status.startswith("disabled_"):
            return 0.5, 0.5, 0.0
        stage = self._status
        if stage == "shadow_mode" or stage == "rolled_back":
            w_leg, w_comp, w_grd = 0.5, 0.5, 0.0
        elif stage == "graduating":
            # Linear ramp from (0.5/0.5/0.0) to (0.4/0.4/0.2) over 100 chains
            prog = max(0, min(100, int(self._graduation_progress))) / 100.0
            w_leg = 0.5 - 0.1 * prog
            w_comp = 0.5 - 0.1 * prog
            w_grd = 0.2 * prog
        elif stage == "active":
            w_leg, w_comp, w_grd = 0.2, 0.3, 0.5
        else:
            w_leg, w_comp, w_grd = 0.5, 0.5, 0.0
        # E3: β-dispersion secondary gate
        if w_grd > 0.05 and self._beta_dispersion_ema < 0.05:
            w_grd = 0.05
            # Compensate — boost legacy + compound proportionally
            w_leg += (w_leg / (w_leg + w_comp)) * (1 - w_leg - w_comp - w_grd) \
                if (w_leg + w_comp) > 1e-6 else 0
            w_comp = max(0.0, 1.0 - w_leg - w_grd)
        # E4: per-domain bonus
        if w_grd > 0 and current_domain:
            dom_thresh = int(COMPOSITION_DEFAULTS["domain_obs_threshold"])
            well_grounded_in_domain = sum(
                1 for p in self._primitives.values()
                if current_domain in p.by_domain
                and int(p.by_domain[current_domain][2]) >= dom_thresh)
            if well_grounded_in_domain >= 2:
                w_grd *= 1.15
        # Renormalize to sum=1
        total = w_leg + w_comp + w_grd
        if total > 1e-6:
            w_leg, w_comp, w_grd = w_leg / total, w_comp / total, w_grd / total
        return float(w_leg), float(w_comp), float(w_grd)

    # ── P11: Kin Protocol — cross-Titan grounding transfer ───────────

    def export_kin_snapshot(self) -> dict:
        """P11: return signable snapshot of grounding + HAOV state for
        Kin Protocol consumption by peer Titans. Includes schema_version so
        future kin protocol upgrades remain backward-compatible.

        Excludes Titan-private state: pending knowledge requests, watchdog
        window, signal tallies, conflict logs — all are per-Titan and not
        meaningful as priors.
        """
        try:
            primitives_out = {}
            for p_id, p in self._primitives.items():
                primitives_out[p_id] = {
                    "alpha": round(float(p.alpha), 4),
                    "beta": round(float(p.beta), 4),
                    "n_samples": int(p.n_samples),
                    "V": round(float(p.V), 4),
                    "confidence": round(float(p.confidence), 4),
                    "haov_rules": list(p.haov_rules),
                    "by_domain": {
                        d: [round(float(e[0]), 4), round(float(e[1]), 4),
                            int(e[2])]
                        for d, e in p.by_domain.items()
                    },
                }
            hypotheses_out = {}
            for h_id, h in self._hypotheses.items():
                hypotheses_out[h_id] = {
                    "status": h.status,
                    "effect_size": round(float(h.effect_size), 4),
                    "confidence_multiplier": round(
                        float(h.confidence_multiplier), 4),
                    "test_count": int(h.test_count),
                }
            return {
                "kin_protocol_version": 1,
                "schema": "meta_cgn_snapshot_v1",
                "titan_id": self._titan_id,
                "exported_ts": time.time(),
                "primitives": primitives_out,
                "hypotheses": hypotheses_out,
                "stats": self.get_stats_compact(),
            }
        except Exception as e:
            logger.warning("[MetaCGN] export_kin_snapshot failed: %s", e)
            return {}

    def import_kin_snapshot(self, snapshot: dict,
                            confidence_scale: float = 0.5) -> dict:
        """P11: import a peer Titan's kin snapshot as PRIORS (not overrides).

        Strategy (rFP §10):
        - For each primitive, scale the peer's (α,β) by `confidence_scale`
          (default 0.5 — priors are half-strength vs native evidence)
        - Merge by **weighted sum** into local posterior: a_local += scale·a_peer,
          b_local += scale·b_peer. Preserves monotonic growth.
        - Per-domain entries merged the same way.
        - Skip confirmed HAOV hypotheses — they're chain-specific evidence,
          not transferable priors.
        - Schema version check: refuse v != 1.

        Returns summary dict for logging/audit.
        """
        if self._status.startswith("disabled_"):
            return {"imported": False, "reason": "meta_cgn_disabled"}
        try:
            version = int(snapshot.get("kin_protocol_version", 0))
            if version != 1:
                return {"imported": False,
                        "reason": f"unsupported_version_{version}"}
            peer_titan = str(snapshot.get("titan_id", "unknown"))
            if peer_titan == self._titan_id:
                return {"imported": False, "reason": "same_titan_self_import"}
            imported_prims = 0
            imported_domains = 0
            scale = max(0.0, min(1.0, float(confidence_scale)))
            for p_id, p_peer in snapshot.get("primitives", {}).items():
                if p_id not in self._primitives:
                    continue
                p_local = self._primitives[p_id]
                a_peer = float(p_peer.get("alpha", 1.0)) - BETA_PARAM_FLOOR
                b_peer = float(p_peer.get("beta", 1.0)) - BETA_PARAM_FLOOR
                # Scaled prior contribution
                p_local.alpha = max(BETA_PARAM_FLOOR,
                                    p_local.alpha + scale * a_peer)
                p_local.beta = max(BETA_PARAM_FLOOR,
                                   p_local.beta + scale * b_peer)
                # Per-domain merge
                for d, e in p_peer.get("by_domain", {}).items():
                    a_d_peer = float(e[0]) - BETA_PARAM_FLOOR
                    b_d_peer = float(e[1]) - BETA_PARAM_FLOOR
                    entry = p_local.by_domain.get(d,
                                                   [BETA_PARAM_FLOOR,
                                                    BETA_PARAM_FLOOR, 0])
                    a_local = max(BETA_PARAM_FLOOR,
                                  float(entry[0]) + scale * a_d_peer)
                    b_local = max(BETA_PARAM_FLOOR,
                                  float(entry[1]) + scale * b_d_peer)
                    n_local = int(entry[2])   # peer n NOT imported — native only
                    p_local.by_domain[d] = [a_local, b_local, n_local]
                    imported_domains += 1
                p_local.recompute_derived()
                imported_prims += 1
            logger.info("[MetaCGN] Imported kin snapshot from %s: %d prims, "
                        "%d domains (confidence_scale=%.2f)",
                        peer_titan, imported_prims, imported_domains, scale)
            return {
                "imported": True,
                "peer_titan": peer_titan,
                "primitives_imported": imported_prims,
                "domains_merged": imported_domains,
                "confidence_scale": scale,
            }
        except Exception as e:
            logger.warning("[MetaCGN] import_kin_snapshot failed: %s", e)
            return {"imported": False, "reason": f"error: {e}"}

    # ── P10: Cross-consumer signal flow handler ───────────────────────

    def handle_cross_consumer_signal(self, consumer: str, event_type: str,
                                      intensity: float = 1.0,
                                      domain: Optional[str] = None,
                                      narrative_context: Optional[dict] = None
                                      ) -> bool:
        """P10 Layer 1: apply a cross-consumer signal as a pseudo-observation.

        Incoming `META_CGN_SIGNAL {consumer, event_type, intensity, domain,
        narrative_context}` → look up affected primitives in SIGNAL_TO_PRIMITIVE,
        apply a tiny Beta update with weight = P10_SIGNAL_WEIGHT · intensity.

        Returns True if applied, False if signal unknown (no mapping).

        Layer 2 (narrative bridge) HOOK: if intensity ≥ threshold AND
        narrative_context present, future code will trigger DuckDB recall +
        reflection chain + writeback. v1 logs a counter; actual bridge is a
        separate rFP (narrative meta-reasoning).
        """
        if self._status in ("disabled_failsafe",
                            "disabled_boot_selftest_failed"):
            return False
        try:
            self._signals_received += 1
            self._signals_by_consumer[consumer] = \
                self._signals_by_consumer.get(consumer, 0) + 1
            mapping = SIGNAL_TO_PRIMITIVE.get((consumer, event_type))
            if not mapping:
                self._signals_rejected_unknown += 1
                # Surface the orphan via BusHealthMonitor — logs WARN on
                # first occurrence per tuple, then silent. Prevents the
                # 2026-04-14 failure mode where 5 Phase 2 producers emitted
                # signals the consumer silently discarded for hours.
                try:
                    from ..core.bus_health import get_global_monitor
                    _m = get_global_monitor()
                    if _m is not None:
                        _m.record_orphan(consumer, event_type)
                except Exception:
                    pass
                return False
            # Apply pseudo-observations
            intensity = max(0.0, min(1.0, float(intensity)))
            effective_weight = P10_SIGNAL_WEIGHT * intensity
            for prim_id, quality_nudge in mapping.items():
                self.update_primitive_V(
                    primitive_id=prim_id,
                    quality=float(quality_nudge),
                    chain_id=-2,       # sentinel: signal-originated update
                    weight=effective_weight,
                    domain=domain,
                )
            self._signals_applied += 1
            # Layer 2 narrative bridge hook (stub in v1)
            if (intensity >= P10_NARRATIVE_TRIGGER_INTENSITY
                    and narrative_context):
                self._narrative_hooks_deferred += 1
                # Placeholder — future standalone rFP wires DuckDB recall here
                logger.debug("[MetaCGN] Narrative hook deferred: consumer=%s "
                             "event=%s intensity=%.2f (will be handled by "
                             "narrative meta-reasoning rFP)",
                             consumer, event_type, intensity)
            return True
        except Exception as e:
            logger.debug("[MetaCGN] cross-consumer signal failed: %s", e)
            return False

    def _blend_weights_preview_dict(self) -> dict:
        """P9: expose current-stage blend weights for dashboards."""
        try:
            w_leg, w_comp, w_grd = self.compute_blend_weights("general")
            return {
                "w_legacy": round(w_leg, 4),
                "w_compound": round(w_comp, 4),
                "w_grounded": round(w_grd, 4),
                "stage": self._status,
                "beta_dispersion_ema": round(self._beta_dispersion_ema, 4),
            }
        except Exception:
            return {"w_legacy": 0.5, "w_compound": 0.5, "w_grounded": 0.0}

    def log_blend_weights(self, chain_id: int, domain: str,
                           w_leg: float, w_comp: float, w_grd: float,
                           r_leg: float, r_comp: float, r_grd: float,
                           terminal: float) -> None:
        """E5: audit trail for reward blending — lets us chart stage
        transitions + see when w_grounded actually drives reward."""
        try:
            path = os.path.join(self._save_dir, "blend_weights_history.jsonl")
            # Rotate if >500KB (~5000 entries)
            if os.path.exists(path) and os.path.getsize(path) > 500_000:
                with open(path) as f:
                    lines = f.readlines()[-2500:]
                with open(path, "w") as f:
                    f.writelines(lines)
            with open(path, "a") as f:
                f.write(json.dumps({
                    "ts": time.time(),
                    "chain_id": chain_id,
                    "domain": domain,
                    "status": self._status,
                    "w_legacy": round(w_leg, 4),
                    "w_compound": round(w_comp, 4),
                    "w_grounded": round(w_grd, 4),
                    "r_legacy": round(r_leg, 4),
                    "r_compound": round(r_comp, 4),
                    "r_grounded": round(r_grd, 4),
                    "terminal": round(terminal, 4),
                    "beta_dispersion_ema": round(self._beta_dispersion_ema, 4),
                }) + "\n")
        except Exception:
            pass

    def handle_knowledge_request(self, request_payload: dict) -> Optional[dict]:
        """D8.4: META-CGN as responder. Called when another consumer emits
        CGN_KNOWLEDGE_REQ with dst="all" or dst="meta".

        Responds with primitive grounding summary — top-3 primitives by V_eff
        in the requested domain (falls back to pooled V).
        """
        try:
            rid = str(request_payload.get("request_id", ""))
            requestor = str(request_payload.get("requestor", ""))
            if requestor in ("meta_reasoning", "meta"):
                # Don't respond to our own requests
                return None
            ctx = request_payload.get("context", {}) or {}
            domain = str(ctx.get("domain", "general"))
            # Build top-3 by V summary
            rows = []
            for p_id, p in self._primitives.items():
                a_d, b_d = p.alpha, p.beta
                used_domain = False
                if domain and domain in p.by_domain:
                    a_d_try, b_d_try, n_d = p.by_domain[domain]
                    if int(n_d) >= COMPOSITION_DEFAULTS["domain_obs_threshold"]:
                        a_d, b_d = float(a_d_try), float(b_d_try)
                        used_domain = True
                v = _beta_mean(a_d, b_d)
                rows.append((p_id, v, used_domain))
            rows.sort(key=lambda t: t[1], reverse=True)
            top3 = rows[:3]
            avg_conf = sum(self._primitives[p].confidence for p, _, _
                           in top3) / max(1, len(top3))
            summary = "Top primitives by grounded V in domain '{}': {}".format(
                domain,
                ", ".join(f"{p}={v:.3f}" for p, v, _ in top3))
            response = {
                "request_id": rid,
                "topic": str(request_payload.get("query", ""))[:120],
                "confidence": round(avg_conf, 4),
                "summary": summary,
                "source": "meta_cgn_grounding",
                "domain": domain,
                "primitives": [p for p, _, _ in top3],
            }
            self._knowledge_responses_sent += 1
            return response
        except Exception as e:
            logger.debug("[MetaCGN] handle_knowledge_request failed: %s", e)
            return None

    def maybe_exit_impasse(self) -> None:
        """Auto-resolve impasse when signals normalize. Called per chain."""
        if self._impasse_state == "healthy":
            return
        # Decrement α boost counter
        if self._impasse_alpha_boost_remaining > 0:
            self._impasse_alpha_boost_remaining -= 1
        # Check if signal has normalized — require at least 200 chains of
        # data after entering impasse before re-evaluating
        elapsed = (time.time() - self._impasse_detected_ts)
        if elapsed < 60:  # cooldown ~1 min; avoids flapping
            return
        # Conservative: re-evaluate using the same signal logic
        # If signal no longer triggers, we're healed
        if self._impasse_alpha_boost_remaining == 0:
            # Enough time passed; check if we should exit
            logger.info("[MetaCGN] Impasse α-boost window expired — "
                        "returning to healthy")
            self._impasse_state = "healthy"
            # P8 I-P8.1: clear dedup signature so next impasse cycle can emit
            self._current_impasse_req_signature = None
            # Restore original HAOV min_samples
            for h in self._hypotheses.values():
                if h.status == "nascent" and h.min_samples < 30:
                    h.min_samples = 30

    def get_impasse_status(self) -> dict:
        """Telemetry for /v4/meta-cgn/impasse-status."""
        return {
            "state": self._impasse_state,
            "total_fires": self._impasse_total_fires,
            "detected_ts": self._impasse_detected_ts,
            "alpha_boost_remaining": self._impasse_alpha_boost_remaining,
            "v_history_depth": len(self._impasse_v_history),
            "graduation_blockers_unchanged_chains":
                self._graduation_blockers_unchanged_chains,
        }

    def get_failsafe_status(self) -> dict:
        """Telemetry for /v4/meta-cgn/failsafe-status."""
        severity_sum = 0
        by_sig: dict[str, int] = {}
        for r in self._failure_window:
            by_sig[r["signature"]] = max(by_sig.get(r["signature"], 0),
                                         r["severity"])
        severity_sum = sum(by_sig.values())
        return {
            "status": self._status,
            "total_failures": self._total_failures,
            "failsafe_trip_count": self._failsafe_trip_count,
            "cooldown_remaining": self._cooldown_remaining,
            "disabled_reason": self._disabled_reason,
            "window_size": len(self._failure_window),
            "unique_signatures_in_window": len(by_sig),
            "severity_sum_in_window": severity_sum,
            "severity_trip_threshold": self._severity_trip_threshold,
            "last_failure_ts": self._last_failure_ts,
        }

    def get_disagreements(self, last_n: int = 50) -> list:
        """Recent advisor disagreements — for CLI + API."""
        try:
            if not os.path.exists(self._disagreements_log_path):
                return []
            with open(self._disagreements_log_path) as f:
                lines = f.readlines()[-last_n:]
            return [json.loads(l) for l in lines if l.strip()]
        except Exception:
            return []

    # ── Telemetry ──────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Rich stats for /v4/meta-reasoning/audit meta_cgn block."""
        primitives_grounded = sum(
            1 for p in self._primitives.values() if p.confidence > 0.5)
        q_lo = COMPOSITION_DEFAULTS["ci_quantile_lo"]
        q_hi = COMPOSITION_DEFAULTS["ci_quantile_hi"]
        primitive_summary = {}
        # P6: expose α,β, CI, and per-domain sample counts
        n_share_vals = []
        for p_id, c in self._primitives.items():
            lo, hi, ci_w = _beta_ci_width(c.alpha, c.beta, q_lo, q_hi)
            n_share_vals.append(max(0, c.n_samples))
            primitive_summary[p_id] = {
                "V": round(c.V, 4),
                "confidence": round(c.confidence, 4),
                "n": c.n_samples,
                "variance": round(c.variance, 4),
                "alpha": round(float(c.alpha), 3),
                "beta": round(float(c.beta), 3),
                "ci_lo": round(lo, 4),
                "ci_hi": round(hi, 4),
                "ci_width": round(ci_w, 4),
                "domains_tracked": sorted(c.by_domain.keys()),
            }
        # I2: usage shares + Gini coefficient (time-series evidence for healing)
        n_total = sum(n_share_vals) or 1
        usage_shares = {
            p_id: round(self._primitives[p_id].n_samples / n_total, 4)
            for p_id in self._primitives
        }
        usage_gini = round(_gini(n_share_vals), 4)
        # Phase 2 graduation tightening: require per-primitive minimum sample
        # count + confirmed hypotheses, not just aggregate update count.
        primitives_well_sampled = sum(
            1 for p in self._primitives.values() if p.n_samples >= 50)
        confirmed_hypotheses = sum(
            1 for h in self._hypotheses.values() if h.status == "confirmed")
        return {
            "status": self._status,
            "registered": self._registered,
            "consumer_name": CONSUMER_NAME,
            "feature_dims": FEATURE_DIMS,
            "action_dims": ACTION_DIMS,
            "primitives_total": NUM_PRIMITIVES,
            "primitives_grounded": primitives_grounded,
            "primitives_well_sampled": primitives_well_sampled,
            "primitive_V_summary": primitive_summary,
            "transitions_sent": self._total_transitions_sent,
            "updates_applied": self._total_updates_applied,
            "compositions_computed": self._total_compositions,
            "disagreements_logged": self._total_disagreements,
            "uptime_seconds": int(time.time() - self._start_ts),
            "haov": self.get_haov_stats(),
            # P4: graduation + rollback state
            "graduation": {
                "status": self._status,
                "progress": self._graduation_progress,
                "rolled_back_count": self._rolled_back_count,
                "chains_since_graduation": self._chains_since_graduation,
                "graduation_ts": self._graduation_ts,
            },
            # P4 I-2: shadow-mode quality metric
            "shadow_quality": self.shadow_quality_metric(),
            # P5: failsafe watchdog state
            "failsafe": self.get_failsafe_status(),
            # P5 F8: impasse detection
            "impasse": self.get_impasse_status(),
            "ready_to_graduate": (
                primitives_well_sampled >= 5 and
                self._total_updates_applied >= 2000 and
                confirmed_hypotheses >= 3 and
                self._status == "shadow_mode"
            ),
            # ── P6: β-influence + usage + domain telemetry ──
            "beta_score_dispersion_ema": round(self._beta_dispersion_ema, 4),
            "rerank_samples": self._total_rerank_samples,
            "usage_shares": usage_shares,
            "usage_gini": usage_gini,
            "chains_since_decay": getattr(self, "_chains_since_decay", 0),
            "domain_hits": self._domain_hits,
            "domain_fallbacks": self._domain_fallbacks,
            # ── P7: EUREKA accelerator + advisor-conflict telemetry ──
            "eureka_accelerated_updates": self._eureka_accelerated_updates,
            "eureka_trigger_counts": dict(self._eureka_trigger_counts),
            "conflict_bus_events_emitted": self._conflict_bus_events_emitted,
            "conflict_sigs_throttled": self._conflict_sigs_throttled,
            "chain_counter": self._chain_counter,
            # ── P9: reward blending stage preview ──
            "blend_weights_preview": self._blend_weights_preview_dict(),
            # ── P10: cross-consumer signal telemetry ──
            "signals_received": self._signals_received,
            "signals_applied": self._signals_applied,
            "signals_rejected_unknown": self._signals_rejected_unknown,
            "signals_by_consumer": dict(self._signals_by_consumer),
            "narrative_hooks_deferred": self._narrative_hooks_deferred,
            # ── P8: SOAR-via-CGN full protocol telemetry ──
            "knowledge_requests_emitted": self._knowledge_requests_emitted,
            "knowledge_requests_deduped": self._knowledge_requests_deduped,
            "knowledge_requests_finalized": self._knowledge_requests_finalized,
            "knowledge_requests_empty": self._knowledge_requests_empty,
            "knowledge_responses_received": self._knowledge_responses_received,
            "knowledge_responses_sent": self._knowledge_responses_sent,
            "knowledge_pending": len(self._pending_knowledge_requests),
            "knowledge_provided_by_source": dict(
                self._knowledge_provided_by_source),
            "knowledge_helpful_by_source": dict(
                self._knowledge_helpful_by_source),
        }

    def get_stats_compact(self) -> dict:
        """Small stats dict for saving with state."""
        return {
            "transitions_sent": self._total_transitions_sent,
            "updates_applied": self._total_updates_applied,
            "compositions_computed": self._total_compositions,
            "disagreements_logged": self._total_disagreements,
        }


def _clip01(x: float) -> float:
    """Clip to [0, 1] with safety for NaN/inf."""
    try:
        if not np.isfinite(x):
            return 0.0
        return float(max(0.0, min(1.0, x)))
    except Exception:
        return 0.0
