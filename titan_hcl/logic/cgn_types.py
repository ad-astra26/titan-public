"""
titan_hcl.logic.cgn_types — pure data classes for CGN (no torch dependency).

D-SPEC-78 (SPEC v1.20.0): extracted from `cgn.py` 2026-05-18 to break the
implicit torch dependency for callers that only need the data structures
(arc/session.py grounding, haov_causal_generator type hints). `cgn.py`
re-exports these for back-compat; new callers should import from here
directly to avoid the 100MB libtorch + libtriton mmap cost.

The neural-net classes (SharedValueNet, ConsumerActionNet,
TransitionBuffer, ConceptGroundingNetwork) remain in `cgn.py` — they're
torch nn.Module subclasses and require torch loaded at definition time.
Only `cgn_worker` and `sage` should import from `cgn` directly.

See: titan-docs/rFP_chat_streaming_safety_first_ovg_async.md §9.1 Chunk α.
"""

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("titan.cgn_types")


# ── CGN-felt RFP: shared neuromod-felt helpers (torch-free) ──────────────────
# Imported by BOTH cgn.py (Phase A — materialize the per-concept felt centroid)
# and synthesis/consolidation.py (Phase C — compare lived felt vs the centroid),
# so the canonical key-space + EMA + distance live in ONE place and cannot drift.
# consolidation cannot import torch-heavy cgn.py, which is why these live here.

# Default EMA recency weight for the per-concept felt centroid (config-tunable).
FELT_EMA_ALPHA: float = 0.3
# Default frame-divergence threshold — RMS felt_distance ABOVE this = a new felt
# frame (frame_dependent), not redundant grounding (config-tunable, RFP §5).
FELT_FRAME_DIVERGENCE: float = 0.15
# The neuromod homeostatic centre (neutral setpoint); a key missing on one side
# compares as neutral rather than 0 (matches cgn._build_state_vector default 0.5).
_NEUROMOD_NEUTRAL: float = 0.5
# felt-dict keys that are metadata, NOT neuromod levels (matches consolidation.py).
_FELT_META_KEYS = frozenset({"emotion", "emotion_confidence", "dream_cycle", "ts"})


def normalize_neuromods(felt: Dict[str, Any]) -> Dict[str, float]:
    """Canonicalize a felt/neuromod dict to ONE comparable key-space (RFP §1.2).

    CGN reads ``DA / 5-HT|5HT / NE / GABA / ACh`` while the lived ``neuromod_context``
    uses ``DA / 5HT / NE / ACh / Endorphin / GABA`` (and may nest ``{level, setpoint}``).
    This collapses ``"5-HT" → "5HT"``, flattens a nested ``{"level": x}`` to ``x``,
    drops metadata keys, and keeps every remaining numeric neuromod level. It does NOT
    fabricate missing keys (no neutral floor — emergence over a hardcode). Non-dict /
    empty → ``{}``.
    """
    if not isinstance(felt, dict):
        return {}
    out: Dict[str, float] = {}
    for k, v in felt.items():
        if k in _FELT_META_KEYS:
            continue
        if isinstance(v, dict):
            v = v.get("level")
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        out["5HT" if k == "5-HT" else k] = float(v)
    return out


def felt_ema(prev: Optional[Dict[str, float]], new: Dict[str, float],
             alpha: float = FELT_EMA_ALPHA) -> Dict[str, float]:
    """Per-key exponential moving average of a felt centroid (RFP §1.2 MATERIALIZE).

    First observation → a copy of ``new``; thereafter ``α·new + (1-α)·prev`` for each
    key in ``new``, preserving any prior-only keys. ``new`` is assumed already
    normalized (caller passes ``normalize_neuromods(...)``). Bounded — no growth.
    """
    if not new:
        return dict(prev or {})
    if not prev:
        return dict(new)
    out = dict(prev)
    for k, v in new.items():
        base = prev.get(k)
        out[k] = float(v) if base is None else float(alpha * v + (1.0 - alpha) * base)
    return out


def felt_distance(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    """Normalized felt divergence ∈ [0,1] between two felt-states (RFP §1.2 / §5).

    Both sides are canonicalized via :func:`normalize_neuromods`; the distance is the
    RMS per-key gap over the UNION of present keys, a missing key counting as the
    neutral setpoint (0.5). RMS — not cosine: felt vectors sit near 0.5 with all-positive
    components, so cosine is ≈1 for any pair and blind to the magnitude difference that
    *defines* a felt frame. Empty either side → 0.0 (absence of signal is not a conflict).
    """
    ca = normalize_neuromods(a)
    cb = normalize_neuromods(b)
    if not ca or not cb:
        return 0.0
    keys = set(ca) | set(cb)
    sq = 0.0
    for k in keys:
        d = ca.get(k, _NEUROMOD_NEUTRAL) - cb.get(k, _NEUROMOD_NEUTRAL)
        sq += d * d
    return min(1.0, math.sqrt(sq / len(keys)))

# ── Data Classes ────────────────────────��───────────────────────────────────


@dataclass
class ConceptFeatures:
    """Domain-agnostic concept representation. CGN never sees 'word' or 'pattern'."""
    concept_id: str
    embedding: np.ndarray            # multi-modal tensor (130D for words)
    confidence: float = 0.0
    encounter_count: int = 0
    production_count: int = 0
    context_history: List[dict] = field(default_factory=list)
    associations: Dict[str, float] = field(default_factory=dict)
    age_epochs: int = 0
    cross_modal_conf: float = 0.0
    meaning_contexts: List[dict] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SensoryContext:
    """Captured at the moment of concept encounter."""
    epoch: int = 0
    msl_attention: Optional[np.ndarray] = None   # 51D
    state_132d: Optional[np.ndarray] = None
    neuromods: Dict[str, float] = field(default_factory=dict)
    concept_confidences: Dict[str, float] = field(default_factory=dict)
    encounter_type: str = "comprehension"         # comprehension|production|teaching|reasoning


@dataclass
class GroundingAction:
    """Output of ground() — what to do about a concept."""
    action_index: int = 0
    action_name: str = "reinforce"
    confidence_delta: float = 0.0
    tensor_plasticity: float = 0.0
    association_deltas: Dict[str, float] = field(default_factory=dict)
    context_weight: float = 0.5


@dataclass
class CGNConsumerConfig:
    """Registration config for a cognitive consumer."""
    name: str
    feature_dims: int = 30
    action_dims: int = 8
    action_names: List[str] = field(default_factory=lambda: [
        "reinforce", "explore", "differentiate", "consolidate",
        "associate", "dissociate", "deepen", "stabilize",
    ])
    reward_source: str = "vocabulary_growth_rate"
    max_buffer_size: int = 1000
    consolidation_priority: int = 1


@dataclass
class CGNTransition:
    """Single experience for IQL training."""
    consumer: str
    concept_id: str
    state: np.ndarray          # 30D sensory context summary
    action: int
    action_params: np.ndarray  # continuous action parameters
    reward: float = 0.0
    terminal: bool = False
    timestamp: float = 0.0
    epoch: int = 0
    metadata: dict = field(default_factory=dict)
    _consolidated: bool = False


# ── Generalized HAOV Engine (Phase A2 — CGN-level cognitive primitive) ──────


@dataclass
class GeneralizedHypothesis:
    """A rule hypothesis that works across any CGN consumer."""
    rule: str                      # e.g. "action_3_increases_order" or "warm_means_comfortable"
    consumer: str                  # which consumer owns this hypothesis
    action_context: dict           # consumer-specific: {action: 3} or {word: "warm"}
    predicted_effect: str          # "order_up", "confidence_gain", "quality_improve", etc.
    predicted_magnitude: float     # expected effect size (0-1)
    confidence: float = 0.1       # grows with verification, decays with falsification
    tests: int = 0
    confirmations: int = 0
    falsifications: int = 0
    source: str = ""              # what triggered this: "pattern", "impasse", "cross_insight"


class GeneralizedHAOVTracker:
    """Hypothesis-Act-Observe-Verify tracker — works with any CGN consumer.

    Each consumer registers a domain-specific verifier function:
      ARC: checks grid feature changes (order, entropy, movement)
      Language: checks word confidence change, production success
      Social: checks quality score, neuromod delta

    The tracker manages hypotheses, selects tests, and verifies outcomes.
    Verified rules (>0.6 confidence) are used for action selection and
    grounded in CGN as high-confidence concepts.
    """

    def __init__(self, consumer: str, max_hypotheses: int = 20,
                 config: dict = None):
        self._consumer = consumer
        _cfg = config or {}
        self._max = _cfg.get("max_hypotheses", max_hypotheses)
        self._confirmation_boost = _cfg.get("confirmation_boost", 0.15)
        self._falsification_decay = _cfg.get("falsification_decay", 0.6)
        self._verification_threshold = _cfg.get("verification_threshold", 0.6)
        self._min_magnitude = _cfg.get("min_magnitude", 0.02)
        self._test_probability = _cfg.get("test_probability", 0.25)
        self._hypotheses: List[GeneralizedHypothesis] = []
        self._active_test: Optional[dict] = None
        self._verified_rules: List[GeneralizedHypothesis] = []
        self._verify_fn: Optional[callable] = None
        self._stats = {
            "formed": 0, "tested": 0, "confirmed": 0,
            "falsified": 0, "used_for_action": 0,
        }

    def set_verifier(self, fn: callable) -> None:
        """Register verification function.

        fn(obs_before: dict, obs_after: dict, action_ctx: dict) → (confirmed: bool, error: float)
        """
        self._verify_fn = fn

    def hypothesize(self, action_context: dict,
                    observation: dict) -> Optional[GeneralizedHypothesis]:
        """Form a hypothesis from observed pattern or state.

        Args:
            action_context: consumer-specific action info (e.g., {action: 3} for ARC)
            observation: dict with keys like 'effect', 'magnitude', 'source_pattern', 'rule_name'

        Returns:
            New or existing hypothesis, or None if nothing worth hypothesizing.
        """
        effect = observation.get("effect", "unknown")
        magnitude = observation.get("magnitude", 0.0)
        rule_name = observation.get("rule_name", f"{self._consumer}_{effect}")
        source = observation.get("source", "observation")

        if magnitude < self._min_magnitude:
            return None

        # Check if already tracking this hypothesis
        for h in self._hypotheses:
            if h.rule == rule_name:
                return h

        h = GeneralizedHypothesis(
            rule=rule_name,
            consumer=self._consumer,
            action_context=action_context,
            predicted_effect=effect,
            predicted_magnitude=magnitude,
            source=source,
        )
        self._hypotheses.append(h)
        self._stats["formed"] += 1

        # Evict lowest-confidence if over limit
        if len(self._hypotheses) > self._max:
            self._hypotheses.sort(key=lambda x: x.confidence)
            self._hypotheses.pop(0)

        return h

    def hypothesize_from_impasse(self, impasse: dict) -> Optional[GeneralizedHypothesis]:
        """SOAR bridge: generate hypothesis targeting stuck area."""
        imp_type = impasse.get("type", "stuck")
        severity = impasse.get("severity", 0.5)

        return self.hypothesize(
            action_context={"impasse_type": imp_type},
            observation={
                "effect": f"resolve_{imp_type}",
                "magnitude": severity,
                "rule_name": f"{self._consumer}_impasse_{imp_type}",
                "source": "soar_impasse",
            },
        )

    def select_test(self, available_context: dict) -> Optional[dict]:
        """Select a hypothesis to test. Returns action context or None."""
        if self._active_test:
            return None

        # Probabilistic gate: explore (test hypothesis) vs exploit (follow policy)
        import random
        if random.random() > self._test_probability:
            return None

        available_actions = available_context.get("available_actions", [])

        # Find testable hypotheses
        testable = [h for h in self._hypotheses
                    if 0.05 <= h.confidence <= 0.7
                    and h.tests < 10]

        # Filter to available actions if applicable
        if available_actions:
            testable = [h for h in testable
                        if h.action_context.get("action") in available_actions
                        or "action" not in h.action_context]

        if not testable:
            return None

        # Prefer high-magnitude, low-test-count
        testable.sort(key=lambda h: h.predicted_magnitude / max(1, h.tests + 1),
                      reverse=True)
        target = testable[0]

        self._active_test = {
            "hypothesis": target,
            "pre_observation": available_context.get("observation", {}),
        }
        self._stats["tested"] += 1
        target.tests += 1

        return target.action_context

    def verify(self, action_context: dict, obs_after: dict,
               reward: float) -> Optional[dict]:
        """Verify hypothesis using registered verifier or reward-based fallback."""
        if not self._active_test:
            return None

        test = self._active_test
        h = test["hypothesis"]
        self._active_test = None

        # Use registered verifier if available
        confirmed = False
        prediction_error = 0.0

        if self._verify_fn:
            try:
                confirmed, prediction_error = self._verify_fn(
                    test["pre_observation"], obs_after, action_context)
            except Exception:
                # Fallback: reward-based verification
                confirmed = reward > 0.05
                prediction_error = abs(h.predicted_magnitude - reward)
        else:
            # Fallback: positive reward = confirmed
            confirmed = reward > 0.05
            prediction_error = abs(h.predicted_magnitude - reward)

        # Update hypothesis confidence (configurable dynamics)
        if confirmed:
            h.confirmations += 1
            h.confidence = min(0.95, h.confidence + self._confirmation_boost * (1.0 - h.confidence))
            self._stats["confirmed"] += 1
            if h.confidence > self._verification_threshold and h not in self._verified_rules:
                self._verified_rules.append(h)
        else:
            h.falsifications += 1
            h.confidence = max(0.01, h.confidence * self._falsification_decay)
            self._stats["falsified"] += 1

        return {
            "hypothesis": h.rule,
            "confirmed": confirmed,
            "confidence": h.confidence,
            "prediction_error": prediction_error,
            "tests": h.tests,
        }

    def suggest(self, available_context: dict) -> Optional[dict]:
        """Use verified rules to suggest an action context."""
        available_actions = available_context.get("available_actions", [])

        candidates = [h for h in self._verified_rules if h.confidence > 0.5]
        if available_actions:
            candidates = [h for h in candidates
                          if h.action_context.get("action") in available_actions
                          or "action" not in h.action_context]

        if not candidates:
            return None

        best = max(candidates, key=lambda h: h.confidence * h.predicted_magnitude)
        self._stats["used_for_action"] += 1
        return best.action_context

    def get_verified_concepts(self) -> List[dict]:
        """Return verified hypotheses as concept candidates for CGN grounding."""
        return [
            {
                "concept_id": f"haov_{h.rule}",
                "confidence": h.confidence,
                "consumer": h.consumer,
                "effect": h.predicted_effect,
                "tests": h.tests,
                "confirmations": h.confirmations,
                "source": h.source,
            }
            for h in self._verified_rules if h.confidence > 0.5
        ]

    def get_stats(self) -> dict:
        """Return HAOV statistics."""
        return {
            **self._stats,
            "active_hypotheses": len(self._hypotheses),
            "verified_rules": len(self._verified_rules),
            "top_rules": [
                {"rule": h.rule, "conf": round(h.confidence, 2),
                 "tests": h.tests, "confirmed": h.confirmations}
                for h in sorted(self._verified_rules,
                                key=lambda x: x.confidence, reverse=True)[:5]
            ],
        }

    def reset_episode(self):
        """Reset per-episode state, keep verified rules."""
        self._active_test = None
        self._hypotheses = [h for h in self._hypotheses if h.confidence > 0.05]


