"""
Concept Grounding Network (CGN) — Higher Cognitive Kernel for Titan.

Layer 3 in the cognitive stack: sits between sensory grounding (MSL, 132D state)
and abstract reasoning. Implements a single concept grounding algorithm shared
across all higher cognitive consumers (language, reasoning, creative, social).

Design: Shared V(s) value net (what states are good for concept learning) +
per-consumer Q(s,a) action nets (what to DO about a concept in each domain).
IQL training via shared buffer, dream-consolidated.

ADDITIVE: If CGN fails, consumers fall back to their legacy paths.
CGN enhances and optimizes — never blocks.

First consumer: language (word grounding, associations, tensor seeding).
Future consumers: abstract_reasoning, creative, social.

See: titan-docs/rFP_concept_grounding_network.md
"""

import json
import logging
import math
import os
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("titan.cgn")

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


# ── Neural Networks ─────────────────────────────────────────────────────────


class SharedValueNet(nn.Module):
    """V(s) — estimates value of current grounding state. Shared across consumers."""

    def __init__(self, input_dim: int = 30):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 48),
            nn.LayerNorm(48),
            nn.ReLU(),
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Linear(24, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ConsumerActionNet(nn.Module):
    """Q(s,a) — per-consumer action policy. Selects grounding action + continuous params."""

    def __init__(self, input_dim: int = 30, action_dims: int = 8):
        super().__init__()
        self.action_dims = action_dims
        # Action selection head
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 24),
            nn.ReLU(),
            nn.Linear(24, 12),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(12, action_dims)
        # Continuous parameter heads (4 outputs per action)
        self.param_head = nn.Linear(12, 4)  # conf_delta, plasticity, assoc_delta, ctx_weight

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        action_logits = self.action_head(features)
        params = torch.sigmoid(self.param_head(features))  # All in [0, 1]
        return action_logits, params


# ── Transition Buffer ─────────────────────────��─────────────────────────────


class TransitionBuffer:
    """Shared buffer with per-consumer sub-views. FIFO eviction."""

    def __init__(self, max_size: int = 2000):
        self.max_size = max_size
        self._buffer: List[CGNTransition] = []

    def add(self, transition: CGNTransition) -> None:
        self._buffer.append(transition)
        if len(self._buffer) > self.max_size:
            self._buffer = self._buffer[-self.max_size:]

    def get_consumer_transitions(self, consumer: str,
                                 max_count: int = 1000) -> List[CGNTransition]:
        result = [t for t in self._buffer if t.consumer == consumer]
        return result[-max_count:]

    def get_all(self, max_count: int = 2000) -> List[CGNTransition]:
        return self._buffer[-max_count:]

    def mark_consolidated(self) -> None:
        for t in self._buffer:
            t._consolidated = True

    def size(self) -> int:
        return len(self._buffer)

    def consumer_sizes(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for t in self._buffer:
            counts[t.consumer] = counts.get(t.consumer, 0) + 1
        return counts


# ── Core CGN Class ──────────────────────────────────────────────────────────


# PERSISTENCE_BY_DESIGN: ConceptGroundingNetwork._value_net / _action_nets /
# _consumer_freq / _concept_journeys / _surprise_buffer / _haov_trackers
# persist via _save_state → cgn_state.pt (torch.load/save) and are restored
# via _load_state_dict / dict mutation patterns the AST scanner can't
# statically trace as direct self-assignments.
class ConceptGroundingNetwork:
    """Higher Cognitive Kernel — single concept grounding algorithm for all consumers.

    ADDITIVE design: if this fails, consumers fall back to legacy paths.
    """

    # Action parameter ranges (mapped from sigmoid [0,1])
    CONF_DELTA_RANGE = (-0.05, 0.10)
    PLASTICITY_RANGE = (0.0, 0.5)
    ASSOC_DELTA_RANGE = (-0.1, 0.2)
    CTX_WEIGHT_RANGE = (0.0, 1.0)

    def __init__(self, db_path: str = "./data/inner_memory.db",
                 state_dir: str = "./data/cgn",
                 value_lr: float = 1e-4,
                 policy_lr: float = 3e-4,
                 gamma: float = 0.99,
                 haov_config: dict = None):
        self._db_path = db_path
        self._state_dir = state_dir
        self._gamma = gamma
        self._haov_config = haov_config or {}

        # Consumer registry
        self._consumers: Dict[str, CGNConsumerConfig] = {}
        self._action_nets: Dict[str, ConsumerActionNet] = {}
        self._action_optimizers: Dict[str, torch.optim.Adam] = {}

        # Shared value net
        self._value_net = SharedValueNet(input_dim=30)
        self._value_optimizer = torch.optim.Adam(
            self._value_net.parameters(), lr=value_lr)

        # Shared buffer
        self._buffer = TransitionBuffer(max_size=2000)

        # Statistics
        self._total_groundings = 0
        self._total_rewards = 0.0
        self._consolidation_count = 0
        self._boot_time = time.time()

        # Sigma: online V(s) micro-updates (continuous gradient learning)
        self._value_lr = value_lr
        self._online_lr = 0.001  # 10x smaller than dream consolidation
        self._consumer_freq: Dict[str, int] = {}  # record_outcome call count per consumer

        # BUG-CGN-SILENT-UNREGISTERED-CONSUMER Phase 0 observability (2026-04-21).
        # record_outcome() silently no-ops when called with a consumer that
        # was never CGN_REGISTER'd — the for-loop scans transitions, finds no
        # match, exits without logging. On T2/T3 this hides language_worker's
        # 7+2 ad-hoc "language"/"social" sites that ship transitions for
        # consumers nobody registered. Likely root of T2/T3 meta-reasoning
        # stuck-in-shadow (w_grounded=0.057 vs T1=0.535).
        # Phase 0 = visibility ONLY (no behavior change). Counters drive
        # arch_map cgn diagnostics + a /v4/cgn/silent-drops endpoint.
        self._unregistered_outcome_attempts: Dict[str, int] = {}
        self._unmatched_outcome_attempts: Dict[str, int] = {}
        self._silent_drop_last_log_ts: Dict[str, float] = {}
        self._silent_drop_log_interval_sec: float = 60.0  # throttle warns to ≤1/min/consumer

        # Concept lifecycle: track concept journey across consumers
        self._concept_journeys: Dict[str, dict] = {}  # concept_id → {first_consumer, consumers_seen, ...}

        # Shared surprise signal: any consumer records surprise, all benefit
        self._surprise_buffer: List[dict] = []

        # Generalized HAOV: per-consumer hypothesis trackers
        self._haov_trackers: Dict[str, GeneralizedHAOVTracker] = {}

        # Telemetry log (persistent, append-only)
        self._telemetry_path = os.path.join(state_dir, "cgn_telemetry.jsonl")

        # Ensure state directory exists
        os.makedirs(state_dir, exist_ok=True)

        # Try to load saved state
        self._load_state()

        logger.info("[CGN] Initialized (consumers=%d, buffer=%d, consolidations=%d)",
                    len(self._consumers), self._buffer.size(),
                    self._consolidation_count)

    # ── Consumer Registration ─────────────────────────���─────────────────

    def register_consumer(self, config: CGNConsumerConfig) -> str:
        """Register a cognitive consumer. Returns consumer handle (name)."""
        if config.name in self._consumers:
            logger.debug("[CGN] Consumer '%s' already registered", config.name)
            return config.name

        self._consumers[config.name] = config

        # Create per-consumer action net
        net = ConsumerActionNet(
            input_dim=config.feature_dims,
            action_dims=config.action_dims)
        self._action_nets[config.name] = net
        self._action_optimizers[config.name] = torch.optim.Adam(
            net.parameters(), lr=3e-4)

        # Create per-consumer HAOV tracker (generalized hypothesis testing)
        if config.name not in self._haov_trackers:
            self._haov_trackers[config.name] = GeneralizedHAOVTracker(
                consumer=config.name, config=self._haov_config)

        logger.info("[CGN] Registered consumer '%s' (features=%dD, actions=%d: %s)",
                    config.name, config.feature_dims, config.action_dims,
                    ", ".join(config.action_names))
        return config.name

    # ── Core: ground() ───────────────────────���──────────────────────────

    def ground(self, consumer: str, concept: ConceptFeatures,
               sensory_ctx: SensoryContext) -> GroundingAction:
        """Core method: decide how to adjust a concept's grounding.

        Returns GroundingAction with confidence_delta, tensor_plasticity,
        association_deltas, and context_weight.
        """
        if consumer not in self._consumers:
            return GroundingAction()  # Safe fallback

        config = self._consumers[consumer]
        net = self._action_nets[consumer]

        # Build input features
        state_vec = self._build_state_vector(concept, sensory_ctx)
        state_tensor = torch.FloatTensor(state_vec).unsqueeze(0)

        # Forward pass
        with torch.no_grad():
            action_logits, params = net(state_tensor)

        # Select action (softmax sampling with temperature)
        temperature = 0.5
        probs = torch.softmax(action_logits / temperature, dim=-1)
        action_idx = torch.multinomial(probs, 1).item()
        action_name = config.action_names[min(action_idx, len(config.action_names) - 1)]

        # Map continuous params from [0, 1] to action-specific ranges
        p = params.squeeze(0).numpy()
        conf_delta = self._map_range(p[0], *self.CONF_DELTA_RANGE)
        plasticity = self._map_range(p[1], *self.PLASTICITY_RANGE)
        assoc_delta = self._map_range(p[2], *self.ASSOC_DELTA_RANGE)
        ctx_weight = self._map_range(p[3], *self.CTX_WEIGHT_RANGE)

        # Action-specific adjustments
        if action_name == "reinforce":
            conf_delta = max(0.0, conf_delta)
            plasticity *= 0.3  # Minimal tensor change
        elif action_name == "explore":
            conf_delta *= 0.5  # Lower confidence adjustment
            plasticity = max(0.2, plasticity)  # Ensure some plasticity
        elif action_name == "differentiate":
            plasticity = max(0.3, plasticity)  # Push tensor away
            conf_delta = 0.0  # Don't change confidence during differentiation
        elif action_name == "consolidate":
            ctx_weight = max(0.7, ctx_weight)  # High context weight
            plasticity *= 0.2  # Low plasticity during consolidation
        elif action_name == "associate":
            assoc_delta = max(0.02, assoc_delta)  # Ensure positive association
        elif action_name == "dissociate":
            assoc_delta = min(-0.02, assoc_delta)  # Ensure negative
        elif action_name == "deepen":
            plasticity = max(0.3, plasticity)
            conf_delta = max(0.0, conf_delta)
        elif action_name == "stabilize":
            plasticity = min(0.05, plasticity)  # Lock in
            conf_delta = max(0.0, conf_delta) * 0.5

        # Record transition (reward added later via record_outcome)
        transition = CGNTransition(
            consumer=consumer,
            concept_id=concept.concept_id,
            state=state_vec,
            action=action_idx,
            action_params=p,
            reward=0.0,
            timestamp=time.time(),
            epoch=sensory_ctx.epoch,
            metadata={"action_name": action_name,
                      "encounter_type": sensory_ctx.encounter_type},
        )
        self._buffer.add(transition)
        self._total_groundings += 1

        # Concept lifecycle: track journey across consumers
        cid = concept.concept_id
        if cid not in self._concept_journeys:
            self._concept_journeys[cid] = {
                "first_consumer": consumer, "first_ts": time.time(),
                "consumers_seen": set(),
            }
        self._concept_journeys[cid]["consumers_seen"].add(consumer)

        # Telemetry: log every 10th grounding (reduce noise)
        if self._total_groundings % 10 == 0:
            self._log_telemetry({
                "event": "grounding",
                "consumer": consumer,
                "concept_id": concept.concept_id,
                "action": action_name,
                "confidence_delta": round(conf_delta, 4),
                "encounter_type": sensory_ctx.encounter_type,
                "epoch": sensory_ctx.epoch,
            })

        return GroundingAction(
            action_index=action_idx,
            action_name=action_name,
            confidence_delta=round(conf_delta, 4),
            tensor_plasticity=round(plasticity, 4),
            association_deltas={},  # Populated by consumer from context
            context_weight=round(ctx_weight, 4),
        )

    # ── Delayed Reward ──────────────────────────────────────────────────

    def record_outcome(self, consumer: str, concept_id: str,
                       reward: float, outcome_context: dict = None) -> None:
        """Record delayed reward for a previous grounding decision.

        Includes Sigma-style online V(s) micro-update: one gradient step
        per outcome, with frequency-scaled learning rate to prevent
        dominant consumers from overwhelming the shared value net.
        """
        # Phase 0 observability for BUG-CGN-SILENT-UNREGISTERED-CONSUMER.
        # Distinguish two failure modes that previously both fell through silently:
        #   (a) consumer was never registered at all → count + log throttled
        #   (b) consumer registered but no matching transition for this concept
        #       (already-rewarded or evicted from buffer) → count separately
        # No behavior change — telemetry only.
        if consumer not in self._consumers:
            self._unregistered_outcome_attempts[consumer] = (
                self._unregistered_outcome_attempts.get(consumer, 0) + 1)
            _now = time.time()
            _last = self._silent_drop_last_log_ts.get(consumer, 0.0)
            if _now - _last >= self._silent_drop_log_interval_sec:
                logger.warning(
                    "[CGN] record_outcome called with UNREGISTERED consumer=%r "
                    "(concept=%s, reward=%.3f) — total unregistered attempts=%d. "
                    "Likely BUG-CGN-SILENT-UNREGISTERED-CONSUMER. Sender must "
                    "send CGN_REGISTER before CGN_TRANSITION/record_outcome.",
                    consumer, concept_id, reward,
                    self._unregistered_outcome_attempts[consumer])
                self._silent_drop_last_log_ts[consumer] = _now
            return  # no transition match possible

        # Find most recent transition for this concept
        for t in reversed(self._buffer._buffer):
            if t.consumer == consumer and t.concept_id == concept_id and t.reward == 0.0:
                t.reward = reward
                t.metadata.update(outcome_context or {})
                self._total_rewards += reward

                # Sigma: one V(s) micro-update (continuous gradient learning)
                try:
                    self._consumer_freq[consumer] = self._consumer_freq.get(consumer, 0) + 1
                    freq = self._consumer_freq[consumer]
                    total_freq = max(1, sum(self._consumer_freq.values()))
                    n_consumers = max(1, len(self._consumer_freq))
                    # Scale lr inversely by relative frequency to prevent dominant consumers
                    freq_scale = min(2.0, total_freq / max(1, freq * n_consumers))
                    scaled_lr = self._online_lr * freq_scale

                    state_t = torch.FloatTensor(t.state).unsqueeze(0)
                    v_pred = self._value_net(state_t).squeeze()
                    target = torch.tensor(reward, dtype=torch.float32)
                    loss = torch.nn.functional.mse_loss(v_pred, target)
                    self._value_optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self._value_net.parameters(), 0.5)
                    # Apply with scaled lr, then restore dream lr
                    for pg in self._value_optimizer.param_groups:
                        pg['lr'] = scaled_lr
                    self._value_optimizer.step()
                    for pg in self._value_optimizer.param_groups:
                        pg['lr'] = self._value_lr
                except Exception:
                    pass  # Non-critical — don't break reward recording

                return
        # Loop exited without a match: registered consumer, but no pending
        # transition for this concept (already rewarded, evicted from buffer,
        # or concept_id mismatch). Track separately from unregistered attempts.
        self._unmatched_outcome_attempts[consumer] = (
            self._unmatched_outcome_attempts.get(consumer, 0) + 1)

    def get_silent_drop_telemetry(self) -> dict:
        """Phase 0 observability accessor for BUG-CGN-SILENT-UNREGISTERED-CONSUMER.

        Returns dict with cumulative counts per consumer:
          - unregistered: record_outcome called for a consumer that never
            sent CGN_REGISTER. Bug signal — these writes never landed.
          - unmatched: registered consumer + record_outcome but no matching
            pending transition. Usually benign (already rewarded / evicted).
          - registered: list of currently registered consumer names (for diff).
        """
        return {
            "unregistered": dict(self._unregistered_outcome_attempts),
            "unmatched": dict(self._unmatched_outcome_attempts),
            "registered": sorted(self._consumers.keys()),
            "consumer_freq": dict(self._consumer_freq),
        }

    # ── Dream Consolidation ─────────────────────────────────────��───────

    def consolidate(self, dream_phase: bool = False) -> dict:
        """Train policy nets from shared buffer.

        Called during dream phase for full consolidation,
        or periodically for lightweight online updates.
        """
        if self._buffer.size() < 10:
            return {"trained": False, "reason": "insufficient_data",
                    "buffer_size": self._buffer.size()}

        stats = {"trained": True, "consumers": {}}

        if not dream_phase:
            # Lightweight online update — V(s) only
            v_loss = self._train_value_net(batch_size=16, steps=5)
            stats["v_loss"] = round(v_loss, 6)
            return stats

        # Full dream consolidation
        # 1. Train shared V(s) from all transitions
        v_loss = self._train_value_net(batch_size=32, steps=20)
        stats["v_loss"] = round(v_loss, 6)

        # 2. Train each consumer's Q(s,a) with advantage
        # Dynamic priority: reward velocity + static priority (Sigma-inspired)
        sorted_consumers = sorted(
            self._consumers.items(),
            key=lambda x: (-self._compute_reward_velocity(x[0]),
                           x[1].consolidation_priority))

        for name, config in sorted_consumers:
            transitions = self._buffer.get_consumer_transitions(name)
            if len(transitions) < 5:
                stats["consumers"][name] = {"skipped": True,
                                            "transitions": len(transitions)}
                continue

            q_loss = self._train_consumer_policy(name, batch_size=32, steps=15)
            stats["consumers"][name] = {
                "q_loss": round(q_loss, 6),
                "transitions": len(transitions),
            }

        # 3. Decay low-confidence concepts (gentle)
        self._decay_unused_concepts(factor=0.995)

        # 4. Mark consolidated
        self._buffer.mark_consolidated()
        self._consolidation_count += 1

        # Telemetry: log every consolidation
        self._log_telemetry({
            "event": "consolidation",
            "dream_phase": True,
            "v_loss": round(v_loss, 6),
            "consumers": {k: v for k, v in stats.get("consumers", {}).items()},
            "consolidation_number": self._consolidation_count,
        })

        # 5. C4a: Cross-consumer insight transfer during dreams
        cross_insights = {}
        try:
            for consumer_name in self._consumers:
                insights = self.get_cross_insights(consumer_name)
                if insights:
                    cross_insights[consumer_name] = insights
            if cross_insights:
                self._log_telemetry({
                    "event": "cross_consumer_transfer",
                    "insights": {
                        k: [{"source": i.get("source_consumer", "?"),
                             "concepts": i.get("top_concepts", [])[:3],
                             "avg_reward": round(i.get("avg_reward", 0), 3)}
                            for i in v]
                        for k, v in cross_insights.items()
                    },
                })
                logger.info("[CGN] Cross-consumer transfer: %s",
                            {k: len(v) for k, v in cross_insights.items()})
        except Exception as _xc_err:
            logger.debug("[CGN] Cross-consumer transfer error: %s", _xc_err)

        stats["cross_insights"] = {k: len(v) for k, v in cross_insights.items()}

        # 6. Save state
        self._save_state()

        stats["consolidation_number"] = self._consolidation_count
        stats["buffer_size"] = self._buffer.size()

        logger.info("[CGN] Dream consolidation #%d: V_loss=%.4f, consumers=%s",
                    self._consolidation_count, v_loss,
                    {k: v.get("q_loss", "skip") for k, v in stats["consumers"].items()})

        return stats

    # ── Bootstrap ───────────────────────────────────────────────────────

    # UNUSED_PUBLIC_API: cold-start seeding path designed in the original
    # CGN rFP (2026-04-03). No live caller — CGN bootstraps from per-
    # consumer replay buffer instead of synthetic seed. Preserved as
    # public API for future migrations / rebuild scenarios.
    def bootstrap_concepts(self, consumer: str,
                           concepts: List[ConceptFeatures]) -> int:
        """One-time initialization for concepts that predate CGN.

        Seeds the buffer with synthetic transitions so the value net
        has initial data to learn from.
        """
        if consumer not in self._consumers:
            return 0

        count = 0
        for concept in concepts:
          try:
            # Create synthetic sensory context from concept's existing data
            synth_ctx = SensoryContext(
                epoch=concept.age_epochs,
                encounter_type="bootstrap",
            )
            state_vec = self._build_state_vector(concept, synth_ctx)

            # Assign synthetic reward based on concept quality
            reward = 0.0
            if concept.confidence > 0.8:
                reward += 0.05
            if concept.cross_modal_conf > 0.1:
                reward += 0.05
            if concept.production_count > 5:
                reward += 0.03
            if len(concept.context_history) > 0:
                reward += 0.02

            transition = CGNTransition(
                consumer=consumer,
                concept_id=concept.concept_id,
                state=state_vec,
                action=0,  # reinforce (default)
                action_params=np.array([0.5, 0.1, 0.0, 0.5]),
                reward=reward,
                timestamp=time.time(),
                epoch=concept.age_epochs,
                metadata={"type": "bootstrap"},
            )
            self._buffer.add(transition)
            count += 1
          except Exception as _boot_err:
            logger.debug("[CGN] Bootstrap skip '%s': %s", concept.concept_id, _boot_err)

        if count > 0:
            logger.info("[CGN] Bootstrapped %d concepts for '%s'", count, consumer)

        return count

    # ── Cross-Domain Insights ─────────────────────────��─────────────────

    def get_cross_insights(self, consumer: str) -> List[dict]:
        """Retrieve insights from OTHER consumers."""
        insights = []
        for other_name, other_config in self._consumers.items():
            if other_name == consumer:
                continue
            transitions = self._buffer.get_consumer_transitions(other_name, max_count=100)
            if not transitions:
                continue

            # Find high-reward transitions from other consumers
            high_reward = [t for t in transitions if t.reward > 0.05]
            if high_reward:
                avg_reward = sum(t.reward for t in high_reward) / len(high_reward)
                top_concepts = list({t.concept_id for t in high_reward})[:5]
                insights.append({
                    "source_consumer": other_name,
                    "high_reward_count": len(high_reward),
                    "avg_reward": round(avg_reward, 4),
                    "top_concepts": top_concepts,
                })

        return insights

    # ── Statistics ─────────────────────────────────���────────────────────

    def _log_telemetry(self, event: dict) -> None:
        """Append CGN event to persistent telemetry log."""
        try:
            event["timestamp"] = time.time()
            event["total_groundings"] = self._total_groundings
            event["buffer_size"] = self._buffer.size()
            with open(self._telemetry_path, "a") as f:
                f.write(json.dumps(event) + "\n")
        except Exception:
            pass

    def get_stats(self) -> dict:
        """Return CGN statistics for monitoring/API."""
        return {
            "total_groundings": self._total_groundings,
            "total_rewards": round(self._total_rewards, 4),
            "avg_reward": round(self._total_rewards / max(1, self._total_groundings), 4),
            "buffer_size": self._buffer.size(),
            "buffer_by_consumer": self._buffer.consumer_sizes(),
            "consumers": list(self._consumers.keys()),
            "consolidations": self._consolidation_count,
            "uptime_hours": round((time.time() - self._boot_time) / 3600, 2),
        }

    # ── Generalized HAOV API ────────────────────────────────────────────

    def register_verifier(self, consumer: str, verify_fn: callable) -> None:
        """Register a domain-specific verification function for a consumer's HAOV.

        verify_fn(obs_before: dict, obs_after: dict, action_ctx: dict) → (confirmed: bool, error: float)
        """
        if consumer in self._haov_trackers:
            self._haov_trackers[consumer].set_verifier(verify_fn)
            logger.info("[CGN] HAOV verifier registered for '%s'", consumer)

    def get_haov(self, consumer: str) -> Optional[GeneralizedHAOVTracker]:
        """Get HAOV tracker for a consumer (or None)."""
        return self._haov_trackers.get(consumer)

    # ── SOAR Impasse Detection ─────────────────────────────────────────

    def detect_impasse(self, consumer: str) -> Optional[dict]:
        """SOAR-inspired impasse detection for a consumer.

        Returns impasse dict or None. When impasse is detected AND
        HAOV tracker exists, generates targeted hypothesis.

        Impasse types:
          stuck — no positive reward in recent window (reasoning)
          plateau — same concept repeated with no improvement (language)
          declining — reward trend is negative (social)
        """
        transitions = self._buffer.get_consumer_transitions(consumer, max_count=50)
        if len(transitions) < 10:
            return None

        recent = transitions[-20:]
        rewarded = [t for t in recent if t.reward > 0.05]

        # Type 1: stuck — no positive rewards
        if len(rewarded) == 0:
            severity = 1.0 - (len([t for t in recent if t.reward > 0.01]) / len(recent))
            result = {"type": "stuck", "severity": severity, "consumer": consumer}
            # SOAR → HAOV bridge: generate subgoal hypothesis
            if consumer in self._haov_trackers:
                self._haov_trackers[consumer].hypothesize_from_impasse(result)
            return result

        # Type 2: plateau — same concept repeated with stagnant reward
        concept_counts: Dict[str, int] = {}
        for t in recent:
            concept_counts[t.concept_id] = concept_counts.get(t.concept_id, 0) + 1
        dominant = max(concept_counts.items(), key=lambda x: x[1])
        if dominant[1] >= len(recent) * 0.6:  # One concept dominates >60%
            dominant_rewards = [t.reward for t in recent if t.concept_id == dominant[0]]
            if max(dominant_rewards) - min(dominant_rewards) < 0.02:
                result = {"type": "plateau", "concept": dominant[0],
                          "consumer": consumer, "repetitions": dominant[1]}
                if consumer in self._haov_trackers:
                    self._haov_trackers[consumer].hypothesize_from_impasse(result)
                return result

        # Type 3: declining — reward trend is negative
        if len(rewarded) >= 4:
            mid = len(rewarded) // 2
            old_avg = sum(t.reward for t in rewarded[:mid]) / mid
            new_avg = sum(t.reward for t in rewarded[mid:]) / max(1, len(rewarded) - mid)
            if new_avg < old_avg * 0.6:  # >40% decline
                result = {"type": "declining", "trend": new_avg - old_avg,
                          "consumer": consumer}
                if consumer in self._haov_trackers:
                    self._haov_trackers[consumer].hypothesize_from_impasse(result)
                return result

        return None

    # ── Social Policy Inference ──────────────────────────────────────────

    def infer_social_action(self, sensory_ctx: SensoryContext,
                            user_features: dict = None) -> dict:
        """Infer best social action from learned Q(s,a) policy.

        Returns dict with action_name, action_index, confidence, q_values.
        Used by social_x_gateway and agno_hooks for soft-gated decisions.

        user_features keys:
            familiarity (float 0-1), interaction_count (int),
            social_valence (float -1..1), mention_count (int),
            net_sentiment (float -1..1), social_felt_tensor (list[float] 30D)
        """
        fallback = {
            "action_name": "engage_cautiously",
            "action_index": 1,
            "confidence": 0.0,
            "q_values": {},
        }

        if "social" not in self._consumers:
            return fallback

        # Check minimum training data
        social_transitions = sum(
            1 for t in self._buffer._buffer
            if t.consumer == "social"
        ) if hasattr(self._buffer, '_buffer') else 0
        if social_transitions < 10:
            return fallback

        config = self._consumers["social"]
        net = self._action_nets["social"]
        uf = user_features or {}

        # Build concept features from user profile
        # Map user attributes to ConceptFeatures fields
        felt_tensor = uf.get("social_felt_tensor", [0.5] * 30)
        if len(felt_tensor) < 30:
            felt_tensor = felt_tensor + [0.5] * (30 - len(felt_tensor))

        concept = ConceptFeatures(
            concept_id=uf.get("user_id", "unknown"),
            embedding=np.array(felt_tensor[:130] if len(felt_tensor) >= 130
                               else felt_tensor + [0.5] * (130 - len(felt_tensor)),
                               dtype=np.float32),
            confidence=uf.get("familiarity", 0.0),
            encounter_count=uf.get("interaction_count", 0),
            production_count=uf.get("mention_count", 0),
            age_epochs=uf.get("relationship_age_epochs", 0),
            cross_modal_conf=max(0.0, min(1.0,
                (uf.get("social_valence", 0.0) + 1.0) / 2.0)),  # map -1..1 to 0..1
            associations={},
            context_history=[],
            meaning_contexts=[],
        )

        try:
            state_vec = self._build_state_vector(concept, sensory_ctx)
            state_tensor = torch.FloatTensor(state_vec).unsqueeze(0)

            with torch.no_grad():
                action_logits, _params = net(state_tensor)

            temperature = 0.5
            probs = torch.softmax(action_logits / temperature, dim=-1).squeeze(0)
            best_idx = torch.argmax(probs).item()
            best_name = config.action_names[min(best_idx,
                                                len(config.action_names) - 1)]
            confidence = probs[best_idx].item()

            q_values = {
                config.action_names[i]: round(probs[i].item(), 4)
                for i in range(min(len(config.action_names), probs.shape[0]))
            }

            return {
                "action_name": best_name,
                "action_index": best_idx,
                "confidence": round(confidence, 4),
                "q_values": q_values,
            }
        except Exception as e:
            logger.debug("[CGN] infer_social_action failed: %s", e)
            return fallback

    # ── Language Consumer Helpers ────────────────────────────────────────

    # DEPRECATED: use language_pipeline.load_concept_from_db(db_path, word)
    # instead. The standalone function works without a local CGN instance
    # (needed by consumer-client-only workers). This instance method is
    # retained for back-compat with any remaining dual-path callers.
    def load_concept(self, consumer: str, word: str) -> Optional[ConceptFeatures]:
        """Load a word from vocabulary DB as ConceptFeatures."""
        try:
            conn = sqlite3.connect(self._db_path, timeout=5.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            row = conn.execute(
                "SELECT word, felt_tensor, confidence, times_encountered, "
                "times_produced, learning_phase, created_at, "
                "COALESCE(sensory_context, '[]'), "
                "COALESCE(meaning_contexts, '[]'), "
                "COALESCE(cross_modal_conf, 0.0) "
                "FROM vocabulary WHERE word=?", (word,)).fetchone()
            conn.close()

            if not row:
                return None

            # Defensive parsing — handle binary/corrupt data gracefully
            try:
                ft_raw = row[1]
                if isinstance(ft_raw, bytes):
                    ft = [0.5] * 130  # Binary tensor from old code
                elif ft_raw:
                    ft = json.loads(ft_raw)
                else:
                    ft = [0.5] * 130
            except (json.JSONDecodeError, TypeError):
                ft = [0.5] * 130

            try:
                contexts = json.loads(row[7]) if row[7] and isinstance(row[7], str) else []
            except (json.JSONDecodeError, TypeError):
                contexts = []

            try:
                meanings = json.loads(row[8]) if row[8] and isinstance(row[8], str) else []
            except (json.JSONDecodeError, TypeError):
                meanings = []

            # Extract associations from meaning_contexts
            associations = {}
            for m in meanings:
                for a in m.get("associations", []):
                    if isinstance(a, (list, tuple)) and len(a) >= 2:
                        associations[a[0]] = associations.get(a[0], 0) + 0.1

            # Approximate age in epochs from creation time
            age = int((time.time() - (row[6] or time.time())) / 1.15)  # ~1.15s per epoch

            return ConceptFeatures(
                concept_id=word,
                embedding=np.array(ft, dtype=np.float32),
                confidence=row[2],
                encounter_count=row[3],
                production_count=row[4],
                context_history=[{"ctx": c} for c in contexts[-10:]],
                associations=associations,
                age_epochs=max(0, age),
                cross_modal_conf=row[9],
                meaning_contexts=meanings,
                extra={"learning_phase": row[5]},
            )
        except Exception as e:
            logger.debug("[CGN] load_concept('%s') failed: %s", word, e)
            return None

    # DEPRECATED: use language_pipeline.apply_grounding_action_to_db(
    #   db_path, word, action, state_132d) instead. Standalone function
    # for consumer-client-only workers. Retained for back-compat.
    def apply_grounding_action(self, word: str, action: GroundingAction,
                               sensory_ctx: SensoryContext = None) -> bool:
        """Apply a GroundingAction to a word in the vocabulary DB."""
        try:
            conn = sqlite3.connect(self._db_path, timeout=5.0)
            conn.execute("PRAGMA journal_mode=WAL")
            row = conn.execute(
                "SELECT confidence, felt_tensor, "
                "COALESCE(cross_modal_conf, 0.0), "
                "COALESCE(sensory_context, '[]') "
                "FROM vocabulary WHERE word=?", (word,)).fetchone()

            if not row:
                conn.close()
                return False

            def _db_float(v, default=0.0):
                if isinstance(v, dict):
                    return float(v.get("confidence", v.get("level", default)))
                if isinstance(v, bytes):
                    import struct as _st
                    try: return _st.unpack('<f', v)[0]
                    except Exception: return default
                try: return float(v) if v is not None else default
                except (TypeError, ValueError): return default
            old_conf = _db_float(row[0])
            try:
                ft = json.loads(row[1]) if row[1] and isinstance(row[1], str) else [0.5] * 130
            except (json.JSONDecodeError, TypeError):
                ft = [0.5] * 130
            old_xm = _db_float(row[2])
            try:
                contexts = json.loads(row[3]) if row[3] and isinstance(row[3], str) else []
            except (json.JSONDecodeError, TypeError):
                contexts = []

            # Apply confidence delta
            new_conf = max(0.0, min(1.0, old_conf + float(action.confidence_delta)))

            # Apply tensor plasticity (blend toward current state)
            if action.tensor_plasticity > 0.01 and sensory_ctx and sensory_ctx.state_132d is not None:
                state = sensory_ctx.state_132d
                p = action.tensor_plasticity
                for i in range(min(len(ft), len(state), 130)):
                    ft[i] = ft[i] * (1 - p) + float(state[i]) * p

            # Update cross_modal_conf (grounding actions increase it)
            xm_boost = 0.0
            if action.action_name in ("reinforce", "deepen", "consolidate"):
                xm_boost = 0.01
            elif action.action_name == "differentiate":
                xm_boost = 0.02  # Differentiation is high-quality grounding
            new_xm = min(1.0, old_xm + xm_boost)

            # Write back
            conn.execute(
                "UPDATE vocabulary SET confidence=?, felt_tensor=?, "
                "cross_modal_conf=? WHERE word=?",
                (new_conf, json.dumps(ft), new_xm, word))
            conn.commit()
            conn.close()
            return True

        except Exception as e:
            logger.debug("[CGN] apply_grounding_action('%s') failed: %s", word, e)
            return False

    # ── Internal: Feature Building ─────────────────────────────���────────

    def _build_state_vector(self, concept: ConceptFeatures,
                            ctx: SensoryContext) -> np.ndarray:
        """Build 30D state vector for V(s) and Q(s,a) input.

        Layout:
          [0:5]   neuromods: DA, 5HT, NE, GABA, ACh
          [5:11]  MSL summary: top-3 attention + entropy + I_conf + convergence
          [11:20] state summary: body_avg, mind_feel, mind_think, spirit_avg,
                                 outer_avg, drift, trajectory, chi, pi_norm
          [20:29] concept summary: confidence, enc_norm, prod_norm, age_norm,
                                   xm_conf, assoc_density, ctx_diversity,
                                   tensor_mag, embedding_stability
          [29]    encounter_type encoding
        """
        vec = np.zeros(30, dtype=np.float32)

        # Neuromods [0:5] — handle both flat (float) and nested ({level, setpoint}) formats
        nm = ctx.neuromods or {}

        def _nm_val(key, alt_key=None, default=0.5):
            v = nm.get(key, nm.get(alt_key, default) if alt_key else default)
            if isinstance(v, dict):
                return float(v.get("level", default))
            return float(v)

        vec[0] = _nm_val("DA")
        vec[1] = _nm_val("5-HT", "5HT")
        vec[2] = _nm_val("NE")
        vec[3] = _nm_val("GABA")
        vec[4] = _nm_val("ACh")

        # MSL summary [5:11]
        if ctx.msl_attention is not None and len(ctx.msl_attention) >= 6:
            attn = ctx.msl_attention
            # Top-3 attention values
            top3 = sorted(attn, reverse=True)[:3]
            vec[5:8] = top3
            # Entropy of attention
            attn_norm = attn / (np.sum(attn) + 1e-10)
            entropy = -np.sum(attn_norm * np.log(attn_norm + 1e-10))
            vec[8] = min(1.0, entropy / 4.0)  # Normalize
        cc = ctx.concept_confidences or {}

        def _cc_val(v):
            if isinstance(v, dict):
                return float(v.get("confidence", v.get("level", 0.0)))
            return float(v) if v is not None else 0.0

        vec[9] = _cc_val(cc.get("I", 0.0))
        vec[10] = (sum(_cc_val(v) for v in cc.values()) / max(len(cc), 1)) if cc else 0.0

        # State summary [11:20]
        if ctx.state_132d is not None and len(ctx.state_132d) >= 65:
            s = ctx.state_132d
            vec[11] = float(np.mean(s[:5]))     # body avg
            vec[12] = float(np.mean(s[5:15]))   # mind feel avg
            vec[13] = float(np.mean(s[15:20]))  # mind think avg
            vec[14] = float(np.mean(s[20:65]))  # spirit avg
            if len(s) > 65:
                vec[15] = float(np.mean(s[65:95]))  # outer avg
            vec[16] = float(np.std(s[:65]))  # drift proxy
        vec[17] = 0.5  # trajectory placeholder
        vec[18] = 0.5  # chi placeholder
        vec[19] = min(1.0, ctx.epoch / 500000.0) if ctx.epoch else 0.0

        # Concept summary [20:29]
        def _safe_f(v, default=0.0):
            if isinstance(v, dict):
                return float(v.get("confidence", v.get("level", default)))
            if isinstance(v, bytes):
                return default
            try:
                return float(v) if v is not None else default
            except (TypeError, ValueError):
                return default
        vec[20] = _safe_f(concept.confidence)
        vec[21] = min(1.0, _safe_f(concept.encounter_count) / 100.0)
        vec[22] = min(1.0, _safe_f(concept.production_count) / 50.0)
        vec[23] = min(1.0, _safe_f(concept.age_epochs) / 300000.0)
        vec[24] = _safe_f(concept.cross_modal_conf)
        vec[25] = min(1.0, len(concept.associations) / 10.0)
        vec[26] = min(1.0, len(concept.context_history) / 10.0)
        if concept.embedding is not None and len(concept.embedding) > 0:
            vec[27] = min(1.0, float(np.linalg.norm(concept.embedding)) / 10.0)
        vec[28] = min(1.0, len(concept.meaning_contexts) / 5.0)

        # Encounter type [29]
        _enc_map = {"comprehension": 0.2, "production": 0.4,
                    "teaching": 0.6, "reasoning": 0.8, "bootstrap": 0.1,
                    "arc_discovery": 0.9, "social_interaction": 0.3}
        vec[29] = _enc_map.get(ctx.encounter_type, 0.5)

        return vec

    @staticmethod
    def _map_range(value: float, low: float, high: float) -> float:
        """Map [0, 1] → [low, high]."""
        return low + value * (high - low)

    # ── Internal: Training ───────────────────────��──────────────────────

    def _train_value_net(self, batch_size: int = 32, steps: int = 10) -> float:
        """Train shared V(s) via TD(0) from buffer."""
        all_t = self._buffer.get_all()
        if len(all_t) < batch_size:
            return 0.0

        total_loss = 0.0
        for _ in range(steps):
            # Sample batch
            indices = np.random.choice(len(all_t), min(batch_size, len(all_t)),
                                       replace=False)
            states = torch.FloatTensor(np.array([all_t[i].state for i in indices]))
            rewards = torch.FloatTensor([all_t[i].reward for i in indices])

            # V(s) prediction
            v_pred = self._value_net(states)

            # Target: reward (simple MC return since we don't have next-state easily)
            loss = nn.functional.mse_loss(v_pred, rewards)

            self._value_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._value_net.parameters(), 1.0)
            self._value_optimizer.step()
            total_loss += loss.item()

        return total_loss / max(steps, 1)

    def _train_consumer_policy(self, consumer: str,
                               batch_size: int = 32,
                               steps: int = 15) -> float:
        """Train consumer Q(s,a) with REINFORCE + advantage (reward - V(s))."""
        transitions = self._buffer.get_consumer_transitions(consumer)
        if len(transitions) < batch_size:
            return 0.0

        net = self._action_nets[consumer]
        optimizer = self._action_optimizers[consumer]
        total_loss = 0.0

        for _ in range(steps):
            indices = np.random.choice(len(transitions),
                                       min(batch_size, len(transitions)),
                                       replace=False)
            states = torch.FloatTensor(
                np.array([transitions[i].state for i in indices]))
            actions = torch.LongTensor([transitions[i].action for i in indices])
            rewards = torch.FloatTensor([transitions[i].reward for i in indices])

            # Compute advantage = reward - V(s)
            with torch.no_grad():
                baselines = self._value_net(states)
            advantages = rewards - baselines

            # Forward
            action_logits, _ = net(states)
            log_probs = torch.log_softmax(action_logits, dim=-1)
            selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

            # REINFORCE loss
            policy_loss = -(selected_log_probs * advantages).mean()

            # Entropy bonus for exploration
            entropy = -(torch.softmax(action_logits, dim=-1) *
                        log_probs).sum(dim=-1).mean()
            loss = policy_loss - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        return total_loss / max(steps, 1)

    def _decay_unused_concepts(self, factor: float = 0.995) -> None:
        """Gentle decay of concepts not recently encountered."""
        # Implemented via DB query — decay cross_modal_conf for stale words
        try:
            conn = sqlite3.connect(self._db_path, timeout=5.0)
            conn.execute("PRAGMA journal_mode=WAL")
            # Decay words not encountered in last 24h
            cutoff = time.time() - 86400
            conn.execute(
                "UPDATE vocabulary SET cross_modal_conf = cross_modal_conf * ? "
                "WHERE cross_modal_conf > 0.01 AND "
                "(last_encountered IS NULL OR last_encountered < ?)",
                (factor, cutoff))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug("[CGN] Decay failed: %s", e)

    # ── Sigma / SOAR / Surprise Primitives ────────────────────────────

    def _compute_reward_velocity(self, consumer: str) -> float:
        """Compute reward velocity for dynamic consolidation priority.

        Higher velocity = consumer is learning faster = should consolidate first.
        Returns float (positive = improving, negative = declining, 0 = cold/stable).
        """
        transitions = self._buffer.get_consumer_transitions(consumer, max_count=50)
        rewarded = [t for t in transitions if t.reward > 0.0]
        if len(rewarded) < 10:
            return 0.0
        # Compare mean reward of recent half vs older half
        mid = len(rewarded) // 2
        old_mean = sum(t.reward for t in rewarded[:mid]) / max(1, mid)
        new_mean = sum(t.reward for t in rewarded[mid:]) / max(1, len(rewarded) - mid)
        return new_mean - old_mean

    def record_surprise(self, consumer: str, concept_id: str,
                        magnitude: float, context: dict = None) -> None:
        """Record a high-surprise event. All consumers can query shared surprises.

        Surprise is a universal learning signal — unexpected ARC patterns,
        unexpected persona responses, unexpected word usage all generate surprise.
        """
        self._surprise_buffer.append({
            "consumer": consumer,
            "concept_id": concept_id,
            "magnitude": magnitude,
            "ts": time.time(),
            "context": context or {},
        })
        # Keep last 100 surprises
        if len(self._surprise_buffer) > 100:
            self._surprise_buffer = self._surprise_buffer[-100:]

    # UNUSED_PUBLIC_API: cross-consumer surprise sharing surface. CGN collects
    # per-consumer surprises via record_outcome, but no consumer currently
    # queries peers. Reserved for planned higher-cognition arc (cross-domain
    # analogical reasoning — see rFP_cgn_orchestrator_promotion).
    def get_shared_surprises(self, consumer: str,
                             max_count: int = 10) -> List[dict]:
        """Get surprise events from OTHER consumers (cross-domain surprise sharing)."""
        return [s for s in self._surprise_buffer[-50:]
                if s["consumer"] != consumer][-max_count:]

    # UNUSED_PUBLIC_API: concept-journey telemetry surface. Populated by
    # the CGN sigma tracker; not yet surfaced via /v4/concept-lifecycle API
    # endpoint. Reserved for observability dashboards.
    def get_concept_lifecycle(self, concept_id: str) -> Optional[dict]:
        """Get concept's journey across consumers."""
        journey = self._concept_journeys.get(concept_id)
        if not journey:
            return None
        return {
            "concept_id": concept_id,
            "first_consumer": journey["first_consumer"],
            "first_ts": journey["first_ts"],
            "consumers_seen": list(journey.get("consumers_seen", set())),
            "cross_domain": len(journey.get("consumers_seen", set())) > 1,
        }

    # ── Persistence ──────────────────────────────────────────────────

    def _save_state(self) -> None:
        """Save CGN state (nets + buffer + stats) to disk."""
        try:
            path = os.path.join(self._state_dir, "cgn_state.pt")
            state = {
                "value_net": self._value_net.state_dict(),
                "total_groundings": self._total_groundings,
                "total_rewards": self._total_rewards,
                "consolidation_count": self._consolidation_count,
                "consumers": {},
            }
            for name, net in self._action_nets.items():
                state["consumers"][name] = {
                    "action_net": net.state_dict(),
                    "config": {
                        "name": self._consumers[name].name,
                        "feature_dims": self._consumers[name].feature_dims,
                        "action_dims": self._consumers[name].action_dims,
                        "action_names": self._consumers[name].action_names,
                    },
                }

            # Save buffer (last 500 transitions for cold-start)
            buf_transitions = self._buffer.get_all(500)
            state["buffer"] = [{
                "consumer": t.consumer, "concept_id": t.concept_id,
                "state": t.state.tolist(), "action": t.action,
                "action_params": t.action_params.tolist(),
                "reward": t.reward, "epoch": t.epoch,
            } for t in buf_transitions]

            # Sigma / lifecycle / surprise persistence
            state["consumer_freq"] = self._consumer_freq
            state["concept_journeys"] = {
                k: {**v, "consumers_seen": list(v.get("consumers_seen", set()))}
                for k, v in self._concept_journeys.items()
            }
            state["surprise_buffer"] = self._surprise_buffer[-50:]

            # HAOV: save verified rules per consumer
            state["haov"] = {}
            for name, tracker in self._haov_trackers.items():
                state["haov"][name] = {
                    "verified_rules": [
                        {"rule": h.rule, "consumer": h.consumer,
                         "action_context": h.action_context,
                         "predicted_effect": h.predicted_effect,
                         "predicted_magnitude": h.predicted_magnitude,
                         "confidence": h.confidence, "tests": h.tests,
                         "confirmations": h.confirmations,
                         "falsifications": h.falsifications, "source": h.source}
                        for h in tracker._verified_rules
                    ],
                    "stats": tracker._stats,
                }

            torch.save(state, path)
            logger.info("[CGN] State saved: %d consumers, %d buffer entries",
                        len(state["consumers"]), len(buf_transitions))

            # Save HAOV stats as JSON sidecar (torch-free API access)
            if state.get("haov"):
                import json as _json
                haov_path = os.path.join(self._state_dir, "haov_stats.json")
                try:
                    with open(haov_path, "w") as _hf:
                        _json.dump(state["haov"], _hf)
                except Exception:
                    pass  # Non-critical
        except Exception as e:
            logger.warning("[CGN] Save failed: %s", e)

    def _load_state(self) -> None:
        """Load CGN state from disk if available."""
        path = os.path.join(self._state_dir, "cgn_state.pt")
        if not os.path.exists(path):
            return

        try:
            state = torch.load(path, map_location="cpu", weights_only=False)

            self._value_net.load_state_dict(state["value_net"])
            self._total_groundings = state.get("total_groundings", 0)
            self._total_rewards = state.get("total_rewards", 0.0)
            self._consolidation_count = state.get("consolidation_count", 0)

            # Restore consumer nets
            for name, cstate in state.get("consumers", {}).items():
                cfg = cstate.get("config", {})
                config = CGNConsumerConfig(
                    name=cfg.get("name", name),
                    feature_dims=cfg.get("feature_dims", 30),
                    action_dims=cfg.get("action_dims", 8),
                    action_names=cfg.get("action_names", []),
                )
                self.register_consumer(config)
                self._action_nets[name].load_state_dict(cstate["action_net"])

            # Restore buffer
            for bt in state.get("buffer", []):
                t = CGNTransition(
                    consumer=bt["consumer"],
                    concept_id=bt["concept_id"],
                    state=np.array(bt["state"], dtype=np.float32),
                    action=bt["action"],
                    action_params=np.array(bt["action_params"], dtype=np.float32),
                    reward=bt["reward"],
                    epoch=bt.get("epoch", 0),
                    timestamp=time.time(),
                )
                self._buffer.add(t)

            # Sigma / lifecycle / surprise restoration (backward-compatible)
            self._consumer_freq = state.get("consumer_freq", {})
            raw_journeys = state.get("concept_journeys", {})
            self._concept_journeys = {
                k: {**v, "consumers_seen": set(v.get("consumers_seen", []))}
                for k, v in raw_journeys.items()
            }
            self._surprise_buffer = state.get("surprise_buffer", [])

            # HAOV: restore verified rules per consumer
            for name, hdata in state.get("haov", {}).items():
                if name in self._haov_trackers:
                    tracker = self._haov_trackers[name]
                    for rd in hdata.get("verified_rules", []):
                        h = GeneralizedHypothesis(
                            rule=rd["rule"], consumer=rd["consumer"],
                            action_context=rd.get("action_context", {}),
                            predicted_effect=rd.get("predicted_effect", "unknown"),
                            predicted_magnitude=rd.get("predicted_magnitude", 0.0),
                            confidence=rd.get("confidence", 0.5),
                            tests=rd.get("tests", 0),
                            confirmations=rd.get("confirmations", 0),
                            falsifications=rd.get("falsifications", 0),
                            source=rd.get("source", "persisted"),
                        )
                        tracker._verified_rules.append(h)
                        tracker._hypotheses.append(h)
                    tracker._stats = hdata.get("stats", tracker._stats)

            logger.info("[CGN] State loaded: groundings=%d, rewards=%.2f, "
                        "consolidations=%d, buffer=%d",
                        self._total_groundings, self._total_rewards,
                        self._consolidation_count, self._buffer.size())
        except Exception as e:
            logger.warning("[CGN] Load failed (starting fresh): %s", e)
