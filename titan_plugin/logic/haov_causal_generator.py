"""Causal hypothesis generator — Phase 1 (v1) of H.4.

Per-consumer pattern miner that watches CGN transitions and proposes causal
hypotheses ("if action X is taken, effect Y follows") to feed the existing
HAOV machinery via GeneralizedHAOVTracker.hypothesize().

The HAOV pipeline (test pump, verifiers, dest map) needs zero changes — this
module only adds a second feeding source alongside the impasse path.

Design lock: titan-docs/rFP_cgn_consolidated.md §2.9 (DESIGN-LOCKED 2026-04-28 PM).

Phase 1 (this module): per-consumer detector with anti-pattern path + staleness
decay. Phase 2 will add cross-consumer abstraction via the 12-dim shared
ConsumerActionNet embedding space; that is OUT of scope for this module.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Deque, Dict, List, Optional, Tuple

if TYPE_CHECKING:  # avoid circular import — cgn.py depends on this module
    from titan_plugin.logic.cgn import CGNTransition

logger = logging.getLogger(__name__)


# ── Effect bucketing — α-style reward buckets ──────────────────────────────
# Shared across all consumers as the default effect-signature.  Per-consumer
# extractors can override with richer β-style effects when transition.metadata
# carries domain-specific signals.

_EFFECT_THRESHOLDS = (0.30, 0.05)  # strong / moderate boundary


def _bucket_reward(reward: float) -> Optional[str]:
    """Return reward-magnitude bucket label, or None if below noise floor."""
    a = abs(reward)
    strong, moderate = _EFFECT_THRESHOLDS
    if a < moderate:
        return None  # noise — generator skips
    sign = "positive" if reward > 0 else "negative"
    tier = "strong" if a >= strong else "moderate"
    return f"{tier}_{sign}"


# ── Per-consumer effect extractors (β-style domain-specific) ───────────────
# Each extractor receives the transition + reward, returns an effect-signature
# string or None.  Default behavior: reward-bucket.  Consumers may include
# domain-specific metadata fields (e.g., language: conf_delta in metadata) to
# produce richer effect signatures; when those fields are absent, extractors
# fall back to reward-bucket.  This is an additive contract — existing
# consumers need no changes.


def _extract_default(transition: "CGNTransition", reward: float) -> Optional[str]:
    return _bucket_reward(reward)


def _extract_language(transition: "CGNTransition", reward: float) -> Optional[str]:
    md = transition.metadata or {}
    cd = md.get("conf_delta")
    if isinstance(cd, (int, float)) and abs(cd) >= 0.05:
        sign = "rose" if cd > 0 else "dropped"
        return f"next_conf_{sign}"
    return _bucket_reward(reward)


def _extract_social(transition: "CGNTransition", reward: float) -> Optional[str]:
    md = transition.metadata or {}
    sd = md.get("sentiment_delta")
    if isinstance(sd, (int, float)) and abs(sd) >= 0.05:
        sign = "warmer" if sd > 0 else "colder"
        return f"reply_{sign}"
    return _bucket_reward(reward)


def _extract_coding(transition: "CGNTransition", reward: float) -> Optional[str]:
    md = transition.metadata or {}
    rt = md.get("sandbox_runtime_delta_ms")
    if isinstance(rt, (int, float)) and abs(rt) >= 10:
        sign = "faster" if rt < 0 else "slower"  # negative ms = faster = good
        return f"runtime_{sign}"
    return _bucket_reward(reward)


def _extract_emotional(transition: "CGNTransition", reward: float) -> Optional[str]:
    md = transition.metadata or {}
    ud = md.get("urgency_delta")
    if isinstance(ud, (int, float)) and abs(ud) >= 0.05:
        sign = "rose" if ud > 0 else "fell"
        return f"next_urgency_{sign}"
    return _bucket_reward(reward)


def _extract_knowledge(transition: "CGNTransition", reward: float) -> Optional[str]:
    md = transition.metadata or {}
    qd = md.get("quality_delta")
    if isinstance(qd, (int, float)) and abs(qd) >= 0.05:
        sign = "rose" if qd > 0 else "dropped"
        return f"concept_quality_{sign}"
    return _bucket_reward(reward)


def _extract_dreaming(transition: "CGNTransition", reward: float) -> Optional[str]:
    md = transition.metadata or {}
    cd = md.get("compactness_delta")
    if isinstance(cd, (int, float)) and abs(cd) >= 0.05:
        sign = "tighter" if cd > 0 else "looser"
        return f"cluster_{sign}"
    return _bucket_reward(reward)


def _extract_reasoning(transition: "CGNTransition", reward: float) -> Optional[str]:
    md = transition.metadata or {}
    dd = md.get("depth_delta")
    if isinstance(dd, (int, float)) and abs(dd) >= 1:
        sign = "deeper" if dd > 0 else "shallower"
        return f"chain_{sign}"
    return _bucket_reward(reward)


def _extract_self_model(transition: "CGNTransition", reward: float) -> Optional[str]:
    md = transition.metadata or {}
    id_ = md.get("introspection_depth_delta")
    if isinstance(id_, (int, float)) and abs(id_) >= 1:
        sign = "deeper" if id_ > 0 else "shallower"
        return f"introspection_{sign}"
    return _bucket_reward(reward)


def _extract_meta(transition: "CGNTransition", reward: float) -> Optional[str]:
    md = transition.metadata or {}
    cs = md.get("chain_success")
    if isinstance(cs, bool):
        return "chain_success" if cs else "chain_failure"
    return _bucket_reward(reward)


EFFECT_EXTRACTORS: Dict[str, Callable[["CGNTransition", float], Optional[str]]] = {
    "language": _extract_language,
    "social": _extract_social,
    "coding": _extract_coding,
    "emotional": _extract_emotional,
    "knowledge": _extract_knowledge,
    "dreaming": _extract_dreaming,
    "reasoning": _extract_reasoning,
    "reasoning_strategy": _extract_reasoning,
    "self_model": _extract_self_model,
    "meta": _extract_meta,
}


def extract_effect(consumer: str, transition: "CGNTransition", reward: float) -> Optional[str]:
    """Resolve effect signature for a (consumer, transition) — public entry."""
    return EFFECT_EXTRACTORS.get(consumer, _extract_default)(transition, reward)


def action_signature(transition: "CGNTransition") -> str:
    """Stable action signature for a transition.

    Uses metadata['action_name'] when populated, else falls back to the
    integer action index.  The result is a per-consumer string — different
    consumers naturally produce disjoint signature spaces in v1.
    """
    md = transition.metadata or {}
    name = md.get("action_name")
    if isinstance(name, str) and name:
        return name
    return f"action_{int(transition.action)}"


# ── CausalCandidate ────────────────────────────────────────────────────────


@dataclass
class CausalCandidate:
    """A pattern observed N times in the recent window, not yet promoted."""
    action_sig: str
    effect_sig: str
    is_anti_pattern: bool
    observed_n: int = 0
    reward_sum: float = 0.0
    first_seen_idx: int = 0
    last_seen_idx: int = 0
    promoted: bool = False  # set True after maybe_promote returns this candidate

    @property
    def reward_mean(self) -> float:
        return self.reward_sum / max(1, self.observed_n)

    def to_observation(self, consumer: str) -> dict:
        """Build the dict expected by GeneralizedHAOVTracker.hypothesize()."""
        prefix = "negative_" if self.is_anti_pattern else ""
        rule_name = (
            f"{consumer}_{self.action_sig}_causes_{prefix}{self.effect_sig}"
        )
        effect_label = f"{prefix}{self.effect_sig}"
        return {
            "effect": effect_label,
            "magnitude": min(1.0, abs(self.reward_mean)),
            "rule_name": rule_name,
            "source": "causal_pattern",
        }


# ── CausalGenerator ────────────────────────────────────────────────────────


class CausalGenerator:
    """Per-consumer sliding-window pattern miner.

    Watches (action_sig, effect_sig, reward) tuples for one consumer.  When
    the same (action_sig, effect_sig) pair shows up at least `min_n` times
    inside the most-recent `window_size` transitions, the pair becomes
    eligible for promotion to a causal hypothesis.  `maybe_promote()` returns
    the observation dict; the caller (cgn.py) feeds it to the consumer's
    HAOV tracker via `tracker.hypothesize(action_context, observation)`.

    Anti-patterns ("if I do X, NEGATIVE Y follows") use the same machinery —
    `observe_negative()` is the entry point; promoted hypotheses get a
    `negative_` prefix in their effect label and rule name.

    Staleness decay: candidates that haven't been re-counted in the window
    naturally evict.  `decay_stale()` applies a multiplicative decay to
    counts — used when the cgn_worker tick wants to slow-bleed unmoving
    candidates regardless of window position.
    """

    def __init__(
        self,
        consumer: str,
        *,
        window_size: int = 30,
        min_n: int = 5,
        magnitude_threshold: float = 0.05,
        anti_pattern_enabled: bool = True,
        staleness_decay_per_tick: float = 0.999,
    ) -> None:
        self._consumer = consumer
        self._window_size = max(1, int(window_size))
        self._min_n = max(1, int(min_n))
        self._magnitude_threshold = float(magnitude_threshold)
        self._anti_pattern_enabled = bool(anti_pattern_enabled)
        self._staleness_decay = float(staleness_decay_per_tick)

        # Sliding window of recent (action_sig, effect_sig, is_anti) keys.
        # Eviction = decrement candidate count when oldest entry leaves.
        self._window: Deque[Tuple[str, str, bool]] = deque(maxlen=self._window_size)

        # Active candidates — (action_sig, effect_sig, is_anti) → CausalCandidate.
        self._candidates: Dict[Tuple[str, str, bool], CausalCandidate] = {}

        # Monotonic transition counter for first/last-seen indices.
        self._idx: int = 0

        # Stats — surfaced via get_stats(), used by arch_map causal-generator.
        self._stats: Dict[str, int] = {
            "observed": 0,
            "below_threshold": 0,
            "candidates_active": 0,
            "promoted": 0,
            "anti_patterns_promoted": 0,
            "decayed_out": 0,
        }

    # ── observation entrypoints ────────────────────────────────────────

    def observe(self, transition: "CGNTransition", reward: float) -> None:
        """Record a positive-effect transition.  No-op if reward is sub-threshold."""
        if reward <= self._magnitude_threshold:
            self._stats["below_threshold"] += 1
            return
        self._record(transition, reward, is_anti=False)

    def observe_negative(self, transition: "CGNTransition", reward: float) -> None:
        """Record a negative-effect (anti-pattern) transition.

        No-op when anti-pattern detection is disabled or |reward| is sub-threshold.
        """
        if not self._anti_pattern_enabled:
            return
        if reward >= -self._magnitude_threshold:
            self._stats["below_threshold"] += 1
            return
        self._record(transition, reward, is_anti=True)

    # ── promotion check ───────────────────────────────────────────────

    def maybe_promote(self) -> Optional[dict]:
        """Return one ready-to-hypothesize observation dict, or None.

        A candidate is ready when observed_n >= min_n and it has not been
        promoted yet.  Idempotent: subsequent calls for the same candidate
        return None until the candidate's window-count grows again.
        """
        for key, cand in self._candidates.items():
            if cand.promoted or cand.observed_n < self._min_n:
                continue
            cand.promoted = True
            self._stats["promoted"] += 1
            if cand.is_anti_pattern:
                self._stats["anti_patterns_promoted"] += 1
            return cand.to_observation(self._consumer)
        return None

    # ── staleness decay ───────────────────────────────────────────────

    def decay_stale(self) -> int:
        """Apply multiplicative decay to candidate window-counts.

        Used by the cgn_worker tick to slow-bleed candidates that aren't
        getting fresh observations.  Returns the number of candidates that
        decayed below 1 and were evicted.
        """
        if self._staleness_decay >= 1.0:
            return 0
        evicted = 0
        for key in list(self._candidates.keys()):
            cand = self._candidates[key]
            cand.observed_n = int(cand.observed_n * self._staleness_decay)
            if cand.observed_n < 1:
                self._candidates.pop(key, None)
                evicted += 1
        if evicted:
            self._stats["decayed_out"] += evicted
        self._stats["candidates_active"] = len(self._candidates)
        return evicted

    # ── telemetry ─────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Snapshot of internal counters for arch_map / observability."""
        return {
            "consumer": self._consumer,
            "window_size": self._window_size,
            "min_n": self._min_n,
            "candidates_active": len(self._candidates),
            "candidates_ready": sum(
                1 for c in self._candidates.values()
                if c.observed_n >= self._min_n and not c.promoted),
            "transitions_observed": self._stats["observed"],
            "below_threshold_skips": self._stats["below_threshold"],
            "promoted_total": self._stats["promoted"],
            "anti_patterns_promoted": self._stats["anti_patterns_promoted"],
            "decayed_out": self._stats["decayed_out"],
            "top_candidates": [
                {
                    "action": c.action_sig,
                    "effect": c.effect_sig,
                    "anti": c.is_anti_pattern,
                    "n": c.observed_n,
                    "reward_mean": round(c.reward_mean, 3),
                    "promoted": c.promoted,
                }
                for c in sorted(
                    self._candidates.values(),
                    key=lambda x: x.observed_n,
                    reverse=True,
                )[:5]
            ],
        }

    # ── internal ──────────────────────────────────────────────────────

    def _record(self, transition: "CGNTransition", reward: float, *, is_anti: bool) -> None:
        action_sig = action_signature(transition)
        effect_sig = extract_effect(self._consumer, transition, reward)
        if effect_sig is None:  # below-threshold — extractor returned None
            self._stats["below_threshold"] += 1
            return

        self._idx += 1
        self._stats["observed"] += 1
        key = (action_sig, effect_sig, is_anti)

        # Eviction: if window is full, drop the oldest key BEFORE appending.
        if len(self._window) == self._window.maxlen:
            old_key = self._window[0]  # leftmost — leaving on next append
            old_cand = self._candidates.get(old_key)
            if old_cand is not None:
                old_cand.observed_n -= 1
                # Don't subtract reward_sum precisely — running average drifts
                # slightly under eviction.  Acceptable for v1; tightening to
                # exact mean would require storing per-instance rewards.
                if old_cand.observed_n <= 0:
                    self._candidates.pop(old_key, None)

        # Append (this evicts the oldest from the deque automatically).
        self._window.append(key)

        # Increment / create candidate.
        cand = self._candidates.get(key)
        if cand is None:
            cand = CausalCandidate(
                action_sig=action_sig,
                effect_sig=effect_sig,
                is_anti_pattern=is_anti,
                observed_n=0,
                reward_sum=0.0,
                first_seen_idx=self._idx,
                last_seen_idx=self._idx,
            )
            self._candidates[key] = cand
        cand.observed_n += 1
        cand.reward_sum += reward
        cand.last_seen_idx = self._idx

        # If this candidate had previously been promoted but its count climbed
        # back up to threshold, allow re-promotion (e.g., after staleness
        # decay knocked it down and fresh evidence brought it back).
        if cand.promoted and cand.observed_n < self._min_n:
            cand.promoted = False

        self._stats["candidates_active"] = len(self._candidates)


# ── Multi-consumer manager (convenience wrapper) ───────────────────────────


class CausalGeneratorRegistry:
    """Per-consumer registry of CausalGenerators owned by ConceptGroundingNetwork.

    Mirrors the shape of CGN's `_haov_trackers` dict.  cgn.py instantiates
    one of these alongside the HAOV tracker registry; record_outcome() routes
    transitions through `.observe_for(consumer, ...)`.
    """

    def __init__(self, defaults: Optional[dict] = None,
                 per_consumer: Optional[Dict[str, dict]] = None) -> None:
        self._defaults = defaults or {}
        self._per_consumer = per_consumer or {}
        self._generators: Dict[str, CausalGenerator] = {}

    def get_or_create(self, consumer: str) -> CausalGenerator:
        gen = self._generators.get(consumer)
        if gen is not None:
            return gen
        cfg = dict(self._defaults)
        cfg.update(self._per_consumer.get(consumer, {}))
        gen = CausalGenerator(
            consumer,
            window_size=int(cfg.get("window_size", 30)),
            min_n=int(cfg.get("min_n", 5)),
            magnitude_threshold=float(cfg.get("magnitude_threshold", 0.05)),
            anti_pattern_enabled=bool(cfg.get("anti_pattern_enabled", True)),
            staleness_decay_per_tick=float(cfg.get("staleness_decay_per_tick", 0.999)),
        )
        self._generators[consumer] = gen
        return gen

    def observe_for(self, consumer: str, transition: "CGNTransition",
                    reward: float) -> Optional[dict]:
        """Route a transition + reward to the consumer's generator.

        Returns a ready-to-hypothesize observation dict if a candidate was
        promoted by this transition, else None.  Anti-patterns and positive
        patterns share the same call site — direction is selected by reward
        sign.
        """
        gen = self.get_or_create(consumer)
        if reward > 0:
            gen.observe(transition, reward)
        elif reward < 0:
            gen.observe_negative(transition, reward)
        return gen.maybe_promote()

    def decay_stale_all(self) -> int:
        """Per-tick staleness decay across all known generators."""
        total = 0
        for gen in self._generators.values():
            total += gen.decay_stale()
        return total

    def get_stats(self) -> Dict[str, dict]:
        return {c: g.get_stats() for c, g in self._generators.items()}
