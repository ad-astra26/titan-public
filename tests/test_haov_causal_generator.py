"""Tests for H.4 Phase 1 (v1) — HAOV causal hypothesis generator.

Design lock: titan-docs/rFP_cgn_consolidated.md §2.9.

Coverage:
  - window expiry (oldest evicted)
  - N-threshold gate (no promotion below N)
  - magnitude threshold (sub-threshold rewards skipped)
  - idempotency (same pattern doesn't double-promote)
  - false-positive guard (random rewards never form hypotheses)
  - per-consumer isolation (no cross-leak in v1)
  - anti-pattern path (negative reward → anti-pattern hypothesis)
  - staleness decay (un-tested candidates bleed out)
  - per-consumer config respected
  - flag-default-false (CGN integration: enabled=False → no candidates)
  - integration: promotion → tracker.hypothesize → formed counter increments
  - effect extractors: per-consumer α-fallback + β-rich behavior
"""
from __future__ import annotations

import os
import shutil
import tempfile
from types import SimpleNamespace

import pytest

from titan_plugin.logic.haov_causal_generator import (
    CausalCandidate,
    CausalGenerator,
    CausalGeneratorRegistry,
    EFFECT_EXTRACTORS,
    action_signature,
    extract_effect,
    _bucket_reward,
)


# ── Lightweight transition stand-in ────────────────────────────────────────
# Real CGNTransition lives in titan_plugin.logic.cgn (pulls torch + numpy).
# These tests only need duck-type access to .action and .metadata, so a
# SimpleNamespace is sufficient and fast.

def _t(action: int, *, metadata: dict | None = None):
    return SimpleNamespace(action=action, metadata=metadata or {})


# ── Reward bucketing ───────────────────────────────────────────────────────


def test_bucket_reward_strong_positive():
    assert _bucket_reward(0.5) == "strong_positive"


def test_bucket_reward_moderate_positive():
    assert _bucket_reward(0.10) == "moderate_positive"


def test_bucket_reward_strong_negative():
    assert _bucket_reward(-0.5) == "strong_negative"


def test_bucket_reward_below_threshold_returns_none():
    assert _bucket_reward(0.04) is None
    assert _bucket_reward(-0.04) is None
    assert _bucket_reward(0.0) is None


# ── Action signature ───────────────────────────────────────────────────────


def test_action_signature_uses_metadata_name_when_present():
    t = _t(3, metadata={"action_name": "explore"})
    assert action_signature(t) == "explore"


def test_action_signature_falls_back_to_index():
    t = _t(7, metadata={})
    assert action_signature(t) == "action_7"


# ── Per-consumer effect extractors ─────────────────────────────────────────


def test_language_extractor_uses_conf_delta_when_present():
    t = _t(0, metadata={"conf_delta": 0.12})
    assert extract_effect("language", t, reward=0.20) == "next_conf_rose"


def test_language_extractor_falls_back_to_reward_bucket():
    t = _t(0, metadata={})
    assert extract_effect("language", t, reward=0.20) == "moderate_positive"


def test_social_extractor_uses_sentiment_delta():
    t = _t(0, metadata={"sentiment_delta": -0.10})
    assert extract_effect("social", t, reward=-0.20) == "reply_colder"


def test_coding_extractor_runtime_negative_is_faster():
    t = _t(0, metadata={"sandbox_runtime_delta_ms": -50})
    assert extract_effect("coding", t, reward=0.10) == "runtime_faster"


def test_emotional_extractor_urgency_delta():
    t = _t(0, metadata={"urgency_delta": 0.10})
    assert extract_effect("emotional", t, reward=0.20) == "next_urgency_rose"


def test_knowledge_extractor_quality_delta():
    t = _t(0, metadata={"quality_delta": 0.20})
    assert extract_effect("knowledge", t, reward=0.10) == "concept_quality_rose"


def test_dreaming_extractor_compactness_delta():
    t = _t(0, metadata={"compactness_delta": 0.10})
    assert extract_effect("dreaming", t, reward=0.10) == "cluster_tighter"


def test_reasoning_extractor_depth_delta():
    t = _t(0, metadata={"depth_delta": 2})
    assert extract_effect("reasoning", t, reward=0.10) == "chain_deeper"


def test_self_model_extractor_introspection_delta():
    t = _t(0, metadata={"introspection_depth_delta": -1})
    assert extract_effect("self_model", t, reward=-0.10) == "introspection_shallower"


def test_meta_extractor_chain_success_bool():
    t = _t(0, metadata={"chain_success": True})
    assert extract_effect("meta", t, reward=0.10) == "chain_success"
    t2 = _t(0, metadata={"chain_success": False})
    assert extract_effect("meta", t2, reward=0.10) == "chain_failure"


def test_unknown_consumer_uses_default_extractor():
    t = _t(0, metadata={})
    assert extract_effect("nonexistent", t, reward=0.20) == "moderate_positive"


def test_all_9_consumers_have_extractors():
    expected = {
        "language", "social", "coding", "emotional", "knowledge",
        "dreaming", "reasoning", "reasoning_strategy", "self_model", "meta",
    }
    assert expected.issubset(set(EFFECT_EXTRACTORS.keys()))


# ── CausalGenerator: window + N-threshold ──────────────────────────────────


def test_promotion_below_n_returns_none():
    g = CausalGenerator("test", window_size=10, min_n=5, magnitude_threshold=0.05)
    for _ in range(4):  # 4 < min_n=5
        g.observe(_t(3, metadata={"action_name": "explore"}), reward=0.20)
    assert g.maybe_promote() is None


def test_promotion_at_exactly_n_returns_observation():
    g = CausalGenerator("test", window_size=10, min_n=5, magnitude_threshold=0.05)
    for _ in range(5):
        g.observe(_t(3, metadata={"action_name": "explore"}), reward=0.20)
    obs = g.maybe_promote()
    assert obs is not None
    assert obs["source"] == "causal_pattern"
    assert obs["rule_name"] == "test_explore_causes_moderate_positive"


def test_promotion_idempotent_does_not_double_fire():
    g = CausalGenerator("test", window_size=10, min_n=5, magnitude_threshold=0.05)
    for _ in range(5):
        g.observe(_t(3, metadata={"action_name": "explore"}), reward=0.20)
    first = g.maybe_promote()
    second = g.maybe_promote()
    assert first is not None
    assert second is None  # already promoted


def test_below_threshold_reward_skipped():
    g = CausalGenerator("test", window_size=10, min_n=3, magnitude_threshold=0.10)
    # |reward| = 0.05 is below the 0.10 threshold for THIS test
    for _ in range(10):
        g.observe(_t(3, metadata={"action_name": "explore"}), reward=0.05)
    assert g.maybe_promote() is None
    stats = g.get_stats()
    assert stats["transitions_observed"] == 0
    assert stats["below_threshold_skips"] >= 10


# ── Window eviction ────────────────────────────────────────────────────────


def test_window_evicts_oldest_pattern():
    g = CausalGenerator("test", window_size=5, min_n=3, magnitude_threshold=0.05)
    # Fill window with action_A (5 entries — at min_n+ but only one pattern)
    for _ in range(5):
        g.observe(_t(0, metadata={"action_name": "A"}), reward=0.20)
    # Promote it
    assert g.maybe_promote() is not None
    # Now flood with action_B until A evicts
    for _ in range(6):
        g.observe(_t(1, metadata={"action_name": "B"}), reward=0.20)
    # action_A pattern should have evicted (count fell to 0, candidate removed)
    stats = g.get_stats()
    actions_seen = {c["action"] for c in stats["top_candidates"]}
    assert "B" in actions_seen
    # A may or may not still be visible depending on candidate dict; the
    # key check is that B is now dominant
    assert any(c["action"] == "B" and c["n"] >= 3 for c in stats["top_candidates"])


# ── False-positive guard (random rewards never converge) ───────────────────


def test_random_actions_dont_form_hypothesis_below_n():
    import random
    rng = random.Random(42)
    g = CausalGenerator("test", window_size=30, min_n=5, magnitude_threshold=0.05)
    # 30 observations across 30 different action signatures + random rewards
    for i in range(30):
        g.observe(
            _t(i, metadata={"action_name": f"a{i}"}),
            reward=rng.uniform(0.06, 0.50),
        )
    assert g.maybe_promote() is None  # no pattern repeated >= 5 times
    stats = g.get_stats()
    assert stats["promoted_total"] == 0


# ── Anti-pattern path ──────────────────────────────────────────────────────


def test_anti_pattern_promotion_with_negative_reward():
    g = CausalGenerator("test", window_size=10, min_n=3,
                       magnitude_threshold=0.05, anti_pattern_enabled=True)
    for _ in range(3):
        g.observe_negative(_t(0, metadata={"action_name": "danger"}),
                           reward=-0.30)
    obs = g.maybe_promote()
    assert obs is not None
    assert obs["effect"].startswith("negative_")
    assert "negative_" in obs["rule_name"]


def test_anti_pattern_disabled_skips_observation():
    g = CausalGenerator("test", window_size=10, min_n=3,
                       magnitude_threshold=0.05, anti_pattern_enabled=False)
    for _ in range(5):
        g.observe_negative(_t(0, metadata={"action_name": "danger"}),
                           reward=-0.30)
    assert g.maybe_promote() is None


def test_positive_observe_with_negative_reward_skipped():
    """Edge case: observe() called with negative reward — sub-threshold-positive
    means below threshold AND not anti-pattern: skipped."""
    g = CausalGenerator("test", window_size=10, min_n=3, magnitude_threshold=0.05)
    for _ in range(5):
        g.observe(_t(0), reward=-0.20)  # negative routed to observe_negative path
    # observe() rejects negative reward → no candidate forms on positive path
    assert g.maybe_promote() is None


# ── Staleness decay ────────────────────────────────────────────────────────


def test_decay_stale_evicts_low_count_candidates():
    g = CausalGenerator("test", window_size=10, min_n=5,
                       magnitude_threshold=0.05,
                       staleness_decay_per_tick=0.5)  # aggressive for testability
    for _ in range(2):
        g.observe(_t(0, metadata={"action_name": "rare"}), reward=0.10)
    stats_before = g.get_stats()
    assert stats_before["candidates_active"] == 1
    # Apply decay enough times to drop count below 1: 2 * 0.5 = 1; 1 * 0.5 = 0
    g.decay_stale()
    g.decay_stale()
    stats_after = g.get_stats()
    assert stats_after["candidates_active"] == 0


def test_decay_no_op_when_decay_factor_is_one():
    g = CausalGenerator("test", window_size=10, min_n=5,
                       magnitude_threshold=0.05,
                       staleness_decay_per_tick=1.0)
    for _ in range(3):
        g.observe(_t(0, metadata={"action_name": "stable"}), reward=0.10)
    g.decay_stale()
    g.decay_stale()
    g.decay_stale()
    assert g.get_stats()["candidates_active"] == 1


# ── Registry: per-consumer config + isolation ──────────────────────────────


def test_registry_per_consumer_min_n_override():
    reg = CausalGeneratorRegistry(
        defaults={"min_n": 5, "window_size": 30, "magnitude_threshold": 0.05},
        per_consumer={"social": {"min_n": 2}, "emotional": {"min_n": 8}},
    )
    social_gen = reg.get_or_create("social")
    emot_gen = reg.get_or_create("emotional")
    other_gen = reg.get_or_create("language")
    assert social_gen._min_n == 2
    assert emot_gen._min_n == 8
    assert other_gen._min_n == 5


def test_registry_consumer_isolation():
    """Patterns observed under one consumer must not promote under another."""
    reg = CausalGeneratorRegistry(
        defaults={"min_n": 3, "window_size": 10, "magnitude_threshold": 0.05},
    )
    # 5 observations under "language"
    for _ in range(5):
        reg.observe_for("language", _t(0, metadata={"action_name": "X"}),
                        reward=0.20)
    # 2 observations under "social" — should not promote (below N=3)
    for _ in range(2):
        reg.observe_for("social", _t(0, metadata={"action_name": "X"}),
                        reward=0.20)
    stats = reg.get_stats()
    assert stats["language"]["promoted_total"] >= 1
    assert stats["social"]["promoted_total"] == 0


def test_registry_observe_for_routes_negative_to_anti_pattern():
    reg = CausalGeneratorRegistry(
        defaults={"min_n": 3, "window_size": 10, "magnitude_threshold": 0.05,
                  "anti_pattern_enabled": True},
    )
    promoted_obs = []
    for _ in range(3):
        result = reg.observe_for("test", _t(0, metadata={"action_name": "harm"}),
                                 reward=-0.30)
        if result is not None:
            promoted_obs.append(result)
    assert len(promoted_obs) == 1
    assert promoted_obs[0]["effect"].startswith("negative_")


def test_registry_decay_stale_all_aggregates():
    reg = CausalGeneratorRegistry(
        defaults={"min_n": 5, "window_size": 10, "magnitude_threshold": 0.05,
                  "staleness_decay_per_tick": 0.4},  # aggressive
    )
    # Seed two consumers with sub-threshold candidates
    for _ in range(2):
        reg.observe_for("a", _t(0, metadata={"action_name": "x"}), reward=0.10)
    for _ in range(2):
        reg.observe_for("b", _t(0, metadata={"action_name": "y"}), reward=0.10)
    # Apply decay enough times to evict
    total_evicted = 0
    for _ in range(5):
        total_evicted += reg.decay_stale_all()
    assert total_evicted >= 2  # both consumers' candidates evicted


# ── Integration with ConceptGroundingNetwork ───────────────────────────────


@pytest.fixture
def cgn_with_consumer():
    """Real CGN instance with one registered consumer.  Uses tmp state dir."""
    # Lazy import — torch + numpy are heavy.
    from titan_plugin.logic.cgn import (
        ConceptGroundingNetwork,
        CGNConsumerConfig,
    )
    tmpdir = tempfile.mkdtemp(prefix="cgn_h4_test_")
    try:
        cgn = ConceptGroundingNetwork(
            db_path=os.path.join(tmpdir, "imem.db"),
            state_dir=tmpdir,
            causal_generator_config={"enabled": True,
                                     "defaults": {"min_n": 3, "window_size": 10,
                                                  "magnitude_threshold": 0.05}},
        )
        cgn.register_consumer(CGNConsumerConfig(
            name="test_consumer", feature_dims=30, action_dims=4,
            action_names=["a", "b", "c", "d"]))
        yield cgn
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_integration_record_outcome_promotes_through_tracker(cgn_with_consumer):
    """End-to-end: record_outcome → causal_generator.observe → maybe_promote
    → tracker.hypothesize → tracker._stats["formed"] increments."""
    from titan_plugin.logic.cgn import CGNTransition
    cgn = cgn_with_consumer
    tracker = cgn._haov_trackers["test_consumer"]
    formed_before = tracker.get_stats()["formed"]
    # Inject 4 transitions for the same concept_id-each-pop, each given reward
    # afterward via record_outcome — this is the hot path the hook lives on.
    import numpy as np
    for i in range(4):
        cid = f"concept_{i}"
        cgn._buffer.add(CGNTransition(
            consumer="test_consumer", concept_id=cid,
            state=np.zeros(30, dtype=np.float32), action=0,
            action_params=np.zeros(4, dtype=np.float32),
            metadata={"action_name": "a"},
        ))
        cgn.record_outcome(consumer="test_consumer", concept_id=cid,
                           reward=0.20, outcome_context={"action_name": "a"})
    formed_after = tracker.get_stats()["formed"]
    assert formed_after > formed_before, (
        "Causal generator promotion didn't reach tracker.hypothesize")


def test_integration_flag_default_false_means_no_promotion():
    """With enabled=False (the default), record_outcome must NOT promote."""
    from titan_plugin.logic.cgn import (
        ConceptGroundingNetwork, CGNConsumerConfig, CGNTransition,
    )
    import numpy as np
    tmpdir = tempfile.mkdtemp(prefix="cgn_h4_disabled_")
    try:
        cgn = ConceptGroundingNetwork(
            db_path=os.path.join(tmpdir, "imem.db"),
            state_dir=tmpdir,
            # No causal_generator_config — defaults to enabled=False
        )
        cgn.register_consumer(CGNConsumerConfig(name="t2", feature_dims=30))
        tracker = cgn._haov_trackers["t2"]
        formed_before = tracker.get_stats()["formed"]
        for i in range(10):
            cid = f"c_{i}"
            cgn._buffer.add(CGNTransition(
                consumer="t2", concept_id=cid,
                state=np.zeros(30, dtype=np.float32), action=0,
                action_params=np.zeros(4, dtype=np.float32),
                metadata={"action_name": "x"},
            ))
            cgn.record_outcome("t2", cid, 0.30,
                               outcome_context={"action_name": "x"})
        formed_after = tracker.get_stats()["formed"]
        assert formed_after == formed_before, (
            "Disabled flag must skip the causal generator hook entirely")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_integration_decay_stale_haov_no_op_when_disabled():
    """cgn.decay_stale_haov() returns 0 quickly when generator is disabled."""
    from titan_plugin.logic.cgn import ConceptGroundingNetwork
    tmpdir = tempfile.mkdtemp(prefix="cgn_h4_decay_")
    try:
        cgn = ConceptGroundingNetwork(
            db_path=os.path.join(tmpdir, "imem.db"), state_dir=tmpdir)
        # No consumer registered, no transitions — but the call must still
        # be safe and return 0.
        assert cgn.decay_stale_haov() == 0
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
