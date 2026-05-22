"""Regression test for BUG-CGN-KNOWLEDGE-HAOV-NO-TOPIC-FIELD-20260517.

The causal-pattern path in ConceptGroundingNetwork.record_outcome forms HAOV
hypotheses for each consumer. For the `knowledge` consumer the resulting
hypothesis MUST carry `action_context["topic"]` (= the CGN concept_id, which
knowledge_worker records as topic[:50]) — otherwise knowledge_worker's
CGN_HAOV_VERIFY_REQ handler has no topic to query `knowledge_concepts` and the
verification silently returns confirmed=False forever (the dormant-signal bug).
"""
import numpy as np
import pytest

from titan_hcl.logic.cgn import ConceptGroundingNetwork, CGNConsumerConfig
from titan_hcl.logic.cgn_types import CGNTransition


def _mk_cgn(tmp_path):
    return ConceptGroundingNetwork(
        state_dir=str(tmp_path),
        causal_generator_config={
            "enabled": True,
            "window_size": 30,
            "min_n": 5,
            "magnitude_threshold": 0.05,
            "anti_pattern_enabled": True,
            "per_consumer": {"knowledge": {"min_n": 5, "window_size": 30}},
        },
    )


def _drive_promotion(cgn, consumer, topic, *, action=2, reward=0.6, n=7):
    """Seed n matching transitions + positive outcomes to promote a pattern."""
    for _ in range(n):
        t = CGNTransition(
            consumer=consumer,
            concept_id=topic,
            state=np.zeros(30, dtype=np.float32),
            action=action,
            action_params=np.zeros(8, dtype=np.float32),
            reward=0.0,
            metadata={"action_name": "deepen", "quality": reward},
        )
        cgn._buffer.add(t)
        cgn.record_outcome(consumer, topic, reward)


def test_knowledge_hypothesis_carries_topic(tmp_path):
    cgn = _mk_cgn(tmp_path)
    cgn.register_consumer(CGNConsumerConfig(name="knowledge"))
    topic = "quantum_entanglement"

    _drive_promotion(cgn, "knowledge", topic)

    tracker = cgn._haov_trackers.get("knowledge")
    assert tracker is not None, "register_consumer should create a HAOV tracker"
    assert tracker._hypotheses, (
        "causal generator should have promoted a knowledge hypothesis "
        "after >= min_n matching transitions")
    assert any(
        h.action_context.get("topic") == topic for h in tracker._hypotheses), (
        "knowledge HAOV hypothesis must carry action_context['topic'] = "
        "concept_id (BUG-CGN-KNOWLEDGE-HAOV-NO-TOPIC-FIELD) — got: "
        f"{[h.action_context for h in tracker._hypotheses]}")


def test_select_test_returns_topic_for_knowledge(tmp_path):
    """The HAOV pump calls tracker.select_test() and expects test_ctx.topic;
    confirm the round-trip surfaces topic for the knowledge consumer."""
    cgn = _mk_cgn(tmp_path)
    cgn.register_consumer(CGNConsumerConfig(name="knowledge"))
    topic = "general_relativity"

    _drive_promotion(cgn, "knowledge", topic)

    tracker = cgn._haov_trackers["knowledge"]
    # select_test has a probabilistic explore gate (test_probability=0.25) +
    # returns None while a test is active — pin it deterministic so we exercise
    # the round-trip the pump relies on (empty available_actions must NOT filter
    # out the knowledge hypothesis, which carries an "action" key from the
    # causal-pattern path; the action filter only applies when available_actions
    # is non-empty).
    tracker._test_probability = 1.0
    test_ctx = tracker.select_test({"available_actions": []})
    assert isinstance(test_ctx, dict) and test_ctx.get("topic") == topic, (
        f"select_test should surface topic for the knowledge pump; got {test_ctx}")


def test_non_knowledge_consumer_unaffected(tmp_path):
    """Other consumers keep their existing action_context shape (no topic)."""
    cgn = _mk_cgn(tmp_path)
    cgn.register_consumer(CGNConsumerConfig(name="language"))

    _drive_promotion(cgn, "language", "warm")

    tracker = cgn._haov_trackers["language"]
    assert tracker._hypotheses
    # language hypotheses must NOT get a spurious topic key from the fix.
    assert all("topic" not in h.action_context for h in tracker._hypotheses)
