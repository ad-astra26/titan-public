"""
Regression tests for BUG #3 Phase C §23.1 completion — the 3 remaining
peer CGN_CROSS_INSIGHT publishers (reasoning, self_model, reasoning_strategy).

Yesterday's Q2 fix (commit 21d3a717) wired 5 of 8 consumers:
meta, coding, language, social, knowledge. The remaining 3 (reasoning,
self_model, reasoning_strategy) were deferred per §23.1 "no
CGNConsumerClient instances yet." Today's fix uses the module-level
`emit_chain_outcome_insight()` helper at each existing CGN_TRANSITION
emit site with a meaningful reward, matching the language_worker /
knowledge_worker wiring pattern.

Wired sites (spirit_worker.py):
1. :4059 reasoning_strategy (outcome_score reward)
2. :5642 self_model (self-prediction: +0.5 confirmed / -0.1 miss)
3. :6278 self_model (introspection: confidence * 0.3)
4. :5683 reasoning (ARC episode reward scaled)
5. :8583 reasoning (kin-exchange hypothesis confidence * 0.3)

2026-06-10: the emit gate changed from the fixed `|reward-0.5|>0.3` to an
EMERGENT reward-surprise gate (|reward - running_mean| >= k·running_std) — see
cgn_consumer_client.emit_chain_outcome_insight. So these tests now warm up a
per-consumer baseline first; the INTENT (accept the 3 consumers, filter
non-informative outcomes, pass surprising negatives) is unchanged.
"""

import queue

import pytest

from titan_hcl.logic.cgn_consumer_client import (
    emit_chain_outcome_insight, _PEER_REWARD_MIN_N,
    _PEER_REWARD_HIST, _PEER_CROSS_INSIGHT_RATE_STATE,
)


def _reset(consumer):
    _PEER_REWARD_HIST.pop(consumer, None)
    _PEER_CROSS_INSIGHT_RATE_STATE.pop(consumer, None)


def _warmup(consumer, q, base=0.5):
    """Feed MIN_N baseline samples so the surprise gate has a baseline.
    Warm-up emits nothing (the gate stays silent until a baseline forms)."""
    for _ in range(_PEER_REWARD_MIN_N):
        emit_chain_outcome_insight(q, "spirit", consumer, base)


def test_emit_chain_outcome_insight_accepts_reasoning_strategy_consumer():
    """Module-level helper accepts the 3 new consumer names; a reward that
    surprises the consumer's baseline emits a chain_outcome CGN_CROSS_INSIGHT."""
    for consumer in ("reasoning_strategy", "self_model", "reasoning"):
        _reset(consumer)
        q = queue.Queue()
        _warmup(consumer, q, base=0.5)
        assert q.empty(), "warm-up must not emit"
        result = emit_chain_outcome_insight(
            q, "spirit", consumer, terminal_reward=0.9, ctx={"test": True})
        assert result is True, f"{consumer} should emit on a surprising reward"
        msg = q.get_nowait()
        assert msg["type"] == "CGN_CROSS_INSIGHT"
        assert msg["payload"]["origin_consumer"] == consumer
        assert msg["payload"]["insight_type"] == "chain_outcome"
        _reset(consumer)


def test_emit_chain_outcome_insight_filters_non_informative_rewards():
    """Emergent gate: a reward WITHIN the consumer's running baseline is filtered
    (relative surprise — replaces the old fixed |r-0.5|>0.3)."""
    c = "reasoning_test1"
    _reset(c)
    q = queue.Queue()
    _warmup(c, q, base=0.5)
    # Within baseline (|0.52 - 0.5| < 1σ floor of 0.05) → filtered
    result = emit_chain_outcome_insight(q, "spirit", c, terminal_reward=0.52)
    assert result is False
    assert q.empty()
    # Surprising (|0.9 - 0.5| >> 1σ) → passes
    result = emit_chain_outcome_insight(q, "spirit", c, terminal_reward=0.9)
    assert result is True
    _reset(c)


def test_bug3_wire_points_migrated_off_deleted_spirit_worker():
    """D-SPEC-116: the BUG #3 §23.1 wire points were source-comment markers
    inside spirit_worker.py, now DELETED. The capability (emit_chain_outcome_
    insight) migrated with the engines and lives in knowledge_worker +
    language_worker; behavior is covered by the test_emit_chain_outcome_insight_*
    tests in this file. The obsolete source-grep on spirit_worker.py is retired."""
    import importlib.util
    import inspect
    assert importlib.util.find_spec("titan_hcl.modules.spirit_worker") is None
    from titan_hcl.modules import knowledge_worker, language_worker
    assert ("emit_chain_outcome_insight" in inspect.getsource(knowledge_worker)
            or "emit_chain_outcome_insight" in inspect.getsource(language_worker))


def test_emit_chain_outcome_insight_negative_rewards_pass_informative_filter():
    """Negative rewards (self_model's -0.1 miss) surprise a ~0.5 baseline and
    emit. |(-0.1) - 0.5| = 0.6 >> 1σ → informative."""
    c = "self_model_neg_test"
    _reset(c)
    q = queue.Queue()
    _warmup(c, q, base=0.5)
    result = emit_chain_outcome_insight(
        q, "spirit", c, terminal_reward=-0.1,
        ctx={"source": "self_prediction", "confirmed": False})
    assert result is True, "negative rewards should surprise a positive baseline"
    msg = q.get_nowait()
    assert msg["payload"]["terminal_reward"] == -0.1
    _reset(c)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-p", "no:anchorpy"])
