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
"""

import pytest


def test_emit_chain_outcome_insight_accepts_reasoning_strategy_consumer():
    """Module-level helper accepts the 3 new consumer names."""
    from titan_plugin.logic.cgn_consumer_client import emit_chain_outcome_insight
    import queue

    for consumer in ("reasoning_strategy", "self_model", "reasoning"):
        q = queue.Queue()
        # Use a high-magnitude reward to pass the informative filter
        result = emit_chain_outcome_insight(
            q, "spirit", consumer, terminal_reward=0.9, ctx={"test": True})
        # Rate-gate is per-consumer; fresh queue → first emit succeeds
        assert result is True, f"{consumer} emit should return True for informative reward"
        msg = q.get_nowait()
        assert msg["type"] == "CGN_CROSS_INSIGHT"
        assert msg["payload"]["origin_consumer"] == consumer
        assert msg["payload"]["insight_type"] == "chain_outcome"


def test_emit_chain_outcome_insight_filters_non_informative_rewards():
    """Rewards near 0.5 (|r - 0.5| <= 0.3) are filtered as non-informative."""
    from titan_plugin.logic.cgn_consumer_client import (
        emit_chain_outcome_insight, _PEER_CROSS_INSIGHT_RATE_STATE)
    import queue

    # Clear rate state so tests don't interfere
    _PEER_CROSS_INSIGHT_RATE_STATE.pop("reasoning_test1", None)
    q = queue.Queue()

    # Non-informative (reward 0.6, delta=0.1 <= 0.3) → filtered
    result = emit_chain_outcome_insight(
        q, "spirit", "reasoning_test1", terminal_reward=0.6)
    assert result is False
    assert q.empty()

    # Informative (reward 0.9, delta=0.4 > 0.3) → passes
    result = emit_chain_outcome_insight(
        q, "spirit", "reasoning_test1", terminal_reward=0.9)
    assert result is True


def test_bug3_wire_points_exist_in_source():
    """Every BUG #3 Phase C §23.1 wire point has a reference comment +
    an emit_chain_outcome_insight invocation. Static check ensures the
    fixes are preserved across refactors."""
    from pathlib import Path
    src = (Path(__file__).parent.parent
           / "titan_plugin" / "modules" / "spirit_worker.py").read_text()

    # BUG #3 reference markers must all appear
    bug3_markers = src.count("BUG #3 Phase C §23.1")
    assert bug3_markers >= 5, \
        f"Expected >=5 BUG #3 Phase C wire points, found {bug3_markers}"

    # Each of the 3 consumer names must appear in an emit_chain_outcome_insight context
    # (check for adjacency — the consumer_name arg follows "name,")
    assert '"reasoning_strategy",' in src
    assert '"self_model",' in src
    assert '"reasoning",' in src


def test_emit_chain_outcome_insight_negative_rewards_pass_informative_filter():
    """Negative rewards (like self_model's -0.1 miss) need to pass the
    informative filter too. |(-0.1) - 0.5| = 0.6 > 0.3 → informative."""
    from titan_plugin.logic.cgn_consumer_client import (
        emit_chain_outcome_insight, _PEER_CROSS_INSIGHT_RATE_STATE)
    import queue

    _PEER_CROSS_INSIGHT_RATE_STATE.pop("self_model_neg_test", None)
    q = queue.Queue()

    result = emit_chain_outcome_insight(
        q, "spirit", "self_model_neg_test", terminal_reward=-0.1,
        ctx={"source": "self_prediction", "confirmed": False})
    assert result is True, "negative rewards should pass the informative filter"
    msg = q.get_nowait()
    assert msg["payload"]["terminal_reward"] == -0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-p", "no:anchorpy"])
