"""Emergent reward-surprise gate for peer CGN_CROSS_INSIGHT emission
(cgn_consumer_client.emit_chain_outcome_insight) — the language/social emitter
that feeds emot_cgn's `received` counter.

Replaces the old fixed `|reward-0.5|>0.3` gate, which the agent's reward regime
(rewards cluster well inside [0.2,0.8]) rarely crossed → the peer cross-insight
channel was muted fleet-wide (diagnosed 2026-06-10). The new gate is RELATIVE:
emit when a reward surprises THIS consumer's own running baseline
(|reward - running_mean| >= k·running_std) — no fixed reward constant.

Pins:
  1. warm-up (< MIN_N samples) builds a baseline WITHOUT emitting.
  2. after warm-up, a reward that surprises the baseline emits (insight_type
     == chain_outcome, so emot's receiver accepts it).
  3. a reward WITHIN the baseline does not emit (relative, not fixed-threshold).

Run: python -m pytest tests/test_cgn_cross_insight_gate.py -v -p no:anchorpy
"""
import warnings
warnings.filterwarnings("ignore")

from titan_hcl.logic.cgn_consumer_client import (
    emit_chain_outcome_insight, _PEER_REWARD_MIN_N,
    _PEER_REWARD_HIST, _PEER_CROSS_INSIGHT_RATE_STATE,
)


class _Q:
    def __init__(self):
        self.sent = []

    def put_nowait(self, m):
        self.sent.append(m)


def _fresh(consumer):
    """Clear the per-consumer module-level state (history + rate gate)."""
    _PEER_REWARD_HIST.pop(consumer, None)
    _PEER_CROSS_INSIGHT_RATE_STATE.pop(consumer, None)


def test_warmup_builds_baseline_without_emitting():
    c = "ut_warmup"
    _fresh(c)
    q = _Q()
    for _ in range(_PEER_REWARD_MIN_N):
        emit_chain_outcome_insight(q, "ut", c, 0.30)
    # baseline still forming → nothing emitted, but the window filled
    assert len(q.sent) == 0
    assert len(_PEER_REWARD_HIST[c]) == _PEER_REWARD_MIN_N
    _fresh(c)


def test_surprising_reward_emits_after_warmup():
    c = "ut_surprise"
    _fresh(c)
    q = _Q()
    for _ in range(_PEER_REWARD_MIN_N):
        emit_chain_outcome_insight(q, "ut", c, 0.30)   # baseline ~0.30
    assert len(q.sent) == 0
    ok = emit_chain_outcome_insight(q, "ut", c, 0.80)   # far from baseline
    assert ok is True
    assert len(q.sent) == 1
    assert q.sent[0]["payload"]["insight_type"] == "chain_outcome"
    assert q.sent[0]["payload"]["origin_consumer"] == c
    _fresh(c)


def test_unsurprising_reward_does_not_emit():
    c = "ut_calm"
    _fresh(c)
    q = _Q()
    for _ in range(_PEER_REWARD_MIN_N):
        emit_chain_outcome_insight(q, "ut", c, 0.30)
    ok = emit_chain_outcome_insight(q, "ut", c, 0.31)   # within baseline
    assert ok is False
    assert len(q.sent) == 0
    _fresh(c)


if __name__ == "__main__":
    test_warmup_builds_baseline_without_emitting()
    test_surprising_reward_emits_after_warmup()
    test_unsurprising_reward_does_not_emit()
    print("OK — emergent reward-surprise cross-insight gate checks passed")
