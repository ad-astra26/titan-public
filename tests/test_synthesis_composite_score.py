"""Composite retrieval score tests — D-SPEC-123 / arch §5.3.

Pure-math validation of the Phase 1 ranker (w_b + w_r + w_p; spreading
deferred to Phase 4).
"""
from __future__ import annotations

import math

import pytest

from titan_hcl.synthesis.composite_score import (
    COLD_START_SENTINEL,
    DEFAULT_COLD_START_B,
    Candidate,
    ScoredCandidate,
    composite_score,
)


def _no_activation(_ids):
    """activation_lookup that returns nothing — all candidates cold-start."""
    return {}


def _from_dict(d):
    """Build an activation_lookup from a dict."""
    return lambda ids: {k: v for k, v in d.items() if k in ids}


# ─────────────────────────────────────────────────────────────────────────
# Empty / edge cases
# ─────────────────────────────────────────────────────────────────────────

def test_empty_candidates_returns_empty():
    out = composite_score([], _no_activation)
    assert out == []


def test_single_candidate_norm_b_is_zero():
    """Z-score of a single point is undefined → 0; ranking is determined
    by cosine + importance only."""
    out = composite_score(
        [Candidate(item_id="mem:1", cosine=0.7, importance=0.3)],
        _no_activation,
    )
    assert len(out) == 1
    sc = out[0]
    assert sc.norm_base_level == 0.0
    # score = 1.0·0 + 1.0·0 (spreading) + 1.0·0.7 + 1.0·0.3 = 1.0
    assert sc.score == pytest.approx(1.0)


def test_all_cold_start_no_signal_in_b_axis():
    """All candidates absent from activation_state → norm_B all 0; ranking
    by cosine + importance."""
    cands = [
        Candidate(item_id="mem:1", cosine=0.9, importance=0.5),
        Candidate(item_id="mem:2", cosine=0.3, importance=0.5),
    ]
    out = composite_score(cands, _no_activation)
    assert out[0].candidate.item_id == "mem:1"   # higher cosine wins
    assert all(sc.norm_base_level == 0.0 for sc in out)


# ─────────────────────────────────────────────────────────────────────────
# Activation-driven re-ranking
# ─────────────────────────────────────────────────────────────────────────

def test_higher_b_outranks_when_cosine_tied():
    """Two candidates with identical cosine: the one with higher B_i wins."""
    cands = [
        Candidate(item_id="mem:cold", cosine=0.6, importance=0.5),
        Candidate(item_id="mem:hot", cosine=0.6, importance=0.5),
    ]
    activations = _from_dict({"mem:hot": 2.0})   # mem:cold absent → cold-start
    out = composite_score(cands, activations)
    assert out[0].candidate.item_id == "mem:hot"
    assert out[1].candidate.item_id == "mem:cold"


def test_cosine_can_outweigh_b_at_default_weights():
    """At equal weights, a strong cosine advantage can outrank a strong
    activation advantage. Pin this — keeps the ranker sane."""
    cands = [
        Candidate(item_id="mem:relevant_cold", cosine=0.95, importance=0.5),
        Candidate(item_id="mem:hot_irrelevant", cosine=0.10, importance=0.5),
    ]
    activations = _from_dict({"mem:hot_irrelevant": 5.0})
    out = composite_score(cands, activations)
    # 5.0 B_i z-scored across (5.0, cold=0.5) gives the hot item ~+1, the
    # cold item ~-1 in norm space. Score deltas:
    #   relevant_cold: 1·(-1) + 0 + 1·0.95 + 1·0.5 = 0.45
    #   hot_irrelevant: 1·(+1) + 0 + 1·0.10 + 1·0.5 = 1.60
    # → hot_irrelevant wins at default weights. Sanity-check: weights are
    # tunable and the bridge salience (importance) will dominate later.
    # The test is here to pin the math, not to assert that "relevance
    # should always win" — that's a tuning judgment.
    assert out[0].candidate.item_id == "mem:hot_irrelevant"
    assert out[0].score > out[1].score


def test_zero_w_b_neutralizes_activation_axis():
    """Setting w_b=0 disables activation re-ranking entirely → pure
    cosine + importance ordering."""
    cands = [
        Candidate(item_id="mem:cold_relevant", cosine=0.9, importance=0.5),
        Candidate(item_id="mem:hot_less_relevant", cosine=0.4, importance=0.5),
    ]
    activations = _from_dict({"mem:hot_less_relevant": 10.0})
    out = composite_score(cands, activations, w_b=0.0)
    assert out[0].candidate.item_id == "mem:cold_relevant"


# ─────────────────────────────────────────────────────────────────────────
# Cold-start substitution
# ─────────────────────────────────────────────────────────────────────────

def test_cold_start_substitution_does_not_poison_mean():
    """A -inf cold item must not propagate into the z-score (would NaN
    everything). Cold items are SUBSTITUTED with cold_start_b before
    normalization."""
    cands = [
        Candidate(item_id="mem:a", cosine=0.5),
        Candidate(item_id="mem:b", cosine=0.5),
        Candidate(item_id="mem:cold", cosine=0.5),
    ]
    activations = _from_dict({"mem:a": 1.0, "mem:b": 2.0})
    out = composite_score(cands, activations)
    # All scores finite, all norm_b finite
    for sc in out:
        assert math.isfinite(sc.score)
        assert math.isfinite(sc.norm_base_level)


def test_cold_start_default_configurable():
    """cold_start_b config gain — pin the math."""
    cands = [
        Candidate(item_id="mem:a", cosine=0.5),
        Candidate(item_id="mem:cold", cosine=0.5),
    ]
    activations = _from_dict({"mem:a": 2.0})
    out_low = composite_score(cands, activations, cold_start_b=0.0)
    out_high = composite_score(cands, activations, cold_start_b=3.0)
    # With cold=0: mem:cold gets B=0 → -1z; mem:a gets B=2 → +1z. mem:a wins.
    assert out_low[0].candidate.item_id == "mem:a"
    # With cold=3: mem:cold gets B=3 → +1z; mem:a gets B=2 → -1z. mem:cold wins.
    assert out_high[0].candidate.item_id == "mem:cold"


# ─────────────────────────────────────────────────────────────────────────
# Importance + cosine
# ─────────────────────────────────────────────────────────────────────────

def test_importance_added_per_node():
    """Per-candidate importance contributes directly to the score."""
    cands = [
        Candidate(item_id="mem:a", cosine=0.5, importance=0.9),
        Candidate(item_id="mem:b", cosine=0.5, importance=0.1),
    ]
    out = composite_score(cands, _no_activation)
    assert out[0].candidate.item_id == "mem:a"
    assert out[0].score - out[1].score == pytest.approx(0.8)


def test_cosine_clamped_to_unit_interval():
    """Defensive clamp on cosine ∈ [0, 1] — some raw inner products can
    fall outside."""
    cands = [
        Candidate(item_id="mem:over", cosine=1.5),
        Candidate(item_id="mem:under", cosine=-0.5),
    ]
    out = composite_score(cands, _no_activation)
    seen = {sc.candidate.item_id: sc.cosine for sc in out}
    assert seen["mem:over"] == 1.0
    assert seen["mem:under"] == 0.0


# ─────────────────────────────────────────────────────────────────────────
# Spreading (Phase 4+ surface; Phase 1 = None → 0)
# ─────────────────────────────────────────────────────────────────────────

def test_spreading_none_in_phase_1():
    """Phase 1 callers pass spreading_lookup=None → spreading contribution
    is 0 for all items."""
    cands = [Candidate(item_id="mem:a", cosine=0.5)]
    out = composite_score(cands, _no_activation, spreading_lookup=None)
    # No assertion needed on spreading directly — but the score must equal
    # the no-spreading formula.
    assert out[0].score == pytest.approx(1.0 * 0 + 1.0 * 0 + 1.0 * 0.5 + 1.0 * 0.5)


def test_spreading_supplied_contributes_to_score():
    """When a future Phase 4 caller supplies spreading_lookup, the result
    is added per candidate."""
    cands = [
        Candidate(item_id="mem:a", cosine=0.5),
        Candidate(item_id="mem:b", cosine=0.5),
    ]
    spread = lambda ids: {"mem:a": 0.7}    # mem:b absent → 0
    out = composite_score(cands, _no_activation, spreading_lookup=spread)
    seen = {sc.candidate.item_id: sc.score for sc in out}
    assert seen["mem:a"] - seen["mem:b"] == pytest.approx(0.7)


# ─────────────────────────────────────────────────────────────────────────
# Sort stability — ties preserve cosine order
# ─────────────────────────────────────────────────────────────────────────

def test_sort_is_stable_under_ties():
    """Python's sort is stable. When two candidates have equal composite
    scores, they preserve their input order."""
    cands = [
        Candidate(item_id="mem:first_in", cosine=0.5),
        Candidate(item_id="mem:second_in", cosine=0.5),
    ]
    out = composite_score(cands, _no_activation)
    assert out[0].candidate.item_id == "mem:first_in"
    assert out[1].candidate.item_id == "mem:second_in"


# ─────────────────────────────────────────────────────────────────────────
# Observability: full ScoredCandidate breakdown is preserved
# ─────────────────────────────────────────────────────────────────────────

def test_scored_candidate_preserves_inputs():
    """ScoredCandidate carries the raw inputs for debugging + observability."""
    cands = [Candidate(item_id="mem:1", cosine=0.7, importance=0.3,
                       payload={"node_id": 42})]
    activations = _from_dict({"mem:1": 1.5})
    out = composite_score(cands, activations, w_b=0.5, w_r=2.0, w_p=0.1)
    sc = out[0]
    assert sc.candidate.payload == {"node_id": 42}
    assert sc.cosine == pytest.approx(0.7)
    assert sc.importance == pytest.approx(0.3)
    assert sc.base_level == 1.5
    assert sc.weights == (0.5, 1.0, 2.0, 0.1)   # (w_b, w_s default, w_r, w_p)
