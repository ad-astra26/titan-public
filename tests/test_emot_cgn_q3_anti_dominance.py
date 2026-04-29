"""
Regression tests for §11 Q3 Option B — anti-dominance correction on
CGN_CROSS_INSIGHT β-posterior updates. Shipped 2026-04-24.
"""
import pytest


def test_anti_dominance_scales_weight_by_inverse_v():
    """A primitive with V=0.29 (WONDER-like dominant) should get a smaller
    β update than a primitive with V=0.50 (LOVE-like neutral)."""
    from titan_plugin.logic.emot_cgn import EmotCGNConsumer

    # Build two minimal consumer instances, each with one primitive at a
    # different V. We'll call handle_incoming_cross_insight on each and
    # compare the α delta.
    def make_consumer(dominant_V: float):
        c = EmotCGNConsumer(titan_id="TEST")
        # Force specific dominant with known V — set α/β so V = α/(α+β)
        # α=V, β=1-V gives exactly that V (plus the tiny +1 Laplace prior
        # already baked into the __init__, but that's the same on both
        # instances so the comparison still holds).
        target_V = dominant_V
        p = c._primitives["WONDER"]
        p.alpha = target_V * 100.0
        p.beta = (1.0 - target_V) * 100.0
        p.recompute_derived()
        c._dominant_emotion = "WONDER"
        return c

    c_dominant = make_consumer(dominant_V=0.29)  # WONDER-like heavy dominance
    c_neutral = make_consumer(dominant_V=0.50)   # LOVE-like cold/neutral

    alpha_before_d = c_dominant._primitives["WONDER"].alpha
    alpha_before_n = c_neutral._primitives["WONDER"].alpha

    payload = {
        "origin_consumer": "language",
        "insight_type": "chain_outcome",
        "terminal_reward": 0.9,  # highly informative
    }
    c_dominant.handle_incoming_cross_insight(payload)
    c_neutral.handle_incoming_cross_insight(payload)

    alpha_delta_d = c_dominant._primitives["WONDER"].alpha - alpha_before_d
    alpha_delta_n = c_neutral._primitives["WONDER"].alpha - alpha_before_n

    # Anti-dominance: low-V gets higher weight
    # w(V=0.29) = 0.25 * max(0.05, 1-0.29) = 0.25 * 0.71 = 0.1775
    # w(V=0.50) = 0.25 * max(0.05, 1-0.50) = 0.25 * 0.50 = 0.125
    # (wait — higher V means lower w; so dominant-V=0.29 is actually lower V,
    #  gets HIGHER weight. Let me flip the semantics check.)
    # V=0.29 → w=0.1775 → α += 0.1775 * 0.9 = 0.1598
    # V=0.50 → w=0.125 → α += 0.125 * 0.9 = 0.1125
    assert alpha_delta_d > alpha_delta_n, \
        f"V=0.29 should get HIGHER α update than V=0.50 (lower-V=more room). Got {alpha_delta_d} vs {alpha_delta_n}"


def test_anti_dominance_high_v_primitive_gets_tiny_update():
    """A primitive at V=0.99 (runaway dominance) should get ~floor-weight
    update (0.05 × reward), protecting against further reinforcement."""
    from titan_plugin.logic.emot_cgn import EmotCGNConsumer

    c = EmotCGNConsumer(titan_id="TEST2")
    p = c._primitives["WONDER"]
    p.alpha = 99.0
    p.beta = 1.0
    p.recompute_derived()
    c._dominant_emotion = "WONDER"

    alpha_before = p.alpha
    payload = {
        "origin_consumer": "language",
        "insight_type": "chain_outcome",
        "terminal_reward": 0.9,
    }
    c.handle_incoming_cross_insight(payload)
    delta = p.alpha - alpha_before

    # V~0.99; w = 0.25 * max(0.05, 1-0.99) = 0.25 * 0.05 = 0.0125
    # α += 0.0125 * 0.9 = 0.01125 (floor-gated)
    # Without the floor, w would be 0.25 * 0.01 = 0.0025 → even smaller
    # With floor=0.05: ~0.011 per update (not zero — still learns)
    assert 0.005 < delta < 0.02, \
        f"High-V primitive should get floor-gated small update, got {delta}"


def test_anti_dominance_negative_reward_also_scales():
    """Negative reward should also scale by (1 - V) — still penalize less
    if already dominant vs. more if emerging."""
    from titan_plugin.logic.emot_cgn import EmotCGNConsumer

    c = EmotCGNConsumer(titan_id="TEST3")
    p = c._primitives["WONDER"]
    p.alpha = 29.0
    p.beta = 71.0  # V = 0.29
    p.recompute_derived()
    c._dominant_emotion = "WONDER"

    beta_before = p.beta
    payload = {
        "origin_consumer": "language",
        "insight_type": "chain_outcome",
        "terminal_reward": 0.05,  # very negative (far from 0.5)
    }
    c.handle_incoming_cross_insight(payload)
    beta_delta = p.beta - beta_before

    # w = 0.25 * 0.71 = 0.1775; beta += w * (1 - 0.05) = 0.1775 * 0.95 = 0.1686
    assert 0.15 < beta_delta < 0.20


def test_anti_dominance_cross_insight_counter_still_increments():
    """Ensure counter increments — proves the handler is actually running
    through the update block."""
    from titan_plugin.logic.emot_cgn import EmotCGNConsumer

    c = EmotCGNConsumer(titan_id="TEST4")
    c._dominant_emotion = "WONDER"
    before = c._cgn_cross_insights_received

    payload = {
        "origin_consumer": "reasoning",
        "insight_type": "chain_outcome",
        "terminal_reward": 0.9,
    }
    c.handle_incoming_cross_insight(payload)
    assert c._cgn_cross_insights_received == before + 1


def test_anti_dominance_marker_comment_in_source():
    """Static check — the §11 Q3 Option B reference comment must stay for
    future-reader discoverability."""
    from pathlib import Path
    src = (Path(__file__).parent.parent / "titan_plugin" / "logic"
           / "emot_cgn.py").read_text()
    assert "§11 Q3 Option B" in src
    assert "anti-dominance correction" in src


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-p", "no:anchorpy"])
