"""Phase 3E C3 tests — social engage-bias from verified HAOV concepts.

The C3 application logic lives on SocialXGateway (the reply-gate decision point):
`_verified_concept_engage_bias` (pure) + `set_verified_haov_concepts` (cache).
These prove the bias is bounded, impasse-filtered, and ZERO when there are no
concept-grounding verified concepts (the current fleet state) — so wiring C3 now
cannot perturb the live mainnet reply gate until such a rule actually crystallizes.

Run: python -m pytest tests/test_haov_c3_engage_bias.py -v -p no:anchorpy --tb=short
"""
from titan_hcl.logic.social_x_gateway import SocialXGateway

_bias = SocialXGateway._verified_concept_engage_bias


def test_bias_zero_when_no_concepts():
    assert _bias([], "tell me about symmetry") == 0.0
    assert _bias(None, "anything") == 0.0


def test_impasse_rules_are_filtered_out():
    # Today's only verified rules are impasse-type (effect "resolve_*") — they
    # carry no teachable concept and must contribute ZERO bias.
    impasse = [{"source": "haov_verified", "rule": "meta_impasse_stuck",
                "effect": "resolve_stuck", "confidence": 0.95}]
    assert _bias(impasse, "i feel stuck and declining") == 0.0


def test_concept_match_returns_bounded_bias():
    concepts = [{"source": "haov_verified", "rule": "language_symmetry",
                 "effect": "grounded", "confidence": 0.8}]
    b = _bias(concepts, "what can you tell me about symmetry today")
    assert b == round(0.8 * 0.2, 4)  # 0.16


def test_no_textual_overlap_zero():
    concepts = [{"rule": "language_symmetry", "effect": "grounded",
                 "confidence": 0.8}]
    assert _bias(concepts, "completely unrelated conversation") == 0.0


def test_bias_capped_at_0_2():
    concepts = [{"rule": "x_symmetry", "effect": "g", "confidence": 5.0}]
    assert _bias(concepts, "symmetry") == 0.2


def test_picks_max_confidence_match():
    concepts = [
        {"rule": "a_symmetry", "effect": "g", "confidence": 0.3},
        {"rule": "b_balance", "effect": "g", "confidence": 0.9},
    ]
    # text mentions both → max confidence wins
    assert _bias(concepts, "i value symmetry and balance") == round(0.9 * 0.2, 4)


def test_set_verified_concepts_caps_at_16():
    g = SocialXGateway.__new__(SocialXGateway)  # no __init__ (avoid heavy boot)
    g.set_verified_haov_concepts([{"rule": f"r{i}"} for i in range(40)])
    assert len(g._verified_haov_concepts) == 16
    g.set_verified_haov_concepts(None)
    assert g._verified_haov_concepts == []
