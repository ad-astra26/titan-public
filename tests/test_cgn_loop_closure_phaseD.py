"""RFP_cgn_loop_closure §7.D (INV-LOOP-6) — learning closes to behaviour.

Confirmed HAOV rules must influence actions, counted by used_for_action. The
apply happens in a consumer process (C2 language teaching / C3 social engage-
bias); the counter lives on the GeneralizedHAOVTracker in cgn_worker (G21). The
consumer emits CGN_HAOV_RULE_APPLIED → cgn_worker calls mark_used_for_action.
These tests pin:
  1. mark_used_for_action increments used_for_action (the cross-process count).
  2. suggest() still both selects AND counts (in-process arc/session path) —
     unchanged.
  3. _matched_haov_source attributes an applied engage-bias to the right
     source consumer + rule (for the emit payload), skipping impasse rules.

Run: python -m pytest tests/test_cgn_loop_closure_phaseD.py -v -p no:anchorpy
"""
import warnings
warnings.filterwarnings("ignore")

from titan_hcl.logic.cgn_types import GeneralizedHAOVTracker, GeneralizedHypothesis
from titan_hcl.logic.social_x_gateway import SocialXGateway


def _tracker_with_verified_rule():
    t = GeneralizedHAOVTracker(consumer="language")
    h = GeneralizedHypothesis(
        rule="pattern_epigenetics",
        consumer="language",
        action_context={"action": "engage", "concept": "epigenetics"},
        predicted_effect="grounds_concept",
        predicted_magnitude=0.8,
    )
    h.confidence = 0.75
    t._verified_rules.append(h)
    return t


def test_mark_used_for_action_increments():
    """The C2/C3 cross-process apply-count: cgn_worker calls this when a
    consumer reports it applied one of this tracker's verified rules."""
    t = _tracker_with_verified_rule()
    assert t._stats["used_for_action"] == 0
    new_total = t.mark_used_for_action()
    assert new_total == 1
    assert t._stats["used_for_action"] == 1
    # Batched count (count>1 in the payload).
    t.mark_used_for_action(3)
    assert t._stats["used_for_action"] == 4


def test_suggest_still_selects_and_counts():
    """Back-compat: suggest() (arc/session in-process path) still both picks a
    rule AND increments used_for_action — Phase D didn't change it."""
    t = _tracker_with_verified_rule()
    out = t.suggest({"available_actions": ["engage", "defer"]})
    assert out is not None
    assert out.get("action") == "engage"
    assert t._stats["used_for_action"] == 1


def test_matched_haov_source_attributes_and_skips_impasse():
    """C3 attribution: the matched concept-grounding rule's source + rule are
    returned (for the CGN_HAOV_RULE_APPLIED payload); impasse rules are skipped;
    no textual match → None."""
    concepts = [
        {"source_consumer": "knowledge", "rule": "resolve_impasse_42",
         "effect": "resolve_stuck", "confidence": 0.9},          # impasse → skip
        {"source_consumer": "language", "rule": "pattern_epigenetics",
         "effect": "grounds_concept", "confidence": 0.7},        # concept-grounding
    ]
    m = SocialXGateway._matched_haov_source(concepts, "a thread about epigenetics")
    assert m is not None
    assert m[0] == "language"
    assert m[1] == "pattern_epigenetics"
    # No overlap → None (no rule applied).
    assert SocialXGateway._matched_haov_source(concepts, "unrelated chatter") is None
    # Empty → None.
    assert SocialXGateway._matched_haov_source([], "epigenetics") is None


def test_engage_bias_zero_does_not_emit():
    """The emit only fires when the bias actually applies (>0). With no matching
    verified concept the bias is 0 and _matched_haov_source returns None, so no
    used_for_action credit is spuriously claimed (apply ≠ delivery)."""
    concepts = [{"source_consumer": "language", "rule": "pattern_epigenetics",
                 "effect": "grounds_concept", "confidence": 0.7}]
    bias = SocialXGateway._verified_concept_engage_bias(concepts, "weather today")
    assert bias == 0.0
    assert SocialXGateway._matched_haov_source(concepts, "weather today") is None


if __name__ == "__main__":
    test_mark_used_for_action_increments()
    test_suggest_still_selects_and_counts()
    test_matched_haov_source_attributes_and_skips_impasse()
    test_engage_bias_zero_does_not_emit()
    print("OK — Phase D learning→behaviour checks passed")
