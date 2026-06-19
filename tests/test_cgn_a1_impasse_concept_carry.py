"""A1 (RFP_cgn_loop_closure §7.D-A1) — hypothesize_from_impasse must carry the
plateau concept into action_context (topic + concept) and a concept-specific
rule_name, so the verified rule is a genuine concept-grounding rule (applicable
via suggest(), shareable via get_cross_insights). stuck/declining impasses are
consumer-wide (no concept) and must stay byte-identical to the legacy form.
"""
from titan_hcl.logic.cgn_types import GeneralizedHAOVTracker


def test_plateau_impasse_carries_concept():
    tr = GeneralizedHAOVTracker("knowledge")
    h = tr.hypothesize_from_impasse(
        {"type": "plateau", "severity": 0.5, "concept": "quantum_entanglement"})
    assert h is not None
    # concept-bearing action_context (mirrors the causal path topic convention)
    assert h.action_context.get("topic") == "quantum_entanglement"
    assert h.action_context.get("concept") == "quantum_entanglement"
    assert h.action_context.get("impasse_type") == "plateau"
    # concept-specific rule name → distinct concepts are distinct rules
    assert h.rule == "knowledge_plateau_quantum_entanglement"
    assert h.source == "soar_impasse"
    # verification key (impasse_type) preserved → _local_haov_verify still routes
    assert h.predicted_effect == "resolve_plateau"


def test_distinct_concepts_form_distinct_rules():
    tr = GeneralizedHAOVTracker("knowledge")
    h1 = tr.hypothesize_from_impasse(
        {"type": "plateau", "severity": 0.5, "concept": "topic_a"})
    h2 = tr.hypothesize_from_impasse(
        {"type": "plateau", "severity": 0.5, "concept": "topic_b"})
    assert h1.rule != h2.rule
    assert h1.rule == "knowledge_plateau_topic_a"
    assert h2.rule == "knowledge_plateau_topic_b"


def test_stuck_impasse_stays_consumer_wide():
    tr = GeneralizedHAOVTracker("meta")
    h = tr.hypothesize_from_impasse({"type": "stuck", "severity": 0.7})
    assert h is not None
    assert h.action_context == {"impasse_type": "stuck"}
    assert "topic" not in h.action_context
    assert h.rule == "meta_impasse_stuck"


def test_declining_impasse_unchanged():
    tr = GeneralizedHAOVTracker("reasoning")
    h = tr.hypothesize_from_impasse({"type": "declining", "severity": 0.4})
    assert h is not None
    assert h.action_context == {"impasse_type": "declining"}
    assert h.rule == "reasoning_impasse_declining"
