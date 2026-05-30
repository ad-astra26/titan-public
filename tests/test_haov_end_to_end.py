"""End-to-end HAOV loop integration test (rFP_haov_efficacy_closure).

Drives the FULL loop through REAL cgn + cgn_worker objects for a consumer that
was dead-routed to "spirit" pre-Phase-1 (`meta`, rFP §1.2 F4):

    impasse → form → test → in-process verify (F4 fix) → confirm →
    crystallize → C1 cross-insight delivery (F1 fix)

This is the deterministic proof the rewire works — it does NOT depend on live
HAOV event cadence (which is exactly why the fleet soak cannot demonstrate the
loop within a session window: a freshly-booted Titan replays frozen disk state
and forms new impasses only slowly).

Run: python -m pytest tests/test_haov_end_to_end.py -v -p no:anchorpy --tb=short
"""
import numpy as np

from titan_hcl.logic.cgn import ConceptGroundingNetwork, CGNConsumerConfig
from titan_hcl.logic.cgn_types import CGNTransition
from titan_hcl.modules.cgn_worker import _local_haov_verify


def _feed(cgn, consumer, rewards, concept="c0"):
    for r in rewards:
        cgn._buffer.add(CGNTransition(
            consumer=consumer, concept_id=concept,
            state=np.zeros(30, dtype=np.float32), action=0,
            action_params=np.zeros(8, dtype=np.float32),
            reward=float(r), metadata={}))


def test_dead_routed_consumer_closes_loop_end_to_end(tmp_path):
    cgn = ConceptGroundingNetwork(state_dir=str(tmp_path))
    # `meta` routed to the dead "spirit" dst pre-Phase-1 — its verify path was
    # black-holed (rFP §1.2 F4). Post-fix it must verify IN-PROCESS.
    cgn.register_consumer(CGNConsumerConfig(name="meta"))
    tracker = cgn._haov_trackers["meta"]
    tracker._test_probability = 1.0  # deterministic select (bypass explore gate)

    # 1. FORM — stuck impasse → SOAR→HAOV bridge spawns a hypothesis.
    _feed(cgn, "meta", [0.0] * 14)
    imp = cgn.detect_impasse("meta", form_hypothesis=True)
    assert imp and imp["type"] == "stuck"
    assert any(h.source == "soar_impasse" for h in tracker._hypotheses)

    # 2. RESOLVE — rewards recover, so the impasse condition clears.
    _feed(cgn, "meta", [0.6] * 25)

    # 3. TEST + VERIFY — the F4 fix: impasse hypotheses verify in-process with
    #    NO bus round-trip and NO dead "spirit" dst.
    tracker.select_test({"available_actions": []})
    assert _local_haov_verify(cgn, tracker, "meta") is True

    # 4. CRYSTALLIZE — repeated confirmation lifts confidence past threshold.
    for _ in range(12):
        if tracker._active_test is None:
            tracker.select_test({"available_actions": []})
        _local_haov_verify(cgn, tracker, "meta")
        if tracker._verified_rules:
            break

    stats = tracker.get_stats()
    assert stats["confirmed"] > 0
    assert len(tracker._verified_rules) >= 1
    assert tracker._verified_rules[0].confidence > 0.5
    assert tracker._verified_rules[0].rule.startswith("meta_impasse_")

    # 5. APPLY CHANNEL (C1/F1) — the verified rule reaches OTHER consumers.
    insights = cgn.get_cross_insights("language")
    haov = [i for i in insights
            if i.get("source") == "haov_verified"
            and i.get("source_consumer") == "meta"]
    assert haov, "verified meta rule must flow cross-consumer via C1"
    assert haov[0]["rule"].startswith("meta_impasse_")
    assert haov[0]["confidence"] > 0.5


def test_loop_discriminates_unresolved_impasse(tmp_path):
    """Negative control — if the impasse NEVER resolves, the loop falsifies and
    does NOT crystallize (it discriminates, it doesn't rubber-stamp)."""
    cgn = ConceptGroundingNetwork(state_dir=str(tmp_path))
    cgn.register_consumer(CGNConsumerConfig(name="meta"))
    tracker = cgn._haov_trackers["meta"]
    tracker._test_probability = 1.0

    _feed(cgn, "meta", [0.0] * 14)
    cgn.detect_impasse("meta", form_hypothesis=True)
    _feed(cgn, "meta", [0.0] * 25)  # impasse PERSISTS — no recovery

    for _ in range(12):
        if tracker._active_test is None:
            tracker.select_test({"available_actions": []})
        _local_haov_verify(cgn, tracker, "meta")

    stats = tracker.get_stats()
    assert stats["falsified"] > 0
    assert len(tracker._verified_rules) == 0
    # And nothing leaks into the cross-insight channel.
    haov = [i for i in cgn.get_cross_insights("language")
            if i.get("source") == "haov_verified"]
    assert haov == []
