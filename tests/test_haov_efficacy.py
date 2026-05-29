"""Phase 1 tests for rFP_haov_efficacy_closure — F4 verify rewire + F2.

Covers:
  - cgn.verify_impasse_resolution() — generic in-process impasse verifier
    (confirms when the impasse clears, falsifies while it persists).
  - cgn.detect_impasse(form_hypothesis=False) — side-effect-free check (must not
    spawn new hypotheses while verifying an existing one).
  - cgn_worker._local_haov_verify() — impasse hypotheses verify locally (no bus),
    moving the tracker's confirmed/falsified stats; consumers with a live
    specialist verifier still route via the bus.
  - cgn_worker._HAOV_DEST_MAP — no dead "spirit"/"dreaming" routes remain.

Run: python -m pytest tests/test_haov_efficacy.py -v -p no:anchorpy --tb=short
"""
import numpy as np
import pytest

from titan_hcl.logic.cgn import ConceptGroundingNetwork, CGNConsumerConfig
from titan_hcl.logic.cgn_types import CGNTransition


def _mk_cgn(tmp_path):
    return ConceptGroundingNetwork(state_dir=str(tmp_path))


def _feed(cgn, consumer, rewards, *, concept="c0"):
    """Append transitions with the given rewards to the consumer's buffer."""
    for r in rewards:
        cgn._buffer.add(CGNTransition(
            consumer=consumer,
            concept_id=concept,
            state=np.zeros(30, dtype=np.float32),
            action=0,
            action_params=np.zeros(8, dtype=np.float32),
            reward=float(r),
            metadata={},
        ))


# ── verify_impasse_resolution ───────────────────────────────────────────────

def test_impasse_persists_falsifies(tmp_path):
    cgn = _mk_cgn(tmp_path)
    cgn.register_consumer(CGNConsumerConfig(name="meta"))
    _feed(cgn, "meta", [0.0] * 14)  # all-zero reward → stuck
    assert cgn.detect_impasse("meta", form_hypothesis=False)["type"] == "stuck"
    confirmed, reward = cgn.verify_impasse_resolution("meta", "stuck")
    assert confirmed is False and reward == 0.0


def test_impasse_resolved_confirms(tmp_path):
    cgn = _mk_cgn(tmp_path)
    cgn.register_consumer(CGNConsumerConfig(name="meta"))
    # 14 stuck, then 20 healthy rewards → recent window no longer stuck.
    _feed(cgn, "meta", [0.0] * 14)
    _feed(cgn, "meta", [0.4] * 20)
    assert cgn.detect_impasse("meta", form_hypothesis=False) is None or \
        cgn.detect_impasse("meta", form_hypothesis=False)["type"] != "stuck"
    confirmed, reward = cgn.verify_impasse_resolution("meta", "stuck")
    assert confirmed is True and reward > 0.05


def test_detect_impasse_form_false_has_no_side_effect(tmp_path):
    cgn = _mk_cgn(tmp_path)
    cgn.register_consumer(CGNConsumerConfig(name="meta"))
    _feed(cgn, "meta", [0.0] * 14)
    tracker = cgn._haov_trackers["meta"]
    before = len(tracker._hypotheses)
    cgn.detect_impasse("meta", form_hypothesis=False)
    assert len(tracker._hypotheses) == before, "no hypothesis must be spawned"
    # Sanity: form_hypothesis=True DOES spawn one.
    cgn.detect_impasse("meta", form_hypothesis=True)
    assert len(tracker._hypotheses) == before + 1


# ── _local_haov_verify + dest map (cgn_worker) ──────────────────────────────

def test_local_verify_handles_impasse_hypothesis(tmp_path):
    from titan_hcl.modules.cgn_worker import _local_haov_verify
    cgn = _mk_cgn(tmp_path)
    cgn.register_consumer(CGNConsumerConfig(name="meta"))
    _feed(cgn, "meta", [0.0] * 14)
    tracker = cgn._haov_trackers["meta"]
    # Form an impasse hypothesis + make it the active test.
    h = tracker.hypothesize_from_impasse(
        {"type": "stuck", "severity": 0.9, "consumer": "meta"})
    assert h is not None and h.source == "soar_impasse"
    tracker._active_test = {"hypothesis": h, "pre_observation": {}}
    falsified_before = tracker._stats["falsified"]

    handled = _local_haov_verify(cgn, tracker, "meta")
    assert handled is True, "impasse hypothesis must verify in-process"
    # Still stuck → falsified count moved, active_test consumed.
    assert tracker._active_test is None
    assert tracker._stats["falsified"] == falsified_before + 1


def test_local_verify_confirms_when_resolved(tmp_path):
    from titan_hcl.modules.cgn_worker import _local_haov_verify
    cgn = _mk_cgn(tmp_path)
    cgn.register_consumer(CGNConsumerConfig(name="meta"))
    _feed(cgn, "meta", [0.0] * 14)
    tracker = cgn._haov_trackers["meta"]
    h = tracker.hypothesize_from_impasse(
        {"type": "stuck", "severity": 0.9, "consumer": "meta"})
    tracker._active_test = {"hypothesis": h, "pre_observation": {}}
    _feed(cgn, "meta", [0.4] * 20)  # impasse now resolved
    confirmed_before = tracker._stats["confirmed"]

    assert _local_haov_verify(cgn, tracker, "meta") is True
    assert tracker._stats["confirmed"] == confirmed_before + 1


def test_local_verify_routes_specialist_consumers_via_bus(tmp_path):
    """language has a live specialist verifier → concept-grounding hypotheses
    must NOT be verified locally (caller routes via bus → returns False)."""
    from titan_hcl.modules.cgn_worker import _local_haov_verify
    cgn = _mk_cgn(tmp_path)
    cgn.register_consumer(CGNConsumerConfig(name="language"))
    tracker = cgn._haov_trackers["language"]
    # A concept-grounding (non-impasse) hypothesis for a specialist consumer.
    h = tracker.hypothesize(
        action_context={"topic": "warm"},
        observation={"effect": "confidence_gain", "magnitude": 0.5,
                     "rule_name": "warm_means_comfortable", "source": "pattern"})
    tracker._active_test = {"hypothesis": h, "pre_observation": {}}
    assert _local_haov_verify(cgn, tracker, "language") is False


# ── Phase 2 / F1 / C1 — verified rules flow into get_cross_insights ─────────

def test_c1_verified_rules_appear_in_cross_insights(tmp_path):
    cgn = _mk_cgn(tmp_path)
    cgn.register_consumer(CGNConsumerConfig(name="language"))
    cgn.register_consumer(CGNConsumerConfig(name="social"))
    # Manufacture a verified rule on language by forming + confirming.
    ltr = cgn._haov_trackers["language"]
    h = ltr.hypothesize_from_impasse(
        {"type": "stuck", "severity": 0.9, "consumer": "language"})
    h.confidence = 0.8  # above the verification threshold
    ltr._verified_rules.append(h)

    # social asks for cross-insights → should see language's verified rule.
    insights = cgn.get_cross_insights("social")
    haov = [i for i in insights if i.get("source") == "haov_verified"]
    assert haov, "C1: verified HAOV rules must surface in get_cross_insights"
    assert haov[0]["source_consumer"] == "language"
    assert haov[0]["rule"] == h.rule
    # A consumer never sees its OWN verified rules as cross-insights.
    own = [i for i in cgn.get_cross_insights("language")
           if i.get("source") == "haov_verified"
           and i.get("source_consumer") == "language"]
    assert not own


def test_c1_low_confidence_rules_excluded(tmp_path):
    cgn = _mk_cgn(tmp_path)
    cgn.register_consumer(CGNConsumerConfig(name="language"))
    cgn.register_consumer(CGNConsumerConfig(name="social"))
    ltr = cgn._haov_trackers["language"]
    h = ltr.hypothesize_from_impasse(
        {"type": "stuck", "severity": 0.5, "consumer": "language"})
    h.confidence = 0.4  # below the 0.5 gate
    ltr._verified_rules.append(h)
    haov = [i for i in cgn.get_cross_insights("social")
            if i.get("source") == "haov_verified"]
    assert not haov, "confidence <= 0.5 verified rules must not surface"


def test_dest_map_has_no_dead_routes():
    from titan_hcl.modules.cgn_worker import _HAOV_DEST_MAP
    assert "spirit" not in _HAOV_DEST_MAP.values(), "dead spirit route remains"
    assert "dreaming" not in _HAOV_DEST_MAP, "phantom dreaming entry remains"
    assert set(_HAOV_DEST_MAP) == {"language", "knowledge", "emotional"}
