"""Tests for the EXPERIENCE_RECORD Record-stage restoration + enrichment.

rFP_experience_distillation_phase_c_restoration_and_enrichment.md (D-SPEC-101).

Covers:
  - bus.emit_experience_record frame shape + bus-hygiene invariants
    (targeted dst=cognitive_worker, never dst=all; per-coalesce_key min-interval).
  - The 3 Phase-1 ExperiencePlugins (knowledge / self_model / meta_reasoning).
  - cognitive_worker._dispatch_experience_record enrich-and-record against a stub.

Run standalone:  python -m pytest tests/test_experience_record_loop.py -v -p no:anchorpy
"""
import time

import titan_hcl.bus as bus


class _Sink:
    """Captures frames a producer would put on its send_queue."""

    def __init__(self):
        self.frames = []

    def put_nowait(self, frame):
        self.frames.append(frame)


def test_emit_frame_is_targeted_and_well_formed():
    sink = _Sink()
    ok = bus.emit_experience_record(
        sink, "language", domain="language",
        action_taken="hello world", outcome_score=0.8,
        context={"source": "direct_composition"}, epoch_id=42,
        coalesce_key="t_targeted",
    )
    assert ok is True
    assert len(sink.frames) == 1
    f = sink.frames[0]
    assert f["type"] == bus.EXPERIENCE_RECORD
    # Bus-hygiene: targeted to the single consumer, NEVER dst="all".
    assert f["dst"] == "cognitive_worker"
    assert f["dst"] != "all"
    p = f["payload"]
    assert p["domain"] == "language"
    assert p["action_taken"] == "hello world"
    assert abs(p["outcome_score"] - 0.8) < 1e-9
    assert p["epoch_id"] == 42
    assert p["context"]["source"] == "direct_composition"
    assert "ts" in p


def test_coalesce_guard_blocks_rapid_same_key():
    sink = _Sink()
    k = "t_coalesce_%f" % time.time()
    a = bus.emit_experience_record(sink, "x", domain="knowledge",
                                   action_taken="a", outcome_score=0.5,
                                   coalesce_key=k, min_interval_s=5.0)
    b = bus.emit_experience_record(sink, "x", domain="knowledge",
                                   action_taken="b", outcome_score=0.5,
                                   coalesce_key=k, min_interval_s=5.0)
    assert a is True
    assert b is False  # coalesced within the interval
    assert len(sink.frames) == 1


def test_distinct_coalesce_keys_do_not_starve():
    sink = _Sink()
    suffix = "%f" % time.time()
    a = bus.emit_experience_record(sink, "lang", domain="language",
                                   action_taken="compose", outcome_score=0.6,
                                   coalesce_key="compose_" + suffix,
                                   min_interval_s=5.0)
    b = bus.emit_experience_record(sink, "lang", domain="language",
                                   action_taken="comprehend", outcome_score=0.6,
                                   coalesce_key="comprehend_" + suffix,
                                   min_interval_s=5.0)
    assert a is True and b is True
    assert len(sink.frames) == 2  # distinct sub-streams both pass


def test_empty_domain_rejected():
    sink = _Sink()
    assert bus.emit_experience_record(sink, "x", domain="",
                                      action_taken="a", outcome_score=0.5) is False
    assert sink.frames == []


def test_phase1_plugins_perception_and_score():
    from titan_hcl.logic.experience_plugins import (
        KnowledgePlugin, SelfModelPlugin, MetaReasoningPlugin)
    state = [0.1 * (i % 10) for i in range(130)]
    ctx = {"inner_state": state,
           "hormonal_snapshot": {"CURIOSITY": 0.7, "REFLECTION": 0.6}}
    for plug, expected_domain in (
        (KnowledgePlugin(), "knowledge"),
        (SelfModelPlugin(), "self_model"),
        (MetaReasoningPlugin(), "meta_reasoning"),
    ):
        assert plug.domain == expected_domain
        pk = plug.extract_perception_key(ctx)
        assert isinstance(pk, list) and len(pk) == 20
        assert all(isinstance(v, float) for v in pk)
        s = plug.compute_outcome_score({"confidence": 0.9, "coherence": 0.9,
                                        "strength": 0.9})
        assert 0.0 <= s <= 1.0
        summary = plug.summarize_for_distillation(
            [{"outcome_score": 0.8, "context": {}}])
        assert "pattern" in summary


def test_dispatch_enrich_and_record_against_stub():
    """_dispatch_experience_record enriches + calls record_outcome."""
    from titan_hcl.modules.cognitive_worker import (
        _dispatch_experience_record)

    captured = {}

    class _StubPlugin:
        def extract_perception_key(self, ctx):
            return [0.5] * 8

    class _StubOrch:
        _plugins = {"language": _StubPlugin()}

        def record_outcome(self, **kw):
            captured.update(kw)
            return 1

    state_refs = {
        "exp_orchestrator": _StubOrch(),
        "consciousness": {"latest_epoch": {"state_vector": [0.3] * 130}},
        "_inner_body_state": [0.5] * 5,
        "_inner_mind_state": [0.5] * 15,
        "_inner_spirit_state": [0.5] * 45,
        "neural_nervous_system": None,
        "coordinator": None,
    }
    _dispatch_experience_record(state_refs, {
        "domain": "language",
        "action_taken": "test sentence",
        "outcome_score": 0.7,
        "context": {"source": "direct_composition"},
        "epoch_id": 5,
    })
    assert captured.get("domain") == "language"
    assert captured.get("action_taken") == "test sentence"
    assert abs(captured.get("outcome_score") - 0.7) < 1e-9
    assert captured.get("is_dreaming") is False
    assert len(captured.get("inner_state_132d")) == 130


def test_dispatch_no_orchestrator_is_safe():
    from titan_hcl.modules.cognitive_worker import (
        _dispatch_experience_record)
    # Must not raise when orchestrator absent.
    _dispatch_experience_record({"exp_orchestrator": None}, {"domain": "x"})


def test_set_dream_subsystems_is_additive_no_clobber():
    """Distill-stage wiring: set_dream_subsystems must be additive so the
    coordinator-init neuromod call + the later exp_orchestrator call coexist.
    The clobber bug left coordinator._exp_orchestrator=None → distill_cycle
    never ran on dream (rFP_experience_distillation_phase_c)."""
    from titan_hcl.logic.inner_coordinator import InnerTrinityCoordinator
    c = InnerTrinityCoordinator.__new__(InnerTrinityCoordinator)
    c._exp_orchestrator = None
    c._e_mem = None
    c._neuromod_system = None
    c.set_dream_subsystems(neuromod_system="NM")  # init-time (neuromod only)
    assert c._neuromod_system == "NM"
    assert c._exp_orchestrator is None
    c.set_dream_subsystems(exp_orchestrator="EO", e_mem="EM")  # later wire
    assert c._neuromod_system == "NM", "clobber regression: neuromod lost"
    assert c._exp_orchestrator == "EO"
    assert c._e_mem == "EM"
