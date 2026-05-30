"""Regression guard: every persistence producer's save method must EXIST.

Session 2026-05-30 (fleet-wide persistence hardening) caught a class of bug
where a worker called a save method that does not exist on its engine
(``prediction_engine.save_state()`` where the real method is ``_save_state``).
The call sat inside a best-effort ``try/except``, so it failed silently and the
state simply never persisted — observed as novelty_state.json frozen ~16 days,
and the graceful-shutdown flush guarded on ``hasattr(pe, "save_state")`` which
was always False.

This test asserts the save method referenced by each producer's periodic
checkpoint / shutdown flush is actually present on the target class, so a rename
or a wrong-name typo fails loudly at CI time instead of silently in production.
"""
import pytest


def test_prediction_engine_has_save_method():
    from titan_hcl.logic.prediction_engine import PredictionEngine
    # self_reflection_worker periodic checkpoint + _save_state_on_shutdown both
    # call pe._save_state(); save_state (no underscore) must NOT be relied upon.
    assert hasattr(PredictionEngine, "_save_state")


def test_cgn_has_save_method():
    from titan_hcl.logic.cgn import ConceptGroundingNetwork
    # cgn_worker._maybe_checkpoint_state -> cgn._save_state()
    assert hasattr(ConceptGroundingNetwork, "_save_state")


def test_sovereignty_tracker_has_save_method():
    from titan_hcl.logic.sovereignty import SovereigntyTracker
    # sovereignty_worker periodic checkpoint -> tracker._save_state()
    assert hasattr(SovereigntyTracker, "_save_state")


def test_social_pressure_meter_has_save_method():
    from titan_hcl.logic.social_pressure import SocialPressureMeter
    # social_worker SAVE_NOW + periodic checkpoint -> meter.save_state()
    assert hasattr(SocialPressureMeter, "save_state")


def test_intuition_convergence_has_save_method():
    from titan_hcl.logic.intuition_convergence import IntuitionConvergenceDetector
    # cognitive_worker._persist_engine_state -> intuition_convergence.save_state()
    assert hasattr(IntuitionConvergenceDetector, "save_state")


def test_maker_engine_save_load_wired_in_init():
    """maker_engine had save_state/load_state but they were NEVER called.

    load_state() must run in __init__ and save_state() at the end of run().
    Assert both methods exist and __init__ invokes load_state.
    """
    import inspect
    from titan_hcl.logic.maker_engine import MakerRelationshipEngine
    assert hasattr(MakerRelationshipEngine, "save_state")
    assert hasattr(MakerRelationshipEngine, "load_state")
    init_src = inspect.getsource(MakerRelationshipEngine.__init__)
    assert "load_state" in init_src, "load_state() not wired into __init__"
    run_src = inspect.getsource(MakerRelationshipEngine.run)
    assert "save_state" in run_src, "save_state() not wired into run()"


def test_edge_detector_persistence_callables():
    from titan_hcl.logic.edge_detector_persistence import (
        save_edge_detector_state, load_edge_detector_state)
    assert callable(save_edge_detector_state)
    assert callable(load_edge_detector_state)


def test_maker_engine_save_load_roundtrip(tmp_path):
    from titan_hcl.logic.maker_engine import MakerRelationshipEngine
    p = str(tmp_path / "maker_engine_state.json")
    e = MakerRelationshipEngine(memory=None)
    e._topic_scores = {"netflix": {"category": "interests", "occurrences": 4}}
    e._promoted_topics = {"netflix"}
    e._last_run_ts = 99.0
    e.save_state(p)
    e2 = MakerRelationshipEngine(memory=None)
    e2.load_state(p)
    assert e2._topic_scores.get("netflix", {}).get("occurrences") == 4
    assert "netflix" in e2._promoted_topics
    assert e2._last_run_ts == 99.0


def test_intuition_convergence_save_load_roundtrip(tmp_path):
    from titan_hcl.logic.intuition_convergence import IntuitionConvergenceDetector
    p = str(tmp_path / "intuition_convergence_state.json")
    d = IntuitionConvergenceDetector()
    d._total_convergence_events = 7
    d._learned_weight = 0.55
    d.save_state(p)
    d2 = IntuitionConvergenceDetector()
    import json
    with open(p) as f:
        d2.from_dict(json.load(f))
    assert d2._total_convergence_events == 7
    assert abs(d2._learned_weight - 0.55) < 1e-6


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
