"""
tests/test_dream_consolidation_offtick.py — D-SPEC-105 / v1.43.0.

Off-tick dream-consolidation suite restoration on the live Phase C dream path
(rFP_dream_consolidation_suite_offtick_restoration). Covers:

  - BEGIN_DREAMING spawns a daemon thread that runs _on_dream_begin off-tick.
  - Single-flight guard: a second spawn while one is alive is skipped.
  - END_DREAMING runs _on_dream_end off-tick, after the begin-thread finishes.
  - tick() no longer calls _run_dream_distillation inline (no double-distill —
    distillation lives inside _on_dream_begin which runs in the thread).
  - join_consolidation blocks until the in-flight thread finishes (shutdown).
  - NS waking _train_all is gated off while is_dreaming (thread-safety §8).
"""
import tempfile
import threading
import time

import pytest


def _make_coordinator():
    from titan_hcl.logic.inner_state import InnerState
    from titan_hcl.logic.spirit_state import SpiritState
    from titan_hcl.logic.observables import ObservableEngine
    from titan_hcl.logic.inner_coordinator import InnerTrinityCoordinator

    inner = InnerState()
    spirit = SpiritState()
    obs_engine = ObservableEngine()
    coord = InnerTrinityCoordinator(inner, spirit, obs_engine)
    return coord, inner


class TestOffTickConsolidationThread:
    def test_begin_spawns_thread_and_runs_on_dream_begin(self):
        coord, _ = _make_coordinator()
        ran = threading.Event()
        coord._on_dream_begin = lambda: ran.set()

        coord._spawn_dream_consolidation()
        coord.join_consolidation(timeout=5.0)

        assert ran.is_set()
        assert coord._consolidation_thread is not None
        assert not coord._consolidation_thread.is_alive()

    def test_thread_is_daemon(self):
        coord, _ = _make_coordinator()
        block = threading.Event()
        coord._on_dream_begin = lambda: block.wait(timeout=5.0)

        coord._spawn_dream_consolidation()
        try:
            assert coord._consolidation_thread.daemon is True
        finally:
            block.set()
            coord.join_consolidation(timeout=5.0)

    def test_single_flight_guard_skips_overlapping_spawn(self):
        coord, _ = _make_coordinator()
        block = threading.Event()
        call_count = {"n": 0}

        def _slow_begin():
            call_count["n"] += 1
            block.wait(timeout=5.0)

        coord._on_dream_begin = _slow_begin

        coord._spawn_dream_consolidation()       # starts + blocks in _on_dream_begin
        first_thread = coord._consolidation_thread
        # Wait until the first thread has actually entered _on_dream_begin so the
        # single-flight guard sees it alive.
        for _ in range(50):
            if call_count["n"] >= 1:
                break
            time.sleep(0.01)

        coord._spawn_dream_consolidation()       # must be skipped (single-flight)
        assert coord._consolidation_thread is first_thread

        block.set()
        coord.join_consolidation(timeout=5.0)
        assert call_count["n"] == 1              # second spawn never ran the body

    def test_dream_end_runs_off_tick(self):
        coord, _ = _make_coordinator()
        seen = {}
        coord._on_dream_end = lambda summary: seen.update(summary or {})

        coord._spawn_dream_end({"distilled_insights": [], "marker": 7})
        coord.join_consolidation(timeout=5.0)

        assert seen.get("marker") == 7

    def test_dream_end_waits_for_in_flight_begin(self):
        coord, _ = _make_coordinator()
        order = []
        begin_block = threading.Event()

        def _begin():
            begin_block.wait(timeout=5.0)
            order.append("begin")

        def _end(summary):
            order.append("end")

        coord._on_dream_begin = _begin
        coord._on_dream_end = _end

        coord._spawn_dream_consolidation()
        # end is requested while begin is still blocked — it must join begin first
        coord._spawn_dream_end({})
        time.sleep(0.05)
        assert order == []          # neither ran yet (begin blocked, end joining)

        begin_block.set()
        coord.join_consolidation(timeout=5.0)
        # the dream-end thread joined the begin thread, so begin completes first
        assert order == ["begin", "end"]

    def test_join_consolidation_no_thread_is_noop(self):
        coord, _ = _make_coordinator()
        # No thread ever spawned — must not raise.
        coord.join_consolidation(timeout=0.1)

    def test_on_dream_begin_runs_mini_reasoner_when_wired(self):
        """_on_dream_begin must invoke _mini_registry.consolidate_all when the
        registry is wired onto the coordinator. Regression guard: the registry
        was created in cognitive_worker but never assigned to the coordinator,
        so the `if hasattr(self,'_mini_registry')` guard was always False and
        the mini-reasoner dream consolidation silently no-op'd (D-SPEC-105)."""
        coord, _ = _make_coordinator()
        calls = {"consolidate": 0, "save": 0}

        class MockMiniRegistry:
            def consolidate_all(self, boost_factor=2.0):
                calls["consolidate"] += 1
                return {"body_x": {"samples": 4, "loss": 0.1}}

            def save_all(self):
                calls["save"] += 1

        coord._mini_registry = MockMiniRegistry()
        # All other engines remain unset → their guarded blocks no-op.
        coord._on_dream_begin()

        assert calls["consolidate"] == 1
        assert calls["save"] == 1


class TestTickDreamTransitionWiring:
    """tick() spawns the off-tick suite on BEGIN_DREAMING and does NOT run the
    distillation inline anymore (no double-distill)."""

    def _tensors(self):
        return {
            "inner_body": [0.5] * 5,
            "inner_mind": [0.5] * 5,
            "inner_spirit": [0.5] * 5,
        }

    def _wire_dreaming(self, coord, inner, transition):
        class MockDreaming:
            def __init__(self):
                self.last_fatigue = 0.0
                self._last_fatigue_breakdown = {}
                self.last_readiness = 0.0

            def check_transition(self, *a, **k):
                return transition

            def begin_dreaming(self, _inner):
                inner.is_dreaming = True

            def end_dreaming(self, _inner):
                inner.is_dreaming = False
                return {"distilled_insights": [], "duration_s": 1.0}

        coord.dreaming = MockDreaming()

    def test_begin_dreaming_spawns_suite_not_inline_distill(self):
        coord, inner = _make_coordinator()
        self._wire_dreaming(coord, inner, "BEGIN_DREAMING")

        spawned = {"n": 0}
        inline_distill = {"n": 0}
        coord._spawn_dream_consolidation = lambda: spawned.__setitem__(
            "n", spawned["n"] + 1)
        coord._run_dream_distillation = lambda: inline_distill.__setitem__(
            "n", inline_distill["n"] + 1)

        coord.tick(self._tensors())

        assert spawned["n"] == 1            # off-tick suite spawned
        assert inline_distill["n"] == 0     # NOT run inline by tick (no double-distill)

    def test_end_dreaming_spawns_dream_end(self):
        coord, inner = _make_coordinator()
        inner.is_dreaming = True
        self._wire_dreaming(coord, inner, "END_DREAMING")

        end_calls = {"n": 0}
        coord._spawn_dream_end = lambda summary: end_calls.__setitem__(
            "n", end_calls["n"] + 1)

        coord.tick(self._tensors())
        assert end_calls["n"] == 1


class TestNSWakingTrainGate:
    """NS online _train_all is suspended while is_dreaming so the off-tick
    consolidation suite has exclusive access to the program nets (§8)."""

    def _make_ns(self, tmpdir):
        from titan_hcl.logic.neural_nervous_system import NeuralNervousSystem
        cfg = {
            "warmup_steps": 0,
            "train_every_n": 3,
            "batch_size": 4,
            "save_every_n": 100000,
            "programs": {
                "REFLEX": {"enabled": True, "fire_threshold": 0.3,
                           "input_features": "standard"},
                "FOCUS": {"enabled": True, "fire_threshold": 0.25,
                          "input_features": "standard"},
            },
        }
        return NeuralNervousSystem(cfg, data_dir=tmpdir)

    def _observables(self):
        parts = ["inner_body", "inner_mind", "inner_spirit",
                 "outer_body", "outer_mind", "outer_spirit"]
        return {p: {"coherence": 0.9, "magnitude": 0.7, "velocity": 0.05,
                    "direction": 0.95, "polarity": 0.1} for p in parts}

    def test_train_all_suppressed_while_dreaming(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ns = self._make_ns(tmpdir)
            calls = {"n": 0}
            ns._train_all = lambda: calls.__setitem__("n", calls["n"] + 1)

            ns.set_dreaming(True)
            ns.update_observation_space(observables=self._observables())
            for _ in range(12):  # well past train_every_n=3
                ns.evaluate(self._observables())
            assert calls["n"] == 0   # no waking training during sleep

    def test_train_all_resumes_when_awake(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ns = self._make_ns(tmpdir)
            calls = {"n": 0}
            ns._train_all = lambda: calls.__setitem__("n", calls["n"] + 1)

            ns.set_dreaming(False)
            ns.update_observation_space(observables=self._observables())
            for _ in range(12):
                ns.evaluate(self._observables())
            assert calls["n"] >= 1   # online training fires while awake


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-p", "no:anchorpy"])
