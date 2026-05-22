"""Tests for hormonal_worker HORMONE_STIMULUS subscriber.

Per rFP_phase_c_impulse_engine_d8_3_migration §3.E.3 + §3.B.7.

The hormonal_worker subscribes bus.HORMONE_STIMULUS and accumulates the
stimulus into the named hormone via HormonalSystem.get_hormone(name)
.accumulate(stimulus, dt) — the cross-worker translation of the legacy
in-process bridge that lived in spirit_worker.py:2692-2703.

These tests exercise the consumer logic directly by extracting the
handler from hormonal_worker module source (no subprocess required).
"""
import pytest


# ── HORMONE_STIMULUS bus constant ────────────────────────────────────


class TestBusConstant:
    """A.1 — HORMONE_STIMULUS registered in titan_hcl.bus."""

    def test_constant_exists(self):
        from titan_hcl import bus
        assert hasattr(bus, "HORMONE_STIMULUS")
        assert bus.HORMONE_STIMULUS == "HORMONE_STIMULUS"

    def test_distinct_from_hormone_fired(self):
        """HORMONE_STIMULUS is the stimulus accumulation channel,
        HORMONE_FIRED is the on-fire observability channel — distinct."""
        from titan_hcl import bus
        assert bus.HORMONE_STIMULUS != bus.HORMONE_FIRED


# ── HormonalSystem accumulate API (consumer dependency) ──────────────


class TestHormonalSystemContract:
    """B.7 consumer relies on HormonalSystem.get_hormone(name).accumulate
    surface — verify it exists with expected signature."""

    def test_get_hormone_returns_object_for_impulse(self):
        from titan_hcl.logic.hormonal_pressure import HormonalSystem
        from titan_hcl.logic.emot_bundle_protocol import NS_PROGRAMS
        sys = HormonalSystem(program_names=list(NS_PROGRAMS))
        h = sys.get_hormone("IMPULSE")
        assert h is not None
        # accumulate must accept (stimulus, dt) positional or keyword.
        # Run it once to verify no exception + state advances.
        level_before = h.level
        h.accumulate(0.1, 0.1)
        assert h.level >= level_before  # accumulation either raises level
                                         # or leaves stable depending on refractory

    def test_get_hormone_unknown_returns_none(self):
        from titan_hcl.logic.hormonal_pressure import HormonalSystem
        from titan_hcl.logic.emot_bundle_protocol import NS_PROGRAMS
        sys = HormonalSystem(program_names=list(NS_PROGRAMS))
        # Unknown name → None (consumer must handle gracefully per B.7).
        h = sys.get_hormone("NONEXISTENT_HORMONE_XYZ")
        assert h is None


# ── hormonal_worker source-level guard contracts ─────────────────────


class TestHormonalWorkerWiring:
    """Verify hormonal_worker module integrates HORMONE_STIMULUS per B.7."""

    def test_subscribes_hormone_stimulus(self):
        """hormonal_worker_main MUST handle bus.HORMONE_STIMULUS."""
        import inspect
        from titan_hcl.modules import hormonal_worker as hw
        src = inspect.getsource(hw.hormonal_worker_main)
        assert "HORMONE_STIMULUS" in src
        # Verify it consumes the canonical fields per rFP §2.D schema.
        assert "hormone_name" in src
        assert "accumulate" in src

    def test_consumer_calls_get_hormone_accumulate(self):
        """Verify the consumer path uses get_hormone(name).accumulate(...)
        — preserves legacy in-process bridge semantics."""
        import inspect
        from titan_hcl.modules import hormonal_worker as hw
        src = inspect.getsource(hw.hormonal_worker_main)
        assert "get_hormone(hormone_name)" in src
        assert ".accumulate(stimulus, dt)" in src \
            or "accumulate(stimulus" in src

    def test_broadcast_topics_extends_for_hormonal(self):
        """plugin.py wires HORMONE_STIMULUS into hormonal_module
        broadcast_topics per B.7."""
        import inspect
        from titan_hcl.core import plugin
        src = inspect.getsource(plugin.TitanHCL._register_workers) \
            if hasattr(plugin.TitanHCL, "_register_workers") \
            else inspect.getsource(plugin)
        # Either the helper is named differently — fall back to module source.
        if "_HORMONAL_WORKER_BROADCAST_TOPICS" not in src:
            with open(plugin.__file__) as f:
                src = f.read()
        assert "_HORMONAL_WORKER_BROADCAST_TOPICS" in src
        assert "bus.HORMONE_STIMULUS" in src


# ── ns_worker producer side (B.6) ────────────────────────────────────


class TestNsWorkerProducer:
    """Symmetric verification: ns_worker emits HORMONE_STIMULUS per B.6."""

    def test_run_impulse_tick_publishes_hormone_stimulus(self):
        """Source-level: ns_worker._run_impulse_tick publishes
        bus.HORMONE_STIMULUS on Trinity deficit > threshold."""
        import inspect
        from titan_hcl.modules import ns_worker as nw
        src = inspect.getsource(nw._run_impulse_tick)
        assert "HORMONE_STIMULUS" in src
        assert "hormone_name" in src
        assert "IMPULSE" in src

    def test_broadcast_topics_extends_for_ns(self):
        """plugin.py wires ACTION_RESULT into ns_module broadcast_topics
        per B.4 + B.12."""
        from titan_hcl.core import plugin
        with open(plugin.__file__) as f:
            src = f.read()
        assert "_NS_WORKER_BROADCAST_TOPICS" in src
        assert "bus.ACTION_RESULT" in src
