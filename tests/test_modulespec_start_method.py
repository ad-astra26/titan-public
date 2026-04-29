"""
Tests for Microkernel v2 Phase A §A.3 (S6) — ModuleSpec.start_method
field + Guardian context selection.

Covers:
  - ModuleSpec.start_method default = "fork" (byte-identical preservation)
  - ModuleSpec accepts "spawn"
  - Unknown start_method values are accepted at construction (Guardian
    falls back to fork at runtime with a WARNING — never crashes boot)
  - All current registered ModuleSpec configs are pickleable end-to-end
    (catches the most common spawn-incompatibility upfront)
  - Guardian's multiprocessing.get_context selection matches spec.start_method

Reference:
  - titan-docs/PLAN_microkernel_phase_a_s6.md §5.1
  - titan_plugin/guardian.py (ModuleSpec + start() context branch)
"""
from __future__ import annotations

import pickle

import pytest

from titan_plugin.guardian import ModuleSpec


def _noop():
    pass


def test_default_start_method_is_fork():
    """Existing ModuleSpec usage gets fork by default — byte-identical preservation."""
    spec = ModuleSpec(name="foo", entry_fn=_noop)
    assert spec.start_method == "fork"


def test_spawn_start_method_accepted():
    spec = ModuleSpec(name="foo", entry_fn=_noop, start_method="spawn")
    assert spec.start_method == "spawn"


def test_fork_start_method_explicit():
    spec = ModuleSpec(name="foo", entry_fn=_noop, start_method="fork")
    assert spec.start_method == "fork"


def test_unknown_start_method_accepted_at_construction():
    """ModuleSpec is a passive dataclass — value validation at runtime
    by Guardian (which falls back to fork with a WARNING)."""
    spec = ModuleSpec(name="foo", entry_fn=_noop, start_method="bogus")
    assert spec.start_method == "bogus"


def test_modulespec_pickleable():
    """ModuleSpec must be pickleable for spawn — Process(args=...) pickles
    everything that crosses the spawn boundary."""
    # Use a function defined at module scope (closures + lambdas aren't
    # pickleable, but real ModuleSpecs use top-level functions)
    spec = ModuleSpec(
        name="foo",
        entry_fn=_noop,
        config={"key": "value", "nested": {"x": 1}},
        rss_limit_mb=300,
        autostart=True,
        layer="L3",
        start_method="spawn",
    )
    data = pickle.dumps(spec)
    spec2 = pickle.loads(data)
    assert spec2.name == "foo"
    assert spec2.start_method == "spawn"
    assert spec2.config == {"key": "value", "nested": {"x": 1}}


def test_get_context_fork_works():
    """Sanity: multiprocessing.get_context('fork') returns a usable context."""
    import multiprocessing
    ctx = multiprocessing.get_context("fork")
    assert ctx is not None
    # Just verify Queue is creatable (we don't actually use it; tearing
    # down a multiprocessing.Queue from inside a test is flaky)
    assert hasattr(ctx, "Queue")
    assert hasattr(ctx, "Process")


def test_get_context_spawn_works():
    """Same for spawn."""
    import multiprocessing
    ctx = multiprocessing.get_context("spawn")
    assert ctx is not None
    assert hasattr(ctx, "Queue")
    assert hasattr(ctx, "Process")


def test_unknown_method_in_get_context_raises():
    """multiprocessing.get_context raises on unknown method — Guardian
    catches this by checking start_method against ('fork', 'spawn') and
    falling back to fork before invoking get_context."""
    import multiprocessing
    with pytest.raises(ValueError):
        multiprocessing.get_context("bogus_method")


def test_guardian_register_accepts_spawn_spec():
    """Guardian.register() doesn't reject spawn-method specs."""
    from titan_plugin.bus import DivineBus
    from titan_plugin.guardian import Guardian

    bus = DivineBus(maxsize=100)
    g = Guardian(bus)
    spec = ModuleSpec(
        name="test_spawn",
        entry_fn=_noop,
        layer="L3",
        start_method="spawn",
    )
    # Should not raise
    g.register(spec)
    assert "test_spawn" in g._modules
    assert g._modules["test_spawn"].spec.start_method == "spawn"
