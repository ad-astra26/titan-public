"""Phase 9 Chunk 9L — Guardian wave-based boot regression test.

Verifies that `Guardian._compute_boot_waves` produces:
  - Dependency-correct topological ordering (a module's MODULE deps
    always ship in an earlier wave),
  - Layer-ascending tie-break (L1 → L2 → L3, then alphabetic) inside
    each topo level,
  - Wave-size capping per `boot_wave_size`.

Per RFP §3F.2.6 Chunk 9L acceptance: "topological dep order honored,
layer-ascending + alphabetic tie-break within wave, wave size respected."
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from titan_hcl.bus import DivineBus
from titan_hcl.guardian_hcl.core import Guardian
from titan_hcl.guardian_hcl.module_registry import ModuleInfo, ModuleSpec
from titan_hcl.supervision import (
    Dependency,
    DependencyAction,
    DependencyKind,
    DependencySeverity,
)


def _mk_spec(name: str, *, layer: str = "L2",
             deps: list[str] | None = None,
             autostart: bool = True) -> ModuleSpec:
    return ModuleSpec(
        name=name,
        entry_fn=lambda *a, **k: None,
        layer=layer,
        autostart=autostart,
        lazy=False,
        dependencies=[
            Dependency(
                name=d, kind=DependencyKind.MODULE,
                severity=DependencySeverity.CRITICAL,
                action=DependencyAction.ENSURE_RUNNING,
                check=lambda: True,
            )
            for d in (deps or [])
        ],
    )


def _make_guardian_with_modules(specs: list[ModuleSpec],
                                wave_size: int = 5) -> Guardian:
    bus = MagicMock(spec=DivineBus)
    bus.subscribe.return_value = MagicMock()
    g = Guardian(bus, config={"boot_wave_size": wave_size})
    for spec in specs:
        g._modules[spec.name] = ModuleInfo(spec=spec)
    return g


def test_no_deps_alphabetic_within_layer() -> None:
    """With no deps, modules sort by (layer, name) and batch into waves."""
    specs = [
        _mk_spec("z_l1_a", layer="L1"),
        _mk_spec("a_l3_x", layer="L3"),
        _mk_spec("m_l2_b", layer="L2"),
        _mk_spec("a_l1_b", layer="L1"),
    ]
    g = _make_guardian_with_modules(specs, wave_size=10)
    waves = g._compute_boot_waves([s.name for s in specs])
    assert waves == [["a_l1_b", "z_l1_a", "m_l2_b", "a_l3_x"]]


def test_wave_size_caps_per_topo_level() -> None:
    """A topo level larger than wave_size splits into multiple waves."""
    specs = [_mk_spec(f"mod_{i}", layer="L2") for i in range(12)]
    g = _make_guardian_with_modules(specs, wave_size=5)
    waves = g._compute_boot_waves([s.name for s in specs])
    assert len(waves) == 3
    assert len(waves[0]) == 5 and len(waves[1]) == 5 and len(waves[2]) == 2


def test_dependency_topo_order_enforced() -> None:
    """A module's deps always ship in an earlier wave."""
    specs = [
        _mk_spec("memory", layer="L2"),
        _mk_spec("meditation", layer="L2", deps=["memory"]),
        _mk_spec("agno_worker", layer="L2", deps=["memory"]),
    ]
    g = _make_guardian_with_modules(specs, wave_size=10)
    waves = g._compute_boot_waves([s.name for s in specs])
    # memory MUST be in an earlier wave than meditation + agno_worker
    flat = [n for w in waves for n in w]
    assert flat.index("memory") < flat.index("meditation")
    assert flat.index("memory") < flat.index("agno_worker")
    # Two topo levels (memory alone, then the two dependents)
    assert waves[0] == ["memory"]
    assert sorted(waves[1]) == ["agno_worker", "meditation"]


def test_l1_before_l2_inside_same_topo_level() -> None:
    """When deps don't force ordering, L1 ships before L2 ships before L3."""
    specs = [
        _mk_spec("heavy", layer="L3"),
        _mk_spec("medium", layer="L2"),
        _mk_spec("light", layer="L1"),
    ]
    g = _make_guardian_with_modules(specs, wave_size=10)
    waves = g._compute_boot_waves([s.name for s in specs])
    assert waves == [["light", "medium", "heavy"]]


def test_external_dep_ignored_for_topo() -> None:
    """Deps targeting modules not in the autostart set are ignored."""
    specs = [
        _mk_spec("a", layer="L2", deps=["external_unknown"]),
    ]
    g = _make_guardian_with_modules(specs, wave_size=10)
    waves = g._compute_boot_waves(["a"])
    assert waves == [["a"]]


def test_chain_of_deps_each_in_own_wave() -> None:
    """A→B→C dependency chain produces 3 waves."""
    specs = [
        _mk_spec("c", layer="L2", deps=["b"]),
        _mk_spec("b", layer="L2", deps=["a"]),
        _mk_spec("a", layer="L2"),
    ]
    g = _make_guardian_with_modules(specs, wave_size=10)
    waves = g._compute_boot_waves(["a", "b", "c"])
    assert waves == [["a"], ["b"], ["c"]]


def test_cycle_fallback_does_not_loop() -> None:
    """A circular dep (A→B, B→A) shouldn't infinite-loop; falls back to flat."""
    specs = [
        _mk_spec("a", layer="L2", deps=["b"]),
        _mk_spec("b", layer="L2", deps=["a"]),
    ]
    g = _make_guardian_with_modules(specs, wave_size=10)
    waves = g._compute_boot_waves(["a", "b"])
    # Should produce a single wave with both members (alphabetic fallback).
    flat = [n for w in waves for n in w]
    assert set(flat) == {"a", "b"}
    assert len(flat) == 2
