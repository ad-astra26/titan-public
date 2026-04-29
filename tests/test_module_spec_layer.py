"""
Tests for Microkernel v2 Phase A §A.5 — layer assignment on ModuleSpec.

Covers:
  - ModuleSpec has a `layer` field with default L3
  - Guardian.register() validates layer ∈ {L0, L1, L2, L3}
  - Every Guardian-registered module matches LAYER_CANON
  - layer_stats() returns accurate per-layer counts
  - get_layer() / get_modules_by_layer() queries

Does NOT boot a real TitanCore — uses mocked Guardian + mocked entry_fn
to avoid touching heavy imports.

Reference: titan-docs/PLAN_microkernel_phase_a.md §4.2
"""
from __future__ import annotations

import pytest

from titan_plugin._layer_canon import LAYER_CANON, VALID_LAYERS, validate_layer
from titan_plugin.bus import DivineBus
from titan_plugin.guardian import Guardian, ModuleSpec, ModuleState


def _noop_entry(recv_q, send_q, name, config):  # pragma: no cover — never invoked in these tests
    pass


@pytest.fixture
def guardian():
    bus = DivineBus(maxsize=100)
    g = Guardian(bus)
    yield g


# ── ModuleSpec layer field ──────────────────────────────────────────


def test_module_spec_default_layer_is_l3():
    spec = ModuleSpec(name="test", entry_fn=_noop_entry)
    assert spec.layer == "L3"


def test_module_spec_explicit_layer():
    spec = ModuleSpec(name="test", entry_fn=_noop_entry, layer="L1")
    assert spec.layer == "L1"


# ── Validation ──────────────────────────────────────────────────────


def test_valid_layers_frozenset():
    assert VALID_LAYERS == frozenset({"L0", "L1", "L2", "L3"})


def test_validate_layer_accepts_all_valid():
    for layer in ("L0", "L1", "L2", "L3"):
        assert validate_layer(layer) == layer


def test_validate_layer_rejects_invalid():
    for bad in ("L4", "L-1", "l1", "", "L1 ", "kernel"):
        with pytest.raises(ValueError, match="Invalid layer"):
            validate_layer(bad)


def test_register_rejects_invalid_layer(guardian):
    spec = ModuleSpec(name="bogus", entry_fn=_noop_entry, layer="L5")
    with pytest.raises(ValueError, match="Invalid layer"):
        guardian.register(spec)
    # Module must NOT be in the registry after failed register
    assert "bogus" not in guardian._modules


def test_register_accepts_all_valid_layers(guardian):
    for i, layer in enumerate(("L0", "L1", "L2", "L3")):
        guardian.register(ModuleSpec(
            name=f"mod_{i}", entry_fn=_noop_entry, layer=layer))
    assert len(guardian._modules) == 4


# ── Canon conformance ──────────────────────────────────────────────


def test_layer_canon_covers_17_production_modules():
    """Every module that plugin.py / legacy_core.py registers must have a canon entry.

    17 modules total: 16 from v5_core.py + warning_monitor (added 2026-04-25
    silent-swallow runtime visibility worker, plugin.py:317).
    """
    expected_modules = {
        "imw", "observatory_writer",
        "body", "mind", "spirit",
        "memory", "rl", "cgn", "emot_cgn", "language",
        "meta_teacher", "timechain",
        "llm", "media", "backup", "knowledge",
        "warning_monitor",
    }
    assert set(LAYER_CANON.keys()) == expected_modules


def test_layer_canon_distribution():
    """L1 has 4, L2 has 7, L3 has 6, L0 has 0."""
    from collections import Counter
    counts = Counter(LAYER_CANON.values())
    assert counts["L1"] == 4  # body, mind, spirit, imw
    assert counts["L2"] == 7  # memory, rl, cgn, emot_cgn, language, meta_teacher, timechain
    assert counts["L3"] == 6  # llm, media, backup, knowledge, observatory_writer, warning_monitor
    assert counts["L0"] == 0  # L0 is in-process kernel, no Guardian modules


def test_layer_canon_trinity_daemons_at_l1():
    for name in ("body", "mind", "spirit"):
        assert LAYER_CANON[name] == "L1"


def test_layer_canon_cgn_at_l2():
    """CGN is the L2 concept-value state registry per project_cgn_as_higher_state_registry.md."""
    assert LAYER_CANON["cgn"] == "L2"


def test_layer_canon_knowledge_at_l3():
    """rFP L3 explicitly lists 'Knowledge search'."""
    assert LAYER_CANON["knowledge"] == "L3"


def test_layer_canon_timechain_at_l2():
    """TimeChain is cognitive substrate, not a Trinity daemon."""
    assert LAYER_CANON["timechain"] == "L2"


def test_layer_canon_imw_at_l1():
    """IMW writes inner_memory.db which is L1's DB per rFP Q2."""
    assert LAYER_CANON["imw"] == "L1"


def test_layer_canon_observatory_writer_at_l3():
    """observatory.db is L3's DB per rFP Q2."""
    assert LAYER_CANON["observatory_writer"] == "L3"


# ── Query methods ──────────────────────────────────────────────────


def test_get_layer_returns_registered_layer(guardian):
    guardian.register(ModuleSpec(name="body", entry_fn=_noop_entry, layer="L1"))
    guardian.register(ModuleSpec(name="llm", entry_fn=_noop_entry, layer="L3"))
    assert guardian.get_layer("body") == "L1"
    assert guardian.get_layer("llm") == "L3"


def test_get_layer_unknown_module_returns_none(guardian):
    assert guardian.get_layer("nonexistent") is None


def test_get_modules_by_layer_returns_sorted(guardian):
    for name in ("zeta", "alpha", "mu"):
        guardian.register(ModuleSpec(name=name, entry_fn=_noop_entry, layer="L2"))
    result = guardian.get_modules_by_layer("L2")
    assert result == ["alpha", "mu", "zeta"]


def test_get_modules_by_layer_empty_returns_empty_list(guardian):
    assert guardian.get_modules_by_layer("L0") == []


def test_layer_stats_structure(guardian):
    stats = guardian.layer_stats()
    assert set(stats.keys()) == {"L0", "L1", "L2", "L3"}
    for bucket in stats.values():
        assert set(bucket.keys()) == {"total", "running", "crashed", "disabled"}


def test_layer_stats_counts_totals(guardian):
    # 2 L1, 3 L2, 1 L3
    for name in ("body", "mind"):
        guardian.register(ModuleSpec(name=name, entry_fn=_noop_entry, layer="L1"))
    for name in ("memory", "rl", "cgn"):
        guardian.register(ModuleSpec(name=name, entry_fn=_noop_entry, layer="L2"))
    guardian.register(ModuleSpec(name="llm", entry_fn=_noop_entry, layer="L3"))
    stats = guardian.layer_stats()
    assert stats["L1"]["total"] == 2
    assert stats["L2"]["total"] == 3
    assert stats["L3"]["total"] == 1
    assert stats["L0"]["total"] == 0
    # Nothing is running (no process started)
    for layer in ("L0", "L1", "L2", "L3"):
        assert stats[layer]["running"] == 0


def test_layer_stats_counts_by_state(guardian):
    for name in ("alpha", "beta", "gamma"):
        guardian.register(ModuleSpec(name=name, entry_fn=_noop_entry, layer="L2"))
    # Manually set states (simulating post-boot scenarios)
    guardian._modules["alpha"].state = ModuleState.RUNNING
    guardian._modules["beta"].state = ModuleState.CRASHED
    guardian._modules["gamma"].state = ModuleState.DISABLED
    stats = guardian.layer_stats()
    assert stats["L2"]["total"] == 3
    assert stats["L2"]["running"] == 1
    assert stats["L2"]["crashed"] == 1
    assert stats["L2"]["disabled"] == 1


# ── get_status exposes layer ──────────────────────────────────────


def test_get_status_includes_layer(guardian):
    guardian.register(ModuleSpec(name="body", entry_fn=_noop_entry, layer="L1"))
    status = guardian.get_status()
    assert "body" in status
    assert status["body"]["layer"] == "L1"
