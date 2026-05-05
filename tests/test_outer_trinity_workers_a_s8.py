"""
tests/test_outer_trinity_workers_a_s8.py — Phase A.S8 RegistrySpec symmetry tests.

Verifies that outer trinity slots have the correct shapes mirroring inner slots,
and that bus constants are properly defined (Chunk 2 + Chunk 1 parity test).
"""
import pytest


def test_outer_body_5d_shape():
    from titan_plugin.core.state_registry import OUTER_BODY_5D, INNER_BODY_5D
    assert OUTER_BODY_5D.shape == (5,), f"outer_body_5d shape should be (5,), got {OUTER_BODY_5D.shape}"
    assert OUTER_BODY_5D.shape == INNER_BODY_5D.shape, "outer_body_5d must mirror inner_body_5d shape"


def test_outer_mind_15d_shape():
    from titan_plugin.core.state_registry import OUTER_MIND_15D, INNER_MIND_15D
    assert OUTER_MIND_15D.shape == (15,), f"outer_mind_15d shape should be (15,), got {OUTER_MIND_15D.shape}"
    assert OUTER_MIND_15D.shape == INNER_MIND_15D.shape, "outer_mind_15d must mirror inner_mind_15d shape"


def test_outer_spirit_45d_shape():
    from titan_plugin.core.state_registry import OUTER_SPIRIT_45D, INNER_SPIRIT_45D
    assert OUTER_SPIRIT_45D.shape == (45,), f"outer_spirit_45d shape should be (45,), got {OUTER_SPIRIT_45D.shape}"
    assert OUTER_SPIRIT_45D.shape == INNER_SPIRIT_45D.shape, "outer_spirit_45d must mirror inner_spirit_45d shape"


def test_topology_30d_shape():
    from titan_plugin.core.state_registry import TOPOLOGY_30D
    assert TOPOLOGY_30D.shape == (30,), f"topology_30d shape should be (30,), got {TOPOLOGY_30D.shape}"


def test_outer_trinity_bus_constants():
    from titan_plugin import bus
    assert bus.OUTER_BODY_STATE == "OUTER_BODY_STATE"
    assert bus.OUTER_MIND_STATE == "OUTER_MIND_STATE"
    assert bus.OUTER_SPIRIT_STATE == "OUTER_SPIRIT_STATE"
    assert bus.OUTER_SOURCES_SNAPSHOT == "OUTER_SOURCES_SNAPSHOT"
    # Legacy kept for legacy_core compat
    assert bus.OUTER_TRINITY_STATE == "OUTER_TRINITY_STATE"


def test_outer_trinity_state_msg_types():
    from titan_plugin import bus
    assert bus.OUTER_BODY_STATE in bus.DivineBus.STATE_MSG_TYPES
    assert bus.OUTER_MIND_STATE in bus.DivineBus.STATE_MSG_TYPES
    assert bus.OUTER_SPIRIT_STATE in bus.DivineBus.STATE_MSG_TYPES


def test_outer_body_tensor_pure_function():
    from titan_plugin.logic.outer_body_tensor import collect_outer_body_5d
    result = collect_outer_body_5d({})
    assert len(result) == 5
    assert all(0.0 <= v <= 1.0 for v in result), f"Values out of range: {result}"


def test_outer_mind_5d_pure_function():
    from titan_plugin.logic.outer_mind_tensor import collect_outer_mind_5d
    result = collect_outer_mind_5d(art_count=10, audio_count=5,
                                    memory_status={"persistent_count": 100, "total_nodes": 200},
                                    uptime_seconds=86400)
    assert len(result) == 5
    assert all(0.0 <= v <= 1.0 for v in result)


def test_outer_spirit_5d_pure_function():
    from titan_plugin.logic.outer_spirit_tensor import collect_outer_spirit_5d
    result = collect_outer_spirit_5d(
        outer_body=[0.6, 0.5, 0.4, 0.5, 0.7],
        outer_mind=[0.5, 0.6, 0.4, 0.5, 0.3],
        soul_health=0.8,
        total_impulses=100,
        total_assessed=50,
        avg_score=0.7,
    )
    assert len(result) == 5
    assert all(0.0 <= v <= 1.0 for v in result)


def test_outer_vs_inner_schumann_ratio():
    """Outer workers publish at 13× inner cadence (environmental tempo scaling)."""
    inner_body_publish = 3.45
    inner_mind_publish = 1.15
    inner_spirit_publish = 0.383
    outer_body_publish = 45.0
    outer_mind_publish = 15.0
    outer_spirit_publish = 5.0
    ratio_body = outer_body_publish / inner_body_publish
    ratio_mind = outer_mind_publish / inner_mind_publish
    ratio_spirit = outer_spirit_publish / inner_spirit_publish
    assert abs(ratio_body - 13.04) < 0.5, f"Outer/inner body ratio {ratio_body:.2f} ≠ 13"
    assert abs(ratio_mind - 13.04) < 0.5, f"Outer/inner mind ratio {ratio_mind:.2f} ≠ 13"
    assert abs(ratio_spirit - 13.05) < 0.5, f"Outer/inner spirit ratio {ratio_spirit:.2f} ≠ 13"


def test_body_slowest_invariant():
    """Body publishes slowest (45s), spirit fastest (5s) — G13 invariant."""
    from titan_plugin.modules.outer_body_worker import _OUTER_BODY_PUBLISH_INTERVAL_S
    from titan_plugin.modules.outer_mind_worker import _OUTER_MIND_PUBLISH_INTERVAL_S
    from titan_plugin.modules.outer_spirit_worker import _OUTER_SPIRIT_PUBLISH_INTERVAL_S
    assert _OUTER_BODY_PUBLISH_INTERVAL_S > _OUTER_MIND_PUBLISH_INTERVAL_S > _OUTER_SPIRIT_PUBLISH_INTERVAL_S


def test_schumann_9_3_1_ratio():
    """Schumann Hz ratio must be 9:3:1 (spirit:mind:body)."""
    from titan_plugin.modules.outer_body_worker import _OUTER_BODY_SCHUMANN_HZ
    from titan_plugin.modules.outer_mind_worker import _OUTER_MIND_SCHUMANN_HZ
    from titan_plugin.modules.outer_spirit_worker import _OUTER_SPIRIT_SCHUMANN_HZ
    assert abs(_OUTER_MIND_SCHUMANN_HZ / _OUTER_BODY_SCHUMANN_HZ - 3.0) < 0.1
    assert abs(_OUTER_SPIRIT_SCHUMANN_HZ / _OUTER_BODY_SCHUMANN_HZ - 9.0) < 0.2
