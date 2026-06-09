"""
test_life_force_inputs_builder — coverage for compute_life_force_inputs.

§4.G chunk G4. Validates the 16-input aggregation extracted from
cognitive_worker.py:2370-2473 (Track 1 drift retirement). One test per
input field + meta-shape + graceful-degradation paths.

Pure function — no SHM, no bus, no subprocess. Mocks upstream sources
(pi_monitor, coordinator, NNS, expression_state_reader, etc.).
"""
from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from titan_hcl.logic.life_force_inputs_builder import (
    _STUB_ANCHOR_FRESHNESS,
    _STUB_INFRASTRUCTURE_HEALTH,
    _STUB_SOL_BALANCE,
    _STUB_SOVEREIGNTY_INDEX,
    compute_life_force_inputs,
)


# ── Test fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def cold_inputs() -> dict[str, Any]:
    """All upstream sources cold/None — every output should land on its fallback."""
    return dict(
        coordinator=None,
        pi_monitor=None,
        neural_nervous_system=None,
        latest_epoch={},
        consciousness=None,
        topology_snap=None,
        expression_state_reader=None,
        vocab_db_path="/nonexistent/path/that/will/fail.db",
    )


@pytest.fixture
def warm_pi_monitor():
    """Healthy pi_monitor with heartbeat + age."""
    return SimpleNamespace(heartbeat_ratio=0.20, developmental_age=300)


@pytest.fixture
def warm_nm_sys():
    """Healthy neuromodulator system."""
    nm = MagicMock()
    nm.get_modulation.return_value = {"learning_rate_gain": 1.3}
    nm._emotion_confidence = 0.78
    nm.modulators = {}  # compute_neuromodulator_homeostasis with empty returns 0.5
    return nm


@pytest.fixture
def warm_coordinator(warm_nm_sys):
    return SimpleNamespace(neuromodulator_system=warm_nm_sys)


@pytest.fixture
def warm_nns():
    """Healthy NeuralNervousSystem with hormonal stats."""
    nns = MagicMock()
    nns.get_stats.return_value = {
        "hormonal_system": {
            "hormones": {
                "cortisol": {"level": 0.4, "setpoint": 0.5},
                "dopamine": {"level": 0.55, "setpoint": 0.5},
            }
        }
    }
    return nns


@pytest.fixture
def warm_latest_epoch_130d():
    """state_vector with all 130 dims populated (passes all coherence gates)."""
    return {"state_vector": [0.5] * 130}


@pytest.fixture
def warm_topology_snap():
    """topology_30d with full 30D layout (passes topology_grounding)."""
    return {"values": [0.5] * 30}


# ── 16 tests, one per input field ─────────────────────────────────────


def test_pi_heartbeat_ratio_from_monitor(cold_inputs, warm_pi_monitor):
    out = compute_life_force_inputs(**{**cold_inputs, "pi_monitor": warm_pi_monitor})
    assert out["pi_heartbeat_ratio"] == pytest.approx(0.20)


def test_pi_heartbeat_ratio_cold_fallback(cold_inputs):
    out = compute_life_force_inputs(**cold_inputs)
    assert out["pi_heartbeat_ratio"] == 0.0


def test_developmental_age_from_monitor(cold_inputs, warm_pi_monitor):
    out = compute_life_force_inputs(**{**cold_inputs, "pi_monitor": warm_pi_monitor})
    assert out["developmental_age"] == 300
    assert isinstance(out["developmental_age"], int)


def test_sovereignty_index_stubbed(cold_inputs):
    out = compute_life_force_inputs(**cold_inputs)
    assert out["sovereignty_index"] == _STUB_SOVEREIGNTY_INDEX == 0


def test_spirit_coherence_from_sv(cold_inputs, warm_latest_epoch_130d):
    out = compute_life_force_inputs(
        **{**cold_inputs, "latest_epoch": warm_latest_epoch_130d}
    )
    # All 0.5 vector → average inner_spirit + outer_spirit coherence = some [0,1]
    assert 0.0 <= out["spirit_coherence"] <= 1.0


def test_spirit_coherence_short_sv_fallback(cold_inputs):
    out = compute_life_force_inputs(
        **{**cold_inputs, "latest_epoch": {"state_vector": [0.5] * 50}}
    )
    # Short vector (< 130) → fallback 0.5
    assert out["spirit_coherence"] == 0.5


def test_vocabulary_size_db_missing_fallback(cold_inputs):
    out = compute_life_force_inputs(**cold_inputs)
    assert out["vocabulary_size"] == 0


def test_learning_rate_gain_from_nm_sys(cold_inputs, warm_coordinator):
    out = compute_life_force_inputs(**{**cold_inputs, "coordinator": warm_coordinator})
    assert out["learning_rate_gain"] == pytest.approx(1.3)


def test_learning_rate_gain_cold_fallback(cold_inputs):
    out = compute_life_force_inputs(**cold_inputs)
    assert out["learning_rate_gain"] == 1.0


def test_emotional_coherence_from_nm_sys(cold_inputs, warm_coordinator):
    out = compute_life_force_inputs(**{**cold_inputs, "coordinator": warm_coordinator})
    assert out["emotional_coherence"] == pytest.approx(0.78)


def test_neuromodulator_homeostasis_cold_fallback(cold_inputs):
    out = compute_life_force_inputs(**cold_inputs)
    # No nm_sys → 0.5 fallback path
    assert out["neuromodulator_homeostasis"] == 0.5


def test_mind_coherence_from_sv(cold_inputs, warm_latest_epoch_130d):
    out = compute_life_force_inputs(
        **{**cold_inputs, "latest_epoch": warm_latest_epoch_130d}
    )
    assert 0.0 <= out["mind_coherence"] <= 1.0


def test_expression_fire_rate_no_reader_fallback(cold_inputs):
    out = compute_life_force_inputs(**cold_inputs)
    # No expression_state_reader → empty stats → compute_expression_fire_rate returns 0.0
    assert out["expression_fire_rate"] == 0.0


def test_expression_fire_rate_from_reader(cold_inputs):
    import msgpack
    reader = MagicMock()
    reader.read_variable.return_value = msgpack.packb(
        {"recent_fire_rate": 0.42}, use_bin_type=True
    )
    out = compute_life_force_inputs(
        **{**cold_inputs, "expression_state_reader": reader}
    )
    assert 0.0 <= out["expression_fire_rate"] <= 1.0


def test_sol_balance_stubbed(cold_inputs):
    out = compute_life_force_inputs(**cold_inputs)
    assert out["sol_balance"] == _STUB_SOL_BALANCE == 13.0


def test_anchor_freshness_stubbed(cold_inputs):
    out = compute_life_force_inputs(**cold_inputs)
    assert out["anchor_freshness"] == _STUB_ANCHOR_FRESHNESS == 0.5


def test_anchor_freshness_real_passthrough(cold_inputs):
    # When cognitive_worker passes a cached freshness (linear-over-24h of
    # timechain_state.recent_anchor_age_s), the builder uses it, not the stub
    # (BUG-LIFEFORCE-INPUT-STUBS).
    out = compute_life_force_inputs(**{**cold_inputs, "anchor_freshness": 0.75})
    assert out["anchor_freshness"] == 0.75


def test_hormonal_vitality_from_nns(cold_inputs, warm_nns):
    out = compute_life_force_inputs(
        **{**cold_inputs, "neural_nervous_system": warm_nns}
    )
    assert 0.0 <= out["hormonal_vitality"] <= 1.0


def test_body_coherence_from_sv(cold_inputs, warm_latest_epoch_130d):
    out = compute_life_force_inputs(
        **{**cold_inputs, "latest_epoch": warm_latest_epoch_130d}
    )
    assert 0.0 <= out["body_coherence"] <= 1.0


def test_topology_grounding_from_snap(cold_inputs, warm_topology_snap):
    out = compute_life_force_inputs(
        **{**cold_inputs, "topology_snap": warm_topology_snap}
    )
    # All-0.5 inner_lower vs balanced 0.5 ref → cosine = 1.0
    assert out["topology_grounding"] == pytest.approx(1.0, abs=1e-3)


def test_topology_grounding_short_snap_fallback(cold_inputs):
    out = compute_life_force_inputs(
        **{**cold_inputs, "topology_snap": {"values": [0.5] * 10}}  # < 20
    )
    assert out["topology_grounding"] == 0.5


def test_infrastructure_health_stubbed(cold_inputs):
    out = compute_life_force_inputs(**cold_inputs)
    assert out["infrastructure_health"] == _STUB_INFRASTRUCTURE_HEALTH == 0.8


# ── Schema / shape tests ──────────────────────────────────────────────


def test_schema_has_all_16_keys(cold_inputs):
    out = compute_life_force_inputs(**cold_inputs)
    expected = {
        # Spirit (4)
        "pi_heartbeat_ratio", "developmental_age", "sovereignty_index",
        "spirit_coherence",
        # Mind (6)
        "vocabulary_size", "learning_rate_gain", "emotional_coherence",
        "neuromodulator_homeostasis", "mind_coherence", "expression_fire_rate",
        # Body (6)
        "sol_balance", "anchor_freshness", "hormonal_vitality",
        "body_coherence", "topology_grounding", "infrastructure_health",
    }
    assert set(out.keys()) == expected
    assert len(out) == 16


def test_msgpack_roundtrip(cold_inputs):
    import msgpack
    out = compute_life_force_inputs(**cold_inputs)
    # Builder output must msgpack-encode cleanly (Publisher contract).
    encoded = msgpack.packb(out, use_bin_type=True)
    decoded = msgpack.unpackb(encoded, raw=False)
    assert decoded == out


def test_pure_function_no_side_effects(cold_inputs, warm_pi_monitor):
    """Same inputs → same outputs (deterministic, no hidden state)."""
    out1 = compute_life_force_inputs(**{**cold_inputs, "pi_monitor": warm_pi_monitor})
    out2 = compute_life_force_inputs(**{**cold_inputs, "pi_monitor": warm_pi_monitor})
    assert out1 == out2


def test_parity_with_pre_extraction_kwargs():
    """The output dict must be a SUPERSET of life_force_engine.evaluate kwargs.

    The pre-extraction call at cognitive_worker.py:2474 passed exactly these
    15 named kwargs to LifeForceEngine.evaluate (note: infrastructure_health
    was NOT passed pre-extraction; engine used its default 0.8 — we now ship
    it explicitly for parity with the SPEC §7.1 schema). Builder must produce
    them all so life_force_worker can pass through unchanged.
    """
    out = compute_life_force_inputs(
        coordinator=None,
        pi_monitor=None,
        neural_nervous_system=None,
        latest_epoch={},
    )
    evaluate_kwargs = {
        "pi_heartbeat_ratio", "developmental_age", "sovereignty_index",
        "spirit_coherence", "vocabulary_size", "learning_rate_gain",
        "emotional_coherence", "neuromodulator_homeostasis", "mind_coherence",
        "expression_fire_rate", "sol_balance", "anchor_freshness",
        "hormonal_vitality", "body_coherence", "topology_grounding",
        "infrastructure_health",
    }
    assert evaluate_kwargs.issubset(set(out.keys()))
