"""
test_life_force_worker — unit tests for the life_force_worker subprocess.

§4.G chunk G6. Validates dispatch loop handlers + SHM publish + bus
emit + drain handlers + SAVE_NOW persistence + FATIGUE edge-debounce.

Pattern mirrors tests/test_metabolism_worker.py (does not exist as such
in the repo today, but follows the same per-handler unit-test approach
used in test_dream_state_worker.py + test_neuromod_worker.py).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import msgpack
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ── _init_life_force_engine ───────────────────────────────────────────


def test_init_engine_default_config():
    from titan_hcl.modules.life_force_worker import _init_life_force_engine
    engine = _init_life_force_engine({})
    assert engine is not None
    assert engine._metabolic_drain == 0.0
    assert engine._drain_passive_decay == pytest.approx(0.9992)


def test_init_engine_config_override():
    from titan_hcl.modules.life_force_worker import _init_life_force_engine
    cfg = {"life_force": {"drain_passive_decay": 0.999}}
    engine = _init_life_force_engine(cfg)
    assert engine is not None
    assert engine._drain_passive_decay == pytest.approx(0.999)


# ── Persistence ───────────────────────────────────────────────────────


def test_save_and_load_roundtrip(tmp_path):
    from titan_hcl.modules import life_force_worker as mod
    from titan_hcl.modules.life_force_worker import (
        _init_life_force_engine, _save_persisted, _try_load_persisted,
    )

    # Swap persist path to tmp
    orig = mod._PERSIST_PATH
    mod._PERSIST_PATH = tmp_path / "life_force_state.json"
    try:
        engine = _init_life_force_engine({})
        engine._metabolic_drain = 0.42
        engine._conviction_counter = 17
        engine._total_evaluations = 99
        _save_persisted(engine)

        assert mod._PERSIST_PATH.exists()
        # New engine, restore
        engine2 = _init_life_force_engine({})
        _try_load_persisted(engine2)
        assert engine2._metabolic_drain == pytest.approx(0.42)
        assert engine2._conviction_counter == 17
        assert engine2._total_evaluations == 99
    finally:
        mod._PERSIST_PATH = orig


def test_try_load_persisted_missing_file_no_crash(tmp_path):
    from titan_hcl.modules import life_force_worker as mod
    from titan_hcl.modules.life_force_worker import _try_load_persisted

    orig = mod._PERSIST_PATH
    mod._PERSIST_PATH = tmp_path / "nonexistent.json"
    try:
        engine = MagicMock()
        engine.restore_state = MagicMock()
        _try_load_persisted(engine)
        # restore_state must NOT have been called (no file → cold boot)
        engine.restore_state.assert_not_called()
    finally:
        mod._PERSIST_PATH = orig


def test_save_persisted_none_engine_no_crash(tmp_path):
    from titan_hcl.modules import life_force_worker as mod
    from titan_hcl.modules.life_force_worker import _save_persisted

    orig = mod._PERSIST_PATH
    mod._PERSIST_PATH = tmp_path / "life_force_state.json"
    try:
        _save_persisted(None)  # must not raise
        assert not mod._PERSIST_PATH.exists()
    finally:
        mod._PERSIST_PATH = orig


# ── _handle_epoch_tick — drives evaluate ──────────────────────────────


def _make_send_queue():
    """Captures emitted messages for assertions."""
    sent = []
    sq = MagicMock()
    sq.put = lambda msg: sent.append(msg)
    return sq, sent


def _make_inputs_reader(payload: dict):
    """Mocked StateRegistryReader-shaped object returning msgpack(payload)."""
    reader = MagicMock()
    reader.read_variable.return_value = msgpack.packb(payload, use_bin_type=True)
    return reader


def _make_cold_state():
    """Mutable closure dicts used by _handle_epoch_tick."""
    return {
        "latest_chi": {"result": None},
        "is_dreaming_box": {"value": False},
        "fatigue_armed": {"value": True},
        "fatigue_emit_count": {"value": 0},
    }


def test_handle_epoch_tick_no_inputs_no_evaluate_no_emit():
    """Cold-boot — no SHM inputs yet → engine.evaluate NOT called."""
    from titan_hcl.modules.life_force_worker import _handle_epoch_tick
    sq, sent = _make_send_queue()
    state = _make_cold_state()
    engine = MagicMock()
    engine.evaluate = MagicMock()
    reader = MagicMock()
    reader.read_variable.return_value = None  # cold

    _handle_epoch_tick(
        engine, reader, state["latest_chi"], state["is_dreaming_box"],
        state["fatigue_armed"], state["fatigue_emit_count"], sq, "life_force",
    )
    engine.evaluate.assert_not_called()
    # No CHI_UPDATED / NUDGE / FATIGUE emit
    assert not any(m.get("type") in {"CHI_UPDATED", "NEUROMOD_EXTERNAL_NUDGE",
                                     "FATIGUE_LEVEL_CRITICAL"} for m in sent)


def test_handle_epoch_tick_with_inputs_calls_evaluate():
    """Inputs SHM has payload → engine.evaluate called with 16 kwargs."""
    from titan_hcl.modules.life_force_worker import _handle_epoch_tick
    sq, sent = _make_send_queue()
    state = _make_cold_state()
    chi_result = {
        "total": 0.65, "spirit": {}, "mind": {}, "body": {},
        "circulation": 0.05, "weights": {},
        "state": "HEALTHY", "developmental_phase": "YOUTH",
        "contemplation": {},
    }
    engine = MagicMock()
    engine.evaluate = MagicMock(return_value=chi_result)
    engine._metabolic_drain = 0.3

    inputs = {
        "pi_heartbeat_ratio": 0.2, "developmental_age": 100,
        "sovereignty_index": 0, "spirit_coherence": 0.5,
        "vocabulary_size": 50, "learning_rate_gain": 1.0,
        "emotional_coherence": 0.5, "neuromodulator_homeostasis": 0.6,
        "mind_coherence": 0.5, "expression_fire_rate": 0.1,
        "sol_balance": 13.0, "anchor_freshness": 0.5,
        "hormonal_vitality": 0.5, "body_coherence": 0.5,
        "topology_grounding": 0.5, "infrastructure_health": 0.8,
        "schema_version": 1, "ts": 1234567890.0,
    }
    reader = _make_inputs_reader(inputs)

    _handle_epoch_tick(
        engine, reader, state["latest_chi"], state["is_dreaming_box"],
        state["fatigue_armed"], state["fatigue_emit_count"], sq, "life_force",
    )
    # evaluate called once with 16 kwargs (schema_version + ts filtered out)
    engine.evaluate.assert_called_once()
    call_kwargs = engine.evaluate.call_args.kwargs
    assert "schema_version" not in call_kwargs
    assert "ts" not in call_kwargs
    assert len(call_kwargs) == 16
    # latest_chi cached
    assert state["latest_chi"]["result"] == chi_result


def test_handle_epoch_tick_emits_chi_updated():
    from titan_hcl.modules.life_force_worker import _handle_epoch_tick
    sq, sent = _make_send_queue()
    state = _make_cold_state()
    chi_result = {"total": 0.5, "state": "HEALTHY", "contemplation": {}}
    engine = MagicMock()
    engine.evaluate = MagicMock(return_value=chi_result)
    engine._metabolic_drain = 0.0
    reader = _make_inputs_reader({"pi_heartbeat_ratio": 0.0})
    _handle_epoch_tick(
        engine, reader, state["latest_chi"], state["is_dreaming_box"],
        state["fatigue_armed"], state["fatigue_emit_count"], sq, "life_force",
    )
    chi_msgs = [m for m in sent if m.get("type") == "CHI_UPDATED"]
    assert len(chi_msgs) == 1
    assert chi_msgs[0]["payload"] == chi_result
    assert chi_msgs[0]["dst"] == "all"


def test_handle_epoch_tick_emits_chi_health_nudge():
    """Per-evaluate NEUROMOD_EXTERNAL_NUDGE closes §4.Q D-SPEC-54 orphan."""
    from titan_hcl.modules.life_force_worker import _handle_epoch_tick
    sq, sent = _make_send_queue()
    state = _make_cold_state()
    engine = MagicMock()
    engine.evaluate = MagicMock(return_value={"total": 0.5, "contemplation": {}})
    engine._metabolic_drain = 0.5

    reader = _make_inputs_reader({"x": 1})
    _handle_epoch_tick(
        engine, reader, state["latest_chi"], state["is_dreaming_box"],
        state["fatigue_armed"], state["fatigue_emit_count"], sq, "life_force",
    )
    nudges = [m for m in sent if m.get("type") == "NEUROMOD_EXTERNAL_NUDGE"]
    assert len(nudges) == 1
    p = nudges[0]["payload"]
    assert p["source"] == "life_force_chi_health"
    # chi_health = max(0.1, 1.0 - 0.5*0.6) = 0.7
    assert p["chi_health"] == pytest.approx(0.7)


def test_handle_epoch_tick_chi_health_floor_at_min():
    """chi_health floor at 0.1 (max-out drain)."""
    from titan_hcl.modules.life_force_worker import _handle_epoch_tick
    sq, sent = _make_send_queue()
    state = _make_cold_state()
    engine = MagicMock()
    engine.evaluate = MagicMock(return_value={"total": 0.1, "contemplation": {}})
    engine._metabolic_drain = 1.5  # > 0.8 cap in real engine, but stub for test

    reader = _make_inputs_reader({"x": 1})
    _handle_epoch_tick(
        engine, reader, state["latest_chi"], state["is_dreaming_box"],
        state["fatigue_armed"], state["fatigue_emit_count"], sq, "life_force",
    )
    nudges = [m for m in sent if m.get("type") == "NEUROMOD_EXTERNAL_NUDGE"]
    assert nudges[0]["payload"]["chi_health"] == pytest.approx(0.1)


# ── FATIGUE_LEVEL_CRITICAL edge-debounce ──────────────────────────────


def test_fatigue_emits_on_upward_crossing():
    """Drain crosses 0.7 → P1 single-shot fires once."""
    from titan_hcl.modules.life_force_worker import _handle_epoch_tick
    sq, sent = _make_send_queue()
    state = _make_cold_state()
    engine = MagicMock()
    engine.evaluate = MagicMock(return_value={"total": 0.3, "contemplation": {}, "state": "SURVIVAL"})
    engine._state = "SURVIVAL"
    engine._metabolic_drain = 0.72  # > threshold

    reader = _make_inputs_reader({"x": 1})
    _handle_epoch_tick(
        engine, reader, state["latest_chi"], state["is_dreaming_box"],
        state["fatigue_armed"], state["fatigue_emit_count"], sq, "life_force",
    )
    fatigues = [m for m in sent if m.get("type") == "FATIGUE_LEVEL_CRITICAL"]
    assert len(fatigues) == 1
    assert fatigues[0]["payload"]["drain"] == pytest.approx(0.72)
    assert state["fatigue_armed"]["value"] is False  # debounced
    assert state["fatigue_emit_count"]["value"] == 1


def test_fatigue_does_not_emit_when_disarmed():
    """While disarmed, even drain>0.7 must not re-emit."""
    from titan_hcl.modules.life_force_worker import _handle_epoch_tick
    sq, sent = _make_send_queue()
    state = _make_cold_state()
    state["fatigue_armed"]["value"] = False  # already fired once

    engine = MagicMock()
    engine.evaluate = MagicMock(return_value={"total": 0.3, "contemplation": {}})
    engine._metabolic_drain = 0.75  # > threshold but disarmed

    reader = _make_inputs_reader({"x": 1})
    _handle_epoch_tick(
        engine, reader, state["latest_chi"], state["is_dreaming_box"],
        state["fatigue_armed"], state["fatigue_emit_count"], sq, "life_force",
    )
    fatigues = [m for m in sent if m.get("type") == "FATIGUE_LEVEL_CRITICAL"]
    assert len(fatigues) == 0


def test_fatigue_re_arms_below_reset_threshold():
    """When drain falls ≤0.6, re-arm for next upward crossing."""
    from titan_hcl.modules.life_force_worker import _handle_epoch_tick
    sq, sent = _make_send_queue()
    state = _make_cold_state()
    state["fatigue_armed"]["value"] = False  # already fired

    engine = MagicMock()
    engine.evaluate = MagicMock(return_value={"total": 0.5, "contemplation": {}})
    engine._metabolic_drain = 0.55  # < reset threshold

    reader = _make_inputs_reader({"x": 1})
    _handle_epoch_tick(
        engine, reader, state["latest_chi"], state["is_dreaming_box"],
        state["fatigue_armed"], state["fatigue_emit_count"], sq, "life_force",
    )
    assert state["fatigue_armed"]["value"] is True  # re-armed


def test_fatigue_no_oscillation_in_band():
    """Drain in (0.6, 0.7) band → no toggle in either direction."""
    from titan_hcl.modules.life_force_worker import _handle_epoch_tick
    sq, sent = _make_send_queue()
    state = _make_cold_state()

    engine = MagicMock()
    engine.evaluate = MagicMock(return_value={"total": 0.5, "contemplation": {}})
    engine._metabolic_drain = 0.65  # in hysteresis band

    reader = _make_inputs_reader({"x": 1})
    _handle_epoch_tick(
        engine, reader, state["latest_chi"], state["is_dreaming_box"],
        state["fatigue_armed"], state["fatigue_emit_count"], sq, "life_force",
    )
    fatigues = [m for m in sent if m.get("type") == "FATIGUE_LEVEL_CRITICAL"]
    assert len(fatigues) == 0
    assert state["fatigue_armed"]["value"] is True  # stays armed


# ── _handle_query — action dispatch ────────────────────────────────────


def test_handle_query_get_stats():
    from titan_hcl.modules.life_force_worker import _handle_query
    sq, sent = _make_send_queue()
    engine = MagicMock()
    engine.get_stats.return_value = {"total_evaluations": 42, "current_state": "HEALTHY"}

    msg = {"type": "QUERY", "rid": "req-1", "src": "life_force_proxy",
           "payload": {"action": "get_stats"}}
    _handle_query(msg, engine, {"result": None}, sq, "life_force")

    rsps = [m for m in sent if m.get("type") == "RESPONSE"]
    assert len(rsps) == 1
    assert rsps[0]["rid"] == "req-1"
    assert rsps[0]["payload"]["result"]["total_evaluations"] == 42


def test_handle_query_get_chi_history():
    from titan_hcl.modules.life_force_worker import _handle_query
    sq, sent = _make_send_queue()
    engine = MagicMock()
    engine._chi_history = [{"ts": 1.0, "total": 0.5}, {"ts": 2.0, "total": 0.6}]

    msg = {"type": "QUERY", "rid": "req-2", "src": "life_force_proxy",
           "payload": {"action": "get_chi_history", "limit": 10}}
    _handle_query(msg, engine, {"result": None}, sq, "life_force")

    rsps = [m for m in sent if m.get("type") == "RESPONSE"]
    assert len(rsps) == 1
    assert rsps[0]["payload"]["result"] == engine._chi_history


def test_handle_query_get_contemplation_status_from_latest():
    from titan_hcl.modules.life_force_worker import _handle_query
    sq, sent = _make_send_queue()
    engine = MagicMock()
    latest = {"result": {"contemplation": {"active": True, "phase": 2,
                                            "conviction": 50}}}
    msg = {"type": "QUERY", "rid": "req-3", "src": "life_force_proxy",
           "payload": {"action": "get_contemplation_status"}}
    _handle_query(msg, engine, latest, sq, "life_force")
    rsps = [m for m in sent if m.get("type") == "RESPONSE"]
    assert rsps[0]["payload"]["result"]["active"] is True
    assert rsps[0]["payload"]["result"]["phase"] == 2


def test_handle_query_get_contemplation_status_fallback():
    from titan_hcl.modules.life_force_worker import _handle_query
    sq, sent = _make_send_queue()
    engine = MagicMock()
    engine._contemplation_phase = 0
    engine._conviction_counter = 0
    msg = {"type": "QUERY", "rid": "req-4", "src": "life_force_proxy",
           "payload": {"action": "get_contemplation_status"}}
    _handle_query(msg, engine, {"result": None}, sq, "life_force")
    rsps = [m for m in sent if m.get("type") == "RESPONSE"]
    assert rsps[0]["payload"]["result"]["active"] is False
    assert rsps[0]["payload"]["result"]["phase"] == 0


def test_handle_query_unknown_action_returns_error():
    from titan_hcl.modules.life_force_worker import _handle_query
    sq, sent = _make_send_queue()
    engine = MagicMock()
    msg = {"type": "QUERY", "rid": "req-5", "src": "x",
           "payload": {"action": "no_such_action"}}
    _handle_query(msg, engine, {"result": None}, sq, "life_force")
    rsps = [m for m in sent if m.get("type") == "RESPONSE"]
    assert "error" in rsps[0]["payload"]


def test_handle_query_no_rid_no_response():
    """rid is None → response not emitted (request-only call)."""
    from titan_hcl.modules.life_force_worker import _handle_query
    sq, sent = _make_send_queue()
    engine = MagicMock()
    engine.get_stats.return_value = {}
    msg = {"type": "QUERY", "rid": None, "src": "x",
           "payload": {"action": "get_stats"}}
    _handle_query(msg, engine, {"result": None}, sq, "life_force")
    rsps = [m for m in sent if m.get("type") == "RESPONSE"]
    assert len(rsps) == 0


# ── _read_inputs_from_shm ─────────────────────────────────────────────


def test_read_inputs_from_shm_none_reader():
    from titan_hcl.modules.life_force_worker import _read_inputs_from_shm
    assert _read_inputs_from_shm(None) is None


def test_read_inputs_from_shm_cold_boot():
    from titan_hcl.modules.life_force_worker import _read_inputs_from_shm
    reader = MagicMock()
    reader.read_variable.return_value = None
    assert _read_inputs_from_shm(reader) is None


def test_read_inputs_from_shm_decodes_msgpack():
    from titan_hcl.modules.life_force_worker import _read_inputs_from_shm
    reader = MagicMock()
    payload = {"x": 1, "y": 2.0}
    reader.read_variable.return_value = msgpack.packb(payload, use_bin_type=True)
    assert _read_inputs_from_shm(reader) == payload


def test_read_inputs_from_shm_corrupted_returns_none():
    from titan_hcl.modules.life_force_worker import _read_inputs_from_shm
    reader = MagicMock()
    reader.read_variable.return_value = b"\xff\xff garbage"
    assert _read_inputs_from_shm(reader) is None


# ── SPEC-level structural assertions ──────────────────────────────────


def test_subscribe_topics_match_spec():
    """SPEC §9.B life_force_worker subscribes: KERNEL_EPOCH_TICK,
    DREAM_STATE_CHANGED, MEDITATION_COMPLETE, EXPRESSION_FIRED,
    NEUROMOD_STATS_UPDATED, MODULE_SHUTDOWN, SAVE_NOW + bus.QUERY."""
    from titan_hcl.modules.life_force_worker import (
        _LIFE_FORCE_WORKER_SUBSCRIBE_TOPICS,
    )
    expected = {
        "QUERY", "KERNEL_EPOCH_TICK", "DREAM_STATE_CHANGED",
        "MEDITATION_COMPLETE", "EXPRESSION_FIRED",
        "NEUROMOD_STATS_UPDATED", "MODULE_SHUTDOWN", "SAVE_NOW",
    }
    assert set(_LIFE_FORCE_WORKER_SUBSCRIBE_TOPICS) == expected


def test_module_name_constant():
    from titan_hcl.modules.life_force_worker import MODULE_NAME
    assert MODULE_NAME == "life_force"
