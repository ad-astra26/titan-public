"""
Tests for self_reflection_worker META_INTROSPECT_REQUEST handler — Chunk B
of rFP_meta_reasoning_self_reasoning_resolver_migration.

SPEC §9.B + D-SPEC-70 v1.15.0. Closes F-8 fleet-wide.

Coverage:
  - Handler runs sr.introspect(**payload) with all expected kwargs
  - Handler writes result to inner_self_insight.bin SHM slot
  - select_introspection_mode override propagates to effective_sub_mode
  - sr=None branch returns silently (engine init failed at boot)
  - Non-dict result from introspect logged + SHM not updated
  - Optional fields preserved in SHM payload
  - chi_coh extracted from echoed msl_data
  - Existing predict + meta_audit dispatch paths unchanged (regression guard)

Reference:
  - titan_hcl/modules/self_reflection_worker.py _handle_meta_introspect_request
  - titan_hcl/api/shm_reader_bank.py read_inner_self_insight
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import msgpack
import pytest

from titan_hcl import bus
from titan_hcl.api.shm_reader_bank import ShmReaderBank
from titan_hcl.modules.self_reflection_worker import (
    _dispatch_msg,
    _handle_meta_introspect_request,
)


@pytest.fixture
def shm_root(tmp_path, monkeypatch):
    monkeypatch.setenv("TITAN_SHM_ROOT", str(tmp_path))
    monkeypatch.setenv("TITAN_ID", "T_TEST_INTRO_HANDLER")
    return tmp_path


@pytest.fixture
def fake_sr():
    """SelfReasoningEngine stub returning a deterministic result."""
    sr = MagicMock()
    sr.select_introspection_mode = MagicMock(return_value="coherence_check")
    sr.introspect = MagicMock(return_value={
        "primitive": "INTROSPECT",
        "sub_mode": "coherence_check",
        "confidence": 0.7,
        "mode_trigger": "DA_phasic_above_0.4",
        "inner_avg": 0.4,
        "outer_avg": 0.3,
        "note": "test introspect output",
    })
    return sr


def _make_trigger_payload(sub_mode: str = "state_audit",
                            chi_coh: float | None = 0.61) -> dict:
    """Trigger payload echoed by cognitive_worker._prim_introspect."""
    msl_data = {}
    if chi_coh is not None:
        msl_data["chi_coherence"] = chi_coh
    return {
        "sub_mode": sub_mode,
        "epoch": 12345,
        "neuromods": {
            "DA": 0.51, "5HT": 0.32, "NE": 0.19,
            "ACh": 0.44, "Endorphin": 0.08, "GABA": 0.27,
        },
        "msl_data": msl_data,
        "reasoning_stats": {"chains": 42},
        "language_stats": {"vocab": 100},
        "coordinator_data": {"foo": "bar"},
        "state_132d": [0.1] * 132,
        "ts": time.time(),
    }


def test_handler_calls_introspect_with_expected_kwargs(shm_root, fake_sr):
    """sr.introspect receives all 8 kwargs threaded from the trigger payload."""
    state_refs = {"self_reasoning": fake_sr}
    payload = _make_trigger_payload(sub_mode="state_audit")

    _handle_meta_introspect_request(payload, state_refs, "self_reflection_worker")

    fake_sr.introspect.assert_called_once()
    kwargs = fake_sr.introspect.call_args.kwargs
    assert kwargs["epoch"] == 12345
    assert kwargs["neuromods"]["DA"] == pytest.approx(0.51)
    assert kwargs["msl_data"] == {"chi_coherence": 0.61}
    assert kwargs["reasoning_stats"] == {"chains": 42}
    assert kwargs["language_stats"] == {"vocab": 100}
    assert kwargs["coordinator_data"] == {"foo": "bar"}
    assert kwargs["state_132d"] == [0.1] * 132
    # select_introspection_mode override: state_audit → coherence_check
    assert kwargs["sub_mode"] == "coherence_check"


def test_handler_writes_shm_slot(shm_root, fake_sr):
    """Handler write is visible via ShmReaderBank.read_inner_self_insight."""
    state_refs = {"self_reasoning": fake_sr}
    payload = _make_trigger_payload(sub_mode="state_audit", chi_coh=0.61)

    _handle_meta_introspect_request(payload, state_refs, "self_reflection_worker")

    bank = ShmReaderBank()
    out = bank.read_inner_self_insight()
    assert out is not None
    assert out["primitive"] == "INTROSPECT"
    assert out["sub_mode"] == "state_audit"
    assert out["effective_sub_mode"] == "coherence_check"
    assert out["confidence"] == pytest.approx(0.7)
    assert out["mode_trigger"] == "DA_phasic_above_0.4"
    assert out["epoch"] == 12345
    assert out["chi_coh"] == pytest.approx(0.61)
    assert out["cold_start"] is False
    assert out["neuromods"]["DA"] == pytest.approx(0.51)
    assert out["note"] == "test introspect output"


def test_handler_state_audit_no_override_when_select_returns_same(shm_root):
    """If select_introspection_mode returns 'state_audit', effective_sub_mode
    stays as state_audit (no upgrade)."""
    sr = MagicMock()
    sr.select_introspection_mode = MagicMock(return_value="state_audit")
    sr.introspect = MagicMock(return_value={
        "primitive": "INTROSPECT", "confidence": 0.5,
    })
    state_refs = {"self_reasoning": sr}
    payload = _make_trigger_payload(sub_mode="state_audit")

    _handle_meta_introspect_request(payload, state_refs, "self_reflection_worker")

    kwargs = sr.introspect.call_args.kwargs
    assert kwargs["sub_mode"] == "state_audit"

    bank = ShmReaderBank()
    out = bank.read_inner_self_insight()
    assert out["effective_sub_mode"] == "state_audit"


def test_handler_non_state_audit_skips_override(shm_root, fake_sr):
    """Non-state_audit sub_mode passes through to sr.introspect without
    consulting select_introspection_mode."""
    state_refs = {"self_reasoning": fake_sr}
    payload = _make_trigger_payload(sub_mode="maker_alignment")

    _handle_meta_introspect_request(payload, state_refs, "self_reflection_worker")

    # select_introspection_mode should NOT be called (only fires for state_audit)
    fake_sr.select_introspection_mode.assert_not_called()
    kwargs = fake_sr.introspect.call_args.kwargs
    assert kwargs["sub_mode"] == "maker_alignment"


def test_handler_silent_when_sr_is_none(shm_root):
    """If self_reasoning is None (engine init failed at boot), handler
    returns silently — no SHM write, no crash."""
    state_refs = {"self_reasoning": None}
    payload = _make_trigger_payload()
    # Should not raise
    _handle_meta_introspect_request(payload, state_refs, "self_reflection_worker")

    bank = ShmReaderBank()
    # SHM untouched — cold-start state (None or empty)
    out = bank.read_inner_self_insight()
    assert out is None or out == {}


def test_handler_non_dict_result_logged_no_shm_write(shm_root):
    """If sr.introspect returns a non-dict, log a warning + skip SHM write."""
    sr = MagicMock()
    sr.select_introspection_mode = MagicMock(return_value="state_audit")
    sr.introspect = MagicMock(return_value="not a dict")
    state_refs = {"self_reasoning": sr}
    payload = _make_trigger_payload()

    _handle_meta_introspect_request(payload, state_refs, "self_reflection_worker")

    bank = ShmReaderBank()
    out = bank.read_inner_self_insight()
    assert out is None or out == {}


def test_handler_introspect_exception_logged_no_shm_write(shm_root):
    """If sr.introspect raises, log + skip SHM write — no propagation."""
    sr = MagicMock()
    sr.select_introspection_mode = MagicMock(return_value="state_audit")
    sr.introspect = MagicMock(side_effect=RuntimeError("simulated"))
    state_refs = {"self_reasoning": sr}
    payload = _make_trigger_payload()

    # Should not raise
    _handle_meta_introspect_request(payload, state_refs, "self_reflection_worker")

    bank = ShmReaderBank()
    out = bank.read_inner_self_insight()
    assert out is None or out == {}


def test_handler_chi_coh_null_when_msl_missing(shm_root, fake_sr):
    """If msl_data has no chi_coherence, SHM chi_coh field is None."""
    state_refs = {"self_reasoning": fake_sr}
    payload = _make_trigger_payload(chi_coh=None)

    _handle_meta_introspect_request(payload, state_refs, "self_reflection_worker")

    bank = ShmReaderBank()
    out = bank.read_inner_self_insight()
    assert out["chi_coh"] is None


def test_dispatcher_routes_meta_introspect_to_handler(shm_root, fake_sr):
    """The _dispatch_msg entry point routes META_INTROSPECT_REQUEST to the
    new handler (closes the wiring contract — broker delivery flows here)."""
    state_refs = {"self_reasoning": fake_sr}
    payload = _make_trigger_payload()
    msg = {"type": bus.META_INTROSPECT_REQUEST, "payload": payload}
    send_queue = MagicMock()

    _dispatch_msg(msg, bus.META_INTROSPECT_REQUEST, state_refs,
                  send_queue, "self_reflection_worker")

    fake_sr.introspect.assert_called_once()
    # Fire-and-forget — no response published
    send_queue.put.assert_not_called()
    send_queue.put_nowait.assert_not_called()


def test_existing_cgn_knowledge_req_predict_path_unchanged(shm_root):
    """Regression guard: the existing predict resolver path through
    CGN_KNOWLEDGE_REQ still routes correctly (separate codepath)."""
    pe = MagicMock()
    pe.get_stats = MagicMock(return_value={
        "predictions_made": 10, "errors_computed": 5,
        "avg_error": 0.3, "novelty_ema": 0.5,
    })
    state_refs = {
        "self_reasoning": None,
        "coding_explorer": None,
        "prediction_engine": pe,
    }
    msg = {
        "type": bus.CGN_KNOWLEDGE_REQ,
        "src": "meta_service",
        "payload": {
            "correlation_id": "test-123",
            "kind": "prediction",
            "name": "recall",
            "consumer_id": "test",
        },
    }
    send_queue = MagicMock()

    _dispatch_msg(msg, bus.CGN_KNOWLEDGE_REQ, state_refs,
                  send_queue, "self_reflection_worker")

    # send_queue.put called with CGN_KNOWLEDGE_RESP — existing wiring intact
    send_queue.put.assert_called_once()
    sent_msg = send_queue.put.call_args.args[0]
    assert sent_msg["type"] == bus.CGN_KNOWLEDGE_RESP
    assert sent_msg["payload"]["correlation_id"] == "test-123"
    assert sent_msg["payload"]["kind"] == "prediction"
