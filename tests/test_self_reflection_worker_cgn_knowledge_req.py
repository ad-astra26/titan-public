"""Chunk B.4 (RFP_meta-reasoning_CGN_FIX.md §4.2 rows 8/9) — verify
self_reflection_worker hosts the Session 3 CGN_KNOWLEDGE_REQ responder
for kind ∈ {prediction, self_reasoning} per Track 2 v1.2.1 commit B8
canonical relocation.
"""
from __future__ import annotations

import pytest

from titan_hcl import bus
from titan_hcl.modules.self_reflection_worker import (
    _SELF_REFLECTION_WORKER_SUBSCRIBE_TOPICS,
    _build_prediction_response,
    _build_self_reasoning_response,
)


# ──────────────────────────────────────────────────────────────────────
# Subscribe-topic registration
# ──────────────────────────────────────────────────────────────────────


def test_cgn_knowledge_req_in_self_reflection_subscribe_topics():
    """self_reflection_worker must subscribe to CGN_KNOWLEDGE_REQ for
    Session 3 prediction + self_reasoning kind dispatch."""
    assert bus.CGN_KNOWLEDGE_REQ in _SELF_REFLECTION_WORKER_SUBSCRIBE_TOPICS


# ──────────────────────────────────────────────────────────────────────
# _build_prediction_response
# ──────────────────────────────────────────────────────────────────────


class _FakePredictionEngine:
    def __init__(self, novelty=0.42, predictions=100, surprises=15,
                 avg_error=0.07):
        self._novelty = novelty
        self._stats = {
            "total_predictions": predictions,
            "total_surprises": surprises,
            "avg_error": avg_error,
        }

    def get_stats(self):
        return self._stats

    def get_novelty_signal(self):
        return self._novelty


def test_prediction_response_with_engine_returns_real_stats():
    rsp = _build_prediction_response(
        _FakePredictionEngine(), "default",
        {"question_type": "evaluate_trajectory", "consumer_id": "emotional"})
    assert rsp["engine"] == "prediction"
    assert rsp["name"] == "default"
    assert rsp["novelty_signal"] == pytest.approx(0.42)
    assert rsp["total_predictions"] == 100
    assert rsp["total_surprises"] == 15
    assert rsp["avg_error"] == pytest.approx(0.07)
    assert rsp["question_type"] == "evaluate_trajectory"
    assert rsp["consumer_id"] == "emotional"


def test_prediction_response_with_none_engine_graceful():
    rsp = _build_prediction_response(None, "default", {})
    assert rsp["engine"] == "unavailable"


def test_prediction_response_without_novelty_method_falls_back_to_zero():
    class _NoNoveltyMethod:
        def get_stats(self):
            return {"total_predictions": 5}
    rsp = _build_prediction_response(_NoNoveltyMethod(), "default", {})
    assert rsp["novelty_signal"] == 0.0
    assert rsp["total_predictions"] == 5


def test_prediction_response_with_broken_engine_returns_zeros():
    class _Broken:
        def get_stats(self):
            raise RuntimeError("boom")
        def get_novelty_signal(self):
            raise RuntimeError("boom")
    rsp = _build_prediction_response(_Broken(), "default", {})
    assert rsp["engine"] == "prediction"
    assert rsp["total_predictions"] == 0
    assert rsp["novelty_signal"] == 0.0


# ──────────────────────────────────────────────────────────────────────
# _build_self_reasoning_response
# ──────────────────────────────────────────────────────────────────────


class _FakeSelfReasoning:
    def get_stats(self):
        return {
            "introspection_depth": 0.65,
            "observations_total": 250,
            "audit_count": 12,
            "last_audit_ts": 1700000000.0,
        }


def test_self_reasoning_response_with_engine_returns_real_stats():
    rsp = _build_self_reasoning_response(
        _FakeSelfReasoning(), "meta_audit",
        {"question_type": "introspect_state", "consumer_id": "reflection"})
    assert rsp["engine"] == "self_reasoning"
    assert rsp["name"] == "meta_audit"
    assert rsp["introspection_depth"] == pytest.approx(0.65)
    assert rsp["observations_total"] == 250
    assert rsp["audit_count"] == 12
    assert rsp["last_audit_ts"] == pytest.approx(1700000000.0)
    assert rsp["question_type"] == "introspect_state"


def test_self_reasoning_response_with_none_engine_graceful():
    rsp = _build_self_reasoning_response(None, "predict", {})
    assert rsp["engine"] == "unavailable"
    assert rsp["name"] == "predict"


def test_self_reasoning_response_without_get_stats_returns_defaults():
    class _Bare:
        pass
    rsp = _build_self_reasoning_response(_Bare(), "meta_audit", {})
    assert rsp["engine"] == "self_reasoning"
    assert rsp["introspection_depth"] == 0.0
    assert rsp["observations_total"] == 0
