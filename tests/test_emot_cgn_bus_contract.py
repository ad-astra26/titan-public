"""Contract tests for EMOT-CGN bus messages (Phase 1.6d, rFP §10 ADR).

Per `memory/feedback_architectural_decisions_no_drift.md` Rule 3:
"Bus contract FIRST, implementation SECOND." These tests pin the shape +
routing + structural invariants of the 3 EMOT-CGN bus message types
BEFORE any producer/consumer code uses them. Protects against future
silent refactors + sets the contract for Phase 1.6e worker subscribers.
"""
import pytest
from titan_plugin.bus import (
    EMOT_CHAIN_EVIDENCE,
    EMOT_CGN_SIGNAL,
    FELT_CLUSTER_UPDATE,
    emit_emot_chain_evidence,
    emit_felt_cluster_update,
    emit_emot_cgn_signal,
)


class _Q:
    """Minimal send-queue mock that records put() + put_nowait() calls."""
    def __init__(self):
        self.sent = []
    def put(self, m): self.sent.append(m)
    def put_nowait(self, m): self.sent.append(m)


# ── Constants ─────────────────────────────────────────────────────

def test_emot_chain_evidence_constant_matches_name():
    """Bus type string matches constant name (convention for arch_map)."""
    assert EMOT_CHAIN_EVIDENCE == "EMOT_CHAIN_EVIDENCE"


def test_felt_cluster_update_constant_matches_name():
    assert FELT_CLUSTER_UPDATE == "FELT_CLUSTER_UPDATE"


def test_emot_cgn_signal_constant_matches_name():
    assert EMOT_CGN_SIGNAL == "EMOT_CGN_SIGNAL"


# ── EMOT_CHAIN_EVIDENCE ────────────────────────────────────────────

def test_emit_chain_evidence_happy_path():
    q = _Q()
    ok = emit_emot_chain_evidence(
        q, src="spirit", chain_id=42,
        dominant_at_start="FLOW", dominant_at_end="WONDER",
        terminal_reward=0.82,
        ctx={"DA": 0.7, "5HT": 0.6})
    assert ok is True
    assert len(q.sent) == 1
    msg = q.sent[0]
    assert msg["type"] == EMOT_CHAIN_EVIDENCE
    assert msg["src"] == "spirit"
    assert msg["dst"] == "emot_cgn"
    assert "ts" in msg
    p = msg["payload"]
    assert p["chain_id"] == 42
    assert p["dominant_at_start"] == "FLOW"
    assert p["dominant_at_end"] == "WONDER"
    assert p["terminal_reward"] == 0.82
    assert p["ctx"]["DA"] == 0.7


def test_emit_chain_evidence_ctx_none_safe():
    """Missing ctx must not crash — defaults to empty dict."""
    q = _Q()
    ok = emit_emot_chain_evidence(
        q, src="spirit", chain_id=1,
        dominant_at_start="GRIEF", dominant_at_end="GRIEF",
        terminal_reward=0.2, ctx=None)
    assert ok is True
    assert q.sent[0]["payload"]["ctx"] == {}


def test_emit_chain_evidence_invalid_sender_returns_false():
    """Sender without put_nowait/publish → False, no crash."""
    class NotASender:
        pass
    ok = emit_emot_chain_evidence(
        NotASender(), src="spirit", chain_id=1,
        dominant_at_start="FLOW", dominant_at_end="FLOW",
        terminal_reward=0.5)
    assert ok is False


def test_emit_chain_evidence_coerces_types():
    """Numeric types coerced; string fields coerced; works with mixed."""
    q = _Q()
    ok = emit_emot_chain_evidence(
        q, src="spirit", chain_id="99",  # string chain_id
        dominant_at_start="FLOW", dominant_at_end="WONDER",
        terminal_reward="0.5")  # string reward
    assert ok is True
    assert q.sent[0]["payload"]["chain_id"] == 99
    assert q.sent[0]["payload"]["terminal_reward"] == 0.5


# ── FELT_CLUSTER_UPDATE ────────────────────────────────────────────

def test_emit_felt_cluster_update_with_150d_feature_vec():
    q = _Q()
    ok = emit_felt_cluster_update(
        q, src="spirit",
        feature_vec_150d=[0.5] * 150)
    assert ok is True
    msg = q.sent[0]
    assert msg["type"] == FELT_CLUSTER_UPDATE
    assert msg["dst"] == "emot_cgn"
    assert "feature_vec_150d" in msg["payload"]
    assert len(msg["payload"]["feature_vec_150d"]) == 150


def test_emit_felt_cluster_update_with_130d_felt_tensor():
    """Fallback path: worker reconstructs 150D from 130D + local context."""
    q = _Q()
    ok = emit_felt_cluster_update(
        q, src="spirit",
        felt_tensor_130d=[0.5] * 130)
    assert ok is True
    assert "felt_tensor_130d" in q.sent[0]["payload"]
    assert "feature_vec_150d" not in q.sent[0]["payload"]


def test_emit_felt_cluster_update_requires_at_least_one_vec():
    """Neither vec provided → emission refused (no silent no-op)."""
    q = _Q()
    ok = emit_felt_cluster_update(q, src="spirit")
    assert ok is False
    assert len(q.sent) == 0


def test_emit_felt_cluster_update_invalid_sender_returns_false():
    class NotASender:
        pass
    ok = emit_felt_cluster_update(
        NotASender(), src="spirit",
        feature_vec_150d=[0.5] * 150)
    assert ok is False


# ── Destination routing (critical for Phase 1.6e worker subscription) ──

def test_all_emot_worker_inputs_route_to_emot_cgn():
    """Phase 1.6d ADR: producer→worker events MUST route dst='emot_cgn'
    so the new standalone worker can subscribe. This test is a load-bearing
    contract for Phase 1.6e subscriber wiring."""
    q = _Q()
    emit_emot_chain_evidence(
        q, src="spirit", chain_id=1,
        dominant_at_start="FLOW", dominant_at_end="FLOW",
        terminal_reward=0.5)
    emit_felt_cluster_update(q, src="spirit", feature_vec_150d=[0.5] * 150)
    assert all(m["dst"] == "emot_cgn" for m in q.sent)


def test_emot_cgn_signal_still_routes_to_spirit_pre_migration():
    """During Phase 1.6a-d transition, EMOT_CGN_SIGNAL continues to route
    dst='spirit' (consumed by meta_reasoning._emot_cgn). Phase 1.6e will
    flip the routing to dst='emot_cgn' alongside the consumer migration.
    This test PINS the pre-migration behavior to catch premature flips."""
    q = _Q()
    ok = emit_emot_cgn_signal(
        q, src="spirit", consumer="emot_cgn",
        event_type="cluster_assignment", intensity=0.5)
    assert ok is True  # orphan check passes (cluster_assignment mapped)
    assert q.sent[0]["dst"] == "spirit"   # NOT yet emot_cgn
