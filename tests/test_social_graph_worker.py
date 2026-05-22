"""tests/test_social_graph_worker.py — social_graph_worker dispatch tests.

Per PLAN_microkernel_phase_c_social_graph_worker_extraction.md §7.1 +
SPEC v1.7.1 §9.B social_graph_worker + D-SPEC-50.

Focuses on the action-dispatch contract: every action the proxy can
emit reaches the right SocialGraph method with the right arguments and
produces the right RESPONSE shape. Subprocess + heartbeat lifecycle
covered separately by the lifecycle test file (PLAN §7.4) which needs
real fork machinery.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from queue import Queue
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from titan_hcl import bus  # noqa: E402
from titan_hcl.core.social_graph import SocialGraph  # noqa: E402
from titan_hcl.modules.social_graph_worker import (  # noqa: E402
    MODULE_NAME,
    _handle_query,
    _serialize_profile,
    _SOCIAL_GRAPH_WORKER_SUBSCRIBE_TOPICS,
)


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def temp_db_path():
    """Real SocialGraph backed by a temp SQLite file."""
    with tempfile.TemporaryDirectory() as td:
        yield str(Path(td) / "social_graph.db")


@pytest.fixture
def social_graph(temp_db_path):
    return SocialGraph(db_path=temp_db_path)


@pytest.fixture
def send_queue():
    return Queue()


def _build_query(action: str, **payload):
    """Build a bus.QUERY-shaped msg dict."""
    return {
        "type": bus.QUERY,
        "src": "social_graph_proxy",
        "dst": "social_graph",
        "rid": f"rid-{action}",
        "payload": {"action": action, **payload},
        "ts": 1234567890.0,
    }


def _drain_responses(q: Queue) -> list[dict]:
    """Drain queue without blocking."""
    out: list[dict] = []
    while True:
        try:
            out.append(q.get_nowait())
        except Exception:
            break
    return out


def _find_response(msgs: list[dict], rid: str) -> dict | None:
    for m in msgs:
        if m.get("type") == bus.RESPONSE and m.get("rid") == rid:
            return m
    return None


# ── Subscribe topics contract ─────────────────────────────────────────


def test_subscribe_topics_minimal():
    """Worker subscribes only to QUERY (dispatch) + MODULE_SHUTDOWN +
    SAVE_NOW lifecycle. Verifies PLAN §3.1 contract."""
    assert _SOCIAL_GRAPH_WORKER_SUBSCRIBE_TOPICS == [
        bus.QUERY, bus.MODULE_SHUTDOWN, bus.SAVE_NOW,
    ]


def test_module_name_constant():
    assert MODULE_NAME == "social_graph"


# ── Write action dispatch ─────────────────────────────────────────────


def test_record_interaction_writes_and_emits_event(social_graph, send_queue):
    msg = _build_query("record_interaction", user_id="alice", quality=0.8)
    _handle_query(msg, social_graph, send_queue, MODULE_NAME)

    # SocialGraph state mutated
    profile = social_graph._cache.get("alice")
    assert profile is not None
    assert profile.interaction_count == 1
    assert profile.like_score > 0  # quality >= 0.6 → like_score increments

    # SOCIAL_INTERACTION_RECORDED bus event emitted
    msgs = _drain_responses(send_queue)
    event = next((m for m in msgs
                  if m.get("type") == bus.SOCIAL_INTERACTION_RECORDED), None)
    assert event is not None
    assert event["payload"]["user_id"] == "alice"
    assert event["payload"]["quality"] == 0.8

    # RESPONSE returned
    resp = _find_response(msgs, "rid-record_interaction")
    assert resp is not None
    assert resp["payload"]["ok"] is True


def test_get_or_create_user_returns_serialized_profile(social_graph, send_queue):
    msg = _build_query(
        "get_or_create_user", user_id="bob", platform="x",
        display_name="Bob",
    )
    _handle_query(msg, social_graph, send_queue, MODULE_NAME)

    resp = _find_response(
        _drain_responses(send_queue), "rid-get_or_create_user")
    assert resp is not None
    profile_data = resp["payload"]["profile"]
    assert profile_data["user_id"] == "bob"
    assert profile_data["platform"] == "x"
    assert profile_data["display_name"] == "Bob"


def test_should_engage_returns_level_string(social_graph, send_queue):
    msg = _build_query("should_engage", user_id="charlie")
    _handle_query(msg, social_graph, send_queue, MODULE_NAME)

    resp = _find_response(
        _drain_responses(send_queue), "rid-should_engage")
    assert resp is not None
    # New user with _ENGAGEMENT_CURIOUS=0.2 → "minimal" engagement_level
    assert resp["payload"]["level"] in ("ignore", "minimal", "neutral", "warm")


def test_record_donation_emits_event(social_graph, send_queue):
    msg = _build_query(
        "record_donation", tx_signature="tx_abc", sender_address="addr_xyz",
        amount_sol=0.10, memo="thanks",
    )
    _handle_query(msg, social_graph, send_queue, MODULE_NAME)

    msgs = _drain_responses(send_queue)
    event = next((m for m in msgs
                  if m.get("type") == bus.SOCIAL_DONATION_RECORDED), None)
    assert event is not None
    assert event["payload"]["amount_sol"] == 0.10
    # 0.10 SOL → mood_delta=0.10, memory_weight=5.0 per DONATION_TIERS
    assert event["payload"]["mood_delta"] == 0.10
    assert event["payload"]["memory_weight"] == 5.0


def test_record_inspiration_emits_event(social_graph, send_queue):
    msg = _build_query(
        "record_inspiration", tx_signature="tx_insp_1",
        sender_address="addr_ins", message="be brave", amount_sol=0.01,
    )
    _handle_query(msg, social_graph, send_queue, MODULE_NAME)

    msgs = _drain_responses(send_queue)
    event = next((m for m in msgs
                  if m.get("type") == bus.SOCIAL_INSPIRATION_RECORDED), None)
    assert event is not None
    assert event["payload"]["message"] == "be brave"


def test_record_edge_writes(social_graph, send_queue):
    # Pre-seed two users
    social_graph.get_or_create_user("alice")
    social_graph.get_or_create_user("bob")

    msg = _build_query("record_edge", user_a="alice", user_b="bob")
    _handle_query(msg, social_graph, send_queue, MODULE_NAME)

    # Verify edge exists
    conns = social_graph.get_user_connections("alice")
    assert len(conns) == 1
    assert conns[0]["user_id"] == "bob"


def test_ledger_record_writes(social_graph, send_queue):
    msg = _build_query(
        "ledger_record", tweet_id="t_1", user_name="alice",
        action_kind="reply", mention_text="hi",
    )
    _handle_query(msg, social_graph, send_queue, MODULE_NAME)
    assert social_graph.ledger_has_tweet("t_1", action="reply")


def test_ledger_cleanup_returns_removed_count(social_graph, send_queue):
    # Insert a stale row
    import time
    with __import__("sqlite3").connect(social_graph._db_path) as conn:
        conn.execute(
            "INSERT INTO engagement_ledger "
            "(tweet_id, user_name, action, timestamp) VALUES (?, ?, ?, ?)",
            ("t_old", "alice", "reply", time.time() - 86400 * 30),
        )
        conn.commit()
    msg = _build_query("ledger_cleanup", max_age_seconds=86400)
    _handle_query(msg, social_graph, send_queue, MODULE_NAME)
    resp = _find_response(
        _drain_responses(send_queue), "rid-ledger_cleanup")
    assert resp is not None
    assert resp["payload"]["removed"] >= 1


# ── Parameterized read dispatch ──────────────────────────────────────


def test_get_top_users_returns_serialized_list(social_graph, send_queue):
    social_graph.get_or_create_user("alice")
    social_graph.record_interaction("alice", quality=0.9)
    msg = _build_query("get_top_users", limit=5)
    _handle_query(msg, social_graph, send_queue, MODULE_NAME)

    resp = _find_response(
        _drain_responses(send_queue), "rid-get_top_users")
    assert resp is not None
    users = resp["payload"]["users"]
    assert isinstance(users, list)
    assert any(u.get("user_id") == "alice" for u in users)


def test_get_user_connections_via_dispatch(social_graph, send_queue):
    social_graph.get_or_create_user("alice")
    social_graph.get_or_create_user("bob")
    social_graph.record_edge("alice", "bob")
    msg = _build_query("get_user_connections", user_id="alice")
    _handle_query(msg, social_graph, send_queue, MODULE_NAME)
    resp = _find_response(
        _drain_responses(send_queue), "rid-get_user_connections")
    assert resp is not None
    assert len(resp["payload"]["connections"]) == 1


def test_ledger_has_tweet_via_dispatch(social_graph, send_queue):
    social_graph.ledger_record("t_777", "alice", "reply")
    msg = _build_query("ledger_has_tweet", tweet_id="t_777",
                       action_kind="reply")
    _handle_query(msg, social_graph, send_queue, MODULE_NAME)
    resp = _find_response(
        _drain_responses(send_queue), "rid-ledger_has_tweet")
    assert resp is not None
    assert resp["payload"]["has"] is True


# ── Unknown action handling ──────────────────────────────────────────


def test_unknown_action_returns_error_response(social_graph, send_queue):
    msg = _build_query("nonexistent_action", foo="bar")
    _handle_query(msg, social_graph, send_queue, MODULE_NAME)
    resp = _find_response(
        _drain_responses(send_queue), "rid-nonexistent_action")
    assert resp is not None
    assert "error" in resp["payload"]
    assert "unknown_action" in resp["payload"]["error"]


def test_exception_in_handler_returns_error_response(social_graph, send_queue):
    """Bad input → controlled error response, no crash."""
    msg = _build_query("record_interaction", user_id="alice",
                       quality="not_a_number")
    _handle_query(msg, social_graph, send_queue, MODULE_NAME)
    resp = _find_response(
        _drain_responses(send_queue), "rid-record_interaction")
    assert resp is not None
    assert "error" in resp["payload"]


# ── _serialize_profile helper ────────────────────────────────────────


def test_serialize_profile_round_trip(social_graph):
    p = social_graph.get_or_create_user("alice")
    data = _serialize_profile(p)
    assert isinstance(data, dict)
    assert data["user_id"] == "alice"
    assert "engagement_level" in data


def test_serialize_profile_none_returns_empty_dict():
    assert _serialize_profile(None) == {}


# ── Fire-and-forget (no rid) dispatch ────────────────────────────────


def test_no_rid_skips_response(social_graph, send_queue):
    """Fire-and-forget caller (no rid) — handler still does work, no
    RESPONSE frame is enqueued, but event broadcasts still fire."""
    msg = _build_query("record_interaction", user_id="alice", quality=0.7)
    msg["rid"] = None  # explicit no-rid
    _handle_query(msg, social_graph, send_queue, MODULE_NAME)

    msgs = _drain_responses(send_queue)
    # No RESPONSE
    assert not any(m.get("type") == bus.RESPONSE for m in msgs)
    # But SOCIAL_INTERACTION_RECORDED still fires
    assert any(m.get("type") == bus.SOCIAL_INTERACTION_RECORDED for m in msgs)
