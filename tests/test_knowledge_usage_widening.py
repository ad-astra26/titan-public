"""
Tests for the KP-v2 follow-up fix bundle (2026-04-21):

  * BUG-KP-WEBSEARCH-HEALTH-DEFAULTS — WebSearchHelper accepts + forwards
    `budgets` to HealthTracker so it doesn't clobber knowledge_worker's
    shared health.json on write.
  * BUG-KNOWLEDGE-USAGE-ZERO-T2T3 root — spirit_worker:8770 dst routing
    fix (not tested directly here because spirit_worker is subprocess
    entry; see test_spirit_knowledge_usage_routing.py for the dst shape
    guard).
  * knowledge_gate enhancement — check_topic_confidence_with_match()
    returns (confidence, matched_topic) so callers can attribute
    CGN_KNOWLEDGE_USAGE emissions to the correct concept row.
  * Coverage widening — social_x_gateway + agno_hooks emit
    CGN_KNOWLEDGE_USAGE whenever a knowledge concept contributes to a
    downstream output (post grounding, chat context, experience narrative).
"""

from __future__ import annotations

import os
import sqlite3
import tempfile

import pytest

from titan_plugin.logic.knowledge_gate import (
    check_topic_confidence,
    check_topic_confidence_with_match,
)


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def knowledge_db(tmp_path):
    """Fresh knowledge_concepts table with a few rows."""
    db_path = str(tmp_path / "inner_memory.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE knowledge_concepts (
            topic TEXT PRIMARY KEY,
            confidence REAL,
            summary TEXT
        )
    """)
    conn.executemany(
        "INSERT INTO knowledge_concepts (topic, confidence, summary) "
        "VALUES (?, ?, ?)",
        [
            ("cognitive flexibility", 0.82, "ability to switch mental sets"),
            ("serendipity", 0.65, "fortunate discovery"),
            ("ephemeral nature", 0.40, "short-lived"),
            ("low confidence term", 0.05, "should not beat others"),
        ])
    conn.commit()
    conn.close()
    return db_path


# ── knowledge_gate enhancement ────────────────────────────────────────

def test_check_topic_confidence_with_match_returns_topic(knowledge_db):
    conf, topic = check_topic_confidence_with_match(
        ["serendipity"], db_path=knowledge_db)
    assert 0.64 < conf < 0.66
    assert topic == "serendipity"


def test_check_topic_confidence_with_match_picks_highest(knowledge_db):
    # Both 'cognitive' + 'serendipity' match; 'cognitive' has higher conf
    conf, topic = check_topic_confidence_with_match(
        ["cognitive", "serendipity"], db_path=knowledge_db)
    assert 0.81 < conf < 0.83
    assert "cognitive" in topic


def test_check_topic_confidence_with_match_no_match(knowledge_db):
    conf, topic = check_topic_confidence_with_match(
        ["quantum", "chromosome"], db_path=knowledge_db)
    assert conf == 0.0
    assert topic == ""


def test_check_topic_confidence_with_match_empty_topics(knowledge_db):
    conf, topic = check_topic_confidence_with_match([], db_path=knowledge_db)
    assert conf == 0.0
    assert topic == ""


def test_check_topic_confidence_with_match_missing_db():
    conf, topic = check_topic_confidence_with_match(
        ["anything"], db_path="/tmp/does_not_exist_xyz.db")
    assert conf == 0.0
    assert topic == ""


def test_legacy_check_topic_confidence_still_returns_float(knowledge_db):
    # Backward-compat: existing callers of check_topic_confidence must
    # still get a bare float.
    v = check_topic_confidence(["serendipity"], db_path=knowledge_db)
    assert isinstance(v, float)
    assert 0.64 < v < 0.66


# ── WebSearchHelper budgets forwarding ────────────────────────────────

def test_websearch_helper_accepts_budgets():
    from titan_plugin.logic.agency.helpers.web_search import WebSearchHelper
    budgets = {"wiktionary": 52_428_800, "wikipedia_direct": 104_857_600}
    h = WebSearchHelper(budgets=budgets)
    assert h._budgets == budgets


def test_websearch_helper_budgets_default_empty():
    from titan_plugin.logic.agency.helpers.web_search import WebSearchHelper
    h = WebSearchHelper()  # no budgets
    assert h._budgets == {}


def test_websearch_helper_forwards_budgets_to_health_tracker(tmp_path):
    from titan_plugin.logic.agency.helpers.web_search import WebSearchHelper
    budgets = {"wiktionary": 52_428_800}
    h = WebSearchHelper(
        health_path=str(tmp_path / "h.json"),
        decision_log_path=str(tmp_path / "d.jsonl"),
        budgets=budgets,
    )
    tracker = h._ensure_health()
    assert tracker is not None
    # HealthTracker stores budgets as `_default_budgets`
    assert tracker._default_budgets == budgets


def test_websearch_helper_no_budgets_no_default_budgets(tmp_path):
    """Regression: the old behaviour must stay safe when caller doesn't
    pass budgets — HealthTracker's _default_budgets becomes empty."""
    from titan_plugin.logic.agency.helpers.web_search import WebSearchHelper
    h = WebSearchHelper(
        health_path=str(tmp_path / "h.json"),
        decision_log_path=str(tmp_path / "d.jsonl"),
    )
    tracker = h._ensure_health()
    assert tracker._default_budgets == {}


# ── CGN_KNOWLEDGE_USAGE emission shape via social_x_gateway helper ────

def test_social_x_gateway_emit_knowledge_usage_happy_path():
    """The gateway helper emits a properly shaped CGN_KNOWLEDGE_USAGE
    when a grounded concept contributed to post publish.
    """
    from titan_plugin.logic.social_x_gateway import SocialXGateway

    gw = SocialXGateway.__new__(SocialXGateway)  # bypass __init__

    published = []
    class _Bus:
        def publish(self, msg):
            published.append(msg)

    gw._emit_knowledge_usage(_Bus(), "cognitive flexibility",
                              reward=0.3, consumer="social")
    assert len(published) == 1
    msg = published[0]
    assert msg["type"] == "CGN_KNOWLEDGE_USAGE"
    assert msg["dst"] == "knowledge"
    assert msg["src"] == "social_x_gateway"
    assert msg["payload"]["topic"] == "cognitive flexibility"
    assert msg["payload"]["reward"] == 0.3
    assert msg["payload"]["consumer"] == "social"


def test_social_x_gateway_emit_knowledge_usage_empty_topic_noop():
    from titan_plugin.logic.social_x_gateway import SocialXGateway
    gw = SocialXGateway.__new__(SocialXGateway)
    published = []
    class _Bus:
        def publish(self, msg):
            published.append(msg)
    gw._emit_knowledge_usage(_Bus(), "", reward=0.3, consumer="social")
    assert published == []


def test_social_x_gateway_emit_knowledge_usage_none_bus_safe():
    from titan_plugin.logic.social_x_gateway import SocialXGateway
    gw = SocialXGateway.__new__(SocialXGateway)
    # Should not raise even though bus is None
    gw._emit_knowledge_usage(None, "cognitive flexibility",
                              reward=0.3, consumer="social")


def test_social_x_gateway_emit_knowledge_usage_callable_bus():
    """Bus can be a plain callable (worker-IPC shape)."""
    from titan_plugin.logic.social_x_gateway import SocialXGateway
    gw = SocialXGateway.__new__(SocialXGateway)
    received = []
    gw._emit_knowledge_usage(lambda m: received.append(m),
                              "cognitive flexibility",
                              reward=0.3, consumer="social")
    assert len(received) == 1
    assert received[0]["type"] == "CGN_KNOWLEDGE_USAGE"
    assert received[0]["dst"] == "knowledge"


def test_social_x_gateway_emit_knowledge_usage_bus_exception_swallowed():
    """A bus that throws must not break the emit."""
    from titan_plugin.logic.social_x_gateway import SocialXGateway
    gw = SocialXGateway.__new__(SocialXGateway)
    class _BadBus:
        def publish(self, _msg):
            raise RuntimeError("bus explode")
    gw._emit_knowledge_usage(_BadBus(), "cognitive flexibility",
                              reward=0.3, consumer="social")
    # No raise → test passes


# ── spirit_worker:8770 dst routing regression guard ──────────────────

def test_spirit_worker_knowledge_usage_dst_is_knowledge():
    """Regression guard: AST-check that every CGN_KNOWLEDGE_USAGE emit in
    spirit_worker uses dst="knowledge" (not `str(winner.source)` which
    silently drops via the bus).
    """
    import ast
    import pathlib
    root = pathlib.Path(__file__).parent.parent
    source = (root / "titan_plugin" / "modules" / "spirit_worker.py").read_text()
    tree = ast.parse(source)
    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # Find _send_msg(..., "CGN_KNOWLEDGE_USAGE", ..., dst_expr, ...)
        func = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if func != "_send_msg":
            continue
        if len(node.args) < 4:
            continue
        msg_type_arg = node.args[1]
        if not (isinstance(msg_type_arg, ast.Constant)
                and msg_type_arg.value == "CGN_KNOWLEDGE_USAGE"):
            continue
        dst_arg = node.args[3]
        # dst must be a literal string == "knowledge"
        if not (isinstance(dst_arg, ast.Constant)
                and dst_arg.value == "knowledge"):
            bad.append((node.lineno, ast.unparse(dst_arg)))
    assert not bad, (
        f"Found CGN_KNOWLEDGE_USAGE emits with non-'knowledge' dst: {bad}. "
        f"Per feedback_bus_dst_must_have_subscriber.md, dst must be the "
        f"consumer module name — \"knowledge\" in this case — else "
        f"DivineBus silently drops the message.")
