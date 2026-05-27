"""Phase 6 — `x_oracle` TruthOraclePlug tests (§P6.E).

Covers `titan_hcl/synthesis/oracles/x_oracle.py`. The plug takes any
``GatewayLike`` instance; tests inject a fake gateway with scripted
responses so no real X API traffic is required.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import pytest

from titan_hcl.synthesis.oracles.x_oracle import (
    DEFAULT_MIN_DISTINCT_AUTHORS_EVENT,
    DEFAULT_MIN_DISTINCT_AUTHORS_TRENDING,
    DEFAULT_PER_CALL_COST_SOL,
    SUPPORTED_DOMAINS,
    XOracle,
)
from titan_hcl.synthesis.plugs import OracleClaim


# ─────────────────────────────────────────────────────────────────────────
# Fake gateway
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class FakeGateway:
    """Scriptable stand-in for SocialXGateway."""

    search_response: dict = field(default_factory=dict)
    recent_response: dict = field(default_factory=dict)
    search_calls: list = field(default_factory=list)
    recent_calls: list = field(default_factory=list)

    def search_tweets(self, query, query_type="Latest", count=20, *, api_key=""):
        self.search_calls.append({"query": query, "query_type": query_type, "count": count, "api_key": api_key})
        return self.search_response

    def fetch_recent_tweets(self, user_name, count=10, *, api_key=""):
        self.recent_calls.append({"user_name": user_name, "count": count, "api_key": api_key})
        return self.recent_response


def _now_tweets(authors: list[str], now: float | None = None) -> list[dict]:
    """Build a list of tweet dicts with author + recent timestamp."""
    now = now if now is not None else time.time()
    return [{"id": f"id{i}", "author_name": a, "timestamp": now - i * 60} for i, a in enumerate(authors)]


# ─────────────────────────────────────────────────────────────────────────
# Protocol surface
# ─────────────────────────────────────────────────────────────────────────


def test_oracle_id_and_cost_class():
    o = XOracle(FakeGateway())
    assert o.oracle_id == "x_api"
    assert o.cost_class == "metered"


def test_supported_domains():
    assert SUPPORTED_DOMAINS == frozenset(
        {"topic_trending", "account_exists", "post_real", "x_event_real"}
    )


def test_can_handle_surface():
    o = XOracle(FakeGateway())
    for d in SUPPORTED_DOMAINS:
        assert o.can_handle(d) is True
    for d in ("code_correctness", "web_fact", "solana_tx_confirmed"):
        assert o.can_handle(d) is False


def test_verify_unsupported_domain_returns_unknown():
    o = XOracle(FakeGateway())
    v = o.verify(OracleClaim(domain="random", payload={}))
    assert v.verdict == "unknown"
    assert v.evidence_ref == "domain_unsupported"


# ─────────────────────────────────────────────────────────────────────────
# topic_trending
# ─────────────────────────────────────────────────────────────────────────


def test_topic_trending_meets_threshold_is_true():
    gw = FakeGateway(search_response={"tweets": _now_tweets(["a", "b", "c", "d", "e"])})
    o = XOracle(gw, min_distinct_authors_trending=5)
    v = o.verify(OracleClaim(domain="topic_trending", payload={"topic": "AI"}))
    assert v.verdict == "true"
    assert v.evidence_ref == "AI"
    assert v.cost == DEFAULT_PER_CALL_COST_SOL
    # Gateway was called with our topic.
    assert gw.search_calls[0]["query"] == "AI"


def test_topic_trending_below_threshold_is_false():
    gw = FakeGateway(search_response={"tweets": _now_tweets(["a", "b"])})
    o = XOracle(gw, min_distinct_authors_trending=5)
    v = o.verify(OracleClaim(domain="topic_trending", payload={"topic": "AI"}))
    assert v.verdict == "false"


def test_topic_trending_empty_tweets_is_false():
    gw = FakeGateway(search_response={"tweets": []})
    o = XOracle(gw)
    v = o.verify(OracleClaim(domain="topic_trending", payload={"topic": "x"}))
    assert v.verdict == "false"


def test_topic_trending_old_tweets_filtered_out():
    """Tweets older than 24h shouldn't count toward 'trending now'."""
    old_ts = time.time() - (48 * 3600)
    gw = FakeGateway(
        search_response={
            "tweets": [
                {"id": "1", "author_name": "a", "timestamp": old_ts},
                {"id": "2", "author_name": "b", "timestamp": old_ts},
                {"id": "3", "author_name": "c", "timestamp": old_ts},
                {"id": "4", "author_name": "d", "timestamp": old_ts},
                {"id": "5", "author_name": "e", "timestamp": old_ts},
            ]
        }
    )
    o = XOracle(gw, min_distinct_authors_trending=5)
    v = o.verify(OracleClaim(domain="topic_trending", payload={"topic": "x"}))
    assert v.verdict == "false"  # all filtered out by recency


def test_topic_trending_dedups_same_author():
    """5 tweets from 2 authors → only 2 distinct → below default threshold."""
    gw = FakeGateway(
        search_response={
            "tweets": _now_tweets(["alice", "alice", "alice", "bob", "bob"])
        }
    )
    o = XOracle(gw, min_distinct_authors_trending=5)
    v = o.verify(OracleClaim(domain="topic_trending", payload={"topic": "x"}))
    assert v.verdict == "false"


def test_topic_trending_missing_topic_is_unknown():
    gw = FakeGateway()
    o = XOracle(gw)
    v = o.verify(OracleClaim(domain="topic_trending", payload={}))
    assert v.verdict == "unknown"
    assert v.evidence_ref == "missing_topic"
    assert gw.search_calls == []


def test_topic_trending_gateway_error_is_unknown():
    gw = FakeGateway(search_response={"status": "error", "message": "rate limited"})
    o = XOracle(gw)
    v = o.verify(OracleClaim(domain="topic_trending", payload={"topic": "x"}))
    assert v.verdict == "unknown"
    assert v.evidence_ref == "gateway_error"
    assert v.cost == DEFAULT_PER_CALL_COST_SOL


def test_topic_trending_gateway_raises_is_unknown():
    class Boom:
        def search_tweets(self, *a, **kw): raise RuntimeError("api down")
        def fetch_recent_tweets(self, *a, **kw): return {}

    o = XOracle(Boom())
    v = o.verify(OracleClaim(domain="topic_trending", payload={"topic": "x"}))
    assert v.verdict == "unknown"
    assert v.evidence_ref == "gateway_exception"


# ─────────────────────────────────────────────────────────────────────────
# x_event_real
# ─────────────────────────────────────────────────────────────────────────


def test_x_event_real_meets_lower_threshold_is_true():
    """Event-real threshold is 3 distinct authors (vs 5 for trending)."""
    gw = FakeGateway(search_response={"tweets": _now_tweets(["a", "b", "c"])})
    o = XOracle(gw, min_distinct_authors_event=3)
    v = o.verify(
        OracleClaim(domain="x_event_real", payload={"event_query": "earthquake"})
    )
    assert v.verdict == "true"
    assert v.evidence_ref == "earthquake"


def test_x_event_real_below_threshold_is_false():
    gw = FakeGateway(search_response={"tweets": _now_tweets(["a", "b"])})
    o = XOracle(gw, min_distinct_authors_event=3)
    v = o.verify(
        OracleClaim(domain="x_event_real", payload={"event_query": "x"})
    )
    assert v.verdict == "false"


def test_x_event_real_missing_query_is_unknown():
    o = XOracle(FakeGateway())
    v = o.verify(OracleClaim(domain="x_event_real", payload={}))
    assert v.verdict == "unknown"
    assert v.evidence_ref == "missing_event_query"


# ─────────────────────────────────────────────────────────────────────────
# account_exists
# ─────────────────────────────────────────────────────────────────────────


def test_account_exists_returns_true_when_tweets_returned():
    gw = FakeGateway(recent_response={"tweets": _now_tweets(["alice"])})
    o = XOracle(gw)
    v = o.verify(OracleClaim(domain="account_exists", payload={"handle": "alice"}))
    assert v.verdict == "true"
    assert v.evidence_ref == "alice"


def test_account_exists_strips_at_prefix():
    """Handle with leading @ should still work."""
    gw = FakeGateway(recent_response={"tweets": _now_tweets(["alice"])})
    o = XOracle(gw)
    v = o.verify(OracleClaim(domain="account_exists", payload={"handle": "@alice"}))
    assert v.verdict == "true"
    # Stored handle is "alice" (no @).
    assert gw.recent_calls[0]["user_name"] == "alice"


def test_account_exists_returns_false_on_user_not_found():
    gw = FakeGateway(
        recent_response={"status": "error", "message": "user not found"}
    )
    o = XOracle(gw)
    v = o.verify(OracleClaim(domain="account_exists", payload={"handle": "ghost"}))
    assert v.verdict == "false"
    assert v.evidence_ref == "ghost"


def test_account_exists_returns_true_when_user_object_present_even_if_no_tweets():
    """Silent account (0 posts) but the user object is in the response."""
    gw = FakeGateway(recent_response={"tweets": [], "user": {"user_name": "alice"}})
    o = XOracle(gw)
    v = o.verify(OracleClaim(domain="account_exists", payload={"handle": "alice"}))
    assert v.verdict == "true"


def test_account_exists_returns_false_on_empty_response():
    gw = FakeGateway(recent_response={"tweets": []})
    o = XOracle(gw)
    v = o.verify(OracleClaim(domain="account_exists", payload={"handle": "ghost"}))
    assert v.verdict == "false"


def test_account_exists_missing_handle_is_unknown():
    o = XOracle(FakeGateway())
    v = o.verify(OracleClaim(domain="account_exists", payload={}))
    assert v.verdict == "unknown"
    assert v.evidence_ref == "missing_handle"


def test_account_exists_gateway_error_is_unknown():
    gw = FakeGateway(recent_response={"status": "error", "message": "rate limited"})
    o = XOracle(gw)
    v = o.verify(OracleClaim(domain="account_exists", payload={"handle": "alice"}))
    assert v.verdict == "unknown"
    assert v.evidence_ref == "gateway_error"


# ─────────────────────────────────────────────────────────────────────────
# post_real
# ─────────────────────────────────────────────────────────────────────────


def test_post_real_returns_true_on_id_match():
    gw = FakeGateway(
        search_response={"tweets": [{"id": "12345", "author_name": "a"}]}
    )
    o = XOracle(gw)
    v = o.verify(OracleClaim(domain="post_real", payload={"post_id": "12345"}))
    assert v.verdict == "true"
    assert v.evidence_ref == "12345"


def test_post_real_returns_true_with_id_str_field():
    """twitterapi.io sometimes uses 'id_str' instead of 'id'."""
    gw = FakeGateway(
        search_response={"tweets": [{"id_str": "67890", "author_name": "b"}]}
    )
    o = XOracle(gw)
    v = o.verify(OracleClaim(domain="post_real", payload={"post_id": "67890"}))
    assert v.verdict == "true"


def test_post_real_returns_false_when_id_not_in_results():
    gw = FakeGateway(
        search_response={"tweets": [{"id": "99999", "author_name": "a"}]}
    )
    o = XOracle(gw)
    v = o.verify(OracleClaim(domain="post_real", payload={"post_id": "11111"}))
    assert v.verdict == "false"


def test_post_real_returns_false_on_empty_response():
    gw = FakeGateway(search_response={"tweets": []})
    o = XOracle(gw)
    v = o.verify(OracleClaim(domain="post_real", payload={"post_id": "11111"}))
    assert v.verdict == "false"


def test_post_real_missing_id_is_unknown():
    o = XOracle(FakeGateway())
    v = o.verify(OracleClaim(domain="post_real", payload={}))
    assert v.verdict == "unknown"
    assert v.evidence_ref == "missing_post_id"
