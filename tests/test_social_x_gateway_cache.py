"""Tests for SocialXGateway TTL cache + 3 new public methods.

Closes BUG-X-API-LEAK-FROM-DISCOVER-MENTIONS-20260430:
fleet-wide ~4500 calls/day projected → ~800-1200/day target via TTL cache
on user/mentions (300s) + last_tweets (300s) + advanced_search (120s).

Tests:
  - cache_key stable across payload reorder
  - GET hits cache on second call within TTL
  - GET misses cache after TTL expiry
  - WRITE methods (POST) NEVER cache
  - LRU eviction at max_size
  - Failed/error responses NOT cached
  - 3 new public methods (fetch_user_relationships / fetch_recent_tweets /
    search_tweets) delegate correctly to _call_x_api
  - SocialPostHelper init guard fires when _explicit_opt_in=False
"""
import time
from unittest.mock import patch, MagicMock

import pytest

from titan_plugin.logic.social_x_gateway import SocialXGateway


@pytest.fixture
def gateway(tmp_path):
    """Fresh gateway with mocked config + temp DB."""
    db = tmp_path / "social_x.db"
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text("[social_x]\nenabled = true\n")
    g = SocialXGateway(db_path=str(db), config_path=str(cfg_path),
                       telemetry_path=str(tmp_path / "telemetry.jsonl"))
    return g


def _stub_config(g, ttls: dict):
    """Patch _load_config to return cache TTLs."""
    base_cfg = {
        "enabled": True, "consumers": {}, "cache": {"enabled": True, "max_size": 8, **ttls},
        "max_posts_per_hour": 100, "max_posts_per_day": 1000,
        "min_post_interval": 0, "max_replies_per_hour": 100,
        "max_replies_per_day": 1000, "min_reply_interval": 0,
        "max_likes_per_hour": 100, "max_likes_per_day": 1000,
        "max_searches_per_hour": 100, "max_post_length": 500,
        "quality_gate": False, "session": "", "proxy": "",
        "api_key": "test", "user_name": "iamtitanai",
        "url_domain": "", "limits": {}, "replies": {},
        "voice": {},
    }
    g._load_config = lambda: base_cfg


# ── Cache key stability ────────────────────────────────────────────────

def test_cache_key_filters_auth_fields(gateway):
    """Cache key must NOT include api_key/proxy/session — same query, different
    auth must hit the same cache entry."""
    k1 = gateway._cache_key_for("twitter/user/mentions", "GET",
                                 {"userName": "iamtitanai", "count": 20,
                                  "X-API-Key": "key1"})
    k2 = gateway._cache_key_for("twitter/user/mentions", "GET",
                                 {"userName": "iamtitanai", "count": 20,
                                  "X-API-Key": "key2"})
    assert k1 == k2


def test_cache_key_payload_order_independent(gateway):
    k1 = gateway._cache_key_for("twitter/user/mentions", "GET",
                                 {"userName": "iamtitanai", "count": 20})
    k2 = gateway._cache_key_for("twitter/user/mentions", "GET",
                                 {"count": 20, "userName": "iamtitanai"})
    assert k1 == k2


def test_cache_key_different_payloads_distinct(gateway):
    k1 = gateway._cache_key_for("twitter/user/mentions", "GET",
                                 {"userName": "u1", "count": 20})
    k2 = gateway._cache_key_for("twitter/user/mentions", "GET",
                                 {"userName": "u2", "count": 20})
    assert k1 != k2


# ── TTL resolution from config ─────────────────────────────────────────

def test_ttl_for_endpoint_from_config(gateway):
    _stub_config(gateway, {"ttl_user_mentions": 300, "ttl_user_last_tweets": 600})
    assert gateway._ttl_for_endpoint("twitter/user/mentions") == 300
    assert gateway._ttl_for_endpoint("twitter/user/last_tweets") == 600


def test_ttl_zero_when_unset(gateway):
    _stub_config(gateway, {})
    assert gateway._ttl_for_endpoint("twitter/user/mentions") == 0


def test_ttl_zero_when_cache_disabled(gateway):
    _stub_config(gateway, {"ttl_user_mentions": 300})
    gateway._load_config = lambda: {"cache": {"enabled": False, "ttl_user_mentions": 300}}
    assert gateway._ttl_for_endpoint("twitter/user/mentions") == 0


# ── Cache hit/miss/expiry ──────────────────────────────────────────────

def test_cache_hit_within_ttl(gateway):
    _stub_config(gateway, {"ttl_user_mentions": 300})
    fake_response = {"status": "success", "tweets": [{"id": "1", "text": "hi"}]}

    def fake_get(url, params=None, headers=None, timeout=None):
        m = MagicMock()
        m.status_code = 200
        m.json.return_value = fake_response
        return m

    with patch("httpx.get", side_effect=fake_get) as mocked:
        r1 = gateway._call_x_api("twitter/user/mentions", method="GET",
                                  payload={"userName": "iamtitanai", "count": 20},
                                  api_key="test")
        r2 = gateway._call_x_api("twitter/user/mentions", method="GET",
                                  payload={"userName": "iamtitanai", "count": 20},
                                  api_key="test")
    assert r1 == r2 == fake_response
    assert mocked.call_count == 1  # second call hit cache
    stats = gateway.get_cache_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1


def test_cache_miss_after_expiry(gateway):
    _stub_config(gateway, {"ttl_user_mentions": 1})
    fake_response = {"status": "success", "tweets": []}

    def fake_get(url, params=None, headers=None, timeout=None):
        m = MagicMock()
        m.status_code = 200
        m.json.return_value = fake_response
        return m

    with patch("httpx.get", side_effect=fake_get) as mocked:
        gateway._call_x_api("twitter/user/mentions", method="GET",
                            payload={"userName": "u", "count": 20}, api_key="test")
        # Force cache expiry by manipulating timestamp
        for k, (_ts, v) in list(gateway._api_cache.items()):
            gateway._api_cache[k] = (time.time() - 10, v)
        gateway._call_x_api("twitter/user/mentions", method="GET",
                            payload={"userName": "u", "count": 20}, api_key="test")
    assert mocked.call_count == 2  # cache expired → second call hit API


def test_post_never_cached(gateway):
    _stub_config(gateway, {"ttl_create_tweet_v2": 0})
    fake_response = {"status": "success", "tweet_id": "abc"}

    def fake_post(url, json=None, headers=None, timeout=None):
        m = MagicMock()
        m.status_code = 200
        m.json.return_value = fake_response
        return m

    with patch("httpx.post", side_effect=fake_post) as mocked:
        gateway._call_x_api("twitter/create_tweet_v2", method="POST",
                            payload={"text": "hello"}, api_key="test")
        gateway._call_x_api("twitter/create_tweet_v2", method="POST",
                            payload={"text": "hello"}, api_key="test")
    assert mocked.call_count == 2  # POSTs never cached


def test_error_response_not_cached(gateway):
    _stub_config(gateway, {"ttl_user_mentions": 300})
    err_response = {"status": "error", "message": "rate limited"}

    def fake_get(url, params=None, headers=None, timeout=None):
        m = MagicMock()
        m.status_code = 429
        m.json.return_value = err_response
        return m

    with patch("httpx.get", side_effect=fake_get) as mocked:
        r1 = gateway._call_x_api("twitter/user/mentions", method="GET",
                                  payload={"userName": "u", "count": 20},
                                  api_key="test")
        r2 = gateway._call_x_api("twitter/user/mentions", method="GET",
                                  payload={"userName": "u", "count": 20},
                                  api_key="test")
    # Errors must NOT be cached — both calls hit the API
    assert mocked.call_count == 2


# ── LRU eviction ───────────────────────────────────────────────────────

def test_lru_eviction_at_max_size(gateway):
    _stub_config(gateway, {"ttl_user_mentions": 300})

    def fake_get(url, params=None, headers=None, timeout=None):
        m = MagicMock()
        m.status_code = 200
        m.json.return_value = {"status": "success",
                                 "userName": params.get("userName", "")}
        return m

    with patch("httpx.get", side_effect=fake_get):
        # max_size = 8 from _stub_config
        for i in range(10):
            gateway._call_x_api(
                "twitter/user/mentions", method="GET",
                payload={"userName": f"u{i}", "count": 20},
                api_key="test")
    assert len(gateway._api_cache) == 8
    stats = gateway.get_cache_stats()
    assert stats["evictions"] >= 2


# ── 3 new public methods ───────────────────────────────────────────────

def test_fetch_user_relationships_delegates(gateway):
    _stub_config(gateway, {"ttl_user_followers": 600})

    captured = {}

    def fake_get(url, params=None, headers=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        m = MagicMock()
        m.status_code = 200
        m.json.return_value = {"status": "success", "followers": []}
        return m

    with patch("httpx.get", side_effect=fake_get):
        result = gateway.fetch_user_relationships(
            user_name="iamtitanai", relationship="followers",
            count=50, api_key="test")
    assert result["status"] == "success"
    assert captured["url"] == "https://api.twitterapi.io/twitter/user/followers"
    assert captured["params"]["userName"] == "iamtitanai"
    assert captured["params"]["count"] == 50


def test_fetch_user_relationships_invalid_relationship_errors(gateway):
    _stub_config(gateway, {})
    result = gateway.fetch_user_relationships(
        user_name="x", relationship="invalid", count=10, api_key="test")
    assert result["status"] == "error"


def test_fetch_recent_tweets_delegates(gateway):
    _stub_config(gateway, {"ttl_user_last_tweets": 300})

    captured = {}

    def fake_get(url, params=None, headers=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        m = MagicMock()
        m.status_code = 200
        m.json.return_value = {"status": "success", "tweets": []}
        return m

    with patch("httpx.get", side_effect=fake_get):
        result = gateway.fetch_recent_tweets(
            user_name="iamtitanai", count=10, api_key="test")
    assert result["status"] == "success"
    assert captured["url"] == "https://api.twitterapi.io/twitter/user/last_tweets"
    assert captured["params"]["userName"] == "iamtitanai"


def test_search_tweets_delegates(gateway):
    _stub_config(gateway, {"ttl_tweet_advanced_search": 120})

    captured = {}

    def fake_get(url, params=None, headers=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        m = MagicMock()
        m.status_code = 200
        m.json.return_value = {"status": "success", "tweets": []}
        return m

    with patch("httpx.get", side_effect=fake_get):
        result = gateway.search_tweets(query="from:foo", count=20, api_key="test")
    assert result["status"] == "success"
    assert captured["params"]["query"] == "from:foo"
    assert captured["params"]["queryType"] == "Latest"


def test_three_methods_share_cache(gateway):
    """Calling fetch_recent_tweets twice with same args hits cache once."""
    _stub_config(gateway, {"ttl_user_last_tweets": 300})

    def fake_get(url, params=None, headers=None, timeout=None):
        m = MagicMock()
        m.status_code = 200
        m.json.return_value = {"status": "success", "tweets": [{"id": "1"}]}
        return m

    with patch("httpx.get", side_effect=fake_get) as mocked:
        gateway.fetch_recent_tweets(user_name="x", count=10, api_key="test")
        gateway.fetch_recent_tweets(user_name="x", count=10, api_key="test")
    assert mocked.call_count == 1


# ── SocialPostHelper disabled-by-default guard ─────────────────────────

def test_social_post_helper_blocked_by_default():
    from titan_plugin.logic.agency.helpers.social_post import SocialPostHelper
    with pytest.raises(ValueError, match="DISABLED"):
        SocialPostHelper(api_key="x")


def test_social_post_helper_explicit_opt_in_works():
    """For testing only — _explicit_opt_in=True bypasses the guard."""
    from titan_plugin.logic.agency.helpers.social_post import SocialPostHelper
    h = SocialPostHelper(api_key="test", _explicit_opt_in=True)
    assert h._api_key == "test"
