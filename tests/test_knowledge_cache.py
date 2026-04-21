"""Unit tests for titan_plugin.logic.knowledge_cache (KP-2).

Each test uses a fresh tmp DB path so runs are hermetic. Covers:
  * resolve_ttl() mapping — success types + failure taxonomy + no-cache
  * put() + get() round-trip for every query type
  * TTL expiry → miss + opportunistic deletion
  * hit counters, bytes-saved telemetry, per-day counter reset
  * evict_expired() removes only past-TTL rows
  * evict_lru() enforces size_cap + prefers least-recently-used
  * stats() shape + derived hit_rate
"""

import json
import os
import tempfile
import time

import pytest

from titan_plugin.logic.knowledge_cache import (
    CacheEntry,
    KnowledgeCache,
    TTL_FAILURE,
    TTL_SUCCESS,
    resolve_ttl,
)
from titan_plugin.logic.knowledge_router import QueryType


@pytest.fixture
def cache():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "search_cache.db")
        yield KnowledgeCache(db_path=path, size_cap=100)


# ── resolve_ttl ──────────────────────────────────────────────────────

class TestResolveTTL:
    def test_success_per_type(self):
        assert resolve_ttl(QueryType.DICTIONARY, True) == 30 * 86400
        assert resolve_ttl(QueryType.WIKIPEDIA_LIKE, True) == 7 * 86400
        assert resolve_ttl(QueryType.CONCEPTUAL, True) == 24 * 3600
        assert resolve_ttl(QueryType.TECHNICAL, True) == 12 * 3600
        assert resolve_ttl(QueryType.NEWS, True) == 3600

    def test_failure_per_error_type(self):
        assert resolve_ttl(QueryType.DICTIONARY, False, "rate_limit") == 300
        assert resolve_ttl(QueryType.DICTIONARY, False, "http_5xx") == 600
        assert resolve_ttl(QueryType.DICTIONARY, False, "empty") == 3600
        assert resolve_ttl(QueryType.DICTIONARY, False, "http_4xx") == 3600
        assert resolve_ttl(QueryType.DICTIONARY, False, "parse_error") == 3600

    def test_no_cache_failures(self):
        # timeout + network must NOT be cached (retry next call)
        assert resolve_ttl(QueryType.DICTIONARY, False, "timeout") == 0
        assert resolve_ttl(QueryType.DICTIONARY, False, "network") == 0

    def test_unknown_error_type_is_no_cache(self):
        assert resolve_ttl(QueryType.DICTIONARY, False, "banana") == 0


# ── put + get round-trip ─────────────────────────────────────────────

class TestPutGet:
    def test_success_roundtrip(self, cache):
        payload = {"raw_text": "A musical pitch", "structured": {"en": [...]}}
        ok = cache.put(
            query_hash="h1", query_text="chi", query_type=QueryType.DICTIONARY,
            backend="wiktionary", result_payload=payload, success=True,
            quality_score=0.85, bytes_consumed=512)
        assert ok is True

        entry = cache.get("h1")
        assert entry is not None
        assert entry.query_text == "chi"
        assert entry.query_type == "dictionary"
        assert entry.backend == "wiktionary"
        assert entry.success is True
        assert entry.quality_score == 0.85
        assert entry.bytes_consumed == 512
        assert entry.hit_count == 1  # incremented by get()
        assert entry.ts_last_hit > 0
        # Payload survived JSON round-trip
        assert json.loads(entry.result_json)["raw_text"] == "A musical pitch"

    def test_miss_returns_none(self, cache):
        assert cache.get("nonexistent") is None

    def test_failure_roundtrip(self, cache):
        ok = cache.put(
            query_hash="h2", query_text="unknown word", query_type=QueryType.DICTIONARY,
            backend="wiktionary", result_payload={}, success=False,
            error_type="empty")
        assert ok is True
        entry = cache.get("h2")
        assert entry is not None
        assert entry.success is False
        assert entry.error_type == "empty"
        # TTL should match TTL_FAILURE["empty"]
        assert entry.ttl_seconds == TTL_FAILURE["empty"]

    def test_no_cache_for_timeout_returns_false(self, cache):
        ok = cache.put(
            query_hash="h3", query_text="x", query_type=QueryType.DICTIONARY,
            backend="wiktionary", result_payload={}, success=False,
            error_type="timeout")
        assert ok is False
        assert cache.get("h3") is None

    def test_no_cache_for_network_returns_false(self, cache):
        ok = cache.put(
            query_hash="h4", query_text="x", query_type=QueryType.DICTIONARY,
            backend="wiktionary", result_payload={}, success=False,
            error_type="network")
        assert ok is False

    def test_replace_existing(self, cache):
        cache.put("h5", "x", QueryType.DICTIONARY, "wiktionary",
                  {"v": 1}, True, bytes_consumed=100)
        cache.put("h5", "x", QueryType.DICTIONARY, "wiktionary",
                  {"v": 2}, True, bytes_consumed=200)
        entry = cache.get("h5")
        assert json.loads(entry.result_json)["v"] == 2
        assert entry.bytes_consumed == 200
        # Hit count reset on replace (row is new)
        assert entry.hit_count == 1


# ── TTL expiry ───────────────────────────────────────────────────────

class TestTTLExpiry:
    def test_expired_entry_evicted_on_get(self, cache, monkeypatch):
        cache.put("h_exp", "x", QueryType.DICTIONARY, "wiktionary",
                  {}, True, bytes_consumed=10)
        # Tamper with ts_cached to force expiry
        import sqlite3
        conn = sqlite3.connect(cache.db_path)
        # Set ts_cached to 40 days ago → past TTL_SUCCESS[DICTIONARY] = 30 days
        conn.execute("UPDATE search_cache SET ts_cached = ? WHERE query_hash = ?",
                     (time.time() - 40 * 86400, "h_exp"))
        conn.commit()
        conn.close()
        # Now a get() should evict + return None
        assert cache.get("h_exp") is None
        # Subsequent get still None (deleted)
        assert cache.get("h_exp") is None

    def test_fresh_entry_not_expired(self, cache):
        cache.put("h_fresh", "x", QueryType.NEWS, "news_api",
                  {}, True, bytes_consumed=10)
        entry = cache.get("h_fresh")
        assert entry is not None
        assert entry.is_expired is False


# ── Counters + bytes saved ───────────────────────────────────────────

class TestCountersTelemetry:
    def test_hits_misses_counters(self, cache):
        cache.put("hit1", "x", QueryType.DICTIONARY, "w", {}, True,
                  bytes_consumed=100)
        cache.get("hit1")  # hit
        cache.get("hit1")  # hit
        cache.get("not_cached")  # miss
        s = cache.stats()
        assert s["hits_24h"] == 2
        assert s["misses_24h"] == 1
        assert abs(s["hit_rate"] - 2 / 3) < 0.001

    def test_bytes_saved_accumulates_on_hits(self, cache):
        cache.put("h_b", "x", QueryType.DICTIONARY, "w", {},
                  True, bytes_consumed=500)
        cache.get("h_b")
        cache.get("h_b")
        s = cache.stats()
        assert s["bytes_saved_24h_estimate"] == 1000  # 2 hits × 500

    def test_counter_reset_on_day_rollover(self, cache):
        cache.put("h", "x", QueryType.DICTIONARY, "w", {},
                  True, bytes_consumed=10)
        cache.get("h")
        assert cache.stats()["hits_24h"] == 1
        # Simulate day rollover by mutating counter epoch
        cache._counter_day_epoch = cache._current_day_epoch() - 1
        cache._maybe_reset_counters()
        assert cache._hits_today == 0
        assert cache._misses_today == 0
        assert cache._bytes_saved_today == 0


# ── Eviction ─────────────────────────────────────────────────────────

class TestEviction:
    def test_evict_expired_removes_past_ttl_only(self, cache):
        import sqlite3
        cache.put("fresh", "a", QueryType.DICTIONARY, "w", {}, True)
        cache.put("stale", "b", QueryType.NEWS, "w", {}, True)  # 1h TTL
        conn = sqlite3.connect(cache.db_path)
        conn.execute("UPDATE search_cache SET ts_cached = ? WHERE query_hash = ?",
                     (time.time() - 7200, "stale"))  # 2h ago
        conn.commit()
        conn.close()
        evicted = cache.evict_expired()
        assert evicted == 1
        assert cache.get("fresh") is not None
        assert cache.get("stale") is None

    def test_evict_lru_enforces_size_cap(self, cache):
        # size_cap = 100 (from fixture)
        for i in range(120):
            cache.put(f"key{i:03d}", f"q{i}", QueryType.DICTIONARY, "w",
                      {}, True, bytes_consumed=10)
        # Force LRU by hitting some entries (they survive)
        for i in range(50, 60):
            cache.get(f"key{i:03d}")
        evicted = cache.evict_lru()
        assert evicted >= 20  # had 120, cap=100 → at least 20 evicted
        remaining = cache.stats()["entries"]
        assert remaining <= 100
        # The hit entries should still be present (most recent ts_last_hit)
        for i in range(50, 60):
            assert cache.get(f"key{i:03d}") is not None

    def test_evict_lru_nothing_under_cap(self, cache):
        cache.put("one", "x", QueryType.DICTIONARY, "w", {}, True)
        assert cache.evict_lru() == 0

    def test_evict_lru_explicit_max(self, cache):
        for i in range(50):
            cache.put(f"k{i}", f"q{i}", QueryType.DICTIONARY, "w", {}, True)
        evicted = cache.evict_lru(max_entries=10)
        assert evicted == 40
        assert cache.stats()["entries"] == 10


# ── stats shape ──────────────────────────────────────────────────────

class TestStatsShape:
    def test_empty_cache_stats(self, cache):
        s = cache.stats()
        assert s["entries"] == 0
        assert s["hits_24h"] == 0
        assert s["misses_24h"] == 0
        assert s["hit_rate"] == 0.0
        assert s["by_query_type"] == {}
        assert s["by_backend"] == {}

    def test_stats_by_type_and_backend(self, cache):
        cache.put("a", "x", QueryType.DICTIONARY, "wiktionary", {}, True)
        cache.put("b", "y", QueryType.DICTIONARY, "wiktionary", {}, True)
        cache.put("c", "z", QueryType.CONCEPTUAL, "searxng_ddg_brave_wiki",
                  {}, True)
        s = cache.stats()
        assert s["by_query_type"] == {"dictionary": 2, "conceptual": 1}
        assert s["by_backend"] == {"wiktionary": 2, "searxng_ddg_brave_wiki": 1}


# ── CacheEntry ───────────────────────────────────────────────────────

class TestCacheEntry:
    def test_is_expired_true_for_past_ttl(self):
        e = CacheEntry(
            query_hash="h", query_text="x", query_type="dictionary",
            backend="w", result_json="{}", success=True,
            ts_cached=time.time() - 7200, ttl_seconds=3600)
        assert e.is_expired is True

    def test_is_expired_false_for_fresh(self):
        e = CacheEntry(
            query_hash="h", query_text="x", query_type="dictionary",
            backend="w", result_json="{}", success=True,
            ts_cached=time.time() - 100, ttl_seconds=3600)
        assert e.is_expired is False

    def test_age_seconds(self):
        t = time.time() - 42
        e = CacheEntry(
            query_hash="h", query_text="x", query_type="dictionary",
            backend="w", result_json="{}", success=True,
            ts_cached=t, ttl_seconds=3600)
        assert 41 <= e.age_seconds <= 44
