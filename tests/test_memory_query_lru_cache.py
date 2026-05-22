"""Tests for Phase B (rFP §3.4.1) §B5 query LRU+TTL cache.

QueryCache class behavior + integration through _handle_query.
"""
from __future__ import annotations

import time
import threading

import pytest

from titan_hcl.modules._memory_dispatch import QueryCache


# ── QueryCache unit tests ───────────────────────────────────────────────────


class TestQueryCache:
    def test_set_and_get_returns_value(self):
        c = QueryCache(maxsize=8, ttl_s=10.0)
        c.set(("hello", 5), [{"id": 1}])
        assert c.get(("hello", 5)) == [{"id": 1}]

    def test_get_missing_returns_none(self):
        c = QueryCache(maxsize=8, ttl_s=10.0)
        assert c.get(("nope", 1)) is None

    def test_ttl_expiry_returns_none_and_drops_entry(self):
        c = QueryCache(maxsize=8, ttl_s=0.05)
        c.set(("k",), "v")
        assert c.get(("k",)) == "v"
        time.sleep(0.1)
        assert c.get(("k",)) is None
        # Internal store cleared on miss-via-expiry.
        assert c.stats()["size"] == 0

    def test_lru_eviction_when_size_exceeds_maxsize(self):
        c = QueryCache(maxsize=2, ttl_s=10.0)
        c.set("a", 1)
        c.set("b", 2)
        c.set("c", 3)  # evicts 'a' (LRU)
        assert c.get("a") is None
        assert c.get("b") == 2
        assert c.get("c") == 3

    def test_lru_get_promotes_to_most_recently_used(self):
        c = QueryCache(maxsize=2, ttl_s=10.0)
        c.set("a", 1)
        c.set("b", 2)
        # Touch 'a' so it becomes MRU; next set evicts 'b' instead of 'a'.
        c.get("a")
        c.set("c", 3)
        assert c.get("a") == 1
        assert c.get("b") is None
        assert c.get("c") == 3

    def test_set_overwrites_existing_key_without_eviction(self):
        c = QueryCache(maxsize=2, ttl_s=10.0)
        c.set("a", 1)
        c.set("b", 2)
        c.set("a", 99)  # no eviction — same key
        assert c.get("a") == 99
        assert c.get("b") == 2

    def test_invalidate_clears_all_entries(self):
        c = QueryCache(maxsize=8, ttl_s=10.0)
        c.set("a", 1)
        c.set("b", 2)
        c.set("c", 3)
        n = c.invalidate()
        assert n == 3
        assert c.stats()["size"] == 0

    def test_stats_track_hits_misses_evictions(self):
        c = QueryCache(maxsize=2, ttl_s=10.0)
        c.set("a", 1)
        c.get("a")            # hit
        c.get("a")            # hit
        c.get("b")            # miss
        c.set("b", 2)
        c.set("c", 3)         # eviction of 'a'
        s = c.stats()
        assert s["hits"] == 2
        assert s["misses"] == 1
        assert s["evictions"] == 1
        assert s["hit_rate"] == pytest.approx(2 / 3)

    def test_zero_or_negative_maxsize_raises(self):
        with pytest.raises(ValueError):
            QueryCache(maxsize=0, ttl_s=10.0)
        with pytest.raises(ValueError):
            QueryCache(maxsize=-1, ttl_s=10.0)

    def test_zero_or_negative_ttl_raises(self):
        with pytest.raises(ValueError):
            QueryCache(maxsize=8, ttl_s=0)
        with pytest.raises(ValueError):
            QueryCache(maxsize=8, ttl_s=-1.0)

    def test_concurrent_get_set_thread_safe(self):
        """50 threads hammering get/set on overlapping keys must not corrupt."""
        c = QueryCache(maxsize=64, ttl_s=10.0)
        n_threads = 50
        ops_per_thread = 100

        def worker(tid):
            for i in range(ops_per_thread):
                key = (tid % 8, i % 8)  # overlap across threads
                c.set(key, (tid, i))
                _ = c.get(key)

        ts = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(timeout=10.0)
        # No assertion on values — just that no exception was raised + cache
        # is in a coherent state.
        s = c.stats()
        assert s["size"] <= 64


# ── _handle_query integration ───────────────────────────────────────────────


def test_handle_query_uses_cache_on_repeat_call():
    """Second identical query for same (text, top_k) returns cached result
    without invoking memory.query a second time."""
    from queue import Queue
    from titan_hcl.modules._memory_dispatch import (
        InFlightRegistry, WorkerContext,
    )
    from titan_hcl.modules.memory_worker import _handle_query
    from titan_hcl import bus

    call_count = {"n": 0}

    class _StubMemory:
        async def query(self, text, top_k=5):
            call_count["n"] += 1
            return [{"id": 1, "text": text, "top_k": top_k}]

    send_queue: Queue = Queue()
    cache = QueryCache(maxsize=8, ttl_s=10.0)
    ctx = WorkerContext(
        memory=_StubMemory(),
        send_queue=send_queue,
        name="memory",
        config={},
        in_flight=InFlightRegistry(),
        write_lock=threading.RLock(),
        query_cache=cache,
    )
    msg = {
        "type": bus.QUERY,
        "src": "memory_proxy", "dst": "memory",
        "ts": 0, "rid": "rid-1",
        "payload": {"action": "query", "text": "hello", "top_k": 3},
    }
    # First call — cache miss, memory.query invoked.
    _handle_query(msg, ctx)
    assert call_count["n"] == 1
    out1 = send_queue.get_nowait()
    assert out1["payload"]["results"] == [
        {"id": 1, "text": "hello", "top_k": 3}]

    # Second identical call — cache hit, memory.query NOT invoked again.
    msg2 = dict(msg, rid="rid-2")
    _handle_query(msg2, ctx)
    assert call_count["n"] == 1
    out2 = send_queue.get_nowait()
    assert out2["payload"]["results"] == [
        {"id": 1, "text": "hello", "top_k": 3}]
    assert out2["rid"] == "rid-2"  # response uses NEW rid

    # Cache stats reflect 1 miss + 1 hit.
    s = cache.stats()
    assert s["misses"] == 1
    assert s["hits"] == 1


def test_handle_query_cache_disabled_when_query_cache_none():
    """If WorkerContext.query_cache is None (test path), every query hits the backend."""
    from queue import Queue
    from titan_hcl.modules._memory_dispatch import (
        InFlightRegistry, WorkerContext,
    )
    from titan_hcl.modules.memory_worker import _handle_query
    from titan_hcl import bus

    call_count = {"n": 0}

    class _StubMemory:
        async def query(self, text, top_k=5):
            call_count["n"] += 1
            return []

    ctx = WorkerContext(
        memory=_StubMemory(),
        send_queue=Queue(),
        name="memory",
        config={},
        in_flight=InFlightRegistry(),
        write_lock=threading.RLock(),
        query_cache=None,
    )
    msg = {
        "type": bus.QUERY,
        "src": "memory_proxy", "dst": "memory",
        "ts": 0, "rid": "r1",
        "payload": {"action": "query", "text": "x", "top_k": 5},
    }
    _handle_query(msg, ctx)
    _handle_query(msg, ctx)
    assert call_count["n"] == 2  # no cache → backend called twice
