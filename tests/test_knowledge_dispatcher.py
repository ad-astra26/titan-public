"""Unit tests for titan_plugin.logic.knowledge_dispatcher (KP-3).

Covers: end-to-end dispatch flow, cache-hit short-circuit, chain-fallback
semantics, INTERNAL_REJECTED short-circuit, SearXNG→Sage delegation,
request coalescing (Essential A), and dispatch_sync adapter.
"""

import asyncio
import os
import tempfile
from unittest.mock import AsyncMock

import pytest

from titan_plugin.logic.knowledge_backends import BackendResult
from titan_plugin.logic.knowledge_cache import KnowledgeCache
from titan_plugin.logic.knowledge_dispatcher import (
    DispatchResult,
    dispatch,
    dispatch_sync,
)
from titan_plugin.logic.knowledge_router import QueryType


@pytest.fixture
def cache():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "search_cache.db")
        yield KnowledgeCache(db_path=path, size_cap=100)


# ── Helpers ──────────────────────────────────────────────────────────

def _mk_backend_fn(results_by_query):
    """Build a fake backend fetcher that returns pre-programmed results.

    results_by_query: dict[str, BackendResult] — key is the normalized
    query string the backend will be called with.
    """
    async def fn(topic, timeout=10.0):
        r = results_by_query.get(topic)
        if r is None:
            return BackendResult(backend="mock", query=topic,
                                 success=False, error_type="empty")
        # Return a fresh copy with this backend's name/query
        return BackendResult(
            backend=r.backend, query=topic, success=r.success,
            raw_text=r.raw_text, structured=r.structured,
            error_type=r.error_type, bytes_consumed=r.bytes_consumed,
            latency_ms=r.latency_ms, status_code=r.status_code,
        )
    return fn


class FakeSage:
    """Minimal duck-typed Sage for delegation tests."""
    def __init__(self, response_text="[SAGE_RESEARCH_FINDINGS]: result"):
        self.response_text = response_text
        self.calls = []

    async def research(self, topic: str) -> str:
        self.calls.append(topic)
        return self.response_text


# ── Internal rejected short-circuit ──────────────────────────────────

class TestInternalRejected:
    @pytest.mark.asyncio
    async def test_underscore_name_rejected(self, cache):
        out = await dispatch("inner_spirit", cache=cache)
        assert out.rejected is True
        assert out.query_type == QueryType.INTERNAL_REJECTED
        assert out.result is None
        assert out.attempts == []
        assert cache.stats()["entries"] == 0

    @pytest.mark.asyncio
    async def test_empty_topic_rejected(self, cache):
        out = await dispatch("", cache=cache)
        assert out.rejected is True
        assert out.normalized == ""

    @pytest.mark.asyncio
    async def test_whitespace_rejected(self, cache):
        out = await dispatch("   \t  ", cache=cache)
        assert out.rejected is True


# ── Direct-REST backend dispatch ─────────────────────────────────────

class TestDirectRESTDispatch:
    @pytest.mark.asyncio
    async def test_dictionary_cache_miss_then_hit(
            self, cache, monkeypatch):
        from titan_plugin.logic import knowledge_dispatcher as kd
        # Mock wiktionary to succeed
        success = BackendResult(
            backend="wiktionary", query="chi",
            success=True, raw_text="[noun] A Greek letter",
            bytes_consumed=42)
        mock = _mk_backend_fn({"chi": success})
        monkeypatch.setitem(kd.BACKEND_REGISTRY, "wiktionary", mock)

        # First call — cache miss, fetches
        out1 = await dispatch("chi", cache=cache)
        assert out1.success
        assert out1.backend_used == "wiktionary"
        assert out1.cache_hit is False
        assert out1.query_type == QueryType.DICTIONARY
        assert out1.attempts == [("wiktionary", "success")]
        assert out1.bytes_consumed_total == 42

        # Second call — cache hit
        out2 = await dispatch("chi", cache=cache)
        assert out2.success
        assert out2.cache_hit is True
        assert out2.backend_used == "wiktionary"
        assert out2.attempts == [("wiktionary", "cache_hit")]
        assert out2.result.raw_text == "[noun] A Greek letter"

    @pytest.mark.asyncio
    async def test_chain_fallback_to_second_backend(
            self, cache, monkeypatch):
        from titan_plugin.logic import knowledge_dispatcher as kd
        wikt_fail = _mk_backend_fn({})  # empty → error_type="empty"
        dict_ok = _mk_backend_fn({"obscureword": BackendResult(
            backend="free_dictionary", query="obscureword", success=True,
            raw_text="[noun] A real definition", bytes_consumed=20)})
        monkeypatch.setitem(kd.BACKEND_REGISTRY, "wiktionary", wikt_fail)
        monkeypatch.setitem(kd.BACKEND_REGISTRY, "free_dictionary", dict_ok)

        out = await dispatch("obscureword", cache=cache)
        assert out.success
        assert out.backend_used == "free_dictionary"
        # wiktionary attempted first (empty), then free_dictionary (success)
        assert out.attempts[0] == ("wiktionary", "empty")
        assert out.attempts[1] == ("free_dictionary", "success")

    @pytest.mark.asyncio
    async def test_all_backends_fail_returns_failure(
            self, cache, monkeypatch):
        from titan_plugin.logic import knowledge_dispatcher as kd
        failing = _mk_backend_fn({})  # every query returns empty
        monkeypatch.setitem(kd.BACKEND_REGISTRY, "wiktionary", failing)
        monkeypatch.setitem(kd.BACKEND_REGISTRY, "free_dictionary", failing)
        monkeypatch.setitem(kd.BACKEND_REGISTRY, "wikipedia_direct", failing)

        # "ontology" is a real single-word dictionary query → walks full
        # dictionary chain (wiktionary → free_dictionary → wikipedia_direct)
        out = await dispatch("ontology", cache=cache)
        assert out.query_type == QueryType.DICTIONARY
        assert out.success is False
        assert out.result is None or out.result.success is False
        assert len(out.attempts) == 3  # all 3 dictionary-chain backends tried
        assert all(a[1] == "empty" for a in out.attempts)

    @pytest.mark.asyncio
    async def test_dictionary_phrase_transforms_query(
            self, cache, monkeypatch):
        from titan_plugin.logic import knowledge_dispatcher as kd
        captured = []

        async def mock_wikt(topic, timeout=10.0):
            captured.append(topic)
            return BackendResult(
                backend="wiktionary", query=topic, success=True,
                raw_text=f"Def: {topic}", bytes_consumed=10)

        monkeypatch.setitem(kd.BACKEND_REGISTRY, "wiktionary", mock_wikt)
        out = await dispatch("own meaning", cache=cache)
        # Dispatcher strips "meaning" before looking up in Wiktionary
        assert captured == ["own"]
        assert out.success


# ── Sage delegation ──────────────────────────────────────────────────

class TestSageDelegation:
    @pytest.mark.asyncio
    async def test_conceptual_routes_to_sage(self, cache):
        sage = FakeSage(response_text=(
            "[SAGE_RESEARCH_FINDINGS]: Conceptual answer"))
        out = await dispatch(
            "hypothesis generation critical thinking",
            cache=cache, sage=sage)
        assert out.success
        assert out.backend_used.startswith("searxng")
        assert out.result.raw_text == "Conceptual answer"
        assert len(sage.calls) == 1

    @pytest.mark.asyncio
    async def test_no_sage_skips_searxng_chain(self, cache):
        # Without sage + no BACKEND_REGISTRY match → all chain entries
        # skipped → result=None
        out = await dispatch(
            "hypothesis generation critical thinking",
            cache=cache, sage=None)
        assert out.success is False
        assert all(a[1] == "skipped" for a in out.attempts)

    @pytest.mark.asyncio
    async def test_sage_empty_response(self, cache):
        sage = FakeSage(response_text="")
        out = await dispatch(
            "hypothesis generation critical thinking",
            cache=cache, sage=sage)
        assert out.success is False


# ── Request coalescing (Essential A) ─────────────────────────────────

class TestRequestCoalescing:
    @pytest.mark.asyncio
    async def test_concurrent_same_query_coalesces(
            self, cache, monkeypatch):
        from titan_plugin.logic import knowledge_dispatcher as kd
        call_count = 0

        async def slow_backend(topic, timeout=10.0):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.05)
            return BackendResult(
                backend="wiktionary", query=topic, success=True,
                raw_text=f"def_{topic}", bytes_consumed=10)

        monkeypatch.setitem(kd.BACKEND_REGISTRY, "wiktionary", slow_backend)

        inflight = {}
        # Launch 5 concurrent dispatches of the same query
        coros = [dispatch("hypothesis", cache=cache, inflight=inflight)
                 for _ in range(5)]
        results = await asyncio.gather(*coros)

        # Only ONE real backend call should have happened
        assert call_count == 1
        # All 5 results should be success
        assert all(r.success for r in results)
        # Exactly one is the "owner" (coalesced=False), the rest attached
        coalesced = sum(r.coalesced for r in results)
        assert coalesced == 4
        assert sum(not r.coalesced for r in results) == 1

    @pytest.mark.asyncio
    async def test_different_queries_no_coalescing(
            self, cache, monkeypatch):
        from titan_plugin.logic import knowledge_dispatcher as kd
        call_count = 0

        async def slow_backend(topic, timeout=10.0):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.02)
            return BackendResult(
                backend="wiktionary", query=topic, success=True,
                raw_text=f"def_{topic}", bytes_consumed=10)

        monkeypatch.setitem(kd.BACKEND_REGISTRY, "wiktionary", slow_backend)
        inflight = {}
        r1, r2 = await asyncio.gather(
            dispatch("hypothesis", cache=cache, inflight=inflight),
            dispatch("ontology", cache=cache, inflight=inflight),
        )
        assert call_count == 2
        assert r1.success and r2.success
        assert not r1.coalesced and not r2.coalesced

    @pytest.mark.asyncio
    async def test_inflight_cleared_after_completion(
            self, cache, monkeypatch):
        from titan_plugin.logic import knowledge_dispatcher as kd

        async def fast_backend(topic, timeout=10.0):
            return BackendResult(
                backend="wiktionary", query=topic, success=True,
                raw_text="x", bytes_consumed=1)

        monkeypatch.setitem(kd.BACKEND_REGISTRY, "wiktionary", fast_backend)
        inflight = {}
        await dispatch("hypothesis", cache=cache, inflight=inflight)
        # inflight dict should be empty after dispatch completes
        assert inflight == {}


# ── dispatch_sync adapter ────────────────────────────────────────────

class TestDispatchSync:
    def test_sync_adapter(self, cache, monkeypatch):
        from titan_plugin.logic import knowledge_dispatcher as kd

        async def mock_wikt(topic, timeout=10.0):
            return BackendResult(
                backend="wiktionary", query=topic, success=True,
                raw_text="Sync def", bytes_consumed=7)

        monkeypatch.setitem(kd.BACKEND_REGISTRY, "wiktionary", mock_wikt)

        # Spin up an event loop in a thread
        import threading
        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=loop.run_forever, daemon=True)
        thread.start()
        try:
            out = dispatch_sync("hypothesis", async_loop=loop, cache=cache)
            assert out.success
            assert out.result.raw_text == "Sync def"
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=2.0)


# ── DispatchResult shape ─────────────────────────────────────────────

class TestDispatchResult:
    def test_success_property(self):
        r = DispatchResult(topic="t", normalized="t",
                           query_type=QueryType.DICTIONARY,
                           result=BackendResult(backend="w", query="t",
                                                success=True))
        assert r.success is True

    def test_rejected_is_not_success(self):
        r = DispatchResult(topic="inner_spirit", normalized="inner_spirit",
                           query_type=QueryType.INTERNAL_REJECTED,
                           rejected=True)
        assert r.success is False

    def test_none_result_is_not_success(self):
        r = DispatchResult(topic="t", normalized="t",
                           query_type=QueryType.DICTIONARY)
        assert r.success is False
