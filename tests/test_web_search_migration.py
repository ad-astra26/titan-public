"""Unit tests for KP-3.5: raw_results dispatch + WebSearchHelper migration.

Covers:
  * fetch_searxng_raw success + error paths
  * dispatch(raw_results=True) routes SearXNG chain entries through
    fetch_searxng_raw instead of sage delegation
  * WebSearchHelper.execute() returns the legacy shape, driven by the
    dispatcher underneath
"""

import os
import tempfile
from unittest.mock import AsyncMock

import httpx
import pytest

from titan_plugin.logic.agency.helpers.web_search import WebSearchHelper
from titan_plugin.logic.knowledge_backends import (
    BackendResult,
    fetch_searxng_raw,
)
from titan_plugin.logic.knowledge_cache import KnowledgeCache
from titan_plugin.logic.knowledge_dispatcher import dispatch
from titan_plugin.logic.knowledge_router import QueryType


def _patch_httpx_get(response_or_exc):
    """Patch httpx.AsyncClient.get. Same helper shape as other test files."""
    class _FakeClient:
        def __init__(self, *a, **kw):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            return False
        async def get(self, *a, **kw):
            if isinstance(response_or_exc, Exception):
                raise response_or_exc
            return response_or_exc
    from unittest.mock import patch
    return patch.object(httpx, "AsyncClient", _FakeClient)


def _mk_response(status_code, json_data=None, content=b"{}"):
    resp = AsyncMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.content = content
    resp.json = lambda: json_data if json_data is not None else {}
    return resp


@pytest.fixture
def cache():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "cache.db")
        yield KnowledgeCache(db_path=path, size_cap=100)


# ── fetch_searxng_raw ────────────────────────────────────────────────

class TestFetchSearxngRaw:
    @pytest.mark.asyncio
    async def test_success_formats_results(self):
        payload = {"results": [
            {"title": "Python docs", "content": "Python programming",
             "url": "https://python.org"},
            {"title": "Real Python", "content": "Tutorial site",
             "url": "https://realpython.com"},
        ]}
        fake = _mk_response(200, payload, content=b"...")
        with _patch_httpx_get(fake):
            r = await fetch_searxng_raw("python", searxng_url="http://x/search")
        assert r.success
        assert r.backend == "searxng_raw"
        assert "Python docs" in r.raw_text
        assert "python.org" in r.raw_text
        assert "Real Python" in r.raw_text
        assert r.structured["results"][0]["title"] == "Python docs"

    @pytest.mark.asyncio
    async def test_empty_results(self):
        fake = _mk_response(200, {"results": []})
        with _patch_httpx_get(fake):
            r = await fetch_searxng_raw("xyz", searxng_url="http://x/search")
        assert not r.success
        assert r.error_type == "empty"

    @pytest.mark.asyncio
    async def test_max_results_caps(self):
        payload = {"results": [
            {"title": f"T{i}", "content": f"C{i}", "url": f"u{i}"}
            for i in range(20)
        ]}
        fake = _mk_response(200, payload)
        with _patch_httpx_get(fake):
            r = await fetch_searxng_raw("q", searxng_url="http://x",
                                        max_results=3)
        # Only 3 results formatted into raw_text
        assert r.raw_text.count("\n") <= 4  # 1 header + 3 lines
        assert "T0" in r.raw_text and "T1" in r.raw_text and "T2" in r.raw_text
        assert "T5" not in r.raw_text

    @pytest.mark.asyncio
    async def test_5xx_error(self):
        fake = _mk_response(503)
        with _patch_httpx_get(fake):
            r = await fetch_searxng_raw("q", searxng_url="http://x")
        assert r.error_type == "http_5xx"

    @pytest.mark.asyncio
    async def test_timeout(self):
        with _patch_httpx_get(httpx.TimeoutException("timeout")):
            r = await fetch_searxng_raw("q", searxng_url="http://x")
        assert r.error_type == "timeout"


# ── dispatch(raw_results=True) ───────────────────────────────────────

class TestRawDispatchMode:
    @pytest.mark.asyncio
    async def test_conceptual_uses_searxng_raw(self, cache):
        # Stub fetch_searxng_raw at the dispatcher's import location
        from titan_plugin.logic import knowledge_dispatcher as kd
        call_args = []

        async def fake_raw(topic, searxng_url="", max_results=5, timeout=10.0):
            call_args.append((topic, searxng_url))
            return BackendResult(
                backend="searxng_raw", query=topic, success=True,
                raw_text=f"Raw result for {topic}", bytes_consumed=50)

        original = kd.fetch_searxng_raw
        kd.fetch_searxng_raw = fake_raw
        try:
            out = await dispatch(
                "hypothesis generation critical thinking",
                cache=cache, sage=None,
                raw_results=True,
                searxng_url="http://x/search",
            )
        finally:
            kd.fetch_searxng_raw = original

        assert out.success
        assert out.result.raw_text.startswith("Raw result for")
        assert call_args[0][1] == "http://x/search"
        # Backend slot records which named chain entry invoked raw path
        assert out.backend_used.startswith("searxng")

    @pytest.mark.asyncio
    async def test_raw_mode_without_url_skips_searxng(self, cache):
        # No searxng_url + no sage → all searxng_* chain entries skipped
        out = await dispatch(
            "hypothesis generation critical thinking",
            cache=cache, sage=None,
            raw_results=True,
            searxng_url="",
        )
        assert not out.success
        assert all(a[1] == "skipped" for a in out.attempts)

    @pytest.mark.asyncio
    async def test_dictionary_unaffected_by_raw_flag(
            self, cache, monkeypatch):
        # raw_results shouldn't change behaviour for direct-REST backends
        from titan_plugin.logic import knowledge_dispatcher as kd
        hits = []

        async def mock_wikt(topic, timeout=10.0):
            hits.append(topic)
            return BackendResult(
                backend="wiktionary", query=topic, success=True,
                raw_text="def", bytes_consumed=10)

        monkeypatch.setitem(kd.BACKEND_REGISTRY, "wiktionary", mock_wikt)
        out = await dispatch("hypothesis", cache=cache,
                             raw_results=True,
                             searxng_url="http://x")
        assert out.success
        assert out.backend_used == "wiktionary"
        assert hits == ["hypothesis"]


# ── WebSearchHelper ──────────────────────────────────────────────────

class TestWebSearchHelper:
    @pytest.mark.asyncio
    async def test_success_returns_legacy_shape(self, monkeypatch):
        from titan_plugin.logic import knowledge_dispatcher as kd

        async def fake_raw(topic, searxng_url="", max_results=5, timeout=10.0):
            return BackendResult(
                backend="searxng_raw", query=topic, success=True,
                raw_text=f"Found 1 result for '{topic}':\n- Title: snippet (url)",
                bytes_consumed=30)

        monkeypatch.setattr(kd, "fetch_searxng_raw", fake_raw)

        with tempfile.TemporaryDirectory() as tmp:
            helper = WebSearchHelper(
                searxng_url="http://localhost:8080/search",
                cache_db_path=os.path.join(tmp, "cache.db"))
            out = await helper.execute({
                "query": "hypothesis generation critical thinking"})

        assert out["success"] is True
        assert out["error"] is None
        assert "Title" in out["result"]
        assert out["enrichment_data"]["mind"] == [0]
        assert out["enrichment_data"]["backend"].startswith("searxng")
        assert out["enrichment_data"]["query_type"] == "conceptual"
        assert out["enrichment_data"]["cache_hit"] is False

    @pytest.mark.asyncio
    async def test_empty_query_rejected(self):
        helper = WebSearchHelper()
        out = await helper.execute({"query": ""})
        assert out["success"] is False
        assert "No query" in out["error"]

    @pytest.mark.asyncio
    async def test_internal_name_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            helper = WebSearchHelper(
                cache_db_path=os.path.join(tmp, "cache.db"))
            out = await helper.execute({"query": "inner_spirit"})
        assert out["success"] is False
        assert "rejected" in out["error"].lower() or "internal" in out["error"].lower()

    @pytest.mark.asyncio
    async def test_cache_hit_reflected_in_enrichment(self, monkeypatch):
        from titan_plugin.logic import knowledge_dispatcher as kd

        call_count = 0

        async def fake_raw(topic, searxng_url="", max_results=5, timeout=10.0):
            nonlocal call_count
            call_count += 1
            return BackendResult(
                backend="searxng_raw", query=topic, success=True,
                raw_text="cached content", bytes_consumed=20)

        monkeypatch.setattr(kd, "fetch_searxng_raw", fake_raw)

        with tempfile.TemporaryDirectory() as tmp:
            helper = WebSearchHelper(
                cache_db_path=os.path.join(tmp, "cache.db"))
            out1 = await helper.execute({"query": "hypothesis generation critical"})
            out2 = await helper.execute({"query": "hypothesis generation critical"})

        assert out1["enrichment_data"]["cache_hit"] is False
        assert out2["enrichment_data"]["cache_hit"] is True
        # Second call served from cache → only 1 backend hit
        assert call_count == 1

    def test_status_without_url(self):
        helper = WebSearchHelper(searxng_url="")
        assert helper.status() == "unavailable"

    def test_status_with_url(self):
        helper = WebSearchHelper(searxng_url="http://x/search")
        assert helper.status() == "available"

    def test_agno_tool_properties_preserved(self):
        helper = WebSearchHelper()
        assert helper.name == "web_search"
        assert "search" in helper.capabilities
        assert helper.enriches == ["mind"]
        assert helper.requires_sandbox is False
