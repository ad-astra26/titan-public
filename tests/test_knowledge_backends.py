"""Unit tests for titan_plugin.logic.knowledge_backends (KP-1).

Mocks httpx at the AsyncClient level so tests stay hermetic — no real
network, no DNS, no external state. Covers success + each documented
error branch (timeout, 429, 404 empty, 5xx, parse_error, network).
"""

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from titan_plugin.logic.knowledge_backends import (
    BACKEND_REGISTRY,
    BackendResult,
    fetch_free_dictionary,
    fetch_wikipedia_summary,
    fetch_wiktionary,
)


# ── Helpers ──────────────────────────────────────────────────────────

def _mk_response(status_code: int, json_data=None, content_bytes: bytes = b""):
    """Build a minimal fake httpx.Response-like mock."""
    resp = AsyncMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.content = content_bytes or (
        b'{}' if json_data is None else str(json_data).encode())
    resp.json = lambda: json_data if json_data is not None else {}
    return resp


def _patch_httpx_get(response_or_exc):
    """Patch AsyncClient so .get() returns/raises what we specify.

    Accepts a fake response OR an Exception to raise.
    """
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

    return patch.object(httpx, "AsyncClient", _FakeClient)


# ── Wiktionary ───────────────────────────────────────────────────────

class TestWiktionary:
    @pytest.mark.asyncio
    async def test_success_flattens_definitions(self):
        payload = {"en": [{
            "partOfSpeech": "noun",
            "definitions": [
                {"definition": "A musical pitch"},
                {"definition": "A Greek letter"},
            ],
        }, {
            "partOfSpeech": "verb",
            "definitions": [{"definition": "To pitch upwards"}],
        }]}
        fake = _mk_response(200, json_data=payload,
                            content_bytes=b'{"en":[]}')
        with _patch_httpx_get(fake):
            r = await fetch_wiktionary("chi")
        assert r.success
        assert r.backend == "wiktionary"
        assert "[noun] A musical pitch" in r.raw_text
        assert "[noun] A Greek letter" in r.raw_text
        assert "[verb] To pitch upwards" in r.raw_text
        assert r.structured == payload
        assert r.error_type == ""
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_404_is_empty(self):
        fake = _mk_response(404, json_data={}, content_bytes=b"{}")
        with _patch_httpx_get(fake):
            r = await fetch_wiktionary("nonexistent_word_xyz")
        assert not r.success
        assert r.error_type == "empty"

    @pytest.mark.asyncio
    async def test_429_is_rate_limit(self):
        fake = _mk_response(429, content_bytes=b"Too Many Requests")
        with _patch_httpx_get(fake):
            r = await fetch_wiktionary("own")
        assert r.error_type == "rate_limit"

    @pytest.mark.asyncio
    async def test_503_is_5xx(self):
        fake = _mk_response(503, content_bytes=b"Service Unavailable")
        with _patch_httpx_get(fake):
            r = await fetch_wiktionary("own")
        assert r.error_type == "http_5xx"

    @pytest.mark.asyncio
    async def test_timeout(self):
        with _patch_httpx_get(httpx.TimeoutException("timeout")):
            r = await fetch_wiktionary("own")
        assert r.error_type == "timeout"
        assert not r.success

    @pytest.mark.asyncio
    async def test_network_error(self):
        with _patch_httpx_get(httpx.ConnectError("refused")):
            r = await fetch_wiktionary("own")
        assert r.error_type == "network"

    @pytest.mark.asyncio
    async def test_no_english_entries_is_empty(self):
        fake = _mk_response(200, json_data={"la": [{"definitions": [
            {"definition": "Latin only"}]}]})
        with _patch_httpx_get(fake):
            r = await fetch_wiktionary("something")
        assert not r.success
        assert r.error_type == "empty"

    @pytest.mark.asyncio
    async def test_raw_text_truncated(self):
        # 60 definitions × 200 chars each = 12 000 chars → truncated at 5000
        huge_defs = [{"definition": "x" * 200} for _ in range(60)]
        payload = {"en": [{"partOfSpeech": "n", "definitions": huge_defs}]}
        fake = _mk_response(200, json_data=payload)
        with _patch_httpx_get(fake):
            r = await fetch_wiktionary("run")
        assert r.success
        assert len(r.raw_text) < 5100
        assert "truncated" in r.raw_text


# ── Free Dictionary ──────────────────────────────────────────────────

class TestFreeDictionary:
    @pytest.mark.asyncio
    async def test_success(self):
        payload = [{
            "word": "own",
            "meanings": [{
                "partOfSpeech": "verb",
                "definitions": [{"definition": "Have as property"}],
            }],
        }]
        fake = _mk_response(200, json_data=payload)
        with _patch_httpx_get(fake):
            r = await fetch_free_dictionary("own")
        assert r.success
        assert r.backend == "free_dictionary"
        assert "[verb] Have as property" in r.raw_text
        assert r.structured == {"entries": payload}

    @pytest.mark.asyncio
    async def test_404_is_empty(self):
        fake = _mk_response(404)
        with _patch_httpx_get(fake):
            r = await fetch_free_dictionary("notaword")
        assert r.error_type == "empty"

    @pytest.mark.asyncio
    async def test_payload_not_list_is_empty(self):
        # Sometimes returns dict for "word not found"
        fake = _mk_response(200, json_data={"title": "No Definitions"})
        with _patch_httpx_get(fake):
            r = await fetch_free_dictionary("foo")
        assert not r.success
        assert r.error_type == "empty"


# ── Wikipedia summary ────────────────────────────────────────────────

class TestWikipediaSummary:
    @pytest.mark.asyncio
    async def test_success(self):
        payload = {
            "title": "Mitochondrial biogenesis",
            "description": "Growth of mitochondria",
            "extract": "Mitochondrial biogenesis is the process by which cells "
                       "increase mitochondrial mass.",
            "content_urls": {"desktop": {"page": "https://..."}},
            "type": "standard",
        }
        fake = _mk_response(200, json_data=payload)
        with _patch_httpx_get(fake):
            r = await fetch_wikipedia_summary("Mitochondrial biogenesis")
        assert r.success
        assert r.backend == "wikipedia_direct"
        assert r.structured["title"] == "Mitochondrial biogenesis"
        assert r.structured["extract"].startswith("Mitochondrial")
        assert "Mitochondrial biogenesis" in r.raw_text

    @pytest.mark.asyncio
    async def test_empty_extract(self):
        fake = _mk_response(200, json_data={"title": "Foo", "extract": ""})
        with _patch_httpx_get(fake):
            r = await fetch_wikipedia_summary("Foo")
        assert r.error_type == "empty"

    @pytest.mark.asyncio
    async def test_404(self):
        fake = _mk_response(404)
        with _patch_httpx_get(fake):
            r = await fetch_wikipedia_summary("NonexistentTopicXYZ")
        assert r.error_type == "empty"

    @pytest.mark.asyncio
    async def test_parse_error(self):
        # JSON that breaks
        resp = AsyncMock(spec=httpx.Response)
        resp.status_code = 200
        resp.content = b"not json"
        def _bad():
            raise ValueError("bad json")
        resp.json = _bad
        with _patch_httpx_get(resp):
            r = await fetch_wikipedia_summary("test")
        assert r.error_type == "parse_error"


# ── Registry ─────────────────────────────────────────────────────────

class TestBackendRegistry:
    def test_three_backends_registered(self):
        assert set(BACKEND_REGISTRY.keys()) == {
            "wiktionary", "free_dictionary", "wikipedia_direct"}

    def test_values_are_callable(self):
        for name, fn in BACKEND_REGISTRY.items():
            assert callable(fn), f"{name} not callable"


# ── BackendResult shape ──────────────────────────────────────────────

class TestBackendResult:
    def test_defaults(self):
        r = BackendResult(backend="x", query="y")
        assert r.success is False
        assert r.raw_text == ""
        assert r.structured is None
        assert r.error_type == ""
        assert r.bytes_consumed == 0
        assert r.latency_ms == 0.0
        assert r.status_code == 0
