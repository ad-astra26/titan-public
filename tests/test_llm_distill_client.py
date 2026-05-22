"""Tests for the shared /v4/llm-distill HTTP client helper (Chunk χ)."""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from titan_hcl.logic.llm_distill_client import (
    distill_via_http_sync,
    distill_via_http_async,
)


def _mock_httpx_response(status_payload: dict, status_code: int = 200):
    """Make a fake httpx Response with .json() returning the payload."""
    mock = MagicMock()
    mock.status_code = status_code
    mock.json = MagicMock(return_value=status_payload)
    return mock


# ── Sync ───────────────────────────────────────────────────────────


class TestDistillSync:
    def test_no_internal_key_returns_empty(self):
        result = distill_via_http_sync(
            text="hi", instruction="echo",
            api_base="http://127.0.0.1:7777", internal_key="")
        assert result == ""

    def test_ok_response_returns_text(self):
        fake_resp = _mock_httpx_response({"status": "ok", "text": "distilled"})
        with patch("httpx.post", return_value=fake_resp) as mock_post:
            result = distill_via_http_sync(
                text="raw", instruction="distill",
                api_base="http://127.0.0.1:7777", internal_key="key",
                model="gemma3:4b", max_tokens=100, temperature=0.3,
                consumer="test")
        assert result == "distilled"
        # Verify payload
        call = mock_post.call_args
        assert call.args[0] == "http://127.0.0.1:7777/v4/llm-distill"
        payload = call.kwargs["json"]
        assert payload["text"] == "raw"
        assert payload["instruction"] == "distill"
        assert payload["model"] == "gemma3:4b"
        assert payload["max_tokens"] == 100
        assert payload["temperature"] == 0.3
        assert payload["consumer"] == "test"
        # Auth header
        assert call.kwargs["headers"]["X-Titan-Internal-Key"] == "key"

    def test_error_status_returns_empty(self):
        fake_resp = _mock_httpx_response({"status": "timeout", "error": "x"})
        with patch("httpx.post", return_value=fake_resp):
            result = distill_via_http_sync(
                text="hi", instruction="echo",
                api_base="http://127.0.0.1:7777", internal_key="key")
        assert result == ""

    def test_exception_returns_empty(self):
        with patch("httpx.post", side_effect=ConnectionError("net down")):
            result = distill_via_http_sync(
                text="hi", instruction="echo",
                api_base="http://127.0.0.1:7777", internal_key="key")
        assert result == ""

    def test_strips_whitespace(self):
        fake_resp = _mock_httpx_response({"status": "ok", "text": "  hello  "})
        with patch("httpx.post", return_value=fake_resp):
            result = distill_via_http_sync(
                text="hi", instruction="echo",
                api_base="http://127.0.0.1:7777", internal_key="key")
        assert result == "hello"

    def test_optional_fields_omitted_when_none(self):
        fake_resp = _mock_httpx_response({"status": "ok", "text": "x"})
        with patch("httpx.post", return_value=fake_resp) as mock_post:
            distill_via_http_sync(
                text="hi", instruction="echo",
                api_base="http://127.0.0.1:7777", internal_key="key")
        payload = mock_post.call_args.kwargs["json"]
        assert "model" not in payload
        assert "max_tokens" not in payload
        assert "temperature" not in payload


# ── Async ──────────────────────────────────────────────────────────


class _FakeAsyncContext:
    """Async context manager wrapping an httpx.AsyncClient mock."""
    def __init__(self, client_mock):
        self._client = client_mock
    async def __aenter__(self):
        return self._client
    async def __aexit__(self, *a):
        return False


class TestDistillAsync:
    @pytest.mark.asyncio
    async def test_no_internal_key_returns_empty(self):
        result = await distill_via_http_async(
            text="hi", instruction="echo",
            api_base="http://127.0.0.1:7777", internal_key="")
        assert result == ""

    @pytest.mark.asyncio
    async def test_ok_response_returns_text(self):
        fake_resp = MagicMock()
        fake_resp.json = MagicMock(return_value={"status": "ok", "text": "abc"})
        client_mock = MagicMock()
        client_mock.post = AsyncMock(return_value=fake_resp)
        with patch("httpx.AsyncClient", return_value=_FakeAsyncContext(client_mock)):
            result = await distill_via_http_async(
                text="raw", instruction="distill",
                api_base="http://127.0.0.1:7777", internal_key="key",
                consumer="meditation_scoring")
        assert result == "abc"
        call = client_mock.post.await_args
        assert call.args[0] == "http://127.0.0.1:7777/v4/llm-distill"
        assert call.kwargs["json"]["consumer"] == "meditation_scoring"

    @pytest.mark.asyncio
    async def test_async_exception_returns_empty(self):
        with patch("httpx.AsyncClient", side_effect=RuntimeError("boom")):
            result = await distill_via_http_async(
                text="hi", instruction="echo",
                api_base="http://127.0.0.1:7777", internal_key="key")
        assert result == ""
