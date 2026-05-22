"""Tests for D-SPEC-88 HTTP proxy endpoints (Chunk ω).

Covers the handler logic of /v4/llm-distill + /v4/llm-score:
  • Pydantic request validation
  • no_proxy / ok / timeout / error response shapes
  • Forwarding of optional model + max_tokens + temperature overrides

We don't boot the full FastAPI app — these are direct function tests of the
endpoint handlers with a mocked `request.app.state.titan_hcl.llm_proxy`.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Optional
from unittest.mock import AsyncMock

import pytest

from titan_hcl.api.llm_proxy_endpoints import (
    DistillRequest,
    DistillResponse,
    ScoreRequest,
    ScoreResponse,
    llm_distill_endpoint,
    llm_score_endpoint,
)


def _make_request(llm_proxy: Optional[object]):
    """Build a minimal mock `request` carrying `app.state.titan_hcl.llm_proxy`."""
    plugin = SimpleNamespace(llm_proxy=llm_proxy) if llm_proxy is not None else None
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(titan_hcl=plugin)))


# ── Pydantic validation ────────────────────────────────────────────


class TestDistillRequestValidation:
    def test_minimum_valid_request(self):
        req = DistillRequest(text="hello", instruction="summarize")
        assert req.text == "hello"
        assert req.instruction == "summarize"
        assert req.model is None
        assert req.max_tokens is None
        assert req.temperature is None
        assert req.pre_hook is False
        assert req.post_hook is False
        assert req.consumer == "external"

    def test_full_request(self):
        req = DistillRequest(
            text="content", instruction="classify",
            model="gemma3:4b", max_tokens=200, temperature=0.5,
            pre_hook=True, post_hook=True,
            channel="agent", timeout_s=10.0,
            consumer="events_teacher",
        )
        assert req.model == "gemma3:4b"
        assert req.max_tokens == 200
        assert req.temperature == 0.5
        assert req.consumer == "events_teacher"

    def test_empty_text_rejected(self):
        with pytest.raises(Exception):
            DistillRequest(text="", instruction="x")

    def test_max_tokens_bounds(self):
        with pytest.raises(Exception):
            DistillRequest(text="x", max_tokens=0)
        with pytest.raises(Exception):
            DistillRequest(text="x", max_tokens=99999)

    def test_temperature_bounds(self):
        with pytest.raises(Exception):
            DistillRequest(text="x", temperature=-1.0)
        with pytest.raises(Exception):
            DistillRequest(text="x", temperature=2.5)

    def test_timeout_bounds(self):
        with pytest.raises(Exception):
            DistillRequest(text="x", timeout_s=0.5)
        with pytest.raises(Exception):
            DistillRequest(text="x", timeout_s=999.0)


# ── /v4/llm-distill handler ────────────────────────────────────────


class TestDistillEndpoint:
    @pytest.mark.asyncio
    async def test_no_plugin_returns_no_proxy(self):
        req = DistillRequest(text="hi", instruction="echo")
        request = _make_request(None)
        resp = await llm_distill_endpoint(req, request)
        assert resp.status == "no_proxy"
        assert "not available" in (resp.error or "")

    @pytest.mark.asyncio
    async def test_no_llm_proxy_attr_returns_no_proxy(self):
        plugin = SimpleNamespace()   # plugin object exists but has no llm_proxy
        request = SimpleNamespace(app=SimpleNamespace(
            state=SimpleNamespace(titan_hcl=plugin)))
        req = DistillRequest(text="hi")
        resp = await llm_distill_endpoint(req, request)
        assert resp.status == "no_proxy"

    @pytest.mark.asyncio
    async def test_ok_response(self):
        mock_proxy = SimpleNamespace(distill=AsyncMock(return_value="distilled output"))
        request = _make_request(mock_proxy)
        req = DistillRequest(text="raw items", instruction="distill")
        resp = await llm_distill_endpoint(req, request)
        assert resp.status == "ok"
        assert resp.text == "distilled output"
        assert resp.latency_ms > 0
        assert resp.consumer == "external"
        # Verify proxy was called with the right args
        mock_proxy.distill.assert_awaited_once()
        kwargs = mock_proxy.distill.await_args.kwargs
        assert kwargs["text"] == "raw items"
        assert kwargs["instruction"] == "distill"

    @pytest.mark.asyncio
    async def test_forwards_max_tokens_and_temperature(self):
        mock_proxy = SimpleNamespace(distill=AsyncMock(return_value="ok"))
        request = _make_request(mock_proxy)
        req = DistillRequest(
            text="t", instruction="i",
            max_tokens=800, temperature=0.3,
            model="deepseek-v3.1:671b",
        )
        await llm_distill_endpoint(req, request)
        kwargs = mock_proxy.distill.await_args.kwargs
        assert kwargs["max_tokens"] == 800
        assert kwargs["temperature"] == 0.3
        assert kwargs["model"] == "deepseek-v3.1:671b"

    @pytest.mark.asyncio
    async def test_empty_proxy_result_returns_timeout(self):
        mock_proxy = SimpleNamespace(distill=AsyncMock(return_value=""))
        request = _make_request(mock_proxy)
        req = DistillRequest(text="x", consumer="events_teacher")
        resp = await llm_distill_endpoint(req, request)
        assert resp.status == "timeout"
        assert resp.consumer == "events_teacher"
        assert resp.error and "empty result" in resp.error

    @pytest.mark.asyncio
    async def test_proxy_raises_returns_error(self):
        mock_proxy = SimpleNamespace(
            distill=AsyncMock(side_effect=RuntimeError("bus down")))
        request = _make_request(mock_proxy)
        req = DistillRequest(text="x")
        resp = await llm_distill_endpoint(req, request)
        assert resp.status == "error"
        assert "bus down" in (resp.error or "")

    @pytest.mark.asyncio
    async def test_forwards_pre_post_hooks(self):
        mock_proxy = SimpleNamespace(distill=AsyncMock(return_value="ok"))
        request = _make_request(mock_proxy)
        req = DistillRequest(
            text="x", pre_hook=True, post_hook=True, channel="ovg-test")
        await llm_distill_endpoint(req, request)
        kwargs = mock_proxy.distill.await_args.kwargs
        assert kwargs["pre_hook"] is True
        assert kwargs["post_hook"] is True
        assert kwargs["channel"] == "ovg-test"


# ── /v4/llm-score handler ──────────────────────────────────────────


class TestScoreEndpoint:
    @pytest.mark.asyncio
    async def test_score_no_plugin(self):
        req = ScoreRequest(prompt="rate 0.0-1.0: ...")
        request = _make_request(None)
        resp = await llm_score_endpoint(req, request)
        assert resp.status == "no_proxy"

    @pytest.mark.asyncio
    async def test_score_ok(self):
        mock_proxy = SimpleNamespace(score=AsyncMock(return_value="0.85"))
        request = _make_request(mock_proxy)
        req = ScoreRequest(prompt="rate this", consumer="persona_endurance")
        resp = await llm_score_endpoint(req, request)
        assert resp.status == "ok"
        assert resp.text == "0.85"
        assert resp.consumer == "persona_endurance"

    @pytest.mark.asyncio
    async def test_score_forwards_model(self):
        mock_proxy = SimpleNamespace(score=AsyncMock(return_value="yes"))
        request = _make_request(mock_proxy)
        req = ScoreRequest(prompt="?", model="gemma3:4b")
        await llm_score_endpoint(req, request)
        kwargs = mock_proxy.score.await_args.kwargs
        assert kwargs["model"] == "gemma3:4b"


# ── llm_proxy.distill signature regression (D-SPEC-88) ─────────────


class TestLLMProxyDistillSignature:
    """Ensure LLMProxy.distill still accepts max_tokens + temperature kwargs."""

    def test_signature_accepts_new_kwargs(self):
        import inspect
        from titan_hcl.proxies.llm_proxy import LLMProxy
        sig = inspect.signature(LLMProxy.distill)
        params = sig.parameters
        assert "max_tokens" in params
        assert "temperature" in params
        # both should be optional
        assert params["max_tokens"].default is None
        assert params["temperature"].default is None
