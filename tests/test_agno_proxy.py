"""
tests/test_agno_proxy.py — AgnoProxy unit tests (post D-SPEC-73).

After D-SPEC-73 (SPEC v1.18.0), AgnoProxy has TWO backends behind one
public surface:

  - bus_client=DivineBus   →  parent-process backend (publish + subscribe
                              reply_queue + poll for CHAT_RESPONSE by rid)
  - bridge=AgnoBridgeClient → api_subprocess backend (forwards to bridge)

The legacy `_ensure_reply_queue` + `bus.request_async(msg_type=...)` path
was deleted (no shim) because it 500'd fleet-wide on the api_subprocess
kernel_rpc boundary.

Tests cover:
  - Construction XOR validation (exactly one of bus_client / bridge)
  - bus backend: publish CHAT_REQUEST + drain CHAT_RESPONSE by rid
  - bus backend: timeout returns 504, no reply received
  - bus backend: error payload → 500 envelope
  - bridge backend: forwards to AgnoBridgeClient.chat
  - OVG headers extraction from response
  - get_stats() reports backend + counters

Heavyweight CHAT_STREAM bus-backend path is exercised end-to-end in
agno_proxy itself (parent-mode stream uses DivineBus.subscribe + publish
in-process — works in the parent context; deferred coverage to integration
tests since the production path is bridge-backend).
"""
from __future__ import annotations

import asyncio
import threading
import time
from queue import Queue
from typing import Optional

import pytest

from titan_hcl import bus
from titan_hcl.proxies.agno_proxy import (
    DEFAULT_REQUEST_TIMEOUT_S,
    AgnoProxy,
)


# ────────────────────────────────────────────────────────────────────────
# Fakes
# ────────────────────────────────────────────────────────────────────────

class _FakeBus:
    """Fake DivineBus exposing publish + subscribe matching the parent-
    process backend contract (publish CHAT_REQUEST → drain reply_queue
    for matching CHAT_RESPONSE by rid).
    """
    def __init__(self):
        self.published: list = []
        self._reply_queue: Queue = Queue()

    def publish(self, msg):
        self.published.append(msg)

    def subscribe(self, name: str, reply_only: bool = False) -> Queue:
        return self._reply_queue

    def inject_reply(self, rid: str, payload: dict,
                     msg_type: str = "CHAT_RESPONSE"):
        """Push a reply onto the queue keyed by rid."""
        self._reply_queue.put({
            "type": msg_type,
            "src": "agno_worker",
            "dst": "agno_proxy",
            "rid": rid,
            "payload": payload,
            "ts": time.time(),
        })


class _FakeBridge:
    """Fake AgnoBridgeClient — captures forwards + returns canned results."""
    def __init__(self):
        self.chat_calls: list = []
        self.stream_calls: list = []
        self._next_chat: dict = {"status_code": 200,
                                 "body": {"response": "fake reply"},
                                 "extra_headers": None}
        self._next_stream: list = []
        self.stats = {"bridge": "agno_fake", "requests": 0}

    async def chat(self, message, **kwargs):
        self.chat_calls.append({"message": message, **kwargs})
        return self._next_chat

    async def chat_stream(self, message, **kwargs):
        self.stream_calls.append({"message": message, **kwargs})
        for item in self._next_stream:
            yield item

    def get_stats(self):
        return dict(self.stats)


# ────────────────────────────────────────────────────────────────────────
# Construction — XOR validation
# ────────────────────────────────────────────────────────────────────────

class TestConstruction:

    def test_with_bus_client_only(self):
        p = AgnoProxy(bus_client=_FakeBus())
        assert p._bus is not None and p._bridge is None
        assert p._timeout == DEFAULT_REQUEST_TIMEOUT_S

    def test_with_bridge_only(self):
        p = AgnoProxy(bridge=_FakeBridge())
        assert p._bus is None and p._bridge is not None

    def test_with_both_raises(self):
        with pytest.raises(ValueError, match="EXACTLY ONE"):
            AgnoProxy(bus_client=_FakeBus(), bridge=_FakeBridge())

    def test_with_neither_raises(self):
        with pytest.raises(ValueError, match="EXACTLY ONE"):
            AgnoProxy()

    def test_custom_timeout(self):
        p = AgnoProxy(bus_client=_FakeBus(), request_timeout_s=42.0)
        assert p._timeout == 42.0

    def test_positional_bus_client_legacy_call(self):
        """Backwards-compat — bus_client is positional-allowed."""
        p = AgnoProxy(_FakeBus())
        assert p._bus is not None and p._bridge is None


# ────────────────────────────────────────────────────────────────────────
# Bridge backend — pure delegation
# ────────────────────────────────────────────────────────────────────────

class TestBridgeBackend:

    @pytest.mark.asyncio
    async def test_chat_forwards_to_bridge(self):
        fake = _FakeBridge()
        fake._next_chat = {"status_code": 200,
                           "body": {"response": "from-bridge"},
                           "extra_headers": None}
        proxy = AgnoProxy(bridge=fake)
        result = await proxy.chat("hi", user_id="alice", session_id="s1",
                                  channel="web", is_maker=False,
                                  claims_sub="sub-1")
        assert result["body"]["response"] == "from-bridge"
        assert len(fake.chat_calls) == 1
        call = fake.chat_calls[0]
        assert call["message"] == "hi"
        assert call["user_id"] == "alice"
        assert call["channel"] == "web"

    @pytest.mark.asyncio
    async def test_chat_stream_forwards_to_bridge(self):
        fake = _FakeBridge()
        fake._next_stream = [
            {"chunk": "A", "done": False},
            {"chunk": "B", "done": True,
             "ovg_headers": {"X-Titan-Verified": "true"}},
        ]
        proxy = AgnoProxy(bridge=fake)
        collected = []
        async for c in proxy.chat_stream("hi"):
            collected.append(c)
        assert len(collected) == 2
        assert collected[0]["chunk"] == "A"
        assert collected[1]["done"] is True
        assert collected[1]["ovg_headers"]["X-Titan-Verified"] == "true"

    @pytest.mark.asyncio
    async def test_chat_bridge_exception_returns_500_envelope(self):
        class BrokenBridge:
            async def chat(self, *a, **kw):
                raise RuntimeError("bridge broken")
            def get_stats(self): return {"bridge": "broken"}

        proxy = AgnoProxy(bridge=BrokenBridge())
        result = await proxy.chat("hi")
        assert result["status_code"] == 500
        assert "bridge broken" in result["body"]["detail"]

    def test_get_stats_with_bridge(self):
        fake = _FakeBridge()
        fake.stats = {"bridge": "agno", "requests": 5, "errors": 0}
        proxy = AgnoProxy(bridge=fake)
        stats = proxy.get_stats()
        assert stats["proxy"] == "agno_proxy"
        assert stats["backend"] == "bridge"
        assert stats["requests"] == 5


# ────────────────────────────────────────────────────────────────────────
# Bus backend — publish + subscribe rid round-trip
# ────────────────────────────────────────────────────────────────────────

class TestBusBackend:

    @pytest.mark.asyncio
    async def test_chat_publishes_chat_request_and_resolves_by_rid(self):
        bus_obj = _FakeBus()
        proxy = AgnoProxy(bus_client=bus_obj, request_timeout_s=2.0)

        # Spawn injection AFTER chat publishes (chat polls reply_queue with
        # 0.5s wait per iter; inject within first iter)
        async def inject_later():
            await asyncio.sleep(0.1)
            assert len(bus_obj.published) == 1
            published = bus_obj.published[0]
            assert published["type"] == bus.CHAT_REQUEST
            assert published["dst"] == "agno_worker"
            rid = published["rid"]
            bus_obj.inject_reply(rid, {
                "response": "parent-mode reply",
                "session_id": "s1",
                "mode": "chat",
                "mood": "calm",
            })

        sim = asyncio.create_task(inject_later())
        result = await proxy.chat("hello", session_id="s1")
        await sim
        assert result["status_code"] == 200
        assert result["body"]["response"] == "parent-mode reply"
        assert result["body"]["session_id"] == "s1"

    @pytest.mark.asyncio
    async def test_chat_timeout_returns_504(self):
        bus_obj = _FakeBus()
        proxy = AgnoProxy(bus_client=bus_obj, request_timeout_s=0.8)
        result = await proxy.chat("never answered")
        assert result["status_code"] == 504
        assert result["body"]["error"] == "agno_timeout"

    @pytest.mark.asyncio
    async def test_chat_worker_error_payload_500(self):
        bus_obj = _FakeBus()
        proxy = AgnoProxy(bus_client=bus_obj, request_timeout_s=2.0)

        async def inject_err():
            await asyncio.sleep(0.1)
            rid = bus_obj.published[0]["rid"]
            bus_obj.inject_reply(rid, {"error": "agent_arun_failed"})

        sim = asyncio.create_task(inject_err())
        result = await proxy.chat("buggy")
        await sim
        assert result["status_code"] == 500
        assert "agent_arun_failed" in result["body"]["detail"]

    @pytest.mark.asyncio
    async def test_chat_extracts_ovg_headers(self):
        bus_obj = _FakeBus()
        proxy = AgnoProxy(bus_client=bus_obj, request_timeout_s=2.0)

        async def inject_with_ovg():
            await asyncio.sleep(0.1)
            rid = bus_obj.published[0]["rid"]
            bus_obj.inject_reply(rid, {
                "response": "verified",
                "ovg_data": {
                    "verified": True,
                    "block_height": 999,
                    "merkle_root": "rootZ",
                    "signature": "sigZ",
                },
            })

        sim = asyncio.create_task(inject_with_ovg())
        result = await proxy.chat("verify me")
        await sim
        hdrs = result["extra_headers"]
        assert hdrs["X-Titan-Verified"] == "true"
        assert hdrs["X-Titan-Block-Height"] == "999"
        assert hdrs["X-Titan-Merkle-Root"] == "rootZ"
        assert hdrs["X-Titan-Signature"] == "sigZ"

    @pytest.mark.asyncio
    async def test_chat_skips_unmatched_rid_messages(self):
        """Reply for a different rid on the queue must not satisfy our call."""
        bus_obj = _FakeBus()
        proxy = AgnoProxy(bus_client=bus_obj, request_timeout_s=2.0)

        async def inject_mismatched_then_correct():
            await asyncio.sleep(0.1)
            real_rid = bus_obj.published[0]["rid"]
            # Wrong rid first
            bus_obj.inject_reply("other-rid",
                                 {"response": "ghost"})
            await asyncio.sleep(0.05)
            # Then correct rid
            bus_obj.inject_reply(real_rid, {"response": "correct"})

        sim = asyncio.create_task(inject_mismatched_then_correct())
        result = await proxy.chat("strict-rid")
        await sim
        assert result["body"]["response"] == "correct"

    def test_get_stats_with_bus(self):
        proxy = AgnoProxy(bus_client=_FakeBus())
        stats = proxy.get_stats()
        assert stats["proxy"] == "agno_proxy"
        assert stats["backend"] == "bus"
        assert stats["requests"] == 0
