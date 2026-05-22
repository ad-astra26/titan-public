"""
D-SPEC-73 (SPEC v1.18.0) — AgnoBridgeClient unit tests.

Covers:
  - sync chat: rid match round-trip, timeout → 504, send_queue full → 503
  - stream chat: chunk dispatch by rid, done sentinel, overflow handling
  - handle_response: rid mismatch returns False (falls through), type mismatch
    returns False, claimed messages return True
  - pending_count() + get_stats() diagnostics
"""
import asyncio
from queue import Queue

import pytest

from titan_hcl import bus
from titan_hcl.api.agno_bridge_client import (
    AgnoBridgeClient, _STREAM_OVERFLOW,
)


# ────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────

@pytest.fixture
def send_q():
    """Real mp.Queue-like for outbound capture."""
    return Queue(maxsize=128)


@pytest.fixture
def bridge(send_q):
    """Bridge with short timeout to keep tests fast."""
    return AgnoBridgeClient(send_queue=send_q, request_timeout_s=1.5,
                            name="api")


# ────────────────────────────────────────────────────────────────────────
# handle_response — claim semantics
# ────────────────────────────────────────────────────────────────────────

def test_handle_response_returns_false_on_type_mismatch(bridge):
    """Non-CHAT_RESPONSE / non-CHAT_STREAM_CHUNK types fall through."""
    assert bridge.handle_response({"type": bus.RESPONSE, "rid": "abc"}) is False
    assert bridge.handle_response({"type": bus.QUERY, "rid": "abc"}) is False
    assert bridge.handle_response({"type": "RANDOM", "rid": "abc"}) is False


def test_handle_response_returns_false_when_rid_not_pending(bridge):
    """CHAT_RESPONSE/STREAM_CHUNK with unknown rid → not claimed."""
    assert bridge.handle_response({"type": bus.CHAT_RESPONSE, "rid": "unknown"}) is False
    msg = {"type": bus.CHAT_STREAM_CHUNK, "rid": "unknown",
           "payload": {"request_id": "unknown"}}
    assert bridge.handle_response(msg) is False


def test_handle_response_rejects_non_dict(bridge):
    assert bridge.handle_response(None) is False
    assert bridge.handle_response("garbage") is False


# ────────────────────────────────────────────────────────────────────────
# Sync chat — rid round-trip
# ────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_chat_sync_roundtrip_resolves_via_rid(bridge, send_q):
    """End-to-end: chat() publishes CHAT_REQUEST + resolves on CHAT_RESPONSE."""
    # Capture the outbound msg in a side-thread to read the rid, then inject
    # a synthetic CHAT_RESPONSE.
    async def simulate_worker():
        # Wait until chat() has enqueued CHAT_REQUEST + registered the rid
        await asyncio.sleep(0.05)
        outbound = send_q.get_nowait()
        assert outbound["type"] == bus.CHAT_REQUEST
        rid = outbound["rid"]
        # Inject CHAT_RESPONSE on the listener-thread side
        bridge.handle_response({
            "type": bus.CHAT_RESPONSE,
            "rid": rid,
            "payload": {"response": "hello back",
                        "session_id": "default",
                        "mode": "chat",
                        "mood": "calm"},
        })

    sim = asyncio.create_task(simulate_worker())
    result = await bridge.chat("hello agno", user_id="alice",
                               session_id="default", channel="web")
    await sim

    assert result["status_code"] == 200
    assert result["body"]["response"] == "hello back"
    assert result["body"]["session_id"] == "default"
    assert result["body"]["mode"] == "chat"


@pytest.mark.asyncio
async def test_chat_sync_timeout_returns_504(bridge):
    """No CHAT_RESPONSE within timeout → 504 envelope, registry cleared."""
    result = await bridge.chat("never answered")
    assert result["status_code"] == 504
    assert result["body"]["error"] == "agno_timeout"
    assert bridge.pending_count() == 0


@pytest.mark.asyncio
async def test_chat_sync_envelope_on_error_payload(bridge, send_q):
    """agno_worker error payload → 500 envelope with details."""
    async def simulate_error_worker():
        await asyncio.sleep(0.05)
        outbound = send_q.get_nowait()
        bridge.handle_response({
            "type": bus.CHAT_RESPONSE,
            "rid": outbound["rid"],
            "payload": {"error": "agent_arun_failed"},
        })

    sim = asyncio.create_task(simulate_error_worker())
    result = await bridge.chat("buggy")
    await sim
    assert result["status_code"] == 500
    assert result["body"]["error"] == "agno_worker_error"
    assert "agent_arun_failed" in result["body"]["detail"]


@pytest.mark.asyncio
async def test_chat_sync_send_queue_full_returns_503():
    """Send queue rejection → 503 + registry cleared."""
    class _FullQueue:
        def put_nowait(self, *_, **__):
            raise RuntimeError("queue full")

    b = AgnoBridgeClient(send_queue=_FullQueue(), request_timeout_s=2.0)
    result = await b.chat("hello")
    assert result["status_code"] == 503
    assert result["body"]["error"] == "send_queue_full"
    assert b.pending_count() == 0


# ────────────────────────────────────────────────────────────────────────
# Sync chat — OVG headers
# ────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_chat_sync_extracts_ovg_headers(bridge, send_q):
    """CHAT_RESPONSE.payload.ovg_data → X-Titan-* extra_headers."""
    async def worker():
        await asyncio.sleep(0.05)
        out = send_q.get_nowait()
        bridge.handle_response({
            "type": bus.CHAT_RESPONSE,
            "rid": out["rid"],
            "payload": {
                "response": "ok",
                "ovg_data": {
                    "verified": True,
                    "block_height": 123456,
                    "merkle_root": "abc123",
                    "signature": "deadbeef",
                },
            },
        })

    sim = asyncio.create_task(worker())
    result = await bridge.chat("hi")
    await sim
    hdrs = result["extra_headers"]
    assert hdrs["X-Titan-Verified"] == "true"
    assert hdrs["X-Titan-Block-Height"] == "123456"
    assert hdrs["X-Titan-Merkle-Root"] == "abc123"
    assert hdrs["X-Titan-Signature"] == "deadbeef"


# ────────────────────────────────────────────────────────────────────────
# Stream chat — chunk dispatch + done
# ────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_chat_stream_drains_chunks_until_done(bridge, send_q):
    """Stream emits chunks by rid, terminates on done=True."""
    async def worker():
        await asyncio.sleep(0.05)
        outbound = send_q.get_nowait()
        assert outbound["type"] == bus.CHAT_STREAM_REQUEST
        rid = outbound["rid"]
        for piece in ["Hello", " ", "world"]:
            bridge.handle_response({
                "type": bus.CHAT_STREAM_CHUNK,
                "payload": {"request_id": rid, "chunk": piece, "done": False},
            })
            await asyncio.sleep(0.01)
        bridge.handle_response({
            "type": bus.CHAT_STREAM_CHUNK,
            "payload": {"request_id": rid, "chunk": "", "done": True,
                        "ovg_headers": {"X-Titan-Verified": "true"}},
        })

    sim = asyncio.create_task(worker())
    chunks: list[dict] = []
    async for payload in bridge.chat_stream("multi-token"):
        chunks.append(payload)
    await sim

    # 3 content chunks + 1 done frame
    assert len(chunks) == 4
    assert chunks[0]["chunk"] == "Hello"
    assert chunks[1]["chunk"] == " "
    assert chunks[2]["chunk"] == "world"
    assert chunks[3]["done"] is True
    assert chunks[3]["ovg_headers"]["X-Titan-Verified"] == "true"
    # Pending registry cleared
    assert bridge.pending_count() == 0


@pytest.mark.asyncio
async def test_chat_stream_timeout_yields_error_frame(bridge):
    """No chunks within timeout → terminal error frame, registry cleared."""
    chunks: list[dict] = []
    async for payload in bridge.chat_stream("never streamed"):
        chunks.append(payload)
    assert chunks[-1]["done"] is True
    assert chunks[-1]["error"] == "agno_stream_timeout"
    assert bridge.pending_count() == 0


@pytest.mark.asyncio
async def test_chat_stream_ignores_chunks_for_unknown_rid(bridge, send_q):
    """Chunks with rid mismatch fall through; ours still receives correctly."""
    async def worker():
        await asyncio.sleep(0.05)
        outbound = send_q.get_nowait()
        rid = outbound["rid"]
        # Inject for OTHER rid first — should not arrive on ours
        bridge.handle_response({
            "type": bus.CHAT_STREAM_CHUNK,
            "payload": {"request_id": "other-rid", "chunk": "ghost",
                        "done": False},
        })
        # Then real chunk
        bridge.handle_response({
            "type": bus.CHAT_STREAM_CHUNK,
            "payload": {"request_id": rid, "chunk": "real", "done": True},
        })

    sim = asyncio.create_task(worker())
    chunks: list[dict] = []
    async for payload in bridge.chat_stream("test"):
        chunks.append(payload)
    await sim
    # Only the real chunk
    assert len(chunks) == 1
    assert chunks[0]["chunk"] == "real"
    assert chunks[0]["done"] is True


# ────────────────────────────────────────────────────────────────────────
# Diagnostics
# ────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_stats_reflects_activity(bridge, send_q):
    """get_stats() counters increment correctly."""
    async def worker():
        await asyncio.sleep(0.05)
        outbound = send_q.get_nowait()
        bridge.handle_response({
            "type": bus.CHAT_RESPONSE,
            "rid": outbound["rid"],
            "payload": {"response": "ok"},
        })

    sim = asyncio.create_task(worker())
    await bridge.chat("hi")
    await sim

    stats = bridge.get_stats()
    assert stats["bridge"] == "agno"
    assert stats["requests"] == 1
    assert stats["streams"] == 0
    assert stats["errors"] == 0
    assert stats["pending_responses"] == 0
    assert stats["pending_streams"] == 0


def test_pending_count_empty_on_construct(bridge):
    assert bridge.pending_count() == 0
    assert bridge.get_stats()["requests"] == 0
