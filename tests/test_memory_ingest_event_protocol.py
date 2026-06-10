"""Tests for Phase B (rFP §3.4.1) MEMORY_INGEST_REQUEST/COMPLETED event
protocol replacing the work-RPC `add` action.

Covers:
  * memory_proxy.add_memory publishes MEMORY_INGEST_REQUEST one-way
    (no work-RPC, no rid, no waiter).
  * memory_proxy.add_memory_with_completion registers a Future for the
    request_id BEFORE publishing, then resolves on matching
    MEMORY_INGEST_COMPLETED broadcast.
  * memory_proxy.inject_memory backward-compat alias returns the
    full dict (node_id + weight + status + cognified).
  * memory_worker._handle_memory_ingest_request runs inject_memory under
    write_lock and emits MEMORY_INGEST_COMPLETED with matching request_id.
  * memory_worker handler skips COMPLETED broadcast when producer omits
    request_id (fire-and-forget producers like spirit_worker).
  * Timeout cancels the Future + cleans up the registry.
  * Concurrent ingest requests don't interleave their Futures.
"""
from __future__ import annotations

import asyncio
import threading
import time
from queue import Queue
from unittest.mock import MagicMock

import pytest

from titan_hcl import bus
from titan_hcl.modules._memory_dispatch import (
    InFlightRegistry,
    QueryCache,
    WorkerContext,
)
from titan_hcl.modules.memory_worker import _handle_memory_ingest_request
from titan_hcl.proxies.memory_proxy import (
    MemoryProxy,
    _IngestCompletionRegistry,
)


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def stub_bus():
    """In-process bus stub: subscribers stored in a dict, publish fans out
    to every subscriber whose `types` filter includes the msg type."""

    class _StubBus:
        def __init__(self):
            self._subs: dict[str, list[tuple[Queue, frozenset | None]]] = {}
            self.published: list[dict] = []

        def subscribe(self, name, reply_only=False, types=None):
            q: Queue = Queue()
            allowed = frozenset(types) if types is not None else None
            self._subs.setdefault(name, []).append((q, allowed))
            return q

        def publish(self, msg):
            self.published.append(msg)
            mt = msg.get("type", "")
            for queues in self._subs.values():
                for q, allowed in queues:
                    if allowed is None or mt in allowed:
                        q.put(msg)

        def request(self, *a, **kw):
            raise RuntimeError("test stub bus does not implement request")

        async def request_async(self, *a, **kw):
            raise RuntimeError("test stub bus does not implement request_async")

    return _StubBus()


@pytest.fixture
def stub_guardian():
    g = MagicMock()
    g.is_module_started = MagicMock(return_value=True)
    return g


@pytest.fixture
def proxy(stub_bus, stub_guardian, monkeypatch):
    # Avoid touching SHM / titan_id resolution + ensure_started kernel call.
    monkeypatch.setattr(
        "titan_hcl.proxies.memory_proxy.resolve_titan_id",
        lambda: "test_titan",
    )
    monkeypatch.setattr(
        "titan_hcl.proxies.memory_proxy.ensure_shm_root",
        lambda tid: __import__("pathlib").Path("/tmp/test_shm"),
    )
    p = MemoryProxy(stub_bus, stub_guardian)
    # Bypass the lazy module-start handshake — we don't care for protocol tests.
    p._ensure_started = lambda: None  # type: ignore
    return p


# ── add_memory (one-way publish) ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_add_memory_publishes_one_way_event(proxy, stub_bus):
    """add_memory should publish exactly one MEMORY_INGEST_REQUEST and return None."""
    result = await proxy.add_memory("hello world", source="test", weight=2.5)
    assert result is None
    # Find the published MEMORY_INGEST_REQUEST.
    requests = [m for m in stub_bus.published
                if m.get("type") == bus.MEMORY_INGEST_REQUEST]
    assert len(requests) == 1
    req = requests[0]
    assert req["dst"] == "memory"
    assert req["src"] == "memory_proxy"
    payload = req["payload"]
    assert payload["text"] == "hello world"
    assert payload["source"] == "test"
    assert payload["weight"] == 2.5
    assert "request_id" in payload
    assert len(payload["request_id"]) > 0


@pytest.mark.asyncio
async def test_add_memory_empty_text_skips_publish(proxy, stub_bus):
    result = await proxy.add_memory("")
    assert result is None
    assert not [m for m in stub_bus.published
                if m.get("type") == bus.MEMORY_INGEST_REQUEST]


# ── add_memory_with_completion (Future-bound round-trip) ────────────────────


@pytest.mark.asyncio
async def test_add_memory_with_completion_resolves_on_matching_broadcast(
        proxy, stub_bus):
    """Publish REQUEST, simulate worker COMPLETED broadcast, verify dict returned."""
    # Race-free pattern: register-before-publish is internal to the proxy.
    # We simulate the worker by intercepting the publish + broadcasting
    # COMPLETED with matching request_id from a helper thread.

    completed_result = {
        "node_id": 42, "weight": 5.25, "status": "persistent", "cognified": True,
    }

    def _worker_responder():
        # Wait for the REQUEST then publish a matching COMPLETED.
        for _ in range(100):  # 1s budget
            for m in list(stub_bus.published):
                if m.get("type") == bus.MEMORY_INGEST_REQUEST:
                    rid = m["payload"]["request_id"]
                    completed = {
                        "type": bus.MEMORY_INGEST_COMPLETED,
                        "src": "memory", "dst": "all",
                        "ts": time.time(), "rid": None,
                        "payload": {
                            "request_id": rid, "success": True,
                            "source": m["payload"].get("source"),
                            **completed_result,
                        },
                    }
                    stub_bus.publish(completed)
                    return
            time.sleep(0.01)

    t = threading.Thread(target=_worker_responder, daemon=True)
    t.start()
    result = await proxy.add_memory_with_completion(
        "test text", source="api", weight=5.0, timeout=2.0)
    t.join(timeout=2.0)
    assert result["success"] is True
    assert result["node_id"] == 42
    assert result["weight"] == 5.25
    assert result["status"] == "persistent"
    assert result["cognified"] is True
    assert result["source"] == "api"


@pytest.mark.asyncio
async def test_add_memory_with_completion_times_out_on_no_response(proxy, stub_bus):
    with pytest.raises(asyncio.TimeoutError):
        await proxy.add_memory_with_completion("never-served", timeout=0.3)
    # Registry must be cleaned up (no orphans).
    assert proxy._ingest_completion.in_flight_count() == 0


@pytest.mark.asyncio
async def test_inject_memory_alias_returns_dict(proxy, stub_bus):
    """inject_memory alias must return the full dict shape (api/maker.py contract)."""
    def _worker_responder():
        for _ in range(100):
            for m in list(stub_bus.published):
                if m.get("type") == bus.MEMORY_INGEST_REQUEST:
                    rid = m["payload"]["request_id"]
                    stub_bus.publish({
                        "type": bus.MEMORY_INGEST_COMPLETED,
                        "src": "memory", "dst": "all",
                        "ts": time.time(), "rid": None,
                        "payload": {
                            "request_id": rid, "success": True,
                            "source": "maker",
                            "node_id": 7, "weight": 5.25,
                            "status": "persistent", "cognified": True,
                        },
                    })
                    return
            time.sleep(0.01)

    t = threading.Thread(target=_worker_responder, daemon=True)
    t.start()
    result = await proxy.inject_memory("text", source="maker", weight=5.0, timeout=2.0)
    t.join(timeout=2.0)
    # api/maker.py uses result["node_id"] AND result["weight"].
    assert result["node_id"] == 7
    assert result["weight"] == 5.25


# ── Concurrent requests ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_concurrent_requests_have_distinct_request_ids(proxy, stub_bus):
    """4 concurrent add_memory calls publish 4 distinct request_ids."""
    await asyncio.gather(*[
        proxy.add_memory(f"text {i}", source=f"src{i}") for i in range(4)
    ])
    requests = [m for m in stub_bus.published
                if m.get("type") == bus.MEMORY_INGEST_REQUEST]
    assert len(requests) == 4
    rids = [m["payload"]["request_id"] for m in requests]
    assert len(set(rids)) == 4


# ── Worker handler ──────────────────────────────────────────────────────────


def _make_worker_ctx(memory):
    send_queue: Queue = Queue()
    return WorkerContext(
        memory=memory,
        send_queue=send_queue,
        name="memory",
        config={},
        in_flight=InFlightRegistry(),
        write_lock=threading.RLock(),
        query_cache=QueryCache(maxsize=8, ttl_s=10.0),
    ), send_queue


def test_handle_memory_ingest_request_runs_inject_and_broadcasts_completed():
    captured = {}

    class _StubMemory:
        async def inject_memory(self, text, source="bus", weight=1.0,
                                neuromod_context=None):
            captured.update(
                text=text, source=source, weight=weight,
                neuromod_context=neuromod_context)
            return {"node_id": 99, "weight": weight + 0.25,
                    "status": "persistent", "cognified": True}

    ctx, send_queue = _make_worker_ctx(_StubMemory())
    msg = {
        "type": bus.MEMORY_INGEST_REQUEST,
        "src": "memory_proxy", "dst": "memory",
        "ts": time.time(), "rid": None,
        "payload": {
            "request_id": "rid-abc",
            "text": "hello",
            "source": "test",
            "weight": 2.0,
            "neuromod_context": {"dopamine": 0.8},
        },
    }
    _handle_memory_ingest_request(msg, ctx)
    # Must have called inject with right params.
    assert captured["text"] == "hello"
    assert captured["source"] == "test"
    assert captured["weight"] == 2.0
    assert captured["neuromod_context"] == {"dopamine": 0.8}
    # Must have published COMPLETED.
    out = send_queue.get_nowait()
    assert out["type"] == bus.MEMORY_INGEST_COMPLETED
    assert out["dst"] == "all"
    assert out["payload"]["request_id"] == "rid-abc"
    assert out["payload"]["success"] is True
    assert out["payload"]["node_id"] == 99
    assert out["payload"]["weight"] == 2.25


def test_handle_memory_ingest_request_no_request_id_skips_completed_broadcast():
    """Fire-and-forget producer (spirit_worker dream bridge) omits request_id —
    handler must not noise every subscriber with an unfilterable COMPLETED."""

    class _StubMemory:
        async def inject_memory(self, text, source="bus", weight=1.0,
                                neuromod_context=None):
            return {"node_id": 101, "weight": weight + 0.25,
                    "status": "persistent", "cognified": True}

    ctx, send_queue = _make_worker_ctx(_StubMemory())
    msg = {
        "type": bus.MEMORY_INGEST_REQUEST,
        "src": "spirit_worker", "dst": "memory",
        "ts": time.time(), "rid": None,
        "payload": {
            # NO request_id — pure fire-and-forget.
            "text": "dream-bridge insight",
            "source": "dream",
            "weight": 1.5,
        },
    }
    _handle_memory_ingest_request(msg, ctx)
    assert send_queue.empty()


def test_handle_memory_ingest_request_handler_error_broadcasts_failure():
    class _StubMemory:
        async def inject_memory(self, **kwargs):
            raise RuntimeError("simulated failure")

    ctx, send_queue = _make_worker_ctx(_StubMemory())
    msg = {
        "type": bus.MEMORY_INGEST_REQUEST,
        "src": "memory_proxy", "dst": "memory",
        "ts": time.time(), "rid": None,
        "payload": {
            "request_id": "rid-fail",
            "text": "will-error",
            "source": "test",
            "weight": 1.0,
        },
    }
    _handle_memory_ingest_request(msg, ctx)
    out = send_queue.get_nowait()
    assert out["type"] == bus.MEMORY_INGEST_COMPLETED
    assert out["payload"]["request_id"] == "rid-fail"
    assert out["payload"]["success"] is False
    assert "RuntimeError" in out["payload"]["error"]


def test_handle_memory_ingest_request_empty_text_with_request_id_broadcasts_failure():
    class _StubMemory:
        async def inject_memory(self, **kwargs):  # pragma: no cover
            pytest.fail("inject_memory should not be called for empty text")

    ctx, send_queue = _make_worker_ctx(_StubMemory())
    msg = {
        "type": bus.MEMORY_INGEST_REQUEST,
        "src": "memory_proxy", "dst": "memory",
        "ts": time.time(), "rid": None,
        "payload": {
            "request_id": "rid-empty",
            "text": "",
            "source": "test",
        },
    }
    _handle_memory_ingest_request(msg, ctx)
    out = send_queue.get_nowait()
    assert out["payload"]["success"] is False
    assert out["payload"]["error"] == "empty text"


# ── Registry isolation ──────────────────────────────────────────────────────


def test_ingest_completion_registry_cancel_releases_slot(stub_bus):
    reg = _IngestCompletionRegistry(stub_bus)
    reg.register("rid-1")
    assert reg.in_flight_count() == 1
    reg.cancel("rid-1")
    assert reg.in_flight_count() == 0
    # Re-registering same rid after cancel should succeed.
    reg.register("rid-1")
    assert reg.in_flight_count() == 1


def test_ingest_completion_registry_duplicate_rid_raises(stub_bus):
    reg = _IngestCompletionRegistry(stub_bus)
    reg.register("rid-dup")
    with pytest.raises(RuntimeError, match="already in-flight"):
        reg.register("rid-dup")


# ── add_to_mempool felt plumbing (Phase C / RFP_synthesis_engram_grounding §7.C) ──
# Regression guard for the proxy hop the offline tests originally missed (the
# felt crux failed live on T3: MemoryProxy.add_to_mempool rejected neuromod_context).


@pytest.mark.asyncio
async def test_add_to_mempool_forwards_neuromod_context_in_payload(proxy, stub_bus):
    felt = {"DA": 0.9, "NE": 0.8, "emotion": "wonder"}
    await proxy.add_to_mempool("u", "a", user_identifier="x", neuromod_context=felt)
    evts = [m for m in stub_bus.published if m.get("type") == bus.MEMORY_MEMPOOL_ADD]
    assert len(evts) == 1
    payload = evts[0]["payload"]
    assert payload["user_prompt"] == "u"
    assert payload["agent_response"] == "a"
    assert payload["user_identifier"] == "x"
    assert payload["neuromod_context"] == felt


@pytest.mark.asyncio
async def test_add_to_mempool_neuromod_context_defaults_none(proxy, stub_bus):
    await proxy.add_to_mempool("u", "a")
    evts = [m for m in stub_bus.published if m.get("type") == bus.MEMORY_MEMPOOL_ADD]
    assert len(evts) == 1
    assert evts[0]["payload"]["neuromod_context"] is None


def test_handle_mempool_add_forwards_neuromod_context_to_core():
    """memory_worker._handle_mempool_add forwards payload neuromod_context to core
    add_to_mempool (the worker hop of the felt chain)."""
    import threading
    from types import SimpleNamespace
    from titan_hcl.modules.memory_worker import _handle_mempool_add

    captured: dict = {}

    class _Mem:
        async def add_to_mempool(self, user_prompt, agent_response,
                                 user_identifier="Anonymous", neuromod_context=None):
            captured.update(neuromod_context=neuromod_context, up=user_prompt)

    ctx = SimpleNamespace(memory=_Mem(), write_lock=threading.Lock(),
                          name="memory", send_queue=None)
    felt = {"DA": 0.7}
    _handle_mempool_add(
        {"payload": {"user_prompt": "u", "agent_response": "a",
                     "user_identifier": "x", "neuromod_context": felt}}, ctx)
    assert captured["neuromod_context"] == felt
    assert captured["up"] == "u"
