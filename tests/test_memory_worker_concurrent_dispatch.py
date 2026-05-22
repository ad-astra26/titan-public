"""
Unit tests for `titan_hcl.modules._memory_dispatch` — Phase A of
`PLAN_phase_c_memory_worker_concurrent_dispatch.md`.

Covers:
  * InFlightRegistry register/resolve/cancel/concurrent
  * ActionRouter classification
  * Concurrent reads don't serialize behind writes
  * Writer pool serializes writes
  * Meditation thread lazy-spawn + handler invocation
  * Clean shutdown
  * Unknown action returns error response without hanging caller
"""

from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import TimeoutError as FutureTimeoutError
from unittest.mock import MagicMock

import pytest

from titan_hcl import bus
from titan_hcl.modules._memory_dispatch import (
    READ_ACTIONS,
    SPECIAL_ACTIONS,
    WRITE_ACTIONS,
    ActionRouter,
    InFlightRegistry,
    WorkerContext,
    ensure_thread_loop,
)


# ── InFlightRegistry ───────────────────────────────────────────────────────


class TestInFlightRegistry:
    def test_register_returns_future_and_records(self):
        reg = InFlightRegistry()
        fut = reg.register("rid-abc")
        assert reg.in_flight_count() == 1
        assert not fut.done()

    def test_register_empty_rid_raises(self):
        reg = InFlightRegistry()
        with pytest.raises(ValueError):
            reg.register("")

    def test_register_duplicate_rid_raises(self):
        reg = InFlightRegistry()
        reg.register("rid-1")
        with pytest.raises(RuntimeError):
            reg.register("rid-1")

    def test_resolve_matching_rid_sets_future_result(self):
        reg = InFlightRegistry()
        fut = reg.register("rid-xyz")
        msg = {"type": bus.RESPONSE, "rid": "rid-xyz", "payload": {"ok": True}}
        assert reg.resolve(msg) is True
        assert fut.done()
        assert fut.result(timeout=0) == msg
        assert reg.in_flight_count() == 0

    def test_resolve_non_matching_rid_returns_false(self):
        reg = InFlightRegistry()
        reg.register("rid-1")
        msg = {"type": bus.RESPONSE, "rid": "rid-other", "payload": {}}
        assert reg.resolve(msg) is False
        assert reg.in_flight_count() == 1  # original entry retained

    def test_resolve_missing_rid_returns_false(self):
        reg = InFlightRegistry()
        msg = {"type": bus.RESPONSE, "payload": {}}
        assert reg.resolve(msg) is False

    def test_cancel_removes_entry(self):
        reg = InFlightRegistry()
        reg.register("rid-cancel")
        reg.cancel("rid-cancel")
        assert reg.in_flight_count() == 0
        # Subsequent resolve should be a no-op
        assert reg.resolve({"rid": "rid-cancel", "type": bus.RESPONSE}) is False

    def test_register_after_cancel_works(self):
        reg = InFlightRegistry()
        reg.register("rid-A")
        reg.cancel("rid-A")
        # Re-register same rid is fine after cancel
        fut2 = reg.register("rid-A")
        assert not fut2.done()

    def test_future_timeout_does_not_block_cancel(self):
        reg = InFlightRegistry()
        fut = reg.register("rid-T")
        with pytest.raises(FutureTimeoutError):
            fut.result(timeout=0.05)
        reg.cancel("rid-T")
        assert reg.in_flight_count() == 0

    def test_concurrent_register_and_resolve(self):
        """Stress test: many threads registering + resolving simultaneously."""
        reg = InFlightRegistry()
        N = 50
        results: dict = {}
        barrier = threading.Barrier(N + 1)

        def register_and_wait(i: int) -> None:
            rid = f"rid-{i}"
            fut = reg.register(rid)
            barrier.wait()  # release with resolver
            try:
                results[i] = fut.result(timeout=2.0)
            except FutureTimeoutError:
                results[i] = "TIMEOUT"

        threads = [threading.Thread(target=register_and_wait, args=(i,))
                   for i in range(N)]
        for t in threads:
            t.start()
        barrier.wait()  # all registered

        # Resolve in shuffled order
        import random
        order = list(range(N))
        random.shuffle(order)
        for i in order:
            reg.resolve({"type": bus.RESPONSE, "rid": f"rid-{i}",
                         "payload": {"i": i}})

        for t in threads:
            t.join(timeout=3.0)

        assert len(results) == N
        for i in range(N):
            assert results[i]["payload"]["i"] == i, f"thread {i} got wrong result"
        assert reg.in_flight_count() == 0


# ── ensure_thread_loop ─────────────────────────────────────────────────────


class TestEnsureThreadLoop:
    def test_returns_loop_for_calling_thread(self):
        results = {}

        def grab_loop(key: str) -> None:
            results[key] = ensure_thread_loop()

        t1 = threading.Thread(target=grab_loop, args=("a",))
        t2 = threading.Thread(target=grab_loop, args=("b",))
        t1.start(); t2.start()
        t1.join(); t2.join()

        # Each thread gets its OWN loop (different object identity)
        assert results["a"] is not results["b"]
        assert not results["a"].is_closed()
        assert not results["b"].is_closed()

    def test_same_thread_reuses_loop(self):
        results = {}

        def grab_twice() -> None:
            results["first"] = ensure_thread_loop()
            results["second"] = ensure_thread_loop()

        t = threading.Thread(target=grab_twice)
        t.start(); t.join()
        assert results["first"] is results["second"]


# ── ActionRouter classification ────────────────────────────────────────────


class TestActionRouterClassify:
    def test_read_actions_classified_as_read(self):
        for action in READ_ACTIONS:
            assert ActionRouter.classify(action) == "read", action

    def test_write_actions_classified_as_write(self):
        for action in WRITE_ACTIONS:
            assert ActionRouter.classify(action) == "write", action

    def test_special_actions_classified_as_meditation(self):
        for action in SPECIAL_ACTIONS:
            assert ActionRouter.classify(action) == "meditation", action

    def test_unknown_action_classified_as_unknown(self):
        assert ActionRouter.classify("nope") == "unknown"
        assert ActionRouter.classify("") == "unknown"


# ── ActionRouter dispatch ──────────────────────────────────────────────────


def _make_ctx(send_queue=None):
    """Build a WorkerContext with a real Queue (subprocess.Queue-compatible
    enough for tests since we only call put_nowait / get)."""
    if send_queue is None:
        send_queue = queue.Queue()
    return WorkerContext(
        memory=MagicMock(),
        send_queue=send_queue,
        name="memory",
        config={},
        in_flight=InFlightRegistry(),
        write_lock=threading.RLock(),
    )


def _make_query_msg(action: str, **payload_kwargs):
    return {
        "type": bus.QUERY,
        "src": "test-caller",
        "rid": "rid-test",
        "payload": {"action": action, **payload_kwargs},
    }


class TestActionRouterDispatch:
    def test_read_action_routes_to_read_pool(self):
        ctx = _make_ctx()
        called_on = {}
        done = threading.Event()

        def handle_query(msg, ctx):
            called_on["thread_name"] = threading.current_thread().name
            done.set()

        router = ActionRouter(
            ctx, handle_query=handle_query,
            handle_memory_add=MagicMock(), handle_mempool_add=MagicMock())
        try:
            router.dispatch(_make_query_msg("query", text="x"))
            assert done.wait(timeout=2.0)
            assert called_on["thread_name"].startswith("mem-read")
        finally:
            router.shutdown(timeout=2.0)

    def test_ingest_request_routes_to_writer_pool(self):
        """Phase B (D-SPEC-44): the bus.QUERY action='add' work-RPC was
        retired; writes now flow through bus.MEMORY_INGEST_REQUEST routed
        to the writer pool via msg_type dispatch."""
        ctx = _make_ctx()
        called_on = {}
        done = threading.Event()

        def handle_ingest(msg, ctx):
            called_on["thread_name"] = threading.current_thread().name
            done.set()

        router = ActionRouter(
            ctx, handle_query=MagicMock(),
            handle_memory_add=MagicMock(), handle_mempool_add=MagicMock(),
            handle_memory_ingest_request=handle_ingest)
        try:
            router.dispatch({
                "type": bus.MEMORY_INGEST_REQUEST, "src": "memory_proxy",
                "payload": {"request_id": "r1", "text": "x"},
            })
            assert done.wait(timeout=2.0)
            assert called_on["thread_name"].startswith("mem-write")
        finally:
            router.shutdown(timeout=2.0)

    def test_meditation_action_routes_to_meditation_thread(self):
        ctx = _make_ctx()
        called_on = {}
        done = threading.Event()

        def handle_query(msg, ctx):
            called_on["thread_name"] = threading.current_thread().name
            done.set()

        router = ActionRouter(
            ctx, handle_query=handle_query,
            handle_memory_add=MagicMock(), handle_mempool_add=MagicMock())
        try:
            router.dispatch(_make_query_msg("run_meditation"))
            assert done.wait(timeout=2.0)
            assert called_on["thread_name"] == "mem-meditation"
        finally:
            router.shutdown(timeout=2.0)

    def test_memory_add_event_routes_to_writer_pool(self):
        ctx = _make_ctx()
        called_on = {}
        done = threading.Event()

        def handle_add(msg, ctx):
            called_on["thread_name"] = threading.current_thread().name
            done.set()

        router = ActionRouter(
            ctx, handle_query=MagicMock(),
            handle_memory_add=handle_add, handle_mempool_add=MagicMock())
        try:
            router.dispatch({
                "type": bus.MEMORY_ADD, "src": "spirit",
                "payload": {"text": "hi"},
            })
            assert done.wait(timeout=2.0)
            assert called_on["thread_name"].startswith("mem-write")
        finally:
            router.shutdown(timeout=2.0)

    def test_mempool_add_event_routes_to_writer_pool(self):
        ctx = _make_ctx()
        called_on = {}
        done = threading.Event()

        def handle_mempool(msg, ctx):
            called_on["thread_name"] = threading.current_thread().name
            done.set()

        router = ActionRouter(
            ctx, handle_query=MagicMock(),
            handle_memory_add=MagicMock(), handle_mempool_add=handle_mempool)
        try:
            router.dispatch({
                "type": bus.MEMORY_MEMPOOL_ADD, "src": "agno_hooks",
                "payload": {"user_prompt": "x", "agent_response": "y"},
            })
            assert done.wait(timeout=2.0)
            assert called_on["thread_name"].startswith("mem-write")
        finally:
            router.shutdown(timeout=2.0)

    def test_unknown_action_sends_error_response(self):
        send_q: queue.Queue = queue.Queue()
        ctx = _make_ctx(send_queue=send_q)
        router = ActionRouter(
            ctx, handle_query=MagicMock(),
            handle_memory_add=MagicMock(), handle_mempool_add=MagicMock())
        try:
            router.dispatch(_make_query_msg("not_a_real_action"))
            resp = send_q.get(timeout=2.0)
            assert resp["type"] == bus.RESPONSE
            assert resp["dst"] == "test-caller"
            assert resp["rid"] == "rid-test"
            assert "unknown action" in resp["payload"]["error"]
        finally:
            router.shutdown(timeout=2.0)

    def test_handler_exception_sends_error_response(self):
        send_q: queue.Queue = queue.Queue()
        ctx = _make_ctx(send_queue=send_q)

        def boom(msg, ctx):
            raise ValueError("kaboom")

        router = ActionRouter(
            ctx, handle_query=boom,
            handle_memory_add=MagicMock(), handle_mempool_add=MagicMock())
        try:
            router.dispatch(_make_query_msg("query", text="x"))
            resp = send_q.get(timeout=2.0)
            assert resp["type"] == bus.RESPONSE
            assert "ValueError" in resp["payload"]["error"]
        finally:
            router.shutdown(timeout=2.0)


# ── Concurrent dispatch behavior ──────────────────────────────────────────


class TestConcurrentDispatch:
    def test_4_concurrent_reads_dont_serialize(self):
        """4 reads with 0.5s simulated FAISS latency should complete in <1.5s
        (not 4×0.5=2s). The read pool has 8 workers; 4 fit easily."""
        ctx = _make_ctx()
        latch = threading.Event()
        started = threading.Barrier(5)  # 4 readers + main

        def slow_query(msg, ctx):
            started.wait(timeout=2.0)
            time.sleep(0.5)

        router = ActionRouter(
            ctx, handle_query=slow_query,
            handle_memory_add=MagicMock(), handle_mempool_add=MagicMock())
        try:
            t_start = time.time()
            for i in range(4):
                router.dispatch(_make_query_msg("query", text=f"q{i}"))
            # Sync all readers to start together
            started.wait(timeout=2.0)
            # Wait for them all to finish
            # ThreadPoolExecutor doesn't expose "all done" without futures;
            # poll for completion by submitting a barrier check
            elapsed_at = []
            def mark_done(msg, ctx):
                elapsed_at.append(time.time() - t_start)
                latch.set()
            # Submit a fifth read; it should also complete fast
            router._handle_query = mark_done  # rebind for the fifth msg
            router.dispatch(_make_query_msg("query", text="fifth"))
            assert latch.wait(timeout=3.0)
            # All 4 originals + 5th finished; check time
            elapsed = time.time() - t_start
            # 4 ran concurrently (~0.5s) + 5th (mostly immediate)
            # Generous upper bound: <1.5s
            assert elapsed < 1.5, f"elapsed={elapsed:.2f}s — reads serialized?"
        finally:
            router.shutdown(timeout=2.0)

    def test_write_does_not_block_reads(self):
        """A 1s simulated write must not block concurrent quick reads.

        Phase B (D-SPEC-44): writes go through MEMORY_INGEST_REQUEST → writer
        pool; reads stay on the read pool via bus.QUERY action='query'."""
        ctx = _make_ctx()
        read_done = threading.Event()
        write_started = threading.Event()
        write_done = threading.Event()

        def handle_query(msg, ctx):
            # Reads only — writes no longer flow through this handler.
            assert write_started.wait(timeout=1.0)
            time.sleep(0.05)
            read_done.set()

        def handle_ingest(msg, ctx):
            write_started.set()
            time.sleep(1.0)
            write_done.set()

        router = ActionRouter(
            ctx, handle_query=handle_query,
            handle_memory_add=MagicMock(), handle_mempool_add=MagicMock(),
            handle_memory_ingest_request=handle_ingest)
        try:
            t_start = time.time()
            router.dispatch({
                "type": bus.MEMORY_INGEST_REQUEST, "src": "memory_proxy",
                "payload": {"request_id": "r1", "text": "x"},
            })
            assert write_started.wait(timeout=1.0)
            router.dispatch(_make_query_msg("query", text="q"))
            assert read_done.wait(timeout=2.0)
            elapsed_read = time.time() - t_start
            assert not write_done.is_set(), \
                "write finished before read — test is racy or write isn't blocking"
            assert elapsed_read < 0.9, \
                f"read latency={elapsed_read:.2f}s — write blocked the read pool?"
            assert write_done.wait(timeout=2.0)
        finally:
            router.shutdown(timeout=3.0)

    def test_writer_pool_serializes_writes(self):
        """3 writes back-to-back must execute in order (not concurrently).

        Phase B (D-SPEC-44): exercises MEMORY_INGEST_REQUEST → writer pool;
        max_workers=1 guarantees in-order execution."""
        ctx = _make_ctx()
        order: list[int] = []
        lock = threading.Lock()
        all_done = threading.Event()

        def handle_ingest(msg, ctx):
            i = msg["payload"]["i"]
            time.sleep(0.05)
            with lock:
                order.append(i)
                if len(order) == 3:
                    all_done.set()

        router = ActionRouter(
            ctx, handle_query=MagicMock(),
            handle_memory_add=MagicMock(), handle_mempool_add=MagicMock(),
            handle_memory_ingest_request=handle_ingest)
        try:
            for i in range(3):
                router.dispatch({
                    "type": bus.MEMORY_INGEST_REQUEST, "src": "memory_proxy",
                    "payload": {"request_id": f"r{i}", "text": f"w{i}", "i": i},
                })
            assert all_done.wait(timeout=3.0)
            # max_workers=1 + dispatch in order ⇒ exec in order
            assert order == [0, 1, 2], f"writes ran out of order: {order}"
        finally:
            router.shutdown(timeout=2.0)


# ── Meditation thread + InFlightRegistry round-trip ────────────────────────


class TestMeditationAnchorRoundTrip:
    """Simulates the meditation → ANCHOR_REQUEST → main-loop → resolve flow.

    The meditation handler runs on `mem-meditation`, registers a rid in the
    InFlightRegistry, "sends" an ANCHOR_REQUEST, and waits on the Future.
    A mock "main loop" thread receives the ANCHOR_REQUEST off send_queue,
    flips it to a RESPONSE, and calls `registry.resolve(...)` on it.
    """

    def test_round_trip_resolves_via_registry(self):
        send_q: queue.Queue = queue.Queue()
        registry = InFlightRegistry()
        ctx = WorkerContext(
            memory=MagicMock(), send_queue=send_q, name="memory",
            config={}, in_flight=registry, write_lock=threading.RLock())

        anchor_payload_received = {}
        anchor_response = {"tx_signature": "SIG_FAKE_12345", "error": None}

        def meditation_handler(msg, ctx):
            # Simulate _request_anchor_via_kernel
            import uuid as _uuid
            rid = _uuid.uuid4().hex
            fut = ctx.in_flight.register(rid)
            ctx.send_queue.put_nowait({
                "type": bus.ANCHOR_REQUEST, "src": ctx.name, "dst": "kernel",
                "rid": rid,
                "payload": {"state_root": "MERKLE_DEADBEEF", "promoted_count": 1},
            })
            try:
                reply = fut.result(timeout=2.0)
            except FutureTimeoutError:
                anchor_payload_received["error"] = "timeout"
                return
            anchor_payload_received.update(reply.get("payload", {}))

        router = ActionRouter(
            ctx, handle_query=meditation_handler,
            handle_memory_add=MagicMock(), handle_mempool_add=MagicMock())

        try:
            # Kick off meditation
            router.dispatch(_make_query_msg("run_meditation"))

            # Mock main loop: read ANCHOR_REQUEST off send_queue, build
            # a RESPONSE with matching rid, resolve via registry
            req = send_q.get(timeout=2.0)
            assert req["type"] == bus.ANCHOR_REQUEST
            assert req["rid"] is not None

            response = {
                "type": bus.RESPONSE,
                "src": "kernel",
                "dst": ctx.name,
                "rid": req["rid"],
                "payload": anchor_response,
            }
            assert registry.resolve(response) is True

            # Meditation thread should have unwound and recorded the result.
            # Give it a moment to finish.
            deadline = time.time() + 2.0
            while time.time() < deadline and "tx_signature" not in anchor_payload_received:
                time.sleep(0.01)

            assert anchor_payload_received["tx_signature"] == "SIG_FAKE_12345"
            assert registry.in_flight_count() == 0
        finally:
            router.shutdown(timeout=3.0)

    def test_round_trip_timeout_cancels_cleanly(self):
        """If the main loop never sends the RESPONSE, the meditation thread's
        Future times out and the registry purges the rid via cancel."""
        send_q: queue.Queue = queue.Queue()
        registry = InFlightRegistry()
        ctx = WorkerContext(
            memory=MagicMock(), send_queue=send_q, name="memory",
            config={}, in_flight=registry, write_lock=threading.RLock())

        timed_out = threading.Event()

        def meditation_handler(msg, ctx):
            import uuid as _uuid
            rid = _uuid.uuid4().hex
            fut = ctx.in_flight.register(rid)
            ctx.send_queue.put_nowait({
                "type": bus.ANCHOR_REQUEST, "src": ctx.name, "dst": "kernel",
                "rid": rid, "payload": {}})
            try:
                fut.result(timeout=0.2)
            except FutureTimeoutError:
                ctx.in_flight.cancel(rid)
                timed_out.set()

        router = ActionRouter(
            ctx, handle_query=meditation_handler,
            handle_memory_add=MagicMock(), handle_mempool_add=MagicMock())
        try:
            router.dispatch(_make_query_msg("run_meditation"))
            assert timed_out.wait(timeout=2.0)
            assert registry.in_flight_count() == 0
        finally:
            router.shutdown(timeout=2.0)


# ── Clean shutdown ─────────────────────────────────────────────────────────


class TestShutdown:
    def test_shutdown_joins_all_pools(self):
        ctx = _make_ctx()
        router = ActionRouter(
            ctx, handle_query=MagicMock(),
            handle_memory_add=MagicMock(), handle_mempool_add=MagicMock())
        # Force the meditation thread to spawn
        router._ensure_meditation_thread()
        assert router._meditation_thread is not None
        assert router._meditation_thread.is_alive()

        router.shutdown(timeout=3.0)

        assert not router._meditation_thread.is_alive()
        # Pool executors don't have a public is_shutdown attr in all py
        # versions, but submit should raise on a shutdown pool
        with pytest.raises(RuntimeError):
            router._read_pool.submit(lambda: None)
        with pytest.raises(RuntimeError):
            router._writer_pool.submit(lambda: None)

    def test_shutdown_idempotent(self):
        """Calling shutdown twice doesn't crash."""
        ctx = _make_ctx()
        router = ActionRouter(
            ctx, handle_query=MagicMock(),
            handle_memory_add=MagicMock(), handle_mempool_add=MagicMock())
        router.shutdown(timeout=2.0)
        router.shutdown(timeout=2.0)  # no crash
