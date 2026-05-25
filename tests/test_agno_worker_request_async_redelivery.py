"""Regression — `_WorkerBusClient` dispatcher routes messages safely.

BUG-AGNO-SILENT-HANG (2026-05-25, D-SPEC-128):
Pre-fix, `_WorkerBusClient.request_async` raced with the worker's main
loop on the shared `recv_queue`. Non-matching messages popped during
`request_async` were silently dropped → stole CHAT_REQUEST + other
load-bearing messages during OVG.verify_safety_async calls →
fleet-wide silent agno_worker degradation (T1 + T3 chat capability
lost until restart).

Spec-correct fix: a dispatcher thread inside `_WorkerBusClient`
exclusively reads `recv_queue` and routes:
  - rid-matching messages → resolve the request_async's Future
  - everything else → consumer_queue (read by the worker main loop)

Tests:
  1. matching reply resolves the request_async Future
  2. CHAT_REQUEST arriving during request_async lands in consumer_queue
     (NOT lost, NOT delivered to request_async)
  3. multiple non-matching messages arrive in original order
  4. timeout path returns None + leaves consumer_queue intact
  5. broadcast events (no rid) land in consumer_queue
  6. request_async cleanup removes its rid entry from the waiter dict
  7. dispatcher stops cleanly on .stop()
"""
from __future__ import annotations

import asyncio
import queue
import time
import unittest

from titan_hcl.modules.agno_worker import _build_worker_bus_client


def _make_client(name: str = "agno_worker"):
    """Construct a _WorkerBusClient.

    Uses `queue.Queue` (single-process, immediately consistent) instead
    of `multiprocessing.Queue` (feeder-thread, non-deterministic
    timing) so test assertions on queue state are reliable. The
    dispatcher only calls `.get(timeout=...)` and `.put_nowait()` on
    the queues; both queue types support these.
    """
    send_q = queue.Queue()
    recv_q = queue.Queue()
    client = _build_worker_bus_client(send_q, recv_q, name)
    return client, send_q, recv_q


def _drain(q) -> list:
    """Pop everything from a queue (no block); preserves order.

    Sleep first to let the dispatcher loop flush any pending routes.
    """
    time.sleep(0.3)  # dispatcher poll = 0.2s, give a margin
    out = []
    while True:
        try:
            out.append(q.get_nowait())
        except queue.Empty:
            return out


class TestRequestAsyncDispatcher(unittest.TestCase):

    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.clients_to_stop = []

    def tearDown(self):
        for client in self.clients_to_stop:
            try:
                client.stop()
            except Exception:
                pass
        self.loop.close()

    def _run(self, coro):
        return self.loop.run_until_complete(coro)

    def _make(self, name="agno_worker"):
        client, send_q, recv_q = _make_client(name)
        self.clients_to_stop.append(client)
        return client, send_q, recv_q

    # ── Happy path ────────────────────────────────────────────────

    def test_matching_reply_resolves_future(self):
        """rid match flows through the dispatcher → request_async Future."""
        client, send_q, recv_q = self._make()

        async def driver():
            task = asyncio.create_task(
                client.request_async(
                    src="ovg_proxy", dst="output_verifier",
                    payload={"action": "verify_safety"},
                    timeout=2.0,
                )
            )
            await asyncio.sleep(0.1)
            # Find the published rid.
            published = []
            while True:
                try:
                    published.append(send_q.get_nowait())
                except queue.Empty:
                    break
            self.assertEqual(len(published), 1)
            rid = published[0]["rid"]
            # Put matching reply on raw recv_queue — dispatcher routes
            # it to the future.
            recv_q.put({"type": "QUERY_RESPONSE", "rid": rid,
                        "payload": {"ok": True}})
            return await task

        reply = self._run(driver())
        self.assertIsNotNone(reply)
        self.assertEqual(reply["payload"], {"ok": True})
        # Matching reply went to future; nothing in consumer_queue.
        self.assertEqual(_drain(client.consumer_queue), [])

    # ── THE BUG: CHAT_REQUEST during request_async lands in consumer ──

    def test_chat_request_during_request_async_routed_to_consumer(self):
        """The smoking-gun fix for BUG-AGNO-SILENT-HANG.

        A CHAT_REQUEST arriving while request_async is awaiting its OVG
        reply must land in consumer_queue (where the main loop reads
        it), NOT be delivered to request_async (rid mismatch) NOR
        silently dropped.
        """
        client, send_q, recv_q = self._make()

        chat_msg = {
            "type": "CHAT_REQUEST",
            "rid": "chat_rid_42",
            "payload": {"message": "hello", "session_id": "s1"},
            "src": "api", "dst": "agno_worker",
        }

        async def driver():
            task = asyncio.create_task(
                client.request_async(
                    src="ovg_proxy", dst="output_verifier",
                    payload={"action": "verify_safety"},
                    timeout=2.0,
                )
            )
            await asyncio.sleep(0.1)
            published = []
            while True:
                try:
                    published.append(send_q.get_nowait())
                except queue.Empty:
                    break
            ovg_rid = published[0]["rid"]
            # CHAT_REQUEST arrives → dispatcher routes to consumer_queue
            # (no rid match for the OVG waiter).
            recv_q.put(chat_msg)
            await asyncio.sleep(0.3)  # let dispatcher process
            # Then the matching OVG reply.
            recv_q.put({"type": "QUERY_RESPONSE", "rid": ovg_rid,
                        "payload": {"ok": True}})
            return await task

        reply = self._run(driver())
        self.assertIsNotNone(reply)
        self.assertEqual(reply["payload"], {"ok": True})
        # CHAT_REQUEST MUST be in consumer_queue (main loop reads from it).
        consumer = _drain(client.consumer_queue)
        self.assertEqual(len(consumer), 1)
        self.assertEqual(consumer[0]["type"], "CHAT_REQUEST")
        self.assertEqual(consumer[0]["rid"], "chat_rid_42")

    # ── Multiple non-matching messages preserve order ─────────────

    def test_multiple_non_matching_messages_preserve_order(self):
        """Dispatcher is single-threaded → consumer_queue gets messages
        in raw_recv arrival order."""
        client, send_q, recv_q = self._make()

        msgs = [
            {"type": "CHAT_REQUEST", "rid": "c1", "payload": {}},
            {"type": "KERNEL_EPOCH_TICK", "rid": None, "payload": {"epoch": 1}},
            {"type": "CHAT_REQUEST", "rid": "c2", "payload": {}},
        ]

        async def driver():
            task = asyncio.create_task(
                client.request_async(
                    src="ovg_proxy", dst="output_verifier",
                    payload={"action": "x"}, timeout=2.0,
                )
            )
            await asyncio.sleep(0.1)
            published = []
            while True:
                try:
                    published.append(send_q.get_nowait())
                except queue.Empty:
                    break
            ovg_rid = published[0]["rid"]
            for m in msgs:
                recv_q.put(m)
            await asyncio.sleep(0.4)
            recv_q.put({"type": "QUERY_RESPONSE", "rid": ovg_rid,
                        "payload": {"ok": True}})
            return await task

        reply = self._run(driver())
        self.assertIsNotNone(reply)
        consumer = _drain(client.consumer_queue)
        self.assertEqual(len(consumer), 3)
        self.assertEqual(
            [r.get("rid") or r["type"] for r in consumer],
            ["c1", "KERNEL_EPOCH_TICK", "c2"],
        )

    # ── Timeout path ──────────────────────────────────────────────

    def test_timeout_path_returns_none_and_keeps_consumer_intact(self):
        """When deadline expires with no matching reply, request_async
        returns None. Any non-matching messages that arrived during the
        wait remain in consumer_queue."""
        client, send_q, recv_q = self._make()

        async def driver():
            task = asyncio.create_task(
                client.request_async(
                    src="ovg_proxy", dst="output_verifier",
                    payload={"action": "x"}, timeout=0.5,
                )
            )
            await asyncio.sleep(0.05)
            # Discard published QUERY.
            while True:
                try:
                    send_q.get_nowait()
                except queue.Empty:
                    break
            # Non-matching message arrives during the wait.
            recv_q.put({"type": "CHAT_REQUEST", "rid": "lost?",
                        "payload": {}})
            return await task

        reply = self._run(driver())
        self.assertIsNone(reply)
        consumer = _drain(client.consumer_queue)
        self.assertEqual(len(consumer), 1)
        self.assertEqual(consumer[0]["type"], "CHAT_REQUEST")

    # ── Broadcasts (no rid) → consumer_queue ──────────────────────

    def test_broadcast_messages_routed_to_consumer(self):
        """No-rid messages always route to consumer_queue, regardless
        of any in-flight request_async."""
        client, send_q, recv_q = self._make()

        async def driver():
            task = asyncio.create_task(
                client.request_async(
                    src="ovg_proxy", dst="output_verifier",
                    payload={"action": "x"}, timeout=1.5,
                )
            )
            await asyncio.sleep(0.05)
            published = []
            while True:
                try:
                    published.append(send_q.get_nowait())
                except queue.Empty:
                    break
            ovg_rid = published[0]["rid"]
            for i in range(3):
                recv_q.put({"type": "NEUROMOD_STATS_UPDATED",
                            "rid": None, "payload": {"epoch": i}})
            await asyncio.sleep(0.3)
            recv_q.put({"type": "QUERY_RESPONSE", "rid": ovg_rid,
                        "payload": {"ok": True}})
            return await task

        reply = self._run(driver())
        self.assertIsNotNone(reply)
        consumer = _drain(client.consumer_queue)
        self.assertEqual(len(consumer), 3)
        self.assertEqual(
            [r["type"] for r in consumer],
            ["NEUROMOD_STATS_UPDATED"] * 3)

    # ── Waiter cleanup ────────────────────────────────────────────

    def test_completed_request_async_unregisters_waiter(self):
        """After request_async returns (success or timeout), its rid
        entry is removed from `_rid_waiters` — no leak."""
        client, send_q, recv_q = self._make()

        async def driver():
            task = asyncio.create_task(
                client.request_async(
                    src="ovg_proxy", dst="output_verifier",
                    payload={"action": "x"}, timeout=1.0,
                )
            )
            await asyncio.sleep(0.1)
            published = []
            while True:
                try:
                    published.append(send_q.get_nowait())
                except queue.Empty:
                    break
            ovg_rid = published[0]["rid"]
            recv_q.put({"type": "QUERY_RESPONSE", "rid": ovg_rid,
                        "payload": {"ok": True}})
            await task

        self._run(driver())
        # Internal contract: registry empty after successful reply.
        self.assertEqual(client._rid_waiters, {})

    def test_timeout_unregisters_waiter(self):
        """Timeout path also cleans up its rid waiter."""
        client, _send_q, _recv_q = self._make()

        async def driver():
            await client.request_async(
                src="ovg_proxy", dst="output_verifier",
                payload={"action": "x"}, timeout=0.2,
            )

        self._run(driver())
        self.assertEqual(client._rid_waiters, {})

    # ── Dispatcher stop ───────────────────────────────────────────

    def test_dispatcher_stop_exits_cleanly(self):
        """client.stop() signals the dispatcher to exit. Subsequent
        messages may not be dispatched, but no exception thrown."""
        client, _send_q, recv_q = self._make()
        client.stop()
        # Give the dispatcher one poll cycle to notice the stop.
        time.sleep(0.3)
        # Dispatcher thread should be done (or about to be).
        self.assertFalse(client._dispatcher.is_alive() and not client._stop.is_set())

    # ── Reply_queue kwarg back-compat ─────────────────────────────

    def test_reply_queue_kwarg_ignored_safely(self):
        """Old callers that pass `reply_queue=some_queue` should still
        work — the parameter is preserved for back-compat but ignored
        (dispatcher routes by rid via futures, not via the passed-in
        queue)."""
        client, send_q, recv_q = self._make()
        # A random queue the caller might pass — should be untouched.
        bogus_q: queue.Queue = queue.Queue()

        async def driver():
            task = asyncio.create_task(
                client.request_async(
                    src="ovg_proxy", dst="output_verifier",
                    payload={"action": "x"},
                    timeout=1.0,
                    reply_queue=bogus_q,  # legacy positional / kwarg
                )
            )
            await asyncio.sleep(0.1)
            published = []
            while True:
                try:
                    published.append(send_q.get_nowait())
                except queue.Empty:
                    break
            ovg_rid = published[0]["rid"]
            # Reply goes to RAW recv (dispatcher's input), not bogus_q.
            recv_q.put({"type": "QUERY_RESPONSE", "rid": ovg_rid,
                        "payload": {"ok": True}})
            return await task

        reply = self._run(driver())
        self.assertIsNotNone(reply)
        # bogus_q untouched.
        self.assertTrue(bogus_q.empty())


if __name__ == "__main__":
    unittest.main()
