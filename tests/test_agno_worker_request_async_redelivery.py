"""Regression — `_WorkerBusClient.request_async` must NOT drop non-matching
messages popped from the shared reply_queue (which IS the worker's main
recv_queue in production).

BUG-AGNO-SILENT-HANG (2026-05-25): pre-fix `request_async` silently
dropped non-matching messages. This stole CHAT_REQUEST + other
load-bearing messages whenever OVG.verify_safety_async ran concurrent
with chat arrival → silent fleet-wide agno_worker degradation
(T1 + T3 chat capability lost until restart).

This test exercises the race directly with an in-process queue +
synthetic CHAT_REQUEST messages, verifying:
  1. matching reply IS returned to caller (existing happy path)
  2. non-matching message is RE-QUEUED, not dropped
  3. main-loop consumer (simulated) can still find it after the call
  4. multiple non-matching messages preserve pop-order on re-queue
  5. timeout path also re-queues (deadline reached, no match found)
"""
from __future__ import annotations

import asyncio
import queue
import time
import unittest
from typing import Any

from titan_hcl.modules.agno_worker import _build_worker_bus_client


def _make_client(name: str = "agno_worker"):
    """Construct a _WorkerBusClient.

    Uses `queue.Queue` (single-process, immediately consistent) instead
    of `multiprocessing.Queue` (feeder-thread, non-deterministic timing)
    so test assertions on queue state are reliable. The _WorkerBusClient
    only calls `.put_nowait()` + `.get(block, timeout)` on the queues,
    which both queue types support — production uses mp.Queue, tests
    use queue.Queue, the request_async semantics are identical.
    """
    send_q = queue.Queue()
    recv_q = queue.Queue()
    client = _build_worker_bus_client(send_q, recv_q, name)
    return client, send_q, recv_q


def _drain(q) -> list:
    """Pop everything from a queue (no block); preserves order.

    Adds a tiny `time.sleep()` before draining to let any pending
    finally-block re-queue operations land before we read.
    """
    time.sleep(0.05)
    out = []
    while True:
        try:
            out.append(q.get_nowait())
        except queue.Empty:
            return out


class TestRequestAsyncRedelivery(unittest.TestCase):

    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    def _run(self, coro):
        return self.loop.run_until_complete(coro)

    # ── Happy path (existing behavior preserved) ─────────────────

    def test_matching_reply_returned(self):
        """rid match returns the reply, leaves queue empty."""
        client, send_q, recv_q = _make_client()

        async def driver():
            # Spawn the request, then put the matching reply.
            task = asyncio.create_task(
                client.request_async(
                    src="ovg_proxy", dst="output_verifier",
                    payload={"action": "verify_safety"},
                    timeout=2.0, reply_queue=recv_q,
                )
            )
            # Drain the published QUERY to find the rid.
            await asyncio.sleep(0.1)
            published = _drain(send_q)
            self.assertEqual(len(published), 1)
            rid = published[0]["rid"]
            # Put matching reply.
            recv_q.put({"type": "QUERY_RESPONSE", "rid": rid, "payload": {"ok": True}})
            return await task

        reply = self._run(driver())
        self.assertIsNotNone(reply)
        self.assertEqual(reply["payload"], {"ok": True})
        # Queue empty after happy path.
        self.assertEqual(_drain(recv_q), [])

    # ── THE BUG: non-matching messages must NOT be dropped ───────

    def test_non_matching_message_is_requeued_not_dropped(self):
        """The smoking-gun test for BUG-AGNO-SILENT-HANG.

        A CHAT_REQUEST arrives while request_async is awaiting its OVG
        reply. Pre-fix: CHAT_REQUEST is popped + silently dropped → main
        loop never sees it → 90s AgnoBridge timeout. Post-fix: popped +
        re-queued → main loop can still process it after the call.
        """
        client, send_q, recv_q = _make_client()

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
                    timeout=1.5, reply_queue=recv_q,
                )
            )
            await asyncio.sleep(0.1)
            published = _drain(send_q)
            ovg_rid = published[0]["rid"]
            # CHAT_REQUEST arrives while request_async is polling.
            recv_q.put(chat_msg)
            await asyncio.sleep(0.2)
            # Then the matching OVG reply arrives.
            recv_q.put({"type": "QUERY_RESPONSE", "rid": ovg_rid,
                        "payload": {"ok": True}})
            return await task

        reply = self._run(driver())
        # The OVG reply IS returned (matching rid).
        self.assertIsNotNone(reply)
        self.assertEqual(reply["payload"], {"ok": True})
        # CHAT_REQUEST MUST be present in the queue after the call —
        # the main loop will pick it up on next recv_queue.get().
        remaining = _drain(recv_q)
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0]["type"], "CHAT_REQUEST")
        self.assertEqual(remaining[0]["rid"], "chat_rid_42")

    # ── Multiple non-matching messages preserve pop-order ────────

    def test_multiple_non_matching_messages_preserve_order(self):
        """Pop order = re-put order. Critical for chat-stream ordering
        and KERNEL_EPOCH_TICK monotonicity."""
        client, send_q, recv_q = _make_client()

        msgs = [
            {"type": "CHAT_REQUEST", "rid": "c1", "payload": {}},
            {"type": "KERNEL_EPOCH_TICK", "rid": "t1", "payload": {"epoch": 1}},
            {"type": "CHAT_REQUEST", "rid": "c2", "payload": {}},
        ]

        async def driver():
            task = asyncio.create_task(
                client.request_async(
                    src="ovg_proxy", dst="output_verifier",
                    payload={"action": "x"},
                    timeout=1.5, reply_queue=recv_q,
                )
            )
            await asyncio.sleep(0.1)
            published = _drain(send_q)
            ovg_rid = published[0]["rid"]
            # 3 non-matching messages arrive (in order).
            for m in msgs:
                recv_q.put(m)
            await asyncio.sleep(0.3)
            # Matching reply.
            recv_q.put({"type": "QUERY_RESPONSE", "rid": ovg_rid,
                        "payload": {"ok": True}})
            return await task

        reply = self._run(driver())
        self.assertIsNotNone(reply)
        remaining = _drain(recv_q)
        # All 3 non-matching messages present, in original order.
        self.assertEqual(len(remaining), 3)
        self.assertEqual([r["rid"] for r in remaining], ["c1", "t1", "c2"])

    # ── Timeout path also re-queues ──────────────────────────────

    def test_timeout_path_still_redelivers_non_matching(self):
        """When deadline expires (no matching reply), non-matching
        messages popped during the wait MUST still be re-queued."""
        client, send_q, recv_q = _make_client()

        async def driver():
            task = asyncio.create_task(
                client.request_async(
                    src="ovg_proxy", dst="output_verifier",
                    payload={"action": "x"},
                    timeout=0.5, reply_queue=recv_q,
                )
            )
            await asyncio.sleep(0.05)
            _drain(send_q)  # discard published QUERY
            # Put a non-matching message; never put a matching reply.
            recv_q.put({"type": "CHAT_REQUEST", "rid": "lost",
                        "payload": {"msg": "x"}})
            return await task

        reply = self._run(driver())
        # Timeout → None returned.
        self.assertIsNone(reply)
        # CHAT_REQUEST MUST be in the queue (not lost).
        remaining = _drain(recv_q)
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0]["type"], "CHAT_REQUEST")

    # ── Broadcast messages also re-queued (NOT dropped) ──────────

    def test_broadcast_messages_redelivered(self):
        """Broadcast events (NEUROMOD_STATS_UPDATED, etc.) are not
        load-bearing for chat but consuming them here would mask
        other bugs. Re-queue them too — they reach their normal
        subscriber via the main loop."""
        client, send_q, recv_q = _make_client()

        async def driver():
            task = asyncio.create_task(
                client.request_async(
                    src="ovg_proxy", dst="output_verifier",
                    payload={"action": "x"},
                    timeout=1.0, reply_queue=recv_q,
                )
            )
            await asyncio.sleep(0.05)
            published = _drain(send_q)
            ovg_rid = published[0]["rid"]
            for i in range(3):
                recv_q.put({
                    "type": "NEUROMOD_STATS_UPDATED",
                    "rid": None,  # broadcasts have no rid
                    "payload": {"epoch": i},
                })
            await asyncio.sleep(0.2)
            recv_q.put({"type": "QUERY_RESPONSE", "rid": ovg_rid,
                        "payload": {"ok": True}})
            return await task

        reply = self._run(driver())
        self.assertIsNotNone(reply)
        remaining = _drain(recv_q)
        self.assertEqual(len(remaining), 3)
        self.assertEqual(
            [r["type"] for r in remaining],
            ["NEUROMOD_STATS_UPDATED"] * 3)


if __name__ == "__main__":
    unittest.main()
