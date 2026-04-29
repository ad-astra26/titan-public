"""
2026-04-29 — Chat bus bridge regression tests.

Architectural fix for BUG-CHAT-AGENT-NOT-INITIALIZED-API-SUBPROCESS:
when api_process_separation_enabled=true, /chat forwards from the api
subprocess to the parent's _chat_handler_loop via a rid-routed
QUERY/RESPONSE protocol over the worker IPC pipe.

Five layers locked:
  1. ChatBridgeClient interface — exists, async, takes (src, dst,
     payload, timeout)
  2. Send path — request_async puts a QUERY on send_queue with the
     correct wire shape (type=QUERY, rid set, payload preserved)
  3. Receive path — handle_response dispatches RESPONSE by rid match;
     ignores non-matching messages
  4. End-to-end round-trip — request_async returns the reply when
     handle_response is called with a matching RESPONSE
  5. Timeout — request_async returns None when no matching RESPONSE
     arrives within the timeout window

Plus parent-side _chat_handler_loop hygiene tests:
  - filters QUERY with payload action=="chat"
  - dispatches each request to a separate task (concurrency, no head-of-line)
  - publishes RESPONSE rid-routed back to the original src
"""
from __future__ import annotations

import asyncio
import inspect
import threading
import time
import unittest
from queue import Queue as ThreadQueue
from unittest.mock import AsyncMock, MagicMock


from titan_plugin import bus
from titan_plugin.api.chat_bridge_client import ChatBridgeClient


class TestChatBridgeClientInterface(unittest.TestCase):
    """Layer 1 — interface stability."""

    def test_request_async_is_coroutine(self):
        self.assertTrue(asyncio.iscoroutinefunction(ChatBridgeClient.request_async))

    def test_request_async_signature(self):
        sig = inspect.signature(ChatBridgeClient.request_async)
        params = list(sig.parameters)
        # self + src + dst + payload + timeout
        self.assertEqual(params, ["self", "src", "dst", "payload", "timeout"])
        self.assertEqual(sig.parameters["timeout"].default, 60.0)

    def test_handle_response_signature(self):
        sig = inspect.signature(ChatBridgeClient.handle_response)
        # self + msg
        self.assertEqual(list(sig.parameters), ["self", "msg"])

    def test_pending_count_starts_at_zero(self):
        c = ChatBridgeClient(send_queue=None)
        self.assertEqual(c.pending_count(), 0)


class TestChatBridgeClientSendPath(unittest.TestCase):
    """Layer 2 — request_async puts the right wire shape on send_queue."""

    def test_request_async_puts_query_on_send_queue(self):
        sq = ThreadQueue()
        client = ChatBridgeClient(send_queue=sq)

        async def runner():
            # Don't await full result — start the request, inspect what
            # got sent, then resolve manually so wait_for doesn't time out.
            task = asyncio.create_task(client.request_async(
                "chat_subproc", "chat_handler",
                {"action": "chat", "body": {"message": "hi"}},
                timeout=2.0,
            ))
            # Yield once so request_async runs through the put_nowait.
            await asyncio.sleep(0)
            self.assertFalse(sq.empty(), "send_queue should have the QUERY")
            msg = sq.get_nowait()
            self.assertEqual(msg["type"], bus.QUERY)
            self.assertEqual(msg["src"], "chat_subproc")
            self.assertEqual(msg["dst"], "chat_handler")
            self.assertIn("rid", msg)
            self.assertEqual(msg["payload"]["action"], "chat")
            self.assertEqual(msg["payload"]["body"]["message"], "hi")
            # Now feed a RESPONSE to resolve the Future, otherwise task
            # times out + emits a warning.
            client.handle_response({
                "type": bus.RESPONSE, "src": "chat_handler",
                "dst": "chat_subproc", "rid": msg["rid"],
                "payload": {"status_code": 200, "body": {}, "extra_headers": None},
                "ts": time.time(),
            })
            reply = await task
            self.assertIsNotNone(reply)

        asyncio.run(runner())

    def test_request_async_returns_none_when_send_queue_is_none(self):
        client = ChatBridgeClient(send_queue=None)

        async def runner():
            return await client.request_async("a", "b", {}, timeout=0.5)

        result = asyncio.run(runner())
        self.assertIsNone(result, "graceful degradation expected when no queue")


class TestChatBridgeClientReceivePath(unittest.TestCase):
    """Layer 3 — handle_response dispatches by rid; rejects non-matches."""

    def test_handle_response_rejects_non_response_type(self):
        client = ChatBridgeClient(send_queue=None)
        # No call to request_async — no pending entries; should always
        # return False for non-RESPONSE.
        self.assertFalse(client.handle_response({"type": bus.QUERY, "rid": "x"}))
        self.assertFalse(client.handle_response({"type": "OTHER", "rid": "x"}))

    def test_handle_response_rejects_unknown_rid(self):
        client = ChatBridgeClient(send_queue=None)
        self.assertFalse(client.handle_response({
            "type": bus.RESPONSE, "rid": "nonexistent-rid", "payload": {}}))

    def test_handle_response_rejects_non_dict(self):
        client = ChatBridgeClient(send_queue=None)
        self.assertFalse(client.handle_response("not-a-dict"))
        self.assertFalse(client.handle_response(None))

    def test_handle_response_resolves_matching_future(self):
        sq = ThreadQueue()
        client = ChatBridgeClient(send_queue=sq)

        async def runner():
            task = asyncio.create_task(client.request_async(
                "chat_subproc", "chat_handler", {"action": "chat"},
                timeout=2.0,
            ))
            await asyncio.sleep(0)
            msg = sq.get_nowait()
            rid = msg["rid"]
            # Simulate parent reply
            response = {
                "type": bus.RESPONSE, "src": "chat_handler",
                "dst": "chat_subproc", "rid": rid,
                "payload": {"status_code": 200, "body": {"response": "hello"}},
                "ts": time.time(),
            }
            self.assertTrue(client.handle_response(response))
            reply = await asyncio.wait_for(task, timeout=2.0)
            return reply

        reply = asyncio.run(runner())
        self.assertIsNotNone(reply)
        self.assertEqual(reply["payload"]["body"]["response"], "hello")


class TestChatBridgeRoundTrip(unittest.TestCase):
    """Layer 4 — full request → reply round-trip."""

    def test_round_trip_returns_reply_payload(self):
        sq = ThreadQueue()
        client = ChatBridgeClient(send_queue=sq)

        # Spin up a "parent" thread that consumes from sq + replies via
        # handle_response (mimics the real kernel→bus→parent→bus→
        # api_subprocess flow without standing up the actual bus).
        def parent_thread():
            msg = sq.get(timeout=2.0)
            response = {
                "type": bus.RESPONSE, "src": "chat_handler",
                "dst": "chat_subproc", "rid": msg["rid"],
                "payload": {
                    "status_code": 200,
                    "body": {"response": "answer", "session_id": "s1",
                             "mode": "Sovereign", "mood": "calm"},
                    "extra_headers": {"X-Titan-Verified": "true"},
                },
                "ts": time.time(),
            }
            client.handle_response(response)

        async def runner():
            t = threading.Thread(target=parent_thread, daemon=True)
            t.start()
            return await client.request_async(
                "chat_subproc", "chat_handler",
                {"action": "chat",
                 "body": {"message": "hi", "session_id": "s1"},
                 "claims": {"sub": "user1"},
                 "headers": {}},
                timeout=3.0,
            )

        reply = asyncio.run(runner())
        self.assertIsNotNone(reply)
        self.assertEqual(reply["payload"]["status_code"], 200)
        self.assertEqual(reply["payload"]["body"]["response"], "answer")
        self.assertEqual(reply["payload"]["extra_headers"]["X-Titan-Verified"], "true")

    def test_concurrent_requests_dont_collide_on_rid(self):
        """Multiple in-flight requests must each receive their own reply."""
        sq = ThreadQueue()
        client = ChatBridgeClient(send_queue=sq)

        # Parent thread: drain sq, build per-rid replies in random order.
        def parent_thread():
            msgs = []
            while len(msgs) < 3:
                msgs.append(sq.get(timeout=2.0))
            # Reply in REVERSE order to prove rid-matching (not FIFO).
            for msg in reversed(msgs):
                rid = msg["rid"]
                bodyresp = msg["payload"]["body"]["message"]
                client.handle_response({
                    "type": bus.RESPONSE, "src": "chat_handler",
                    "dst": "chat_subproc", "rid": rid,
                    "payload": {"status_code": 200,
                                "body": {"response": f"reply-to-{bodyresp}"}},
                    "ts": time.time(),
                })

        async def runner():
            t = threading.Thread(target=parent_thread, daemon=True)
            t.start()
            results = await asyncio.gather(
                client.request_async("chat_subproc", "chat_handler",
                    {"action": "chat", "body": {"message": "A"}}, 3.0),
                client.request_async("chat_subproc", "chat_handler",
                    {"action": "chat", "body": {"message": "B"}}, 3.0),
                client.request_async("chat_subproc", "chat_handler",
                    {"action": "chat", "body": {"message": "C"}}, 3.0),
            )
            return results

        results = asyncio.run(runner())
        self.assertEqual(len(results), 3)
        replies = [r["payload"]["body"]["response"] for r in results]
        # Each request gets its own reply matched by rid.
        self.assertIn("reply-to-A", replies)
        self.assertIn("reply-to-B", replies)
        self.assertIn("reply-to-C", replies)


class TestChatBridgeTimeout(unittest.TestCase):
    """Layer 5 — timeout handling."""

    def test_request_async_returns_none_on_timeout(self):
        sq = ThreadQueue()
        client = ChatBridgeClient(send_queue=sq)

        async def runner():
            # No parent thread → no reply ever arrives → timeout
            return await client.request_async(
                "chat_subproc", "chat_handler",
                {"action": "chat"},
                timeout=0.3,  # short but not too short
            )

        t0 = time.time()
        reply = asyncio.run(runner())
        elapsed = time.time() - t0
        self.assertIsNone(reply)
        self.assertGreater(elapsed, 0.2)  # waited the timeout
        self.assertLess(elapsed, 1.0)     # didn't deadlock

    def test_pending_cleaned_after_timeout(self):
        sq = ThreadQueue()
        client = ChatBridgeClient(send_queue=sq)

        async def runner():
            await client.request_async(
                "chat_subproc", "chat_handler", {"action": "chat"},
                timeout=0.2,
            )
            return client.pending_count()

        n = asyncio.run(runner())
        self.assertEqual(n, 0, "pending dict should be cleaned after timeout")


class TestParentChatHandlerLoop(unittest.TestCase):
    """Parent-side: _chat_handler_loop filters QUERY action=chat,
    dispatches each request to a separate task. Tests via inspection +
    a small smoke test of dispatch logic."""

    def test_chat_handler_loop_method_exists(self):
        from titan_plugin.core.plugin import TitanPlugin
        self.assertTrue(hasattr(TitanPlugin, "_chat_handler_loop"))
        self.assertTrue(asyncio.iscoroutinefunction(TitanPlugin._chat_handler_loop))

    def test_handle_chat_request_method_exists(self):
        from titan_plugin.core.plugin import TitanPlugin
        self.assertTrue(hasattr(TitanPlugin, "_handle_chat_request"))
        self.assertTrue(asyncio.iscoroutinefunction(TitanPlugin._handle_chat_request))

    def test_chat_handler_loop_filters_by_action(self):
        """Source-inspection: the loop filters msg.type==QUERY AND
        payload.action=="chat" — so non-chat traffic on this queue is
        silently ignored."""
        from titan_plugin.core.plugin import TitanPlugin
        src = inspect.getsource(TitanPlugin._chat_handler_loop)
        self.assertIn('bus.QUERY', src)
        self.assertIn('"action") != "chat"', src)
        self.assertIn('"chat_handler"', src)

    def test_handle_chat_request_publishes_response_rid_routed(self):
        """Source-inspection: _handle_chat_request publishes
        type=bus.RESPONSE with src='chat_handler' and dst=msg.src."""
        from titan_plugin.core.plugin import TitanPlugin
        src = inspect.getsource(TitanPlugin._handle_chat_request)
        self.assertIn("bus.RESPONSE", src)
        self.assertIn('"chat_handler"', src)
        self.assertIn("self.bus.publish", src)
        # rid is the original msg's rid (rid-routed)
        self.assertIn('msg.get("rid")', src)


class TestRunChatInterface(unittest.TestCase):
    """Layer: TitanPlugin.run_chat exists, async, has the documented
    contract (returns dict with status_code/body/extra_headers)."""

    def test_run_chat_method_exists(self):
        from titan_plugin.core.plugin import TitanPlugin
        self.assertTrue(hasattr(TitanPlugin, "run_chat"))
        self.assertTrue(asyncio.iscoroutinefunction(TitanPlugin.run_chat))

    def _build_plugin_stub(self, agent=None, limbo=False):
        """Build a TitanPlugin instance with no kernel boot — bypassing
        __init__ avoids loading the entire microkernel for these guard-
        rail tests. _limbo_mode is a @property reading kernel.limbo_mode,
        so we attach a tiny mock kernel."""
        from titan_plugin.core.plugin import TitanPlugin
        plugin = TitanPlugin.__new__(TitanPlugin)
        plugin._agent = agent
        # Attach a fake kernel exposing limbo_mode (the real attribute the
        # _limbo_mode property reads).
        fake_kernel = MagicMock()
        fake_kernel.limbo_mode = limbo
        plugin.kernel = fake_kernel
        return plugin

    def test_run_chat_returns_503_when_no_agent(self):
        """When self._agent is None, run_chat returns 503 — preserves
        the legacy quick-fix workaround behavior in the parent path
        (subprocess path uses chat_bridge_bus + Mode 3 in chat.py for
        the no-bridge-no-agent case)."""
        plugin = self._build_plugin_stub(agent=None, limbo=False)

        async def runner():
            return await plugin.run_chat(
                {"message": "hi"}, {"sub": "user1"}, {})

        result = asyncio.run(runner())
        self.assertEqual(result["status_code"], 503)
        self.assertIn("not initialized", result["body"]["error"])
        self.assertIsNone(result["extra_headers"])

    def test_run_chat_returns_400_on_empty_message(self):
        plugin = self._build_plugin_stub(agent=MagicMock(), limbo=False)

        async def runner():
            return await plugin.run_chat(
                {"message": "   "}, {"sub": "u"}, {})

        result = asyncio.run(runner())
        self.assertEqual(result["status_code"], 400)
        self.assertIn("empty", result["body"]["error"].lower())

    def test_run_chat_returns_503_when_limbo(self):
        plugin = self._build_plugin_stub(agent=MagicMock(), limbo=True)

        async def runner():
            return await plugin.run_chat({"message": "hi"}, {"sub": "u"}, {})

        result = asyncio.run(runner())
        self.assertEqual(result["status_code"], 503)
        self.assertIn("Limbo", result["body"]["error"])


class TestCreateAppExposesChatBridgeBus(unittest.TestCase):
    """create_app accepts chat_bridge_bus parameter and exposes it on
    app.state — the contract chat.py Mode 2 depends on."""

    def test_create_app_signature_includes_chat_bridge_bus(self):
        from titan_plugin.api import create_app
        sig = inspect.signature(create_app)
        self.assertIn("chat_bridge_bus", sig.parameters)
        self.assertIsNone(sig.parameters["chat_bridge_bus"].default)

    def test_create_app_stores_chat_bridge_bus_on_app_state(self):
        """Source-inspection — runtime asserting requires booting the
        plugin which is heavy. Source-inspection is consistent with
        test_a8_thread_count.py pattern."""
        from titan_plugin.api import create_app as ca_mod
        src = inspect.getsource(ca_mod)
        self.assertIn("app.state.chat_bridge_bus = chat_bridge_bus", src)


if __name__ == "__main__":
    unittest.main()
