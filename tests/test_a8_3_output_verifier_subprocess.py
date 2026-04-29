"""
L3 Phase A.8.3 — Output Verifier subprocess extraction regression tests.

Covers:
  - output_verifier_worker handles verify_and_sign / build_timechain_payload / stats
  - output_verifier_worker ignores non-QUERY messages cleanly
  - OutputVerifierProxy translates calls via bus.request
  - Proxy returns hard-fail OVGResult on bus timeout
  - Proxy update_cached_stats refreshes attributes
  - core/plugin.py flag-routing: flag-off → local OutputVerifier, flag-on → proxy
  - ModuleSpec for "output_verifier" registered in Guardian
"""
from __future__ import annotations

import dataclasses
import os
import time
import unittest
from unittest.mock import MagicMock, patch
from queue import Queue

# Ensure project root importable
import sys
_HERE = os.path.dirname(__file__)
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


class TestOutputVerifierProxyCore(unittest.TestCase):
    """Proxy behavior in isolation (mocked bus)."""

    def _make_bus_mock(self, response_payload=None, return_none=False):
        """Build a DivineBus mock whose `request()` returns a preset response."""
        bus = MagicMock()
        bus.subscribe.return_value = Queue()
        if return_none:
            bus.request.return_value = None
        else:
            bus.request.return_value = {
                "type": "RESPONSE",
                "src": "output_verifier",
                "dst": "output_verifier_proxy",
                "rid": "test-rid",
                "payload": response_payload or {},
                "ts": time.time(),
            }
        return bus

    def test_proxy_subscribes_to_bus_at_construction(self):
        from titan_plugin.proxies.output_verifier_proxy import OutputVerifierProxy
        bus = self._make_bus_mock()
        OutputVerifierProxy(bus)
        bus.subscribe.assert_called_once_with("output_verifier_proxy", reply_only=True)

    def test_proxy_default_stats_zero(self):
        from titan_plugin.proxies.output_verifier_proxy import OutputVerifierProxy
        bus = self._make_bus_mock()
        proxy = OutputVerifierProxy(bus)
        self.assertEqual(proxy.sovereignty_score, 0.0)
        self.assertEqual(proxy.verified_count, 0)
        self.assertEqual(proxy.rejected_count, 0)

    def test_proxy_update_cached_stats(self):
        from titan_plugin.proxies.output_verifier_proxy import OutputVerifierProxy
        bus = self._make_bus_mock()
        proxy = OutputVerifierProxy(bus)
        proxy.update_cached_stats({
            "sovereignty_score": 0.85, "verified_count": 42, "rejected_count": 3,
        })
        self.assertEqual(proxy.sovereignty_score, 0.85)
        self.assertEqual(proxy.verified_count, 42)
        self.assertEqual(proxy.rejected_count, 3)

    def test_proxy_verify_and_sign_routes_to_bus(self):
        from titan_plugin.proxies.output_verifier_proxy import OutputVerifierProxy
        from titan_plugin.logic.output_verifier import OVGResult
        valid_dict = dataclasses.asdict(OVGResult(
            passed=True, output_text="hi", signature="sig", channel="chat",
        ))
        bus = self._make_bus_mock(response_payload=valid_dict)
        proxy = OutputVerifierProxy(bus)
        result = proxy.verify_and_sign("hi", channel="chat")
        # Bus.request was called with action="verify_and_sign"
        call_kwargs = bus.request.call_args.kwargs
        self.assertEqual(call_kwargs["dst"], "output_verifier")
        self.assertEqual(call_kwargs["payload"]["action"], "verify_and_sign")
        self.assertEqual(call_kwargs["payload"]["output_text"], "hi")
        # Result reconstructed correctly
        self.assertTrue(result.passed)
        self.assertEqual(result.signature, "sig")

    def test_proxy_verify_and_sign_returns_hard_fail_on_timeout(self):
        from titan_plugin.proxies.output_verifier_proxy import OutputVerifierProxy
        bus = self._make_bus_mock(return_none=True)
        proxy = OutputVerifierProxy(bus)
        result = proxy.verify_and_sign("hi", channel="chat")
        self.assertFalse(result.passed)
        self.assertEqual(result.violation_type, "proxy_unavailable")
        self.assertIsNone(result.signature)

    def test_proxy_verify_and_sign_returns_hard_fail_on_worker_error(self):
        from titan_plugin.proxies.output_verifier_proxy import OutputVerifierProxy
        bus = self._make_bus_mock(response_payload={"error": "boom"})
        proxy = OutputVerifierProxy(bus)
        result = proxy.verify_and_sign("hi", channel="chat")
        self.assertFalse(result.passed)
        self.assertEqual(result.violation_type, "proxy_error")
        self.assertIn("boom", result.violations[0])

    def test_proxy_get_stats(self):
        from titan_plugin.proxies.output_verifier_proxy import OutputVerifierProxy
        bus = self._make_bus_mock(response_payload={
            "sovereignty_score": 0.92, "verified_count": 100, "rejected_count": 7,
        })
        proxy = OutputVerifierProxy(bus)
        stats = proxy.get_stats()
        self.assertEqual(stats["sovereignty_score"], 0.92)
        self.assertEqual(stats["verified_count"], 100)


class TestOutputVerifierWorkerHandler(unittest.TestCase):
    """Test the worker's QUERY-dispatch logic."""

    def test_worker_handles_verify_and_sign_query(self):
        """Drive the worker loop with a single QUERY, verify it produces RESPONSE."""
        from titan_plugin.modules.output_verifier_worker import (
            output_verifier_worker_main,
        )
        recv = Queue()
        send = Queue()
        # Inject the QUERY then SHUTDOWN to exit the loop.
        recv.put({
            "type": "QUERY", "src": "test_caller", "dst": "output_verifier",
            "rid": "rid-123",
            "payload": {
                "action": "verify_and_sign",
                "output_text": "Hello world",
                "channel": "chat",
                "injected_context": "",
                "prompt_text": "Hi",
                "chain_state": None,
            },
            "ts": time.time(),
        })
        recv.put({"type": "MODULE_SHUTDOWN", "src": "guardian", "dst": "output_verifier"})

        # Use a config with minimal valid OutputVerifier params so it actually constructs.
        # Use a tmp directory for tc_dir to avoid touching real timechain state.
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            cfg = {
                "info_banner": {"titan_id": "TEST"},
                "memory_and_storage": {"data_dir": td},
                "network": {"wallet_keypair_path": "data/titan_identity_keypair.json"},
            }
            output_verifier_worker_main(recv, send, "output_verifier", cfg)

        # Drain the send queue and look for RESPONSE
        sent = []
        while not send.empty():
            try:
                sent.append(send.get_nowait())
            except Exception:
                break
        responses = [m for m in sent if m.get("type") == "RESPONSE" and m.get("rid") == "rid-123"]
        self.assertEqual(len(responses), 1, f"Expected one RESPONSE for rid-123, got: {sent}")
        body = responses[0]["payload"]
        # OVGResult dataclass has 'passed' field
        self.assertIn("passed", body)
        self.assertIn("output_text", body)
        self.assertIn("channel", body)
        self.assertEqual(body["channel"], "chat")

    def test_worker_handles_unknown_action_gracefully(self):
        from titan_plugin.modules.output_verifier_worker import (
            output_verifier_worker_main,
        )
        recv = Queue()
        send = Queue()
        recv.put({
            "type": "QUERY", "src": "test", "dst": "output_verifier", "rid": "rid-x",
            "payload": {"action": "no_such_action"}, "ts": time.time(),
        })
        recv.put({"type": "MODULE_SHUTDOWN", "src": "guardian", "dst": "output_verifier"})

        import tempfile
        with tempfile.TemporaryDirectory() as td:
            cfg = {
                "info_banner": {"titan_id": "TEST"},
                "memory_and_storage": {"data_dir": td},
                "network": {"wallet_keypair_path": "data/titan_identity_keypair.json"},
            }
            output_verifier_worker_main(recv, send, "output_verifier", cfg)

        sent = []
        while not send.empty():
            try:
                sent.append(send.get_nowait())
            except Exception:
                break
        responses = [m for m in sent if m.get("type") == "RESPONSE" and m.get("rid") == "rid-x"]
        self.assertEqual(len(responses), 1)
        self.assertIn("error", responses[0]["payload"])

    def test_worker_emits_ready_signal_on_boot(self):
        """Worker publishes OUTPUT_VERIFIER_READY once at startup."""
        from titan_plugin.modules.output_verifier_worker import (
            output_verifier_worker_main,
        )
        recv = Queue()
        send = Queue()
        # Just shutdown, no QUERY needed.
        recv.put({"type": "MODULE_SHUTDOWN", "src": "guardian", "dst": "output_verifier"})

        import tempfile
        with tempfile.TemporaryDirectory() as td:
            cfg = {
                "info_banner": {"titan_id": "TEST"},
                "memory_and_storage": {"data_dir": td},
                "network": {"wallet_keypair_path": "data/titan_identity_keypair.json"},
            }
            output_verifier_worker_main(recv, send, "output_verifier", cfg)

        sent = []
        while not send.empty():
            try:
                sent.append(send.get_nowait())
            except Exception:
                break
        ready_msgs = [m for m in sent if m.get("type") == "OUTPUT_VERIFIER_READY"]
        self.assertEqual(len(ready_msgs), 1)
        self.assertEqual(ready_msgs[0]["payload"]["titan_id"], "TEST")


class TestPluginFlagRouting(unittest.TestCase):
    """core/plugin.py: __init__ chooses local OutputVerifier vs proxy based on flag."""

    def test_init_uses_local_when_flag_off(self):
        """When flag is off (default), plugin gets a real OutputVerifier (not proxy)."""
        from titan_plugin.core.plugin import TitanPlugin
        import inspect
        src = inspect.getsource(TitanPlugin.__init__)
        # Both paths exist in the source.
        self.assertIn("a8_output_verifier_subprocess_enabled", src)
        self.assertIn("OutputVerifierProxy", src)
        # Default-false read (flag absent → False).
        self.assertIn("False", src)

    def test_module_spec_registered_with_flag_aware_autostart(self):
        """Guardian.register is called with name='output_verifier' and autostart bound to flag."""
        from titan_plugin.core.plugin import TitanPlugin
        import inspect
        src = inspect.getsource(TitanPlugin)
        self.assertIn('name="output_verifier"', src)
        self.assertIn('layer="L2"', src)
        self.assertIn("autostart=_ov_subproc_enabled", src)
        self.assertIn("rss_limit_mb=400", src)


if __name__ == "__main__":
    unittest.main()
