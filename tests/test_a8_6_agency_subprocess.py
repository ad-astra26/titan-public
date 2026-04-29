"""
L3 Phase A.8.6 — Agency subprocess extraction regression tests.

Covers:
  - agency_worker handles handle_intent / dispatch_from_nervous_signals /
    assess / agency_stats / assessment_stats QUERY actions
  - agency_worker emits MODULE_READY + AGENCY_READY on boot
  - agency_worker emits AGENCY_STATS + ASSESSMENT_STATS broadcasts
  - agency_worker ignores non-QUERY messages cleanly
  - AgencyProxy translates calls via async-friendly bus.request
  - AssessmentProxy translates calls via async-friendly bus.request
  - Both proxies return hard-fail (None / [] / neutral) on bus timeout
  - Proxy update_cached_stats refreshes attributes
  - core/plugin.py flag-routing: flag-off → local AgencyModule + SelfAssessment,
    flag-on → AgencyProxy + AssessmentProxy
  - ModuleSpec for "agency_worker" registered in Guardian with flag-aware autostart
  - Bus constants AGENCY_READY / AGENCY_STATS / ASSESSMENT_STATS exist
  - _RegistryFacade satisfies _agency._registry.list_helper_names() call from
    plugin._handle_impulse
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
import tempfile
import unittest
from queue import Queue
from unittest.mock import MagicMock, patch

# Ensure project root importable
_HERE = os.path.dirname(__file__)
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ── Bus constants present ────────────────────────────────────────────


class TestBusConstants(unittest.TestCase):
    def test_agency_ready_constant_exists(self):
        from titan_plugin.bus import AGENCY_READY
        self.assertEqual(AGENCY_READY, "AGENCY_READY")

    def test_agency_stats_constant_exists(self):
        from titan_plugin.bus import AGENCY_STATS
        self.assertEqual(AGENCY_STATS, "AGENCY_STATS")

    def test_assessment_stats_constant_exists(self):
        from titan_plugin.bus import ASSESSMENT_STATS
        self.assertEqual(ASSESSMENT_STATS, "ASSESSMENT_STATS")


# ── Proxy unit tests (mocked bus) ────────────────────────────────────


class TestAgencyProxyCore(unittest.TestCase):
    """AgencyProxy behavior in isolation (mocked bus)."""

    def _make_bus_mock(self, response_payload=None, return_none=False):
        bus = MagicMock()
        bus.subscribe.return_value = Queue()
        if return_none:
            bus.request.return_value = None
        else:
            bus.request.return_value = {
                "type": "RESPONSE", "src": "agency_worker",
                "dst": "agency_proxy", "rid": "test-rid",
                "payload": response_payload or {},
                "ts": time.time(),
            }

        # 2026-04-29 — bus.request_async is the new canonical async path
        # (routes through bus_ipc_pool). Tests retain `bus.request` mocks
        # for call_args assertions; we wrap as async so
        # `await self._bus.request_async(...)` invokes the same MagicMock.
        async def _request_async(*args, **kwargs):
            return bus.request(*args, **kwargs)
        bus.request_async = _request_async

        return bus

    def test_proxy_subscribes_to_bus_at_construction(self):
        from titan_plugin.proxies.agency_proxy import AgencyProxy
        bus = self._make_bus_mock()
        AgencyProxy(bus)
        bus.subscribe.assert_called_once_with("agency_proxy", reply_only=True)

    def test_proxy_default_stats(self):
        from titan_plugin.proxies.agency_proxy import AgencyProxy
        bus = self._make_bus_mock()
        proxy = AgencyProxy(bus)
        stats = proxy.get_stats()
        self.assertEqual(stats["action_count"], 0)
        self.assertEqual(stats["llm_calls_this_hour"], 0)
        self.assertEqual(stats["registered_helpers"], [])

    def test_proxy_update_cached_stats(self):
        from titan_plugin.proxies.agency_proxy import AgencyProxy
        bus = self._make_bus_mock()
        proxy = AgencyProxy(bus)
        proxy.update_cached_stats({
            "action_count": 5,
            "llm_calls_this_hour": 2,
            "budget_per_hour": 10,
            "budget_remaining": 8,
            "registered_helpers": ["web_search", "art_generate"],
            "helper_statuses": {"web_search": "available"},
            "recent_actions": 3,
        })
        stats = proxy.get_stats()
        self.assertEqual(stats["action_count"], 5)
        self.assertEqual(stats["registered_helpers"], ["web_search", "art_generate"])
        self.assertEqual(stats["helper_statuses"]["web_search"], "available")

    def test_proxy_handle_intent_routes_to_bus(self):
        from titan_plugin.proxies.agency_proxy import AgencyProxy
        action_result = {
            "action_id": 1, "impulse_id": 7, "posture": "research",
            "helper": "web_search", "success": True, "result": "found",
            "enrichment_data": {}, "error": None, "reasoning": "test",
            "trinity_before": {}, "ts": time.time(),
        }
        bus = self._make_bus_mock(response_payload={"action_result": action_result})
        proxy = AgencyProxy(bus)
        result = asyncio.get_event_loop().run_until_complete(
            proxy.handle_intent({"posture": "research", "impulse_id": 7})
        )
        # Bus.request was called against the worker
        call_args = bus.request.call_args
        # First positional or kwarg-form — proxy uses positional in to_thread
        all_args = list(call_args.args) + list(call_args.kwargs.values())
        # Confirm "agency_worker" dst appears
        self.assertIn("agency_worker", all_args)
        # Confirm payload action
        payload = next((a for a in all_args
                        if isinstance(a, dict) and a.get("action") == "handle_intent"), None)
        self.assertIsNotNone(payload)
        self.assertEqual(payload["intent"]["impulse_id"], 7)
        # Result reconstructed
        self.assertEqual(result["impulse_id"], 7)
        self.assertEqual(result["helper"], "web_search")

    def test_proxy_handle_intent_returns_none_on_timeout(self):
        from titan_plugin.proxies.agency_proxy import AgencyProxy
        bus = self._make_bus_mock(return_none=True)
        proxy = AgencyProxy(bus)
        result = asyncio.get_event_loop().run_until_complete(
            proxy.handle_intent({"posture": "research"})
        )
        self.assertIsNone(result)

    def test_proxy_handle_intent_returns_none_on_worker_error(self):
        from titan_plugin.proxies.agency_proxy import AgencyProxy
        bus = self._make_bus_mock(response_payload={"error": "boom"})
        proxy = AgencyProxy(bus)
        result = asyncio.get_event_loop().run_until_complete(
            proxy.handle_intent({"posture": "research"})
        )
        self.assertIsNone(result)

    def test_proxy_dispatch_from_nervous_signals_routes_to_bus(self):
        from titan_plugin.proxies.agency_proxy import AgencyProxy
        results = [
            {"action_id": 1, "impulse_id": -1, "posture": "creativity",
             "helper": "art_generate", "success": True, "result": "art.png",
             "enrichment_data": {}, "error": None, "reasoning": "auto",
             "trinity_before": {}, "ts": time.time()}
        ]
        bus = self._make_bus_mock(response_payload={"action_results": results})
        proxy = AgencyProxy(bus)
        out = asyncio.get_event_loop().run_until_complete(
            proxy.dispatch_from_nervous_signals(
                outer_signals=[{"system": "CREATIVITY", "urgency": 0.8,
                                "helpers": ["art_generate"]}],
                trinity_snapshot={},
            )
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["helper"], "art_generate")

    def test_proxy_dispatch_returns_empty_on_timeout(self):
        from titan_plugin.proxies.agency_proxy import AgencyProxy
        bus = self._make_bus_mock(return_none=True)
        proxy = AgencyProxy(bus)
        out = asyncio.get_event_loop().run_until_complete(
            proxy.dispatch_from_nervous_signals([], {})
        )
        self.assertEqual(out, [])

    def test_proxy_registry_facade_returns_cached_helpers(self):
        """_handle_impulse reads self._agency._registry.list_helper_names()."""
        from titan_plugin.proxies.agency_proxy import AgencyProxy
        bus = self._make_bus_mock()
        proxy = AgencyProxy(bus)
        proxy.update_cached_stats({
            "registered_helpers": ["web_search", "art_generate", "infra_inspect"],
            "helper_statuses": {"web_search": "available"},
        })
        names = proxy._registry.list_helper_names()
        self.assertEqual(names, ["web_search", "art_generate", "infra_inspect"])
        statuses = proxy._registry.get_all_statuses()
        self.assertEqual(statuses["web_search"], "available")


class TestAssessmentProxyCore(unittest.TestCase):
    """AssessmentProxy behavior in isolation (mocked bus)."""

    def _make_bus_mock(self, response_payload=None, return_none=False):
        bus = MagicMock()
        bus.subscribe.return_value = Queue()
        if return_none:
            bus.request.return_value = None
        else:
            bus.request.return_value = {
                "type": "RESPONSE", "src": "agency_worker",
                "dst": "assessment_proxy", "rid": "test-rid",
                "payload": response_payload or {},
                "ts": time.time(),
            }

        # 2026-04-29 — bus.request_async wrapper (see TestAgencyProxyCore
        # _make_bus_mock for rationale).
        async def _request_async(*args, **kwargs):
            return bus.request(*args, **kwargs)
        bus.request_async = _request_async

        return bus

    def test_proxy_subscribes_at_construction(self):
        from titan_plugin.proxies.assessment_proxy import AssessmentProxy
        bus = self._make_bus_mock()
        AssessmentProxy(bus)
        bus.subscribe.assert_called_once_with("assessment_proxy", reply_only=True)

    def test_proxy_assess_routes_to_bus(self):
        from titan_plugin.proxies.assessment_proxy import AssessmentProxy
        body_assessment = {
            "action_id": 1, "impulse_id": 7, "score": 0.78,
            "reflection": "ok", "enrichment": {"mind": {0: 0.05}},
            "mood_delta": 0.02, "threshold_direction": "lower",
            "ts": time.time(),
        }
        bus = self._make_bus_mock(response_payload={"assessment": body_assessment})
        proxy = AssessmentProxy(bus)
        result = asyncio.get_event_loop().run_until_complete(
            proxy.assess({"action_id": 1, "impulse_id": 7, "success": True,
                          "result": "found something", "helper": "web_search",
                          "posture": "research"})
        )
        self.assertAlmostEqual(result["score"], 0.78)
        self.assertEqual(result["threshold_direction"], "lower")

    def test_proxy_assess_returns_neutral_on_timeout(self):
        from titan_plugin.proxies.assessment_proxy import AssessmentProxy
        bus = self._make_bus_mock(return_none=True)
        proxy = AssessmentProxy(bus)
        result = asyncio.get_event_loop().run_until_complete(
            proxy.assess({"action_id": 99, "impulse_id": 99, "success": True})
        )
        self.assertEqual(result["score"], 0.5)
        self.assertEqual(result["threshold_direction"], "hold")
        self.assertIn("proxy_neutral", result["reflection"])

    def test_proxy_assess_returns_neutral_on_worker_error(self):
        from titan_plugin.proxies.assessment_proxy import AssessmentProxy
        bus = self._make_bus_mock(response_payload={"error": "scoring crashed"})
        proxy = AssessmentProxy(bus)
        result = asyncio.get_event_loop().run_until_complete(
            proxy.assess({"action_id": 1, "impulse_id": 1, "success": True})
        )
        self.assertEqual(result["score"], 0.5)
        self.assertIn("scoring crashed", result["reflection"])

    def test_proxy_update_cached_stats(self):
        from titan_plugin.proxies.assessment_proxy import AssessmentProxy
        bus = self._make_bus_mock()
        proxy = AssessmentProxy(bus)
        proxy.update_cached_stats({
            "total": 25, "avg_score": 0.62, "recent": [{"score": 0.7}],
        })
        stats = proxy.get_stats()
        self.assertEqual(stats["total"], 25)
        self.assertAlmostEqual(stats["avg_score"], 0.62)


# ── Worker handler tests (Queue-driven, no real subprocess) ──────────


class TestAgencyWorkerHandler(unittest.TestCase):
    """Drive worker_main with sync Queues + assert RESPONSE shape."""

    def _make_cfg(self, td: str) -> dict:
        return {
            "info_banner": {"titan_id": "TEST"},
            "memory_and_storage": {"data_dir": td},
            "inference": {"ollama_cloud_api_key": "", "venice_api_key": ""},
            "agency": {"enabled": True, "llm_budget_per_hour": 10},
            "stealth_sage": {},
            "expressive": {"output_path": td},
            "audio": {"max_duration_seconds": 30, "sample_rate": 44100},
            "knowledge_pipeline": {"budgets": {}},
            "network": {"wallet_keypair_path": "data/titan_identity_keypair.json"},
        }

    def _drain(self, q):
        sent = []
        while not q.empty():
            try:
                sent.append(q.get_nowait())
            except Exception:
                break
        return sent

    def test_worker_emits_module_ready_and_agency_ready_on_boot(self):
        from titan_plugin.modules.agency_worker import agency_worker_main
        recv, send = Queue(), Queue()
        recv.put({"type": "MODULE_SHUTDOWN", "src": "guardian", "dst": "agency_worker"})
        with tempfile.TemporaryDirectory() as td:
            agency_worker_main(recv, send, "agency_worker", self._make_cfg(td))
        sent = self._drain(send)
        ready_module = [m for m in sent if m.get("type") == "MODULE_READY"]
        ready_agency = [m for m in sent if m.get("type") == "AGENCY_READY"]
        self.assertEqual(len(ready_module), 1)
        self.assertEqual(len(ready_agency), 1)
        self.assertEqual(ready_agency[0]["payload"]["titan_id"], "TEST")
        self.assertIn("helpers", ready_agency[0]["payload"])

    def test_worker_handles_handle_intent_query(self):
        from titan_plugin.modules.agency_worker import agency_worker_main
        recv, send = Queue(), Queue()
        recv.put({
            "type": "QUERY", "src": "test_caller", "dst": "agency_worker",
            "rid": "rid-intent",
            "payload": {
                "action": "handle_intent",
                "intent": {
                    "posture": "meditate",
                    "urgency": 0.5,
                    "impulse_id": 42,
                    "source_layer": "spirit",
                    "source_dims": [0],
                    "deficit_values": [0.3],
                    "trinity_snapshot": {},
                },
            },
            "ts": time.time(),
        })
        recv.put({"type": "MODULE_SHUTDOWN", "src": "guardian", "dst": "agency_worker"})
        with tempfile.TemporaryDirectory() as td:
            agency_worker_main(recv, send, "agency_worker", self._make_cfg(td))
        sent = self._drain(send)
        responses = [m for m in sent if m.get("type") == "RESPONSE"
                     and m.get("rid") == "rid-intent"]
        self.assertEqual(len(responses), 1)
        body = responses[0]["payload"]
        # action_result is either dict (helper executed) or None (no helper /
        # rule-based fallback returned None). With no LLM key + no real
        # registry availability, expect None or a dict with no_suitable_helper
        # / helper_not_found error.
        ar = body.get("action_result")
        if ar is not None:
            self.assertIn("posture", ar)
            self.assertEqual(ar["posture"], "meditate")

    def test_worker_handles_assess_query(self):
        from titan_plugin.modules.agency_worker import agency_worker_main
        recv, send = Queue(), Queue()
        recv.put({
            "type": "QUERY", "src": "test", "dst": "agency_worker",
            "rid": "rid-assess",
            "payload": {
                "action": "assess",
                "action_result": {
                    "action_id": 1, "impulse_id": 7, "posture": "research",
                    "helper": "web_search", "success": True,
                    "result": "found something useful (long enough)",
                    "enrichment_data": {}, "error": None,
                },
            },
            "ts": time.time(),
        })
        recv.put({"type": "MODULE_SHUTDOWN", "src": "guardian", "dst": "agency_worker"})
        with tempfile.TemporaryDirectory() as td:
            agency_worker_main(recv, send, "agency_worker", self._make_cfg(td))
        sent = self._drain(send)
        responses = [m for m in sent if m.get("type") == "RESPONSE"
                     and m.get("rid") == "rid-assess"]
        self.assertEqual(len(responses), 1)
        assessment = responses[0]["payload"].get("assessment")
        self.assertIsNotNone(assessment)
        self.assertIn("score", assessment)
        self.assertIn("threshold_direction", assessment)
        # No LLM available — should hit heuristic fallback successfully
        self.assertGreaterEqual(assessment["score"], 0.0)
        self.assertLessEqual(assessment["score"], 1.0)

    def test_worker_handles_dispatch_from_nervous_signals_query(self):
        from titan_plugin.modules.agency_worker import agency_worker_main
        recv, send = Queue(), Queue()
        recv.put({
            "type": "QUERY", "src": "test", "dst": "agency_worker",
            "rid": "rid-dispatch",
            "payload": {
                "action": "dispatch_from_nervous_signals",
                "outer_signals": [],  # empty list = no executions, returns []
                "trinity_snapshot": {},
            },
            "ts": time.time(),
        })
        recv.put({"type": "MODULE_SHUTDOWN", "src": "guardian", "dst": "agency_worker"})
        with tempfile.TemporaryDirectory() as td:
            agency_worker_main(recv, send, "agency_worker", self._make_cfg(td))
        sent = self._drain(send)
        responses = [m for m in sent if m.get("type") == "RESPONSE"
                     and m.get("rid") == "rid-dispatch"]
        self.assertEqual(len(responses), 1)
        self.assertEqual(responses[0]["payload"].get("action_results"), [])

    def test_worker_handles_agency_stats_query(self):
        from titan_plugin.modules.agency_worker import agency_worker_main
        recv, send = Queue(), Queue()
        recv.put({
            "type": "QUERY", "src": "test", "dst": "agency_worker",
            "rid": "rid-stats",
            "payload": {"action": "agency_stats"},
            "ts": time.time(),
        })
        recv.put({"type": "MODULE_SHUTDOWN", "src": "guardian", "dst": "agency_worker"})
        with tempfile.TemporaryDirectory() as td:
            agency_worker_main(recv, send, "agency_worker", self._make_cfg(td))
        sent = self._drain(send)
        responses = [m for m in sent if m.get("type") == "RESPONSE"
                     and m.get("rid") == "rid-stats"]
        self.assertEqual(len(responses), 1)
        stats = responses[0]["payload"].get("stats")
        self.assertIsNotNone(stats)
        self.assertIn("action_count", stats)
        self.assertIn("registered_helpers", stats)

    def test_worker_handles_assessment_stats_query(self):
        from titan_plugin.modules.agency_worker import agency_worker_main
        recv, send = Queue(), Queue()
        recv.put({
            "type": "QUERY", "src": "test", "dst": "agency_worker",
            "rid": "rid-asstats",
            "payload": {"action": "assessment_stats"},
            "ts": time.time(),
        })
        recv.put({"type": "MODULE_SHUTDOWN", "src": "guardian", "dst": "agency_worker"})
        with tempfile.TemporaryDirectory() as td:
            agency_worker_main(recv, send, "agency_worker", self._make_cfg(td))
        sent = self._drain(send)
        responses = [m for m in sent if m.get("type") == "RESPONSE"
                     and m.get("rid") == "rid-asstats"]
        self.assertEqual(len(responses), 1)
        stats = responses[0]["payload"].get("stats")
        self.assertIn("total", stats)
        self.assertIn("avg_score", stats)

    def test_worker_handles_unknown_action_gracefully(self):
        from titan_plugin.modules.agency_worker import agency_worker_main
        recv, send = Queue(), Queue()
        recv.put({
            "type": "QUERY", "src": "test", "dst": "agency_worker",
            "rid": "rid-unknown",
            "payload": {"action": "no_such_action"},
            "ts": time.time(),
        })
        recv.put({"type": "MODULE_SHUTDOWN", "src": "guardian", "dst": "agency_worker"})
        with tempfile.TemporaryDirectory() as td:
            agency_worker_main(recv, send, "agency_worker", self._make_cfg(td))
        sent = self._drain(send)
        responses = [m for m in sent if m.get("type") == "RESPONSE"
                     and m.get("rid") == "rid-unknown"]
        self.assertEqual(len(responses), 1)
        self.assertIn("error", responses[0]["payload"])


# ── Plugin flag-routing + ModuleSpec registration ────────────────────


class TestPluginFlagRouting(unittest.TestCase):
    def test_boot_agency_has_both_paths(self):
        """_boot_agency contains flag-on (proxy) and flag-off (local) branches."""
        from titan_plugin.core.plugin import TitanPlugin
        import inspect
        src = inspect.getsource(TitanPlugin._boot_agency)
        self.assertIn("a8_agency_subprocess_enabled", src)
        self.assertIn("AgencyProxy", src)
        self.assertIn("AssessmentProxy", src)
        # Local path also intact
        self.assertIn("AgencyModule", src)
        self.assertIn("SelfAssessment", src)

    def test_module_spec_registered_with_flag_aware_autostart(self):
        from titan_plugin.core.plugin import TitanPlugin
        import inspect
        src = inspect.getsource(TitanPlugin)
        self.assertIn('name="agency_worker"', src)
        self.assertIn('layer="L3"', src)
        self.assertIn("autostart=_ag_subproc_enabled", src)
        self.assertIn("agency_worker_main", src)

    def test_agency_loop_handles_stats_broadcasts(self):
        """_agency_loop refreshes proxy cached stats on AGENCY_STATS / ASSESSMENT_STATS."""
        from titan_plugin.core.plugin import TitanPlugin
        import inspect
        src = inspect.getsource(TitanPlugin._agency_loop)
        self.assertIn("AGENCY_STATS", src)
        self.assertIn("ASSESSMENT_STATS", src)
        self.assertIn("AGENCY_READY", src)
        self.assertIn("update_cached_stats", src)

    def test_config_flag_default_false(self):
        """titan_params.toml has the flag, default false, with docstring."""
        params_path = os.path.join(_ROOT, "titan_plugin", "titan_params.toml")
        with open(params_path) as f:
            content = f.read()
        self.assertIn("a8_agency_subprocess_enabled", content)
        # Default OFF
        self.assertIn("a8_agency_subprocess_enabled = false", content)
        # Has explanatory docstring
        self.assertIn("Phase A.8.6", content)


if __name__ == "__main__":
    unittest.main()
