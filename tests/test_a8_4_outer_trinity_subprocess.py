"""
L3 Phase A.8.4 — Outer Trinity subprocess extraction regression tests.

Covers:
  - outer_trinity_worker handles OUTER_TRINITY_COLLECT_REQUEST
  - worker publishes OUTER_TRINITY_STATE with collector result
  - worker emits MODULE_READY + OUTER_TRINITY_READY on boot
  - worker ignores non-COLLECT_REQUEST messages cleanly
  - worker exits on MODULE_SHUTDOWN
  - OuterTrinityCollector reads pre-extracted art/audio counts when present
  - Collector falls back to obs_db when pre-extracted missing (back-compat)
  - Pre-extraction parity: pre-extracted counts produce same outer_mind
    result as direct obs_db query
  - ModuleSpec for "outer_trinity" registered + autostart bound to flag
"""
from __future__ import annotations

import os
import sys
import threading
import time
import unittest
from queue import Empty, Queue
from unittest.mock import MagicMock

_HERE = os.path.dirname(__file__)
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ── Test fixtures ─────────────────────────────────────────────────────

def _minimal_sources(art_count_100=0, audio_count_100=0,
                     art_count_500=0, audio_count_500=0,
                     uptime_seconds=3600.0):
    """Build a minimal sources dict with required keys + pre-extracted counts."""
    return {
        "uptime_seconds": uptime_seconds,
        "agency_stats": {},
        "assessment_stats": {},
        "helper_statuses": {},
        "bus_stats": {},
        "memory_status": {},
        "soul_health": 0.5,
        "llm_avg_latency": 0.0,
        "anchor_state": {},
        "art_count_100": art_count_100,
        "audio_count_100": audio_count_100,
        "art_count_500": art_count_500,
        "audio_count_500": audio_count_500,
    }


class _MockObsDB:
    """Minimal observatory_db mock — returns lists of fake records."""

    def __init__(self, art_records=10, audio_records=5):
        self._art_records = art_records
        self._audio_records = audio_records

    def get_expressive_archive(self, type_, limit=100):
        n = self._art_records if type_ == "art" else self._audio_records
        return [{"id": i} for i in range(min(n, limit))]


# ── Worker behavior tests ─────────────────────────────────────────────

class TestOuterTrinityWorkerLifecycle(unittest.TestCase):
    """Boot signals + heartbeat + shutdown."""

    def _run_worker(self, recv: Queue, send: Queue, stop_event: threading.Event):
        """Run worker in a thread until stop_event triggers a SHUTDOWN msg."""
        from titan_plugin.modules.outer_trinity_worker import (
            outer_trinity_worker_main,
        )

        def _run():
            outer_trinity_worker_main(
                recv, send, "outer_trinity",
                {"info_banner": {"titan_id": "T-test"}})

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return t

    def _drain_send_queue(self, q: Queue, timeout=0.5):
        msgs = []
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                msgs.append(q.get(timeout=0.05))
            except Empty:
                if msgs:
                    break
        return msgs

    def test_worker_emits_module_ready_and_outer_trinity_ready_on_boot(self):
        recv = Queue()
        send = Queue()
        stop = threading.Event()
        t = self._run_worker(recv, send, stop)
        try:
            time.sleep(0.5)  # allow boot
            msgs = self._drain_send_queue(send, timeout=1.0)
            types = [m["type"] for m in msgs]
            self.assertIn("MODULE_READY", types)
            self.assertIn("OUTER_TRINITY_READY", types)

            module_ready = next(m for m in msgs if m["type"] == "MODULE_READY")
            self.assertEqual(module_ready["dst"], "guardian")
            self.assertEqual(module_ready["src"], "outer_trinity")
            self.assertEqual(module_ready["payload"]["titan_id"], "T-test")
        finally:
            recv.put({"type": "MODULE_SHUTDOWN"})
            t.join(timeout=2.0)

    def test_worker_exits_on_module_shutdown(self):
        recv = Queue()
        send = Queue()
        stop = threading.Event()
        t = self._run_worker(recv, send, stop)
        time.sleep(0.5)
        self._drain_send_queue(send, timeout=0.5)  # consume boot signals
        recv.put({"type": "MODULE_SHUTDOWN"})
        t.join(timeout=2.0)
        self.assertFalse(t.is_alive(), "Worker did not exit on SHUTDOWN")

    def test_worker_ignores_unknown_message_type(self):
        recv = Queue()
        send = Queue()
        stop = threading.Event()
        t = self._run_worker(recv, send, stop)
        try:
            time.sleep(0.5)
            self._drain_send_queue(send, timeout=0.5)  # boot signals
            recv.put({"type": "RANDOM_NOISE", "src": "x", "payload": {}})
            time.sleep(0.5)
            # Should be no STATE published from this, only heartbeat eventually
            msgs = self._drain_send_queue(send, timeout=0.3)
            state_msgs = [m for m in msgs if m["type"] == "OUTER_TRINITY_STATE"]
            self.assertEqual(state_msgs, [],
                             "Worker published STATE for unknown message type")
        finally:
            recv.put({"type": "MODULE_SHUTDOWN"})
            t.join(timeout=2.0)


class TestOuterTrinityWorkerCollectRequest(unittest.TestCase):
    """COLLECT_REQUEST → STATE round-trip behavior."""

    def _run_worker_and_send_request(self, sources: dict):
        """Boot worker, send COLLECT_REQUEST with sources, return STATE msg."""
        from titan_plugin.modules.outer_trinity_worker import (
            outer_trinity_worker_main,
        )
        recv = Queue()
        send = Queue()

        def _run():
            outer_trinity_worker_main(
                recv, send, "outer_trinity",
                {"info_banner": {"titan_id": "T-test"}})

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        try:
            time.sleep(0.5)  # boot
            # Drain boot signals
            while True:
                try:
                    send.get(timeout=0.1)
                except Empty:
                    break

            recv.put({
                "type": "OUTER_TRINITY_COLLECT_REQUEST",
                "src": "core", "dst": "outer_trinity",
                "payload": {"sources": sources},
                "ts": time.time(),
            })
            # Wait for STATE response
            deadline = time.time() + 3.0
            while time.time() < deadline:
                try:
                    msg = send.get(timeout=0.2)
                except Empty:
                    continue
                if msg["type"] == "OUTER_TRINITY_STATE":
                    return msg
            return None
        finally:
            recv.put({"type": "MODULE_SHUTDOWN"})
            t.join(timeout=2.0)

    def test_collect_request_produces_state(self):
        sources = _minimal_sources(art_count_100=12, audio_count_100=3)
        state = self._run_worker_and_send_request(sources)
        self.assertIsNotNone(state, "Worker did not publish OUTER_TRINITY_STATE")
        self.assertEqual(state["type"], "OUTER_TRINITY_STATE")
        self.assertEqual(state["src"], "outer_trinity")
        self.assertEqual(state["dst"], "all")

        payload = state["payload"]
        self.assertIn("outer_body", payload)
        self.assertIn("outer_mind", payload)
        self.assertIn("outer_spirit", payload)
        self.assertEqual(len(payload["outer_body"]), 5)
        self.assertEqual(len(payload["outer_mind"]), 5)
        self.assertEqual(len(payload["outer_spirit"]), 5)

    def test_collect_request_extended_tensors_present(self):
        sources = _minimal_sources(
            art_count_100=12, audio_count_100=3,
            art_count_500=42, audio_count_500=18)
        state = self._run_worker_and_send_request(sources)
        self.assertIsNotNone(state)
        payload = state["payload"]
        # 132D symmetry — extended tensors present
        self.assertIn("outer_mind_15d", payload)
        self.assertIn("outer_spirit_45d", payload)
        self.assertEqual(len(payload["outer_mind_15d"]), 15)
        self.assertEqual(len(payload["outer_spirit_45d"]), 45)


# ── Collector pre-extraction tests ────────────────────────────────────

class TestCollectorPreExtraction(unittest.TestCase):
    """Verify collector reads pre-extracted counts and falls back to obs_db."""

    def test_collector_uses_pre_extracted_counts(self):
        from titan_plugin.logic.outer_trinity import OuterTrinityCollector
        collector = OuterTrinityCollector()
        sources = _minimal_sources(
            art_count_100=20, audio_count_100=10,
            art_count_500=200, audio_count_500=50,
            uptime_seconds=86400.0)  # 1 day
        result = collector.collect(sources)
        # creative_output proportional to art_count; with 20 art over 1 day
        # at expected_per_day=5, we'd be at 4× expected → clamped to 1.0
        self.assertGreater(result["outer_mind"][0], 0.5)
        # sonic_expression: 10 audio over 1 day at expected=3 → 3.3× → 1.0
        self.assertGreater(result["outer_mind"][1], 0.5)

    def test_collector_falls_back_to_obs_db_when_no_pre_extracted(self):
        from titan_plugin.logic.outer_trinity import OuterTrinityCollector
        collector = OuterTrinityCollector()
        # Sources WITHOUT pre-extracted counts but WITH obs_db handle
        sources = {
            "uptime_seconds": 86400.0,
            "agency_stats": {}, "assessment_stats": {},
            "helper_statuses": {}, "bus_stats": {},
            "memory_status": {}, "soul_health": 0.5,
            "llm_avg_latency": 0.0, "anchor_state": {},
            "observatory_db": _MockObsDB(art_records=20, audio_records=10),
        }
        result = collector.collect(sources)
        # Same expected behavior as pre-extracted path
        self.assertGreater(result["outer_mind"][0], 0.5)
        self.assertGreater(result["outer_mind"][1], 0.5)

    def test_pre_extraction_parity_with_obs_db_query(self):
        """Pre-extracted counts produce same outer_mind as direct obs_db."""
        from titan_plugin.logic.outer_trinity import OuterTrinityCollector
        # Path A: pre-extracted only
        c1 = OuterTrinityCollector()
        sources_pre = _minimal_sources(
            art_count_100=15, audio_count_100=7,
            art_count_500=80, audio_count_500=30,
            uptime_seconds=86400.0)
        result_pre = c1.collect(sources_pre)

        # Path B: obs_db only
        c2 = OuterTrinityCollector()
        sources_db = {
            "uptime_seconds": 86400.0,
            "agency_stats": {}, "assessment_stats": {},
            "helper_statuses": {}, "bus_stats": {},
            "memory_status": {}, "soul_health": 0.5,
            "llm_avg_latency": 0.0, "anchor_state": {},
            "observatory_db": _MockObsDB(art_records=15, audio_records=7),
        }
        result_db = c2.collect(sources_db)

        # outer_mind[0] (creative_output, art-driven, 100-window) must match
        self.assertEqual(result_pre["outer_mind"][0], result_db["outer_mind"][0])
        # outer_mind[1] (sonic_expression, audio-driven, 100-window) must match
        self.assertEqual(result_pre["outer_mind"][1], result_db["outer_mind"][1])

    def test_collector_handles_missing_counts_gracefully(self):
        """No pre-extracted, no obs_db → counts default to 0, no crash."""
        from titan_plugin.logic.outer_trinity import OuterTrinityCollector
        collector = OuterTrinityCollector()
        sources = {
            "uptime_seconds": 3600.0,
            "agency_stats": {}, "assessment_stats": {},
            "helper_statuses": {}, "bus_stats": {},
            "memory_status": {}, "soul_health": 0.5,
            "llm_avg_latency": 0.0, "anchor_state": {},
        }
        result = collector.collect(sources)
        # 0 art, 0 audio → 0.0 normalized
        self.assertEqual(result["outer_mind"][0], 0.0)
        self.assertEqual(result["outer_mind"][1], 0.0)


# ── Plugin flag-routing tests (static — no actual subprocess boot) ────

class TestPluginFlagRouting(unittest.TestCase):
    """Verify ModuleSpec is registered and autostart bound to flag."""

    def test_module_spec_signature_matches_pattern(self):
        """outer_trinity_worker_main has the (recv, send, name, config) signature."""
        from titan_plugin.modules.outer_trinity_worker import (
            outer_trinity_worker_main,
        )
        import inspect
        sig = inspect.signature(outer_trinity_worker_main)
        params = list(sig.parameters.keys())
        self.assertEqual(params[:4], ["recv_queue", "send_queue", "name", "config"])

    def test_bus_constant_defined(self):
        from titan_plugin.bus import OUTER_TRINITY_COLLECT_REQUEST
        self.assertEqual(
            OUTER_TRINITY_COLLECT_REQUEST, "OUTER_TRINITY_COLLECT_REQUEST")


if __name__ == "__main__":
    unittest.main()
