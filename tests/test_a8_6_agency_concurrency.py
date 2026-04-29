"""
L3 Phase A.8.6 — Agency subprocess concurrency / ordering tests.

The CRITICAL §A.8.6 OBS gate is `OBS-a8-agency-impulse-ordering`:
"All IMPULSE messages produce ACTION_RESULT in correct sequence; zero
ordering inversions over 7 days." This test file is the unit-level
precondition for that gate — it runs the agency_worker loop (in-process,
single thread) under a sequence of QUERY messages and verifies the
RESPONSE rid matches the QUERY rid for every round-trip in order.

Note: parent's _agency_loop is single-threaded (await on each handle_intent
before processing next IMPULSE), so subprocess-side concurrency isn't
even possible — this test verifies the worker's serial-in / serial-out
contract holds even under back-to-back QUERY arrivals.

Real subprocess (mp.Process + mp.Queue) is exercised via the existing
Guardian smoke tests when the plugin boots in flag-on mode; we don't
duplicate that here. This file focuses on the worker's deterministic
QUERY → RESPONSE rid pairing.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from queue import Queue

_HERE = os.path.dirname(__file__)
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


class TestAgencyWorkerQueryOrdering(unittest.TestCase):
    """Worker processes QUERY messages serially + emits RESPONSE in order."""

    def _make_cfg(self, td: str) -> dict:
        return {
            "info_banner": {"titan_id": "TEST"},
            "memory_and_storage": {"data_dir": td},
            "inference": {"ollama_cloud_api_key": "", "venice_api_key": ""},
            "agency": {"enabled": True, "llm_budget_per_hour": 100},
            "stealth_sage": {},
            "expressive": {"output_path": td},
            "audio": {"max_duration_seconds": 30, "sample_rate": 44100},
            "knowledge_pipeline": {"budgets": {}},
            "network": {"wallet_keypair_path": "data/titan_identity_keypair.json"},
        }

    def test_ten_concurrent_queries_responses_in_order(self):
        """Inject 10 handle_intent QUERYs back-to-back; expect 10 RESPONSEs in
        the same rid order. This is the unit-level guarantor for the
        OBS-a8-agency-impulse-ordering 7d soak gate."""
        from titan_plugin.modules.agency_worker import agency_worker_main
        recv, send = Queue(), Queue()

        # Inject 10 IMPULSEs back-to-back, each with a distinct rid + impulse_id
        for i in range(10):
            recv.put({
                "type": "QUERY", "src": "test_caller", "dst": "agency_worker",
                "rid": f"rid-{i:02d}",
                "payload": {
                    "action": "handle_intent",
                    "intent": {
                        "posture": "meditate",
                        "urgency": 0.3,
                        "impulse_id": i,
                        "source_layer": "spirit",
                        "source_dims": [0],
                        "deficit_values": [0.2],
                        "trinity_snapshot": {},
                    },
                },
                "ts": time.time(),
            })
        recv.put({"type": "MODULE_SHUTDOWN", "src": "guardian", "dst": "agency_worker"})

        with tempfile.TemporaryDirectory() as td:
            agency_worker_main(recv, send, "agency_worker", self._make_cfg(td))

        # Drain — extract RESPONSEs in send order
        responses = []
        while not send.empty():
            try:
                m = send.get_nowait()
            except Exception:
                break
            if m.get("type") == "RESPONSE":
                responses.append(m)

        # Expect exactly 10 RESPONSEs, rid order preserved
        self.assertEqual(len(responses), 10,
                         f"Expected 10 RESPONSEs, got {len(responses)}: "
                         f"{[r.get('rid') for r in responses]}")
        for i, r in enumerate(responses):
            self.assertEqual(r["rid"], f"rid-{i:02d}",
                             f"Response #{i} has wrong rid — ordering inverted: "
                             f"got {r['rid']}, expected rid-{i:02d}")

    def test_mixed_action_types_responses_in_order(self):
        """Worker handles different actions (handle_intent, agency_stats, assess)
        interleaved + RESPONSE rid order matches QUERY arrival order."""
        from titan_plugin.modules.agency_worker import agency_worker_main
        recv, send = Queue(), Queue()

        # 5 mixed QUERYs in deterministic order
        recv.put({
            "type": "QUERY", "src": "t", "dst": "agency_worker",
            "rid": "Q1", "payload": {"action": "agency_stats"},
            "ts": time.time(),
        })
        recv.put({
            "type": "QUERY", "src": "t", "dst": "agency_worker",
            "rid": "Q2",
            "payload": {
                "action": "handle_intent",
                "intent": {"posture": "rest", "impulse_id": 1, "trinity_snapshot": {}},
            },
            "ts": time.time(),
        })
        recv.put({
            "type": "QUERY", "src": "t", "dst": "agency_worker",
            "rid": "Q3", "payload": {"action": "assessment_stats"},
            "ts": time.time(),
        })
        recv.put({
            "type": "QUERY", "src": "t", "dst": "agency_worker",
            "rid": "Q4",
            "payload": {
                "action": "assess",
                "action_result": {
                    "action_id": 1, "impulse_id": 1, "posture": "rest",
                    "helper": "infra_inspect", "success": True,
                    "result": "system_ok", "enrichment_data": {},
                    "error": None,
                },
            },
            "ts": time.time(),
        })
        recv.put({
            "type": "QUERY", "src": "t", "dst": "agency_worker",
            "rid": "Q5", "payload": {"action": "agency_stats"},
            "ts": time.time(),
        })
        recv.put({"type": "MODULE_SHUTDOWN", "src": "guardian", "dst": "agency_worker"})

        with tempfile.TemporaryDirectory() as td:
            agency_worker_main(recv, send, "agency_worker", self._make_cfg(td))

        # Drain RESPONSE messages in send order
        responses = []
        while not send.empty():
            try:
                m = send.get_nowait()
            except Exception:
                break
            if m.get("type") == "RESPONSE":
                responses.append(m["rid"])

        # Order preserved
        self.assertEqual(responses, ["Q1", "Q2", "Q3", "Q4", "Q5"],
                         f"RESPONSE rid order violated: {responses}")


class TestAgencyWorkerStatsBroadcast(unittest.TestCase):
    """AGENCY_STATS + ASSESSMENT_STATS broadcasts emitted at boot if loop runs
    long enough (the 60s tick is too long for unit tests, so we verify the
    broadcast structure is right when triggered)."""

    def test_module_ready_payload_has_helper_count(self):
        from titan_plugin.modules.agency_worker import agency_worker_main
        recv, send = Queue(), Queue()
        recv.put({"type": "MODULE_SHUTDOWN", "src": "guardian", "dst": "agency_worker"})
        with tempfile.TemporaryDirectory() as td:
            cfg = {
                "info_banner": {"titan_id": "T-X"},
                "memory_and_storage": {"data_dir": td},
                "inference": {"ollama_cloud_api_key": "", "venice_api_key": ""},
                "agency": {"enabled": True, "llm_budget_per_hour": 10},
                "stealth_sage": {}, "expressive": {"output_path": td},
                "audio": {"max_duration_seconds": 30, "sample_rate": 44100},
                "knowledge_pipeline": {"budgets": {}},
                "network": {"wallet_keypair_path": "data/titan_identity_keypair.json"},
            }
            agency_worker_main(recv, send, "agency_worker", cfg)
        sent = []
        while not send.empty():
            try:
                sent.append(send.get_nowait())
            except Exception:
                break
        ready = [m for m in sent if m.get("type") == "MODULE_READY"]
        agency_ready = [m for m in sent if m.get("type") == "AGENCY_READY"]
        self.assertEqual(len(ready), 1)
        self.assertEqual(len(agency_ready), 1)
        # Both carry helper_count in payload
        self.assertIn("helper_count", ready[0]["payload"])
        self.assertIn("helper_count", agency_ready[0]["payload"])
        # AGENCY_READY exposes helpers list (used by parent to seed proxy cache)
        self.assertIn("helpers", agency_ready[0]["payload"])
        self.assertIsInstance(agency_ready[0]["payload"]["helpers"], list)


if __name__ == "__main__":
    unittest.main()
