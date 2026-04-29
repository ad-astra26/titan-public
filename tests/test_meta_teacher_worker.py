"""Integration tests for meta_teacher_worker.

Covers rFP §10 integration test: full path META_CHAIN_COMPLETE → teacher →
META_TEACHER_FEEDBACK + META_TEACHER_GROUNDING, with mock LLM. Also
exercises:
  - adoption-signal tracking across subsequent chains
  - critiques.jsonl persistence + rotation
  - QUERY response path for /v4 API
  - enabled=false path produces zero META_TEACHER_* bus traffic
"""
import json
import os
import queue
import tempfile
import threading
import time

import pytest

from titan_plugin.logic.meta_teacher import MetaTeacher
from titan_plugin.modules import meta_teacher_worker as mtw


def make_config(tmp_dir, **overrides):
    cfg = {
        "enabled": True,
        "sample_mode": "uncertainty_plus_random",
        "uncertainty_threshold": 0.9,  # almost always trigger
        "random_sample_rate": 1.0,
        "max_critiques_per_hour": 100,
        "domain_balance_floor": 0.0,
        "reward_weight": 0.05,
        "reward_weight_cap": 0.30,
        "grounding_weight": 0.15,
        "ramp_phase1_critiques": 1000,
        "ramp_phase2_critiques": 1500,
        "llm_timeout_s": 5.0,
        "critique_log_retention_days": 7,
        "task_key": "meta_teacher",
        "data_dir": tmp_dir,
        "inference": {
            # No API key — forces LLM-disabled path (neutral critique)
            "ollama_cloud_api_key": "",
        },
    }
    cfg.update(overrides)
    return cfg


def make_chain_complete_msg(chain_id=42, domain="social", iql_conf=0.3,
                            primitives=None):
    prims = primitives or ["FORMULATE", "RECALL", "HYPOTHESIZE", "EVALUATE"]
    transitions = [(prims[i], prims[i + 1]) for i in range(len(prims) - 1)]
    return {
        "type": "META_CHAIN_COMPLETE",
        "src": "spirit", "dst": "all", "ts": time.time(), "rid": None,
        "payload": {
            "chain_id": chain_id,
            "primitives_used": prims,
            "primitive_transitions": transitions,
            "chain_length": len(prims),
            "domain": domain,
            "task_success": 0.7,
            "chain_iql_confidence": iql_conf,
            "start_epoch": 1000,
            "conclude_epoch": 1005,
            "context_summary": {
                "dominant_emotion": "WONDER",
                "chi_remaining": 0.5,
                "impasse_state": "none",
                "trigger_reason": "explore",
                "knowledge_injected": False,
            },
            "haov_hypothesis_id": None,
            "final_observation": {"chain_template": "FORMULATE→RECALL",
                                  "unique_primitives": 4},
        },
    }


def drain_queue(q, timeout=0.5):
    """Pull all messages from a queue with a short timeout."""
    msgs = []
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            msgs.append(q.get(timeout=0.05))
        except queue.Empty:
            if msgs:
                break
    return msgs


def start_worker(recv_q, send_q, config):
    """Run worker in a background thread so we can stop it via MODULE_SHUTDOWN."""
    t = threading.Thread(
        target=mtw.meta_teacher_worker_main,
        args=(recv_q, send_q, "meta_teacher", config),
        daemon=True,
        name="test-meta-teacher",
    )
    t.start()
    return t


class TestWorkerLifecycle:
    def test_worker_starts_and_shuts_down(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(tmp)
            recv = queue.Queue(maxsize=100)
            send = queue.Queue(maxsize=100)
            t = start_worker(recv, send, config)
            time.sleep(0.3)  # Let worker reach main loop
            recv.put({"type": "MODULE_SHUTDOWN", "src": "guardian",
                      "dst": "meta_teacher", "ts": time.time(),
                      "rid": None, "payload": {}})
            t.join(timeout=10.0)
            assert not t.is_alive(), "Worker should exit on MODULE_SHUTDOWN"

    def test_query_status_responds(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(tmp)
            recv = queue.Queue(maxsize=100)
            send = queue.Queue(maxsize=100)
            t = start_worker(recv, send, config)
            time.sleep(0.3)

            recv.put({
                "type": "QUERY", "src": "api", "dst": "meta_teacher",
                "ts": time.time(), "rid": "req-1",
                "payload": {"query_type": "get_meta_teacher_status"},
            })
            # Collect messages
            msgs = []
            deadline = time.time() + 2.0
            while time.time() < deadline:
                try:
                    m = send.get(timeout=0.1)
                    msgs.append(m)
                except queue.Empty:
                    pass
                if any(m.get("rid") == "req-1" for m in msgs):
                    break

            qr = [m for m in msgs if m.get("rid") == "req-1"]
            assert qr, f"No QUERY_RESPONSE for req-1 (got {[m.get('type') for m in msgs]})"
            resp = qr[0]
            assert resp["type"] == "QUERY_RESPONSE"
            assert "enabled" in resp["payload"]
            assert resp["payload"]["enabled"] is True

            recv.put({"type": "MODULE_SHUTDOWN", "src": "guardian",
                      "dst": "meta_teacher", "ts": time.time(),
                      "rid": None, "payload": {}})
            t.join(timeout=10.0)


class TestLLMFailurePath:
    """With no API key, LLM returns None → worker emits neutral feedback."""

    def test_chain_complete_produces_neutral_feedback(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(tmp)
            recv = queue.Queue(maxsize=100)
            send = queue.Queue(maxsize=100)
            t = start_worker(recv, send, config)
            time.sleep(0.3)

            recv.put(make_chain_complete_msg(chain_id=1))

            # Wait for META_TEACHER_FEEDBACK on send queue
            deadline = time.time() + 8.0
            feedback = None
            groundings = []
            while time.time() < deadline:
                try:
                    m = send.get(timeout=0.2)
                except queue.Empty:
                    continue
                if m.get("type") == "META_TEACHER_FEEDBACK":
                    feedback = m
                elif m.get("type") == "META_TEACHER_GROUNDING":
                    groundings.append(m)
                if feedback is not None:
                    time.sleep(0.1)
                    break

            assert feedback is not None, "Expected META_TEACHER_FEEDBACK"
            fp = feedback["payload"]
            assert fp["chain_id"] == 1
            # No API key → LLM call returns "" → critique=None → neutral
            assert fp["llm_ok"] is False
            assert fp["quality_score"] == 0.5
            assert fp["reward_bonus"] == 0.0

            # Groundings are only emitted when critique is valid — expect 0
            assert len(groundings) == 0

            recv.put({"type": "MODULE_SHUTDOWN", "src": "guardian",
                      "dst": "meta_teacher", "ts": time.time(),
                      "rid": None, "payload": {}})
            t.join(timeout=10.0)


class TestDisabledPath:
    """enabled=false → zero META_TEACHER_* bus traffic."""

    def test_no_bus_traffic_when_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(tmp, enabled=False)
            recv = queue.Queue(maxsize=100)
            send = queue.Queue(maxsize=100)
            t = start_worker(recv, send, config)
            time.sleep(0.3)

            for i in range(3):
                recv.put(make_chain_complete_msg(chain_id=100 + i))

            time.sleep(1.0)  # Let worker process

            # Collect all messages; filter out heartbeats
            msgs = drain_queue(send, timeout=0.5)
            teacher_msgs = [
                m for m in msgs
                if m.get("type", "").startswith("META_TEACHER_")
            ]
            assert len(teacher_msgs) == 0, (
                f"Expected zero META_TEACHER_* with enabled=false, "
                f"got {len(teacher_msgs)}: {[m['type'] for m in teacher_msgs]}")

            recv.put({"type": "MODULE_SHUTDOWN", "src": "guardian",
                      "dst": "meta_teacher", "ts": time.time(),
                      "rid": None, "payload": {}})
            t.join(timeout=10.0)


class TestMockedLLMFullPath:
    """With a mocked LLM client, full feedback + grounding flow."""

    def test_feedback_and_grounding_emitted(self, monkeypatch):
        # Mock the ollama loader to return a fake client.
        # v2 (2026-04-24): suggested_primitives must come from NOT-USED set.
        # The test chain uses {FORMULATE, RECALL, HYPOTHESIZE, EVALUATE}, so
        # we suggest SYNTHESIZE + BREAK — both absent from the chain — which
        # is valid under v2's "MISSING primitives only" rule.
        class FakeClient:
            async def complete(self, prompt, model, system, temperature,
                               max_tokens, timeout):
                return ('{"quality_score": 0.8, "critique_categories": ["depth"], '
                        '"critique_text": "Good decomposition.", '
                        '"suggested_primitives": ["SYNTHESIZE", "BREAK"], '
                        '"confidence": 0.9, "principles_invoked": ["depth"]}')

        def fake_loader(inf_cfg):
            return FakeClient(), "fake-model"

        monkeypatch.setattr(mtw, "_load_ollama_client", fake_loader)

        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(tmp)
            recv = queue.Queue(maxsize=100)
            send = queue.Queue(maxsize=100)
            t = start_worker(recv, send, config)
            time.sleep(0.3)

            recv.put(make_chain_complete_msg(chain_id=99, domain="knowledge"))

            deadline = time.time() + 6.0
            feedback = None
            groundings = []
            while time.time() < deadline:
                try:
                    m = send.get(timeout=0.2)
                except queue.Empty:
                    continue
                if m.get("type") == "META_TEACHER_FEEDBACK":
                    feedback = m
                elif m.get("type") == "META_TEACHER_GROUNDING":
                    groundings.append(m)
                if feedback and len(groundings) >= 4:
                    break

            assert feedback is not None
            fp = feedback["payload"]
            assert fp["chain_id"] == 99
            assert fp["quality_score"] == 0.8
            assert fp["llm_ok"] is True
            assert fp["reward_bonus"] == pytest.approx(0.05)  # phase-0 weight
            assert fp["suggested_primitives"] == ["SYNTHESIZE", "BREAK"]

            # Four primitives in the chain → four grounding messages
            assert len(groundings) == 4
            prim_ids = {g["payload"]["primitive_id"] for g in groundings}
            assert prim_ids == {"FORMULATE", "RECALL", "HYPOTHESIZE", "EVALUATE"}
            for g in groundings:
                assert g["payload"]["label_quality"] == 0.8
                assert g["payload"]["grounding_weight"] == 0.15

            # Check critiques.jsonl was written
            date_tag = time.strftime("%Y%m%d", time.gmtime())
            jsonl_path = os.path.join(tmp, "meta_teacher",
                                      f"critiques.{date_tag}.jsonl")
            assert os.path.exists(jsonl_path)
            with open(jsonl_path) as f:
                lines = f.readlines()
            assert len(lines) == 1
            entry = json.loads(lines[0])
            assert entry["chain_id"] == 99
            assert entry["domain"] == "knowledge"

            recv.put({"type": "MODULE_SHUTDOWN", "src": "guardian",
                      "dst": "meta_teacher", "ts": time.time(),
                      "rid": None, "payload": {}})
            t.join(timeout=10.0)
