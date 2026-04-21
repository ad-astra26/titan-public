"""Tests for emot_cgn_worker + shm-mirror integration (Phase 1.6g).

Covers:
  - Worker entry fn smoke (boot + MODULE_READY + shutdown)
  - Bus message handlers (EMOT_CHAIN_EVIDENCE / FELT_CLUSTER_UPDATE /
    CGN_CROSS_INSIGHT) consume payloads and mutate EmotCGNConsumer state
  - ShmEmotWriter → ShmEmotReader roundtrip
  - Worker writes shm after state-changing events
  - Consumer ShmEmotReader falls back to None when shm absent
"""
from __future__ import annotations

import os
import tempfile
import threading
import time
from queue import Queue

import pytest

from titan_plugin.logic.emot_shm_protocol import (
    ShmEmotReader, ShmEmotWriter, STATE_SIZE, GROUNDING_SIZE,
)
from titan_plugin.logic.emotion_cluster import EMOT_PRIMITIVES


# ── ShmEmotWriter + ShmEmotReader roundtrip ─────────────────────────

def test_writer_creates_fixed_size_files():
    with tempfile.TemporaryDirectory() as tmp:
        sp = os.path.join(tmp, "state.bin")
        gp = os.path.join(tmp, "grounding.bin")
        ShmEmotWriter(state_path=sp, grounding_path=gp)
        assert os.path.getsize(sp) == STATE_SIZE
        assert os.path.getsize(gp) == GROUNDING_SIZE


def test_state_roundtrip_preserves_all_fields():
    with tempfile.TemporaryDirectory() as tmp:
        sp = os.path.join(tmp, "state.bin")
        gp = os.path.join(tmp, "grounding.bin")
        w = ShmEmotWriter(state_path=sp, grounding_path=gp)
        w.write_state(dominant_idx=5, is_active=True, cgn_registered=True,
                      V_beta=0.72, V_blended=0.68, cluster_confidence=0.85,
                      cluster_distance=1.2, total_updates=123,
                      cross_insights_sent=4, cross_insights_received=2)
        r = ShmEmotReader(state_path=sp, grounding_path=gp)
        s = r.read_state()
        assert s["dominant_idx"] == 5
        assert s["is_active"] is True
        assert s["cgn_registered"] is True
        assert abs(s["V_beta"] - 0.72) < 1e-4
        assert abs(s["V_blended"] - 0.68) < 1e-4
        assert abs(s["cluster_confidence"] - 0.85) < 1e-4
        assert s["total_updates"] == 123
        assert s["cross_insights_sent"] == 4


def test_grounding_roundtrip_preserves_8_primitives():
    with tempfile.TemporaryDirectory() as tmp:
        sp = os.path.join(tmp, "state.bin")
        gp = os.path.join(tmp, "grounding.bin")
        w = ShmEmotWriter(state_path=sp, grounding_path=gp)
        prims = [{"alpha": float(i) + 1.0, "beta": 1.0,
                  "V": i / 10.0, "confidence": i / 20.0,
                  "n_samples": i * 10} for i in range(8)]
        w.write_grounding(prims)
        r = ShmEmotReader(state_path=sp, grounding_path=gp)
        g = r.read_grounding()
        assert len(g) == 8
        for i, p in enumerate(g):
            assert abs(p["alpha"] - (i + 1.0)) < 1e-4
            assert p["n_samples"] == i * 10


def test_has_new_state_poll_cheap_change_detection():
    with tempfile.TemporaryDirectory() as tmp:
        sp = os.path.join(tmp, "state.bin")
        gp = os.path.join(tmp, "grounding.bin")
        w = ShmEmotWriter(state_path=sp, grounding_path=gp)
        w.write_state(dominant_idx=0, is_active=False, cgn_registered=False,
                      V_beta=0.5, V_blended=0.5, cluster_confidence=0.0,
                      cluster_distance=0.0, total_updates=0,
                      cross_insights_sent=0, cross_insights_received=0)
        r = ShmEmotReader(state_path=sp, grounding_path=gp)
        assert r.has_new_state() is True  # fresh reader sees version > 0
        r.read_state()
        assert r.has_new_state() is False  # after read, same version → no change
        w.write_state(dominant_idx=1, is_active=False, cgn_registered=False,
                      V_beta=0.6, V_blended=0.6, cluster_confidence=0.1,
                      cluster_distance=0.0, total_updates=1,
                      cross_insights_sent=0, cross_insights_received=0)
        assert r.has_new_state() is True  # writer advanced version


def test_reader_returns_none_when_shm_absent():
    """Worker not booted → reader graceful fallback."""
    r = ShmEmotReader(state_path="/tmp/nonexistent_emot_state.bin",
                      grounding_path="/tmp/nonexistent_emot_grounding.bin")
    assert r.read_state() is None
    assert r.read_grounding() is None
    assert r.has_new_state() is False


# ── Worker boot / shutdown ─────────────────────────────────────────

class _BusQueueMock:
    """Stand-in for the multiprocessing Queue the Guardian uses."""
    def __init__(self):
        self.q = Queue()
    def put_nowait(self, msg): self.q.put_nowait(msg)
    def put(self, msg): self.q.put(msg)
    def get(self, timeout=5.0):
        from queue import Empty
        return self.q.get(timeout=timeout)
    def empty(self): return self.q.empty()


def test_worker_main_emits_module_ready_then_exits_on_shutdown(tmp_path):
    """Smoke: run emot_cgn_worker_main in a thread with mock queues,
    send MODULE_SHUTDOWN, verify it boots + exits cleanly."""
    from titan_plugin.modules.emot_cgn_worker import emot_cgn_worker_main
    recv = _BusQueueMock()
    send = _BusQueueMock()
    config = {
        "titan_id": "T_TEST",
        "emot_cgn_save_dir": str(tmp_path / "emot_cgn"),
    }
    t = threading.Thread(
        target=emot_cgn_worker_main,
        args=(recv, send, "emot_cgn", config),
        daemon=True,
    )
    t.start()
    # Give worker time to boot + send MODULE_READY
    time.sleep(1.5)
    recv.put({"type": "MODULE_SHUTDOWN", "src": "guardian", "dst": "emot_cgn",
              "ts": time.time(), "payload": {}})
    t.join(timeout=10.0)
    assert not t.is_alive(), "worker did not exit on MODULE_SHUTDOWN"
    # Drain send queue + check for MODULE_READY
    saw_ready = False
    while not send.empty():
        msg = send.q.get_nowait()
        if msg.get("type") == "MODULE_READY":
            saw_ready = True
    assert saw_ready, "worker did not emit MODULE_READY"


def test_worker_handles_emot_chain_evidence(tmp_path):
    """Chain evidence → worker's EmotCGNConsumer β-posterior updated +
    shm file grows version. Tmp shm paths via config (tests don't touch
    /dev/shm/titan/ which is shared with real running Titans)."""
    state_path = str(tmp_path / "state.bin")
    grounding_path = str(tmp_path / "grounding.bin")

    from titan_plugin.modules.emot_cgn_worker import emot_cgn_worker_main
    recv = _BusQueueMock()
    send = _BusQueueMock()
    config = {"titan_id": "T_TEST",
              "emot_cgn_save_dir": str(tmp_path / "emot_cgn"),
              "shm_state_path": state_path,
              "shm_grounding_path": grounding_path}
    t = threading.Thread(target=emot_cgn_worker_main,
                         args=(recv, send, "emot_cgn", config), daemon=True)
    t.start()
    time.sleep(1.5)  # boot

    # Send EMOT_CHAIN_EVIDENCE
    recv.put({"type": "EMOT_CHAIN_EVIDENCE", "src": "spirit",
              "dst": "emot_cgn", "ts": time.time(),
              "payload": {"chain_id": 1, "dominant_at_start": "FLOW",
                          "dominant_at_end": "WONDER",
                          "terminal_reward": 0.9,
                          "ctx": {"DA": 0.8, "5HT": 0.7}}})
    time.sleep(1.0)

    # Verify state.bin exists + has non-zero version
    r = ShmEmotReader(state_path=state_path, grounding_path=grounding_path)
    state = r.read_state()
    assert state is not None
    assert state["total_updates"] >= 1, "chain evidence did not increment"

    # Shutdown
    recv.put({"type": "MODULE_SHUTDOWN", "src": "guardian", "dst": "emot_cgn",
              "ts": time.time(), "payload": {}})
    t.join(timeout=10.0)
    assert not t.is_alive()


def test_worker_handles_felt_cluster_update_with_130d(tmp_path):
    """FELT_CLUSTER_UPDATE with only 130D → worker builds 150D locally."""
    state_path = str(tmp_path / "state.bin")
    grounding_path = str(tmp_path / "grounding.bin")

    from titan_plugin.modules.emot_cgn_worker import emot_cgn_worker_main
    recv = _BusQueueMock()
    send = _BusQueueMock()
    config = {"titan_id": "T_TEST",
              "emot_cgn_save_dir": str(tmp_path / "emot_cgn"),
              "shm_state_path": state_path,
              "shm_grounding_path": grounding_path}
    t = threading.Thread(target=emot_cgn_worker_main,
                         args=(recv, send, "emot_cgn", config), daemon=True)
    t.start()
    time.sleep(1.5)

    recv.put({"type": "FELT_CLUSTER_UPDATE", "src": "spirit",
              "dst": "emot_cgn", "ts": time.time(),
              "payload": {"felt_tensor_130d": [0.5] * 130}})
    time.sleep(1.0)

    r = ShmEmotReader(state_path=state_path, grounding_path=grounding_path)
    state = r.read_state()
    assert state is not None
    # dominant_idx should be a valid primitive (clusterer assigned something)
    assert 0 <= state["dominant_idx"] < len(EMOT_PRIMITIVES)

    recv.put({"type": "MODULE_SHUTDOWN", "src": "guardian", "dst": "emot_cgn",
              "ts": time.time(), "payload": {}})
    t.join(timeout=10.0)


def test_worker_ignores_own_cross_insight_origin(tmp_path):
    """CGN_CROSS_INSIGHT with origin='emotional' → skipped (own bounce-back)."""
    state_path = str(tmp_path / "state.bin")
    grounding_path = str(tmp_path / "grounding.bin")

    from titan_plugin.modules.emot_cgn_worker import emot_cgn_worker_main
    recv = _BusQueueMock()
    send = _BusQueueMock()
    config = {"titan_id": "T_TEST",
              "emot_cgn_save_dir": str(tmp_path / "emot_cgn"),
              "shm_state_path": state_path,
              "shm_grounding_path": grounding_path}
    t = threading.Thread(target=emot_cgn_worker_main,
                         args=(recv, send, "emot_cgn", config), daemon=True)
    t.start()
    time.sleep(1.5)

    # Send own-origin insight: worker should skip without error
    recv.put({"type": "CGN_CROSS_INSIGHT", "src": "emot_cgn",
              "dst": "emot_cgn", "ts": time.time(),
              "payload": {"origin_consumer": "emotional",
                          "insight_type": "chain_outcome",
                          "terminal_reward": 0.9}})
    time.sleep(0.5)

    recv.put({"type": "MODULE_SHUTDOWN", "src": "guardian", "dst": "emot_cgn",
              "ts": time.time(), "payload": {}})
    t.join(timeout=10.0)
    assert not t.is_alive()  # no crash
