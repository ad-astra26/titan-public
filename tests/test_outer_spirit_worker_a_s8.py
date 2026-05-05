"""
tests/test_outer_spirit_worker_a_s8.py — Phase A.S8 outer_spirit_worker smoke tests.

Verifies boot signals, OUTER_SOURCES_SNAPSHOT handling, FILTER_DOWN, Observer
Principle SHM reads (safe fallback), and MODULE_SHUTDOWN exit.
"""
import inspect
import threading
import time
from queue import Empty, Queue


def _boot_worker(extra_config=None):
    from titan_plugin.modules.outer_spirit_worker import outer_spirit_worker_main
    recv = Queue()
    send = Queue()
    config = {"info_banner": {"titan_id": "T-test"}}
    if extra_config:
        config.update(extra_config)
    t = threading.Thread(
        target=outer_spirit_worker_main,
        args=(recv, send, "outer_spirit", config),
        daemon=True,
    )
    t.start()
    return recv, send, t


def _wait_for_ready(send: Queue, timeout=3.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            msg = send.get(timeout=0.1)
            if msg.get("type") == "MODULE_READY":
                return True
        except Empty:
            pass
    return False


def _drain(q: Queue, timeout=0.5) -> list:
    msgs = []
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            msgs.append(q.get(timeout=0.05))
        except Empty:
            if msgs:
                break
    return msgs


def test_entry_point_signature():
    from titan_plugin.modules.outer_spirit_worker import outer_spirit_worker_main
    params = list(inspect.signature(outer_spirit_worker_main).parameters.keys())
    assert params[:4] == ["recv_queue", "send_queue", "name", "config"]


def test_worker_emits_module_ready():
    recv, send, t = _boot_worker()
    try:
        assert _wait_for_ready(send), "outer_spirit_worker did not emit MODULE_READY within 3s"
    finally:
        recv.put({"type": "MODULE_SHUTDOWN"})
        t.join(timeout=3.0)


def test_worker_exits_on_shutdown():
    recv, send, t = _boot_worker()
    _wait_for_ready(send)
    recv.put({"type": "MODULE_SHUTDOWN"})
    t.join(timeout=3.0)
    assert not t.is_alive(), "outer_spirit_worker did not exit on MODULE_SHUTDOWN"


def test_outer_sources_snapshot_accepted():
    """Plugin sources accepted; spirit reads body+mind from SHM with safe fallback."""
    recv, send, t = _boot_worker()
    try:
        _wait_for_ready(send)
        recv.put({
            "type": "OUTER_SOURCES_SNAPSHOT",
            "payload": {
                "soul_health": 0.9,
                "uptime_seconds": 14400.0,
                "impulse_stats": {"total": 50, "assessed": 40, "avg_score": 0.75},
                "assessment_stats": {"avg_score": 0.75},
            },
        })
        time.sleep(0.3)
        # No OUTER_SPIRIT_STATE expected before 5s publish interval in test context
        msgs = _drain(send, timeout=0.5)
        spirit_msgs = [m for m in msgs if m.get("type") == "OUTER_SPIRIT_STATE"]
        assert spirit_msgs == [] or len(spirit_msgs) <= 1
    finally:
        recv.put({"type": "MODULE_SHUTDOWN"})
        t.join(timeout=3.0)


def test_filter_down_accepted():
    """FILTER_DOWN modulates spirit severity multipliers without crash."""
    recv, send, t = _boot_worker()
    try:
        _wait_for_ready(send)
        recv.put({
            "type": "FILTER_DOWN",
            "payload": {"multipliers": [0.4, 0.5, 0.6, 0.7, 0.8]},
        })
        time.sleep(0.2)
    finally:
        recv.put({"type": "MODULE_SHUTDOWN"})
        t.join(timeout=3.0)


def test_shm_observer_returns_valid_data():
    """Observer Principle: SHM reads return correct shape with valid float values.
    On live systems returns real SHM data; on cold boot falls back to [0.5]*N.
    Either way: length must match and all values must be in [0, 1]."""
    from titan_plugin.modules.outer_spirit_worker import (
        _read_outer_shm_5d, _read_outer_shm_15d,
    )
    result_5d = _read_outer_shm_5d("outer_body_5d")
    assert len(result_5d) == 5, f"Expected 5 values, got {len(result_5d)}"
    assert all(0.0 <= v <= 1.0 for v in result_5d), f"Values out of [0,1]: {result_5d}"

    result_15d = _read_outer_shm_15d("outer_mind_15d")
    assert len(result_15d) == 15, f"Expected 15 values, got {len(result_15d)}"
    assert all(0.0 <= v <= 1.0 for v in result_15d), f"Values out of [0,1]: {result_15d}"


def test_unknown_message_ignored():
    recv, send, t = _boot_worker()
    try:
        _wait_for_ready(send)
        recv.put({"type": "COMPLETELY_UNKNOWN", "payload": {}})
        time.sleep(0.2)
        msgs = _drain(send, timeout=0.3)
        spirit_msgs = [m for m in msgs if m.get("type") == "OUTER_SPIRIT_STATE"]
        assert spirit_msgs == []
    finally:
        recv.put({"type": "MODULE_SHUTDOWN"})
        t.join(timeout=3.0)
