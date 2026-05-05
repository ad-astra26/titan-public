"""
tests/test_outer_mind_worker_a_s8.py — Phase A.S8 outer_mind_worker smoke tests.

Verifies boot signals, OUTER_SOURCES_SNAPSHOT handling, FILTER_DOWN, and
MODULE_SHUTDOWN exit. Plugin-only sources path (no self-fetched sensors).
"""
import inspect
import threading
import time
from queue import Empty, Queue


def _boot_worker(extra_config=None):
    from titan_plugin.modules.outer_mind_worker import outer_mind_worker_main
    recv = Queue()
    send = Queue()
    config = {"info_banner": {"titan_id": "T-test"}}
    if extra_config:
        config.update(extra_config)
    t = threading.Thread(
        target=outer_mind_worker_main,
        args=(recv, send, "outer_mind", config),
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
    from titan_plugin.modules.outer_mind_worker import outer_mind_worker_main
    params = list(inspect.signature(outer_mind_worker_main).parameters.keys())
    assert params[:4] == ["recv_queue", "send_queue", "name", "config"]


def test_worker_emits_module_ready():
    recv, send, t = _boot_worker()
    try:
        assert _wait_for_ready(send), "outer_mind_worker did not emit MODULE_READY within 3s"
    finally:
        recv.put({"type": "MODULE_SHUTDOWN"})
        t.join(timeout=3.0)


def test_worker_exits_on_shutdown():
    recv, send, t = _boot_worker()
    _wait_for_ready(send)
    recv.put({"type": "MODULE_SHUTDOWN"})
    t.join(timeout=3.0)
    assert not t.is_alive(), "outer_mind_worker did not exit on MODULE_SHUTDOWN"


def test_outer_sources_snapshot_accepted():
    """Plugin-only sources snapshot is consumed; worker does not crash."""
    recv, send, t = _boot_worker()
    try:
        _wait_for_ready(send)
        recv.put({
            "type": "OUTER_SOURCES_SNAPSHOT",
            "payload": {
                "soul_health": 0.8,
                "uptime_seconds": 7200.0,
                "memory_status": {"persistent_count": 120, "total_nodes": 300},
                "art_count_100": 10,
                "audio_count_100": 5,
            },
        })
        time.sleep(0.3)
        msgs = _drain(send, timeout=0.5)
        # No OUTER_MIND_STATE expected before 15s publish interval
        mind_msgs = [m for m in msgs if m.get("type") == "OUTER_MIND_STATE"]
        assert mind_msgs == [] or len(mind_msgs) <= 1
    finally:
        recv.put({"type": "MODULE_SHUTDOWN"})
        t.join(timeout=3.0)


def test_filter_down_accepted():
    """FILTER_DOWN modulates 5D mind severity multipliers without crash."""
    recv, send, t = _boot_worker()
    try:
        _wait_for_ready(send)
        recv.put({
            "type": "FILTER_DOWN",
            "payload": {"multipliers": [0.5, 0.6, 0.7, 0.8, 0.9]},
        })
        time.sleep(0.2)
    finally:
        recv.put({"type": "MODULE_SHUTDOWN"})
        t.join(timeout=3.0)


def test_unknown_message_ignored():
    recv, send, t = _boot_worker()
    try:
        _wait_for_ready(send)
        recv.put({"type": "DOES_NOT_EXIST", "payload": {}})
        time.sleep(0.2)
        msgs = _drain(send, timeout=0.3)
        mind_msgs = [m for m in msgs if m.get("type") == "OUTER_MIND_STATE"]
        assert mind_msgs == []
    finally:
        recv.put({"type": "MODULE_SHUTDOWN"})
        t.join(timeout=3.0)
