"""
tests/test_outer_body_worker_a_s8.py — Phase A.S8 outer_body_worker smoke tests.

Verifies boot signals, OUTER_SOURCES_SNAPSHOT handling, FILTER_DOWN, and
MODULE_SHUTDOWN exit. Does NOT wait for a 45s publish cycle — that is covered
by the Schumann ratio tests in test_outer_trinity_workers_a_s8.py.
"""
import inspect
import threading
import time
from queue import Empty, Queue


def _boot_worker(extra_config=None):
    """Boot outer_body_worker in a daemon thread; return (recv, send, thread)."""
    from titan_plugin.modules.outer_body_worker import outer_body_worker_main
    recv = Queue()
    send = Queue()
    config = {"info_banner": {"titan_id": "T-test"}}
    if extra_config:
        config.update(extra_config)

    t = threading.Thread(
        target=outer_body_worker_main,
        args=(recv, send, "outer_body", config),
        daemon=True,
    )
    t.start()
    return recv, send, t


def _drain(q: Queue, timeout=0.8) -> list:
    msgs = []
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            msgs.append(q.get(timeout=0.05))
        except Empty:
            if msgs:
                break
    return msgs


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


def test_entry_point_signature():
    from titan_plugin.modules.outer_body_worker import outer_body_worker_main
    params = list(inspect.signature(outer_body_worker_main).parameters.keys())
    assert params[:4] == ["recv_queue", "send_queue", "name", "config"]


def test_worker_emits_module_ready():
    recv, send, t = _boot_worker()
    try:
        assert _wait_for_ready(send), "outer_body_worker did not emit MODULE_READY within 3s"
    finally:
        recv.put({"type": "MODULE_SHUTDOWN"})
        t.join(timeout=3.0)


def test_worker_exits_on_shutdown():
    recv, send, t = _boot_worker()
    _wait_for_ready(send)
    recv.put({"type": "MODULE_SHUTDOWN"})
    t.join(timeout=3.0)
    assert not t.is_alive(), "outer_body_worker did not exit on MODULE_SHUTDOWN"


def test_outer_sources_snapshot_accepted():
    """OUTER_SOURCES_SNAPSHOT is consumed without crash or spurious output."""
    recv, send, t = _boot_worker()
    try:
        _wait_for_ready(send)
        recv.put({
            "type": "OUTER_SOURCES_SNAPSHOT",
            "payload": {
                "soul_health": 0.75,
                "uptime_seconds": 3600.0,
                "agency_stats": {"active": 2},
                "bus_stats": {"total": 100},
            },
        })
        time.sleep(0.3)
        # Drain; expect no crash msgs or unexpected STATE output
        msgs = _drain(send, timeout=0.5)
        state_msgs = [m for m in msgs if m.get("type") == "OUTER_BODY_STATE"]
        # No publish expected this fast (45s interval) — just no crash
        assert state_msgs == [] or len(state_msgs) <= 1
    finally:
        recv.put({"type": "MODULE_SHUTDOWN"})
        t.join(timeout=3.0)


def test_filter_down_accepted():
    """FILTER_DOWN message with 5 multipliers accepted without crash."""
    recv, send, t = _boot_worker()
    try:
        _wait_for_ready(send)
        recv.put({
            "type": "FILTER_DOWN",
            "payload": {"multipliers": [0.9, 0.8, 0.7, 0.6, 1.0]},
        })
        time.sleep(0.2)
    finally:
        recv.put({"type": "MODULE_SHUTDOWN"})
        t.join(timeout=3.0)


def test_unknown_message_ignored():
    """Unknown message types do not cause crash or unexpected output."""
    recv, send, t = _boot_worker()
    try:
        _wait_for_ready(send)
        recv.put({"type": "COMPLETELY_UNKNOWN_TYPE", "payload": {}})
        time.sleep(0.2)
        msgs = _drain(send, timeout=0.3)
        state_msgs = [m for m in msgs if m.get("type") == "OUTER_BODY_STATE"]
        assert state_msgs == []
    finally:
        recv.put({"type": "MODULE_SHUTDOWN"})
        t.join(timeout=3.0)
