"""
SPEC §8.0.quat parity tests (D-SPEC-131-Py, RFP_Phase_C_python_fix).

These tests pin the Condition+predicate refactor of BusSocketClient inbound
signaling. The fix replaces a dual-purpose `threading.Event` (data-arrived
AND stopping) with a single `threading.Condition(self._inbound_lock)` whose
waiters re-check an explicit multi-predicate on every wakeup. Matches the
Rust broker D-SPEC-131 split (per-subscriber data-wake vs close-state
primitives) for cross-language SPEC §8.0.quat parity.

Invariants asserted:

  1. Data arrival wakes a pending `SocketQueue.get()` and returns the msg.
  2. `stop()` wakes a pending `SocketQueue.get()` and raises `QueueEmpty`
     within a tight bound (no missed-wake).
  3. `get(timeout=N)` respects the timeout when no signal arrives.
  4. A spurious `_wake_cond.notify_all()` (predicate components both false)
     does NOT cause `get()` to return — the predicate re-evaluates and
     callers re-enter wait. (Replaces the old Event-pattern's vulnerability
     to false-positive wakes after a drain.)
"""
from __future__ import annotations

import queue
import threading
import time

from titan_hcl.core.bus_socket import BusSocketClient, SocketQueue


def _make_client(name: str = "test_cond") -> BusSocketClient:
    """Build a client without starting any threads — we drive the inbound
    deque manually via `_deliver_to_inbound` and observe `SocketQueue.get`
    wake/timeout semantics directly."""
    return BusSocketClient(
        titan_id="T1",
        authkey=b"\x00" * 32,
        name=name,
        sock_path="/tmp/test_bus_cond.sock",
        topics=None,
    )


# ── Invariant 1: data arrival wakes a pending get() ────────────────────


def test_inbound_data_wakes_get():
    """`_deliver_to_inbound` must wake a blocked `SocketQueue.get()` and the
    msg must be returned."""
    client = _make_client("test_data_wake")
    q = SocketQueue(client)

    result: dict = {}

    def reader() -> None:
        result["msg"] = q.get(timeout=2.0)

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    # Give the reader thread time to enter wait_for. 50 ms is generous.
    time.sleep(0.05)

    payload = {"type": "TEST_MSG", "n": 42}
    client._deliver_to_inbound(payload)

    t.join(timeout=1.0)
    assert not t.is_alive(), "reader did not wake within 1s of deliver"
    assert result["msg"] == payload


# ── Invariant 2: stop() wakes a pending get() within a tight bound ─────


def test_stop_wakes_pending_get():
    """`stop()` must wake a blocked `SocketQueue.get()` such that it raises
    `QueueEmpty` within 200 ms. No missed-wake."""
    client = _make_client("test_stop_wake")
    q = SocketQueue(client)

    raised: dict = {}

    def reader() -> None:
        try:
            q.get(timeout=5.0)
        except queue.Empty:
            raised["empty"] = True

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    time.sleep(0.05)

    t_stop_start = time.time()
    client.stop(timeout=0.5)
    t.join(timeout=0.5)
    elapsed = time.time() - t_stop_start

    assert not t.is_alive(), "reader did not wake within 500 ms of stop()"
    assert raised.get("empty") is True, "get() should have raised QueueEmpty"
    assert elapsed < 0.2, f"stop()→QueueEmpty took {elapsed*1000:.0f} ms (>200 ms)"


# ── Invariant 3: get(timeout=N) respects timeout under no signal ───────


def test_get_with_timeout_respects_timeout_under_no_signal():
    """With no data + no stop, `get(timeout=0.1)` must raise `QueueEmpty`
    after ~100 ms (within a 50 ms tolerance band — scheduling jitter)."""
    client = _make_client("test_timeout")
    q = SocketQueue(client)

    t_start = time.time()
    try:
        q.get(timeout=0.1)
        raised = False
    except queue.Empty:
        raised = True
    elapsed = time.time() - t_start

    assert raised, "get() should have raised QueueEmpty after timeout"
    assert 0.09 <= elapsed <= 0.20, (
        f"timeout=0.1 returned in {elapsed*1000:.0f} ms (expected 90-200 ms)"
    )


# ── Invariant 4: spurious notify_all does NOT cause get() to return ────


def test_wake_cond_predicate_does_not_false_positive():
    """A spurious `_wake_cond.notify_all()` with no data and no stop signal
    must NOT cause `get()` to return — the predicate re-evaluates as False
    and the caller re-enters wait. This is the load-bearing invariant the
    Condition+predicate pattern adds vs the old Event-based pattern."""
    client = _make_client("test_spurious")
    q = SocketQueue(client)

    returned: dict = {}

    def reader() -> None:
        try:
            q.get(timeout=0.5)
            returned["ok"] = True
        except queue.Empty:
            returned["empty"] = True

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    time.sleep(0.05)

    # Fire a spurious wake with neither predicate component true.
    with client._inbound_lock:
        assert not client._inbound, "deque must be empty for this invariant"
        assert not client._stop_event.is_set(), "stop must be clear"
        client._wake_cond.notify_all()

    # Reader must NOT have returned yet — give the wake time to propagate.
    time.sleep(0.1)
    assert "ok" not in returned, "spurious notify caused false-positive return"
    assert "empty" not in returned, "reader exited before timeout fired"

    # Now let the timeout fire — reader must wake via QueueEmpty.
    t.join(timeout=1.0)
    assert not t.is_alive(), "reader did not wake on timeout after spurious notify"
    assert returned.get("empty") is True
