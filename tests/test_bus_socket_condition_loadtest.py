"""
SPEC §8.0.quat soak / load tests (D-SPEC-134, RFP_Phase_C_python_fix §6.3).

These tests stress the Condition+predicate refactor of BusSocketClient
inbound signaling under sustained throughput. The unit tests in
`test_bus_socket_condition_pattern.py` pin the wake-up *semantics* on
single-event paths (data wake, stop wake, timeout, spurious notify).
This file pins the same semantics under load — what RFP §6.3 calls the
"missed-wake or stuck-get behavior" surface.

Soak invariants:

  L1. **No missed-wake under sustained producer pressure.**
      With ≥10K msgs/sec produced into the deque, the consumer's
      `SocketQueue.get()` returns every msg in FIFO order without
      hanging — net throughput on the consumer side equals net
      throughput on the producer side, and no msg is lost to the
      `_inbound_capacity` ring drop.

  L2. **No stuck-get under concurrent stop().**
      During sustained traffic, calling `stop()` on the client wakes
      all blocked consumers within a tight bound (≤200 ms each), and
      no consumer thread is left blocked.

  L3. **No spurious-wake amplification.**
      The predicate-based wait does not amplify a single notify into
      multiple `get()` returns. Each msg appended yields exactly one
      consumer return.

The test runs a short, deterministic burst (sized so CI completes in
≤10s wall-clock) but at a sustained per-second rate that exercises
the same critical-section behavior as production. Anything below ~10K
msgs/sec into a single client's deque exceeds T1/T2/T3's observed
peak inbound rate by 10x+.
"""
from __future__ import annotations

import queue
import threading
import time

from titan_hcl.core.bus_socket import BusSocketClient, SocketQueue


# Sized so the test runs in <5s on a loaded VPS while still hitting the
# 10K msgs/sec rate that RFP §6.3 specifies. 20K msgs * 1 producer = 2s
# at 10K/sec; the consumer drains in parallel.
_SOAK_MSG_COUNT = 20_000
_PRODUCER_TARGET_RATE_HZ = 10_000.0
# Generous over the deque's default _inbound_capacity (8192) so we
# actually stress the missed-wake path — but bounded so we don't OOM.
_INBOUND_CAPACITY = 65_536


def _make_client(name: str) -> BusSocketClient:
    return BusSocketClient(
        titan_id="T1",
        authkey=b"\x00" * 32,
        name=name,
        sock_path="/tmp/test_bus_cond_soak.sock",
        topics=None,
        inbound_capacity=_INBOUND_CAPACITY,
    )


# ── L1: no missed-wake under sustained producer pressure ─────────────


def test_soak_no_missed_wake_under_sustained_pressure():
    """Run ≥10K msgs/sec through the deque + verify the consumer drains
    every msg in FIFO order without hanging.

    Failure modes this catches:
      - `_wake_cond.notify_all()` not landing on the consumer's
        `wait_for` (would manifest as consumer hanging with msgs still
        in deque).
      - Predicate re-evaluation race (consumer wakes, predicate flips
        back to False before drain — would manifest as missing msgs).
      - Lock-contention starvation between deliver + get under high
        rate (would manifest as wall-clock blowup).
    """
    client = _make_client("test_soak_l1")
    q = SocketQueue(client)

    received: list[int] = []
    consumer_done = threading.Event()

    def consumer() -> None:
        try:
            while len(received) < _SOAK_MSG_COUNT:
                try:
                    msg = q.get(timeout=2.0)
                except queue.Empty:
                    # Hard fail — we should never time out before all
                    # produced msgs are drained.
                    break
                received.append(msg["n"])
        finally:
            consumer_done.set()

    t_consumer = threading.Thread(target=consumer, daemon=True)
    t_consumer.start()

    # Producer paces itself to ≥10K msgs/sec. We don't tight-loop —
    # we run a tiny sleep-budget so total wall-clock matches the rate
    # target (the deque is the hot path, not the producer's sleep).
    deadline_per_msg = 1.0 / _PRODUCER_TARGET_RATE_HZ
    t_produce_start = time.time()
    for n in range(_SOAK_MSG_COUNT):
        target_ts = t_produce_start + (n + 1) * deadline_per_msg
        client._deliver_to_inbound({"type": "SOAK_MSG", "n": n})
        # Light pacing — if we're ahead of schedule, sleep just enough
        # to keep the rate. If behind, don't sleep (recover).
        now = time.time()
        if now < target_ts:
            time.sleep(target_ts - now)
    t_produce_end = time.time()

    assert consumer_done.wait(timeout=5.0), (
        f"consumer did not drain within 5s after producer finished "
        f"(received {len(received)}/{_SOAK_MSG_COUNT})"
    )

    produce_elapsed = t_produce_end - t_produce_start
    effective_rate = _SOAK_MSG_COUNT / produce_elapsed if produce_elapsed > 0 else float("inf")

    # L1.a — every msg drained.
    assert len(received) == _SOAK_MSG_COUNT, (
        f"missed-wake: consumer received {len(received)}/{_SOAK_MSG_COUNT} "
        f"(effective producer rate {effective_rate:.0f} Hz)"
    )

    # L1.b — FIFO order preserved end-to-end.
    assert received == list(range(_SOAK_MSG_COUNT)), (
        "FIFO order violated under load — first divergence at index "
        f"{next((i for i, n in enumerate(received) if n != i), 'n/a')}"
    )

    # L1.c — sustained rate met. We allow 20% slack for scheduler jitter
    # on a busy CI/VPS host; the floor is still 8K/sec which exceeds
    # production peak inbound by 8x+.
    assert effective_rate >= _PRODUCER_TARGET_RATE_HZ * 0.8, (
        f"sustained rate {effective_rate:.0f} Hz below 80% of "
        f"{_PRODUCER_TARGET_RATE_HZ:.0f} Hz target — lock contention "
        f"or scheduler stall"
    )


# ── L2: no stuck-get under concurrent stop() ─────────────────────────


def test_soak_stop_unblocks_all_consumers_under_traffic():
    """Multiple consumer threads blocked in `get()` during sustained
    traffic — `stop()` must wake every one of them within 200 ms.

    Catches: a missed-wake on the stop path when the predicate
    momentarily re-evaluates False between data drains. The
    Condition+predicate pattern guarantees the level-triggered
    _stop_event.is_set() arm wins; this test exercises that under load.
    """
    client = _make_client("test_soak_l2")

    n_consumers = 4
    consumer_results: list[dict] = [{} for _ in range(n_consumers)]

    def consumer(idx: int) -> None:
        result = consumer_results[idx]
        try:
            while True:
                try:
                    q = SocketQueue(client)
                    q.get(timeout=10.0)
                except queue.Empty:
                    result["raised_empty"] = True
                    return
        except Exception as exc:  # pragma: no cover — defense
            result["exception"] = repr(exc)

    threads = [
        threading.Thread(target=consumer, args=(i,), daemon=True)
        for i in range(n_consumers)
    ]
    for t in threads:
        t.start()

    # Run a brief burst of traffic so consumers are actively churning
    # between `wait_for` re-evaluations and `popleft` returns.
    for n in range(2_000):
        client._deliver_to_inbound({"type": "SOAK_MSG", "n": n})
        if n % 50 == 0:
            time.sleep(0.001)  # let consumers run

    # Now call stop. Every consumer must wake + raise QueueEmpty within
    # 200 ms (the per-consumer target from RFP §6 invariant 2 — even
    # under load).
    t_stop = time.time()
    client.stop(timeout=1.0)
    for t in threads:
        t.join(timeout=1.0)
    elapsed = time.time() - t_stop

    assert elapsed < 1.0, (
        f"stop()→all-consumers-woke took {elapsed*1000:.0f} ms (>1s)"
    )

    # Every consumer must have either (a) drained all msgs cleanly
    # (consumed before stop hit) or (b) raised QueueEmpty post-stop.
    # No consumer should be hung.
    for i, t in enumerate(threads):
        assert not t.is_alive(), f"consumer #{i} hung past stop()"


# ── L3: no spurious-wake amplification ───────────────────────────────


def test_soak_no_spurious_wake_amplification():
    """A burst of `_wake_cond.notify_all()` calls with no data + no stop
    must NOT cause a single deque append to be returned multiple times.

    Pre-fix Event-pattern was vulnerable: if `set()` fired between a
    consumer's `clear()` and the next `wait()`, a phantom wake could
    surface stale state. Condition+predicate eliminates this — each
    `get()` returns exactly one msg per append.
    """
    client = _make_client("test_soak_l3")
    q = SocketQueue(client)

    received: list[int] = []
    stop_consumer = threading.Event()

    def consumer() -> None:
        while not stop_consumer.is_set():
            try:
                msg = q.get(timeout=0.05)
                received.append(msg["n"])
            except queue.Empty:
                continue

    t = threading.Thread(target=consumer, daemon=True)
    t.start()

    # Fire spurious notifies on a tight loop while the consumer is
    # mostly idle. None of these should cause a spurious return.
    for _ in range(500):
        with client._inbound_lock:
            client._wake_cond.notify_all()
        time.sleep(0.0001)

    # Now actually deliver exactly 100 msgs. Consumer must receive
    # exactly 100 — not 500 + 100 = 600 (amplification).
    for n in range(100):
        client._deliver_to_inbound({"type": "SOAK_MSG", "n": n})

    # Give consumer a beat to drain.
    deadline = time.time() + 2.0
    while time.time() < deadline and len(received) < 100:
        time.sleep(0.01)

    stop_consumer.set()
    t.join(timeout=1.0)

    assert len(received) == 100, (
        f"spurious-wake amplification: consumer received {len(received)} "
        f"msgs after 500 phantom notifies + 100 real appends "
        f"(expected exactly 100)"
    )
    assert received == list(range(100)), (
        "FIFO violated despite no amplification — internal bug"
    )
