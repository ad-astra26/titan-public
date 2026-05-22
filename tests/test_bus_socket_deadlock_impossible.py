"""
SPEC §8.0.ter deadlock-impossible regression test.

This is the test that EXISTS specifically because the pre-§8.0.ter code
silently wedged T3's L2 Python plugin event loop for 13.5h on 2026-05-13
→ 14. py-spy diagnosed MainThread frozen in `send_frame` from
`_heartbeat_loop` (kernel.py:749) inside an asyncio coroutine. Every other
orchestrator loop (`_meditation_loop`, `_publish_outer_sources_loop`,
`_guardian_handler_loop`, `_sovereignty_loop`) was frozen too — they all
share the same asyncio event loop.

Pre-fix: a slow broker can wedge the entire asyncio event loop because
synchronous `publish()` blocks on `sock.sendall()`.

Post-fix: `publish()` returns after enqueue; the writer thread is the
sole socket-toucher; the asyncio event loop is structurally immune to
broker backpressure.

This test:
  1. Sets up an asyncio loop on the main thread.
  2. Inside an asyncio task, publishes to a BusSocketClient whose
     send_frame is patched to block forever.
  3. Concurrently runs a sibling task that simply sleeps 0.2s.
  4. Asserts: the sibling task completes within a generous timeout.

Pre-§8.0.ter behavior: sibling task NEVER completes (event loop frozen
on send_frame in the publish task). Test FAILS.
Post-§8.0.ter behavior: publish task returns immediately after enqueue;
event loop is free to run the sibling task. Test PASSES.

Verified by the rFP §6 acceptance gate: this test was checked to FAIL
against the pre-fix code before locking the post-fix pass.
"""
from __future__ import annotations

import asyncio
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from titan_hcl.core.bus_socket import BusSocketClient


def _make_client(name: str = "deadlock_test") -> BusSocketClient:
    return BusSocketClient(
        titan_id="T1",
        authkey=b"\x00" * 32,
        name=name,
        sock_path="/tmp/test_bus_deadlock.sock",
        topics=None,
    )


@pytest.mark.asyncio
async def test_main_thread_publish_during_broker_stall():
    """SPEC §8.0.ter deadlock-impossible regression.

    Reproduces the exact T3 2026-05-14 deadlock scenario in miniature:
      - asyncio event loop on MainThread
      - coroutine publishes to BusSocketClient
      - broker (mocked send_frame) blocks forever
      - sibling coroutine must still complete

    Pre-§8.0.ter (BUG-T3-PARENT-DEADLOCK): publish call blocks MainThread
    in sock.sendall(), sibling coroutine starves, asyncio loop is dead.
    Post-§8.0.ter: publish enqueues + returns; MainThread free; sibling
    completes.
    """
    client = _make_client()

    # Inject a fake live socket so the writer thread has something
    # to (try to) send through. The writer will block on send_frame
    # forever — that's the point. The asyncio MainThread MUST NOT
    # share that fate.
    fake_sock = MagicMock()
    client._sock = fake_sock

    # send_frame blocks forever — simulates a totally stalled broker.
    # The writer thread will be wedged here; the publisher (asyncio
    # MainThread) MUST NOT be.
    stall_event = threading.Event()

    def block_forever(sock, payload):
        stall_event.wait()  # never set during the test — blocks indefinitely

    sibling_completed = asyncio.Event()

    async def publisher_task():
        """Inside the asyncio MainThread loop — publishes once.
        Pre-fix this would call sock.sendall() and never return,
        wedging the loop."""
        # Use direct client.publish — same call path as the kernel's
        # _heartbeat_loop that triggered the production deadlock.
        client.publish({
            "type": "DEADLOCK_TEST",
            "src": "publisher_task",
            "dst": "guardian",
            "payload": {"trigger": "deadlock_regression"},
        })

    async def sibling_task():
        """Independent coroutine that just sleeps. Must complete
        regardless of what the publisher task does to the bus."""
        await asyncio.sleep(0.2)
        sibling_completed.set()

    with patch(
        "titan_hcl.core.bus_socket.send_frame",
        side_effect=block_forever,
    ):
        # Start the writer thread — it'll block on send_frame and
        # NEVER drain. That's exactly the production failure mode.
        client._writer_thread = threading.Thread(
            target=client._writer_loop, daemon=True,
            name=f"bus-writer-{client.name}")
        client._writer_thread.start()
        try:
            # Run BOTH tasks on the SAME asyncio loop. The whole
            # point of this test is that the publisher task does not
            # wedge the loop, so the sibling task can complete.
            await asyncio.wait_for(
                asyncio.gather(publisher_task(), sibling_task()),
                timeout=2.0,
            )
        finally:
            # Unblock the wedged writer thread so it can clean up.
            stall_event.set()
            client._stop_event.set()
            client._outbound_event.set()
            client._writer_thread.join(timeout=2.0)

    assert sibling_completed.is_set(), (
        "SPEC §8.0.ter VIOLATION: asyncio MainThread was wedged by "
        "publish() during broker stall. Sibling task never completed. "
        "This is the exact failure mode py-spy diagnosed on T3 PID 3827138 "
        "at 2026-05-14 05:48 UTC (13.5h hang). The fix MUST ensure "
        "publish() returns after enqueue, NEVER blocks on socket I/O "
        "from the caller's thread."
    )


@pytest.mark.asyncio
async def test_main_thread_loop_remains_responsive_under_sustained_publish_storm():
    """Variant of the deadlock test with a publish STORM (100 publishes
    in tight loop) against a stalled broker. Pre-fix the first publish
    would wedge; sibling never runs. Post-fix all 100 enqueue + return;
    sibling runs in interleaved fashion with the storm."""
    client = _make_client()
    fake_sock = MagicMock()
    client._sock = fake_sock

    stall_event = threading.Event()

    def block_forever(sock, payload):
        stall_event.wait()

    sibling_counter = {"count": 0}

    async def publisher_task():
        for i in range(100):
            client.publish({
                "type": "STORM",
                "src": "publisher",
                "dst": "y",
                "payload": {"i": i},
            })
            # Tiny yield so the loop has a chance to schedule the sibling.
            await asyncio.sleep(0)

    async def sibling_task():
        for _ in range(10):
            await asyncio.sleep(0.01)
            sibling_counter["count"] += 1

    with patch(
        "titan_hcl.core.bus_socket.send_frame",
        side_effect=block_forever,
    ):
        client._writer_thread = threading.Thread(
            target=client._writer_loop, daemon=True,
            name=f"bus-writer-{client.name}")
        client._writer_thread.start()
        try:
            await asyncio.wait_for(
                asyncio.gather(publisher_task(), sibling_task()),
                timeout=2.0,
            )
        finally:
            stall_event.set()
            client._stop_event.set()
            client._outbound_event.set()
            client._writer_thread.join(timeout=2.0)

    assert sibling_counter["count"] == 10, (
        f"Sibling task only ticked {sibling_counter['count']}/10 — the "
        f"asyncio loop was wedged by the publish storm. SPEC §8.0.ter "
        f"requires publish() to return after enqueue without blocking."
    )
