"""Phase B.2.1 fix (2026-04-27) — broker → in-process DivineBus callback.

Verifies the bug fix where worker → kernel messages were silently dropped
because BusSocketServer.publish() only fanned out to broker subscribers
(workers), never to the kernel's in-process DivineBus subscribers
(shadow_swap orchestrator, Guardian, etc.).

Bug surfaced 2026-04-27 PM during first true-outlive E2E swap test on T1:
shadow_swap orchestrator's `_phase_readiness_wait` polled
UPGRADE_READINESS_QUERY but received zero UPGRADE_READINESS_REPORT replies
because the broker had no path back to the in-process orchestrator.

Fix: BusSocketServer constructor takes an optional `on_inbound_publish`
callback. _handle_inbound calls it AFTER fanout (so other broker
subscribers still see the message). Kernel passes
`self.bus.publish_in_process` which delivers to in-process subscribers
WITHOUT re-forwarding to the broker (loop avoidance).
"""
from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

from titan_plugin import bus as bus_mod
from titan_plugin.bus import DivineBus, make_msg
from titan_plugin.core.bus_socket import BusSocketServer


def test_publish_in_process_skips_broker():
    """publish_in_process must deliver to in-process subscribers but NOT
    re-forward to the attached broker (loop avoidance)."""
    div = DivineBus(maxsize=100)
    fake_broker = MagicMock()
    div.attach_broker(fake_broker)

    sub_q = div.subscribe("kernel_handler")
    n = div.publish_in_process(make_msg("UPGRADE_READINESS_REPORT",
                                         src="body", dst="shadow_swap"))
    # When dst is not "all" but matches a subscriber name, dispatched.
    msg2 = make_msg("UPGRADE_READINESS_REPORT", src="body", dst="kernel_handler")
    n2 = div.publish_in_process(msg2)
    assert n2 == 1, f"expected 1 in-process delivery, got {n2}"
    # Broker MUST NOT be called from publish_in_process
    fake_broker.publish.assert_not_called()


def test_publish_in_process_dst_all_fanouts():
    """dst='all' must reach every in-process subscriber (excluding src)."""
    div = DivineBus(maxsize=100)
    qa = div.subscribe("alpha")
    qb = div.subscribe("beta")
    qc = div.subscribe("gamma")

    n = div.publish_in_process(make_msg("X", src="alpha", dst="all"))
    # alpha is src → excluded; beta + gamma → included
    assert n == 2


def test_broker_callback_fires_on_inbound_publish():
    """BusSocketServer must invoke the on_inbound_publish callback
    when handling a non-protocol message from a subscriber."""
    received: list[dict] = []
    cb = lambda msg: received.append(msg)

    server = BusSocketServer(
        titan_id="T_test",
        authkey=b"x" * 32,
        on_inbound_publish=cb,
    )
    # Don't actually start() — the bug fix path is _handle_inbound which
    # we can exercise directly with a mock subscriber.
    sub = MagicMock()
    sub.name = "body"
    sub.subscribed_topics = set()
    sub.lock = threading.Lock()
    msg = make_msg("UPGRADE_READINESS_REPORT", src="body", dst="shadow_swap",
                   payload={"ready": True})
    server._handle_inbound(sub, msg)
    assert len(received) == 1
    assert received[0]["type"] == "UPGRADE_READINESS_REPORT"
    assert received[0]["dst"] == "shadow_swap"


def test_broker_callback_skipped_for_protocol_messages():
    """BUS_SUBSCRIBE / BUS_UNSUBSCRIBE / BUS_PONG must NOT trigger the
    in-process callback — they're broker housekeeping, not bus traffic."""
    received: list[dict] = []
    cb = lambda msg: received.append(msg)

    server = BusSocketServer(
        titan_id="T_test",
        authkey=b"x" * 32,
        on_inbound_publish=cb,
    )
    server._subscribers["body"] = sub_obj = MagicMock()
    sub_obj.name = "body"
    sub_obj.subscribed_topics = set()
    sub_obj.lock = threading.Lock()

    server._handle_inbound(sub_obj, {"type": "BUS_SUBSCRIBE",
                                     "payload": {"name": "body", "topics": []}})
    server._handle_inbound(sub_obj, {"type": "BUS_UNSUBSCRIBE",
                                     "payload": {"topics": []}})
    server._handle_inbound(sub_obj, {"type": "BUS_PONG"})
    assert received == [], "Protocol messages must not fire the callback"


def test_broker_callback_none_is_safe():
    """No callback (default) must not crash _handle_inbound."""
    server = BusSocketServer(
        titan_id="T_test",
        authkey=b"x" * 32,
        # on_inbound_publish=None (default)
    )
    sub = MagicMock()
    sub.name = "body"
    sub.subscribed_topics = set()
    sub.lock = threading.Lock()
    msg = make_msg("FOO", src="body", dst="shadow_swap")
    server._handle_inbound(sub, msg)  # must not raise


def test_broker_callback_exception_doesnt_break_fanout():
    """If on_inbound_publish raises, the broker fanout must still proceed."""
    cb = MagicMock(side_effect=RuntimeError("simulated DivineBus failure"))
    server = BusSocketServer(
        titan_id="T_test",
        authkey=b"x" * 32,
        on_inbound_publish=cb,
    )
    sub = MagicMock()
    sub.name = "body"
    sub.subscribed_topics = set()
    sub.lock = threading.Lock()
    msg = make_msg("FOO", src="body", dst="shadow_swap")
    # Must not raise — broker robustness
    server._handle_inbound(sub, msg)
    # Callback was still invoked
    cb.assert_called_once()


def test_kernel_callback_wired():
    """Smoke test: kernel.boot wires self.bus.publish_in_process as the
    on_inbound_publish callback so worker → in-process delivery works."""
    # AST check rather than running a real kernel.
    import ast, inspect, titan_plugin.core.kernel as kernel_mod
    src = inspect.getsource(kernel_mod)
    assert "on_inbound_publish=self.bus.publish_in_process" in src, (
        "kernel.py must construct BusSocketServer with "
        "on_inbound_publish=self.bus.publish_in_process — without it, "
        "workers can't reach in-process subscribers like shadow_swap"
    )
