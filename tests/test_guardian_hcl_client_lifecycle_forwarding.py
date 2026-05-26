"""
test_guardian_hcl_client_lifecycle_forwarding — verify GuardianHCLClient
forwards start/stop/restart as bus publishes targeted at guardian_hcl.

Phase 6 / SPEC §11.B.4 / D-SPEC-135 / v1.62.0. We don't boot guardian_hcl
here — we subscribe to the lifecycle topics ourselves and confirm the
client publishes the right messages with the right dst routing.
"""
import time

import pytest

from titan_hcl.bus import (
    DivineBus, MODULE_RESTART_REQUEST, MODULE_START_REQUEST,
    MODULE_STOP_REQUEST,
)
from titan_hcl.guardian_hcl_client import GuardianHCLClient


@pytest.fixture
def bus_and_client():
    bus = DivineBus(maxsize=1000)
    client = GuardianHCLClient(bus)
    yield bus, client
    client.shutdown()


def _wait_for_msg(queue, mtype: str, name: str, timeout: float = 1.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            msg = queue.get(timeout=0.1)
        except Exception:
            continue
        if msg.get("type") == mtype and (
            msg.get("payload", {}).get("name") == name
        ):
            return msg
    return None


def test_start_forwards_to_guardian_hcl_lifecycle(bus_and_client):
    bus, client = bus_and_client
    queue = bus.subscribe(
        "guardian_hcl_lifecycle",
        types=[MODULE_START_REQUEST],
        reply_only=True,
    )
    client.start("memory")
    msg = _wait_for_msg(queue, MODULE_START_REQUEST, "memory", timeout=1.0)
    assert msg is not None
    assert msg["payload"]["name"] == "memory"
    assert msg.get("dst") == "guardian_hcl_lifecycle"


def test_stop_forwards_with_reason(bus_and_client):
    bus, client = bus_and_client
    queue = bus.subscribe(
        "guardian_hcl_lifecycle",
        types=[MODULE_STOP_REQUEST],
        reply_only=True,
    )
    client.stop("rl", reason="rebalance")
    msg = _wait_for_msg(queue, MODULE_STOP_REQUEST, "rl", timeout=1.0)
    assert msg is not None
    assert msg["payload"]["reason"] == "rebalance"


def test_restart_forwards_kwargs(bus_and_client):
    bus, client = bus_and_client
    queue = bus.subscribe(
        "guardian_hcl_lifecycle",
        types=[MODULE_RESTART_REQUEST],
        reply_only=True,
    )
    client.restart_module("spirit", reason="hot_reload", force=True, attempt=2)
    msg = _wait_for_msg(queue, MODULE_RESTART_REQUEST, "spirit", timeout=1.0)
    assert msg is not None
    assert msg["payload"]["reason"] == "hot_reload"
    assert msg["payload"]["force"] is True
    assert msg["payload"]["attempt"] == 2


def test_register_raises(bus_and_client):
    _, client = bus_and_client
    with pytest.raises(RuntimeError, match="not callable from the titan_hcl process"):
        client.register({"name": "anything"})
