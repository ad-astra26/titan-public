"""
test_guardian_hcl_client_cache — GuardianHCLClient event-driven cache.

Phase 6 / SPEC §11.B.4 / D-SPEC-135 / v1.62.0. The thin client (no real
Guardian) serves status reads from a cache populated by bus events. This
test exercises the dispatcher loop without booting any subprocess.
"""
import time

import pytest

from titan_hcl.bus import (
    DivineBus, MODULE_CRASHED, MODULE_READY, SUPERVISION_CHILD_RESTARTED,
    make_msg,
)
from titan_hcl.guardian_hcl_client import GuardianHCLClient


def _wait_for(predicate, timeout: float = 2.0, step: float = 0.05) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(step)
    return False


@pytest.fixture
def client_with_bus():
    bus = DivineBus(maxsize=1000)
    client = GuardianHCLClient(bus)
    yield bus, client
    client.shutdown()


def test_module_ready_marks_running(client_with_bus):
    bus, client = client_with_bus
    bus.publish(make_msg(
        MODULE_READY, src="guardian", dst="guardian_hcl_client_cache",
        payload={"name": "memory"},
    ))
    assert _wait_for(lambda: client.is_running("memory")), (
        "GuardianHCLClient cache should mark module RUNNING within 2s of "
        "MODULE_READY")


def test_module_crashed_marks_not_running(client_with_bus):
    bus, client = client_with_bus
    bus.publish(make_msg(
        MODULE_READY, src="guardian", dst="guardian_hcl_client_cache",
        payload={"name": "rl"},
    ))
    assert _wait_for(lambda: client.is_running("rl"))
    bus.publish(make_msg(
        MODULE_CRASHED, src="guardian", dst="guardian_hcl_client_cache",
        payload={"name": "rl"},
    ))
    assert _wait_for(lambda: not client.is_running("rl"))


def test_restart_count_increments(client_with_bus):
    bus, client = client_with_bus
    for _ in range(3):
        bus.publish(make_msg(
            SUPERVISION_CHILD_RESTARTED, src="guardian",
            dst="guardian_hcl_client_cache",
            payload={"name": "spirit"},
        ))
    assert _wait_for(
        lambda: client._modules.get("spirit", {}).get("restart_count", 0) == 3
    )
