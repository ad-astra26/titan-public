"""Unit tests for Phase B per-module hot-reload — SPEC §8.3 + §11.B.3.

Covers the Guardian-side orchestrator without spawning real subprocesses:
- Validation (Step 1): unknown module / not running / reload in-flight all
  return status="failed" with the right reason.
- Idempotency (rFP §4.4): re-issuing during in-flight returns status="failed"
  reason="reload_in_flight".
- Suppression contract (SPEC §11.B.3): restart_async returns None +
  monitor_tick skips restart paths while info.reload_in_flight=True.
- Message routing: MODULE_READY + BUS_WORKER_ADOPT_REQUEST routed to the
  reload orchestrator's queues when matching in-flight state.
- Dispatch (_dispatch_reload_request): malformed payload emits failed ACK
  without spawning.
- ACK shape: _emit_reload_ack publishes a SPEC §8.3-conformant payload.

Full subprocess integration (happy path + rollback through real spawn)
requires worker-side ADOPTION_REQUEST emission on Phase B reload which is
Chunk 2.5 follow-up — see rFP §4 + commit history.
"""
from __future__ import annotations

import asyncio
import threading
import time
from unittest.mock import MagicMock

import pytest

from titan_hcl import bus
from titan_hcl.bus import (
    BUS_WORKER_ADOPT_REQUEST,
    DivineBus,
    MODULE_READY,
    MODULE_RELOAD_ACK,
    MODULE_RELOAD_REQUEST,
    make_msg,
)
from titan_hcl.guardian_hcl import (
    Guardian,
    ModuleInfo,
    ModuleSpec,
    ModuleState,
    ReloadState,
)


def _spec(name: str) -> ModuleSpec:
    return ModuleSpec(
        name=name, layer="L3", entry_fn=lambda *a, **kw: None,
        autostart=False, restart_on_crash=True,
    )


def _make_guardian_with_running_module(name: str = "fake_worker"):
    """Build a Guardian + registered module in artificial RUNNING state
    (no real subprocess). Returns (guardian, info).
    """
    g = Guardian(DivineBus())
    g.register(_spec(name))
    info = g._modules[name]
    info.state = ModuleState.RUNNING
    info.pid = 99999  # synthetic
    info.process = MagicMock()
    info.process.is_alive.return_value = True
    info.last_heartbeat = time.time()
    return g, info


# ── Validation (Step 1) — async public API ───────────────────────────────


def test_reload_unknown_module_returns_failed():
    g = Guardian(DivineBus())
    result = asyncio.run(g.reload_module("does_not_exist"))
    assert result["status"] == "failed"
    assert result["reason"] == "unknown_module"
    assert result["module_name"] == "does_not_exist"
    assert "swap_id" in result
    assert isinstance(result["total_elapsed_ms"], int)
    g.stop_all()


def test_reload_not_running_returns_failed():
    g = Guardian(DivineBus())
    g.register(_spec("not_yet_started"))
    # state is STOPPED by default
    result = asyncio.run(g.reload_module("not_yet_started"))
    assert result["status"] == "failed"
    assert result["reason"].startswith("not_running:state=")
    g.stop_all()


def test_reload_in_flight_returns_failed_with_reload_in_flight_reason():
    g, info = _make_guardian_with_running_module()
    # Simulate an in-flight reload already registered
    g._reloads_in_flight["fake_worker"] = ReloadState(
        swap_id="existing", module_name="fake_worker",
        old_pid=99999, new_module_path=None, started_ts=time.time(),
    )
    info.reload_in_flight = True

    result = asyncio.run(g.reload_module("fake_worker"))
    assert result["status"] == "failed"
    assert result["reason"] == "reload_in_flight"
    g.stop_all()


def test_reload_signature_matches_rfp_phase_b_44():
    """rFP §4.4 signature: async def reload_module(self, module_name,
    new_module_path=None, timeout_s=30.0) -> dict."""
    import inspect
    sig = inspect.signature(Guardian.reload_module)
    params = list(sig.parameters.keys())
    assert params == ["self", "module_name", "new_module_path", "timeout_s"]
    assert sig.parameters["new_module_path"].default is None
    assert sig.parameters["timeout_s"].default == 30.0
    assert asyncio.iscoroutinefunction(Guardian.reload_module)


# ── §11.B.3 suppression contract ─────────────────────────────────────────


def test_restart_async_returns_none_during_reload_in_flight():
    """SPEC §11.B.3 — restart_async is skipped when info.reload_in_flight."""
    g, info = _make_guardian_with_running_module()
    info.reload_in_flight = True
    # Mock restart to fail loud if called
    g.restart = MagicMock(side_effect=AssertionError("restart called!"))

    fut = g.restart_async("fake_worker", reason="heartbeat_timeout")
    assert fut is None
    g.restart.assert_not_called()
    g.stop_all()


def test_restart_async_works_when_reload_not_in_flight():
    """Sanity baseline — without reload_in_flight, restart_async works."""
    g, info = _make_guardian_with_running_module()
    g.restart = MagicMock(return_value=True)

    fut = g.restart_async("fake_worker", reason="test")
    assert fut is not None
    fut.result(timeout=2.0)
    g.stop_all()


def test_monitor_tick_skips_dead_process_restart_during_reload():
    """SPEC §11.B.3 — monitor_tick MUST skip restart paths while
    reload_in_flight=True even if process appears dead."""
    g, info = _make_guardian_with_running_module()
    info.reload_in_flight = True
    info.process.is_alive.return_value = False  # process "died"

    # Mock restart_async to detect any unwanted invocation
    g.restart_async = MagicMock()

    g.monitor_tick()
    g.restart_async.assert_not_called(), (
        "monitor_tick must NOT issue restart for reload-in-flight module"
    )
    g.stop_all()


def test_monitor_tick_skips_heartbeat_timeout_restart_during_reload():
    g, info = _make_guardian_with_running_module()
    info.reload_in_flight = True
    # Make heartbeat ancient — would normally trigger restart
    info.last_heartbeat = time.time() - 999.0
    g.restart_async = MagicMock()

    g.monitor_tick()
    g.restart_async.assert_not_called()
    g.stop_all()


# ── Message routing into in-flight reload queues ─────────────────────────


def test_module_ready_routed_to_in_flight_ready_queue():
    g, info = _make_guardian_with_running_module()
    rs = ReloadState(
        swap_id="sw1", module_name="fake_worker",
        old_pid=99999, new_module_path=None, started_ts=time.time(),
    )
    rs.new_pid = 88888
    g._reloads_in_flight["fake_worker"] = rs
    info.reload_in_flight = True

    g.bus.publish(make_msg(
        MODULE_READY, "fake_worker", "guardian", {"src": "fake_worker"}
    ))
    g._process_guardian_messages()

    # MODULE_READY should be available on rs.ready_q
    msg = rs.ready_q.get(timeout=1.0)
    assert msg["type"] == MODULE_READY
    g.stop_all()


def test_adoption_request_routed_to_in_flight_adoption_queue_when_pid_matches():
    g, info = _make_guardian_with_running_module()
    rs = ReloadState(
        swap_id="sw1", module_name="fake_worker",
        old_pid=99999, new_module_path=None, started_ts=time.time(),
    )
    rs.new_pid = 88888
    g._reloads_in_flight["fake_worker"] = rs
    info.reload_in_flight = True

    g.bus.publish(make_msg(
        BUS_WORKER_ADOPT_REQUEST, "fake_worker", "guardian", {
            "name": "fake_worker", "pid": 88888,
            "start_method": "spawn", "boot_ts": time.time(),
        },
        rid="req-1",
    ))
    g._process_guardian_messages()

    msg = rs.adoption_q.get(timeout=1.0)
    assert msg["type"] == BUS_WORKER_ADOPT_REQUEST
    assert msg["payload"]["pid"] == 88888
    g.stop_all()


def test_adoption_request_routes_by_name_when_pid_mismatches():
    """BUG-PHASE-B-FIRST-RELOAD-ADOPTION-ROUTING-MISS-20260514 fix —
    routing in `_process_guardian_messages` is now name-only (no pid
    check), eliminating the memory-visibility race where the message
    handler thread could read rs.new_pid as the initial None while the
    orchestrator's executor-thread write was pending. Defensive pid
    validation now lives in the orchestrator's adoption_q drain loop
    (step 4 of _reload_module_sync).
    """
    g, info = _make_guardian_with_running_module()
    rs = ReloadState(
        swap_id="sw1", module_name="fake_worker",
        old_pid=99999, new_module_path=None, started_ts=time.time(),
    )
    rs.new_pid = 88888
    g._reloads_in_flight["fake_worker"] = rs

    # adopt_worker MUST NOT be called when name matches an in-flight reload
    g.adopt_worker = MagicMock(side_effect=AssertionError("must not run"))

    g.bus.publish(make_msg(
        BUS_WORKER_ADOPT_REQUEST, "fake_worker", "guardian", {
            "name": "fake_worker", "pid": 77777,  # pid mismatch — still routed
            "start_method": "spawn", "boot_ts": time.time(),
        },
        rid="req-2",
    ))
    g._process_guardian_messages()

    g.adopt_worker.assert_not_called()
    routed = rs.adoption_q.get(timeout=1.0)
    assert routed["payload"]["pid"] == 77777, (
        "frame routed verbatim — orchestrator validates pid on pickup"
    )
    g.stop_all()


def test_adoption_request_falls_through_when_name_not_in_flight():
    """When a worker by the same name is NOT in reload-in-flight,
    BUS_WORKER_ADOPT_REQUEST falls through to the cross-kernel
    adopt_worker() path (Phase B.2.1 shadow-swap behavior, unchanged)."""
    g, info = _make_guardian_with_running_module()
    # NOTE: no entry in _reloads_in_flight for "fake_worker"
    g.adopt_worker = MagicMock(return_value=False)

    g.bus.publish(make_msg(
        BUS_WORKER_ADOPT_REQUEST, "fake_worker", "guardian", {
            "name": "fake_worker", "pid": 77777,
            "start_method": "spawn", "boot_ts": time.time(),
        },
        rid="req-3",
    ))
    g._process_guardian_messages()

    g.adopt_worker.assert_called_once_with("fake_worker", 77777)
    g.stop_all()


# ── _dispatch_reload_request from MODULE_RELOAD_REQUEST ──────────────────


def test_dispatch_reload_request_malformed_emits_failed_ack():
    g = Guardian(DivineBus())
    # Subscribe to MODULE_RELOAD_ACK so we can observe what guardian emits
    ack_q = g.bus.subscribe("test_observer", types=[MODULE_RELOAD_ACK])
    # Note: ack is published with dst="all" so subscribe-all observer
    # would also see it; we use a named observer + verify the type.

    g._dispatch_reload_request(make_msg(
        MODULE_RELOAD_REQUEST, "maker_cli", "guardian", {
            # missing module_name → malformed
            "swap_id": "abc-123",
        }
    ))
    # Drain the observer queue
    found = []
    for _ in range(5):
        msgs = g.bus.drain(ack_q, max_msgs=10)
        found.extend(msgs)
        if found:
            break
        time.sleep(0.05)
    statuses = [
        m["payload"].get("status") for m in found
        if m.get("type") == MODULE_RELOAD_ACK
    ]
    assert "failed" in statuses, (
        f"expected failed ACK on malformed dispatch; got {statuses}"
    )
    failed_acks = [
        m["payload"] for m in found
        if m.get("type") == MODULE_RELOAD_ACK
        and m["payload"].get("status") == "failed"
    ]
    assert any(
        a.get("reason") == "malformed_request" for a in failed_acks
    ), f"expected malformed_request reason; got {failed_acks}"
    g.stop_all()


# ── ACK payload shape per SPEC §8.3 ──────────────────────────────────────


def test_emit_reload_ack_payload_shape_matches_spec_8_3():
    """Per SPEC §8.3 MODULE_RELOAD_ACK row:
    {swap_id, module_name, status, reason, total_elapsed_ms, ts}.
    """
    g = Guardian(DivineBus())
    observer = g.bus.subscribe("observer", types=[MODULE_RELOAD_ACK])
    started = time.time()
    g._emit_reload_ack(
        swap_id="abc-123",
        module_name="m",
        status="spawning",
        reason=None,
        started_ts=started,
    )
    # Allow the bus to deliver
    msgs = []
    for _ in range(5):
        msgs = g.bus.drain(observer, max_msgs=10)
        if msgs:
            break
        time.sleep(0.02)
    ack_msgs = [m for m in msgs if m.get("type") == MODULE_RELOAD_ACK]
    assert len(ack_msgs) == 1
    payload = ack_msgs[0]["payload"]
    assert set(payload.keys()) == {
        "swap_id", "module_name", "status", "reason",
        "total_elapsed_ms", "ts",
    }
    assert payload["swap_id"] == "abc-123"
    assert payload["module_name"] == "m"
    assert payload["status"] == "spawning"
    assert payload["reason"] is None
    assert isinstance(payload["total_elapsed_ms"], int)
    assert payload["total_elapsed_ms"] >= 0
    assert payload["ts"] >= started
    g.stop_all()


def test_reload_result_helper_matches_ack_shape():
    """The dict returned by reload_module() mirrors the ACK payload."""
    started = time.time() - 0.5
    result = Guardian._reload_result(
        "swap-1", "modA", "rolled_back", "adoption:timeout", started
    )
    assert result == {
        "swap_id": "swap-1",
        "module_name": "modA",
        "status": "rolled_back",
        "reason": "adoption:timeout",
        "total_elapsed_ms": result["total_elapsed_ms"],
        "ts": result["ts"],
    }
    assert result["total_elapsed_ms"] >= 500  # 0.5s elapsed


# ── Reload constants exposed correctly ───────────────────────────────────


def test_reload_constants_loaded_from_phase_c_constants():
    from titan_hcl._phase_c_constants import (
        ADOPTION_TIMEOUT_S,
        MODULE_RELOAD_DEFAULT_TIMEOUT_S,
        MODULE_RELOAD_HAPPY_PATH_S,
    )
    assert MODULE_RELOAD_HAPPY_PATH_S == 10.0
    assert MODULE_RELOAD_DEFAULT_TIMEOUT_S == 30.0
    assert ADOPTION_TIMEOUT_S == 30.0


def test_module_reload_constants_in_bus():
    assert bus.MODULE_RELOAD_REQUEST == "MODULE_RELOAD_REQUEST"
    assert bus.MODULE_RELOAD_ACK == "MODULE_RELOAD_ACK"


def test_module_reload_ack_in_boot_buffered_types():
    """Phase A foresight: MODULE_RELOAD_ACK must be pre-listed in
    BOOT_BUFFERED_TYPES so an initiator subscription transient lapse
    doesn't lose the terminal status."""
    from titan_hcl.core.bus_socket import BOOT_BUFFERED_TYPES
    assert "MODULE_RELOAD_ACK" in BOOT_BUFFERED_TYPES


# ── Phase 2B routing-DOWN — D-SPEC-93 v1.32.0 ───────────────────────


def test_reload_step6_module_shutdown_includes_target_pid_and_swap_id():
    """SPEC §11.B.3.1 (D-SPEC-93, v1.32.0) — Guardian's Step 6
    MODULE_SHUTDOWN publish must carry payload.target_pid=old_pid +
    payload.swap_id=reload swap_id. This is the routing-key field
    that the worker-side bus recv filter checks.

    Verifies guardian.py:1266-1271 — without it the dual-pid name-
    aliased subscription window during reload causes NEW worker to
    receive the shutdown intended for OLD → Guardian's restart:
    died_exitcode_0 cascade.
    """
    from titan_hcl.guardian_hcl import Guardian
    import inspect
    src = inspect.getsource(Guardian._reload_module_sync)
    # The Step 6 publish must include both fields explicitly.
    assert '"target_pid": old_pid' in src, (
        "guardian._reload_module_sync Step 6 must set target_pid=old_pid "
        "in the MODULE_SHUTDOWN payload (D-SPEC-93)")
    assert '"swap_id": swap_id' in src, (
        "guardian._reload_module_sync Step 6 must set swap_id in the "
        "MODULE_SHUTDOWN payload (D-SPEC-93)")
    assert '"reason": "reload"' in src, (
        "guardian._reload_module_sync Step 6 reason must be \"reload\" "
        "(D-SPEC-93 contract)")


def test_reload_step8_finalizes_state_running_after_module_ready():
    """Companion to D-SPEC-93 — Guardian._reload_module_sync Step 8 must
    explicitly set info.state = ModuleState.RUNNING after consuming
    MODULE_READY from rs.ready_q.

    Live-discovered race 2026-05-19 on T3: when NEW boots faster than
    Step 6's 10s SIGKILL grace, _process_guardian_messages sets
    state=RUNNING first; then Step 7's atomic swap unconditionally
    overwrites state=STARTING; Step 8 reads MODULE_READY from
    pre-populated ready_q and returns success but state stays STARTING
    forever. Fix: Step 8 finalizes state=RUNNING explicitly after
    successful ready_q.get.

    Source-level check: the post-ready_q block must transition info.state
    when it is not already RUNNING. We assert the source contains the
    sentinel "state finalized RUNNING" log line, which is uniquely
    produced by this finalization path.
    """
    from titan_hcl.guardian_hcl import Guardian
    import inspect
    src = inspect.getsource(Guardian._reload_module_sync)
    assert "state finalized RUNNING" in src, (
        "guardian._reload_module_sync Step 8 must finalize info.state = "
        "ModuleState.RUNNING after successful ready_q.get to defend "
        "against the Step 7 atomic-swap-overwrites-RUNNING race "
        "(D-SPEC-93 companion fix)")
    assert "info.state = ModuleState.RUNNING" in src, (
        "guardian._reload_module_sync must contain explicit state=RUNNING "
        "assignment in Step 8 post-ready path")


def test_dual_pid_name_aliased_subscription_only_target_receives_shutdown():
    """Integration test for D-SPEC-93 — the bug scenario in production.

    Reload Step 3 spawns NEW alongside OLD; both subscribe to the broker
    under the same `name`. Step 6 publishes MODULE_SHUTDOWN to
    `dst=module_name` with `payload.target_pid=old_pid`. The bus recv
    filter at `BusSocketClient._handle_inbound` drops the frame at the
    NEW client (whose os.getpid() != target_pid) so only OLD processes
    the shutdown signal cleanly.

    Without the filter (the pre-D-SPEC-93 state), BOTH NEW + OLD
    receive MODULE_SHUTDOWN → NEW exits prematurely → reload's adoption
    invariant violated → module enters crash-loop.
    """
    import os
    import unittest.mock as _mock
    from titan_hcl import bus as _bus
    from titan_hcl.core.bus_socket import BusSocketClient

    # Two clients sharing the same name (simulating the reload's dual-
    # pid window: NEW + OLD both subscribed as "knowledge").
    old_client = BusSocketClient(
        titan_id="testT", authkey=b"k" * 32, name="knowledge")
    new_client = BusSocketClient(
        titan_id="testT", authkey=b"k" * 32, name="knowledge")

    fake_old_pid = os.getpid()
    fake_new_pid = os.getpid() + 1

    # Guardian Step 6 publish: target_pid=old_pid.
    shutdown_frame = {
        "type": _bus.MODULE_SHUTDOWN,
        "src": "guardian",
        "dst": "knowledge",
        "payload": {
            "reason": "reload",
            "target_pid": fake_old_pid,
            "swap_id": "swap-integration-test",
        },
    }

    # OLD pid client → os.getpid() == target_pid → frame delivered.
    old_client._handle_inbound(shutdown_frame)
    with old_client._inbound_lock:
        old_types = [m.get("type") for m in old_client._inbound]
    assert old_types == [_bus.MODULE_SHUTDOWN], (
        "OLD pid client must RECEIVE the MODULE_SHUTDOWN intended for it")

    # NEW pid client → simulate fake_new_pid via os.getpid patch → drop.
    with _mock.patch("titan_hcl.core.bus_socket.os.getpid",
                     return_value=fake_new_pid):
        new_client._handle_inbound(shutdown_frame)
    with new_client._inbound_lock:
        new_types = [m.get("type") for m in new_client._inbound]
    assert new_types == [], (
        "NEW pid client must DROP the MODULE_SHUTDOWN intended for OLD "
        "(D-SPEC-93 — prevents the dual-pid fanout crash-loop)")
