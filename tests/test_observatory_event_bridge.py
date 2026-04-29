"""
Tests for Microkernel v2 Phase A §A.4 (S5) — OBSERVATORY_EVENT bridge.

Validates that kernel-side bus.publish(OBSERVATORY_EVENT, ...) flows
correctly to the API subprocess's event_bus.emit() for WebSocket clients.

The actual cross-process bridge involves:
  - Kernel: publishes OBSERVATORY_EVENT with {"event_type", "data"} payload
  - Bus: routes to the api subprocess's recv_queue
  - Subprocess: bus_listener_loop translates to event_bus.emit(type, data)

This test exercises the SUBPROCESS-SIDE translation logic in isolation:
given an OBSERVATORY_EVENT message, verify event_bus.emit is called with
the correct (event_type, data) args.

Reference:
  - titan-docs/PLAN_microkernel_phase_a_s5.md §5.3
  - titan_plugin/api/api_subprocess.py:_bus_listener_loop
  - titan_plugin/bus.py:OBSERVATORY_EVENT
"""
from __future__ import annotations

import asyncio
import threading
import time
from queue import Queue, Empty

import pytest

from titan_plugin.bus import OBSERVATORY_EVENT, make_msg


def test_observatory_event_constant_registered():
    """The bus message type is registered for cross-module use."""
    assert OBSERVATORY_EVENT == "OBSERVATORY_EVENT"


def test_make_msg_with_observatory_event():
    """Bus framing helper produces correct shape for OBSERVATORY_EVENT."""
    msg = make_msg(
        OBSERVATORY_EVENT, "spirit", "api",
        {"event_type": "mood_update", "data": {"mood": "calm"}},
    )
    assert msg["type"] == "OBSERVATORY_EVENT"
    assert msg["src"] == "spirit"
    assert msg["dst"] == "api"
    assert msg["payload"]["event_type"] == "mood_update"
    assert msg["payload"]["data"] == {"mood": "calm"}


def test_bridge_translation_logic():
    """Simulate the api_subprocess bus_listener_loop translation:
    OBSERVATORY_EVENT bus message → event_bus.emit(type, data)."""
    from titan_plugin.api.events import EventBus

    event_bus = EventBus()
    captured = []

    # Subscriber that records what arrives
    sub_q = event_bus.subscribe()

    async def collect_events():
        # Consume a few events with timeout
        for _ in range(3):
            try:
                ev = await asyncio.wait_for(sub_q.get(), timeout=2.0)
                captured.append(ev)
            except asyncio.TimeoutError:
                break

    async def emit_translated():
        # Simulate: OBSERVATORY_EVENT bus messages arriving via recv_queue
        bus_msgs = [
            make_msg(OBSERVATORY_EVENT, "spirit", "api",
                     {"event_type": "sphere_pulse",
                      "data": {"clock": "inner_body", "pulse_count": 42}}),
            make_msg(OBSERVATORY_EVENT, "spirit", "api",
                     {"event_type": "dream_state",
                      "data": {"is_dreaming": True}}),
            make_msg(OBSERVATORY_EVENT, "core", "api",
                     {"event_type": "mood_update",
                      "data": {"mood": "joyful"}}),
        ]
        # Translation: extract event_type + data, call event_bus.emit
        for msg in bus_msgs:
            payload = msg.get("payload", {})
            ev_type = payload.get("event_type", "unknown")
            ev_data = payload.get("data", {})
            await event_bus.emit(ev_type, ev_data)

    async def runner():
        # Start collector first
        collect_task = asyncio.create_task(collect_events())
        await asyncio.sleep(0.1)
        # Then emit translated events
        await emit_translated()
        await collect_task

    asyncio.run(runner())

    # 3 events received in correct order with correct event_type + data
    assert len(captured) == 3
    assert captured[0]["type"] == "sphere_pulse"
    assert captured[0]["data"]["pulse_count"] == 42
    assert captured[1]["type"] == "dream_state"
    assert captured[1]["data"]["is_dreaming"] is True
    assert captured[2]["type"] == "mood_update"
    assert captured[2]["data"]["mood"] == "joyful"
