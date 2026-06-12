"""Per-device presence for the Titan ⇄ App event channel.

RFP_titan_app_event_channel Phase 1. The phone reports its reachability on every
``POST /console/app/heartbeat`` — foreground/background, last-seen, battery — so a sender
knows whether a held long-poll exists or only the WorkManager cadence will deliver. Stored
at ``~/.titan/devices/<id>/presence.json``; sole-writer = the Console Agent (AG-EVT-1).
"""
from __future__ import annotations

import time

from .events import _device_dir, _safe_id
from .pairing import _read_json, _write_json

_STATES = {"foreground", "background", "active", "inactive"}


def put(ctx, device_id: str, hb: dict) -> dict:
    """Record a heartbeat. Returns the stored presence record."""
    _safe_id(device_id)
    hb = hb if isinstance(hb, dict) else {}
    state = hb.get("state")
    rec = {
        "state": state if state in _STATES else "unknown",
        "last_seen": time.time(),
        "battery": hb.get("battery"),
        "ack_cursor": hb.get("ack_cursor"),
    }
    _write_json(_device_dir(ctx, device_id) / "presence.json", rec)
    return rec


def get(ctx, device_id: str) -> dict | None:
    """The last heartbeat for a device, or None if it has never reported."""
    _safe_id(device_id)
    return _read_json(_device_dir(ctx, device_id) / "presence.json", None)
