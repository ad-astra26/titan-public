"""
SPEC §8.0.ter Chunk 4 — warning_monitor wiring for high-water WARN.

The BusSocketClient emits a rate-limited log line when its outbound buffer
crosses OUTBOUND_BUFFER_HIGH_WATER (1000 frames):

    [WARNING] [bus_client.<name>] outbound buffer high water: <N> frames queued
        (threshold=<H>). Writer thread may be blocked by slow broker drain.

`warning_monitor_worker` tails the brain log + groups events by the first
`[TAG]` in the message text. With our log line format, the TAG match
yields `bus_client.<name>` as the aggregation key — which then surfaces
via `/v4/warning-monitor` (read from `data/warning_monitor/state.json`).

This test PINS the integration contract: if either side drifts (log line
format changes OR TAG_RE changes), the test fails and signals that the
end-to-end backpressure visibility is broken. No new code wiring needed
for warning_monitor — the existing brain-log tail handles our line by
construction; this test is the lockdown.

Phase C T3 note: T3 logs to journald only, not /tmp/titan_brain.log. The
warning_monitor on T3 will miss these unless the worker is updated to
read from journald. That's a separate scope (out of this rFP).
"""
from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

from titan_hcl.core.bus_socket import (
    BusSocketClient,
    OUTBOUND_BUFFER_HIGH_WATER,
)
from titan_hcl.modules.warning_monitor_worker import (
    LOG_RE,
    _extract_key,
)


def _make_client(name: str = "outer_trinity") -> BusSocketClient:
    return BusSocketClient(
        titan_id="T1",
        authkey=b"\x00" * 32,
        name=name,
        sock_path="/tmp/test_high_water.sock",
        topics=None,
    )


def test_warning_monitor_picks_up_high_water_warn_key(caplog):
    """End-to-end pin: emit the high-water WARN, parse the captured log
    line through warning_monitor's own helpers, assert the aggregation
    key is `bus_client.<name>`.

    If this test breaks, either:
      - BusSocketClient's log format drifted from `[bus_client.<name>] ...`
      - warning_monitor's LOG_RE or _extract_key drifted
    Either way the /v4/warning-monitor endpoint will silently miss
    backpressure events — a SPEC §8.0.ter visibility regression.
    """
    client = _make_client(name="outer_trinity")
    caplog.set_level(logging.WARNING, logger="titan_hcl.core.bus_socket")

    # Trigger the high-water WARN by publishing past the threshold.
    for i in range(OUTBOUND_BUFFER_HIGH_WATER + 1):
        client.publish({"type": f"M{i}", "src": "x", "dst": "y", "payload": {}})

    # Find the rate-limited WARN line from BusSocketClient.
    high_water_warns = [
        r for r in caplog.records
        if "outbound buffer high water" in r.getMessage()
    ]
    assert high_water_warns, (
        "BusSocketClient must emit at least one [bus_client.<name>] "
        "outbound buffer high water WARN once depth crosses "
        f"OUTBOUND_BUFFER_HIGH_WATER ({OUTBOUND_BUFFER_HIGH_WATER})"
    )

    # The actual logger formats the line with the standard logging
    # layout. We synthesize the brain-log format manually (since
    # caplog gives us the rendered message but not the full log layout)
    # and confirm warning_monitor's parser extracts the right key.
    rendered = high_water_warns[0].getMessage()
    # Brain-log line shape: "HH:MM:SS [WARNING] <rendered>" — the
    # warning_monitor LOG_RE matches "HH:MM:SS \[LEVEL\] <rest>".
    fake_brain_log_line = f"07:20:00 [WARNING] {rendered}"
    m = LOG_RE.match(fake_brain_log_line)
    assert m is not None, (
        f"warning_monitor LOG_RE failed to parse our brain-log line: "
        f"{fake_brain_log_line!r}"
    )
    assert m.group("level") == "WARNING"
    key = _extract_key(m.group("rest"))
    assert key == "bus_client.outer_trinity", (
        f"warning_monitor must extract 'bus_client.outer_trinity' as the "
        f"aggregation key; got {key!r}. This means /v4/warning-monitor "
        f"will surface this client's backpressure events under that key."
    )


def test_warning_monitor_extract_key_handles_titan_HCL_dot_name():
    """Variant of above for the kernel's own bus client name 'titan_HCL'
    — the most important high-water source operationally (kernel publishes
    heartbeats every 10s; if blocked, the asyncio loop is wedged).
    The dot in the tag must not break key extraction."""
    rendered = ("[bus_client.titan_HCL] outbound buffer high water: 1500 "
                "frames queued (threshold=1000). Writer thread may be "
                "blocked by slow broker drain.")
    fake_line = f"07:25:00 [WARNING] {rendered}"
    m = LOG_RE.match(fake_line)
    assert m is not None
    assert _extract_key(m.group("rest")) == "bus_client.titan_HCL"


def test_warning_monitor_key_unique_per_client_name():
    """Two clients with different names must group under separate keys
    so operators can see WHICH module is backpressured."""
    for client_name, expected_key in [
        ("outer_trinity", "bus_client.outer_trinity"),
        ("memory_proxy", "bus_client.memory_proxy"),
        ("cognitive_worker", "bus_client.cognitive_worker"),
        ("titan_HCL", "bus_client.titan_HCL"),
    ]:
        rendered = f"[bus_client.{client_name}] outbound buffer high water: 1000 frames queued (threshold=1000)."
        fake_line = f"07:30:00 [WARNING] {rendered}"
        m = LOG_RE.match(fake_line)
        assert m is not None, f"LOG_RE failed for client={client_name}"
        key = _extract_key(m.group("rest"))
        assert key == expected_key, (
            f"client={client_name} → expected key={expected_key!r}, got {key!r}"
        )
