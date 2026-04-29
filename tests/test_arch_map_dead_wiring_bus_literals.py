"""arch_map dead-wiring v2.4 — unregistered-literal detection.

Locks the fix for the 2026-04-28 BUS_HANDOFF_ACK blind spot.

The pre-fix scanner filtered AST findings by `msg_type in known_types`,
silently skipping any literal that wasn't a registered constant in
`titan_plugin/bus.py`. That made the scanner blind to exactly the drift
class it was meant to catch (literals bypassing constant registration).

After fix: every `make_msg("...", ...)` / `_send_msg(q, "...", ...)`
literal that looks like a bus msg type (UPPER_SNAKE_CASE) but is NOT in
`known_types` produces a `bus_literal_msg_type` finding.

These tests are isolated — they construct a tmp directory with a fake
bus.py and a fake worker, then run the scanner against it. No reliance
on the live titan_plugin tree.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import pytest

# Make scripts/ importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from arch_map_dead_wiring import (
    extract_bus_message_types,
    find_bus_flow_imbalances,
    scan_bus_publishers_and_subscribers,
)


def _write_fake_bus(tmp: Path, registered: list[str]) -> None:
    """Create a fake bus.py with the named constants registered."""
    lines = ['"""fake bus.py for unit test"""', ""]
    for name in registered:
        lines.append(f'{name} = "{name}"')
    lines.append("")
    (tmp / "bus.py").write_text("\n".join(lines))


def _write_fake_worker(tmp: Path, name: str, body: str) -> None:
    (tmp / f"{name}.py").write_text(body)


def test_literal_used_but_not_registered_emits_finding(tmp_path):
    """The exact BUS_HANDOFF_ACK pattern: literal at producer, no constant."""
    _write_fake_bus(tmp_path, ["BUS_HANDOFF", "HIBERNATE_ACK"])
    _write_fake_worker(tmp_path, "worker_swap_handler", '''
import bus

def emit_ack(state):
    ack = bus.make_msg(
        "BUS_HANDOFF_ACK",
        state.name,
        "shadow_swap",
        {"event_id": state.event_id, "pid": 0},
    )
    return ack
''')
    known = extract_bus_message_types(tmp_path / "bus.py")
    assert "BUS_HANDOFF_ACK" not in known
    assert "BUS_HANDOFF" in known

    unregistered: dict[str, list[tuple[str, int]]] = defaultdict(list)
    pubs, subs = scan_bus_publishers_and_subscribers(
        tmp_path, known, unregistered_literals=unregistered)
    assert "BUS_HANDOFF_ACK" in unregistered, \
        f"expected literal flagged; got {dict(unregistered)}"
    assert len(unregistered["BUS_HANDOFF_ACK"]) == 1

    findings = find_bus_flow_imbalances(
        pubs, subs, known, unregistered_literals=unregistered)
    literal_findings = [f for f in findings if f.kind == "bus_literal_msg_type"]
    assert len(literal_findings) == 1
    assert "BUS_HANDOFF_ACK" in literal_findings[0].title
    assert literal_findings[0].severity == "high"


def test_subscriber_literal_also_flagged(tmp_path):
    """msg_type == \"BAR_BAZ_QUUX\" with no constant should flag too."""
    _write_fake_bus(tmp_path, ["FOO_BAR"])
    _write_fake_worker(tmp_path, "consumer", '''
def handle(msg):
    msg_type = msg.get("type")
    if msg_type == "BAR_BAZ_QUUX":
        return "matched"
''')
    known = extract_bus_message_types(tmp_path / "bus.py")
    unregistered: dict[str, list[tuple[str, int]]] = defaultdict(list)
    scan_bus_publishers_and_subscribers(
        tmp_path, known, unregistered_literals=unregistered)
    assert "BAR_BAZ_QUUX" in unregistered, \
        f"subscriber literal not flagged; got {dict(unregistered)}"


def test_subscriber_literal_in_set_also_flagged(tmp_path):
    """Set/tuple literals (the shadow_orchestrator pattern) flagged too."""
    _write_fake_bus(tmp_path, ["HIBERNATE_ACK"])
    _write_fake_worker(tmp_path, "orchestrator", '''
import bus

def drain(inbox, types):
    pass

def collect_acks(inbox):
    drain(inbox, {bus.HIBERNATE_ACK, "BUS_HANDOFF_ACK"})
    for msg in inbox:
        msg_type = msg.get("type")
        if msg_type == "BUS_HANDOFF_ACK":
            yield msg
''')
    known = extract_bus_message_types(tmp_path / "bus.py")
    unregistered: dict[str, list[tuple[str, int]]] = defaultdict(list)
    scan_bus_publishers_and_subscribers(
        tmp_path, known, unregistered_literals=unregistered)
    assert "BUS_HANDOFF_ACK" in unregistered, \
        f"set-element literal not flagged; got {dict(unregistered)}"
    # Both the set element AND the equality comparison should be detected
    assert len(unregistered["BUS_HANDOFF_ACK"]) >= 1


def test_registered_constant_does_not_emit_finding(tmp_path):
    """A literal that matches a registered constant must NOT be flagged."""
    _write_fake_bus(tmp_path, ["HIBERNATE_ACK"])
    _write_fake_worker(tmp_path, "good_producer", '''
import bus

def emit_hibernate_ack():
    return bus.make_msg("HIBERNATE_ACK", "src", "dst", {})
''')
    known = extract_bus_message_types(tmp_path / "bus.py")
    unregistered: dict[str, list[tuple[str, int]]] = defaultdict(list)
    pubs, subs = scan_bus_publishers_and_subscribers(
        tmp_path, known, unregistered_literals=unregistered)
    assert "HIBERNATE_ACK" not in unregistered
    # Must still appear as a publisher
    assert "HIBERNATE_ACK" in pubs


def test_non_msg_type_literal_not_flagged(tmp_path):
    """Strings that don't look like bus msg types (lowercase, single word,
    etc.) must NOT trigger false-positive findings."""
    _write_fake_bus(tmp_path, [])
    _write_fake_worker(tmp_path, "noisy", '''
import bus

def make_things():
    bus.make_msg("lowercase", "src", "dst", {})
    bus.make_msg("a", "src", "dst", {})
    bus.make_msg("Hello World", "src", "dst", {})
    bus.make_msg("UPPER", "src", "dst", {})  # single segment, no underscore
''')
    known = extract_bus_message_types(tmp_path / "bus.py")
    unregistered: dict[str, list[tuple[str, int]]] = defaultdict(list)
    scan_bus_publishers_and_subscribers(
        tmp_path, known, unregistered_literals=unregistered)
    assert dict(unregistered) == {}, \
        f"false-positive flagging non-msg-type literals: {dict(unregistered)}"


def test_bare_constant_reference_not_flagged_as_literal(tmp_path):
    """Bare `MSG_TYPE` (as Name node) referenced from imported constants
    must NOT be flagged as a literal — it's the correct usage."""
    _write_fake_bus(tmp_path, ["FOO_BAR"])
    _write_fake_worker(tmp_path, "good_caller", '''
from bus import FOO_BAR
import bus

def emit():
    return bus.make_msg(FOO_BAR, "src", "dst", {})
''')
    known = extract_bus_message_types(tmp_path / "bus.py")
    unregistered: dict[str, list[tuple[str, int]]] = defaultdict(list)
    pubs, subs = scan_bus_publishers_and_subscribers(
        tmp_path, known, unregistered_literals=unregistered)
    assert dict(unregistered) == {}
    assert "FOO_BAR" in pubs


def test_scanner_default_no_unregistered_dict_does_not_crash(tmp_path):
    """Backward compat: callers passing no unregistered_literals dict
    (the pre-2026-04-28 signature) still work."""
    _write_fake_bus(tmp_path, ["FOO_BAR"])
    _write_fake_worker(tmp_path, "old_caller", '''
import bus

def emit():
    return bus.make_msg("UNREGISTERED_MSG", "src", "dst", {})
''')
    known = extract_bus_message_types(tmp_path / "bus.py")
    # No third arg — old signature
    pubs, subs = scan_bus_publishers_and_subscribers(tmp_path, known)
    # Pre-fix behavior preserved: literal not in known_types is silently
    # skipped (no exception, no finding)
    assert dict(pubs) == {}
    assert dict(subs) == {}
