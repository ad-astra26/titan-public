"""
Tests for titan_plugin/bus_specs.py — declarative priority + coalesce table.

Covers:
- BusMsgSpec dataclass: frozen, hashable, sane defaults
- get_spec lookup: known type returns its spec, unknown returns DEFAULT_SPEC
- coalesce_key extraction: tuple of fields, missing fields → None entry, never raises
- Priority bounds: every shipped spec is in [0, 3]
- Drift detection: audit_against_bus_constants finds zero issues against
  current bus.py constants (the table never claims a constant that doesn't exist)
- P0/P1/P3 spec correctness (P2 is the implicit default)
"""
from __future__ import annotations

import pytest

from titan_plugin.bus_specs import (
    DEFAULT_SPEC,
    MSG_SPECS,
    BusMsgSpec,
    all_priorities_in_range,
    audit_against_bus_constants,
    coalesce_key,
    get_spec,
)


# ── BusMsgSpec dataclass shape ─────────────────────────────────────────────


def test_busmsgspec_is_frozen():
    """Frozen dataclass — broker assumes specs are immutable across hot-path reads."""
    spec = BusMsgSpec("FOO")
    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        spec.priority = 99  # type: ignore[misc]


def test_busmsgspec_is_hashable():
    """Specs must be hashable — required if they ever land in sets / dict keys."""
    spec = BusMsgSpec("FOO", priority=1, coalesce=("src", "type"))
    {spec}  # must not raise
    assert hash(spec) == hash(BusMsgSpec("FOO", priority=1, coalesce=("src", "type")))


def test_busmsgspec_default_priority_is_2():
    """P2 default = drop-oldest under pressure (the safe default for events)."""
    spec = BusMsgSpec("FOO")
    assert spec.priority == 2
    assert spec.coalesce is None


# ── get_spec lookup ────────────────────────────────────────────────────────


def test_get_spec_known_type_returns_table_entry():
    spec = get_spec("EPOCH_TICK")
    assert spec.name == "EPOCH_TICK"
    assert spec.priority == 0


def test_get_spec_unknown_type_returns_default():
    spec = get_spec("NEVER_REGISTERED_FAKE_TYPE_zzz")
    assert spec is DEFAULT_SPEC
    assert spec.priority == 2
    assert spec.coalesce is None


def test_default_spec_is_p2_no_coalesce():
    assert DEFAULT_SPEC.priority == 2
    assert DEFAULT_SPEC.coalesce is None


# ── Coalesce key extraction ────────────────────────────────────────────────


def test_coalesce_key_none_when_spec_has_no_coalesce():
    spec = get_spec("EPOCH_TICK")
    assert spec.coalesce is None
    assert coalesce_key(spec, {"src": "x", "type": "EPOCH_TICK"}) is None


def test_coalesce_key_tuple_when_spec_specifies():
    spec = get_spec("BODY_STATE")
    assert spec.coalesce == ("src", "type")
    msg = {"src": "body_worker", "type": "BODY_STATE", "payload": {"x": 1}}
    assert coalesce_key(spec, msg) == ("body_worker", "BODY_STATE")


def test_coalesce_key_missing_field_yields_none_entry_no_raise():
    """Broker hot path must NEVER raise on coalesce computation."""
    spec = BusMsgSpec("FOO", priority=2, coalesce=("src", "missing_field"))
    msg = {"src": "x", "type": "FOO"}  # 'missing_field' not present
    assert coalesce_key(spec, msg) == ("x", None)


# ── Priority bounds ────────────────────────────────────────────────────────


def test_all_shipped_priorities_in_range():
    """Every spec in the shipped MSG_SPECS table must have priority in [0, 3]."""
    issues = all_priorities_in_range()
    assert issues == [], f"priority issues: {issues}"


def test_default_spec_priority_in_range():
    assert 0 <= DEFAULT_SPEC.priority <= 3


# ── Drift detection ────────────────────────────────────────────────────────


def test_audit_against_bus_constants_clean():
    """Every spec.name must correspond to a constant in bus.py with matching value.

    This is the main drift guard — surfaces typos, renames, or stale entries
    immediately. New entries that depend on constants added in later chunks
    must NOT be in MSG_SPECS until those constants land.
    """
    issues = audit_against_bus_constants()
    assert issues == [], f"bus_specs ↔ bus.py drift: {issues}"


# ── Priority class spot-checks ─────────────────────────────────────────────


def test_p0_lifecycle_messages_present():
    """Critical lifecycle messages must be P0 — never droppable under any pressure."""
    for name in ("EPOCH_TICK", "MODULE_HEARTBEAT", "MODULE_READY",
                 "MODULE_SHUTDOWN", "MODULE_CRASHED"):
        assert get_spec(name).priority == 0, f"{name} must be P0"


def test_p1_trinity_state_coalesces():
    """Body/Mind/Spirit state updates coalesce by (src, type) — graceful design."""
    for name in ("BODY_STATE", "MIND_STATE", "SPIRIT_STATE", "OUTER_OBSERVATION"):
        spec = get_spec(name)
        assert spec.priority == 1, f"{name} should be P1 (Trinity-tier)"
        assert spec.coalesce == ("src", "type"), f"{name} should coalesce by (src, type)"


def test_p3_observatory_event_drops_newest():
    """OBSERVATORY_EVENT is P3 — under hard pressure, drop newest (let queued land)."""
    spec = get_spec("OBSERVATORY_EVENT")
    assert spec.priority == 3


def test_table_has_no_p_outside_range():
    """No spec should accidentally have priority > 3 or < 0."""
    for name, spec in MSG_SPECS.items():
        assert 0 <= spec.priority <= 3, f"{name} priority={spec.priority} OOR"


# ── Phase B.2.1 supervision-transfer messages ─────────────────────────────


def test_b2_1_adoption_messages_registered_p0():
    """BUS_WORKER_ADOPT_REQUEST/ACK + BUS_HANDOFF_CANCELED are kernel-critical (P0)."""
    for name in ("BUS_WORKER_ADOPT_REQUEST", "BUS_WORKER_ADOPT_ACK",
                 "BUS_HANDOFF_CANCELED"):
        spec = get_spec(name)
        assert spec.priority == 0, f"{name} must be P0 (got {spec.priority})"
        assert spec.coalesce is None, f"{name} must NOT coalesce (adoption is per-event)"


def test_b2_1_constants_match_bus_module():
    """B.2.1 constants in bus.py match their bus_specs.MSG_SPECS keys."""
    from titan_plugin import bus
    for name in ("BUS_WORKER_ADOPT_REQUEST", "BUS_WORKER_ADOPT_ACK",
                 "BUS_HANDOFF_CANCELED"):
        assert hasattr(bus, name), f"bus.py missing constant {name}"
        assert getattr(bus, name) == name, f"bus.{name} value should equal '{name}'"
        assert name in MSG_SPECS, f"bus_specs.MSG_SPECS missing {name}"


def test_b2_1_audit_finds_zero_drift():
    """audit_against_bus_constants stays clean after B.2.1 additions."""
    issues = audit_against_bus_constants()
    assert issues == [], f"drift detected: {issues}"


def test_b2_1_priorities_in_range():
    """B.2.1 specs respect priority bounds (re-check via all_priorities_in_range)."""
    issues = all_priorities_in_range()
    assert issues == [], f"B.2.1 priority bounds violation: {issues}"
