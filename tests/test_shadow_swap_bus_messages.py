"""
Tests for Microkernel v2 Phase B.1 bus message types.

Verifies the 10 new message types exist, are properly named, are unique,
and don't collide with existing types. Per feedback_bus_emit_use_constants.md
all emit/match sites must reference these constants, never string literals.

rFP: titan-docs/rFP_microkernel_v2_shadow_core.md §347-357
PLAN: titan-docs/PLAN_microkernel_phase_b1_shadow_swap.md §1
"""
import pytest

from titan_plugin import bus


# Canonical list of B.1 message-type constants (the contract).
B1_MESSAGE_CONSTANTS = (
    "SYSTEM_UPGRADE_QUEUED",
    "SYSTEM_UPGRADE_PENDING",
    "SYSTEM_UPGRADE_PENDING_DEFERRED",
    "SYSTEM_UPGRADE_STARTING",
    "SYSTEM_RESUMED",
    "UPGRADE_READINESS_QUERY",
    "UPGRADE_READINESS_REPORT",
    "HIBERNATE",
    "HIBERNATE_ACK",
    "HIBERNATE_CANCEL",
)


class TestB1MessageTypesExist:
    """Each constant is defined on the bus module and is a non-empty str."""

    @pytest.mark.parametrize("name", B1_MESSAGE_CONSTANTS)
    def test_constant_exists(self, name):
        assert hasattr(bus, name), f"bus.{name} not defined"
        value = getattr(bus, name)
        assert isinstance(value, str), f"bus.{name} must be a str, got {type(value).__name__}"
        assert value, f"bus.{name} must be non-empty"


class TestB1MessageTypesNamingConvention:
    """Constants are SCREAMING_SNAKE_CASE matching their value (canonical convention)."""

    @pytest.mark.parametrize("name", B1_MESSAGE_CONSTANTS)
    def test_value_matches_name(self, name):
        # Canonical Titan bus convention: STATE_SNAPSHOT_REQUEST = "STATE_SNAPSHOT_REQUEST"
        assert getattr(bus, name) == name, (
            f"bus.{name} value {getattr(bus, name)!r} must equal its name "
            f"(per Titan canonical convention)"
        )


class TestB1MessageTypesUniqueness:
    """No B.1 type collides with another B.1 type or any pre-existing constant."""

    def test_b1_types_unique_among_themselves(self):
        values = [getattr(bus, n) for n in B1_MESSAGE_CONSTANTS]
        assert len(values) == len(set(values)), (
            f"Duplicate values among B.1 constants: {values}"
        )

    def test_b1_types_dont_collide_with_existing(self):
        # Collect every public string-valued constant in bus that isn't a B.1 one.
        b1_set = set(B1_MESSAGE_CONSTANTS)
        existing = {}
        for attr in dir(bus):
            if attr.startswith("_") or attr in b1_set:
                continue
            value = getattr(bus, attr, None)
            # Only consider module-level UPPER_CASE str constants
            if isinstance(value, str) and attr.isupper() and attr.replace("_", "").isalnum():
                existing[attr] = value
        for name in B1_MESSAGE_CONSTANTS:
            value = getattr(bus, name)
            colliding = [k for k, v in existing.items() if v == value]
            assert not colliding, (
                f"B.1 constant {name}={value!r} collides with existing constant(s): {colliding}"
            )


class TestB1NotInStateMsgTypes:
    """B.1 messages are NOT state messages — they're routed events with rid/payload.

    State messages route through the SharedBlackboard (latest-value, no queue).
    B.1 messages need real queues + request/response (rid) semantics.
    """

    @pytest.mark.parametrize("name", B1_MESSAGE_CONSTANTS)
    def test_not_in_state_msg_types(self, name):
        from titan_plugin.bus import DivineBus
        value = getattr(bus, name)
        assert value not in DivineBus.STATE_MSG_TYPES, (
            f"B.1 type {name}={value!r} must NOT be in STATE_MSG_TYPES "
            f"(needs queue routing, not blackboard latest-value)"
        )


class TestB1MakeMsgRoundTrip:
    """make_msg builds valid envelopes with B.1 types — sanity check integration."""

    def test_make_msg_with_each_b1_type(self):
        from titan_plugin.bus import make_msg

        for name in B1_MESSAGE_CONSTANTS:
            mtype = getattr(bus, name)
            msg = make_msg(mtype, src="orchestrator", dst="all", payload={"n": 1})
            assert msg["type"] == mtype
            assert msg["src"] == "orchestrator"
            assert msg["dst"] == "all"
            assert msg["payload"] == {"n": 1}
            assert "ts" in msg
            assert isinstance(msg["ts"], float)


class TestB1NoStringLiteralsInBus:
    """Per feedback_bus_emit_use_constants.md, bus.py itself should
    only USE these strings via the constant names, never as literals
    in handler/match code (this test file is a literal-OK exception).

    This is a static-grep heuristic — guards against future regressions
    where a developer might write bus.publish(make_msg("HIBERNATE", ...))
    with a string literal instead of bus.HIBERNATE.

    Scope: only checks bus.py itself (not the whole codebase — that's
    arch_map's job). Allows the constant-definition lines themselves.
    """

    def test_bus_py_uses_constants_not_literals(self):
        from pathlib import Path
        bus_py = Path(__file__).parent.parent / "titan_plugin" / "bus.py"
        text = bus_py.read_text()
        for name in B1_MESSAGE_CONSTANTS:
            value = getattr(bus, name)
            quoted = f'"{value}"'
            # Allowed line: the constant definition (e.g. `HIBERNATE = "HIBERNATE"`)
            allowed_pattern = f"{name} = {quoted}"
            # Find every "QUOTED" occurrence
            count_quoted = text.count(quoted)
            count_in_definition = text.count(allowed_pattern)
            stray_literals = count_quoted - count_in_definition
            assert stray_literals == 0, (
                f"bus.py contains {stray_literals} string-literal "
                f"references to {value!r} outside its constant definition. "
                f"Use bus.{name} instead."
            )
