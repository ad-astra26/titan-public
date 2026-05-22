"""
Test SPEC §8.5 src migration in state_register.py — D3 chunk of
rFP_phase_c_definitive_runtime_closure.

Verifies that BODY_STATE / MIND_STATE / SPIRIT_STATE handlers honor the
canonical `payload.src ∈ {inner, outer}` field added by Phase C, routing
inner publishes to inner_* fields and outer publishes (from Rust outer
trinity daemons) to outer_* fields.

Closes the missing-chunk gap noted in audit/PHASE_C_AUDIT_C-S6_20260506:
"Python state_register MIND_STATE/BODY_STATE/SPIRIT_STATE handlers ignore
payload.src; legacy OUTER_*_STATE handlers never fire because Rust
publishes canonical SPEC §8.5 events".
"""
from __future__ import annotations

import time

import pytest

from titan_hcl import bus
from titan_hcl.logic.state_register import OuterState, StateRegister


# ── Inner src (default) — preserves Phase A+B behavior ───────────────


def test_body_state_inner_default_src_routes_to_body_tensor() -> None:
    sr = OuterState()
    sr.is_active = True
    msg = {
        "type": bus.BODY_STATE,
        "payload": {
            "values": [0.6, 0.5, 0.4, 0.3, 0.7],
            "center_dist": 0.42,
        },
    }
    sr._process_bus_message(msg)
    assert sr._state["body_tensor"] == [0.6, 0.5, 0.4, 0.3, 0.7]
    assert sr._state["body_center_dist"] == 0.42
    # outer_body should be UNCHANGED (default)
    assert sr._state.get("outer_body") in ([0.5] * 5, None)


def test_mind_state_inner_default_src_routes_to_mind_tensor_15d() -> None:
    sr = OuterState()
    sr.is_active = True
    msg = {
        "type": bus.MIND_STATE,
        "payload": {
            "values": [0.6, 0.5, 0.4, 0.3, 0.7],
            "values_15d": [0.5] * 15,
        },
    }
    sr._process_bus_message(msg)
    assert sr._state["mind_tensor"] == [0.6, 0.5, 0.4, 0.3, 0.7]
    assert sr._state["mind_tensor_15d"] == [0.5] * 15
    # outer fields untouched
    assert sr._state.get("outer_mind") in ([0.5] * 5, None)
    assert sr._state.get("outer_mind_15d") is None


def test_spirit_state_inner_default_src_preserves_consciousness() -> None:
    sr = OuterState()
    sr.is_active = True
    msg = {
        "type": bus.SPIRIT_STATE,
        "payload": {
            "values": [0.6, 0.5, 0.4, 0.3, 0.7],
            "values_45d": [0.5] * 45,
            "consciousness": {
                "epoch_number": 1234,
                "drift": 0.42,
                "trajectory": 0.3,
                "curvature": 1.5,
                "density": 0.95,
            },
        },
    }
    sr._process_bus_message(msg)
    assert sr._state["spirit_tensor"] == [0.6, 0.5, 0.4, 0.3, 0.7]
    assert sr._state["spirit_tensor_45d"] == [0.5] * 45
    assert sr._state["consciousness"]["epoch_number"] == 1234
    # Outer fields untouched
    assert sr._state.get("outer_spirit") in ([0.5] * 5, None)
    assert sr._state.get("outer_spirit_45d") is None


# ── Outer src — Rust outer-trinity daemon canonical SPEC §8.5 ────────


def test_body_state_outer_src_routes_to_outer_body() -> None:
    """Rust outer-body-rs publishes BODY_STATE+src=outer with values=[5]."""
    sr = OuterState()
    sr.is_active = True
    msg = {
        "type": bus.BODY_STATE,
        "payload": {
            "src": "outer",
            "values": [0.7, 0.6, 0.5, 0.4, 0.3],
            "ts": time.time(),
        },
    }
    sr._process_bus_message(msg)
    assert sr._state["outer_body"] == [0.7, 0.6, 0.5, 0.4, 0.3]
    # Inner body_tensor should be UNCHANGED (default 0.5*5)
    assert sr._state["body_tensor"] == [0.5] * 5


def test_mind_state_outer_src_routes_15d_values_to_outer_mind_15d() -> None:
    """Rust outer-mind-rs publishes MIND_STATE+src=outer with values=[15]
    (the canonical 15D tensor — NOT a 5D base + values_15d kwarg)."""
    sr = OuterState()
    sr.is_active = True
    fifteen_distinct = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                       0.9, 0.55, 0.45, 0.35, 0.25, 0.15, 0.05]
    msg = {
        "type": bus.MIND_STATE,
        "payload": {
            "src": "outer",
            "values": fifteen_distinct,
            "ts": time.time(),
        },
    }
    sr._process_bus_message(msg)
    assert sr._state["outer_mind_15d"] == fifteen_distinct
    # outer_mind 5D base = first 5 of the 15D
    assert sr._state["outer_mind"] == fifteen_distinct[:5]
    # Inner mind_tensor unchanged
    assert sr._state["mind_tensor"] == [0.5] * 5


def test_spirit_state_outer_src_routes_45d_values_to_outer_spirit_45d() -> None:
    """Rust outer-spirit-rs publishes SPIRIT_STATE+src=outer with values=[45]
    (the canonical 45D tensor)."""
    sr = OuterState()
    sr.is_active = True
    forty_five = [round(0.1 + i * 0.02, 3) for i in range(45)]
    msg = {
        "type": bus.SPIRIT_STATE,
        "payload": {
            "src": "outer",
            "values": forty_five,
            "ts": time.time(),
        },
    }
    sr._process_bus_message(msg)
    assert sr._state["outer_spirit_45d"] == forty_five
    assert sr._state["outer_spirit"] == forty_five[:5]
    # Inner spirit_tensor + consciousness UNCHANGED
    assert sr._state["spirit_tensor"] == [0.5] * 5
    assert sr._state["consciousness"]["epoch_number"] == 0


def test_spirit_state_outer_src_does_not_overwrite_consciousness() -> None:
    """Outer src publishes from Rust have NO consciousness payload — must
    not zero out an inner-published consciousness state."""
    sr = OuterState()
    sr.is_active = True
    # Inner first → populate consciousness
    sr._process_bus_message({
        "type": bus.SPIRIT_STATE,
        "payload": {
            "values": [0.5] * 5,
            "consciousness": {"epoch_number": 999, "density": 0.88},
        },
    })
    assert sr._state["consciousness"]["epoch_number"] == 999

    # Then outer publishes 45D
    sr._process_bus_message({
        "type": bus.SPIRIT_STATE,
        "payload": {
            "src": "outer",
            "values": [0.7] * 45,
        },
    })
    # Consciousness UNTOUCHED
    assert sr._state["consciousness"]["epoch_number"] == 999
    assert sr._state["consciousness"]["density"] == 0.88
    # outer_spirit_45d populated
    assert sr._state["outer_spirit_45d"] == [0.7] * 45


# ── Willing-slice merge regression (mirror legacy OUTER_MIND_STATE behavior) ──


def test_mind_state_outer_src_preserves_willing_slice_when_set() -> None:
    """Mirror legacy OUTER_MIND_STATE willing-slice preservation:
    if existing outer_mind_15d[10:15] has any non-default value, prefer
    those over the incoming willing values."""
    sr = OuterState()
    sr.is_active = True
    # Pre-populate outer_mind_15d with custom willing slice
    sr._state["outer_mind_15d"] = [
        0.5, 0.5, 0.5, 0.5, 0.5,  # thinking
        0.5, 0.5, 0.5, 0.5, 0.5,  # feeling
        0.9, 0.8, 0.7, 0.6, 0.5,  # willing — NON-default
    ]
    incoming = [0.1] * 15  # would overwrite if no merge
    sr._process_bus_message({
        "type": bus.MIND_STATE,
        "payload": {"src": "outer", "values": incoming},
    })
    result = sr._state["outer_mind_15d"]
    # Thinking + feeling [0:10] from incoming
    assert result[:10] == [0.1] * 10
    # Willing [10:15] preserved from prior state
    assert result[10:15] == [0.9, 0.8, 0.7, 0.6, 0.5]


def test_mind_state_outer_src_overwrites_willing_when_default() -> None:
    """When existing outer_mind_15d[10:15] is all default (0.5), incoming
    willing values DO overwrite — no merge needed."""
    sr = OuterState()
    sr.is_active = True
    sr._state["outer_mind_15d"] = [0.5] * 15  # all default
    incoming = [round(0.1 + i * 0.05, 3) for i in range(15)]
    sr._process_bus_message({
        "type": bus.MIND_STATE,
        "payload": {"src": "outer", "values": incoming},
    })
    assert sr._state["outer_mind_15d"] == incoming


# ── Legacy OUTER_*_STATE handlers still work (Phase A+B regression) ──


def test_legacy_outer_mind_state_handler_still_works() -> None:
    """Phase A+B path (l0_rust_enabled=false) still emits OUTER_MIND_STATE
    via outer_mind_worker. These handlers must still route correctly."""
    sr = OuterState()
    sr.is_active = True
    msg = {
        "type": bus.OUTER_MIND_STATE,
        "payload": {
            "outer_mind": [0.6, 0.5, 0.4, 0.3, 0.7],
            "outer_mind_15d": [round(0.1 + i * 0.05, 3) for i in range(15)],
        },
    }
    sr._process_bus_message(msg)
    assert sr._state["outer_mind"] == [0.6, 0.5, 0.4, 0.3, 0.7]
    expected_15d = [round(0.1 + i * 0.05, 3) for i in range(15)]
    assert sr._state["outer_mind_15d"] == expected_15d


def test_legacy_outer_spirit_state_handler_still_works() -> None:
    sr = OuterState()
    sr.is_active = True
    msg = {
        "type": bus.OUTER_SPIRIT_STATE,
        "payload": {
            "outer_spirit": [0.6, 0.5, 0.4, 0.3, 0.7],
            "outer_spirit_45d": [round(0.1 + i * 0.01, 3) for i in range(45)],
        },
    }
    sr._process_bus_message(msg)
    assert sr._state["outer_spirit"] == [0.6, 0.5, 0.4, 0.3, 0.7]
    assert len(sr._state["outer_spirit_45d"]) == 45


def test_legacy_outer_body_state_handler_still_works() -> None:
    sr = OuterState()
    sr.is_active = True
    msg = {
        "type": bus.OUTER_BODY_STATE,
        "payload": {"outer_body": [0.6, 0.5, 0.4, 0.3, 0.7]},
    }
    sr._process_bus_message(msg)
    assert sr._state["outer_body"] == [0.6, 0.5, 0.4, 0.3, 0.7]


# ── Backward-compat alias ────────────────────────────────────────────


def test_state_register_alias_resolves_to_outer_state() -> None:
    assert StateRegister is OuterState
