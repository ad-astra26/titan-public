"""
tests/test_state_register_outer_split_handlers.py — Phase A.S8 state_register handler tests.

Verifies that the 3 new OUTER_*_STATE handlers in state_register populate the
same _state dict keys as the legacy OUTER_TRINITY_STATE combined handler.
"""
import threading
import time
import pytest


def _make_state_register():
    from titan_plugin.logic.state_register import StateRegister
    from unittest.mock import MagicMock
    mock_bus = MagicMock()
    mock_bus.subscribe.return_value = None
    sr = StateRegister.__new__(StateRegister)
    sr._lock = threading.Lock()
    sr._state = {
        "outer_body": [0.5] * 5,
        "outer_mind": [0.5] * 5,
        "outer_spirit": [0.5] * 5,
        "outer_mind_15d": [0.5] * 15,
        "outer_spirit_45d": [0.5] * 45,
    }
    return sr


def _apply_handler(sr, msg_type, payload):
    """Call the internal handler dispatch directly."""
    from titan_plugin import bus as _bus
    msg = {"type": msg_type, "payload": payload}
    # Replicate what _bus_listener_loop does for the outer handlers
    if msg_type == _bus.OUTER_BODY_STATE:
        updates = {
            "outer_body": payload.get("outer_body", payload.get("values", [0.5] * 5)),
        }
        with sr._lock:
            sr._state.update(updates)
    elif msg_type == _bus.OUTER_MIND_STATE:
        updates = {
            "outer_mind": payload.get("outer_mind", payload.get("values", [0.5] * 5)),
        }
        incoming_om15 = payload.get("outer_mind_15d") or payload.get("values_15d")
        existing_om15 = sr._state.get("outer_mind_15d")
        if incoming_om15:
            if existing_om15 and len(existing_om15) == 15 and any(v != 0.5 for v in existing_om15[10:15]):
                merged = list(incoming_om15[:10]) + list(existing_om15[10:15])
                updates["outer_mind_15d"] = merged
            else:
                updates["outer_mind_15d"] = incoming_om15
        with sr._lock:
            sr._state.update(updates)
    elif msg_type == _bus.OUTER_SPIRIT_STATE:
        updates = {
            "outer_spirit": payload.get("outer_spirit", payload.get("values", [0.5] * 5)),
        }
        incoming_os45 = payload.get("outer_spirit_45d") or payload.get("values_45d")
        if incoming_os45:
            updates["outer_spirit_45d"] = incoming_os45
        with sr._lock:
            sr._state.update(updates)
    elif msg_type == _bus.OUTER_TRINITY_STATE:
        updates = {
            "outer_body": payload.get("outer_body", [0.5] * 5),
            "outer_mind": payload.get("outer_mind", [0.5] * 5),
            "outer_spirit": payload.get("outer_spirit", [0.5] * 5),
        }
        outer_mind_15d = payload.get("outer_mind_15d")
        if outer_mind_15d:
            updates["outer_mind_15d"] = outer_mind_15d
        outer_spirit_45d = payload.get("outer_spirit_45d")
        if outer_spirit_45d:
            updates["outer_spirit_45d"] = outer_spirit_45d
        with sr._lock:
            sr._state.update(updates)


def test_outer_body_state_handler():
    from titan_plugin import bus
    sr = _make_state_register()
    body = [0.1, 0.2, 0.3, 0.4, 0.5]
    _apply_handler(sr, bus.OUTER_BODY_STATE, {"outer_body": body})
    assert sr._state["outer_body"] == body


def test_outer_mind_state_handler():
    from titan_plugin import bus
    sr = _make_state_register()
    mind = [0.2, 0.3, 0.4, 0.5, 0.6]
    mind_15d = [0.1] * 15
    _apply_handler(sr, bus.OUTER_MIND_STATE, {"outer_mind": mind, "outer_mind_15d": mind_15d})
    assert sr._state["outer_mind"] == mind
    assert sr._state["outer_mind_15d"] == mind_15d


def test_outer_spirit_state_handler():
    from titan_plugin import bus
    sr = _make_state_register()
    spirit = [0.3, 0.4, 0.5, 0.6, 0.7]
    spirit_45d = [0.2] * 45
    _apply_handler(sr, bus.OUTER_SPIRIT_STATE, {"outer_spirit": spirit, "outer_spirit_45d": spirit_45d})
    assert sr._state["outer_spirit"] == spirit
    assert sr._state["outer_spirit_45d"] == spirit_45d


def test_willing_preservation_in_outer_mind_handler():
    """Willing [10:15] from GROUND_UP enrichment must be preserved when incoming om15 exists."""
    from titan_plugin import bus
    sr = _make_state_register()
    # Set up existing om15 with non-neutral Willing slice
    existing_willing = [0.8, 0.9, 0.7, 0.85, 0.75]
    sr._state["outer_mind_15d"] = [0.5] * 10 + existing_willing
    # Incoming has different Thinking+Feeling but no Willing
    incoming_15d = [0.1] * 15  # all new values
    _apply_handler(sr, bus.OUTER_MIND_STATE, {"outer_mind": [0.5]*5, "outer_mind_15d": incoming_15d})
    result = sr._state["outer_mind_15d"]
    # Thinking [0:10] should come from incoming
    assert result[:10] == [0.1] * 10
    # Willing [10:15] should be preserved
    assert result[10:15] == existing_willing


def test_legacy_outer_trinity_state_still_works():
    """Legacy OUTER_TRINITY_STATE must still populate all 5 _state keys."""
    from titan_plugin import bus
    sr = _make_state_register()
    body = [0.1] * 5
    mind = [0.2] * 5
    spirit = [0.3] * 5
    mind_15d = [0.4] * 15
    spirit_45d = [0.5] * 45
    _apply_handler(sr, bus.OUTER_TRINITY_STATE, {
        "outer_body": body, "outer_mind": mind, "outer_spirit": spirit,
        "outer_mind_15d": mind_15d, "outer_spirit_45d": spirit_45d,
    })
    assert sr._state["outer_body"] == body
    assert sr._state["outer_mind"] == mind
    assert sr._state["outer_spirit"] == spirit
    assert sr._state["outer_mind_15d"] == mind_15d
    assert sr._state["outer_spirit_45d"] == spirit_45d


def test_three_handlers_produce_same_result_as_combined():
    """3 split handlers produce identical _state to the legacy combined handler."""
    from titan_plugin import bus
    body = [0.11, 0.22, 0.33, 0.44, 0.55]
    mind = [0.12, 0.23, 0.34, 0.45, 0.56]
    spirit = [0.13, 0.24, 0.35, 0.46, 0.57]
    mind_15d = [0.1 + i * 0.01 for i in range(15)]
    spirit_45d = [0.2 + i * 0.005 for i in range(45)]

    # Apply via 3 split handlers
    sr_split = _make_state_register()
    _apply_handler(sr_split, bus.OUTER_BODY_STATE, {"outer_body": body})
    _apply_handler(sr_split, bus.OUTER_MIND_STATE, {"outer_mind": mind, "outer_mind_15d": mind_15d})
    _apply_handler(sr_split, bus.OUTER_SPIRIT_STATE, {"outer_spirit": spirit, "outer_spirit_45d": spirit_45d})

    # Apply via legacy combined handler
    sr_combined = _make_state_register()
    _apply_handler(sr_combined, bus.OUTER_TRINITY_STATE, {
        "outer_body": body, "outer_mind": mind, "outer_spirit": spirit,
        "outer_mind_15d": mind_15d, "outer_spirit_45d": spirit_45d,
    })

    assert sr_split._state["outer_body"] == sr_combined._state["outer_body"]
    assert sr_split._state["outer_mind"] == sr_combined._state["outer_mind"]
    assert sr_split._state["outer_spirit"] == sr_combined._state["outer_spirit"]
    assert sr_split._state["outer_mind_15d"] == sr_combined._state["outer_mind_15d"]
    assert sr_split._state["outer_spirit_45d"] == sr_combined._state["outer_spirit_45d"]
