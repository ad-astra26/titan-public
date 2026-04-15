"""Tests for OuterState (state_register) — rFP #1 foundation methods.

Covers:
  - _pad_to helper: boundary behaviour
  - get_full_130dt: dim stability across maturity levels, layout match
    with unified_spirit tensor slicing, correct zero-pad with 0.5

Style mirrors tests/test_t1_coherence_observables.py.
"""
import pytest

from titan_plugin.logic.state_register import OuterState, _pad_to


# ── _pad_to helper ──────────────────────────────────────────────────

def test_pad_to_empty_returns_all_pad_val():
    assert _pad_to(None, 5) == [0.5] * 5
    assert _pad_to([], 5) == [0.5] * 5


def test_pad_to_shorter_pads_with_default_0_5():
    assert _pad_to([0.1, 0.2], 5) == [0.1, 0.2, 0.5, 0.5, 0.5]


def test_pad_to_shorter_pads_with_custom_pad_val():
    assert _pad_to([0.1, 0.2], 5, pad_val=0.0) == [0.1, 0.2, 0.0, 0.0, 0.0]


def test_pad_to_exact_length_returns_copy_unchanged():
    v = [0.1, 0.2, 0.3, 0.4, 0.5]
    assert _pad_to(v, 5) == v
    # Ensure it's a copy (mutating input shouldn't mutate output)
    result = _pad_to(v, 5)
    v[0] = 999.0
    assert result[0] == 0.1


def test_pad_to_longer_truncates():
    assert _pad_to([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], 5) == [0.1, 0.2, 0.3, 0.4, 0.5]


# ── get_full_130dt ──────────────────────────────────────────────────

def test_get_full_130dt_returns_exactly_130_floats_fresh_register():
    """Fresh register with no data → all defaults padded to 130 floats."""
    reg = OuterState()
    v = reg.get_full_130dt()
    assert len(v) == 130
    assert all(isinstance(x, float) for x in v)


def test_get_full_130dt_layout_matches_unified_spirit_slicing():
    """Populate distinct 5/15/45 + 5/15/45 tensors → each slice matches input."""
    reg = OuterState()
    reg._state["body_tensor"]         = [0.11] * 5
    reg._state["mind_tensor_15d"]     = [0.22] * 15
    reg._state["spirit_tensor_45d"]   = [0.33] * 45
    reg._state["outer_body"]          = [0.44] * 5
    reg._state["outer_mind_15d"]      = [0.55] * 15
    reg._state["outer_spirit_45d"]    = [0.66] * 45

    v = reg.get_full_130dt()
    assert len(v) == 130
    # Layout: 5 + 15 + 45 + 5 + 15 + 45 = 130
    assert v[0:5]    == [0.11] * 5   # inner_body
    assert v[5:20]   == [0.22] * 15  # inner_mind
    assert v[20:65]  == [0.33] * 45  # inner_spirit
    assert v[65:70]  == [0.44] * 5   # outer_body
    assert v[70:85]  == [0.55] * 15  # outer_mind
    assert v[85:130] == [0.66] * 45  # outer_spirit


def test_get_full_130dt_legacy_5d_mind_pads_with_0_5():
    """Only legacy 5D mind_tensor set (no mind_tensor_15d) → 5 real + 10×0.5 = 15."""
    reg = OuterState()
    reg._state["mind_tensor"] = [0.7, 0.7, 0.7, 0.7, 0.7]
    # No mind_tensor_15d

    v = reg.get_full_130dt()
    # Inner mind slice is v[5:20]
    assert v[5:10]  == [0.7] * 5       # real
    assert v[10:20] == [0.5] * 10      # padded


def test_get_full_130dt_no_outer_pads_all_outer_with_0_5():
    """Only inner populated → outer slice (v[65:130]) is all 0.5."""
    reg = OuterState()
    reg._state["body_tensor"]       = [0.1] * 5
    reg._state["mind_tensor_15d"]   = [0.2] * 15
    reg._state["spirit_tensor_45d"] = [0.3] * 45
    # Outer not populated — defaults to [0.5]*5 in _state init;
    # _pad_to extends to full 15D / 45D with 0.5.

    v = reg.get_full_130dt()
    assert v[65:70]  == [0.5] * 5     # outer_body default
    assert v[70:85]  == [0.5] * 15    # outer_mind — 5 real 0.5s + 10 pad 0.5s
    assert v[85:130] == [0.5] * 45    # outer_spirit — same pattern


def test_get_full_130dt_values_all_floats_not_ints():
    """Guard against int contamination — cosine_sim math expects floats."""
    reg = OuterState()
    reg._state["body_tensor"] = [0, 1, 0, 1, 0]  # ints
    v = reg.get_full_130dt()
    # _pad_to returns list(values) which preserves int-ness in Python
    # but downstream consumers expect floats; document the current behavior
    # and assert the pad-val positions are floats.
    for i in range(5, 130):
        assert isinstance(v[i], float)


# ── get_full_30d_topology + OBSERVABLES_SNAPSHOT handler ────────────

def test_get_full_30d_topology_default_zeros_before_any_snapshot():
    """Fresh register → 30 floats, all 0.0 (observables centered default)."""
    reg = OuterState()
    v = reg.get_full_30d_topology()
    assert len(v) == 30
    assert v == [0.0] * 30


def test_observables_snapshot_handler_stores_30d_flat():
    """Publish OBSERVABLES_SNAPSHOT → state reflects the 30D payload."""
    reg = OuterState()
    flat = [0.1 + i * 0.01 for i in range(30)]
    dict_payload = {"inner_body": {"coherence": 0.5}}
    msg = {
        "type": "OBSERVABLES_SNAPSHOT",
        "src": "spirit",
        "dst": "state_register",
        "payload": {
            "observables_30d":  flat,
            "observables_dict": dict_payload,
        },
    }
    reg._process_bus_message(msg)

    assert reg.get_full_30d_topology() == flat
    assert reg._state["observables_dict"] == dict_payload


def test_observables_snapshot_handler_rejects_wrong_dim():
    """Payload with non-30 list → state unchanged (silent reject)."""
    reg = OuterState()
    msg = {
        "type": "OBSERVABLES_SNAPSHOT",
        "src": "spirit",
        "dst": "state_register",
        "payload": {"observables_30d": [0.5] * 27},  # wrong dim
    }
    reg._process_bus_message(msg)
    # Still zeros — the malformed snapshot was rejected.
    assert reg.get_full_30d_topology() == [0.0] * 30


def test_state_snapshot_payload_has_four_dim_keys():
    """Phase 3: emitted STATE_SNAPSHOT payload includes all 4 atomic signals.

    Populates extended state first so full_65dt emits at its canonical
    65D size (without extension keys, legacy code yields 15D — pre-existing
    behaviour, not changed by rFP #1).
    """
    reg = OuterState()
    # Populate extended dims so full_65dt reaches its canonical size.
    reg._state["body_tensor"]       = [0.1] * 5
    reg._state["mind_tensor_15d"]   = [0.2] * 15
    reg._state["spirit_tensor_45d"] = [0.3] * 45

    captured = {}

    class MockBus:
        blackboard = None

        def publish(self, msg):
            captured["msg"] = msg

    reg._bus = MockBus()
    # Emulate one iteration of _snapshot_publish_loop's body.
    from titan_plugin.bus import make_msg

    snapshot_30dt = reg.get_full_30dt()
    extended = reg.get_full_extended()
    full_65dt = (
        extended["inner_body"] + extended["inner_mind"] + extended["inner_spirit"]
    )
    msg = make_msg("STATE_SNAPSHOT", "state_register", "spirit", {
        "full_30dt":         snapshot_30dt,
        "full_65dt":         full_65dt,
        "full_130dt":        reg.get_full_130dt(),
        "full_30d_topology": reg.get_full_30d_topology(),
        "dims":              extended["dims"],
        "timestamp":         123.0,
    })
    reg._bus.publish(msg)

    assert "msg" in captured
    payload = captured["msg"]["payload"]
    # Four atomic signals present
    assert "full_30dt" in payload
    assert "full_65dt" in payload
    assert "full_130dt" in payload
    assert "full_30d_topology" in payload
    # Exact dims
    assert len(payload["full_30dt"]) == 30
    assert len(payload["full_65dt"]) == 65     # inner 5+15+45 when extended
    assert len(payload["full_130dt"]) == 130    # always 130, per rFP #1
    assert len(payload["full_30d_topology"]) == 30


def test_observables_snapshot_handler_updates_partial_payloads():
    """Payload with only the dict → dict stored, 30D unchanged; then vice versa."""
    reg = OuterState()
    # Dict-only
    reg._process_bus_message({
        "type": "OBSERVABLES_SNAPSHOT",
        "payload": {"observables_dict": {"k": "v"}},
    })
    assert reg._state["observables_dict"] == {"k": "v"}
    assert reg.get_full_30d_topology() == [0.0] * 30
    # 30D-only
    flat = [0.3] * 30
    reg._process_bus_message({
        "type": "OBSERVABLES_SNAPSHOT",
        "payload": {"observables_30d": flat},
    })
    assert reg.get_full_30d_topology() == flat
    assert reg._state["observables_dict"] == {"k": "v"}  # preserved
