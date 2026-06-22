"""RFP_g18 §7.D — trinity STATE types no longer broadcast (D-SPEC-162).

After §7.C (cognitive_worker → SHM) + §7.B (state_register retired), nothing
consumes BODY/MIND/SPIRIT_STATE off the bus. §7.D makes these types fully skip
the dst="all" fan-out (in both publish() and publish_in_process()) and shrinks
the _HIGH_RATE_BROADCAST_TYPES stopgap to the EVENT set — without deleting it
(INV-G18-4: deleting it re-opens the T3 wildcard queue-overflow for EVENTs).

Run isolated: pytest tests/test_g18_broadcast_drop.py -p no:anchorpy
"""
from titan_hcl.bus import (
    DivineBus, BODY_STATE, MIND_STATE, SPIRIT_STATE, SPHERE_PULSE,
)


def test_state_types_skip_fanout_even_for_filtered_subscriber():
    bus = DivineBus()
    # a subscriber that EXPLICITLY opts in to BODY_STATE via a broadcast filter
    # (the old stopgap still delivered to these) — §7.D must drop it anyway.
    bus.subscribe("opted_in", types=[BODY_STATE, MIND_STATE, SPIRIT_STATE])
    for t, dim in ((BODY_STATE, 5), (MIND_STATE, 15), (SPIRIT_STATE, 45)):
        n = bus.publish({"type": t, "dst": "all", "src": "inner",
                         "values": [0.9] * dim})
        assert n == 0, f"{t} was fanned out ({n}) — must be 0 under §7.D"


def test_state_types_dropped_on_incoming_broker_path():
    """publish_in_process is the live cutover point (Rust broker → _client)."""
    bus = DivineBus()
    bus.subscribe("wild")  # wildcard
    n = bus.publish_in_process(
        {"type": SPIRIT_STATE, "dst": "all", "src": "outer", "values": [0.5] * 45})
    assert n == 0


def test_blackboard_latest_value_kept_for_state(monkeypatch):
    """RFP §6 Q3 — the in-kernel blackboard latest-value write is preserved."""
    bus = DivineBus()
    written = {}
    monkeypatch.setattr(bus._blackboard, "write",
                        lambda k, v: written.__setitem__(k, v))
    bus.publish({"type": BODY_STATE, "dst": "all", "src": "inner",
                 "values": [0.7] * 5})
    assert "inner_BODY_STATE" in written  # blackboard still written before the skip


def test_stopgap_shrunk_not_deleted():
    """INV-G18-4: STATE types removed from the stopgap; EVENT types remain."""
    hr = DivineBus._HIGH_RATE_BROADCAST_TYPES
    # STATE types fully dropped → no longer in the stopgap
    assert BODY_STATE not in hr
    assert MIND_STATE not in hr
    assert SPIRIT_STATE not in hr
    # EVENT types still capped (stopgap shrunk, NOT deleted)
    assert SPHERE_PULSE in hr          # EVENT (R1) — observatory SSE
    assert "BIG_PULSE" in hr
    assert "PI_HEARTBEAT_UPDATED" in hr
    assert "TOPOLOGY_STATE_UPDATED" in hr
    assert len(hr) >= 4  # never emptied


def test_state_no_broadcast_set_is_the_three_trinity_types():
    assert DivineBus._STATE_NO_BROADCAST_TYPES == frozenset(
        {BODY_STATE, MIND_STATE, SPIRIT_STATE})


def test_event_type_still_fans_out_to_a_filtered_subscriber():
    """An EVENT type (SPHERE_PULSE) must still reach a subscriber that opts in
    — it is bus-legal under G18; only the migrated STATE types are dropped."""
    bus = DivineBus()
    bus.subscribe("sse_consumer", types=[SPHERE_PULSE])
    n = bus.publish({"type": SPHERE_PULSE, "dst": "all", "src": "trinity",
                     "clock_name": "body", "pulse_count": 3})
    assert n >= 1, "SPHERE_PULSE (EVENT) must still deliver to an opted-in sub"
