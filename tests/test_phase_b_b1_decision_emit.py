"""Phase B (§7.B B.1) — agno emits SELF_LEARN_DECISION for NON-verifiable turns.

A `direct`/`research`/`IDK` action has no synchronous oracle, so the agno PreHook
stashes its `(features, action)` (keyed by a fresh reasoning_id) for an async reward
(turn-judge / user / Maker) to join later. The verifiable `tool`/`skill_delegate`
lane is skipped (Phase 1 trains it directly at the oracle verdict).
"""
from titan_hcl import bus as _bus_mod
from titan_hcl.modules.agno_hooks import _emit_nonverifiable_decision
from titan_hcl.synthesis.outer_meta_policy import OUTER_ACTIONS


class _FakeBus:
    def __init__(self):
        self.published = []

    def publish(self, msg):
        self.published.append(msg)


class _FakePlugin:
    def __init__(self, bus=None):
        self.bus = bus


_FEATS = [0.4, 0.0, 0.0, 0.0, 1.0]


def test_b1_emits_decision_for_direct():
    bus = _FakeBus()
    rid = _emit_nonverifiable_decision(
        _FakePlugin(bus), _FEATS, OUTER_ACTIONS.index("direct"),
        "What does sovereignty mean to you?", "maker")
    assert isinstance(rid, str) and len(rid) > 0
    assert len(bus.published) == 1
    msg = bus.published[0]
    assert msg["type"] == _bus_mod.SELF_LEARN_DECISION
    p = msg["payload"]
    assert p["parent_tool_call_tx"] == rid       # the stash key = reasoning_id
    assert p["action"] == OUTER_ACTIONS.index("direct")
    assert p["features"] == _FEATS
    assert p["turn_id"] == "maker"
    assert isinstance(p["goal_class"], str)


def test_b1_emits_for_research_and_idk():
    for name in ("research", "IDK"):
        bus = _FakeBus()
        rid = _emit_nonverifiable_decision(
            _FakePlugin(bus), _FEATS, OUTER_ACTIONS.index(name), "topic", "u")
        assert rid is not None and len(bus.published) == 1


def test_b1_skips_verifiable_tool_action():
    bus = _FakeBus()
    rid = _emit_nonverifiable_decision(
        _FakePlugin(bus), _FEATS, OUTER_ACTIONS.index("tool"), "compute 8!", "u")
    assert rid is None
    assert bus.published == []


def test_b1_no_bus_is_graceful():
    rid = _emit_nonverifiable_decision(
        _FakePlugin(bus=None), _FEATS, OUTER_ACTIONS.index("direct"), "hi", "u")
    assert rid is None


def test_b1_unique_reasoning_ids():
    bus = _FakeBus()
    plugin = _FakePlugin(bus)
    direct = OUTER_ACTIONS.index("direct")
    rids = {_emit_nonverifiable_decision(plugin, _FEATS, direct, "q", "u")
            for _ in range(5)}
    assert len(rids) == 5  # each turn gets a fresh id
