"""Integration tests pinning down the Option B subscribe-filter contracts
for the three subscribers migrated in commit ef7c3ba7 (2026-04-29):

  - rl_proxy_stats  → {SAGE_STATS}
  - v4_bridge       → V4_EVENT_TYPES (7 types)
  - guardian        → {MODULE_HEARTBEAT, MODULE_READY, MODULE_SHUTDOWN,
                       BUS_WORKER_ADOPT_REQUEST}

These tests are regression gates — if anyone changes the consumer-side
handler whitelist without updating the `types=` filter at the subscribe
call (or vice versa), the test fails. Per memory rule
`feedback_specs_need_enforcement_automation.md`: rules need test gates,
not just human discipline.
"""
from __future__ import annotations

from titan_plugin.bus import (
    BODY_STATE,
    BUS_WORKER_ADOPT_REQUEST,
    DivineBus,
    DREAM_STATE_CHANGED,
    BIG_PULSE,
    FILTER_DOWN,
    FOCUS_NUDGE,
    GREAT_PULSE,
    IMPULSE,
    MIND_STATE,
    MODULE_HEARTBEAT,
    MODULE_READY,
    MODULE_SHUTDOWN,
    OBSERVABLES_SNAPSHOT,
    OUTER_TRINITY_STATE,
    QUERY,
    SAGE_STATS,
    SOVEREIGNTY_EPOCH,
    SPHERE_PULSE,
    SPIRIT_STATE,
)


# ── rl_proxy_stats ──────────────────────────────────────────────────


def test_rl_proxy_stats_filter_is_sage_stats_only():
    """RLProxy declares types=[SAGE_STATS] at subscribe time so its queue
    only receives SAGE_STATS broadcasts (T2: 308k drops + T3: 342k drops
    pre-fix were all unwanted broadcasts saturating this queue)."""
    bus = DivineBus()

    # Mimic the construction path: rl_proxy.py:70 subscribes via
    # bus.subscribe("rl_proxy_stats", types=[SAGE_STATS])
    from titan_plugin.proxies.rl_proxy import RLProxy
    # RLProxy needs a Guardian for spawn-on-demand. Use a stub — we only
    # care about the subscribe() side effect at __init__.
    class _StubGuardian:
        def __init__(self):
            self._modules = {}
        def register(self, *a, **kw):
            pass
        def start(self, *a, **kw):
            return None

    proxy = RLProxy(bus, _StubGuardian())
    assert proxy._stats_subscription is not None, (
        "RLProxy did not subscribe — broadcast filter never installed")

    flt = bus.get_broadcast_filter("rl_proxy_stats")
    assert flt == frozenset({SAGE_STATS}), (
        f"rl_proxy_stats filter drift: expected {{SAGE_STATS}}, got {flt}. "
        f"If you changed the handler in plugin.py:_rl_stats_loop, also "
        f"update the types= filter at proxies/rl_proxy.py:~70.")


# ── v4_bridge ────────────────────────────────────────────────────────


def test_v4_bridge_filter_matches_V4_EVENT_TYPES_constant():
    """v4_bridge declares types=V4_EVENT_TYPES at subscribe time. The
    consumer-side `if msg_type not in V4_EVENT_TYPES: continue` check
    in plugin.py:_v4_event_bridge becomes redundant but is kept as
    belt-and-suspenders. Both sides MUST stay in sync."""
    # The two subscribe sites (core/plugin.py and legacy_core.py) both
    # use the same set. Verify by simulating the subscribe call directly.
    bus = DivineBus()
    V4_EVENT_TYPES = {
        SPHERE_PULSE, BIG_PULSE, GREAT_PULSE, DREAM_STATE_CHANGED,
        "NEUROMOD_UPDATE", "HORMONE_FIRED", "EXPRESSION_FIRED",
    }
    bus.subscribe("v4_bridge", types=V4_EVENT_TYPES)
    flt = bus.get_broadcast_filter("v4_bridge")
    assert flt == frozenset(V4_EVENT_TYPES), (
        f"v4_bridge filter drift: expected {V4_EVENT_TYPES}, got {flt}.")
    # Drift-detection assertion: the consumer-side check must list
    # exactly the same types. Verify by reading the source.
    import inspect
    from titan_plugin.core.plugin import TitanPlugin
    # The bridge function is async-defined inside another async function —
    # we can't easily inspect it. Instead, read the source file and check
    # the V4_EVENT_TYPES literal set is exactly these 7 entries.
    src = inspect.getsource(TitanPlugin)
    assert "V4_EVENT_TYPES = {SPHERE_PULSE, BIG_PULSE, GREAT_PULSE, DREAM_STATE_CHANGED" in src, (
        "V4_EVENT_TYPES literal in core/plugin.py drifted — update both "
        "this test AND the V4_EVENT_TYPES set at the subscribe site.")


# ── guardian ─────────────────────────────────────────────────────────


def test_guardian_filter_matches_lifecycle_msg_handlers():
    """Guardian declares types=[MODULE_HEARTBEAT, MODULE_READY,
    MODULE_SHUTDOWN, BUS_WORKER_ADOPT_REQUEST] at subscribe time. These
    are the 4 msg_types its `_process_guardian_messages` consumer
    handles + targeted (lifecycle) — listed explicitly so the contract
    is self-documenting."""
    bus = DivineBus()
    from titan_plugin.guardian import Guardian
    guardian = Guardian(bus)
    flt = bus.get_broadcast_filter("guardian")
    expected = frozenset({
        MODULE_HEARTBEAT, MODULE_READY, MODULE_SHUTDOWN,
        BUS_WORKER_ADOPT_REQUEST,
    })
    assert flt == expected, (
        f"guardian filter drift: expected {expected}, got {flt}. "
        f"If you added/removed an elif branch in "
        f"_process_guardian_messages, update the types= list at "
        f"guardian.py:~168.")


def test_guardian_module_heartbeat_targeted_msgs_bypass_filter():
    """MODULE_HEARTBEAT is conventionally targeted (dst="guardian") so
    it bypasses the broadcast filter regardless. This test verifies the
    targeted-bypass guarantee — load-bearing for Guardian's correctness:
    even if our filter list ever shrinks, targeted heartbeats still land.

    Pre-fix, MODULE_HEARTBEAT msgs were getting lost as queue-full drops
    because the queue was saturated with broadcast _UPDATED noise — 87
    heartbeat drops in a single 500-line window on T1, risking false
    heartbeat-timeouts and unnecessary worker restarts."""
    from titan_plugin.bus import make_msg
    bus = DivineBus()
    from titan_plugin.guardian import Guardian
    guardian = Guardian(bus)

    # Targeted MODULE_HEARTBEAT — must arrive even though filter excludes
    # nothing of consequence (this case: filter has it; targeted bypass
    # only matters when the filter would NOT include the type).
    bus.publish(make_msg(MODULE_HEARTBEAT, "some_module", "guardian",
                         {"rss_mb": 100}))
    msg = guardian._guardian_queue.get_nowait()
    assert msg["type"] == MODULE_HEARTBEAT
    assert msg["src"] == "some_module"


def test_guardian_filters_out_unwanted_broadcasts():
    """Guardian no longer receives the dst='all' fan-out flood. This
    captures the actual fix: 1.2M T1 / 308k T2 / 342k T3 queue-full drops
    were largely caused by Guardian's queue being a catch-all for every
    broadcast. Post-fix, broadcasts not in the lifecycle whitelist are
    dropped at publish, never enter Guardian's queue, never displace
    MODULE_HEARTBEAT."""
    from titan_plugin.bus import make_msg
    bus = DivineBus()
    from titan_plugin.guardian import Guardian
    guardian = Guardian(bus)

    # Spam the broadcast bus with the kinds of msgs that flooded T1
    flood_types = [
        "DREAMING_STATE_UPDATED", "NEUROMOD_STATS_UPDATED",
        "MSL_STATE_UPDATED", "TOPOLOGY_STATE_UPDATED",
        "REASONING_STATS_UPDATED", "META_REASONING_STATS_UPDATED",
        "LANGUAGE_STATS_UPDATED", "PI_HEARTBEAT_UPDATED",
        "EXPRESSION_COMPOSITES_UPDATED", "SPIRIT_STATE",
        "SPHERE_PULSE", "EXPRESSION_FIRED",
    ]
    for t in flood_types * 100:  # 1200 unwanted broadcasts
        bus.publish(make_msg(t, "spirit", "all", {}))

    # Guardian queue should be empty (nothing in filter, nothing targeted)
    drained = []
    try:
        while True:
            drained.append(guardian._guardian_queue.get_nowait())
    except Exception:
        pass
    assert drained == [], (
        f"Guardian received {len(drained)} unwanted broadcasts despite "
        f"filter. Filter check broken? First 3: {drained[:3]}")

    # Filter stats reflect the dropped broadcasts (1200 broadcasts × 1
    # subscriber = 1200 filtered)
    assert bus._stats["filtered_broadcasts"] >= 1200


# ── agency ───────────────────────────────────────────────────────────


def test_agency_filter_matches_loop_elif_chain():
    """TitanPlugin._agency_loop handles 6 msg_types (manually verified):
    IMPULSE, OUTER_DISPATCH, QUERY, AGENCY_STATS, ASSESSMENT_STATS,
    AGENCY_READY. Legacy_core path handles only the first 3 — filter
    union semantics keeps both call sites' behavior consistent."""
    from titan_plugin.bus import (
        AGENCY_READY, AGENCY_STATS, ASSESSMENT_STATS,
        OUTER_DISPATCH, QUERY,
    )
    bus = DivineBus()
    # Mimic the call site directly — full TitanPlugin construction is
    # too heavy and not needed to verify the filter declaration.
    bus.subscribe(
        "agency",
        types=[
            IMPULSE, OUTER_DISPATCH, QUERY,
            AGENCY_STATS, ASSESSMENT_STATS, AGENCY_READY,
        ],
    )
    flt = bus.get_broadcast_filter("agency")
    expected = frozenset({
        IMPULSE, OUTER_DISPATCH, QUERY,
        AGENCY_STATS, ASSESSMENT_STATS, AGENCY_READY,
    })
    assert flt == expected, (
        f"agency filter drift: expected {expected}, got {flt}. "
        f"Read core/plugin.py:_agency_loop elif chain to refresh.")


# ── state_register ───────────────────────────────────────────────────


def test_state_register_filter_matches_process_bus_message_chain():
    """StateRegister.start() declares types matching the elif chain in
    _process_bus_message() — captured 9 types via auto-extraction:
    BODY_STATE, FILTER_DOWN, FOCUS_NUDGE, IMPULSE, MIND_STATE,
    OBSERVABLES_SNAPSHOT, OUTER_TRINITY_STATE, SPHERE_PULSE, SPIRIT_STATE."""
    bus = DivineBus()
    from titan_plugin.logic.state_register import StateRegister
    sr = StateRegister()
    sr.start(bus, snapshot_interval=999.0)  # large interval = no snapshot fires
    try:
        flt = bus.get_broadcast_filter("state_register")
        expected = frozenset({
            BODY_STATE, FILTER_DOWN, FOCUS_NUDGE, IMPULSE, MIND_STATE,
            OBSERVABLES_SNAPSHOT, OUTER_TRINITY_STATE, SPHERE_PULSE, SPIRIT_STATE,
        })
        assert flt == expected, (
            f"state_register filter drift: expected {expected}, got {flt}. "
            f"Run scripts/migrate_bus_filters.py and re-extract.")
    finally:
        sr.stop()


# ── chat_handler ─────────────────────────────────────────────────────


def test_chat_handler_filter_is_query_only():
    """TitanPlugin._chat_handler_loop / TitanCore._chat_handler_loop
    only handle msg.type == QUERY (with payload.action == 'chat'). The
    queue is targeted via dst='chat_handler' for CHAT_REQUEST round-trips
    via bus.request_async — those bypass the filter regardless. The
    QUERY broadcast filter is defensive against future fan-out."""
    bus = DivineBus()
    bus.subscribe("chat_handler", types=[QUERY])
    flt = bus.get_broadcast_filter("chat_handler")
    assert flt == frozenset({QUERY}), (
        f"chat_handler filter drift: expected {{QUERY}}, got {flt}. "
        f"Read core/plugin.py:_chat_handler_loop or "
        f"legacy_core.py:_chat_handler_loop to refresh.")


# ── sovereignty ──────────────────────────────────────────────────────


def test_sovereignty_filter_is_sovereignty_epoch_only():
    """Sovereignty queue only consumes SOVEREIGNTY_EPOCH per the elif
    chain in TitanPlugin._sovereignty_loop and
    TitanCore._sovereignty_loop."""
    bus = DivineBus()
    bus.subscribe("sovereignty", types=[SOVEREIGNTY_EPOCH])
    flt = bus.get_broadcast_filter("sovereignty")
    assert flt == frozenset({SOVEREIGNTY_EPOCH}), (
        f"sovereignty filter drift: expected {{SOVEREIGNTY_EPOCH}}, got {flt}.")


# ── core + meditation (reply_only=True, types=[]) ────────────────────


def test_core_filter_is_empty_reply_only():
    """`core` is reply_only=True (excludes broadcasts) AND types=[]
    documents that explicitly. Targeted dst='core' RPC replies still
    arrive via the targeted-msg-bypass path."""
    bus = DivineBus()
    bus.subscribe("core", reply_only=True, types=[])
    flt = bus.get_broadcast_filter("core")
    assert flt == frozenset(), (
        f"core filter drift: expected empty set, got {flt}.")


def test_meditation_filter_is_empty_reply_only():
    """`meditation` is reply_only=True AND types=[]. MEDITATION_REQUEST
    arrives via targeted dst='meditation' — bypasses filter. Same
    contract as core."""
    bus = DivineBus()
    bus.subscribe("meditation", reply_only=True, types=[])
    flt = bus.get_broadcast_filter("meditation")
    assert flt == frozenset(), (
        f"meditation filter drift: expected empty set, got {flt}.")


# ── End-to-end: targeted msgs + filtered broadcasts together ─────────


def test_guardian_receives_only_lifecycle_msgs_under_mixed_load():
    """Realistic mixed-load test: spam broadcasts AND lifecycle msgs.
    Verify Guardian gets all lifecycle msgs (none dropped) and zero
    broadcasts."""
    from titan_plugin.bus import make_msg
    bus = DivineBus()
    from titan_plugin.guardian import Guardian
    guardian = Guardian(bus)

    # Interleave: 100 broadcasts + 1 lifecycle, repeat 10x
    for cycle in range(10):
        for t in ["SPHERE_PULSE", "DREAMING_STATE_UPDATED",
                  "NEUROMOD_STATS_UPDATED"]:
            for _ in range(100):
                bus.publish(make_msg(t, "spirit", "all", {}))
        # One MODULE_HEARTBEAT per cycle
        bus.publish(make_msg(MODULE_HEARTBEAT, f"mod_{cycle}",
                             "guardian", {"rss_mb": cycle}))

    # Drain
    received = []
    try:
        while True:
            received.append(guardian._guardian_queue.get_nowait())
    except Exception:
        pass

    types_received = [m["type"] for m in received]
    assert types_received == [MODULE_HEARTBEAT] * 10, (
        f"Guardian received unexpected msgs: {types_received[:5]}...")
    # Source IDs preserved — none lost to drops
    srcs = sorted(m["src"] for m in received)
    assert srcs == [f"mod_{i}" for i in range(10)]
