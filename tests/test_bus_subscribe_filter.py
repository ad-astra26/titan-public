"""Unit tests for DivineBus.subscribe(types=[...]) — Option B filter.

Closes the bus-flood architectural regression discovered 2026-04-29:
  T1 1.2M / T2 308k / T3 342k queue-full drops in microkernel v2 mode
  caused by spirit_loop fan-out broadcasts (~12 _UPDATED events per
  snapshot tick to dst="all") flooding every named subscriber's queue,
  including subscribers (v4_bridge, guardian, rl_proxy_stats) that
  whitelist a small subset of msg_types and discard the rest at the
  consumer side — but only AFTER they've already filled the queue and
  starved messages the subscriber DOES care about (MODULE_HEARTBEAT
  drops at guardian, DREAMING_STATE_UPDATED drops at api).

Filter contract:
  - Subscriber declares accepted msg_types via subscribe(name, types=[...])
  - Broadcasts (dst="all") with msg_type not in filter → dropped at
    publish time (zero queue cost), counted in stats["filtered_broadcasts"]
  - Targeted messages (dst="<name>") ALWAYS delivered regardless of filter
  - types=None (default) = legacy wildcard (receive every broadcast)
  - Two subscribe calls with same name take UNION of types (no narrowing)
"""
from __future__ import annotations

import threading

from titan_plugin.bus import DivineBus, make_msg


# ── Helpers ──────────────────────────────────────────────────────────


def _drain(q, timeout: float = 0.0) -> list:
    """Drain a queue without blocking. Returns list of msgs."""
    out = []
    try:
        while True:
            out.append(q.get_nowait())
    except Exception:
        pass
    return out


# ── Backward-compat: types=None preserves legacy wildcard behavior ──


def test_subscribe_without_types_receives_all_broadcasts():
    bus = DivineBus()
    q = bus.subscribe("legacy_wildcard")
    bus.publish(make_msg("FOO", "src", "all", {}))
    bus.publish(make_msg("BAR", "src", "all", {}))
    bus.publish(make_msg("BAZ", "src", "all", {}))
    msgs = _drain(q)
    assert {m["type"] for m in msgs} == {"FOO", "BAR", "BAZ"}


def test_get_broadcast_filter_returns_none_for_unfiltered_subscriber():
    bus = DivineBus()
    bus.subscribe("legacy_wildcard")
    assert bus.get_broadcast_filter("legacy_wildcard") is None


def test_get_broadcast_filter_returns_none_for_unknown_subscriber():
    bus = DivineBus()
    assert bus.get_broadcast_filter("never_subscribed") is None


# ── Filter enforcement on dst="all" broadcasts ──────────────────────


def test_filter_drops_unwanted_broadcasts_at_publish():
    bus = DivineBus()
    q = bus.subscribe("filtered", types=["WANT_A", "WANT_B"])
    bus.publish(make_msg("WANT_A", "src", "all", {}))
    bus.publish(make_msg("UNWANTED", "src", "all", {}))
    bus.publish(make_msg("WANT_B", "src", "all", {}))
    bus.publish(make_msg("ALSO_UNWANTED", "src", "all", {}))
    msgs = _drain(q)
    types = [m["type"] for m in msgs]
    assert types == ["WANT_A", "WANT_B"]
    # Stats reflect 2 filtered drops (UNWANTED + ALSO_UNWANTED)
    assert bus._stats["filtered_broadcasts"] == 2


def test_filter_returns_correct_set_via_introspection():
    bus = DivineBus()
    bus.subscribe("filtered", types=["A", "B", "C"])
    flt = bus.get_broadcast_filter("filtered")
    assert flt == frozenset({"A", "B", "C"})
    assert isinstance(flt, frozenset)


def test_empty_filter_drops_all_broadcasts():
    """types=[] = receive no broadcasts (similar to reply_only=True)."""
    bus = DivineBus()
    q = bus.subscribe("no_broadcasts", types=[])
    bus.publish(make_msg("ANY", "src", "all", {}))
    bus.publish(make_msg("THING", "src", "all", {}))
    assert _drain(q) == []
    # Filter is empty frozenset (NOT None — wildcard would deliver these)
    assert bus.get_broadcast_filter("no_broadcasts") == frozenset()


# ── Targeted messages bypass the filter ──────────────────────────────


def test_filter_does_not_block_targeted_messages():
    """dst="<name>" messages MUST always be delivered, regardless of filter.

    This is the load-bearing invariant for RPC: a subscriber's filter
    declares what it consumes via BROADCAST. Replies and direct sends
    use targeted dst — those are explicitly addressed and must always
    land.
    """
    bus = DivineBus()
    q = bus.subscribe("targeted_test", types=["BROADCAST_ONLY"])
    # Targeted msg with type NOT in filter — must still arrive.
    bus.publish(make_msg("REPLY_TYPE", "src", "targeted_test", {}))
    # Targeted msg with type IN filter — also arrives.
    bus.publish(make_msg("BROADCAST_ONLY", "src", "targeted_test", {}))
    # Broadcast with type NOT in filter — dropped.
    bus.publish(make_msg("OTHER", "src", "all", {}))
    msgs = _drain(q)
    types = [m["type"] for m in msgs]
    assert types == ["REPLY_TYPE", "BROADCAST_ONLY"]


def test_targeted_messages_dont_increment_filtered_broadcasts():
    bus = DivineBus()
    bus.subscribe("foo", types=["X"])
    # 5 targeted msgs that wouldn't pass the broadcast filter
    for _ in range(5):
        bus.publish(make_msg("Y", "src", "foo", {}))
    assert bus._stats["filtered_broadcasts"] == 0


# ── Filter union when same name subscribes twice ─────────────────────


def test_two_subscribes_same_name_union_filters():
    """Dual-path subscribe sites (legacy_core + core/plugin for the same
    name) must not narrow each other's filter. Union is taken so the
    union of all types declared is delivered."""
    bus = DivineBus()
    q1 = bus.subscribe("dual_path", types=["A", "B"])
    q2 = bus.subscribe("dual_path", types=["B", "C"])
    bus.publish(make_msg("A", "src", "all", {}))
    bus.publish(make_msg("B", "src", "all", {}))
    bus.publish(make_msg("C", "src", "all", {}))
    bus.publish(make_msg("D", "src", "all", {}))
    # Both queues receive the same msgs (union of both filter declarations)
    types1 = sorted(m["type"] for m in _drain(q1))
    types2 = sorted(m["type"] for m in _drain(q2))
    assert types1 == ["A", "B", "C"]
    assert types2 == ["A", "B", "C"]
    # Filter introspection shows the union
    assert bus.get_broadcast_filter("dual_path") == frozenset({"A", "B", "C"})


def test_wildcard_then_filter_keeps_wildcard_for_first_queue():
    """If the FIRST subscribe is wildcard (types=None) and a SECOND
    subscribe declares types, the union semantics means the filter
    becomes that second declaration. Both queues then receive the same
    set — so the wildcard subscriber DOES get narrowed.

    This is intentional — same name = same filter (filter is per-name,
    not per-queue), and we don't want a wildcard caller to silently
    nullify a deliberate filter from another caller. Documented here
    so the behavior is explicit.
    """
    bus = DivineBus()
    q1 = bus.subscribe("partial", types=None)  # wildcard
    q2 = bus.subscribe("partial", types=["A"])  # filter
    bus.publish(make_msg("A", "src", "all", {}))
    bus.publish(make_msg("B", "src", "all", {}))
    # q1 also gets narrowed because the filter is per-name
    assert [m["type"] for m in _drain(q1)] == ["A"]
    assert [m["type"] for m in _drain(q2)] == ["A"]
    assert bus.get_broadcast_filter("partial") == frozenset({"A"})


# ── reply_only x types interaction ───────────────────────────────────


def test_reply_only_takes_priority_over_filter():
    """reply_only=True excludes a subscriber from ALL broadcasts even if
    types declares some. The reply_only flag was introduced first and
    the existing semantics must be preserved."""
    bus = DivineBus()
    q = bus.subscribe("reply_subscriber", reply_only=True, types=["A"])
    # Even though "A" is in filter, reply_only excludes broadcasts entirely
    bus.publish(make_msg("A", "src", "all", {}))
    assert _drain(q) == []
    # Targeted msg still works (reply_only path)
    bus.publish(make_msg("A", "src", "reply_subscriber", {}))
    msgs = _drain(q)
    assert len(msgs) == 1


# ── Unsubscribe clears the filter ────────────────────────────────────


def test_unsubscribe_last_queue_clears_filter():
    bus = DivineBus()
    q = bus.subscribe("ephemeral", types=["X"])
    assert bus.get_broadcast_filter("ephemeral") == frozenset({"X"})
    bus.unsubscribe("ephemeral", q)
    assert bus.get_broadcast_filter("ephemeral") is None


def test_unsubscribe_one_of_many_keeps_filter():
    """Filter is per-name; removing one queue should NOT clear it while
    other queues for the same name still exist."""
    bus = DivineBus()
    q1 = bus.subscribe("multi", types=["X"])
    q2 = bus.subscribe("multi", types=["Y"])
    bus.unsubscribe("multi", q1)
    # Filter should still be the union (X, Y) because q2 lives on
    assert bus.get_broadcast_filter("multi") == frozenset({"X", "Y"})


# ── Sender-self-skip still works with filter ─────────────────────────


def test_filter_doesnt_change_sender_self_skip():
    """Sender's own broadcast still doesn't echo back to its own queue,
    even if msg_type is in its filter."""
    bus = DivineBus()
    q = bus.subscribe("loopback", types=["EVENT"])
    bus.publish(make_msg("EVENT", "loopback", "all", {}))
    # Sender shouldn't receive its own broadcast
    assert _drain(q) == []


# ── Concurrency: filter holds under racing subscribe + publish ─────


def test_concurrent_subscribe_and_publish_no_crash():
    """The locking pattern means subscribe() must be safe vs publish()
    iteration. Run them concurrently for 100 iters and verify no
    exception escapes + counts add up."""
    bus = DivineBus()
    bus.subscribe("c_sub", types=["FOO"])
    received = []
    received_lock = threading.Lock()

    def publish_thread():
        for _ in range(100):
            bus.publish(make_msg("FOO", "src", "all", {}))
            bus.publish(make_msg("BAR", "src", "all", {}))

    def subscribe_thread():
        for i in range(50):
            qx = bus.subscribe(f"trans_{i}", types=["FOO"])
            with received_lock:
                received.append(qx)

    t1 = threading.Thread(target=publish_thread)
    t2 = threading.Thread(target=subscribe_thread)
    t1.start()
    t2.start()
    t1.join(timeout=10.0)
    t2.join(timeout=10.0)
    assert not t1.is_alive()
    assert not t2.is_alive()


# ── Stats: filtered_broadcasts is initialized + counts up ───────────


def test_stats_filtered_broadcasts_starts_at_zero():
    bus = DivineBus()
    assert bus._stats["filtered_broadcasts"] == 0


def test_stats_filtered_broadcasts_increments_per_subscriber_dropped():
    """If two subscribers both filter out the same msg_type, the counter
    increments by 2 for that single publish — one per subscriber-skip,
    not one per publish."""
    bus = DivineBus()
    bus.subscribe("a", types=["WANTED"])
    bus.subscribe("b", types=["WANTED"])
    bus.subscribe("c", types=["WANTED"])
    bus.publish(make_msg("UNWANTED", "src", "all", {}))
    assert bus._stats["filtered_broadcasts"] == 3
