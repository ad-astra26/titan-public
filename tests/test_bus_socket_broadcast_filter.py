"""Tests for BusSocketServer broadcast filter at publish time.

Post-rFP_worker_broadcast_topics_completion §4.C contract (2026-05-12):
the `_HIGH_RATE_BROADCAST_TYPES` stopgap was retired (deleted entirely)
after all 21 workers declared explicit `ModuleSpec.broadcast_topics` and
both the Python broker (this file) + Rust DivineBus broker (titan-rust/
crates/titan-bus/src/broker.rs `fanout`) implement the same filter.

Contract:
  • dst != "all"  → targeted name-based routing — bypasses topics filter.
  • dst == "all" + subscribed_topics non-empty → strict opt-in (msg type
    must be in the set).
  • dst == "all" + subscribed_topics EMPTY → WARN+drop (loud-fail). Every
    worker MUST declare `broadcast_topics` on its ModuleSpec; reaching
    this branch is a SPEC violation surfaced via logger.warning.

History — superseded contracts:
  • 2026-04-30: added `_HIGH_RATE_BROADCAST_TYPES` frozenset as stopgap
    fallback for unmigrated subscribe-all workers (closed fleet-wide
    queue-full flood + heap leak).
  • 2026-05-09: expanded the stopgap with 6 Rust-only Schumann-rate
    types after C-S5 unstuck Phase C inner-trinity Rust daemons.
  • 2026-05-12: deleted entirely after lockstep tests confirm all
    workers have explicit filters (per rFP §4.C + soak gate).

See bus_socket.py:publish() docstring.
"""
import logging

import pytest

from titan_hcl import bus
from titan_hcl.core.bus_socket import BusSocketServer, BrokerSubscriber


@pytest.fixture
def server():
    """A real BusSocketServer (not started) for testing publish() in isolation."""
    s = BusSocketServer(titan_id="T1", authkey=b"x" * 32, on_inbound_publish=lambda m: None)
    yield s


def _make_sub(server, name, subscribed_topics=None, reply_only=False):
    """Inject a fake subscriber into the server's _subscribers dict.

    D-SPEC-42 (SPEC v1.4.0): `reply_only=True` declares the subscriber
    consumes ONLY targeted dst=<name> messages — broker silently skips
    it from dst='all' fan-out (no enqueue, no warn, no drop counter).
    """
    sub = BrokerSubscriber(name=name, conn=None, addr="test")  # conn None — we don't actually send
    if subscribed_topics:
        sub.subscribed_topics.update(subscribed_topics)
    sub.reply_only = bool(reply_only)
    with server._subs_lock:
        server._subscribers[name] = sub
    return sub


def _enqueue_count(server, sub, msg):
    """Track how many times _enqueue_to is called for this sub by patching."""
    calls = []
    original = server._enqueue_to
    def spy(s, m):
        if s is sub:
            calls.append(m.get("type"))
    server._enqueue_to = spy
    try:
        server.publish(msg)
    finally:
        server._enqueue_to = original
    return calls


# ── Post-§4.C: empty topics = WARN + drop on ALL broadcasts ────────────────

@pytest.mark.parametrize("msg_type", [
    bus.SPHERE_PULSE,
    bus.PI_HEARTBEAT_UPDATED,
    bus.BIG_PULSE,
    bus.SPIRIT_STATE,
    bus.TOPOLOGY_STATE_UPDATED,
    bus.MEDITATION_COMPLETE,
    bus.MODULE_READY,
    "ANY_CUSTOM_TYPE",
])
def test_empty_topics_subscriber_drops_all_broadcasts(server, caplog, msg_type):
    """Post-§4.C: a subscriber with empty subscribed_topics receives NO
    broadcasts (former 'subscribe-all' mode is retired). The drop logs
    a WARN so regressions are visible.
    """
    sub = _make_sub(server, "no_topics_worker")  # empty subscribed_topics
    with caplog.at_level(logging.WARNING, logger="titan_hcl.core.bus_socket"):
        calls = _enqueue_count(server, sub,
            {"type": msg_type, "dst": "all", "payload": {}})
    assert calls == [], (
        f"{msg_type} delivered to empty-topics sub — post-§4.C contract broken"
    )
    assert any(
        "empty broadcast_topics" in rec.message
        for rec in caplog.records
    ), "expected WARN about empty broadcast_topics"


# ── Explicit topic subscription — strict opt-in ────────────────────────────

def test_explicit_topic_filter_blocks_unsubscribed_types(server):
    """Worker subscribed to MEDITATION_COMPLETE doesn't receive SPHERE_PULSE."""
    sub = _make_sub(server, "backup",
                    subscribed_topics={bus.MEDITATION_COMPLETE, bus.BACKUP_TRIGGER_MANUAL})
    calls = _enqueue_count(server, sub,
        {"type": bus.SPHERE_PULSE, "dst": "all", "payload": {}})
    assert calls == []


def test_explicit_topic_filter_passes_subscribed_type(server):
    """Worker subscribed to MEDITATION_COMPLETE receives that type."""
    sub = _make_sub(server, "backup",
                    subscribed_topics={bus.MEDITATION_COMPLETE, bus.BACKUP_TRIGGER_MANUAL})
    calls = _enqueue_count(server, sub,
        {"type": bus.MEDITATION_COMPLETE, "dst": "all", "payload": {}})
    assert calls == [bus.MEDITATION_COMPLETE]


def test_explicit_topic_can_opt_into_former_high_rate(server):
    """Workers that explicitly opt into formerly-blocked high-rate types
    DO receive them — the filter is fully under per-worker control now.
    """
    sub = _make_sub(server, "spirit_consumer",
                    subscribed_topics={bus.SPHERE_PULSE})
    calls = _enqueue_count(server, sub,
        {"type": bus.SPHERE_PULSE, "dst": "all", "payload": {}})
    assert calls == [bus.SPHERE_PULSE]


# ── Targeted (named) routing — filter does NOT apply ────────────────────────

def test_targeted_message_bypasses_filter_even_with_empty_topics(server):
    """dst='backup' SPHERE_PULSE goes through even though sub has empty
    topics — targeted routing is for RPC RESPONSE / SHUTDOWN / etc. and
    MUST work regardless of broadcast filter.
    """
    sub = _make_sub(server, "backup")  # empty topics
    calls = _enqueue_count(server, sub,
        {"type": bus.SPHERE_PULSE, "dst": "backup", "payload": {}})
    assert calls == [bus.SPHERE_PULSE]


def test_targeted_to_other_subscriber_not_delivered(server):
    """dst='backup' is NOT delivered to 'other_worker'."""
    _make_sub(server, "backup", subscribed_topics={bus.MEDITATION_COMPLETE})
    other = _make_sub(server, "other_worker",
                      subscribed_topics={bus.MEDITATION_COMPLETE})
    calls_other = _enqueue_count(server, other,
        {"type": bus.MEDITATION_COMPLETE, "dst": "backup", "payload": {}})
    assert calls_other == []


# ── §4.C stopgap retirement: dead-code check ───────────────────────────────

def test_high_rate_broadcast_types_is_retired(server):
    """Post-§4.C: the `_HIGH_RATE_BROADCAST_TYPES` frozenset MUST be
    deleted entirely. Any future regression that re-introduces the
    stopgap violates `feedback_no_quick_patches_only_spec_correct_solutions
    .md` — file a new SPEC-correct bug instead.
    """
    assert not hasattr(server, "_HIGH_RATE_BROADCAST_TYPES"), (
        "_HIGH_RATE_BROADCAST_TYPES stopgap was retired per rFP §4.C — "
        "if a high-rate broadcast is flooding queues, fix the producer or "
        "extend per-worker broadcast_topics; do NOT re-add the stopgap."
    )


# ── D-SPEC-42 (SPEC v1.4.0): reply_only silent skip ────────────────────────

@pytest.mark.parametrize("msg_type", [
    bus.SPHERE_PULSE,
    bus.MEDITATION_COMPLETE,
    bus.MODULE_READY,
    "ANY_CUSTOM_TYPE",
])
def test_reply_only_subscriber_silently_skipped_on_broadcast(
    server, caplog, msg_type
):
    """D-SPEC-42: reply_only=True subscriber receives NO broadcasts AND
    the broker does NOT log a WARN about empty topics. The skip is silent
    by design — these subscribers (RPC reply queues, writer services,
    proxy aliases) declared they don't consume broadcasts.
    """
    sub = _make_sub(server, "rpc_reply_queue", reply_only=True)
    with caplog.at_level(logging.WARNING, logger="titan_hcl.core.bus_socket"):
        calls = _enqueue_count(server, sub,
            {"type": msg_type, "dst": "all", "payload": {}})
    assert calls == [], (
        f"{msg_type} delivered to reply_only sub — D-SPEC-42 contract broken"
    )
    # Crucially: NO WARN. This is the architectural difference from the
    # empty-topics-no-reply_only regression case above.
    assert not any(
        "empty broadcast_topics" in rec.message
        for rec in caplog.records
    ), (
        "reply_only=True must produce ZERO warn output — silent skip is "
        "the SPEC v1.4.0 contract per D-SPEC-42"
    )


def test_reply_only_subscriber_receives_targeted(server):
    """D-SPEC-42: reply_only=True subscriber DOES receive targeted
    dst=<name> messages (RPC RESPONSE, MODULE_SHUTDOWN, etc.). The
    flag only affects broadcast fan-out — targeted routing is
    unconditional.
    """
    sub = _make_sub(server, "rpc_reply_queue", reply_only=True)
    calls = _enqueue_count(server, sub,
        {"type": "RESPONSE", "dst": "rpc_reply_queue",
         "payload": {"result": "ok"}})
    assert calls == ["RESPONSE"]


def test_reply_only_takes_precedence_over_topics(server):
    """D-SPEC-42: if a subscriber declares BOTH reply_only=True AND
    topics=[non-empty] (pathological — canonical pattern is
    reply_only=True ∧ topics=[]), reply_only wins. The subscriber
    is silently skipped from broadcasts.
    """
    sub = _make_sub(
        server,
        "weird_pattern_sub",
        subscribed_topics={bus.MEDITATION_COMPLETE},
        reply_only=True,
    )
    calls = _enqueue_count(server, sub,
        {"type": bus.MEDITATION_COMPLETE, "dst": "all", "payload": {}})
    assert calls == [], (
        "reply_only=True must take precedence over topics filter — "
        "matches Rust broker.rs::fanout dispatch order"
    )


def test_brokersubscriber_default_reply_only_is_false(server):
    """D-SPEC-42: backward compatibility — BrokerSubscriber instances
    default `reply_only=False` (broadcast-consumer intent), preserving
    byte-identical behavior for pre-v1.4.0 clients that omit the field
    from BUS_SUBSCRIBE payload.
    """
    sub = _make_sub(server, "default_sub",
                    subscribed_topics={bus.MEDITATION_COMPLETE})
    assert sub.reply_only is False
    # And the subscriber DOES receive its subscribed broadcast.
    calls = _enqueue_count(server, sub,
        {"type": bus.MEDITATION_COMPLETE, "dst": "all", "payload": {}})
    assert calls == [bus.MEDITATION_COMPLETE]
