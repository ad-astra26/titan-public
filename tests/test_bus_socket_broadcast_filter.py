"""Tests for BusSocketServer broadcast filter at publish time.

Closes the 2026-04-30 fleet-wide queue-full flood + heap leak:
broker was delivering ALL `dst="all"` broadcasts to ALL subscribers,
ignoring subscribed_topics. SPHERE_PULSE / SPIRIT_STATE / PI_HEARTBEAT_UPDATED
high-rate broadcasts saturated every worker queue and grew BrokerSubscriber
coalesce_index dicts unbounded.

Fix: publish() now filters at fan-out time:
  - subscribed_topics non-empty → opt-in (msg type must be in set)
  - subscribed_topics empty → block HIGH_RATE_BROADCAST_TYPES; allow others
  - dst == <name> → name-based routing, filter does NOT apply

See bus_socket.py:_HIGH_RATE_BROADCAST_TYPES + publish() docstring.
"""
import pytest

from titan_plugin import bus
from titan_plugin.core.bus_socket import BusSocketServer, BrokerSubscriber


@pytest.fixture
def server():
    """A real BusSocketServer (not started) for testing publish() in isolation."""
    s = BusSocketServer(titan_id="T1", authkey=b"x" * 32, on_inbound_publish=lambda m: None)
    yield s


def _make_sub(server, name, subscribed_topics=None):
    """Inject a fake subscriber into the server's _subscribers dict."""
    sub = BrokerSubscriber(name=name, conn=None, addr="test")  # conn is None — we don't actually send
    if subscribed_topics:
        sub.subscribed_topics.update(subscribed_topics)
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


# ── HIGH_RATE filter ────────────────────────────────────────────────────────

def test_high_rate_broadcast_blocked_for_subscribe_all(server):
    """SPHERE_PULSE broadcast NOT delivered to subscriber with empty topics."""
    sub = _make_sub(server, "test_worker")
    calls = _enqueue_count(server, sub,
        {"type": bus.SPHERE_PULSE, "dst": "all", "payload": {}})
    assert calls == [], "SPHERE_PULSE should be blocked for subscribe-all"


def test_pi_heartbeat_blocked_for_subscribe_all(server):
    sub = _make_sub(server, "w")
    calls = _enqueue_count(server, sub,
        {"type": bus.PI_HEARTBEAT_UPDATED, "dst": "all", "payload": {}})
    assert calls == []


def test_big_pulse_blocked_for_subscribe_all(server):
    sub = _make_sub(server, "w")
    calls = _enqueue_count(server, sub,
        {"type": bus.BIG_PULSE, "dst": "all", "payload": {}})
    assert calls == []


def test_spirit_state_blocked_for_subscribe_all(server):
    sub = _make_sub(server, "w")
    calls = _enqueue_count(server, sub,
        {"type": bus.SPIRIT_STATE, "dst": "all", "payload": {}})
    assert calls == []


def test_topology_state_blocked_for_subscribe_all(server):
    sub = _make_sub(server, "w")
    calls = _enqueue_count(server, sub,
        {"type": bus.TOPOLOGY_STATE_UPDATED, "dst": "all", "payload": {}})
    assert calls == []


# ── Low-rate broadcast still flows ──────────────────────────────────────────

def test_low_rate_broadcast_delivered_to_subscribe_all(server):
    """MEDITATION_COMPLETE (not in high-rate set) STILL flows on subscribe-all."""
    sub = _make_sub(server, "w")
    calls = _enqueue_count(server, sub,
        {"type": bus.MEDITATION_COMPLETE, "dst": "all", "payload": {}})
    assert calls == [bus.MEDITATION_COMPLETE]


# ── Explicit topic subscription ─────────────────────────────────────────────

def test_explicit_topic_filter_blocks_other_types(server):
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


def test_explicit_topic_can_opt_into_high_rate(server):
    """Worker that explicitly subscribed to SPHERE_PULSE STILL receives it."""
    sub = _make_sub(server, "spirit_consumer",
                    subscribed_topics={bus.SPHERE_PULSE})
    calls = _enqueue_count(server, sub,
        {"type": bus.SPHERE_PULSE, "dst": "all", "payload": {}})
    assert calls == [bus.SPHERE_PULSE], (
        "Workers with SPHERE_PULSE in subscribed_topics MUST still receive it")


# ── Targeted (named) routing — filter does NOT apply ────────────────────────

def test_targeted_message_bypasses_filter(server):
    """dst='backup' SPHERE_PULSE goes through even though high-rate."""
    sub = _make_sub(server, "backup")  # empty topics
    calls = _enqueue_count(server, sub,
        {"type": bus.SPHERE_PULSE, "dst": "backup", "payload": {}})
    assert calls == [bus.SPHERE_PULSE], (
        "Targeted messages bypass broadcast filter")


def test_targeted_to_other_subscriber_not_delivered(server):
    """dst='backup' is NOT delivered to 'other_worker'."""
    backup = _make_sub(server, "backup")
    other = _make_sub(server, "other_worker")
    calls_other = _enqueue_count(server, other,
        {"type": bus.MEDITATION_COMPLETE, "dst": "backup", "payload": {}})
    assert calls_other == []


# ── HIGH_RATE_BROADCAST_TYPES is comprehensive but not too aggressive ──────

def test_high_rate_set_is_frozen(server):
    """HIGH_RATE_BROADCAST_TYPES is a frozenset (immutable, hashable)."""
    assert isinstance(server._HIGH_RATE_BROADCAST_TYPES, frozenset)


def test_high_rate_set_includes_documented_types(server):
    """All 5 known high-rate types are in the filter set."""
    expected = {
        bus.SPHERE_PULSE,
        bus.PI_HEARTBEAT_UPDATED,
        bus.BIG_PULSE,
        bus.SPIRIT_STATE,
        bus.TOPOLOGY_STATE_UPDATED,
    }
    assert expected.issubset(server._HIGH_RATE_BROADCAST_TYPES)


def test_low_rate_types_NOT_in_high_rate_set(server):
    """MEDITATION_COMPLETE / BACKUP_TRIGGER_MANUAL / etc. should pass through."""
    safe_types = {
        bus.MEDITATION_COMPLETE,
        bus.BACKUP_TRIGGER_MANUAL,
        bus.MODULE_SHUTDOWN,
        bus.MODULE_READY,
    }
    for t in safe_types:
        assert t not in server._HIGH_RATE_BROADCAST_TYPES, (
            f"{t} must NOT be in high-rate set — it would break low-rate flow")
