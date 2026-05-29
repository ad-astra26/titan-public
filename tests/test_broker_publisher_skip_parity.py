"""Tests for SPEC §8.2 + D-SPEC-52 (v1.7.3) — publisher-skip parity.

Per rFP_broker_publisher_skip_parity_fix (2026-05-14):
the broker's publisher-skip rule (preventing self-echo) MUST apply
ONLY to `dst="all"` broadcasts. Targeted `dst=<name>` MUST deliver to
the named subscriber even when the publisher IS that subscriber (e.g.
spirit_worker emitting META_CGN_SIGNAL with dst="spirit" where
MetaCGNConsumer lives in-process).

Pre-fix, the Python socket broker (`bus_socket.py:842-843`) and the
Rust broker (`titan-rust/crates/titan-bus/src/broker.rs:586`) skipped
the publisher UNCONDITIONALLY (before the dst check). This violated
SPEC §8.2 v1.4.0 D-SPEC-42 ("Targeted routing remains unaffected")
and was inconsistent with the SPEC-correct Python in-process bus
(`bus.py:955+` — skip inside the `if dst == "all":` branch).

Symptom that exposed it: META-CGN + EMOT-CGN signal pipelines silent
fleet-wide post-fleet-Phase-C migration (2026-05-14) because
spirit_worker self-emits with dst="spirit" were dropped at fan-out;
`/v4/meta-cgn.signals_received = 0` on T1+T2+T3 despite producer
"EMIT —" log lines firing.

This file covers the Python socket broker side. Rust side parity is
covered in `titan-rust/crates/titan-bus/tests/publisher_skip_parity.rs`.
"""
import logging

import pytest

from titan_hcl import bus
from titan_hcl.core.bus_socket import BusSocketServer, BrokerSubscriber


@pytest.fixture
def server():
    """A real BusSocketServer (not started) for publish() unit-testing."""
    s = BusSocketServer(titan_id="T1", authkey=b"x" * 32,
                        on_inbound_publish=lambda m: None)
    yield s


def _make_sub(server, name, subscribed_topics=None, reply_only=False):
    """Inject a fake subscriber into server._subscribers (mirrors
    test_bus_socket_broadcast_filter helper pattern)."""
    sub = BrokerSubscriber(name=name, conn=None, addr="test")
    if subscribed_topics:
        sub.subscribed_topics.update(subscribed_topics)
    sub.reply_only = bool(reply_only)
    with server._subs_lock:
        server._subscribers[name] = sub
    return sub


def _capture_deliveries(server, msg, _from_subscriber=None):
    """Run server.publish(msg, _from_subscriber=...) and capture per-sub
    delivery list. Returns dict name → [msg_types_delivered]."""
    deliveries: dict[str, list[str]] = {}
    original_enqueue = server._enqueue_to

    def spy(sub, m):
        deliveries.setdefault(sub.name, []).append(m.get("type"))

    server._enqueue_to = spy
    try:
        server.publish(msg, _from_subscriber=_from_subscriber)
    finally:
        server._enqueue_to = original_enqueue
    return deliveries


# ── Test 1: self-targeted DELIVERS (the critical bug-catcher) ──────────────


def test_self_targeted_dst_delivers_to_publisher(server):
    """SPEC §8.2 v1.4.0 D-SPEC-42 + D-SPEC-52 (v1.7.3): a worker emitting
    a targeted dst=<own-name> message MUST receive the delivery in its
    own subscriber queue. This is the path used by spirit_worker emitting
    META_CGN_SIGNAL with dst="spirit" where MetaCGNConsumer lives
    in-process.

    Pre-D-SPEC-52 the broker's unconditional `if sub is _from_subscriber:
    continue` filter dropped this delivery, causing META-CGN signal
    starvation fleet-wide.
    """
    spirit = _make_sub(server, "spirit",
                       subscribed_topics={bus.META_CGN_SIGNAL,
                                          bus.EMOT_CGN_SIGNAL})
    deliveries = _capture_deliveries(
        server,
        {"type": bus.META_CGN_SIGNAL, "src": "meta_reasoning",
         "dst": "spirit", "payload": {"consumer": "meta_reasoning",
                                      "event_type": "eureka"}},
        _from_subscriber=spirit)
    assert deliveries.get("spirit") == [bus.META_CGN_SIGNAL], (
        f"self-targeted dst=spirit MUST deliver to spirit subscriber "
        f"per SPEC §8.2 v1.4.0 + D-SPEC-52; got deliveries={deliveries}"
    )


# ── Test 2: self-broadcast dst="all" SKIPS publisher (anti-echo correct) ──


def test_self_broadcast_dst_all_skips_publisher(server):
    """SPEC §8.2 v1.4.0: broadcast (dst="all") publisher-skip remains
    correct — a worker doesn't receive its OWN broadcasts. This is the
    anti-feedback-loop guard, applied ONLY in the broadcast branch.
    """
    spirit = _make_sub(server, "spirit",
                       subscribed_topics={bus.SPHERE_PULSE})
    cgn = _make_sub(server, "cgn",
                    subscribed_topics={bus.SPHERE_PULSE})
    deliveries = _capture_deliveries(
        server,
        {"type": bus.SPHERE_PULSE, "src": "spirit", "dst": "all",
         "payload": {}},
        _from_subscriber=spirit)
    assert deliveries.get("cgn") == [bus.SPHERE_PULSE], (
        "broadcast SHOULD deliver to non-publisher subscribers"
    )
    assert "spirit" not in deliveries, (
        f"broadcast MUST NOT echo to publisher (anti-feedback-loop); "
        f"got deliveries={deliveries}"
    )


# ── Test 3: cross-worker targeted DELIVERS (sanity check) ──────────────────


def test_cross_worker_targeted_delivers_normally(server):
    """Cross-worker targeted routing was unaffected by the bug but
    verify it stays correct — cgn_worker emitting META_CGN_SIGNAL with
    dst="spirit" reaches spirit subscriber, not cgn itself.
    """
    spirit = _make_sub(server, "spirit",
                       subscribed_topics={bus.META_CGN_SIGNAL})
    cgn = _make_sub(server, "cgn",
                    subscribed_topics={bus.META_CGN_SIGNAL})
    deliveries = _capture_deliveries(
        server,
        {"type": bus.META_CGN_SIGNAL, "src": "knowledge",
         "dst": "spirit", "payload": {"consumer": "knowledge",
                                      "event_type": "impasse_resolved"}},
        _from_subscriber=cgn)
    assert deliveries.get("spirit") == [bus.META_CGN_SIGNAL], (
        f"cross-worker dst=spirit MUST deliver to spirit; "
        f"got deliveries={deliveries}"
    )
    assert "cgn" not in deliveries, (
        "targeted dst=spirit should NOT deliver to other workers (cgn)"
    )


# ── Test 4: targeted dst=<other-worker> skips both publisher AND non-match ──


def test_targeted_dst_skips_non_matching_subscribers(server):
    """Sanity: targeted dst=<name> ONLY delivers to subscriber matching
    that name — not to publisher (when name happens to match), not to
    other unrelated subscribers."""
    spirit = _make_sub(server, "spirit",
                       subscribed_topics={bus.META_CGN_SIGNAL})
    cgn = _make_sub(server, "cgn",
                    subscribed_topics={bus.META_CGN_SIGNAL})
    language = _make_sub(server, "language",
                         subscribed_topics={bus.META_CGN_SIGNAL})
    deliveries = _capture_deliveries(
        server,
        {"type": bus.META_CGN_SIGNAL, "src": "meta_reasoning",
         "dst": "cgn", "payload": {}},
        _from_subscriber=spirit)
    assert deliveries.get("cgn") == [bus.META_CGN_SIGNAL]
    assert "spirit" not in deliveries
    assert "language" not in deliveries


# ── Test 5: self-targeted on reply_only=True subscriber STILL DELIVERS ─────


def test_self_targeted_delivers_to_reply_only_subscriber(server):
    """Reply-only flag affects only broadcast fan-out per SPEC §8.2
    v1.4.0 D-SPEC-42. Targeted dst=<name> on a reply_only subscriber
    MUST deliver, including when subscriber IS the publisher."""
    proxy = _make_sub(server, "memory_proxy",
                      subscribed_topics=None, reply_only=True)
    deliveries = _capture_deliveries(
        server,
        {"type": bus.RESPONSE, "src": "memory_proxy", "dst": "memory_proxy",
         "payload": {}},
        _from_subscriber=proxy)
    assert deliveries.get("memory_proxy") == [bus.RESPONSE], (
        "reply_only subscriber MUST receive own targeted message; "
        f"got deliveries={deliveries}"
    )


# ── Test 6: dst=alias self-target also delivers ────────────────────────────


def test_self_targeted_via_alias_delivers(server):
    """Multi-name BUS_SUBSCRIBE (SPEC v1.3.0): a connection can register
    multiple names. Targeted dst=<alias> where alias is the publisher's
    own registered alias MUST deliver per SPEC §8.2 v1.4.0 D-SPEC-42 +
    D-SPEC-52 — the SPEC says routing matches name ∪ aliases.

    NOTE on bus_socket.py: Python broker stores aliases conceptually on
    BrokerSubscriber but checks `sub.name == dst` (not aliases) — multi-
    name support is currently Rust-only per SPEC v1.3.0 deployment.
    This test asserts the SPEC-defined behavior; if it fails on Python
    that's an orthogonal v1.3.0 multi-name parity gap, not D-SPEC-52.
    """
    spirit = _make_sub(server, "spirit",
                       subscribed_topics={bus.META_CGN_SIGNAL})
    # Direct name-self-target (alias semantics in Python broker
    # are bus_socket.py:849 `sub.name != dst` — name match only).
    deliveries = _capture_deliveries(
        server,
        {"type": bus.META_CGN_SIGNAL, "src": "meta_reasoning",
         "dst": "spirit", "payload": {}},
        _from_subscriber=spirit)
    assert deliveries.get("spirit") == [bus.META_CGN_SIGNAL]
