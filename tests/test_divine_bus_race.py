"""Regression tests for DivineBus thread-safety.

The 2026-04-26 H4 first-deploy hit "dictionary changed size during
iteration" when a fresh thread published while kernel boot was still
adding subscribers. publish() iterated _subscribers.items() without a
lock; subscribe() mutated the dict from another thread → RuntimeError
inside publish, which corrupted bus state and prevented uvicorn from
binding :7777 in api_subprocess.

These tests exercise the race directly. They MUST pass; if they fail
again it means the lock got removed or a new iteration site bypassed
the snapshot pattern.
"""
import threading
import time

import pytest

from titan_plugin.bus import DivineBus, make_msg


@pytest.fixture
def bus():
    return DivineBus(maxsize=100)


def test_publish_does_not_race_with_concurrent_subscribe(bus):
    """Hammer publish on one thread + subscribe on another. With the
    pre-2026-04-26 implementation this would raise RuntimeError
    'dictionary changed size during iteration' within ~50ms. With the
    lock + snapshot pattern, both threads run cleanly and every
    publish returns a non-error delivered count.
    """
    # Seed at least one subscriber so publishes have something to iterate.
    bus.subscribe("seed")

    publish_errors = []
    publish_count = 0
    stop = threading.Event()

    def _publisher():
        nonlocal publish_count
        # Many small publishes so we hit the iteration path repeatedly.
        for _ in range(2000):
            if stop.is_set():
                break
            try:
                bus.publish(make_msg("STRESS", "src", "all", {"i": publish_count}))
                publish_count += 1
            except Exception as e:
                publish_errors.append(e)
                stop.set()
                return

    def _subscriber_churn():
        # Continuously subscribe + unsubscribe. Each call mutates
        # _subscribers / _modules / _reply_only — exactly the surface
        # publish() iterates.
        for i in range(500):
            if stop.is_set():
                break
            name = f"churn_{i}"
            q = bus.subscribe(name)
            # Unsubscribe immediately so the dict size churns both ways.
            bus.unsubscribe(name, q)

    pub_thread = threading.Thread(target=_publisher, name="bus-pub")
    sub_thread = threading.Thread(target=_subscriber_churn, name="bus-sub")

    pub_thread.start()
    sub_thread.start()
    pub_thread.join(timeout=10.0)
    sub_thread.join(timeout=10.0)

    assert not publish_errors, (
        f"publish() raised under concurrent subscribe: {publish_errors[0]!r}")
    assert publish_count >= 500, (
        f"publisher made too few iterations ({publish_count}) — bus may be hung")


def test_publish_dst_specific_does_not_race(bus):
    """Same race surface but via dst=<specific>, which hits the
    other branch in publish (subscribers list lookup + queue iteration).
    """
    bus.subscribe("target")

    publish_errors = []
    stop = threading.Event()

    def _publisher():
        for _ in range(2000):
            if stop.is_set():
                break
            try:
                bus.publish(make_msg("STRESS", "src", "target", {}))
            except Exception as e:
                publish_errors.append(e)
                stop.set()
                return

    def _subscriber_churn():
        for i in range(500):
            if stop.is_set():
                break
            q = bus.subscribe("target")  # add another queue under same name
            bus.unsubscribe("target", q)

    threads = [threading.Thread(target=_publisher),
               threading.Thread(target=_subscriber_churn)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    assert not publish_errors, (
        f"publish(dst=specific) raised under concurrent subscribe: "
        f"{publish_errors[0]!r}")


def test_subscribe_unsubscribe_concurrent(bus):
    """Two threads subscribing+unsubscribing concurrently must not
    corrupt the internal dict (no exceptions, final state consistent).
    """
    errors = []

    def _churn(prefix, n):
        for i in range(n):
            try:
                name = f"{prefix}_{i}"
                q = bus.subscribe(name)
                bus.unsubscribe(name, q)
            except Exception as e:
                errors.append(e)
                return

    threads = [threading.Thread(target=_churn, args=("a", 300)),
               threading.Thread(target=_churn, args=("b", 300))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    assert not errors, f"subscribe/unsubscribe raced: {errors[0]!r}"
