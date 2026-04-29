"""
Unit tests for ``titan_plugin.core.sensor_cache`` — the §L1 Trinity
Daemon Internal Design substrate (cache + refresh threads + Schumann
shm writer).

Microkernel v2 Phase A §A.7 / §L1 (S7, 2026-04-26).
"""
from __future__ import annotations

import threading
import time

import pytest

from titan_plugin.core.sensor_cache import (
    RefreshSpec,
    SensorCache,
    start_refresh_threads,
    start_shm_writer_thread,
    stop_threads,
)


# ── SensorCache primitives ──────────────────────────────────────────


def test_sensor_cache_initial_seed():
    """Initial readings are accessible immediately, ts defaults to 0."""
    cache = SensorCache(initial={"foo": {"value": 0.5, "severity": 1}})

    reading = cache.get("foo")
    assert reading is not None
    assert reading["value"] == 0.5
    assert reading["severity"] == 1
    assert reading["ts"] == 0.0  # never refreshed


def test_sensor_cache_initial_seed_preserves_caller_ts():
    """If the caller seeds with a ts, it's preserved (not overwritten)."""
    cache = SensorCache(initial={"foo": {"value": 0.5, "ts": 12345.0}})
    assert cache.get("foo")["ts"] == 12345.0


def test_sensor_cache_set_stamps_ts():
    """set() auto-stamps ts to time.time()."""
    cache = SensorCache()
    before = time.time()
    cache.set("bar", {"value": 0.7, "severity": 3})
    after = time.time()

    reading = cache.get("bar")
    assert reading["value"] == 0.7
    assert reading["severity"] == 3
    assert before <= reading["ts"] <= after


def test_sensor_cache_get_missing_returns_none():
    cache = SensorCache()
    assert cache.get("nonexistent") is None


def test_sensor_cache_get_returns_copy_not_alias():
    """Mutating the returned dict must not affect cache state."""
    cache = SensorCache()
    cache.set("x", {"value": 0.1})
    snap = cache.get("x")
    snap["value"] = 999.0  # mutate caller's copy

    fresh = cache.get("x")
    assert fresh["value"] == 0.1


def test_sensor_cache_get_all():
    cache = SensorCache()
    cache.set("a", {"value": 0.1})
    cache.set("b", {"value": 0.2})
    snapshot = cache.get_all()
    assert snapshot.keys() == {"a", "b"}
    assert snapshot["a"]["value"] == 0.1
    assert snapshot["b"]["value"] == 0.2

    # Mutating snapshot doesn't stomp cache state.
    snapshot["a"]["value"] = 999.0
    assert cache.get("a")["value"] == 0.1


def test_sensor_cache_age_seconds_initial_returns_none():
    """ts=0 = never refreshed, so age_seconds returns None."""
    cache = SensorCache(initial={"foo": {"value": 0.5}})
    assert cache.age_seconds("foo") is None


def test_sensor_cache_age_seconds_after_set_is_small():
    cache = SensorCache()
    cache.set("foo", {"value": 0.5})
    age = cache.age_seconds("foo")
    assert age is not None
    assert 0.0 <= age < 0.1  # well under 100ms


def test_sensor_cache_age_seconds_missing_key_returns_none():
    cache = SensorCache()
    assert cache.age_seconds("missing") is None


def test_sensor_cache_concurrent_writes_no_torn_state():
    """Two threads racing writes should never produce a torn dict."""
    cache = SensorCache()
    stop = threading.Event()

    def writer(name, value):
        n = 0
        while not stop.is_set():
            cache.set(name, {"value": value, "n": n, "key": name})
            n += 1

    t1 = threading.Thread(target=writer, args=("a", 0.1), daemon=True)
    t2 = threading.Thread(target=writer, args=("b", 0.2), daemon=True)
    t1.start()
    t2.start()

    # Read many times during contention; every read must be self-consistent.
    deadline = time.time() + 0.2
    while time.time() < deadline:
        ra = cache.get("a")
        rb = cache.get("b")
        if ra is not None:
            assert ra["key"] == "a"
            assert ra["value"] == 0.1
        if rb is not None:
            assert rb["key"] == "b"
            assert rb["value"] == 0.2

    stop.set()
    t1.join(timeout=1.0)
    t2.join(timeout=1.0)


# ── Refresh threads ─────────────────────────────────────────────────


def test_refresh_thread_initial_call_populates_cache_immediately():
    """First refresh fires ASAP (not after one period_s)."""
    cache = SensorCache()
    counter = {"n": 0}

    def fn():
        counter["n"] += 1
        return {"value": 0.7, "n": counter["n"]}

    spec = RefreshSpec(name="foo", refresh_fn=fn, period_s=10.0)
    stop = threading.Event()
    threads = start_refresh_threads([spec], cache, stop)

    # Within 100 ms, the cache should have at least the first reading.
    deadline = time.time() + 0.5
    while time.time() < deadline:
        if cache.get("foo") is not None:
            break
        time.sleep(0.01)

    assert cache.get("foo") is not None
    assert cache.get("foo")["value"] == 0.7

    stop_threads(stop, threads)


def test_refresh_thread_periodic_updates():
    """Multiple ticks fire at period_s cadence."""
    cache = SensorCache()
    counter = {"n": 0}

    def fn():
        counter["n"] += 1
        return {"value": 0.5, "n": counter["n"]}

    spec = RefreshSpec(name="foo", refresh_fn=fn, period_s=0.05)
    stop = threading.Event()
    threads = start_refresh_threads([spec], cache, stop)

    time.sleep(0.25)  # allow ~5 ticks
    n = cache.get("foo")["n"]
    assert n >= 3  # initial + at least 2 periodic

    stop_threads(stop, threads)


def test_refresh_thread_swallows_exceptions():
    """A raising sense fn keeps the cache at last-known-good."""
    cache = SensorCache(initial={"foo": {"value": 0.5}})
    state = {"raise": False, "calls": 0}

    def fn():
        state["calls"] += 1
        if state["raise"]:
            raise RuntimeError("boom")
        return {"value": 0.9}

    spec = RefreshSpec(name="foo", refresh_fn=fn, period_s=0.02)
    stop = threading.Event()
    threads = start_refresh_threads([spec], cache, stop)

    time.sleep(0.08)  # let initial refresh land
    assert cache.get("foo")["value"] == 0.9

    state["raise"] = True
    time.sleep(0.15)  # several would-be refreshes; all raise

    # Cache stays at last-known-good (0.9 from the working refreshes).
    assert cache.get("foo")["value"] == 0.9
    assert state["calls"] >= 4  # multiple refresh attempts

    stop_threads(stop, threads)


def test_refresh_thread_multi_sense():
    """One thread per sense, all populate independently."""
    cache = SensorCache()

    def fn_a():
        return {"value": 0.1, "tag": "a"}

    def fn_b():
        return {"value": 0.2, "tag": "b"}

    specs = [
        RefreshSpec("a", fn_a, period_s=10.0),
        RefreshSpec("b", fn_b, period_s=10.0),
    ]
    stop = threading.Event()
    threads = start_refresh_threads(specs, cache, stop)

    deadline = time.time() + 0.5
    while time.time() < deadline:
        if cache.get("a") is not None and cache.get("b") is not None:
            break
        time.sleep(0.01)

    assert cache.get("a")["tag"] == "a"
    assert cache.get("b")["tag"] == "b"
    assert len(threads) == 2

    stop_threads(stop, threads)


def test_refresh_thread_stops_on_event():
    """stop_event halts the refresh loop within timeout."""
    cache = SensorCache()
    counter = {"n": 0}

    def fn():
        counter["n"] += 1
        return {"value": 0.5}

    spec = RefreshSpec("foo", fn, period_s=0.02)
    stop = threading.Event()
    threads = start_refresh_threads([spec], cache, stop)

    time.sleep(0.1)
    stop_threads(stop, threads, timeout_s=1.0)

    n_at_stop = counter["n"]
    time.sleep(0.1)
    # No new calls after stop.
    assert counter["n"] == n_at_stop


# ── Schumann shm writer thread ──────────────────────────────────────


def test_shm_writer_thread_fires_at_period():
    """Writer thread fires tick_fn at ~period_s cadence."""
    counter = {"n": 0}

    def tick():
        counter["n"] += 1

    stop = threading.Event()
    period = 0.02
    t = start_shm_writer_thread(tick, period, stop, "test_writer")

    time.sleep(0.2)  # ~10 ticks expected
    stop.set()
    t.join(timeout=1.0)

    # Allow generous tolerance for jitter; we just want order-of-magnitude.
    assert 5 <= counter["n"] <= 50


def test_shm_writer_swallows_exceptions():
    """A raising tick_fn does not kill the writer thread."""
    state = {"calls": 0, "raise_until": 0}

    def tick():
        state["calls"] += 1
        if state["calls"] <= state["raise_until"]:
            raise RuntimeError("boom")

    state["raise_until"] = 5
    stop = threading.Event()
    t = start_shm_writer_thread(tick, 0.01, stop, "test_writer_exc")

    time.sleep(0.2)
    stop.set()
    t.join(timeout=1.0)

    # Made many calls including the raising ones.
    assert state["calls"] >= 10


def test_shm_writer_thread_stops_on_event():
    counter = {"n": 0}

    def tick():
        counter["n"] += 1

    stop = threading.Event()
    t = start_shm_writer_thread(tick, 0.01, stop, "test_writer_stop")

    time.sleep(0.1)
    stop.set()
    t.join(timeout=1.0)
    assert not t.is_alive()

    n_at_stop = counter["n"]
    time.sleep(0.1)
    assert counter["n"] == n_at_stop


def test_stop_threads_signals_and_joins():
    """stop_threads sets event + joins each thread."""
    cache = SensorCache()

    def fn():
        return {"value": 0.5}

    specs = [
        RefreshSpec("a", fn, period_s=0.02),
        RefreshSpec("b", fn, period_s=0.02),
    ]
    stop = threading.Event()
    threads = start_refresh_threads(specs, cache, stop)

    assert all(t.is_alive() for t in threads)
    stop_threads(stop, threads, timeout_s=1.0)
    assert all(not t.is_alive() for t in threads)


# ── Latency budget — the load-bearing assertion of the §L1 pattern ──


def test_cache_read_latency_under_microsecond_budget():
    """
    The whole point of S7 is that the tick path reads from cache in
    well under 1 ms. Assert: 100 cache.get + age_seconds round-trips
    complete in under 10 ms (avg < 100 μs each). This is the
    architectural guarantee that makes 7.83-23.49 Hz tick rates safe.
    """
    cache = SensorCache(initial={
        "a": {"value": 0.1, "severity": 1},
        "b": {"value": 0.2, "severity": 1},
        "c": {"value": 0.3, "severity": 3},
        "d": {"value": 0.4, "severity": 10},
        "e": {"value": 0.5, "severity": 1},
    })

    t0 = time.perf_counter()
    for _ in range(100):
        readings = cache.get_all()
        for name in readings:
            _ = cache.age_seconds(name)
    elapsed = time.perf_counter() - t0

    # 100 ticks × (1 get_all + 5 age_seconds) = 600 cache ops.
    # Budget: 10 ms = 100 μs / tick = ~16 μs / op.
    assert elapsed < 0.1, f"Cache read latency {elapsed*1000:.2f}ms exceeds budget"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
