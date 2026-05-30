"""
Tests for the background snapshot builder threads in snapshot_builders.

Fix: T1-COORD-QUERYTHREAD-BACKLOG (2026-04-21). Before this refactor, the
get_coordinator / get_trinity / get_nervous_system handlers built their
snapshots inline in the QueryThread, blocking 460-2144ms per coord build
on T1 under dashboard-poll + ARC load. After: dedicated daemon threads
rebuild continuously in the background; handlers are cache reads.

These tests verify:
  - build_*_snapshot() functions work with empty / full state_refs.
  - start_snapshot_builder_threads() launches 3 threads and populates
    caches within a reasonable window.
  - Build failures are caught and cache keeps serving last-good.
"""

import time
import types
import pytest

# Phase 10E — snapshot builders relocated spirit_loop → logic/snapshot_builders.
from titan_hcl.logic import snapshot_builders


# ---------------------------------------------------------------------------
# build_coordinator_snapshot
# ---------------------------------------------------------------------------

def test_build_coordinator_snapshot_returns_none_when_coordinator_missing():
    assert snapshot_builders.build_coordinator_snapshot({}) is None
    assert snapshot_builders.build_coordinator_snapshot({"coordinator": None}) is None


def test_build_coordinator_snapshot_minimal_coordinator_only():
    """Coordinator alone is sufficient — other subsystems are all optional."""
    coord = types.SimpleNamespace(
        get_stats=lambda: {"commits": 42},
        _meta_engine=None,
        _meta_service=None,
        dreaming=None,
        inner=None,
        nervous_system=None,
    )
    snap = snapshot_builders.build_coordinator_snapshot({"coordinator": coord})
    assert snap is not None
    assert snap["commits"] == 42
    # Always-emit shape-stable keys
    assert snap["meta_reasoning"] == {}
    assert snap["meta_service"] == {}


def test_build_coordinator_snapshot_subsystem_exception_surfaces_at_warn():
    """A subsystem get_stats() throwing does not crash the builder."""
    coord = types.SimpleNamespace(
        get_stats=lambda: {"x": 1},
        _meta_engine=types.SimpleNamespace(
            get_stats=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
            get_audit_stats=lambda: {},
        ),
        _meta_service=None,
        dreaming=None,
        inner=None,
        nervous_system=None,
    )
    snap = snapshot_builders.build_coordinator_snapshot({"coordinator": coord})
    assert snap is not None
    assert snap["meta_reasoning"] == {}


# ---------------------------------------------------------------------------
# D-SPEC-143 — trinity + NS snapshot builders REMOVED (orphaned, 4Hz CPU hog)
# ---------------------------------------------------------------------------

def test_trinity_and_ns_builders_removed():
    """Regression guard: the orphaned trinity/NS builders + their caches/build
    fns are gone (D-SPEC-143 profiling — no reader: spirit_worker QueryThread
    retired, dashboard reads SHM-direct). Only the coordinator builder remains.
    """
    assert not hasattr(snapshot_builders, "build_trinity_snapshot")
    assert not hasattr(snapshot_builders, "build_nervous_system_snapshot")
    assert not hasattr(snapshot_builders, "_TRINITY_SNAPSHOT_CACHE")
    assert not hasattr(snapshot_builders, "_NS_SNAPSHOT_CACHE")
    assert not hasattr(snapshot_builders, "_TRINITY_SNAPSHOT_BUILDER_INTERVAL")
    assert not hasattr(snapshot_builders, "_NS_SNAPSHOT_BUILDER_INTERVAL")
    # coord builder survives
    assert hasattr(snapshot_builders, "build_coordinator_snapshot")
    assert hasattr(snapshot_builders, "_COORD_SNAPSHOT_CACHE")


# ---------------------------------------------------------------------------
# start_snapshot_builder_threads
# ---------------------------------------------------------------------------

def test_builder_threads_populate_coord_cache_quickly():
    """After start_snapshot_builder_threads, the coord cache populates within 2s."""
    # Reset cache to simulate fresh boot.
    snapshot_builders._COORD_SNAPSHOT_CACHE["data"] = None
    snapshot_builders._COORD_SNAPSHOT_CACHE["ts"] = 0.0

    coord = types.SimpleNamespace(
        get_stats=lambda: {"commits": 7},
        _meta_engine=None, _meta_service=None,
        dreaming=None, inner=None,
        nervous_system=types.SimpleNamespace(programs={"p": 1}),
    )
    refs = {"coordinator": coord}
    snapshot_builders.start_snapshot_builder_threads(refs, config={})

    deadline = time.time() + 2.0
    while time.time() < deadline:
        if snapshot_builders._COORD_SNAPSHOT_CACHE["data"] is not None:
            break
        time.sleep(0.05)

    assert snapshot_builders._COORD_SNAPSHOT_CACHE["data"] is not None, \
        "coord cache not populated within 2s"
    assert snapshot_builders._COORD_SNAPSHOT_CACHE["data"]["commits"] == 7


def test_builder_thread_survives_build_exceptions(monkeypatch):
    """If build_fn raises, cache keeps serving last-good; thread stays alive.

    NOTE: `build_coordinator_snapshot` deliberately ISOLATES per-subsystem
    `get_stats` failures (incl. coordinator.get_stats) and returns a partial
    dict with a `coordinator_error` marker rather than raising — so mocking
    coordinator.get_stats to raise no longer reaches the loop's except arm.
    To exercise the loop's genuine raise-resilience contract (the invariant
    that actually matters: a raising build_fn must not clobber last-good and
    must not kill the builder thread), force the build fn itself to raise.
    """
    # Prime cache with known-good data.
    snapshot_builders._COORD_SNAPSHOT_CACHE["data"] = {"last_good": True}
    snapshot_builders._COORD_SNAPSHOT_CACHE["ts"] = time.time()

    # Force the build fn itself to raise (genuine build_fn exception that the
    # loop's try/except must absorb without clobbering the cache).
    def _boom(_state_refs):
        raise RuntimeError("boom")
    monkeypatch.setattr(snapshot_builders, "build_coordinator_snapshot", _boom)

    coord = types.SimpleNamespace(
        get_stats=lambda: {}, _meta_engine=None, _meta_service=None,
        dreaming=None, inner=None, nervous_system=None,
    )
    refs = {"coordinator": coord}
    snapshot_builders.start_snapshot_builder_threads(refs, config={})

    # Wait a couple of builder cycles and verify cache still has last-good data.
    time.sleep(0.5)
    assert snapshot_builders._COORD_SNAPSHOT_CACHE["data"] == {"last_good": True}, \
        "builder exception should not clobber the last-good cache"
