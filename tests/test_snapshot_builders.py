"""
Tests for the background snapshot builder threads in spirit_loop.

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

from titan_plugin.modules import spirit_loop


# ---------------------------------------------------------------------------
# build_coordinator_snapshot
# ---------------------------------------------------------------------------

def test_build_coordinator_snapshot_returns_none_when_coordinator_missing():
    assert spirit_loop.build_coordinator_snapshot({}) is None
    assert spirit_loop.build_coordinator_snapshot({"coordinator": None}) is None


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
    snap = spirit_loop.build_coordinator_snapshot({"coordinator": coord})
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
    snap = spirit_loop.build_coordinator_snapshot({"coordinator": coord})
    assert snap is not None
    assert snap["meta_reasoning"] == {}


# ---------------------------------------------------------------------------
# build_trinity_snapshot
# ---------------------------------------------------------------------------

def test_build_trinity_snapshot_with_empty_refs_returns_default_shape():
    snap = spirit_loop.build_trinity_snapshot({}, config={})
    assert "spirit_tensor" in snap
    assert snap["body_values"] == [0.5] * 5
    assert snap["mind_values"] == [0.5] * 5


def test_build_trinity_snapshot_includes_subsystem_stats_when_present():
    refs = {
        "body_state": {"values": [0.1, 0.2, 0.3, 0.4, 0.5], "center_dist": 0.3},
        "mind_state": {"values": [0.9, 0.8, 0.7, 0.6, 0.5], "center_dist": 0.2},
        "filter_down": types.SimpleNamespace(get_stats=lambda: {"fd": 1}),
        "intuition": types.SimpleNamespace(get_stats=lambda: {"in": 2}),
    }
    snap = spirit_loop.build_trinity_snapshot(refs, config={})
    assert snap["body_values"] == [0.1, 0.2, 0.3, 0.4, 0.5]
    assert snap["body_center_dist"] == 0.3
    assert snap["filter_down"] == {"fd": 1}
    assert snap["intuition"] == {"in": 2}


# ---------------------------------------------------------------------------
# build_nervous_system_snapshot
# ---------------------------------------------------------------------------

def test_build_nervous_system_snapshot_empty_refs_returns_none():
    assert spirit_loop.build_nervous_system_snapshot({}) is None


def test_build_nervous_system_snapshot_uses_neural_when_present():
    refs = {
        "neural_nervous_system": types.SimpleNamespace(
            get_stats=lambda: {"source": "neural"}
        ),
    }
    assert spirit_loop.build_nervous_system_snapshot(refs) == {"source": "neural"}


def test_build_nervous_system_snapshot_falls_back_to_v4_vm():
    coord = types.SimpleNamespace(
        nervous_system=types.SimpleNamespace(programs={"a": 1, "b": 2}),
    )
    snap = spirit_loop.build_nervous_system_snapshot({"coordinator": coord})
    assert snap == {"version": "v4_vm", "programs": ["a", "b"]}


# ---------------------------------------------------------------------------
# start_snapshot_builder_threads
# ---------------------------------------------------------------------------

def test_builder_threads_populate_caches_quickly():
    """After start_snapshot_builder_threads, all 3 caches populate within 2s."""
    # Reset caches to simulate fresh boot.
    spirit_loop._COORD_SNAPSHOT_CACHE["data"] = None
    spirit_loop._COORD_SNAPSHOT_CACHE["ts"] = 0.0
    spirit_loop._TRINITY_SNAPSHOT_CACHE["data"] = None
    spirit_loop._TRINITY_SNAPSHOT_CACHE["ts"] = 0.0
    spirit_loop._NS_SNAPSHOT_CACHE["data"] = None
    spirit_loop._NS_SNAPSHOT_CACHE["ts"] = 0.0

    coord = types.SimpleNamespace(
        get_stats=lambda: {"commits": 7},
        _meta_engine=None, _meta_service=None,
        dreaming=None, inner=None,
        nervous_system=types.SimpleNamespace(programs={"p": 1}),
    )
    refs = {"coordinator": coord}
    spirit_loop.start_snapshot_builder_threads(refs, config={})

    deadline = time.time() + 2.0
    while time.time() < deadline:
        if (spirit_loop._COORD_SNAPSHOT_CACHE["data"] is not None
                and spirit_loop._TRINITY_SNAPSHOT_CACHE["data"] is not None
                and spirit_loop._NS_SNAPSHOT_CACHE["data"] is not None):
            break
        time.sleep(0.05)

    assert spirit_loop._COORD_SNAPSHOT_CACHE["data"] is not None, \
        "coord cache not populated within 2s"
    assert spirit_loop._COORD_SNAPSHOT_CACHE["data"]["commits"] == 7
    assert spirit_loop._TRINITY_SNAPSHOT_CACHE["data"] is not None, \
        "trinity cache not populated within 2s"
    assert spirit_loop._NS_SNAPSHOT_CACHE["data"] is not None, \
        "NS cache not populated within 2s"
    assert spirit_loop._NS_SNAPSHOT_CACHE["data"]["version"] == "v4_vm"


def test_builder_thread_survives_build_exceptions():
    """If build_fn raises, cache keeps serving last-good; thread stays alive."""
    # Prime cache with known-good data.
    spirit_loop._COORD_SNAPSHOT_CACHE["data"] = {"last_good": True}
    spirit_loop._COORD_SNAPSHOT_CACHE["ts"] = time.time()

    # Refs whose coordinator raises on get_stats.
    coord = types.SimpleNamespace(
        get_stats=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        _meta_engine=None, _meta_service=None,
        dreaming=None, inner=None, nervous_system=None,
    )
    refs = {"coordinator": coord}
    spirit_loop.start_snapshot_builder_threads(refs, config={})

    # Wait a couple of builder cycles and verify cache still has last-good data.
    time.sleep(0.5)
    assert spirit_loop._COORD_SNAPSHOT_CACHE["data"] == {"last_good": True}, \
        "builder exception should not clobber the last-good cache"
