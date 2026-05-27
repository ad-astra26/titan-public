"""Phase 7 — BufferCache unit tests (D-SPEC-PHASE7 / INV-Syn-16/17).

Covers:
- get returns None on unset; cached value otherwise
- set updates dict + emits bus payload (op=set)
- clear drops dict entry + emits bus payload (op=clear)
- bus emit failure soft-fails (no raise) per INV-Syn-17
- lazy hydrate from snapshot on first cold access
- stale snapshot (> warm_threshold_s) skips hydration
- missing snapshot is silent (no raise)
- corrupt snapshot is silent
- unknown buffer raises ValueError on set/clear (programmer error)
- writes_emitted / clears_emitted counters advance
"""
from __future__ import annotations

import json
import os

import pytest

from titan_hcl.synthesis.buffer_cache import (
    BufferCache, DEFAULT_HYDRATION_WARM_THRESHOLD_S,
)


@pytest.fixture()
def cache_state(tmp_path):
    emitted: list[dict] = []
    snapshot_path = str(tmp_path / "buffers_snapshot.json")
    clock = [1000.0]

    def _emit(payload):
        emitted.append(payload)

    def _clock():
        return clock[0]

    cache = BufferCache(
        bus_emit=_emit,
        snapshot_path=snapshot_path,
        hydration_warm_threshold_s=DEFAULT_HYDRATION_WARM_THRESHOLD_S,
        clock=_clock,
    )
    return cache, emitted, snapshot_path, clock


# ── get/set basic ─────────────────────────────────────────────────────


def test_get_returns_none_on_unset(cache_state):
    cache, _, _, _ = cache_state
    assert cache.get("alice:s1", "goal") is None


def test_set_updates_cache_and_emits(cache_state):
    cache, emitted, _, _ = cache_state
    cache.set("alice:s1", "goal", content="debug", concept_ids=["rust"])
    row = cache.get("alice:s1", "goal")
    assert row["content"] == "debug"
    assert row["concept_ids"] == ["rust"]
    assert row["ts"] == 1000.0
    assert len(emitted) == 1
    payload = emitted[0]
    assert payload["op"] == "set"
    assert payload["chat_id"] == "alice:s1"
    assert payload["buffer_name"] == "goal"
    assert payload["content"] == "debug"
    assert payload["concept_ids"] == ["rust"]
    assert payload["ts"] == 1000.0


def test_clear_drops_cache_entry_and_emits(cache_state):
    cache, emitted, _, _ = cache_state
    cache.set("alice:s1", "goal", content="x", concept_ids=[])
    emitted.clear()
    cache.clear("alice:s1", "goal")
    assert cache.get("alice:s1", "goal") is None
    assert len(emitted) == 1
    assert emitted[0]["op"] == "clear"
    assert emitted[0]["buffer_name"] == "goal"


def test_get_all_returns_all_buffers_for_chat(cache_state):
    cache, _, _, _ = cache_state
    cache.set("alice:s1", "goal", content="g", concept_ids=[])
    cache.set("alice:s1", "retrieval", content="r", concept_ids=[])
    all_bufs = cache.get_all("alice:s1")
    assert set(all_bufs.keys()) == {"goal", "retrieval"}


# ── INV-Syn-17: bus emit failure soft-fails ────────────────────────────


def test_set_bus_emit_failure_does_not_raise(tmp_path):
    def _emit(payload):
        raise RuntimeError("bus broken")

    cache = BufferCache(
        bus_emit=_emit,
        snapshot_path=str(tmp_path / "s.json"),
    )
    # Must NOT raise — local cache stays correct, persistence is lost but logged.
    cache.set("alice:s1", "goal", content="x", concept_ids=[])
    assert cache.get("alice:s1", "goal")["content"] == "x"


def test_clear_bus_emit_failure_does_not_raise(tmp_path):
    def _emit(payload):
        raise RuntimeError("bus broken")

    cache = BufferCache(
        bus_emit=_emit,
        snapshot_path=str(tmp_path / "s.json"),
    )
    cache.clear("alice:s1", "goal")  # absent + bus broken → no raise


# ── lazy hydration ────────────────────────────────────────────────────


def test_hydrate_from_snapshot_on_first_cold_get(cache_state):
    cache, _, snapshot_path, clock = cache_state
    # Write a snapshot directly (simulating synthesis_worker output).
    snap_payload = {
        "version": 1,
        "ts": 1000.0,
        "writes_seen": 1,
        "clears_seen": 0,
        "chat_count": 1,
        "chats": {
            "alice:s1": {
                "goal": {
                    "content": "from snapshot",
                    "concept_ids": ["snap_c"],
                    "embedding_hash": "abc",
                    "updated_at": 999.0,
                },
            },
        },
    }
    with open(snapshot_path, "w") as f:
        json.dump(snap_payload, f)
    # First read of this cold chat hydrates.
    row = cache.get("alice:s1", "goal")
    assert row["content"] == "from snapshot"
    assert row["concept_ids"] == ["snap_c"]


def test_hydrate_skips_stale_snapshot(cache_state):
    cache, _, snapshot_path, clock = cache_state
    snap_payload = {
        "version": 1, "ts": 0.0, "chats": {
            "alice:stale": {
                "goal": {
                    "content": "old",
                    "concept_ids": [],
                    "embedding_hash": "",
                    "updated_at": 1.0,   # epoch-ish; clock far ahead → stale
                },
            },
        },
    }
    with open(snapshot_path, "w") as f:
        json.dump(snap_payload, f)
    # Clock is far ahead → stale → hydration skips.
    clock[0] = 10_000_000.0
    assert cache.get("alice:stale", "goal") is None


def test_hydrate_missing_snapshot_silent(cache_state):
    cache, _, _, _ = cache_state
    # No snapshot file written → get is silently None (no raise).
    assert cache.get("alice:s99", "goal") is None


def test_hydrate_corrupt_snapshot_silent(cache_state):
    cache, _, snapshot_path, _ = cache_state
    with open(snapshot_path, "w") as f:
        f.write("not-json{{{")
    assert cache.get("alice:s99", "goal") is None


def test_hydrate_only_runs_once_per_chat(cache_state):
    cache, _, snapshot_path, clock = cache_state
    snap_payload = {
        "version": 1, "ts": 1000.0, "chats": {
            "alice:s1": {"goal": {
                "content": "v1", "concept_ids": [],
                "embedding_hash": "", "updated_at": 1000.0,
            }},
        },
    }
    with open(snapshot_path, "w") as f:
        json.dump(snap_payload, f)
    assert cache.get("alice:s1", "goal")["content"] == "v1"
    # Snapshot changes — but cache stays at v1 (already hydrated).
    snap_payload["chats"]["alice:s1"]["goal"]["content"] = "v2"
    with open(snapshot_path, "w") as f:
        json.dump(snap_payload, f)
    assert cache.get("alice:s1", "goal")["content"] == "v1"


# ── validation ────────────────────────────────────────────────────────


def test_set_unknown_buffer_raises(cache_state):
    cache, _, _, _ = cache_state
    with pytest.raises(ValueError):
        cache.set("x", "brain", content="x", concept_ids=[])


def test_set_empty_chat_raises(cache_state):
    cache, _, _, _ = cache_state
    with pytest.raises(ValueError):
        cache.set("", "goal", content="x", concept_ids=[])


def test_clear_unknown_buffer_raises(cache_state):
    cache, _, _, _ = cache_state
    with pytest.raises(ValueError):
        cache.clear("x", "brain")


def test_get_unknown_buffer_returns_none(cache_state):
    """get is permissive — used by tools that may pass through unknown
    names (caller-side validation lives in the tool layer)."""
    cache, _, _, _ = cache_state
    assert cache.get("x", "brain") is None


# ── counters ──────────────────────────────────────────────────────────


def test_counters_advance(cache_state):
    cache, _, _, _ = cache_state
    cache.set("a:b", "goal", content="x", concept_ids=[])
    cache.set("a:b", "retrieval", content="y", concept_ids=[])
    cache.clear("a:b", "imaginal")
    stats = cache.stats()
    assert stats["writes_emitted"] == 2
    assert stats["clears_emitted"] == 1


def test_chats_cached_count(cache_state):
    cache, _, _, _ = cache_state
    cache.set("alice:s1", "goal", content="x", concept_ids=[])
    cache.set("bob:s1", "goal", content="x", concept_ids=[])
    assert cache.stats()["chats_cached"] == 2
