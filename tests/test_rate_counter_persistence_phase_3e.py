"""
Tests for Phase 3.E wave 3b — atomic JSON snapshot persistence of in-memory
rate-counter deques (D-SPEC-87 follow-up).

agency._history (action ledger) + output_verifier._rejection_timestamps
both reset to empty on restart pre-fix, so actions_this_day / rejected_per_day
took 24h to warm up after every restart → outer_mind willing[10,12,13] frozen.

Wave 3b wraps each deque in a RollingStateStore (titan_hcl/core/
rolling_state_persistence.py) — atomic tempfile+rename writes, 24h
stale-discard on load, save_every_n batching to avoid I/O spam.

Per CLAUDE.md run with:
``python -m pytest tests/test_rate_counter_persistence_phase_3e.py -v -p no:anchorpy``
"""
from __future__ import annotations

import time

import pytest

from titan_hcl.core.rolling_state_persistence import RollingStateStore


@pytest.fixture()
def isolate_data_dir(tmp_path, monkeypatch):
    """Force RollingStateStore to write under tmp_path/data/dim_history."""
    # Patch the resolution helper so the store writes under tmp_path.
    import titan_hcl.core.rolling_state_persistence as rsp

    real = rsp._titan_data_dir

    def fake(titan_id=None):
        d = tmp_path / "data"
        d.mkdir(parents=True, exist_ok=True)
        return d

    monkeypatch.setattr(rsp, "_titan_data_dir", fake)
    yield tmp_path
    monkeypatch.setattr(rsp, "_titan_data_dir", real)


def test_rolling_store_round_trip_actions(isolate_data_dir):
    """Append 10 actions → close → re-open → reload preserves entries."""
    store = RollingStateStore(
        name="agency_action_history",
        max_entries=500,
        max_age_s=86400.0,
        save_every_n=1,
    )
    now = time.time()
    actions = [
        {"ts": now - i, "action_type": "post", "success": True}
        for i in range(10)
    ]
    for a in actions:
        store.append_and_save(a, actions)
    # Re-open and load
    store2 = RollingStateStore(
        name="agency_action_history",
        max_entries=500,
        max_age_s=86400.0,
    )
    restored = store2.load()
    assert len(restored) == 10
    assert restored[0]["action_type"] == "post"


def test_rolling_store_drops_stale_entries_on_load(isolate_data_dir):
    """Entries older than max_age_s should be discarded by load()."""
    store = RollingStateStore(
        name="stale_test",
        max_entries=500,
        max_age_s=10.0,  # 10s window
        save_every_n=1,
    )
    now = time.time()
    fresh = {"ts": now, "v": "fresh"}
    stale = {"ts": now - 30, "v": "stale"}
    # Save both
    store.save([fresh, stale])
    store2 = RollingStateStore(
        name="stale_test",
        max_entries=500,
        max_age_s=10.0,
    )
    restored = store2.load()
    assert len(restored) == 1
    assert restored[0]["v"] == "fresh"


def test_rolling_store_handles_corrupted_file(isolate_data_dir):
    """Bad JSON on disk → returns [] without crashing."""
    store = RollingStateStore(name="corrupt_test", max_entries=10)
    store.path.write_text("not valid json {{{")
    restored = store.load()
    assert restored == []


def test_rolling_store_atomic_write_no_partial_files(isolate_data_dir):
    """After save, no .tmp files left behind (atomic rename completes)."""
    store = RollingStateStore(name="atomic_test", max_entries=10)
    store.save([{"ts": time.time(), "v": 1}])
    # No stray .tmp files
    tmps = list(store.path.parent.glob(".atomic_test.*.tmp"))
    assert tmps == []
    # Final file exists
    assert store.path.exists()


def test_rejection_timestamps_restored_as_floats(isolate_data_dir):
    """The output_verifier integration packs {ts: float} entries — verify
    that shape survives round-trip and float type is preserved."""
    store = RollingStateStore(
        name="output_verifier_rejections",
        max_entries=1000,
        max_age_s=86400.0,
    )
    now = time.time()
    entries = [{"ts": now - i * 60} for i in range(5)]  # 5 rejections in last 5 min
    store.save(entries)

    store2 = RollingStateStore(
        name="output_verifier_rejections",
        max_entries=1000,
        max_age_s=86400.0,
    )
    restored = store2.load()
    assert len(restored) == 5
    for e in restored:
        assert isinstance(e["ts"], float)


def test_first_append_persists_immediately_for_cold_start_durability(isolate_data_dir):
    """First append should persist immediately (no batching) so cold-start
    state survives an immediate crash. Subsequent appends within save_every_s
    can batch."""
    store = RollingStateStore(
        name="cold_start_durability", max_entries=10, save_every_n=5,
        save_every_s=60.0)
    accumulated = []
    first = {"ts": time.time(), "v": "first"}
    accumulated.append(first)
    store.append_and_save(first, accumulated)
    # First save fires immediately because _last_save_ts=0 → time_since > 60s.
    store2 = RollingStateStore(name="cold_start_durability", max_entries=10)
    restored = store2.load()
    assert len(restored) == 1
    assert restored[0]["v"] == "first"
