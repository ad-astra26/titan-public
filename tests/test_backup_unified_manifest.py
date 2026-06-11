"""Tests for SPEC §24.3 unified backup manifest (Arweave plane).

Per rFP_backup_diff_baseline_unified_v1 §5.2 test-coverage requirements:
  - schema round-trip
  - atomic-write per §11.H.2 (.bak + .bak.prev rotation)
  - walk-backward semantics
  - prev_event_id linkage / chain-break detection
  - baseline-trigger first-wins (month_boundary OR depth_cap)
"""

import json
import os
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from titan_hcl.logic.backup_unified_manifest import (
    UNIFIED_MANIFEST_SCHEMA_VERSION,
    UnifiedManifest,
    _atomic_write_json,
    make_event,
    new_event_id,
)


# ── helpers ───────────────────────────────────────────────────────────────


def _make_personality(tx="ar_tx_p", root="00" * 32, size=1024, mode="full",
                      diff_against=None, skipped=None, tier="none"):
    return {
        "tx_id": tx, "merkle_root": root, "size_bytes": size,
        "diff_mode": mode, "diff_against_event_id": diff_against,
        "skipped_files": skipped or [], "encryption_tier": tier,
    }


def _make_timechain(tx="ar_tx_t", root="11" * 32, size=2048, mode="full",
                    block_range=(0, 1000), prev_offset=0):
    return {
        "tx_id": tx, "merkle_root": root, "size_bytes": size,
        "diff_mode": mode, "block_range": list(block_range),
        "prev_offset_bytes": prev_offset,
    }


def _baseline_event(event_id=None, prev=None, trigger="month_boundary", ts=None):
    return make_event(
        event_id=event_id or new_event_id(),
        event_type="baseline",
        prev_event_id=prev,
        baseline_trigger=trigger,
        personality=_make_personality(),
        timechain=_make_timechain(),
        ts_unix=ts,
    )


def _incremental_event(prev_id, event_id=None, ts=None,
                       diff_against=None):
    return make_event(
        event_id=event_id or new_event_id(),
        event_type="incremental",
        prev_event_id=prev_id,
        baseline_trigger=None,
        personality=_make_personality(
            mode="incremental", diff_against=diff_against),
        timechain=_make_timechain(mode="tail"),
        ts_unix=ts,
    )


# ── schema construction + round-trip ─────────────────────────────────────


def test_fresh_manifest_has_empty_event_list(tmp_path):
    m = UnifiedManifest(titan_id="T1", base_dir=str(tmp_path))
    assert m.events == []
    assert m.current_baseline_event_id is None
    assert m.current_baseline_date is None


def test_save_and_reload_round_trip(tmp_path):
    m1 = UnifiedManifest(titan_id="T1", base_dir=str(tmp_path))
    ev = _baseline_event(trigger="first_event")
    m1.append_event(ev)
    m1.save()

    m2 = UnifiedManifest.load(titan_id="T1", base_dir=str(tmp_path))
    assert len(m2.events) == 1
    assert m2.events[0]["event_id"] == ev["event_id"]
    assert m2.current_baseline_event_id == ev["event_id"]


def test_schema_version_present_on_save(tmp_path):
    m = UnifiedManifest(titan_id="T1", base_dir=str(tmp_path))
    m.append_event(_baseline_event(trigger="first_event"))
    m.save()
    with open(m.path) as f:
        loaded = json.load(f)
    assert loaded["schema_version"] == UNIFIED_MANIFEST_SCHEMA_VERSION


def test_load_rejects_cross_titan_contamination(tmp_path):
    """SPEC §24.3 — titan_id mismatch is a load-bearing safety check."""
    m1 = UnifiedManifest(titan_id="T1", base_dir=str(tmp_path))
    m1.append_event(_baseline_event(trigger="first_event"))
    m1.save()

    # Try loading the T1 file as T2 — must reject
    t2_path = os.path.join(str(tmp_path), "backup_unified_manifest_T2.json")
    os.replace(m1.path, t2_path)
    with pytest.raises(ValueError, match="titan_id mismatch"):
        UnifiedManifest.load(titan_id="T2", base_dir=str(tmp_path))


# ── atomic-write per §11.H.2 ─────────────────────────────────────────────


def test_atomic_write_creates_bak_on_second_save(tmp_path):
    """SPEC §11.H.2 — 2-generation .bak retention."""
    m = UnifiedManifest(titan_id="T1", base_dir=str(tmp_path))
    m.append_event(_baseline_event(trigger="first_event"))
    m.save()
    assert os.path.exists(m.path)
    assert not os.path.exists(m.path + ".bak")  # first save, no prior

    # Second save — primary becomes .bak
    e2 = _incremental_event(prev_id=m.get_latest_event()["event_id"])
    m.append_event(e2)
    m.save()
    assert os.path.exists(m.path)
    assert os.path.exists(m.path + ".bak")


def test_atomic_write_rotates_bak_to_bak_prev(tmp_path):
    """SPEC §11.H.2 — 3rd write rotates .bak → .bak.prev."""
    m = UnifiedManifest(titan_id="T1", base_dir=str(tmp_path))
    m.append_event(_baseline_event(trigger="first_event"))
    m.save()
    e2 = _incremental_event(prev_id=m.get_latest_event()["event_id"])
    m.append_event(e2)
    m.save()
    e3 = _incremental_event(prev_id=m.get_latest_event()["event_id"])
    m.append_event(e3)
    m.save()
    assert os.path.exists(m.path)
    assert os.path.exists(m.path + ".bak")
    assert os.path.exists(m.path + ".bak.prev")


def test_load_falls_through_to_bak_on_primary_corruption(tmp_path):
    """SPEC §11.H.4 — boot integrity check falls back to .bak."""
    m = UnifiedManifest(titan_id="T1", base_dir=str(tmp_path))
    m.append_event(_baseline_event(trigger="first_event"))
    m.save()
    e2 = _incremental_event(prev_id=m.get_latest_event()["event_id"])
    m.append_event(e2)
    m.save()
    # Now corrupt the primary
    with open(m.path, "w") as f:
        f.write("{not json")
    m2 = UnifiedManifest.load(titan_id="T1", base_dir=str(tmp_path))
    # .bak has 1 event (post-first-save, pre-second-save state)
    assert len(m2.events) == 1


def test_load_raises_when_all_3_corrupt(tmp_path):
    """SPEC §11.H.4 — both backups also corrupted → escalate, do not silent-reset."""
    m = UnifiedManifest(titan_id="T1", base_dir=str(tmp_path))
    m.append_event(_baseline_event(trigger="first_event"))
    m.save()
    # Save twice more to create both .bak and .bak.prev
    e2 = _incremental_event(prev_id=m.get_latest_event()["event_id"])
    m.append_event(e2)
    m.save()
    e3 = _incremental_event(prev_id=m.get_latest_event()["event_id"])
    m.append_event(e3)
    m.save()
    # Corrupt all 3
    for p in (m.path, m.path + ".bak", m.path + ".bak.prev"):
        with open(p, "w") as f:
            f.write("{not json")
    with pytest.raises(ValueError, match="corrupted on all 3 generations"):
        UnifiedManifest.load(titan_id="T1", base_dir=str(tmp_path))


# ── append_event chain enforcement ──────────────────────────────────────


def test_append_rejects_invalid_prev_event_id(tmp_path):
    m = UnifiedManifest(titan_id="T1", base_dir=str(tmp_path))
    m.append_event(_baseline_event(trigger="first_event"))
    # Try appending an incremental with WRONG prev_event_id
    bad = _incremental_event(prev_id="nonexistent_uuid")
    with pytest.raises(ValueError, match="prev_event_id chain break"):
        m.append_event(bad)


def test_first_event_requires_prev_none(tmp_path):
    m = UnifiedManifest(titan_id="T1", base_dir=str(tmp_path))
    # First-ever append with non-None prev → chain break
    bad = _baseline_event(prev="some_uuid", trigger="first_event")
    with pytest.raises(ValueError, match="prev_event_id chain break"):
        m.append_event(bad)


def test_baseline_updates_current_baseline_pointers(tmp_path):
    m = UnifiedManifest(titan_id="T1", base_dir=str(tmp_path))
    e1 = _baseline_event(trigger="first_event", ts=1700000000.0)
    m.append_event(e1)
    assert m.current_baseline_event_id == e1["event_id"]
    assert m.current_baseline_date == "2023-11-14"  # UTC date of ts=1700000000


def test_incremental_does_not_change_baseline_pointer(tmp_path):
    m = UnifiedManifest(titan_id="T1", base_dir=str(tmp_path))
    e1 = _baseline_event(trigger="first_event")
    m.append_event(e1)
    e2 = _incremental_event(prev_id=e1["event_id"])
    m.append_event(e2)
    assert m.current_baseline_event_id == e1["event_id"]


# ── walk_chain semantics ────────────────────────────────────────────────


def test_walk_chain_yields_newest_first(tmp_path):
    m = UnifiedManifest(titan_id="T1", base_dir=str(tmp_path))
    e1 = _baseline_event(trigger="first_event")
    m.append_event(e1)
    e2 = _incremental_event(prev_id=e1["event_id"])
    m.append_event(e2)
    e3 = _incremental_event(prev_id=e2["event_id"])
    m.append_event(e3)
    walked = list(m.walk_chain())
    assert [w["event_id"] for w in walked] == [
        e3["event_id"], e2["event_id"], e1["event_id"]]


def test_walk_chain_from_specific_event(tmp_path):
    m = UnifiedManifest(titan_id="T1", base_dir=str(tmp_path))
    e1 = _baseline_event(trigger="first_event")
    m.append_event(e1)
    e2 = _incremental_event(prev_id=e1["event_id"])
    m.append_event(e2)
    e3 = _incremental_event(prev_id=e2["event_id"])
    m.append_event(e3)
    walked = list(m.walk_chain(from_event_id=e2["event_id"]))
    assert [w["event_id"] for w in walked] == [e2["event_id"], e1["event_id"]]


def test_get_baseline_for_event_finds_predecessor_baseline(tmp_path):
    m = UnifiedManifest(titan_id="T1", base_dir=str(tmp_path))
    b1 = _baseline_event(trigger="first_event")
    m.append_event(b1)
    i1 = _incremental_event(prev_id=b1["event_id"])
    m.append_event(i1)
    i2 = _incremental_event(prev_id=i1["event_id"])
    m.append_event(i2)
    result = m.get_baseline_for_event(i2["event_id"])
    assert result["event_id"] == b1["event_id"]


def test_incrementals_since_baseline_chronological_order(tmp_path):
    m = UnifiedManifest(titan_id="T1", base_dir=str(tmp_path))
    b1 = _baseline_event(trigger="first_event")
    m.append_event(b1)
    i1 = _incremental_event(prev_id=b1["event_id"])
    m.append_event(i1)
    i2 = _incremental_event(prev_id=i1["event_id"])
    m.append_event(i2)
    incs = m.incrementals_since_baseline()
    assert [i["event_id"] for i in incs] == [i1["event_id"], i2["event_id"]]


# ── should_rebase trigger (first-wins per Maker Q3) ─────────────────────


def test_should_rebase_true_on_first_event(tmp_path):
    m = UnifiedManifest(titan_id="T1", base_dir=str(tmp_path))
    do, reason = m.should_rebase()
    assert do is True
    assert reason == "first_event"


def test_should_rebase_false_mid_month_under_depth_cap(tmp_path):
    m = UnifiedManifest(titan_id="T1", base_dir=str(tmp_path))
    b1 = _baseline_event(trigger="first_event",
                          ts=datetime(2026, 6, 1, 12, tzinfo=timezone.utc).timestamp())
    m.append_event(b1)
    for _ in range(5):
        m.append_event(_incremental_event(prev_id=m.get_latest_event()["event_id"]))
    # Now check on 2026-06-15 (mid-month, well under depth cap of 30)
    do, reason = m.should_rebase(now=datetime(2026, 6, 15, 12, tzinfo=timezone.utc))
    assert do is False
    assert reason is None


def test_should_rebase_true_on_month_boundary(tmp_path):
    m = UnifiedManifest(titan_id="T1", base_dir=str(tmp_path))
    b1 = _baseline_event(trigger="first_event",
                          ts=datetime(2026, 6, 1, 12, tzinfo=timezone.utc).timestamp())
    m.append_event(b1)
    # 1st of next month — should rebase
    do, reason = m.should_rebase(now=datetime(2026, 7, 1, 12, tzinfo=timezone.utc))
    assert do is True
    assert reason == "month_boundary"


def test_should_rebase_suppressed_when_baseline_too_fresh(tmp_path):
    """§24.2 fresh-baseline guard (2026-06-01): a first_event baseline that
    lands days before the 1st must NOT immediately re-baseline on the 1st —
    that's a full re-ship of barely-changed tiers (the T1 05-29→06-01 case).
    The next month boundary (baseline aged past the grace) still rebases."""
    m = UnifiedManifest(titan_id="T1", base_dir=str(tmp_path))
    # baseline 2026-05-29 (3 days before the June boundary)
    b1 = _baseline_event(trigger="first_event",
                         ts=datetime(2026, 5, 29, 12, tzinfo=timezone.utc).timestamp())
    m.append_event(b1)
    do, reason = m.should_rebase(now=datetime(2026, 6, 1, 9, tzinfo=timezone.utc))
    assert do is False, "fresh (3d) baseline must NOT re-baseline on month boundary"
    assert reason is None
    # …but a baseline aged past the grace DOES rebase on the next boundary.
    m2 = UnifiedManifest(titan_id="T1", base_dir=str(tmp_path / "aged"))
    b2 = _baseline_event(trigger="first_event",
                         ts=datetime(2026, 4, 22, 12, tzinfo=timezone.utc).timestamp())
    m2.append_event(b2)
    do2, reason2 = m2.should_rebase(now=datetime(2026, 6, 1, 9, tzinfo=timezone.utc))
    assert do2 is True and reason2 == "month_boundary"


def test_should_rebase_false_on_same_day_as_baseline(tmp_path):
    """Don't double-rebase if a baseline already landed today."""
    m = UnifiedManifest(titan_id="T1", base_dir=str(tmp_path))
    b1 = _baseline_event(trigger="month_boundary",
                          ts=datetime(2026, 7, 1, 0, tzinfo=timezone.utc).timestamp())
    m.append_event(b1)
    # Later on the SAME 1st — already rebased today, no double-rebase
    do, reason = m.should_rebase(now=datetime(2026, 7, 1, 23, tzinfo=timezone.utc))
    assert do is False


def test_should_rebase_true_on_depth_cap_first_wins(tmp_path):
    """Per SPEC §24.2 Maker Q3 — first-wins of (month) OR (30 incrementals)."""
    m = UnifiedManifest(titan_id="T1", base_dir=str(tmp_path))
    b1 = _baseline_event(trigger="first_event",
                          ts=datetime(2026, 6, 5, 12, tzinfo=timezone.utc).timestamp())
    m.append_event(b1)
    # 30 incrementals — hit the depth cap mid-month
    for _ in range(30):
        m.append_event(_incremental_event(prev_id=m.get_latest_event()["event_id"]))
    do, reason = m.should_rebase(now=datetime(2026, 6, 20, 12, tzinfo=timezone.utc))
    assert do is True
    assert reason == "depth_cap"


def test_should_rebase_month_boundary_wins_over_depth(tmp_path):
    """If both conditions fire, month_boundary is reported first (first-wins)."""
    m = UnifiedManifest(titan_id="T1", base_dir=str(tmp_path))
    b1 = _baseline_event(trigger="first_event",
                          ts=datetime(2026, 6, 5, 12, tzinfo=timezone.utc).timestamp())
    m.append_event(b1)
    for _ in range(30):
        m.append_event(_incremental_event(prev_id=m.get_latest_event()["event_id"]))
    # 1st of July — both trigger; month boundary wins per code order
    do, reason = m.should_rebase(now=datetime(2026, 7, 1, 12, tzinfo=timezone.utc))
    assert do is True
    assert reason == "month_boundary"


# ── Mode-aware cadence (RFP_backup_redesign_spine, 2026-06-11) ────────────


def test_should_rebase_mainnet_none_no_calendar_baseline(tmp_path):
    """mainnet (cadence='none'): NO calendar trigger — even an old baseline on the
    1st of the month must NOT rebase. Incrementals run as long as possible; a full
    Arweave baseline fires only on depth_cap/self_heal (don't waste SOL)."""
    m = UnifiedManifest(titan_id="T1", base_dir=str(tmp_path))
    b1 = _baseline_event(trigger="first_event",
                         ts=datetime(2026, 4, 1, 12, tzinfo=timezone.utc).timestamp())
    m.append_event(b1)
    for _ in range(10):
        m.append_event(_incremental_event(prev_id=m.get_latest_event()["event_id"]))
    # 1st of a later month + a 2-month-old baseline → 'monthly' WOULD rebase here;
    # 'none' must NOT.
    do, reason = m.should_rebase(
        now=datetime(2026, 6, 1, 12, tzinfo=timezone.utc),
        cadence="none", depth_cap=90)
    assert do is False and reason is None


def test_should_rebase_mainnet_depth_cap_90(tmp_path):
    """mainnet depth_cap=90 — 89 incrementals stay incremental; the 90th rebases
    (the restore-chain-length safety, not a calendar churn)."""
    m = UnifiedManifest(titan_id="T1", base_dir=str(tmp_path))
    b1 = _baseline_event(trigger="first_event",
                         ts=datetime(2026, 6, 1, 12, tzinfo=timezone.utc).timestamp())
    m.append_event(b1)
    for _ in range(89):
        m.append_event(_incremental_event(prev_id=m.get_latest_event()["event_id"]))
    do, _ = m.should_rebase(now=datetime(2026, 6, 30, 12, tzinfo=timezone.utc),
                            cadence="none", depth_cap=90)
    assert do is False, "89 < 90 depth cap → still incremental"
    m.append_event(_incremental_event(prev_id=m.get_latest_event()["event_id"]))  # 90th
    do, reason = m.should_rebase(now=datetime(2026, 6, 30, 12, tzinfo=timezone.utc),
                                 cadence="none", depth_cap=90)
    assert do is True and reason == "depth_cap"


def test_should_rebase_weekly_cadence(tmp_path):
    """devnet/local (cadence='weekly'): re-baseline once the baseline is ≥7d old;
    a fresh (<7d) baseline stays incremental."""
    m = UnifiedManifest(titan_id="T1", base_dir=str(tmp_path))
    b1 = _baseline_event(trigger="first_event",
                         ts=datetime(2026, 6, 1, 12, tzinfo=timezone.utc).timestamp())
    m.append_event(b1)
    # 3 days old → no weekly rebase
    do, _ = m.should_rebase(now=datetime(2026, 6, 4, 12, tzinfo=timezone.utc),
                            cadence="weekly", depth_cap=30)
    assert do is False, "3d-old baseline (<7d) → no weekly rebase"
    # 7 days old → week_boundary
    do, reason = m.should_rebase(now=datetime(2026, 6, 8, 12, tzinfo=timezone.utc),
                                 cadence="weekly", depth_cap=30)
    assert do is True and reason == "week_boundary"


# ── devnet/local retention: prune_before_current_baseline (2026-06-11) ────


def test_prune_before_current_baseline_keeps_baseline_and_after(tmp_path):
    """Drops events PRECEDING the current baseline; keeps the baseline + its
    incrementals + the latest. Returns the pruned events for tarball cleanup."""
    m = UnifiedManifest(titan_id="T1", base_dir=str(tmp_path))
    b1 = _baseline_event(trigger="first_event")
    m.append_event(b1)
    i1 = _incremental_event(prev_id=b1["event_id"]); m.append_event(i1)
    b2 = _baseline_event(prev=i1["event_id"], trigger="week_boundary"); m.append_event(b2)
    i2 = _incremental_event(prev_id=b2["event_id"]); m.append_event(i2)
    assert m.current_baseline_event_id == b2["event_id"]

    pruned = m.prune_before_current_baseline()
    pruned_ids = [e["event_id"] for e in pruned]
    assert pruned_ids == [b1["event_id"], i1["event_id"]]          # everything before b2
    kept_ids = [e["event_id"] for e in m.events]
    assert kept_ids == [b2["event_id"], i2["event_id"]]            # baseline + after kept
    assert m.current_baseline_event_id == b2["event_id"]           # baseline untouched
    # persisted
    m2 = UnifiedManifest.load(titan_id="T1", base_dir=str(tmp_path))
    assert [e["event_id"] for e in m2.events] == kept_ids


def test_prune_before_current_baseline_noop_when_baseline_is_oldest(tmp_path):
    """No-op (returns []) when the current baseline is already the first event —
    can never remove the baseline or leave zero backups."""
    m = UnifiedManifest(titan_id="T1", base_dir=str(tmp_path))
    b1 = _baseline_event(trigger="first_event"); m.append_event(b1)
    m.append_event(_incremental_event(prev_id=b1["event_id"]))
    assert m.prune_before_current_baseline() == []
    assert len(m.events) == 2
    # fresh manifest (no baseline) → also a no-op
    empty = UnifiedManifest(titan_id="T1", base_dir=str(tmp_path / "e"))
    assert empty.prune_before_current_baseline() == []


# ── make_event validation ────────────────────────────────────────────────


def test_make_event_rejects_invalid_event_type():
    with pytest.raises(ValueError, match="event_type must be"):
        make_event(event_id="x", event_type="garbage", prev_event_id=None,
                   baseline_trigger=None, personality=_make_personality(),
                   timechain=_make_timechain())


def test_make_event_rejects_baseline_without_trigger():
    with pytest.raises(ValueError, match="baseline_trigger"):
        make_event(event_id="x", event_type="baseline", prev_event_id=None,
                   baseline_trigger=None, personality=_make_personality(),
                   timechain=_make_timechain())


def test_make_event_rejects_incremental_with_trigger():
    with pytest.raises(ValueError, match="incremental events must have baseline_trigger=None"):
        make_event(event_id="x", event_type="incremental", prev_event_id="prev",
                   baseline_trigger="month_boundary",
                   personality=_make_personality(),
                   timechain=_make_timechain())


def test_make_event_requires_personality_subfields():
    with pytest.raises(ValueError, match="event.personality missing required field"):
        make_event(event_id="x", event_type="incremental", prev_event_id="prev",
                   baseline_trigger=None,
                   personality={"tx_id": "ar"},  # missing merkle_root etc.
                   timechain=_make_timechain())
