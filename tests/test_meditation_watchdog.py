"""
tests/test_meditation_watchdog.py — unit tests for MeditationWatchdog.

Covers F1-F8 detection, I1 self-test, I3 safety floor, and diagnose-first
classification per rFP_self_healing_meditation_cadence.md.

Run:
    source test_env/bin/activate
    python -m pytest tests/test_meditation_watchdog.py -v -p no:anchorpy
"""
import os
import time

import pytest

from titan_plugin.logic.meditation_watchdog import (
    MeditationWatchdog, WatchdogAlert,
)


def make_tracker(last_ts=0.0, count=0, in_meditation=False, last_epoch=0):
    return {
        "last_ts": last_ts,
        "count": count,
        "count_since_nft": 0,
        "in_meditation": in_meditation,
        "last_epoch": last_epoch,
    }


# ── No-alert baseline ────────────────────────────────────────────────

def test_no_alerts_on_fresh_meditation():
    wd = MeditationWatchdog("T1")
    now = time.time()
    tracker = make_tracker(last_ts=now - 60, count=5)  # 1 min ago
    assert wd.check(tracker, now) == []


def test_no_alerts_when_never_meditated_pre_bootstrap():
    """First boot: count=0, last_ts=0. No alerts until bootstrap grace expires."""
    wd = MeditationWatchdog("T1")
    now = time.time()
    tracker = make_tracker(last_ts=0.0, count=0)
    assert wd.check(tracker, now) == []


# ── F1 + F2: overdue detection ───────────────────────────────────────

def test_f1_f2_overdue_fires():
    wd = MeditationWatchdog("T1", bootstrap_hours=6.0, min_alert_hours=3.0)
    now = time.time()
    # 13h since last (2*bootstrap=12h, floor=3h → threshold max(3,12)=12h)
    tracker = make_tracker(last_ts=now - 13 * 3600, count=5)
    alerts = wd.check(tracker, now)
    f12 = [a for a in alerts if a.failure_mode == "F1_F2_OVERDUE"]
    assert len(f12) == 1
    assert f12[0].severity == "HIGH"
    assert f12[0].diagnostic["time_since_hours"] > 12


def test_f1_f2_not_fired_below_threshold():
    wd = MeditationWatchdog("T1", bootstrap_hours=12.0, min_alert_hours=3.0)
    now = time.time()
    # 10h < 12h bootstrap threshold — no alert
    tracker = make_tracker(last_ts=now - 10 * 3600, count=5)
    assert [a for a in wd.check(tracker, now) if a.failure_mode == "F1_F2_OVERDUE"] == []


# ── I3: safety floor ─────────────────────────────────────────────────

def test_i3_safety_floor_blocks_premature_alert():
    """Hyperactive Titan: avg gap 30min → 2*avg=1h. Floor=3h should suppress alerts under 3h."""
    wd = MeditationWatchdog("T1", bootstrap_hours=12.0, min_alert_hours=3.0, gap_window=50)
    for _ in range(10):
        wd._gaps.append(1800.0)  # 30-min gaps
    # expected_interval() returns 2 * 1800 = 3600 = 1h
    assert wd.expected_interval() == 3600.0
    now = time.time()
    # 2h overdue — below 3h floor → NO alert despite 2*avg=1h
    tracker = make_tracker(last_ts=now - 2 * 3600, count=10)
    f12 = [a for a in wd.check(tracker, now) if a.failure_mode == "F1_F2_OVERDUE"]
    assert f12 == []


def test_i3_safety_floor_lets_alert_through_above_floor():
    wd = MeditationWatchdog("T1", bootstrap_hours=12.0, min_alert_hours=3.0)
    for _ in range(10):
        wd._gaps.append(1800.0)
    now = time.time()
    # 4h overdue — above 3h floor → alert fires
    tracker = make_tracker(last_ts=now - 4 * 3600, count=10)
    f12 = [a for a in wd.check(tracker, now) if a.failure_mode == "F1_F2_OVERDUE"]
    assert len(f12) == 1


# ── F3 + F6: stuck in_meditation ─────────────────────────────────────

def test_f3_f6_stuck_fires_after_threshold():
    wd = MeditationWatchdog("T1", stuck_threshold_seconds=600.0)
    now = time.time()
    tracker = make_tracker(last_ts=now - 60, count=5, in_meditation=True)
    # First check: just entered in_meditation, not stuck yet
    assert [a for a in wd.check(tracker, now) if a.failure_mode == "F3_F6_STUCK"] == []
    # 15 min later, still in_meditation → stuck
    alerts_later = wd.check(tracker, now + 15 * 60)
    f3 = [a for a in alerts_later if a.failure_mode == "F3_F6_STUCK"]
    assert len(f3) == 1
    assert f3[0].severity == "HIGH"


def test_f3_resets_after_meditation_completes():
    wd = MeditationWatchdog("T1", stuck_threshold_seconds=600.0)
    now = time.time()
    # Enter meditation
    wd.check(make_tracker(last_ts=now - 60, count=5, in_meditation=True), now)
    # Complete — in_meditation drops, count increments
    wd.check(make_tracker(last_ts=now + 15 * 60, count=6, in_meditation=False), now + 16 * 60)
    # New meditation starts fresh — should NOT be stuck immediately
    alerts = wd.check(
        make_tracker(last_ts=now + 15 * 60, count=6, in_meditation=True),
        now + 17 * 60,
    )
    assert [a for a in alerts if a.failure_mode == "F3_F6_STUCK"] == []


# ── F4: backup count lag ─────────────────────────────────────────────

def test_f4_backup_lag_fires():
    wd = MeditationWatchdog("T1")
    now = time.time()
    tracker = make_tracker(last_ts=now - 60, count=10)
    alerts = wd.check(tracker, now, backup_state_count=7)
    f4 = [a for a in alerts if a.failure_mode == "F4_BACKUP_LAG"]
    assert len(f4) == 1
    assert f4[0].severity == "MEDIUM"
    assert f4[0].diagnostic["lag"] == 3


def test_f4_no_fire_under_lag_threshold():
    wd = MeditationWatchdog("T1")
    now = time.time()
    tracker = make_tracker(last_ts=now - 60, count=10)
    # lag=1 < 2
    assert [a for a in wd.check(tracker, now, backup_state_count=9)
            if a.failure_mode == "F4_BACKUP_LAG"] == []


# ── F7: not distilling streak ────────────────────────────────────────

def test_f7_not_distilling_streak():
    wd = MeditationWatchdog("T1")
    for _ in range(3):
        wd.record_meditation(time.time(), promoted=0)
    tracker = make_tracker(last_ts=time.time() - 60, count=3)
    alerts = wd.check(tracker, time.time())
    f7 = [a for a in alerts if a.failure_mode == "F7_NOT_DISTILLING"]
    assert len(f7) == 1
    assert f7[0].diagnostic["zero_promoted_streak"] == 3


def test_f7_resets_on_any_promotion():
    wd = MeditationWatchdog("T1")
    for _ in range(3):
        wd.record_meditation(time.time(), promoted=0)
    wd.record_meditation(time.time(), promoted=2)  # success resets
    tracker = make_tracker(last_ts=time.time() - 60, count=4)
    alerts = wd.check(tracker, time.time())
    assert [a for a in alerts if a.failure_mode == "F7_NOT_DISTILLING"] == []


# ── expected_interval: bootstrap + emergent ──────────────────────────

def test_expected_interval_bootstrap_under_5_samples():
    wd = MeditationWatchdog("T1", bootstrap_hours=12.0)
    for _ in range(4):
        wd._gaps.append(3 * 3600)
    assert wd.expected_interval() == 12 * 3600


def test_expected_interval_emergent_from_gaps():
    wd = MeditationWatchdog("T1", bootstrap_hours=12.0)
    for _ in range(10):
        wd._gaps.append(4 * 3600)
    # 2 * 4h = 8h
    assert wd.expected_interval() == 8 * 3600


def test_expected_interval_wide_window_stability():
    """Wide window (50) should not be perturbed by a single short gap."""
    wd = MeditationWatchdog("T1", gap_window=50)
    for _ in range(49):
        wd._gaps.append(4 * 3600)
    wd._gaps.append(1 * 3600)  # one anomalous short gap
    avg_hours = sum(wd._gaps) / len(wd._gaps) / 3600
    expected_hours = wd.expected_interval() / 3600
    # Should be close to 2 * 3.94 = ~7.88h, not massively perturbed
    assert abs(expected_hours - 2 * avg_hours) < 0.01


# ── Diagnose-first classification ────────────────────────────────────

def test_classify_overdue_natural_calm():
    wd = MeditationWatchdog("T1")
    assert wd.classify_overdue({}, drain_flat_12h=False, gaba_flat_12h=False) == "natural_calm"


def test_classify_overdue_stuck():
    wd = MeditationWatchdog("T1")
    assert wd.classify_overdue({}, drain_flat_12h=True, gaba_flat_12h=True) == "stuck"


def test_classify_overdue_partial_still_natural():
    """Only one flat signal → conservative: still 'natural_calm' (avoids false force-trigger)."""
    wd = MeditationWatchdog("T1")
    assert wd.classify_overdue({}, drain_flat_12h=True, gaba_flat_12h=False) == "natural_calm"


# ── I1: self-test on boot ────────────────────────────────────────────

def test_i1_self_test_passes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    wd = MeditationWatchdog("T1", stuck_threshold_seconds=600.0)
    assert wd.self_test() is True
    assert wd._selftest_done is True
    assert wd._selftest_pass is True
    assert (tmp_path / "data" / "meditation_watchdog_selftest.log").exists()


def test_i1_self_test_does_not_leak_state(tmp_path, monkeypatch):
    """After self_test, real watchdog state must be clean (no fake stuck timer)."""
    monkeypatch.chdir(tmp_path)
    wd = MeditationWatchdog("T1", stuck_threshold_seconds=600.0)
    wd.self_test()
    # Now run a real check with in_meditation=False — should be empty
    now = time.time()
    alerts = wd.check(make_tracker(last_ts=now - 60, count=5, in_meditation=False), now)
    assert [a for a in alerts if a.failure_mode == "F3_F6_STUCK"] == []


# ── WatchdogAlert dataclass sanity ───────────────────────────────────

def test_watchdog_alert_serializable():
    alert = WatchdogAlert(
        severity="HIGH", failure_mode="F1_F2_OVERDUE",
        detail="test", diagnostic={"a": 1}, ts=1234.5,
    )
    import json
    serialized = json.dumps({
        "severity": alert.severity, "failure_mode": alert.failure_mode,
        "detail": alert.detail, "diagnostic": alert.diagnostic, "ts": alert.ts,
    })
    assert "F1_F2_OVERDUE" in serialized
