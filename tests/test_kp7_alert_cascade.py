"""
KP-7 alert cascade tests — budget + circuit-open alerts out of HealthTracker.

Covers the callback path added in KP-7:
  * budget_warning fires once when daily consumption crosses 80%
  * budget_exceeded fires once when consumption crosses 100%
  * circuit_open fires once when the breaker transitions CLOSED → OPEN
  * alerted_today set dedups same-kind alerts within a UTC day
  * day rollover resets the dedup set so alerts fire fresh the next day
  * alerts fire OUTSIDE the tracker lock so callbacks may take their own locks
  * a raising callback does NOT break record_attempt
  * budget_daily_bytes == 0 means unlimited → no alerts fire
  * on_alert is None → code path is a no-op (no crash)
"""

from __future__ import annotations

import os
import tempfile
import threading
import time

import pytest

from titan_plugin.logic.knowledge_health import (
    ALERT_KIND_BUDGET_EXCEEDED,
    ALERT_KIND_BUDGET_WARNING,
    ALERT_KIND_CIRCUIT_OPEN,
    HealthTracker,
)


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def tmp_paths():
    d = tempfile.mkdtemp(prefix="kp7_")
    yield {
        "health": os.path.join(d, "h.json"),
        "decisions": os.path.join(d, "d.jsonl"),
    }


def _make_tracker(tmp_paths, *, budgets=None, on_alert=None, cb_fail_threshold=3):
    return HealthTracker(
        health_path=tmp_paths["health"],
        decision_log_path=tmp_paths["decisions"],
        budgets=budgets or {},
        cb_fail_threshold=cb_fail_threshold,
        cb_cooldown_seconds=30,
        on_alert=on_alert,
    )


# ── Budget warning (80%) ──────────────────────────────────────────────

def test_budget_warning_fires_once_on_threshold_crossing(tmp_paths):
    events = []
    tracker = _make_tracker(tmp_paths, budgets={"wiktionary": 1000},
                             on_alert=lambda k, b, c: events.append((k, b, c)))
    # Well below 80% → no alert
    tracker.record_attempt("wiktionary", success=True, bytes_consumed=500)
    assert events == []
    # Cross 80% (500 → 850, pct 0.85 ≥ 0.80)
    tracker.record_attempt("wiktionary", success=True, bytes_consumed=350)
    assert len(events) == 1
    kind, backend, ctx = events[0]
    assert kind == ALERT_KIND_BUDGET_WARNING
    assert backend == "wiktionary"
    assert ctx["bytes_consumed"] == 850
    assert ctx["budget_bytes"] == 1000
    assert 80.0 <= ctx["pct"] <= 90.0


def test_budget_warning_deduped_within_day(tmp_paths):
    events = []
    tracker = _make_tracker(tmp_paths, budgets={"wiktionary": 1000},
                             on_alert=lambda k, b, c: events.append((k, b, c)))
    # Cross 80%, then continue consuming below 100% → only ONE warning
    tracker.record_attempt("wiktionary", success=True, bytes_consumed=850)
    tracker.record_attempt("wiktionary", success=True, bytes_consumed=50)
    tracker.record_attempt("wiktionary", success=True, bytes_consumed=50)
    warnings = [e for e in events if e[0] == ALERT_KIND_BUDGET_WARNING]
    assert len(warnings) == 1


# ── Budget exceeded (100%) ────────────────────────────────────────────

def test_budget_exceeded_fires_on_crossing(tmp_paths):
    events = []
    tracker = _make_tracker(tmp_paths, budgets={"wikipedia_direct": 1000},
                             on_alert=lambda k, b, c: events.append((k, b, c)))
    # Cross 100% in one shot (0 → 1100)
    tracker.record_attempt("wikipedia_direct", success=True, bytes_consumed=1100)
    kinds = [e[0] for e in events]
    # We cross 80% AND 100% in one step; both alerts fire for observability
    assert ALERT_KIND_BUDGET_EXCEEDED in kinds
    exceeded = next(e for e in events if e[0] == ALERT_KIND_BUDGET_EXCEEDED)
    assert exceeded[2]["bytes_consumed"] == 1100
    assert exceeded[2]["pct"] >= 100.0


def test_budget_exceeded_deduped_within_day(tmp_paths):
    events = []
    tracker = _make_tracker(tmp_paths, budgets={"searxng_raw": 500},
                             on_alert=lambda k, b, c: events.append((k, b, c)))
    tracker.record_attempt("searxng_raw", success=True, bytes_consumed=600)
    tracker.record_attempt("searxng_raw", success=True, bytes_consumed=100)
    tracker.record_attempt("searxng_raw", success=True, bytes_consumed=100)
    exceeded = [e for e in events if e[0] == ALERT_KIND_BUDGET_EXCEEDED]
    assert len(exceeded) == 1


# ── Circuit breaker open transition ───────────────────────────────────

def test_circuit_open_alert_fires_on_threshold(tmp_paths):
    events = []
    tracker = _make_tracker(
        tmp_paths, budgets={}, cb_fail_threshold=3,
        on_alert=lambda k, b, c: events.append((k, b, c)))
    # 3 consecutive errors → circuit opens
    for _ in range(3):
        tracker.record_attempt("free_dictionary", success=False,
                                error_type="http_5xx")
    circuit_alerts = [e for e in events if e[0] == ALERT_KIND_CIRCUIT_OPEN]
    assert len(circuit_alerts) == 1
    _, backend, ctx = circuit_alerts[0]
    assert backend == "free_dictionary"
    assert ctx["consecutive_errors"] >= 3
    assert ctx["last_error_type"] == "http_5xx"
    assert ctx["previous_state"] == "closed"


def test_circuit_open_alert_deduped_within_day(tmp_paths):
    events = []
    tracker = _make_tracker(
        tmp_paths, budgets={}, cb_fail_threshold=2,
        on_alert=lambda k, b, c: events.append((k, b, c)))
    # First breach: closed → open (fires)
    tracker.record_attempt("wiktionary", success=False, error_type="timeout")
    tracker.record_attempt("wiktionary", success=False, error_type="timeout")
    # Now circuit is open + cooldown period; another failure stays open
    # without firing a second alert for the same day
    tracker.record_attempt("wiktionary", success=False, error_type="timeout")
    circuit_alerts = [e for e in events if e[0] == ALERT_KIND_CIRCUIT_OPEN]
    assert len(circuit_alerts) == 1


# ── Day rollover ──────────────────────────────────────────────────────

def test_alerted_today_resets_on_day_rollover(tmp_paths):
    events = []
    tracker = _make_tracker(tmp_paths, budgets={"wiktionary": 1000},
                             on_alert=lambda k, b, c: events.append((k, b, c)))
    tracker.record_attempt("wiktionary", success=True, bytes_consumed=850)
    assert len([e for e in events if e[0] == ALERT_KIND_BUDGET_WARNING]) == 1

    # Force day rollover by rewinding the backend's day epoch
    with tracker._lock:
        h = tracker._backends["wiktionary"]
        h.counter_day_epoch -= 1  # yesterday
        h.alerted_today.add("stale")  # sentinel to prove we clear it

    # Next record resets the day + clears alerted_today, so the same 80%
    # threshold fires again on a fresh day
    tracker.record_attempt("wiktionary", success=True, bytes_consumed=850)
    assert len([e for e in events if e[0] == ALERT_KIND_BUDGET_WARNING]) == 2


# ── Callback safety ───────────────────────────────────────────────────

def test_callback_fires_outside_lock(tmp_paths):
    """Callback must run after the tracker lock is released so it can
    safely call back into the tracker without deadlocking."""
    observed_lock = []

    def cb(_kind, _backend, _ctx):
        # Try to acquire the lock — non-blocking; succeed iff we're outside it
        acquired = tracker._lock.acquire(blocking=False)
        observed_lock.append(acquired)
        if acquired:
            tracker._lock.release()

    tracker = _make_tracker(tmp_paths, budgets={"wiktionary": 1000},
                             on_alert=cb)
    tracker.record_attempt("wiktionary", success=True, bytes_consumed=850)
    assert observed_lock == [True]  # callback was NOT holding the tracker lock


def test_callback_exception_does_not_break_record_attempt(tmp_paths):
    def exploding_cb(_kind, _backend, _ctx):
        raise RuntimeError("boom")

    tracker = _make_tracker(tmp_paths, budgets={"wiktionary": 500},
                             on_alert=exploding_cb)
    # Should NOT raise despite the callback exploding
    tracker.record_attempt("wiktionary", success=True, bytes_consumed=500)
    # Counters still advanced
    with tracker._lock:
        assert tracker._backends["wiktionary"].requests_today == 1
        assert tracker._backends["wiktionary"].bytes_consumed_today == 500


# ── Edge cases ────────────────────────────────────────────────────────

def test_no_alert_when_budget_zero_unlimited(tmp_paths):
    events = []
    tracker = _make_tracker(tmp_paths, budgets={"wiktionary": 0},
                             on_alert=lambda k, b, c: events.append((k, b, c)))
    # Even a huge consumption never triggers a budget alert
    tracker.record_attempt("wiktionary", success=True,
                            bytes_consumed=10_000_000_000)
    assert [e for e in events if e[0].startswith("budget_")] == []


def test_no_callback_wired_is_safe(tmp_paths):
    tracker = _make_tracker(tmp_paths, budgets={"wiktionary": 100},
                             on_alert=None)
    tracker.record_attempt("wiktionary", success=True, bytes_consumed=200)
    # No crash; counters still advanced
    with tracker._lock:
        assert tracker._backends["wiktionary"].bytes_consumed_today == 200


def test_alerted_today_persists_through_save_load(tmp_paths):
    """alerted_today must survive health.json round-trip so a worker restart
    within the same UTC day doesn't re-fire alerts already sent."""
    events = []
    tracker = _make_tracker(tmp_paths, budgets={"wiktionary": 1000},
                             on_alert=lambda k, b, c: events.append((k, b, c)))
    tracker.record_attempt("wiktionary", success=True, bytes_consumed=850)
    assert len(events) == 1
    # Simulate worker restart — new tracker loads state from health.json
    events2 = []
    tracker2 = _make_tracker(tmp_paths, budgets={"wiktionary": 1000},
                              on_alert=lambda k, b, c: events2.append((k, b, c)))
    # Crossing 80% again shouldn't re-fire because alerted_today was persisted
    tracker2.record_attempt("wiktionary", success=True, bytes_consumed=10)
    warnings = [e for e in events2 if e[0] == ALERT_KIND_BUDGET_WARNING]
    assert warnings == []
