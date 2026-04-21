"""Unit tests for titan_plugin.logic.knowledge_health (KP-4).

Covers: circuit breaker state transitions, budget counters + persistence
across "restart", decision-log write + rotation, near-duplicate detection.
"""

import json
import os
import tempfile
import time

import pytest

from titan_plugin.logic.knowledge_health import (
    CIRCUIT_CLOSED,
    CIRCUIT_HALF_OPEN,
    CIRCUIT_OPEN,
    HealthTracker,
)


@pytest.fixture
def tmpdirs():
    with tempfile.TemporaryDirectory() as tmp:
        yield {
            "health": os.path.join(tmp, "health.json"),
            "decisions": os.path.join(tmp, "logs", "decisions.jsonl"),
        }


# ── Circuit breaker ──────────────────────────────────────────────────

class TestCircuitBreaker:
    def test_closed_by_default(self, tmpdirs):
        h = HealthTracker(health_path=tmpdirs["health"],
                          decision_log_path=tmpdirs["decisions"])
        assert h.should_attempt("wiktionary") is True

    def test_opens_after_consecutive_errors(self, tmpdirs):
        h = HealthTracker(health_path=tmpdirs["health"],
                          decision_log_path=tmpdirs["decisions"],
                          cb_fail_threshold=3,
                          cb_cooldown_seconds=10)
        for _ in range(3):
            h.record_attempt("wikt", success=False, error_type="http_5xx")
        assert h.should_attempt("wikt") is False
        snap = h.snapshot()["backends"]["wikt"]
        assert snap["circuit_state"] == CIRCUIT_OPEN

    def test_success_resets_consecutive_counter(self, tmpdirs):
        h = HealthTracker(health_path=tmpdirs["health"],
                          decision_log_path=tmpdirs["decisions"],
                          cb_fail_threshold=3)
        h.record_attempt("wikt", success=False, error_type="timeout")
        h.record_attempt("wikt", success=False, error_type="timeout")
        h.record_attempt("wikt", success=True)
        # Counter reset — 2 more failures shouldn't open
        h.record_attempt("wikt", success=False, error_type="timeout")
        h.record_attempt("wikt", success=False, error_type="timeout")
        assert h.should_attempt("wikt") is True

    def test_half_open_after_cooldown(self, tmpdirs, monkeypatch):
        h = HealthTracker(health_path=tmpdirs["health"],
                          decision_log_path=tmpdirs["decisions"],
                          cb_fail_threshold=2,
                          cb_cooldown_seconds=5)
        h.record_attempt("wikt", success=False, error_type="http_5xx")
        h.record_attempt("wikt", success=False, error_type="http_5xx")
        assert h.should_attempt("wikt") is False

        # Force cooldown to elapse by tampering with circuit_opened_ts
        snap = h.snapshot()["backends"]["wikt"]
        assert snap["circuit_state"] == CIRCUIT_OPEN
        # Monkeypatch time
        h._backends["wikt"].circuit_opened_ts = time.time() - 10
        assert h.should_attempt("wikt") is True  # transitions to half_open
        snap = h.snapshot()["backends"]["wikt"]
        assert snap["circuit_state"] == CIRCUIT_HALF_OPEN

    def test_half_open_success_closes_circuit(self, tmpdirs):
        h = HealthTracker(health_path=tmpdirs["health"],
                          decision_log_path=tmpdirs["decisions"],
                          cb_fail_threshold=2)
        h.record_attempt("wikt", success=False, error_type="http_5xx")
        h.record_attempt("wikt", success=False, error_type="http_5xx")
        h._backends["wikt"].circuit_opened_ts = time.time() - 1000
        h.should_attempt("wikt")  # transitions to half_open
        h.record_attempt("wikt", success=True)
        assert h._backends["wikt"].circuit_state == CIRCUIT_CLOSED
        assert h._backends["wikt"].consecutive_errors == 0

    def test_half_open_failure_reopens(self, tmpdirs):
        h = HealthTracker(health_path=tmpdirs["health"],
                          decision_log_path=tmpdirs["decisions"],
                          cb_fail_threshold=2)
        h.record_attempt("wikt", success=False, error_type="http_5xx")
        h.record_attempt("wikt", success=False, error_type="http_5xx")
        h._backends["wikt"].circuit_opened_ts = time.time() - 1000
        h.should_attempt("wikt")  # half_open
        h.record_attempt("wikt", success=False, error_type="http_5xx")
        assert h._backends["wikt"].circuit_state == CIRCUIT_OPEN


# ── Budget counters + persistence ────────────────────────────────────

class TestBudget:
    def test_budget_unlimited_by_default(self, tmpdirs):
        h = HealthTracker(health_path=tmpdirs["health"],
                          decision_log_path=tmpdirs["decisions"])
        assert h.check_budget("wikt") is True

    def test_budget_enforced(self, tmpdirs):
        h = HealthTracker(health_path=tmpdirs["health"],
                          decision_log_path=tmpdirs["decisions"],
                          budgets={"wikt": 1000})  # 1000 bytes
        # Below budget
        h.record_attempt("wikt", success=True, bytes_consumed=500)
        assert h.check_budget("wikt") is True
        # At budget (equal is OK per < semantic)
        h.record_attempt("wikt", success=True, bytes_consumed=499)
        assert h.check_budget("wikt") is True
        # Exceed
        h.record_attempt("wikt", success=True, bytes_consumed=100)
        assert h.check_budget("wikt") is False

    def test_counters_survive_restart(self, tmpdirs):
        h1 = HealthTracker(health_path=tmpdirs["health"],
                           decision_log_path=tmpdirs["decisions"])
        h1.record_attempt("wikt", success=True, bytes_consumed=500)
        h1.record_attempt("wikt", success=False, error_type="timeout")

        # "Restart" — new instance reads from disk
        h2 = HealthTracker(health_path=tmpdirs["health"],
                           decision_log_path=tmpdirs["decisions"])
        snap = h2.snapshot()["backends"]["wikt"]
        assert snap["requests_today"] == 2
        assert snap["errors_today"] == 1
        assert snap["bytes_consumed_today"] == 500
        assert snap["total_requests_lifetime"] == 2

    def test_day_rollover_resets_counters(self, tmpdirs):
        h = HealthTracker(health_path=tmpdirs["health"],
                          decision_log_path=tmpdirs["decisions"])
        h.record_attempt("wikt", success=True, bytes_consumed=500)
        # Simulate day rollover
        h._backends["wikt"].counter_day_epoch = (
            h._current_day_epoch() - 1)
        # Next check / record sees new day
        assert h.check_budget("wikt") is True  # also resets
        snap = h.snapshot()["backends"]["wikt"]
        assert snap["requests_today"] == 0
        assert snap["bytes_consumed_today"] == 0
        # lifetime retained
        assert snap["total_requests_lifetime"] == 1

    def test_reset_budget_single_backend(self, tmpdirs):
        h = HealthTracker(health_path=tmpdirs["health"],
                          decision_log_path=tmpdirs["decisions"])
        h.record_attempt("wikt", success=True, bytes_consumed=500)
        h.record_attempt("wiki", success=True, bytes_consumed=300)
        h.reset_budget("wikt")
        assert h.snapshot()["backends"]["wikt"]["bytes_consumed_today"] == 0
        assert h.snapshot()["backends"]["wiki"]["bytes_consumed_today"] == 300

    def test_reset_budget_all(self, tmpdirs):
        h = HealthTracker(health_path=tmpdirs["health"],
                          decision_log_path=tmpdirs["decisions"])
        h.record_attempt("wikt", success=True, bytes_consumed=500)
        h.record_attempt("wiki", success=True, bytes_consumed=300)
        h.reset_budget()  # all
        snap = h.snapshot()["backends"]
        assert snap["wikt"]["bytes_consumed_today"] == 0
        assert snap["wiki"]["bytes_consumed_today"] == 0


# ── Decision log ─────────────────────────────────────────────────────

class TestDecisionLog:
    def test_append_writes_line(self, tmpdirs):
        h = HealthTracker(health_path=tmpdirs["health"],
                          decision_log_path=tmpdirs["decisions"])
        h.append_decision({"query": "test", "ts": time.time()})
        assert os.path.exists(tmpdirs["decisions"])
        with open(tmpdirs["decisions"]) as f:
            line = f.readline()
        entry = json.loads(line)
        assert entry["query"] == "test"

    def test_rotation_at_size_cap(self, tmpdirs):
        h = HealthTracker(health_path=tmpdirs["health"],
                          decision_log_path=tmpdirs["decisions"],
                          decision_log_max_bytes=200)  # tiny cap
        for i in range(50):
            h.append_decision({"query": f"q{i}", "extra": "x" * 50})
        # Primary file exists + rotation file should exist
        assert os.path.exists(tmpdirs["decisions"])
        assert os.path.exists(tmpdirs["decisions"] + ".1")


# ── Near-duplicate detection ─────────────────────────────────────────

class TestNearDup:
    def test_identical_not_flagged(self, tmpdirs):
        h = HealthTracker(health_path=tmpdirs["health"],
                          decision_log_path=tmpdirs["decisions"])
        h.note_query_for_near_dup("python async")
        assert h.note_query_for_near_dup("python async") is None

    def test_flags_near_duplicate(self, tmpdirs):
        h = HealthTracker(health_path=tmpdirs["health"],
                          decision_log_path=tmpdirs["decisions"],
                          near_dup_jaccard=0.5)
        h.note_query_for_near_dup("python async await")
        match = h.note_query_for_near_dup("python async")
        # 2 of 2 tokens overlap out of 3 total → Jaccard 2/3 = 0.67 > 0.5
        assert match == "python async await"

    def test_no_match_below_threshold(self, tmpdirs):
        h = HealthTracker(health_path=tmpdirs["health"],
                          decision_log_path=tmpdirs["decisions"],
                          near_dup_jaccard=0.8)
        h.note_query_for_near_dup("python async")
        # "docker kubernetes" — zero overlap
        assert h.note_query_for_near_dup("docker kubernetes") is None


# ── snapshot ─────────────────────────────────────────────────────────

class TestSnapshot:
    def test_snapshot_shape(self, tmpdirs):
        h = HealthTracker(health_path=tmpdirs["health"],
                          decision_log_path=tmpdirs["decisions"])
        h.record_attempt("wikt", success=True, bytes_consumed=100,
                         latency_ms=250)
        snap = h.snapshot()
        assert "ts" in snap
        assert "backends" in snap
        assert "wikt" in snap["backends"]
        assert snap["backends"]["wikt"]["avg_latency_ms"] == 250
        assert snap["backends"]["wikt"]["requests_today"] == 1
