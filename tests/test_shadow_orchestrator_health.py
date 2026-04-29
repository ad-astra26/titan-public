"""Tests for the multi-criterion shadow health gate (BUG-B1-WEAK-HEALTH-CHECK fix).

Pre-fix: shadow boot success = HTTP 200 on /health alone. Post-fix: also
verify module roster / critical workers / no crash-loops / fresh
heartbeats / dynamic-state advance. Each criterion has a dedicated test
that simulates it failing in isolation, plus a happy-path test where all
pass.

We don't spawn a real shadow kernel — that's an integration concern.
Instead we mock the HTTP layer to return synthetic /v4/state payloads
shaped like the real endpoint produces, then assert the gate makes the
right pass/fail decision.
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from titan_plugin.core import shadow_orchestrator as so


def _mk_module(state="running", restart_count=0, heartbeat_age=1.0,
               rss_mb=100.0, layer="L1"):
    """Build a guardian-module dict shaped like /v4/state returns."""
    return {
        "state": state, "pid": 12345, "rss_mb": rss_mb,
        "uptime": 100.0, "restart_count": restart_count,
        "restarts_in_window": 0, "last_heartbeat_age": heartbeat_age,
        "layer": layer,
    }


def _mk_state_response(modules: dict, spirit_uptime: float | None = None) -> dict:
    """Build a /v4/state payload with the given modules dict."""
    if spirit_uptime is not None and "spirit" in modules:
        modules["spirit"]["uptime"] = spirit_uptime
    return {
        "status": "ok",
        "data": {"v4": True, "guardian": modules},
    }


def _healthy_modules() -> dict:
    """Return a roster of 18 healthy modules covering all critical names."""
    names = ["spirit", "body", "mind", "timechain", "memory", "imw", "api",
             "warning_monitor", "observatory_writer", "rl", "llm", "media",
             "language", "meta_teacher", "cgn", "knowledge", "backup", "emot_cgn"]
    return {n: _mk_module() for n in names}


class TestMultiCriterionHealth(unittest.TestCase):
    def setUp(self):
        # Test criteria with smoke disabled by default (smoke is timing-sensitive
        # and tested separately). Layer 2 shadow_db_integrity_check also disabled
        # — these tests use mocked HTTP and don't create a real shadow data dir,
        # so the gate would fail with shadow_dir_missing. Layer 2 has its own
        # dedicated tests at tests/test_shadow_swap_layer2_db_integrity.py.
        self.criteria = so.HealthCriteria(
            smoke_test_enabled=False,
            shadow_db_integrity_check_enabled=False,
        )

    def _patch_http(self, health_status=200, state_payload=None):
        """Patch urllib so /health returns health_status and /v4/state returns state_payload."""
        def _urlopen(url, timeout=2):
            class Resp:
                def __init__(self, status, body):
                    self.status = status
                    self._body = body
                def read(self):
                    return self._body
                def __enter__(self):
                    return self
                def __exit__(self, *a):
                    pass
            if "/health" in str(url):
                return Resp(health_status, b"{}")
            if "/v4/state" in str(url):
                body = json.dumps(state_payload or {}).encode()
                return Resp(200 if state_payload else 500, body)
            return Resp(404, b"")
        return mock.patch("urllib.request.urlopen", side_effect=_urlopen)

    def test_happy_path_all_criteria_pass(self):
        with self._patch_http(state_payload=_mk_state_response(_healthy_modules())):
            passed, diag = so._check_multi_criterion_health(7779, self.criteria)
        self.assertTrue(passed, msg=f"diag: {diag}")
        self.assertTrue(diag["all_pre_smoke_passed"])
        self.assertTrue(diag["checks"]["health_endpoint"]["pass"])
        self.assertTrue(diag["checks"]["min_modules_running"]["pass"])
        self.assertTrue(diag["checks"]["critical_modules_running"]["pass"])
        self.assertTrue(diag["checks"]["no_crash_loops"]["pass"])
        self.assertTrue(diag["checks"]["heartbeats_fresh"]["pass"])

    def test_fails_when_health_endpoint_returns_500(self):
        with self._patch_http(health_status=500):
            passed, diag = so._check_multi_criterion_health(7779, self.criteria)
        self.assertFalse(passed)
        self.assertFalse(diag["checks"]["health_endpoint"]["pass"])

    def test_fails_when_critical_module_missing(self):
        # spirit is critical — drop it
        modules = _healthy_modules()
        modules["spirit"]["state"] = "stopped"
        with self._patch_http(state_payload=_mk_state_response(modules)):
            passed, diag = so._check_multi_criterion_health(7779, self.criteria)
        self.assertFalse(passed)
        crit = diag["checks"]["critical_modules_running"]
        self.assertFalse(crit["pass"])
        self.assertIn("spirit", crit["missing"])

    def test_fails_when_module_in_crash_loop(self):
        modules = _healthy_modules()
        # cgn has crashed twice
        modules["cgn"]["restart_count"] = 2
        with self._patch_http(state_payload=_mk_state_response(modules)):
            passed, diag = so._check_multi_criterion_health(7779, self.criteria)
        self.assertFalse(passed)
        crash_check = diag["checks"]["no_crash_loops"]
        self.assertFalse(crash_check["pass"])
        names = [m["name"] for m in crash_check["modules_above_threshold"]]
        self.assertIn("cgn", names)

    def test_fails_when_heartbeat_stale(self):
        modules = _healthy_modules()
        # body's heartbeat is 60s old — over the 30s threshold
        modules["body"]["last_heartbeat_age"] = 60.0
        with self._patch_http(state_payload=_mk_state_response(modules)):
            passed, diag = so._check_multi_criterion_health(7779, self.criteria)
        self.assertFalse(passed)
        hb = diag["checks"]["heartbeats_fresh"]
        self.assertFalse(hb["pass"])
        names = [m["name"] for m in hb["stale_modules"]]
        self.assertIn("body", names)

    def test_fails_when_too_few_modules_running(self):
        # Reduce healthy count below 14 by stopping 6 modules
        modules = _healthy_modules()
        for name in ["warning_monitor", "rl", "llm", "media", "meta_teacher", "knowledge"]:
            modules[name]["state"] = "stopped"
        with self._patch_http(state_payload=_mk_state_response(modules)):
            passed, diag = so._check_multi_criterion_health(7779, self.criteria)
        # 12 modules running < min_modules_running=14
        self.assertFalse(passed)
        mr = diag["checks"]["min_modules_running"]
        self.assertFalse(mr["pass"])
        self.assertEqual(mr["running_count"], 12)

    def test_diagnosis_is_serializable_json(self):
        """Diagnosis must be JSON-safe for shadow_swap_history.jsonl audit log."""
        modules = _healthy_modules()
        modules["spirit"]["state"] = "stopped"
        with self._patch_http(state_payload=_mk_state_response(modules)):
            _, diag = so._check_multi_criterion_health(7779, self.criteria)
        json.dumps(diag)  # must not raise


class TestSmokeAdvancing(unittest.TestCase):
    """Verify smoke step detects whether dynamic state is actually progressing."""

    def setUp(self):
        # Fast smoke for tests
        self.criteria = so.HealthCriteria(smoke_interval_s=0.05)

    def test_passes_when_uptime_advances(self):
        # First call: spirit.uptime=100; second call: 200
        states = [
            _mk_state_response(_healthy_modules(), spirit_uptime=100.0),
            _mk_state_response(_healthy_modules(), spirit_uptime=200.0),
        ]
        with mock.patch.object(so, "_fetch_state_json", side_effect=states):
            passed, diag = so._check_smoke_advancing(7779, self.criteria)
        self.assertTrue(passed, msg=f"diag: {diag}")
        self.assertGreater(diag["sample2"], diag["sample1"])

    def test_fails_when_uptime_stuck(self):
        states = [
            _mk_state_response(_healthy_modules(), spirit_uptime=100.0),
            _mk_state_response(_healthy_modules(), spirit_uptime=100.0),  # no advance
        ]
        with mock.patch.object(so, "_fetch_state_json", side_effect=states):
            passed, diag = so._check_smoke_advancing(7779, self.criteria)
        self.assertFalse(passed)
        self.assertEqual(diag["error"], "field_did_not_advance")

    def test_fails_when_field_missing(self):
        states = [
            _mk_state_response(_healthy_modules()),  # no uptime override → field missing
            _mk_state_response(_healthy_modules()),
        ]
        # Use a guaranteed-missing field for this test
        criteria = so.HealthCriteria(
            smoke_interval_s=0.05,
            smoke_field_path="data.guardian.spirit.does_not_exist",
        )
        with mock.patch.object(so, "_fetch_state_json", side_effect=states):
            passed, diag = so._check_smoke_advancing(7779, criteria)
        self.assertFalse(passed)
        self.assertEqual(diag["error"], "field_unreadable")


class TestWaitForHealth(unittest.TestCase):
    """End-to-end test of the polling _wait_for_health gate with multi-criterion."""

    def test_returns_pass_with_diag_on_success(self):
        """First poll passes → return (True, diag) immediately."""
        criteria = so.HealthCriteria(smoke_test_enabled=False)
        with mock.patch.object(so, "_check_multi_criterion_health",
                               return_value=(True, {"checks": {"all": {"pass": True}}})):
            passed, diag = so._wait_for_health(7779, timeout=5.0, criteria=criteria)
        self.assertTrue(passed)
        self.assertIn("checks", diag)

    def test_returns_fail_with_last_diag_on_timeout(self):
        """Never passes → return (False, last_diagnosis) after timeout."""
        criteria = so.HealthCriteria(smoke_test_enabled=False)
        fail_diag = {"checks": {"health_endpoint": {"pass": False, "error": "boom"}}}
        with mock.patch.object(so, "_check_multi_criterion_health",
                               return_value=(False, fail_diag)):
            passed, diag = so._wait_for_health(7779, timeout=0.5, criteria=criteria)
        self.assertFalse(passed)
        self.assertFalse(diag["checks"]["health_endpoint"]["pass"])

    def test_smoke_failure_keeps_polling(self):
        """If pre-smoke passes but smoke fails, keep polling (smoke needs warm-up)."""
        criteria = so.HealthCriteria(
            smoke_test_enabled=True, smoke_interval_s=0.01,
        )
        # Multi-criterion always passes, but smoke fails
        with mock.patch.object(so, "_check_multi_criterion_health",
                               return_value=(True, {"checks": {"all": {"pass": True}}})), \
             mock.patch.object(so, "_check_smoke_advancing",
                               return_value=(False, {"error": "field_did_not_advance"})):
            passed, _diag = so._wait_for_health(7779, timeout=0.3, criteria=criteria)
        # Smoke never passes within timeout → overall fail
        self.assertFalse(passed)


class TestHealthCriteriaDefaults(unittest.TestCase):
    def test_default_criteria_includes_critical_modules(self):
        c = so.HealthCriteria()
        for name in ("spirit", "body", "mind", "timechain", "memory", "imw", "api"):
            self.assertIn(name, c.critical_modules)

    def test_default_min_running_is_reasonable(self):
        # 18 total modules typical; require 14 = 78%
        c = so.HealthCriteria()
        self.assertGreaterEqual(c.min_modules_running, 10)
        self.assertLessEqual(c.min_modules_running, 18)


if __name__ == "__main__":
    unittest.main()
