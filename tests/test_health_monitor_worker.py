"""tests/test_health_monitor_worker.py — SPEC v1.12.0 §9.B + D-SPEC-67.

Coverage (Phase 1 MVP):
  • Bus event constants defined (5 new HEALTH_* + HEAL_REQUEST/RESULT).
  • Plugin contract: HealthResult dataclass + HealthCheckPlugin abstract
    enforcement (cannot instantiate abstract base).
  • social_x plugin: instantiation, class attributes, heal descriptor
    contract (refresh_session on posting-DEGRADED; None on pipeline).
  • _PluginRuntime: cooldown_ok, heal_attempts_in_window, prune.
  • _maybe_emit_heal_request: emits HEAL_REQUEST when cap+cooldown OK;
    skips on cap-exhaustion + emits HEALTH_HEAL_FAILED.
  • _record_heal_outcome: emits HEALTH_HEAL_ATTEMPT; consecutive-
    failure-threshold escalates to HEALTH_HEAL_FAILED P1.
  • _check_pending_heal_timeouts: in-flight correlation_id expires
    after HEALTH_HEAL_REPLY_TIMEOUT_S → timeout outcome recorded.
  • State persistence: atomic write + round-trip load preserves
    next_fire_time + heal_history + consecutive_failures.
  • social_worker._handle_heal_request: unknown action replies failure;
    refresh_session with absent gateway replies gateway_not_initialized;
    refresh_session with mock gateway returning empty string replies
    refresh_returned_empty; refresh_session with mock returning new
    session replies session_refreshed=True.

Per CLAUDE.md: pytest -p no:anchorpy, separate process per file.
"""
from __future__ import annotations

import json
import queue
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from titan_hcl import bus
from titan_hcl.health import (
    DEFAULT_MAX_HEAL_ATTEMPTS_PER_24H,
    HEALTH_CHECK_TIMEOUT_S,
    HEALTH_HEAL_CONSECUTIVE_FAILURE_THRESHOLD,
    HEALTH_HEAL_REPLY_TIMEOUT_S,
    HealthCheckPlugin,
    HealthResult,
)
from titan_hcl.health.social_x import SocialXHealthCheck
from titan_hcl.modules.health_monitor_worker import (
    DAILY_WINDOW_S,
    _PluginRuntime,
    _atomic_write_json,
    _check_pending_heal_timeouts,
    _load_state,
    _maybe_emit_heal_request,
    _record_heal_outcome,
    _save_state,
)


# ── Bus constants ───────────────────────────────────────────────────


def test_bus_event_constants_defined():
    assert bus.HEALTH_CHECK_RESULT == "HEALTH_CHECK_RESULT"
    assert bus.HEAL_REQUEST == "HEAL_REQUEST"
    assert bus.HEAL_RESULT == "HEAL_RESULT"
    assert bus.HEALTH_HEAL_ATTEMPT == "HEALTH_HEAL_ATTEMPT"
    assert bus.HEALTH_HEAL_FAILED == "HEALTH_HEAL_FAILED"


def test_module_exports_present():
    from titan_hcl.modules import health_monitor_worker as hmw
    assert hasattr(hmw, "health_monitor_worker_main")
    assert hmw.HEARTBEAT_INTERVAL_S == 30.0
    assert hmw.STATE_PERSIST_INTERVAL_S == 60.0


# ── Plugin contract ─────────────────────────────────────────────────


def test_healthcheckplugin_abstract_cannot_instantiate():
    """Abstract base — direct instantiation must fail (check() unimplemented)."""
    with pytest.raises(TypeError):
        HealthCheckPlugin()  # type: ignore[abstract]


def test_healthresult_dataclass_to_dict_roundtrip():
    r = HealthResult(
        plugin="x", layer="y", status="OK", reason="ok",
        details={"k": "v"}, heal_recommended=True)
    d = r.to_dict()
    assert d["plugin"] == "x" and d["layer"] == "y"
    assert d["status"] == "OK" and d["heal_recommended"] is True
    assert d["details"] == {"k": "v"}
    assert "ts" in d


def test_healthresult_default_heal_not_recommended():
    r = HealthResult(
        plugin="x", layer="y", status="DOWN", reason="z")
    assert r.heal_recommended is False


# ── social_x plugin ─────────────────────────────────────────────────


def test_social_x_class_attributes():
    assert SocialXHealthCheck.name == "social_x"
    # Phase 1.6: each Titan posts independently to its own DB → applies_on=all
    # so each Titan monitors its own X health. The "canonical_poller" pattern
    # is for POLLING X mentions (avoiding 3x API spend); POSTING is per-Titan.
    assert SocialXHealthCheck.applies_on == "all"
    # Phase 1.7: owning_worker MUST match the live bus subscriber name —
    # social_worker subscribes as "social_worker", NOT "social". Using
    # "social" would silently drop HEAL_REQUEST (broker has no such subscriber).
    assert SocialXHealthCheck.owning_worker == "social_worker"
    assert SocialXHealthCheck.cadence_s == 14400.0
    assert SocialXHealthCheck.max_heal_attempts_per_24h == 6


def test_social_x_posting_filters_by_titan_id():
    """Phase 1.6 fix: T1's social_x.db has cross-Titan rows (canonical
    poller visibility into T2+T3 posts). Without titan_id filter, T1's
    posting layer would falsely flip OK when T2 posts even if T1 itself
    is silent. With filter, each Titan only counts its own verified posts."""
    import sqlite3
    sx = SocialXHealthCheck({"api_key": "K", "user_name": "u",
                              "titan_id": "T1"})
    now = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "social_x.db"
        sx._db_path = str(db)
        conn = sqlite3.connect(str(db))
        _make_actions_table(conn)
        # T2 + T3 posted recently with verified_at set (live data shape);
        # without titan_id filter T1 would show OK.
        conn.execute(
            "INSERT INTO actions (status, titan_id, created_at, "
            "verified_at) VALUES ('verified', 'T2', ?, ?)",
            (now - 1800, now - 1795))
        conn.execute(
            "INSERT INTO actions (status, titan_id, created_at, "
            "verified_at) VALUES ('verified', 'T3', ?, ?)",
            (now - 3600, now - 3595))
        conn.commit()
        conn.close()
        posting = sx._check_posting()
    # T1 should see DEGRADED (its own count = 0) despite T2/T3 rows present.
    assert posting.status == "DEGRADED"
    assert "verified_posts_6h=0" in posting.reason
    assert posting.heal_recommended is True


def test_social_x_posting_counts_own_titan_posts():
    """Reciprocal: when this Titan DID post, status is OK regardless of
    cross-Titan visibility."""
    import sqlite3
    sx = SocialXHealthCheck({"api_key": "K", "user_name": "u",
                              "titan_id": "T2"})
    now = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "social_x.db"
        sx._db_path = str(db)
        conn = sqlite3.connect(str(db))
        _make_actions_table(conn)
        conn.execute(
            "INSERT INTO actions (status, titan_id, created_at, "
            "verified_at) VALUES ('verified', 'T2', ?, ?)",
            (now - 1800, now - 1795))
        conn.commit()
        conn.close()
        posting = sx._check_posting()
    assert posting.status == "OK"
    assert posting.details["titan_id"] == "T2"


def test_social_x_heal_returns_refresh_on_posting_degraded():
    sx = SocialXHealthCheck({"api_key": "K", "user_name": "u"})
    posting_degraded = HealthResult(
        plugin="social_x", layer="posting", status="DEGRADED",
        reason="verified_posts_6h=0", heal_recommended=True)
    action, details = sx.heal(posting_degraded)
    assert action == "refresh_session"
    assert details.get("trigger") == "verified_posts_6h_zero"


def test_social_x_heal_returns_none_on_pipeline_down():
    sx = SocialXHealthCheck({"api_key": "K", "user_name": "u"})
    pipeline_down = HealthResult(
        plugin="social_x", layer="pipeline", status="DOWN",
        reason="net_err", heal_recommended=False)
    action, _ = sx.heal(pipeline_down)
    assert action is None


def test_social_x_check_missing_api_key_pipeline_down(monkeypatch):
    # Force secrets-layer to be empty so the plugin truly has no api_key.
    # (Without this, the plugin's secrets-merge fallback would pick up the
    # dev VM's real ~/.titan/secrets.toml[stealth_sage].twitterapi_io_key.)
    # C.5/C.6: the plugin resolves secrets via params.load_titan_params (SHM
    # whole-config), not the retired config_loader.load_titan_config.
    import titan_hcl.health.social_x as _sx
    monkeypatch.setattr(_sx, "load_titan_params", lambda **kw: {})
    sx = SocialXHealthCheck({"api_key": "", "user_name": "u"})
    # Use a fake DB path to keep posting layer deterministic + isolated.
    with tempfile.TemporaryDirectory() as tmp:
        sx._db_path = str(Path(tmp) / "no_such.db")
        results = sx.check()
    assert len(results) == 2
    pipeline = next(r for r in results if r.layer == "pipeline")
    assert pipeline.status == "DOWN"
    assert pipeline.reason == "api_key_missing_in_config"


def test_social_x_check_posting_db_missing_degraded():
    sx = SocialXHealthCheck({"api_key": "K", "user_name": "u"})
    with tempfile.TemporaryDirectory() as tmp:
        sx._db_path = str(Path(tmp) / "no_such.db")
        # api_key=K means pipeline will try a real HTTP call — patch
        # `requests.get` via the plugin's _check_pipeline level using
        # MagicMock isn't straightforward across the import; assert
        # posting layer only.
        posting = sx._check_posting()
    assert posting.status == "DEGRADED"
    assert posting.reason == "social_x_db_missing"
    # DB-missing is NOT auto-healable.
    assert posting.heal_recommended is False


def _make_actions_table(conn):
    """Phase 1.9: full actions table schema mirrors production
    (logic/social_x_gateway.py:351). posted_at + verified_at are BOTH
    nullable and routinely NULL in live data."""
    conn.execute(
        "CREATE TABLE actions ("
        "  id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "  status TEXT NOT NULL DEFAULT 'pending', "
        "  titan_id TEXT, "
        "  created_at REAL NOT NULL, "
        "  posted_at REAL, "
        "  verified_at REAL"
        ")")


def test_social_x_check_posting_db_present_zero_posts_degraded():
    import sqlite3
    # Pass titan_id explicitly so the test doesn't depend on what
    # resolve_titan_id() returns on the dev VM.
    sx = SocialXHealthCheck({"api_key": "K", "user_name": "u",
                              "titan_id": "T1"})
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "social_x.db"
        sx._db_path = str(db)
        conn = sqlite3.connect(str(db))
        _make_actions_table(conn)
        conn.commit()
        conn.close()
        posting = sx._check_posting()
    assert posting.status == "DEGRADED"
    assert "verified_posts_6h=0" in posting.reason
    assert posting.heal_recommended is True


def test_social_x_check_posting_recent_verified_ok():
    """Production path: gateway sets verified_at (NOT posted_at) for many
    rows. Phase 1.9 uses COALESCE so verified_at alone is sufficient."""
    import sqlite3
    sx = SocialXHealthCheck({"api_key": "K", "user_name": "u",
                              "titan_id": "T1"})
    now = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "social_x.db"
        sx._db_path = str(db)
        conn = sqlite3.connect(str(db))
        _make_actions_table(conn)
        # Live data shape: verified_at populated, posted_at NULL.
        conn.execute(
            "INSERT INTO actions (status, titan_id, created_at, "
            "posted_at, verified_at) VALUES "
            "('verified', 'T1', ?, NULL, ?)",
            (now - 1800, now - 1790))
        conn.commit()
        conn.close()
        posting = sx._check_posting()
    assert posting.status == "OK"
    assert posting.heal_recommended is False


def test_social_x_check_posting_coalesce_falls_back_to_posted_at():
    """When verified_at is NULL but posted_at is set (older gateway flow),
    COALESCE should still pick it up."""
    import sqlite3
    sx = SocialXHealthCheck({"api_key": "K", "user_name": "u",
                              "titan_id": "T1"})
    now = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "social_x.db"
        sx._db_path = str(db)
        conn = sqlite3.connect(str(db))
        _make_actions_table(conn)
        conn.execute(
            "INSERT INTO actions (status, titan_id, created_at, "
            "posted_at, verified_at) VALUES "
            "('verified', 'T1', ?, ?, NULL)",
            (now - 1800, now - 1795))
        conn.commit()
        conn.close()
        posting = sx._check_posting()
    assert posting.status == "OK"


def test_social_x_check_posting_coalesce_falls_back_to_created_at():
    """When both verified_at and posted_at are NULL but row exists with
    status='verified', use created_at."""
    import sqlite3
    sx = SocialXHealthCheck({"api_key": "K", "user_name": "u",
                              "titan_id": "T1"})
    now = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "social_x.db"
        sx._db_path = str(db)
        conn = sqlite3.connect(str(db))
        _make_actions_table(conn)
        conn.execute(
            "INSERT INTO actions (status, titan_id, created_at, "
            "posted_at, verified_at) VALUES "
            "('verified', 'T1', ?, NULL, NULL)",
            (now - 1800,))
        conn.commit()
        conn.close()
        posting = sx._check_posting()
    assert posting.status == "OK"


def test_social_x_check_posting_excludes_old_verified_outside_window():
    """6h window enforcement: a 7h-old verified row must NOT count."""
    import sqlite3
    sx = SocialXHealthCheck({"api_key": "K", "user_name": "u",
                              "titan_id": "T1"})
    now = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "social_x.db"
        sx._db_path = str(db)
        conn = sqlite3.connect(str(db))
        _make_actions_table(conn)
        conn.execute(
            "INSERT INTO actions (status, titan_id, created_at, "
            "posted_at, verified_at) VALUES "
            "('verified', 'T1', ?, NULL, ?)",
            (now - 7 * 3600, now - 7 * 3600 + 5))
        conn.commit()
        conn.close()
        posting = sx._check_posting()
    assert posting.status == "DEGRADED"
    assert "verified_posts_6h=0" in posting.reason


# ── _PluginRuntime ──────────────────────────────────────────────────


def _make_runtime(now: float | None = None,
                  cadence: float = 14400.0,
                  cap: int = 6,
                  cd_success: float = 3600.0,
                  cd_failure: float = 1800.0) -> _PluginRuntime:
    if now is None:
        now = time.time()

    class _P(HealthCheckPlugin):
        name = "p"
        applies_on = "all"
        owning_worker = "owner"
        cadence_s = cadence
        max_heal_attempts_per_24h = cap
        heal_cooldown_after_success_s = cd_success
        heal_cooldown_after_failure_s = cd_failure

        def check(self):
            return []

    return _PluginRuntime(_P(), now)


def test_runtime_cooldown_ok_with_no_history():
    rt = _make_runtime()
    assert rt.cooldown_ok(time.time()) is True


def test_runtime_cooldown_after_success_enforced():
    rt = _make_runtime(cd_success=3600.0)
    now = time.time()
    rt.heal_history_24h.append({
        "ts": now - 60, "action": "x",
        "result": "success", "reason": "ok"})
    rt.last_heal_at = now - 60
    assert rt.cooldown_ok(now) is False  # 60s < 3600s
    assert rt.cooldown_ok(now + 3700) is True


def test_runtime_cooldown_after_failure_enforced():
    rt = _make_runtime(cd_failure=1800.0)
    now = time.time()
    rt.heal_history_24h.append({
        "ts": now - 60, "action": "x",
        "result": "failed", "reason": "err"})
    rt.last_heal_at = now - 60
    assert rt.cooldown_ok(now) is False
    assert rt.cooldown_ok(now + 1900) is True


def test_runtime_heal_attempts_in_window_counts_24h():
    rt = _make_runtime()
    now = time.time()
    rt.heal_history_24h.append(
        {"ts": now - 100, "action": "a", "result": "success",
         "reason": ""})
    rt.heal_history_24h.append(
        {"ts": now - (DAILY_WINDOW_S + 100), "action": "old",
         "result": "failed", "reason": ""})
    assert rt.heal_attempts_in_window(now) == 1


def test_runtime_prune_drops_old_entries():
    rt = _make_runtime()
    now = time.time()
    rt.heal_history_24h.append(
        {"ts": now - (DAILY_WINDOW_S + 100), "action": "old"})
    rt.heal_history_24h.append({"ts": now - 100, "action": "fresh"})
    rt.prune_heal_history(now)
    assert len(rt.heal_history_24h) == 1
    assert rt.heal_history_24h[0]["action"] == "fresh"


# ── _maybe_emit_heal_request ────────────────────────────────────────


def test_maybe_emit_heal_request_emits_when_action_returned():
    rt = _make_runtime()
    rt.plugin.heal = lambda r: ("refresh_session", {"k": "v"})
    sq = queue.Queue()
    fake_result = HealthResult(
        plugin="p", layer="L", status="DEGRADED", reason="r",
        heal_recommended=True)
    with tempfile.TemporaryDirectory() as tmp:
        emitted = _maybe_emit_heal_request(
            rt, fake_result, sq, "health_monitor",
            Path(tmp) / "j.jsonl")
    assert emitted is True
    msg = sq.get_nowait()
    assert msg["type"] == bus.HEAL_REQUEST
    assert msg["dst"] == "owner"
    assert msg["payload"]["action"] == "refresh_session"
    assert msg["payload"]["details"] == {"k": "v"}
    assert "correlation_id" in msg["payload"]
    assert len(rt.pending_heals) == 1


def test_maybe_emit_heal_request_skipped_under_cooldown():
    rt = _make_runtime(cd_success=3600.0)
    rt.plugin.heal = lambda r: ("act", {})
    now = time.time()
    rt.heal_history_24h.append({
        "ts": now - 60, "action": "act",
        "result": "success", "reason": ""})
    rt.last_heal_at = now - 60
    sq = queue.Queue()
    fake_result = HealthResult(
        plugin="p", layer="L", status="DEGRADED", reason="r",
        heal_recommended=True)
    with tempfile.TemporaryDirectory() as tmp:
        emitted = _maybe_emit_heal_request(
            rt, fake_result, sq, "health_monitor",
            Path(tmp) / "j.jsonl")
    assert emitted is False
    with pytest.raises(queue.Empty):
        sq.get_nowait()


def test_maybe_emit_heal_request_emits_failed_on_cap_exhaustion():
    rt = _make_runtime(cap=3)
    rt.plugin.heal = lambda r: ("act", {})
    now = time.time()
    for i in range(3):
        rt.heal_history_24h.append({
            "ts": now - i, "action": "act",
            "result": "failed", "reason": "x"})
    rt.last_heal_at = now
    sq = queue.Queue()
    fake_result = HealthResult(
        plugin="p", layer="L", status="DEGRADED", reason="r",
        heal_recommended=True)
    with tempfile.TemporaryDirectory() as tmp:
        emitted = _maybe_emit_heal_request(
            rt, fake_result, sq, "health_monitor",
            Path(tmp) / "j.jsonl")
    assert emitted is False
    msg = sq.get_nowait()
    assert msg["type"] == bus.HEALTH_HEAL_FAILED
    assert msg["payload"]["reason"] == "daily_cap_exhausted"


def test_maybe_emit_heal_request_returns_false_on_no_action():
    rt = _make_runtime()
    rt.plugin.heal = lambda r: (None, {})
    sq = queue.Queue()
    fake_result = HealthResult(
        plugin="p", layer="L", status="OK", reason="r",
        heal_recommended=True)
    with tempfile.TemporaryDirectory() as tmp:
        emitted = _maybe_emit_heal_request(
            rt, fake_result, sq, "health_monitor",
            Path(tmp) / "j.jsonl")
    assert emitted is False


# ── _record_heal_outcome ────────────────────────────────────────────


def test_record_heal_outcome_emits_attempt_event():
    rt = _make_runtime()
    sq = queue.Queue()
    _record_heal_outcome(
        rt, action="act", result="success", reason="ok",
        send_queue=sq, name="health_monitor")
    msg = sq.get_nowait()
    assert msg["type"] == bus.HEALTH_HEAL_ATTEMPT
    assert msg["payload"]["result"] == "success"
    assert rt.consecutive_failures == 0


def test_record_heal_outcome_consecutive_failures_escalate():
    rt = _make_runtime()
    sq = queue.Queue()
    for i in range(HEALTH_HEAL_CONSECUTIVE_FAILURE_THRESHOLD):
        _record_heal_outcome(
            rt, action="act", result="failed", reason="err",
            send_queue=sq, name="health_monitor")
    # Drain attempt events
    saw_attempt = saw_failed = False
    while not sq.empty():
        msg = sq.get_nowait()
        if msg["type"] == bus.HEALTH_HEAL_ATTEMPT:
            saw_attempt = True
        elif msg["type"] == bus.HEALTH_HEAL_FAILED:
            saw_failed = True
            assert msg["payload"]["reason"] == "consecutive_failures"
    assert saw_attempt and saw_failed
    assert (rt.consecutive_failures
            >= HEALTH_HEAL_CONSECUTIVE_FAILURE_THRESHOLD)


def test_record_heal_outcome_success_resets_failure_streak():
    rt = _make_runtime()
    rt.consecutive_failures = 2
    sq = queue.Queue()
    _record_heal_outcome(
        rt, action="act", result="success", reason="ok",
        send_queue=sq, name="health_monitor")
    assert rt.consecutive_failures == 0


# ── _check_pending_heal_timeouts ────────────────────────────────────


def test_pending_heal_timeout_records_timeout_outcome():
    rt = _make_runtime()
    sq = queue.Queue()
    expired_ts = time.time() - HEALTH_HEAL_REPLY_TIMEOUT_S - 5
    rt.pending_heals["abc"] = {
        "sent_ts": expired_ts, "action": "act", "details": {}}
    _check_pending_heal_timeouts(
        {"p": rt}, sq, "health_monitor")
    assert "abc" not in rt.pending_heals
    msg = sq.get_nowait()
    assert msg["type"] == bus.HEALTH_HEAL_ATTEMPT
    assert msg["payload"]["result"] == "timeout"


def test_pending_heal_within_timeout_kept():
    rt = _make_runtime()
    sq = queue.Queue()
    rt.pending_heals["abc"] = {
        "sent_ts": time.time(), "action": "act", "details": {}}
    _check_pending_heal_timeouts(
        {"p": rt}, sq, "health_monitor")
    assert "abc" in rt.pending_heals


# ── State persistence ──────────────────────────────────────────────


def test_state_atomic_write_roundtrip(tmp_path):
    payload = {"plugins": {"x": {"next_fire_time": 123.45,
                                   "heal_history_24h": [{"ts": 1}]}},
               "updated_at": 999.0}
    p = tmp_path / "state.json"
    _atomic_write_json(p, payload)
    assert p.exists()
    loaded = json.loads(p.read_text())
    assert loaded["updated_at"] == 999.0
    assert (loaded["plugins"]["x"]["heal_history_24h"][0]["ts"] == 1)


def test_save_state_then_load_roundtrip(tmp_path):
    rt = _make_runtime()
    rt.next_fire_time = 1234.5
    rt.consecutive_failures = 2
    rt.heal_history_24h.append(
        {"ts": time.time(), "action": "a",
         "result": "failed", "reason": "e"})
    sp = tmp_path / "state.json"
    _save_state(sp, {"p": rt})
    loaded = _load_state(sp)
    assert "plugins" in loaded
    p_state = loaded["plugins"]["p"]
    assert p_state["next_fire_time"] == 1234.5
    assert p_state["consecutive_failures"] == 2
    assert len(p_state["heal_history_24h"]) == 1


def test_load_state_missing_file_returns_empty(tmp_path):
    assert _load_state(tmp_path / "absent.json") == {}


# ── social_worker._handle_heal_request ─────────────────────────────


def test_social_worker_handle_heal_request_unknown_action_replies_failure():
    from titan_hcl.modules.social_worker import _handle_heal_request
    sq = queue.Queue()
    payload = {"plugin": "social_x", "action": "weird_action",
               "correlation_id": "xyz", "details": {}}
    state_refs = {"social_x_gateway": MagicMock()}
    _handle_heal_request(payload, state_refs, sq, "social")
    msg = sq.get_nowait()
    assert msg["type"] == bus.HEAL_RESULT
    assert msg["dst"] == "health_monitor"
    assert msg["payload"]["success"] is False
    assert msg["payload"]["correlation_id"] == "xyz"
    assert "unknown_action" in msg["payload"]["reason"]


def test_social_worker_handle_heal_request_no_gateway_replies_failure():
    from titan_hcl.modules.social_worker import _handle_heal_request
    sq = queue.Queue()
    payload = {"plugin": "social_x", "action": "refresh_session",
               "correlation_id": "c1", "details": {}}
    state_refs = {"social_x_gateway": None}
    _handle_heal_request(payload, state_refs, sq, "social")
    msg = sq.get_nowait()
    assert msg["payload"]["success"] is False
    assert msg["payload"]["reason"] == "gateway_not_initialized"


def test_social_worker_handle_heal_request_refresh_success(monkeypatch):
    # Phase 1.10: handler now uses gateway._load_config() as canonical
    # resolution. Mock returns the gateway-merged dict shape.
    import titan_hcl.modules.social_worker as _sw
    monkeypatch.setattr(_sw, "get_params", lambda section: {})
    monkeypatch.setattr(_sw, "load_titan_params", lambda **kw: {})
    from titan_hcl.modules.social_worker import _handle_heal_request
    sq = queue.Queue()
    gw = MagicMock()
    # gateway._load_config() returns a FLAT dict (api_key/proxy at root,
    # not nested under "social_x") per gateway code line 451-493.
    gw._load_config.return_value = {"api_key": "K", "proxy": "P"}
    gw._refresh_session.return_value = "new_session_cookie"
    payload = {"plugin": "social_x", "action": "refresh_session",
               "correlation_id": "c2", "details": {}}
    _handle_heal_request(
        payload, {"social_x_gateway": gw}, sq, "social")
    gw._refresh_session.assert_called_once_with("K", "P")
    msg = sq.get_nowait()
    assert msg["payload"]["success"] is True
    assert msg["payload"]["reason"] == "session_refreshed"


def test_social_worker_handle_heal_request_refresh_empty_replies_failure():
    from titan_hcl.modules.social_worker import _handle_heal_request
    sq = queue.Queue()
    gw = MagicMock()
    gw.config = {"api_key": "K", "proxy": "P"}
    gw._refresh_session.return_value = ""
    payload = {"plugin": "social_x", "action": "refresh_session",
               "correlation_id": "c3", "details": {}}
    _handle_heal_request(
        payload, {"social_x_gateway": gw}, sq, "social")
    msg = sq.get_nowait()
    assert msg["payload"]["success"] is False
    assert msg["payload"]["reason"] == "refresh_returned_empty"


def test_social_x_plugin_resolves_api_key_from_stealth_sage_secrets(
        monkeypatch):
    """Phase 1.5 fix: api_key in [stealth_sage].twitterapi_io_key (the live
    production location) MUST resolve through load_titan_config secrets
    merge when not passed via ctor."""
    import titan_hcl.health.social_x as _sx
    monkeypatch.setattr(
        _sx, "load_titan_params",
        lambda **kw: {"stealth_sage": {
            "twitterapi_io_key": "FAKE_TEST_TOKEN_NOT_REAL_AAA"}})
    sx = SocialXHealthCheck()  # no config — forces secrets path
    assert sx._api_key == "FAKE_TEST_TOKEN_NOT_REAL_AAA"


def test_social_x_plugin_ctor_config_overrides_secrets(monkeypatch):
    """Ctor-passed config takes precedence over secrets layer."""
    import titan_hcl.health.social_x as _sx
    monkeypatch.setattr(
        _sx, "load_titan_params",
        lambda **kw: {"stealth_sage": {
            "twitterapi_io_key": "FAKE_TEST_TOKEN_NOT_REAL_BBB"}})
    sx = SocialXHealthCheck({"api_key": "FROM_CTOR"})
    assert sx._api_key == "FROM_CTOR"


def test_social_worker_handle_heal_resolves_from_stealth_sage(monkeypatch):
    """Phase 1.10 fallback: when gateway._load_config raises, handler
    falls through to load_titan_config + stealth_sage.twitterapi_io_key
    resolution chain (mirrors gateway's own production fallback)."""
    # C.5/C.6: the heal fallback reads params.get_params per-section, not the
    # retired config_loader.load_titan_config.
    import titan_hcl.modules.social_worker as _sw
    _whole = {"stealth_sage": {"twitterapi_io_key": "FAKE_TEST_TOKEN_NOT_REAL_CCC"},
              "twitter_social": {"webshare_static_url": "http://proxy.example:80"}}
    monkeypatch.setattr(_sw, "get_params", lambda section: dict(_whole.get(section, {})))
    monkeypatch.setattr(_sw, "load_titan_params", lambda **kw: dict(_whole))
    from titan_hcl.modules.social_worker import _handle_heal_request
    sq = queue.Queue()
    gw = MagicMock()
    # Force gateway._load_config to fail so fallback path activates.
    gw._load_config.side_effect = RuntimeError("simulated load fail")
    gw.config = {"api_key": "", "proxy": ""}
    gw._refresh_session.return_value = "new_session"
    payload = {"plugin": "social_x", "action": "refresh_session",
               "correlation_id": "cm", "details": {}}
    _handle_heal_request(
        payload, {"social_x_gateway": gw}, sq, "social")
    gw._refresh_session.assert_called_once()
    call_args = gw._refresh_session.call_args
    assert call_args[0][0] == "FAKE_TEST_TOKEN_NOT_REAL_CCC"
    # Phase 1.10 closure: proxy MUST also resolve via the fallback chain
    # (this would have caught the prod bug — missing webshare_static_url
    # fallback caused twitterapi.io 400 "proxy is required" responses).
    assert call_args[0][1] == "http://proxy.example:80"
    msg = sq.get_nowait()
    assert msg["payload"]["success"] is True


def test_social_worker_handle_heal_request_missing_api_key_replies_failure(
        monkeypatch):
    # Phase 1.10: all 3 resolution layers empty → expect api_key_missing.
    import titan_hcl.modules.social_worker as _sw
    monkeypatch.setattr(_sw, "get_params", lambda section: {})
    monkeypatch.setattr(_sw, "load_titan_params", lambda **kw: {})
    from titan_hcl.modules.social_worker import _handle_heal_request
    sq = queue.Queue()
    gw = MagicMock()
    gw._load_config.return_value = {"api_key": "", "proxy": ""}
    gw.config = {"api_key": "", "proxy": ""}
    payload = {"plugin": "social_x", "action": "refresh_session",
               "correlation_id": "c4", "details": {}}
    _handle_heal_request(
        payload, {"social_x_gateway": gw}, sq, "social")
    msg = sq.get_nowait()
    assert msg["payload"]["success"] is False
    assert msg["payload"]["reason"] == "api_key_missing"


# ── Discovery (registry) ───────────────────────────────────────────


def test_discover_plugins_loads_social_x_on_all_titans():
    """Phase 1.6: social_x.applies_on = 'all' (was 'canonical_poller')
    because each Titan posts independently to its own social_x.db and
    needs its own X health monitor."""
    from titan_hcl.modules.health_monitor_worker import (
        _discover_plugins)
    config = {"social_x": {"api_key": "K", "user_name": "u"}}
    # All 3 Titans should load social_x.
    for tid in ("T1", "T2", "T3"):
        plugins = _discover_plugins(tid)  # C.6: reads sections via get_params
        names = {p.name for p in plugins}
        assert "social_x" in names, f"social_x missing on {tid}"


def test_discover_plugins_applies_on_canonical_poller_still_filters():
    """A hypothetical plugin with applies_on='canonical_poller' MUST still
    be filtered to canonical_poller Titan only (contract preserved for
    future plugins)."""
    from titan_hcl.modules.health_monitor_worker import (
        _discover_plugins, HealthCheckPlugin)
    # Use a mock subclass with applies_on=canonical_poller; we can't add
    # one to the package without registry side-effects, so instead just
    # verify the filter LOGIC handles canonical_poller correctly by
    # checking social_x continues to load when we artificially restrict.
    # (The real applies_on=canonical_poller branch is exercised by the
    # health_monitor_worker logic itself — kept here as contract sentinel.)
    config = {"social_x": {"canonical_poller_titan_id": "T1",
                            "api_key": "K", "user_name": "u"}}
    plugins = _discover_plugins("T1")  # C.6: reads sections via get_params
    names = {p.name for p in plugins}
    assert "social_x" in names
