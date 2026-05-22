"""tests/test_arch_map_health_monitor.py — SPEC v1.12.0 D-SPEC-67 Phase 2.

Coverage:
  • _hm_read_state: T1 local file read, missing-file → None, malformed
    JSON → None.
  • _hm_read_journal_tail: T1 local read of jsonl, missing-file → [],
    malformed lines skipped.
  • _hm_ago: past + future + 'never' formatting.
  • _hm_status_icon: OK/DEGRADED/DOWN/? mappings.
  • _check_social_x_health: missing-state → warn; present-OK → ok;
    present-DEGRADED → warn; present-DOWN → fail.
  • run_x_check: rc=0 on empty + populated journal; --hours filter works.
  • run_health_monitor: rc=0 with state; rc=1 with no data anywhere;
    --plugin filter.

Per CLAUDE.md: pytest -p no:anchorpy, separate process per file.
"""
from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import pytest


@pytest.fixture
def am(monkeypatch, tmp_path):
    """Load scripts/arch_map.py as a module + redirect PROJECT_ROOT to tmp."""
    spec = importlib.util.spec_from_file_location(
        "arch_map_under_test",
        Path(__file__).resolve().parents[1] / "scripts" / "arch_map.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)
    return mod


@pytest.fixture
def hm_dir(am):
    d = am.PROJECT_ROOT / "data" / "health_monitor"
    d.mkdir(parents=True)
    return d


def _write_state(hm_dir: Path, payload: dict) -> None:
    (hm_dir / "state.json").write_text(json.dumps(payload))


def _write_journal(hm_dir: Path, events: list[dict]) -> None:
    with open(hm_dir / "events.jsonl", "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


# ── _hm_ago ─────────────────────────────────────────────────────────


def test_hm_ago_none_returns_never(am):
    assert am._hm_ago(None) == "never"
    assert am._hm_ago(0) == "never"


def test_hm_ago_past_seconds_minutes_hours_days(am):
    now = 10_000_000.0
    assert am._hm_ago(now - 30, now) == "30s"
    assert am._hm_ago(now - 600, now) == "10m"
    assert am._hm_ago(now - 3 * 3600, now) == "3.0h"
    assert am._hm_ago(now - 2 * 86400, now) == "2.0d"


def test_hm_ago_future_renders_in_prefix(am):
    now = 10_000_000.0
    assert am._hm_ago(now + 30, now) == "in 30s"
    assert am._hm_ago(now + 600, now) == "in 10m"
    assert am._hm_ago(now + 3 * 3600, now) == "in 3.0h"


# ── _hm_status_icon ─────────────────────────────────────────────────


def test_hm_status_icon(am):
    assert am._hm_status_icon("OK") == "✓"
    assert am._hm_status_icon("DEGRADED") == "⚠"
    assert am._hm_status_icon("DOWN") == "✗"
    assert am._hm_status_icon("WHATEVER") == "?"


# ── _hm_read_state ──────────────────────────────────────────────────


def test_hm_read_state_missing_returns_none(am):
    assert am._hm_read_state("T1") is None


def test_hm_read_state_present_roundtrip(am, hm_dir):
    _write_state(hm_dir, {"plugins": {"x": {"a": 1}}, "updated_at": 42.0})
    s = am._hm_read_state("T1")
    assert s is not None
    assert s["plugins"]["x"]["a"] == 1
    assert s["updated_at"] == 42.0


def test_hm_read_state_malformed_returns_none(am, hm_dir):
    (hm_dir / "state.json").write_text("{not valid json")
    assert am._hm_read_state("T1") is None


# ── _hm_read_journal_tail ───────────────────────────────────────────


def test_hm_read_journal_missing_returns_empty(am):
    assert am._hm_read_journal_tail("T1") == []


def test_hm_read_journal_skips_malformed_lines(am, hm_dir):
    with open(hm_dir / "events.jsonl", "w") as f:
        f.write(json.dumps({"kind": "x", "ts": 1, "payload": {}}) + "\n")
        f.write("malformed line\n")
        f.write(json.dumps({"kind": "y", "ts": 2, "payload": {}}) + "\n")
    events = am._hm_read_journal_tail("T1")
    assert len(events) == 2
    assert events[0]["kind"] == "x"
    assert events[1]["kind"] == "y"


def test_hm_read_journal_respects_max_lines(am, hm_dir):
    with open(hm_dir / "events.jsonl", "w") as f:
        for i in range(100):
            f.write(json.dumps({"kind": "k", "ts": i, "payload": {}})
                    + "\n")
    events = am._hm_read_journal_tail("T1", max_lines=20)
    assert len(events) == 20
    # Tail: ts 80..99
    assert events[0]["ts"] == 80
    assert events[-1]["ts"] == 99


# ── _check_social_x_health ──────────────────────────────────────────


def test_check_social_x_health_missing_state_warns(am):
    r = am._check_social_x_health()
    assert r["status"] == "warn"
    assert r["subsystem"] == "social_x_health"


def test_check_social_x_health_plugin_absent_warns(am, hm_dir):
    _write_state(hm_dir, {"plugins": {}, "updated_at": 1.0})
    r = am._check_social_x_health()
    assert r["status"] == "warn"
    assert "not loaded" in r["details"]


def test_check_social_x_health_ok_status(am, hm_dir):
    now = time.time()
    _write_state(hm_dir, {
        "plugins": {"social_x": {
            "last_result": {"plugin": "social_x", "layer": "posting",
                            "status": "OK", "reason": "v=3",
                            "ts": now - 60},
            "consecutive_failures": 0,
            "heal_history_24h": [],
        }},
        "updated_at": now})
    r = am._check_social_x_health()
    assert r["status"] == "ok"
    assert r["last_status"] == "OK"


def test_check_social_x_health_degraded_warns(am, hm_dir):
    now = time.time()
    _write_state(hm_dir, {
        "plugins": {"social_x": {
            "last_result": {"plugin": "social_x", "layer": "posting",
                            "status": "DEGRADED", "reason": "v=0",
                            "ts": now - 60},
            "consecutive_failures": 1,
            "heal_history_24h": [{"ts": now - 30, "action": "x",
                                   "result": "failed", "reason": "e"}],
        }},
        "updated_at": now})
    r = am._check_social_x_health()
    assert r["status"] == "warn"
    assert r["last_status"] == "DEGRADED"
    assert r["consecutive_failures"] == 1
    assert r["heals_24h"] == 1


def test_check_social_x_health_down_fails(am, hm_dir):
    now = time.time()
    _write_state(hm_dir, {
        "plugins": {"social_x": {
            "last_result": {"plugin": "social_x", "layer": "pipeline",
                            "status": "DOWN", "reason": "net_err",
                            "ts": now - 60},
            "consecutive_failures": 0,
            "heal_history_24h": [],
        }},
        "updated_at": now})
    r = am._check_social_x_health()
    assert r["status"] == "fail"
    assert r["last_status"] == "DOWN"


# ── run_x_check ─────────────────────────────────────────────────────


def test_run_x_check_no_journal_rc_zero(am, capsys):
    rc = am.run_x_check([])
    captured = capsys.readouterr()
    assert rc == 0
    assert "no health-monitor journal" in captured.out


def test_run_x_check_populated_journal_renders(am, hm_dir, capsys):
    now = time.time()
    _write_journal(hm_dir, [
        {"kind": "check_result", "ts": now - 60,
         "payload": {"plugin": "social_x", "layer": "pipeline",
                     "status": "OK", "reason": "ok"}},
        {"kind": "check_result", "ts": now - 30,
         "payload": {"plugin": "social_x", "layer": "posting",
                     "status": "DEGRADED", "reason": "v=0"}},
        {"kind": "heal_request", "ts": now - 25,
         "payload": {"plugin": "social_x", "action": "refresh_session",
                     "owning_worker": "social",
                     "correlation_id": "c1"}},
        {"kind": "heal_result", "ts": now - 20,
         "payload": {"plugin": "social_x", "action": "refresh_session",
                     "success": True, "reason": "ok",
                     "correlation_id": "c1"}},
    ])
    rc = am.run_x_check([])
    captured = capsys.readouterr()
    assert rc == 0
    assert "CHECK RESULTS (2)" in captured.out
    assert "HEAL REQUESTS (1)" in captured.out
    assert "HEAL RESULTS (1)" in captured.out


def test_run_x_check_hours_filter_excludes_old(am, hm_dir, capsys):
    now = time.time()
    _write_journal(hm_dir, [
        {"kind": "check_result", "ts": now - 7200,  # 2h ago
         "payload": {"plugin": "social_x", "layer": "pipeline",
                     "status": "OK", "reason": "ok"}},
    ])
    rc = am.run_x_check(["--hours", "1"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "no social_x events in last 1h" in captured.out


def test_run_x_check_help_rc_zero(am, capsys):
    rc = am.run_x_check(["--help"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "x_check" in captured.out


# ── run_health_monitor ──────────────────────────────────────────────


def test_run_health_monitor_no_data_rc_one(am, capsys):
    rc = am.run_health_monitor([])
    captured = capsys.readouterr()
    assert rc == 1
    assert "No health_monitor data" in captured.out


def test_run_health_monitor_with_state_rc_zero(am, hm_dir, capsys):
    now = time.time()
    _write_state(hm_dir, {
        "plugins": {"social_x": {
            "last_result": {"plugin": "social_x", "layer": "posting",
                            "status": "OK", "reason": "v=3",
                            "ts": now - 60},
            "consecutive_failures": 0,
            "heal_history_24h": [],
            "next_fire_time": now + 3600,
        }},
        "updated_at": now})
    rc = am.run_health_monitor([])
    captured = capsys.readouterr()
    assert rc == 0
    assert "social_x" in captured.out
    assert "status=OK" in captured.out


def test_run_health_monitor_plugin_filter(am, hm_dir, capsys):
    now = time.time()
    _write_state(hm_dir, {
        "plugins": {
            "social_x": {
                "last_result": {"plugin": "social_x", "layer": "posting",
                                "status": "OK", "reason": "v=3",
                                "ts": now - 60},
                "consecutive_failures": 0,
                "heal_history_24h": [],
            },
            "other_plugin": {
                "last_result": {"plugin": "other_plugin", "layer": "x",
                                "status": "OK", "reason": "ok",
                                "ts": now - 60},
                "consecutive_failures": 0,
                "heal_history_24h": [],
            },
        },
        "updated_at": now})
    rc = am.run_health_monitor(["--plugin", "social_x"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "social_x" in captured.out
    assert "other_plugin" not in captured.out
