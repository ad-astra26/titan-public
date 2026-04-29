"""Tests for /v4/warning-monitor — endpoint that surfaces the
warning_monitor_worker's rolling state.json to arch_map runtime audits.

Bug surfaced 2026-04-25: `arch_map silent-swallows --runtime` probed
http://<titan>:7777/v4/warning-monitor and got HTTP 404 on all 3 Titans.
The worker was correctly persisting state.json but no API endpoint
exposed it.

Fix: dashboard.py now serves the snapshot, mirroring the file-based
read pattern used by /v4/imw-health.
"""
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def fake_state_file():
    """Create a fixture state.json with realistic shape."""
    with tempfile.TemporaryDirectory() as td:
        state_dir = Path(td) / "data" / "warning_monitor"
        state_dir.mkdir(parents=True)
        state_path = state_dir / "state.json"
        snap = {
            "saved_ts": 1777115713.5,
            "aggregated": {
                "DivineBus": {
                    "count": 132,
                    "first_seen_ts": 1777110829.27,
                    "last_seen_ts": 1777115683.37,
                    "last_msg": "[DivineBus] Request timed out",
                    "by_level": {"WARNING": 132},
                    "rate_1m": 1,
                },
                "swallow:logic.cgn.cross_consumer_transfer_error": {
                    "count": 5,
                    "first_seen_ts": 1777115000.0,
                    "last_seen_ts": 1777115500.0,
                    "last_msg": "RuntimeError: timeout",
                    "by_level": {"SWALLOW": 5},
                    "rate_1m": 0,
                },
            },
        }
        state_path.write_text(json.dumps(snap))
        old_cwd = os.getcwd()
        os.chdir(td)
        try:
            yield state_path
        finally:
            os.chdir(old_cwd)


@pytest.fixture
def app_client():
    """Build a minimal FastAPI app + TestClient that mounts the
    dashboard router. Avoids spinning up the full TitanPlugin."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from titan_plugin.api import dashboard

    app = FastAPI()
    app.include_router(dashboard.router)
    return TestClient(app)


def test_warning_monitor_endpoint_returns_state(app_client, fake_state_file):
    resp = app_client.get("/v4/warning-monitor")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    data = body["data"]
    assert "aggregated" in data
    assert "DivineBus" in data["aggregated"]
    assert data["aggregated"]["DivineBus"]["count"] == 132
    # The endpoint adds the state_file_age_sec field for staleness checks
    assert "state_file_age_sec" in data
    assert isinstance(data["state_file_age_sec"], (int, float))


def test_warning_monitor_endpoint_swallow_keys_passthrough(app_client,
                                                              fake_state_file):
    """arch_map silent-swallows --runtime filters by `swallow:` prefix —
    the endpoint must surface those keys without mangling."""
    resp = app_client.get("/v4/warning-monitor")
    body = resp.json()
    swallow_keys = {k: v for k, v in body["data"]["aggregated"].items()
                    if k.startswith("swallow:")}
    assert "swallow:logic.cgn.cross_consumer_transfer_error" in swallow_keys
    assert swallow_keys["swallow:logic.cgn.cross_consumer_transfer_error"]["count"] == 5


def test_warning_monitor_endpoint_error_when_file_missing(app_client):
    """Worker hasn't started yet (no state.json) → returns the dashboard's
    error envelope (status_code=500 + status=error) so arch_map's runtime
    probe surfaces it as PROBE-FAIL rather than silently treating "no
    swallow keys" as PASS."""
    with tempfile.TemporaryDirectory() as td:
        old_cwd = os.getcwd()
        os.chdir(td)
        try:
            resp = app_client.get("/v4/warning-monitor")
        finally:
            os.chdir(old_cwd)
    assert resp.status_code == 500
    body = resp.json()
    assert body["status"] == "error"
    detail = body.get("detail", "").lower()
    assert "warning_monitor" in detail or "not found" in detail


def test_warning_monitor_endpoint_marks_stale_state(app_client):
    """State.json older than 5 min should be flagged stale."""
    with tempfile.TemporaryDirectory() as td:
        state_dir = Path(td) / "data" / "warning_monitor"
        state_dir.mkdir(parents=True)
        state_path = state_dir / "state.json"
        state_path.write_text(json.dumps({"saved_ts": 0, "aggregated": {}}))
        # Backdate the mtime
        old_mtime = os.path.getmtime(state_path) - 600
        os.utime(state_path, (old_mtime, old_mtime))
        old_cwd = os.getcwd()
        os.chdir(td)
        try:
            resp = app_client.get("/v4/warning-monitor")
        finally:
            os.chdir(old_cwd)
    body = resp.json()
    assert body["status"] == "ok"
    assert "stale_warning" in body["data"]
    assert body["data"]["state_file_age_sec"] >= 300
