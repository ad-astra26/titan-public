"""Tests for the phone → Titan context/presence uplink (RFP_titan_mobile_app Phase 3 / AG6).

Proves: per-sensor opt-in flags default OFF (INV-OPT-IN), the field-level gate (un-opted-in
fields are never stored), settings patch + cadence clamp, the per-device context log + latest,
read_latest picking the newest across devices, and the dispatch auth gate (device-signed
/console/context + /console/presence[/settings]). Stops at persistence — no cognition (AG8).
"""
import base64
import json
import time

import pytest

from titan_console import _ed25519, events, pairing, presence
from titan_console.agent import dispatch
from titan_console.context import Context

_INTERNAL_KEY = "TEST-INTERNAL-KEY"


def _ctx(tmp_path):
    (tmp_path / "secrets.toml").write_text(f'[api]\ninternal_key = "{_INTERNAL_KEY}"\n')
    return Context(install_root=tmp_path, titan_id="T1",
                   secrets_path=tmp_path / "secrets.toml")


def _b64(b):
    return base64.b64encode(b).decode()


@pytest.fixture(autouse=True)
def _clear():
    pairing._clear_caches()
    events._reset_registries()
    yield
    pairing._clear_caches()
    events._reset_registries()


def _register_device(ctx, device_id, t0=1000.0):
    _, payload = pairing.mint_pairing(ctx, now=t0)
    token = payload["pairing_token"]
    seed, pub = _ed25519.keygen()
    pairing.submit_device(ctx, {"pairing_token": token, "device_pubkey": _b64(pub),
                                "device_id": device_id, "label": device_id}, now=t0 + 1)
    code = pairing.code6(base64.b64decode(token), pub)
    pairing.confirm_device(ctx, token, code, now=t0 + 2)
    return seed


def _signed_headers(seed, device_id, method, path, body=b""):
    ts = str(int(time.time()))
    msg = pairing.canonical_request(method, path, ts, pairing.body_sha256_hex(body)).encode()
    return {"x-device-id": device_id, "x-timestamp": ts,
            "x-signature": _b64(_ed25519.sign(msg, seed))}


# ── settings (opt-in flags) ──────────────────────────────────────────────────
def test_settings_default_all_off(tmp_path):
    ctx = _ctx(tmp_path)
    s = presence.get_settings(ctx)
    assert s["location_enabled"] is False and s["time_enabled"] is False
    assert s["motion_enabled"] is False and s["battery_enabled"] is False
    assert s["cadence_minutes"] == 15


def test_set_settings_patches_and_clamps(tmp_path):
    ctx = _ctx(tmp_path)
    s = presence.set_settings(ctx, {"location_enabled": True, "cadence_minutes": 99999})
    assert s["location_enabled"] is True and s["cadence_minutes"] == 1440
    s2 = presence.set_settings(ctx, {"cadence_minutes": 0})
    assert s2["cadence_minutes"] == 1 and s2["location_enabled"] is True  # prior flag persists


# ── ingest gating ────────────────────────────────────────────────────────────
def test_ingest_all_off_stores_nothing(tmp_path):
    ctx = _ctx(tmp_path)
    res = presence.ingest(ctx, "dev-1", [{"lat": 50.08, "lon": 14.43, "ts": 1.0}])
    assert res["accepted"] == 0
    assert presence.read_latest(ctx) == {}


def test_ingest_location_opt_in_stores(tmp_path):
    ctx = _ctx(tmp_path)
    _register_device(ctx, "dev-1")
    presence.set_settings(ctx, {"location_enabled": True})
    res = presence.ingest(ctx, "dev-1", [{"lat": 50.08, "lon": 14.43, "accuracy": 12, "ts": 5.0}])
    assert res["accepted"] == 1
    latest = presence.read_latest(ctx)
    assert latest["lat"] == 50.08 and latest["lon"] == 14.43 and latest["device_id"] == "dev-1"


def test_ingest_field_gate_drops_unopted(tmp_path):
    ctx = _ctx(tmp_path)
    _register_device(ctx, "dev-1")
    presence.set_settings(ctx, {"location_enabled": True})  # battery NOT opted in
    presence.ingest(ctx, "dev-1", [{"lat": 1.0, "lon": 2.0, "battery": 80, "ts": 5.0}])
    latest = presence.read_latest(ctx)
    assert latest["lat"] == 1.0 and "battery" not in latest  # battery gated out


def test_ingest_missing_ts_gets_server_clock(tmp_path):
    ctx = _ctx(tmp_path)
    _register_device(ctx, "dev-1")
    presence.set_settings(ctx, {"time_enabled": True})
    presence.ingest(ctx, "dev-1", [{"tz": "Europe/Prague", "local_time": "19:40"}])
    latest = presence.read_latest(ctx)
    assert latest["tz"] == "Europe/Prague" and isinstance(latest["ts"], (int, float))


def test_context_log_appends(tmp_path):
    ctx = _ctx(tmp_path)
    _register_device(ctx, "dev-1")
    presence.set_settings(ctx, {"location_enabled": True})
    presence.ingest(ctx, "dev-1", [{"lat": 1.0, "lon": 1.0, "ts": 1.0}])
    presence.ingest(ctx, "dev-1", [{"lat": 2.0, "lon": 2.0, "ts": 2.0}])
    log = (presence._device_dir(ctx, "dev-1") / "context_log.jsonl").read_text().strip().split("\n")
    assert len(log) == 2 and json.loads(log[-1])["lat"] == 2.0
    assert presence.read_latest(ctx)["lat"] == 2.0  # latest = newest ts


def test_read_latest_newest_across_devices(tmp_path):
    ctx = _ctx(tmp_path)
    _register_device(ctx, "dev-1", t0=1000.0)
    _register_device(ctx, "dev-2", t0=2000.0)
    presence.set_settings(ctx, {"location_enabled": True})
    presence.ingest(ctx, "dev-1", [{"lat": 1.0, "lon": 1.0, "ts": 100.0}])
    presence.ingest(ctx, "dev-2", [{"lat": 9.0, "lon": 9.0, "ts": 200.0}])  # newer
    assert presence.read_latest(ctx)["device_id"] == "dev-2"


# ── dispatch auth gate ───────────────────────────────────────────────────────
def test_context_route_requires_device_auth(tmp_path):
    ctx = _ctx(tmp_path)
    body = json.dumps({"samples": [{"lat": 1.0}]}).encode()
    s, r = dispatch(ctx, "POST", "/console/context", {}, body, {}, True)
    assert s == 401


def test_presence_read_requires_device_auth(tmp_path):
    ctx = _ctx(tmp_path)
    s, _ = dispatch(ctx, "GET", "/console/presence", {}, b"", {}, True)
    assert s == 401


def test_signed_context_and_presence_roundtrip(tmp_path):
    ctx = _ctx(tmp_path)
    seed = _register_device(ctx, "dev-1")
    # opt-in to location over a signed settings POST
    sbody = json.dumps({"location_enabled": True}).encode()
    h = _signed_headers(seed, "dev-1", "POST", "/console/presence/settings", sbody)
    s, _ = dispatch(ctx, "POST", "/console/presence/settings", {}, sbody, h, True)
    assert s == 200
    # upload a signed context sample
    cbody = json.dumps({"samples": [{"lat": 50.08, "lon": 14.43, "ts": 5.0}]}).encode()
    h = _signed_headers(seed, "dev-1", "POST", "/console/context", cbody)
    s, r = dispatch(ctx, "POST", "/console/context", {}, cbody, h, True)
    assert s == 200 and r["accepted"] == 1
    # read it back signed
    h = _signed_headers(seed, "dev-1", "GET", "/console/presence")
    s, r = dispatch(ctx, "GET", "/console/presence", {}, b"", h, True)
    assert s == 200 and r["lat"] == 50.08 and r["device_id"] == "dev-1"
