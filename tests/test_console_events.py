"""Tests for the Titan ⇄ App event channel — Phase 1 (RFP_titan_app_event_channel §7.1).

Proves the durable per-device queue (enqueue/pending/drain/ack), deliver-once + dedupe,
the long-poll wake, the monotonic cursor surviving a prune, the device_id path-traversal
guard, presence + heartbeat-ack, the HealthMonitor self-enqueue, and the dispatch auth
gate on all three routes (device-signed drain/heartbeat, internal-key local-only enqueue).
"""
import base64
import threading
import time

import pytest

from titan_console import _ed25519, alerts, events, pairing, presence
from titan_console.agent import dispatch
from titan_console.context import Context

_INTERNAL_KEY = "TEST-INTERNAL-KEY"


def _ctx(tmp_path):
    # _titan_dir → secrets_path.parent, so all queue/presence/devices state lands in tmp.
    (tmp_path / "secrets.toml").write_text(
        f'[api]\ninternal_key = "{_INTERNAL_KEY}"\n')
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


def _register_signed_device(ctx, t0=1000.0):
    _, payload = pairing.mint_pairing(ctx, now=t0)
    token = payload["pairing_token"]
    seed, pub = _ed25519.keygen()
    pairing.submit_device(ctx, {"pairing_token": token, "device_pubkey": _b64(pub),
                                "device_id": "dev-1", "label": "Maker phone"}, now=t0 + 1)
    code = pairing.code6(base64.b64decode(token), pub)
    pairing.confirm_device(ctx, token, code, now=t0 + 2)
    return seed, pub


def _sign(seed, method, path, ts, body=b""):
    msg = pairing.canonical_request(method, path, str(ts),
                                    pairing.body_sha256_hex(body)).encode()
    return _b64(_ed25519.sign(msg, seed))


def _signed_headers(seed, method, path, body=b""):
    ts = str(int(time.time()))
    return {"x-device-id": "dev-1", "x-timestamp": ts,
            "x-signature": _sign(seed, method, path, ts, body)}


# ── queue unit behaviour ─────────────────────────────────────────────────────
def test_enqueue_pending_ack_happy_path(tmp_path):
    ctx = _ctx(tmp_path)
    events.enqueue(ctx, "dev-1", type="message", payload={"text": "hi"})
    events.enqueue(ctx, "dev-1", type="message", payload={"text": "again"})
    fresh, cursor = events.pending(ctx, "dev-1", 0)
    assert [e["seq"] for e in fresh] == [1, 2] and cursor == 2
    assert fresh[0]["payload"] == {"text": "hi"}
    assert events.ack(ctx, "dev-1", 2) == 2
    fresh2, cursor2 = events.pending(ctx, "dev-1", 2)
    assert fresh2 == [] and cursor2 == 2


def test_deliver_once_dedupe(tmp_path):
    ctx = _ctx(tmp_path)
    events.enqueue(ctx, "dev-1", type="message", payload={"t": 1}, dedupe_key="k1")
    events.enqueue(ctx, "dev-1", type="message", payload={"t": 2}, dedupe_key="k1")
    fresh, _ = events.pending(ctx, "dev-1", 0)
    assert len(fresh) == 1 and fresh[0]["payload"] == {"t": 1}  # second is a no-op


def test_cursor_is_monotonic_across_prune(tmp_path):
    ctx = _ctx(tmp_path)
    events.enqueue(ctx, "dev-1", type="message", payload={})
    events.enqueue(ctx, "dev-1", type="message", payload={})
    events.ack(ctx, "dev-1", 2)                       # outbox now empty
    e3 = events.enqueue(ctx, "dev-1", type="message", payload={})
    assert e3["seq"] == 3                             # seq did NOT reset to 1
    fresh, cursor = events.pending(ctx, "dev-1", 2)
    assert [e["seq"] for e in fresh] == [3] and cursor == 3


def test_partial_ack_keeps_unacked(tmp_path):
    ctx = _ctx(tmp_path)
    for _ in range(3):
        events.enqueue(ctx, "dev-1", type="message", payload={})
    assert events.ack(ctx, "dev-1", 1) == 1
    fresh, cursor = events.pending(ctx, "dev-1", 0)
    assert [e["seq"] for e in fresh] == [2, 3]


def test_drain_long_poll_wakes_on_enqueue(tmp_path):
    ctx = _ctx(tmp_path)
    _register_signed_device(ctx)  # not required by drain() but mirrors reality
    result = {}

    def _drain():
        t0 = time.time()
        evs, cur = events.drain(ctx, "dev-1", 0, wait_s=5)
        result["evs"], result["cur"], result["dt"] = evs, cur, time.time() - t0

    th = threading.Thread(target=_drain)
    th.start()
    time.sleep(0.2)                       # let the drain park on the waiter
    events.enqueue(ctx, "dev-1", type="message", payload={"text": "wake"})
    th.join(timeout=3)
    assert not th.is_alive()
    assert len(result["evs"]) == 1 and result["cur"] == 1
    assert result["dt"] < 3               # woke well before the 5s timeout


def test_drain_instant_when_wait_zero(tmp_path):
    ctx = _ctx(tmp_path)
    t0 = time.time()
    evs, cur = events.drain(ctx, "dev-1", 0, wait_s=0)
    assert evs == [] and cur == 0 and (time.time() - t0) < 1  # WorkManager-style instant drain


def test_device_id_path_traversal_rejected(tmp_path):
    ctx = _ctx(tmp_path)
    for bad in ("../evil", "a/b", "x" * 65, "", "dev 1"):
        with pytest.raises(ValueError):
            events.enqueue(ctx, bad, type="message", payload={})


# ── presence ─────────────────────────────────────────────────────────────────
def test_presence_put_get(tmp_path):
    ctx = _ctx(tmp_path)
    presence.put(ctx, "dev-1", {"state": "background", "battery": 42, "ack_cursor": 7})
    rec = presence.get(ctx, "dev-1")
    assert rec["state"] == "background" and rec["battery"] == 42 and rec["ack_cursor"] == 7
    assert rec["last_seen"] > 0
    assert presence.get(ctx, "dev-2") is None


# ── HealthMonitor self-enqueue ───────────────────────────────────────────────
def test_health_transition_enqueues_to_device(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    _register_signed_device(ctx)
    monkeypatch.setattr(alerts, "titan_status",
                        lambda c: {"up": False, "why_down": "boom"})
    mon = alerts.HealthMonitor(ctx)
    mon._last_up = True                                   # prime "was up" → this is a down edge
    res = mon.check_once()
    assert res["transition"] == "down"
    fresh, _ = events.pending(ctx, "dev-1", 0)
    assert len(fresh) == 1 and fresh[0]["type"] == "health"
    assert fresh[0]["payload"]["up"] is False and fresh[0]["urgency"] == "high"


def test_health_no_transition_no_enqueue(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    _register_signed_device(ctx)
    monkeypatch.setattr(alerts, "titan_status", lambda c: {"up": True})
    mon = alerts.HealthMonitor(ctx)
    mon._last_up = True                                   # still up → no edge
    mon.check_once()
    assert events.pending(ctx, "dev-1", 0) == ([], 0)


# ── dispatch routes + auth ───────────────────────────────────────────────────
def test_route_roundtrip_enqueue_drain_ack(tmp_path):
    ctx = _ctx(tmp_path)
    seed, _ = _register_signed_device(ctx)
    # 1) Titan enqueues (local + internal key)
    body = b'{"device_id":"dev-1","type":"message","payload":{"text":"It is 6"}}'
    s, r = dispatch(ctx, "POST", "/console/events/enqueue", {}, body,
                    {"x-titan-internal-key": _INTERNAL_KEY}, True)
    assert s == 200 and r["seq"] == 1
    # 2) Phone drains (device-signed)
    s, r = dispatch(ctx, "GET", "/console/events", {"since": ["0"]}, b"",
                    _signed_headers(seed, "GET", "/console/events"), True)
    assert s == 200 and len(r["events"]) == 1 and r["cursor"] == 1
    assert r["events"][0]["payload"] == {"text": "It is 6"}
    # 3) Phone acks via heartbeat → event pruned
    hb = b'{"state":"background","ack_cursor":1}'
    s, r = dispatch(ctx, "POST", "/console/app/heartbeat", {}, hb,
                    _signed_headers(seed, "POST", "/console/app/heartbeat", hb), True)
    assert s == 200 and r["ok"] is True
    assert events.pending(ctx, "dev-1", 0) == ([], 0)
    assert presence.get(ctx, "dev-1")["state"] == "background"


def test_events_drain_requires_device_signature(tmp_path):
    ctx = _ctx(tmp_path)
    _register_signed_device(ctx)
    # unsigned, non-local → blocked by the AD-5 gate
    s, _ = dispatch(ctx, "GET", "/console/events", {}, b"", {}, False)
    assert s == 401
    # unsigned even on localhost → the route still demands a device signature
    s, _ = dispatch(ctx, "GET", "/console/events", {}, b"", {}, True)
    assert s == 401


def test_enqueue_requires_internal_key(tmp_path):
    ctx = _ctx(tmp_path)
    _register_signed_device(ctx)
    body = b'{"device_id":"dev-1","type":"message","payload":{}}'
    # no key → 401
    s, _ = dispatch(ctx, "POST", "/console/events/enqueue", {}, body, {}, True)
    assert s == 401
    # wrong key → 401
    s, _ = dispatch(ctx, "POST", "/console/events/enqueue", {}, body,
                    {"x-titan-internal-key": "WRONG"}, True)
    assert s == 401


def test_enqueue_is_local_only(tmp_path):
    ctx = _ctx(tmp_path)
    _register_signed_device(ctx)
    body = b'{"device_id":"dev-1","type":"message","payload":{}}'
    # even WITH the right key, a non-local enqueue is refused (the AD-5 gate fires first)
    s, _ = dispatch(ctx, "POST", "/console/events/enqueue", {}, body,
                    {"x-titan-internal-key": _INTERNAL_KEY}, False)
    assert s == 401


def test_enqueue_unknown_device(tmp_path):
    ctx = _ctx(tmp_path)
    body = b'{"device_id":"ghost","type":"message","payload":{}}'
    s, _ = dispatch(ctx, "POST", "/console/events/enqueue", {}, body,
                    {"x-titan-internal-key": _INTERNAL_KEY}, True)
    assert s == 404
