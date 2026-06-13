"""v0 app-inbox consumer — RFP_titan_app_event_channel §7.3 (minimal, observability-only).

When the Maker taps a Channel-2 action button or a feedback chip, or sets his
availability ("busy"), the response lands durably in the Console Agent's per-device
inbox / presence record (the transport — survives a kernel-down, AG-EVT-3). This is the
KERNEL-side reader: it periodically drains the inbox over the local internal-key route,
LOGS each item, and acks it — so the phone→kernel round-trip is observable. It logs the
declared availability on change too.

🚩 SCOPE FENCE (load-bearing): this is **observability ONLY**. It does NOT act on busy
(no ``if busy: don't_speak`` — that hardcoded mute is exactly INV-MIS-EMERGENCE's
forbidden case), does NOT shape speak-frequency, does NOT learn. The real consumer —
self-regulation + learning what "busy" *means* — is RFP_missions_and_the_maker_model
Phase 4. This proves the wire and gives that future engine a live feed to attach to.
"""
from __future__ import annotations

import json
import logging
import os
import ssl
import threading
import urllib.request as _ur

logger = logging.getLogger(__name__)

_DEFAULT_INTERVAL_S = 30
_last_avail: dict[str, str] = {}  # device_id → last-logged availability (throttle log spam)


def _titan_dir() -> str:
    return os.path.expanduser("~/.titan")


def _internal_key() -> str | None:
    import tomllib
    sp = os.path.join(_titan_dir(), "secrets.toml")
    if not os.path.exists(sp):
        return None
    try:
        with open(sp, "rb") as f:
            return (tomllib.load(f).get("api") or {}).get("internal_key")
    except Exception:
        return None


def _device_ids() -> list[str]:
    dp = os.path.join(_titan_dir(), "devices.json")
    if not os.path.exists(dp):
        return []
    try:
        with open(dp) as f:
            return [d.get("device_id") for d in json.load(f)
                    if isinstance(d, dict) and d.get("device_id")]
    except Exception:
        return []


def _sslctx() -> ssl.SSLContext:
    c = ssl.create_default_context()
    c.check_hostname = False
    c.verify_mode = ssl.CERT_NONE  # self-signed pinned cert on loopback
    return c


def _http(method: str, path: str, key: str, body: dict | None = None) -> dict:
    port = int(os.environ.get("TITAN_CONSOLE_PORT", "7799"))
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = _ur.Request(f"https://127.0.0.1:{port}{path}", data=data, method=method,
                      headers={"Content-Type": "application/json",
                               "X-Titan-Internal-Key": key})
    with _ur.urlopen(req, timeout=5, context=_sslctx()) as r:
        return json.loads(r.read().decode())


def _log_availability(did: str) -> None:
    """Log the Maker's declared availability only when it CHANGES (no per-cycle spam)."""
    pp = os.path.join(_titan_dir(), "devices", did, "presence.json")
    if not os.path.exists(pp):
        return
    try:
        with open(pp) as f:
            av = json.load(f).get("availability") or "available"
    except Exception:
        return
    if _last_avail.get(did) != av:
        _last_avail[did] = av
        logger.info("[AppInbox] %s availability → %s (observed; meaning learned in missions P4)",
                    did[:8], av)


def drain_once(key: str) -> int:
    """Drain + log + ack every device's inbox once. Returns the number of items logged."""
    n = 0
    for did in _device_ids():
        _log_availability(did)
        try:
            resp = _http("GET", f"/console/events/inbox?device_id={did}&since=0", key)
        except Exception:
            continue
        items = resp.get("items", [])
        if not items:
            continue
        cursor = max(int(it.get("seq", 0)) for it in items)
        for it in items:
            logger.info(
                "[AppInbox] %s %s in_reply_to=%s action=%s reaction=%s stars=%s",
                did[:8], it.get("kind"), it.get("in_reply_to"),
                it.get("action_id"), it.get("reaction"), it.get("stars"))
            n += 1
        try:
            _http("POST", "/console/events/inbox/ack", key,
                  {"device_id": did, "cursor": cursor})
        except Exception:
            pass  # un-acked items just re-log next cycle — bounded, harmless
    return n


def _loop(interval_s: int, stop_event: threading.Event) -> None:
    while not stop_event.wait(interval_s):
        key = _internal_key()
        if not key:
            continue  # no internal key → can't reach the console; idle quietly
        try:
            drain_once(key)
        except Exception as e:
            logger.debug("[AppInbox] drain cycle error: %s", e)


def start_app_inbox_consumer(interval_s: int = _DEFAULT_INTERVAL_S) -> threading.Event:
    """Spawn the daemon drain loop. Returns its stop Event. Idempotent-safe to call once
    per worker boot. I/O-bound (a short HTTP poll) → releases the GIL, never starves the
    worker heartbeat."""
    stop = threading.Event()
    threading.Thread(target=_loop, args=(interval_s, stop),
                     name="app_inbox_consumer", daemon=True).start()
    logger.info("[AppInbox] v0 consumer started (interval=%ds, observability-only — "
                "self-regulation is missions RFP Phase 4)", interval_s)
    return stop
