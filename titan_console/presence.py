"""Per-device presence for the Titan ⇄ App event channel.

RFP_titan_app_event_channel Phase 1. The phone reports its reachability on every
``POST /console/app/heartbeat`` — foreground/background, last-seen, battery — so a sender
knows whether a held long-poll exists or only the WorkManager cadence will deliver. Stored
at ``~/.titan/devices/<id>/presence.json``; sole-writer = the Console Agent (AG-EVT-1).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .events import _device_dir, _lock_for, _safe_id
from .pairing import _read_json, _titan_dir, _write_json, registered_device_ids

_STATES = {"foreground", "background", "active", "inactive"}
# The Maker's declared availability (RFP §7.3 3b). Transport-level only.
_AVAILABILITY = {"available", "busy", "dnd"}


def put(ctx, device_id: str, hb: dict) -> dict:
    """Record a heartbeat. Returns the stored presence record."""
    _safe_id(device_id)
    hb = hb if isinstance(hb, dict) else {}
    state = hb.get("state")
    # availability is the Maker's *declared* status (RFP §7.3 3b) — a transport signal,
    # NOT a coded gate on Titan's speech. Default "available". What "busy" *means* is
    # learned in the cognition (missions RFP P4), never an if-busy mute here.
    avail = hb.get("availability")
    until = hb.get("availability_until")
    rec = {
        "state": state if state in _STATES else "unknown",
        "last_seen": time.time(),
        "battery": hb.get("battery"),
        "ack_cursor": hb.get("ack_cursor"),
        "availability": avail if avail in _AVAILABILITY else "available",
        "availability_until": until if isinstance(until, (int, float)) else None,
    }
    _write_json(_device_dir(ctx, device_id) / "presence.json", rec)
    return rec


def get(ctx, device_id: str) -> dict | None:
    """The last heartbeat for a device, or None if it has never reported."""
    _safe_id(device_id)
    return _read_json(_device_dir(ctx, device_id) / "presence.json", None)


# ── Context / presence uplink (RFP_titan_mobile_app Phase 3, AG6 INV-OPT-IN) ───────────────
# The phone uploads where-the-Maker-is samples (location / dual-time / motion / battery) via
# POST /console/context. Each sample FIELD is gated by a per-sensor opt-in flag (default OFF —
# the one place a flag defaults off, per SPEC AG6). The flags live in a CONSOLE-LOCAL store
# (NOT config.toml): the kernel's config_schema validator is fail-closed on undeclared keys,
# so a phone toggle must never touch the kernel-validated config — the decoupled Console Agent
# owns its own presence gates. Sole-writer = the Console Agent (AG7). STOPS at persistence +
# readout — NO cognition (that is Phase 4, deliberately not wired here; AG8).

# Per-field opt-in → which sample keys it admits.
_SENSOR_FIELDS = {
    "location_enabled": ("lat", "lon", "accuracy"),
    "time_enabled": ("tz", "local_time"),
    "motion_enabled": ("motion",),
    "battery_enabled": ("battery",),
}
_DEFAULT_SETTINGS = {
    "location_enabled": False,
    "time_enabled": False,
    "motion_enabled": False,
    "battery_enabled": False,
    "cadence_minutes": 15,  # the app honors this for its background sample cadence
}


def _presence_dir(ctx) -> Path:
    d = _titan_dir(ctx) / "presence"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_settings(ctx) -> dict:
    """The per-sensor opt-in flags + cadence (defaults all-off, AG6)."""
    stored = _read_json(_presence_dir(ctx) / "settings.json", {})
    out = dict(_DEFAULT_SETTINGS)
    if isinstance(stored, dict):
        for k in _DEFAULT_SETTINGS:
            if k in stored:
                out[k] = stored[k]
    return out


def set_settings(ctx, patch: dict) -> dict:
    """Patch the opt-in flags (booleans) / cadence (int 1..1440). Returns the merged settings."""
    cur = get_settings(ctx)
    if isinstance(patch, dict):
        for k, v in patch.items():
            if k in ("location_enabled", "time_enabled", "motion_enabled", "battery_enabled"):
                cur[k] = bool(v)
            elif k == "cadence_minutes":
                try:
                    cur[k] = max(1, min(1440, int(v)))
                except (TypeError, ValueError):
                    pass
    _write_json(_presence_dir(ctx) / "settings.json", cur)
    return cur


def _gate_sample(sample: dict, settings: dict) -> dict | None:
    """Keep only fields whose sensor is opted-in (+ ts). None if nothing survives the gate."""
    if not isinstance(sample, dict):
        return None
    kept: dict = {}
    for flag, fields in _SENSOR_FIELDS.items():
        if settings.get(flag):
            for f in fields:
                if sample.get(f) is not None:
                    kept[f] = sample[f]
    if not kept:
        return None
    # ts is provenance, always retained (server clock if the sample omits it).
    ts = sample.get("ts")
    kept["ts"] = ts if isinstance(ts, (int, float)) else time.time()
    return kept


def ingest(ctx, device_id: str, samples) -> dict:
    """Persist opt-in-gated context samples for a device. Appends to the per-device
    ``context_log.jsonl`` (sole-writer, AG7) + rewrites ``context_latest.json``."""
    _safe_id(device_id)
    settings = get_settings(ctx)
    rows = samples if isinstance(samples, list) else []
    accepted = [g for g in (_gate_sample(s, settings) for s in rows) if g is not None]
    if not accepted:
        return {"accepted": 0, "device_id": device_id}
    d = _device_dir(ctx, device_id)
    d.mkdir(parents=True, exist_ok=True)
    with _lock_for(f"ctx:{device_id}"):
        with (d / "context_log.jsonl").open("a", encoding="utf-8") as fh:
            for g in accepted:
                fh.write(json.dumps(g, separators=(",", ":")) + "\n")
        _write_json(d / "context_latest.json", accepted[-1])
    return {"accepted": len(accepted), "device_id": device_id}


def read_latest(ctx) -> dict:
    """The newest context sample across all registered devices — 'where is the Maker'.
    Returns {} when no device has ever uploaded. No cognition; a flat readout (AG8)."""
    best: dict | None = None
    best_dev: str | None = None
    for dev in registered_device_ids(ctx):
        try:
            rec = _read_json(_device_dir(ctx, dev) / "context_latest.json", None)
        except ValueError:
            continue
        if isinstance(rec, dict) and rec.get("ts") is not None:
            if best is None or rec["ts"] > best.get("ts", 0):
                best, best_dev = rec, dev
    if best is None:
        return {}
    out = dict(best)
    out["device_id"] = best_dev
    return out
