"""Per-device outbound event queue for the Titan ⇄ App channel.

RFP_titan_app_event_channel Phase 1. The Console Agent owns a durable, sole-writer
per-device queue (AG-EVT-1). Anything that needs to reach a paired phone — Titan's
cognition (an internal-key ``/console/events/enqueue`` call) or the Console Agent
itself (``HealthMonitor``) — appends an event; the phone drains it over a held
long-poll (``GET /console/events``) and acks a monotonic cursor so each event is
delivered exactly once (AG-EVT-2). The queue lives entirely in the Console Agent,
so it works while the kernel is down (AG-EVT-3 / AG2).

Storage (under ``~/.titan/devices/<device_id>/``; redirected by ``ctx.secrets_path``
in tests because the path comes from ``pairing._titan_dir``):
  outbox.jsonl   one JSON event per line — appended on enqueue, rewritten-minus-acked
                 on ack. Atomic single-writer (AG-EVT-1).
  cursor.json    {"next_seq": int} — the monotonic sequence allocator. It never resets
                 on prune, so a phone's cursor stays valid after the outbox empties.

Concurrency: the Console Agent is a ``ThreadingHTTPServer`` — enqueue/ack/drain may be
called on different request threads at once. A per-device lock serializes writes; a
per-device ``threading.Event`` wakes a held long-poll the instant an event lands. The
held wait runs on its own request thread, so it never blocks other console calls.
"""
from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

from .pairing import _find_device, _read_json, _titan_dir, _write_json

# A drain holds a request thread for at most this long (the route clamps ``wait`` to it).
_MAX_WAIT_S = 25

# device_id is filesystem-bound (it names a directory) → strict allowlist, no '.'/'/'
# so a phone-supplied id can never traverse out of ~/.titan/devices/.
_DEVICE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# Per-device concurrency primitives, created on demand under the registry lock.
_registry_lock = threading.Lock()
_locks: dict[str, threading.Lock] = {}
_waiters: dict[str, threading.Event] = {}


def _safe_id(device_id: str) -> str:
    """Return ``device_id`` iff it is a safe directory name, else raise (path-traversal guard)."""
    if not isinstance(device_id, str) or not _DEVICE_ID_RE.match(device_id):
        raise ValueError(f"invalid device_id: {device_id!r}")
    return device_id


def _device_dir(ctx, device_id: str) -> Path:
    return _titan_dir(ctx) / "devices" / _safe_id(device_id)


def _lock_for(sid: str) -> threading.Lock:
    with _registry_lock:
        lk = _locks.get(sid)
        if lk is None:
            lk = _locks[sid] = threading.Lock()
        return lk


def _waiter_for(sid: str) -> threading.Event:
    with _registry_lock:
        ev = _waiters.get(sid)
        if ev is None:
            ev = _waiters[sid] = threading.Event()
        return ev


def _read_events(outbox: Path) -> list:
    out: list = []
    try:
        with open(outbox) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue  # tolerate a torn tail line; never lose the rest
    except OSError:
        pass
    return out


def _rewrite(outbox: Path, events: list) -> None:
    outbox.parent.mkdir(parents=True, exist_ok=True)
    tmp = outbox.with_suffix(outbox.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(e) + "\n" for e in events))
    tmp.replace(outbox)  # atomic single-writer (AG-EVT-1)


def enqueue(ctx, device_id: str, *, type: str, payload=None,
            urgency: str = "normal", dedupe_key: str | None = None) -> dict:
    """Append an event to a device's queue and wake any held long-poll. Returns the
    stored event. A ``dedupe_key`` that matches an already-pending (un-acked) event is a
    no-op returning that event — no double delivery (AG-EVT-2)."""
    sid = _safe_id(device_id)
    waiter = _waiter_for(sid)
    with _lock_for(sid):
        d = _device_dir(ctx, device_id)
        d.mkdir(parents=True, exist_ok=True)
        outbox = d / "outbox.jsonl"
        if dedupe_key:
            for e in _read_events(outbox):
                if e.get("dedupe_key") == dedupe_key:
                    return e
        cur = _read_json(d / "cursor.json", {"next_seq": 1})
        seq = int(cur.get("next_seq", 1))
        cur["next_seq"] = seq + 1
        # Reserve the seq BEFORE appending: a crash here leaves a harmless gap; the
        # reverse order could reuse a seq and silently merge two events.
        _write_json(d / "cursor.json", cur)
        evt = {"seq": seq, "type": type, "payload": payload,
               "urgency": urgency, "ts": time.time(), "dedupe_key": dedupe_key}
        with open(outbox, "a") as f:
            f.write(json.dumps(evt) + "\n")
    waiter.set()
    return evt


def pending(ctx, device_id: str, since: int) -> tuple[list, int]:
    """Events with ``seq > since`` plus the new cursor (highest seq the phone now holds)."""
    _safe_id(device_id)
    events = _read_events(_device_dir(ctx, device_id) / "outbox.jsonl")
    fresh = [e for e in events if int(e.get("seq", 0)) > since]
    cursor = since
    for e in fresh:
        cursor = max(cursor, int(e.get("seq", 0)))
    return fresh, cursor


def drain(ctx, device_id: str, since: int, wait_s: int) -> tuple[list, int]:
    """Long-poll drain: return pending events immediately, else hold up to ``wait_s``
    (clamped to ``_MAX_WAIT_S``) for an enqueue, then return. ``wait_s<=0`` = instant
    drain (what the WorkManager periodic job uses)."""
    sid = _safe_id(device_id)
    waiter = _waiter_for(sid)
    waiter.clear()  # clear before reading → no lost wakeup if an enqueue races the wait
    fresh, cursor = pending(ctx, device_id, since)
    if fresh or wait_s <= 0:
        return fresh, cursor
    waiter.wait(timeout=min(wait_s, _MAX_WAIT_S))
    return pending(ctx, device_id, since)


def ack(ctx, device_id: str, cursor: int) -> int:
    """Prune every delivered event (``seq <= cursor``). Returns how many were pruned."""
    sid = _safe_id(device_id)
    with _lock_for(sid):
        outbox = _device_dir(ctx, device_id) / "outbox.jsonl"
        events = _read_events(outbox)
        keep = [e for e in events if int(e.get("seq", 0)) > cursor]
        pruned = len(events) - len(keep)
        if pruned:
            _rewrite(outbox, keep)
    return pruned


def _reset_registries() -> None:
    """Test hygiene — drop the per-device locks/waiters (process-global otherwise)."""
    with _registry_lock:
        _locks.clear()
        _waiters.clear()
