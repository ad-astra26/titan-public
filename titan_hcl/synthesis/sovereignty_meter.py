"""SovereigntyRatioMeter — the headline sovereignty metric (RFP_synthesis_decision_authority P3).

Rolling-window aggregator of the ONE per-reply sovereignty score
`S = 0.7·E + 0.3·V` (see `synthesis/sovereignty_score.py`). Each completed chat
turn contributes one `S_reply ∈ [0,1]`; this meter holds the timestamped marks
and reports the **rolling-mean S** per window (24h / 7d / all) plus a trend
(this window vs the prior equal-length window).

History: this replaces the Phase-10 `recall_satisfied / knowledge_moments` count
ratio — the RFP collapses the four disagreeing sovereignty scores into this one
metric. The class name + the durable `on_record`/boot-seed plumbing are
**reused** (RFP §7.P3 "reuse the plumbing, swap the formula"); only the recorded
quantity changes (discrete moment/satisfied marks → a continuous per-reply S).

INV-SDA-3: one sovereignty metric. INV-Syn-25: observation only — the meter holds
ephemeral, timestamped marks rebuildable from the durable
`synthesis.duckdb::sovereignty_marks` source; it is never a decision source of
truth. The *computation* of S is the cheap pure `compute_sovereignty_score`
(agno-side, so the reply's TIMECHAIN_COMMIT can anchor it); the meter only
records + aggregates (synthesis-side, off the hot path — INV-SDA-11).
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Callable, Optional

__all__ = [
    "SovereigntyRatioMeter",
    "WINDOW_SECONDS",
    "boot_seed_from_marks",
]

WINDOW_SECONDS = {
    "24h": 24 * 3600,
    "7d": 7 * 24 * 3600,
    "all": None,  # unbounded
}


class SovereigntyRatioMeter:
    """Rolling-window mean of the per-reply sovereignty score S. Thread-safe."""

    def __init__(
        self,
        *,
        windows: Optional[list] = None,
        max_marks: int = 50000,
        clock=None,
        on_record: Optional[Callable[[float, float], None]] = None,
    ) -> None:
        self._windows = list(windows) if windows else ["24h", "7d", "all"]
        import time as _time
        self._clock = clock or _time.time
        # Each mark is (ts, s) — one completed reply's sovereignty score.
        self._marks: deque = deque(maxlen=max_marks)
        self._lock = threading.Lock()
        # G9 durable persistence: an optional callback fired on every NEW mark —
        # `on_record(ts, s)`. synthesis_worker injects a SynthesisWriter INSERT
        # into synthesis.duckdb::sovereignty_marks (INV-Syn-28). Default None ⇒
        # the class stays pure (unit tests unaffected). Boot-seed replays with
        # `_persist=False` so it never re-fires.
        self._on_record = on_record

    # ── recording ─────────────────────────────────────────────────────

    def record_reply(
        self, s: float, ts: Optional[float] = None, *, _persist: bool = True,
    ) -> None:
        """Record one completed reply's sovereignty score `s ∈ [0,1]`.

        `_persist=False` (boot-seed replay only) appends the mark WITHOUT firing
        the durable `on_record` callback — replaying durable rows must not
        re-write them. The callback fires OUTSIDE the lock (a slow SynthesisWriter
        submit must not stall recording)."""
        t = float(ts if ts is not None else self._clock())
        sv = _clamp01(s)
        with self._lock:
            self._marks.append((t, sv))
        if _persist and self._on_record is not None:
            try:
                self._on_record(t, sv)
            except Exception:
                pass

    # ── compute ───────────────────────────────────────────────────────

    def compute(self, now_ts: Optional[float] = None) -> dict:
        """Return {window: {replies, sovereignty, trend}} for each window.

        `sovereignty` = rolling-mean S over the window (0.0 when no replies);
        `trend` = this window's mean minus the prior equal-length window's mean
        (None for the unbounded 'all' window)."""
        now = float(now_ts if now_ts is not None else self._clock())
        with self._lock:
            marks = list(self._marks)
        out: dict = {}
        for w in self._windows:
            span = WINDOW_SECONDS.get(w)
            out[w] = self._window_stats(marks, now, span)
        return out

    def _window_stats(self, marks, now, span) -> dict:
        if span is None:
            lo, prev_lo = float("-inf"), None
        else:
            lo = now - span
            prev_lo = now - 2 * span
        cur = [s for (t, s) in marks if t >= lo]
        replies = len(cur)
        mean_s = (sum(cur) / replies) if replies else 0.0
        trend = None
        if span is not None:
            prev = [s for (t, s) in marks if prev_lo <= t < lo]
            prev_mean = (sum(prev) / len(prev)) if prev else 0.0
            trend = round(mean_s - prev_mean, 4)
        return {
            "replies": replies,
            "sovereignty": round(mean_s, 4),
            "trend": trend,
        }


def _clamp01(x: float) -> float:
    try:
        x = float(x)
    except Exception:
        return 0.0
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def boot_seed_from_marks(
    meter: "SovereigntyRatioMeter",
    query_fn: Callable[[float, int], object],
    *,
    since_ts: float,
    cap: int = 50000,
) -> dict:
    """Reseed `meter` from the durable `sovereignty_marks` store (G9 / INV-Syn-25).

    The meter holds ephemeral rolling-window marks that a `synthesis_worker`
    respawn zeros (crash-loop audit §5.3 "respawn zeros windows"). INV-Syn-25
    makes metrics a rebuildable projection over a **canonical durable source**;
    the source is `synthesis.duckdb::sovereignty_marks(ts, s)`, written by the
    `on_record` callback. On boot we replay every in-window `(ts, s)` row via
    `record_reply(s, ts, _persist=False)` so the replay never re-writes the rows.

    `query_fn(since_ts, limit)` returns an iterable of `(ts, s)` rows; the caller
    supplies it as a **SynthesisWriter-serialized** read (INV-Syn-28). `since_ts`
    bounds the replay to the longest standard rolling window; `cap` is surfaced
    when hit — no silent truncation. Never raises.

    Returns `{scanned, replies, capped, window_since_ts}`.
    """
    out = {
        "scanned": 0,
        "replies": 0,
        "capped": False,
        "window_since_ts": float(since_ts),
    }
    try:
        rows = list(query_fn(float(since_ts), int(cap) + 1))
    except Exception:
        return out
    if len(rows) > cap:
        out["capped"] = True
        rows = rows[:cap]
    for r in rows:
        try:
            ts = float(r[0])
            s = float(r[1])
        except Exception:
            continue
        out["scanned"] += 1
        try:
            meter.record_reply(s, ts, _persist=False)
            out["replies"] += 1
        except Exception:
            continue
    return out
