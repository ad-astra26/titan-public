"""SovereigntyRatioMeter — the headline metric (Synthesis Engine Phase 10, §1.2).

`sovereignty_ratio = recall_satisfied / knowledge_moments` over rolling windows.

- **knowledge_moment** = a chat turn that needed knowledge (≥1 retrieval surfaced
  OR ≥1 tool call).
- **recall_satisfied** = that moment was answered by memory rather than fresh LLM
  re-derivation: a delegated compiled skill (P8 `parent_skill_id`) OR ≥1 cited
  recall (`used_by_llm=True`, P9 strict gate INV-Syn-23).

Honest only because of P9's strict gate. INV-Syn-25: observation only — the meter
holds ephemeral, timestamped event marks; it is never a decision source of truth.

Self-contained + testable: the synthesis_worker records moments/satisfactions as
they happen (chat-turn boundary, skill delegation, cited recall); `compute(now_ts)`
buckets the marks into rolling windows and reports the ratio + a trend (this window
vs the prior equal-length window).
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
    """Rolling-window recall-vs-rederivation meter. Thread-safe; bounded memory."""

    def __init__(
        self,
        *,
        windows: Optional[list] = None,
        max_marks: int = 50000,
        clock=None,
        on_record: Optional[Callable[[str, float, Optional[str]], None]] = None,
    ) -> None:
        self._windows = list(windows) if windows else ["24h", "7d", "all"]
        import time as _time
        self._clock = clock or _time.time
        # Each mark is a timestamp. recall_satisfied marks also tagged by kind.
        self._knowledge: deque = deque(maxlen=max_marks)
        self._satisfied: deque = deque(maxlen=max_marks)   # (ts, kind)
        self._lock = threading.Lock()
        # G9 durable persistence (Phase F): an optional callback fired on every
        # NEW mark — `on_record(mark_type, ts, kind)`, mark_type ∈
        # {"knowledge","satisfied"}, kind = satisfied-kind or None. synthesis_worker
        # injects a SynthesisWriter INSERT into synthesis.duckdb::sovereignty_marks
        # (INV-Syn-28). Default None ⇒ the class stays pure (unit tests + non-synthesis
        # uses unaffected). Boot-seed replays with `_persist=False` so it never re-fires.
        self._on_record = on_record

    # ── recording ─────────────────────────────────────────────────────

    def record_knowledge_moment(
        self, ts: Optional[float] = None, *, _persist: bool = True,
    ) -> None:
        """A chat turn that needed knowledge (≥1 retrieval surfaced OR ≥1 tool call).

        `_persist=False` (boot-seed replay only) appends the mark WITHOUT firing
        the durable `on_record` callback — replaying durable rows must not re-write
        them. The callback fires OUTSIDE the lock (a slow SynthesisWriter submit
        must not stall recording)."""
        t = float(ts if ts is not None else self._clock())
        with self._lock:
            self._knowledge.append(t)
        if _persist and self._on_record is not None:
            try:
                self._on_record("knowledge", t, None)
            except Exception:
                pass

    def record_recall_satisfied(
        self, kind: str = "cited_recall", ts: Optional[float] = None,
        *, _persist: bool = True,
    ) -> None:
        """The moment was answered by memory. kind ∈ {'skill_delegation',
        'cited_recall', …} — open-ended (BRAIN Phase 12 adds 'brain_grounded').
        `_persist=False` = boot-seed replay (no re-fire of `on_record`)."""
        t = float(ts if ts is not None else self._clock())
        k = str(kind)
        with self._lock:
            self._satisfied.append((t, k))
        if _persist and self._on_record is not None:
            try:
                self._on_record("satisfied", t, k)
            except Exception:
                pass

    # ── compute ───────────────────────────────────────────────────────

    def compute(self, now_ts: Optional[float] = None) -> dict:
        """Return {window: {knowledge_moments, recall_satisfied, ratio,
        skill_delegations, cited_recalls, trend}} for each configured window."""
        now = float(now_ts if now_ts is not None else self._clock())
        with self._lock:
            knowledge = list(self._knowledge)
            satisfied = list(self._satisfied)
        out: dict = {}
        for w in self._windows:
            span = WINDOW_SECONDS.get(w)
            out[w] = self._window_stats(knowledge, satisfied, now, span)
        return out

    def _window_stats(self, knowledge, satisfied, now, span) -> dict:
        if span is None:
            lo, prev_lo = float("-inf"), None
        else:
            lo = now - span
            prev_lo = now - 2 * span
        km = sum(1 for t in knowledge if t >= lo)
        sat_marks = [(t, k) for (t, k) in satisfied if t >= lo]
        rs = len(sat_marks)
        skill_d = sum(1 for _, k in sat_marks if k == "skill_delegation")
        cited = sum(1 for _, k in sat_marks if k == "cited_recall")
        ratio = (rs / km) if km > 0 else 0.0
        ratio = min(1.0, ratio)  # satisfied can't exceed moments meaningfully
        trend = None
        if span is not None:
            prev_km = sum(1 for t in knowledge if prev_lo <= t < lo)
            prev_rs = sum(1 for (t, _) in satisfied if prev_lo <= t < lo)
            prev_ratio = (prev_rs / prev_km) if prev_km > 0 else 0.0
            trend = round(ratio - min(1.0, prev_ratio), 4)
        return {
            "knowledge_moments": km,
            "recall_satisfied": rs,
            "skill_delegations": skill_d,
            "cited_recalls": cited,
            "ratio": round(ratio, 4),
            "trend": trend,
        }


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
    Phase F's source is `synthesis.duckdb::sovereignty_marks(ts, mark_type, kind)`,
    written by the `on_record` callback. On boot we replay every in-window row:
    `mark_type='knowledge'` → `record_knowledge_moment(ts)`; `'satisfied'` →
    `record_recall_satisfied(kind, ts)` — with `_persist=False` so the replay
    never re-writes the rows it just read.

    `query_fn(since_ts, limit)` returns an iterable of `(ts, mark_type, kind)`
    rows; the caller supplies it as a **SynthesisWriter-serialized** read so this
    function never touches a DuckDB handle off the writer thread (INV-Syn-28).
    `since_ts` bounds the replay to the longest standard rolling window (the
    unbounded "all"-window full-journey aggregate is the genesis-spine's job,
    arch §18 TARGET); `cap` is surfaced when hit — no silent truncation. Never
    raises.

    Returns `{scanned, knowledge_moments, recall_satisfied, capped,
    window_since_ts}`.
    """
    out = {
        "scanned": 0,
        "knowledge_moments": 0,
        "recall_satisfied": 0,
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
            mark_type = str(r[1])
            kind = r[2]
        except Exception:
            continue
        out["scanned"] += 1
        try:
            if mark_type == "knowledge":
                meter.record_knowledge_moment(ts, _persist=False)
                out["knowledge_moments"] += 1
            elif mark_type == "satisfied":
                meter.record_recall_satisfied(
                    kind=str(kind) if kind is not None else "cited_recall",
                    ts=ts, _persist=False)
                out["recall_satisfied"] += 1
        except Exception:
            continue
    return out
