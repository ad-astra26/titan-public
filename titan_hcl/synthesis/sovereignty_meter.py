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
from typing import Optional

__all__ = [
    "SovereigntyRatioMeter",
    "WINDOW_SECONDS",
    "boot_seed_from_conversation_chain",
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
    ) -> None:
        self._windows = list(windows) if windows else ["24h", "7d", "all"]
        import time as _time
        self._clock = clock or _time.time
        # Each mark is a timestamp. recall_satisfied marks also tagged by kind.
        self._knowledge: deque = deque(maxlen=max_marks)
        self._satisfied: deque = deque(maxlen=max_marks)   # (ts, kind)
        self._lock = threading.Lock()

    # ── recording ─────────────────────────────────────────────────────

    def record_knowledge_moment(self, ts: Optional[float] = None) -> None:
        """A chat turn that needed knowledge (≥1 retrieval surfaced OR ≥1 tool call)."""
        with self._lock:
            self._knowledge.append(float(ts if ts is not None else self._clock()))

    def record_recall_satisfied(
        self, kind: str = "cited_recall", ts: Optional[float] = None,
    ) -> None:
        """The moment was answered by memory. kind ∈ {'skill_delegation','cited_recall'}."""
        with self._lock:
            self._satisfied.append(
                (float(ts if ts is not None else self._clock()), str(kind)))

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


def boot_seed_from_conversation_chain(
    meter: "SovereigntyRatioMeter",
    *,
    data_dir: str,
    since_ts: float,
    cap: int = 5000,
    index_db_path: Optional[str] = None,
) -> dict:
    """Reseed `meter` from in-window conversation-fork TXs (G9 / INV-Syn-25).

    The meter holds ephemeral rolling-window marks that a worker respawn zeros
    (the crash-loop audit §5.3 "respawn zeros windows" gap). INV-Syn-25 makes
    metrics a **rebuildable projection over the Timechain**, so on boot we
    replay the per-turn `sovereignty{needed,satisfied}` signal that arch §7
    conversation-fork TXs now carry: `needed` → `record_knowledge_moment(ts)`;
    `satisfied` → `record_recall_satisfied('cited_recall', ts)`.

    Reads the conversation fork (resolved BY NAME per INV-Syn-26 — id is
    chain-local) from the read-only `block_index` + chain `.bin` payload. The
    conversation fork is ~one block per turn (the batched flood is the *episodic*
    fork), so each block's content IS the per-turn TX; any batch envelope that
    lacks a top-level `sovereignty` dict is skipped. Bounded by `since_ts` (the
    longest standard rolling window — the unbounded "all" window's full-journey
    accuracy is the genesis-spine aggregate's job, arch §18 TARGET) and `cap`
    (logged when hit — no silent truncation). Read-only; never raises.

    Returns `{scanned, knowledge_moments, recall_satisfied, capped,
    window_since_ts}`.
    """
    import os
    import sqlite3
    from pathlib import Path

    out = {
        "scanned": 0,
        "knowledge_moments": 0,
        "recall_satisfied": 0,
        "capped": False,
        "window_since_ts": float(since_ts),
    }
    path = index_db_path or os.path.join(data_dir, "timechain", "index.db")
    if not os.path.exists(path):
        return out
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5.0)
        conn.row_factory = sqlite3.Row
    except Exception:
        return out
    try:
        rows = list(conn.execute(
            "SELECT bi.block_hash, bi.fork_id, bi.file_offset, bi.timestamp "
            "FROM block_index bi "
            "JOIN fork_registry fr ON bi.fork_id = fr.fork_id "
            "WHERE fr.fork_name = 'conversation' "
            "  AND bi.thought_type = 'conversation' "
            "  AND bi.timestamp > ? "
            "ORDER BY bi.timestamp ASC LIMIT ?",
            [float(since_ts), int(cap) + 1],
        ))
    except sqlite3.Error:
        rows = []
    finally:
        try:
            conn.close()
        except Exception:
            pass

    if len(rows) > cap:
        out["capped"] = True
        rows = rows[:cap]
    if not rows:
        return out

    try:
        from titan_hcl.synthesis.chain_reader import read_block_content_at
    except Exception:
        return out

    ddir = Path(data_dir)
    for r in rows:
        out["scanned"] += 1
        try:
            content = read_block_content_at(
                ddir, int(r["fork_id"]), int(r["file_offset"]))
        except Exception:
            continue
        if not isinstance(content, dict):
            continue
        sov = content.get("sovereignty")
        if not isinstance(sov, dict):
            continue
        ts = float(r["timestamp"])
        try:
            if sov.get("needed"):
                meter.record_knowledge_moment(ts)
                out["knowledge_moments"] += 1
            if sov.get("satisfied"):
                meter.record_recall_satisfied(kind="cited_recall", ts=ts)
                out["recall_satisfied"] += 1
        except Exception:
            continue
    return out
