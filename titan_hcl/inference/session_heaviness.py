"""titan_hcl/inference/session_heaviness.py — Phase C (session-aware routing).

RFP_load_adaptive_inference_routing §7.C. A process-local, O(1) tracker of per-session
"heaviness" — how deep/long/costly a conversation is — so the adaptive router keeps the
strongest model (gemma) for heavy sessions and offloads short/light ones first
(INV-AR-QUALITY-FOLLOWS-DEPTH). Read at the routing chokepoint, updated once per completed
turn in the PostHook. In-memory only (§Ad-4: stats_ref-style, NEVER an agno_sessions.db
query on the chat hot path — G18 / INV-AR-OFF-HOT-PATH). Leaf module (no titan_hcl imports)
so both agno_worker and agno_hooks can use it without a circular import.

The heaviness SCALAR ∈ [0,1] is a config-weighted blend of {turn_count, wall_duration,
token_volume, goal_class} normalized against config caps. Weights/caps live in
[inference.autoscale]; this module carries only sane defaults so it is a no-op-safe leaf.
"""
from __future__ import annotations

import time
from typing import Optional


class SessionHeaviness:
    """Per-session {turns, first_ts, last_ts, tokens}, TTL-evicted. All O(1)."""

    def __init__(self, ttl_s: float = 3600.0, max_sessions: int = 4096) -> None:
        self._ttl_s = float(ttl_s)
        self._max = int(max_sessions)
        self._d: dict[str, dict] = {}

    def note_turn(self, session_id: str, tokens: int, now: Optional[float] = None) -> None:
        """Fold one completed turn into the session's running stats."""
        if not session_id:
            return
        t = float(now if now is not None else time.time())
        rec = self._d.get(session_id)
        if rec is None:
            # opportunistic TTL sweep only when we grow (bounded, amortized O(1))
            if len(self._d) >= self._max:
                self._evict(t)
            rec = {"turns": 0, "first_ts": t, "last_ts": t, "tokens": 0}
            self._d[session_id] = rec
        rec["turns"] += 1
        rec["last_ts"] = t
        rec["tokens"] += max(0, int(tokens))

    def raw(self, session_id: str, now: Optional[float] = None) -> tuple[int, float, int]:
        """(turns_so_far, wall_duration_s, cumulative_tokens) for `session_id`.
        A session gone stale past TTL reads as fresh (0,0,0) — a returning user after
        a long gap is NOT treated as a deep session."""
        rec = self._d.get(session_id)
        if rec is None:
            return (0, 0.0, 0)
        t = float(now if now is not None else time.time())
        if (t - float(rec["last_ts"])) > self._ttl_s:
            return (0, 0.0, 0)
        return (int(rec["turns"]), max(0.0, t - float(rec["first_ts"])), int(rec["tokens"]))

    def _evict(self, now: float) -> None:
        stale = [s for s, r in self._d.items()
                 if (now - float(r["last_ts"])) > self._ttl_s]
        for s in stale:
            self._d.pop(s, None)
        # if still over cap after TTL sweep, drop the oldest-touched sessions
        if len(self._d) >= self._max:
            for s, _ in sorted(self._d.items(),
                               key=lambda kv: kv[1]["last_ts"])[: self._max // 8 + 1]:
                self._d.pop(s, None)


def _norm(x: float, cap: float) -> float:
    return 0.0 if cap <= 0 else max(0.0, min(1.0, float(x) / float(cap)))


def compute_heaviness(turns: int, duration_s: float, tokens: int,
                      goal_class: str, cfg: Optional[dict] = None) -> float:
    """Blend the raw session signals into a heaviness scalar ∈ [0,1]. Each component
    is normalized against a config cap then weighted; goal_class adds an optional
    per-class bump (config map, default none). Weights need not sum to 1 — the result
    is clamped. All knobs live in [inference.autoscale]; defaults make it sensible OOTB."""
    c = cfg or {}

    def _f(k, d):
        try:
            return float(c.get(k, d))
        except Exception:
            return d

    turn_n = _norm(turns, _f("heaviness_turns_cap", 8.0))
    dur_n = _norm(duration_s, _f("heaviness_duration_cap_s", 600.0))
    tok_n = _norm(tokens, _f("heaviness_tokens_cap", 6000.0))
    gc_map = c.get("heaviness_goal_class_weights", {}) or {}
    try:
        gc = max(0.0, min(1.0, float(gc_map.get(str(goal_class or ""), 0.0))))
    except Exception:
        gc = 0.0
    w_turn = _f("heaviness_w_turns", 0.4)
    w_dur = _f("heaviness_w_duration", 0.25)
    w_tok = _f("heaviness_w_tokens", 0.25)
    w_gc = _f("heaviness_w_goal_class", 0.1)
    score = w_turn * turn_n + w_dur * dur_n + w_tok * tok_n + w_gc * gc
    return max(0.0, min(1.0, score))


# Process-global singleton (one agno_worker process = one tracker).
_TRACKER = SessionHeaviness()


def note_turn(session_id: str, tokens: int, now: Optional[float] = None) -> None:
    _TRACKER.note_turn(session_id, tokens, now)


def raw(session_id: str, now: Optional[float] = None) -> tuple[int, float, int]:
    return _TRACKER.raw(session_id, now)
