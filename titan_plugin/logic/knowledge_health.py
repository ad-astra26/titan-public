"""
knowledge_health — per-backend health tracker for the knowledge pipeline.

Owns four concerns, all persisted across knowledge_worker restart:

  1. Circuit breaker per backend (Essential B)
     Three states — closed / open / half_open — with N consecutive-error
     threshold + M-second cooldown. On success in half-open → closed.
     On failure in half-open → re-open.

  2. Per-backend daily bandwidth budget + rolling counters (Essential C)
     Requests + errors + bytes_consumed per backend, scoped to UTC day,
     survive process restart by reloading from disk. Explicit reset via
     /v4/search-pipeline/budget-reset (KP-5) or Maker override.

  3. Decision log (Essential D)
     data/logs/knowledge_router_decisions.jsonl — rolling JSONL with one
     line per dispatch() decision. Auto-rotates at size cap. Feeds KP-8's
     smart routing learning + arch_map analytics.

  4. Near-duplicate detection (Optional E)
     Rolling window of recent normalized queries. On cache miss, compare
     Jaccard similarity; WARN when > 0.8 against a non-matching hash so
     we see whether callers send semantically-same queries with
     punctuation / stopword differences.

Persistence: health state goes to data/knowledge_pipeline_health.json via
atomic write (write-tmp-then-rename). Load at __init__ restores counters
+ circuit state. Decision log is append-only.

Per rFP_knowledge_pipeline_v2.md §3.3 + §4.5 + KP-4 essentials.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ── Defaults ─────────────────────────────────────────────────────────

DEFAULT_CB_FAIL_THRESHOLD = 5          # consecutive errors to open circuit
DEFAULT_CB_COOLDOWN_SECONDS = 300      # 5 min before half-open probe
DEFAULT_DECISION_LOG_MAX_BYTES = 50 * 1024 * 1024   # 50 MB rotate
DEFAULT_DECISION_LOG_RETENTION = 2     # keep .1 and .2 rotations
DEFAULT_NEAR_DUP_WINDOW = 20           # last 20 normalized queries tracked
DEFAULT_NEAR_DUP_JACCARD = 0.8         # warn above this similarity

# KP-7 — alert thresholds + kinds
BUDGET_WARNING_PCT = 0.80              # fire budget_warning at 80% consumed
ALERT_KIND_BUDGET_WARNING = "budget_warning"
ALERT_KIND_BUDGET_EXCEEDED = "budget_exceeded"
ALERT_KIND_CIRCUIT_OPEN = "circuit_open"

# Callback signature: (kind: str, backend: str, ctx: dict) -> None
AlertCallback = Callable[[str, str, dict], None]


# ── Circuit breaker states ───────────────────────────────────────────

CIRCUIT_CLOSED = "closed"
CIRCUIT_OPEN = "open"
CIRCUIT_HALF_OPEN = "half_open"


# ── Per-backend health record ────────────────────────────────────────

@dataclass
class BackendHealth:
    """Mutable per-backend health state. Serialized to health.json."""
    name: str
    circuit_state: str = CIRCUIT_CLOSED
    consecutive_errors: int = 0
    circuit_opened_ts: float = 0.0
    last_success_ts: float = 0.0
    last_error_ts: float = 0.0
    last_error_type: str = ""
    requests_today: int = 0
    errors_today: int = 0
    bytes_consumed_today: int = 0
    budget_daily_bytes: int = 0            # 0 = unlimited
    avg_latency_ms: float = 0.0
    counter_day_epoch: int = 0
    total_requests_lifetime: int = 0
    total_errors_lifetime: int = 0
    # KP-7 — alerts fired today (dedup set, resets at day rollover)
    alerted_today: Set[str] = field(default_factory=set)

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()}
        # Sets don't serialize to JSON — encode as list
        d["alerted_today"] = sorted(list(self.alerted_today))
        return d

    @classmethod
    def from_dict(cls, name: str, d: dict) -> "BackendHealth":
        return cls(
            name=name,
            circuit_state=d.get("circuit_state", CIRCUIT_CLOSED),
            consecutive_errors=int(d.get("consecutive_errors", 0)),
            circuit_opened_ts=float(d.get("circuit_opened_ts", 0.0)),
            last_success_ts=float(d.get("last_success_ts", 0.0)),
            last_error_ts=float(d.get("last_error_ts", 0.0)),
            last_error_type=d.get("last_error_type", ""),
            requests_today=int(d.get("requests_today", 0)),
            errors_today=int(d.get("errors_today", 0)),
            bytes_consumed_today=int(d.get("bytes_consumed_today", 0)),
            budget_daily_bytes=int(d.get("budget_daily_bytes", 0)),
            avg_latency_ms=float(d.get("avg_latency_ms", 0.0)),
            counter_day_epoch=int(d.get("counter_day_epoch", 0)),
            total_requests_lifetime=int(d.get("total_requests_lifetime", 0)),
            total_errors_lifetime=int(d.get("total_errors_lifetime", 0)),
            alerted_today=set(d.get("alerted_today", []) or []),
        )


# ── Tracker ──────────────────────────────────────────────────────────

class HealthTracker:
    """Thread-safe health tracker + decision logger + near-dup detector.

    Single instance shared by every dispatch() call (passed via dispatcher
    kwarg). knowledge_worker instantiates one at boot and keeps it alive
    for the subprocess lifetime; WebSearchHelper instantiates its own
    (separate process in titan_main). Both share data/knowledge_pipeline
    _health.json via atomic writes — last-writer-wins is acceptable since
    each process is the authoritative counter for its own backends.
    """

    def __init__(self,
                 health_path: str = "data/knowledge_pipeline_health.json",
                 decision_log_path: str = "data/logs/knowledge_router_decisions.jsonl",
                 budgets: Optional[Dict[str, int]] = None,
                 cb_fail_threshold: int = DEFAULT_CB_FAIL_THRESHOLD,
                 cb_cooldown_seconds: int = DEFAULT_CB_COOLDOWN_SECONDS,
                 decision_log_max_bytes: int = DEFAULT_DECISION_LOG_MAX_BYTES,
                 near_dup_window: int = DEFAULT_NEAR_DUP_WINDOW,
                 near_dup_jaccard: float = DEFAULT_NEAR_DUP_JACCARD,
                 on_alert: Optional[AlertCallback] = None):
        self.health_path = health_path
        self.decision_log_path = decision_log_path
        self.cb_fail_threshold = int(cb_fail_threshold)
        self.cb_cooldown_seconds = int(cb_cooldown_seconds)
        self.decision_log_max_bytes = int(decision_log_max_bytes)
        self.near_dup_window = int(near_dup_window)
        self.near_dup_jaccard = float(near_dup_jaccard)
        self._default_budgets = dict(budgets or {})
        self._on_alert = on_alert       # KP-7 alert cascade callback

        self._lock = Lock()
        self._backends: Dict[str, BackendHealth] = {}
        self._recent_queries: deque = deque(maxlen=self.near_dup_window)
        self._load()

        # Ensure decision-log directory exists
        os.makedirs(os.path.dirname(self.decision_log_path) or ".",
                    exist_ok=True)

    # ── Day-epoch helpers ───────────────────────────────────────────

    @staticmethod
    def _current_day_epoch() -> int:
        return int(time.time() // 86400)

    def _maybe_reset_daily(self, h: BackendHealth) -> None:
        """Reset daily counters if the UTC day has rolled over."""
        today = self._current_day_epoch()
        if h.counter_day_epoch != today:
            h.counter_day_epoch = today
            h.requests_today = 0
            h.errors_today = 0
            h.bytes_consumed_today = 0
            h.alerted_today = set()   # fresh day → fresh alerts

    def _ensure_backend(self, name: str) -> BackendHealth:
        h = self._backends.get(name)
        if h is None:
            h = BackendHealth(
                name=name,
                budget_daily_bytes=int(self._default_budgets.get(name, 0)),
                counter_day_epoch=self._current_day_epoch(),
            )
            self._backends[name] = h
        self._maybe_reset_daily(h)
        return h

    # ── Persistence ─────────────────────────────────────────────────

    def _load(self) -> None:
        """Read health.json if present; tolerate missing/malformed file."""
        try:
            if not os.path.exists(self.health_path):
                return
            with open(self.health_path, "r") as f:
                data = json.load(f)
            for name, d in (data.get("backends") or {}).items():
                h = BackendHealth.from_dict(name, d)
                # Respect in-process budget defaults if saved budget is 0
                if h.budget_daily_bytes == 0:
                    h.budget_daily_bytes = int(
                        self._default_budgets.get(name, 0))
                self._maybe_reset_daily(h)
                self._backends[name] = h
            logger.info("[KnowledgeHealth] Loaded %d backends from %s",
                        len(self._backends), self.health_path)
        except Exception as e:
            logger.warning("[KnowledgeHealth] Load failed (%s): %s — "
                           "starting fresh", self.health_path, e)

    def _save(self) -> None:
        """Atomic write: tmp-then-rename. Caller must hold self._lock."""
        try:
            os.makedirs(os.path.dirname(self.health_path) or ".",
                        exist_ok=True)
            tmp = self.health_path + ".tmp"
            payload = {
                "ts": time.time(),
                "backends": {
                    name: h.to_dict() for name, h in self._backends.items()},
            }
            with open(tmp, "w") as f:
                json.dump(payload, f, default=str)
            os.replace(tmp, self.health_path)
        except Exception as e:
            logger.warning("[KnowledgeHealth] Save failed: %s", e)

    # ── Circuit breaker ─────────────────────────────────────────────

    def should_attempt(self, backend_name: str) -> bool:
        """True if dispatcher should attempt this backend right now.

        Returns False when circuit is OPEN and cooldown hasn't elapsed.
        Transitions OPEN → HALF_OPEN once cooldown passes; returns True
        to allow the probe request.
        """
        now = time.time()
        with self._lock:
            h = self._ensure_backend(backend_name)
            if h.circuit_state == CIRCUIT_CLOSED:
                return True
            if h.circuit_state == CIRCUIT_HALF_OPEN:
                # Caller is already probing; don't allow a second concurrent
                return True
            # OPEN — check cooldown
            elapsed = now - h.circuit_opened_ts
            if elapsed >= self.cb_cooldown_seconds:
                h.circuit_state = CIRCUIT_HALF_OPEN
                logger.info(
                    "[KnowledgeHealth] Circuit %s: OPEN → HALF_OPEN "
                    "after %.0fs cooldown", backend_name, elapsed)
                self._save()
                return True
            return False

    # ── Budget ──────────────────────────────────────────────────────

    def check_budget(self, backend_name: str) -> bool:
        """True if backend still has daily budget available."""
        with self._lock:
            h = self._ensure_backend(backend_name)
            if h.budget_daily_bytes <= 0:
                return True  # 0 = unlimited
            return h.bytes_consumed_today < h.budget_daily_bytes

    def set_budget(self, backend_name: str, daily_bytes: int) -> None:
        """Runtime budget adjustment (e.g. from /v4 API or config reload)."""
        with self._lock:
            h = self._ensure_backend(backend_name)
            h.budget_daily_bytes = int(daily_bytes)
            self._save()

    def reset_budget(self, backend_name: Optional[str] = None) -> None:
        """Manual reset of today's counter — Maker override."""
        with self._lock:
            targets = ([backend_name] if backend_name
                       else list(self._backends.keys()))
            for name in targets:
                h = self._ensure_backend(name)
                h.requests_today = 0
                h.errors_today = 0
                h.bytes_consumed_today = 0
            self._save()
        logger.info("[KnowledgeHealth] Budget reset: %s",
                    backend_name or "ALL backends")

    # ── Recording ───────────────────────────────────────────────────

    def record_attempt(self, backend_name: str, *,
                        success: bool, error_type: str = "",
                        bytes_consumed: int = 0,
                        latency_ms: float = 0.0) -> None:
        """Update counters + circuit state for one completed attempt.

        error_type="" means success. Circuit state transitions:
          * closed + success → closed (errors counter reset)
          * closed + failure → consecutive_errors++
                               → if >= threshold → OPEN  (fires alert)
          * half_open + success → closed (recovery)
          * half_open + failure → OPEN (cooldown restarts, fires alert)

        Budget alerts (KP-7):
          * Crosses 80% of budget_daily_bytes → fires budget_warning once
          * Crosses 100% → fires budget_exceeded once (pipeline-blocking)
        """
        now = time.time()
        alerts_to_fire: List[tuple] = []   # (kind, ctx) collected under lock,
                                            # fired AFTER release to avoid
                                            # deadlocks if callback calls back
        with self._lock:
            h = self._ensure_backend(backend_name)
            h.requests_today += 1
            h.total_requests_lifetime += 1

            # Budget threshold detection BEFORE adding new bytes — lets us
            # detect the exact crossing.
            budget = h.budget_daily_bytes
            pct_before = (h.bytes_consumed_today / budget) if budget > 0 else 0.0
            h.bytes_consumed_today += int(bytes_consumed)
            pct_after = (h.bytes_consumed_today / budget) if budget > 0 else 0.0

            if budget > 0:
                if (pct_after >= 1.0 and pct_before < 1.0
                        and ALERT_KIND_BUDGET_EXCEEDED not in h.alerted_today):
                    h.alerted_today.add(ALERT_KIND_BUDGET_EXCEEDED)
                    alerts_to_fire.append((ALERT_KIND_BUDGET_EXCEEDED, {
                        "backend": backend_name,
                        "bytes_consumed": h.bytes_consumed_today,
                        "budget_bytes": budget,
                        "pct": round(pct_after * 100, 1),
                    }))
                elif (pct_after >= BUDGET_WARNING_PCT
                      and pct_before < BUDGET_WARNING_PCT
                      and ALERT_KIND_BUDGET_WARNING not in h.alerted_today):
                    h.alerted_today.add(ALERT_KIND_BUDGET_WARNING)
                    alerts_to_fire.append((ALERT_KIND_BUDGET_WARNING, {
                        "backend": backend_name,
                        "bytes_consumed": h.bytes_consumed_today,
                        "budget_bytes": budget,
                        "pct": round(pct_after * 100, 1),
                    }))

            # Latency EMA (alpha=0.2 — responsive to recent)
            if latency_ms > 0:
                if h.avg_latency_ms == 0:
                    h.avg_latency_ms = latency_ms
                else:
                    h.avg_latency_ms = 0.8 * h.avg_latency_ms + 0.2 * latency_ms

            # Circuit state transitions
            prev_state = h.circuit_state
            if success:
                h.last_success_ts = now
                if h.circuit_state == CIRCUIT_HALF_OPEN:
                    logger.info(
                        "[KnowledgeHealth] Circuit %s: HALF_OPEN → CLOSED "
                        "(recovery)", backend_name)
                h.circuit_state = CIRCUIT_CLOSED
                h.consecutive_errors = 0
            else:
                h.errors_today += 1
                h.total_errors_lifetime += 1
                h.last_error_ts = now
                h.last_error_type = error_type

                if h.circuit_state == CIRCUIT_HALF_OPEN:
                    # Probe failed — re-open
                    h.circuit_state = CIRCUIT_OPEN
                    h.circuit_opened_ts = now
                    logger.warning(
                        "[KnowledgeHealth] Circuit %s: HALF_OPEN → OPEN "
                        "(probe failed %s)", backend_name, error_type)
                else:
                    h.consecutive_errors += 1
                    if (h.consecutive_errors >= self.cb_fail_threshold
                            and h.circuit_state == CIRCUIT_CLOSED):
                        h.circuit_state = CIRCUIT_OPEN
                        h.circuit_opened_ts = now
                        logger.warning(
                            "[KnowledgeHealth] Circuit %s: CLOSED → OPEN "
                            "(%d consecutive errors, last=%s)",
                            backend_name, h.consecutive_errors, error_type)

            # Fire circuit-open alert on ANY transition to OPEN (once/day)
            if (prev_state != CIRCUIT_OPEN
                    and h.circuit_state == CIRCUIT_OPEN
                    and ALERT_KIND_CIRCUIT_OPEN not in h.alerted_today):
                h.alerted_today.add(ALERT_KIND_CIRCUIT_OPEN)
                alerts_to_fire.append((ALERT_KIND_CIRCUIT_OPEN, {
                    "backend": backend_name,
                    "consecutive_errors": h.consecutive_errors,
                    "last_error_type": h.last_error_type,
                    "previous_state": prev_state,
                }))

            self._save()

        # Fire alerts outside the lock — callbacks may do I/O (Telegram HTTP,
        # bus publish) that shouldn't block other record_attempt callers.
        for kind, ctx in alerts_to_fire:
            self._fire_alert(kind, backend_name, ctx)

    def _fire_alert(self, kind: str, backend: str, ctx: dict) -> None:
        """Invoke the on_alert callback if wired. Never raises."""
        if self._on_alert is None:
            return
        try:
            self._on_alert(kind, backend, ctx)
        except Exception as e:
            logger.warning(
                "[KnowledgeHealth] on_alert callback failed (kind=%s, "
                "backend=%s): %s", kind, backend, e)

    # ── Decision log ────────────────────────────────────────────────

    def append_decision(self, entry: dict) -> None:
        """Append a single decision entry to the rolling JSONL log.

        Auto-rotates at decision_log_max_bytes. Best-effort: logs error
        but never raises back to dispatcher.
        """
        try:
            line = json.dumps(entry, default=str) + "\n"
            # Rotate if file would exceed cap
            try:
                size = os.path.getsize(self.decision_log_path)
            except OSError:
                size = 0
            if size + len(line) > self.decision_log_max_bytes:
                self._rotate_decision_log()
            with open(self.decision_log_path, "a") as f:
                f.write(line)
        except Exception as e:
            logger.debug("[KnowledgeHealth] append_decision error: %s", e)

    def _rotate_decision_log(self) -> None:
        """Shift .jsonl → .jsonl.1 → .jsonl.2 → delete."""
        try:
            base = self.decision_log_path
            for i in range(DEFAULT_DECISION_LOG_RETENTION, 0, -1):
                src = f"{base}.{i}" if i > 0 else base
                dst = f"{base}.{i + 1}"
                if i == DEFAULT_DECISION_LOG_RETENTION and os.path.exists(dst):
                    os.remove(dst)
                if os.path.exists(src):
                    os.replace(src, dst)
            if os.path.exists(base):
                os.replace(base, f"{base}.1")
            logger.info("[KnowledgeHealth] Rotated decision log at %s", base)
        except Exception as e:
            logger.warning("[KnowledgeHealth] Rotation failed: %s", e)

    # ── Near-duplicate detection (Optional E) ──────────────────────

    def note_query_for_near_dup(self, normalized: str) -> Optional[str]:
        """Track recent normalized queries + return the best near-duplicate.

        Returns the previous normalized string if Jaccard(tokens) >
        near_dup_jaccard AND it's not identical. None otherwise.
        """
        if not normalized:
            return None
        tokens_now = set(normalized.split())
        best_match = None
        best_jaccard = 0.0
        for prev in self._recent_queries:
            if prev == normalized:
                continue
            tokens_prev = set(prev.split())
            if not tokens_now or not tokens_prev:
                continue
            inter = len(tokens_now & tokens_prev)
            union = len(tokens_now | tokens_prev)
            j = inter / max(1, union)
            if j > best_jaccard:
                best_jaccard = j
                best_match = prev
        self._recent_queries.append(normalized)
        if best_match is not None and best_jaccard >= self.near_dup_jaccard:
            logger.info(
                "[KnowledgeHealth] near-dup: '%s' ~ '%s' (Jaccard=%.2f)",
                normalized[:40], best_match[:40], best_jaccard)
            return best_match
        return None

    # ── Snapshot ────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        """Full health state for /v4/search-pipeline/health (KP-5)."""
        with self._lock:
            # Refresh day epochs + derived fields
            for h in self._backends.values():
                self._maybe_reset_daily(h)
            return {
                "ts": time.time(),
                "backends": {
                    name: h.to_dict() for name, h in self._backends.items()
                },
                "recent_query_sample": list(self._recent_queries)[-5:],
            }


__all__ = [
    "BackendHealth",
    "CIRCUIT_CLOSED",
    "CIRCUIT_HALF_OPEN",
    "CIRCUIT_OPEN",
    "HealthTracker",
]
