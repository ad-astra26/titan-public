"""
Meta-Reasoning Consumer Service — router, aggregator, rate-limiter, cache.

Lives inside spirit_worker's process. Accepts META_REASON_REQUEST from all 8
CGN consumers, enqueues / rate-limits / caches, runs a meta-reasoning chain
(stubbed in Session 1 — returns failure_mode="not_yet_implemented"), and
emits META_REASON_RESPONSE back to the consumer's home worker.

See rFP: titan-docs/rFP_meta_service_interface.md §4 (Service Interface).

Session 1 scope (this commit):
  - queue + aggregator (pending_request tracking)
  - per-consumer + global rate-limiter (sliding window)
  - in-process LRU cache (cosine-match on context_vector)
  - outcome ingestion (store raw; accumulator semantics land in later commit)
  - /v4/meta-service status export
  - DRY-RUN response: all requests immediately resolve with
    failure_mode="not_yet_implemented"

Session 2+ will replace the dry-run with actual chain execution via the
Recruitment Layer (meta_recruitment.py, Commit 4) + compositional sub-modes
(Commit 5) + signed reward blending (Commit 6).
"""
from __future__ import annotations

import collections
import hashlib
import logging
import threading
import time
import tomllib
from pathlib import Path
from typing import Callable, Optional

from ..bus import (
    META_REASON_REQUEST,
    META_REASON_RESPONSE,
    META_REASON_OUTCOME,
)
from .meta_service_client import (
    CONTEXT_VECTOR_DIM,
    FAILURE_MODES,
    KNOWN_CONSUMERS,
    KNOWN_QUESTION_TYPES,
    OUTCOME_REWARD_MAX,
    OUTCOME_REWARD_MIN,
)

logger = logging.getLogger(__name__)


def _load_config() -> dict:
    """Read [meta_service_interface] from titan_params.toml (fail-safe)."""
    try:
        path = Path(__file__).parent.parent / "titan_params.toml"
        with open(path, "rb") as f:
            cfg = tomllib.load(f)
        return cfg.get("meta_service_interface", {}) or {}
    except Exception as e:
        logger.warning(
            "[MetaService] config read failed (%s), using defaults", e)
        return {}


def _context_hash(vec: list) -> str:
    """Stable hash of a context vector for cache keys (first 8 hex chars)."""
    try:
        payload = ",".join(f"{float(x):.4f}" for x in vec)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]
    except Exception:
        return "invalid"


def _cosine_sim(a: list, b: list) -> float:
    """Cosine similarity of two equal-length float vectors. Zero on error."""
    try:
        if len(a) != len(b):
            return 0.0
        dot = 0.0
        na = 0.0
        nb = 0.0
        for x, y in zip(a, b):
            xf = float(x)
            yf = float(y)
            dot += xf * yf
            na += xf * xf
            nb += yf * yf
        if na <= 0 or nb <= 0:
            return 0.0
        return dot / ((na ** 0.5) * (nb ** 0.5))
    except Exception:
        return 0.0


class _RateTracker:
    """Sliding-window rate tracker. Thread-safe, bounded memory."""

    def __init__(self, window_s: float = 60.0):
        self._window_s = float(window_s)
        self._ts: dict = {}  # key → deque[timestamps]
        self._lock = threading.Lock()

    def record(self, key: str) -> None:
        now = time.time()
        with self._lock:
            dq = self._ts.setdefault(key, collections.deque())
            dq.append(now)
            self._prune_unlocked(dq, now)

    def count(self, key: str) -> int:
        now = time.time()
        with self._lock:
            dq = self._ts.get(key)
            if dq is None:
                return 0
            self._prune_unlocked(dq, now)
            return len(dq)

    def _prune_unlocked(self, dq, now: float) -> None:
        cutoff = now - self._window_s
        while dq and dq[0] < cutoff:
            dq.popleft()


class _ThoughtCache:
    """LRU-ish cache for near-duplicate meta-requests (rFP §4.5)."""

    def __init__(self, max_entries: int, ttl_s: float,
                 cosine_threshold: float):
        self._max = int(max_entries)
        self._ttl = float(ttl_s)
        self._thresh = float(cosine_threshold)
        # (consumer_id, question_type) → list[(context_vec, response_payload,
        #                                      cached_at, outcome_reward)]
        self._entries: dict = {}
        self._lock = threading.Lock()

    def lookup(self, consumer_id: str, question_type: str,
               context_vector: list) -> Optional[dict]:
        now = time.time()
        key = (consumer_id, question_type)
        with self._lock:
            bucket = self._entries.get(key)
            if not bucket:
                return None
            # Evict expired in-place
            bucket[:] = [e for e in bucket if now - e[2] <= self._ttl]
            best = None
            best_sim = self._thresh  # must exceed threshold
            for e in bucket:
                sim = _cosine_sim(context_vector, e[0])
                if sim > best_sim:
                    # Only consider non-negative outcome (rFP §4.5)
                    if e[3] is None or e[3] >= 0.0:
                        best = e
                        best_sim = sim
            return dict(best[1]) if best else None

    def store(self, consumer_id: str, question_type: str,
              context_vector: list, response_payload: dict) -> None:
        key = (consumer_id, question_type)
        entry = (list(context_vector), dict(response_payload),
                 time.time(), None)
        with self._lock:
            bucket = self._entries.setdefault(key, [])
            bucket.append(entry)
            # Trim per-key oldest-first
            while len(bucket) > max(1, self._max // 10):
                bucket.pop(0)
            # Global cap across all keys
            total = sum(len(v) for v in self._entries.values())
            if total > self._max:
                # Pop oldest across all buckets
                oldest_key = None
                oldest_t = float("inf")
                for k, v in self._entries.items():
                    if v and v[0][2] < oldest_t:
                        oldest_t = v[0][2]
                        oldest_key = k
                if oldest_key:
                    self._entries[oldest_key].pop(0)

    def update_outcome(self, consumer_id: str, question_type: str,
                       context_vector: list,
                       outcome_reward: float) -> None:
        """Attach outcome reward to most-recent matching entry."""
        key = (consumer_id, question_type)
        with self._lock:
            bucket = self._entries.get(key)
            if not bucket:
                return
            # Find closest-context entry and replace its outcome
            best_idx = -1
            best_sim = -1.0
            for i, e in enumerate(bucket):
                sim = _cosine_sim(context_vector, e[0])
                if sim > best_sim:
                    best_sim = sim
                    best_idx = i
            if best_idx >= 0 and best_sim >= self._thresh:
                vec, resp, t_cached, _ = bucket[best_idx]
                bucket[best_idx] = (vec, resp, t_cached, float(outcome_reward))

    def size(self) -> int:
        with self._lock:
            return sum(len(v) for v in self._entries.values())


class MetaService:
    """Router + queue + cache + rate-limiter for consumer → meta requests."""

    def __init__(self, response_emitter: Optional[Callable] = None,
                 outcome_sink: Optional[Callable] = None,
                 recruitment: Optional["object"] = None):
        """
        Args:
            response_emitter: callable(msg_dict) invoked to publish the
                META_REASON_RESPONSE. Usually spirit_worker's send_queue.
                put_nowait. If None, responses are only captured in stats.
            outcome_sink: callable(payload_dict) invoked whenever a
                META_REASON_OUTCOME is ingested. Session 1: no-op;
                Commit 6 wires this to the dynamic-reward accumulator.
            recruitment: optional MetaRecruitment instance. Session 1
                exposes it through get_status() for observability; Session
                2 uses it for chain dispatch when resolving primitives.
        """
        cfg = _load_config()
        self._per_consumer_rpm = int(cfg.get("per_consumer_requests_per_min", 10))
        self._global_rpm = int(cfg.get("global_requests_per_min", 60))
        self._queue_max = int(cfg.get("queue_max_depth", 20))
        self._backpressure = int(cfg.get("backpressure_threshold", 15))
        self._alpha_ramp_enabled = bool(cfg.get("alpha_ramp_enabled", False))
        self._cache_enabled = bool(cfg.get("cache_enabled", True))

        self._rate = _RateTracker(window_s=60.0)
        self._cache = _ThoughtCache(
            max_entries=int(cfg.get("cache_max_entries", 200)),
            ttl_s=float(cfg.get("cache_ttl_seconds", 300)),
            cosine_threshold=float(cfg.get("cache_cosine_match_threshold", 0.85)),
        )
        self._home_worker_map: dict = dict(
            cfg.get("consumer_home_worker", {}))

        self._queue: collections.deque = collections.deque()
        self._pending: dict = {}  # request_id → entry dict
        self._outcomes: collections.deque = collections.deque(maxlen=1000)

        self._response_emitter = response_emitter
        self._recruitment = recruitment
        self._lock = threading.Lock()

        # Dynamic reward accumulator (rFP §7 / Upgrade D). Instantiated
        # locally unless caller provided an external one (for test injection).
        # outcome_sink — if caller gives one, we chain it AFTER the internal
        # accumulator update so external observers still receive records.
        try:
            from .meta_dynamic_rewards import DynamicRewardAccumulator
            self._rewards = DynamicRewardAccumulator(
                alpha_ramp_enabled=self._alpha_ramp_enabled,
                phase_0_end=int(cfg.get("alpha_phase_0_end", 500)),
                phase_1_end=int(cfg.get("alpha_phase_1_end", 2000)),
                phase_2_end=int(cfg.get("alpha_phase_2_end", 5000)),
                phase_3_end=int(cfg.get("alpha_phase_3_end", 10000)),
                cold_start_n=int(cfg.get("dynamic_cold_start_n", 10)),
            )
        except Exception as e:
            logger.warning("[MetaService] DynamicRewardAccumulator init: %s", e)
            self._rewards = None

        # Wrap outcome_sink so internal accumulator always receives first,
        # then chain to caller-provided sink.
        _external_sink = outcome_sink
        def _internal_and_chain(record: dict) -> None:
            if self._rewards is not None:
                try:
                    self._rewards.ingest_outcome_record(record)
                except Exception as e:
                    logger.debug("[MetaService] accumulator ingest: %s", e)
            if _external_sink is not None:
                try:
                    _external_sink(record)
                except Exception as e:
                    logger.debug("[MetaService] external outcome sink: %s", e)
        self._outcome_sink = _internal_and_chain

        # Telemetry (rFP §11.1)
        self._stats = {
            "requests_received": 0,
            "requests_dry_run_resolved": 0,
            "requests_completed": 0,  # Session 2+: real-chain resolutions
            "requests_failed": 0,
            "cache_hits": 0,
            "cache_stores": 0,
            "rate_limited": 0,
            "backpressure_events": 0,
            "queue_overflows": 0,
            "outcomes_received": 0,
            "outcomes_invalid": 0,
            "per_consumer_received": collections.Counter(),
            "per_consumer_outcomes": collections.Counter(),
            "per_consumer_outcome_sum": collections.defaultdict(float),
            "per_question_type_received": collections.Counter(),
            "started_at": time.time(),
        }
        logger.info(
            "[MetaService] initialized: per_consumer_rpm=%d global_rpm=%d "
            "queue_max=%d backpressure=%d cache=%s alpha_ramp=%s",
            self._per_consumer_rpm, self._global_rpm, self._queue_max,
            self._backpressure, self._cache_enabled, self._alpha_ramp_enabled)

    # ── Request ingestion ───────────────────────────────────────────

    def handle_request(self, msg: dict) -> Optional[str]:
        """Process incoming META_REASON_REQUEST. Returns failure_mode (if any
        synchronous rejection) or None when the request enters the queue
        or is served from cache.

        Session 1 dry-run: every queued request is immediately resolved with
        failure_mode="not_yet_implemented" via the response_emitter so the
        end-to-end wiring can be exercised without running real chains yet.
        """
        payload = msg.get("payload") or {}
        consumer_id = payload.get("consumer_id", "")
        question_type = payload.get("question_type", "")
        request_id = payload.get("request_id", "")
        context_vector = payload.get("context_vector") or []
        src = msg.get("src", "")

        # Schema sanity (client already validated — defense in depth)
        if (consumer_id not in KNOWN_CONSUMERS
                or question_type not in KNOWN_QUESTION_TYPES
                or not request_id
                or len(context_vector) != CONTEXT_VECTOR_DIM):
            self._stats["requests_failed"] += 1
            logger.warning(
                "[MetaService] schema-invalid request rejected: "
                "consumer=%s qt=%s req_id=%s ctx_len=%d",
                consumer_id, question_type, request_id, len(context_vector))
            return "schema_invalid"

        with self._lock:
            self._stats["requests_received"] += 1
            self._stats["per_consumer_received"][consumer_id] += 1
            self._stats["per_question_type_received"][question_type] += 1

        # Rate-limit checks (per-consumer then global)
        per_consumer = self._rate.count(f"c:{consumer_id}")
        global_count = self._rate.count("global")
        if per_consumer >= self._per_consumer_rpm:
            self._stats["rate_limited"] += 1
            self._emit_response(
                request_id=request_id,
                consumer_id=consumer_id,
                src=src,
                failure_mode="rate_limited",
                insight=None,
                reason=f"per-consumer cap {self._per_consumer_rpm}/min",
            )
            return "rate_limited"
        if global_count >= self._global_rpm:
            self._stats["rate_limited"] += 1
            self._emit_response(
                request_id=request_id,
                consumer_id=consumer_id,
                src=src,
                failure_mode="rate_limited",
                insight=None,
                reason=f"global cap {self._global_rpm}/min",
            )
            return "rate_limited"

        # Queue depth + backpressure check
        with self._lock:
            qlen = len(self._queue)
        if qlen >= self._queue_max:
            self._stats["queue_overflows"] += 1
            self._emit_response(
                request_id=request_id,
                consumer_id=consumer_id,
                src=src,
                failure_mode="rate_limited",
                insight=None,
                reason=f"queue full ({self._queue_max})",
            )
            return "rate_limited"
        if qlen >= self._backpressure:
            self._stats["backpressure_events"] += 1
            # Not a hard reject — just record the signal; consumer may see
            # rate_limited soon. Still let this one through.

        # Record the request in rate trackers
        self._rate.record(f"c:{consumer_id}")
        self._rate.record("global")

        # Cache lookup
        if self._cache_enabled:
            cached = self._cache.lookup(consumer_id, question_type,
                                         context_vector)
            if cached is not None:
                self._stats["cache_hits"] += 1
                cached_insight = cached.get("insight")
                cached_failure = cached.get("failure_mode")
                self._emit_response(
                    request_id=request_id,
                    consumer_id=consumer_id,
                    src=src,
                    failure_mode=cached_failure,
                    insight=cached_insight,
                    reason="cache_hit",
                    extra={"cache_hit": True},
                )
                return None

        # Enqueue for async processing. Session 1: resolve dry-run inline.
        entry = {
            "request_id": request_id,
            "consumer_id": consumer_id,
            "question_type": question_type,
            "context_vector": list(context_vector),
            "time_budget_ms": int(payload.get("time_budget_ms", 0)),
            "constraints": dict(payload.get("constraints") or {}),
            "payload_snippet": str(payload.get("payload_snippet", ""))[:256],
            "src": src,
            "t_enqueue": time.time(),
        }
        with self._lock:
            self._queue.append(request_id)
            self._pending[request_id] = entry

        # SESSION 1 DRY-RUN: immediately resolve
        self._resolve_dry_run(request_id)
        return None

    def _resolve_dry_run(self, request_id: str) -> None:
        """Session 1: drain the queue by returning not_yet_implemented."""
        with self._lock:
            entry = self._pending.pop(request_id, None)
            try:
                self._queue.remove(request_id)
            except ValueError:
                pass
        if entry is None:
            return
        self._emit_response(
            request_id=request_id,
            consumer_id=entry["consumer_id"],
            src=entry["src"],
            failure_mode="not_yet_implemented",
            insight=None,
            reason="session_1_dry_run",
        )
        self._stats["requests_dry_run_resolved"] += 1

    # ── Response emission ───────────────────────────────────────────

    def _emit_response(
        self,
        request_id: str,
        consumer_id: str,
        src: str,
        failure_mode: Optional[str],
        insight: Optional[dict],
        reason: str = "",
        extra: Optional[dict] = None,
    ) -> None:
        """Build and publish a META_REASON_RESPONSE via response_emitter."""
        if failure_mode and failure_mode not in FAILURE_MODES:
            # Internal bug — log but still emit (consumer's failure_mode
            # handling branch will cover unknown values).
            logger.warning(
                "[MetaService] unknown failure_mode=%r in emit for %s",
                failure_mode, request_id)

        home_worker = self._home_worker_map.get(consumer_id)
        if not home_worker:
            # Fallback: try to reply to msg.src if the consumer came from a
            # registered worker. Better than silent drop.
            home_worker = src or "spirit"
            logger.debug(
                "[MetaService] no home_worker mapping for consumer='%s', "
                "falling back to src=%s", consumer_id, home_worker)

        payload = {
            "consumer_id": consumer_id,
            "request_id": request_id,
            "insight": insight,
            "recruitment_trace": [],
            "timechain_trace": [],
            "confidence": 0.0,
            "chain_id": 0,
            "processing_time_ms": 0,
            "failure_mode": failure_mode,
            "reason": reason,
        }
        if extra:
            payload.update(extra)

        msg = {
            "type": META_REASON_RESPONSE,
            "src": "spirit",
            "dst": home_worker,
            "ts": time.time(),
            "rid": None,
            "payload": payload,
        }
        if self._response_emitter is not None:
            try:
                self._response_emitter(msg)
            except Exception as e:
                logger.warning(
                    "[MetaService] response_emitter failed for %s: %s",
                    request_id, e)

    # ── Outcome ingestion ───────────────────────────────────────────

    def handle_outcome(self, msg: dict) -> bool:
        """Ingest META_REASON_OUTCOME. Returns True on accept, False on
        schema rejection."""
        payload = msg.get("payload") or {}
        consumer_id = payload.get("consumer_id", "")
        request_id = payload.get("request_id", "")
        reward = payload.get("outcome_reward")

        if consumer_id not in KNOWN_CONSUMERS or not request_id:
            self._stats["outcomes_invalid"] += 1
            return False
        try:
            reward_f = float(reward)
        except (TypeError, ValueError):
            self._stats["outcomes_invalid"] += 1
            return False
        if not (OUTCOME_REWARD_MIN <= reward_f <= OUTCOME_REWARD_MAX):
            self._stats["outcomes_invalid"] += 1
            logger.warning(
                "[MetaService] outcome out of [-1,+1]: consumer=%s "
                "reward=%s req_id=%s", consumer_id, reward, request_id)
            return False

        record = {
            "t_received": time.time(),
            "request_id": request_id,
            "consumer_id": consumer_id,
            "outcome_reward": reward_f,
            "actual_primitive_used":
                payload.get("actual_primitive_used"),
            "context": str(payload.get("context", ""))[:256],
        }
        with self._lock:
            self._outcomes.append(record)
            self._stats["outcomes_received"] += 1
            self._stats["per_consumer_outcomes"][consumer_id] += 1
            self._stats["per_consumer_outcome_sum"][consumer_id] += reward_f

        if self._outcome_sink is not None:
            try:
                self._outcome_sink(record)
            except Exception as e:
                logger.debug(
                    "[MetaService] outcome_sink error: %s (consumer=%s)",
                    e, consumer_id)
        return True

    # ── Status export (for /v4/meta-service) ────────────────────────

    def get_status(self) -> dict:
        """Snapshot for the /v4/meta-service endpoint (rFP §11.1)."""
        with self._lock:
            qlen = len(self._queue)
            pending = len(self._pending)
            per_consumer_recv = dict(self._stats["per_consumer_received"])
            per_consumer_out = dict(self._stats["per_consumer_outcomes"])
            per_consumer_sum = dict(self._stats["per_consumer_outcome_sum"])
            per_qt = dict(self._stats["per_question_type_received"])
            uptime = time.time() - self._stats["started_at"]

        avg_outcome = {
            c: (per_consumer_sum.get(c, 0.0)
                / max(1, per_consumer_out.get(c, 1)))
            for c in per_consumer_out
        }

        recruitment_stats = None
        if self._recruitment is not None:
            try:
                recruitment_stats = self._recruitment.get_stats()
            except Exception as e:
                logger.debug(
                    "[MetaService] recruitment stats error: %s", e)

        rewards_stats = None
        if self._rewards is not None:
            try:
                rewards_stats = self._rewards.get_stats()
            except Exception as e:
                logger.debug(
                    "[MetaService] rewards stats error: %s", e)

        return {
            "session_phase": "session_1_dry_run",
            "alpha_ramp_enabled": self._alpha_ramp_enabled,
            "uptime_seconds": round(uptime, 1),
            "queue_depth": qlen,
            "pending_requests": pending,
            "queue_max_depth": self._queue_max,
            "backpressure_threshold": self._backpressure,
            "per_consumer_requests_per_min": self._per_consumer_rpm,
            "global_requests_per_min": self._global_rpm,
            "cache": {
                "enabled": self._cache_enabled,
                "current_size": self._cache.size() if self._cache_enabled
                                                   else 0,
                "hits": self._stats["cache_hits"],
                "stores": self._stats["cache_stores"],
            },
            "counters": {
                "requests_received": self._stats["requests_received"],
                "requests_dry_run_resolved":
                    self._stats["requests_dry_run_resolved"],
                "requests_completed": self._stats["requests_completed"],
                "requests_failed": self._stats["requests_failed"],
                "rate_limited": self._stats["rate_limited"],
                "backpressure_events": self._stats["backpressure_events"],
                "queue_overflows": self._stats["queue_overflows"],
                "outcomes_received": self._stats["outcomes_received"],
                "outcomes_invalid": self._stats["outcomes_invalid"],
            },
            "per_consumer_requests": per_consumer_recv,
            "per_consumer_outcomes": per_consumer_out,
            "per_consumer_avg_outcome_reward": {
                c: round(v, 4) for c, v in avg_outcome.items()
            },
            "per_question_type_requests": per_qt,
            "home_worker_map": dict(self._home_worker_map),
            "recruitment": recruitment_stats,
            "rewards": rewards_stats,
        }
