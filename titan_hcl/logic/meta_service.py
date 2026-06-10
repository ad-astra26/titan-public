"""
Meta-Reasoning Consumer Service — router, aggregator, rate-limiter, cache.

Lives inside cognitive_worker's process (migrated from spirit_worker during
D8-3 retirement, commit 72f95a6b 2026-05-16). Accepts META_REASON_REQUEST
from all 8 CGN consumers, enqueues / rate-limits / caches, runs a
meta-reasoning chain via the Recruitment Layer (Session 3 live-dispatch
resolvers), and emits META_REASON_RESPONSE back to the consumer's home
worker.

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

import asyncio
import collections
import hashlib
import inspect
import logging
import threading
import time
import tomllib
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional

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
from titan_hcl.utils.silent_swallow import swallow_warn

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Session 3 — live-dispatch infrastructure (RFP_meta-reasoning_CGN_FIX.md
# PART A §4.1). MetaService dispatches a META_REASON_REQUEST through the
# Recruitment Layer, the chosen resolver (async coroutine per SPEC G19+G22
# + §8.0.ter D-SPEC-48) publishes a downstream bus event (CGN_KNOWLEDGE_REQ
# / TIMECHAIN_QUERY / META_LANGUAGE_REQUEST) and awaits the matching
# response via a correlation_id-keyed asyncio.Future. When the target
# worker publishes the response, the cognitive_worker bus loop forwards it
# to MetaService.handle_response which resolves the matching Future
# thread-safely. Periodic sweep_timeouts() emits failure_mode=
# "resolver_timeout" for stale dispatches.
# ──────────────────────────────────────────────────────────────────────


# Maps consumer-facing question_type → primitive (RFP §4.2 mapping).
# Used by _dispatch_to_resolver to seed the Recruitment Layer's Thompson
# β-posterior selector with the right primitive context.
QUESTION_TYPE_TO_PRIMITIVE: Dict[str, str] = {
    "formulate_strategy":   "FORMULATE",
    "recall_context":       "RECALL",
    "evaluate_option":      "EVALUATE",
    "evaluate_trajectory":  "EVALUATE",
    "hypothesize_cause":    "HYPOTHESIZE",
    "synthesize_insight":   "SYNTHESIZE",
    "break_impasse":        "BREAK",
    "introspect_state":     "INTROSPECT",
    "spirit_self_nudge":    "SPIRIT_SELF",
}


class _PendingResponseRegistry:
    """Tracks correlation_id → asyncio.Future for resolver response awaits.

    The MetaService dispatcher runs in a dedicated daemon asyncio loop so the
    sync cognitive_worker bus handler can schedule resolver coroutines via
    `asyncio.run_coroutine_threadsafe`. The resolver awaits an entry of this
    registry; the bus handler (running on the cognitive_worker thread) resolves
    the matching Future via `loop.call_soon_threadsafe`.

    SPEC anchors:
      - Preamble G19: async response with ≤5s timeout
      - §8.0.ter D-SPEC-48: non-blocking publish — `put_nowait`
      - §4.1 RFP_meta-reasoning_CGN_FIX.md: correlation_id-keyed cache
    """

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop
        # Mutated only from the loop thread (resolver) or via
        # call_soon_threadsafe (bus handler).
        self._pending: Dict[str, asyncio.Future] = {}
        # Metadata for timeout sweep + telemetry — mirrors _pending key.
        self._meta: Dict[str, dict] = {}

    @staticmethod
    def next_correlation_id() -> str:
        """Generate a fresh correlation_id (uuid4 hex, 32 chars)."""
        return uuid.uuid4().hex

    def register(self, correlation_id: str, meta: Optional[dict] = None
                 ) -> asyncio.Future:
        """Create + register a pending Future for the given correlation_id.
        Called from inside a resolver coroutine (loop thread).
        """
        fut = self._loop.create_future()
        self._pending[correlation_id] = fut
        if meta is not None:
            self._meta[correlation_id] = dict(meta)
            self._meta[correlation_id]["dispatch_ts"] = time.time()
        return fut

    def resolve(self, correlation_id: str, response: dict) -> bool:
        """Resolve a pending Future from any thread.

        Returns True if the correlation_id matched a registered Future
        (regardless of whether it was already done). False if unknown or
        stale (already swept or unknown correlation_id).
        """
        fut = self._pending.pop(correlation_id, None)
        self._meta.pop(correlation_id, None)
        if fut is None:
            return False
        if fut.done():
            # Already resolved (race with timeout sweep); benign.
            return True
        # Schedule set_result on the loop thread — Futures from
        # asyncio.AbstractEventLoop are not thread-safe for set_result.
        self._loop.call_soon_threadsafe(_safe_set_future_result, fut, response)
        return True

    def discard(self, correlation_id: str) -> None:
        """Drop a pending entry without resolving (e.g., resolver shutdown)."""
        self._pending.pop(correlation_id, None)
        self._meta.pop(correlation_id, None)

    def sweep_timeouts(self, timeout_s: float) -> list:
        """Return list of correlation_ids whose dispatch_ts is older than
        `timeout_s`. Caller is responsible for resolving them with a
        timeout response and removing from registry.
        """
        now = time.time()
        stale: list = []
        # Snapshot to avoid mutation during iteration; tolerate races.
        for cid, meta in list(self._meta.items()):
            dispatch_ts = meta.get("dispatch_ts", 0.0)
            if now - dispatch_ts > timeout_s:
                stale.append(cid)
        return stale

    def size(self) -> int:
        return len(self._pending)

    def metadata(self, correlation_id: str) -> Optional[dict]:
        return self._meta.get(correlation_id)


def _safe_set_future_result(fut: asyncio.Future, result: Any) -> None:
    """Called via call_soon_threadsafe — runs on the loop thread. Guards
    against double-resolution (race with timeout sweep)."""
    if not fut.done():
        fut.set_result(result)


def _default_sub_mode_for(primitive: str) -> Optional[str]:
    """Default sub_mode pick for a primitive — first entry in SUB_MODES
    list. The Thompson β-posterior selector in MetaRecruitment then chooses
    the recruiter for (primitive, sub_mode) from RECRUITMENT_CATALOG,
    weighted by past outcome success. Once outcomes accumulate (α-ramp
    Phase 2+), the selector picks higher-V sub_modes.
    """
    # Import here to avoid cycle (meta_reasoning imports meta_service via
    # the in-process MetaCGNConsumer integration).
    try:
        from .meta_reasoning import SUB_MODES
    except ImportError:
        return None
    modes = SUB_MODES.get(primitive)
    if not modes:
        return None
    return modes[0]


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
                META_REASON_RESPONSE. Usually cognitive_worker's send_queue.
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
        # Phase A (RFP_cgn_enhancements §9.1) — optional callback that receives
        # concept-grounding requests (those carrying a grounding_payload) and
        # appends them to the MetaReasoningEngine's _pending_groundings queue
        # (drained by should_trigger_meta Path #0). Wired by cognitive_worker
        # after both MetaService and the engine exist. None → grounding
        # requests fall through to the normal one-shot resolver (back-compat).
        self._grounding_sink: Optional[Callable] = None
        self._lock = threading.Lock()

        # ── Session 3 live-dispatch infrastructure ─────────────────────
        # Dedicated asyncio event loop running in a daemon thread so the
        # sync cognitive_worker bus handler can schedule async resolver
        # coroutines via `asyncio.run_coroutine_threadsafe`. The loop
        # owns all _pending_registry mutations; bus-thread interactions
        # use `call_soon_threadsafe`.
        # SPEC anchors: Preamble G19 (async resolver, ≤5s) + §8.0.ter
        # D-SPEC-48 (non-blocking publish).
        self._dispatch_timeout_s: float = float(
            cfg.get("resolver_dispatch_timeout_s", 5.0))
        self._loop = asyncio.new_event_loop()
        self._loop_ready = threading.Event()
        self._loop_thread = threading.Thread(
            target=self._run_dispatch_loop,
            daemon=True,
            name="meta-service-loop",
        )
        self._loop_thread.start()
        self._loop_ready.wait(timeout=2.0)
        if not self._loop_ready.is_set():
            logger.error(
                "[MetaService] dispatch loop failed to start within 2s")
        self._pending_registry = _PendingResponseRegistry(self._loop)
        # request_id → correlation_id reverse lookup (used by
        # sweep_timeouts to emit META_REASON_RESPONSE for stale dispatches).
        self._request_to_correlation: Dict[str, str] = {}
        # correlation_id → request_id (used by handle_response to find
        # the originating request — emit_response needs request_id).
        self._correlation_to_request: Dict[str, str] = {}

        # Dynamic reward accumulator (rFP §7 / Upgrade D). Instantiated
        # locally unless caller provided an external one (for test injection).
        # outcome_sink — if caller gives one, we chain it AFTER the internal
        # accumulator update so external observers still receive records.
        try:
            from .meta_dynamic_rewards import DynamicRewardAccumulator
            # RFP_meta-reasoning_CGN_FIX.md §4.4 — gentler 5-tier schedule
            # with α=0.10 warm-up + time-escape hatch.
            self._rewards = DynamicRewardAccumulator(
                alpha_ramp_enabled=self._alpha_ramp_enabled,
                phase_warmup_end=int(cfg.get("alpha_phase_warmup_end", 500)),
                phase_0_end=int(cfg.get("alpha_phase_0_end", 2000)),
                phase_1_end=int(cfg.get("alpha_phase_1_end", 5000)),
                phase_2_end=int(cfg.get("alpha_phase_2_end", 10000)),
                phase_3_end=int(cfg.get("alpha_phase_3_end", 20000)),
                cold_start_n=int(cfg.get("dynamic_cold_start_n", 10)),
                time_escape_enabled=bool(
                    cfg.get("alpha_time_escape_enabled", True)),
                time_escape_seconds_per_step=float(
                    cfg.get("alpha_time_escape_seconds_per_step", 604800.0)),
                time_escape_increment=float(
                    cfg.get("alpha_time_escape_increment", 0.10)),
                time_escape_cap=float(
                    cfg.get("alpha_time_escape_cap", 1.0)),
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
                    swallow_warn('[MetaService] accumulator ingest', e,
                                 key="logic.meta_service.accumulator_ingest", throttle=100)
            if _external_sink is not None:
                try:
                    _external_sink(record)
                except Exception as e:
                    swallow_warn('[MetaService] external outcome sink', e,
                                 key="logic.meta_service.external_outcome_sink", throttle=100)
        self._outcome_sink = _internal_and_chain

        # Telemetry (rFP §11.1 + Session 3 dispatch counters)
        self._stats = {
            "requests_received": 0,
            "requests_dry_run_resolved": 0,
            "requests_completed": 0,           # Session 3 live-dispatch successes
            "requests_failed": 0,
            "cache_hits": 0,
            "cache_stores": 0,
            "rate_limited": 0,
            "backpressure_events": 0,
            "queue_overflows": 0,
            "outcomes_received": 0,
            "outcomes_invalid": 0,
            # Session 3 dispatch telemetry — RFP_meta-reasoning_CGN_FIX.md §13.1
            "dispatches_scheduled": 0,         # resolver coroutines kicked off
            "dispatches_resolved": 0,          # response arrived in time
            "dispatches_timed_out": 0,         # 5s timeout without response
            "dispatches_resolver_unavailable": 0,
            "dispatches_resolver_error": 0,
            "responses_correlated": 0,         # handle_response correlation_id matched
            "responses_uncorrelated": 0,       # stale / unknown correlation_id
            "per_consumer_received": collections.Counter(),
            "per_consumer_outcomes": collections.Counter(),
            "per_consumer_outcome_sum": collections.defaultdict(float),
            "per_question_type_received": collections.Counter(),
            "per_category_dispatched": collections.Counter(),
            "started_at": time.time(),
        }
        logger.info(
            "[MetaService] initialized: per_consumer_rpm=%d global_rpm=%d "
            "queue_max=%d backpressure=%d cache=%s alpha_ramp=%s "
            "dispatch_timeout=%.1fs loop_thread=%s",
            self._per_consumer_rpm, self._global_rpm, self._queue_max,
            self._backpressure, self._cache_enabled, self._alpha_ramp_enabled,
            self._dispatch_timeout_s, self._loop_thread.name)

    # ── Session 3 dispatch loop (daemon thread) ─────────────────────

    def _run_dispatch_loop(self) -> None:
        """Run the dedicated asyncio loop in a daemon thread.

        Started from __init__. Loop processes resolver coroutines scheduled
        via `asyncio.run_coroutine_threadsafe`. Loop shutdown via close().
        """
        try:
            asyncio.set_event_loop(self._loop)
            self._loop_ready.set()
            self._loop.run_forever()
        except Exception as e:
            logger.error("[MetaService] dispatch loop crashed: %s", e)
        finally:
            try:
                self._loop.close()
            except Exception:
                pass

    def close(self) -> None:
        """Graceful shutdown of the dispatch loop. Idempotent."""
        if not getattr(self, "_loop", None):
            return
        try:
            if self._loop.is_running():
                self._loop.call_soon_threadsafe(self._loop.stop)
            if self._loop_thread.is_alive():
                self._loop_thread.join(timeout=2.0)
        except Exception as e:
            logger.warning("[MetaService] close error: %s", e)

    @property
    def pending_registry(self) -> "_PendingResponseRegistry":
        """Expose the pending registry to resolver factory closures.
        Resolvers call `pending_registry.register(corr_id)` to create the
        Future they await.
        """
        return self._pending_registry

    # ── Session 3 dispatch flow (rFP §4.1) ──────────────────────────

    def _dispatch_to_resolver(self, request_id: str) -> None:
        """Live-dispatch path replacing Session 1 _resolve_dry_run.

        Maps question_type → primitive → Thompson-selected recruiter →
        resolver coroutine. Schedules the resolver on the dedicated
        asyncio loop. The resolver publishes a downstream bus event with
        a fresh correlation_id and awaits the matching response Future
        registered in `_pending_registry`. On resolution (success / timeout
        / error), `_finalize_dispatch` emits the META_REASON_RESPONSE
        and updates telemetry.

        Falls back to `_resolve_dry_run` if any prerequisite is missing
        (unknown question_type, no recruitment layer, unregistered
        resolver). Fallback preserves Session 1 behavior for backward
        compatibility.
        """
        with self._lock:
            entry = self._pending.get(request_id)
        if entry is None:
            return  # already resolved by some other path

        question_type = entry.get("question_type", "")
        primitive = QUESTION_TYPE_TO_PRIMITIVE.get(question_type)
        if not primitive:
            logger.warning(
                "[MetaService] no primitive mapped for question_type=%r — "
                "falling back to dry-run resolution", question_type)
            self._resolve_dry_run(request_id)
            return
        if self._recruitment is None:
            logger.debug(
                "[MetaService] no recruitment layer wired — dry-run fallback")
            self._resolve_dry_run(request_id)
            return

        # Pick a sub_mode default — first entry in SUB_MODES[primitive].
        # Thompson β-selector then chooses the recruiter for (primitive,
        # sub_mode) from RECRUITMENT_CATALOG, weighted by past outcome
        # success. Session 3 cold-start: all β posteriors are
        # uninformative until outcomes accumulate, so selection is
        # effectively uniform random.
        sub_mode = _default_sub_mode_for(primitive)
        if sub_mode is None:
            logger.warning(
                "[MetaService] no SUB_MODES entry for primitive=%r — dry-run",
                primitive)
            self._resolve_dry_run(request_id)
            return

        recruiter = None
        try:
            recruiter = self._recruitment.select_recruiter(primitive, sub_mode)
        except Exception as e:
            logger.warning(
                "[MetaService] select_recruiter raised: %s — dry-run fallback",
                e)

        if not recruiter:
            self._stats["dispatches_resolver_unavailable"] += 1
            self._emit_response_for_failure(
                request_id, "resolver_unavailable",
                f"no recruiter for {primitive}.{sub_mode}")
            return

        # Recruiter is a string like "reasoning.DECOMPOSE" or "<self:...>".
        # Skip self-recursive meta chains here — DELEGATE.full_chain etc.
        # route via a different in-engine path (rFP §14.2 recursion cap).
        if recruiter.startswith("<self:"):
            logger.debug(
                "[MetaService] recruiter=%s is self-recursive — dry-run",
                recruiter)
            self._resolve_dry_run(request_id)
            return

        if "." in recruiter:
            category, _, name = recruiter.partition(".")
        else:
            category, name = recruiter, "default"

        resolver_fn = self._recruitment._resolvers.get(category) \
            if self._recruitment else None
        if resolver_fn is None:
            self._stats["dispatches_resolver_unavailable"] += 1
            self._emit_response_for_failure(
                request_id, "resolver_unavailable",
                f"no resolver registered for category={category!r}")
            return

        # Build the resolver context — what each resolver needs to construct
        # its downstream dispatch payload + correlation_id.
        ctx = {
            "request_id": request_id,
            "consumer_id": entry.get("consumer_id", ""),
            "question_type": question_type,
            "primitive": primitive,
            "sub_mode": sub_mode,
            "recruiter": recruiter,
            "context_vector": list(entry.get("context_vector") or []),
            "time_budget_ms": int(entry.get("time_budget_ms", 0)),
            "constraints": dict(entry.get("constraints") or {}),
            "payload_snippet": entry.get("payload_snippet", ""),
        }

        self._stats["dispatches_scheduled"] += 1
        self._stats["per_category_dispatched"][category] += 1

        # Schedule resolver coroutine on the dedicated dispatch loop.
        # The coroutine returns a normalized result dict; _finalize_dispatch
        # consumes it and emits META_REASON_RESPONSE.
        coro = self._run_resolver(request_id, category, name, ctx, resolver_fn)
        asyncio.run_coroutine_threadsafe(coro, self._loop)

    async def _run_resolver(self, request_id: str, category: str,
                            name: str, ctx: dict,
                            resolver_fn: Callable) -> None:
        """Run a resolver coroutine + emit the META_REASON_RESPONSE.

        Runs inside the dedicated dispatch event loop. The resolver itself
        is responsible for publishing a downstream bus event with a
        correlation_id, registering a pending Future, awaiting the response,
        and returning a normalized dict:
            {"success": bool, "output": dict|None, "recruiter": str,
             "reason": str, "failure_mode": Optional[str]}

        On any exception or timeout-shaped result, _finalize_dispatch emits
        the appropriate failure_mode META_REASON_RESPONSE.
        """
        try:
            if inspect.iscoroutinefunction(resolver_fn):
                result = await resolver_fn(name, ctx)
            else:
                # Sync resolver — legacy shells from Session 2. Wrap so we
                # don't block the loop on a sync call (these only return
                # immediately-prepared dicts, so this is safe).
                result = resolver_fn(name, ctx)
        except asyncio.TimeoutError:
            self._stats["dispatches_timed_out"] += 1
            self._emit_response_for_failure(
                request_id, "resolver_timeout",
                f"resolver {category}.{name} timed out")
            return
        except Exception as e:
            self._stats["dispatches_resolver_error"] += 1
            logger.warning(
                "[MetaService] resolver %s.%s raised: %s",
                category, name, e)
            self._emit_response_for_failure(
                request_id, "resolver_error", f"{type(e).__name__}: {e}")
            return

        # Resolver returned — classify success vs failure.
        if not isinstance(result, dict):
            self._stats["dispatches_resolver_error"] += 1
            self._emit_response_for_failure(
                request_id, "resolver_error",
                f"resolver returned non-dict: {type(result).__name__}")
            return

        failure_mode = result.get("failure_mode")
        success = bool(result.get("success", False))
        output = result.get("output")

        if failure_mode == "resolver_timeout":
            self._stats["dispatches_timed_out"] += 1
            self._emit_response_for_failure(
                request_id, "resolver_timeout",
                result.get("reason", "resolver awaited timeout"))
            return
        if failure_mode == "resolver_error":
            self._stats["dispatches_resolver_error"] += 1
            self._emit_response_for_failure(
                request_id, "resolver_error",
                result.get("reason", "resolver internal error"))
            return
        if not success or output is None:
            # Resolver completed but signaled non-success (e.g. unknown
            # primitive name). Surface as low_confidence so consumer still
            # learns something from the trip.
            self._emit_response_for_failure(
                request_id, "low_confidence",
                result.get("reason", "resolver returned no usable output"))
            return

        # Success — emit META_REASON_RESPONSE with the resolver output.
        self._stats["dispatches_resolved"] += 1
        self._stats["requests_completed"] += 1
        with self._lock:
            self._pending.pop(request_id, None)
            try:
                self._queue.remove(request_id)
            except ValueError as _swallow_exc:
                swallow_warn('[MetaService] queue.remove on dispatch resolve',
                             _swallow_exc,
                             key="logic.meta_service.queue_remove_on_resolve",
                             throttle=100)

        # Cache the result for future cosine-similar requests
        if self._cache_enabled:
            try:
                self._cache.store(
                    ctx.get("consumer_id", ""), ctx.get("question_type", ""),
                    ctx.get("context_vector") or [],
                    {"insight": output, "failure_mode": None},
                )
                self._stats["cache_stores"] += 1
            except Exception as _swallow_exc:
                swallow_warn('[MetaService] cache store on resolve',
                             _swallow_exc,
                             key="logic.meta_service.cache_store_on_resolve",
                             throttle=100)

        self._emit_response(
            request_id=request_id,
            consumer_id=ctx.get("consumer_id", ""),
            src=ctx.get("src", ""),
            failure_mode=None,
            insight=output,
            reason=result.get("reason", "resolved"),
            extra={"recruiter": result.get("recruiter", ctx.get("recruiter"))},
        )

    def _emit_response_for_failure(self, request_id: str,
                                   failure_mode: str, reason: str) -> None:
        """Pop the request from _pending and emit META_REASON_RESPONSE
        with the given failure_mode. Centralized so all dispatch failure
        paths share the same cleanup."""
        with self._lock:
            entry = self._pending.pop(request_id, None)
            try:
                self._queue.remove(request_id)
            except ValueError as _swallow_exc:
                swallow_warn(
                    '[MetaService] queue.remove on dispatch failure',
                    _swallow_exc,
                    key="logic.meta_service.queue_remove_on_failure",
                    throttle=100)
        if entry is None:
            return
        self._stats["requests_failed"] += 1
        self._emit_response(
            request_id=request_id,
            consumer_id=entry.get("consumer_id", ""),
            src=entry.get("src", ""),
            failure_mode=failure_mode,
            insight=None,
            reason=reason,
        )

    # ── Session 3 response correlation (cognitive_worker bus handler entry) ─

    def handle_response(self, msg: dict) -> bool:
        """Route an incoming CGN_KNOWLEDGE_RESP / TIMECHAIN_QUERY_RESP /
        META_LANGUAGE_RESPONSE back to the resolver that awaits it.

        Called from cognitive_worker's sync bus loop. Looks up the
        correlation_id in the pending registry and resolves the matching
        Future thread-safely. The resolver's awaiting coroutine then
        completes inside the dispatch loop and `_run_resolver` emits the
        META_REASON_RESPONSE.

        Returns True if the response was correlated to a pending Future
        (regardless of resolver outcome). False if the correlation_id is
        unknown (stale / duplicate / not from us).
        """
        payload = msg.get("payload") or {}
        correlation_id = payload.get("correlation_id")
        if not correlation_id:
            return False
        ok = self._pending_registry.resolve(correlation_id, payload)
        if ok:
            self._stats["responses_correlated"] += 1
        else:
            self._stats["responses_uncorrelated"] += 1
        return ok

    def sweep_timeouts(self) -> int:
        """Time out any pending dispatches older than `_dispatch_timeout_s`.

        Called periodically from cognitive_worker (e.g., every Nth bus tick).
        For each stale correlation_id, resolves the Future with a
        timeout-shaped payload so the resolver's `asyncio.wait_for` raises
        TimeoutError and `_run_resolver` emits failure_mode=resolver_timeout.

        Returns the number of timeouts swept.
        """
        stale = self._pending_registry.sweep_timeouts(self._dispatch_timeout_s)
        for cid in stale:
            # Synthetic timeout payload — resolver will see this as a
            # non-response and surface resolver_timeout. We could also
            # cancel the resolver coroutine outright, but resolving with a
            # synthetic timeout payload keeps the codepath uniform.
            self._pending_registry.resolve(
                cid, {"correlation_id": cid, "_timeout": True})
        return len(stale)

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

        # Phase A (RFP_cgn_enhancements §9.1) — concept-grounding route.
        # If the request carries a grounding_payload and a sink is wired, hand
        # it to the chain-trigger queue (should_trigger_meta Path #0) instead
        # of the one-shot resolver, and ack the consumer. This is the learning-
        # event → Level-A chain trigger. Skips the thought-cache (each grounding
        # is a unique trigger, not a re-answerable query).
        grounding = payload.get("grounding_payload")
        if grounding and self._grounding_sink is not None:
            try:
                self._grounding_sink({
                    "consumer": consumer_id,
                    "concept_id": str(grounding.get("concept_id", ""))[:128],
                    "felt_state_ref": grounding.get("felt_state_ref"),
                    "context_vector": list(context_vector),
                    "question_type": question_type,
                    "entry_primitive": QUESTION_TYPE_TO_PRIMITIVE.get(
                        question_type, ""),
                    "request_id": request_id,
                })
                self._stats["grounding_enqueued"] = (
                    self._stats.get("grounding_enqueued", 0) + 1)
            except Exception as _gs_err:
                logger.warning(
                    "[MetaService] grounding_sink failed for %s.%s: %s",
                    consumer_id, question_type, _gs_err)
            # Ack — the chain runs async; the consumer does not block on it.
            self._emit_response(
                request_id=request_id,
                consumer_id=consumer_id,
                src=src,
                failure_mode=None,
                insight={"grounding_queued": True},
                reason="grounding_enqueued",
            )
            return None

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

        # Session 3 live-dispatch (RFP_meta-reasoning_CGN_FIX.md §4.1). The
        # resolver runs on the dedicated asyncio loop, publishes a downstream
        # bus event, awaits the response Future, and emits the
        # META_REASON_RESPONSE asynchronously. _dispatch_to_resolver falls
        # back to _resolve_dry_run if the question_type has no primitive
        # mapping or the recruitment layer is unwired — preserving Session 1
        # behavior for backward compatibility.
        self._dispatch_to_resolver(request_id)
        return None

    def _resolve_dry_run(self, request_id: str) -> None:
        """Session 1: drain the queue by returning not_yet_implemented."""
        with self._lock:
            entry = self._pending.pop(request_id, None)
            try:
                self._queue.remove(request_id)
            except ValueError as _swallow_exc:
                swallow_warn('[logic.meta_service] MetaService._resolve_dry_run: self._queue.remove(request_id)', _swallow_exc,
                             key='logic.meta_service.MetaService._resolve_dry_run.line444', throttle=100)
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
                swallow_warn('[MetaService] recruitment stats error', e,
                             key="logic.meta_service.recruitment_stats_error", throttle=100)

        rewards_stats = None
        if self._rewards is not None:
            try:
                rewards_stats = self._rewards.get_stats()
            except Exception as e:
                swallow_warn('[MetaService] rewards stats error', e,
                             key="logic.meta_service.rewards_stats_error", throttle=100)

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
