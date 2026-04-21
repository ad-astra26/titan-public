"""
knowledge_dispatcher — shared orchestrator for the knowledge pipeline.

Composes knowledge_router (classification) + knowledge_cache (persistence) +
knowledge_backends (direct REST) + optional StealthSageResearcher
(SearXNG-based backends) into a single async entry point used by both
knowledge_worker (subprocess consumer) and WebSearchHelper (agno-tool in
titan_main — wired in KP-3.5).

Per rFP_knowledge_pipeline_v2.md §3 — the architectural win is one canonical
pipeline for every external-knowledge query in the codebase.

KP-3 scope: dispatch flow + request coalescing (Essential A) + SearXNG
delegation to existing Sage. Error taxonomy lift + circuit breaker +
decision log + budget tracking wrap on top in KP-4.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from titan_plugin.logic.knowledge_backends import (
    BACKEND_REGISTRY,
    BackendResult,
    fetch_searxng_raw,
)
from titan_plugin.logic.knowledge_cache import KnowledgeCache
from titan_plugin.logic.knowledge_health import HealthTracker
from titan_plugin.logic.knowledge_router import (
    QueryType,
    classify_query,
    normalize_query,
    query_hash,
    route,
)
from titan_plugin.logic.knowledge_routing_learner import RoutingLearner

logger = logging.getLogger(__name__)


# ── Dispatch outcome ─────────────────────────────────────────────────

@dataclass
class DispatchResult:
    """End-to-end outcome of a single dispatch() call.

    `attempts` records (backend_name, outcome) pairs for every backend the
    dispatcher tried, in order. Outcome is "cache_hit", "success", the
    backend's error_type on fetch failure, or "skipped" when a backend in
    the chain isn't implemented (e.g. news_api without a key).

    Used by KP-4's decision log + KP-8's smart-routing learning.
    """
    topic: str
    normalized: str
    query_type: QueryType
    backend_used: str = ""
    cache_hit: bool = False
    result: Optional[BackendResult] = None
    rejected: bool = False                       # True if INTERNAL_REJECTED
    attempts: List[Tuple[str, str]] = field(default_factory=list)
    bytes_consumed_total: int = 0
    latency_ms_total: float = 0.0
    coalesced: bool = False                      # True if request_coalescing
                                                  # attached to an in-flight
                                                  # duplicate
    learned_chain: bool = False                   # True if RoutingLearner
                                                  # (KP-8) reordered the chain
                                                  # away from the static order

    @property
    def success(self) -> bool:
        return (self.result is not None
                and self.result.success
                and not self.rejected)


# ── Sage delegation adapter ──────────────────────────────────────────

# Optional dependency type — duck-typed so imports stay cheap. We treat
# anything with an async research(topic) -> str method as a Sage-shaped
# researcher.
SageLike = Any


async def _invoke_sage_backend(sage: SageLike, topic: str,
                                backend_name: str,
                                timeout: float) -> BackendResult:
    """Wrap sage.research(topic) as a BackendResult.

    Sage returns a string like "[SAGE_RESEARCH_FINDINGS]: <body>" on
    success, empty string on no-results/error. It already bundles
    discovery (SearXNG) + scrape + LLM distillation. We treat its entire
    pipeline as a single backend from the dispatcher's point of view.
    """
    result = BackendResult(backend=backend_name, query=topic)
    t0 = time.monotonic()
    try:
        findings = await asyncio.wait_for(sage.research(topic), timeout=timeout)
        result.latency_ms = round((time.monotonic() - t0) * 1000, 1)
        if findings:
            summary = (findings or "").replace(
                "[SAGE_RESEARCH_FINDINGS]: ", "").strip()
            if summary:
                result.success = True
                result.raw_text = summary
                result.bytes_consumed = len(summary.encode("utf-8"))
                return result
        result.error_type = "empty"
        return result
    except asyncio.TimeoutError:
        result.latency_ms = round((time.monotonic() - t0) * 1000, 1)
        result.error_type = "timeout"
        return result
    except Exception as e:
        result.latency_ms = round((time.monotonic() - t0) * 1000, 1)
        result.error_type = "network"
        result.error_msg = str(e)[:200]
        return result


# ── Per-backend query transformation ─────────────────────────────────

def _transform_for_backend(topic: str, qt: QueryType, backend: str) -> str:
    """Prepare the query string for a specific backend.

    Dictionary_phrase queries ("own meaning", "chi definition") need to
    drop the definition marker for REST dictionary backends, which look
    up single words. SearXNG and Wikipedia handle phrases directly.
    """
    if backend in ("wiktionary", "free_dictionary"):
        if qt == QueryType.DICTIONARY_PHRASE:
            tokens = topic.strip().lower().split()
            if tokens:
                return tokens[0]
    return topic


# ── Core dispatch ────────────────────────────────────────────────────

async def _run_chain(topic: str,
                     normalized: str,
                     qt: QueryType,
                     cache: Optional[KnowledgeCache],
                     sage: Optional[SageLike],
                     timeout_per_backend: float,
                     raw_results: bool = False,
                     searxng_url: str = "",
                     health: Optional[HealthTracker] = None,
                     learner: Optional[RoutingLearner] = None) -> DispatchResult:
    """Walk the routing chain until one backend yields a cacheable hit.

    Order of operations per backend:
      1. Cache lookup by (normalized, qt, backend) — immediate return on hit.
      2. Fetch via BACKEND_REGISTRY (direct REST) OR sage delegation
         (SearXNG family) OR skip if unimplemented.
      3. Cache the result (cache.put() short-circuits for 0-TTL errors).
      4. Return on success; otherwise continue to next backend.

    If a learner is wired, the static chain from route() is reordered by
    observed reputation per (qt, backend). Cold-start returns the static
    chain unchanged; warm stats surface promotions + demotions per rFP §3.4.

    If the entire chain exhausts, return a DispatchResult with result=None
    so the caller can decide whether to fall back to an unstructured path.
    """
    t_start = time.monotonic()
    out = DispatchResult(topic=topic, normalized=normalized, query_type=qt)
    static_chain = route(topic, qt)

    if not static_chain:
        out.latency_ms_total = round((time.monotonic() - t_start) * 1000, 1)
        return out

    # KP-8 smart routing learning: reorder chain by observed reputation.
    # learner.learned_chain() returns None when stats are cold so we fall
    # back to the static rFP §3.1 preference order.
    chain = static_chain
    if learner is not None:
        learned = learner.learned_chain(qt, static_chain)
        if learned is not None:
            chain = learned
            out.learned_chain = True

    for backend_name in chain:
        qhash = query_hash(normalized, qt, backend_name)

        # 0) Circuit breaker — skip backends that are currently OPEN
        if health is not None and not health.should_attempt(backend_name):
            out.attempts.append((backend_name, "circuit_open"))
            continue

        # 0b) Budget check — skip if daily bytes budget exhausted
        if health is not None and not health.check_budget(backend_name):
            out.attempts.append((backend_name, "budget_exceeded"))
            continue

        # 1) Cache
        if cache is not None:
            cached = cache.get(qhash)
            if cached is not None:
                out.backend_used = backend_name
                out.cache_hit = True
                out.attempts.append((backend_name, "cache_hit"))
                # Rehydrate a BackendResult from cached payload for uniform
                # downstream handling. Payload is the dict shape we put().
                import json as _json
                try:
                    payload = _json.loads(cached.result_json)
                except Exception:
                    payload = {}
                out.result = BackendResult(
                    backend=backend_name,
                    query=topic,
                    success=cached.success,
                    raw_text=payload.get("raw_text", ""),
                    structured=payload.get("structured"),
                    error_type=cached.error_type,
                    bytes_consumed=cached.bytes_consumed,
                    latency_ms=0.0,
                    status_code=payload.get("status_code", 0),
                )
                out.bytes_consumed_total += cached.bytes_consumed
                out.latency_ms_total = round(
                    (time.monotonic() - t_start) * 1000, 1)
                return out

        # 2) Fetch
        fetch_topic = _transform_for_backend(topic, qt, backend_name)
        fetcher: Optional[Callable[..., Awaitable[BackendResult]]] = (
            BACKEND_REGISTRY.get(backend_name))

        if fetcher is not None:
            result = await fetcher(fetch_topic, timeout=timeout_per_backend)
        elif (raw_results
              and backend_name.startswith("searxng")
              and searxng_url):
            # Raw mode (WebSearchHelper / KP-3.5): bypass Sage distillation,
            # return SearXNG hits as title/snippet/url blocks. Cheaper + no
            # LLM inference.
            result = await fetch_searxng_raw(
                fetch_topic, searxng_url=searxng_url,
                timeout=timeout_per_backend)
        elif backend_name.startswith("searxng") and sage is not None:
            result = await _invoke_sage_backend(
                sage, fetch_topic, backend_name, timeout=timeout_per_backend)
        else:
            # Backend named in chain but not implemented (e.g. news_api
            # without a key, or searxng_* with neither sage nor raw mode).
            out.attempts.append((backend_name, "skipped"))
            continue

        out.bytes_consumed_total += result.bytes_consumed

        # 3) Health telemetry — update circuit state + counters regardless
        # of cache outcome. Fires BEFORE cache write so a failed write
        # doesn't skew the health record.
        if health is not None:
            health.record_attempt(
                backend_name,
                success=result.success,
                error_type=result.error_type,
                bytes_consumed=result.bytes_consumed,
                latency_ms=result.latency_ms,
            )

        # 3b) Routing learner — record per-(qt, backend) outcome so future
        # chains can reorder by reputation. Quality score arrives from the
        # knowledge_worker gate later via learner.record_outcome() with a
        # refreshed quality arg; dispatcher sends 0.0 here so only attempt/
        # success/latency counters move. See KP-8.
        if learner is not None:
            learner.record_outcome(
                query_type=qt,
                backend=backend_name,
                success=result.success,
                latency_ms=result.latency_ms,
                quality=0.0,
            )

        # 4) Cache
        if cache is not None:
            payload_dict = {
                "raw_text": result.raw_text,
                "structured": result.structured,
                "status_code": result.status_code,
            }
            cache.put(
                query_hash=qhash,
                query_text=normalized,
                query_type=qt,
                backend=backend_name,
                result_payload=payload_dict,
                success=result.success,
                error_type=result.error_type,
                bytes_consumed=result.bytes_consumed,
            )

        out.attempts.append(
            (backend_name, "success" if result.success
                          else result.error_type or "unknown"))

        # 4) Success → return
        if result.success:
            out.backend_used = backend_name
            out.result = result
            out.latency_ms_total = round(
                (time.monotonic() - t_start) * 1000, 1)
            return out

    # Chain exhausted
    out.latency_ms_total = round((time.monotonic() - t_start) * 1000, 1)
    return out


async def dispatch(topic: str,
                   cache: Optional[KnowledgeCache] = None,
                   sage: Optional[SageLike] = None,
                   classify_override: Optional[QueryType] = None,
                   timeout_per_backend: float = 10.0,
                   inflight: Optional[Dict[str, asyncio.Future]] = None,
                   raw_results: bool = False,
                   searxng_url: str = "",
                   health: Optional[HealthTracker] = None,
                   learner: Optional[RoutingLearner] = None,
                   requestor: str = ""
                   ) -> DispatchResult:
    """Run the full knowledge-pipeline flow for a topic.

    Steps:
      1. Normalize + classify (skip if classify_override supplied).
      2. INTERNAL_REJECTED short-circuit — no cache, no fetch, no
         bandwidth. Caller inspects `.rejected` + emits its own telemetry.
      3. Request coalescing (Essential A) — if `inflight` dict is provided
         and the same normalized query is already running, await that
         future instead of launching a duplicate.
      4. Walk the routing chain (cache → fetch → cache-write → next).

    `inflight` is caller-owned. knowledge_worker runs sequential message
    processing so the dict will mostly stay empty; WebSearchHelper (KP-3.5)
    processes agno-tool calls concurrently and benefits more. The dict is
    keyed by normalized string + query-type so different types of the same
    string don't falsely coalesce.
    """
    t_start = time.monotonic()
    normalized = normalize_query(topic)

    if not normalized:
        # Empty/whitespace → treat as rejected
        return DispatchResult(
            topic=topic, normalized="",
            query_type=QueryType.INTERNAL_REJECTED,
            rejected=True,
            latency_ms_total=round((time.monotonic() - t_start) * 1000, 1),
        )

    qt = classify_override if classify_override is not None else classify_query(topic)

    if qt == QueryType.INTERNAL_REJECTED:
        rejected_out = DispatchResult(
            topic=topic, normalized=normalized, query_type=qt,
            rejected=True,
            latency_ms_total=round((time.monotonic() - t_start) * 1000, 1),
        )
        if health is not None:
            health.append_decision({
                "ts": time.time(),
                "requestor": requestor,
                "query": normalized,
                "query_type": qt.value,
                "rejected": True,
                "backend_used": "",
                "cache_hit": False,
                "bytes": 0,
                "latency_ms": rejected_out.latency_ms_total,
                "attempts": [],
            })
        return rejected_out

    # Near-duplicate telemetry (Optional E) — WARN when a caller sends a
    # semantically-same query with different hashes so we can tighten
    # normalization later.
    if health is not None:
        health.note_query_for_near_dup(normalized)

    # Coalesce only if caller provided an inflight dict.
    if inflight is not None:
        coalesce_key = f"{normalized}\x1f{qt.value}"
        existing = inflight.get(coalesce_key)
        if existing is not None and not existing.done():
            try:
                result = await existing
            except Exception as e:
                logger.warning(
                    "[KnowledgeDispatcher] Coalesced wait failed "
                    "key=%s err=%s", coalesce_key[:60], e)
                # Fall through to our own fetch attempt
            else:
                # Clone the shared result with coalesced=True
                return DispatchResult(
                    topic=topic,
                    normalized=result.normalized,
                    query_type=result.query_type,
                    backend_used=result.backend_used,
                    cache_hit=result.cache_hit,
                    result=result.result,
                    rejected=result.rejected,
                    attempts=list(result.attempts),
                    bytes_consumed_total=result.bytes_consumed_total,
                    latency_ms_total=round(
                        (time.monotonic() - t_start) * 1000, 1),
                    coalesced=True,
                )

        # No existing in-flight — we own this key.
        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        inflight[coalesce_key] = fut
        try:
            out = await _run_chain(topic, normalized, qt, cache, sage,
                                    timeout_per_backend,
                                    raw_results=raw_results,
                                    searxng_url=searxng_url,
                                    health=health,
                                    learner=learner)
            if not fut.done():
                fut.set_result(out)
            _maybe_log_decision(health, out, requestor)
            return out
        except Exception as e:
            if not fut.done():
                fut.set_exception(e)
            raise
        finally:
            # Remove after future resolution so late arrivals don't block
            # forever on a completed future. Keep briefly to let coalescers
            # already awaiting fut pick it up.
            inflight.pop(coalesce_key, None)

    # No coalescing requested — straight chain walk
    out = await _run_chain(topic, normalized, qt, cache, sage,
                            timeout_per_backend,
                            raw_results=raw_results,
                            searxng_url=searxng_url,
                            health=health,
                            learner=learner)
    _maybe_log_decision(health, out, requestor)
    return out


def _maybe_log_decision(health: Optional[HealthTracker],
                         out: DispatchResult,
                         requestor: str) -> None:
    """Write one decision-log entry if a HealthTracker is wired."""
    if health is None:
        return
    try:
        health.append_decision({
            "ts": time.time(),
            "requestor": requestor,
            "query": out.normalized,
            "query_type": out.query_type.value,
            "rejected": out.rejected,
            "backend_used": out.backend_used,
            "cache_hit": out.cache_hit,
            "coalesced": out.coalesced,
            "success": out.success,
            "bytes": out.bytes_consumed_total,
            "latency_ms": out.latency_ms_total,
            "attempts": out.attempts,
            "learned_chain": out.learned_chain,
            "error_type": (out.result.error_type
                           if out.result is not None else ""),
        })
    except Exception:
        # Decision log is best-effort — never let it break dispatch
        pass


def dispatch_sync(topic: str,
                  async_loop: asyncio.AbstractEventLoop,
                  cache: Optional[KnowledgeCache] = None,
                  sage: Optional[SageLike] = None,
                  timeout: float = 45.0,
                  timeout_per_backend: float = 10.0,
                  inflight: Optional[Dict[str, asyncio.Future]] = None,
                  raw_results: bool = False,
                  searxng_url: str = "",
                  health: Optional[HealthTracker] = None,
                  learner: Optional[RoutingLearner] = None,
                  requestor: str = ""
                  ) -> DispatchResult:
    """Synchronous adapter for sync callers (e.g. knowledge_worker's main loop).

    Dispatches dispatch() to `async_loop` via run_coroutine_threadsafe and
    blocks the caller up to `timeout` seconds. `timeout_per_backend` is the
    per-backend HTTP timeout passed into the async dispatch.
    """
    fut = asyncio.run_coroutine_threadsafe(
        dispatch(topic=topic, cache=cache, sage=sage,
                 timeout_per_backend=timeout_per_backend,
                 inflight=inflight, raw_results=raw_results,
                 searxng_url=searxng_url, health=health,
                 learner=learner,
                 requestor=requestor),
        async_loop)
    return fut.result(timeout=timeout)


__all__ = [
    "DispatchResult",
    "dispatch",
    "dispatch_sync",
]
