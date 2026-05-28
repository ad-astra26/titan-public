"""
Meta-Recruitment Resolvers — Session 3 LIVE DISPATCH
(RFP_meta-reasoning_CGN_FIX.md PART A §4.1 + §4.2).

Each of the 10 KNOWN_RESOLVER_CATEGORIES is bound to an **async coroutine**
that publishes a downstream bus event (CGN_KNOWLEDGE_REQ with `kind`
discriminator, OR TIMECHAIN_QUERY for the timechain category) tagged
with a fresh correlation_id, registers a pending Future via the
MetaService PendingResponseRegistry, and awaits the matching response
within `resolver_dispatch_timeout_s` (default 5.0s per Preamble G19).

SPEC anchors:
  - SPEC §8.2 D-SPEC-42 + D-SPEC-52 — targeted dst routing; self-target
    (resolver in cognitive_worker.meta_service → cognitive_worker for kinds
    meta_wisdom / chain_archive / language; → memory for episodic_memory /
    semantic_graph) delivers correctly. Post-D8-3 spirit_worker retirement
    (commit 72f95a6b 2026-05-16) all resolver dispatch lives in cognitive_worker.
  - SPEC §8.0.ter D-SPEC-48 — publish-non-blocking; resolvers use
    `send_queue.put_nowait` only (never blocking I/O on the dispatch
    event loop thread).
  - SPEC Preamble G19 — async response path, timeout ≤5s.
  - SPEC Preamble G22 — no new sync `bus.request` patterns. The
    pending-future correlation pattern is event-publish + async-await,
    not a new RPC handler.

Replaces the Session 2 SHELL resolvers that returned
`{"action": "deferred_to_chain", ...}` markers. Session 3 dispatches
real bus events and returns the target worker's actual computed output,
which the consumer's outcome computer (Session 2 signed-outcome path)
then folds into the α-blended dynamic reward (rFP §4.4).

Worker target table (per current SPEC §9.B post fleet Phase C migration
2026-05-14):

  category               kind                target worker          handler
  --------               ----                -------------          -------
  reasoning              "reasoning"         cognitive_worker       Chunk B.3
  pattern_primitives     "pattern_primitives"cognitive_worker       Chunk B.3
  chain_archive          "chain_archive"     cognitive_worker       Chunk B.3
  prediction_engine      "prediction"        self_reflection_worker Chunk B.4
  self_reasoning         "self_reasoning"    self_reflection_worker Chunk B.4
  meta_wisdom            "meta_wisdom"       cognitive_worker       Chunk B.5 ext
  episodic_memory        "episodic_memory"   memory                 Chunk B.5 ext
  semantic_graph         "semantic_graph"    memory                 Chunk B.5 ext
  language_reasoner      "language"          cognitive_worker       Chunk B.5 ext
  timechain              (uses TIMECHAIN_QUERY)  kernel-rs           existing
  (Post-D8-3 2026-05-16: spirit_worker retired — all dispatch lives in
  cognitive_worker (MetaService instance), memory_worker, or kernel-rs.)
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional

from titan_hcl import bus

logger = logging.getLogger(__name__)


# Resolver dispatch timeout matches MetaService._dispatch_timeout_s default.
# Could be wired from config — kept as a module constant here so resolvers
# can be unit-tested without a full MetaService instance.
_RESOLVER_DISPATCH_TIMEOUT_S = 5.0


# ── Shared async dispatch helper ─────────────────────────────────────


def _make_bus_dispatch_resolver(
    *,
    category: str,
    bus_event: str,
    dst_worker: str,
    valid_names: Optional[set] = None,
    payload_kind: Optional[str] = None,
    # RFP_meta-reasoning_CGN_FIX.md Chunk B.7b Fix1c: src is the virtual
    # "meta_service" identity. cognitive_worker registers "meta_service"
    # as a BUS_SUBSCRIBE alias per SPEC v1.3.0 D-SPEC multi-name so
    # handler responses with dst=msg.src deliver back here regardless of
    # which physical worker hosts the MetaService instance. Future
    # MetaService relocations only update the alias registration site.
    src_name: str = "meta_service",
    send_queue: Any,
    pending_registry: Any,
    timeout_s: float = _RESOLVER_DISPATCH_TIMEOUT_S,
) -> Callable:
    """Build an async resolver coroutine that publishes a bus event with a
    fresh correlation_id and awaits the matching response Future.

    Args:
      category: resolver category for the recruiter prefix (e.g., "reasoning")
      bus_event: bus message type to publish (CGN_KNOWLEDGE_REQ or
        TIMECHAIN_QUERY)
      dst_worker: targeted dst per SPEC §8.2 (cognitive_worker /
        self_reflection_worker / spirit / kernel-rs / etc.)
      valid_names: optional set of acceptable post-dot recruiter names
        for early validation; None = accept any name
      payload_kind: optional `kind` discriminator in the payload for
        CGN_KNOWLEDGE_REQ multiplexing (e.g., "reasoning"). For
        TIMECHAIN_QUERY this is None (kind not used).
      src_name: source name placed in the bus envelope (default
        "meta_service" — matches MetaService dispatcher identity)
      send_queue: cognitive_worker's send_queue (mp.Queue-like) — must
        support put_nowait per §8.0.ter D-SPEC-48
      pending_registry: MetaService._PendingResponseRegistry instance —
        resolvers register/await Futures here
      timeout_s: dispatch timeout per Preamble G19 (≤5s)

    Returns the async coroutine `_resolve(name, ctx)`.
    """
    def _factory():
        async def _resolve(name: str, ctx: Optional[dict] = None
                           ) -> Dict[str, Any]:
            name = (name or "").strip()
            ctx = ctx or {}
            recruiter = f"{category}.{name}" if name else category

            # Optional name validation (early rejection for unknown
            # primitive sub-keys — cheap, doesn't burn bus traffic).
            if valid_names is not None and name and name not in valid_names:
                return {
                    "success": False,
                    "output": None,
                    "recruiter": recruiter,
                    "reason": f"unknown_name:{name}",
                }

            # Generate correlation_id + register pending Future BEFORE
            # publishing so a fast response can't race ahead.
            cid = pending_registry.next_correlation_id()
            fut = pending_registry.register(cid, meta={
                "category": category,
                "kind": payload_kind,
                "name": name,
            })

            # Build the bus payload. CGN_KNOWLEDGE_REQ carries `kind`
            # discriminator + name + context; TIMECHAIN_QUERY carries
            # query parameters lifted from ctx.
            if bus_event == bus.CGN_KNOWLEDGE_REQ:
                payload = {
                    "kind": payload_kind,
                    "name": name,
                    "correlation_id": cid,
                    "consumer_id": ctx.get("consumer_id", ""),
                    "question_type": ctx.get("question_type", ""),
                    "primitive": ctx.get("primitive", ""),
                    "sub_mode": ctx.get("sub_mode", ""),
                    "context_vector": ctx.get("context_vector") or [],
                    # Per spirit_worker P8 D8.4 legacy compat — populate
                    # requestor + topic for spirit_worker-hosted kinds
                    # that share the handler path.
                    "requestor": "meta_service",
                    "topic": ctx.get("payload_snippet", "")[:64],
                }
            elif bus_event == bus.TIMECHAIN_QUERY:
                payload = {
                    "op": name or "recall",  # recall/check/compare/aggregate/similar
                    "correlation_id": cid,
                    "consumer_id": ctx.get("consumer_id", ""),
                    "question_type": ctx.get("question_type", ""),
                    "topic": ctx.get("payload_snippet", "")[:64],
                    "limit": 10,
                }
            else:
                payload = {
                    "correlation_id": cid,
                    "name": name,
                    "ctx": ctx,
                }

            try:
                send_queue.put_nowait({
                    "type": bus_event,
                    "src": src_name,
                    "dst": dst_worker,
                    "payload": payload,
                })
            except Exception as e:
                pending_registry.discard(cid)
                logger.warning(
                    "[meta_resolvers] %s publish to dst=%s failed: %s",
                    category, dst_worker, e)
                return {
                    "success": False,
                    "output": None,
                    "recruiter": recruiter,
                    "reason": f"publish_failed: {type(e).__name__}",
                    "failure_mode": "resolver_error",
                }

            # Await the response Future. Timeout per Preamble G19.
            try:
                response = await asyncio.wait_for(fut, timeout=timeout_s)
            except asyncio.TimeoutError:
                pending_registry.discard(cid)
                return {
                    "success": False,
                    "output": None,
                    "recruiter": recruiter,
                    "reason": "dispatch_timeout",
                    "failure_mode": "resolver_timeout",
                }

            # MetaService.sweep_timeouts() resolves stale futures with a
            # synthetic _timeout=True marker. Surface that as
            # failure_mode=resolver_timeout too.
            if isinstance(response, dict) and response.get("_timeout"):
                return {
                    "success": False,
                    "output": None,
                    "recruiter": recruiter,
                    "reason": "sweep_timeout",
                    "failure_mode": "resolver_timeout",
                }

            # Successful response — unwrap the worker's output field if
            # present (Chunk B.3/B.4 handlers wrap output under
            # response.output; legacy paths may put it at top-level).
            if isinstance(response, dict):
                handler_output = response.get("output", response)
                handler_failure = response.get("failure")
                if handler_failure:
                    return {
                        "success": False,
                        "output": handler_output,
                        "recruiter": recruiter,
                        "reason": f"worker_reported_failure:{handler_failure}",
                    }
            else:
                handler_output = response

            return {
                "success": True,
                "output": handler_output,
                "recruiter": recruiter,
                "reason": "live_dispatch",
            }

        return _resolve

    return _factory()


# ── Individual resolver factories (Session 3) ─────────────────────────


def _make_reasoning_resolver(send_queue, pending_registry) -> Callable:
    """reasoning.{DECOMPOSE,COMPARE,CONTRAST,IF_THEN,ANALOGIZE,GENERALIZE,
    consistency_check} — dispatches to cognitive_worker (SPEC §9.B §4.A).
    """
    return _make_bus_dispatch_resolver(
        category="reasoning",
        bus_event=bus.CGN_KNOWLEDGE_REQ,
        dst_worker="cognitive_worker",
        payload_kind="reasoning",
        valid_names={"DECOMPOSE", "COMPARE", "CONTRAST", "IF_THEN",
                     "ANALOGIZE", "GENERALIZE", "consistency_check"},
        send_queue=send_queue,
        pending_registry=pending_registry,
    )


def _make_pattern_primitives_resolver(send_queue, pending_registry) -> Callable:
    """pattern_primitives.{extract_structure,merge,abstract,match,extrapolate}
    — dispatches to cognitive_worker."""
    return _make_bus_dispatch_resolver(
        category="pattern_primitives",
        bus_event=bus.CGN_KNOWLEDGE_REQ,
        dst_worker="cognitive_worker",
        payload_kind="pattern_primitives",
        valid_names={"extract_structure", "merge", "abstract",
                     "match", "extrapolate"},
        send_queue=send_queue,
        pending_registry=pending_registry,
    )


def _make_language_reasoner_resolver(send_queue, pending_registry) -> Callable:
    """language_reasoner.{formulate_query,...} — dispatches to
    cognitive_worker as a TRANSITIONAL target (no extracted language_worker
    yet per L2 separation strategy §4). Retargets when language_worker
    carve-out ships. NOT routed through spirit_worker per the D8 retirement
    invariant (`feedback_phase_c_spirit_worker_d8_retirement.md`)."""
    return _make_bus_dispatch_resolver(
        category="language_reasoner",
        bus_event=bus.CGN_KNOWLEDGE_REQ,
        dst_worker="cognitive_worker",
        payload_kind="language",
        send_queue=send_queue,
        pending_registry=pending_registry,
    )


def _make_meta_wisdom_resolver(send_queue, pending_registry) -> Callable:
    """meta_wisdom.{query_by_embedding,store_wisdom} — dispatches to
    cognitive_worker (MetaCGNConsumer lives inside MetaReasoningEngine
    whose canonical home is cognitive_worker per SPEC §9.B §4.A SHIPPED
    2026-05-05). NOT routed through spirit_worker per D8 retirement."""
    return _make_bus_dispatch_resolver(
        category="meta_wisdom",
        bus_event=bus.CGN_KNOWLEDGE_REQ,
        dst_worker="cognitive_worker",
        payload_kind="meta_wisdom",
        valid_names={"query_by_embedding", "store_wisdom"},
        send_queue=send_queue,
        pending_registry=pending_registry,
    )


def _make_chain_archive_resolver(send_queue, pending_registry) -> Callable:
    """chain_archive.{query,...} — dispatches to cognitive_worker (which
    owns the chain_archive DB)."""
    return _make_bus_dispatch_resolver(
        category="chain_archive",
        bus_event=bus.CGN_KNOWLEDGE_REQ,
        dst_worker="cognitive_worker",
        payload_kind="chain_archive",
        send_queue=send_queue,
        pending_registry=pending_registry,
    )


def _make_episodic_memory_resolver(send_queue, pending_registry) -> Callable:
    """episodic_memory.search — dispatches to memory_worker (Phase C
    SHIPPED 2026-05-13 Phase B per SPEC §3.4.1). Canonical home for
    episodic memory queries."""
    return _make_bus_dispatch_resolver(
        category="episodic_memory",
        bus_event=bus.CGN_KNOWLEDGE_REQ,
        dst_worker="memory",  # ModuleSpec name="memory" per plugin.py:638
        payload_kind="episodic_memory",
        send_queue=send_queue,
        pending_registry=pending_registry,
    )


def _make_semantic_graph_resolver(send_queue, pending_registry) -> Callable:
    """semantic_graph.neighbors — dispatches to memory_worker (Kuzu graph
    is owned by memory_worker per SPEC §9.B §3.4.1)."""
    return _make_bus_dispatch_resolver(
        category="semantic_graph",
        bus_event=bus.CGN_KNOWLEDGE_REQ,
        dst_worker="memory",  # ModuleSpec name="memory" per plugin.py:638
        payload_kind="semantic_graph",
        send_queue=send_queue,
        pending_registry=pending_registry,
    )


def _make_prediction_engine_resolver(send_queue, pending_registry) -> Callable:
    """prediction_engine (bare) — dispatches to self_reflection_worker
    (Track 2 v1.2.1 commit B8 canonical home for PredictionEngine)."""
    return _make_bus_dispatch_resolver(
        category="prediction_engine",
        bus_event=bus.CGN_KNOWLEDGE_REQ,
        dst_worker="self_reflection_worker",  # Fix3: bus name per plugin.py:978
        payload_kind="prediction",
        send_queue=send_queue,
        pending_registry=pending_registry,
    )


def _make_self_reasoning_resolver(send_queue, pending_registry) -> Callable:
    """self_reasoning.{predict,meta_audit} — dispatches to
    self_reflection_worker."""
    return _make_bus_dispatch_resolver(
        category="self_reasoning",
        bus_event=bus.CGN_KNOWLEDGE_REQ,
        dst_worker="self_reflection_worker",  # Fix3: bus name per plugin.py:978
        payload_kind="self_reasoning",
        valid_names={"predict", "meta_audit"},
        send_queue=send_queue,
        pending_registry=pending_registry,
    )


def _make_timechain_resolver(send_queue, pending_registry) -> Callable:
    """timechain.{recall,check,compare,aggregate,similar} — dispatches via
    existing TIMECHAIN_QUERY bus event to kernel-rs (Phase C L1)."""
    return _make_bus_dispatch_resolver(
        category="timechain",
        bus_event=bus.TIMECHAIN_QUERY,
        dst_worker="timechain",  # Fix3: ModuleSpec name="timechain" per plugin.py:1404 (NOT kernel-rs — TIMECHAIN_QUERY is handled by timechain_worker)
        payload_kind=None,  # TIMECHAIN_QUERY doesn't use `kind`
        valid_names={"recall", "check", "compare", "aggregate", "similar"},
        send_queue=send_queue,
        pending_registry=pending_registry,
    )


# ── Phase 9 (INV-Syn-22): RECALL → EngineRecall in-process routing ────

# Map the legacy resolver categories onto EngineRecall granularities (arch
# §13.2). These three categories carry the §13.2 RECALL sub-modes that have a
# direct EngineRecall granularity equivalent: episodic_specific→turn,
# semantic_neighbors→concept, chain_archive→archive. (procedural + autobiographical
# granularities are reachable directly on EngineRecall; they have no bus-resolver
# category to wrap here.)
_GRANULARITY_BY_CATEGORY: Dict[str, str] = {
    "episodic_memory": "turn",
    "semantic_graph": "concept",
    "chain_archive": "archive",
}


def _query_text_from_ctx(ctx: Optional[dict]) -> str:
    ctx = ctx or {}
    for key in ("query_text", "payload_snippet", "topic", "question"):
        v = ctx.get(key)
        if isinstance(v, str) and v.strip():
            return v
    return ""


def _make_engine_recall_resolver(
    *,
    category: str,
    granularity: str,
    engine_recall: Any,
    legacy_resolver: Callable,
    soft_fallback: bool,
) -> Callable:
    """Wrap a legacy bus-dispatch resolver so RECALL resolves in-process through
    the SC-op-backed EngineRecall first (INV-Syn-22). On a None result (engine
    disabled / no embedder / fork empty / failure) the wrapper falls back to the
    legacy bus resolver ONLY while `soft_fallback` is True (the parity-soak
    window; removed at cascade-close). EngineRecall.recall is sync + in-process
    (read-only handles, watermark-gated activation — NO sync bus.request)."""

    async def _resolve(name: str, ctx: Optional[dict] = None) -> Dict[str, Any]:
        recruiter = f"{category}.{(name or '').strip()}" if name else category
        query = _query_text_from_ctx(ctx)
        results = None
        if query:
            try:
                results = engine_recall.recall(query, granularity=granularity)
            except Exception as exc:  # pragma: no cover — defensive
                logger.warning(
                    "[meta_resolvers] EngineRecall(%s) raised: %s",
                    granularity, exc,
                )
                results = None
        if results is not None:
            import dataclasses
            out = [
                dataclasses.asdict(r) if dataclasses.is_dataclass(r) else r
                for r in results
            ]
            return {
                "success": True,
                "output": out,
                "recruiter": recruiter,
                "reason": f"engine_recall:{granularity}",
            }
        if soft_fallback:
            return await legacy_resolver(name, ctx)
        return {
            "success": False,
            "output": None,
            "recruiter": recruiter,
            "reason": f"engine_recall_empty:{granularity}",
        }

    return _resolve


# ── Category factory map ──────────────────────────────────────────────


_CATEGORY_FACTORIES: Dict[str, Callable[..., Callable]] = {
    "reasoning": _make_reasoning_resolver,
    "pattern_primitives": _make_pattern_primitives_resolver,
    "language_reasoner": _make_language_reasoner_resolver,
    "meta_wisdom": _make_meta_wisdom_resolver,
    "chain_archive": _make_chain_archive_resolver,
    "episodic_memory": _make_episodic_memory_resolver,
    "semantic_graph": _make_semantic_graph_resolver,
    "prediction_engine": _make_prediction_engine_resolver,
    "self_reasoning": _make_self_reasoning_resolver,
    "timechain": _make_timechain_resolver,
}


def register_default_resolvers(
    recruitment: Any,
    send_queue: Any = None,
    pending_registry: Any = None,
    # Legacy kwarg preserved for backward compat with callers that pass
    # reasoning_engine; unused in Session 3 dispatch (all categories now
    # use bus dispatch instead of in-process engine references).
    reasoning_engine: Any = None,
    # Phase 9 (INV-Syn-22): the cognitive_worker's read-only EngineRecall.
    # When provided + recall_engine_enabled, the episodic_memory / semantic_graph
    # / chain_archive categories resolve RECALL in-process through EngineRecall,
    # falling back to the legacy bus resolver while recall_soft_fallback is True
    # (the parity-soak window). When None, legacy dispatch is unchanged.
    engine_recall: Any = None,
    recall_engine_enabled: bool = True,
    recall_soft_fallback: bool = True,
) -> Dict[str, bool]:
    """Bind Session 3 live-dispatch resolvers to the MetaRecruitment catalog.
    Returns dict {category: registered_bool}.

    Session 3 contract: each resolver is an async coroutine bound to the
    MetaService dispatch loop. send_queue + pending_registry are REQUIRED
    for dispatch to work. If either is None, registration falls back to
    a graceful-error sync resolver that reports `resolver_unavailable`
    (preserves catalog_health_check stale=0 invariant).

    Idempotent — re-registering overwrites the previous binding for a
    category.
    """
    registered: Dict[str, bool] = {}
    if recruitment is None:
        logger.warning(
            "[meta_resolvers] register_default_resolvers called with "
            "recruitment=None — skipping")
        return registered

    if send_queue is None or pending_registry is None:
        # Fallback: register synchronous error resolvers so the catalog
        # still reports covered, but every dispatch returns failure_mode=
        # resolver_unavailable. This mirrors Session 2 behavior when
        # send_queue isn't wired (e.g., test harness without bus).
        logger.warning(
            "[meta_resolvers] send_queue=%s pending_registry=%s — "
            "registering graceful-error resolvers (Session 3 dispatch "
            "needs both wired)",
            "wired" if send_queue is not None else "None",
            "wired" if pending_registry is not None else "None")
        for category in _CATEGORY_FACTORIES.keys():
            recruitment.register_resolver(
                category, _make_unavailable_resolver(category))
            registered[category] = True
        return registered

    use_engine = engine_recall is not None and recall_engine_enabled
    for category, factory in _CATEGORY_FACTORIES.items():
        try:
            resolver = factory(
                send_queue=send_queue,
                pending_registry=pending_registry,
            )
            # Phase 9 INV-Syn-22: route RECALL categories through EngineRecall
            # in-process, with parity-soak fallback to the legacy resolver.
            if use_engine and category in _GRANULARITY_BY_CATEGORY:
                resolver = _make_engine_recall_resolver(
                    category=category,
                    granularity=_GRANULARITY_BY_CATEGORY[category],
                    engine_recall=engine_recall,
                    legacy_resolver=resolver,
                    soft_fallback=recall_soft_fallback,
                )
            recruitment.register_resolver(category, resolver)
            registered[category] = True
        except Exception as e:
            logger.warning(
                "[meta_resolvers] register %s failed: %s", category, e)
            registered[category] = False

    if use_engine:
        logger.info(
            "[meta_resolvers] INV-Syn-22: RECALL operator wired to EngineRecall "
            "for %s (soft_fallback=%s)",
            sorted(_GRANULARITY_BY_CATEGORY.keys()), recall_soft_fallback,
        )

    logger.info(
        "[meta_resolvers] Session 3 LIVE resolver registration: %d/%d "
        "categories bound (async bus-dispatch via correlation_id Futures)",
        sum(1 for v in registered.values() if v),
        len(_CATEGORY_FACTORIES))
    return registered


def _make_unavailable_resolver(category: str) -> Callable:
    """Sync fallback resolver that always reports unavailable — used when
    send_queue / pending_registry aren't wired (boot bring-up window or
    test harness)."""
    def _resolve(name: str, ctx: Optional[dict] = None) -> Optional[dict]:
        return {
            "success": False,
            "output": None,
            "recruiter": f"{category}.{name or ''}",
            "reason": "dispatch_infrastructure_not_wired",
            "failure_mode": "resolver_unavailable",
        }
    return _resolve


def get_supported_categories() -> list:
    """Return the list of categories this module can resolve.
    Used by tests + /v4/meta-service/recruitment for honesty checks."""
    return sorted(_CATEGORY_FACTORIES.keys())
