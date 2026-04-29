"""
Knowledge Worker — Guardian-supervised CGN consumer for knowledge acquisition.

The 4th CGN consumer: detects knowledge gaps, searches internal memory first,
then uses Stealth Sage (SearXNG → scrape → LLM distillation) for external
research. Quality-gated results are grounded as concepts and distributed to
other consumers via the CGN shared value landscape.

Bus protocol:
  CGN_KNOWLEDGE_REQ     (any → knowledge)  — knowledge gap request
  CGN_KNOWLEDGE_RESP    (knowledge → any)  — grounded knowledge response
  CGN_TRANSITION        (knowledge → cgn)  — experience transitions
  CGN_REGISTER          (knowledge → cgn)  — consumer registration at boot
  QUERY get_knowledge_stats                — on-demand stats

See: titan-docs/rFP_cgn_cognitive_kernel_v2.md §4
"""

import asyncio
import json
import logging
import os
import sqlite3
import sys
import threading
import time
from queue import Empty
from titan_plugin import bus

logger = logging.getLogger(__name__)

# ── Knowledge concepts table schema ──────────────────────────────────

_KNOWLEDGE_SCHEMA = """
CREATE TABLE IF NOT EXISTS knowledge_concepts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT NOT NULL,
    felt_tensor TEXT DEFAULT '[]',
    confidence REAL DEFAULT 0.0,
    source TEXT DEFAULT 'unknown',
    source_url TEXT,
    summary TEXT,
    encounter_count INTEGER DEFAULT 1,
    times_used INTEGER DEFAULT 0,
    quality_score REAL DEFAULT 0.0,
    associations TEXT DEFAULT '[]',
    neuromod_at_acquisition TEXT DEFAULT '{}',
    requesting_consumer TEXT,
    created_at REAL NOT NULL,
    last_used_at REAL,
    UNIQUE(topic)
);
CREATE INDEX IF NOT EXISTS idx_knowledge_topic ON knowledge_concepts(topic);
CREATE INDEX IF NOT EXISTS idx_knowledge_conf ON knowledge_concepts(confidence DESC);
"""

# ── Quality gate heuristic thresholds ────────────────────────────────

_MIN_SUMMARY_LENGTH = 30       # Reject trivially short distillations
_MIN_KEYWORD_OVERLAP = 0.15    # At least 15% of query words in summary
_MAX_SUMMARY_LENGTH = 3000     # Reject overly long (likely garbage)


def knowledge_worker_main(recv_queue, send_queue, name: str, config: dict) -> None:
    """Main loop for the Knowledge Worker process.

    Args:
        recv_queue: receives messages from DivineBus
        send_queue: sends messages back to DivineBus
        name: module name ("knowledge")
        config: dict from merged [knowledge] + [inference] + [stealth_sage]
    """
    project_root = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    logger.info("[KnowledgeWorker] Initializing Knowledge Gatherer...")
    init_start = time.time()

    # ── Database setup (auto-migration) ────────────────────────────────
    db_path = config.get("db_path", "data/inner_memory.db")
    _ensure_schema(db_path)

    # ── CGN consumer client (for policy inference) ─────────────────────
    from titan_plugin.logic.cgn_consumer_client import CGNConsumerClient
    cgn_client = CGNConsumerClient(
        "knowledge", send_queue=send_queue, module_name=name,
        state_dir=config.get("cgn_state_dir", "data/cgn"),
        shm_path=config.get("shm_path", "/dev/shm/cgn_live_weights.bin"))

    # ── Register "knowledge" consumer with CGN Worker ──────────────────
    _send_msg(send_queue, "CGN_REGISTER", name, "cgn", {
        "name": "knowledge",
        "feature_dims": 30,
        "action_dims": 6,
        "action_names": [
            "recall_internal",
            "research_shallow",
            "research_deep",
            "consolidate",
            "distribute",
            "defer",
        ],
        "reward_source": "downstream_usage",
        "max_buffer_size": 500,
        "consolidation_priority": 2,
    })

    # ── Stealth Sage researcher (async) ────────────────────────────────
    sage = None
    sage_config = _build_sage_config(config)
    try:
        from titan_plugin.logic.sage.researcher import StealthSageResearcher
        sage = StealthSageResearcher(sage_config)
        # Wire Ollama Cloud for distillation
        _wire_ollama_cloud(sage, config)
        logger.info("[KnowledgeWorker] Stealth Sage initialized "
                    "(searxng=%s)", sage._searxng_host)
    except Exception as e:
        logger.warning("[KnowledgeWorker] Stealth Sage init failed: %s "
                       "(internal recall only)", e)

    # ── Async event loop for Sage research (runs in thread) ────────────
    _async_loop = asyncio.new_event_loop()
    _async_thread = threading.Thread(
        target=_async_loop.run_forever, daemon=True, name="knowledge-async")
    _async_thread.start()

    # ── KP-2 persistent cache + KP-3 dispatcher ─────────────────────────
    # Replaces Stage A in-memory cache + internal-name filter. All cache
    # semantics (TTL, LRU, failure-caching) live in KnowledgeCache; all
    # routing + internal-rejection + backend-chain walking live in
    # knowledge_dispatcher. Worker's job is now: message dispatch + CGN
    # policy + internal recall + grounding.
    from titan_plugin.logic.knowledge_cache import KnowledgeCache
    from titan_plugin.logic.knowledge_dispatcher import (
        dispatch_sync, DispatchResult)
    from titan_plugin.logic.knowledge_health import HealthTracker
    from titan_plugin.logic.knowledge_router import (
        QueryType, classify_query)
    from titan_plugin.logic.knowledge_routing_learner import RoutingLearner
    _cache_path = config.get("search_cache_path", "data/search_cache.db")
    _cache_size_cap = int(config.get("search_cache_size_cap", 10_000))
    _knowledge_cache = KnowledgeCache(
        db_path=_cache_path, size_cap=_cache_size_cap)
    # KP-4 — health tracker (circuit breaker + budget + decision log +
    # near-dup). Budgets come from config's [knowledge_pipeline.budgets];
    # see KP-7 for the alert cascade. Defaults (0 = unlimited) ship first
    # so a misconfigured deploy doesn't hard-stop the pipeline.
    _health_path = config.get(
        "pipeline_health_path", "data/knowledge_pipeline_health.json")
    _decision_log_path = config.get(
        "pipeline_decision_log_path",
        "data/logs/knowledge_router_decisions.jsonl")
    _budgets_cfg = dict(config.get("budgets", {})) if config.get("budgets") else {}
    # budgets are stored as MB in config; HealthTracker wants bytes
    _budgets_bytes = {k: int(v) * 1024 * 1024 for k, v in _budgets_cfg.items()
                      if isinstance(v, (int, float))}
    # KP-7 alert cascade: HealthTracker invokes this whenever a budget
    # threshold (80% or 100%) is crossed, OR a backend's circuit breaker
    # transitions to OPEN. We fire a bus event (observability) + a Telegram
    # notification (Maker awareness). Dedup handled inside HealthTracker
    # (once per backend per day) and maker_notify (24h cooldown per class).
    _telegram_alerts_enabled = bool(
        config.get("telegram_alerts_enabled", True))

    def _kp_alert_callback(_kind: str, _backend: str, _ctx: dict) -> None:
        # Bus event — routed to "core" so logs / observatory pick it up.
        # INTENTIONAL_BROADCAST: observability-only, no subscriber handles.
        # Bus events ALWAYS fire (they drive arch_map + dashboard); the
        # kill-switch below only gates the Telegram hop.
        try:
            _evt_type = {
                "budget_warning": "SEARCH_PIPELINE_BUDGET_WARNING",
                "budget_exceeded": "SEARCH_PIPELINE_BUDGET_EXCEEDED",
                "circuit_open": "SEARCH_PIPELINE_DEGRADED",
            }.get(_kind, "SEARCH_PIPELINE_ALERT")
            _send_msg(send_queue, _evt_type, name, "core", {
                "kind": _kind, "backend": _backend, **_ctx,
            })
        except Exception as _bus_err:
            logger.debug("[KnowledgePipeline] alert bus publish: %s", _bus_err)
        # Telegram via maker_notify (dedup at-most-daily per class) — KP-7
        # kill-switch: [knowledge_pipeline].telegram_alerts_enabled = false
        # mutes Telegram without suppressing the bus event.
        if not _telegram_alerts_enabled:
            return
        try:
            from titan_plugin.utils.maker_notify import notify_maker
            _titan_id = config.get("titan_id",
                                   os.environ.get("TITAN_ID", "T?"))
            _pct = _ctx.get("pct", 0.0)
            _bytes_mb = _ctx.get("bytes_consumed", 0) / (1024 * 1024)
            _budget_mb = _ctx.get("budget_bytes", 0) / (1024 * 1024)
            if _kind == "budget_warning":
                _txt = (f"⚠ *Search pipeline* backend `{_backend}` at "
                        f"{_pct:.0f}% of daily budget "
                        f"({_bytes_mb:.1f}MB / {_budget_mb:.0f}MB). "
                        f"Will hard-stop at 100%.")
                notify_maker(f"search_budget_warn_{_backend}", _titan_id, _txt)
            elif _kind == "budget_exceeded":
                _txt = (f"🚨 *Search pipeline* backend `{_backend}` "
                        f"BUDGET EXCEEDED — {_bytes_mb:.1f}MB used, "
                        f"pipeline blocked until reset. "
                        f"POST /v4/search-pipeline/budget-reset to override.")
                notify_maker(f"search_budget_exceed_{_backend}", _titan_id,
                             _txt, force=False)
            elif _kind == "circuit_open":
                _txt = (f"⚠ *Search pipeline* backend `{_backend}` circuit "
                        f"OPENED — {_ctx.get('consecutive_errors', 0)} "
                        f"consecutive errors (last: "
                        f"`{_ctx.get('last_error_type', '?')}`). "
                        f"Auto-probe in 5 min.")
                notify_maker(f"search_circuit_{_backend}", _titan_id, _txt)
        except Exception as _tg_err:
            logger.debug(
                "[KnowledgePipeline] Telegram notify failed: %s", _tg_err)

    _health = HealthTracker(
        health_path=_health_path,
        decision_log_path=_decision_log_path,
        budgets=_budgets_bytes,
        on_alert=_kp_alert_callback,
    )
    # KP-8 routing learner — per (query_type, backend) reputation tracking
    # with 7-day rolling window. Dispatcher consults learner.learned_chain()
    # at every dispatch to reorder the static rFP §3.1 chain. Cold-start
    # (any backend with < min_samples in the window) falls back to static.
    # KP-8.1 — chain_reordered events → SEARCH_PIPELINE_CHAIN_REORDERED bus
    # so observatory + arch_map see promotion/demotion decisions.
    def _kp_routing_event_callback(_kind: str, _ctx: dict) -> None:
        try:
            _evt_type = {
                "chain_reordered": "SEARCH_PIPELINE_CHAIN_REORDERED",
            }.get(_kind, "SEARCH_PIPELINE_LEARNER_EVENT")
            # INTENTIONAL_BROADCAST: observability-only; observatory + arch_map
            # subscribe. Payload includes static/reordered chains + demoted list.
            _send_msg(send_queue, _evt_type, name, "core", {
                "kind": _kind, **_ctx,
            })
        except Exception as _re_err:
            logger.debug(
                "[KnowledgePipeline] routing event publish: %s", _re_err)

    _routing_learner = RoutingLearner(
        db_path=config.get("routing_stats_path",
                            "data/knowledge_routing_stats.db"),
        window_days=int(config.get("routing_learner_window_days", 7)),
        min_samples=int(config.get("routing_learner_min_samples", 10)),
        demote_threshold=float(config.get(
            "routing_learner_demote_threshold", 0.10)),
        enabled=bool(config.get("routing_learner_enabled", True)),
        on_event=_kp_routing_event_callback,
    )
    # In-flight coalescing dict lives inside the async loop's thread;
    # sequential message handling means it mostly stays empty but it
    # protects against overlapping dispatch_sync calls if Sage research
    # triggers nested dispatches in the future.
    _dispatch_inflight: dict = {}

    # ── Stats ──────────────────────────────────────────────────────────
    _stats = {
        "requests_received": 0,
        "internal_recalls": 0,
        "external_researches": 0,
        "concepts_grounded": 0,
        "quality_rejected": 0,
        "deferred": 0,
        "cache_hits": 0,                    # from router cache, not Stage A
        "rejected_internal_names": 0,       # from router.INTERNAL_REJECTED
        "backend_attempts": {},             # {backend_name: int} per-session
    }
    _last_heartbeat = time.time()
    _heartbeat_interval = 5.0

    # ── META-CGN Producer #10 EdgeDetector: knowledge.concept_grounded ──
    # Primed from DB at boot so already-grounded topics don't re-fire.
    # observe_first_time(topic) ensures one emission per unique topic per
    # Titan lifetime. No persistence file — DB knowledge_concepts IS durable.
    _p10_edge_detector = None
    try:
        from titan_plugin.logic.meta_cgn import EdgeDetector as _P10EdgeDet
        _p10_edge_detector = _P10EdgeDet()
        _p10_primed = 0
        try:
            _p10_conn = sqlite3.connect(db_path, timeout=5.0)
            for _p10_row in _p10_conn.execute(
                    "SELECT topic FROM knowledge_concepts"):
                _p10_edge_detector._seen.add(str(_p10_row[0]))
                _p10_primed += 1
            _p10_conn.close()
        except Exception as _p10_prime_err:
            logger.debug("[META-CGN] Producer #10 prime query: %s", _p10_prime_err)
        logger.info(
            "[META-CGN] Producer #10 EdgeDetector primed (%d topics already grounded)",
            _p10_primed)
    except Exception as _p10_init_err:
        logger.warning(
            "[META-CGN] Producer #10 EdgeDetector init failed: %s — "
            "knowledge.concept_grounded will not emit", _p10_init_err)

    # Stage A's in-memory cache + internal-name filter are REMOVED here —
    # both responsibilities now live in knowledge_cache + knowledge_router.
    # KP-3 cutover 2026-04-20. See rFP_knowledge_pipeline_v2.md §6.

    # ── Background heartbeat thread (keeps Guardian alive during long Sage research) ──
    _hb_stop = threading.Event()

    def _heartbeat_loop():
        while not _hb_stop.is_set():
            _send_heartbeat(send_queue, name)
            _hb_stop.wait(15.0)

    _hb_thread = threading.Thread(target=_heartbeat_loop, daemon=True,
                                  name="knowledge-heartbeat")
    _hb_thread.start()

    init_ms = (time.time() - init_start) * 1000
    logger.info("[KnowledgeWorker] Ready in %.0fms (sage=%s, db=%s)",
                init_ms, "OK" if sage else "DISABLED", db_path)
    _send_msg(send_queue, bus.MODULE_READY, name, "guardian", {})

    # ── F-phase (rFP §16.3): Meta-Reasoning Consumer Service wire ────
    # Session 2 wire-now-gate-later: request at action-dispatch time,
    # outcome after research concludes (neutral 0.0 reward in Session 2
    # since advice not applied; Session 3 switches to
    # compute_outcome_knowledge(research_result)).
    _kn_meta_pending: dict = {}  # request_id → (t_sent, topic, pre_conf)
    try:
        from titan_plugin.logic.meta_service_client import (
            register_response_handler as _kn_register_mrh,
        )

        def _kn_meta_response_handler(payload: dict) -> None:
            req_id = payload.get("request_id", "")
            failure = payload.get("failure_mode")
            if failure:
                logger.info(
                    "[KnMeta] response req_id=%s failure=%s "
                    "(dry-run expected)", req_id[:8], failure)
            else:
                insight = payload.get("insight") or {}
                logger.info(
                    "[KnMeta] response req_id=%s sugg=%s",
                    req_id[:8],
                    insight.get("suggested_action") if insight else None)
            _kn_meta_pending.pop(req_id, None)

        _kn_register_mrh("knowledge", _kn_meta_response_handler)
        logger.info("[KnowledgeWorker] F-phase meta response handler registered")
    except Exception as _knh_err:
        logger.warning(
            "[KnowledgeWorker] Meta response handler registration: %s",
            _knh_err)

    # ── Main loop ──────────────────────────────────────────────────────
    # ── Microkernel v2 Phase B.1 §6 — readiness/hibernate reporter ──
    from titan_plugin.core.readiness_reporter import trivial_reporter
    def _b1_save_state():
        return []
    _b1_reporter = trivial_reporter(
        worker_name=name, layer="L3", send_queue=send_queue,
        save_state_cb=_b1_save_state,
    )

    while True:
        try:
            msg = recv_queue.get(timeout=_heartbeat_interval)
        except Empty:
            _send_heartbeat(send_queue, name)
            _last_heartbeat = time.time()
            continue

        msg_type = msg.get("type", "")
        payload = msg.get("payload", {})

        # Heartbeat
        now = time.time()
        if now - _last_heartbeat > _heartbeat_interval:
            _send_heartbeat(send_queue, name)
            _last_heartbeat = now

        # ── CGN_KNOWLEDGE_REQ — knowledge gap request ─────────────
        if msg_type == bus.CGN_KNOWLEDGE_REQ:
            # Heartbeat immediately — this handler can take 30s+
            _send_heartbeat(send_queue, name)
            _last_heartbeat = time.time()
            try:
                # P8 compat: accept both "topic" and "query" (META-CGN emits query)
                topic = (payload.get("topic") or payload.get("query")
                         or "").strip()
                requestor = payload.get("requestor", msg.get("src", "?"))
                urgency = float(payload.get("urgency", 0.5))
                neuromods = payload.get("neuromods", {})
                # P8 I-P8.2: correlate responses to pending requests via request_id
                _request_id = str(payload.get("request_id", ""))

                if not topic:
                    logger.debug("[KnowledgeWorker] Empty topic, skipping")
                    continue

                # Pre-classification via router — rejects Titan-internal
                # names upfront (supersedes Stage A's inline filter) and
                # establishes the query_type used by the dispatcher for
                # backend selection + cache TTL.
                _qt = classify_query(topic)
                if _qt == QueryType.INTERNAL_REJECTED:
                    logger.warning(
                        "[KnowledgeWorker] Rejected internal-name query: '%s' "
                        "(requestor=%s) — Titan-internal names not searchable "
                        "externally.", topic[:60], requestor)
                    _stats["rejected_internal_names"] = (
                        _stats.get("rejected_internal_names", 0) + 1)
                    continue

                _stats["requests_received"] += 1
                logger.info("[KnowledgeWorker] Request: '%s' from %s "
                            "(urgency=%.2f, type=%s)", topic[:60], requestor,
                            urgency, _qt.value)

                # Phase 1: Internal recall (always first)
                internal = _search_internal(db_path, topic)

                if internal and internal[0]["confidence"] > 0.4:
                    # Known with decent confidence — distribute
                    _stats["internal_recalls"] += 1
                    _distribute(send_queue, name, requestor, topic,
                                internal[0], source="internal_recall",
                                request_id=_request_id)
                    _send_transition(send_queue, name, cgn_client,
                                     topic, 0, neuromods,
                                     reward=0.0)  # Delayed reward
                    logger.info("[KnowledgeWorker] Internal recall: '%s' "
                                "(conf=%.2f)", topic[:40],
                                internal[0]["confidence"])
                    continue

                # Phase 2: CGN policy decides action
                action_result = {}
                try:
                    action_result = cgn_client.infer_action(
                        sensory_ctx={
                            "neuromods": neuromods,
                            "epoch": payload.get("epoch", 0),
                        },
                        features={
                            "topic_novelty": 1.0 - (
                                internal[0]["confidence"]
                                if internal else 0.0),
                            "urgency": urgency,
                        })
                except Exception:
                    pass  # New consumer, no weights yet

                action_name = action_result.get("action_name", "research_shallow")
                action_idx = action_result.get("action_index", 1)

                # ── F-phase (rFP §16.3): consult meta on action choice ──
                # Session 2 dry-run: meta returns not_yet_implemented;
                # knowledge falls back to CGN policy (above). Accumulates
                # request traffic for Session 3 chain execution.
                _kn_req_id = ""
                try:
                    from titan_plugin.logic.meta_service_client import (
                        send_meta_request as _kn_send,
                    )
                    from titan_plugin.logic.meta_consumer_contexts import (
                        build_knowledge_meta_context_30d as _kn_ctx,
                    )
                    _kn_pre_conf = (internal[0]["confidence"]
                                    if internal else 0.0)
                    _kn_req_id = _kn_send(
                        consumer_id="knowledge",
                        question_type="formulate_strategy",
                        context_vector=_kn_ctx(
                            topic=topic,
                            confidence={"internal": _kn_pre_conf},
                            urgency=urgency,
                            neuromods=neuromods),
                        time_budget_ms=3000,
                        constraints={
                            "confidence_threshold": 0.3,
                            "allow_timechain_query": True,
                        },
                        payload_snippet=f"topic={topic[:40]} action={action_name}",
                        send_queue=send_queue, src=name)
                    _kn_meta_pending[_kn_req_id] = (
                        time.time(), topic[:40], _kn_pre_conf)
                except Exception as _kn_err:
                    logger.debug("[KnMeta] pre-research request skipped: %s",
                                 _kn_err)

                if action_name == "defer":
                    _stats["deferred"] += 1
                    logger.info("[KnowledgeWorker] Deferred: '%s'",
                                topic[:40])
                    _send_transition(send_queue, name, cgn_client,
                                     topic, 5, neuromods, reward=0.0)
                    continue

                if action_name in ("research_shallow", "research_deep"):
                    # External research via knowledge_dispatcher — walks the
                    # router's backend chain for _qt (cache → direct REST →
                    # Sage delegation). Sage is still the fallback for
                    # conceptual/technical queries; dictionary/wikipedia
                    # queries bypass Sage entirely via direct REST backends.
                    _stats["external_researches"] += 1
                    _send_heartbeat(send_queue, name)
                    try:
                        dispatch_res: DispatchResult = dispatch_sync(
                            topic=topic,
                            async_loop=_async_loop,
                            cache=_knowledge_cache,
                            sage=sage,
                            timeout=45.0,
                            timeout_per_backend=10.0,
                            inflight=_dispatch_inflight,
                            health=_health,
                            learner=_routing_learner,
                            requestor=requestor,
                        )
                    except Exception as _disp_err:
                        logger.warning("[KnowledgeWorker] Dispatch error: %s",
                                       _disp_err)
                        dispatch_res = None

                    # Track per-backend attempt telemetry (feeds KP-4 health)
                    if dispatch_res is not None:
                        for _bk, _outcome in dispatch_res.attempts:
                            _stats["backend_attempts"][_bk] = (
                                _stats["backend_attempts"].get(_bk, 0) + 1)
                        if dispatch_res.cache_hit:
                            _stats["cache_hits"] += 1

                    # Translate DispatchResult → legacy (findings, source) form
                    findings = ""
                    _source_backend = "dispatcher"
                    if (dispatch_res is not None
                            and dispatch_res.result is not None
                            and dispatch_res.result.success):
                        findings = dispatch_res.result.raw_text
                        _source_backend = dispatch_res.backend_used or "dispatcher"

                    if findings:
                        # Heartbeat after research completes
                        _send_heartbeat(send_queue, name)
                        _last_heartbeat = time.time()

                        # summary is what the backend gave us. For SearXNG
                        # backends Sage already stripped the prefix in
                        # _invoke_sage_backend; direct REST backends return
                        # plain text.
                        summary = findings

                        # Quality gate
                        quality = _quality_score(topic, summary)
                        if quality < 0.3:
                            _stats["quality_rejected"] += 1
                            logger.info("[KnowledgeWorker] Quality rejected "
                                        "'%s' (score=%.2f, backend=%s)",
                                        topic[:40], quality, _source_backend)
                            _send_transition(
                                send_queue, name, cgn_client,
                                topic, action_idx, neuromods,
                                reward=-0.05)
                            continue

                        # Ground as concept — `source` field records the
                        # actual backend used so downstream analytics see
                        # wiktionary/wikipedia_direct/searxng_* distinctly.
                        concept = _ground_concept(
                            db_path, topic, summary, quality,
                            source=_source_backend,
                            requestor=requestor,
                            neuromods=neuromods)
                        _stats["concepts_grounded"] += 1
                        # KP-8 — feed quality back to routing learner so
                        # reputation reflects real downstream-usable signal.
                        if (_source_backend
                                and _source_backend != "dispatcher"
                                and dispatch_res is not None):
                            _routing_learner.record_quality(
                                dispatch_res.query_type,
                                _source_backend, quality)

                        # ── META-CGN producer #10: knowledge.concept_grounded ──
                        # v3 Phase D rollout (rFP § 12 row 10). Fires exactly
                        # once per first grounding of a topic. MONOCULTURE-AWARE
                        # weights deviate from rFP spec — see meta_cgn.py mapping.
                        try:
                            if _p10_edge_detector is not None and \
                                    _p10_edge_detector.observe_first_time(str(topic)):
                                from titan_plugin.bus import emit_meta_cgn_signal
                                _p10_sent = emit_meta_cgn_signal(
                                    send_queue,
                                    src="knowledge",
                                    consumer="knowledge",
                                    event_type="concept_grounded",
                                    intensity=min(1.0, max(0.1, float(quality))),
                                    domain=str(topic)[:40],
                                    reason=f"first grounding of topic '{topic}' "
                                           f"(quality={quality:.2f}, requestor={requestor})",
                                )
                                if _p10_sent:
                                    logger.info(
                                        "[META-CGN] knowledge.concept_grounded EMIT — "
                                        "topic=%s quality=%.2f requestor=%s",
                                        topic, quality, requestor)
                                else:
                                    logger.warning(
                                        "[META-CGN] Producer #10 knowledge.concept_grounded DROPPED by bus "
                                        "— topic=%s (rate-gate or queue-full; signal missed)",
                                        topic)
                        except Exception as _p10_err:
                            logger.warning(
                                "[META-CGN] Producer #10 knowledge.concept_grounded emit FAILED "
                                "— topic=%s err=%s (signal missed)",
                                topic, _p10_err)

                        # Cache persistence is owned by knowledge_cache (KP-2),
                        # populated inside dispatch_sync above. No inline
                        # cache write needed here.

                        # ── CONCEPT RESONANCE CASCADE ──
                        # Immediate cross-consumer effects:
                        # 1. Distribute to requestor
                        _distribute(send_queue, name, requestor, topic,
                                    concept, source="external_research",
                                    request_id=_request_id)
                        # 2. Broadcast to language → priority teaching
                        if requestor != "language":
                            _distribute(send_queue, name, "language",
                                        topic, concept,
                                        source="knowledge_cascade")
                        # 3. CGN_SURPRISE → Sigma micro-update on V(s)
                        #    All consumers' value landscapes shift
                        _send_msg(send_queue, "CGN_SURPRISE", name,
                                  "cgn", {
                            "consumer": "knowledge",
                            "concept_id": topic[:50],
                            "magnitude": quality,
                            "context": {
                                "type": "knowledge_grounded",
                                "source": _source_backend,
                                "summary_len": len(summary),
                                "requestor": requestor,
                            },
                        })
                        # 4. Concept lifecycle: birth event
                        _log_concept_lifecycle(
                            send_queue, name, topic,
                            "grounded", quality, requestor)
                        # 5. TimeChain: knowledge acquired → declarative fork
                        send_queue.put({"type": bus.TIMECHAIN_COMMIT, "src": name,
                            "dst": "timechain", "ts": time.time(), "payload": {
                            "fork": "declarative", "thought_type": "declarative",
                            "source": "knowledge_research",
                            "content": {"topic": topic[:100],
                                "summary_len": len(summary),
                                "quality": round(quality, 3),
                                "search_source": _source_backend,
                                "requestor": requestor},
                            "significance": quality,
                            "novelty": 0.9,
                            "coherence": 0.5,
                            "tags": [t.strip() for t in topic.lower().split()[:3]] + ["knowledge"],
                            "db_ref": f"knowledge_concepts:{topic[:50]}",
                            "neuromods": neuromods or {},
                            "chi_available": 0.5, "attention": 0.5,
                            "i_confidence": 0.5, "chi_coherence": 0.3,
                        }})

                        # Positive transition (reward delayed from usage)
                        _send_transition(
                            send_queue, name, cgn_client,
                            topic, action_idx, neuromods,
                            reward=0.1)

                        logger.info("[KnowledgeWorker] Grounded '%s' "
                                    "(quality=%.2f, %d chars)",
                                    topic[:40], quality, len(summary))
                    else:
                        logger.info("[KnowledgeWorker] No findings for '%s'",
                                    topic[:40])
                        _send_transition(
                            send_queue, name, cgn_client,
                            topic, action_idx, neuromods,
                            reward=-0.02)

                elif action_name == "consolidate" and internal:
                    # Merge with existing — bump encounter count
                    _consolidate_existing(db_path, topic)
                    _stats["internal_recalls"] += 1
                    # The consolidate branch only has one _distribute call;
                    # request_id propagation handled at line-level below.
                    _distribute(send_queue, name, requestor, topic,
                                internal[0], source="consolidated",
                                request_id=_request_id)

                else:
                    # Fallback: try dispatcher (was direct Sage call).
                    # Same pipeline as the research_shallow/deep path, just
                    # entered via the CGN fallback branch (consolidate/defer
                    # didn't apply + no internal hit with conf > 0.4).
                    _stats["external_researches"] += 1
                    _send_heartbeat(send_queue, name)
                    try:
                        dispatch_res = dispatch_sync(
                            topic=topic,
                            async_loop=_async_loop,
                            cache=_knowledge_cache,
                            sage=sage,
                            timeout=45.0,
                            timeout_per_backend=10.0,
                            inflight=_dispatch_inflight,
                            health=_health,
                            learner=_routing_learner,
                            requestor=requestor,
                        )
                    except Exception:
                        dispatch_res = None

                    if (dispatch_res is not None
                            and dispatch_res.result is not None
                            and dispatch_res.result.success):
                        # Track per-backend attempts + cache hits here too
                        for _bk, _outcome in dispatch_res.attempts:
                            _stats["backend_attempts"][_bk] = (
                                _stats["backend_attempts"].get(_bk, 0) + 1)
                        if dispatch_res.cache_hit:
                            _stats["cache_hits"] += 1
                        _source_backend = (
                            dispatch_res.backend_used or "dispatcher")
                        summary = dispatch_res.result.raw_text
                        quality = _quality_score(topic, summary)
                        if quality >= 0.3:
                            concept = _ground_concept(
                                db_path, topic, summary, quality,
                                source=_source_backend,
                                requestor=requestor,
                                neuromods=neuromods)
                            _stats["concepts_grounded"] += 1
                            # KP-8 — quality feedback into routing learner
                            if (_source_backend
                                    and _source_backend != "dispatcher"
                                    and dispatch_res is not None):
                                _routing_learner.record_quality(
                                    dispatch_res.query_type,
                                    _source_backend, quality)
                            # Resonance cascade (same as primary path)
                            _distribute(
                                send_queue, name, requestor, topic,
                                concept, source="external_research",
                                request_id=_request_id)
                            if requestor != "language":
                                _distribute(send_queue, name,
                                            "language", topic, concept,
                                            source="knowledge_cascade")
                            _send_msg(send_queue, "CGN_SURPRISE",
                                      name, "cgn", {
                                "consumer": "knowledge",
                                "concept_id": topic[:50],
                                "magnitude": quality,
                                "context": {
                                    "type": "knowledge_grounded",
                                    "source": _source_backend,
                                    "requestor": requestor,
                                },
                            })
                            _log_concept_lifecycle(
                                send_queue, name, topic,
                                "grounded", quality, requestor)
                            # TimeChain: fallback research → declarative
                            send_queue.put({"type": bus.TIMECHAIN_COMMIT,
                                "src": name, "dst": "timechain",
                                "ts": time.time(), "payload": {
                                "fork": "declarative",
                                "thought_type": "declarative",
                                "source": "knowledge_research",
                                "content": {"topic": topic[:100],
                                    "quality": round(quality, 3),
                                    "search_source": _source_backend,
                                    "requestor": requestor},
                                "significance": quality,
                                "novelty": 0.9, "coherence": 0.5,
                                "tags": [t.strip() for t in topic.lower().split()[:3]] + ["knowledge"],
                                "db_ref": f"knowledge_concepts:{topic[:50]}",
                                "neuromods": neuromods or {},
                                "chi_available": 0.5, "attention": 0.5,
                                "i_confidence": 0.5, "chi_coherence": 0.3,
                            }})

                # ── F-phase (rFP §16.3): meta outcome after research ──
                # Session 2: reward = 0.0 (advice not yet applied).
                # Session 3: switch to compute_outcome_knowledge with
                # actual pre/post confidence + bandwidth delta.
                if _kn_req_id:
                    try:
                        from titan_plugin.logic.meta_service_client import (
                            send_meta_outcome as _kn_out,
                        )
                        _kn_out(
                            request_id=_kn_req_id,
                            consumer_id="knowledge",
                            outcome_reward=0.0,
                            actual_primitive_used=None,
                            context=f"session_2_dry action={action_name}",
                            send_queue=send_queue, src=name)
                        _kn_meta_pending.pop(_kn_req_id, None)
                    except Exception as _kn_out_err:
                        logger.debug(
                            "[KnMeta] outcome skipped: %s", _kn_out_err)

            except Exception as e:
                logger.warning("[KnowledgeWorker] Request error: %s", e)

        # ── META_REASON_RESPONSE (F-phase rFP §4.3) ────────────────
        # Routed here per [meta_service_interface.consumer_home_worker]
        # knowledge = "knowledge".
        elif msg_type == bus.META_REASON_RESPONSE:
            try:
                from titan_plugin.logic.meta_service_client import (
                    dispatch_meta_response as _kn_dispatch,
                )
                _kn_dispatch(msg, logger_obj=logger)
            except Exception as _kn_disp_err:
                logger.warning(
                    "[KnMeta] response dispatch error: %s", _kn_disp_err)

        # ── CGN_KNOWLEDGE_USAGE — downstream usage reward ─────────
        # API_STUB: handler ready, awaiting CGN consumers (language/social/
        # reasoning) to send back usage events when they reference knowledge
        # concepts. Wired in CGN-EXTRACT (next session). Tracked as I-003.
        elif msg_type == bus.CGN_KNOWLEDGE_USAGE:
            try:
                topic = payload.get("topic", "")
                reward = float(payload.get("reward", 0.1))
                if topic:
                    _record_usage(db_path, topic)
                    _send_transition(send_queue, name, cgn_client,
                                     topic, 4, {}, reward=reward)
                    # KP-8 — usage is the strongest signal that a routing
                    # decision landed a useful concept. Look up the source
                    # backend that grounded this topic + feed the learner.
                    try:
                        _usage_source = ""
                        _usage_conn = sqlite3.connect(db_path, timeout=2.0)
                        _usage_row = _usage_conn.execute(
                            "SELECT source FROM knowledge_concepts "
                            "WHERE topic LIKE ? LIMIT 1",
                            (f"%{topic}%",)).fetchone()
                        _usage_conn.close()
                        if _usage_row and _usage_row[0]:
                            _usage_source = str(_usage_row[0])
                        if _usage_source and _usage_source != "dispatcher":
                            _qt_usage = classify_query(topic)
                            if _qt_usage != QueryType.INTERNAL_REJECTED:
                                _routing_learner.record_usage(
                                    _qt_usage, _usage_source)
                    except Exception as _lu_err:
                        logger.debug(
                            "[KnowledgeWorker] Learner usage update: %s",
                            _lu_err)
                    logger.info("[KnowledgeWorker] Usage reward: '%s' → %.2f",
                                topic[:40], reward)
            except Exception as e:
                logger.debug("[KnowledgeWorker] Usage error: %s", e)

        # ── CGN_HAOV_VERIFY_REQ — knowledge hypothesis verification ──
        # CGN Worker asks knowledge_worker to verify a knowledge hypothesis.
        # Verifier: did the knowledge concept gain downstream usage or
        # confidence since the hypothesis was formed?
        elif msg_type == bus.CGN_HAOV_VERIFY_REQ:
            try:
                _haov_p = msg.get("payload", {})
                _haov_consumer = _haov_p.get("consumer", "")
                if _haov_consumer == "knowledge":
                    _obs_b = _haov_p.get("obs_before", {})
                    _topic = _haov_p.get("test_ctx", {}).get("topic",
                             _haov_p.get("test_ctx", {}).get("concept", ""))
                    _conf_b = float(_obs_b.get("confidence", 0))
                    _usage_b = int(_obs_b.get("usage", 0))

                    # Query current state from DB
                    _conf_a = _conf_b
                    _usage_a = _usage_b
                    try:
                        import sqlite3 as _hv_sql
                        _hv_db = _hv_sql.connect(db_path, timeout=2.0)
                        _hv_db.execute("PRAGMA journal_mode=WAL")
                        if _topic:
                            _hv_row = _hv_db.execute(
                                "SELECT confidence, times_used FROM knowledge_concepts "
                                "WHERE topic = ? ORDER BY updated_at DESC LIMIT 1",
                                (_topic,)
                            ).fetchone()
                            if _hv_row:
                                _conf_a = float(_hv_row[0])
                                _usage_a = int(_hv_row[1])
                        _hv_db.close()
                    except Exception:
                        pass

                    _confirmed = (_conf_a > _conf_b + 0.05) or (_usage_a > _usage_b)
                    _error = abs(_conf_a - _conf_b)
                    _send_msg(send_queue, bus.CGN_HAOV_VERIFY_RSP, name, "cgn", {
                        "consumer": "knowledge",
                        "test_ctx": _haov_p.get("test_ctx"),
                        "obs_after": {"confidence": _conf_a, "usage": _usage_a},
                        "reward": min(1.0, _conf_a) if _confirmed else 0.0,
                        "confirmed": _confirmed,
                        "error": _error,
                    })
                    logger.info("[HAOV] Knowledge verify: topic='%s' conf %.3f→%.3f "
                                "usage %d→%d confirmed=%s",
                                (_topic or "?")[:40], _conf_b, _conf_a,
                                _usage_b, _usage_a, _confirmed)
            except Exception as _haov_err:
                logger.debug("[HAOV] Knowledge verification error: %s", _haov_err)

        # ── QUERY — stats and diagnostics ─────────────────────────
        elif msg_type == bus.QUERY:
            from titan_plugin.core.profiler import handle_memory_profile_query
            if handle_memory_profile_query(msg, send_queue, name):
                continue
            action = payload.get("action", "")
            if action == "get_knowledge_stats":
                stats = _get_stats(db_path, _stats)
                _send_response(send_queue, name, msg.get("src", ""),
                               stats, msg.get("rid"))
            elif action == "knowledge_search":
                topic = payload.get("topic", "")
                results = _search_internal(db_path, topic)
                _send_response(send_queue, name, msg.get("src", ""),
                               {"results": results}, msg.get("rid"))

        # ── SEARCH_PIPELINE_BUDGET_RESET — Maker override (KP-5) ──
        elif msg_type == bus.SEARCH_PIPELINE_BUDGET_RESET:
            try:
                _rb_backend = (payload.get("backend") or "").strip() or None
                _health.reset_budget(_rb_backend)
                logger.info(
                    "[KnowledgeWorker] Budget reset by Maker (backend=%s)",
                    _rb_backend or "ALL")
            except Exception as _rb_err:
                logger.warning(
                    "[KnowledgeWorker] Budget reset error: %s", _rb_err)

        # ── CGN_WEIGHTS_MAJOR — weights updated (client auto-reloads from SHM)
        elif msg_type == bus.CGN_WEIGHTS_MAJOR:
            logger.debug("[KnowledgeWorker] Weights updated (v=%s)",
                         payload.get("shm_version", "?"))

        # ── CGN_CROSS_INSIGHT (Plug B, rFP §20) ────────────────────
        # Emotional cross-insights from emot_cgn_worker reach us via
        # the bus's dst="all" broadcast. Forward to local CGN client
        # so its EMA of emotional-outcome rewards updates → surfaces
        # in state_vec slot 18 on next ground() call.
        elif msg_type == bus.CGN_CROSS_INSIGHT:
            try:
                if cgn_client is not None:
                    cgn_client.note_incoming_cross_insight(
                        msg.get("payload", {}))
            except Exception as _ci_err:
                logger.debug("[KnowledgeWorker] cross-insight note error: %s",
                             _ci_err)

        # ── rFP_titan_meta_outer_layer — bus-RPC handlers for meta-outer ──
        # Meta-reasoning queries knowledge_worker (which owns the Kuzu DB
        # connection; single-process lock) via these 3 handlers. Read-only,
        # rate-limited by existing search infra, small response payloads.
        elif msg_type == bus.KNOWLEDGE_QUERY_CONCEPT:
            try:
                topic = payload.get("topic", "")
                rid = msg.get("rid") or payload.get("rid", "")
                requestor = msg.get("src", "unknown")
                if topic:
                    rows = _search_internal(db_path, str(topic), max_results=1)
                    concept = rows[0] if rows else None
                else:
                    concept = None
                _send_response(send_queue, name, requestor,
                               {"concept": concept,
                                "confidence": (concept or {}).get(
                                    "confidence", 0.0) if concept else 0.0},
                               rid)
            except Exception as _kq_err:
                logger.debug("[KnowledgeWorker] KNOWLEDGE_QUERY_CONCEPT err: %s",
                             _kq_err)
        elif msg_type == bus.KNOWLEDGE_SEARCH:
            try:
                query = payload.get("query", "")
                rid = msg.get("rid") or payload.get("rid", "")
                requestor = msg.get("src", "unknown")
                max_r = int(payload.get("max_results", 5))
                rows = _search_internal(db_path, str(query),
                                         max_results=max(1, min(max_r, 20)))
                _send_response(send_queue, name, requestor,
                               {"results": rows}, rid)
            except Exception as _ks_err:
                logger.debug("[KnowledgeWorker] KNOWLEDGE_SEARCH err: %s",
                             _ks_err)
        elif msg_type == bus.KNOWLEDGE_CONCEPTS_FOR_PERSON:
            try:
                person_id = payload.get("person_id", "")
                rid = msg.get("rid") or payload.get("rid", "")
                requestor = msg.get("src", "unknown")
                limit = int(payload.get("limit", 5))
                # v1 heuristic: find concepts whose topic appears in the
                # person's felt_experiences. Walk events_teacher.db then
                # intersect with knowledge_concepts. Bounded to limit rows.
                concepts: list = []
                try:
                    ev_db = "data/events_teacher.db"
                    if os.path.exists(ev_db):
                        ev_conn = sqlite3.connect(
                            f"file:{ev_db}?mode=ro", uri=True, timeout=2.0)
                        ev_conn.row_factory = sqlite3.Row
                        topics = [r[0] for r in ev_conn.execute(
                            "SELECT DISTINCT topic FROM felt_experiences "
                            "WHERE author = ? ORDER BY created_at DESC LIMIT ?",
                            (person_id, max(1, min(limit * 3, 30)))
                        ).fetchall()]
                        ev_conn.close()
                        for t in topics:
                            rows = _search_internal(db_path, str(t),
                                                     max_results=1)
                            if rows:
                                concepts.append(rows[0])
                                if len(concepts) >= limit:
                                    break
                except Exception as _cfp_err:
                    logger.debug("[KnowledgeWorker] CFP probe err: %s",
                                 _cfp_err)
                _send_response(send_queue, name, requestor,
                               {"concepts": concepts}, rid)
            except Exception as _cfp_err:
                logger.debug(
                    "[KnowledgeWorker] KNOWLEDGE_CONCEPTS_FOR_PERSON err: %s",
                    _cfp_err)

        # ── MODULE_SHUTDOWN ───────────────────────────────────────
        # ── Microkernel v2 Phase B.1 §6 — shadow swap dispatch ────
        if _b1_reporter.handles(msg_type):
            _b1_reporter.handle(msg)
            if _b1_reporter.should_exit():
                break
            continue

        # ── Microkernel v2 Phase B.2.1 — supervision-transfer dispatch ──
        from titan_plugin.core import worker_swap_handler as _swap
        if _swap.maybe_dispatch_swap_msg(msg):
            continue

        elif msg_type == bus.MODULE_SHUTDOWN:
            logger.info("[KnowledgeWorker] Shutdown: %s",
                        payload.get("reason", "?"))
            _async_loop.call_soon_threadsafe(_async_loop.stop)
            break


# ── Internal Recall ───────────────────────────────────────────────────

def _search_internal(db_path: str, topic: str, max_results: int = 5) -> list:
    """Search internal memory for a topic. Returns list of dicts."""
    results = []
    try:
        conn = sqlite3.connect(db_path, timeout=2.0)
        conn.row_factory = sqlite3.Row

        # 1. Check knowledge_concepts first
        try:
            rows = conn.execute(
                "SELECT * FROM knowledge_concepts "
                "WHERE topic LIKE ? ORDER BY confidence DESC LIMIT ?",
                (f"%{topic}%", max_results)
            ).fetchall()
            for r in rows:
                results.append({
                    "source": "knowledge_concepts",
                    "topic": r["topic"],
                    "summary": r["summary"] or "",
                    "confidence": r["confidence"],
                    "times_used": r["times_used"],
                    "quality_score": r["quality_score"],
                })
        except sqlite3.OperationalError:
            pass  # Table may not exist on first run

        # 2. Check vocabulary (word-level knowledge)
        try:
            words = topic.lower().split()
            for word in words[:3]:  # Check first 3 words
                row = conn.execute(
                    "SELECT word, confidence, word_type, learning_phase "
                    "FROM vocabulary WHERE word = ?", (word,)
                ).fetchone()
                if row:
                    results.append({
                        "source": "vocabulary",
                        "topic": row["word"],
                        "summary": f"{row['word_type']} word, "
                                   f"phase={row['learning_phase']}",
                        "confidence": row["confidence"],
                        "times_used": 0,
                        "quality_score": row["confidence"],
                    })
        except Exception:
            pass

        # 3. Check meta_wisdom (crystallized reasoning insights)
        try:
            wisdom_rows = conn.execute(
                "SELECT pattern_type, pattern_description, confidence "
                "FROM meta_wisdom "
                "WHERE pattern_description LIKE ? "
                "ORDER BY confidence DESC LIMIT 3",
                (f"%{topic}%",)
            ).fetchall()
            for r in wisdom_rows:
                results.append({
                    "source": "meta_wisdom",
                    "topic": r["pattern_type"],
                    "summary": r["pattern_description"],
                    "confidence": r["confidence"],
                    "times_used": 0,
                    "quality_score": r["confidence"],
                })
        except Exception:
            pass

        conn.close()
    except Exception as e:
        logger.debug("[KnowledgeWorker] Internal search error: %s", e)

    return results


# ── Concept Grounding ─────────────────────────────────────────────────

def _ground_concept(db_path: str, topic: str, summary: str,
                    quality: float, source: str = "unknown",
                    requestor: str = "", neuromods: dict = None) -> dict:
    """Ground a new knowledge concept in inner_memory.db."""
    now = time.time()
    concept = {
        "topic": topic,
        "summary": summary,
        "confidence": min(0.5, quality),  # Start modest, grows with usage
        "quality_score": quality,
        "source": source,
        "requesting_consumer": requestor,
        "neuromod_at_acquisition": json.dumps(
            {k: round(v, 3) for k, v in (neuromods or {}).items()}),
        "created_at": now,
    }

    try:
        conn = sqlite3.connect(db_path, timeout=2.0)
        conn.execute("PRAGMA journal_mode=WAL")

        # Upsert: insert or update if topic exists
        existing = conn.execute(
            "SELECT id, encounter_count, confidence FROM knowledge_concepts "
            "WHERE topic = ?", (topic,)
        ).fetchone()

        conn.close()
        from titan_plugin.persistence import get_client
        client = get_client(caller_name="knowledge_worker")
        if existing:
            # Update existing — bump encounter, merge summary if better
            new_conf = min(1.0, existing[2] + 0.05)
            client.write(
                "UPDATE knowledge_concepts SET "
                "summary = CASE WHEN length(?) > length(summary) THEN ? "
                "  ELSE summary END, "
                "confidence = ?, quality_score = ?, "
                "encounter_count = encounter_count + 1, "
                "neuromod_at_acquisition = ?, "
                "last_used_at = ? "
                "WHERE topic = ?",
                (summary, summary, new_conf, quality,
                 concept["neuromod_at_acquisition"], now, topic),
                table="knowledge_concepts",
            )
            concept["confidence"] = new_conf
        else:
            client.write(
                "INSERT INTO knowledge_concepts "
                "(topic, summary, confidence, source, quality_score, "
                "requesting_consumer, neuromod_at_acquisition, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (topic, summary, concept["confidence"], source,
                 quality, requestor,
                 concept["neuromod_at_acquisition"], now),
                table="knowledge_concepts",
            )
    except Exception as e:
        logger.warning("[KnowledgeWorker] Ground concept error: %s", e)

    return concept


def _consolidate_existing(db_path: str, topic: str) -> None:
    """Bump encounter count for an existing concept."""
    try:
        from titan_plugin.persistence import get_client
        get_client(caller_name="knowledge_worker").write(
            "UPDATE knowledge_concepts SET encounter_count = encounter_count + 1, "
            "last_used_at = ? WHERE topic = ?", (time.time(), topic),
            table="knowledge_concepts")
    except Exception:
        pass


def _record_usage(db_path: str, topic: str) -> None:
    """Record downstream usage of a knowledge concept."""
    try:
        from titan_plugin.persistence import get_client
        get_client(caller_name="knowledge_worker").write(
            "UPDATE knowledge_concepts SET times_used = times_used + 1, "
            "confidence = MIN(1.0, confidence + 0.02), "
            "last_used_at = ? WHERE topic LIKE ?",
            (time.time(), f"%{topic}%"),
            table="knowledge_concepts")
    except Exception:
        pass


# ── Quality Gate (Heuristic) ──────────────────────────────────────────

def _quality_score(topic: str, summary: str) -> float:
    """Heuristic quality score for research results (0.0-1.0).

    Will be replaced by IQL net once training data accumulates.
    """
    if not summary or len(summary) < _MIN_SUMMARY_LENGTH:
        return 0.0
    if len(summary) > _MAX_SUMMARY_LENGTH:
        return 0.1  # Suspiciously long

    score = 0.3  # Base score for non-empty summary

    # Keyword overlap with topic
    topic_words = set(topic.lower().split())
    summary_words = set(summary.lower().split())
    if topic_words:
        overlap = len(topic_words & summary_words) / len(topic_words)
        score += min(0.3, overlap)

    # Length bonus (prefer 100-500 char summaries)
    if 100 <= len(summary) <= 500:
        score += 0.2
    elif 50 <= len(summary) <= 1000:
        score += 0.1

    # Penalty for boilerplate indicators
    boilerplate = ["click here", "subscribe", "cookie", "javascript",
                   "enable javascript", "403", "404", "access denied"]
    for b in boilerplate:
        if b in summary.lower():
            score -= 0.15

    return max(0.0, min(1.0, score))


# ── Distribution ──────────────────────────────────────────────────────

def _distribute(send_queue, name: str, requestor: str, topic: str,
                concept: dict, source: str = "",
                request_id: str = "") -> None:
    """Distribute grounded knowledge to requestor and broadcast.

    P8: carry request_id through for aggregation-window correlation. Empty
    string (default) means pre-P8 legacy call — handled as non-aggregated.
    """
    payload = {
        "topic": topic,
        "summary": concept.get("summary", ""),
        "confidence": concept.get("confidence", 0.0),
        "quality_score": concept.get("quality_score", 0.0),
        "source": source,
        "requestor": requestor,
    }
    if request_id:
        payload["request_id"] = request_id
    _send_msg(send_queue, bus.CGN_KNOWLEDGE_RESP, name, requestor, payload)


# ── Concept Lifecycle Telemetry ────────────────────────────────────────

def _log_concept_lifecycle(send_queue, name: str, topic: str,
                           event: str, quality: float,
                           consumer: str) -> None:
    """Log a concept lifecycle event for observatory visualization.

    Events: grounded, taught, used_in_social, used_in_reasoning,
            used_in_chat, confidence_boost
    """
    try:
        import json
        entry = {
            "topic": topic[:100],
            "event": event,
            "consumer": consumer,
            "quality": quality,
            "ts": time.time(),
        }
        path = "./data/concept_lifecycle.jsonl"
        with open(path, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


# ── CGN Transition ────────────────────────────────────────────────────

def _send_transition(send_queue, name: str, cgn_client,
                     topic: str, action_idx: int,
                     neuromods: dict, reward: float = 0.0) -> None:
    """Send a CGN transition for the knowledge consumer."""
    try:
        import numpy as np
        # Build minimal 30D state vector from neuromods
        state = np.zeros(30, dtype=np.float32)
        nm_keys = ["DA", "5HT", "NE", "ACh", "GABA", "Endorphin"]
        for i, k in enumerate(nm_keys):
            state[i] = float(neuromods.get(k, 0.5))
        # Topic hash as feature (deterministic)
        state[6] = (hash(topic) % 1000) / 1000.0

        _send_msg(send_queue, "CGN_TRANSITION", name, "cgn", {
            "consumer": "knowledge",
            "concept_id": topic[:50],
            "state": state.tolist(),
            "action": action_idx,
            "action_params": [0.0, 0.0, 0.0, 0.0],
            "reward": reward,
            "timestamp": time.time(),
            "epoch": 0,
            "metadata": {"topic": topic[:100]},
        })
        # Upgrade III peer publishing (audit 2026-04-23 Q2) — broadcast
        # knowledge chain-outcome so emot_cgn + other peer consumers can
        # learn from it. Rate-gated + informative filter inside.
        try:
            from titan_plugin.logic.cgn_consumer_client import (
                emit_chain_outcome_insight)
            emit_chain_outcome_insight(
                send_queue, name, "knowledge", float(reward),
                ctx={"topic": topic[:60]})
        except Exception:
            pass
    except Exception as e:
        logger.debug("[KnowledgeWorker] Transition error: %s", e)


# ── Stats ─────────────────────────────────────────────────────────────

def _get_stats(db_path: str, worker_stats: dict) -> dict:
    """Get comprehensive knowledge stats."""
    stats = dict(worker_stats)
    try:
        conn = sqlite3.connect(db_path, timeout=2.0)
        row = conn.execute(
            "SELECT COUNT(*), AVG(confidence), SUM(times_used) "
            "FROM knowledge_concepts"
        ).fetchone()
        stats["total_concepts"] = row[0] or 0
        stats["avg_confidence"] = round(row[1] or 0, 3)
        stats["total_usage"] = row[2] or 0

        # Recent concepts
        recent = conn.execute(
            "SELECT topic, confidence, source, times_used "
            "FROM knowledge_concepts ORDER BY created_at DESC LIMIT 5"
        ).fetchall()
        stats["recent_concepts"] = [
            {"topic": r[0], "confidence": r[1],
             "source": r[2], "times_used": r[3]}
            for r in recent
        ]
        conn.close()
    except Exception:
        stats["total_concepts"] = 0
        stats["avg_confidence"] = 0.0
        stats["total_usage"] = 0
        stats["recent_concepts"] = []

    return stats


# ── Schema Migration ──────────────────────────────────────────────────

def _ensure_schema(db_path: str) -> None:
    """Create knowledge_concepts table if it doesn't exist."""
    try:
        conn = sqlite3.connect(db_path, timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(_KNOWLEDGE_SCHEMA)
        conn.close()
        logger.info("[KnowledgeWorker] Schema OK (knowledge_concepts)")
    except Exception as e:
        logger.error("[KnowledgeWorker] Schema migration failed: %s", e)


# ── Sage Config Builder ──────────────────────────────────────────────

def _build_sage_config(config: dict) -> dict:
    """Build config dict for StealthSageResearcher from our merged config."""
    # Firecrawl: if API key present, used for premium scraping ONLY when the
    # user explicitly opts in via scrape_strategy = "firecrawl". Otherwise the
    # user's explicit choice from [stealth_sage].scrape_strategy in config.toml
    # is respected. Default falls back to "firecrawl" if a key is present
    # (legacy behavior), else "fast".
    # 2026-04-09 fix: this used to hard-override scrape_strategy whenever a
    # firecrawl_api_key was set, ignoring the user's explicit config choice.
    # That made all 3 Titans silently switch to firecrawl, hit 402 Payment
    # Required (credits depleted), and fall back to fast scrape via the
    # internal chain — wasteful + noisy. Now config.get() respects the
    # explicit setting.
    firecrawl_key = config.get("firecrawl_api_key", "").strip()
    default_strategy = "firecrawl" if firecrawl_key else "fast"
    return {
        "searxng_host": config.get("searxng_host", "http://localhost:8080"),
        "searxng_top_num_urls": config.get("searxng_top_num_urls", 3),
        "research_timeout_seconds": config.get("research_timeout_seconds", 30),
        "webshare_rotating_url": config.get("webshare_rotating_url", ""),
        "twitterapi_io_key": config.get("twitterapi_io_key", ""),
        "firecrawl_api_key": firecrawl_key,
        "scrape_strategy": config.get("scrape_strategy", default_strategy),
        "_inference": {
            "inference_provider": config.get("inference_provider", "ollama_cloud"),
            "ollama_cloud_base_url": config.get("ollama_cloud_base_url", ""),
            "ollama_cloud_api_key": config.get("ollama_cloud_api_key", ""),
        },
    }


def _wire_ollama_cloud(sage, config: dict) -> None:
    """Wire Ollama Cloud client into Sage for distillation."""
    # Sage uses cloud distillation via its own httpx calls when
    # _cloud_api_key is set. For Ollama Cloud, we override directly.
    base_url = config.get("ollama_cloud_base_url", "")
    api_key = config.get("ollama_cloud_api_key", "")
    model = config.get("ollama_cloud_chat_model", "deepseek-v3.1:671b")

    if base_url and api_key:
        # Override Sage's cloud inference to use Ollama Cloud
        sage._cloud_api_key = api_key
        sage._cloud_base_url = f"{base_url.rstrip('/')}/chat/completions"
        sage._cloud_model = model
        logger.info("[KnowledgeWorker] Sage distillation via Ollama Cloud "
                    "(%s)", model)


# ── Bus Helpers ───────────────────────────────────────────────────────

def _send_msg(send_queue, msg_type: str, src: str, dst: str,
              payload: dict) -> None:
    """Send a bus message."""
    send_queue.put({
        "type": msg_type, "src": src, "dst": dst,
        "payload": payload, "ts": time.time(),
    })


# Heartbeat throttle (Phase E Fix 2): 3s min interval per process.
_last_hb_ts: float = 0.0


def _send_heartbeat(send_queue, name: str) -> None:
    """Send heartbeat to Guardian (throttled to ≤1 per 3s)."""
    global _last_hb_ts
    now = time.time()
    if now - _last_hb_ts < 3.0:
        return
    _last_hb_ts = now
    try:
        import psutil
        rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        rss_mb = 0
    _send_msg(send_queue, bus.MODULE_HEARTBEAT, name, "guardian", {"rss_mb": round(rss_mb, 1)})


def _send_response(send_queue, name: str, dst: str,
                   data: dict, rid: str = None) -> None:
    """Send a QUERY response."""
    msg = {
        "type": "QUERY_RESPONSE", "src": name, "dst": dst,
        "payload": data, "ts": time.time(),
    }
    if rid:
        msg["rid"] = rid
    send_queue.put(msg)
