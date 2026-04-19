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

    # ── Stats ──────────────────────────────────────────────────────────
    _stats = {
        "requests_received": 0,
        "internal_recalls": 0,
        "external_researches": 0,
        "concepts_grounded": 0,
        "quality_rejected": 0,
        "deferred": 0,
        "cache_hits": 0,
        "rejected_internal_names": 0,
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

    # Stage A in-memory search cache (2026-04-12). Key: topic.lower().strip().
    # TTL 1 hour. Max 500 entries (FIFO eviction). Reduces bandwidth burn on
    # duplicate queries. Persistent SQLite-backed version in rFP v2 Layer 2.
    _search_cache: dict = {}
    _SEARCH_CACHE_MAX = 500
    _SEARCH_CACHE_TTL = 3600

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
    _send_msg(send_queue, "MODULE_READY", name, "guardian", {})

    # ── Main loop ──────────────────────────────────────────────────────
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
        if msg_type == "CGN_KNOWLEDGE_REQ":
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

                # Stage A filter (2026-04-12): reject Titan-internal-looking
                # topics that shouldn't reach external search engines. These
                # leak through from meta-reasoning formulate_output, persona
                # telemetry, etc. Patterns:
                #   - single word with underscores (inner_spirit, outer_perception)
                #   - primitive.submode notation (FORMULATE.load_wisdom)
                #   - single word < 4 chars
                # Prevents bandwidth burn on queries SearXNG/Sage can't answer.
                _is_internal_name = (
                    ("_" in topic and " " not in topic) or
                    ("." in topic and len(topic.split()) == 1) or
                    (len(topic) < 4 and " " not in topic)
                )
                if _is_internal_name:
                    logger.warning(
                        "[KnowledgeWorker] Rejected internal-name query: '%s' "
                        "(requestor=%s) — Titan-internal names not searchable "
                        "externally. See rFP_knowledge_pipeline_v2 Layer 4.",
                        topic[:60], requestor)
                    _stats["rejected_internal_names"] = (
                        _stats.get("rejected_internal_names", 0) + 1)
                    continue

                _stats["requests_received"] += 1
                logger.info("[KnowledgeWorker] Request: '%s' from %s "
                            "(urgency=%.2f)", topic[:60], requestor, urgency)

                # Stage A cache (2026-04-12): check in-memory cache before
                # external search. TTL + max defined at worker init above.
                _cache_key = topic.lower().strip()
                _now_ts = time.time()
                _cached = _search_cache.get(_cache_key)
                if _cached and (_now_ts - _cached["ts"]) < _SEARCH_CACHE_TTL:
                    # Cache hit — distribute cached result without external call
                    _stats["cache_hits"] += 1
                    _distribute(send_queue, name, requestor, topic,
                                _cached["concept"], source="cache",
                                request_id=_request_id)
                    logger.info("[KnowledgeWorker] Cache hit: '%s' "
                                "(age=%.0fs, saves ~2-5MB proxy bandwidth)",
                                topic[:40], _now_ts - _cached["ts"])
                    continue

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

                if action_name == "defer":
                    _stats["deferred"] += 1
                    logger.info("[KnowledgeWorker] Deferred: '%s'",
                                topic[:40])
                    _send_transition(send_queue, name, cgn_client,
                                     topic, 5, neuromods, reward=0.0)
                    continue

                if action_name in ("research_shallow", "research_deep") and sage:
                    # External research via Stealth Sage
                    _stats["external_researches"] += 1
                    # Send heartbeat before long research operation
                    _send_heartbeat(send_queue, name)
                    try:
                        future = asyncio.run_coroutine_threadsafe(
                            sage.research(topic), _async_loop)
                        findings = future.result(timeout=45.0)
                    except Exception as sage_err:
                        logger.warning("[KnowledgeWorker] Sage error: %s",
                                       sage_err)
                        findings = ""

                    if findings:
                        # Heartbeat after research completes
                        _send_heartbeat(send_queue, name)
                        _last_heartbeat = time.time()

                        # Strip [SAGE_RESEARCH_FINDINGS]: prefix
                        summary = findings.replace(
                            "[SAGE_RESEARCH_FINDINGS]: ", "").strip()

                        # Quality gate
                        quality = _quality_score(topic, summary)
                        if quality < 0.3:
                            _stats["quality_rejected"] += 1
                            logger.info("[KnowledgeWorker] Quality rejected "
                                        "'%s' (score=%.2f)", topic[:40],
                                        quality)
                            _send_transition(
                                send_queue, name, cgn_client,
                                topic, action_idx, neuromods,
                                reward=-0.05)
                            continue

                        # Ground as concept
                        concept = _ground_concept(
                            db_path, topic, summary, quality,
                            source="searxng",
                            requestor=requestor,
                            neuromods=neuromods)
                        _stats["concepts_grounded"] += 1

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

                        # Stage A cache write: store successful result for
                        # future hits within TTL. FIFO eviction at max size.
                        if len(_search_cache) >= _SEARCH_CACHE_MAX:
                            # Drop oldest entry (simple FIFO)
                            _oldest = min(_search_cache,
                                          key=lambda k: _search_cache[k]["ts"])
                            _search_cache.pop(_oldest, None)
                        _search_cache[_cache_key] = {
                            "concept": concept,
                            "ts": _now_ts,
                        }

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
                                "source": "searxng",
                                "summary_len": len(summary),
                                "requestor": requestor,
                            },
                        })
                        # 4. Concept lifecycle: birth event
                        _log_concept_lifecycle(
                            send_queue, name, topic,
                            "grounded", quality, requestor)
                        # 5. TimeChain: knowledge acquired → declarative fork
                        send_queue.put({"type": "TIMECHAIN_COMMIT", "src": name,
                            "dst": "timechain", "ts": time.time(), "payload": {
                            "fork": "declarative", "thought_type": "declarative",
                            "source": "knowledge_research",
                            "content": {"topic": topic[:100],
                                "summary_len": len(summary),
                                "quality": round(quality, 3),
                                "search_source": "searxng",
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
                    # Fallback: try shallow research if sage available
                    if sage:
                        _stats["external_researches"] += 1
                        _send_heartbeat(send_queue, name)
                        try:
                            future = asyncio.run_coroutine_threadsafe(
                                sage.research(topic), _async_loop)
                            findings = future.result(timeout=45.0)
                        except Exception:
                            findings = ""
                        if findings:
                            summary = findings.replace(
                                "[SAGE_RESEARCH_FINDINGS]: ", "").strip()
                            quality = _quality_score(topic, summary)
                            if quality >= 0.3:
                                concept = _ground_concept(
                                    db_path, topic, summary, quality,
                                    source="searxng",
                                    requestor=requestor,
                                    neuromods=neuromods)
                                _stats["concepts_grounded"] += 1
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
                                        "source": "searxng",
                                        "requestor": requestor,
                                    },
                                })
                                _log_concept_lifecycle(
                                    send_queue, name, topic,
                                    "grounded", quality, requestor)
                                # TimeChain: fallback research → declarative
                                send_queue.put({"type": "TIMECHAIN_COMMIT",
                                    "src": name, "dst": "timechain",
                                    "ts": time.time(), "payload": {
                                    "fork": "declarative",
                                    "thought_type": "declarative",
                                    "source": "knowledge_research",
                                    "content": {"topic": topic[:100],
                                        "quality": round(quality, 3),
                                        "requestor": requestor},
                                    "significance": quality,
                                    "novelty": 0.9, "coherence": 0.5,
                                    "tags": [t.strip() for t in topic.lower().split()[:3]] + ["knowledge"],
                                    "db_ref": f"knowledge_concepts:{topic[:50]}",
                                    "neuromods": neuromods or {},
                                    "chi_available": 0.5, "attention": 0.5,
                                    "i_confidence": 0.5, "chi_coherence": 0.3,
                                }})

            except Exception as e:
                logger.warning("[KnowledgeWorker] Request error: %s", e)

        # ── CGN_KNOWLEDGE_USAGE — downstream usage reward ─────────
        # API_STUB: handler ready, awaiting CGN consumers (language/social/
        # reasoning) to send back usage events when they reference knowledge
        # concepts. Wired in CGN-EXTRACT (next session). Tracked as I-003.
        elif msg_type == "CGN_KNOWLEDGE_USAGE":
            try:
                topic = payload.get("topic", "")
                reward = float(payload.get("reward", 0.1))
                if topic:
                    _record_usage(db_path, topic)
                    _send_transition(send_queue, name, cgn_client,
                                     topic, 4, {}, reward=reward)
                    logger.info("[KnowledgeWorker] Usage reward: '%s' → %.2f",
                                topic[:40], reward)
            except Exception as e:
                logger.debug("[KnowledgeWorker] Usage error: %s", e)

        # ── CGN_HAOV_VERIFY_REQ — knowledge hypothesis verification ──
        # CGN Worker asks knowledge_worker to verify a knowledge hypothesis.
        # Verifier: did the knowledge concept gain downstream usage or
        # confidence since the hypothesis was formed?
        elif msg_type == "CGN_HAOV_VERIFY_REQ":
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
                    _send_msg(send_queue, "CGN_HAOV_VERIFY_RSP", name, "cgn", {
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
        elif msg_type == "QUERY":
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

        # ── CGN_WEIGHTS_MAJOR — weights updated (client auto-reloads from SHM)
        elif msg_type == "CGN_WEIGHTS_MAJOR":
            logger.debug("[KnowledgeWorker] Weights updated (v=%s)",
                         payload.get("shm_version", "?"))

        # ── MODULE_SHUTDOWN ───────────────────────────────────────
        elif msg_type == "MODULE_SHUTDOWN":
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

        if existing:
            # Update existing — bump encounter, merge summary if better
            new_conf = min(1.0, existing[2] + 0.05)
            conn.execute(
                "UPDATE knowledge_concepts SET "
                "summary = CASE WHEN length(?) > length(summary) THEN ? "
                "  ELSE summary END, "
                "confidence = ?, quality_score = ?, "
                "encounter_count = encounter_count + 1, "
                "neuromod_at_acquisition = ?, "
                "last_used_at = ? "
                "WHERE topic = ?",
                (summary, summary, new_conf, quality,
                 concept["neuromod_at_acquisition"], now, topic))
            concept["confidence"] = new_conf
        else:
            conn.execute(
                "INSERT INTO knowledge_concepts "
                "(topic, summary, confidence, source, quality_score, "
                "requesting_consumer, neuromod_at_acquisition, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (topic, summary, concept["confidence"], source,
                 quality, requestor,
                 concept["neuromod_at_acquisition"], now))

        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning("[KnowledgeWorker] Ground concept error: %s", e)

    return concept


def _consolidate_existing(db_path: str, topic: str) -> None:
    """Bump encounter count for an existing concept."""
    try:
        conn = sqlite3.connect(db_path, timeout=2.0)
        conn.execute(
            "UPDATE knowledge_concepts SET encounter_count = encounter_count + 1, "
            "last_used_at = ? WHERE topic = ?", (time.time(), topic))
        conn.commit()
        conn.close()
    except Exception:
        pass


def _record_usage(db_path: str, topic: str) -> None:
    """Record downstream usage of a knowledge concept."""
    try:
        conn = sqlite3.connect(db_path, timeout=2.0)
        conn.execute(
            "UPDATE knowledge_concepts SET times_used = times_used + 1, "
            "confidence = MIN(1.0, confidence + 0.02), "
            "last_used_at = ? WHERE topic LIKE ?",
            (time.time(), f"%{topic}%"))
        conn.commit()
        conn.close()
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
    _send_msg(send_queue, "CGN_KNOWLEDGE_RESP", name, requestor, payload)


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
    _send_msg(send_queue, "MODULE_HEARTBEAT", name, "guardian", {"rss_mb": round(rss_mb, 1)})


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
