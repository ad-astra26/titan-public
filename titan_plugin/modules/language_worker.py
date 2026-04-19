"""
Language Worker — Guardian-managed module for language composition, teaching,
and vocabulary management.

Owns: CompositionEngine, LanguageTeacher, WordSelector, GrammarValidator,
      PatternLibrary, vocabulary DB, teacher scheduling, self-hearing deltas.

Bus protocol:
  SPEAK_REQUEST     (spirit → language)  — compose sentence from felt-state
  SPEAK_RESULT      (language → spirit)  — sentence + perturbation deltas
  TEACHER_SIGNALS   (language → spirit)  — MSL concept signals + vocab updates
  LLM_TEACHER_REQUEST  (language → llm)  — teacher prompt for inference
  LLM_TEACHER_RESPONSE (llm → language)  — teacher response
  LANGUAGE_STATS_UPDATE (language → all)  — periodic stats broadcast
  QUERY get_language_stats               — on-demand stats for inner-trinity

Entry point: language_worker_main(recv_queue, send_queue, name, config)

Phase 1: Skeleton — initializes engines, handles QUERY, logs other messages.
         Spirit_worker still does all language work inline.
Phase 2+: Full SPEAK and teacher path migration.
"""
import json
import logging
import os
import sys
import threading
import time

import numpy as np

logger = logging.getLogger(__name__)


# ── Phase D.1 — META_LANGUAGE reward helpers ───────────────────────────
def _count_grounded_words(db_path: str) -> int:
    """Count words with confidence >= 0.3 (grounded threshold).

    Used as a baseline/snapshot measure when registering a pending
    META_LANGUAGE_REWARD and when computing the delta ~60s later.
    """
    try:
        import sqlite3 as _cg_sql
        _cg_db = _cg_sql.connect(db_path, timeout=5.0)
        _cg_n = _cg_db.execute(
            "SELECT COUNT(*) FROM vocabulary WHERE confidence >= 0.3"
        ).fetchone()[0]
        _cg_db.close()
        return int(_cg_n)
    except Exception:
        return 0


def _measure_meta_lang_reward(
    vocab_delta: int, grounded_delta: int, primitives: list
) -> float:
    """Normalize vocab/grounded delta into a reward in [0, 1].

    Heuristic (tuned to reward both new-word acquisition AND deepening
    of existing-word grounding):
      - +0.3 per new vocab word (cap at 0.6)
      - +0.2 per newly grounded word (cap at 0.4)
      - +0.1 base if chain contained a diversification primitive
        (RECALL/COMPARE/DECOMPOSE/DISTILL) — encourages the META_LANGUAGE
        loop to pull the policy out of FORMULATE monoculture.
    """
    r = 0.0
    if vocab_delta > 0:
        r += min(0.6, 0.3 * vocab_delta)
    if grounded_delta > 0:
        r += min(0.4, 0.2 * grounded_delta)
    if any(p in ("RECALL", "COMPARE", "DECOMPOSE", "DISTILL")
           for p in (primitives or [])):
        r += 0.1
    return min(1.0, r)


def language_worker_main(recv_queue, send_queue, name: str, config: dict) -> None:
    """Main loop for the Language module process.

    Args:
        recv_queue: receives messages from DivineBus (bus->worker)
        send_queue: sends messages back to DivineBus (worker->bus)
        name: module name ("language")
        config: dict from [language] config section + data_dir
    """
    from queue import Empty

    # Project root for imports
    project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    logger.info("[LanguageWorker] Initializing language subsystem...")
    init_start = time.time()

    # ── Load per-Titan configuration ─────────────────────────────────
    from titan_plugin.logic.language_config import load_config
    lang_config = load_config()
    titan_id = lang_config.get("titan_id", "T1")
    logger.info("[LanguageWorker] Config loaded for %s: interval=%d, bootstrap_thresh=%d",
                titan_id,
                lang_config.get("teacher_interval_compositions", 5),
                lang_config.get("bootstrap_vocab_threshold", 50))

    # ── Initialize language engines ──────────────────────────────────
    composition_engine = None
    persistent_teacher = None
    grammar_validator = None
    pattern_library = None

    try:
        from titan_plugin.logic.grammar_patterns import GrammarPatternLibrary
        pattern_library = GrammarPatternLibrary()
    except Exception as e:
        logger.warning("[LanguageWorker] PatternLibrary init failed: %s", e)

    try:
        from titan_plugin.logic.composition_engine import CompositionEngine
        composition_engine = CompositionEngine(pattern_library=pattern_library)
        logger.info("[LanguageWorker] CompositionEngine ready")
    except Exception as e:
        logger.warning("[LanguageWorker] CompositionEngine init failed: %s", e)

    try:
        from titan_plugin.logic.language_teacher import LanguageTeacher
        persistent_teacher = LanguageTeacher()
        logger.info("[LanguageWorker] LanguageTeacher ready")
    except Exception as e:
        logger.warning("[LanguageWorker] LanguageTeacher init failed: %s", e)

    try:
        from titan_plugin.logic.grammar_validator import GrammarValidator
        data_dir = config.get("data_dir", "./data")
        grammar_validator = GrammarValidator(
            db_path=os.path.join(data_dir, "grammar_rules.db"))
        logger.info("[LanguageWorker] GrammarValidator ready")
    except Exception as e:
        logger.warning("[LanguageWorker] GrammarValidator init failed: %s", e)

    # ── Initialize vocabulary cache from DB ──────────────────────────
    from titan_plugin.logic.language_pipeline import (
        load_vocabulary, update_language_stats,
    )
    data_dir = config.get("data_dir", "./data")
    db_path = os.path.join(data_dir, "inner_memory.db")

    # ── Auto-migrate vocabulary schema (Phase 4) ────────────────────
    try:
        import sqlite3 as _mig_sql
        _mig_db = _mig_sql.connect(db_path, timeout=5.0)
        _mig_cols = {r[1] for r in _mig_db.execute("PRAGMA table_info(vocabulary)")}
        for _mc, _mt, _md in [("sensory_context", "TEXT", "'[]'"),
                              ("meaning_contexts", "TEXT", "'[]'"),
                              ("cross_modal_conf", "REAL", "0.0")]:
            if _mc not in _mig_cols:
                _mig_db.execute(f"ALTER TABLE vocabulary ADD COLUMN {_mc} {_mt} DEFAULT {_md}")
                logger.info("[LanguageWorker] Migrated: added %s column", _mc)
        _mig_db.commit()
        _mig_db.close()
    except Exception as _mig_err:
        logger.warning("[LanguageWorker] Schema migration: %s", _mig_err)

    cached_vocab = load_vocabulary(db_path)
    logger.info("[LanguageWorker] Vocabulary cache: %d words loaded from DB", len(cached_vocab))

    # ── META-CGN Producer #7 EdgeDetector: language.vocab_expanded ──
    # rFP v3 § 12 row 7. Fires on observe_new_max("vocab_size", n) when
    # vocab grows. No persistence file needed — the vocab DB IS the
    # durable source of truth; we prime the detector's _max to current
    # vocab size at boot so only POST-BOOT growth fires (no false
    # emission on worker restart).
    try:
        from titan_plugin.logic.meta_cgn import EdgeDetector as _P7EdgeDet
        _vocab_edge_detector = _P7EdgeDet()
        _vocab_edge_detector._max["vocab_size"] = float(len(cached_vocab))
        logger.info(
            "[META-CGN] Producer #7 EdgeDetector primed at vocab_size=%d (no false emit on boot)",
            len(cached_vocab))
    except Exception as _p7_init_err:
        logger.warning(
            "[META-CGN] Producer #7 EdgeDetector init failed: %s — vocab_expanded will not emit",
            _p7_init_err)
        _vocab_edge_detector = None

    # ── META-CGN Producer #8 EdgeDetector: language.concept_grounded ──
    # rFP v3 § 12 row 8. Fires per-word when vocabulary confidence crosses
    # ≥ 0.5 for the first time. Primed at boot with ALL current words ≥ 0.5
    # so only POST-BOOT groundings fire (no false emissions on restart).
    # Monoculture-aware weights per rebalance (see SIGNAL_TO_PRIMITIVE comment):
    # FORMULATE 0.20 (anti — T1/T3 monoculture), SYNTHESIZE 0.70 (boost),
    # HYPOTHESIZE 0.65 (boost), RECALL 0.30 (anti — T2 RECALL monoculture).
    try:
        _concept_edge_detector = _P7EdgeDet()
        import sqlite3 as _p8_sql
        _p8_primed = 0
        try:
            _p8_conn = _p8_sql.connect(db_path, timeout=5.0)
            for _p8_row in _p8_conn.execute(
                    "SELECT word FROM vocabulary WHERE confidence >= 0.5"):
                _concept_edge_detector._crossed[str(_p8_row[0])] = True
                _p8_primed += 1
            _p8_conn.close()
        except Exception as _p8_prime_err:
            logger.debug("[META-CGN] Producer #8 prime query: %s", _p8_prime_err)
        logger.info(
            "[META-CGN] Producer #8 EdgeDetector primed (%d words already ≥ 0.5 conf)",
            _p8_primed)
    except Exception as _p8_init_err:
        logger.warning(
            "[META-CGN] Producer #8 EdgeDetector init failed: %s — concept_grounded will not emit",
            _p8_init_err)
        _concept_edge_detector = None

    # ── Language stats (refreshed every 30s, broadcast for inner-trinity) ──
    language_stats = update_language_stats(db_path, cached_vocab)

    # ── Teacher state — Phase 1 fix for I-010 (Language Teacher silent) ──
    # Persisted state file survives RSS-triggered worker restarts. Without
    # persistence, teacher_compositions_since reset to 0 on every restart and
    # could never reach the threshold under high-restart conditions, causing
    # 33-44h teacher silence on T1/T2 (T3 was lucky due to lower restart rate).
    teacher_state_path = os.path.join(data_dir, "teacher_state.json")

    def _load_teacher_state():
        # On fresh-boot (no state file), default last_fire_time to NOW so the
        # time-based fallback starts counting from worker start, not from epoch
        # (which would make time_since_last_teach huge on first iteration).
        try:
            if os.path.exists(teacher_state_path):
                with open(teacher_state_path) as f:
                    state = json.load(f)
                    # Migrate old format: if last_fire_time is 0 or missing, set
                    # to current time so silence alert and fallback don't fire
                    # immediately on next boot of older saved state.
                    if state.get("last_fire_time", 0.0) <= 0:
                        state["last_fire_time"] = time.time()
                    return state
        except Exception as e:
            logger.warning("[LanguageWorker] teacher_state load failed: %s", e)
        return {"compositions_since": 0, "last_fire_time": time.time()}

    def _save_teacher_state(comp_since, last_fire):
        try:
            tmp = teacher_state_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"compositions_since": comp_since,
                           "last_fire_time": last_fire}, f)
            os.replace(tmp, teacher_state_path)  # Atomic
        except Exception as e:
            logger.debug("[LanguageWorker] teacher_state save failed: %s", e)

    teacher_queue = []
    teacher_pending_since = 0.0
    teacher_no_response_count = 0
    _teacher_state = _load_teacher_state()
    teacher_compositions_since = _teacher_state.get("compositions_since", 0)
    teacher_last_fire_time = _teacher_state.get("last_fire_time", time.time())
    teacher_interval = lang_config.get("teacher_interval_compositions", 5)
    teacher_fallback_timeout = lang_config.get("teacher_fallback_timeout_s", 1800)  # 30 min
    teacher_silence_alert_threshold = lang_config.get("teacher_silence_alert_s", 7200)  # 2h
    teacher_silence_alerted = False  # rate-limited alert flag
    if teacher_compositions_since > 0 or teacher_last_fire_time > 0:
        logger.info(
            "[LanguageWorker] Teacher state restored: comp_since=%d, last_fire=%.0fs ago",
            teacher_compositions_since,
            time.time() - teacher_last_fire_time if teacher_last_fire_time > 0 else 0,
        )
    bootstrap_speak_attempts = 0
    bootstrap_last_trigger = 0.0

    # Conversation state
    conversation_pending = None
    conversation_timeout = lang_config.get("conversation_timeout_s", 600)
    conversation_stats = {"asked": 0, "answered": 0, "timed_out": 0, "avg_score": 0}
    recent_teacher_questions = []

    # Word grounding state (Phase 4)
    _grounding_last_check = 0.0
    _grounding_interval = 300.0  # Check every 5 minutes
    _grounded_words = set()  # Words already grounded (avoid re-triggering)

    # Word consolidation state (Phase 5) — discovers words needing teaching
    _consolidation_last_check = 0.0
    _consolidation_interval = 600.0  # Check every 10 minutes
    _recently_consolidated = set()  # Avoid re-queuing same words
    _cached_concept_confs = {}  # MSL concept confidences (updated via SPEAK_RESULT)

    # Phase D.1 — META_LANGUAGE_REWARD pending tracking.
    # chain_id → {vocab_baseline, grounded_baseline, ts, primitives}
    # On each background tick, any entry older than META_LANG_MEASURE_DELAY_S
    # gets its vocab/grounded delta measured, normalized to a reward in [0, 1],
    # and emitted as META_LANGUAGE_REWARD. Entries older than
    # META_LANG_PENDING_MAX_AGE_S are dropped as orphans.
    meta_lang_pending: dict = {}
    META_LANG_MEASURE_DELAY_S = 60.0
    META_LANG_PENDING_MAX_AGE_S = 300.0
    # Phase D.1 fix: track grounding EVENTS (not vocab count).
    # _count_grounded_words() saturates because Phase 4c re-grounds
    # existing words without changing the count above threshold.
    # This counter increments on every REASONING_LINKED grounding event.
    _d1_grounding_events: int = 0

    # ── Concept Grounding Network (CGN) — Cognitive Kernel v2 ────────
    # CGN Worker (Guardian module) owns training, buffer, Sigma, HAOV.
    # ── CGN Consumer Client (A1 migration) ──────────────────────────
    # Language worker uses CGNConsumerClient for:
    #   - ground() forward pass (fast, 0.5ms, from /dev/shm weights)
    # Vocabulary DB ops (load_concept, apply_grounding_action) use standalone
    # functions from language_pipeline.py. No local ConceptGroundingNetwork.
    # All transitions forwarded to CGN Worker via bus.
    cgn = None           # CGNConsumerClient("language") for language grounding
    cgn_social = None    # CGNConsumerClient("social") for social inference
    _cgn_local_groundings = 0
    try:
        from titan_plugin.logic.cgn_consumer_client import CGNConsumerClient
        cgn = CGNConsumerClient(
            "language", send_queue=send_queue, module_name=name,
            state_dir=os.path.join(data_dir, "cgn"),
            shm_path="/dev/shm/cgn_live_weights.bin")
        cgn_social = CGNConsumerClient(
            "social", send_queue=send_queue, module_name=name,
            state_dir=os.path.join(data_dir, "cgn"),
            shm_path="/dev/shm/cgn_live_weights.bin")
        logger.info("[LanguageWorker] CGN ConsumerClients loaded "
                    "(language + social, from /dev/shm)")
    except Exception as _cgn_err:
        logger.warning("[LanguageWorker] CGN client init failed: %s", _cgn_err)
        cgn = None

    # Standalone vocabulary DB functions (extracted from CGN)
    from titan_plugin.logic.language_pipeline import (
        load_concept_from_db, apply_grounding_action_to_db,
    )

    def _cgn_forward_outcome(consumer, concept_id, reward, outcome_context=None):
        """Forward a delayed reward to CGN Worker via bus."""
        try:
            _send_msg(send_queue, "CGN_TRANSITION", name, "cgn", {
                "type": "outcome",
                "consumer": consumer,
                "concept_id": concept_id,
                "reward": reward,
                "outcome_context": outcome_context or {},
            })
        except Exception:
            pass

    init_ms = (time.time() - init_start) * 1000
    logger.info("[LanguageWorker] Language subsystem ready in %.0fms "
                "(vocab=%d, engines=%s/%s/%s, cgn=%s)",
                init_ms, len(cached_vocab),
                "CE" if composition_engine else "-",
                "LT" if persistent_teacher else "-",
                "GV" if grammar_validator else "-",
                "ON" if cgn else "OFF")

    # ── Signal ready to Guardian ─────────────────────────────────────
    _send_msg(send_queue, "MODULE_READY", name, "guardian", {})

    # ── Background heartbeat thread (prevents Guardian timeout) ──────
    _hb_stop = threading.Event()

    def _heartbeat_loop():
        while not _hb_stop.is_set():
            _send_heartbeat(send_queue, name)
            _hb_stop.wait(30.0)

    hb_thread = threading.Thread(target=_heartbeat_loop, daemon=True,
                                 name="language-heartbeat")
    hb_thread.start()

    # ── Timing ───────────────────────────────────────────────────────
    last_stats_broadcast = time.time()
    stats_broadcast_interval = 30.0  # Match spirit_worker's 30s update cycle

    # ── Main loop ────────────────────────────────────────────────────
    while True:
        # Heartbeat every iteration (throttled to 3s min in helper).
        # MUST be at top, NOT only in Empty or per-msg-type, because broadcast
        # messages (TITAN_SELF_STATE dst="all", etc.) that fall through the
        # elif chain without matching any explicit msg_type would starve the
        # Empty path AND skip every explicit per-msg heartbeat call. See
        # media_worker.py Option A fix 2026-04-15 + worker audit.
        _send_heartbeat(send_queue, name)

        try:
            msg = recv_queue.get(timeout=5.0)
        except Empty:
            # Periodic work on timeout
            now = time.time()

            # Broadcast language stats every 30s
            if now - last_stats_broadcast >= stats_broadcast_interval:
                try:
                    language_stats = update_language_stats(db_path, cached_vocab)
                    _send_msg(send_queue, "LANGUAGE_STATS_UPDATE", name, "all", language_stats)
                except Exception as e:
                    logger.debug("[LanguageWorker] Stats broadcast error: %s", e)
                last_stats_broadcast = now

            # Teacher trigger check (Phase 3) — I-010 fix: ALSO fire on
            # time-based fallback. The original count-based trigger is fragile
            # because the counter resets on worker restart and SPEAK rate can
            # be too low to reach the threshold between restarts. The fallback
            # ensures the teacher fires AT LEAST every teacher_fallback_timeout
            # (default 30 min) when there's a non-empty queue.
            time_since_last_teach = max(0.0, now - teacher_last_fire_time)
            should_trigger = teacher_queue and teacher_pending_since == 0 and (
                teacher_compositions_since >= teacher_interval
                or time_since_last_teach >= teacher_fallback_timeout
            )
            if should_trigger:
                try:
                    _trigger_reason = ("count" if teacher_compositions_since >= teacher_interval
                                       else f"fallback({time_since_last_teach:.0f}s)")
                    _check_teacher_trigger(
                        send_queue, name, persistent_teacher,
                        teacher_queue, cached_vocab,
                        lang_config, recent_teacher_questions,
                        None,  # neuromod_state not available in timeout
                        cgn_instance=cgn)
                    teacher_pending_since = now
                    teacher_last_fire_time = now
                    teacher_silence_alerted = False  # reset alert flag
                    _save_teacher_state(teacher_compositions_since, teacher_last_fire_time)
                    logger.info(
                        "[LanguageWorker] Teacher fired (reason=%s, queue=%d, comp_since=%d)",
                        _trigger_reason, len(teacher_queue), teacher_compositions_since)
                except Exception as e:
                    logger.debug("[LanguageWorker] Teacher trigger error: %s", e)

            # Silence alert: warn if teacher hasn't fired in 2+ hours despite
            # having a non-empty queue (rate-limited to one alert per silence
            # period — clears once teacher fires successfully).
            if (teacher_queue and time_since_last_teach >= teacher_silence_alert_threshold
                    and not teacher_silence_alerted):
                logger.warning(
                    "[LanguageWorker] TEACHER SILENCE ALERT — no teaching for %.1fh, "
                    "queue=%d, comp_since=%d, interval=%d. Investigate why trigger conditions never met.",
                    time_since_last_teach / 3600.0, len(teacher_queue),
                    teacher_compositions_since, teacher_interval)
                teacher_silence_alerted = True

            # Bootstrap check
            if not teacher_queue and teacher_pending_since == 0:
                from titan_plugin.logic.language_pipeline import should_bootstrap
                if should_bootstrap(
                    vocab_size=len(cached_vocab),
                    bootstrap_speak_attempts=bootstrap_speak_attempts,
                    compositions_since_teach=teacher_compositions_since,
                    teacher_queue_empty=not teacher_queue,
                    teacher_pending=teacher_pending_since > 0,
                    last_bootstrap_trigger=bootstrap_last_trigger,
                    threshold=lang_config.get("bootstrap_vocab_threshold", 50),
                    cooldown_s=lang_config.get("bootstrap_cooldown_s", 60),
                ):
                    try:
                        from titan_plugin.logic.language_teacher import BOOTSTRAP_MODE
                        _bt_prompt = persistent_teacher.build_prompt(
                            BOOTSTRAP_MODE, [], [], patterns_to_avoid=[])
                        _send_msg(send_queue, "LLM_TEACHER_REQUEST", name, "llm", {
                            "prompt": _bt_prompt["prompt"],
                            "system": _bt_prompt["system"],
                            "mode": BOOTSTRAP_MODE,
                            "original": "",
                            "max_tokens": _bt_prompt.get("max_tokens", 40),
                            "sentences": [],
                            "neuromod_gate": "",
                        })
                        teacher_pending_since = now
                        teacher_last_fire_time = now  # I-010 fix: update on bootstrap fire
                        teacher_silence_alerted = False
                        _save_teacher_state(teacher_compositions_since, teacher_last_fire_time)
                        bootstrap_last_trigger = now
                        bootstrap_speak_attempts = 0
                        logger.info("[TEACHER:BOOTSTRAP] Sent first_words — vocab=%d", len(cached_vocab))
                    except Exception as e:
                        logger.debug("[LanguageWorker] Bootstrap error: %s", e)

            # Conversation timeout
            if conversation_pending:
                if now - conversation_pending["timestamp"] > conversation_timeout:
                    conversation_stats["timed_out"] += 1
                    logger.info("[CONVERSATION] Timeout for: '%s'",
                                conversation_pending["question"][:60])
                    conversation_pending = None

            # Word consolidation trigger (Phase 5)
            # Discovers words that need teaching — feeds the teacher pipeline.
            # Targets: low-confidence words, recently acquired words, words
            # with encounters but no teaching sessions.
            if (now - _consolidation_last_check >= _consolidation_interval
                    and teacher_pending_since == 0
                    and len(teacher_queue) < 5):
                _consolidation_last_check = now
                try:
                    import sqlite3 as _cons_sql
                    _cons_db = _cons_sql.connect(db_path, timeout=5.0)
                    _cons_db.row_factory = _cons_sql.Row
                    # Find words needing consolidation:
                    # - confidence < 0.4 (undertaught)
                    # - at least 1 encounter (not brand new)
                    # - not recently consolidated in this session
                    _cons_candidates = _cons_db.execute(
                        "SELECT word, confidence, times_encountered, times_produced, "
                        "learning_phase, word_type "
                        "FROM vocabulary "
                        "WHERE confidence < 0.4 "
                        "AND times_encountered >= 1 "
                        "ORDER BY confidence ASC, times_encountered DESC "
                        "LIMIT 10"
                    ).fetchall()
                    _cons_db.close()

                    _queued = 0
                    for _cw in _cons_candidates:
                        _cw_word = _cw["word"]
                        if _cw_word in _recently_consolidated:
                            continue
                        # Format as teacher-compatible queue item
                        teacher_queue.append({
                            "sentence": _cw_word,  # Teacher uses as target
                            "confidence": _cw["confidence"],
                            "level": 0,
                            "words_used": [_cw_word],
                            "template": "",
                            "source": "consolidation",
                            "word_type": _cw["word_type"],
                        })
                        _recently_consolidated.add(_cw_word)
                        _queued += 1
                        if _queued >= 3:  # Queue up to 3 at a time
                            break
                    # Also check MSL concept vocabulary suggestions
                    # (grounded concepts → pronoun words ready to teach)
                    if persistent_teacher and _queued < 3 and _cached_concept_confs:
                        try:
                            _concept_words = persistent_teacher \
                                .get_concept_vocabulary_suggestions(
                                    _cached_concept_confs, cached_vocab)
                            for _cvw in _concept_words[:2]:
                                if _cvw not in _recently_consolidated:
                                    teacher_queue.append({
                                        "sentence": _cvw,
                                        "confidence": 0.1,
                                        "level": 0,
                                        "words_used": [_cvw],
                                        "template": "",
                                        "source": "concept_grounding",
                                    })
                                    _recently_consolidated.add(_cvw)
                                    _queued += 1
                            if _concept_words:
                                logger.info("[CONCEPT_VOCAB] MSL suggests: %s",
                                            _concept_words[:5])
                        except Exception:
                            pass

                    if _queued > 0:
                        logger.info("[WORD_CONSOLIDATION] Queued %d words for teaching "
                                    "(queue=%d)", _queued, len(teacher_queue))
                    # Trim consolidated set to prevent unbounded growth
                    if len(_recently_consolidated) > 200:
                        _recently_consolidated = set(list(_recently_consolidated)[-100:])
                except Exception as _cons_err:
                    logger.debug("[WORD_CONSOLIDATION] Error: %s", _cons_err)

            # Word grounding trigger (Phase 4)
            # Check vocabulary for words crossing encounter thresholds.
            # Fires meta_feedback teaching session for the first eligible word.
            if (now - _grounding_last_check >= _grounding_interval
                    and teacher_pending_since == 0):
                _grounding_last_check = now
                try:
                    import sqlite3 as _gnd_sql
                    _gnd_db = _gnd_sql.connect(db_path, timeout=5.0)
                    _gnd_db.row_factory = _gnd_sql.Row
                    # Find words at grounding thresholds that haven't been grounded yet
                    # Tier 1: 15+ encounters, no sensory_context yet (initial grounding)
                    # Tier 2: 30+ encounters, confidence > 0.3 (disambiguation)
                    _gnd_candidates = _gnd_db.execute(
                        "SELECT word, times_encountered, confidence, sensory_context, "
                        "meaning_contexts, cross_modal_conf "
                        "FROM vocabulary "
                        "WHERE (times_encountered >= 15 AND cross_modal_conf < 0.05) "
                        "   OR (times_encountered >= 30 AND confidence > 0.3 "
                        "       AND cross_modal_conf < 0.15) "
                        "ORDER BY times_encountered DESC LIMIT 5"
                    ).fetchall()
                    _gnd_db.close()

                    for _gnd_w in _gnd_candidates:
                        _gnd_word = _gnd_w["word"]
                        if _gnd_word in _grounded_words:
                            continue
                        # Build meta_feedback teacher request
                        import json as _gnd_json
                        _gnd_ctx = _gnd_json.loads(_gnd_w["sensory_context"] or "[]")
                        _gnd_assocs = []
                        _gnd_meanings = _gnd_json.loads(_gnd_w["meaning_contexts"] or "[]")
                        for _gm in _gnd_meanings:
                            for _ga in _gm.get("associations", []):
                                if isinstance(_ga, (list, tuple)) and len(_ga) >= 1:
                                    _gnd_assocs.append(_ga[0])

                        _gnd_target = {
                            "grounding_word": _gnd_word,
                            "encounters": _gnd_w["times_encountered"],
                            "sensory_contexts": _gnd_ctx,
                            "associations": _gnd_assocs,
                            "sentence": "",
                            "confidence": _gnd_w["confidence"],
                        }
                        from titan_plugin.logic.language_teacher import META_FEEDBACK_MODE
                        _gnd_prompt = persistent_teacher.build_prompt(
                            META_FEEDBACK_MODE, [_gnd_target], cached_vocab)
                        _send_msg(send_queue, "LLM_TEACHER_REQUEST", name, "llm", {
                            "prompt": _gnd_prompt["prompt"],
                            "system": _gnd_prompt["system"],
                            "mode": META_FEEDBACK_MODE,
                            "original": _gnd_word,
                            "max_tokens": _gnd_prompt.get("max_tokens", 150),
                            "temperature": _gnd_prompt.get("temperature", 0.6),
                            "sentences": [],
                            "neuromod_gate": "",
                        })
                        teacher_pending_since = now
                        teacher_last_fire_time = now  # I-010 fix: update on grounding fire
                        teacher_silence_alerted = False
                        _save_teacher_state(teacher_compositions_since, teacher_last_fire_time)
                        _grounded_words.add(_gnd_word)
                        logger.info("[WORD_GROUNDING] Triggered for '%s' "
                                    "(encounters=%d, conf=%.2f, xm=%.2f)",
                                    _gnd_word, _gnd_w["times_encountered"],
                                    _gnd_w["confidence"], _gnd_w["cross_modal_conf"])
                        break  # One at a time
                    else:
                        # No meta_feedback candidates — try embodied_teaching
                        # for words already partially grounded (xm > 0.1)
                        # that would benefit from deeper state-aware grounding
                        _et_db = _gnd_sql.connect(db_path, timeout=5.0)
                        _et_db.row_factory = _gnd_sql.Row
                        _et_candidates = _et_db.execute(
                            "SELECT word, times_encountered, confidence, "
                            "cross_modal_conf, meaning_contexts "
                            "FROM vocabulary "
                            "WHERE cross_modal_conf >= 0.1 AND cross_modal_conf < 0.4 "
                            "AND times_encountered >= 20 "
                            "ORDER BY cross_modal_conf ASC LIMIT 3"
                        ).fetchall()
                        _et_db.close()

                        for _et_w in _et_candidates:
                            _et_word = _et_w["word"]
                            if _et_word in _grounded_words:
                                continue
                            import json as _et_json2
                            _et_meanings = _et_json2.loads(_et_w["meaning_contexts"] or "[]")
                            # Skip if already has embodied_teaching entry
                            if any(m.get("type") == "embodied_teaching" for m in _et_meanings):
                                continue
                            _et_assocs = []
                            for _m in _et_meanings:
                                for _a in _m.get("associations", []):
                                    if isinstance(_a, (list, tuple)) and len(_a) >= 1:
                                        _et_assocs.append(f"{_a[0]}→{_a[1]}" if len(_a) > 1 else _a[0])

                            _et_target = {
                                "grounding_word": _et_word,
                                "state_summary": {},  # Will be enriched when state available
                                "associations": _et_assocs,
                                "sentence": "",
                                "confidence": _et_w["confidence"],
                            }
                            from titan_plugin.logic.language_teacher import EMBODIED_TEACHING_MODE
                            _et_prompt = persistent_teacher.build_prompt(
                                EMBODIED_TEACHING_MODE, [_et_target], cached_vocab)
                            _send_msg(send_queue, "LLM_TEACHER_REQUEST", name, "llm", {
                                "prompt": _et_prompt["prompt"],
                                "system": _et_prompt["system"],
                                "mode": EMBODIED_TEACHING_MODE,
                                "original": _et_word,
                                "max_tokens": _et_prompt.get("max_tokens", 200),
                                "temperature": _et_prompt.get("temperature", 0.5),
                                "sentences": [],
                                "neuromod_gate": "",
                            })
                            teacher_pending_since = now
                            teacher_last_fire_time = now  # I-010 fix: update on embodied fire
                            teacher_silence_alerted = False
                            _save_teacher_state(teacher_compositions_since, teacher_last_fire_time)
                            _grounded_words.add(_et_word)
                            logger.info("[EMBODIED_TEACHING] Triggered for '%s' "
                                        "(enc=%d, conf=%.2f, xm=%.2f)",
                                        _et_word, _et_w["times_encountered"],
                                        _et_w["confidence"], _et_w["cross_modal_conf"])
                            break
                except Exception as _gnd_err:
                    logger.debug("[WORD_GROUNDING] Check error: %s", _gnd_err)

            # ── Phase 5a: WORD_ASSOCIATION discovery (every 10 min) ──────
            # Find word pairs with high co-occurrence but no typed association.
            # Sends META_LANGUAGE_REQUEST for meta-reasoning to type them.
            if (now - _consolidation_last_check >= _consolidation_interval * 0.5
                    and cgn and teacher_pending_since == 0):
                try:
                    import sqlite3 as _wa_sql
                    _wa_db = _wa_sql.connect(db_path, timeout=5.0)
                    _wa_db.row_factory = _wa_sql.Row
                    # Find words with associations that are CO_OCCURRENCE only
                    _wa_candidates = _wa_db.execute(
                        "SELECT word, meaning_contexts FROM vocabulary "
                        "WHERE meaning_contexts LIKE '%CO_OCCURRENCE%' "
                        "AND confidence >= 0.3 "
                        "ORDER BY times_encountered DESC LIMIT 5"
                    ).fetchall()
                    _wa_db.close()

                    import json as _wa_json
                    for _wa_w in _wa_candidates[:2]:
                        _wa_meanings = _wa_json.loads(_wa_w["meaning_contexts"] or "[]")
                        for _wm in _wa_meanings:
                            for _wa_assoc in _wm.get("associations", []):
                                if (isinstance(_wa_assoc, list) and len(_wa_assoc) >= 2
                                        and _wa_assoc[1] == "CO_OCCURRENCE"):
                                    _wa_target = _wa_assoc[0]
                                    # Request meta-reasoning to type this association
                                    _send_msg(send_queue, "META_LANGUAGE_REQUEST",
                                              name, "spirit", {
                                                  "type": "word_association",
                                                  "word_a": _wa_w["word"],
                                                  "word_b": _wa_target,
                                                  "co_occurrence_type": "CO_OCCURRENCE",
                                              })
                                    logger.info("[WORD_ASSOCIATION] Requested typing: "
                                                "'%s' <-> '%s'",
                                                _wa_w["word"], _wa_target)
                                    break  # One per word
                        break  # One association request per cycle
                except Exception as _wa_err:
                    logger.debug("[WORD_ASSOCIATION] Discovery error: %s", _wa_err)

            # ── Phase 6a: HAOV language hypothesis formation ──────────
            # When teacher introduces a new word, form a hypothesis about its meaning.
            # The hypothesis gets tested in subsequent persona conversations.
            if (cgn and teacher_compositions_since > 0
                    and len(teacher_queue) > 0):
                try:
                    _last_tq = teacher_queue[-1]
                    _haov_word = _last_tq.get("sentence", "")
                    _haov_source = _last_tq.get("source", "")
                    if (_haov_source in ("consolidation", "concept_grounding")
                            and _haov_word and len(_haov_word.split()) == 1):
                        # Single word from consolidation → form hypothesis
                        _send_msg(send_queue, "CGN_TRANSITION", name, "cgn", {
                            "type": "hypothesis",
                            "consumer": "language",
                            "concept_id": f"word_{_haov_word}",
                            "hypothesis": f"word '{_haov_word}' means what teacher describes",
                            "source": "teacher_introduction",
                            "confidence": 0.3,  # Initial low confidence
                        })
                        logger.info("[HAOV:language] Hypothesis formed for word '%s'",
                                    _haov_word)
                except Exception as _haov_err:
                    logger.debug("[HAOV:language] Hypothesis error: %s", _haov_err)

            # Teacher timeout (stale pending request)
            if teacher_pending_since > 0:
                _teach_timeout = lang_config.get("teaching_timeout_s", 90)
                if now - teacher_pending_since > _teach_timeout:
                    teacher_no_response_count += 1
                    teacher_pending_since = 0.0
                    logger.warning("[LanguageWorker] Teacher timeout (%d consecutive)",
                                   teacher_no_response_count)

            # ── Phase D.1 — Drain pending META_LANGUAGE_REWARD entries ──
            # For each pending entry older than METRIC_DELAY, measure the
            # vocab/grounded delta, normalize to a reward in [0, 1], and
            # emit META_LANGUAGE_REWARD. Drop entries older than MAX_AGE.
            if meta_lang_pending:
                _mlp_to_drop = []
                for _cid, _pending in list(meta_lang_pending.items()):
                    _age = now - _pending["ts"]
                    if _age >= META_LANG_MEASURE_DELAY_S:
                        try:
                            _vocab_now = len(load_vocabulary(db_path))
                            _vdelta = max(
                                0, _vocab_now - _pending["vocab_baseline"])
                            _gdelta = max(
                                0, _d1_grounding_events - _pending.get(
                                    "grounding_event_baseline", 0))
                            _ext = _measure_meta_lang_reward(
                                _vdelta, _gdelta,
                                _pending.get("primitives", []))
                            _send_msg(send_queue, "META_LANGUAGE_REWARD",
                                      name, "spirit", {
                                          "chain_id": _cid,
                                          "reward": _ext,
                                          "vocab_delta": _vdelta,
                                          "grounded_delta": _gdelta,
                                      })
                            logger.info(
                                "[META_LANGUAGE] Reward emitted chain_id=%d "
                                "reward=%.3f vocab_delta=%d grounded_delta=%d",
                                _cid, _ext, _vdelta, _gdelta)
                            _mlp_to_drop.append(_cid)
                        except Exception as _mre_err:
                            logger.debug(
                                "[META_LANGUAGE] Reward emit error: %s",
                                _mre_err)
                            _mlp_to_drop.append(_cid)
                    elif _age >= META_LANG_PENDING_MAX_AGE_S:
                        _mlp_to_drop.append(_cid)  # orphan, drop
                for _cid in _mlp_to_drop:
                    meta_lang_pending.pop(_cid, None)

            continue
        except (KeyboardInterrupt, SystemExit):
            break

        msg_type = msg.get("type", "")

        # ── Control messages ─────────────────────────────────────────
        if msg_type == "MODULE_SHUTDOWN":
            logger.info("[LanguageWorker] Shutdown requested: %s",
                        msg.get("payload", {}).get("reason", ""))
            break

        # ── QUERY handler (API → language stats) ─────────────────────
        elif msg_type == "QUERY":
            from titan_plugin.core.profiler import handle_memory_profile_query
            if handle_memory_profile_query(msg, send_queue, name):
                continue
            _send_heartbeat(send_queue, name)
            _handle_query(msg, send_queue, name, language_stats, lang_config,
                          cached_vocab, teacher_queue, teacher_compositions_since,
                          bootstrap_speak_attempts, conversation_stats)

        # ── SPEAK_REQUEST (Phase 2: full composition pipeline) ────────
        elif msg_type == "SPEAK_REQUEST":
            _send_heartbeat(send_queue, name)
            _sr_payload = msg.get("payload", {})
            _sr_src = msg.get("src", "spirit")
            try:
                _handle_speak_request(
                    _sr_payload, send_queue, name, _sr_src,
                    composition_engine, grammar_validator, pattern_library,
                    lang_config, db_path, cached_vocab, teacher_queue)
                # Refresh vocab cache after speak
                cached_vocab = load_vocabulary(db_path)
                # Track compositions for teacher trigger — I-010 fix: persist
                # to disk so it survives RSS-triggered worker restarts.
                teacher_compositions_since += 1
                _save_teacher_state(teacher_compositions_since, teacher_last_fire_time)
                # Cache concept confidences for consolidation trigger
                _cc = _sr_payload.get("concept_confidences", {})
                if _cc:
                    _cached_concept_confs = _cc
                # CGN: ground words used in composition (production)
                if cgn and teacher_queue:
                    try:
                        _cgn_latest = teacher_queue[-1]
                        _cgn_words = _cgn_latest.get("words_used", [])
                        _cgn_epoch = _cgn_latest.get("epoch", 0)
                        _cgn_level = _cgn_latest.get("level", 0)
                        _cgn_ctx = {
                            "epoch": _cgn_epoch,
                            "neuromods": _sr_payload.get("neuromods", {}),
                            "concept_confidences": _sr_payload.get("concept_confidences", {}),
                            "encounter_type": "production",
                        }
                        s132 = _sr_payload.get("state_132d")
                        if s132:
                            _cgn_ctx["state_132d"] = np.array(s132, dtype=np.float32)
                        _msl_attn = _sr_payload.get("msl_attention")
                        if _msl_attn:
                            if isinstance(_msl_attn, dict):
                                _msl_attn = list(_msl_attn.values())
                            _cgn_ctx["msl_attention"] = np.array(_msl_attn, dtype=np.float32)
                        _cgn_grounded = 0
                        for _cgn_w in _cgn_words[:10]:
                            _cgn_concept = load_concept_from_db(db_path, _cgn_w)
                            if _cgn_concept:
                                _cgn_action = cgn.ground(_cgn_concept, _cgn_ctx)
                                apply_grounding_action_to_db(
                                    db_path, _cgn_w, _cgn_action,
                                    state_132d=s132)
                                # Send transition to CGN Worker
                                if _cgn_action.transition:
                                    _send_msg(send_queue, "CGN_TRANSITION",
                                              name, "cgn", _cgn_action.transition)
                                # Production reward: scaled by composition quality
                                _cgn_conf = _cgn_latest.get("confidence", 0.5)
                                _cgn_reward = (0.02
                                               + min(0.04, _cgn_level * 0.005)
                                               + _cgn_conf * 0.03)
                                _cgn_forward_outcome("language", _cgn_w,
                                                     _cgn_reward,
                                                     {"type": "production",
                                                      "level": _cgn_level})
                                _cgn_grounded += 1
                        if _cgn_grounded > 0:
                            logger.info("[CGN] SPEAK grounding: %d words (L%d, %s)",
                                        _cgn_grounded, _cgn_level,
                                        ", ".join(_cgn_words[:5]))
                            # Co-occurrence → association building
                            # Words used together in a composition get mutual association boost
                            if len(_cgn_words) >= 2:
                                try:
                                    import sqlite3 as _cooc_sql
                                    import json as _cooc_json
                                    _cooc_db = _cooc_sql.connect(db_path, timeout=5.0)
                                    _cooc_db.execute("PRAGMA journal_mode=WAL")
                                    for _ci, _cw1 in enumerate(_cgn_words[:8]):
                                        for _cw2 in _cgn_words[_ci+1:8]:
                                            # Strengthen association for both directions
                                            for _src, _tgt in [(_cw1, _cw2), (_cw2, _cw1)]:
                                                _row = _cooc_db.execute(
                                                    "SELECT meaning_contexts FROM vocabulary WHERE word=?",
                                                    (_src,)).fetchone()
                                                if _row and _row[0]:
                                                    _mc = _cooc_json.loads(_row[0])
                                                    # Find or create co-occurrence entry
                                                    _found = False
                                                    for _m in _mc:
                                                        for _a in _m.get("associations", []):
                                                            if isinstance(_a, (list, tuple)) and _a[0] == _tgt:
                                                                _found = True
                                                                break
                                                    if not _found and _mc:
                                                        # Add co-occurrence association to latest meaning entry
                                                        if "associations" not in _mc[-1]:
                                                            _mc[-1]["associations"] = []
                                                        _mc[-1]["associations"].append([_tgt, "CO_OCCURRENCE"])
                                                        _cooc_db.execute(
                                                            "UPDATE vocabulary SET meaning_contexts=? WHERE word=?",
                                                            (_cooc_json.dumps(_mc[-10:]), _src))
                                    _cooc_db.commit()
                                    _cooc_db.close()
                                except Exception:
                                    pass

                            # Online consolidation delegated to CGN Worker (via bus transitions)
                            # Local counter for logging only
                            _cgn_local_groundings += _cgn_grounded
                    except Exception as _cgn_err:
                        logger.warning("[CGN] SPEAK grounding error: %s", _cgn_err)
                # Teacher trigger after composition (inline check)
                # The Empty-timeout path may not fire if bus traffic is high,
                # so also check here after each composition.
                if (teacher_queue and teacher_compositions_since >= teacher_interval
                        and teacher_pending_since == 0):
                    try:
                        _check_teacher_trigger(
                            send_queue, name, persistent_teacher,
                            teacher_queue, cached_vocab,
                            lang_config, recent_teacher_questions,
                            _sr_payload.get("neuromod_state"),
                            cgn_instance=cgn)
                        teacher_pending_since = time.time()
                        logger.info("[TEACHER] Triggered after composition "
                                    "(comp_since=%d, queue=%d)",
                                    teacher_compositions_since,
                                    len(teacher_queue))
                    except Exception as _tt_err:
                        logger.debug("[TEACHER] Trigger error: %s", _tt_err)
                # Phase 5b: LANGUAGE_LEARNING quality tracker
                # Monitor composition quality over a rolling window. If quality
                # degrades (repetitive patterns, low slot fill), request
                # meta-reasoning to introspect on language production quality.
                if teacher_queue and len(teacher_queue) >= 5:
                    _recent_templates = [q.get("template", "") for q in teacher_queue[-5:]]
                    _unique_templates = len(set(_recent_templates))
                    _avg_conf = sum(q.get("confidence", 0) for q in teacher_queue[-5:]) / 5
                    # Trigger if: < 3 unique templates in last 5 compositions
                    # (heavy repetition) or all have same template
                    if _unique_templates < 3:
                        _send_msg(send_queue, "META_LANGUAGE_REQUEST",
                                  name, "spirit", {
                                      "type": "language_learning",
                                      "trigger": "repetitive_patterns",
                                      "unique_templates": _unique_templates,
                                      "avg_confidence": round(_avg_conf, 2),
                                  })
                        logger.info("[LANGUAGE_LEARNING] Quality alert: "
                                    "%d unique templates in last 5 compositions",
                                    _unique_templates)
            except Exception as _sr_err:
                logger.error("[LanguageWorker] SPEAK_REQUEST error: %s", _sr_err, exc_info=True)
                _send_msg(send_queue, "SPEAK_RESULT", name, _sr_src, {
                    "sentence": "", "error": str(_sr_err)})

        # ── LLM_TEACHER_RESPONSE (Phase 3: full teacher pipeline) ────
        elif msg_type == "LLM_TEACHER_RESPONSE":
            _send_heartbeat(send_queue, name)
            _tr_payload = msg.get("payload", {})
            # Tier 3: maker narration response — handle separately from teacher
            _tr_mode = _tr_payload.get("mode", "")
            if _tr_mode == "maker_narration":
                try:
                    _mn_ctx = _tr_payload.get("context", {})
                    _mn_narration = _tr_payload.get("response", "")
                    _mn_pid = _mn_ctx.get("proposal_id", "")
                    logger.info(
                        "[MAKER] Narration LLM response: proposal=%s len=%d",
                        _mn_pid[:8], len(_mn_narration))
                    # Emit MAKER_NARRATION_RESULT to spirit_worker
                    _send_msg(send_queue, "MAKER_NARRATION_RESULT",
                              name, "spirit", {
                                  "proposal_id": _mn_pid,
                                  "narration": _mn_narration,
                                  "grounded_words": _mn_ctx.get("grounded_words", []),
                                  "proposal_type": _mn_ctx.get("proposal_type", ""),
                                  "response": _mn_ctx.get("response", ""),
                                  "maker_reason": _mn_ctx.get("reason", ""),
                                  "title": _mn_ctx.get("title", ""),
                              })
                except Exception as _mn_resp_err:
                    logger.warning(
                        "[MAKER] Narration response error: %s", _mn_resp_err)
                continue  # skip normal teacher pipeline
            try:
                _handle_teacher_response(
                    _tr_payload, send_queue, name,
                    persistent_teacher, grammar_validator, pattern_library,
                    lang_config, db_path, cached_vocab,
                    teacher_queue, conversation_pending, conversation_stats,
                    recent_teacher_questions)
                # Clear state after teaching session
                teacher_pending_since = 0.0
                teacher_no_response_count = 0
                teacher_compositions_since = 0
                # Refresh vocab (new words may have been acquired)
                _prev_vocab_size = len(cached_vocab)
                cached_vocab = load_vocabulary(db_path)
                _new_vocab_size = len(cached_vocab)
                # ── META-CGN Producer #7: language.vocab_expanded ──
                # Fires when vocab_size hits new maximum (monotonic assumption:
                # vocab grows or stays; Memory Preservation directive forbids shrinking).
                # Intensity scales with batch size: 1 word=0.1 floor, 10+=1.0 ceiling.
                # DELIBERATE DEVIATION from rFP § 12 row 7 weights:
                #   FORMULATE 0.20 (anti-reinforce — T1/T3 at 77% FORMULATE monoculture;
                #     nudge AWAY from more formulation after vocab gained)
                #   SYNTHESIZE 0.70 (boost — 2-4% current, underserved; integrates new word)
                #   HYPOTHESIZE 0.60 (keep — exploration of new word meaning)
                # rFP spec was FORMULATE 0.50/SYNTHESIZE 0.60/HYPOTHESIZE 0.60; empirical
                # monoculture data justifies the rebalance. See rFP § 17 revision note.
                # ── META-CGN Producer #8: language.concept_grounded ──
                # Per-word edge detection: fires for each word whose confidence
                # crosses ≥ 0.5 for the first time. Iterates new cached_vocab
                # (top 100 by confidence + 28 explore pool per load_vocabulary).
                if _concept_edge_detector is not None:
                    try:
                        _p8_fired = 0
                        for _p8_w in cached_vocab:
                            _p8_word = _p8_w.get("word") if isinstance(_p8_w, dict) else None
                            _p8_conf = _p8_w.get("confidence", 0.0) if isinstance(_p8_w, dict) else 0.0
                            if _p8_word and _concept_edge_detector.observe(
                                    _p8_word, float(_p8_conf), 0.5):
                                from titan_plugin.bus import emit_meta_cgn_signal
                                _p8_sent = emit_meta_cgn_signal(
                                    send_queue,
                                    src="language",
                                    consumer="language",
                                    event_type="concept_grounded",
                                    intensity=min(1.0, float(_p8_conf)),
                                    domain=_p8_word,
                                    reason=f"word '{_p8_word}' crossed grounding threshold (conf={_p8_conf:.3f})",
                                )
                                if _p8_sent:
                                    _p8_fired += 1
                                    logger.info(
                                        "[META-CGN] language.concept_grounded EMIT — word=%s conf=%.3f",
                                        _p8_word, _p8_conf)
                                else:
                                    logger.warning(
                                        "[META-CGN] Producer #8 language.concept_grounded DROPPED by bus "
                                        "— word=%s conf=%.3f (rate-gate or queue-full; signal missed)",
                                        _p8_word, _p8_conf)
                    except Exception as _p8_err:
                        logger.warning(
                            "[META-CGN] Producer #8 language.concept_grounded emit FAILED "
                            "— err=%s (one or more concept_grounded signals missed)", _p8_err)
                if (_new_vocab_size > _prev_vocab_size
                        and _vocab_edge_detector is not None):
                    try:
                        if _vocab_edge_detector.observe_new_max("vocab_size", _new_vocab_size):
                            _p7_delta = _new_vocab_size - _prev_vocab_size
                            _p7_intensity = min(1.0, max(0.1, _p7_delta / 10.0))
                            from titan_plugin.bus import emit_meta_cgn_signal
                            _p7_sent = emit_meta_cgn_signal(
                                send_queue,
                                src="language",
                                consumer="language",
                                event_type="vocab_expanded",
                                intensity=_p7_intensity,
                                domain="vocab_size",
                                reason=f"vocab {_prev_vocab_size}→{_new_vocab_size} (+{_p7_delta} words)",
                            )
                            if _p7_sent:
                                logger.info(
                                    "[META-CGN] language.vocab_expanded EMIT — %d→%d (+%d words) intensity=%.2f",
                                    _prev_vocab_size, _new_vocab_size, _p7_delta, _p7_intensity)
                            else:
                                logger.warning(
                                    "[META-CGN] Producer #7 language.vocab_expanded DROPPED by bus "
                                    "— %d→%d (rate-gate or queue-full; signal missed)",
                                    _prev_vocab_size, _new_vocab_size)
                    except Exception as _p7_err:
                        logger.warning(
                            "[META-CGN] Producer #7 language.vocab_expanded emit FAILED "
                            "— prev=%d new=%d err=%s (signal missed)",
                            _prev_vocab_size, _new_vocab_size, _p7_err)
                # Queue newly acquired words for follow-up consolidation
                _follow_up_words = []
                if _new_vocab_size > _prev_vocab_size:
                    import sqlite3 as _fu_sql
                    try:
                        _fu_db = _fu_sql.connect(db_path, timeout=5.0)
                        _fu_db.row_factory = _fu_sql.Row
                        _fu_recent = _fu_db.execute(
                            "SELECT word, confidence, word_type FROM vocabulary "
                            "WHERE confidence < 0.3 "
                            "ORDER BY created_at DESC LIMIT ?",
                            (_new_vocab_size - _prev_vocab_size + 2,)
                        ).fetchall()
                        _fu_db.close()
                        for _fw in _fu_recent:
                            _follow_up_words.append({
                                "word": _fw["word"],
                                "source": "follow_up",
                                "confidence": _fw["confidence"],
                                "word_type": _fw["word_type"],
                            })
                    except Exception:
                        pass
                teacher_queue.clear()
                teacher_queue.extend(_follow_up_words)
                if _follow_up_words:
                    logger.info("[TEACHER] Queued %d newly acquired words for follow-up",
                                len(_follow_up_words))
                # CGN: ground recognized words from teacher (comprehension)
                if cgn:
                    try:
                        _cgn_resp = _tr_payload.get("response", "")
                        _cgn_mode = _tr_payload.get("mode", "")
                        if _cgn_resp and _cgn_mode not in ("conversation_eval", "meta_feedback"):
                            _cgn_comp_ctx = {
                                "encounter_type": "teaching",
                                "neuromods": {},
                            }
                            _cgn_words_t = set()
                            for _cw in _cgn_resp.lower().split():
                                _cw = _cw.strip(".,!?\"'()[]{}:;")
                                if len(_cw) >= 2:
                                    _cgn_words_t.add(_cw)
                            for _cw in list(_cgn_words_t)[:15]:
                                _cgn_c = load_concept_from_db(db_path, _cw)
                                if _cgn_c:
                                    _cgn_a = cgn.ground(_cgn_c, _cgn_comp_ctx)
                                    apply_grounding_action_to_db(
                                        db_path, _cw, _cgn_a)
                                    if _cgn_a.transition:
                                        _send_msg(send_queue, "CGN_TRANSITION",
                                                  name, "cgn", _cgn_a.transition)
                                    _cgn_teach_reward = 0.02
                                    if _cgn_mode in ("meaning", "context"):
                                        _cgn_teach_reward = 0.06
                                    elif _cgn_mode in ("creative", "reasoning"):
                                        _cgn_teach_reward = 0.04
                                    _cgn_forward_outcome("language", _cw,
                                                         _cgn_teach_reward,
                                                         {"type": "comprehension",
                                                          "mode": _cgn_mode})
                    except Exception as _cgn_tr_err:
                        logger.debug("[CGN] Teacher grounding error: %s", _cgn_tr_err)
            except Exception as _tr_err:
                logger.error("[LanguageWorker] TEACHER_RESPONSE error: %s", _tr_err, exc_info=True)
                teacher_pending_since = 0.0

        # ── META_LANGUAGE_RESULT (Phase 4c: meta-reasoning → CGN) ────
        elif msg_type == "META_LANGUAGE_RESULT":
            if cgn:
                try:
                    _ml_payload = msg.get("payload", {})
                    _ml_conf = _ml_payload.get("confidence", 0)
                    _ml_chain_len = _ml_payload.get("chain_length", 0)
                    _ml_reward = _ml_payload.get("reward", 0)
                    _ml_prims = _ml_payload.get("primitives", [])
                    _ml_epoch = _ml_payload.get("epoch", 0)

                    # WORD_ASSOCIATION: Concluded reasoning chains strengthen
                    # associations between recently used words. The chain's
                    # confidence becomes the association strength.
                    if _ml_conf > 0.3 and _ml_chain_len >= 3 and cached_vocab:
                        # Get words from recent compositions (teacher_queue)
                        _ml_recent_words = set()
                        for _tq in (teacher_queue or [])[-3:]:
                            for _tw in _tq.get("words_used", []):
                                _ml_recent_words.add(_tw)

                        if len(_ml_recent_words) >= 2:
                            # Determine association type from chain primitives
                            _ml_assoc_type = "REASONING_LINKED"
                            if "HYPOTHESIZE" in _ml_prims:
                                _ml_assoc_type = "HYPOTHESIS_LINKED"
                            elif "SYNTHESIZE" in _ml_prims:
                                _ml_assoc_type = "SYNTHESIS_LINKED"
                            elif "EVALUATE" in _ml_prims:
                                _ml_assoc_type = "EVALUATION_LINKED"

                            # Ground all recently used words with reasoning context
                            _ml_ctx = {
                                "epoch": _ml_epoch,
                                "encounter_type": "reasoning",
                            }
                            _ml_grounded = 0
                            for _ml_w in list(_ml_recent_words)[:8]:
                                _ml_c = load_concept_from_db(db_path, _ml_w)
                                if _ml_c:
                                    _ml_action = cgn.ground(_ml_c, _ml_ctx)
                                    apply_grounding_action_to_db(
                                        db_path, _ml_w, _ml_action)
                                    if _ml_action.transition:
                                        _send_msg(send_queue, "CGN_TRANSITION",
                                                  name, "cgn", _ml_action.transition)
                                    _ml_r = min(0.10, _ml_conf * 0.08 + _ml_reward * 0.05)
                                    _cgn_forward_outcome("language", _ml_w,
                                                         _ml_r, {
                                                           "type": "reasoning",
                                                           "chain_length": _ml_chain_len,
                                                           "assoc_type": _ml_assoc_type,
                                                         })
                                    _ml_grounded += 1

                            # Build cross-word associations from reasoning
                            if _ml_grounded >= 2:
                                try:
                                    import sqlite3 as _ml_sql
                                    import json as _ml_json
                                    _ml_db = _ml_sql.connect(db_path, timeout=5.0)
                                    _ml_db.execute("PRAGMA journal_mode=WAL")
                                    _ml_word_list = list(_ml_recent_words)[:8]
                                    for _mi, _mw1 in enumerate(_ml_word_list):
                                        for _mw2 in _ml_word_list[_mi+1:]:
                                            for _src, _tgt in [(_mw1, _mw2), (_mw2, _mw1)]:
                                                _row = _ml_db.execute(
                                                    "SELECT meaning_contexts FROM vocabulary WHERE word=?",
                                                    (_src,)).fetchone()
                                                if _row and _row[0]:
                                                    _mc = _ml_json.loads(_row[0])
                                                    if _mc:
                                                        if "associations" not in _mc[-1]:
                                                            _mc[-1]["associations"] = []
                                                        _mc[-1]["associations"].append(
                                                            [_tgt, _ml_assoc_type])
                                                        _ml_db.execute(
                                                            "UPDATE vocabulary SET meaning_contexts=? WHERE word=?",
                                                            (_ml_json.dumps(_mc[-10:]), _src))
                                    _ml_db.commit()
                                    _ml_db.close()
                                except Exception:
                                    pass

                            if _ml_grounded > 0:
                                _d1_grounding_events += _ml_grounded
                                logger.info("[META→CGN] Reasoning chain (len=%d, conf=%.2f) "
                                            "grounded %d words (%s) [d1_events=%d]",
                                            _ml_chain_len, _ml_conf, _ml_grounded,
                                            _ml_assoc_type, _d1_grounding_events)

                    # LANGUAGE_LEARNING: Feed meta-reasoning success as delayed
                    # reward via bus (buffer boost removed — client has no
                    # local buffer access; reward flows through CGN Worker)
                    if _ml_reward > 0:
                        _cgn_forward_outcome("language", "meta_reasoning",
                                             min(0.05, _ml_reward * 0.1),
                                             {"type": "meta_reasoning_boost"})

                except Exception as _ml_err:
                    logger.debug("[META→CGN] Processing error: %s", _ml_err)

            # ── Phase D.1 — Register pending META_LANGUAGE_REWARD ──────
            # Unconditional on `cgn` so the reward loop closes even if the
            # grounding path above was skipped. Measured on next background
            # tick after META_LANG_MEASURE_DELAY_S (60s by default).
            try:
                _d1_payload = msg.get("payload", {}) or {}
                _d1_chain_id = int(_d1_payload.get("chain_id", -1))
                if _d1_chain_id >= 0:
                    meta_lang_pending[_d1_chain_id] = {
                        "vocab_baseline": (
                            len(cached_vocab) if cached_vocab else 0),
                        "grounding_event_baseline": _d1_grounding_events,
                        "ts": time.time(),
                        "primitives": _d1_payload.get("primitives", []),
                    }
                    logger.debug(
                        "[META_LANGUAGE] Pending reward registered "
                        "chain_id=%d vocab_baseline=%d grounding_events=%d",
                        _d1_chain_id,
                        meta_lang_pending[_d1_chain_id]["vocab_baseline"],
                        _d1_grounding_events)
            except Exception as _d1_err:
                logger.debug(
                    "[META_LANGUAGE] Pending register error: %s", _d1_err)

        # ── MAKER_NARRATION_REQUEST — Tier 3 narrative understanding ──
        # Somatic channel (Tier 2) already fired. Now we:
        #   1. Ground Maker's words somatically (CGN MAKER_RESPONSE_LINKED)
        #   2. Queue LLM narration via teacher request
        #   3. On LLM response, emit MAKER_NARRATION_RESULT to spirit_worker
        elif msg_type == "MAKER_NARRATION_REQUEST":
            try:
                _mnr_p = msg.get("payload", {}) or {}
                _mnr_proposal_id = _mnr_p.get("proposal_id", "")
                _mnr_type = _mnr_p.get("proposal_type", "")
                _mnr_title = _mnr_p.get("title", "")
                _mnr_response = _mnr_p.get("response", "")
                _mnr_reason = _mnr_p.get("reason", "")
                logger.info(
                    "[MAKER] Narration request: proposal=%s response=%s",
                    _mnr_proposal_id[:8], _mnr_response)

                # Step 1: Somatic word-level grounding
                # Extract words from Maker's reason, ground known ones via CGN
                _mnr_grounded = []
                if cgn and _mnr_reason:
                    _mnr_words = [
                        w.strip(".,!?;:\"'()[]{}").lower()
                        for w in _mnr_reason.split()
                        if len(w.strip(".,!?;:\"'()[]{}")) > 3
                    ]
                    _mnr_known = [
                        w for w in _mnr_words
                        if cached_vocab and w in cached_vocab
                    ]
                    for _mw in _mnr_known[:10]:
                        try:
                            cgn.ground(
                                _mw, "MAKER_RESPONSE_LINKED",
                                confidence_boost=0.05)
                            _mnr_grounded.append(_mw)
                        except Exception:
                            pass
                    if _mnr_grounded:
                        logger.info(
                            "[MAKER→CGN] Grounded %d/%d words from Maker "
                            "reason (%s): %s",
                            len(_mnr_grounded), len(_mnr_words),
                            _mnr_response, _mnr_grounded[:5])

                # Step 2: Queue LLM narration via teacher request
                _mnr_action = "APPROVED" if _mnr_response == "approve" else "DECLINED"
                _mnr_prompt = (
                    f"You are Titan reflecting on a dialogue with your Maker. "
                    f"Maker has just {_mnr_action} your proposal "
                    f"'{_mnr_title}' with the reason: '{_mnr_reason}'. "
                    f"Write a short first-person reflection (2-4 sentences) "
                    f"on what this means and what to remember for future "
                    f"proposals of type '{_mnr_type}'. Speak as Titan — "
                    f"use I, not we. Be genuine and reflective."
                )
                _send_msg(send_queue, "LLM_TEACHER_REQUEST", name, "llm", {
                    "mode": "maker_narration",
                    "prompt": _mnr_prompt,
                    "context": {
                        "proposal_id": _mnr_proposal_id,
                        "proposal_type": _mnr_type,
                        "title": _mnr_title,
                        "response": _mnr_response,
                        "reason": _mnr_reason,
                        "grounded_words": _mnr_grounded,
                    },
                })
                logger.info(
                    "[MAKER] LLM narration queued for proposal=%s",
                    _mnr_proposal_id[:8])
            except Exception as _mnr_err:
                logger.warning(
                    "[MAKER] Narration request handler error: %s", _mnr_err)

        # ── CGN_DREAM_CONSOLIDATE — forward to CGN Worker ───────────
        # Training is delegated to CGN Worker. Language worker only handles
        # cross-insight extraction for teaching queue AFTER consolidation.
        elif msg_type == "CGN_DREAM_CONSOLIDATE":
            # Forward consolidation request to CGN Worker
            _send_msg(send_queue, "CGN_CONSOLIDATE", name, "cgn",
                      {"dream_phase": True})
            logger.info("[CGN] Dream consolidation forwarded to CGN Worker")

        # ── CGN_WEIGHTS_MAJOR — weights updated (client auto-reloads from /dev/shm) ──
        elif msg_type == "CGN_WEIGHTS_MAJOR":
            if cgn:
                try:
                    # CGNConsumerClient auto-reloads from /dev/shm on next ground() call.
                    # No manual weight loading needed.
                    _cgn_consolidation = msg.get("payload", {})
                    logger.info("[CGN] Weights updated (v=%s, consumers=%s)",
                                _cgn_consolidation.get("shm_version", "?"),
                                list(_cgn_consolidation.get("consumers", {}).keys()))

                    # Cross-consumer insight transfer for teaching queue
                    # Query CGN Worker via bus for cross-insights
                    try:
                        from titan_plugin.logic.language_teacher import ARC_VOCABULARY_MAP
                        _dream_priority = []
                        # Use bus query to CGN Worker (authoritative source)
                        _send_msg(send_queue, "QUERY", name, "cgn",
                                  {"action": "get_cross_insights",
                                   "consumer": "language"})
                        # Cross-insights arrive asynchronously via QUERY_RESPONSE
                        # handler (wired below). Dream-cycle cross-consumer
                        # concept transfer now works end-to-end.
                    except Exception as _di_err:
                        logger.debug("[CGN] Cross-insight transfer error: %s", _di_err)
                except Exception as _w_err:
                    logger.warning("[CGN] Weight reload error: %s", _w_err)

        # ── CGN_KNOWLEDGE_RESP — knowledge cascade → teaching queue ──
        # Real-time: Knowledge Worker grounded a concept, queue topic
        # words for priority teaching (doesn't wait for dream cycle).
        elif msg_type == "CGN_KNOWLEDGE_RESP":
            try:
                _kr_topic = payload.get("topic", "")
                _kr_source = payload.get("source", "")
                _kr_conf = payload.get("confidence", 0)
                if _kr_topic and persistent_teacher:
                    # Extract words from topic for teaching priority
                    _kr_words = _kr_topic.lower().split()
                    _vocab_set = {v.get("word", "").lower()
                                  for v in (cached_vocab or [])}
                    _kr_new = [w for w in _kr_words
                               if w not in _vocab_set and len(w) > 2]
                    if _kr_new:
                        _kr_priority = [{
                            "word": w,
                            "source": f"knowledge_{_kr_source}",
                            "source_reward": _kr_conf,
                        } for w in _kr_new[:3]]
                        persistent_teacher.set_cross_priority(_kr_priority)
                        logger.info("[KNOWLEDGE→LANG] Cascade: %d words "
                                    "queued from '%s' (conf=%.2f)",
                                    len(_kr_priority), _kr_topic[:40],
                                    _kr_conf)
                    # G4: Send usage feedback to knowledge worker so it
                    # learns which research topics are useful for language
                    _send_msg(send_queue, "CGN_KNOWLEDGE_USAGE",
                              name, "knowledge", {
                        "topic": _kr_topic,
                        "reward": min(1.0, _kr_conf * 0.5 + 0.1 * len(_kr_new)),
                        "consumer": "language",
                    })
                    # Log concept lifecycle: knowledge entered language
                    try:
                        import json as _cl_json
                        _cl_entry = {
                            "topic": _kr_topic[:100],
                            "event": "entered_language",
                            "consumer": "language",
                            "quality": _kr_conf,
                            "ts": time.time(),
                        }
                        with open("./data/concept_lifecycle.jsonl", "a") as _cl_f:
                            _cl_f.write(_cl_json.dumps(_cl_entry) + "\n")
                    except Exception:
                        pass
            except Exception as _kr_err:
                logger.debug("[KNOWLEDGE→LANG] Error: %s", _kr_err)

        # ── QUERY_RESPONSE — CGN cross-insights from dream consolidation ──
        # API_STUB: handler ready, awaits cross-insights flow to be wired in
        # CGN-EXTRACT (next session). Tracked I-003.
        # After CGN_WEIGHTS_MAJOR triggers a QUERY for cross-insights,
        # cgn_worker replies here with high-value concepts from other consumers
        # (reasoning, social, coding) that language hasn't grounded yet.
        elif msg_type == "QUERY_RESPONSE":
            try:
                _qr_data = payload.get("result", payload)
                _qr_action = payload.get("action", "")
                if _qr_action == "get_cross_insights" and persistent_teacher:
                    from titan_plugin.logic.language_teacher import ARC_VOCABULARY_MAP
                    _dream_priority = []
                    _vocab_words = {v.get("word", "").lower()
                                    for v in (cached_vocab or [])}
                    insights = _qr_data if isinstance(_qr_data, list) else \
                        _qr_data.get("insights", [])
                    for _di in insights:
                        _di_source = _di.get("source_consumer", "")
                        for _di_concept in _di.get("top_concepts", [])[:3]:
                            _stripped = _di_concept.replace(
                                "arc_", "").replace("pattern_", "")
                            _mapped = ARC_VOCABULARY_MAP.get(
                                _stripped, [_stripped])
                            for _dw in _mapped:
                                if _dw.lower() not in _vocab_words:
                                    _dream_priority.append({
                                        "word": _dw,
                                        "source": _di_source,
                                        "source_reward": _di.get(
                                            "avg_reward", 0),
                                    })
                                    break
                    if _dream_priority:
                        persistent_teacher.set_cross_priority(_dream_priority)
                        logger.info(
                            "[CGN] Dream cross-insights: %d words queued "
                            "(sources: %s)", len(_dream_priority),
                            {p["source"] for p in _dream_priority})
            except Exception as _qr_err:
                logger.debug("[QUERY_RESPONSE] Error: %s", _qr_err)

        # ── SOCIAL_PERCEPTION (Phase 1: CGN social grounding) ────────
        # Events Teacher → perturbation gate → API → bus → here.
        # Ground vocabulary words from felt_summary with SOCIAL_LINKED
        # associations. This creates emergent social-semantic connections.
        elif msg_type == "SOCIAL_PERCEPTION":
            if cgn:
                try:
                    import re as _sp_re
                    _sp_p = msg.get("payload", {})
                    _sp_summary = _sp_p.get("felt_summary", "")
                    _sp_concepts = _sp_p.get("concept_signals", [])
                    _sp_author = _sp_p.get("author", "")
                    _sp_contagion = _sp_p.get("contagion_type", "")
                    _sp_sentiment = float(_sp_p.get("sentiment", 0.0))
                    _sp_relevance = float(_sp_p.get("relevance", 0.0))

                    # Extract words from felt_summary that exist in vocabulary
                    _sp_words_raw = _sp_re.findall(r'[a-zA-Z]+',
                                                    _sp_summary.lower())
                    _sp_vocab_words = set()
                    _sp_vocab_set = {w["word"] for w in (cached_vocab or [])}
                    for _spw in _sp_words_raw:
                        if _spw in _sp_vocab_set and len(_spw) > 2:
                            _sp_vocab_words.add(_spw)

                    # Build sensory context for CGN
                    _sp_ctx = {
                        "epoch": 0,
                        "neuromods": {},
                        "concept_confidences": {},
                        "encounter_type": "social",
                    }

                    # Ground each matching word with social encounter
                    _sp_grounded = 0
                    for _spw in list(_sp_vocab_words)[:8]:
                        _sp_concept = load_concept_from_db(db_path, _spw)
                        if _sp_concept:
                            _sp_action = cgn.ground(_sp_concept, _sp_ctx)
                            apply_grounding_action_to_db(
                                db_path, _spw, _sp_action)
                            if _sp_action.transition:
                                _send_msg(send_queue, "CGN_TRANSITION",
                                          name, "cgn", _sp_action.transition)
                            # Social reward: scaled by relevance
                            _sp_reward = 0.02 + _sp_relevance * 0.03
                            _cgn_forward_outcome("language", _spw,
                                                 _sp_reward, {
                                                   "type": "social",
                                                   "author": _sp_author,
                                                   "contagion": _sp_contagion,
                                               })
                            _sp_grounded += 1

                    # Add SOCIAL_LINKED associations between co-occurring
                    # vocabulary words in the social context
                    if len(_sp_vocab_words) >= 2 and db_path:
                        import sqlite3 as _sp_sql
                        import json as _sp_json
                        try:
                            _sp_db = _sp_sql.connect(db_path, timeout=5.0)
                            _sp_db.execute("PRAGMA journal_mode=WAL")
                            _sp_list = list(_sp_vocab_words)[:6]
                            for _spi, _spw1 in enumerate(_sp_list):
                                for _spw2 in _sp_list[_spi+1:]:
                                    for _src, _tgt in [(_spw1, _spw2),
                                                       (_spw2, _spw1)]:
                                        _row = _sp_db.execute(
                                            "SELECT meaning_contexts FROM "
                                            "vocabulary WHERE word=?",
                                            (_src,)).fetchone()
                                        if _row and _row[0]:
                                            try:
                                                _mc = _sp_json.loads(_row[0])
                                            except Exception:
                                                _mc = []
                                            # Check if association exists
                                            _found = False
                                            for _m in _mc:
                                                if not isinstance(_m, dict):
                                                    continue
                                                for _a in _m.get(
                                                        "associations", []):
                                                    if (isinstance(_a,
                                                            (list, tuple))
                                                            and _a[0] == _tgt
                                                            and _a[1] ==
                                                            "SOCIAL_LINKED"):
                                                        _found = True
                                                        break
                                            if not _found and _mc:
                                                if "associations" not in \
                                                        _mc[-1]:
                                                    _mc[-1][
                                                        "associations"] = []
                                                _mc[-1][
                                                    "associations"].append(
                                                    [_tgt, "SOCIAL_LINKED"])
                                                _sp_db.execute(
                                                    "UPDATE vocabulary SET "
                                                    "meaning_contexts=? "
                                                    "WHERE word=?",
                                                    (_sp_json.dumps(
                                                        _mc[-10:]), _src))
                            _sp_db.commit()
                            _sp_db.close()
                        except Exception:
                            pass

                    if _sp_grounded > 0:
                        logger.info("[CGN:Social] Grounded %d words from "
                                    "'%s' (%s by %s)",
                                    _sp_grounded, _sp_p.get("topic", "?"),
                                    _sp_contagion, _sp_author)

                    # P4: Sapir-Whorf — queue social vocabulary candidates for teacher
                    _sp_vocab_cands = _sp_p.get("social_vocab_candidates", [])
                    if _sp_vocab_cands and isinstance(_sp_vocab_cands, list):
                        for _sv_word in _sp_vocab_cands[:5]:
                            if isinstance(_sv_word, str) and len(_sv_word) > 3:
                                teacher_queue.append({
                                    "word": _sv_word,
                                    "source": "social_feed",
                                    "author": _sp_author,
                                    "context": _sp_p.get("felt_summary", ""),
                                })
                        if _sp_vocab_cands:
                            logger.info("[LanguageWorker] Sapir-Whorf: queued %d "
                                        "social vocab candidates: %s",
                                        len(_sp_vocab_cands), _sp_vocab_cands[:3])

                    # Wire concept signals to MSL via bus (I, YOU, WE, etc.)
                    if _sp_concepts:
                        _send_msg(send_queue, "QUERY", name, "spirit", {
                            "action": "signal_concept",
                            "payload": {
                                "concepts": _sp_concepts,
                                "source": "social_perception",
                                "quality": 0.2,
                            },
                        })

                    # P4: Handle engagement reciprocity → delayed CGN reward
                    _eng_reward = _sp_p.get("cgn_engagement_reward")
                    if _eng_reward and _eng_reward.get("reward", 0) > 0:
                        _eng_user = _eng_reward.get("target_user", "")
                        _eng_r = float(_eng_reward["reward"])
                        # Scale by config weight
                        _eng_weight = 0.4  # default
                        _eng_max = 0.15
                        try:
                            _eng_weight = lang_config.get(
                                "cgn_social_policy", {}).get(
                                "engagement_reward_weight", 0.4)
                            _eng_max = lang_config.get(
                                "cgn_social_policy", {}).get(
                                "engagement_reward_max", 0.15)
                        except Exception:
                            pass
                        _eng_scaled = min(_eng_max, _eng_r * _eng_weight)
                        if _eng_scaled > 0.001:
                            _cgn_forward_outcome(
                                "social",
                                _eng_user or "engagement",
                                _eng_scaled, {
                                    "type": "engagement_reciprocity",
                                    "post_type": _eng_reward.get("post_type", ""),
                                })
                            logger.info("[CGN:Social] Engagement reciprocity reward: "
                                        "%.4f for user=%s post_type=%s",
                                        _eng_scaled, _eng_user,
                                        _eng_reward.get("post_type", ""))

                except Exception as _sp_err:
                    logger.warning("[CGN:Social] SOCIAL_PERCEPTION error: %s",
                                   _sp_err)

        # ── CGN_SOCIAL_TRANSITION: Social IQL learning ────────────────
        # Record social interaction as CGN transition for the social consumer.
        # Reward = neuromod delta (felt state improvement from interaction).
        # Dream consolidation trains V(s) + Q(s,a) → emergent social wisdom.
        elif msg_type == "CGN_SOCIAL_TRANSITION":
            if cgn_social:
                try:
                    import numpy as _soc_np
                    _soc_p = msg.get("payload", {})
                    _soc_user = _soc_p.get("user_id", "unknown")
                    _soc_reward = float(_soc_p.get("reward", 0.0))
                    _soc_nm_before = _soc_p.get("neuromod_before", {})

                    # Build concept dict for CGNConsumerClient
                    _soc_concept = {
                        "concept_id": _soc_user,
                        "embedding": _soc_np.zeros(130, dtype=_soc_np.float32),
                        "confidence": float(_soc_p.get("quality", 0.5)),
                        "encounter_count": int(_soc_p.get("interaction_count", 1)),
                        "production_count": 0,
                        "context_history": [],
                        "associations": {},
                        "age_epochs": 0,
                        "cross_modal_conf": float(_soc_p.get("familiarity", 0)),
                        "meaning_contexts": [],
                    }

                    _soc_ctx = {
                        "neuromods": _soc_nm_before,
                        "encounter_type": _soc_p.get("encounter_type", "chat"),
                    }

                    # Ground: select social action via policy
                    _soc_action = cgn_social.ground(_soc_concept, _soc_ctx)

                    # Send transition + outcome to CGN Worker
                    if _soc_action.transition:
                        _send_msg(send_queue, "CGN_TRANSITION",
                                  name, "cgn", _soc_action.transition)
                    _cgn_forward_outcome("social", _soc_user,
                                         _soc_reward, {
                                           "type": _soc_p.get("encounter_type", "chat"),
                                           "action_selected": _soc_action.action_name,
                                           "quality": _soc_p.get("quality", 0),
                                         })

                    logger.info("[CGN:Social] %s → %s (reward=%+.4f, fam=%.2f)",
                                _soc_user[:20], _soc_action.action_name,
                                _soc_reward, _soc_p.get("familiarity", 0))

                except Exception as _soc_err:
                    logger.debug("[CGN:Social] Transition error: %s", _soc_err)

        # ── CGN_HAOV_VERIFY_REQ — HAOV verification request from CGN Worker ──
        # CGN Worker detected impasse or has hypothesis to test.
        # Language worker runs domain-specific verification using LIVE state.
        elif msg_type == "CGN_HAOV_VERIFY_REQ":
            _haov_p = msg.get("payload", {})
            _haov_consumer = _haov_p.get("consumer", "")
            try:
                if _haov_consumer == "language":
                    # Language verifier: query live vocabulary confidence
                    _obs_b = _haov_p.get("obs_before", {})
                    _hyp_word = _haov_p.get("test_ctx", {}).get("word", "")
                    _conf_b = _obs_b.get("confidence", 0)

                    # Query live word confidence from vocabulary cache
                    _conf_a = _conf_b
                    _prod_ok = False
                    _qual = 0.0
                    try:
                        import sqlite3 as _hv_sql
                        _hv_db = _hv_sql.connect(db_path, timeout=2.0)
                        _hv_db.execute("PRAGMA journal_mode=WAL")
                        _hv_db.row_factory = _hv_sql.Row
                        if _hyp_word:
                            _hv_row = _hv_db.execute(
                                "SELECT confidence, stage FROM vocabulary "
                                "WHERE word = ?", (_hyp_word,)
                            ).fetchone()
                            if _hv_row:
                                _conf_a = _hv_row["confidence"]
                                _prod_ok = _hv_row["stage"] in ("active", "mastered")
                        # Overall vocab quality = avg confidence of recent words
                        _hv_avg = _hv_db.execute(
                            "SELECT AVG(confidence) FROM vocabulary "
                            "WHERE confidence > 0"
                        ).fetchone()[0] or 0
                        _qual = _hv_avg
                        _hv_db.close()
                    except Exception:
                        pass

                    _confirmed = (_conf_a > _conf_b + 0.01) or _prod_ok or _qual > 0.5
                    _error = abs(_conf_a - _conf_b)
                    _send_msg(send_queue, "CGN_HAOV_VERIFY_RSP", name, "cgn", {
                        "consumer": "language",
                        "test_ctx": _haov_p.get("test_ctx"),
                        "obs_after": {"confidence": _conf_a, "production_success": _prod_ok, "quality": _qual},
                        "reward": _qual if _confirmed else 0.0,
                        "confirmed": _confirmed,
                        "error": _error,
                    })
                    logger.info("[HAOV] Language verify: word=%s conf %.3f→%.3f confirmed=%s",
                                _hyp_word, _conf_b, _conf_a, _confirmed)
            except Exception as _haov_err:
                logger.debug("[HAOV] Verification error: %s", _haov_err)

        # ── EPOCH_TICK (Phase 2+: teacher scheduling) ────────────────
        elif msg_type == "EPOCH_TICK":
            # Phase 1: refresh vocab cache periodically
            now = time.time()
            if now - last_stats_broadcast >= stats_broadcast_interval:
                try:
                    cached_vocab = load_vocabulary(db_path)
                    language_stats = update_language_stats(db_path, cached_vocab)
                    _send_msg(send_queue, "LANGUAGE_STATS_UPDATE", name, "all",
                              language_stats)
                except Exception:
                    pass
                last_stats_broadcast = now

    # ── Cleanup ──────────────────────────────────────────────────────
    logger.info("[LanguageWorker] Exiting")
    _hb_stop.set()


# ── Teacher Trigger ──────────────────────────────────────────────────

def _check_teacher_trigger(
    send_queue, name: str, teacher, teacher_queue: list,
    cached_vocab: list, lang_config: dict,
    recent_questions: list, neuromod_state: dict | None,
    cgn_instance=None,
) -> None:
    """Fire a teacher request when enough compositions accumulated."""
    from titan_plugin.logic.language_pipeline import build_teacher_request

    # ── CGN Cross-Insight Priority Queue ────────────────────────────
    # Query other consumers (reasoning, social) for high-value concepts
    # that language hasn't grounded yet → priority teaching candidates.
    # CGN cross-insights now flow via P4 Concept Resonance Cascade
    # (CGN_KNOWLEDGE_RESP → language teaching queue) and are handled
    # in the CGN_KNOWLEDGE_RESP bus handler. No synchronous query needed.
    # cgn_instance is a CGNConsumerClient (no get_cross_insights method).

    req = build_teacher_request(
        teacher, teacher_queue, cached_vocab,
        neuromod_state or {},
        concept_confidences=None,
        recent_questions=recent_questions,
    )
    if req:
        _send_msg(send_queue, "LLM_TEACHER_REQUEST", name, "llm", req)
        logger.info("[LanguageWorker] Teacher request sent: mode=%s", req.get("mode", "?"))


# ── LLM_TEACHER_RESPONSE Handler ────────────────────────────────────

def _handle_teacher_response(
    payload: dict, send_queue, name: str,
    teacher, grammar_validator, pattern_library,
    lang_config: dict, db_path: str, cached_vocab: list,
    teacher_queue: list, conversation_pending, conversation_stats: dict,
    recent_questions: list,
) -> None:
    """Handle teacher response: comprehension bridge, word acquisition, MSL signals.

    Moved from spirit_worker.py LLM_TEACHER_RESPONSE handler (~416 lines).
    Sends TEACHER_SIGNALS back to spirit_worker with perturbation deltas +
    MSL signals + dynamic recipes + conversation data.
    """
    import json as _json
    import numpy as np
    from titan_plugin.logic.language_pipeline import (
        classify_word_type, persist_teacher_session, _STRIP_CHARS,
    )

    response = payload.get("response", "")
    mode = payload.get("mode", "unknown")
    original = payload.get("original", "")

    # Outputs for TEACHER_SIGNALS
    perturbation_deltas = []
    msl_signals = []
    dynamic_recipes = []
    neuromod_nudge = {}
    words_acquired = 0
    words_recognized = 0
    conversation_question = None
    conversation_eval_data = None

    # ── CONVERSATION MODE: question for Titan ────────────────────────
    if mode == "conversation" and response:
        conversation_question = response.strip()
        recent_questions.append(conversation_question)
        if len(recent_questions) > 10:
            recent_questions[:] = recent_questions[-10:]
        conversation_stats["asked"] = conversation_stats.get("asked", 0) + 1
        neuromod_nudge["Endorphin"] = 0.15  # Response instinct
        logger.info("[CONVERSATION] Question: '%s'", conversation_question[:80])

    # ── CONVERSATION EVAL: score the response ────────────────────────
    if mode == "conversation_eval" and response:
        try:
            eval_text = response.strip()
            score = 0.0
            note = ""
            try:
                parsed = _json.loads(eval_text)
                score = float(parsed.get("score", 0.0))
                note = parsed.get("note", "")
            except (_json.JSONDecodeError, ValueError):
                import re
                m = re.search(r"(\d+\.?\d*)", eval_text)
                if m:
                    score = min(1.0, float(m.group(1)))

            conversation_stats["total_score"] = conversation_stats.get("total_score", 0) + score
            n_answered = max(1, conversation_stats.get("answered", 1))
            conversation_stats["avg_score"] = round(
                conversation_stats["total_score"] / n_answered, 3)

            # MSL YES/NO from conversation quality
            if score > 0.7:
                msl_signals.append({"concept": "YES", "quality": score})
            elif score < 0.3:
                msl_signals.append({"concept": "NO", "quality": 1.0 - score})

            conversation_eval_data = {
                "score": score, "note": note,
                "question": original,
                "response": payload.get("conversation_response", ""),
            }

            logger.info("[CONVERSATION] Eval: score=%.2f %s [avg=%.2f]",
                        score, note[:30], conversation_stats["avg_score"])
        except Exception as e:
            logger.debug("[CONVERSATION] Eval error: %s", e)

        # Send TEACHER_SIGNALS with eval data (skip comprehension for eval)
        _send_msg(send_queue, "TEACHER_SIGNALS", name, "spirit", {
            "perturbation_deltas": [],
            "msl_signals": msl_signals,
            "dynamic_recipes": [],
            "neuromod_nudge": {},
            "words_acquired": 0,
            "words_recognized": 0,
            "conversation_question": None,
            "conversation_eval": conversation_eval_data,
            "mode": mode,
        })
        return  # Don't process eval through comprehension

    # ── META_FEEDBACK MODE: Word grounding via dimensional prescriptions ──
    if mode == "meta_feedback" and response:
        try:
            parsed = teacher.parse_response(mode, response, original, cached_vocab)
            grounding = parsed.get("grounding", {})
            target_word = parsed.get("target_word", original)

            if grounding.get("dimensions"):
                import sqlite3
                import json as _mf_json
                import numpy as np

                conn = sqlite3.connect(db_path, timeout=5.0)
                conn.execute("PRAGMA journal_mode=WAL")
                row = conn.execute(
                    "SELECT felt_tensor, sensory_context, meaning_contexts, "
                    "cross_modal_conf, confidence FROM vocabulary WHERE word=?",
                    (target_word,)).fetchone()

                if row:
                    # 1. Apply dimensional prescriptions to felt_tensor
                    ft = _mf_json.loads(row[0]) if row[0] else [0.5] * 130
                    from titan_plugin.logic.language_teacher import LanguageTeacher
                    dim_map = LanguageTeacher._DIM_MAP
                    plasticity = 0.3  # Blend 30% prescription / 70% existing
                    dims_applied = 0
                    for dim_name, target_val in grounding["dimensions"].items():
                        idx = dim_map.get(dim_name)
                        if idx is not None and idx < len(ft):
                            ft[idx] = ft[idx] * (1 - plasticity) + target_val * plasticity
                            dims_applied += 1
                            perturbation_deltas.append({
                                "dimension": idx,
                                "delta": (target_val - ft[idx]) * plasticity,
                            })

                    # 2. Store associations
                    existing_contexts = _mf_json.loads(row[1]) if row[1] else []
                    existing_meanings = _mf_json.loads(row[2]) if row[2] else []
                    existing_xm = row[3] or 0.0
                    old_conf = row[4] or 0.0

                    # Add new contexts
                    for ctx in grounding.get("contexts", []):
                        if ctx not in existing_contexts:
                            existing_contexts.append(ctx)
                    existing_contexts = existing_contexts[-10:]  # Keep last 10

                    # Add associations as meaning context entry
                    if grounding.get("associations"):
                        meaning_entry = {
                            "associations": grounding["associations"],
                            "dimensions": grounding["dimensions"],
                            "timestamp": time.time(),
                        }
                        existing_meanings.append(meaning_entry)
                        existing_meanings = existing_meanings[-5:]  # Keep last 5

                    # 3. Update cross_modal confidence
                    #    More dimensions grounded → higher cross-modal confidence
                    xm_boost = min(0.1, dims_applied * 0.02)
                    new_xm = min(1.0, existing_xm + xm_boost)

                    # 4. Confidence boost from grounding (replaces hardcoded +0.02)
                    #    Proportional to grounding quality
                    conf_delta = min(0.05, dims_applied * 0.01)
                    new_conf = min(1.0, old_conf + conf_delta)

                    # 5. Write back
                    conn.execute(
                        "UPDATE vocabulary SET felt_tensor=?, sensory_context=?, "
                        "meaning_contexts=?, cross_modal_conf=?, confidence=? "
                        "WHERE word=?",
                        (_mf_json.dumps(ft), _mf_json.dumps(existing_contexts),
                         _mf_json.dumps(existing_meanings), new_xm,
                         new_conf, target_word))
                    conn.commit()
                    conn.close()

                    logger.info("[META_FEEDBACK] Grounded '%s': %d dims, %d assocs, "
                                "conf %.2f→%.2f, xm %.2f→%.2f",
                                target_word, dims_applied,
                                len(grounding.get("associations", [])),
                                old_conf, new_conf, existing_xm, new_xm)

                    # MSL signal: grounding is a form of self-understanding
                    msl_signals.append({"concept": "I", "quality": 0.2})

                else:
                    logger.debug("[META_FEEDBACK] Word '%s' not in vocabulary", target_word)

        except Exception as e:
            logger.warning("[META_FEEDBACK] Processing error: %s", e)

        # Send signals and return (skip normal comprehension)
        _send_msg(send_queue, "TEACHER_SIGNALS", name, "spirit", {
            "perturbation_deltas": perturbation_deltas,
            "msl_signals": msl_signals,
            "dynamic_recipes": [],
            "neuromod_nudge": {"DA": 0.05},  # Grounding reward
            "words_acquired": 0,
            "words_recognized": 1,
            "conversation_question": None,
            "conversation_eval": None,
            "mode": mode,
        })
        return

    # ── EMBODIED_TEACHING MODE: full state-aware dimensional shifts ──
    if mode == "embodied_teaching" and response:
        try:
            parsed = teacher.parse_response(mode, response, original, cached_vocab)
            prescription = parsed.get("prescription", {})
            target_word = parsed.get("target_word", original)

            if prescription.get("shifts"):
                import sqlite3
                import json as _et_json

                conn = sqlite3.connect(db_path, timeout=5.0)
                conn.execute("PRAGMA journal_mode=WAL")
                row = conn.execute(
                    "SELECT felt_tensor, sensory_context, meaning_contexts, "
                    "cross_modal_conf, confidence FROM vocabulary WHERE word=?",
                    (target_word,)).fetchone()

                if row:
                    ft = _et_json.loads(row[0]) if row[0] else [0.5] * 130
                    old_xm = row[3] or 0.0
                    old_conf = row[4] or 0.0
                    existing_meanings = _et_json.loads(row[2]) if row[2] else []

                    # Apply dimensional shifts to felt_tensor
                    plasticity = 0.25  # Embodied teaching: moderate plasticity
                    dims_applied = 0
                    for group, shift in prescription["shifts"].items():
                        start, end = shift["range"]
                        delta = shift["delta"]
                        for i in range(start, min(end, len(ft))):
                            ft[i] = max(0.0, min(1.0, ft[i] + delta * plasticity))
                            dims_applied += 1
                        perturbation_deltas.append({
                            "group": group,
                            "direction": shift["direction"],
                            "magnitude": shift["magnitude"],
                            "delta": delta,
                        })

                    # Neuromod nudges from prescription
                    for nm_name, nm_delta in prescription.get("neuromods", {}).items():
                        neuromod_nudge[nm_name] = nm_delta * 0.1  # Gentle nudge

                    # Store associations + prescription
                    meaning_entry = {
                        "type": "embodied_teaching",
                        "shifts": {k: {"dir": v["direction"], "mag": v["magnitude"]}
                                   for k, v in prescription["shifts"].items()},
                        "associations": prescription.get("associations", []),
                        "timestamp": time.time(),
                    }
                    existing_meanings.append(meaning_entry)
                    existing_meanings = existing_meanings[-5:]

                    # Boost cross_modal and confidence
                    new_xm = min(1.0, old_xm + 0.03)  # Higher boost than meta_feedback
                    new_conf = min(1.0, old_conf + 0.02)

                    conn.execute(
                        "UPDATE vocabulary SET felt_tensor=?, meaning_contexts=?, "
                        "cross_modal_conf=?, confidence=? WHERE word=?",
                        (_et_json.dumps(ft), _et_json.dumps(existing_meanings),
                         new_xm, new_conf, target_word))
                    conn.commit()
                    conn.close()

                    logger.info("[EMBODIED_TEACHING] Grounded '%s': %d dims shifted, "
                                "%d groups, xm %.2f→%.2f",
                                target_word, dims_applied,
                                len(prescription["shifts"]),
                                old_xm, new_xm)

                    msl_signals.append({"concept": "I", "quality": 0.3})

        except Exception as e:
            logger.warning("[EMBODIED_TEACHING] Processing error: %s", e)

        _send_msg(send_queue, "TEACHER_SIGNALS", name, "spirit", {
            "perturbation_deltas": perturbation_deltas,
            "msl_signals": msl_signals,
            "dynamic_recipes": [],
            "neuromod_nudge": neuromod_nudge,
            "words_acquired": 0,
            "words_recognized": 1,
            "conversation_question": None,
            "conversation_eval": None,
            "mode": mode,
        })
        return

    # ── EMBODIED_REASONING MODE: cross-consumer grounding from ARC/social ──
    # Handled by comprehension bridge below, but first emit CGN cross-insight
    # signal so CGN knows this word has been grounded across multiple consumers.
    if mode == "embodied_reasoning" and response:
        _er_target = payload.get("target_word", original)
        if _er_target:
            try:
                _send_msg(send_queue, "CGN_SURPRISE", name, "cgn", {
                    "consumer": "language",
                    "concept_id": _er_target,
                    "magnitude": 0.3,
                    "context": {
                        "source": "embodied_reasoning_grounding",
                        "cross_grounded": True,
                        "original_source": payload.get("source", "discovery"),
                    },
                })
                msl_signals.append({"concept": "I", "quality": 0.4})
                logger.info("[EMBODIED_REASONING] Cross-grounding '%s' — "
                            "CGN surprise sent", _er_target)
            except Exception as _er_err:
                logger.debug("[EMBODIED_REASONING] Signal error: %s", _er_err)
        # Fall through to comprehension bridge for standard word processing

    # ── COMPREHENSION BRIDGE ──────��─────────────────────────���────────
    if response:
        try:
            parsed = teacher.parse_response(mode, response, original, cached_vocab)

            # 1. Word extraction + perturbation computation
            if parsed.get("text"):
                all_words = []
                unknown_words = []
                known_tensors = []

                for raw_w in parsed["text"].lower().split():
                    w = raw_w.strip(_STRIP_CHARS)
                    w = w.replace("**", "").replace("__", "").replace("*", "").replace("`", "").replace("#", "")
                    w = w.replace("\u2026", "").strip("_")
                    if "\u2014" in w or "\u2013" in w or "\u201c" in w or "\u201d" in w:
                        continue
                    if not w or len(w) < 2:
                        continue
                    all_words.append(w)

                    # Check if known word (exists in vocab)
                    known = False
                    known_ft = None
                    for vw in cached_vocab:
                        if vw.get("word") == w:
                            known = True
                            known_ft = vw.get("felt_tensor")
                            break

                    if known:
                        words_recognized += 1
                        if known_ft:
                            known_tensors.append(known_ft)
                        # Compute perturbation for this known word
                        try:
                            from titan_plugin.logic.action_narrator import ActionNarrator
                            _narrator = ActionNarrator()
                            _perturb = _narrator.get_word_perturbation(w)
                            if _perturb:
                                perturbation_deltas.append({
                                    "word": w,
                                    "inner_body": _perturb.get("inner_body", []),
                                    "inner_mind": _perturb.get("inner_mind", []),
                                })
                        except Exception:
                            pass
                    else:
                        # Unknown — candidate for acquisition
                        if mode == "first_words":
                            unknown_words.append(w)
                        elif w not in {"i", "a", "an", "the", "is", "am", "are",
                                      "was", "were", "be", "to", "of", "in",
                                      "it", "that", "this", "for", "on", "at",
                                      "by", "do", "not", "no", "or", "if",
                                      "my", "me", "we", "he", "she", "they",
                                      "his", "her", "its", "our", "you", "your",
                                      "has", "have", "had", "can", "will", "would",
                                      "could", "should", "may", "with", "from",
                                      "as", "but", "so", "when", "than"}:
                            unknown_words.append(w)

                # 2. Contextual acquisition
                if unknown_words:
                    import sqlite3

                    # Build contextual tensor from known word tensors
                    if not known_tensors:
                        # Try DB lookup for known words
                        try:
                            conn = sqlite3.connect(db_path, timeout=5.0)
                            conn.execute("PRAGMA journal_mode=WAL")
                            for kw in all_words:
                                if kw in unknown_words:
                                    continue
                                row = conn.execute(
                                    "SELECT felt_tensor FROM vocabulary WHERE word=? AND felt_tensor IS NOT NULL",
                                    (kw,)).fetchone()
                                if row and row[0]:
                                    ft = _json.loads(row[0]) if isinstance(row[0], str) else row[0]
                                    if ft and len(ft) >= 65:
                                        known_tensors.append(ft)
                            conn.close()
                        except Exception:
                            pass

                    # Build contextual tensor
                    ctx_tensors = [np.array(t) for t in known_tensors if len(t) >= 65]
                    if ctx_tensors:
                        ctx_avg = sum(ctx_tensors) / len(ctx_tensors)
                        # Blend 70% context / 30% average (we don't have live state here)
                        contextual_tensor = ctx_avg.tolist()
                    else:
                        # Bootstrap: pure zero-centered tensor
                        contextual_tensor = [0.5] * 130
                        logger.info("[TEACHER:BOOTSTRAP] Pure state tensors for %d words",
                                    len(unknown_words))

                    # INSERT new words into vocabulary
                    acq_limit = 8 if mode == "first_words" else lang_config.get("max_acquisition_per_session", 3)
                    conn = sqlite3.connect(db_path, timeout=5.0)
                    conn.execute("PRAGMA journal_mode=WAL")
                    for acq_w in unknown_words[:acq_limit]:
                        exists = conn.execute(
                            "SELECT 1 FROM vocabulary WHERE word=?", (acq_w,)).fetchone()
                        if not exists:
                            acq_type = classify_word_type(acq_w)
                            word_tensor = list(contextual_tensor)
                            for wti in range(min(20, len(word_tensor))):
                                noise = (hash((acq_w, wti)) % 100 - 50) / 500.0
                                word_tensor[wti] = max(0.0, min(1.0, word_tensor[wti] + noise))
                            acq_phase = "first_word" if mode == "first_words" else "contextual"
                            conn.execute(
                                "INSERT INTO vocabulary "
                                "(word, word_type, stage, felt_tensor, confidence, "
                                "times_encountered, times_produced, learning_phase, created_at) "
                                "VALUES (?, ?, 1, ?, 0.15, 1, 0, ?, ?)",
                                (acq_w, acq_type, _json.dumps(word_tensor), acq_phase, time.time()))
                            words_acquired += 1
                            dynamic_recipes.append({
                                "word": acq_w,
                                "tensor": word_tensor,
                                "word_type": acq_type,
                                "context": response[:100],
                            })
                            logger.info("[ACQUISITION] '%s' (%s) from %s (conf=0.15)",
                                        acq_w, acq_type,
                                        "bootstrap" if mode == "first_words" else "context")
                            # TimeChain: word learned → declarative fork
                            send_queue.put({"type": "TIMECHAIN_COMMIT", "src": name,
                                "dst": "timechain", "ts": time.time(), "payload": {
                                "fork": "declarative", "thought_type": "declarative",
                                "source": "language_teacher",
                                "content": {"word": acq_w, "word_type": acq_type,
                                    "learning_phase": acq_phase, "confidence": 0.15},
                                "significance": 0.4, "novelty": 0.8, "coherence": 0.5,
                                "tags": [acq_w, "word_acquired", acq_type],
                                "db_ref": f"vocabulary:{acq_w}",
                                "neuromods": dict(_cached_neuromods) if '_cached_neuromods' in dir() else {},
                                "chi_available": 0.5, "attention": 0.5,
                                "i_confidence": 0.5, "chi_coherence": 0.3,
                            }})
                    conn.commit()
                    conn.close()

                if words_recognized > 0 or words_acquired > 0:
                    logger.info("[TEACHER] %s: %d words felt, %d acquired",
                                mode, words_recognized, words_acquired)

            # 3. Mode-specific learning
            if mode == "grammar" and parsed.get("correction") and grammar_validator:
                rule = grammar_validator.learn_from_correction(
                    original, parsed["correction"], source="language_teacher")
                if rule:
                    logger.info("[TEACHER] Grammar rule: '%s' -> '%s'",
                                rule.pattern, rule.replacement)

            if parsed.get("text") and pattern_library:
                l8 = pattern_library.add_pattern(
                    parsed["text"], cached_vocab, source="teacher")
                if l8:
                    logger.info("[L8] Pattern from teacher: '%s'", l8)

            # 4. MSL concept signals from teaching
            try:
                signals = teacher.compute_teaching_signals(mode, response, score=None)
                for sig in signals:
                    msl_signals.append({"concept": sig["concept"], "quality": sig["quality"]})
                if signals:
                    logger.debug("[TEACHER] MSL signals: %s", [s["concept"] for s in signals])
            except Exception:
                pass

            # 4b. Knowledge requests for newly acquired words
            #      Send CGN_KNOWLEDGE_REQ for words we just learned —
            #      the Knowledge Worker can research deeper context/meaning.
            #      2026-04-12 fix: short single-word queries (e.g., "own", "noun")
            #      return 0 results from SearXNG. Enrich with "meaning definition"
            #      suffix for short words so SearXNG returns dictionary-style
            #      results. Preserve original word in `original_word` metadata
            #      so downstream can attribute correctly.
            if words_acquired > 0 and dynamic_recipes:
                for _kr_recipe in dynamic_recipes[:3]:  # Max 3 requests per session
                    _kr_word = _kr_recipe.get("word", "")
                    if _kr_word and len(_kr_word) > 2:
                        # Enrich short words with contextual suffix for SearXNG
                        _kr_query = (
                            f"{_kr_word} meaning definition"
                            if len(_kr_word) <= 6 else _kr_word
                        )
                        try:
                            _send_msg(send_queue, "CGN_KNOWLEDGE_REQ", name,
                                      "knowledge", {
                                "topic": _kr_query,
                                "original_word": _kr_word,  # preserve for attribution
                                "requestor": "language",
                                "urgency": 0.3,  # Low urgency — enrichment, not blocking
                                "neuromods": payload.get("neuromod_state", {}),
                                "context": f"newly acquired word from teacher ({mode})",
                            })
                        except Exception:
                            pass

            # 5. Persist teacher session
            persist_teacher_session(
                db_path, mode, original, response[:500],
                words_recognized,
                correction=parsed.get("correction"),
                pattern_hash=(parsed.get("pattern", {}).get("hash")
                              if parsed.get("pattern") else None),
                neuromod_gate=payload.get("neuromod_gate", ""),
            )

            # 5b. Reasoning → Language reverse feedback
            # When reasoning-sourced teaching succeeds, send confidence
            # feedback back to CGN reasoning consumer via CGN_TRANSITION
            if mode in ("reasoning", "embodied_reasoning") and (
                    words_acquired > 0 or words_recognized > 1):
                _rf_reward = 0.1 * words_acquired + 0.02 * words_recognized
                try:
                    _send_msg(send_queue, "CGN_TRANSITION", name, "cgn", {
                        "type": "outcome",
                        "consumer": "reasoning",
                        "concept_id": f"lang_grounding_{original[:30]}",
                        "reward": min(0.5, _rf_reward),
                        "outcome_context": {
                            "source": "language_reverse_feedback",
                            "mode": mode,
                            "words_acquired": words_acquired,
                            "words_recognized": words_recognized,
                            "original_word": original[:50],
                        },
                    })
                    logger.debug("[REVERSE_FEEDBACK] %s: %.2f reward → "
                                 "reasoning (acq=%d rec=%d)",
                                 mode, _rf_reward,
                                 words_acquired, words_recognized)
                except Exception:
                    pass

            logger.info("[TEACHER] Session complete: mode=%s", mode)

        except Exception as e:
            logger.warning("[TEACHER] Response processing error: %s", e)

    # ── Send TEACHER_SIGNALS to spirit_worker ────────────────────────
    _send_msg(send_queue, "TEACHER_SIGNALS", name, "spirit", {
        "perturbation_deltas": perturbation_deltas,
        "msl_signals": msl_signals,
        "dynamic_recipes": dynamic_recipes,
        "neuromod_nudge": neuromod_nudge,
        "words_acquired": words_acquired,
        "words_recognized": words_recognized,
        "conversation_question": conversation_question,
        "conversation_eval": conversation_eval_data,
        "mode": mode,
    })


# ── SPEAK_REQUEST Handler ────────────────────────────────────────────

def _handle_speak_request(
    payload: dict, send_queue, name: str, dst: str,
    composition_engine, grammar_validator, pattern_library,
    lang_config: dict, db_path: str, cached_vocab: list,
    teacher_queue: list | None = None,
) -> None:
    """Handle SPEAK_REQUEST: compose sentence from felt-state.

    Moved from spirit_worker.py SPEAK handler (lines 3965-4316).

    Performs: vocab load, DA-gated exploration, compose, grammar validate,
    compute perturbation deltas, persist composition, update vocab after speak.

    Sends SPEAK_RESULT back to spirit_worker.
    """
    from titan_plugin.logic.language_pipeline import (
        load_vocabulary, compose_sentence, compute_perturbation_deltas,
        update_vocabulary_after_speak, persist_composition,
    )

    state_132d = payload.get("state_132d", [])
    neuromods = payload.get("neuromods", {})
    concept_confidences = payload.get("concept_confidences")
    visual_context = payload.get("visual_context")
    epoch_id = payload.get("epoch_id", 0)

    if len(state_132d) < 65:
        logger.warning("[LanguageWorker] SPEAK_REQUEST: state_vector too short (%d)", len(state_132d))
        _send_msg(send_queue, "SPEAK_RESULT", name, dst, {"sentence": ""})
        return

    # 1. Load vocabulary from DB (fresh each time)
    vocab = load_vocabulary(db_path)
    if not vocab:
        logger.info("[LanguageWorker] SPEAK: no vocabulary — returning empty")
        _send_msg(send_queue, "SPEAK_RESULT", name, dst, {
            "sentence": "", "bootstrap_needed": True})
        return

    # 2. Experience bias (passed as serializable dict or None)
    experience_bias = None
    eb_data = payload.get("experience_bias")
    if eb_data:
        # Reconstruct minimal ExperienceBias-like object
        try:
            from titan_plugin.logic.experience_orchestrator import ExperienceBias
            experience_bias = ExperienceBias(
                optimal_inner_state=eb_data.get("optimal_inner_state"),
                confidence=eb_data.get("confidence", 0),
                domain=eb_data.get("domain", "language"),
            )
        except Exception:
            pass  # ExperienceBias not available, skip

    # 3. DA-gated exploration + compose + grammar validation
    da_info = neuromods.get("DA", {})
    da_level = da_info.get("level", 0.5)
    da_setpoint = da_info.get("setpoint", 0.5)
    max_level = lang_config.get("composition_max_level", 9)

    # Phase 4: L9 reasoning-powered composition when reasoning COMMIT is available
    reasoning_result = payload.get("reasoning_result")
    result = None
    if (reasoning_result and max_level >= 9
            and hasattr(composition_engine, 'compose_l9')):
        try:
            result = composition_engine.compose_l9(
                felt_state=list(state_132d),
                vocabulary=vocab,
                reasoning_plan=reasoning_result,
                concept_confidences=concept_confidences,
            )
            if result and result.get("sentence") and result.get("level", 0) == 9:
                logger.info("[L9] Reasoning-powered composition: '%s'",
                            result["sentence"][:60])
        except Exception as _l9_err:
            logger.debug("[L9] compose_l9 error: %s", _l9_err)
            result = None

    # Standard L8 composition (or L9 fallback)
    if not result or not result.get("sentence"):
        result = compose_sentence(
            composition_engine, state_132d, vocab,
            da_level=da_level, da_setpoint=da_setpoint,
            grammar_validator=grammar_validator,
            experience_bias=experience_bias,
            visual_context=visual_context,
            concept_confidences=concept_confidences,
            max_level=max_level,
        )

    if not result.get("sentence"):
        logger.debug("[LanguageWorker] SPEAK: compose returned empty")
        _send_msg(send_queue, "SPEAK_RESULT", name, dst, {"sentence": ""})
        return

    sentence = result["sentence"]
    level = result.get("level", 0)
    confidence = result.get("confidence", 0.0)

    logger.info('[LanguageWorker] SPEAK: "%s" (L%d, conf=%.2f)',
                sentence, level, confidence)

    # 4. Compute self-hearing perturbation deltas (NOT applied — spirit does that)
    perturbation_deltas = []
    try:
        from titan_plugin.logic.action_narrator import ActionNarrator
        narrator = ActionNarrator()
        perturbation_deltas = compute_perturbation_deltas(narrator, sentence)
    except Exception as e:
        logger.debug("[LanguageWorker] Perturbation delta error: %s", e)

    # 5. Persist composition to DB
    persist_composition(
        db_path, sentence, level,
        result.get("template", f"L{level}"),
        result.get("words_used", []),
        confidence,
        result.get("slots_filled", 0),
        result.get("slots_total", 0),
        epoch_id,
        result.get("resonance", 0.0),
    )

    # 6. Update vocabulary after speak (advance to producible + confidence boost)
    vocab_updated, words_reinforced = update_vocabulary_after_speak(
        db_path, None, sentence)  # narrator=None here, spirit has the real narrator

    # 7. Word association tracking
    _persist_word_associations(db_path, result.get("words_used", []))

    # 8. Creative journal entry
    _persist_creative_journal(db_path, result, epoch_id)

    # 9. Accumulate to teacher queue (Phase 3: language_worker owns teacher)
    if teacher_queue is not None and len(teacher_queue) < 10:
        teacher_queue.append({
            "sentence": sentence,
            "confidence": confidence,
            "level": level,
            "words_used": result.get("words_used", []),
            "template": result.get("template", ""),
            "epoch": epoch_id,
        })

    # 10. Send SPEAK_RESULT back to spirit_worker
    # Include social contagion context if present (for post annotation)
    _social_ctx = payload.get("social_contagion")
    _speak_result = {
        "sentence": sentence,
        "level": level,
        "confidence": confidence,
        "words_used": result.get("words_used", []),
        "template": result.get("template", ""),
        "resonance": result.get("resonance", 0.0),
        "slots_filled": result.get("slots_filled", 0),
        "slots_total": result.get("slots_total", 0),
        "perturbation_deltas": perturbation_deltas,
        "epoch_id": epoch_id,
        "vocab_updated": vocab_updated,
    }
    if _social_ctx:
        _speak_result["social_contagion"] = _social_ctx
    _send_msg(send_queue, "SPEAK_RESULT", name, dst, _speak_result)

    # TimeChain: SPEAK composition → episodic fork (creative expression)
    if sentence and level >= 3:  # Only L3+ compositions (non-trivial)
        send_queue.put({"type": "TIMECHAIN_COMMIT", "src": name,
            "dst": "timechain", "ts": time.time(), "payload": {
            "fork": "episodic", "thought_type": "episodic",
            "source": "expression_speak",
            "content": {"level": level, "words_used": result.get("words_used", [])[:10],
                "template": result.get("template", "")[:50],
                "confidence": round(confidence, 3),
                "resonance": round(result.get("resonance", 0.0), 3)},
            "significance": min(1.0, level / 10.0),
            "novelty": 0.5, "coherence": 0.6,
            "tags": ["speak", f"L{level}"] + result.get("words_used", [])[:3],
            "neuromods": {}, "chi_available": 0.5,
            "attention": 0.5, "i_confidence": 0.5, "chi_coherence": 0.3,
            "epoch_id": epoch_id,
        }})


def _persist_word_associations(db_path: str, words_used: list) -> None:
    """Track word co-occurrence for association learning."""
    import sqlite3
    try:
        words = [w.strip(".,!?\"'").lower() for w in words_used if len(w.strip(".,!?\"'")) > 2]
        if len(words) < 2:
            return
        conn = sqlite3.connect(db_path, timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS word_associations ("
            "word_a TEXT NOT NULL, word_b TEXT NOT NULL, "
            "co_occurrence INTEGER NOT NULL DEFAULT 1, "
            "reward_sum REAL NOT NULL DEFAULT 0.0, "
            "avg_state_delta REAL NOT NULL DEFAULT 0.0, "
            "last_epoch INTEGER NOT NULL DEFAULT 0, "
            "PRIMARY KEY (word_a, word_b))")
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                wa, wb = sorted([words[i], words[j]])
                conn.execute(
                    "INSERT INTO word_associations "
                    "(word_a, word_b, co_occurrence, reward_sum, avg_state_delta, last_epoch) "
                    "VALUES (?, ?, 1, 0, 0, 0) "
                    "ON CONFLICT(word_a, word_b) DO UPDATE SET "
                    "co_occurrence = co_occurrence + 1",
                    (wa, wb))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.debug("[LanguageWorker] Word association error: %s", e)


def _persist_creative_journal(db_path: str, result: dict, epoch_id: int) -> None:
    """Write creative journal entry for SPEAK composition."""
    import sqlite3
    try:
        import json as _json
        conn = sqlite3.connect(db_path, timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS creative_journal ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "timestamp REAL NOT NULL, action_type TEXT NOT NULL, "
            "creation_summary TEXT, score REAL, state_delta REAL, "
            "words_used TEXT, features TEXT, epoch_id INTEGER)")
        conn.execute(
            "INSERT INTO creative_journal "
            "(timestamp, action_type, creation_summary, score, "
            "state_delta, words_used, features, epoch_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (time.time(), "speak",
             'Composed: "%s"' % result.get("sentence", ""),
             result.get("confidence", 0.0),
             result.get("resonance", 0.0),
             _json.dumps(result.get("words_used", [])),
             "{}",
             epoch_id))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.debug("[LanguageWorker] Creative journal error: %s", e)


# ── QUERY Handler ────────────────────────────────────────────────────

def _handle_query(msg, send_queue, name, language_stats, lang_config,
                  cached_vocab, teacher_queue, compositions_since,
                  bootstrap_attempts, conversation_stats):
    """Handle QUERY messages from API/proxies."""
    payload = msg.get("payload", {})
    action = payload.get("action", "")
    rid = msg.get("rid")
    src = msg.get("src", "")

    try:
        if action == "get_language_stats":
            stats = dict(language_stats)
            stats["teacher_queue_depth"] = len(teacher_queue)
            stats["teacher_interval"] = lang_config.get("teacher_interval_compositions", 5)
            stats["bootstrap_attempts"] = bootstrap_attempts
            stats["compositions_since_teach"] = compositions_since
            stats["conversation_stats"] = conversation_stats
            stats["titan_id"] = lang_config.get("titan_id", "T1")
            stats["vocab_cache_size"] = len(cached_vocab)
            _send_response(send_queue, name, src, stats, rid)

        elif action == "get_status":
            _send_response(send_queue, name, src, {
                "ready": True,
                "titan_id": lang_config.get("titan_id", "T1"),
                "vocab_size": len(cached_vocab),
                "engines": {
                    "composition": True,
                    "teacher": True,
                    "grammar": True,
                },
            }, rid)

        else:
            _send_response(send_queue, name, src,
                           {"error": f"unknown action: {action}"}, rid)

    except Exception as e:
        logger.error("[LanguageWorker] Query error: %s", e, exc_info=True)
        _send_response(send_queue, name, src, {"error": str(e)}, rid)


# ── Message Helpers ──────────────────────────────────────────────────

def _send_msg(send_queue, msg_type: str, src: str, dst: str,
              payload: dict, rid: str = None) -> None:
    """Send a message via the send queue (worker->bus)."""
    try:
        send_queue.put_nowait({
            "type": msg_type, "src": src, "dst": dst,
            "ts": time.time(), "rid": rid, "payload": payload,
        })
    except Exception:
        from titan_plugin.bus import record_send_drop
        record_send_drop(src, dst, msg_type)


def _send_response(send_queue, src: str, dst: str,
                   payload: dict, rid: str) -> None:
    """Send a RESPONSE message back."""
    _send_msg(send_queue, "RESPONSE", src, dst, payload, rid)


# Heartbeat throttle (Phase E Fix 2): 3s min interval per process.
_last_hb_ts: float = 0.0


def _send_heartbeat(send_queue, name: str) -> None:
    """Send heartbeat to Guardian with RSS info (throttled to ≤1 per 3s)."""
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
    _send_msg(send_queue, "MODULE_HEARTBEAT", name, "guardian",
              {"rss_mb": round(rss_mb, 1)})
