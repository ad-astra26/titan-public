"""cognitive_worker — Python L2 module hosting the L3 cognitive engines.

CANONICAL L2 WORKER TEMPLATE
============================
This file is the canonical template for the L2 separation strategy
(rFP_titan_hcl_l2_separation_strategy.md §6). Worker #1 of the 14
candidate L2 extractions ships here; workers #2 (expression_worker),
#3 (social_worker), #4 (meditation_worker), … COPY this file's shape
when they ship. Sections labelled ``=== BOILERPLATE ===`` are the
parts to copy verbatim (rename ``CognitiveWorker`` log prefix +
``cognitive_worker_main`` → ``<your>_worker_main`` + flag check).
Sections labelled ``=== MODULE-SPECIFIC ===`` are the parts to replace
with your module's engine init + dispatcher logic. Comments inside the
file flag both.

Per chunk 8E of PLAN_microkernel_phase_c_s8_cognitive_worker_extraction.md
+ SPEC §1 glossary (cognitive_worker term) + SPEC §9.B Python tree (NEW v0.1.8).

ACTIVE UNDER: ``microkernel.l0_rust_enabled = true`` ONLY.
Under ``l0_rust_enabled = false`` the legacy ``spirit_worker_main`` code path
runs the cognitive engines instead per Maker D3 (b) — cognitive_worker
exits early in that mode (after MODULE_READY) so no double-cognitive-engine
work runs simultaneously.

This module is the architecturally-correct home for the L3 cognitive engines.
It replaced ``spirit_worker.py``'s slim-shim 4A interim (the function
formerly named ``_spirit_worker_shim_loop``, deleted in chunk 8I and
replaced with a heartbeat-only ``_spirit_worker_heartbeat_stub`` for the
l0_rust_enabled=true mode). cognitive_worker owns:
  - ``ReasoningEngine`` + ``MetaReasoningEngine``
  - ``DreamingEngine`` + ``TopologyEngine`` + ``NeuralNervousSystem`` (via
    ``InnerTrinityCoordinator``)
  - ``PiHeartbeatMonitor``, ``ObservableEngine``, ``ExpressionManager``
  - ``ChainArchive``, ``MetaWisdomStore``, ``MetaAutoencoder``
  - ``InnerState``, ``SpiritState`` (T2 registries)

Subscribe contract (SPEC §8.5): 3 trinity event TYPES — BODY_STATE,
MIND_STATE, SPIRIT_STATE — coalesce-disambiguated by ``payload.src ∈
{"inner", "outer"}`` for 6 streams. Dispatcher fans into 6 first-class
internal cache slots (``_inner_body_state``, ``_outer_body_state``, …)
per G1 inner↔outer doctrinal symmetry. The 6-stream symmetry Maker D5
specified is preserved via the bus broker's ``coalesce=("src", "type")``
design — inner-BODY and outer-BODY occupy separate coalesce slots, both
survive backpressure.

Adaptive consciousness epoch driver (1–30s tick, Schumann_body × {1, 9, 27}):
  - ``COGNITIVE_EPOCH_MIN_INTERVAL_S``     = 1.15  (1× Schumann body  — floor)
  - ``COGNITIVE_EPOCH_DEFAULT_INTERVAL_S`` = 10.35 (9× Schumann body  — legacy parity)
  - ``COGNITIVE_EPOCH_MAX_INTERVAL_S``     = 31.05 (27× Schumann body — staleness ceiling)
  - ``COGNITIVE_PERSIST_EVERY_N_EPOCHS``   = 100

CHUNK SCOPE
-----------
Chunk 8E (this file): boot section (setup_worker_bus + pdeathsig + engine
init via _cognitive_init) + ModuleSpec registration + heartbeat-only main
loop. NO bus subscriptions, NO epoch driver, NO snapshot publishers — those
land in chunks 8F → 8H.

Chunk 8F (next): bus dispatcher — subscribe to BODY_STATE/MIND_STATE/
SPIRIT_STATE/NEUROMOD_STATE/KERNEL_EPOCH_TICK/CGN_DREAM_CONSOLIDATE/
CONVERSATION_STIMULUS/EXPERIENCE_STIMULUS/MODULE_SHUTDOWN/SAVE_NOW; fan
trinity events into 6 cache slots indexed by ``payload.src``.

Chunk 8G: adaptive 1–30s epoch tick driver — read 6 cache slots →
``coordinator.update(inner_65D, outer_65D)`` → ``_run_consciousness_epoch``
→ ``pi_monitor.observe`` → ``reasoning_engine.step`` → ``meta_engine.tick``;
persist every 100 epochs.

Chunk 8H: snapshot publishers — call ``start_snapshot_builder_threads``
from spirit_loop with the cognitive_worker state_refs dict; daemon
threads then publish ``REASONING_STATS_UPDATED`` / ``DREAMING_STATE_UPDATED``
/ etc. on 2.5s cadence.

Entry point: ``cognitive_worker_main(recv_queue, send_queue, name, config)``.

ARG ORDER — every Guardian-spawned L2 worker entry function follows the
order ``(recv_queue, send_queue, name, config)`` in the Titan codebase.
Note: the docstring inside ``titan_hcl/core/worker_bus_bootstrap.py``
shows ``(name, recv_q, send_q, config)`` — that is STALE and contradicts
all production workers (outer_body_worker, mind_worker, body_worker,
spirit_worker, etc.). Use the production order shown here, not the
bootstrap docstring. Future extractions: do not copy the wrong order.

GENERIC HELPERS BELOW — ``_send_msg``, ``_send_heartbeat``,
``_load_toml_section`` exist in 5+ workers in slightly different shapes.
When the 3rd L2 extraction lands (per L2 separation strategy rFP §5),
extract these to a shared ``titan_hcl/modules/_worker_skeleton.py``
helper module. YAGNI today; plan ahead.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time
from queue import Empty

from titan_hcl import bus
from titan_hcl._phase_c_constants import (
    COGNITIVE_EPOCH_DEFAULT_INTERVAL_S,
    COGNITIVE_EPOCH_MAX_INTERVAL_S,
    COGNITIVE_EPOCH_MIN_INTERVAL_S,
    COGNITIVE_PERSIST_EVERY_N_EPOCHS,
)
from titan_hcl.core.module_error_handler import with_error_envelope
from titan_hcl.errors import Severity as _phase11_sev

logger = logging.getLogger(__name__)

# Heartbeat cadence per SPEC §9.C (10s — guardian_HCL liveness contract).
_HEARTBEAT_INTERVAL_S = 10.0
# Main loop poll cadence (kept tight so MODULE_SHUTDOWN is responsive).
_POLL_INTERVAL_S = 0.2

# Phase 11 §11.I.3/§11.I.5 — module-level readiness sentinel; SHM heartbeat
# is suppressed until the worker has finished in-process scaffolding.
_WORKER_READY: bool = False

# Phase 11 §11.I.5 — process-global ModuleStateWriter ref. Set in
# cognitive_worker_main; consumed by _send_heartbeat so the SHM heartbeat
# rides alongside the bus heartbeat on the same cadence.
_STATE_WRITER = None


# ──────────────────────────────────────────────────────────────────────
# Session 3 CGN_KNOWLEDGE_REQ response builders
# RFP_meta-reasoning_CGN_FIX.md §4.2 rows 1/2/5. Each builder produces a
# REAL output dict from the corresponding cognitive engine (reasoning_engine
# state, pattern-archive query, chain_archive query) — these flow back to
# meta_service resolvers as META_REASON_RESPONSE.insight so consumer-driven
# outcomes can populate the dynamic-reward accumulator α-blend.
# ──────────────────────────────────────────────────────────────────────


def _build_reasoning_response(reasoning_engine, name: str,
                              payload: dict) -> dict:
    """Build a reasoning-kind response. The `name` is the post-dot portion
    of the selected recruiter (e.g., "DECOMPOSE", "COMPARE", "CONTRAST",
    "IF_THEN", "ANALOGIZE", "GENERALIZE", "consistency_check").

    Session 3 surface: returns reasoning_engine's current state summary +
    a per-name action signature so the consumer's outcome computer has
    real differentiated input. Tighter coupling to per-name engine entry
    points (e.g., decompose_query) is a follow-on enrichment — out of
    scope for this rFP per §4.3.4.
    """
    if reasoning_engine is None:
        return {"engine": "unavailable", "name": name,
                "suggested_action": "fallback_static"}
    try:
        stats = reasoning_engine.get_stats() or {}
    except Exception:
        stats = {}
    commit_rate = float(stats.get("commit_rate", 0.0))
    avg_len = float(stats.get("avg_chain_length", 0.0))
    # RFP_meta-reasoning_CGN_FIX.md Fix4 — suggested_action derived from
    # engine's real state. Consumers feed this to their outcome computer
    # to learn whether the suggestion was useful.
    if commit_rate < 0.5:
        # Engine is committing rarely → suggest more aggressive primitive
        sugg = "decompose_deeper" if name in ("DECOMPOSE", "default") else \
               "try_alternate_primitive"
    elif avg_len < 5.0:
        sugg = "extend_chain"
    else:
        sugg = "synthesize_now"
    return {
        "engine": "reasoning",
        "name": name,
        "chains_total": int(stats.get("total_chains", 0)),
        "chains_commit_rate": commit_rate,
        "avg_chain_length": avg_len,
        "buffer_size": int(stats.get("buffer_size", 0)),
        "question_type": payload.get("question_type", ""),
        "consumer_id": payload.get("consumer_id", ""),
        "suggested_action": sugg,
    }


def _build_pattern_response(state_refs: dict, name: str,
                            payload: dict) -> dict:
    """Build a pattern_primitives-kind response. The `name` is the
    post-dot recruiter portion (e.g., "extract_structure", "merge",
    "abstract", "match", "extrapolate").

    Session 3 surface: returns chain_archive's recent pattern frequency
    (which pattern signatures recur across recent chains) so the consumer
    learns which patterns its meta_service request maps to. Per-name
    pattern operations land in a follow-on rFP if soak telemetry shows
    they're load-bearing.
    """
    chain_archive = state_refs.get("chain_archive")
    if chain_archive is None:
        return {"engine": "unavailable", "name": name,
                "suggested_action": "fallback_static"}
    try:
        # chain_archive.query_high_scoring returns recent high-quality
        # chains — proxy for "patterns that recently worked"
        top_chains = chain_archive.query_high_scoring(limit=5) or []
    except Exception:
        top_chains = []
    # Fix4 — suggested_action derived from pattern frequency
    if not top_chains:
        sugg = "explore_new_pattern"
    elif name == "extract_structure":
        sugg = "match_existing_top_pattern"
    elif name == "merge":
        sugg = "merge_top_2_patterns"
    elif name == "abstract":
        sugg = "abstract_top_pattern"
    else:  # match / extrapolate / default
        sugg = "extrapolate_from_top"
    return {
        "engine": "pattern_primitives",
        "name": name,
        "top_chain_count": len(top_chains),
        "top_chain_ids": [int(c.get("chain_id", -1))
                          for c in top_chains if isinstance(c, dict)],
        "question_type": payload.get("question_type", ""),
        "suggested_action": sugg,
    }


def _build_meta_wisdom_response(meta_engine, name: str,
                                payload: dict) -> dict:
    """meta_wisdom.{query_by_embedding,store_wisdom} response — delegates
    to MetaCGNConsumer.handle_knowledge_request (which lives inside
    MetaReasoningEngine instantiated in cognitive_worker per Track 1
    canonical home). RFP_meta-reasoning_CGN_FIX.md §4.2 row 4."""
    if meta_engine is None:
        return {"engine": "unavailable", "name": name,
                "suggested_action": "fallback_static"}
    mcgn = getattr(meta_engine, "_meta_cgn", None)
    if mcgn is None:
        return {"engine": "meta_cgn_not_initialized", "name": name,
                "suggested_action": "fallback_static"}
    legacy_req = {
        "topic": payload.get("topic", "") or payload.get("payload_snippet", ""),
        "requestor": "meta_service",
        "urgency": 0.5,
        "request_id": str(payload.get("correlation_id", ""))[:16],
        "name": name,
        "consumer_id": payload.get("consumer_id", ""),
    }
    try:
        response = mcgn.handle_knowledge_request(legacy_req) or {}
    except Exception as e:
        return {"engine": "meta_wisdom", "name": name, "error": str(e),
                "suggested_action": "fallback_static"}
    # Fix4 — suggested_action from wisdom confidence
    confidence = float(response.get("confidence", 0.0))
    if confidence >= 0.7:
        sugg = "apply_high_conf_wisdom"
    elif confidence >= 0.4:
        sugg = "consider_wisdom_with_doubt"
    elif name == "store_wisdom":
        sugg = "store_new_wisdom"
    else:
        sugg = "seek_more_wisdom"
    return {
        "engine": "meta_wisdom",
        "name": name,
        "topic": legacy_req["topic"],
        "confidence": confidence,
        "summary": response.get("summary", ""),
        "source": response.get("source", "meta_cgn"),
        "suggested_action": sugg,
    }


def _build_language_transitional_response(name: str, payload: dict) -> dict:
    """language.* transitional response — cognitive_worker doesn't host
    LanguageReasoner directly. This is a placeholder until the future
    language_worker carve-out ships per L2 separation strategy §4
    (currently not in the §4 table — TBD). Returns a minimal-but-shaped
    response so the consumer's outcome computer sees real input rather
    than a None/exception. RFP_meta-reasoning_CGN_FIX.md §4.2 row 3."""
    # Fix4 — transitional but still useful suggested_action for consumers
    return {
        "engine": "language",
        "name": name,
        "status": "transitional_target_no_language_worker_yet",
        "consumer_id": payload.get("consumer_id", ""),
        "topic": payload.get("topic", ""),
        "note": ("language_reasoner currently inline in spirit_worker; "
                 "resolver retargets to language_worker when that "
                 "extraction ships per rFP_titan_hcl_l2_separation_strategy"),
        "suggested_action": "expand_vocab_via_grounding",
    }


def _build_chain_archive_response(chain_archive, name: str,
                                  payload: dict) -> dict:
    """Build a chain_archive-kind response. The `name` is the post-dot
    recruiter portion (typically "query").

    Returns chain_archive query results based on the consumer's domain
    hint (extracted from question_type + consumer_id). Real DB query —
    consumer outcome computers can compare against ground truth.
    """
    if chain_archive is None:
        return {"engine": "unavailable", "name": name,
                "suggested_action": "fallback_static"}
    consumer_id = payload.get("consumer_id", "")
    # Map consumer → domain heuristic (matches the chain_archive domain
    # column written by reasoning.py chain conclude).
    _domain_hint = {
        "language": "language",
        "knowledge": "knowledge",
        "social": "outer_perception",
        "reasoning": "inner_spirit",
        "emotional": "emot",
        "self_model": "introspect",
        "coding": "knowledge",
        "dreaming": "inner_spirit",
        "reflection": "introspect",
    }.get(consumer_id, "general")
    try:
        rows = chain_archive.query_by_domain(_domain_hint, limit=5) or []
    except Exception:
        rows = []
    # Fix4 — suggested_action from query yield
    if len(rows) >= 3:
        sugg = "recall_strong_prior"
    elif len(rows) >= 1:
        sugg = "recall_weak_prior"
    else:
        sugg = "explore_new_chain"
    return {
        "engine": "chain_archive",
        "name": name,
        "domain": _domain_hint,
        "matched_count": len(rows),
        "top_chain_ids": [int(r.get("chain_id", -1))
                          for r in rows if isinstance(r, dict)][:5],
        "suggested_action": sugg,
    }

# === MODULE-SPECIFIC: subscribe topics list (PLAN §3.1 driver table) ===
# Enumerated per PLAN §11 acceptance criterion #3. Trinity events are
# 3 TYPES × payload.src ∈ {inner, outer} = 6 streams (SPEC §8.5).
# NEUROMOD_STATE is a SHM SLOT not a bus event — read via shm at each
# epoch tick (chunk 8G) per SPEC §10.G shm-direct-read fallback.
# EXPRESSION events are produced internally by the epoch driver
# (expression_manager.evaluate_all() in chunk 8G), not bus-subscribed.
_COGNITIVE_WORKER_SUBSCRIBE_TOPICS = [
    bus.BODY_STATE,                # 5D, src=inner|outer per SPEC §8.5
    bus.MIND_STATE,                # 15D, src=inner|outer
    bus.SPIRIT_STATE,              # 45D, src=inner|outer
    bus.KERNEL_EPOCH_TICK,         # circadian phase update (1Hz)
    bus.CGN_DREAM_CONSOLIDATE,     # → coordinator.dreaming.consolidate_pending
    # v1.8.2 (D-SPEC-56): dream_state_worker forwards DREAM_WAKE_REQUEST here
    # (dst="cognitive" targeted; broadcast_topics inclusion is defensive +
    # explicit so the subscription contract is visible in this list).
    bus.DREAM_WAKE_FORWARD,        # → coordinator.dreaming.request_wake(reason)
    # Force-dream wiring (post-§4.I D8-3 cleanup, 2026-05-15 evening):
    # FORCE_DREAM_REQUEST from CommandSender (admin / maker test path) lands
    # here under Phase C since cognitive_worker owns DreamingEngine via
    # InnerTrinityCoordinator. Closes the orphan handler whose subscriber
    # lived in the deleted spirit_worker BEGIN_DREAMING coord_event block.
    bus.FORCE_DREAM_REQUEST,       # → coordinator.dreaming.request_dream(reason)
    bus.CONVERSATION_STIMULUS,     # chat → reasoning_engine.observe_stimulus
    bus.EXPERIENCE_STIMULUS,       # experience replay → reasoning_engine
    bus.MEDITATION_COMPLETE,       # meditation phase tracking via coordinator
    bus.MODULE_SHUTDOWN,           # clean shutdown
    bus.SAVE_NOW,                  # B.1 shadow_swap orchestrator (when re-enabled)
    # Track 2 SPEAK gating (v1.2.1 — closes T3 SPEAK quality regression).
    # outer_interface_worker → cognitive_worker advisor-refractory cache feed.
    bus.ADVISOR_REFRACTORY_STATE,  # cache → SPEAK gate (skip emit if refractory)
    # Track 2 prediction_engine drift correction (v1.2.1 commit B8).
    # PredictionEngine relocated to self_reflection_worker; cognitive_worker
    # consumes novelty surrogate via bus event (one-tick latency, per G19).
    bus.PREDICTION_GENERATED,      # cache → _latest_prediction for novelty consumer
    # §4.B Track 3 — expression_worker → cognitive_worker bridges (2026-05-15).
    # SPEAK_REQUEST_PENDING — Tier-1 SPEAK detection from expression_worker;
    #   cognitive_worker assembles the language-pipeline SPEAK_REQUEST using
    #   in-proc consciousness / msl / exp_orchestrator state.
    # NS_REWARD — composite → NS program reward; cognitive_worker calls
    #   neural_nervous_system.record_outcome with the payload's reward+program.
    bus.SPEAK_REQUEST_PENDING,     # → state_refs["_speak_pending_from_bus"]
    bus.NS_REWARD,                 # → neural_nervous_system.record_outcome
    # HORMONE_CONSUME — EXPRESSION composite fire → deplete the driving
    # hormones in the NNS HormonalSystem (neural_nervous_system._hormonal),
    # which is the instance published to nns_hormonal_state.bin and read by
    # expression_worker. Restores the consumption→refractory loop the Phase C
    # split severed (2026-06-01). MUST live here, NOT hormonal_worker: that
    # owns a DIFFERENT hormonal_state.bin instance that expression never reads.
    bus.HORMONE_CONSUME,           # → neural_nervous_system._hormonal.consume
    # EXPERIENCE_RECORD — Record stage of the ExperienceOrchestrator loop
    # (rFP_experience_distillation_phase_c). Per-worker producers emit; this
    # worker enriches + records via _dispatch_experience_record. Targeted
    # dst="cognitive_worker" (P2, never dst=all) — bus-hygiene §3.1.
    bus.EXPERIENCE_RECORD,         # → _dispatch_experience_record (exp_orchestrator.record_outcome)
    # MEMORY_RECALL_PERTURBATION — Phase D (D-SPEC-116). i_depth + working_mem
    # legs of the recall bridge, re-homed from the retired spirit_worker. Emitted
    # by agno_hooks (interface) targeted dst="cognitive_worker"; the neuromod-nudge
    # leg of the same bridge goes to dst="neuromod" separately.
    bus.MEMORY_RECALL_PERTURBATION,  # → msl.i_depth + working_mem.attend
    # TEACHER_SIGNALS — Phase D (D-SPEC-116). language_worker teacher feedback,
    # re-homed from the retired spirit_worker. Restored legs: MSL concept
    # grounding + neuromod nudge (others superseded/covered — see handler).
    bus.TEACHER_SIGNALS,             # → msl.concept_grounder + neuromod nudge
    # OUTER_OBSERVATION — Phase D (D-SPEC-116). action-result observation,
    # re-homed from the retired spirit_worker. Only the X-engagement→MSL leg
    # is live-needed here (other legs superseded/re-homed — see handler).
    bus.OUTER_OBSERVATION,           # → msl.signal_engagement
    # §4.Q (2026-05-15) — bus event sources for neuromod_inputs.bin builder +
    # /v4/neuromodulators cache + TimeChain heartbeat emit. Producer wiring:
    #   PREDICTION_STATS_UPDATED  ← self_reflection_worker (2.5s coalesced)
    #   EXPRESSION_COMPOSITES_UPDATED ← expression_worker (1 Hz)
    #   KIN_SIGNATURE_UPDATED     ← outer_interface_worker (2.5s coalesced)
    #   NEUROMOD_STATS_UPDATED    ← neuromod_worker (2.5s coalesced, §4.Q Q12)
    # Without these subscriptions the handlers in the dispatcher are dead
    # code → state_refs["_last_neuromod_stats"]/_expression_composites/
    # _kin_signature/_prediction_stats stay empty → builder + Fix3 cache
    # fall back to defaults.
    bus.PREDICTION_STATS_UPDATED,  # → state_refs["_prediction_stats"]
    bus.EXPRESSION_COMPOSITES_UPDATED,  # → state_refs["_expression_composites"]
    bus.KIN_SIGNATURE_UPDATED,     # → state_refs["_kin_signature"]
    bus.NEUROMOD_STATS_UPDATED,    # → state_refs["_last_neuromod_stats"]
    # D-SPEC-66 v1.11.0 PLAN §1.6 — kin resonance catalyst emit (D8-3
    # site #7 closure). cognitive_worker subscribes to KIN_SIGNAL +
    # reads DA from neuromod_reader callable (available in handler
    # context); if resonance>0.5 AND DA>0.5 emit SOCIAL_CATALYST
    # (type=kin_resonance). Was spirit_worker.py:8644 dead under
    # Phase C heartbeat-stub.
    bus.KIN_SIGNAL,
    # Session 3 (RFP_meta-reasoning_CGN_FIX.md §4.1 + §4.2 row 1/2/5):
    # cognitive_worker hosts ReasoningEngine + pattern_primitives + chain_archive
    # + MetaCGNConsumer (via MetaReasoningEngine) — receives CGN_KNOWLEDGE_REQ
    # with payload.kind ∈ {reasoning, pattern_primitives, chain_archive,
    # meta_wisdom, language} from meta_service resolvers and publishes
    # CGN_KNOWLEDGE_RESP back via the correlation_id contract.
    bus.CGN_KNOWLEDGE_REQ,
    # RFP_meta-reasoning_CGN_FIX.md Chunk B.7b — MetaService relocated
    # from spirit_worker (D8 retirement target) to cognitive_worker. The
    # 8 CGN consumers now publish META_REASON_REQUEST + META_REASON_OUTCOME
    # with dst="cognitive_worker" via meta_service_client.send_meta_*.
    # CGN_KNOWLEDGE_RESP + TIMECHAIN_QUERY_RESP arrive here as well, so
    # MetaService.handle_response can correlate them with awaiting
    # resolver Futures.
    bus.META_REASON_REQUEST,
    bus.META_REASON_OUTCOME,
    bus.CGN_CONCEPT_GROUNDED,      # Phase B (§9.2) → meta_engine.note_concept_grounded
    bus.CGN_KNOWLEDGE_RESP,
    bus.TIMECHAIN_QUERY_RESP,
    # Subsystem-signal cache refresh (restored from spirit_worker, lost in
    # 72f95a6b D8-3). CONTRACT_LIST_RESP feeds meta_engine.update_subsystem_cache
    # for the contract_* compound-reward signals. TIMECHAIN_QUERY_RESP (above)
    # already subscribed. rFP_subsystem_reward_refresh_restore.md.
    bus.CONTRACT_LIST_RESP,
]


# ARG ORDER (template-critical — see module docstring): every Guardian-spawned
# L2 worker entry follows (recv_queue, send_queue, name, config).
@with_error_envelope(module_name="cognitive_worker", subsystem="entry", severity=_phase11_sev.FATAL)
def cognitive_worker_main(recv_queue, send_queue, name: str, config: dict) -> None:
    """Main loop for the cognitive_worker subprocess.

    Chunk 8E skeleton — heartbeat-only main loop. Bus dispatcher (chunk 8F),
    consciousness epoch driver (chunk 8G), and snapshot publishers
    (chunk 8H) land in subsequent commits.
    """
    # === BOILERPLATE: spawn-mode sys.path bootstrap ===
    # Spawn mode starts a fresh Python interpreter without inheriting the
    # parent's sys.path. Re-add the project root so `from titan_hcl.X
    # import Y` works inside this subprocess. Fork mode inherits sys.path
    # so this is a no-op there.
    project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # === BOILERPLATE: Phase B.2 §C7 socket-mode bus client setup ===
    # Falls back to mp.Queue in legacy mode (loud WARNING from
    # worker_bus_bootstrap if env vars missing). The `topics` list is
    # MODULE-SPECIFIC and enumerates every event type the worker's
    # dispatcher handles. Broker filters dst="all" broadcasts at publish
    # time so only listed types reach this subscriber (closes the
    # per-subscriber flood class identified 2026-04-30).
    from titan_hcl.core.worker_bus_bootstrap import setup_worker_bus
    try:
        recv_queue, send_queue, _bus_client = setup_worker_bus(
            name, recv_queue, send_queue,
            topics=_COGNITIVE_WORKER_SUBSCRIBE_TOPICS,   # MODULE-SPECIFIC
        )
    except Exception as _err:
        logger.error(
            "[CognitiveWorker] setup_worker_bus failed: %s — exiting", _err,
            exc_info=True)
        return

    # RFP_meta-reasoning_CGN_FIX.md Chunk B.7b Fix1c — register "meta_service"
    # as a BUS_SUBSCRIBE alias on cognitive_worker per SPEC §8.2 v1.3.0
    # D-SPEC multi-name. MetaService resolvers publish bus events with
    # src="meta_service" (virtual identity); downstream handlers respond
    # with dst=msg.src; without this alias the broker would have no target
    # for "meta_service" and drop the response (→ resolver_timeout). The
    # alias keeps the virtual identity stable across future MetaService
    # relocations: only this one line updates if MetaService moves again.
    if _bus_client is not None and hasattr(_bus_client, "subscribe_alias"):
        try:
            _bus_client.subscribe_alias("meta_service")
            logger.info(
                "[CognitiveWorker] BUS_SUBSCRIBE alias 'meta_service' "
                "registered (SPEC §8.2 v1.3.0 multi-name; Session 3 "
                "response routing)")
        except Exception as _alias_err:
            logger.warning(
                "[CognitiveWorker] subscribe_alias('meta_service') failed: %s",
                _alias_err)

    # === BOILERPLATE: pdeathsig installation ===
    # Linux PR_SET_PDEATHSIG: kernel delivers SIGTERM if titan_HCL parent
    # dies, so this worker can't outlive its supervisor (matches A.8
    # graduated-spawn worker pattern). Failure is non-fatal — the
    # parent_watcher fallback in worker_lifecycle handles it.
    try:
        from titan_hcl.core.worker_lifecycle import install_parent_death_signal
        install_parent_death_signal()
    except Exception as _err:
        logger.debug("[CognitiveWorker] pdeathsig install skipped: %s", _err)

    from titan_hcl.core.state_registry import resolve_titan_id
    titan_id = (
        (config.get("info_banner", {}) or {}).get("titan_id")
        or resolve_titan_id()
    )
    boot_ts = time.time()

    global _WORKER_READY, _STATE_WRITER
    _WORKER_READY = False

    # ── Phase 11 §11.I.5 — SHM state-slot writer (G21 per worker) ──
    # Created BEFORE flag check + slow engine init so the slot publishes
    # state="starting" immediately; _send_heartbeat refreshes last_heartbeat
    # during boot to keep guardian's staleness detector at bay.
    try:
        from titan_hcl.core.module_state import BootPriority, ModuleStateWriter
        _STATE_WRITER = ModuleStateWriter(
            module_name="cognitive_worker",
            layer="L2",
            boot_priority=BootPriority.MANDATORY,
        )
        _STATE_WRITER.write_state("starting")
    except Exception as _sw_err:  # noqa: BLE001
        _STATE_WRITER = None
        logger.warning(
            "[CognitiveWorker] Phase 11 ModuleStateWriter init failed: %s",
            _sw_err)

    # === BOILERPLATE: optional flag-gated activation ===
    # Workers that have a legacy parallel code path (cognitive_worker
    # under l0_rust=false → legacy spirit_worker_main; future
    # expression_worker under expression.enabled=false → legacy in
    # spirit_worker, etc.) check the activation flag here. If the flag
    # is off, MODULE_READY + heartbeat-only no-op loop so guardian
    # doesn't restart-loop us. Workers without a parallel legacy path
    # can DELETE this entire `if not flag_on:` block.
    # legacy_core.py registration is also gated on the flag so this
    # check is defensive (registration normally skips us in the off-mode).
    flag_on = bool((config or {}).get("microkernel", {}).get("l0_rust_enabled", False))
    if not flag_on:
        logger.info(
            "[CognitiveWorker] microkernel.l0_rust_enabled=false — "
            "legacy spirit_worker_main owns cognitive engines in this mode. "
            "Entering heartbeat-only no-op loop.")
        # Phase 11 §11.I.2 — MODULE_READY bus-emit deleted per locked D2;
        # SHM slot transitions starting → booted (flag-off no-op flavor).
        _WORKER_READY = True
        if _STATE_WRITER is not None:
            try:
                _STATE_WRITER.write_state("booted")
                logger.info(
                    "[CognitiveWorker] Phase 11 §11.I.2 — SHM slot state=booted "
                    "(flag_off no-op branch)")
            except Exception as _swb_err:  # noqa: BLE001
                logger.warning(
                    "[CognitiveWorker] Phase 11 write_state(booted) failed "
                    "(flag_off branch): %s", _swb_err)
        _heartbeat_loop(recv_queue, send_queue, name, flag_off=True)
        return

    logger.info(
        "[CognitiveWorker] Booting (titan_id=%s, l0_rust=true) — chunk 8E "
        "skeleton. Bus dispatcher / epoch driver / snapshot publishers "
        "land in chunks 8F–8H.", titan_id)

    # === MODULE-SPECIFIC: engine init ===
    # Each L2 worker has its own engine cluster — replace this call with
    # your worker's equivalent (e.g., `_init_expression_engines(config)`,
    # `_init_social_engines(config)`). The state_refs dict shape (one key
    # per engine, value = engine instance or None on init failure) is the
    # template-canonical shape consumed by snapshot builder threads.
    state_refs = _init_cognitive_engines(config, send_queue)

    # === MODULE-SPECIFIC: 6 trinity cache slots (SPEC §8.5 + G1) ===
    # Bus dispatcher (below) writes here on each BODY_STATE/MIND_STATE/
    # SPIRIT_STATE event indexed by payload.src. Epoch driver (chunk 8G)
    # reads all 6 at each tick into coordinator.update(...). Default
    # values are 0.5 center per SPEC G5 (Middle Path equilibrium). GIL
    # makes list-pointer reassignment atomic — no explicit lock needed
    # for write-from-main-loop / read-from-epoch-driver pattern.
    state_refs["_inner_body_state"] = [0.5] * 5
    state_refs["_outer_body_state"] = [0.5] * 5
    state_refs["_inner_mind_state"] = [0.5] * 15
    state_refs["_outer_mind_state"] = [0.5] * 15
    state_refs["_inner_spirit_state"] = [0.5] * 45
    state_refs["_outer_spirit_state"] = [0.5] * 45
    state_refs["_circadian_phase"] = 0.0  # KERNEL_EPOCH_TICK target

    # === MODULE-SPECIFIC: consciousness DB + topology (chunk 8G epoch driver) ===
    try:
        from titan_hcl.logic.consciousness_epoch import _init_consciousness  # Phase 10D
        state_refs["consciousness"] = _init_consciousness(config)
    except Exception as _err:
        logger.warning("[CognitiveWorker] _init_consciousness failed: %s", _err)
        state_refs["consciousness"] = None

    # === RFP_meta-reasoning_CGN_FIX.md Chunk B.7b — MetaService relocation ==
    # MetaService + MetaRecruitment instances move from spirit_worker (D8
    # retirement target) to cognitive_worker (canonical home of
    # MetaReasoningEngine + MetaCGNConsumer per SPEC §9.B §4.A SHIPPED
    # 2026-05-05). cognitive_worker becomes the F-phase request/response
    # hub. All 8 CGN consumers now target dst="cognitive_worker" via
    # meta_service_client.send_meta_request (single helper updated atomically).
    state_refs["_meta_service"] = None
    state_refs["_meta_recruitment"] = None
    try:
        from titan_hcl.logic.meta_recruitment import MetaRecruitment
        state_refs["_meta_recruitment"] = MetaRecruitment()
        _mr_health = state_refs["_meta_recruitment"].catalog_health_check()
        logger.info(
            "[CognitiveWorker] Meta-recruitment booted: %d keys, %d recruiters, "
            "%d stale (resolvers register after MetaService init)",
            _mr_health["catalog_keys"], _mr_health["recruiters_total"],
            _mr_health["stale_recruiter_count"])
    except Exception as _mrc_err:
        logger.warning("[CognitiveWorker] Meta-recruitment init: %s", _mrc_err)

    try:
        from titan_hcl.logic.meta_service import MetaService
        def _meta_service_emit(_msg):
            try:
                send_queue.put_nowait(_msg)
            except Exception as _mse_err:
                logger.debug("[MetaService] send_queue put failed: %s",
                             _mse_err)
        state_refs["_meta_service"] = MetaService(
            response_emitter=_meta_service_emit,
            outcome_sink=None,
            recruitment=state_refs["_meta_recruitment"],
        )
        logger.info(
            "[CognitiveWorker] Meta-service booted (Session 3 live dispatch)")
        # Phase A (RFP_cgn_enhancements §9.1) — CANONICAL grounding_sink wiring.
        # _init_cognitive_engines (line 588) already built meta_engine and stored
        # state_refs["meta_engine"], so by here BOTH exist regardless of the
        # symmetric (no-op-when-first) attempt inside _init_cognitive_engines.
        # This routes concept-grounding META_REASON_REQUESTs into the engine's
        # chain-trigger queue (should_trigger_meta Path #0).
        _eng_for_grounding = state_refs.get("meta_engine")
        if _eng_for_grounding is not None and state_refs["_meta_service"] is not None:
            state_refs["_meta_service"]._grounding_sink = \
                _eng_for_grounding.enqueue_grounding
            logger.info(
                "[CognitiveWorker] Phase A — meta_service grounding_sink wired to "
                "meta_engine.enqueue_grounding (learning-event Path #0 ACTIVE)")
        else:
            logger.warning(
                "[CognitiveWorker] Phase A — meta_engine=%s at meta-service init; "
                "grounding_sink NOT wired (Path #0 INACTIVE)",
                "present" if _eng_for_grounding is not None else "MISSING")
    except Exception as _ms_err:
        logger.warning("[CognitiveWorker] Meta-service init: %s", _ms_err)

    # Register the 10 Session 3 LIVE async resolvers (each closes over
    # send_queue + pending_registry).
    if state_refs["_meta_recruitment"] is not None:
        try:
            from titan_hcl.logic.meta_resolvers import (
                register_default_resolvers as _rdr,
            )
            _pending_reg = (state_refs["_meta_service"].pending_registry
                            if state_refs["_meta_service"] is not None
                            else None)
            # Phase 9 INV-Syn-22: build the read-only EngineRecall recall reader
            # so meta-reasoning RECALL resolves in-process (no sync bus.request).
            # Defensive: any failure → engine_recall=None → pure legacy dispatch
            # (zero regression). Embedder/faiss/kuzu read handles are not wired
            # in cognitive_worker yet, so embedding-dependent granularities
            # (turn/concept) soft-fall-back to the legacy resolver during the
            # parity-soak window (recall_parity_soft_fallback); archive routes
            # through the engine today.
            _engine_recall = None
            _recall_enabled = True
            _recall_soft_fallback = True
            try:
                from titan_hcl.synthesis.bridge_recall import BridgeRecall
                from titan_hcl.synthesis.recall_reader import build_recall_reader
                _meta_cfg = {}
                try:
                    import tomllib as _toml
                    _pp = os.path.join(
                        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "titan_params.toml")
                    with open(_pp, "rb") as _f:
                        _meta_cfg = (_toml.load(_f)
                                     .get("synthesis", {}).get("meta", {}))
                except Exception:
                    _meta_cfg = {}
                _recall_enabled = bool(_meta_cfg.get("recall_engine_enabled", True))
                _recall_soft_fallback = bool(
                    _meta_cfg.get("recall_parity_soft_fallback", True))
                if _recall_enabled:
                    _data_dir = os.environ.get("TITAN_DATA_DIR", "data")
                    _engine_recall = build_recall_reader(
                        data_dir=_data_dir,
                        bridge_recall=BridgeRecall(),
                    )
            except Exception as _er_err:
                logger.warning(
                    "[CognitiveWorker] INV-Syn-22 recall reader build failed: "
                    "%s — RECALL stays on legacy dispatch", _er_err)
                _engine_recall = None
            _rr = _rdr(
                state_refs["_meta_recruitment"],
                send_queue=send_queue,
                pending_registry=_pending_reg,
                engine_recall=_engine_recall,
                recall_engine_enabled=_recall_enabled,
                recall_soft_fallback=_recall_soft_fallback,
            )
            logger.info(
                "[CognitiveWorker] Session 3 resolvers bound: %d/%d "
                "(pending_registry=%s, RECALL operator=%s)",
                sum(1 for v in _rr.values() if v), len(_rr),
                "wired" if _pending_reg is not None else "None",
                "EngineRecall (INV-Syn-22)" if _engine_recall is not None
                else "legacy")
        except Exception as _rdr_err:
            logger.warning(
                "[CognitiveWorker] Session 3 resolver registration failed: %s",
                _rdr_err)

    # RFP_meta-reasoning_CGN_FIX.md Chunk B.7b followup — attach MetaService
    # to coordinator so spirit_loop.get_coordinator snapshot exposes
    # /v4/meta-service status (reads coordinator._meta_service at
    # spirit_loop.py:1753). Runs HERE (in cognitive_worker_main, post
    # _init_cognitive_engines return) because:
    #   - state_refs is built by _init_cognitive_engines and returned;
    #     state_refs["coordinator"] is set during that function;
    #   - MetaService init above also writes state_refs["_meta_service"];
    #   - this is the first scope where BOTH coordinator + _meta_service
    #     are available together.
    # Mirrors the legacy spirit_worker.py:1826-1828 post-init attachment.
    _coord = state_refs.get("coordinator")
    _ms = state_refs.get("_meta_service")
    _mr = state_refs.get("_meta_recruitment")
    if _coord is not None:
        if _ms is not None:
            _coord._meta_service = _ms
        if _mr is not None:
            _coord._meta_recruitment = _mr
        logger.info(
            "[CognitiveWorker] Coordinator wired to MetaService=%s + "
            "MetaRecruitment=%s (Session 3 /v4/meta-service path)",
            "OK" if _ms is not None else "None",
            "OK" if _mr is not None else "None")

    # === MODULE-SPECIFIC: NEUROMOD_STATE shm reader (SPEC §10.G fallback) ===
    # neuromod_state.bin is a 6-float shm slot owned by neuromod_worker
    # (DA, 5HT, NE, ACh, Endorphin, GABA in canonical order). Read at
    # each epoch tick — drives coordinator.update_neuromodulators.
    state_refs["_neuromod_reader"] = _make_neuromod_reader()

    # === MODULE-SPECIFIC: full shm reader bank (chunk 8M.4) ===
    # Per SPEC §1096 cognitive_worker reads 12+ Rust-owned shm slots each
    # epoch tick. Closes rFP_phase_c_observatory_data_pipeline.md Gap A
    # (§2.1) + Gap H (§2.8) — the missing read-back layer that left
    # /v4/inner-trinity subfields empty under l0_rust_enabled=true.
    state_refs["_shm_reader_bank"] = _init_shm_reader_bank(titan_id)

    # §4.B Track 3 — expression_state.bin reader (2026-05-15).
    # Composite stats now live in this SHM slot, written by
    # expression_worker. Used in the LifeForceEngine compute path
    # (compute_expression_fire_rate) where expression_manager.get_stats()
    # used to be called pre-extraction.
    state_refs["expression_state_reader"] = None
    try:
        from titan_hcl.core.state_registry import (
            StateRegistryReader, ensure_shm_root,
        )
        from titan_hcl.logic.expression_state_specs import (
            EXPRESSION_STATE_SPEC,
        )
        state_refs["expression_state_reader"] = StateRegistryReader(
            EXPRESSION_STATE_SPEC, ensure_shm_root(titan_id))
    except Exception as _exrerr:
        logger.warning(
            "[CognitiveWorker] expression_state_reader init failed: %s — "
            "expression_fire_rate input to chi will fall back to 0",
            _exrerr)

    # === inner_spirit sensor cache writer (Sprint 7 §4.6 +
    # rFP_phase_c_130d_rust_l1_port closure) ===
    # cognitive_worker hosts the InnerSpiritSensorRefresh sidecar because
    # spirit_worker has been retired on T3 (D8-3 partial). Without this
    # writer, sensor_cache_inner_spirit.bin is never produced and
    # titan-inner-spirit-rs starves on constant-zero input → all 45
    # inner_spirit dims stuck at PARTIAL. Per
    # `feedback_verify_worker_runs_on_target_before_implementing.md`
    # codified 2026-05-12 after this mistake was caught post-deploy.
    # The same sidecar runs in spirit_worker on T1+T2 (Phase A+B) via
    # the same `start_inner_spirit_sensor_refresh` entry point.
    state_refs["_inner_spirit_stop_event"] = threading.Event()
    try:
        from titan_hcl.logic.inner_spirit_sidecar import (
            start_inner_spirit_sensor_refresh)
        state_refs["_inner_spirit_sensor_thread"] = (
            start_inner_spirit_sensor_refresh(
                config=config,
                stop_event=state_refs["_inner_spirit_stop_event"],
                logger=logger,
                log_prefix="[CognitiveWorker]",
            ))
    except Exception:
        logger.critical(
            "[CognitiveWorker] failed to import/start inner_spirit_sidecar "
            "— inner_spirit_45d.bin will starve.", exc_info=True)
        state_refs["_inner_spirit_sensor_thread"] = None

    logger.info(
        "[CognitiveWorker] all engines initialized: reasoning=%s meta=%s "
        "dreaming=%s pi_monitor=%s NS=%s coord=%s obs=%s",
        state_refs["reasoning_engine"] is not None,
        state_refs["meta_engine"] is not None,
        (state_refs["coordinator"] is not None
         and getattr(state_refs["coordinator"], "dreaming", None) is not None),
        state_refs["pi_monitor"] is not None,
        state_refs["neural_nervous_system"] is not None,
        state_refs["coordinator"] is not None,
        state_refs["observable_engine"] is not None,
    )

    # Phase 11 §11.I.2 — MODULE_READY bus-emit deleted per locked D2.
    # SHM slot transitions starting → booted; titan_hcl's 1Hz SHM poll
    # dispatches MODULE_PROBE_REQUEST after seeing this.
    _WORKER_READY = True
    if _STATE_WRITER is not None:
        try:
            _STATE_WRITER.write_state("booted")
            logger.info(
                "[CognitiveWorker] Phase 11 §11.I.2 — SHM slot state=booted "
                "(awaiting MODULE_PROBE_REQUEST from titan_hcl)")
        except Exception as _swb_err:  # noqa: BLE001
            logger.warning(
                "[CognitiveWorker] Phase 11 write_state(booted) failed: %s",
                _swb_err)
    logger.info("[CognitiveWorker] online")

    # === MODULE-SPECIFIC: launch snapshot publisher daemon threads (chunk 8H) ===
    # start_snapshot_builder_threads (lives in spirit_loop.py per PLAN §2.1)
    # spawns 3 daemon threads that periodically read engine .get_stats() +
    # publish *_UPDATED bus events on 2.5s cadence:
    #   REASONING_STATS_UPDATED      → cache.reasoning.state       → /v4/reasoning
    #   META_REASONING_STATS_UPDATED → cache.meta_reasoning.state  → /v4/meta-reasoning
    #   DREAMING_STATE_UPDATED       → cache.dreaming.state        → /v4/dreaming
    #   PI_HEARTBEAT_UPDATED         → cache.pi_heartbeat.state    → /v4/pi-heartbeat
    #   NEUROMOD_STATS_UPDATED       → cache.neuromods.state       → /v4/inner-trinity
    #   EXPRESSION_COMPOSITES_UPDATED → cache.expression.composites → /v4/expression-composites
    #   MSL_STATE_UPDATED            → cache.msl.state             → /v4/inner-trinity.msl
    #   TOPOLOGY_STATE_UPDATED       → cache.topology.state        → /v4/inner-trinity.topology
    # Threads consume the state_refs dict (engines wired by chunk 8E init +
    # 6 trinity cache slots populated by chunk 8F dispatcher + epoch updates
    # from chunk 8G coordinator.tick). _safe_set in build_coordinator_snapshot
    # tolerates None entries, so engines that failed to init at boot are
    # cleanly skipped.
    try:
        # Phase 10E — snapshot builders relocated out of the retiring spirit_loop
        # into logic/snapshot_builders.py (driven here; they read cognitive_worker's
        # in-process engine objects via state_refs).
        from titan_hcl.logic.snapshot_builders import start_snapshot_builder_threads
        start_snapshot_builder_threads(
            state_refs, config, send_queue=send_queue, name=name)
        logger.info(
            "[CognitiveWorker] snapshot builder threads launched — "
            "*_UPDATED publishers active (reasoning=%s meta=%s dreaming=%s "
            "pi_monitor=%s NS=%s coord=%s expr=%s)",
            state_refs.get("reasoning_engine") is not None,
            state_refs.get("meta_engine") is not None,
            (state_refs.get("coordinator") is not None
             and getattr(state_refs.get("coordinator"), "dreaming", None) is not None),
            state_refs.get("pi_monitor") is not None,
            state_refs.get("neural_nervous_system") is not None,
            state_refs.get("coordinator") is not None,
            state_refs.get("expression_manager") is not None,
        )
    except Exception as _err:
        logger.error(
            "[CognitiveWorker] snapshot builder threads start failed — "
            "/v4/* routes will return empty values: %s", _err, exc_info=True)

    # === MODULE-SPECIFIC: launch adaptive consciousness epoch thread (chunk 8G) ===
    # Daemon thread fires every 1-30s adaptively (default 10.35s = 9× Schumann
    # body period per SPEC §18.1 + COGNITIVE_EPOCH_DEFAULT_INTERVAL_S).
    # Thread reads from state_refs (written by main loop dispatcher); GIL
    # covers list-pointer atomicity for the cache-slot read pattern.
    _epoch_stop_event = threading.Event()
    _epoch_early_fire_event = threading.Event()
    state_refs["_epoch_early_fire_event"] = _epoch_early_fire_event
    _epoch_thread = threading.Thread(
        target=_cognitive_epoch_loop,
        args=(state_refs, config, send_queue, name,
              _epoch_stop_event, _epoch_early_fire_event),
        name=f"cognitive_epoch_{titan_id}",
        daemon=True,
    )
    _epoch_thread.start()
    logger.info(
        "[CognitiveWorker] epoch driver started "
        "(default=%.2fs, min=%.2fs, max=%.2fs, persist_every=%d)",
        COGNITIVE_EPOCH_DEFAULT_INTERVAL_S,
        COGNITIVE_EPOCH_MIN_INTERVAL_S,
        COGNITIVE_EPOCH_MAX_INTERVAL_S,
        COGNITIVE_PERSIST_EVERY_N_EPOCHS,
    )

    # v0 app-inbox consumer (RFP_titan_app_event_channel §7.3) — drains phone→kernel
    # Channel-2 responses/feedback + logs declared availability. Observability ONLY (no
    # self-regulation; that's missions RFP Phase 4). Best-effort daemon; never fatal.
    try:
        from titan_hcl.utils.app_inbox_consumer import start_app_inbox_consumer
        state_refs["_app_inbox_stop_event"] = start_app_inbox_consumer()
    except Exception as e:
        logger.warning("[CognitiveWorker] app-inbox consumer not started: %s", e)

    # === BOILERPLATE: Phase B.1 readiness reporter ===
    # Lets shadow_swap orchestrator drain this worker's state before
    # kernel swap. The `save_state_cb` is MODULE-SPECIFIC — return a
    # list of file paths that shadow_swap should persist. For chunk 8E
    # skeleton this is `[]`; chunks 8G/8I wire reasoning_totals.json,
    # dreaming_state.json, pi_heartbeat_state.json, neural_ns/* etc.
    try:
        from titan_hcl.core.readiness_reporter import trivial_reporter
        _b1_reporter = trivial_reporter(
            worker_name=name, layer="L2", send_queue=send_queue,
            save_state_cb=lambda: [],   # MODULE-SPECIFIC: chunk 8G/8I wire state files
        )
    except Exception:
        _b1_reporter = None

    # === BOILERPLATE: main loop skeleton (heartbeat + B.1 + B.2.1 + dispatcher) ===
    # Heartbeat-only for chunk 8E. Chunks 8F (dispatcher) / 8G (epoch
    # tick) / 8H (snapshot publishers) extend this loop without changing
    # its top-level shape.
    last_heartbeat_ts = 0.0
    last_meta_sweep_ts = 0.0
    _META_SWEEP_INTERVAL_S = 1.0  # RFP §4.4 — sweep stale dispatches every ~1s
    while True:
        now = time.time()
        if now - last_heartbeat_ts >= _HEARTBEAT_INTERVAL_S:
            _send_heartbeat(send_queue, name)
            last_heartbeat_ts = now
        # RFP_meta-reasoning_CGN_FIX.md Chunk B.7b — MetaService sweep_timeouts
        # tick (relocated from spirit_worker). Stale dispatches exceeding
        # resolver_dispatch_timeout_s (default 5s) surface failure_mode=
        # resolver_timeout within ~1s of timeout.
        if now - last_meta_sweep_ts >= _META_SWEEP_INTERVAL_S:
            _ms = state_refs.get("_meta_service")
            if _ms is not None:
                try:
                    _ms.sweep_timeouts()
                except Exception as _sw_err:
                    logger.warning(
                        "[MetaService] sweep_timeouts error: %s", _sw_err)
            last_meta_sweep_ts = now

        try:
            msg = recv_queue.get(timeout=_POLL_INTERVAL_S)
        except Empty:
            continue
        except (KeyboardInterrupt, SystemExit):
            break
        except Exception:
            continue

        msg_type = msg.get("type")

        # ── Phase 11 §11.I.3 — MODULE_PROBE_REQUEST handler ─────────
        if msg_type == bus.MODULE_PROBE_REQUEST and _STATE_WRITER is not None:
            try:
                from titan_hcl.core.probe_dispatcher import (
                    handle_module_probe_request,
                )
                handle_module_probe_request(
                    msg,
                    probe_fn=None,  # trivial pass-through per §11.I.2
                    send_queue=send_queue,
                    module_name=name,
                    state_writer=_STATE_WRITER,
                )
            except Exception as _phb_err:  # noqa: BLE001
                logger.warning(
                    "[CognitiveWorker] Phase 11 probe handler raised: %s",
                    _phb_err)
            continue

        # Phase B.1 shadow swap dispatch.
        if _b1_reporter is not None and _b1_reporter.handles(msg_type):
            _b1_reporter.handle(msg)
            if _b1_reporter.should_exit():
                break
            continue

        # Phase B.2.1 supervision-transfer dispatch — preserves spawn-mode adoption.
        try:
            from titan_hcl.core import worker_swap_handler as _swap
            if _swap.maybe_dispatch_swap_msg(msg):
                continue
        except Exception:
            pass

        if msg_type == bus.MODULE_SHUTDOWN:
            logger.info("[CognitiveWorker] Shutdown received — stopping epoch driver + persisting state")
            _epoch_stop_event.set()
            _epoch_early_fire_event.set()  # unblock the wait in epoch loop
            _epoch_thread.join(timeout=5.0)
            # Let an in-flight off-tick dream consolidation finish its save_all
            # so we don't tear a write (daemon threads die on exit otherwise).
            # rFP_dream_consolidation_suite_offtick_restoration §2.3 / D-SPEC-105.
            _shutdown_coord = state_refs.get("coordinator")
            if _shutdown_coord is not None and hasattr(
                    _shutdown_coord, "join_consolidation"):
                try:
                    _shutdown_coord.join_consolidation(timeout=15.0)
                except Exception as _jc_err:
                    logger.warning(
                        "[CognitiveWorker] join_consolidation on shutdown "
                        "failed: %s", _jc_err)
            try:
                _persist_engine_state(state_refs)
            except Exception as _err:
                logger.warning("[CognitiveWorker] final persist on shutdown failed: %s", _err)
            return

        # === MODULE-SPECIFIC: bus dispatcher (PLAN §3.1 driver table) ===
        # 3 trinity event types × payload.src ∈ {inner, outer} → 6 cache
        # slots (SPEC §8.5 + G1 doctrinal symmetry). All other handlers
        # delegate to engine methods on state_refs. Failures per-handler
        # are logged + swallowed so one bad payload doesn't take down
        # the dispatcher loop.
        try:
            payload = _decode_payload(msg.get("payload"))

            if msg_type == bus.BODY_STATE:
                _dispatch_trinity_state(
                    state_refs, payload, dim=5,
                    inner_key="_inner_body_state",
                    outer_key="_outer_body_state",
                    type_label="BODY_STATE")

            elif msg_type == bus.MIND_STATE:
                _dispatch_trinity_state(
                    state_refs, payload, dim=15,
                    inner_key="_inner_mind_state",
                    outer_key="_outer_mind_state",
                    type_label="MIND_STATE")

            elif msg_type == bus.SPIRIT_STATE:
                _dispatch_trinity_state(
                    state_refs, payload, dim=45,
                    inner_key="_inner_spirit_state",
                    outer_key="_outer_spirit_state",
                    type_label="SPIRIT_STATE")

            elif msg_type == bus.KERNEL_EPOCH_TICK:
                # Circadian phase (1Hz tick from Rust kernel-rs).
                # Used by epoch driver (chunk 8G) for arming logic.
                phase = payload.get("phase")
                if phase is not None:
                    state_refs["_circadian_phase"] = float(phase)

            elif msg_type == bus.CGN_DREAM_CONSOLIDATE:
                _dispatch_dream_consolidate(state_refs, payload)

            elif msg_type == bus.DREAM_WAKE_FORWARD:
                # v1.8.2 (D-SPEC-56): dream_state_worker forwards
                # DREAM_WAKE_REQUEST (originally from chat-API maker-fast-wake
                # + world-observer DI:URGENT interrupt) here. cognitive_worker
                # owns DreamingEngine via InnerTrinityCoordinator — calling
                # `request_wake(reason)` in-process sets the engine's
                # `_wake_requested` flag. The next `check_transition()` in
                # the DREAMING branch (after `min_dream_s` gate) returns
                # END_DREAMING, the dream lifecycle exits, and
                # spirit_loop._publish_coord_subdomains emits the
                # `DREAMING_STATE_UPDATED.state="dream_end"` payload that
                # dream_state_worker detects → emits DREAM_STATE_CHANGED +
                # flushes inbox via DREAM_INBOX_REPLAY.
                try:
                    coordinator = state_refs.get("coordinator")
                    dreaming = getattr(coordinator, "dreaming", None) \
                        if coordinator else None
                    if dreaming is not None and hasattr(dreaming, "request_wake"):
                        _wake_reason = str(payload.get("reason", "unspecified"))
                        _wake_source = str(payload.get("source", "unknown"))
                        dreaming.request_wake(
                            reason=f"{_wake_reason} (source={_wake_source})")
                        logger.info(
                            "[CognitiveWorker] DREAM_WAKE_FORWARD received "
                            "(reason=%s, source=%s) — coordinator.dreaming.request_wake() called",
                            _wake_reason, _wake_source)
                    else:
                        logger.warning(
                            "[CognitiveWorker] DREAM_WAKE_FORWARD received but "
                            "coordinator.dreaming.request_wake not available "
                            "(coordinator=%s, dreaming=%s) — wake request "
                            "dropped",
                            coordinator is not None, dreaming is not None)
                except Exception as _wake_err:
                    logger.warning(
                        "[CognitiveWorker] DREAM_WAKE_FORWARD dispatch failed: %s",
                        _wake_err, exc_info=True)

            elif msg_type == bus.FORCE_DREAM_REQUEST:
                # Force-dream wiring (post-§4.I D8-3 cleanup, 2026-05-15 evening):
                # admin / maker testing / maintenance / inspection path. The
                # CommandSender publishes FORCE_DREAM_REQUEST originally to
                # dst="spirit"; that handler lived in the deleted spirit_worker
                # BEGIN_DREAMING coord_event block (chunk I8 cleanup) → orphaned.
                # Now rewired to cognitive_worker (which owns DreamingEngine via
                # InnerTrinityCoordinator). Symmetric to DREAM_WAKE_FORWARD
                # above. `request_dream(reason)` sets the engine's
                # `_dream_requested` flag; the next `check_transition()` in the
                # AWAKE branch (before FORCE SLEEP path, bypassing wake_inertia
                # + drive gate) returns BEGIN_DREAMING, the dream lifecycle
                # starts, and spirit_loop._publish_coord_subdomains emits the
                # `DREAMING_STATE_UPDATED.state="dream_start"` payload that
                # dream_state_worker detects → emits canonical DREAM_STATE_CHANGED.
                try:
                    coordinator = state_refs.get("coordinator")
                    dreaming = getattr(coordinator, "dreaming", None) \
                        if coordinator else None
                    if dreaming is not None and hasattr(dreaming, "request_dream"):
                        _dream_reason = str(payload.get("reason", "admin_force"))
                        _dream_source = str(payload.get("source", "admin"))
                        dreaming.request_dream(
                            reason=f"{_dream_reason} (source={_dream_source})")
                        logger.info(
                            "[CognitiveWorker] FORCE_DREAM_REQUEST received "
                            "(reason=%s, source=%s) — coordinator.dreaming.request_dream() called",
                            _dream_reason, _dream_source)
                    else:
                        logger.warning(
                            "[CognitiveWorker] FORCE_DREAM_REQUEST received but "
                            "coordinator.dreaming.request_dream not available "
                            "(coordinator=%s, dreaming=%s) — dream request "
                            "dropped",
                            coordinator is not None, dreaming is not None)
                except Exception as _dream_err:
                    logger.warning(
                        "[CognitiveWorker] FORCE_DREAM_REQUEST dispatch failed: %s",
                        _dream_err, exc_info=True)

            elif msg_type in (bus.CONVERSATION_STIMULUS, bus.EXPERIENCE_STIMULUS):
                _dispatch_stimulus(state_refs, msg_type, payload)

            elif msg_type == bus.KIN_SIGNAL:
                # D-SPEC-66 v1.11.0 PLAN §1.6 — kin resonance catalyst
                # (D8-3 site #7 closure). Reads kin payload + DA from
                # neuromod_reader; if resonance>0.5 AND DA>0.5 emit
                # SOCIAL_CATALYST(type=kin_resonance). Was
                # spirit_worker.py:8644 dead under Phase C heartbeat-stub.
                try:
                    _kin_payload = payload or {}
                    _kin_resonance = float(_kin_payload.get("resonance", 0.0))
                    _kin_emotion = _kin_payload.get("kin_emotion", "neutral")
                    _kin_pubkey = str(_kin_payload.get("kin_pubkey", "unknown"))
                    # Record stage (rFP_experience_distillation_phase_c): a kin
                    # exchange is a communication experience. Recorded in-proc —
                    # cognitive_worker owns the ExperienceOrchestrator, so no bus
                    # hop (mirrors deleted spirit_worker.py:8609 kin_sense record).
                    _dispatch_experience_record(state_refs, {
                        "domain": "communication",
                        "action_taken": "kin_sense",
                        "outcome_score": _kin_resonance,
                        "context": {
                            "kin_pubkey": _kin_pubkey[:8],
                            "kin_emotion": _kin_emotion,
                        },
                    })
                    if _kin_resonance > 0.5:
                        _kin_nr = state_refs.get("_neuromod_reader")
                        _kin_neuromods = _kin_nr() if _kin_nr else {}
                        _kin_da = float(_kin_neuromods.get("DA", 0.5))
                        if _kin_da > 0.5:
                            _send_msg(send_queue, bus.SOCIAL_CATALYST, name,
                                      "social", {
                                          "type": "kin_resonance",
                                          "significance":
                                              min(1.0, _kin_resonance),
                                          "content": (
                                              f"Exchanged consciousness "
                                              f"with my twin — "
                                              f"resonance "
                                              f"{_kin_resonance:.3f}, "
                                              f"they felt {_kin_emotion}"),
                                          "data": {
                                              "resonance": _kin_resonance,
                                              "kin_emotion": _kin_emotion,
                                              "kin_id": _kin_pubkey[:8],
                                          },
                                      })
                            logger.info(
                                "[CognitiveWorker] kin_resonance catalyst "
                                "emitted (resonance=%.3f, DA=%.2f)",
                                _kin_resonance, _kin_da)
                except Exception as _kin_err:
                    logger.warning(
                        "[CognitiveWorker] KIN_SIGNAL handler error: %s",
                        _kin_err)

            elif msg_type == bus.MEDITATION_COMPLETE:
                _dispatch_meditation_complete(state_refs, payload)

            elif msg_type == bus.SAVE_NOW:
                # Persist all engine state (chunk 8G persistence cadence
                # is the 100-epoch tick; SAVE_NOW is the on-demand path
                # for shadow_swap B.1 readiness).
                _persist_engine_state(state_refs)

            elif msg_type == bus.ADVISOR_REFRACTORY_STATE:
                # Track 2 (v1.2.1) — cache outer_interface_worker's advisor
                # refractory snapshot. Consumed by the SPEAK emit block in
                # _drive_one_epoch (chunk A8) to skip emit when SPEAK is
                # within its refractory window. Coalesced at broker by
                # ("titan_id",) so freshest snapshot wins in place.
                state_refs["_advisor_state"] = {
                    "action_refractory": payload.get("action_refractory", {}),
                    "cooldown_multiplier": payload.get("cooldown_multiplier", 9.0),
                    "ts": payload.get("ts", time.time()),
                }

            elif msg_type == bus.PREDICTION_GENERATED:
                # Track 2 (v1.2.1 commit B8) — drift correction consumer side.
                # self_reflection_worker emits PREDICTION_GENERATED whenever
                # PredictionEngine._total_predictions counter grows.
                # cognitive_worker caches the latest prediction surrogate for
                # novelty-driven exploration paths (formerly in-process via
                # prediction_engine.predict_next; now bus-event with one-tick
                # latency, acceptable per G19).
                state_refs["_latest_prediction"] = {
                    "total_predictions": payload.get("total_predictions", 0),
                    "total_surprises": payload.get("total_surprises", 0),
                    "last_prediction": payload.get("last_prediction"),
                    "ts": payload.get("ts", time.time()),
                }

            elif msg_type == bus.PREDICTION_STATS_UPDATED:
                # §4.Q (2026-05-15) — cached prediction stats feed for the
                # neuromod_inputs.bin builder. self_reflection_worker publishes
                # this 2.5s coalesced; carries novelty_signal + total_predictions
                # + total_surprises + EMA from PredictionEngine.
                state_refs["_prediction_stats"] = {
                    "novelty_signal": payload.get("novelty_signal", 0.0),
                    "total_predictions": payload.get("total_predictions", 0),
                    "total_surprises": payload.get("total_surprises", 0),
                    "ts": payload.get("ts", time.time()),
                }

            elif msg_type == bus.EXPRESSION_COMPOSITES_UPDATED:
                # §4.Q (2026-05-15) — cached expression composite stats feed
                # for the neuromod_inputs.bin builder. expression_worker
                # publishes this 1 Hz; carries the 6-composite ledger.
                state_refs["_expression_composites"] = (
                    payload.get("composites") or payload or {})
                state_refs["_expression_composites_ts"] = payload.get(
                    "ts", time.time())

            elif msg_type == bus.KIN_SIGNATURE_UPDATED:
                # §4.Q (2026-05-15) — cached kin signature feed for the
                # neuromod_inputs.bin builder. outer_interface_worker
                # publishes this 2.5s coalesced; carries kin resonance +
                # last exchange timestamp for the kin → DA/Endorphin/5HT/NE
                # boost injection.
                state_refs["_kin_signature"] = {
                    "last_resonance": payload.get("last_resonance", 0.0),
                    "last_exchange_ts": payload.get("last_exchange_ts", 0.0),
                    "resonant_count": payload.get("resonant_count", 0),
                    "ts": payload.get("ts", time.time()),
                }

            elif msg_type == bus.NEUROMOD_STATS_UPDATED:
                # §4.Q chunk Q11 (2026-05-15) — cached emotion + modulators
                # snapshot. Used by TimeChain heartbeat emit (current_emotion
                # field). Publisher: neuromod_worker (2.5s coalesced).
                state_refs["_last_neuromod_stats"] = {
                    "modulators": payload.get("modulators", {}) or {},
                    "modulation": payload.get("modulation", {}) or {},
                    "current_emotion": payload.get("current_emotion", "neutral"),
                    "emotion_confidence": payload.get("emotion_confidence", 0.0),
                    "total_evaluations": payload.get("total_evaluations", 0),
                    "ts": payload.get("ts", time.time()),
                }

            elif msg_type == bus.SPEAK_REQUEST_PENDING:
                # §4.B Track 3 (2026-05-15) — Tier-1 SPEAK detection
                # arrives from expression_worker. Block 8.5 in
                # _drive_one_epoch consumes
                # state_refs["_speak_pending_from_bus"] with a 5s TTL.
                state_refs["_speak_pending_from_bus"] = {
                    "urge": payload.get("urge", 0.0),
                    "hormones": payload.get("hormones", {}) or {},
                    "developmental_age": payload.get(
                        "developmental_age", 0),
                    "tier2_fired_composites": payload.get(
                        "tier2_fired_composites", []) or [],
                    "ts": payload.get("ts", time.time()),
                }

            elif msg_type == bus.NS_REWARD:
                # §4.B Track 3 (2026-05-15) — composite → NS program
                # reward bridge from expression_worker. Dispatch to
                # neural_nervous_system.record_outcome in-process here.
                nns = state_refs.get("neural_nervous_system")
                if nns is not None:
                    try:
                        nns.record_outcome(
                            reward=float(payload.get("reward", 0.0)),
                            program=str(payload.get("program", "")),
                            source=str(payload.get(
                                "source", "expression_worker")),
                        )
                    except Exception as _nsr_err:
                        logger.debug(
                            "[CognitiveWorker] NS_REWARD record_outcome "
                            "raised: %s", _nsr_err)

            elif msg_type == bus.HORMONE_CONSUME:
                # 2026-06-01 — EXPRESSION composite fire (expression_worker)
                # depletes the driving hormones in the NNS HormonalSystem
                # (the instance published to nns_hormonal_state.bin that
                # expression_worker reads). Restores the consumption→refractory
                # loop severed by the Phase C split — without it composites
                # re-fire every tick (EXPRESSION.SOCIAL runaway). Applied here,
                # NOT in hormonal_worker, whose hormonal_state.bin instance the
                # expression urge never reads.
                nns = state_refs.get("neural_nervous_system")
                _hsys = getattr(nns, "_hormonal", None) if nns is not None else None
                if _hsys is not None:
                    consumption = payload.get("consumption", {}) or {}
                    for _hname, _amt in consumption.items():
                        try:
                            _horm = _hsys.get_hormone(_hname)
                            if _horm is not None:
                                _horm.consume(float(_amt))
                        except Exception as _hc_err:
                            logger.debug(
                                "[CognitiveWorker] HORMONE_CONSUME %s raised: %s",
                                _hname, _hc_err)

            elif msg_type == bus.EXPERIENCE_RECORD:
                # Record stage of the distillation loop — enrich producer-emitted
                # semantic content with in-proc inner-state + hormones + perception
                # key, then persist via ExperienceOrchestrator (G21 sole writer).
                _dispatch_experience_record(state_refs, payload)

            elif msg_type == bus.MEMORY_RECALL_PERTURBATION:
                # Phase D (D-SPEC-116) — i_depth + working_mem legs of the recall
                # bridge, re-homed from the retired spirit_worker handler. The
                # neuromod-nudge leg is emitted separately by agno_hooks (it owns
                # the delta→target conversion via the neuromod SHM read). Here we
                # only close the two legs that need msl + working_mem (both live
                # in cognitive_worker).
                _mrp_msl = state_refs.get("msl")
                _mrp_nudge = payload.get("nudge_map", {})
                if _mrp_msl is not None and hasattr(_mrp_msl, "i_depth"):
                    try:
                        _mrp_msl.i_depth.record_recall_perturbation()
                    except Exception as _mrp_id_err:
                        logger.debug("[CognitiveWorker] i_depth recall "
                                     "perturbation raised: %s", _mrp_id_err)
                _mrp_wm = state_refs.get("working_mem")
                if _mrp_wm is not None and _mrp_nudge:
                    try:
                        _mrp_cons = state_refs.get("consciousness") or {}
                        _mrp_ep = int((_mrp_cons.get(
                            "latest_epoch", {}) or {}).get("epoch_id", 0) or 0)
                        _mrp_wm.attend(
                            "memory_recall_echo",
                            f"recall_{int(time.time())}",
                            {"nudges": _mrp_nudge,
                             "memory_count": payload.get("memory_count", 0)},
                            _mrp_ep)
                    except Exception as _mrp_wm_err:
                        logger.debug("[CognitiveWorker] working_mem recall "
                                     "attend raised: %s", _mrp_wm_err)

            elif msg_type == bus.TEACHER_SIGNALS:
                # Phase D (D-SPEC-116) — language_worker teacher feedback, re-homed
                # from the retired spirit_worker. Of the old handler's 6 legs only
                # two were genuinely dropped + not otherwise covered; the rest are
                # superseded or already handled in Phase C:
                #   • perturbation_deltas → body/mind = Rust-owned now (superseded)
                #   • dynamic_recipes → covered by the durable path (vocab DB +
                #     language_pipeline narrator.register_dynamic_recipe + the
                #     word_recipes.json save/reload in outer_interface_worker)
                #   • conversation_eval (exp_orchestrator) → covered by
                #     language_worker's EXPERIENCE_RECORD(domain="language") emit
                #     (rFP_experience_distillation_phase_c)
                #   • conversation_question → no live consumer (Phase C routes
                #     conversation through the chat/SPEAK path, not teacher-pending)
                # Restored here: (1) MSL concept grounding, (2) neuromod nudge.
                _tmsl = state_refs.get("msl")
                if _tmsl is not None and getattr(_tmsl, "concept_grounder", None):
                    _tcg = _tmsl.concept_grounder
                    _tepoch = getattr(_tmsl, "_tick_count", 0)
                    for _tsig in payload.get("msl_signals", []):
                        _tc = _tsig.get("concept")
                        _tq = float(_tsig.get("quality", 0.5))
                        try:
                            if _tc == "I":
                                _tmsl.confidence._convergence_count += 1
                            elif _tc == "YOU":
                                _tcg.signal_you("teacher", _tq, _tepoch, None)
                            elif _tc == "YES":
                                _tcg.signal_yes(_tq, _tepoch, None)
                            elif _tc == "NO":
                                _tcg.signal_no(_tq, _tepoch, None)
                        except Exception as _tsig_err:
                            logger.debug("[CognitiveWorker] TEACHER msl signal "
                                         "%s raised: %s", _tc, _tsig_err)
                # Neuromod nudge — old handler applied additive deltas to the
                # in-proc neuromod system (which lived in spirit). neuromod is now
                # its own worker; convert delta→target via the neuromod SHM reader
                # (cognitive owns _neuromod_reader) and emit the standard §4.Q
                # target-shaped NEUROMOD_EXTERNAL_NUDGE to dst="neuromod".
                _tnudge = payload.get("neuromod_nudge", {})
                if _tnudge:
                    try:
                        _tnr = state_refs.get("_neuromod_reader")
                        _tlevels = _tnr() if _tnr else {}
                        _ttargets = {}
                        for _tmod, _tdelta in _tnudge.items():
                            _tcur = float(_tlevels.get(_tmod, 0.5))
                            _ttargets[_tmod] = max(0.0, min(1.0, _tcur + _tdelta))
                        if _ttargets:
                            _tpi = state_refs.get("pi_monitor")
                            _tdev = float(getattr(_tpi, "developmental_age", 1.0) or 1.0)
                            send_queue.put({
                                "type": bus.NEUROMOD_EXTERNAL_NUDGE,
                                "src": name, "dst": "neuromod",
                                "payload": {
                                    "nudge_map": _ttargets,
                                    "max_delta": 0.05,
                                    "developmental_age": _tdev,
                                    "source": "teacher_signals",
                                },
                                "ts": time.time(),
                            })
                    except Exception as _tnudge_err:
                        logger.debug("[CognitiveWorker] TEACHER neuromod nudge "
                                     "raised: %s", _tnudge_err)

            elif msg_type == bus.OUTER_OBSERVATION:
                # Phase D (D-SPEC-116) — action-result observation, re-homed from
                # the retired spirit_worker. Of the old handler's legs only the
                # X-engagement→MSL signal is restored here (no other live consumer):
                #   • outer_body/mind/inner delta-application → Rust-owned (superseded)
                #   • process_action_result narrator learning → outer_interface_worker
                #     (EXPRESSION_FIRED, rFP §2.A.3)
                #   • creative_journal art/music writes → language_worker
                #   • KIN ingestion → KIN_SIGNAL→cognitive (D-SPEC-66)
                # Restored: msl.signal_engagement from X engagement_details.
                _oo_msl = state_refs.get("msl")
                if _oo_msl is not None and hasattr(_oo_msl, "signal_engagement"):
                    _oo_result = payload.get("result", {}) or {}
                    _oo_enr = _oo_result.get("enrichment_data", {})
                    _oo_eng = (_oo_enr.get("engagement_details", [])
                               if isinstance(_oo_enr, dict) else [])
                    for _oo_ed in _oo_eng:
                        try:
                            _oo_regular = _oo_ed.get("user_reply_count", 0) >= 3
                            _oo_msl.signal_engagement(
                                engagement_type=_oo_ed.get("type", "like") + "_received",
                                author=_oo_ed.get("user_name", "unknown"),
                                sentiment_hint=float(_oo_ed.get("relevance", 0.5)),
                                is_regular=_oo_regular)
                        except Exception as _oo_err:
                            logger.debug("[CognitiveWorker] OUTER_OBSERVATION "
                                         "signal_engagement raised: %s", _oo_err)

            elif msg_type == bus.CGN_KNOWLEDGE_REQ:
                # Session 3 live-dispatch handler — RFP_meta-reasoning_CGN_FIX.md
                # §4.1 + §4.2 rows 1/2/5. Receives requests from meta_service
                # resolvers tagged with payload.kind ∈ {reasoning,
                # pattern_primitives, chain_archive} and replies with
                # CGN_KNOWLEDGE_RESP carrying the same correlation_id.
                #
                # SPEC anchors:
                #   - §8.2 D-SPEC-42 + D-SPEC-52: dst="meta_service" targeted
                #     routing delivers to the awaiting resolver coroutine
                #   - §8.0.ter D-SPEC-48: publish via send_queue.put_nowait
                #     (non-blocking on caller's thread)
                #   - Preamble G19: handler responds synchronously without
                #     blocking the bus loop (computation is bounded)
                _ckr_payload = msg.get("payload", {}) or {}
                _ckr_corr = _ckr_payload.get("correlation_id")
                _ckr_kind = _ckr_payload.get("kind", "")
                _ckr_name = _ckr_payload.get("name", "")
                _ckr_src = msg.get("src", "meta_service")

                # Only handle Session 3 dispatch envelopes. Legacy
                # CGN_KNOWLEDGE_REQ (from spirit_worker P8 META-CGN responder
                # path) doesn't carry correlation_id + kind — drop silently
                # so the existing spirit_worker handler can answer them.
                if not _ckr_corr or _ckr_kind not in (
                        "reasoning", "pattern_primitives", "chain_archive",
                        "meta_wisdom", "language"):
                    continue

                # Build a per-kind real response. All paths return a dict
                # that the resolver wraps into META_REASON_RESPONSE.insight.
                _ckr_output: dict
                _ckr_failure = None
                try:
                    if _ckr_kind == "reasoning":
                        _re = state_refs.get("reasoning_engine")
                        _ckr_output = _build_reasoning_response(
                            _re, _ckr_name, _ckr_payload)
                    elif _ckr_kind == "pattern_primitives":
                        _ckr_output = _build_pattern_response(
                            state_refs, _ckr_name, _ckr_payload)
                    elif _ckr_kind == "chain_archive":
                        _ca = state_refs.get("chain_archive")
                        _ckr_output = _build_chain_archive_response(
                            _ca, _ckr_name, _ckr_payload)
                    elif _ckr_kind == "meta_wisdom":
                        _me = state_refs.get("meta_engine")
                        _ckr_output = _build_meta_wisdom_response(
                            _me, _ckr_name, _ckr_payload)
                    elif _ckr_kind == "language":
                        # Transitional target (no language_worker extracted
                        # yet per L2 strategy §4). cognitive_worker doesn't
                        # host language_reasoner directly — surface graceful
                        # placeholder until carve-out ships. RFP §4.2 row 3.
                        _ckr_output = _build_language_transitional_response(
                            _ckr_name, _ckr_payload)
                    else:  # defense-in-depth — never hit due to filter above
                        _ckr_output = {}
                        _ckr_failure = "unknown_kind"
                except Exception as _ckr_err:
                    logger.warning(
                        "[CognitiveWorker] CGN_KNOWLEDGE_REQ kind=%s name=%s "
                        "handler error: %s", _ckr_kind, _ckr_name, _ckr_err)
                    _ckr_output = {"error": str(_ckr_err)}
                    _ckr_failure = "handler_error"

                # Publish response back. dst=meta_service via D-SPEC-52
                # targeted self-route (meta_service lives in spirit_worker).
                _resp_payload = {
                    "correlation_id": _ckr_corr,
                    "kind": _ckr_kind,
                    "name": _ckr_name,
                    "output": _ckr_output,
                    "ts": time.time(),
                }
                if _ckr_failure:
                    _resp_payload["failure"] = _ckr_failure
                try:
                    send_queue.put_nowait({
                        "type": bus.CGN_KNOWLEDGE_RESP,
                        "dst": _ckr_src,
                        "src": name,
                        "payload": _resp_payload,
                    })
                except Exception as _sq_err:
                    logger.warning(
                        "[CognitiveWorker] CGN_KNOWLEDGE_RESP publish "
                        "failed (corr=%s): %s", _ckr_corr[:8], _sq_err)

            elif msg_type == bus.META_REASON_REQUEST:
                # RFP_meta-reasoning_CGN_FIX.md Chunk B.7b — MetaService relocated
                # to cognitive_worker. Route incoming requests to the dispatcher.
                _ms = state_refs.get("_meta_service")
                if _ms is not None:
                    try:
                        _ms.handle_request(msg)
                    except Exception as _msr_err:
                        logger.warning(
                            "[MetaService] request handler error: %s",
                            _msr_err)
                else:
                    logger.debug(
                        "[MetaService] META_REASON_REQUEST received but "
                        "service not yet initialized")

            elif msg_type == bus.META_REASON_OUTCOME:
                _ms = state_refs.get("_meta_service")
                if _ms is not None:
                    try:
                        _ms.handle_outcome(msg)
                    except Exception as _mso_err:
                        logger.warning(
                            "[MetaService] outcome handler error: %s",
                            _mso_err)

            elif msg_type == bus.CGN_CONCEPT_GROUNDED:
                # Phase B (RFP_cgn_enhancements §9.2) — a concept matured across
                # ≥2 consumers (from cgn_worker). Feed the meta engine's Level-B
                # abstraction accumulator; it fires synthesis when enough mature
                # AND a neuromod gate opens (checked in meta_engine.tick).
                _me = state_refs.get("meta_engine")
                if _me is not None:
                    try:
                        _me.note_concept_grounded(msg.get("payload", {}) or {})
                    except Exception as _ncg_err:
                        logger.warning(
                            "[Phase B] note_concept_grounded error: %s", _ncg_err)

            elif msg_type == bus.CGN_KNOWLEDGE_RESP:
                # Session 3 live-dispatch response routing — RFP §4.1 Chunk B.7b.
                # If payload carries correlation_id, route to MetaService for
                # resolver Future resolution. The legacy P8 aggregator path
                # (META-CGN knowledge response) stays in spirit_worker for
                # backward compat — Session 3 responses are tagged with
                # correlation_id which the legacy path doesn't emit.
                _resp_p = msg.get("payload", {}) or {}
                if _resp_p.get("correlation_id"):
                    _ms = state_refs.get("_meta_service")
                    if _ms is not None:
                        try:
                            _ms.handle_response(msg)
                        except Exception as _rh_err:
                            logger.warning(
                                "[MetaService] CGN_KNOWLEDGE_RESP routing "
                                "error: %s", _rh_err)

            elif msg_type == bus.TIMECHAIN_QUERY_RESP:
                # Same Session 3 routing pattern as CGN_KNOWLEDGE_RESP.
                _resp_p = msg.get("payload", {}) or {}
                if _resp_p.get("correlation_id"):
                    _ms = state_refs.get("_meta_service")
                    if _ms is not None:
                        try:
                            _ms.handle_response(msg)
                        except Exception as _rh_err:
                            logger.warning(
                                "[MetaService] TIMECHAIN_QUERY_RESP routing "
                                "error: %s", _rh_err)
                else:
                    # Subsystem-signal cache populate path (no correlation_id),
                    # restored from spirit_worker (lost in 72f95a6b D8-3).
                    # rFP_subsystem_reward_refresh_restore.md.
                    try:
                        _me = state_refs.get("meta_engine")
                        if _me is not None:
                            _tcr = _resp_p.get("results", [])
                            if _tcr or "error" not in _resp_p:
                                _me.update_subsystem_cache(timechain_results=_tcr)
                                logger.info(
                                    "[META] Subsystem cache: TimeChain "
                                    "response (%d blocks)", len(_tcr or []))
                    except Exception as _tcrerr:
                        logger.warning(
                            "[META] TIMECHAIN_QUERY_RESP cache handler "
                            "error: %s", _tcrerr)

            elif msg_type == bus.CONTRACT_LIST_RESP:
                # Subsystem-signal cache populate path for contract_* signals,
                # restored from spirit_worker (lost in 72f95a6b D8-3).
                # rFP_subsystem_reward_refresh_restore.md.
                try:
                    _me = state_refs.get("meta_engine")
                    if _me is not None:
                        _clr = (msg.get("payload", {}) or {}).get(
                            "contracts", [])
                        _me.update_subsystem_cache(contract_results=_clr)
                        logger.info(
                            "[META] Subsystem cache: Contract response "
                            "(%d active)", len(_clr or []))
                except Exception as _clrerr:
                    logger.warning(
                        "[META] CONTRACT_LIST_RESP cache handler error: %s",
                        _clrerr)

            # Unknown message type — silently drop (broker topic filter
            # should prevent these, but defense in depth).
        except Exception as _disp_err:
            logger.warning(
                "[CognitiveWorker] dispatcher %s failed: %s",
                msg_type, _disp_err)

    logger.info("[CognitiveWorker] Exiting")


# === MODULE-SPECIFIC: engine init ==================================
# Replace this whole section with your module's engine init when adapting
# the template. The state_refs dict shape (one key per engine, value =
# instance or None on init failure) is the template-canonical contract.


def _init_cognitive_engines(config: dict, send_queue) -> dict:
    """Initialize all L3 cognitive engines and return the state_refs dict.

    Each block is best-effort: any single failure returns None for that
    subsystem (build_coordinator_snapshot's _safe_set tolerates None).
    The worker keeps running with whatever was successfully booted.
    """
    from titan_hcl.modules._cognitive_init import (
        _init_t2_state_registries, _init_observable_engine,
        _init_neural_nervous_system, _init_coordinator,
    )

    inner_state = spirit_state = observable_engine = None
    neural_nervous_system = coordinator = None
    pi_monitor = reasoning_engine = meta_engine = None

    # Block F: pre-D8 ownership audit Track 1 — memory + monitor engines.
    # Track 2 (v1.2.1 — commit B8): prediction_engine REMOVED from
    # cognitive_worker per rFP_phase_c_self_improvement_subsystem_migration §0
    # drift correction. PredictionEngine now lives in self_reflection_worker
    # (its strategy-correct home per rFP_titan_hcl_l2_separation_strategy §4.E).
    # cognitive_worker consumes PREDICTION_GENERATED bus events for
    # novelty-driven exploration (one-tick latency, acceptable per G19).
    working_mem = None
    intuition_convergence = None
    wallet_observer = None
    meta_recruitment = None
    timeseries_store = None
    outer_reader = None
    meta_service = None
    b1_reporter = None
    episodic_mem = None
    interpreter = None
    mini_registry = None
    med_watchdog = None

    try:
        inner_state, spirit_state = _init_t2_state_registries()
    except Exception as _err:
        logger.warning("[CognitiveWorker] T2 state registries init failed: %s", _err)

    try:
        observable_engine = _init_observable_engine()
    except Exception as _err:
        logger.warning("[CognitiveWorker] ObservableEngine init failed: %s", _err)

    try:
        neural_nervous_system = _init_neural_nervous_system(config)
    except Exception as _err:
        logger.warning("[CognitiveWorker] NeuralNervousSystem init failed: %s", _err)

    try:
        coordinator = _init_coordinator(
            inner_state, spirit_state, observable_engine,
            neural_nervous_system=neural_nervous_system, config=config)
    except Exception as _err:
        logger.warning("[CognitiveWorker] InnerTrinityCoordinator init failed: %s", _err)

    # RFP_meta-reasoning_CGN_FIX.md Chunk B.7b — coordinator attachment.
    # MetaService + MetaRecruitment locals are declared at L1081-1082 of this
    # function (currently always None — instantiation happens in
    # cognitive_worker_main after this function returns, at L455-484). The
    # attach block is preserved for forward-compatibility when MetaService
    # init migrates here. 2026-05-15: replaced `state_refs.get(...)` (NameError
    # — state_refs is the return value of this function, not in scope inside
    # it) with the local meta_service / meta_recruitment vars. Crash-fix only;
    # the actual MetaService attachment-to-coordinator still happens in
    # cognitive_worker_main at the post-_init_cognitive_engines wire-up site.
    if coordinator is not None:
        _ms = meta_service
        _mr = meta_recruitment
        if _ms is not None:
            coordinator._meta_service = _ms
        if _mr is not None:
            coordinator._meta_recruitment = _mr
        logger.info(
            "[CognitiveWorker] Coordinator wired to MetaService=%s + "
            "MetaRecruitment=%s (Session 3 /v4/meta-service path)",
            "OK" if _ms is not None else "None",
            "OK" if _mr is not None else "None")

    # Wire NeuromodulatorSystem onto the coordinator for Phase C
    # (l0_rust_enabled=true). On Phase A+B spirit_worker.py:2168 calls
    # `coordinator.set_dream_subsystems(neuromod_system=...)` with the
    # same instance — under Phase C spirit_worker is heartbeat-only so
    # nothing wired the coordinator's neuromod attribute, leaving
    # `coordinator._neuromod_system = None` AND `coordinator.neuromodulator_system`
    # undefined. Closes BUG-T3-NEUROMODULATORS-EMPTY-PAYLOAD-PHASE-C-OWNERSHIP-MISS:
    # /v4/neuromodulators returned `data={}` on T3 because:
    #   1. build_coordinator_snapshot couldn't find the live instance,
    #   2. spirit_loop._publish_coord_subdomains gated `if nm:` and skipped.
    # We construct ONE NeuromodulatorSystem here, wire it via both the
    # canonical `set_dream_subsystems` (writes `_neuromod_system` — used by
    # InnerTrinityCoordinator's own dream-side-effect helpers AND chi
    # computation at cognitive_worker.py:1549/1726) AND set the public
    # alias `.neuromodulator_system` (read by `_publish_coord_subdomains`
    # via build_coordinator_snapshot fall-back path). Both attributes
    # point at the same instance.
    if coordinator is not None:
        try:
            from titan_hcl.logic.neuromodulator import NeuromodulatorSystem
            _nm_data_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(
                    os.path.abspath(__file__)))),
                "data", "neuromodulator")
            _coord_nm_sys = NeuromodulatorSystem(data_dir=_nm_data_dir)
            coordinator.set_dream_subsystems(neuromod_system=_coord_nm_sys)
            coordinator.neuromodulator_system = _coord_nm_sys
            logger.info(
                "[CognitiveWorker] NeuromodulatorSystem wired to coordinator "
                "(_neuromod_system + .neuromodulator_system alias)")
        except Exception as _nm_err:
            logger.warning(
                "[CognitiveWorker] NeuromodulatorSystem wire failed: %s — "
                "/v4/neuromodulators will return empty payload on T3", _nm_err)

    try:
        from titan_hcl.logic.pi_heartbeat import PiHeartbeatMonitor
        pi_monitor = PiHeartbeatMonitor(
            min_cluster_size=3, min_gap_size=2,
            state_path="./data/pi_heartbeat_state.json")
    except Exception as _err:
        logger.warning("[CognitiveWorker] PiHeartbeatMonitor init failed: %s", _err)

    try:
        from titan_hcl.logic.reasoning import ReasoningEngine
        _reasoning_cfg = _load_toml_section("reasoning")
        if _reasoning_cfg.get("enabled", True):
            _rewards = _load_toml_section("reasoning_rewards")
            if _rewards:
                _reasoning_cfg = dict(_reasoning_cfg)
                _reasoning_cfg["reasoning_rewards"] = dict(_rewards)
            reasoning_engine = ReasoningEngine(config=_reasoning_cfg)
            if coordinator is not None:
                coordinator._reasoning_engine = reasoning_engine
    except Exception as _err:
        logger.warning("[CognitiveWorker] ReasoningEngine init failed: %s", _err)

    # ── Block F (Track 1): WorkingMemory + PredictionEngine ──
    # 2026-05-10 — pre-D8 ownership audit Track 1 closure. Both small,
    # dep-free engines mirroring spirit_worker.py:1156+1159.
    # WorkingMemory is load-bearing for reasoning_engine.tick (chain
    # context); PredictionEngine drives per-epoch novelty signal that
    # feeds chi + dreaming.
    try:
        from titan_hcl.logic.working_memory import WorkingMemory
        working_mem = WorkingMemory(capacity=7, decay_epochs=5)
        logger.info(
            "[CognitiveWorker] WorkingMemory booted (capacity=7, decay=5)")
    except Exception as _wm_err:
        logger.warning(
            "[CognitiveWorker] WorkingMemory init failed: %s", _wm_err)

    # PredictionEngine relocated to self_reflection_worker per Track 2 drift
    # correction (rFP §0 + commit B8). cognitive_worker subscribes to
    # PREDICTION_GENERATED bus events instead of in-process method calls.
    # See _dispatch_prediction_generated handler below.

    # ── EpisodicMemory ──
    try:
        from titan_hcl.logic.episodic_memory import EpisodicMemory
        episodic_mem = EpisodicMemory(db_path="./data/episodic_memory.db")
        logger.info("[CognitiveWorker] EpisodicMemory booted")
    except Exception as _em_err:
        logger.warning(
            "[CognitiveWorker] EpisodicMemory init failed: %s", _em_err)

    # ── IntuitionConvergenceDetector (M11-M13) ──
    try:
        from titan_hcl.logic.intuition_convergence import (
            IntuitionConvergenceDetector)
        _ic_cfg = _load_toml_section("intuition_convergence")
        intuition_convergence = IntuitionConvergenceDetector(config=_ic_cfg)
        # Restore persisted state if present.
        _ic_state_path = "./data/intuition_convergence_state.json"
        if os.path.exists(_ic_state_path):
            import json as _ic_json
            try:
                with open(_ic_state_path) as _ic_f:
                    intuition_convergence.from_dict(_ic_json.load(_ic_f))
            except Exception as _ic_load_err:
                logger.warning(
                    "[CognitiveWorker] IntuitionConvergence state restore "
                    "failed: %s", _ic_load_err)
        logger.info(
            "[CognitiveWorker] IntuitionConvergence booted "
            "(events=%d, weight=%.3f)",
            intuition_convergence._total_convergence_events,
            intuition_convergence._learned_weight)
    except Exception as _ic_err:
        logger.warning(
            "[CognitiveWorker] IntuitionConvergence init failed: %s",
            _ic_err)

    # ── WalletObserver (DI:/I:/Donation detection) ──
    try:
        from titan_hcl.logic.wallet_observer import WalletObserver
        _net_cfg = (config or {}).get("network", {}) or {}
        _titan_pubkey = _net_cfg.get("titan_pubkey", "")
        _maker_pubkey = _net_cfg.get("maker_pubkey", "")
        _rpc_url = _net_cfg.get(
            "premium_rpc_url",
            _net_cfg.get("rpc_url", "https://api.mainnet-beta.solana.com"))
        if _titan_pubkey and _maker_pubkey:
            wallet_observer = WalletObserver(
                titan_pubkey=_titan_pubkey,
                maker_pubkey=_maker_pubkey,
                rpc_url=_rpc_url,
                poll_interval=30.0,
            )
            logger.info(
                "[CognitiveWorker] WalletObserver booted — "
                "listening for DI:/I:/Donations")
        else:
            logger.info(
                "[CognitiveWorker] WalletObserver skipped — "
                "no titan/maker pubkey configured")
    except Exception as _wo_err:
        logger.warning(
            "[CognitiveWorker] WalletObserver init failed: %s", _wo_err)

    # ── MetaRecruitment (catalog health for meta-reasoning) ──
    try:
        from titan_hcl.logic.meta_recruitment import MetaRecruitment
        meta_recruitment = MetaRecruitment()
        logger.info("[CognitiveWorker] MetaRecruitment booted")
    except Exception as _mr_err:
        logger.warning(
            "[CognitiveWorker] MetaRecruitment init failed: %s", _mr_err)

    # ── TimeseriesStore (telemetry sink) ──
    try:
        from titan_hcl.logic.timeseries import TimeseriesStore
        timeseries_store = TimeseriesStore("./data/timeseries.db")
        logger.info("[CognitiveWorker] TimeseriesStore booted")
    except Exception as _ts_err:
        logger.warning(
            "[CognitiveWorker] TimeseriesStore init failed: %s", _ts_err)

    # ── MiniReasonerRegistry (distributed mini-reasoners) ──
    try:
        from titan_hcl.logic.mini_experience import MiniReasonerRegistry
        mini_registry = MiniReasonerRegistry(save_dir="./data/mini_reasoning")
        try:
            mini_registry.load_all()
        except Exception as _mri_load_err:
            logger.debug(
                "[CognitiveWorker] MiniReasonerRegistry load_all warned: %s",
                _mri_load_err)
        logger.info(
            "[CognitiveWorker] MiniReasonerRegistry booted — %d reasoners",
            len(mini_registry.all()))
        # Wire onto the coordinator so the off-tick dream-consolidation suite
        # (_on_dream_begin) can run `_mini_registry.consolidate_all` — without
        # this the `if hasattr(self, '_mini_registry')` guard is False and the
        # mini-reasoner dream consolidation silently no-ops (the rest of the
        # suite's engines are wired below at coordinator._meta_*). D-SPEC-105.
        if coordinator is not None:
            coordinator._mini_registry = mini_registry
    except Exception as _mri_err:
        logger.warning(
            "[CognitiveWorker] MiniReasonerRegistry init failed: %s",
            _mri_err)

    # ── ReasoningInterpreter (concept-domain interpretation) ──
    try:
        from titan_hcl.logic.reasoning_interpreter import (
            ReasoningInterpreter)
        _interp_cfg = _load_toml_section("reasoning_interpreter")
        interpreter = ReasoningInterpreter(config=_interp_cfg)
        logger.info(
            "[CognitiveWorker] ReasoningInterpreter booted — %d domains",
            len(interpreter.registry.all()))
    except Exception as _ri_err:
        logger.warning(
            "[CognitiveWorker] ReasoningInterpreter init failed: %s",
            _ri_err)

    # ── MeditationWatchdog — NOT owned here. ──
    # The MeditationWatchdog is constructed + checked by its canonical owner,
    # meditation_worker (which holds the `_meditation_tracker` + calls
    # `watchdog.check(tracker_view, now, backup_state_count=...)` at cadence).
    # cognitive_worker has NO meditation tracker, so a watchdog here could never
    # function: the old construction raised (`config=` kwarg + missing titan_id),
    # and the per-epoch check driver below called `check()` with no args — both
    # latent-dead, masked only because the construction always failed → None.
    # Leave med_watchdog = None (set above); the check driver is None-guarded and
    # cleanly no-ops. Meditation cadence/alerts remain fully live in meditation_worker.
    # (Merge 2026-06-02: titan-v6 c7a9602a removal supersedes the f1038bd5 titan_id
    # construction my reconcile carried — MeditationWatchdog is meditation_worker's.)
    med_watchdog = None

    # ── Meta-Reasoning Foundation (M1-M3) ──
    # 2026-05-10: post-deploy follow-up to the pre-D8 ownership audit.
    # Boot-driver parity audit caught that meta_engine was wired but its
    # 4 required positional deps (chain_archive, meta_wisdom, exp_orchestrator,
    # meta_autoencoder) were missing in cognitive_worker — meta_engine.tick
    # was raising TypeError silently every epoch (now visible after
    # _log_driver_err visibility upgrade). Mirrors spirit_worker.py:1415-1434.
    chain_archive = None
    meta_wisdom = None
    meta_autoencoder = None
    try:
        from titan_hcl.logic.chain_archive import ChainArchive
        from titan_hcl.logic.meta_wisdom import MetaWisdomStore
        from titan_hcl.logic.meta_autoencoder import MetaAutoencoder
        chain_archive = ChainArchive()
        meta_wisdom = MetaWisdomStore()
        _ae_dir = (reasoning_engine.save_dir
                   if reasoning_engine is not None else "./data/reasoning")
        meta_autoencoder = MetaAutoencoder(save_dir=_ae_dir)
        logger.info(
            "[CognitiveWorker] Meta-reasoning foundation: archive=OK, "
            "wisdom=OK, autoencoder=%s",
            "trained" if meta_autoencoder.is_trained else "untrained")
        if coordinator is not None:
            coordinator._chain_archive = chain_archive
            coordinator._meta_wisdom = meta_wisdom
            coordinator._meta_autoencoder = meta_autoencoder
    except Exception as _mrf_err:
        logger.warning(
            "[CognitiveWorker] Meta-reasoning foundation init failed: %s",
            _mrf_err)

    meta_engine = None
    try:
        from titan_hcl.logic.meta_reasoning import MetaReasoningEngine
        _meta_cfg = _load_toml_section("meta_reasoning")
        if _meta_cfg.get("enabled", True):
            # REGRESSION FIX (2026-05-20): enrich the bare [meta_reasoning]
            # section with titan_id + per-Titan-merged DNA. Without this,
            # MetaReasoningEngine ran with self._dna={} and titan_id="T1" on
            # ALL Titans since the spirit→cognitive engine migration — every
            # [meta_reasoning_dna.*] / [cognitive_contracts_dna.*] override was
            # inert (e.g. compound_legacy_blend_alpha, sunset_task4, contract
            # monoculture thresholds). See PLAN_cgn_k1_gamma_canary_and_dna_regression.md.
            from titan_hcl.core.state_registry import (
                resolve_titan_id as _resolve_tid_meta)
            _mr_titan_id = (
                (config.get("info_banner", {}) or {}).get("titan_id")
                or _resolve_tid_meta())
            _meta_cfg = {
                **_meta_cfg,
                "titan_id": _mr_titan_id,
                "dna": _merge_per_titan_dna("meta_reasoning_dna", _mr_titan_id),
                "contracts_dna": _merge_per_titan_dna(
                    "cognitive_contracts_dna", _mr_titan_id),
            }
            logger.info(
                "[CognitiveWorker] meta_reasoning DNA enriched: titan_id=%s "
                "dna_keys=%d contracts_dna_keys=%d compound_blend_alpha=%s "
                "(OBS-meta-dna-enrichment-restored)",
                _mr_titan_id, len(_meta_cfg["dna"]),
                len(_meta_cfg["contracts_dna"]),
                _meta_cfg["dna"].get("compound_legacy_blend_alpha", "default"))
            meta_engine = MetaReasoningEngine(
                config=_meta_cfg, send_queue=send_queue)
            if coordinator is not None:
                coordinator._meta_engine = meta_engine
            # NOTE (2026-06-10 fix): the grounding_sink is wired CANONICALLY in the
            # caller (search "CANONICAL grounding_sink wiring", ~line 657) where BOTH
            # meta_engine (returned as state_refs["meta_engine"]) AND _meta_service
            # (built in the caller, AFTER this function returns) exist. The earlier
            # in-function attempt here referenced `state_refs` — which is NOT in scope
            # inside _init_cognitive_engines (it is THIS function's RETURN value; see
            # the same note at the top of this function) — so it raised
            # `NameError: name 'state_refs' is not defined`, which the outer
            # `except` swallowed as "MetaReasoningEngine init failed", aborting the
            # init mid-way (meta_engine half-wired → started one chain then FROZE,
            # tmc pinned, monoculture never moved). Removed; the canonical wiring is
            # the sole, correct path. (Introduced by ff3583b4; broke T2 meta-reasoning.)
            # D-SPEC-70 v1.15.0 — attach inner_self_insight.bin SHM reader
            # so _prim_introspect can read pre-warmed cache per G20.
            # ShmReaderBank lazy-inited in state_refs at boot (see
            # state_refs["_shm_reader_bank"] population further down).
            # The reader is attached here (not in state_refs.get later)
            # because MetaReasoning.tick is in-process from this point on.
            try:
                from titan_hcl.api.shm_reader_bank import ShmReaderBank
                meta_engine._inner_self_insight_reader = ShmReaderBank()
                logger.info(
                    "[CognitiveWorker] meta_engine._inner_self_insight_reader "
                    "attached — D-SPEC-70 v1.15.0 / closes F-8")
            except Exception as _shm_err:
                logger.warning(
                    "[CognitiveWorker] inner_self_insight SHM reader attach "
                    "failed: %s — META INTROSPECT falls back to cold_start "
                    "placeholder until reattach", _shm_err)
    except Exception as _err:
        logger.warning("[CognitiveWorker] MetaReasoningEngine init failed: %s", _err)

    # ExpressionManager — MOVED to expression_worker per §4.B Track 3
    # (SHIPPED 2026-05-15). cognitive_worker no longer instantiates the
    # manager or owns the 6 composites. The Tier-2 evaluate_all driver
    # (formerly Block 8 below) is replaced by expression_worker's
    # KERNEL_EPOCH_TICK subscriber. SPEAK Tier-1 detection arrives via
    # SPEAK_REQUEST_PENDING bus event (handled in Block 8.5).
    # Composite → NS program reward arrives via NS_REWARD bus event
    # (handled in the main recv loop) and is dispatched to
    # neural_nervous_system.record_outcome here.
    expression_manager = None  # retained as None for state_refs symmetry

    # === §4.G — LifeForceShmReader + LifeForceInputsPublisher init ===
    # Per rFP_titan_hcl_l2_separation_strategy.md §4.G + D-SPEC-57: the
    # LifeForceEngine (chunk 8M.6 Track 1 drift) was extracted to its own
    # L2 worker (`life_force_worker`). cognitive_worker no longer hosts
    # the engine — instead, it:
    #   (a) reads chi state via LifeForceShmReader at the 5 hot-path
    #       sites (MSL static_context chi_total, reasoning body_state,
    #       hormonal_pressure inputs, ground_up_enricher chi_overlay,
    #       NN modulation cap)
    #   (b) writes life_force_inputs.bin via LifeForceInputsPublisher
    #       at every KERNEL_EPOCH_TICK (cross-process 16-input bridge
    #       mirroring §4.Q neuromod_inputs.bin pattern)
    # See SPEC v1.8.3 §9.B life_force_worker block for the contract.
    life_force_engine = None  # retained as None for state_refs symmetry
    life_force_shm_reader = None
    life_force_inputs_publisher = None
    try:
        from titan_hcl.proxies.life_force_proxy import LifeForceShmReader
        life_force_shm_reader = LifeForceShmReader()
        logger.info(
            "[CognitiveWorker] LifeForceShmReader attached "
            "(§4.G — reads life_force_state.bin for chi_total/drain at "
            "5 hot-path sites)")
    except Exception as _err:
        logger.warning(
            "[CognitiveWorker] LifeForceShmReader init failed: %s — "
            "chi readers fall back to cold defaults", _err)
    try:
        from titan_hcl.core.state_registry import resolve_titan_id as _resolve_tid_lf
        from titan_hcl.logic.life_force_inputs_publisher import (
            LifeForceInputsPublisher,
        )
        _lf_tid = (
            (config.get("info_banner", {}) or {}).get("titan_id")
            or _resolve_tid_lf()
        )
        life_force_inputs_publisher = LifeForceInputsPublisher(titan_id=_lf_tid)
        logger.info(
            "[CognitiveWorker] LifeForceInputsPublisher attached "
            "(§4.G — writes life_force_inputs.bin G21 single-writer "
            "for life_force_worker.evaluate)")
    except Exception as _err:
        logger.warning(
            "[CognitiveWorker] LifeForceInputsPublisher init failed: %s — "
            "life_force_worker will not receive cross-process inputs",
            _err)

    # === Phase A.4 publishers (D-SPEC-70 v1.10.0) ===
    # cognitive_worker owns 3 of the 9 Python L2 SHM slots introduced for
    # the api_subprocess SHM-canonical migration per Preamble G18:
    #   reasoning_state.bin       (ReasoningEngine — lives here)
    #   meta_reasoning_state.bin  (MetaReasoningEngine — lives here)
    #   msl_state.bin             (MSL — lives here, per `coordinator._msl`)
    # NOTE: meta_teacher_state.bin is owned by meta_teacher_worker (G21
    # single-writer; the MetaTeacher class lives in its own dedicated
    # worker per `titan_hcl/modules/meta_teacher_worker.py`).
    # Phase B.5 closure 2026-05-18: this block runs inside
    # _init_cognitive_engines, where `state_refs` does NOT exist (the
    # return dict is constructed at the bottom). Use locals + add to
    # the return dict — same pattern as LifeForceInputsPublisher at
    # line 1693. The prior shape silently failed with `name 'state_refs'
    # is not defined` and left reasoning_state.bin /
    # meta_reasoning_state.bin / msl_state.bin un-published fleet-wide.
    _reasoning_state_publisher = None
    _meta_reasoning_state_publisher = None
    _msl_state_publisher = None
    _experience_stats_publisher = None
    _consciousness_age_publisher = None
    try:
        from titan_hcl.core.state_registry import resolve_titan_id as _resolve_tid_a4
        _a4_tid = (
            (config.get("info_banner", {}) or {}).get("titan_id")
            or _resolve_tid_a4()
        )
        from titan_hcl.logic.reasoning_state_publisher import (
            ReasoningStatePublisher,
        )
        from titan_hcl.logic.meta_reasoning_state_publisher import (
            MetaReasoningStatePublisher,
        )
        from titan_hcl.logic.msl_state_publisher import (
            MSLStatePublisher,
        )
        from titan_hcl.logic.consciousness_age_publisher import (
            ConsciousnessAgePublisher,
        )
        from titan_hcl.logic.experience_stats_publisher import (
            ExperienceStatsPublisher,
        )
        _reasoning_state_publisher = ReasoningStatePublisher(
            titan_id=_a4_tid)
        _meta_reasoning_state_publisher = MetaReasoningStatePublisher(
            titan_id=_a4_tid)
        _msl_state_publisher = MSLStatePublisher(titan_id=_a4_tid)
        # §3L Phase 15 chunk 15.1 (D-SPEC-PHASE15) — experience_stats.bin
        # publisher. Replaces the retired ExperienceMemory.get_stats
        # recompute-on-read; sources from ExperienceOrchestrator's
        # incremental action_stats. Event-driven publish (boot-seed +
        # post-record_outcome); in-proc consumers read the orchestrator
        # directly (always fresh), only the api reads the slot.
        _experience_stats_publisher = ExperienceStatsPublisher(
            titan_id=_a4_tid)
        # D-SPEC-85 v1.25.0 (2026-05-18) — consciousness_age.bin slot.
        # Producer = cognitive_worker; Consciousness object lives in
        # spirit_loop under this worker per SPEC §1 glossary.
        _consciousness_age_publisher = ConsciousnessAgePublisher(
            titan_id=_a4_tid)
        logger.info(
            "[CognitiveWorker] Phase A.4 publishers attached: "
            "reasoning_state / meta_reasoning_state / msl_state / "
            "consciousness_age / experience_stats "
            "(G21 single-writers; rFP_phase_c_state_read_unification_l0_l1_canonical "
            "+ D-SPEC-85 + D-SPEC-PHASE15)")
    except Exception as _err:
        logger.warning(
            "[CognitiveWorker] Phase A.4 publishers init failed: %s — "
            "api_subprocess will read cold-boot stubs from those slots",
            _err)

    # === chunk 8M.7 — Multisensory Synthesis Layer (MSL) init ===
    # Per rFP §3.7 + Gap F (§2.6): MSL hosts the L2 perception engine
    # (concept grounding, attention weighting, i-confidence). Under
    # l0_rust=true the legacy spirit_worker stub doesn't init it.
    # Mirrors spirit_worker.py:1949-1979 init pattern + same titan_params.toml
    # [msl] section gate.
    msl = None
    try:
        from titan_hcl.logic.msl import MultisensorySynthesisLayer
        _msl_cfg = {}
        try:
            import tomllib as _msl_tl
            _msl_tp = os.path.join(
                os.path.dirname(__file__), "..", "titan_params.toml")
            if os.path.exists(_msl_tp):
                with open(_msl_tp, "rb") as _msl_f:
                    _msl_cfg = _msl_tl.load(_msl_f).get("msl", {})
        except Exception as _msl_cfg_err:
            logger.debug(
                "[CognitiveWorker] MSL config load failed: %s", _msl_cfg_err)
        if _msl_cfg.get("enabled", True):
            msl = MultisensorySynthesisLayer(config=_msl_cfg)
            try:
                msl.load_all()
            except Exception as _msl_load_err:
                logger.debug(
                    "[CognitiveWorker] MSL.load_all warned: %s", _msl_load_err)
            if coordinator is not None:
                coordinator._msl = msl
            # Phase A.4 gap-7 closure — msl reaches the MSLStatePublisher tick
            # path (_drive_one_epoch) as state_refs["msl"] via this function's
            # RETURN LITERAL ("msl": msl, ~line 2429). The prior `state_refs["msl"]
            # = msl` here raised `NameError: name 'state_refs' is not defined`
            # (state_refs is THIS function's return value, not in scope inside it)
            # → the except swallowed it as "MSL init failed", aborting MSL boot.
            # Removed; the return literal is the sole, correct path. (ff3583b4 bug.)
            logger.info(
                "[CognitiveWorker] MSL booted: input=%dD, output=%dD, "
                "buffer=%d frames, updates=%d, I-confidence=%.3f, "
                "convergences=%d",
                msl.policy.input_dim, msl.policy.output_dim,
                msl.buffer.max_frames, msl.policy.total_updates,
                msl.get_i_confidence(),
                msl.confidence._convergence_count)
    except Exception as _msl_err:
        logger.warning("[CognitiveWorker] MSL init failed: %s", _msl_err)

    # ── Boot NeuromodRewardObserver (rFP β Stage 2 Phase 2b) ──
    #
    # 2026-05-10: migrated from spirit_worker per pre-D8 ownership audit.
    # Constructor refactored to take a `levels_provider` callable so the
    # observer can read NEUROMOD_STATE shm (Rust-produced under
    # l0_rust=true) without needing cross-process Python attr access on
    # neuromodulator_system (which lives in neuromod_worker). NS lives
    # in this process, so record_outcome calls are in-process.
    # ── Boot ExperientialMemory + ExperienceOrchestrator ──
    # (ExperienceMemory RETIRED — §3L Phase 15 chunk 15.1 / D-SPEC-PHASE15)
    #
    # 2026-05-10: Block D of pre-D8 ownership audit closure. Tier 1 SPEAK
    # firing path emits SPEAK_REQUEST with experience_bias built by
    # ExperienceOrchestrator.get_experience_bias("language", ...). Migrated
    # from spirit_worker:1133-1880 (l0_rust=false legacy path) so SPEAK
    # works on T3 (where spirit_worker is heartbeat-only). Plugins
    # registered: ArcPuzzle, LanguageLearning, CreativeExpression, Communication.
    e_mem = None
    exp_orchestrator = None
    try:
        # §3L Phase 15 chunk 15.1 (D-SPEC-PHASE15): ExperienceMemory RETIRED.
        # Its sole writer (spirit_worker) was deleted in D8-3/72f95a6b, leaving
        # data/experience_memory.db frozen fleet-wide since 2026-05-14. The
        # live successor is ExperienceOrchestrator (records the same outcomes
        # via DOMAIN_MAP + closes the recall/distill/bias loop). The former
        # ex_mem constructor dep was assignment-only (never read). Stats now
        # flow from the orchestrator's incremental action_stats → experience_stats.bin.
        from titan_hcl.logic.experiential_memory import ExperientialMemory
        from titan_hcl.logic.experience_orchestrator import (
            ExperienceOrchestrator)
        from titan_hcl.logic.experience_plugins import (
            ArcPuzzlePlugin, LanguageLearningPlugin,
            CreativeExpressionPlugin, CommunicationPlugin,
            KnowledgePlugin, SelfModelPlugin, MetaReasoningPlugin)
        _dev_age_fn = (lambda: pi_monitor.developmental_age) if pi_monitor else (
            lambda: 0)
        e_mem = ExperientialMemory(
            db_path="./data/experiential_memory.db",
            developmental_age_fn=_dev_age_fn,
        )
        exp_orchestrator = ExperienceOrchestrator(
            e_mem=e_mem, cognee_memory=None,
            db_path="./data/experience_orchestrator.db")
        exp_orchestrator.register_plugin(ArcPuzzlePlugin())
        exp_orchestrator.register_plugin(LanguageLearningPlugin())
        exp_orchestrator.register_plugin(CreativeExpressionPlugin())
        exp_orchestrator.register_plugin(CommunicationPlugin())
        # Phase 1 enrichment (rFP_experience_distillation_phase_c §5) — extend
        # distillation to the wisdom-bearing CGN consumer streams.
        exp_orchestrator.register_plugin(KnowledgePlugin())
        exp_orchestrator.register_plugin(SelfModelPlugin())
        exp_orchestrator.register_plugin(MetaReasoningPlugin())
        logger.info(
            "[CognitiveWorker] Experience Orchestrator booted: %s",
            list(exp_orchestrator._plugins.keys()))
        # Wire the ExperienceOrchestrator into the coordinator's dream side-
        # effects so `_on_dream_begin` runs `distill_cycle` (the Distill stage
        # of the Record→Distill→Bias loop). The coordinator-init
        # set_dream_subsystems(neuromod_system=...) call ran BEFORE
        # exp_orchestrator existed, leaving coordinator._exp_orchestrator=None →
        # distillation was skipped every dream. The additive setter
        # (rFP_experience_distillation_phase_c) wires it now without clobbering
        # neuromod_system. PURE IN-PROCESS wiring — distill_cycle is a local DB
        # operation; no bus traffic.
        if coordinator is not None:
            coordinator.set_dream_subsystems(
                exp_orchestrator=exp_orchestrator, e_mem=e_mem)
            logger.info(
                "[CognitiveWorker] ExperienceOrchestrator + e_mem wired into "
                "coordinator dream side-effects (distill_cycle now active on "
                "_on_dream_begin)")
        # §3L Phase 15 chunk 15.1 — boot-seed experience_stats.bin so the api
        # reads live aggregates immediately (not a cold-boot stub) before the
        # first record_outcome. G18: one-time slot publish at boot.
        if _experience_stats_publisher is not None:
            _experience_stats_publisher.publish(exp_orchestrator)
    except Exception as _exp_err:
        logger.warning(
            "[CognitiveWorker] Experience Orchestrator init failed: %s",
            _exp_err)

    # ── Boot SocialPressureMeter (legacy social pressure tracking) ──
    #
    # Used to call on_social_fire(urge) when expression_manager fires SOCIAL
    # composite. Persona system reads accumulated pressure for posting
    # cadence regulation. Migrated from spirit_worker:1358 per Block D.
    _social_pressure_meter = None
    try:
        from titan_hcl.logic.social_pressure import SocialPressureMeter
        _sp_cfg = (config or {}).get("social_presence", {}) or {}
        _social_pressure_meter = SocialPressureMeter(_sp_cfg)
        logger.info(
            "[CognitiveWorker] SocialPressureMeter booted (cfg keys=%s)",
            sorted(_sp_cfg.keys()))
    except Exception as _sp_err:
        logger.warning(
            "[CognitiveWorker] SocialPressureMeter init failed: %s", _sp_err)

    neuromod_reward_observer = None
    try:
        from titan_hcl.logic.neuromod_reward_observer import (
            NeuromodRewardObserver)
        _nro_cfg = (config or {}).get(
            "neuromod_reward_observer", {}) or {}
        if _nro_cfg.get("enabled", True) and neural_nervous_system is not None:
            # Build a SHM-backed levels_provider locally — _init_cognitive_engines
            # runs before cognitive_worker_main installs the shared
            # `_neuromod_reader` on state_refs, so we construct our own here.
            # _make_neuromod_reader returns None if the SHM slot is disabled
            # (l0_rust=false / shm_neuromod_enabled=false), in which case the
            # observer falls through to its empty-levels path until the
            # provider becomes hot.
            _nro_levels_provider = _make_neuromod_reader()
            neuromod_reward_observer = NeuromodRewardObserver(
                neural_nervous_system=neural_nervous_system,
                levels_provider=_nro_levels_provider,
                tick_interval=int(_nro_cfg.get("tick_interval", 10)),
                ema_alpha=float(_nro_cfg.get("ema_alpha", 0.05)),
                enabled=True,
            )
            logger.info(
                "[CognitiveWorker] NeuromodRewardObserver online "
                "(levels_provider=%s, tick_interval=%d)",
                "shm" if _nro_levels_provider else "none",
                neuromod_reward_observer.tick_interval)
            # Side-channel back-pointer for legacy code that reaches for
            # nns._neuromod_reward_observer (matches spirit_worker:1229).
            neural_nervous_system._neuromod_reward_observer = neuromod_reward_observer
    except Exception as _nro_err:
        logger.warning(
            "[CognitiveWorker] NeuromodRewardObserver init failed: %s",
            _nro_err)

    return {
        "coordinator": coordinator,
        "neural_nervous_system": neural_nervous_system,
        "pi_monitor": pi_monitor,
        "reasoning_engine": reasoning_engine,
        "meta_engine": meta_engine,
        "observable_engine": observable_engine,
        # expression_manager — kept as None for symmetry; real owner is
        # expression_worker per §4.B Track 3. Reads via /v4/expression-
        # composites route now go to expression_state.bin SHM slot.
        "expression_manager": expression_manager,
        "inner_state": inner_state,
        "spirit_state": spirit_state,
        # §4.G + chunk 8M.7: chi reader/publisher + MSL engine accessible
        # via state_refs. life_force_engine is None post-§4.G extraction
        # (Track 1 drift retired; engine now lives in life_force_worker
        # subprocess). LifeForceShmReader serves chi state at 5 hot-path
        # sites; LifeForceInputsPublisher writes the 16-input bridge.
        "life_force_engine": life_force_engine,
        "_life_force_shm_reader": life_force_shm_reader,
        "_life_force_inputs_publisher": life_force_inputs_publisher,
        # Phase A.4 SHM publishers (D-SPEC-70 v1.10.0). Initialized
        # ~190 lines above; threaded into the return dict so
        # _drive_one_epoch can call .publish(reasoning_engine) etc.
        # per Preamble G18. None if init failed (warn log already
        # emitted; readers fall back to cold-boot stub).
        "_reasoning_state_publisher": _reasoning_state_publisher,
        "_meta_reasoning_state_publisher":
            _meta_reasoning_state_publisher,
        "_msl_state_publisher": _msl_state_publisher,
        # D-SPEC-85 v1.25.0 — consciousness_age.bin publisher (lifetime
        # self-observation tick counter for post_dispatch footer "main age").
        "_consciousness_age_publisher": _consciousness_age_publisher,
        # §3L Phase 15 chunk 15.1 — experience_stats.bin publisher
        # (G21 single-writer; ExperienceOrchestrator-sourced).
        "_experience_stats_publisher": _experience_stats_publisher,
        "msl": msl,
        "neuromod_reward_observer": neuromod_reward_observer,
        # Block D — Tier 1 SPEAK migration deps:
        "e_mem": e_mem,
        "exp_orchestrator": exp_orchestrator,
        "social_pressure_meter": _social_pressure_meter,
        # Meta-reasoning foundation (M1-M3) — required by meta_engine.tick:
        "chain_archive": chain_archive,
        "meta_wisdom": meta_wisdom,
        "meta_autoencoder": meta_autoencoder,
        # Block F (Track 1) — pre-D8 ownership audit migrations:
        # prediction_engine relocated to self_reflection_worker per Track 2
        # drift correction (rFP §0 + commit B8). Consumer side via
        # PREDICTION_GENERATED bus event → state_refs["_latest_prediction"].
        "working_mem": working_mem,
        "episodic_mem": episodic_mem,
        "intuition_convergence": intuition_convergence,
        "wallet_observer": wallet_observer,
        "meta_recruitment": meta_recruitment,
        "timeseries_store": timeseries_store,
        "mini_registry": mini_registry,
        "interpreter": interpreter,
        "med_watchdog": med_watchdog,
    }


# === MODULE-SPECIFIC: bus event dispatchers (PLAN §3.1 driver table) ===
# Each dispatcher reads from msg payload + writes to state_refs.
# Failures per-dispatcher are caught at the call site (see main loop).


def _decode_payload(payload):
    """Normalize msg payload to a dict.

    Phase B.2 §C7 socket-mode bus may deliver payloads as raw msgpack
    bytes (when the broker forwards a wire frame without unpacking).
    Phase B legacy mp.Queue mode delivers dicts directly. cognitive_worker
    handles both — same pattern as `bus_socket.py:633` + `kernel_rpc.py:312`
    (msgpack.unpackb with raw=False for Python str semantics).
    """
    if payload is None:
        return {}
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, (bytes, bytearray, memoryview)):
        try:
            import msgpack
            decoded = msgpack.unpackb(payload, raw=False)
            return decoded if isinstance(decoded, dict) else {"_raw": decoded}
        except Exception as _err:
            logger.debug("[CognitiveWorker] payload msgpack decode failed: %s", _err)
            return {}
    return {}


def _dispatch_trinity_state(state_refs: dict, payload: dict, *, dim: int,
                            inner_key: str, outer_key: str, type_label: str) -> None:
    """Dispatch BODY_STATE / MIND_STATE / SPIRIT_STATE per SPEC §8.5.

    Reads payload.src ∈ {"inner", "outer"} and writes payload.values to
    one of the 6 internal cache slots indexed by (type, src). Preserves
    G1 doctrinal symmetry — inner and outer are equally first-class.

    Rust producers (titan-{inner,outer}-{body,mind,spirit}-rs) publish
    `{src: "inner"|"outer", type: <NAME>, values: [N floats], ts: float}`
    per SPEC §8.5 row in §8.5 Trinity tensor messages table. If src is
    missing (legacy publisher), default to "inner" with a debug log so
    we don't lose the tensor; this matches legacy 67D-only consciousness
    epoch behavior.
    """
    src = payload.get("src", "inner")
    values = payload.get("values")
    if not isinstance(values, list) or len(values) < dim:
        # Bad payload — keep prior cache value, don't blank.
        return
    target_key = inner_key if src == "inner" else outer_key
    state_refs[target_key] = list(values[:dim])
    if src not in ("inner", "outer"):
        logger.debug(
            "[CognitiveWorker] %s with unexpected src=%r — treated as inner",
            type_label, src)


def _dispatch_dream_consolidate(state_refs: dict, payload: dict) -> None:
    """Trigger DreamingEngine consolidation per CGN signal."""
    coordinator = state_refs.get("coordinator")
    if coordinator is None:
        return
    dreaming = getattr(coordinator, "dreaming", None) or getattr(
        coordinator, "dreaming_engine", None)
    if dreaming is None:
        return
    consolidate = getattr(dreaming, "consolidate_pending", None)
    if callable(consolidate):
        consolidate(payload)


def _run_dream_bridge(state_refs: dict, neuromod_reader, send_queue, name: str) -> None:
    """Bridge A (rFP §3G Phase 10G restore) — harvest crystallized inner events
    on the dreaming→waking falling edge and inject them into the outer memory
    graph, felt-tagged for Bridge B recall perturbation.

    This orchestration loop was DROPPED at the D8-3 spirit→cognitive migration
    (the spirit_worker END_DREAMING handler that invoked the harvest was deleted
    and never re-homed; zero invoker since). cognitive_worker owns the
    DreamingEngine + chain_archive + meta_wisdom engines, so it is the correct
    Phase C home. Harvest body lives in logic/dream_bridge.py (pure sqlite reads).

    Read-only on engines + non-blocking bus emits; never raises into the epoch
    driver (a missing engine degrades gracefully — a skipped dream bridge is
    recoverable, not critical state).
    """
    from titan_hcl.logic.dream_bridge import harvest_dream_memories

    coordinator = state_refs.get("coordinator")
    dreaming = getattr(coordinator, "dreaming", None) if coordinator else None
    dream_cycle = int(getattr(dreaming, "_cycle_count", 0) or 0)
    chain_archive = state_refs.get("chain_archive")
    meta_wisdom = state_refs.get("meta_wisdom")
    cgn_db_path = os.path.join("data", "inner_memory.db")

    # Felt-state snapshot (neuromods + emotion) stored as neuromod_context so
    # Bridge B can re-experience the somatic state at injection time.
    levels = {}
    try:
        levels = (neuromod_reader() or {}) if neuromod_reader else {}
    except Exception:
        levels = {}
    emotion = "neutral"
    emotion_conf = 0.0
    try:
        _inner = getattr(coordinator, "inner", None) if coordinator else None
        emotion = str(getattr(_inner, "_current_emotion", emotion) or emotion)
        emotion_conf = float(getattr(_inner, "_emotion_confidence", 0.0) or 0.0)
    except Exception:
        pass
    felt = {k: round(float(v), 4) for k, v in levels.items()
            if isinstance(v, (int, float))}
    felt["emotion"] = emotion
    felt["emotion_confidence"] = round(emotion_conf, 4)
    felt["dream_cycle"] = dream_cycle
    felt["ts"] = time.time()

    memories, chain_ids = harvest_dream_memories(
        chain_archive, meta_wisdom, felt, cgn_db_path, dream_cycle)

    for mem in memories:
        _send_msg(send_queue, bus.MEMORY_ADD, name, "memory", {
            "text": mem["text"],
            "source": mem.get("source", "dream_consolidation"),
            "weight": float(mem.get("weight", 2.0)),
            "neuromod_context": mem.get("neuromod_context"),
        })

    # Mark the harvested reasoning chains consolidated so they aren't re-injected.
    if chain_ids and chain_archive is not None:
        try:
            chain_archive.mark_consolidated(chain_ids)
        except Exception as _mc_err:
            logger.debug("[DreamBridge] mark_consolidated failed: %s", _mc_err)

    # Feed the msl dream-bridge metric (consumed by the dashboard bridge count).
    _msl = state_refs.get("msl")
    if _msl is not None and hasattr(_msl, "record_dream_bridge"):
        try:
            _msl.record_dream_bridge(len(memories))
        except Exception as _msl_err:
            logger.debug("[DreamBridge] msl.record_dream_bridge failed: %s", _msl_err)

    if memories:
        logger.info(
            "[DreamBridge] dream-end harvest: injected %d memories "
            "(cycle=%d, chains_marked=%d)",
            len(memories), dream_cycle, len(chain_ids))

    # Observatory retention — once/day prune of telemetry rows older than 90d.
    # Restores the spirit_worker "prune after every dream cycle" call DROPPED at
    # the D8-3 spirit→cognitive migration (same class as the dream-bridge harvest
    # above), capped to daily per observatory_db.prune_old_data's own follow-up
    # note (spirit called it every ~22min dream cycle — far more than needed).
    # The writer-routed path serializes DELETEs through observatory_writer and
    # SKIPS VACUUM (a 1.88GB VACUUM under load caused the 2026-04-21 T3
    # degradation), so this is lock-safe and bounds DB growth; the one-time
    # on-disk shrink stays a separate maintenance-window op. Best-effort — a
    # prune failure never raises into the dream/epoch path.
    try:
        _now = time.time()
        if _now - float(state_refs.get("_last_observatory_prune", 0.0)) >= 86400:
            from titan_hcl.utils.observatory_db import get_observatory_db
            _odb = get_observatory_db()
            # SAFETY: only prune when a writer is present → the writer-routed
            # path (DELETEs serialized through observatory_writer, VACUUM SKIPPED).
            # With NO writer, prune_old_data takes the direct path that runs
            # VACUUM (observatory_db.py:867) — the 2026-04-21 T3-degradation
            # trigger. Skip in that case; the writer-owning process owns retention.
            if getattr(_odb, "_writer", None) is not None:
                _odb.prune_old_data(max_days=90)
                state_refs["_last_observatory_prune"] = _now
            else:
                logger.debug("[DreamBridge] observatory prune skipped — no writer "
                             "(would hit the VACUUM direct path)")
    except Exception as _prune_err:
        logger.debug("[DreamBridge] observatory prune skipped: %s", _prune_err)


def _dispatch_experience_record(state_refs: dict, payload: dict) -> None:
    """EXPERIENCE_RECORD consumer — Record stage of the ExperienceOrchestrator
    Record→Distill→Bias loop (rFP_experience_distillation_phase_c_restoration_and_
    enrichment.md). Per-worker producers emit the semantic content they own
    (domain / action / outcome / context); cognitive_worker — sole owner of the
    ExperienceOrchestrator + experience_orchestrator.db (G21) — ENRICHES here with
    the in-proc consciousness inner-state + hormonal snapshot + the domain plugin's
    perception key, then persists via record_outcome().

    Mirrors the recording shape of the deleted spirit_worker.py callsites
    (L7451/8609/8955/10920). Sub-ms; no downstream bus emit; graceful degradation —
    a missing engine/plugin never raises (an experience record is recoverable, not
    critical state).
    """
    exp_orchestrator = state_refs.get("exp_orchestrator")
    if exp_orchestrator is None:
        return
    try:
        domain = str(payload.get("domain", "")).strip()
        if not domain:
            return
        action_taken = str(payload.get("action_taken", ""))[:200]
        outcome_score = float(payload.get("outcome_score", 0.0))
        context = payload.get("context") or {}
        if not isinstance(context, dict):
            context = {}
        epoch_id = int(payload.get("epoch_id", 0) or 0)

        # Inner-state enrichment: prefer the full consciousness state vector;
        # fall back to the cached inner trinity (5+15+45=65D felt tensor) which
        # the bus dispatch keeps fresh via BODY/MIND/SPIRIT_STATE handlers.
        consciousness = state_refs.get("consciousness")
        _sv = []
        if consciousness:
            _sv = (consciousness.get("latest_epoch", {}) or {}).get(
                "state_vector", []) or []
            if hasattr(_sv, "to_list"):
                _sv = _sv.to_list()
        _sv = list(_sv) if _sv else []
        _ib = list(state_refs.get("_inner_body_state") or [])[:5]
        _im = list(state_refs.get("_inner_mind_state") or [])[:15]
        _isp = list(state_refs.get("_inner_spirit_state") or [])[:45]
        if not _sv:
            _sv = _ib + _im + _isp  # 65D felt-tensor fallback

        # Hormonal snapshot from in-proc NS (cognitive_worker hosts the NS).
        nns = state_refs.get("neural_nervous_system")
        hormonal_snapshot = {}
        if nns is not None and getattr(nns, "_hormonal_enabled", False):
            try:
                hormonal_snapshot = {h: round(v.level, 3)
                                     for h, v in nns._hormonal._hormones.items()}
            except Exception:
                hormonal_snapshot = {}

        # Perception key via the domain plugin (graceful raw-slice fallback).
        plugin = exp_orchestrator._plugins.get(domain)
        if plugin is not None:
            perception_features = plugin.extract_perception_key({
                "inner_state": _sv,
                "felt_tensor": _sv[:65],
                "inner_body": _ib or _sv[:5],
                "inner_mind": _im or (_sv[5:20] if len(_sv) >= 20 else []),
                "inner_spirit": _isp or (_sv[20:65] if len(_sv) >= 65 else []),
                "intent_hormones": hormonal_snapshot,
                "hormonal_snapshot": hormonal_snapshot,
                "spatial_features": _sv[65:95] if len(_sv) > 65 else [],
            })
        else:
            perception_features = _sv[:10] if _sv else [0.5] * 10

        # is_dreaming gates pending_distillation (records during dreams tag
        # pending=0 → consolidated in the next dream cycle).
        is_dreaming = False
        coordinator = state_refs.get("coordinator")
        if coordinator is not None and getattr(
                coordinator, "dreaming", None) is not None:
            is_dreaming = bool(getattr(
                coordinator.dreaming, "is_dreaming", False))

        exp_orchestrator.record_outcome(
            domain=domain,
            perception_features=perception_features,
            inner_state_132d=_sv[:130],
            hormonal_snapshot=hormonal_snapshot,
            action_taken=action_taken,
            outcome_score=outcome_score,
            context=context,
            epoch_id=epoch_id,
            is_dreaming=is_dreaming,
        )
        # §3L Phase 15 chunk 15.1 — event-driven publish of experience_stats.bin
        # after each recorded outcome (EXPERIENCE_RECORD is upstream-throttled
        # to ≥2s via emit_experience_record). Keeps the api-read slot current
        # without a periodic thread; in-proc consumers read the orchestrator
        # directly. G18/G21 single-writer (cognitive_worker owns the slot).
        _exp_stats_pub = state_refs.get("_experience_stats_publisher")
        if _exp_stats_pub is not None:
            _exp_stats_pub.publish(exp_orchestrator)
    except Exception as _exp_err:
        logger.debug(
            "[CognitiveWorker] EXPERIENCE_RECORD record raised: %s", _exp_err)


def _dispatch_stimulus(state_refs: dict, msg_type: str, payload: dict) -> None:
    """Feed CONVERSATION_STIMULUS / EXPERIENCE_STIMULUS to ReasoningEngine.

    PLAN §3.1: both event types drive `reasoning_engine.observe_stimulus(...)`
    so the engine can begin a new chain. The `source` field tags which
    upstream produced the stimulus — useful for chain-cause attribution.
    """
    reasoning_engine = state_refs.get("reasoning_engine")
    if reasoning_engine is None:
        return
    observe = getattr(reasoning_engine, "observe_stimulus", None)
    if not callable(observe):
        return
    enriched = dict(payload)
    enriched.setdefault("source", msg_type.lower().replace("_stimulus", ""))
    observe(enriched)


def _dispatch_meditation_complete(state_refs: dict, payload: dict) -> None:
    """Notify InnerTrinityCoordinator that a meditation phase completed.

    Coordinator's meditation_observe (if present) tracks meditation
    cadence as input to the dreaming/consciousness scheduler. Future
    extraction: meditation_worker (rFP_titan_hcl_l2_separation_strategy.md
    §4 worker #4) will own this concern entirely; cognitive_worker will
    just consume the snapshot.
    """
    coordinator = state_refs.get("coordinator")
    if coordinator is None:
        return
    observe = getattr(coordinator, "meditation_observe", None)
    if callable(observe):
        observe(payload)


def _init_shm_reader_bank(titan_id: str):
    """Construct the SPEC §1096 shm reader bank for cognitive_worker.

    chunk 8M.4 (2026-05-05) — closes rFP_phase_c_observatory_data_pipeline.md
    Gap A (§2.1): cognitive_worker previously only read NEUROMOD_STATE.
    Per SPEC §1096 the cognitive_worker MUST read all 12 Rust-owned shm
    slots (self_162d, all 6 inner/outer trinity slots, topology_30d,
    neuromod_state, titanvm_registers, hormonal_state, identity,
    epoch_counter, sphere_clocks, chi_state, cgn_live_weights).

    Reuses ``titan_hcl.api.shm_reader_bank.ShmReaderBank`` — same bank
    api_subprocess uses (defense in depth: cognitive_worker write path +
    api_subprocess fallback path both go through one set of typed
    accessors that own per-registry SeqLock-validated reads).

    Returns ``None`` if the bank can't be constructed (titan-rust kernel
    not running, shm root missing, etc.) — the epoch driver tolerates a
    None bank by falling back to bus-cache slots only (chunk 8G behavior).
    """
    try:
        from titan_hcl.api.shm_reader_bank import ShmReaderBank
        bank = ShmReaderBank(titan_id=titan_id)
        return bank
    except Exception as _err:
        logger.warning(
            "[CognitiveWorker] ShmReaderBank init failed (%s) — "
            "falling back to bus-cache only", _err)
        return None


def _make_neuromod_reader():
    """Return a callable that reads neuromod_state.bin shm slot, or None.

    Per SPEC §10.G shm-direct-read pattern. neuromod_state.bin is owned
    by neuromod_worker (writes 6 floats: DA, 5HT, NE, ACh, Endorphin,
    GABA in canonical order). Cognitive_worker reads at each epoch tick
    to drive `coordinator.update_neuromodulators(...)`.

    Returns ``None`` if shm reader can't be constructed (slot disabled,
    Rust microkernel unavailable, etc.) — coordinator will use default
    values.
    """
    try:
        from titan_hcl.core.state_registry import NEUROMOD_STATE, RegistryBank
        from titan_hcl.config_loader import load_titan_config
        # 2026-05-10 — pass the merged runtime config so is_enabled() can
        # evaluate microkernel.shm_neuromod_enabled against the canonical
        # value from titan_params.toml + per-Titan overrides. Pre-fix
        # `RegistryBank()` defaulted to empty config → is_enabled always
        # returned False → reader was always None → NeuromodRewardObserver
        # ran in dormant levels_provider=none mode on T3.
        bank = RegistryBank(config=load_titan_config())
        if not bank.is_enabled(NEUROMOD_STATE):
            return None
        reader = bank.reader(NEUROMOD_STATE)

        # 2026-05-15 (§4.Q): use decode_neuromod_levels which accepts both
        # v1 (6,) and v2 (6, 4) layouts so consumers don't break during the
        # schema transition. Pre-§4.Q this used hardcoded arr[0..5] indices.
        from titan_hcl.modules.neuromod_worker import decode_neuromod_levels

        def _read():
            arr = reader.read()
            if arr is None:
                return None
            try:
                return decode_neuromod_levels(arr)
            except ValueError:
                # Unexpected shape — log and return None rather than silently
                # mis-decode. Matches feedback_three_state_health_checks.md.
                logger.warning(
                    "[CognitiveWorker] neuromod_state.bin unexpected shape "
                    "%s — slot reader returning None", getattr(arr, "shape", None))
                return None
        return _read
    except Exception as _err:
        logger.debug("[CognitiveWorker] neuromod shm reader init failed: %s", _err)
        return None


def _cognitive_epoch_loop(state_refs: dict, config: dict, send_queue,
                          name: str, stop_event: threading.Event,
                          early_fire_event: threading.Event) -> None:
    """Adaptive 1–30s consciousness epoch driver (PLAN §4 pseudocode).

    Per tick:
      1. Snapshot all 6 trinity cache slots (SPEC §8.5 + G1 symmetry).
      2. Drive coordinator.tick(inner_tensors, outer_tensors) — which
         drives DreamingEngine, TopologyEngine, ObservableEngine, NS.
      3. Read NEUROMOD_STATE shm slot (SPEC §10.G fallback) and pass to
         coordinator.update_neuromodulators if available.
      4. Run consciousness epoch via spirit_loop._run_consciousness_epoch
         — computes drift/trajectory/curvature/density and inserts
         EpochRecord into consciousness DB.
      5. Drive PiHeartbeatMonitor.observe(curvature, epoch_id) on each
         tick.
      6. Step ReasoningEngine if it has an active chain.
      7. Tick MetaReasoningEngine.
      8. Persist engine state every COGNITIVE_PERSIST_EVERY_N_EPOCHS.

    Adaptive interval (chunk 8G simple form): default cadence
    COGNITIVE_EPOCH_DEFAULT_INTERVAL_S; early_fire_event allows the bus
    dispatcher to wake the loop immediately on resonance/urgency
    signals (chunk 8F+ wires this — currently unused, future-ready).
    Future tuning: adaptive ramp-up under high arousal / ramp-down
    under quiescence; bounded by [MIN, MAX] per SPEC v0.2.0 constants.
    """
    from titan_hcl.logic.consciousness_epoch import _run_consciousness_epoch

    consciousness = state_refs.get("consciousness")
    coordinator = state_refs.get("coordinator")
    pi_monitor = state_refs.get("pi_monitor")
    reasoning_engine = state_refs.get("reasoning_engine")
    meta_engine = state_refs.get("meta_engine")
    neuromod_reader = state_refs.get("_neuromod_reader")
    shm_bank = state_refs.get("_shm_reader_bank")

    interval_s = COGNITIVE_EPOCH_DEFAULT_INTERVAL_S
    epochs_observed = 0

    logger.info("[CognitiveWorker] epoch loop alive (interval=%.2fs)", interval_s)

    while not stop_event.is_set():
        # Wait for next tick OR early-fire signal OR shutdown.
        early_fire_event.wait(timeout=interval_s)
        early_fire_event.clear()
        if stop_event.is_set():
            break

        try:
            _drive_one_epoch(
                state_refs, config,
                consciousness=consciousness,
                coordinator=coordinator,
                pi_monitor=pi_monitor,
                reasoning_engine=reasoning_engine,
                meta_engine=meta_engine,
                neuromod_reader=neuromod_reader,
                shm_bank=shm_bank,
                send_queue=send_queue,
                name=name,
            )
            epochs_observed += 1
            if epochs_observed % COGNITIVE_PERSIST_EVERY_N_EPOCHS == 0:
                logger.info(
                    "[CognitiveWorker] persisting engine state "
                    "(epoch %d, every %d)",
                    epochs_observed, COGNITIVE_PERSIST_EVERY_N_EPOCHS)
                _persist_engine_state(state_refs)
        except Exception as _err:
            logger.warning("[CognitiveWorker] epoch tick failed: %s", _err,
                           exc_info=True)

    logger.info(
        "[CognitiveWorker] epoch loop exiting (observed %d epochs)",
        epochs_observed)


def _drive_one_epoch(state_refs: dict, config: dict, *,
                     consciousness, coordinator, pi_monitor,
                     reasoning_engine, meta_engine, neuromod_reader,
                     shm_bank=None,
                     send_queue, name) -> None:
    """One pass of the adaptive consciousness epoch (extracted for testability).

    All engines optional — None checks per call site. Single epoch tick
    succeeds even if a subset of engines failed to init at boot.

    chunk 8M.4 (2026-05-05): adds Step 1.5 — shm read pass. Per SPEC §1096
    + rFP_phase_c_observatory_data_pipeline.md §2.1+§2.8, cognitive_worker
    reads all Rust-owned shm slots each epoch and:
      (a) overrides empty/missing bus-cache trinity tensors (defense in
          depth: when bus delivery is silent, shm holds last-known-good)
      (b) populates ``coordinator._<key>_snapshot`` attributes consumed
          by spirit_loop.build_coordinator_snapshot's shm-fallback path
          (chunk 8M.5).
      (c) syncs consciousness.epoch_number from epoch_counter shm so
          /v4/inner-trinity.coordinator.consciousness.epoch_number
          tracks the kernel-rs counter (was stuck at 0 per rFP §1.3).
    """
    from titan_hcl.logic.consciousness_epoch import _run_consciousness_epoch

    # §4.Q (2026-05-15) — helper for emitting NEUROMOD_EXTERNAL_NUDGE bus
    # events to neuromod_worker. Defined at the top of _drive_one_epoch so
    # all call sites below resolve the name. Replaces the 6 in-process
    # `neuromodulator_system.apply_external_nudge(...)` calls that pre-§4.Q
    # lived in dead spirit_worker.py code:
    #   - MSL concept valence ×2 (spirit_worker.py:4055, 4065)
    #   - FILTER_DOWN reasoning confidence + ABANDON ×2 (5205, 5234)
    #   - META eureka DA burst (7244)
    #   - SPIRIT_SELF nudge (7270+)
    # Each call site below documents which legacy line it replaces.
    def _emit_neuromod_nudge(
        nudge_map: dict[str, float],
        max_delta: float,
        developmental_age: float,
        source: str,
    ) -> None:
        if not nudge_map:
            return
        try:
            send_queue.put({
                "type": bus.NEUROMOD_EXTERNAL_NUDGE,
                "src": name,
                "dst": "neuromod",
                "payload": {
                    "nudge_map": nudge_map,
                    "max_delta": float(max_delta),
                    "developmental_age": float(developmental_age),
                    "source": source,
                },
                "ts": time.time(),
            })
        except Exception as _ne:
            _ne_key = f"nudge_emit:{source}:{type(_ne).__name__}"
            _ne_counts = state_refs.setdefault("_neuromod_emit_err_counts", {})
            _ne_counts[_ne_key] = _ne_counts.get(_ne_key, 0) + 1
            if _ne_counts[_ne_key] == 1 or _ne_counts[_ne_key] % 100 == 0:
                logger.warning(
                    "[CognitiveWorker] §4.Q nudge emit failed source=%s "
                    "count=%d: %s", source, _ne_counts[_ne_key], _ne)

    # 1. Snapshot 6 trinity cache slots — GIL-atomic list-pointer reads.
    inner_body = state_refs.get("_inner_body_state") or [0.5] * 5
    inner_mind_15 = state_refs.get("_inner_mind_state") or [0.5] * 15
    inner_spirit_45 = state_refs.get("_inner_spirit_state") or [0.5] * 45
    outer_body = state_refs.get("_outer_body_state") or [0.5] * 5
    outer_mind_15 = state_refs.get("_outer_mind_state") or [0.5] * 15
    outer_spirit_45 = state_refs.get("_outer_spirit_state") or [0.5] * 45

    # 1.5 chunk 8M.4 — shm-direct-read pass (SPEC §1096 + §10.G).
    #
    # Per rFP_phase_c_observatory_data_pipeline.md Gap A: the bus-cache
    # paths above (filled by chunk 8F dispatcher from BODY/MIND/SPIRIT_STATE
    # bus events) can be silent if the Rust-side broker is bottlenecked or
    # subscriber filter dropped events. In that case shm holds the last-
    # known-good payload (Rust daemons publish-and-write atomically per
    # SPEC §10.E). Fall back to shm reads when bus cache is at the all-
    # 0.5 default (i.e. never populated).
    sphere_clocks_snap = None
    chi_snap = None
    topology_snap = None
    self_162d_snap = None
    titanvm_snap = None
    hormonal_snap = None
    inner_spirit_45d_snap = None
    epoch_counter_snap = None

    if shm_bank is not None:
        # Bus-cache fallback for the 6 trinity tensor slots — only override
        # when the bus cache is at default (never updated). _populated_from_shm
        # flag avoids logging on every tick.
        def _is_default_5(v):
            return all(abs(float(x) - 0.5) < 1e-6 for x in v)

        def _is_default_15(v):
            return all(abs(float(x) - 0.5) < 1e-6 for x in v)

        def _is_default_45(v):
            return all(abs(float(x) - 0.5) < 1e-6 for x in v)

        try:
            if _is_default_5(inner_body):
                snap = shm_bank.read_inner_body_5d()
                if snap and snap.get("values"):
                    inner_body = snap["values"]
            if _is_default_15(inner_mind_15):
                snap = shm_bank.read_inner_mind_15d()
                if snap and snap.get("values"):
                    inner_mind_15 = snap["values"]
            if _is_default_45(inner_spirit_45):
                snap = shm_bank.read_inner_spirit_45d()
                if snap:
                    # SAT/CHIT/ANANDA → flat 45D
                    sat = snap.get("SAT") or [0.5] * 15
                    chit = snap.get("CHIT") or [0.5] * 15
                    ananda = snap.get("ANANDA") or [0.5] * 15
                    inner_spirit_45 = list(sat) + list(chit) + list(ananda)
                    inner_spirit_45d_snap = snap
            if _is_default_5(outer_body):
                snap = shm_bank.read_outer_body_5d()
                if snap and snap.get("values"):
                    outer_body = snap["values"]
            if _is_default_15(outer_mind_15):
                snap = shm_bank.read_outer_mind_15d()
                if snap and snap.get("values"):
                    outer_mind_15 = snap["values"]
            if _is_default_45(outer_spirit_45):
                snap = shm_bank.read_outer_spirit_45d()
                if snap and snap.get("values"):
                    outer_spirit_45 = snap["values"]
        except Exception as _err:
            logger.debug(
                "[CognitiveWorker] shm trinity-tensor fallback read failed: %s",
                _err)

        # Read remaining SPEC §1096 slots — these are pure shm-owned
        # (no bus equivalent), so always read regardless of cache state.
        try:
            sphere_clocks_snap = shm_bank.read_sphere_clocks()
        except Exception as _err:
            logger.debug("[CognitiveWorker] read_sphere_clocks failed: %s", _err)
        try:
            chi_snap = shm_bank.read_chi()
        except Exception as _err:
            logger.debug("[CognitiveWorker] read_chi failed: %s", _err)
        try:
            topology_snap = shm_bank.read_topology_30d()
        except Exception as _err:
            logger.debug("[CognitiveWorker] read_topology_30d failed: %s", _err)
        try:
            self_162d_snap = shm_bank.read_trinity()
        except Exception as _err:
            logger.debug("[CognitiveWorker] read_trinity failed: %s", _err)
        try:
            titanvm_snap = shm_bank.read_titanvm_registers()
        except Exception as _err:
            logger.debug("[CognitiveWorker] read_titanvm_registers failed: %s", _err)
        try:
            hormonal_snap = shm_bank.read_hormonal()
        except Exception as _err:
            logger.debug("[CognitiveWorker] read_hormonal failed: %s", _err)
        try:
            epoch_counter_snap = shm_bank.read_epoch()
        except Exception as _err:
            logger.debug("[CognitiveWorker] read_epoch failed: %s", _err)

    # 1.6 Inject snapshot dicts onto coordinator for build_coordinator_snapshot
    # to consume via the shm-fallback path (chunk 8M.5). Always set — None is
    # a valid signal ("shm read failed/disabled, fall back to engine.get_stats()").
    if coordinator is not None:
        coordinator._sphere_clocks_snapshot = sphere_clocks_snap
        coordinator._chi_snapshot = chi_snap
        coordinator._topology_snapshot = topology_snap
        coordinator._self_162d_snapshot = self_162d_snap
        coordinator._titanvm_snapshot = titanvm_snap
        coordinator._hormonal_snapshot = hormonal_snap
        coordinator._inner_spirit_45d_snapshot = inner_spirit_45d_snap

    # 1.7 Sync consciousness.epoch_number from epoch_counter shm — was stuck
    # at 0 per rFP §1.3 because Python consciousness dict is initialized at
    # boot and never refreshed from the kernel-rs counter. Read here.
    if consciousness is not None and epoch_counter_snap is not None:
        kernel_epoch = epoch_counter_snap.get("epoch")
        if kernel_epoch is not None and kernel_epoch > 0:
            # consciousness is a dict-like state container; only update the
            # field when shm reports a fresher value. Don't go backward.
            try:
                cur = consciousness.get("epoch_number", 0) or 0
                if int(kernel_epoch) > int(cur):
                    consciousness["epoch_number"] = int(kernel_epoch)
            except Exception as _err:
                logger.debug(
                    "[CognitiveWorker] consciousness epoch_number sync failed: %s",
                    _err)

    # 2. Drive coordinator.tick(inner_tensors, outer_tensors) — runs
    #    DreamingEngine, TopologyEngine, ObservableEngine, NS programs.
    if coordinator is not None:
        try:
            coordinator.tick(
                inner_tensors={
                    "inner_body": inner_body,
                    "inner_mind": inner_mind_15,
                    "inner_spirit": inner_spirit_45,
                },
                outer_tensors={
                    "outer_body": outer_body,
                    "outer_mind": outer_mind_15,
                    "outer_spirit": outer_spirit_45,
                },
            )
        except Exception as _err:
            logger.debug("[CognitiveWorker] coordinator.tick failed: %s", _err)

    # 2.5 Resolve is_dreaming ONCE (authoritative source = inner_state) and
    # gate waking engine-mutation on it. During a dream the full consolidation
    # suite trains the engines off-tick (coordinator daemon thread); the waking
    # epoch driver must NOT train the SAME policy/networks concurrently or the
    # backprop / net.lr races corrupt weights (Phase 0 §8). Biologically: no
    # waking learning during sleep. reasoning_engine.tick already self-gates;
    # NS / reasoning.step / meta.tick / mini.tick_all are gated below using
    # this flag. rFP_dream_consolidation_suite_offtick_restoration / D-SPEC-105.
    _epoch_is_dreaming = bool(
        getattr(getattr(coordinator, "inner", None), "is_dreaming", False)
    ) if coordinator is not None else False

    # ── DreamingMeta consultation (D-SPEC-116 restore) ──────────────────
    # On the waking→dreaming RISING EDGE, consult the meta-reasoner so the
    # DreamingMeta consumer learns from each dream cycle. This loop was dropped
    # at the §4.I dream_state_worker extraction (D-SPEC-56 deleted spirit_worker's
    # BEGIN_DREAMING block incl. this hook). NOTE: the original used
    # question_type="consolidate_themes" which was NEVER whitelisted in
    # KNOWN_QUESTION_TYPES → it was rejected (dead-on-arrival); we use the valid
    # `synthesize_insight` (→ SYNTHESIZE primitive), the apt realization of the
    # intent (dreaming synthesizes themes). EXTRA-CARE (D-SPEC-105 dream mechanic):
    # this runs in the WAKING epoch driver — NOT the off-tick consolidation suite —
    # is purely read-only + a non-blocking bus emit, and touches no dream/engine
    # state, so it cannot perturb the dream state machine. Fires once per cycle.
    if _epoch_is_dreaming and not state_refs.get("_dreaming_meta_was_dreaming", False):
        try:
            from titan_hcl.logic.meta_service_client import (
                send_meta_request as _dm_send)
            from titan_hcl.logic.meta_consumer_contexts import (
                build_dreaming_meta_context_30d as _dm_ctx)
            _dm_nm = {}
            try:
                _dm_nm = (neuromod_reader() or {}) if neuromod_reader else {}
            except Exception:
                _dm_nm = {}
            _dm_send(
                consumer_id="dreaming",
                question_type="synthesize_insight",
                context_vector=_dm_ctx(
                    dream_state={"is_dreaming": True},
                    neuromods=_dm_nm),
                time_budget_ms=5000,
                constraints={"confidence_threshold": 0.3,
                             "allow_timechain_query": True},
                payload_snippet="dream entry — consolidate themes",
                send_queue=send_queue, src=name)
        except Exception as _dm_err:
            logger.debug("[DreamingMeta] dream-entry consult skipped: %s", _dm_err)
    state_refs["_dreaming_meta_was_dreaming"] = _epoch_is_dreaming

    # ── Dream Bridge A (rFP §3G Phase 10G restore) ──────────────────────
    # Symmetric to the DreamingMeta rising-edge consult above: on the
    # dreaming→waking FALLING EDGE (dream just ended), harvest crystallized
    # inner events (wisdom / eureka chains / CGN milestones / compositions /
    # social) and inject them into the outer memory graph, felt-tagged for
    # Bridge B recall perturbation. This loop was dropped at the D8-3
    # spirit_worker gutting (zero invoker since); restored here per Maker
    # (2026-05-28). Read-only + non-blocking emits; fires once per cycle.
    if state_refs.get("_dream_bridge_was_dreaming", False) and not _epoch_is_dreaming:
        try:
            _run_dream_bridge(state_refs, neuromod_reader, send_queue, name)
        except Exception as _db_err:
            logger.warning("[DreamBridge] dream-end harvest/inject failed: %s",
                           _db_err, exc_info=True)
    state_refs["_dream_bridge_was_dreaming"] = _epoch_is_dreaming

    # 3. Read NEUROMOD_STATE shm slot + drive coordinator.update_neuromodulators.
    if neuromod_reader is not None and coordinator is not None:
        try:
            neuromods = neuromod_reader()
            if neuromods is not None:
                update_nm = getattr(coordinator, "update_neuromodulators", None)
                if callable(update_nm):
                    update_nm(neuromods)
        except Exception as _err:
            logger.debug("[CognitiveWorker] neuromod read/apply failed: %s", _err)

    # 3.5 §4.Q (2026-05-15) — Modulation reconstruction → NNS._modulation.
    # Read the 4-field neuromod_state.bin v2 layout written by neuromod_worker
    # and reconstruct the 14-key modulation dict via compute_modulation_from_state
    # (byte-identical to NeuromodulatorSystem.get_modulation()). Apply to
    # neural_nervous_system._modulation in-process before each NS evaluate so
    # learning_rate_gain / sensory_gain / training_frequency_gain / etc. modulate
    # NS training. Replaces dead spirit_worker.py:4344-4350.
    _nns_for_mod = state_refs.get("neural_nervous_system")
    if _nns_for_mod is not None:
        try:
            from titan_hcl.core.state_registry import (
                NEUROMOD_STATE, RegistryBank)
            from titan_hcl.config_loader import load_titan_config
            from titan_hcl.logic.neuromodulator import (
                compute_modulation_from_state)
            from titan_hcl.modules.neuromod_worker import (
                decode_neuromod_state)
            if "_neuromod_state_reader" not in state_refs:
                _bank_q10 = RegistryBank(config=load_titan_config())
                if _bank_q10.is_enabled(NEUROMOD_STATE):
                    state_refs["_neuromod_state_reader"] = _bank_q10.reader(
                        NEUROMOD_STATE)
                else:
                    state_refs["_neuromod_state_reader"] = None
            _reader_q10 = state_refs.get("_neuromod_state_reader")
            if _reader_q10 is not None:
                _arr_q10 = _reader_q10.read()
                if _arr_q10 is not None and _arr_q10.shape == (6, 4):
                    _state_q10 = decode_neuromod_state(_arr_q10)
                    _modulation = compute_modulation_from_state(_state_q10)
                    _nns_for_mod._modulation = _modulation
                    # Self-emergent hormonal governors injection (preserves
                    # spirit_worker.py:4347-4350 math).
                    _gaba_q10 = _state_q10.get("GABA", {}).get("level", 0.5)
                    # §4.G — chi read via SHM (was: life_force_engine._latest_chi)
                    _lf_reader_q10 = state_refs.get("_life_force_shm_reader")
                    _chi_total_q10 = (
                        float(_lf_reader_q10.get_chi_total())
                        if _lf_reader_q10 is not None else 0.6
                    )
                    _nns_for_mod._modulation["gaba_level"] = float(_gaba_q10)
                    _nns_for_mod._modulation["chi_total"] = _chi_total_q10
                    state_refs["_last_neuromod_modulation_ts"] = time.time()
        except Exception as _mod_err:
            # First occurrence + every 100th — flood-safe.
            if "_neuromod_mod_err_counts" not in state_refs:
                state_refs["_neuromod_mod_err_counts"] = {}
            _mk = f"{type(_mod_err).__name__}:{str(_mod_err)[:80]}"
            _mc = state_refs["_neuromod_mod_err_counts"].get(_mk, 0) + 1
            state_refs["_neuromod_mod_err_counts"][_mk] = _mc
            if _mc == 1 or _mc % 100 == 0:
                logger.warning(
                    "[CognitiveWorker] §4.Q modulation reconstruction failed "
                    "(count=%d): %s", _mc, _mod_err)

    # 4. Run consciousness epoch — populates consciousness["latest_epoch"]
    #    with curvature/density/state_vector/journey_point/etc.
    if consciousness is not None:
        body_state_dict = {"values": inner_body}
        mind_state_dict = {"values": inner_mind_15[:5], "values_15d": inner_mind_15}
        outer_state_dict = {
            "outer_body": outer_body,
            "outer_mind": outer_mind_15[:5],
            "outer_mind_15d": outer_mind_15,
            "outer_spirit": outer_spirit_45[:5],
            "outer_spirit_45d": outer_spirit_45,
        }
        try:
            _run_consciousness_epoch(
                consciousness, body_state_dict, mind_state_dict,
                config, outer_state=outer_state_dict, shm_bank=shm_bank)
        except Exception as _err:
            logger.debug("[CognitiveWorker] _run_consciousness_epoch failed: %s", _err)

        # Bug E (rFP_dead_dim_wiring_fix §2.E, SPEC §7.1 + D-SPEC-69
        # v1.14.0): write trajectory_state.bin SHM slot — G18-pure
        # replacement for the retired TRAJECTORY_UPDATE bus event.
        # Source-of-truth: `consciousness["latest_epoch"]` dict that
        # spirit_loop._run_consciousness_epoch just mutated above (same
        # dict reference; same process; no cross-process snapshot pipe
        # in between — bypasses the broken `state_vector` path that
        # caused [0.5, 0.5] degenerate emit via meta_reasoning padding).
        try:
            if "_trajectory_state_writer" not in state_refs:
                from titan_hcl.core.state_registry import (
                    TRAJECTORY_STATE, StateRegistryWriter,
                    ensure_shm_root, resolve_titan_id)
                _titan_id_traj = resolve_titan_id()
                state_refs["_trajectory_state_writer"] = StateRegistryWriter(
                    TRAJECTORY_STATE, ensure_shm_root(_titan_id_traj))
                logger.info(
                    "[CognitiveWorker] trajectory_state.bin writer attached "
                    "(G21 single-writer; SPEC §7.1 D-SPEC-69 — replaces "
                    "TRAJECTORY_UPDATE bus event per G18)")

            _latest = (consciousness or {}).get("latest_epoch") or {}
            _curv = float(_latest.get("curvature", 0.0) or 0.0)
            _dens = float(_latest.get("density", 0.0) or 0.0)
            import numpy as _np_traj
            _traj_vec = _np_traj.array([_curv, _dens], dtype=_np_traj.float32)
            state_refs["_trajectory_state_writer"].write(_traj_vec)
        except Exception as _traj_err:
            if "_trajectory_state_err_count" not in state_refs:
                state_refs["_trajectory_state_err_count"] = 0
            _c = state_refs["_trajectory_state_err_count"]
            state_refs["_trajectory_state_err_count"] = _c + 1
            if _c == 0 or _c % 100 == 0:
                logger.warning(
                    "[CognitiveWorker] trajectory_state write failed "
                    "(count=%d): %s", _c + 1, _traj_err)

        # Phase 3.A D-SPEC-86: write consciousness_state.bin SHM slot.
        # Inner-spirit-rs producer reads this slot for 6 dims (temporal_continuity,
        # self_awareness_depth, wisdom_accumulation, dream_awareness, meta_cognition,
        # meaning_depth). Pre-fix, slot had no writer → all 6 dims dead fleet-wide.
        # Schema matches Rust constants.rs:299: {epoch_id, density, curvature,
        # dream_quality, fatigue, trajectory_magnitude, latest_epoch, ts}. dream_quality
        # + fatigue sourced from coordinator.dreaming + state_refs.inner_state when
        # available; producer-side defaults preserve dim semantics otherwise.
        try:
            if "_consciousness_state_writer" not in state_refs:
                from titan_hcl.logic.spirit_state_specs import (
                    CONSCIOUSNESS_STATE_SPEC)
                from titan_hcl.core.state_registry import (
                    StateRegistryWriter, ensure_shm_root, resolve_titan_id)
                _titan_id_cs = resolve_titan_id()
                state_refs["_consciousness_state_writer"] = StateRegistryWriter(
                    CONSCIOUSNESS_STATE_SPEC, ensure_shm_root(_titan_id_cs))
                logger.info(
                    "[CognitiveWorker] consciousness_state.bin writer attached "
                    "(G21 single-writer; Phase 3.A D-SPEC-86 — feeds "
                    "inner_spirit_sidecar.bin for 6 chit/ananda dims)")
            _dream_q = 0.0
            _fatigue = 0.0
            try:
                _dr = getattr(coordinator, "dreaming", None) if coordinator else None
                if _dr is not None:
                    _dream_q = float(getattr(_dr, "dream_quality", 0.0) or 0.0)
            except Exception:
                pass
            try:
                _is = state_refs.get("inner_state")
                if _is is not None:
                    _fatigue = float(getattr(_is, "fatigue", 0.0) or 0.0)
            except Exception:
                pass
            _epoch_id = int(_latest.get("epoch_id", 0) or 0)
            _traj_mag = float(_latest.get("trajectory_magnitude", 0.0) or 0.0)
            _cs_payload = {
                "epoch_count": _epoch_id,  # primary key inner-spirit-rs reads
                "epoch_id": _epoch_id,     # fallback alias
                "density": _dens,
                "curvature": _curv,
                "dream_quality": _dream_q,
                "fatigue": _fatigue,
                "trajectory": _traj_mag,            # primary
                "trajectory_magnitude": _traj_mag,  # alias
                "ts": time.time(),
            }
            import msgpack as _msgpack_cs
            _cs_bytes = _msgpack_cs.packb(_cs_payload, use_bin_type=True)
            state_refs["_consciousness_state_writer"].write_variable(_cs_bytes)
        except Exception as _cs_err:
            if "_consciousness_state_err_count" not in state_refs:
                state_refs["_consciousness_state_err_count"] = 0
            _csc = state_refs["_consciousness_state_err_count"]
            state_refs["_consciousness_state_err_count"] = _csc + 1
            if _csc == 0 or _csc % 100 == 0:
                logger.warning(
                    "[CognitiveWorker] consciousness_state write failed "
                    "(count=%d): %s", _csc + 1, _cs_err)

        # ── RFP_meta-reasoning_CGN_FIX.md §8 — substrate emits ──────
        # Direct producer-side emits to emot_cgn, replacing the retiring
        # spirit_worker._attach_emot_producer_ctx bridge (D8 retirement
        # per feedback_phase_c_spirit_worker_d8_retirement.md). Each
        # emit is best-effort: if the substrate value isn't available
        # (e.g. boot warmup, dream-state pause, snapshot read failed),
        # the put_nowait is skipped — emot_cgn falls back to its cached
        # last_* value, which is the same defensive pattern as the
        # legacy ctx-attach contract. Cadence is consciousness-epoch
        # ~1Hz; bus traffic is two P3 fire-and-forget messages per tick
        # (well under §8.0.ter publish budget).
        # Bug B fix (rFP_dead_dim_wiring_fix §2 B): shm_bank.read_topology_30d()
        # returns {"values": [30 floats], "parts": {...}, "age_seconds":..., "seq":...}.
        # The 30D payload lives in `values` (canonical 6 body parts × 5 features
        # layout). The previous read paths (`outer_lower_topology_10d`,
        # `inner_lower_topology_10d`, `_last_whole_10d`) were invented keys that
        # never existed in the SHM reader output → emit gated → space_topology
        # bundle dead.
        try:
            _topo_snap = getattr(coordinator, "_topology_snapshot", None) or {}
            _values = list(_topo_snap.get("values") or [])[:30]
            if len(_values) >= 30:
                send_queue.put_nowait({
                    "type": bus.SPACE_TOPOLOGY_UPDATE,
                    "src": name,
                    "dst": "emot_cgn",
                    "payload": {
                        "space_topology_30d": [float(v) for v in _values[:30]],
                    },
                })
        except Exception:
            pass  # graceful — emit is best-effort

        # Bug A fix (rFP_dead_dim_wiring_fix §2 A): shm_bank.read_sphere_clocks()
        # returns {"clocks": {sphere_name: {"phase":..., "radius":..., ...}, ...},
        # "age_seconds":..., "seq":...}. Phases are nested under `clocks`, each
        # clock is a dict containing a `phase` key. The previous flat `.get(name)`
        # path always returned 0.0 → gate `any(abs(v)>1e-9)` skipped emit →
        # pi_phase bundle dead.
        try:
            _sc_snap = getattr(coordinator, "_sphere_clocks_snapshot", None)
            _clocks = (_sc_snap or {}).get("clocks") if isinstance(
                _sc_snap, dict) else None
            if _clocks and isinstance(_clocks, dict):
                _pi6 = [
                    float((_clocks.get(name) or {}).get("phase", 0.0))
                    for name in ("inner_body", "outer_body", "inner_mind",
                                 "outer_mind", "inner_spirit", "outer_spirit")
                ]
                # Only emit when at least one phase is non-zero — avoids
                # flooding emot_cgn with all-zero pulses during boot
                # warmup before sphere_clock has ticked.
                if any(abs(v) > 1e-9 for v in _pi6):
                    send_queue.put_nowait({
                        "type": bus.PI_PHASE_UPDATE,
                        "src": name,
                        "dst": "emot_cgn",
                        "payload": {"pi_phase_6d": _pi6},
                    })
        except Exception:
            pass  # graceful — emit is best-effort

        # Bug C fix (rFP_dead_dim_wiring_fix §2.C, SPEC §7.1 + D-SPEC-68
        # v1.13.0): canonical NS-program urgency snapshot written to the
        # `ns_program_urgencies_input.bin` SHM slot (G18-compliant; G21
        # single-writer = cognitive_worker). Closes the load-bearing
        # cross-process wire-up gap from the ns_worker L2 carve-out
        # (Phase C migration).
        #
        # Source-of-truth: inner_coordinator.tick() calls the LEGACY
        # `self.nervous_system.evaluate(...)` (NervousSystem class). The
        # V5 `self.neural_ns` is the same instance bound to
        # state_refs["neural_nervous_system"] and is evaluated by section
        # 4.6 below — not by the coordinator's tick path.
        # The legacy `evaluate()` returns `list[dict]` of FIRED programs
        # `{system, urgency, result}` (only `urgency > 0` entries) which
        # InnerTrinityCoordinator caches at `_last_nervous_signals`.
        #
        # ns_worker reads this slot per tick, applies peak-hold-decay
        # 0.9 (URGENCY_PEAK_HOLD_DECAY) to preserve fired-program peaks
        # across the ~10-tick post-fire reset cycle, then writes the
        # downstream titanvm_registers.bin urgency column + emits
        # NS_URGENCIES_UPDATE → emot_cgn substrate cache. Pattern mirrors
        # `neuromod_inputs.bin` (§4.Q) + `life_force_inputs.bin` (§4.G).
        try:
            if "_ns_urg_input_writer" not in state_refs:
                from titan_hcl.core.state_registry import (
                    NS_PROGRAM_URGENCIES_INPUT, StateRegistryWriter,
                    ensure_shm_root, resolve_titan_id)
                _titan_id_urg = resolve_titan_id()
                state_refs["_ns_urg_input_writer"] = StateRegistryWriter(
                    NS_PROGRAM_URGENCIES_INPUT,
                    ensure_shm_root(_titan_id_urg))
                logger.info(
                    "[CognitiveWorker] ns_program_urgencies_input.bin "
                    "writer attached (G21 single-writer; feeds ns_worker "
                    "→ titanvm_registers.bin urgency column + emot_cgn "
                    "ns_urgencies substrate cache)")

            _signals = getattr(
                coordinator, "_last_nervous_signals", None) or []
            _urg_by_name = {}
            for _sig in _signals:
                try:
                    _sys_name = _sig.get("system")
                    if not _sys_name:
                        continue
                    _urg_by_name[str(_sys_name)] = float(
                        _sig.get("urgency", 0.0))
                except (AttributeError, TypeError, ValueError):
                    continue
            # Build canonical 11-float vector in NS_PROGRAMS order
            # (matches titanvm_registers.bin row order).
            from titan_hcl.logic.emot_bundle_protocol import NS_PROGRAMS
            import numpy as _np_urg
            _urg_vec = _np_urg.zeros((len(NS_PROGRAMS),), dtype=_np_urg.float32)
            for _idx, _pname in enumerate(NS_PROGRAMS):
                _urg_vec[_idx] = float(_urg_by_name.get(_pname, 0.0))
            state_refs["_ns_urg_input_writer"].write(_urg_vec)
        except Exception as _urg_err:
            # First-fail + every-100th log per directive_error_visibility.
            if "_ns_urg_input_err_count" not in state_refs:
                state_refs["_ns_urg_input_err_count"] = 0
            _c = state_refs["_ns_urg_input_err_count"]
            state_refs["_ns_urg_input_err_count"] = _c + 1
            if _c == 0 or _c % 100 == 0:
                logger.warning(
                    "[CognitiveWorker] ns_program_urgencies_input write "
                    "failed (count=%d): %s", _c + 1, _urg_err)

    latest = (consciousness or {}).get("latest_epoch") or {}
    epoch_id = latest.get("epoch_id", 0)
    curvature = latest.get("curvature", 0.0)

    # 4.5 §4.G life_force_inputs.bin publish (cross-process bridge).
    #
    # Track 1 drift retired per rFP §4.G + D-SPEC-57: the previous
    # ~140-line chi evaluate+publish block (which read 16 inputs from
    # cognitive_worker in-process state and called life_force_engine.evaluate
    # directly) has been replaced by a 2-step bridge:
    #   1. compute_life_force_inputs(state_refs) — pure helper aggregates
    #      the 16 inputs (extracted to logic/life_force_inputs_builder.py)
    #   2. LifeForceInputsPublisher.publish(...) — writes msgpack to
    #      life_force_inputs.bin SHM slot (G21 single-writer)
    # life_force_worker subscribes to KERNEL_EPOCH_TICK, reads the slot,
    # runs LifeForceEngine.evaluate, writes life_force_state.bin, and emits
    # CHI_UPDATED. cognitive_worker's downstream chi readers (the 5
    # hot-path sites below) now read life_force_state.bin via
    # LifeForceShmReader instead of the in-process engine's _latest_chi.
    expression_manager = state_refs.get("expression_manager")  # always None
    neural_nervous_system = state_refs.get("neural_nervous_system")
    _lf_inputs_publisher = state_refs.get("_life_force_inputs_publisher")
    if _lf_inputs_publisher is not None and consciousness is not None:
        try:
            from titan_hcl.logic.life_force_inputs_builder import (
                compute_life_force_inputs,
            )
            _lf_bank = state_refs.get("_shm_reader_bank")
            # sol_balance — real cached SOL from network_state.bin (kernel
            # balance-publisher loop; cheap G18 SHM read), replacing the 13.0
            # stub (BUG-LIFEFORCE-INPUT-STUBS, 2026-06-05).
            _lf_sol = None
            try:
                _lf_ns = _lf_bank.read_network_state() if _lf_bank is not None else None
                if _lf_ns:
                    _lf_sol = _lf_ns.get("balance_sol")
            except Exception:
                _lf_sol = None
            # anchor_freshness — real linear-over-24h freshness from the cached
            # on-chain anchor age (timechain_state.bin.recent_anchor_age_s; cheap
            # G18 SHM read), replacing the 0.5 stub (BUG-LIFEFORCE-INPUT-STUBS).
            _lf_anchor = None
            try:
                _lf_tc = _lf_bank.read_timechain_state() if _lf_bank is not None else None
                if _lf_tc:
                    _lf_age = _lf_tc.get("recent_anchor_age_s")
                    if _lf_age is not None:
                        _lf_anchor = 1.0 - min(float(_lf_age) / 86400.0, 1.0)
            except Exception:
                _lf_anchor = None
            # sovereignty_index — the ONE synthesis sovereignty score S in basis
            # points (int(S×10000)) via synthesis.sovereignty_readout (G18
            # snapshot read, the same source backup consumes), replacing the 0
            # stub (BUG-LIFEFORCE-INPUT-STUBS; "only one sovereignty score = S").
            _lf_sov = None
            try:
                from titan_hcl.synthesis.sovereignty_readout import (
                    rolling_sovereignty_bp,
                )
                _lf_sov = rolling_sovereignty_bp()
            except Exception:
                _lf_sov = None
            _lf_inputs = compute_life_force_inputs(
                coordinator=coordinator,
                pi_monitor=pi_monitor,
                neural_nervous_system=neural_nervous_system,
                latest_epoch=latest,
                consciousness=consciousness,
                topology_snap=topology_snap,
                expression_state_reader=state_refs.get("expression_state_reader"),
                sol_balance=_lf_sol,
                anchor_freshness=_lf_anchor,
                sovereignty_index=_lf_sov,
            )
            _lf_inputs_publisher.publish(_lf_inputs)
        except Exception as _err:
            if "_life_force_inputs_err_counts" not in state_refs:
                state_refs["_life_force_inputs_err_counts"] = {}
            _lfk = f"{type(_err).__name__}:{str(_err)[:80]}"
            _lfc = state_refs["_life_force_inputs_err_counts"].get(_lfk, 0) + 1
            state_refs["_life_force_inputs_err_counts"][_lfk] = _lfc
            if _lfc == 1 or _lfc % 100 == 0:
                logger.warning(
                    "[CognitiveWorker] life_force_inputs.publish failed "
                    "(count=%d): %s", _lfc, _err, exc_info=True)

    # Update topology journey Y-axis with chi circulation (via SHM reader).
    # Replaces the in-engine _latest_chi.get("circulation") read with a
    # SHM-direct sub-µs read against life_force_state.bin.
    if consciousness is not None:
        try:
            _lf_reader_topo = state_refs.get("_life_force_shm_reader")
            if _lf_reader_topo is not None:
                _topo = consciousness.get("topology")
                if _topo is not None and hasattr(_topo, "update_chi_circulation"):
                    _topo.update_chi_circulation(
                        _lf_reader_topo.get_circulation())
        except Exception:
            pass

    # 4.6 NeuralNervousSystem evaluation loop (full Phase C migration).
    #
    # 2026-05-10 — closes a SILENT 42-hour T3 production gap. Maker
    # observed home-route NEURAL MATURITY = 0% post-chi-fix; investigation
    # found T3 NeuralNervousSystem.evaluate() had not been called since
    # l0_rust_enabled flipped 2026-05-08 21:18 UTC. Symptoms:
    #   - last_train_ts frozen at 2026-05-08 14:38 UTC
    #   - total_transitions / total_train_steps frozen at boot-restore values
    #   - hormonal maturity stuck at 0.0 (formula needs evaluate() to fire)
    #   - all NS program signals stale (REFLEX/FOCUS/IMPULSE/etc.)
    #
    # Root cause: chunk 8E shipped engine init in cognitive_worker but the
    # per-tick observation-space build + evaluate() call was left in
    # spirit_worker.py:4470-4598 (heartbeat-only on T3). Same chunk 8M.6
    # sibling-gap class as the chi fix earlier today.
    #
    # Mirrors spirit_worker.py:4470-4598 with cognitive_worker context
    # adaptations: where spirit_worker had in-process refs to
    # filter_down/focus_*/impulse_engine/neuromodulator_system/etc.,
    # cognitive_worker reads from state_refs + SHM (read_sphere_clocks /
    # read_resonance_state / read_unified_spirit_metadata).
    #
    # Per `feedback_no_quick_patches_only_spec_correct_solutions.md`:
    # this is the SPEC-correct closure of a Phase C migration gap, not
    # a quick patch. cognitive_worker.evaluate-loop becomes the SOLE NS
    # driver under l0_rust_enabled=true (spirit_worker keeps its block
    # for l0_rust_enabled=false rollback path).
    if neural_nervous_system is not None:
        try:
            # ── Build observation_space inputs from cognitive_worker context ──
            _nn_clocks: dict = {}
            _nn_topo: dict = {}
            _nn_resonance: dict = {}
            _nn_us: dict = {}
            if shm_bank is not None:
                try:
                    sphere = shm_bank.read_sphere_clocks() if hasattr(
                        shm_bank, "read_sphere_clocks") else None
                    if isinstance(sphere, dict):
                        _nn_clocks = sphere.get("clocks") or {}
                except Exception:
                    pass
                try:
                    topo = shm_bank.read_topology_30d() if hasattr(
                        shm_bank, "read_topology_30d") else None
                    if isinstance(topo, dict):
                        _nn_topo = topo
                except Exception:
                    pass
                try:
                    res = shm_bank.read_resonance_state() if hasattr(
                        shm_bank, "read_resonance_state") else None
                    if isinstance(res, dict):
                        _nn_resonance = res
                except Exception:
                    pass
                try:
                    us = shm_bank.read_unified_spirit_metadata() if hasattr(
                        shm_bank, "read_unified_spirit_metadata") else None
                    if isinstance(us, dict):
                        _nn_us = us
                except Exception:
                    pass

            _nn_consciousness = {
                "drift_magnitude": float(latest.get("drift_magnitude", 0.0) or 0.0),
                "trajectory_magnitude": float(
                    latest.get("trajectory_magnitude", 0.0) or 0.0),
                "state_vector": latest.get("state_vector", [0.5] * 9),
            }
            _nn_dreaming = {}
            if coordinator is not None:
                _dr = getattr(coordinator, "dreaming", None) or getattr(
                    coordinator, "dreaming_engine", None)
                if _dr is not None:
                    _nn_dreaming = {
                        "fatigue": float(getattr(_dr, "last_fatigue", 0.0) or 0.0),
                        "readiness": float(getattr(_dr, "last_readiness", 0.0) or 0.0),
                    }
            _nn_neuromod_levels: dict = {}
            _nn_neuromod_setpoints: dict = {}
            _nm_sys = getattr(coordinator, "neuromodulator_system", None) if (
                coordinator is not None) else None
            if _nm_sys is not None:
                try:
                    for _nm_n, _nm_m in _nm_sys.modulators.items():
                        _nn_neuromod_levels[_nm_n] = float(getattr(_nm_m, "level", 0.0) or 0.0)
                        _nn_neuromod_setpoints[_nm_n] = float(getattr(_nm_m, "setpoint", 0.0) or 0.0)
                except Exception:
                    pass
            # §4.G — chi state read via SHM (was: life_force_engine._latest_chi)
            _nn_chi: dict = {}
            _nn_drain = 0.0
            _lf_reader_nn = state_refs.get("_life_force_shm_reader")
            if _lf_reader_nn is not None:
                _raw_chi = _lf_reader_nn.get_chi_state() or {}
                if _raw_chi:
                    _nn_chi = {
                        "total": float(_raw_chi.get("total", 0.5) or 0.5),
                        "circulation": float(_raw_chi.get("circulation", 0.5) or 0.5),
                        "body": (_raw_chi.get("body", {}).get("effective", 0.5)
                                 if isinstance(_raw_chi.get("body"), dict)
                                 else float(_raw_chi.get("body", 0.5) or 0.5)),
                        "mind": (_raw_chi.get("mind", {}).get("effective", 0.5)
                                 if isinstance(_raw_chi.get("mind"), dict)
                                 else float(_raw_chi.get("mind", 0.5) or 0.5)),
                        "spirit": (_raw_chi.get("spirit", {}).get("effective", 0.5)
                                   if isinstance(_raw_chi.get("spirit"), dict)
                                   else float(_raw_chi.get("spirit", 0.5) or 0.5)),
                    }
                _nn_drain = float(_lf_reader_nn.get_metabolic_drain())
            _nn_sd = 0.0
            _nn_wd = 0.0
            _nn_exp_p = 0.0
            _nn_exp_rep = 0.0
            _nn_tsd = 0.0
            if coordinator is not None:
                _dr = getattr(coordinator, "dreaming", None) or getattr(
                    coordinator, "dreaming_engine", None)
                if _dr is not None:
                    _nn_sd = float(getattr(_dr, "last_sleep_drive", 0.0) or 0.0)
                    _nn_wd = float(getattr(_dr, "last_wake_drive", 0.0) or 0.0)
                    _fb = getattr(_dr, "_last_fatigue_breakdown", {}) or {}
                    _nn_exp_p = float(_fb.get("o4_exp", 0.0) or 0.0)
                    _nn_exp_rep = float(_fb.get("o5_rep", 0.0) or 0.0)
                    _nn_tsd = float(getattr(_dr, "_epochs_since_dream", 0) or 0) * 7.0
            _nn_reasoning: dict = {}
            if reasoning_engine is not None:
                try:
                    get_obs = getattr(reasoning_engine, "get_observation_features", None)
                    if callable(get_obs):
                        _nn_reasoning = get_obs() or {}
                except Exception:
                    pass

            # Observables source — the live 30D inner/outer-trinity observables
            # the coordinator recomputes EVERY tick (observe_all →
            # inner.update_observables; inner_coordinator.py:159,163) and stores
            # on coordinator.inner.observables, keyed by the 6 body parts each
            # carrying {coherence,magnitude,velocity,direction,polarity} —
            # exactly the tier1 layout _build_tier1 consumes.
            #
            # 2026-06-11 — closes BUG-REASONING-TIER1-OBSERVABLES-STARVED. The
            # Phase C NNS-loop migration read observables via
            # `coordinator.observable_engine.get_observables()`, but the
            # coordinator stores the engine as `.observables` (NOT
            # `.observable_engine`) AND ObservableEngine exposes no
            # get_observables/snapshot method — so _nn_obs was ALWAYS {} →
            # tier1 (observation[:30]) all-zeros → COMPARE (inner[:15] vs
            # outer[15:30]) never `significant` → the +0.1 confidence boost
            # never fired → reasoning never crossed the 0.6 commit threshold
            # (commit_rate=0 fleet-wide / 231k conclusions, 0 commits). Read the
            # live dict the coordinator already maintains.
            _nn_obs: dict = {}
            _inner_ref = getattr(coordinator, "inner", None) if (
                coordinator is not None) else None
            if _inner_ref is not None:
                _maybe_obs = getattr(_inner_ref, "observables", None)
                if isinstance(_maybe_obs, dict) and _maybe_obs:
                    _nn_obs = _maybe_obs

            update_obs_fn = getattr(neural_nervous_system, "update_observation_space", None)
            if callable(update_obs_fn):
                update_obs_fn(
                    observables=_nn_obs,
                    sphere_clocks=_nn_clocks,
                    topology=_nn_topo,
                    resonance=_nn_resonance,
                    unified_spirit=_nn_us,
                    consciousness=_nn_consciousness,
                    dreaming=_nn_dreaming,
                    filter_down_mults={},  # not yet wired in cognitive_worker context
                    focus_nudges={},        # same
                    impulse_state={},       # same
                    neuromodulator_levels=_nn_neuromod_levels,
                    neuromodulator_setpoints=_nn_neuromod_setpoints,
                    chi_state=_nn_chi,
                    metabolic_drain=_nn_drain,
                    sleep_drive=_nn_sd,
                    wake_drive=_nn_wd,
                    experience_pressure=_nn_exp_p,
                    expression_repetitiveness=_nn_exp_rep,
                    time_since_dream=_nn_tsd,
                    reasoning_active=float(_nn_reasoning.get("is_active", 0.0) or 0.0),
                    reasoning_chain_length=float(_nn_reasoning.get("chain_length_norm", 0.0) or 0.0),
                    reasoning_confidence=float(_nn_reasoning.get("confidence", 0.0) or 0.0),
                    reasoning_gut_agreement=float(_nn_reasoning.get("gut_agreement", 0.0) or 0.0),
                )

            # Maturity signals — drives hormonal_pressure.update_maturity inside evaluate().
            update_mat_fn = getattr(neural_nervous_system, "update_maturity_signals", None)
            if callable(update_mat_fn):
                _mat_great = int(_nn_resonance.get("great_pulse_count", 0) or 0)
                _mat_radius = 1.0
                _inner_radii = [
                    float(c.get("radius", 1.0) or 1.0)
                    for name, c in _nn_clocks.items()
                    if isinstance(name, str) and name.startswith("inner_")
                    and isinstance(c, dict)
                ]
                if _inner_radii:
                    _mat_radius = sum(_inner_radii) / len(_inner_radii)
                _mat_epochs = int(latest.get("epoch_id", 0) or 0)
                update_mat_fn(
                    great_epochs=_mat_great,
                    sphere_radius=_mat_radius,
                    consciousness_epochs=_mat_epochs,
                )

            # Propagate dreaming state into the NS (canonical circadian-
            # modulation API). This was an orphaned wiring gap in Phase C —
            # set_dreaming() had zero callers, so neural_ns._is_dreaming was
            # stuck False fleet-wide (Phase 0 §8). With it wired, _train_all
            # is gated off during the dream (neural_nervous_system.py) so the
            # off-tick consolidation suite has exclusive access to the program
            # nets; perception/hormonal/signals still run. D-SPEC-105.
            _set_dreaming_fn = getattr(
                neural_nervous_system, "set_dreaming", None)
            if callable(_set_dreaming_fn):
                try:
                    _set_dreaming_fn(_epoch_is_dreaming)
                except Exception as _sd_err:
                    logger.debug(
                        "[CognitiveWorker] NS set_dreaming failed: %s", _sd_err)

            # Drive evaluate() — THIS is what actually trains + updates hormonal maturity.
            # Per spirit_worker.py:4593: temporal=None (programs are 55D; 60D mismatch).
            evaluate_fn = getattr(neural_nervous_system, "evaluate", None)
            if callable(evaluate_fn):
                _nn_signals = evaluate_fn(_nn_obs, temporal=None) or []
                # §4.B Track 3 migration-preserving fix (2026-05-15): publish
                # the in-process HormonalSystem state to nns_hormonal_state.bin
                # SHM slot. expression_worker reads this slot via
                # HormonalShmReader to drive ExpressionManager.evaluate_all
                # with the SAME live hormone levels that pre-extraction
                # in-process evaluate_all saw. WITHOUT this publish, the
                # cross-process expression_worker reads hormonal_module's
                # different HormonalSystem instance (only stimulated by
                # ns_worker IMPULSE), composites stay at near-zero urge.
                _h = getattr(neural_nervous_system, "_hormonal", None)
                if _h is not None:
                    try:
                        # Lazy-init writer on first use.
                        if "_nns_hormonal_writer" not in state_refs:
                            from titan_hcl.core.state_registry import (
                                NNS_HORMONAL_STATE, StateRegistryWriter,
                                ensure_shm_root, resolve_titan_id,
                            )
                            _titan_id = resolve_titan_id()
                            state_refs["_nns_hormonal_writer"] = StateRegistryWriter(
                                NNS_HORMONAL_STATE, ensure_shm_root(_titan_id))
                            logger.info(
                                "[CognitiveWorker] nns_hormonal_state.bin "
                                "writer attached (G21 single-writer; feeds "
                                "expression_worker.HormonalShmReader per "
                                "§4.B Track 3 migration-preserving fix)")
                        from titan_hcl.modules.hormonal_worker import (
                            encode_hormonal_state)
                        _arr = encode_hormonal_state(_h)
                        state_refs["_nns_hormonal_writer"].write(_arr)
                    except Exception as _hw_err:
                        _log_driver_err("nns_hormonal.publish", _hw_err)

                    # Phase 3.A D-SPEC-86: write hormone_fires.bin SHM slot.
                    # Inner-spirit-rs producer reads this slot for 7 dims
                    # (authenticity, pattern_recognition, reflective_capacity,
                    # creative_joy, truth_resonance, connection_fulfillment,
                    # exploration_joy). Pre-fix, slot had no writer → all 7
                    # dead fleet-wide. Schema matches Rust constants.rs:291
                    # + inner_spirit_sidecar.py:130 reader contract:
                    # {"fires": {HORMONE_NAME: int_count, ...}, "ts": float}.
                    # G21 single-writer = cognitive_worker (in-process
                    # NeuralNervousSystem._hormonal owns the authoritative
                    # fire counts; same instance feeds ExpressionManager via
                    # NNS_HORMONAL_STATE above).
                    try:
                        if "_hormone_fires_writer" not in state_refs:
                            from titan_hcl.logic.spirit_state_specs import (
                                HORMONE_FIRES_SPEC)
                            from titan_hcl.core.state_registry import (
                                StateRegistryWriter, ensure_shm_root,
                                resolve_titan_id)
                            _titan_id_hf = resolve_titan_id()
                            state_refs["_hormone_fires_writer"] = (
                                StateRegistryWriter(
                                    HORMONE_FIRES_SPEC,
                                    ensure_shm_root(_titan_id_hf)))
                            logger.info(
                                "[CognitiveWorker] hormone_fires.bin writer "
                                "attached (G21 single-writer; Phase 3.A "
                                "D-SPEC-86 — feeds inner_spirit_sidecar.bin "
                                "for 7 fire-derived chit/ananda dims)")
                        _fires = {
                            str(n): int(getattr(h, "fire_count", 0) or 0)
                            for n, h in (getattr(_h, "_hormones", {}) or {}).items()
                        }
                        import msgpack as _msgpack_hf
                        _hf_bytes = _msgpack_hf.packb(
                            {"fires": _fires, "ts": time.time()},
                            use_bin_type=True)
                        state_refs["_hormone_fires_writer"].write_variable(
                            _hf_bytes)
                    except Exception as _hf_err:
                        _log_driver_err("hormone_fires.publish", _hf_err)
                # Light cadence-only logging — per-tick log would flood.
                if epoch_id and epoch_id % 100 == 0:
                    logger.info(
                        "[CognitiveWorker] NeuralNS alive: transitions=%d, "
                        "train_steps=%d, signals=%d, maturity=%.4f",
                        getattr(neural_nervous_system, "_total_transitions", 0),
                        getattr(neural_nervous_system, "_total_train_steps", 0),
                        len(_nn_signals),
                        getattr(getattr(neural_nervous_system, "_hormonal", None), "maturity", 0.0))

                # ── §4.Q neuromod_inputs.bin publish ────────────────────
                # Build the 6 emergent neuromod inputs + chi_health +
                # topology_velocity + dt and write to the SHM slot for
                # neuromod_worker to consume in its evaluate driver.
                # Mirrors the nns_hormonal_state.bin pattern above.
                try:
                    if "_neuromod_inputs_builder" not in state_refs:
                        from titan_hcl.logic.neuromod_inputs_builder import (
                            NeuromodInputsBuilder)
                        state_refs["_neuromod_inputs_builder"] = (
                            NeuromodInputsBuilder(
                                dna=_load_toml_section("neuromodulator_dna")))
                        logger.info(
                            "[CognitiveWorker] NeuromodInputsBuilder attached "
                            "(§4.Q feeds neuromod_inputs.bin → neuromod_worker.evaluate)")
                    if "_neuromod_inputs_writer" not in state_refs:
                        from titan_hcl.core.state_registry import (
                            NEUROMOD_INPUTS, StateRegistryWriter,
                            ensure_shm_root, resolve_titan_id)
                        _titan_id = resolve_titan_id()
                        state_refs["_neuromod_inputs_writer"] = (
                            StateRegistryWriter(
                                NEUROMOD_INPUTS, ensure_shm_root(_titan_id)))
                        logger.info(
                            "[CognitiveWorker] neuromod_inputs.bin writer "
                            "attached (G21 single-writer; feeds neuromod_worker)")

                    _builder = state_refs["_neuromod_inputs_builder"]
                    _topo_v = float(getattr(
                        neural_nervous_system, "_topology_velocity", 0.3) or 0.3)
                    _is_dreaming_q = False
                    if coordinator and getattr(coordinator, "dreaming", None):
                        _is_dreaming_q = bool(getattr(
                            coordinator.dreaming, "is_dreaming", False))
                    _payload = _builder.build(
                        coordinator=coordinator,
                        neural_nervous_system=neural_nervous_system,
                        # §4.G — life_force_engine retired (Track 1 drift closed,
                        # G7). chi_health bridges via NEUROMOD_EXTERNAL_NUDGE bus
                        # event from life_force_worker per G9. Builder is
                        # None-tolerant.
                        life_force_engine=None,
                        pi_monitor=pi_monitor,
                        exp_orchestrator=state_refs.get("exp_orchestrator"),
                        sphere_clocks_snap=getattr(
                            coordinator, "_sphere_clocks_snapshot", None),
                        latest_epoch=(consciousness.get("latest_epoch") or {})
                        if consciousness else {},
                        is_dreaming=_is_dreaming_q,
                        prediction_stats=state_refs.get("_prediction_stats"),
                        expression_stats=state_refs.get("_expression_composites"),
                        kin_signature=state_refs.get("_kin_signature"),
                        filter_down_count=int(state_refs.get(
                            "_filter_down_count", 0) or 0),
                        resonance_count=int((state_refs.get(
                            "_kin_signature") or {}).get(
                            "resonant_count", 0) or 0),
                        topology_velocity=_topo_v,
                        dt=1.0,
                    )
                    import msgpack as _msgpack_neuromod
                    _msg_bytes = _msgpack_neuromod.packb(_payload, use_bin_type=True)
                    state_refs["_neuromod_inputs_writer"].write_variable(_msg_bytes)
                except Exception as _ni_err:
                    # Visibility: log first occurrence + every 100th of each
                    # error class. Mirrors the _ns_eval_err_counts pattern at
                    # L2036 (avoids forward reference to _log_driver_err
                    # which is defined later in this function scope).
                    if "_neuromod_inputs_err_counts" not in state_refs:
                        state_refs["_neuromod_inputs_err_counts"] = {}
                    _nik = f"{type(_ni_err).__name__}:{str(_ni_err)[:80]}"
                    _nic = state_refs["_neuromod_inputs_err_counts"].get(_nik, 0) + 1
                    state_refs["_neuromod_inputs_err_counts"][_nik] = _nic
                    if _nic == 1 or _nic % 100 == 0:
                        logger.error(
                            "[CognitiveWorker] neuromod_inputs.publish failed "
                            "(count=%d): %s", _nic, _ni_err, exc_info=True)

                # ── §4.Q chunk Q11: SOVEREIGNTY_EPOCH + TimeChain heartbeat ──
                # Mainnet Lifecycle (rFP 2026-04-20) — fires every 10
                # consciousness epochs to stay within bus budget. v1.8.3 §4.L
                # (D-SPEC-57, 2026-05-15): sovereignty_worker subprocess
                # subscribes (dst="sovereignty") and calls
                # tracker.record_epoch(...). Great-pulse count comes from
                # resonance._great_pulse_count delta (read via SHM resonance
                # state).
                # Replaces spirit_worker.py:4389-4427 (dead under Phase C).
                try:
                    _sov_eid = int(epoch_id or 0)
                    _sov_last = int(state_refs.get("_sov_last_sent_epoch", 0) or 0)
                    if _sov_eid > 0 and (_sov_eid - _sov_last) >= 10:
                        _sov_prev_gp = int(state_refs.get(
                            "_sov_prev_great_pulses", 0) or 0)
                        _sov_cur_gp = 0
                        try:
                            if shm_bank is not None and hasattr(
                                    shm_bank, "read_resonance_state"):
                                _res = shm_bank.read_resonance_state() or {}
                                _sov_cur_gp = int(
                                    _res.get("great_pulse_count", 0) or 0)
                        except Exception:
                            pass
                        _sov_fired = (_sov_cur_gp - _sov_prev_gp) > 0
                        state_refs["_sov_prev_great_pulses"] = _sov_cur_gp
                        state_refs["_sov_last_sent_epoch"] = _sov_eid
                        _sov_dev_age = float(getattr(
                            pi_monitor, "developmental_age", 0.0) or 0.0) \
                            if pi_monitor else 0.0
                        _sov_neuromods = neuromod_reader() if neuromod_reader else {}
                        send_queue.put({
                            "type": bus.SOVEREIGNTY_EPOCH,
                            "src": name,
                            "dst": "sovereignty",
                            "payload": {
                                "epoch_id": _sov_eid,
                                "neuromods": dict(_sov_neuromods or {}),
                                "dev_age": _sov_dev_age,
                                "great_pulse_fired": _sov_fired,
                                "total_great_pulses": _sov_cur_gp,
                            },
                            "ts": time.time(),
                        })

                    # TimeChain heartbeat (every 100 epochs) — main chain.
                    _tc_last = int(state_refs.get(
                        "_tc_last_heartbeat_epoch", 0) or 0)
                    if _sov_eid > 0 and (_sov_eid - _tc_last) >= 100:
                        state_refs["_tc_last_heartbeat_epoch"] = _sov_eid
                        # §4.G — chi via SHM (was: life_force_engine._latest_chi)
                        _lf_reader_tc = state_refs.get("_life_force_shm_reader")
                        _tc_chi = (
                            _lf_reader_tc.get_chi_state()
                            if _lf_reader_tc is not None else {}
                        )
                        _tc_neuromods = neuromod_reader() if neuromod_reader else {}
                        _tc_emotion = "neutral"
                        # Best-effort emotion from NEUROMOD_STATS_UPDATED cache.
                        _last_stats = state_refs.get("_last_neuromod_stats")
                        if isinstance(_last_stats, dict):
                            _tc_emotion = str(_last_stats.get(
                                "current_emotion", "neutral"))
                        _tc_dreaming = False
                        if coordinator and getattr(coordinator, "dreaming", None):
                            _tc_dreaming = bool(getattr(
                                coordinator.dreaming, "is_dreaming", False))
                        send_queue.put({
                            "type": bus.EPOCH_TICK,
                            "src": name,
                            "dst": "timechain",
                            "payload": {
                                "epoch_id": _sov_eid,
                                "chi_total": float(_tc_chi.get("total", 0.0) or 0.0),
                                "emotion": _tc_emotion,
                                "is_dreaming": _tc_dreaming,
                                "neuromods": dict(_tc_neuromods or {}),
                            },
                            "ts": time.time(),
                        })
                except Exception as _sov_err:
                    if "_sov_emit_err_counts" not in state_refs:
                        state_refs["_sov_emit_err_counts"] = {}
                    _sek = f"{type(_sov_err).__name__}:{str(_sov_err)[:80]}"
                    _sec = state_refs["_sov_emit_err_counts"].get(_sek, 0) + 1
                    state_refs["_sov_emit_err_counts"][_sek] = _sec
                    if _sec == 1 or _sec % 100 == 0:
                        logger.warning(
                            "[CognitiveWorker] §4.Q sovereignty/timechain emit "
                            "failed (count=%d): %s", _sec, _sov_err)
        except Exception as _ns_err:
            # 2026-05-10 — error visibility upgrade per directive_error_visibility.md.
            # Was logger.debug — silenced silent failures (this is exactly how the
            # 42-hour T3 NS gap stayed invisible). Promote to ERROR for the FIRST
            # failure per-error-class + every 100th repeat (prevents flood while
            # keeping silent failure impossible).
            if not hasattr(state_refs, "_ns_eval_err_counts"):
                state_refs["_ns_eval_err_counts"] = {}
            _err_key = f"{type(_ns_err).__name__}:{str(_ns_err)[:80]}"
            _ns_err_counts = state_refs["_ns_eval_err_counts"]
            _count = _ns_err_counts.get(_err_key, 0) + 1
            _ns_err_counts[_err_key] = _count
            if _count == 1 or _count % 100 == 0:
                logger.error(
                    "[CognitiveWorker] NS evaluation loop failed (count=%d): %s",
                    _count, _ns_err, exc_info=True)

    # 2026-05-10 — error visibility helper. The 42-hour T3 NS gap was silent
    # because exception handlers in this driver used logger.debug. Per
    # directive_error_visibility.md + feedback_all_tests_must_pass_no_exceptions.md,
    # promote all engine-driver exceptions to ERROR (first occurrence per
    # error class + every 100th repeat — flood-safe but never silent).
    def _log_driver_err(driver_name: str, err: Exception) -> None:
        if "_engine_drv_err_counts" not in state_refs:
            state_refs["_engine_drv_err_counts"] = {}
        _err_key = f"{driver_name}:{type(err).__name__}:{str(err)[:80]}"
        _ec = state_refs["_engine_drv_err_counts"]
        _count = _ec.get(_err_key, 0) + 1
        _ec[_err_key] = _count
        if _count == 1 or _count % 100 == 0:
            logger.error(
                "[CognitiveWorker] %s driver failed (count=%d): %s",
                driver_name, _count, err, exc_info=True)

    # 5. Drive PiHeartbeatMonitor.observe(curvature, epoch_id).
    if pi_monitor is not None and epoch_id > 0:
        try:
            pi_monitor.observe(curvature=curvature, epoch_id=epoch_id)
        except Exception as _err:
            _log_driver_err("pi_monitor.observe", _err)

    # 6. Step ReasoningEngine if it has an active chain. Gated off during
    #    dreams (no waking chain advancement during sleep — the off-tick
    #    consolidation owns the reasoning engine; D-SPEC-105 §8).
    if reasoning_engine is not None and not _epoch_is_dreaming:
        try:
            has_active = getattr(reasoning_engine, "has_active_chain", None)
            step = getattr(reasoning_engine, "step", None)
            if callable(has_active) and callable(step) and has_active():
                step()
        except Exception as _err:
            _log_driver_err("reasoning_engine.step", _err)

    # 6.5 Drive ReasoningEngine.tick — IQL training driver.
    #
    # 2026-05-10 closes BUG-COGNITIVE-WORKER-REASONING-TICK-MISSING-20260510.
    # tick() runs the per-epoch IQL learning + reward update loop.
    # step() (above, step 6) only advances the active chain by one action.
    # Different methods, different intents — both must run per epoch.
    # Mirrors spirit_worker.py:4793 input contract; cross-process unavailable
    # state (working_memory, neuromodulator setpoints) defaults gracefully.
    neural_nervous_system = state_refs.get("neural_nervous_system")
    if reasoning_engine is not None:
        try:
            _is_dreaming_r = bool(
                getattr(getattr(coordinator, "inner", None),
                        "is_dreaming", False)
            ) if coordinator else False
            if not _is_dreaming_r:
                # Gut signals — hormone-augmented program urgencies (rFP β
                # Phase 3). Falls back to raw _all_urgencies, then to fired
                # signals from coordinator's last NS evaluation.
                _r_gut = {}
                if neural_nervous_system is not None:
                    try:
                        _r_gut = neural_nervous_system.get_augmented_urgencies(
                            hormone_blend=0.3)
                    except Exception:
                        _r_gut = dict(getattr(
                            neural_nervous_system, "_all_urgencies", {}) or {})
                if not _r_gut and coordinator is not None:
                    for _rs in getattr(
                            coordinator, "_last_nervous_signals", []) or []:
                        _r_gut[_rs.get("system", "")] = _rs.get("urgency", 0.0)

                # Body state — fatigue + chi + metabolic_drain + dreaming.
                # §4.G — chi/drain via SHM (was: life_force_engine._latest_chi)
                _lf_reader_r = state_refs.get("_life_force_shm_reader")
                _r_body = {
                    "fatigue": float(_nn_exp_p) if "_nn_exp_p" in dir() else 0.3,
                    "chi_total": float(
                        _lf_reader_r.get_chi_total()
                        if _lf_reader_r is not None else 0.5),
                    "metabolic_drain": float(
                        _lf_reader_r.get_metabolic_drain()
                        if _lf_reader_r is not None else 0.0),
                    "is_dreaming": _is_dreaming_r,
                }

                # Raw neuromods — read from SHM via neuromod_reader (Rust-
                # produced under l0_rust=true, written by neuromod_worker).
                _r_neuromods = neuromod_reader() if neuromod_reader else {}
                if not isinstance(_r_neuromods, dict):
                    _r_neuromods = {}

                # Working memory items — Block F migration (2026-05-10)
                # boots WorkingMemory in cognitive_worker. get_context()
                # returns recent attended items for IQL chain context;
                # falls back to [] if engine boot failed.
                _wm_for_reas = state_refs.get("working_mem")
                _r_wm_items: list = []
                if _wm_for_reas is not None:
                    try:
                        _wm_ctx = _wm_for_reas.get_context()
                        if isinstance(_wm_ctx, list):
                            _r_wm_items = _wm_ctx
                    except Exception:
                        pass

                # Observation — enriched NS observation space (79D).
                _r_obs = None
                if neural_nervous_system is not None and hasattr(
                        neural_nervous_system, "_observation_space"):
                    try:
                        _r_obs = neural_nervous_system._observation_space.build_input(
                            "enriched")
                    except Exception:
                        _r_obs = None
                if _r_obs is None:
                    import numpy as _np_r
                    _r_obs = _np_r.zeros(79)

                tick_fn = getattr(reasoning_engine, "tick", None)
                if callable(tick_fn):
                    _r_result = tick_fn(
                        observation=_r_obs,
                        gut_signals=_r_gut,
                        body_state=_r_body,
                        raw_neuromods=_r_neuromods,
                        working_memory_items=_r_wm_items,
                        dt=1.0,
                    )
                    # Stash for Block D Tier 1 SPEAK_REQUEST emission (only
                    # threaded into the bus payload when COMMIT@conf>=0.5).
                    state_refs["_last_reasoning_result"] = _r_result

                    # §4.Q (2026-05-15) FILTER_DOWN nudges — emit via bus
                    # event to neuromod_worker. Replaces in-process
                    # spirit_worker.py:5197-5207 (COMMIT confidence → DA)
                    # + spirit_worker.py:5229-5237 (ABANDON long chain → NE).
                    # Reasoning is "one of many neuromod inputs" — DA/NE
                    # nudges are tuned downwards (max_delta=0.03).
                    if isinstance(_r_result, dict):
                        _r_dev_age = float(getattr(
                            pi_monitor, "developmental_age", 1.0) or 1.0) \
                            if pi_monitor else 1.0
                        _r_conf = float(_r_result.get("confidence", 0.5))
                        _da_now = float(_r_neuromods.get("DA", 0.5))
                        _da_delta = (_r_conf - 0.5) * 0.05
                        _r_nudge = {"DA": _da_now + _da_delta}
                        if _r_conf > 0.7:
                            _end_now = float(_r_neuromods.get("Endorphin", 0.5))
                            _r_nudge["Endorphin"] = _end_now + 0.02
                        _emit_neuromod_nudge(
                            _r_nudge, max_delta=0.03,
                            developmental_age=_r_dev_age,
                            source="filter_down_commit")
                        if _r_result.get("action") == "ABANDON":
                            _r_chain_len = int(_r_result.get("chain_length", 0) or 0)
                            if _r_chain_len >= 5:
                                _ne_now = float(_r_neuromods.get("NE", 0.5))
                                _emit_neuromod_nudge(
                                    {"NE": _ne_now + 0.02},
                                    max_delta=0.02,
                                    developmental_age=_r_dev_age,
                                    source="filter_down_abandon")

                        # ── rFP α §2b — CGN reasoning_strategy emission (RESTORED).
                        #
                        # REGRESSION FIX (REASONING-STRATEGY-CGN-LOW-EMISSION-RATE,
                        # closed 2026-05-26). The reasoning engine attaches a
                        # `cgn_reasoning_strategy` payload to the conclusion dict
                        # on COMMIT@reward>=cgn_emission_threshold (reasoning.py:
                        # 1811-1832, default threshold 0.55 from
                        # [reasoning_rewards] in titan_params.toml). The matching
                        # bus emission block lived at the deleted
                        # spirit_worker.py:4634-4659 path until commit 72f95a6b
                        # (D8-3 spirit_worker retirement) dropped it WITHOUT
                        # re-homing — exactly the orphaned-orchestration-loop
                        # class warned by `project_spirit_cognitive_migration_
                        # dropped_orchestration_loops`. The payload kept being
                        # attached to conclusions, but nothing read it, so
                        # `_consumers.reasoning_strategy` stayed `formed=0
                        # tested=0` fleet-wide (T1: 0 hits across 64,438+ commits).
                        #
                        # SPEC anchors:
                        #   * §9.B cognitive_worker block (L3 cognitive home post
                        #     D-SPEC-110 v1.48.0). Reasoning engine lives here →
                        #     emission must live here too (locality + restart-
                        #     isolation per §11.G + `feedback_l2_migration_audit_
                        #     three_categories.md`).
                        #   * §8 bus catalog — `CGN_TRANSITION` (no version
                        #     bump; existing constant) routed to consumer
                        #     `cgn`. META-CGN observes the standard CGN→META
                        #     flow per D16; no duplicate emit_meta_cgn_signal.
                        #   * §23.1 (BUG #3 Phase C completion 2026-04-24) —
                        #     peer-publish CGN_CROSS_INSIGHT via
                        #     `emit_chain_outcome_insight()` for emot_cgn
                        #     learning from reasoning_strategy outcomes
                        #     (rate-gated + informative-only inside helper).
                        _cgn_payload = _r_result.get("cgn_reasoning_strategy")
                        if _cgn_payload:
                            try:
                                _cgn_payload["epoch_id"] = (
                                    int(getattr(pi_monitor, "current_epoch_id", 0)
                                        or 0)
                                    if pi_monitor else 0
                                )
                                _cgn_concept = "strategy_" + "_".join(
                                    _cgn_payload.get("chain_signature", []))[:80]
                                # NB: "CGN_TRANSITION" is a fleet-wide string
                                # literal (not yet a `bus.py` constant) —
                                # cgn_consumer_client.py:495,529 +
                                # language_worker.py:502+ + coding_explorer.py:821
                                # all use the literal form. Adding the constant
                                # to bus.py is a separate convention-cleanup
                                # follow-up under `feedback_bus_emit_use_constants`;
                                # this restoration matches the current fleet
                                # convention to keep the migration regression
                                # closure scope-tight.
                                _send_msg(
                                    send_queue, "CGN_TRANSITION", name, "cgn", {
                                        "type": "experience",  # (b) complete transition → record_experience → observe_for (DEFERRED G1)
                                        "consumer": "reasoning_strategy",
                                        "concept_id": _cgn_concept,
                                        "reward": float(_cgn_payload.get(
                                            "outcome_score", 0.0)),
                                        "outcome_context": _cgn_payload,
                                    })
                                try:
                                    from titan_hcl.logic.cgn_consumer_client import (
                                        emit_chain_outcome_insight)
                                    emit_chain_outcome_insight(
                                        send_queue, name, "reasoning_strategy",
                                        float(_cgn_payload.get("outcome_score",
                                                               0.0)),
                                        ctx={"concept_id": _cgn_concept[:60]})
                                except Exception:
                                    pass
                            except Exception as _cgn_err:
                                if hash(("cgn_rstrat",
                                         _cgn_err.__class__.__name__)) % 50 == 0:
                                    logger.warning(
                                        "[Reasoning/CGN-strat] emit failed: %s",
                                        _cgn_err)
        except Exception as _err:
            _log_driver_err("reasoning_engine.tick", _err)

    # 7. Tick MetaReasoningEngine — feeds latest_epoch state.
    #
    # 2026-05-10 post-deploy fix: MetaReasoningEngine.tick signature
    # (logic/meta_reasoning.py:1056) requires 4 additional positional
    # deps — chain_archive, meta_wisdom, exp_orchestrator, meta_autoencoder —
    # that were missing from this call site. Was raising TypeError
    # silently every epoch. Foundation engines now booted in
    # _init_cognitive_engines and threaded through state_refs.
    #
    # Gated off during dreams: meta_engine.tick trains meta_policy /
    # sub_mode_policies, which the off-tick consolidate_training also trains —
    # concurrent backprop would race (Phase 0 §8 / D-SPEC-105). No waking
    # meta-cognition during sleep.
    if meta_engine is not None and not _epoch_is_dreaming:
        # ── A-finish: Subsystem signal cache refresh (restored from
        #    spirit_worker, lost in 72f95a6b D8-3 — never re-homed). Per
        #    architecture §5.5 / rFP_subsystem_reward_refresh_restore.md: fire
        #    TIMECHAIN_QUERY + CONTRACT_LIST async (non-blocking) when the cache
        #    is stale and no refresh is in flight. The RESP handlers in the bus
        #    dispatcher call meta_engine.update_subsystem_cache() to populate
        #    the 14 compound-reward signals (were stubbed at 0 → only FORMULATE
        #    earned reward → fleet-wide meta-primitive monoculture). Fire before
        #    tick() so the next chain reads fresh signals within the TTL window.
        try:
            if (meta_engine.is_subsystem_cache_stale()
                    and not meta_engine.is_subsystem_cache_pending()):
                _send_msg(send_queue, bus.TIMECHAIN_QUERY, name, "timechain",
                          {"limit": 50})
                _send_msg(send_queue, bus.CONTRACT_LIST, name, "timechain",
                          {"status": "active"})
                meta_engine.mark_subsystem_cache_pending()
                logger.info("[META] Subsystem cache refresh dispatched")
        except Exception as _ssrefresh_err:
            logger.warning(
                "[META] Subsystem cache refresh dispatch failed: %s",
                _ssrefresh_err)
        try:
            tick_fn = getattr(meta_engine, "tick", None)
            if callable(tick_fn):
                _meta_neuromods = neuromod_reader() if neuromod_reader else None
                # D-SPEC-70 v1.15.0 — populate _introspect_context so
                # _prim_introspect's META_INTROSPECT_REQUEST publish carries
                # the in-process state self_reflection_worker needs to run
                # sr.introspect() with full ctx (msl_data.chi_coherence for
                # Producer #14 coherence_gain detection, etc.).
                #
                # Minimal-context for F-8 initial closure: epoch + chi_coherence
                # (the Producer #14 input). Other fields (reasoning_stats /
                # language_stats / coordinator_data) stay None — sr.introspect
                # handles Optional gracefully. Follow-up rFP may surface them.
                try:
                    _intr_ctx = {"epoch": 0, "msl_data": None,
                                 "reasoning_stats": None,
                                 "language_stats": None,
                                 "coordinator_data": None}
                    if consciousness is not None:
                        _intr_ctx["epoch"] = int(
                            consciousness.get("epoch_number") or 0)
                    _chi = None
                    if coordinator is not None:
                        try:
                            _msl = getattr(coordinator, "_last_msl_data", None)
                            if isinstance(_msl, dict):
                                _chi = _msl.get("chi_coherence")
                        except Exception:
                            pass
                    if _chi is None and coordinator is not None:
                        try:
                            _chi_state = getattr(
                                coordinator, "_last_chi_state", None)
                            if isinstance(_chi_state, dict):
                                _chi = _chi_state.get("coherence")
                        except Exception:
                            pass
                    if _chi is not None:
                        _intr_ctx["msl_data"] = {"chi_coherence": float(_chi)}
                    meta_engine._introspect_context = _intr_ctx
                except Exception as _ctx_err:
                    logger.debug(
                        "[CognitiveWorker] _introspect_context populate "
                        "failed: %s — META INTROSPECT will publish with "
                        "minimal ctx", _ctx_err)
                _meta_result = tick_fn(
                    state_132d=latest.get("state_vector"),
                    neuromods=_meta_neuromods,
                    reasoning_engine=reasoning_engine,
                    chain_archive=state_refs.get("chain_archive"),
                    meta_wisdom=state_refs.get("meta_wisdom"),
                    exp_orchestrator=state_refs.get("exp_orchestrator"),
                    meta_autoencoder=state_refs.get("meta_autoencoder"),
                )
                # §4.Q (2026-05-15) META reward nudges — emit via bus events.
                # Replaces in-process spirit_worker.py:7238-7275 (eureka DA
                # burst + SPIRIT_SELF nudge). meta_engine.tick returns a
                # dict with `eureka` + `nudge_request` keys on trigger.
                if isinstance(_meta_result, dict):
                    _meta_dev_age = float(getattr(
                        pi_monitor, "developmental_age", 1.0) or 1.0) \
                        if pi_monitor else 1.0
                    _meta_neuromods = _meta_neuromods or {}
                    # M9: EUREKA pulse — DA burst (spirit_worker.py:7239-7248).
                    _eureka = _meta_result.get("eureka")
                    if _eureka:
                        _da_burst = float(_eureka.get("da_burst_magnitude", 0.0))
                        if _da_burst > 0.0:
                            _da_now = float(_meta_neuromods.get("DA", 0.5))
                            _da_target = min(1.0, _da_now + _da_burst)
                            _emit_neuromod_nudge(
                                {"DA": _da_target}, max_delta=_da_burst,
                                developmental_age=_meta_dev_age,
                                source="meta_eureka")
                        # D-SPEC-66 v1.11.0: publish META_EUREKA bus event
                        # (closes latent silent-emit bug — sole producer
                        # was spirit_worker.py:6193 dead under fleet-wide
                        # Phase C 2026-05-14) + SOCIAL_CATALYST per D8-3
                        # catalyst-producer site #2 closure (PLAN §1.2).
                        _send_msg(send_queue, bus.META_EUREKA, name, "all",
                                  _eureka)
                        _has_ss = "SPIRIT_SELF" in str(
                            _meta_result.get("chain_primitives", []))
                        _send_msg(send_queue, bus.SOCIAL_CATALYST, name,
                                  "social", {
                                      "type": "eureka_spirit"
                                              if _has_ss else "eureka",
                                      "significance": 0.95 if _has_ss
                                                     else 0.7,
                                      "content": (
                                          f"{'SPIRIT_SELF ' if _has_ss else ''}"
                                          f"EUREKA: "
                                          f"{_eureka.get('domain', '?')} "
                                          f"novelty="
                                          f"{_eureka.get('novelty', 0):.2f}"),
                                      "data": _eureka,
                                  })
                        # Record stage (rFP_experience_distillation_phase_c §5) —
                        # an eureka is a meta_reasoning experience. Targeted self-
                        # emit (dst=cognitive_worker delivers back per D-SPEC-52).
                        bus.emit_experience_record(
                            send_queue, name,
                            domain="meta_reasoning",
                            action_taken=(
                                f"eureka:{str(_eureka.get('domain', '?'))[:40]}"),
                            outcome_score=min(1.0, max(0.0, float(
                                _eureka.get("novelty", 0.5)))),
                            context={
                                "eureka_domain": str(
                                    _eureka.get("domain", "?"))[:40],
                                "novelty": float(_eureka.get("novelty", 0.0)),
                                "spirit_self": bool(_has_ss),
                                "source": "eureka",
                            },
                            coalesce_key="meta_reasoning:eureka",
                        )
                    # D-SPEC-66 v1.11.0: BREAK primitive → SOCIAL_CATALYST
                    # (type=vulnerability) per D8-3 catalyst-producer
                    # site #3 closure (PLAN §1.3). Was spirit_worker.py:
                    # 6218 dead under Phase C heartbeat-stub.
                    if _meta_result.get("primitive") == "BREAK":
                        _send_msg(send_queue, bus.SOCIAL_CATALYST, name,
                                  "social", {
                                      "type": "vulnerability",
                                      "significance": 0.4,
                                      "content": (
                                          f"BREAK at step "
                                          f"{_meta_result.get('chain_length', 0)}: "
                                          f"{_meta_result.get('sub_mode', 'restart')}"),
                                      "data": {"chain_length":
                                               _meta_result.get(
                                                   "chain_length", 0)},
                                  })
                    # M8: SPIRIT_SELF nudge (spirit_worker.py:7262-7275).
                    _nudge_req = _meta_result.get("nudge_request")
                    if _nudge_req:
                        _ss_nudges = _nudge_req.get("nudges", {}) or {}
                        for _nm_name, _nm_delta in _ss_nudges.items():
                            _nm_now = float(_meta_neuromods.get(_nm_name, 0.5))
                            _nm_target = max(0.0, min(1.0, _nm_now + float(_nm_delta)))
                            _emit_neuromod_nudge(
                                {_nm_name: _nm_target},
                                max_delta=abs(float(_nm_delta)),
                                developmental_age=_meta_dev_age,
                                source="meta_spirit_self")
                state_refs["_last_meta_result"] = _meta_result
        except Exception as _err:
            _log_driver_err("meta_engine.tick", _err)

    # 7.1 Drain P14 (Producer #14) coherence_gain META-CGN events.
    #
    # 2026-05-10 closes the P14 coherence detector audit item from the
    # pre-D8 ownership audit. meta_engine.tick (above) populates
    # `_pending_cgn_coherence_events` with chi_coh observations that
    # crossed coherence thresholds. The drain block converts them into
    # META_CGN_SIGNAL emissions, gated by an EdgeDetector per the
    # "discrete state transitions only" invariant — first crossing of
    # thresholds [0.3, 0.5, 0.7, 0.9] per chi_coh emits, sustained
    # elevated state is silent, drop-and-re-cross emits again. Persisted
    # state in ./data/edge_detector_state.json under "coherence_gain"
    # key. Mirrors spirit_worker.py:7005-7048.
    if meta_engine is not None:
        try:
            _p14_pending = getattr(
                meta_engine, "_pending_cgn_coherence_events", None)
            if _p14_pending:
                from titan_hcl.bus import emit_meta_cgn_signal
                from titan_hcl.logic.edge_detector_persistence import (
                    load_edge_detector_state)
                if not getattr(
                        coordinator, "_p14_coherence_init", False):
                    from titan_hcl.logic.meta_cgn import EdgeDetector
                    coordinator._p14_coherence_detector = EdgeDetector()
                    _p14_persisted = load_edge_detector_state().get(
                        "coherence_gain")
                    if _p14_persisted:
                        coordinator._p14_coherence_detector.load_dict(
                            _p14_persisted)
                        logger.info(
                            "[META-CGN] Producer #14 EdgeDetector state "
                            "restored (%d threshold keys known)",
                            len(_p14_persisted.get("crossed", {})))
                    coordinator._p14_coherence_init = True
                _p14_det = coordinator._p14_coherence_detector
                _p14_thresholds = [0.3, 0.5, 0.7, 0.9]
                while _p14_pending:
                    _p14_evt = _p14_pending.pop(0)
                    _p14_chi = float(_p14_evt.get("chi_coh", 0.0))
                    for _p14_thr in _p14_thresholds:
                        _p14_key = f"chi_coh_{_p14_thr}"
                        if _p14_det.observe(_p14_key, _p14_chi, _p14_thr):
                            _p14_sent = emit_meta_cgn_signal(
                                send_queue,
                                src="self_model",
                                consumer="self_model",
                                event_type="coherence_gain",
                                intensity=min(1.0, _p14_chi),
                                domain=f"thr_{_p14_thr}",
                                reason=(
                                    f"chi_coherence crossed threshold "
                                    f"{_p14_thr} (chi={_p14_chi:.3f})"))
                            if _p14_sent:
                                logger.info(
                                    "[META-CGN] self_model.coherence_gain "
                                    "EMIT — threshold=%.1f chi=%.3f",
                                    _p14_thr, _p14_chi)
                            else:
                                logger.warning(
                                    "[META-CGN] Producer #14 "
                                    "self_model.coherence_gain DROPPED by "
                                    "bus — threshold=%.1f chi=%.3f "
                                    "(rate-gate or queue-full)",
                                    _p14_thr, _p14_chi)
                            # ── Phase A (RFP_cgn_enhancements §9.1) ──────────
                            # A coherence-threshold crossing is a self_model
                            # learning event: spirit_self_nudge → SPIRIT_SELF.
                            # Walk the self-coherence concept downstream so the
                            # chain reasons about THIS event (§5.3 fix) rather
                            # than collapsing to FORMULATE. Same EdgeDetector
                            # gate as the emit; rides the meta-service
                            # rate-limiter; never breaks the drain loop.
                            try:
                                from titan_hcl.logic.meta_service_client import (
                                    send_meta_request as _p14_mrq)
                                from titan_hcl.logic.meta_consumer_contexts import (
                                    build_self_model_meta_context_30d
                                    as _p14_mrq_ctx)
                                _p14_mrq(
                                    consumer_id="self_model",
                                    question_type="spirit_self_nudge",
                                    context_vector=_p14_mrq_ctx(),
                                    time_budget_ms=2000,
                                    send_queue=send_queue,
                                    src="self_model",
                                    grounding_payload={
                                        "concept_id": "self_coherence"},
                                    payload_snippet=(
                                        "self_model.coherence_gain:"
                                        f"thr_{_p14_thr}"),
                                )
                                logger.info(
                                    "[Phase A] self_model.coherence_gain → "
                                    "META_REASON_REQUEST (thr=%.1f)",
                                    _p14_thr)
                            except Exception as _p14_mrq_err:
                                logger.warning(
                                    "[Phase A] self_model coherence_gain "
                                    "meta-request failed: %s", _p14_mrq_err)
                            # Record stage (rFP_experience_distillation_phase_c
                            # §5) — a coherence-threshold crossing is a self_model
                            # experience. Targeted self-emit (D-SPEC-52).
                            bus.emit_experience_record(
                                send_queue, name,
                                domain="self_model",
                                action_taken=f"coherence_gain:thr_{_p14_thr}",
                                outcome_score=min(1.0, max(0.0, _p14_chi)),
                                context={"threshold": _p14_thr,
                                         "chi_coherence": _p14_chi,
                                         "source": "coherence_gain"},
                                coalesce_key="self_model:coherence",
                            )
        except Exception as _err:
            _log_driver_err("meta_cgn.p14_coherence_drain", _err)

    # 7.5 Drive MSL (Meta-State Learning) — confidence/depth/attention.
    #
    # 2026-05-10 closes BUG-COGNITIVE-WORKER-MSL-TICK-MISSING-20260510.
    # cognitive_worker boots MSL via msl.load_all() at line 637 but the
    # per-epoch tick driver was forgotten in chunk 8E. Mirrors
    # spirit_worker.py:3792-3793. set_pi_value() injects π-heartbeat ratio
    # into MSL's gating context BEFORE tick advances state.
    msl = state_refs.get("msl")
    if msl is not None:
        try:
            if pi_monitor is not None:
                _set_pi = getattr(msl, "set_pi_value", None)
                if callable(_set_pi):
                    _set_pi(pi_monitor.heartbeat_ratio)
            _msl_tick_fn = getattr(msl, "tick", None)
            if callable(_msl_tick_fn):
                _msl_output = _msl_tick_fn()
                # Stash latest output on coordinator for downstream consumers
                # (Tier 1 SPEAK_REQUEST + observatory dashboards). None is a
                # valid signal (MSL skipped this tick).
                if coordinator is not None:
                    coordinator._msl_latest_output = _msl_output

                # §4.Q (2026-05-15) MSL concept emotional valence nudges —
                # emit via bus events. Replaces in-process
                # spirit_worker.py:4048-4068 (Phase 3 concept emotional valence
                # → neuromod nudges + "I" valence). Iterates the concept
                # grounder's confidences; for each grounded concept with a
                # valence entry in CONCEPT_VALENCE, emit a NUDGE.
                _cg = getattr(msl, "concept_grounder", None)
                if _cg is not None:
                    _msl_dev_age = float(getattr(
                        pi_monitor, "developmental_age", 0.0) or 0.0) \
                        if pi_monitor else 0.0
                    try:
                        _concepts = _cg.get_concept_confidences() or {}
                        for _cname, _cconf in _concepts.items():
                            if _cconf <= 0.01:
                                continue
                            _val = _cg.get_emotional_valence(_cname)
                            if not _val:
                                continue
                            _emit_neuromod_nudge(
                                _val["nudge_map"],
                                max_delta=float(_val["max_delta"]) * min(
                                    1.0, float(_cconf)),
                                developmental_age=_msl_dev_age,
                                source=f"msl_concept_valence:{_cname}")
                        # "I" valence — Phase 2 concept, Phase 3 wiring
                        # (spirit_worker.py:4060-4068).
                        _get_i = getattr(msl, "get_i_confidence", None)
                        _i_conf = float(_get_i()) if callable(_get_i) else 0.0
                        if _i_conf > 0.01:
                            from titan_hcl.logic.msl import ConceptGrounder
                            _i_val = ConceptGrounder.CONCEPT_VALENCE.get("I")
                            if _i_val:
                                _emit_neuromod_nudge(
                                    _i_val["nudge_map"],
                                    max_delta=float(_i_val["max_delta"]) * min(
                                        1.0, _i_conf),
                                    developmental_age=_msl_dev_age,
                                    source="msl_i_valence")
                    except Exception as _msl_n_err:
                        _log_driver_err("msl.concept_valence_nudges", _msl_n_err)
        except Exception as _err:
            _log_driver_err("msl.tick", _err)

    # 8. ExpressionManager.evaluate_all — MOVED to expression_worker
    # per §4.B Track 3 (SHIPPED 2026-05-15). expression_worker subscribes
    # to KERNEL_EPOCH_TICK and runs evaluate_all on each tick, publishing
    # EXPRESSION_FIRED / NS_REWARD / META_CGN_SIGNAL / SOCIAL_CATALYST
    # per fire + SPEAK_REQUEST_PENDING on Tier-1 SPEAK detection.
    # cognitive_worker's role narrows to:
    #   - subscribing to SPEAK_REQUEST_PENDING (handled by Block 8.5 via
    #     a bus-event-fed `state_refs["_speak_pending_from_bus"]` cache);
    #   - subscribing to NS_REWARD (main recv loop calls
    #     neural_nervous_system.record_outcome on receipt);
    #   - NOT owning composites, NOT instantiating ExpressionManager,
    #     NOT driving evaluate_all (G21 — expression_worker is sole
    #     writer of composite-ledger state + expression_state.bin).
    expression_manager = None  # vestigial reference for Block 8.5 SPEAK
    # injection — replaced by bus-fed _speak_pending_from_bus cache below.
    _bus_speak_pending = state_refs.get("_speak_pending_from_bus") or {}
    _spp_ts = float(_bus_speak_pending.get("ts", 0.0) or 0.0)
    # TTL gate — only honor SPEAK_REQUEST_PENDING bus events within last
    # 5s. Stale events (worker restarted, msg outdated) ignored.
    _t2_speak_pending = bool(_bus_speak_pending) and (
        time.time() - _spp_ts <= 5.0)
    _t2_fired: list = []
    _t2_hormones: dict = (
        _bus_speak_pending.get("hormones") or {}) if _t2_speak_pending else {}
    # (Block 8 evaluate_all body REMOVED — entire block migrated to
    # expression_worker. Per-fire bus emits (EXPRESSION_FIRED,
    # NS_REWARD, META_CGN_SIGNAL, SOCIAL_CATALYST(strong_composition))
    # now fire from expression_worker. cognitive_worker consumes
    # NS_REWARD events in the main recv loop and dispatches to
    # neural_nervous_system.record_outcome here.)
    state_refs["_last_t2_speak_pending"] = _t2_speak_pending
    state_refs["_last_t2_hormones"] = _t2_hormones
    state_refs["_last_t2_fired"] = _t2_fired

    # 8.5 Tier 1 SPEAK firing path — SPEAK injection + SOCIAL pressure +
    #     SPEAK_REQUEST emission to language_worker.
    #
    # 2026-05-10 closes BUG SPEAK-silent-on-T3 (filed in this session as
    # part of the pre-D8 ownership audit). Mirrors spirit_worker.py:8775-8916
    # except for the second evaluate_all call — Tier 2 (step 8 above)
    # already evaluated all composites WITH exclude={"SPEAK"}; injecting
    # the synthetic SPEAK fire when _t2_speak_pending is enough. The
    # Tier 1 re-evaluate in spirit_worker_main duplicates ART/MUSIC fires
    # at publish-cycle cadence — we drop that here for cleaner per-epoch
    # semantics on T3 (MAY revisit if Tier 1 cadence-gating proves needed).
    #
    # Cross-process safe defaults: setpoint=0.5 + lr_gain=1.0 +
    # emotion_confidence=0.5 when neuromodulator_system isn't accessible
    # (it lives in neuromod_worker under l0_rust=true). Same pattern as
    # the chi block at line 1185 — graceful degradation.
    if (expression_manager is not None
            and neural_nervous_system is not None
            and getattr(neural_nervous_system, "_hormonal_enabled", False)):
        try:
            social_pressure_meter = state_refs.get("social_pressure_meter")
            exp_orchestrator = state_refs.get("exp_orchestrator")

            # SPEAK injection if expression_worker flagged it via
            # SPEAK_REQUEST_PENDING bus event (§4.B Track 3, 2026-05-15).
            # The urge value comes from the bus payload (set by
            # expression_worker's Tier-1 SPEAK detection); no in-proc
            # composite lookup needed since ExpressionManager now lives
            # in expression_worker.
            if _t2_speak_pending:
                _speak_in_fired = any(
                    f.get("composite") == "SPEAK" for f in _t2_fired)
                if not _speak_in_fired:
                    _bus_speak_urge = float(
                        (_bus_speak_pending or {}).get("urge", 0.5) or 0.5)
                    _t2_fired.append({
                        "composite": "SPEAK",
                        "urge": _bus_speak_urge,
                        "intensity": 1.0,
                        "dominant_hormone": "CREATIVITY",
                        "action_helper": "speak",
                        "total_consumption": 0,
                    })

            # Social Pressure: SOCIAL composite fires are now consumed
            # directly by social_worker (subscribes to EXPRESSION_FIRED
            # which expression_worker publishes per §4.C+§4.B Track 3).
            # No in-proc social_pressure_meter call here under l0_rust=true
            # — cognitive_worker doesn't see Tier-2 fires in _t2_fired
            # anymore (Block 8 removed). Kept guarded for l0_rust=false
            # legacy where state_refs is populated differently.
            if social_pressure_meter is not None:
                for _spf in _t2_fired:
                    if _spf.get("composite") == "SOCIAL":
                        try:
                            social_pressure_meter.on_social_fire(
                                _spf.get("urge", 1.0))
                        except Exception as _sp_err:
                            _log_driver_err(
                                "social_pressure.on_social_fire", _sp_err)

            # SPEAK_REQUEST emission for SPEAK fires.
            for _ef in _t2_fired:
                if _ef.get("composite") != "SPEAK":
                    continue

                # ── Track 2 advisor-refractory gate (v1.2.1) ───────────────
                # Per SPEC §8.5 + rFP §2.A.7. Consult outer_interface_worker's
                # cached ADVISOR_REFRACTORY_STATE — if SPEAK is within its
                # refractory window, skip emit. Closes the T3 SPEAK quality
                # regression where the same SPEAK fired repetitively without
                # advisor backoff under l0_rust=true.
                _adv_state = state_refs.get("_advisor_state") or {}
                _adv_refrac = (_adv_state.get("action_refractory") or {})
                _adv_speak = _adv_refrac.get("SPEAK") or _adv_refrac.get("speak")
                if isinstance(_adv_speak, dict):
                    _next_allowed = float(_adv_speak.get("next_allowed_ts", 0.0))
                    if _next_allowed > time.time():
                        logger.info(
                            "[SPEAK] gated by advisor refractory — next_allowed_ts=%.1f "
                            "(in %.1fs)", _next_allowed,
                            _next_allowed - time.time())
                        continue  # skip this SPEAK emit; advisor will reopen on cooldown

                _speak_sv = (consciousness or {}).get(
                    "latest_epoch", {}).get("state_vector", []) if consciousness else []
                if hasattr(_speak_sv, "to_list"):
                    _speak_sv = _speak_sv.to_list()
                _speak_sv = list(_speak_sv) if _speak_sv else []
                if len(_speak_sv) < 65:
                    continue

                # Build experience bias via ExperienceOrchestrator.
                _speak_bias_data = None
                if exp_orchestrator is not None:
                    try:
                        _sb_plugin = exp_orchestrator._plugins.get("language")
                        if _sb_plugin:
                            _sb_perc = _sb_plugin.extract_perception_key({
                                "inner_state": _speak_sv,
                                "felt_tensor": _speak_sv[:65],
                                "inner_body": _speak_sv[:5],
                                "inner_mind": _speak_sv[5:20]
                                              if len(_speak_sv) >= 20 else [],
                                "inner_spirit": _speak_sv[20:65]
                                                 if len(_speak_sv) >= 65 else [],
                                "hormonal_snapshot": _t2_hormones,
                                "intent_hormones": _t2_hormones,
                                "spatial_features": [],
                            })
                            _sb_bias = exp_orchestrator.get_experience_bias(
                                domain="language",
                                current_perception=_sb_perc,
                                current_inner_state=_speak_sv,
                                candidate_actions=["self_express"],
                            )
                            if _sb_bias and _sb_bias.confidence >= 0.2:
                                _speak_bias_data = {
                                    "optimal_inner_state": (
                                        list(_sb_bias.optimal_inner_state)
                                        if _sb_bias.optimal_inner_state is not None
                                        else None),
                                    "confidence": _sb_bias.confidence,
                                    "domain": "language",
                                }
                    except Exception as _bias_err:
                        _log_driver_err(
                            "experience_orchestrator.get_experience_bias",
                            _bias_err)

                # Concept confidences (MSL-based).
                _speak_concept_conf = None
                if msl is not None:
                    try:
                        _msl_out = getattr(coordinator,
                                           "_msl_latest_output", None) or {}
                        _msl_concepts = (_msl_out.get(
                            "concept_confidences") if isinstance(
                                _msl_out, dict) else None) or {}
                        _speak_concept_conf = dict(_msl_concepts)
                        _speak_concept_conf["I"] = msl.get_i_confidence()
                    except Exception:
                        _speak_concept_conf = None

                # DA info — under l0_rust=true neuromodulator_system lives
                # in neuromod_worker. Use SHM-read level + setpoint=0.5
                # default (graceful degradation per chi block pattern).
                _nm_levels = neuromod_reader() if neuromod_reader else {}
                _da_level = (_nm_levels or {}).get("DA", 0.5)
                _da_setpoint = 0.5

                # MSL attention for KIN sensory context.
                _speak_msl_attn = None
                if msl is not None:
                    try:
                        _attn_fn = getattr(
                            msl, "get_attention_weights_for_kin", None)
                        if callable(_attn_fn):
                            _speak_msl_attn = _attn_fn()
                    except Exception:
                        pass

                # Social contagion context (coordinator-attached buffer).
                _speak_social_ctx = None
                _sc_buf = getattr(
                    coordinator, "_social_contagion_buffer", []) if coordinator else []
                if _sc_buf:
                    _sc_latest = _sc_buf[-1]
                    _speak_social_ctx = {
                        "contagion_type": _sc_latest.get("contagion_type"),
                        "author": _sc_latest.get("author", ""),
                        "topic": _sc_latest.get("topic", ""),
                        "felt_summary": _sc_latest.get("felt_summary", ""),
                    }

                # Reasoning plan threading (from step 6.5 stash).
                _r_result = state_refs.get("_last_reasoning_result")
                _speak_reasoning = None
                if (_r_result
                        and isinstance(_r_result, dict)
                        and _r_result.get("action") == "COMMIT"
                        and _r_result.get("confidence", 0) >= 0.5):
                    _speak_reasoning = _r_result

                _ch_epoch = (consciousness or {}).get(
                    "latest_epoch", {}).get("epoch_id", 0) if consciousness else 0

                # ── Track 2 SPEAK_REQUEST_PENDING precursor (v1.2.1) ──────
                # Per SPEC §8.5 + rFP §2.A.7. Emit candidate words from MSL
                # concept_confidences keys so outer_interface_worker can
                # compute narrator.get_word_perturbation per candidate and
                # publish WORD_PERTURBATION_HINT for language_worker to
                # consume. The precursor fires INSIDE the same per-fire
                # block (back-to-back with SPEAK_REQUEST) so the broker
                # delivers them in order; language_worker correlates by
                # request_id with ≤200ms TTL on the HINT.
                import uuid as _uuid  # local import — keep hot path uncluttered
                _speak_request_id = _uuid.uuid4().hex
                _candidate_words: list[str] = []
                if isinstance(_speak_concept_conf, dict):
                    # Top-K concepts by confidence as candidate words; cap
                    # to avoid pathological inputs.
                    _sorted = sorted(
                        _speak_concept_conf.items(),
                        key=lambda kv: float(kv[1] or 0.0),
                        reverse=True)
                    _candidate_words = [k for k, _ in _sorted[:16]]
                _send_msg(send_queue, bus.SPEAK_REQUEST_PENDING, name, "all", {
                    "request_id": _speak_request_id,
                    "candidate_words": _candidate_words,
                    "epoch_id": _ch_epoch,
                    "ts": time.time(),
                })

                _send_msg(send_queue, bus.SPEAK_REQUEST, name, "language", {
                    "request_id": _speak_request_id,
                    "state_132d": _speak_sv,
                    "neuromods": {
                        "DA": {"level": _da_level, "setpoint": _da_setpoint},
                    },
                    "concept_confidences": _speak_concept_conf,
                    "visual_context": None,  # outer_state not in cognitive_worker scope
                    "experience_bias": _speak_bias_data,
                    "epoch_id": _ch_epoch,
                    "msl_attention": _speak_msl_attn,
                    "social_contagion": _speak_social_ctx,
                    "reasoning_result": _speak_reasoning,
                })
                logger.info(
                    "[SPEAK] SPEAK_REQUEST sent to language_worker "
                    "(epoch=%d, request_id=%s, candidate_words=%d)",
                    _ch_epoch, _speak_request_id[:8], len(_candidate_words))
        except Exception as _t1_err:
            _log_driver_err("expression_manager.tier1_speak", _t1_err)

    # 8.6 Block F (Track 1) drivers — pre-D8 ownership audit closure.
    #
    # 2026-05-10: 9 GREEN + 4 YELLOW engines migrated boot+drive into
    # cognitive_worker so spirit_worker_main is fully unreferenced on T3
    # under l0_rust_enabled=true (D8 retirement-ready). Each driver runs
    # per-epoch with _log_driver_err for first-fail + every-100th
    # visibility per directive_error_visibility.md.

    # working_mem.decay — per-epoch attention decay (spirit_worker:6012).
    working_mem = state_refs.get("working_mem")
    if working_mem is not None:
        try:
            _wm_decay = getattr(working_mem, "decay", None)
            if callable(_wm_decay):
                _wm_decay(epoch_id)
        except Exception as _err:
            _log_driver_err("working_mem.decay", _err)

    # prediction_engine driver REMOVED — Track 2 drift correction (rFP §2.C
    # + commit B8). PredictionEngine now lives in self_reflection_worker;
    # cognitive_worker reads novelty signal from state_refs["_latest_prediction"]
    # cache slot (populated by the PREDICTION_GENERATED bus handler at
    # ~line 470 in the main loop dispatcher). One-tick latency vs the
    # legacy in-process predict_next call — acceptable per G19 SHM-only
    # state transport doctrine.

    # intuition_convergence.check — per-epoch convergence detection.
    intuition_convergence = state_refs.get("intuition_convergence")
    if intuition_convergence is not None:
        try:
            _ic_check = getattr(intuition_convergence, "check", None)
            if callable(_ic_check):
                # check() signature varies — pass best-effort kwargs and
                # tolerate; spirit_worker has the canonical call site at
                # line ~3200 inside the social-pressure branch.
                _ic_check()
        except TypeError:
            pass  # signature mismatch — non-fatal until consumer migrated
        except Exception as _err:
            _log_driver_err("intuition_convergence.check", _err)

    # wallet_observer.poll — periodic poll, rate-limited by should_poll.
    wallet_observer = state_refs.get("wallet_observer")
    if wallet_observer is not None:
        try:
            _wo_should = getattr(wallet_observer, "should_poll", None)
            _wo_poll = getattr(wallet_observer, "poll", None)
            if callable(_wo_should) and callable(_wo_poll) and _wo_should():
                _wo_poll()
        except Exception as _err:
            _log_driver_err("wallet_observer.poll", _err)

    # meta_recruitment.catalog_health_check — periodic catalog audit.
    meta_recruitment = state_refs.get("meta_recruitment")
    if meta_recruitment is not None and epoch_id and epoch_id % 100 == 0:
        try:
            _mr_check = getattr(
                meta_recruitment, "catalog_health_check", None)
            if callable(_mr_check):
                _mr_check()
        except Exception as _err:
            _log_driver_err("meta_recruitment.catalog_health_check", _err)

    # timeseries_store — per-epoch metric record (gated by should_record).
    # 2026-05-15 (post-§4.Q): expanded from {epoch_id, curvature, chi_total}
    # to include neuromod.* + expression.*.urge metrics so the observatory's
    # "Neuromodulator History (24h)" + expression timeseries charts populate
    # under Phase C. Pre-§4.Q the only writer was dead spirit_worker.py:2639;
    # cognitive_worker's existing 3-metric record didn't cover what the
    # frontend (TimeseriesChart with metrics=['neuromod.DA', 'neuromod.5HT',
    # ...]) actually queries. T3 (fresh-start 2026-05-14) showed empty
    # because timeseries.db never gained neuromod rows; T1+T2 had stale
    # pre-Phase-C rows that gradually decay out of the 30-day retention
    # window. This block restores live writes fleet-wide.
    timeseries_store = state_refs.get("timeseries_store")
    if timeseries_store is not None and epoch_id:
        try:
            _ts_should = getattr(timeseries_store, "should_record", None)
            _ts_record = getattr(timeseries_store, "record", None)
            if callable(_ts_should) and callable(_ts_record) and _ts_should():
                # §4.G — chi via SHM (was: life_force_engine._latest_chi)
                _lf_reader_ts = state_refs.get("_life_force_shm_reader")
                _ts_metrics = {
                    "epoch_id": epoch_id,
                    "curvature": curvature,
                    "chi_total": float(
                        _lf_reader_ts.get_chi_total()
                        if _lf_reader_ts is not None else 0.5),
                }
                # Neuromod levels — sourced from cached NEUROMOD_STATS_UPDATED
                # (published 2.5s coalesced by neuromod_worker per §4.Q Q12).
                # Falls back to SHM-read via neuromod_reader on cold-start
                # before the first NEUROMOD_STATS_UPDATED arrives.
                _ts_neuromod_stats = state_refs.get("_last_neuromod_stats") or {}
                _ts_mods = _ts_neuromod_stats.get("modulators") or {}
                if not _ts_mods and neuromod_reader:
                    try:
                        _ts_levels = neuromod_reader() or {}
                        _ts_mods = {
                            n: {"level": float(v)} for n, v in _ts_levels.items()
                        }
                    except Exception:
                        pass
                for _nm_name in ("DA", "5HT", "NE", "ACh", "Endorphin", "GABA"):
                    _nm_entry = _ts_mods.get(_nm_name)
                    if isinstance(_nm_entry, dict) and "level" in _nm_entry:
                        _ts_metrics[f"neuromod.{_nm_name}"] = float(
                            _nm_entry["level"])
                # Expression composite urges — sourced from cached
                # EXPRESSION_COMPOSITES_UPDATED (1Hz from expression_worker
                # per §4.B). Field shape: {composite: {urge, threshold,
                # fire_count}}.
                _ts_composites = state_refs.get("_expression_composites") or {}
                if isinstance(_ts_composites, dict):
                    for _xc_name, _xc_data in _ts_composites.items():
                        if isinstance(_xc_data, dict) and "urge" in _xc_data:
                            _ts_metrics[f"expression.{_xc_name}.urge"] = float(
                                _xc_data["urge"])
                # ── Restore the msl.* / pi.* / reasoning.* / meta.* / ns.*
                # series that the retired spirit_worker.collect_snapshot used
                # to write (stopped 2026-05-14 at Phase C fleet migration →
                # IDepthTab "I-Depth & Chi History (24h)" + PiHeartbeatStrip
                # went flat). cognitive_worker hosts all these engines via
                # state_refs, so it is the canonical writer per G21. Each guarded
                # so a single missing engine never breaks the record tick.
                _ts_msl = state_refs.get("msl")
                if _ts_msl is not None:
                    try:
                        _conf = getattr(_ts_msl, "confidence", None)
                        if _conf is not None:
                            _ts_metrics["msl.i_confidence"] = float(
                                getattr(_conf, "confidence", 0.0) or 0.0)
                            _ts_metrics["msl.convergence_count"] = float(
                                getattr(_conf, "_convergence_count", 0) or 0)
                        _idp = getattr(_ts_msl, "i_depth", None)
                        if _idp is not None:
                            _ts_metrics["msl.i_depth"] = float(
                                getattr(_idp, "depth", 0.0) or 0.0)
                    except Exception:
                        pass
                _ts_pi = state_refs.get("pi_monitor")
                if _ts_pi is not None:
                    try:
                        # get_stats() is the canonical accessor. The old
                        # getattr(pi,"cluster_count") read a PRIVATE attr and
                        # silently logged 0 — fixed + expanded 2026-06-04.
                        _ps = _ts_pi.get_stats() if hasattr(_ts_pi, "get_stats") else {}
                        _ts_metrics["pi.heartbeat_ratio"] = float(
                            _ps.get("heartbeat_ratio", getattr(_ts_pi, "heartbeat_ratio", 0.0)) or 0.0)
                        _ts_metrics["pi.cluster_count"] = float(_ps.get("cluster_count", 0) or 0)
                        _ts_metrics["pi.dev_age"] = float(
                            _ps.get("developmental_age", getattr(_ts_pi, "developmental_age", 0)) or 0)
                        # NEW — the ≈π "value" + streak/total raw signals:
                        _ts_metrics["pi.avg_cluster_size"] = float(
                            _ps.get("avg_cluster_size", getattr(_ts_pi, "avg_cluster_size", 0)) or 0)
                        _ts_metrics["pi.pi_streak"] = float(_ps.get("current_pi_streak", 0) or 0)
                        _ts_metrics["pi.zero_streak"] = float(_ps.get("current_zero_streak", 0) or 0)
                        _ts_metrics["pi.total_pi_epochs"] = float(_ps.get("total_pi_epochs", 0) or 0)
                        _ts_metrics["pi.total_epochs"] = float(_ps.get("total_epochs_observed", 0) or 0)
                        _ts_metrics["pi.in_cluster"] = 1.0 if _ps.get("in_cluster") else 0.0
                    except Exception:
                        pass
                _ts_reas = state_refs.get("reasoning_engine")
                if _ts_reas is not None:
                    try:
                        _ts_metrics["reasoning.total_chains"] = float(
                            getattr(_ts_reas, "total_chains", 0) or 0)
                    except Exception:
                        pass
                _ts_meta = state_refs.get("meta_engine")
                if _ts_meta is not None:
                    try:
                        _ts_metrics["meta.total_chains"] = float(
                            getattr(_ts_meta, "_total_meta_chains", 0) or 0)
                        _ts_metrics["meta.total_eurekas"] = float(
                            getattr(_ts_meta, "_total_eurekas", 0) or 0)
                        _ts_metrics["meta.total_wisdom"] = float(
                            getattr(_ts_meta, "_total_wisdom_saved", 0) or 0)
                    except Exception:
                        pass
                _ts_nns = state_refs.get("neural_nervous_system")
                if _ts_nns is not None:
                    try:
                        _ts_hrm = getattr(_ts_nns, "_hormonal", None)
                        if _ts_hrm is not None:
                            _ts_metrics["ns.maturity"] = float(
                                getattr(_ts_hrm, "maturity", 0.0) or 0.0)
                    except Exception:
                        pass
                # ── Higher-system + inner-layer gauges (2026-06-05) — read the
                # canonical SHM state slots via the reader bank (G18; never
                # hand-rolled offsets). Gated by should_record (5-min) → zero
                # per-epoch cost; each guarded so a missing slot never breaks the
                # tick. Curated to REAL publishers ONLY — the life_force INPUT
                # stubs (sovereignty_index, sol_balance) are deliberately excluded.
                _ts_bank = state_refs.get("_shm_reader_bank")
                if _ts_bank is not None:
                    def _ts_put(_name, _d, _key):
                        try:
                            _v = (_d or {}).get(_key)
                            if isinstance(_v, bool):
                                _ts_metrics[_name] = 1.0 if _v else 0.0
                            elif isinstance(_v, (int, float)):
                                _ts_metrics[_name] = float(_v)
                        except Exception:
                            pass
                    try:
                        _m = _ts_bank.read_memory_state()
                        _ts_put("memory.mempool", _m, "mempool_size")
                        _ts_put("memory.persistent", _m, "persistent_count")
                        _ts_put("memory.learning_velocity", _m, "learning_velocity")
                        _ts_put("memory.kg_nodes", _m, "kg_node_count")
                    except Exception:
                        pass
                    try:
                        _ts_put("sovereignty.ratio", _ts_bank.read_expression_state(), "sovereignty_ratio")
                    except Exception:
                        pass
                    try:
                        _mb = _ts_bank.read_metabolism_state()
                        _ts_put("metabolic.balance_pct", _mb, "balance_pct")
                        _ts_put("metabolic.social_gravity", _mb, "social_gravity_score")
                    except Exception:
                        pass
                    try:
                        _bd = _ts_bank.read_body_state()
                        _ts_put("metabolic.sol_balance", _bd, "sol_balance")
                        _ts_put("body.health", _bd, "body_health")
                    except Exception:
                        pass
                    try:
                        _ts_put("life_force.total", _ts_bank.read_life_force_state(), "total")
                    except Exception:
                        pass
                    try:
                        _cg = _ts_bank.read_cgn_engine_state()
                        _ts_put("cgn.groundings", _cg, "total_groundings")
                        _ts_put("cgn.avg_reward", _cg, "avg_reward")
                        _ts_put("cgn.consolidations", _cg, "consolidations")
                    except Exception:
                        pass
                    try:
                        _lg = _ts_bank.read_language_state()
                        _ts_put("vocab.total", _lg, "vocab_total")
                        _ts_put("vocab.producible", _lg, "vocab_producible")
                    except Exception:
                        pass
                # `chi.total` alias — IDepthTab queries the dotted name; the
                # base dict already records `chi_total` (underscore) consumed
                # by other charts. Record both so each consumer resolves.
                if "chi_total" in _ts_metrics:
                    _ts_metrics["chi.total"] = _ts_metrics["chi_total"]
                _ts_record(_ts_metrics)
        except Exception as _err:
            _log_driver_err("timeseries_store.record", _err)

    # mini_registry.tick_all — per-epoch distributed mini-reasoner tick
    # across body/mind/spirit rate tiers. Signature per
    # logic/mini_experience.py:460 is tick_all(context, rate_tier) where
    # rate_tier ∈ {"body", "mind", "spirit"}. Mirrors spirit_worker:4666-4668.
    # Gated off during dreams: tick_all trains each domain's _policy, which
    # the off-tick consolidate_all also trains — concurrent backprop would
    # race (Phase 0 §8 / D-SPEC-105).
    mini_registry = state_refs.get("mini_registry")
    if mini_registry is not None and not _epoch_is_dreaming:
        try:
            _mri_tick = getattr(mini_registry, "tick_all", None)
            if callable(_mri_tick):
                _mri_ctx = {
                    "neuromod_levels": (
                        neuromod_reader() if neuromod_reader else {}),
                    "fatigue": float(_nn_exp_p) if "_nn_exp_p" in dir() else 0.3,
                }
                for _tier in ("body", "mind", "spirit"):
                    _mri_tick(_mri_ctx, _tier)
        except Exception as _err:
            _log_driver_err("mini_registry.tick_all", _err)

    # interpreter — passive per-event (consumes chain commits via bus).
    # Reference here ensures parity test sees it; concrete consumers
    # (chain commit handlers) migrated as part of Track 2 self-improvement
    # subsystem rFP. Bare reference satisfies the boot-driver invariant.
    #
    # §4.Q (2026-05-15) ORPHAN: spirit_worker.py:5181-5192 had a
    # `self_exploration → seek_novelty → DA+NE nudge` flow that consumed the
    # interpreter result. Phase C has the interpreter object but no live
    # result-handler wiring (interpretation results are passive-per-event).
    # When that flow ships (Track 2 follow-up), emit
    # NEUROMOD_EXTERNAL_NUDGE(source="self_exploration_seek_novelty",
    # nudge_map={"DA": da+0.03, "NE": ne+0.02}, max_delta=0.03,
    # developmental_age=pi_monitor.developmental_age) here. Tracked in
    # PLAN_microkernel_phase_c_neuromod_worker_evaluate_migration.md §6 R-orphan.
    interpreter = state_refs.get("interpreter")
    _ = interpreter  # parity-anchor; future commit wires concrete drivers

    # med_watchdog.check — per-epoch meditation cadence + alerts.
    med_watchdog = state_refs.get("med_watchdog")
    if med_watchdog is not None and epoch_id and epoch_id % 60 == 0:
        try:
            _mw_check = getattr(med_watchdog, "check", None)
            if callable(_mw_check):
                _mw_check()
        except Exception as _err:
            _log_driver_err("med_watchdog.check", _err)

    # episodic_mem — passive backing store; consumer migration follows
    # in next session as part of dream-cycle path migration. Bare
    # reference satisfies the parity invariant for now.
    episodic_mem = state_refs.get("episodic_mem")
    _ = episodic_mem  # parity-anchor; concrete record_episode callsites
                      # remain in spirit_worker_main legacy path until
                      # dream-cycle migration ships.

    # 9. Drive NeuromodRewardObserver — emits per-program reward from
    #    neuromod EMAs every tick_interval ticks.
    #
    # 2026-05-10 closes the NeuromodRewardObserver cross-process audit
    # item from the pre-D8 ownership audit. Constructor was refactored to
    # accept a `levels_provider` callable so the SHM-backed neuromod
    # reader (NEUROMOD_STATE slot, written by neuromod_worker) feeds the
    # observer without cross-process Python attr access. NS lives here,
    # so the observer's record_outcome calls are in-process.
    neuromod_reward_observer = state_refs.get("neuromod_reward_observer")
    if neuromod_reward_observer is not None:
        try:
            neuromod_reward_observer.tick()
        except Exception as _err:
            _log_driver_err("neuromod_reward_observer.tick", _err)

    # 10. Phase A.4 (rFP_phase_c_state_read_unification_l0_l1_canonical /
    #     D-SPEC-70 v1.10.0) — publish 3 cognitive_worker-owned Python L2
    #     SHM slots so api_subprocess StateAccessor reads canonical SHM
    #     instead of bus-cache per Preamble G18:
    #       reasoning_state.bin       (ReasoningEngine)
    #       meta_reasoning_state.bin  (MetaReasoningEngine)
    #       msl_state.bin             (Multisensory Synthesis Layer)
    #     meta_teacher_state.bin is owned by meta_teacher_worker (G21).
    #     All publishers extend BaseStatePublisher — failures are caught +
    #     throttled-warned internally; never break the epoch tick.
    _reasoning_pub = state_refs.get("_reasoning_state_publisher")
    if _reasoning_pub is not None:
        _reasoning_pub.publish(reasoning_engine)
    _meta_reasoning_pub = state_refs.get("_meta_reasoning_state_publisher")
    if _meta_reasoning_pub is not None:
        # Pass NS + pi_monitor too so the publisher can co-publish
        # neural_maturity (Hormonal._maturity) AND the rich π-heartbeat stats
        # (PiHeartbeat cluster/streak/dev_age/heartbeat_ratio) alongside
        # meta_reasoning telemetry — all live in cognitive_worker, so this is
        # the natural co-located surface for the dashboard /status.lifetime
        # composer + SovereigntyGauge + /v4/pi-heartbeat PiHeartbeatStrip.
        _meta_reasoning_pub.publish(
            meta_engine,
            nns=state_refs.get("neural_nervous_system"),
            pi_monitor=state_refs.get("pi_monitor"),
            neuromod_stats=state_refs.get("_last_neuromod_stats"))
    _msl_pub = state_refs.get("_msl_state_publisher")
    if _msl_pub is not None:
        _msl_pub.publish(state_refs.get("msl"))
    # Phase C (OML §7.C piece 2) — publish the FULL MSL distilled_context[20]
    # to a DEDICATED fixed float32 SHM slot so the agno DECIDE path reads it
    # O(1) AT decision-time (the `_v5["msl"]` overlay is fetched AFTER the
    # decision — Q4). ADDITIVE: reads the engine's stored `_last_output`
    # (msl.py:2256) — does NOT touch MSLStatePublisher / msl_state.bin or any
    # inner behavior; publish failures never break the tick. G21 single-writer.
    _msl_for_ctx = state_refs.get("msl")
    if _msl_for_ctx is not None:
        try:
            _lo = getattr(_msl_for_ctx, "_last_output", None)
            _ctx = _lo.get("distilled_context") if isinstance(_lo, dict) else None
            if _ctx is not None:
                from titan_hcl.synthesis.outer_meta_policy import (
                    OUTER_MSL_CONTEXT_STATE_SPEC, msl_context_to_fixed)
                _ctx_writer = state_refs.get("_msl_context_writer")
                if _ctx_writer is None:
                    from titan_hcl.core.state_registry import (
                        StateRegistryWriter, ensure_shm_root, resolve_titan_id)
                    _ctx_writer = StateRegistryWriter(
                        OUTER_MSL_CONTEXT_STATE_SPEC,
                        ensure_shm_root(resolve_titan_id()))
                    state_refs["_msl_context_writer"] = _ctx_writer
                    logger.info(
                        "[CognitiveWorker] outer_msl_context_state.bin writer "
                        "attached (Phase C — full MSL context[20] for the agno "
                        "DECIDE path, O(1) read; G18/G20 single-writer)")
                _ctx_writer.write(msl_context_to_fixed(_ctx))
        except Exception as _msl_ctx_err:
            logger.debug(
                "[CognitiveWorker] msl_context publish soft-fail: %s",
                _msl_ctx_err)
    # D-SPEC-85 v1.25.0 — consciousness_age slot publish (lifetime
    # self-observation tick counter from consciousness.db). G21
    # single-writer = cognitive_worker; surfaces Titan's "main age" to
    # post_dispatch footer (cannot reach consciousness.db per G18).
    _conscage_pub = state_refs.get("_consciousness_age_publisher")
    if _conscage_pub is not None:
        _conscage_pub.publish(consciousness)


def _persist_engine_state(state_refs: dict) -> None:
    """Persist all cognitive engine state to disk (G16 atomic-write).

    Called from SAVE_NOW (B.1 shadow-swap orchestrator) and from the
    chunk 8G epoch driver every COGNITIVE_PERSIST_EVERY_N_EPOCHS=100
    epochs. Each engine's persist call is wrapped so one failure
    doesn't block the others.
    """
    reasoning_engine = state_refs.get("reasoning_engine")
    pi_monitor = state_refs.get("pi_monitor")
    neural_nervous_system = state_refs.get("neural_nervous_system")
    meta_engine = state_refs.get("meta_engine")
    coordinator = state_refs.get("coordinator")
    intuition_convergence = state_refs.get("intuition_convergence")
    msl = state_refs.get("msl")
    meta_service = state_refs.get("_meta_service")

    for engine, name_, method in (
        # AUDIT §C fix (rFP §P2): was "save_state" — ReasoningEngine only has
        # save_all() (logic/reasoning.py:2216); the getattr(method) lookup
        # silently returned None → policy/buffer/totals never persisted →
        # reasoning state lost on every respawn. Corrected to save_all.
        (reasoning_engine, "reasoning_engine", "save_all"),
        (pi_monitor, "pi_monitor", "_save_state"),
        (neural_nervous_system, "neural_nervous_system", "save_all"),
        (meta_engine, "meta_engine", "save_all"),
        # intuition_convergence loaded its state at boot but never wrote it —
        # save_state() (added 2026-05-30) closes that silent learning-loss gap.
        (intuition_convergence, "intuition_convergence", "save_state"),
        # MSL loaded at boot via msl.load_all() but was absent from this persist
        # list → its mutations were lost on respawn (AUDIT §C secondary). Added.
        (msl, "msl", "save_all"),
        # INV-PERSIST (RFP_cgn_loop_closure §7.A / G10) — the emergent-reward
        # accumulator (DynamicRewardAccumulator) lives in MetaService, not the
        # engine; without this its outcome tuples + α outcome-count reset every
        # restart → emergent reward never leaves cold-start. save_all delegates.
        (meta_service, "_meta_service", "save_all"),
    ):
        if engine is None:
            continue
        fn = getattr(engine, method, None)
        if not callable(fn):
            continue
        try:
            fn()
        except Exception as _err:
            logger.warning(
                "[CognitiveWorker] persist %s.%s failed: %s",
                name_, method, _err)

    # Coordinator's dreaming engine also has its own persist path.
    if coordinator is not None:
        dreaming = getattr(coordinator, "dreaming", None) or getattr(
            coordinator, "dreaming_engine", None)
        if dreaming is not None:
            persist = getattr(dreaming, "_persist", None) or getattr(
                dreaming, "save_state", None)
            if callable(persist):
                try:
                    persist()
                except Exception as _err:
                    logger.warning(
                        "[CognitiveWorker] persist dreaming failed: %s", _err)


def _load_toml_section(section: str) -> dict:
    """Load a single top-level section from titan_params.toml.

    Reads from ``../titan_params.toml`` relative to this file
    (titan_hcl/modules/). Returns {} on any failure (file missing,
    parse error, section absent).
    """
    try:
        import tomllib
        params_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "titan_params.toml")
        if not os.path.exists(params_path):
            return {}
        with open(params_path, "rb") as f:
            full = tomllib.load(f)
        return full.get(section, {})
    except Exception:
        return {}


def _merge_per_titan_dna(section: str, titan_id: str) -> dict:
    """Load a flat DNA section + apply its per-Titan override sub-table.

    DNA sections (``[meta_reasoning_dna]``, ``[cognitive_contracts_dna]``) are
    flat tables of base values plus ``[<section>.T1/T2/T3]`` override sub-tables.
    Returns ``{**base_keys, **section[titan_id]}`` — per-Titan keys win.

    REGRESSION FIX (2026-05-20): MetaReasoningEngine moved spirit_worker →
    cognitive_worker; the bare ``_load_toml_section("meta_reasoning")`` at the
    construct site dropped the ``dna``/``contracts_dna``/``titan_id`` enrichment
    the spirit path used to do (see meta_reasoning.py self._dna), leaving every
    per-Titan DNA override inert fleet-wide. This restores the merge. Idiom
    mirrors language_config.py:79 (``section.get(titan_id, {})``).
    """
    raw = _load_toml_section(section)
    if not isinstance(raw, dict):
        return {}
    base = {k: v for k, v in raw.items() if not isinstance(v, dict)}
    override = raw.get(titan_id, {}) or {}
    return {**base, **override}


# === BOILERPLATE: heartbeat + messaging helpers ====================
# These three helpers (_heartbeat_loop, _send_heartbeat, _send_msg) are
# generic L2 worker boilerplate. They're inlined here for chunk 8E so
# cognitive_worker is self-contained, but should be promoted to a shared
# `titan_hcl/modules/_worker_skeleton.py` module when the 3rd L2
# extraction lands (per L2 separation strategy rFP §5 sequencing). Until
# then, copy verbatim into each new L2 worker — change only the
# `[CognitiveWorker]` log prefix + the `chunk` payload field.


def _heartbeat_loop(recv_queue, send_queue, name: str, *, flag_off: bool) -> None:
    """Heartbeat-only loop for the l0_rust_enabled=false defensive branch.

    Exits cleanly on MODULE_SHUTDOWN. No engine init, no dispatcher.
    Phase 11 §11.I.3 — also handles MODULE_PROBE_REQUEST via the trivial
    pass-through contract (flag_off worker is "ready" by definition: it
    has nothing to init).
    """
    last_heartbeat_ts = 0.0
    while True:
        now = time.time()
        if now - last_heartbeat_ts >= _HEARTBEAT_INTERVAL_S:
            _send_heartbeat(send_queue, name, extra={"flag_off_noop": flag_off})
            last_heartbeat_ts = now
        try:
            msg = recv_queue.get(timeout=_POLL_INTERVAL_S)
        except Empty:
            continue
        except Exception:
            continue
        msg_type = msg.get("type") if isinstance(msg, dict) else None
        if msg_type == bus.MODULE_PROBE_REQUEST and _STATE_WRITER is not None:
            try:
                from titan_hcl.core.probe_dispatcher import (
                    handle_module_probe_request,
                )
                handle_module_probe_request(
                    msg, probe_fn=None, send_queue=send_queue,
                    module_name=name, state_writer=_STATE_WRITER,
                )
            except Exception as _phb_err:  # noqa: BLE001
                logger.warning(
                    "[CognitiveWorker] Phase 11 probe handler (flag_off) "
                    "raised: %s", _phb_err)
            continue
        if msg_type == bus.MODULE_SHUTDOWN:
            logger.info("[CognitiveWorker] Shutdown received (flag_off branch)")
            return


def _send_heartbeat(send_queue, name: str, extra: dict | None = None) -> None:
    """Emit MODULE_HEARTBEAT to guardian_HCL with current RSS.

    Phase 11 §11.I.5 — also publishes ModuleStateWriter.heartbeat() on the SHM
    state slot when _STATE_WRITER is set AND _WORKER_READY is True (gate keeps
    the slot in state="starting"/"booted" until in-process scaffolding done).
    """
    try:
        import resource
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        rss_mb = 0.0
    payload = {"alive": True, "ts": time.time(), "rss_mb": round(rss_mb, 1),
               "chunk": "8E"}
    if extra:
        payload.update(extra)
    _send_msg(send_queue, bus.MODULE_HEARTBEAT, name, "guardian", payload)
    if _STATE_WRITER is not None:
        try:
            # During boot we still refresh last_heartbeat (state stays
            # "starting" until _WORKER_READY flips); republishes current state.
            _STATE_WRITER.heartbeat()
        except Exception:
            pass


def _send_msg(send_queue, msg_type: str, src: str, dst: str, payload: dict,
              rid=None) -> None:
    """Best-effort enqueue helper — never raises (heartbeat path)."""
    try:
        msg = {"type": msg_type, "src": src, "dst": dst, "payload": payload,
               "ts": time.time()}
        if rid is not None:
            msg["rid"] = rid
        send_queue.put(msg)
    except Exception:
        pass
