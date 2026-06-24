"""
CGN Cognitive Kernel Worker — Guardian-supervised process.

Owns the authoritative CGN instance: V(s) shared value net, per-consumer Q(s,a)
action nets, replay buffer, HAOV trackers, Sigma micro-updates, SOAR impasse
detection. Propagates weights to consumers via /dev/shm (<1ms latency).

Consumers communicate via bus messages:
  CGN_TRANSITION      — experience from any consumer (buffered here)
  CGN_REGISTER        — new consumer registration
  CGN_CONSOLIDATE     — dream phase trigger (full training)
  CGN_KNOWLEDGE_REQ   — route to knowledge worker
  CGN_SURPRISE        — cross-consumer surprise event
  CGN_HAOV_VERIFY_RSP — verification result from consumer
  CGN_INFERENCE_REQ   — policy inference request (for processes without client)

See: titan-docs/rFP_cgn_cognitive_kernel_v2.md
"""

import logging
import os
import sys
import threading
import time
from queue import Empty
from titan_hcl.utils.silent_swallow import swallow_warn
from titan_hcl import bus
from titan_hcl.core.module_error_handler import with_error_envelope
from titan_hcl.errors import Severity as _phase11_sev

logger = logging.getLogger(__name__)


# Phase 11 §11.I.3 / §11.I.5 (Chunk 11N) — module-level readiness sentinel.
# Flipped True after CGN load + SHM writer init complete. Gates SHM
# slot heartbeat: while False, the slot stays at state="starting"/"booted"
# so titan_hcl's 1Hz SHM poll + MODULE_PROBE_REQUEST dispatcher see real
# liveness rather than a boot-time "subscribed-but-not-warm" lie.
from titan_hcl.modules._heartbeat_grace import (
    boot_deadline_from_now, shm_heartbeat_allowed,
)
from titan_hcl.params import load_titan_params

_WORKER_READY: bool = False
_BOOT_DEADLINE = None  # boot-grace deadline (monotonic); None=no grace


# ─────────────────────────────────────────────────────────────────────
# CODE-AUTHORITATIVE CONSUMER MANIFEST (A5 — 2026-04-21)
# ─────────────────────────────────────────────────────────────────────
# The authoritative set of CGN consumers this codebase expects to exist.
# Named consumers present on disk (cgn_state.pt) but NOT in this set
# trigger a WARN at load — this is schema drift and the root cause of
# BUG-CGN-SILENT-UNREGISTERED-CONSUMER (cross-Titan divergence where
# T1 carried legacy "language"/"social" consumers that T2/T3 never had).
#
# Registration paths per consumer:
#   - Statically pre-registered below: reasoning, self_model, coding,
#     reasoning_strategy, emotional, language, social
#   - Dynamically via CGN_REGISTER bus message: knowledge, meta
#
# Adding a new consumer MUST be done in this manifest AND in a static
# pre-registration (or by a CGN_REGISTER from its worker). Adding only
# to disk (via record_outcome from an unregistered name) is exactly
# the bug this manifest prevents.
#
# See: memory/project_cgn_as_higher_state_registry.md (CGN V-layer role
# = L2 analog of StateRegistry/TitanVM at L0/L1 — the registry must
# have code-authoritative schema, not disk-authoritative).
CODE_AUTHORITATIVE_CONSUMERS = frozenset({
    "reasoning",
    "self_model",
    "coding",
    "reasoning_strategy",
    "emotional",
    "language",
    "social",
    "knowledge",
    "meta",
    # Inner↔Outer Felt-Teaching Bridge §7.4 — the felt_teaching_worker consumer
    # (RFP_inner_outer_felt_teaching_bridge). Pre-registered below + self-registers
    # via CGN_REGISTER; listed here so it is code-authoritative (no schema-drift warn).
    "felt_teaching",
})


# H.2/H.3 (2026-04-28): module-level dest map shared between the
# CGN_TRANSITION outcome handler and the periodic test pump (_run_haov_pump).
# Routes CGN_HAOV_VERIFY_REQ to the worker that owns the consumer's
# verifier branch. All 9 registered consumers covered (zero silent drops).
# HAOV verify routing — ONLY the consumers with a live specialist verifier for
# their topic-ful CONCEPT-GROUNDING hypotheses. IMPASSE-sourced hypotheses
# (the dominant class, all consumers) are verified in-process by
# `cgn.verify_impasse_resolution` (see `_local_haov_verify`) and never routed
# here. Per rFP_haov_efficacy_closure (F4): the 7 prior `"spirit"` entries
# black-holed verification after the 2026-05-16 spirit retirement (spirit has no
# CGN_HAOV_VERIFY_REQ subscriber) — they are removed. Consumers absent from this
# map fall through to in-process verification. `dreaming` was a phantom entry
# (never registered) and is dropped.
_HAOV_DEST_MAP = {
    "language": "language",
    "knowledge": "knowledge",
    "emotional": "emot_cgn",
}


def _local_haov_verify(cgn, tracker, consumer):
    """Verify a tracker's active HAOV test IN-PROCESS when no live bus verifier
    applies. Returns True if handled here (caller must NOT emit a bus verify).

    Two cases verify locally (per rFP_haov_efficacy_closure, Option A):
      1. IMPASSE-sourced hypotheses (`source == "soar_impasse"`, any consumer) —
         verified by `cgn.verify_impasse_resolution` (reward-trend resolution).
         This is the dominant class and the fix for F4 (dead `"spirit"` routing).
      2. CONCEPT-GROUNDING hypotheses for consumers with no specialist verifier
         (not in `_HAOV_DEST_MAP`) — fall back to a reward-based check instead of
         black-holing. Consumers WITH a specialist verifier
         (language/knowledge/emot_cgn) return False so the caller routes via bus.
    """
    at = tracker._active_test
    if not at or not isinstance(at, dict):
        return False
    h = at.get("hypothesis")
    if h is None:
        return False
    if getattr(h, "source", "") == "soar_impasse":
        imp_type = ""
        if isinstance(getattr(h, "action_context", None), dict):
            imp_type = h.action_context.get("impasse_type", "")
        if not imp_type and "_impasse_" in getattr(h, "rule", ""):
            imp_type = h.rule.split("_impasse_", 1)[1]
        _confirmed, reward = cgn.verify_impasse_resolution(consumer, imp_type)
        tracker.verify(getattr(h, "action_context", {}), {}, reward)
        return True
    if consumer in _HAOV_DEST_MAP:
        return False  # live specialist verifier — caller routes via bus
    # Concept-grounding hypothesis, no specialist verifier → reward-based local.
    pre = at.get("pre_observation", {}) or {}
    reward = float(pre.get("reward", 0.0)) if isinstance(pre, dict) else 0.0
    tracker.verify(getattr(h, "action_context", {}), {}, reward)
    return True


def _run_haov_pump(cgn, send_queue, name, stuck_timeout_s):
    """H.3 periodic HAOV test pump.

    Walks all registered HAOV trackers and:
      1. Expires _active_test entries older than stuck_timeout_s
         (recovery from lost CGN_HAOV_VERIFY_RSP — pre-H.2 routing
         silently dropped messages, leaving trackers permanently
         unable to test).
      2. Calls select_test() to attempt a new test (decoupled from
         per-consumer outcome events — fixes Defect 4).
      3. Routes successful selects via _HAOV_DEST_MAP and emits
         CGN_HAOV_VERIFY_REQ.

    Returns count of (expired, attempted, sent) for telemetry.
    """
    expired = 0
    attempted = 0
    sent = 0
    now = time.time()
    for consumer_name, tracker in cgn._haov_trackers.items():
        try:
            # 1. Expire stuck active_test
            at = tracker._active_test
            if at and isinstance(at, dict) and at.get("ts"):
                if now - float(at["ts"]) > stuck_timeout_s:
                    tracker._active_test = None
                    expired += 1

            # 2. Attempt a new test
            attempted += 1
            test_ctx = tracker.select_test({"available_actions": []})
            if not test_ctx:
                continue

            # 2b. In-process verification — impasse hypotheses (any consumer) +
            # consumers with no specialist verifier. Fixes F4 (dead "spirit"
            # routing) per rFP_haov_efficacy_closure: no bus round-trip, no
            # dead dst. Counts as a completed verification.
            if _local_haov_verify(cgn, tracker, consumer_name):
                continue

            # 3. Concept-grounding hypothesis → route to a LIVE specialist verifier.
            dest = _HAOV_DEST_MAP.get(consumer_name)
            if dest is None:
                continue  # no specialist verifier; local path handles these
            if tracker._active_test is not None:
                tracker._active_test["ts"] = now

            # Pre-emit guard for consumer="knowledge": knowledge_worker
            # expects test_ctx.topic/concept to query knowledge_concepts.
            # As of 2026-05-22 the causal-pattern path (cgn.py record_outcome)
            # populates action_context.topic = concept_id for the knowledge
            # consumer, so pattern-mined knowledge hypotheses now carry a topic
            # and pass this guard (closes BUG-CGN-KNOWLEDGE-HAOV-NO-TOPIC-FIELD).
            # The guard stays as belt-and-suspenders: impasse-path hypotheses
            # (hypothesize_from_impasse → action_context={"impasse_type": ...})
            # genuinely have no concept to verify, so we still skip those rather
            # than emit a verify the consumer can't process.
            if consumer_name == "knowledge":
                _has_topic = bool(
                    isinstance(test_ctx, dict) and
                    (test_ctx.get("topic") or test_ctx.get("concept")))
                if not _has_topic:
                    logger.debug(
                        "[CGNWorker] HAOV verify skipped for knowledge: "
                        "no topic/concept in test_ctx — hypothesis lacks "
                        "topic field (rFP follow-up needed)")
                    continue

            _send_msg(send_queue, bus.CGN_HAOV_VERIFY_REQ, name, dest, {
                "consumer": consumer_name,
                "test_ctx": test_ctx,
                "obs_before": {},  # no outcome context — pump-driven
                "hypothesis": (
                    tracker._active_test["hypothesis"].rule
                    if tracker._active_test else ""),
            })
            sent += 1
        except Exception as _pump_err:
            logger.debug(
                "[CGNWorker] HAOV pump error for '%s': %s",
                consumer_name, _pump_err)
    return (expired, attempted, sent)


@with_error_envelope(module_name="cgn", subsystem="entry", severity=_phase11_sev.FATAL)
def cgn_worker_main(recv_queue, send_queue, name: str, config: dict) -> None:
    """Main loop for the CGN Cognitive Kernel process.

    Args:
        recv_queue: receives messages from DivineBus (bus->worker)
        send_queue: sends messages back to DivineBus (worker->bus)
        name: module name ("cgn")
        config: dict from [cgn] config section
    """
    # Phase 11 §11.I.5 (Chunk 11N) — module-level readiness flag reset on
    # every entry (fork-mode re-entries inherit parent's True; spawn-mode
    # re-spawns get fresh False; explicit reset covers both).
    global _WORKER_READY, _BOOT_DEADLINE
    _WORKER_READY = False
    _BOOT_DEADLINE = boot_deadline_from_now()

    project_root = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    logger.info("[CGNWorker] Initializing Cognitive Kernel...")
    init_start = time.time()

    # ── Phase 11 §11.I.5 / Chunk 11N — SHM state-slot writer (G21 per worker) ──
    # Constructed BEFORE slow CGN init so the slot publishes state="starting"
    # immediately. state_writer.heartbeat() (via _send_heartbeat below) keeps
    # the slot's last_heartbeat fresh during init so guardian_hcl's staleness
    # detector doesn't kill the worker mid-boot. Transition to "booted" happens
    # after CGN load + SHM writer init completes (search for _state_writer.write_state("booted")).
    _state_writer = None
    try:
        from titan_hcl.core.module_state import (
            BootPriority,
            ModuleStateWriter,
        )
        _state_writer = ModuleStateWriter(
            module_name="cgn",
            layer="L2",
            boot_priority=BootPriority.OPTIONAL_POST_BOOT,
        )
        _state_writer.write_state("starting")
    except Exception as _sw_err:  # noqa: BLE001
        logger.warning(
            "[CGNWorker] Phase 11 ModuleStateWriter init failed "
            "(continuing on legacy path): %s", _sw_err)

    # ── Load CGN ────────────────────────────────────────────────────────
    from titan_hcl.logic.cgn import (
        ConceptGroundingNetwork, CGNConsumerConfig, CGNTransition)
    from titan_hcl.logic.cgn_shm_protocol import ShmWeightWriter

    state_dir = config.get("state_dir", "data/cgn")
    db_path = config.get("db_path", "data/inner_memory.db")
    # Microkernel v2 §A.2 part 2 (S4): shm_path retained for explicit-literal
    # escape hatch. When config supplies the canonical default path, we let
    # the dual-mode resolver pick legacy vs StateRegistry per the
    # shm_cgn_format_alignment_enabled flag.
    shm_path = config.get("shm_path", "/dev/shm/cgn_live_weights.bin")

    # Load HAOV config with per-Titan profile overrides.
    #
    # 2026-04-21: migrated from direct `tomllib.load(titan_params.toml)` to
    # `config_loader.load_titan_config()` so that HAOV values go through the
    # 3-layer merge (titan_params.toml < config.toml < ~/.titan/secrets.toml).
    # This is the CGN-scoped subset of BUG-CONFIG-LOADER-MERGE-TITAN-PARAMS —
    # ~15-20 direct tomllib.load call sites exist repo-wide, but only this
    # one plus a handful in meta_reasoning are CGN/meta-reasoning adjacent.
    # meta_reasoning.py and meta_cgn.py already take their config via the
    # `cfg` parameter (no direct load), so this is the only CGN/meta
    # module that needed the migration.
    haov_config = {}
    try:
        import json as _hc_json
        _params = load_titan_params()
        _haov_raw = _params.get("cgn", {}).get("haov", {})
        # Base config: all non-dict values from [cgn.haov]
        haov_config = {k: v for k, v in _haov_raw.items()
                       if not isinstance(v, dict)}
        # Per-Titan overrides: merge from [cgn.haov.profiles.{titan_id}]
        _tid_path = os.path.join(project_root, "data", "titan_identity.json")
        if os.path.exists(_tid_path):
            with open(_tid_path) as _tid_f:
                _titan_id = _hc_json.load(_tid_f).get("titan_id", "T1")
        else:
            _titan_id = "T1"
        _profiles = _haov_raw.get("profiles", {})
        _titan_profile = _profiles.get(_titan_id, {})
        haov_config.update(_titan_profile)
        logger.info("[CGNWorker] HAOV config for %s: %s", _titan_id, haov_config)
    except Exception as _hc_err:
        logger.info("[CGNWorker] Using default HAOV config: %s", _hc_err)

    # H.4 Phase 1 (v1) — causal generator config from titan_params.
    # Flag-default-false; enables per-consumer pattern miner that proposes
    # causal hypotheses to the existing HAOV machinery.  Design lock:
    # titan-docs/rFP_cgn_consolidated.md §2.9.
    causal_generator_config: dict = {}
    try:
        _cg_cfg = config.get("causal_generator", {}) or {}
        causal_generator_config = {
            "enabled": bool(_cg_cfg.get("enabled", False)),
            "defaults": {
                "window_size": int(_cg_cfg.get("window_size", 30)),
                "min_n": int(_cg_cfg.get("min_n", 5)),
                "magnitude_threshold": float(_cg_cfg.get("magnitude_threshold", 0.05)),
                "anti_pattern_enabled": bool(_cg_cfg.get("anti_pattern_enabled", True)),
                "staleness_decay_per_tick": float(
                    _cg_cfg.get("staleness_decay_per_tick", 0.999)),
                "stale_grace_ticks": int(
                    _cg_cfg.get("stale_grace_ticks", 30)),
            },
            "per_consumer": dict(_cg_cfg.get("per_consumer", {}) or {}),
        }
        logger.info(
            "[CGNWorker] Causal generator config: enabled=%s, defaults=%s, "
            "per_consumer=%s",
            causal_generator_config["enabled"],
            causal_generator_config["defaults"],
            sorted(causal_generator_config["per_consumer"].keys()),
        )
    except Exception as _cg_err:
        logger.info("[CGNWorker] Using default causal_generator config: %s", _cg_err)
        causal_generator_config = {"enabled": False}

    # Canonical IQL learner config (RFP_cgn_canonical_iql_upgrade).
    # Flag-default-false: legacy advantage-REINFORCE until soak-green.
    iql_config: dict = {}
    try:
        iql_config = {
            "enabled": bool(config.get("cgn_iql_enabled", False)),
            "tau": float(config.get("cgn_iql_tau", 0.7)),
            "beta": float(config.get("cgn_iql_beta", 3.0)),
            "adv_clip": float(config.get("cgn_iql_adv_clip", 100.0)),
            "polyak": float(config.get("cgn_iql_polyak", 0.005)),
            "gamma": float(config.get("cgn_iql_gamma", 0.99)),
        }
        logger.info("[CGNWorker] IQL learner: enabled=%s (tau=%.2f beta=%.2f "
                    "gamma=%.2f polyak=%.4f)", iql_config["enabled"],
                    iql_config["tau"], iql_config["beta"],
                    iql_config["gamma"], iql_config["polyak"])
    except Exception as _iq_err:
        logger.info("[CGNWorker] Using default (disabled) IQL config: %s", _iq_err)
        iql_config = {"enabled": False}

    cgn = ConceptGroundingNetwork(
        db_path=db_path, state_dir=state_dir, haov_config=haov_config,
        causal_generator_config=causal_generator_config,
        iql_config=iql_config)
    # _load_state() called in __init__

    # RFP_supervision_lifecycle §7.D / Phase D.1 — bus-INDEPENDENT save of the
    # CGN's learned IQL state on any shutdown (SIGTERM/control-group/PDEATHSIG).
    # _save_state is idempotent + atomic (also called periodically in-loop).
    from titan_hcl.core.worker_shutdown import register_shutdown_save
    register_shutdown_save(name, lambda: cgn._save_state())

    # ── Pre-register "reasoning" consumer for ARC (runs as standalone script) ──
    # §7.D-A2 NOTE: this consumer is ARC-GEOMETRIC (grid ops below) and fed ONLY
    # by the ARC standalone session (arc/session.py get_haov("reasoning")). It is
    # dormant on the live fleet BY DESIGN — the live reasoning faculty is
    # represented by `meta` (meta-reasoning chains) + `reasoning_strategy` (wired
    # IQL-complete from chain commits). Not a wiring bug; do not "wake" it without
    # ARC. (The stray language reverse-feedback emit was retired, not re-homed.)
    if "reasoning" not in cgn._consumers:
        cgn.register_consumer(CGNConsumerConfig(
            name="reasoning",
            feature_dims=30,
            action_dims=8,
            action_names=["explore", "exploit", "rotate", "flip",
                          "translate", "scale", "color_map", "compose"],
            reward_source="arc_episode_score",
            max_buffer_size=500,
            consolidation_priority=3,
        ))
        logger.info("[CGNWorker] Pre-registered 'reasoning' consumer for ARC")

    # ── Pre-register "self_model" consumer for Self-Reasoning ──
    if "self_model" not in cgn._consumers:
        cgn.register_consumer(CGNConsumerConfig(
            name="self_model",
            feature_dims=30,
            action_dims=6,
            action_names=["deepen_awareness", "predict_transition",
                          "compare_profile", "cross_reference",
                          "consolidate_identity", "kin_reflect"],
            reward_source="prediction_accuracy",
            max_buffer_size=500,
            consolidation_priority=2,
        ))
        logger.info("[CGNWorker] Pre-registered 'self_model' consumer for Self-Reasoning")

    # ── Pre-register "coding" consumer for Self-Directed Development ──
    if "coding" not in cgn._consumers:
        cgn.register_consumer(CGNConsumerConfig(
            name="coding",
            feature_dims=30,
            action_dims=6,
            action_names=["decompose", "abstract", "implement",
                          "test", "refactor", "compose"],
            reward_source="sandbox_execution_success",
            max_buffer_size=500,
            consolidation_priority=3,
        ))
        logger.info("[CGNWorker] Pre-registered 'coding' consumer for Self-Directed Development")

    # ── Pre-register "reasoning_strategy" consumer (rFP α §2b) ──
    # Distinct from "reasoning" (ARC-geometric) — this one grounds reasoning
    # *strategy templates* (primitive sequences that led to successful COMMIT).
    # Emitted by spirit_worker from reasoning.py _conclude_chain when
    # outcome_score >= cgn_emission_threshold (default 0.55).
    # Action space is N/A (consumer-only in v1) — action_dims=1 as placeholder.
    if "reasoning_strategy" not in cgn._consumers:
        cgn.register_consumer(CGNConsumerConfig(
            name="reasoning_strategy",
            feature_dims=30,
            action_dims=1,
            action_names=["observe"],
            reward_source="reasoning_chain_outcome",
            max_buffer_size=500,
            consolidation_priority=2,
        ))
        logger.info("[CGNWorker] Pre-registered 'reasoning_strategy' consumer for rFP α")

    # ── Pre-register "emotional" consumer (rFP_emot_cgn_v2, 8th CGN consumer) ──
    # Grounds 8 emotion primitives as CGN concepts so META-CGN HAOV hypotheses
    # involving emotional states work via the shared V(s) landscape (enables
    # Abstract Modelling → Abstract Thinking → True Creativity arc per
    # rFP_cgn_orchestrator_promotion.md §4 + §9).
    # EmotCGNConsumer also sends CGN_REGISTER dynamically at init (idempotent) —
    # pre-registration is a safety net against restart-order gaps.
    if "emotional" not in cgn._consumers:
        cgn.register_consumer(CGNConsumerConfig(
            name="emotional",
            feature_dims=30,
            # RFP_affective_grounding_loop §7.D (D.4) — 8→9: MAKER_PRESENCE appended
            # (matches emotion_cluster.EMOT_PRIMITIVES order = EMOT_PRIMITIVE_INDEX,
            # which the CGN transition's action_idx uses). On an EXISTING Titan this
            # static block is skipped (consumer already restored) — the dynamic
            # CGN_REGISTER(9) from emot_cgn grows the loaded 8-net 8→9 (zero-init
            # append, learned weights preserved); fresh installs register at 9 here.
            action_dims=9,
            action_names=[
                "FLOW", "IMPASSE_TENSION", "RESOLUTION",
                "PEACE", "CURIOSITY", "GRIEF", "WONDER", "LOVE",
                "MAKER_PRESENCE",
            ],
            # v1 uses chain terminal_reward as reward signal. True
            # emotional_coherence metric (e.g., 1 - variance(V_across_primitives))
            # is TUNING-EMOT-COHERENCE in TUNING_DATABASE.md, v1.6+.
            reward_source="terminal_reward",
            max_buffer_size=500,
            consolidation_priority=2,
        ))
        logger.info("[CGNWorker] Pre-registered 'emotional' consumer (rFP_emot_cgn_v2)")

    # ── Pre-register "language" consumer (A5 — 2026-04-21) ──
    # Shapes match T1's historical registration (feature_dims=30, action_dims=8,
    # action_names below) — verified by reading T1 cgn_state.pt during audit.
    # Fixes BUG-CGN-SILENT-UNREGISTERED-CONSUMER — language_worker's
    # CGN_TRANSITION sends for "language" were silently dropped on T2/T3
    # because the consumer was never registered there. T1 carried it in
    # disk state from a prior codebase version.
    if "language" not in cgn._consumers:
        cgn.register_consumer(CGNConsumerConfig(
            name="language",
            feature_dims=30,
            action_dims=8,
            action_names=["reinforce", "explore", "differentiate",
                          "consolidate", "associate", "dissociate",
                          "deepen", "stabilize"],
            reward_source="language_chain_outcome",
            max_buffer_size=500,
            consolidation_priority=2,
        ))
        logger.info("[CGNWorker] Pre-registered 'language' consumer (A5 cross-Titan symmetry)")

    # ── Pre-register "social" consumer (A5 — 2026-04-21) ──
    # Shapes match T1's historical registration. Same fix pattern as language.
    if "social" not in cgn._consumers:
        cgn.register_consumer(CGNConsumerConfig(
            name="social",
            feature_dims=30,
            action_dims=6,
            action_names=["engage_warmly", "engage_cautiously",
                          "respond_briefly", "disengage",
                          "deepen_bond", "protect"],
            reward_source="engagement_reciprocity",
            max_buffer_size=500,
            consolidation_priority=2,
        ))
        logger.info("[CGNWorker] Pre-registered 'social' consumer (A5 cross-Titan symmetry)")

    # ── Pre-register "felt_teaching" consumer ──
    # Inner↔Outer Felt-Teaching Bridge §7.4 (RFP_inner_outer_felt_teaching_bridge):
    # the felt_teaching_worker feeds outer→inner felt-grounding evidence as a FULL
    # CGN consumer (value_net + IQL participant) — but propose-only (record_outcome
    # only; never writes a grounding). Pre-registered here as the safety net so its
    # CGN_TRANSITION sends are not silently dropped (BUG-CGN-SILENT-UNREGISTERED-
    # CONSUMER); the worker also sends CGN_REGISTER at init.
    if "felt_teaching" not in cgn._consumers:
        cgn.register_consumer(CGNConsumerConfig(
            name="felt_teaching",
            feature_dims=30,
            action_dims=8,
            action_names=["reinforce", "explore", "differentiate",
                          "consolidate", "associate", "dissociate",
                          "deepen", "stabilize"],
            reward_source="felt_grounding_evidence",
            max_buffer_size=500,
            consolidation_priority=2,
        ))
        logger.info("[CGNWorker] Pre-registered 'felt_teaching' consumer "
                    "(Inner↔Outer Felt-Teaching Bridge §7.4)")

    # ── Audit: disk consumers vs code-authoritative manifest (A5 — 2026-04-21) ──
    # After all static pre-registrations complete, cross-check: any consumer
    # loaded from cgn_state.pt that isn't in CODE_AUTHORITATIVE_CONSUMERS
    # indicates schema drift (code was updated to drop a consumer, but disk
    # still carries it). Dynamic consumers (knowledge, meta) register via
    # CGN_REGISTER bus msg later — they're expected-absent here and in manifest.
    try:
        disk_consumers = set(cgn._consumers.keys())
        unexpected = disk_consumers - CODE_AUTHORITATIVE_CONSUMERS
        missing_static = (
            CODE_AUTHORITATIVE_CONSUMERS
            - disk_consumers
            - {"knowledge", "meta"}  # Dynamic — registered later via CGN_REGISTER
        )
        if unexpected:
            logger.warning(
                "[CGNWorker] Consumer schema drift — on disk but NOT in "
                "CODE_AUTHORITATIVE_CONSUMERS manifest: %s. This is the "
                "BUG-CGN-SILENT-UNREGISTERED-CONSUMER pattern. Decide: "
                "add to manifest OR purge from disk via controlled reset.",
                sorted(unexpected))
        if missing_static:
            logger.warning(
                "[CGNWorker] Consumers in manifest expected statically but "
                "not yet registered: %s. This should not happen post-A5 "
                "boot — investigate pre-registration block above.",
                sorted(missing_static))
        logger.info(
            "[CGNWorker] Consumer manifest audit: %d registered, "
            "%d unexpected, %d expected-but-absent-static.",
            len(disk_consumers), len(unexpected), len(missing_static))
    except Exception as _audit_err:
        logger.warning("[CGNWorker] Consumer manifest audit failed: %s",
                       _audit_err)

    # ── Initialize /dev/shm weight writer (S4 dual-mode) ───────────────
    # Pass titan_id + full config so the dual-mode resolver can pick
    # legacy (16B header, global path) vs StateRegistry (24B header,
    # per-titan path) based on shm_cgn_format_alignment_enabled flag.
    # Explicit non-default shm_path (test override) still honored.
    _cgn_titan_id = _titan_id  # resolved above from data/titan_identity.json
    _shm_arg = shm_path if shm_path != "/dev/shm/cgn_live_weights.bin" else None
    shm_writer = ShmWeightWriter(
        shm_path=_shm_arg, titan_id=_cgn_titan_id, config=config)
    _write_full_shm(cgn, shm_writer)
    logger.info("[CGNWorker] SHM weights written to %s (v=%d, sr=%s)",
                shm_writer._path, shm_writer.get_version(),
                shm_writer._use_stateregistry)

    # ── Stats ──────────────────────────────────────────────────────────
    _stats = {
        "transitions_received": 0,
        "outcomes_recorded": 0,
        "consolidations": 0,
        "shm_writes": 1,  # Initial write counts
        "impasses_detected": 0,
        "haov_verifications": 0,
    }
    _last_heartbeat = time.time()
    _heartbeat_interval = 5.0
    _last_cgn_stats_pub = 0.0  # rFP §12 / SPEC §23.1 — periodic publisher
    _online_consolidation_counter = 0
    _online_consolidation_threshold = config.get(
        "online_consolidation_every", 50)
    # rFP_dead_dim_wiring_fix §2.F / SPEC §7.1 + D-SPEC-69 v1.14.0:
    # lazy-init StateRegistryWriter for cgn_beta_state.bin slot.
    cgn_beta_state_writer = None
    # Phase A.4 (D-SPEC-70 v1.10.0) — cgn_engine_state.bin publisher.
    # Sibling to existing cgn_live_weights.bin (tensor) + cgn_beta_state.bin
    # (8-float per-consumer reward EMA) — this slot carries engine-level
    # stats consumed by api_subprocess StateAccessor.cgn (replaces cgn.stats
    # bus-cache per Preamble G18).
    cgn_engine_state_publisher = None
    try:
        from titan_hcl.logic.cgn_engine_state_publisher import (
            CGNEngineStatePublisher,
        )
        from titan_hcl.core.state_registry import resolve_titan_id as _resolve_tid_cgn
        cgn_engine_state_publisher = CGNEngineStatePublisher(
            titan_id=_resolve_tid_cgn())
        logger.info(
            "[CGNWorker] cgn_engine_state publisher attached "
            "(G21 single-writer; Phase A.4 / D-SPEC-70)")
    except Exception as _err:
        logger.warning(
            "[CGNWorker] cgn_engine_state publisher init failed: %s — "
            "api_subprocess will read cold-boot stubs from cgn_engine_state",
            _err)

    # H.3 (2026-04-28): periodic HAOV test pump. Defect 4 — test gate
    # was coupled to per-consumer CGN_TRANSITION outcome; consumers that
    # form hypotheses but rarely emit outcomes (meta=184 formed/0 tested
    # in live telemetry) couldn't ever test. The pump walks all trackers
    # at fixed cadence and forces select_test independent of inbound
    # message. Also expires stuck _active_test entries (when verifier
    # response was lost — pre-H.2 this happened constantly because of
    # the silent-drop dest map).
    HAOV_TEST_PUMP_INTERVAL_S = float(
        config.get("haov_test_pump_interval_s", 30.0))
    HAOV_ACTIVE_TEST_TIMEOUT_S = float(
        config.get("haov_active_test_timeout_s", 300.0))  # 5 min
    _last_haov_pump_ts = 0.0

    # H.4 Phase 1 (v1) — staleness-decay tick for causal-generator candidates.
    # Slower cadence than the HAOV test pump; once per minute.  Only candidates
    # that got NO observation since the previous tick are bled (the staleness
    # gate — see CausalGenerator.decay_stale); actively-fed candidates are
    # exempt.  Decay is true multiplicative (factor ~0.999/tick → ~16 h before a
    # genuinely-quiet candidate decays below 1 and evicts).  No-op when
    # cgn.causal_generator.enabled = false.
    CAUSAL_DECAY_INTERVAL_S = float(
        config.get("causal_decay_interval_s", 60.0))
    _last_causal_decay_ts = 0.0

    # rFP_titan_vm_v2 Phase 2 §3.8: emit CGN_STATE_SNAPSHOT every
    # snapshot_interval CGN_TRANSITION events. spirit_worker subscribes
    # and writes payload into state_register.cgn_state; TitanVM bytecode
    # reads the cgn.* observable namespace from there.
    _snapshot_counter = 0
    _snapshot_interval = int(config.get("vm_snapshot_every_n_transitions", 10))

    init_ms = (time.time() - init_start) * 1000
    logger.info("[CGNWorker] Ready in %.0fms (consumers=%s, buffer=%d, "
                "shm=%s)", init_ms,
                list(cgn._consumers.keys()),
                cgn._buffer.size(), shm_path)

    # Phase 11 §11.I.2 — slot transition: starting → booted (D-SPEC-141 / v1.65.0).
    # MODULE_READY bus emit DELETED per locked D2 (no shim, no dual-publish).
    # SHM slot is the contract; titan_hcl's 1Hz poll detects "booted" and
    # dispatches MODULE_PROBE_REQUEST → handler below transitions slot to "running".
    _WORKER_READY = True
    if _state_writer is not None:
        try:
            _state_writer.write_state("booted")
            logger.info(
                "[CGNWorker] Phase 11 §11.I.2 — SHM slot state=booted "
                "(awaiting MODULE_PROBE_REQUEST from titan_hcl)")
        except Exception as _swb_err:  # noqa: BLE001
            logger.warning(
                "[CGNWorker] Phase 11 write_state(booted) failed: %s", _swb_err)

    # ── Main loop ──────────────────────────────────────────────────────
    # ── Microkernel v2 Phase B.1 §6 — readiness/hibernate reporter ──
    from titan_hcl.core.readiness_reporter import trivial_reporter
    def _b1_save_state():
        return []
    _b1_reporter = trivial_reporter(
        worker_name=name, layer="L2", send_queue=send_queue,
        save_state_cb=_b1_save_state,
    )

    # ── Phase 8.Y fold-in: lexicon snapshot exporter (D-SPEC-PHASE8) ────
    # Writes data/cgn_lexicon_snapshot.json on a 5-min cadence + emits
    # CGN_LEXICON_UPDATED so agno_worker refreshes its `plugin.cgn_lexicon`
    # cache (P7 `_ground_for_goal_hook` becomes able to return real
    # concept_ids). Soft-fail throughout.
    _LEXICON_EXPORT_INTERVAL_S = 300.0  # 5 minutes
    _last_lexicon_export = 0.0

    # ── Periodic cgn_state.pt disk checkpoint (CRITICAL persistence) ──
    # cgn_state.pt was otherwise written ONLY on dream consolidation
    # (CGN_CONSOLIDATE) + graceful SAVE_NOW/MODULE_SHUTDOWN. Under shm_pid_dead
    # restart churn the worker dies before any of those fire, so ALL learning
    # since the last dream (groundings, consolidations, value_net, consumer_freq)
    # was lost on every restart — cgn_state.pt was observed frozen for 1.5 days
    # across the fleet. This time-gated checkpoint makes persistence independent
    # of graceful shutdown (loses at most the interval on an ungraceful death).
    # Time-gated (not per-consolidation) to bound torch.save frequency.
    _STATE_CHECKPOINT_INTERVAL_S = 300.0  # 5 min
    _last_state_checkpoint = time.time()  # first checkpoint ~5min after boot
    _ckpt_thread = [None]                 # single-slot non-blocking writer

    def _maybe_export_lexicon_snapshot() -> None:
        nonlocal _last_lexicon_export
        now_ts = time.time()
        if now_ts - _last_lexicon_export < _LEXICON_EXPORT_INTERVAL_S:
            return
        _last_lexicon_export = now_ts
        try:
            from titan_hcl.cgn.lexicon_exporter import export_lexicon
            payload = export_lexicon()
            if payload is None:
                return
            _send_msg(send_queue, bus.CGN_LEXICON_UPDATED, name, "all", {
                "ts": payload.get("ts", now_ts),
                "lexicon_size": payload.get("lexicon_size", 0),
            })
            logger.debug(
                "[CGNWorker] cgn_lexicon_snapshot exported — size=%d",
                payload.get("lexicon_size", 0),
            )
        except Exception as e:
            logger.debug("[CGNWorker] lexicon export failed: %s", e)

    def _maybe_publish_cgn_stats() -> None:
        """Periodic (30s) publish of CGN_STATS_UPDATED + cgn_engine_state SHM.
        Phase C dissolution (2026-05-22): called from BOTH the idle (Empty) path
        AND the message path so it fires regardless of inbound traffic — the
        prior in-message-path-only placement was the recv_queue_except_empty
        periodic trap (`feedback_recv_queue_except_empty_periodic_trap`), which
        left cgn_engine_state empty on low-traffic Titans (verified live T2)."""
        nonlocal _last_cgn_stats_pub
        now = time.time()
        if now - _last_cgn_stats_pub < 30.0:
            return
        try:
            _cs = cgn.get_stats()
            _cs["worker"] = _stats
            try:
                _vm = cgn.get_vm_snapshot()
                if isinstance(_vm, dict):
                    _cs.setdefault("grounded_density",
                                   _vm.get("grounded_density", 0.0))
            except Exception:
                pass
            _send_msg(send_queue, bus.CGN_STATS_UPDATED, name, "all", _cs)
            if cgn_engine_state_publisher is not None:
                try:
                    cgn_engine_state_publisher.publish(cgn)
                except Exception as _ces_err:
                    logger.debug(
                        "[CGNWorker] cgn_engine_state periodic publish: %s",
                        _ces_err)
            _last_cgn_stats_pub = now
        except Exception as _csp_err:
            logger.debug("[CGNWorker] CGN_STATS_UPDATED publish: %s", _csp_err)

    def _maybe_checkpoint_state() -> None:
        """Periodic cgn_state.pt disk checkpoint (every _STATE_CHECKPOINT_INTERVAL_S).
        Called from BOTH the idle (Empty) path AND the message path so it fires
        regardless of inbound traffic (avoids the recv_queue_except_empty trap).
        This is the robust persistence guarantee: cgn survives an ungraceful
        (shm_pid_dead / SIGKILL) death losing at most one interval, rather than
        reverting to the last dream-consolidation snapshot.

        NON-BLOCKING: the torch.save() is offloaded to a single-slot daemon
        thread so a slow disk (swap / IO pressure) never stalls the heartbeat
        and trips a false shm_pid_dead. The single-slot guard prevents pile-up
        if a write runs longer than the interval."""
        nonlocal _last_state_checkpoint
        now_ts = time.time()
        if now_ts - _last_state_checkpoint < _STATE_CHECKPOINT_INTERVAL_S:
            return
        if _ckpt_thread[0] is not None and _ckpt_thread[0].is_alive():
            return  # prior checkpoint still writing — skip, retry next tick
        _last_state_checkpoint = now_ts

        def _do_ckpt():
            try:
                cgn._save_state()
            except Exception as _ckpt_err:  # noqa: BLE001
                logger.warning(
                    "[CGNWorker] periodic state checkpoint failed: %s",
                    _ckpt_err)
        _ckpt_thread[0] = threading.Thread(
            target=_do_ckpt, daemon=True, name="cgn-checkpoint")
        _ckpt_thread[0].start()

    while True:
        # H.3 (2026-04-28): periodic HAOV test pump. Runs at fixed cadence
        # regardless of inbound message activity. Decouples test-trigger
        # from per-consumer CGN_TRANSITION outcome events (Defect 4).
        _now_pump = time.time()
        if _now_pump - _last_haov_pump_ts >= HAOV_TEST_PUMP_INTERVAL_S:
            _last_haov_pump_ts = _now_pump
            try:
                _exp, _att, _snt = _run_haov_pump(
                    cgn, send_queue, name, HAOV_ACTIVE_TEST_TIMEOUT_S)
                if _exp or _snt:
                    logger.info(
                        "[CGNWorker] HAOV pump: expired=%d attempted=%d sent=%d",
                        _exp, _att, _snt)
            except Exception as _pump_outer_err:
                logger.warning(
                    "[CGNWorker] HAOV pump outer error: %s", _pump_outer_err)

        # H.4 Phase 1 (v1) — staleness decay for causal-generator candidates.
        # No-op when cgn.causal_generator.enabled = false (cgn returns 0).
        if _now_pump - _last_causal_decay_ts >= CAUSAL_DECAY_INTERVAL_S:
            _last_causal_decay_ts = _now_pump
            try:
                _evicted = cgn.decay_stale_haov()
                if _evicted:
                    logger.info(
                        "[CGNWorker] causal decay: evicted=%d stale candidates",
                        _evicted)
            except Exception as _decay_err:
                logger.warning(
                    "[CGNWorker] causal decay outer error: %s", _decay_err)

        try:
            msg = recv_queue.get(timeout=_heartbeat_interval)
        except Empty:
            # Heartbeat on idle
            _send_heartbeat(send_queue, name)
            # Phase 11 §11.I.5 — SHM-slot heartbeat sidecar (guardian_hcl
            # staleness detector + observatory /v6/readiness consume this).
            # Suppressed until _WORKER_READY so slot stays starting/booted.
            if _state_writer is not None and _WORKER_READY:
                try:
                    _state_writer.heartbeat()
                except Exception:  # noqa: BLE001
                    pass
            _last_heartbeat = time.time()
            # Periodic stats publish ALSO on the idle path (Phase C — see
            # _maybe_publish_cgn_stats docstring; idle is the common case on a
            # low-traffic Titan, where the in-message-path block never ran).
            _maybe_publish_cgn_stats()
            _maybe_export_lexicon_snapshot()
            _maybe_checkpoint_state()
            continue

        msg_type = msg.get("type", "")
        payload = msg.get("payload", {})

        # ── Phase 11 §11.I.3 — MODULE_PROBE_REQUEST handler ─────────────
        # titan_hcl dispatches this after detecting our SHM slot transitioned
        # to "booted". probe_dispatcher.handle_module_probe_request writes
        # state="probing", runs probe_fn (None=trivial-pass), then writes
        # state="running" or "unhealthy" and publishes MODULE_PROBE_RESPONSE.
        if msg_type == bus.MODULE_PROBE_REQUEST and _state_writer is not None:
            try:
                from titan_hcl.core.probe_dispatcher import (
                    handle_module_probe_request,
                )
                handle_module_probe_request(
                    msg,
                    probe_fn=None,
                    send_queue=send_queue,
                    module_name=name,
                    state_writer=_state_writer,
                )
            except Exception as _probe_err:  # noqa: BLE001
                logger.warning(
                    "[CGNWorker] MODULE_PROBE_REQUEST handler failed: %s",
                    _probe_err)
            continue

        # ── Heartbeat check ──
        now = time.time()
        if now - _last_heartbeat > _heartbeat_interval:
            _send_heartbeat(send_queue, name)
            if _state_writer is not None and shm_heartbeat_allowed(_WORKER_READY, _BOOT_DEADLINE):
                try:
                    _state_writer.heartbeat()
                except Exception:  # noqa: BLE001
                    pass
            _last_heartbeat = now

        # CGN_STATS_UPDATED + cgn_engine_state periodic publish (30s) — also on
        # the message path (high-traffic case). See _maybe_publish_cgn_stats.
        _maybe_publish_cgn_stats()
        # P8.Y fold-in: lexicon snapshot export on 5-min cadence
        _maybe_export_lexicon_snapshot()
        # CRITICAL persistence: periodic cgn_state.pt disk checkpoint
        _maybe_checkpoint_state()

        # ── CGN_TRANSITION — experience from any consumer ──────────
        if msg_type == "CGN_TRANSITION":
            try:
                _ptype = payload.get("type")
                if _ptype in ("outcome", "experience"):
                    if _ptype == "experience":
                        # (b) Single-shot COMPLETE transition (action+reward in one
                        # message). record_experience buffers the action as pending
                        # then self-matches via record_outcome → FULL Sigma + journey
                        # + causal observe_for + (the HAOV select_test below). For
                        # simultaneous-outcome consumers (emotional / meta / self_model
                        # / reasoning_strategy / knowledge); SPEC §6.5 / DEFERRED G1 —
                        # the dormant-consumer wake the two-phase pending→outcome split
                        # otherwise blocked. Falls through to the same downstream.
                        cgn.record_experience(
                            consumer=payload["consumer"],
                            concept_id=payload["concept_id"],
                            reward=payload["reward"],
                            action=payload.get("action", 0),
                            state=payload.get("state"),
                            action_params=payload.get("action_params"),
                            outcome_context=payload.get("outcome_context"),
                            metadata=payload.get("metadata", {}))
                    else:
                        # Delayed reward for an existing pending transition
                        cgn.record_outcome(
                            consumer=payload["consumer"],
                            concept_id=payload["concept_id"],
                            reward=payload["reward"],
                            outcome_context=payload.get("outcome_context"))
                    _stats["outcomes_recorded"] += 1

                    # Sigma micro-update happened inside record_outcome()
                    # Write updated V(s) to /dev/shm
                    if config.get("shm_write_on_every_outcome", True):
                        _write_incremental_shm(cgn, shm_writer)
                        _stats["shm_writes"] += 1

                    # ── Phase B (RFP_cgn_enhancements §9.2) — emit CGN_CONCEPT_GROUNDED
                    # for any concept that just matured across ≥2 consumers (the
                    # Level-B cross-consumer abstraction trigger). cgn queues these
                    # in record_outcome; we drain + publish to cognitive_worker.
                    try:
                        for _mat in cgn.pop_matured_concepts():
                            # CGN-felt RFP Phase B — carry the per-concept felt centroid
                            # (G18 read-down) so synthesis can do a true felt-vector
                            # frame_dependent comparison. {} when no felt accumulated yet.
                            _send_msg(send_queue, bus.CGN_CONCEPT_GROUNDED, name,
                                      "cognitive_worker", {
                                          "concept_id": _mat["concept_id"],
                                          "consumers": _mat["consumers"],
                                          "first_consumer": _mat["first_consumer"],
                                          "felt_centroid": cgn.concept_felt_centroid(
                                              _mat["concept_id"]),
                                      })
                            logger.info(
                                "[CGN] CGN_CONCEPT_GROUNDED — concept=%s consumers=%s "
                                "(Phase B Level-B trigger)",
                                _mat["concept_id"], _mat["consumers"])
                    except Exception as _cg_mat_err:
                        swallow_warn('[CGNWorker] CGN_CONCEPT_GROUNDED emit', _cg_mat_err,
                                     key="modules.cgn_worker.concept_grounded_emit",
                                     throttle=100)

                    # SOAR impasse check (every 10 outcomes)
                    if _stats["outcomes_recorded"] % 10 == 0:
                        _check_impasses(cgn, send_queue, name)

                    # HAOV: try to select and trigger a hypothesis test
                    _haov_consumer = payload["consumer"]
                    _haov_tracker = cgn._haov_trackers.get(_haov_consumer)
                    if _haov_tracker:
                        _test_ctx = _haov_tracker.select_test({
                            "available_actions": [],
                            "observation": payload.get("outcome_context", {}),
                        })
                        if _test_ctx:
                            # In-process verify first (impasse hypotheses + no-
                            # specialist consumers) — fixes F4 dead-routing.
                            if _local_haov_verify(
                                    cgn, _haov_tracker, _haov_consumer):
                                pass  # verified locally; no bus emit
                            else:
                                # Concept-grounding → live specialist verifier only.
                                _haov_dest = _HAOV_DEST_MAP.get(_haov_consumer)
                                if _haov_dest is not None:
                                    if _haov_tracker._active_test is not None:
                                        _haov_tracker._active_test["ts"] = time.time()
                                    _send_msg(send_queue, bus.CGN_HAOV_VERIFY_REQ,
                                              name, _haov_dest, {
                                        "consumer": _haov_consumer,
                                        "test_ctx": _test_ctx,
                                        "obs_before": payload.get("outcome_context", {}),
                                        "hypothesis": _haov_tracker._active_test[
                                            "hypothesis"].rule if _haov_tracker._active_test else "",
                                    })
                                    logger.debug("[CGNWorker] HAOV test → %s: %s",
                                                 _haov_dest, _test_ctx)
                else:
                    # New transition — buffer it
                    t = CGNTransition(
                        consumer=payload.get("consumer", "?"),
                        concept_id=payload.get("concept_id", "?"),
                        state=__import__("numpy").array(
                            payload.get("state", [0.0] * 30),
                            dtype=__import__("numpy").float32),
                        action=payload.get("action", 0),
                        action_params=__import__("numpy").array(
                            payload.get("action_params", [0.0] * 4),
                            dtype=__import__("numpy").float32),
                        reward=payload.get("reward", 0.0),
                        timestamp=payload.get("timestamp", time.time()),
                        epoch=payload.get("epoch", 0),
                        metadata=payload.get("metadata", {}),
                    )
                    cgn._buffer.add(t)
                    _stats["transitions_received"] += 1

                    # Online consolidation check
                    _online_consolidation_counter += 1
                    if (_online_consolidation_counter >=
                            _online_consolidation_threshold):
                        _online_consolidation_counter = 0
                        cgn.consolidate(dream_phase=False)
                        _write_incremental_shm(cgn, shm_writer)
                        _stats["shm_writes"] += 1

                # rFP_titan_vm_v2 Phase 2 §3.8: emit CGN_STATE_SNAPSHOT
                # every N transitions (both outcome and new-transition
                # branches tick the counter — snapshot reflects real
                # activity cadence). Failure here is non-fatal (best-effort
                # observability to TitanVM).
                _snapshot_counter += 1
                if _snapshot_counter >= _snapshot_interval:
                    _snapshot_counter = 0
                    try:
                        _snap = cgn.get_vm_snapshot()
                        _send_msg(send_queue, bus.CGN_STATE_SNAPSHOT,
                                  name, "spirit", _snap)
                    except Exception as _e:
                        swallow_warn('[CGNWorker] snapshot emit failed', _e,
                                     key="modules.cgn_worker.snapshot_emit_failed", throttle=100)
                    # rFP_dead_dim_wiring_fix §2.F / SPEC §7.1 + D-SPEC-69
                    # v1.14.0: write cgn_beta_state.bin SHM slot — G18-pure
                    # replacement for the retired CGN_BETA_SNAPSHOT bus event.
                    # Source: _snap.get(f"{c}_reward_ema") per consumer
                    # (from cgn.get_vm_snapshot — computed from last 64
                    # replay-buffer transitions per consumer). Missing
                    # consumers default to 0.5 (no recent activity).
                    try:
                        if cgn_beta_state_writer is None:
                            from titan_hcl.core.state_registry import (
                                CGN_BETA_STATE, StateRegistryWriter,
                                ensure_shm_root, resolve_titan_id)
                            _titan_id_beta = resolve_titan_id()
                            cgn_beta_state_writer = StateRegistryWriter(
                                CGN_BETA_STATE,
                                ensure_shm_root(_titan_id_beta))
                            logger.info(
                                "[CGNWorker] cgn_beta_state.bin writer "
                                "attached (G21 single-writer; SPEC §7.1 "
                                "D-SPEC-69 — replaces CGN_BETA_SNAPSHOT "
                                "bus event per G18)")
                        from titan_hcl.logic.emot_bundle_protocol import (
                            CGN_CONSUMERS as _CGN_CONSUMERS)
                        import numpy as _np_beta
                        _beta_vec = _np_beta.array(
                            [float(_snap.get(f"{c}_reward_ema", 0.5))
                             for c in _CGN_CONSUMERS],
                            dtype=_np_beta.float32)
                        cgn_beta_state_writer.write(_beta_vec)
                    except Exception as _e:
                        swallow_warn('[CGNWorker] cgn_beta_state write failed', _e,
                                     key="modules.cgn_worker.cgn_beta_state_write_failed", throttle=100)

                    # Phase A.4 — publish engine-level stats slot at same
                    # cadence as cgn_beta_state (per snapshot tick).
                    if cgn_engine_state_publisher is not None:
                        cgn_engine_state_publisher.publish(cgn)

            except Exception as e:
                logger.warning("[CGNWorker] CGN_TRANSITION error: %s", e)

        # ── CGN_REGISTER — new consumer registration ──────────────
        elif msg_type == "CGN_REGISTER":
            try:
                cfg = CGNConsumerConfig(
                    name=payload["name"],
                    feature_dims=payload.get("feature_dims", 30),
                    action_dims=payload.get("action_dims", 8),
                    action_names=payload.get("action_names", []),
                    reward_source=payload.get("reward_source", ""),
                    max_buffer_size=payload.get("max_buffer_size", 500),
                    consolidation_priority=payload.get(
                        "consolidation_priority", 5),
                )
                handle = cgn.register_consumer(cfg)
                _write_full_shm(cgn, shm_writer)
                logger.info("[CGNWorker] Registered consumer '%s' "
                            "(actions=%d)", handle, cfg.action_dims)
            except Exception as e:
                logger.warning("[CGNWorker] CGN_REGISTER error: %s", e)

        # ── CGN_CONSOLIDATE — dream phase full training ───────────
        elif msg_type == "CGN_CONSOLIDATE":
            try:
                logger.info("[CGNWorker] Dream consolidation starting...")
                t0 = time.time()
                stats = cgn.consolidate(dream_phase=True)
                cgn._save_state()
                _write_full_shm(cgn, shm_writer)
                _stats["consolidations"] += 1
                _stats["shm_writes"] += 1

                # Compute consumer affinity matrix (reward correlations)
                affinity = _compute_affinity_matrix(cgn)

                duration_ms = (time.time() - t0) * 1000
                logger.info("[CGNWorker] Consolidation #%d complete (%.0fms): "
                            "v_loss=%.4f consumers=%s affinity_pairs=%d",
                            _stats["consolidations"], duration_ms,
                            stats.get("v_loss", 0),
                            list(stats.get("consumers", {}).keys()),
                            len(affinity))

                # Broadcast to all consumers: weights updated
                _send_msg(send_queue, bus.CGN_WEIGHTS_MAJOR, name, "all", {
                    "consolidation": _stats["consolidations"],
                    "v_loss": stats.get("v_loss", 0),
                    "consumers": {
                        k: v.get("q_loss", 0) if isinstance(v, dict) else 0
                        for k, v in stats.get("consumers", {}).items()
                    },
                    "shm_version": shm_writer.get_version(),
                    "affinity_matrix": affinity,
                })
                # TimeChain: dream consolidation → procedural fork
                send_queue.put({"type": bus.TIMECHAIN_COMMIT, "src": name,
                    "dst": "timechain", "ts": time.time(), "payload": {
                    "fork": "procedural", "thought_type": "procedural",
                    "source": "cgn_dream_consolidation",
                    "content": {
                        "consolidation_num": _stats["consolidations"],
                        "v_loss": round(stats.get("v_loss", 0), 6),
                        "consumers": list(stats.get("consumers", {}).keys()),
                        "buffer_size": stats.get("buffer_used", 0),
                        "affinity_pairs": len(affinity),
                    },
                    "significance": 0.5, "novelty": 0.3,
                    "coherence": 0.8,
                    "tags": ["dream_consolidation", "cgn", "iql"],
                    "neuromods": {}, "chi_available": 0.5,
                    "attention": 0.5, "i_confidence": 0.5,
                    "chi_coherence": 0.3,
                }})
            except Exception as e:
                logger.error("[CGNWorker] Consolidation error: %s", e)

        # ── CGN_SURPRISE — cross-consumer surprise event ──────────
        elif msg_type == "CGN_SURPRISE":
            try:
                cgn.record_surprise(
                    consumer=payload.get("consumer", "?"),
                    concept_id=payload.get("concept_id", "?"),
                    magnitude=payload.get("magnitude", 0.0),
                    context=payload.get("context"))
            except Exception as e:
                swallow_warn('[CGNWorker] Surprise error', e,
                             key="modules.cgn_worker.surprise_error", throttle=100)

        # ── CGN_HAOV_VERIFY_RSP — verification from consumer ─────
        elif msg_type == bus.CGN_HAOV_VERIFY_RSP:
            try:
                consumer = payload["consumer"]
                tracker = cgn._haov_trackers.get(consumer)
                if tracker and payload.get("test_ctx"):
                    tracker.verify(
                        payload["test_ctx"],
                        payload.get("obs_after", {}),
                        payload.get("reward", 0.0))
                    _stats["haov_verifications"] += 1
                    # Sigma micro-update for verified hypothesis
                    _write_incremental_shm(cgn, shm_writer)
                    # TimeChain: HAOV hypothesis verified → procedural fork
                    _tc_hypothesis = payload.get("hypothesis", "")
                    _tc_reward = payload.get("reward", 0.0)
                    _tc_confirmed = _tc_reward > 0
                    send_queue.put({"type": bus.TIMECHAIN_COMMIT, "src": name,
                        "dst": "timechain", "ts": time.time(), "payload": {
                        "fork": "procedural", "thought_type": "procedural",
                        "source": "haov_verification",
                        "content": {"consumer": consumer,
                            "hypothesis": str(_tc_hypothesis)[:200],
                            "reward": round(_tc_reward, 4),
                            "confirmed": _tc_confirmed,
                            "test_ctx": str(payload.get("test_ctx", ""))[:100]},
                        "significance": 0.7 if _tc_confirmed else 0.4,
                        "novelty": 0.6, "coherence": 0.7,
                        "tags": [consumer, "haov",
                                 "confirmed" if _tc_confirmed else "falsified"],
                        "neuromods": {},
                        "chi_available": 0.5, "attention": 0.5,
                        "i_confidence": 0.5, "chi_coherence": 0.3,
                    }})
            except Exception as e:
                swallow_warn('[CGNWorker] HAOV verify error', e,
                             key="modules.cgn_worker.haov_verify_error", throttle=100)

        # ── CGN_HAOV_RULE_APPLIED — a verified rule influenced an action ─
        # RFP_cgn_loop_closure §7.D (INV-LOOP-6): closes learning → behaviour.
        # A consumer (C2 language teaching / C3 social engage-bias) applied a
        # verified haov rule in its own process, then emitted this. We own the
        # trackers (G21), so we bump the SOURCE consumer's used_for_action — the
        # rule's author gets the credit (social acting on language's rule →
        # language.used_for_action++). Published in cgn_engine_state.bin.haov_stats
        # → /v6/cognition/cgn-haov-stats (G7). Monotonic + restart-safe (we own
        # the counter; consumer restarts don't reset it).
        elif msg_type == bus.CGN_HAOV_RULE_APPLIED:
            try:
                _src = payload.get("source_consumer") or payload.get("consumer")
                _tracker = cgn._haov_trackers.get(_src) if _src else None
                if _tracker is not None:
                    _n = max(1, int(payload.get("count", 1) or 1))
                    _new_total = _tracker.mark_used_for_action(_n)
                    _stats["haov_rules_applied"] = (
                        _stats.get("haov_rules_applied", 0) + _n)
                    _write_incremental_shm(cgn, shm_writer)
                    logger.info(
                        "[CGNWorker] HAOV rule APPLIED — source=%s by=%s "
                        "used_for_action=%d (rule=%s)", _src,
                        payload.get("applying_consumer", "?"), _new_total,
                        str(payload.get("rule", ""))[:60])
                else:
                    logger.debug(
                        "[CGNWorker] CGN_HAOV_RULE_APPLIED for unknown source "
                        "consumer=%s — dropped", _src)
            except Exception as e:
                swallow_warn('[CGNWorker] HAOV rule-applied error', e,
                             key="modules.cgn_worker.haov_rule_applied_error", throttle=100)

        # ── CGN_MODEL_CORROBORATION — OFFER-inner (RFP_pattern_logic §7.1/§VC-2) ─
        # A promoted cross-substrate MODEL corroborates the inner HAOV hypotheses it
        # was built from → boost each named rule's confidence via the tracker merge
        # path. HAOV-confidence ONLY — never reward_ema / Q-net / cgn_beta_state, so
        # the emot-coupling (the CRITICAL-NO blast radius) is untouched.
        elif msg_type == bus.CGN_MODEL_CORROBORATION:
            try:
                _consumer = str(payload.get("consumer", "") or "")
                _tracker = cgn._haov_trackers.get(_consumer) if _consumer else None
                _strength = float(payload.get("strength", 0.0) or 0.0)
                if _tracker is not None and _strength > 0.0:
                    _rules = [str(r) for r in (payload.get("rules") or []) if r]
                    _hit = sum(1 for _r in _rules if _tracker.corroborate(_r, _strength))
                    if _hit:
                        _stats["haov_corroborations"] = (
                            _stats.get("haov_corroborations", 0) + _hit)
                        _write_incremental_shm(cgn, shm_writer)
                        logger.info(
                            "[CGNWorker] cross-substrate corroboration — consumer=%s "
                            "rules_hit=%d/%d strength=%.3f", _consumer, _hit,
                            len(_rules), _strength)
                else:
                    logger.debug(
                        "[CGNWorker] CGN_MODEL_CORROBORATION unknown consumer=%s or "
                        "zero strength — dropped", _consumer)
            except Exception as e:
                swallow_warn('[CGNWorker] model-corroboration error', e,
                             key="modules.cgn_worker.model_corroboration_error", throttle=100)

        # ── CGN_INFERENCE_REQ — policy inference for remote processes ─
        # API_STUB: handler ready, awaits remote process senders (T2/T3
        # delegate path). Tracked I-003.
        elif msg_type == bus.CGN_INFERENCE_REQ:
            try:
                result = cgn.infer_social_action(
                    __import__("titan_hcl.logic.cgn",
                               fromlist=["SensoryContext"]).SensoryContext(
                        neuromods=payload.get("neuromods", {}),
                        epoch=payload.get("epoch", 0)),
                    user_features=payload.get("user_features"))
                _send_response(send_queue, name, msg.get("src", ""),
                               result, msg.get("rid"))
            except Exception as e:
                swallow_warn('[CGNWorker] Inference error', e,
                             key="modules.cgn_worker.inference_error", throttle=100)

        # ── CGN_KNOWLEDGE_REQ — route to knowledge worker ─────────
        elif msg_type == bus.CGN_KNOWLEDGE_REQ:
            # Forward to knowledge worker (when it exists)
            _send_msg(send_queue, bus.CGN_KNOWLEDGE_REQ, name,
                      "knowledge", payload)

        # ── QUERY — stats and diagnostics ─────────────────────────
        elif msg_type == bus.QUERY:
            from titan_hcl.core.profiler import handle_memory_profile_query
            if handle_memory_profile_query(msg, send_queue, name):
                continue
            action = payload.get("action", "")
            if action == "get_stats":
                cgn_stats = cgn.get_stats()
                cgn_stats["worker"] = _stats
                cgn_stats["shm_version"] = shm_writer.get_version()
                cgn_stats["affinity_matrix"] = _compute_affinity_matrix(cgn)
                _send_response(send_queue, name, msg.get("src", ""),
                               cgn_stats, msg.get("rid"))
            elif action == "get_cross_insights":
                consumer = payload.get("consumer", "")
                insights = cgn.get_cross_insights(consumer)
                # rFP_haov_efficacy_closure §3E (C1→C2/C3 delivery fix, I-003):
                # the consumers (language_worker:1884, social_worker) listen on
                # QUERY_RESPONSE keyed by payload.action — NOT the generic RESPONSE
                # that _send_response emits. Pre-fix the reply type/shape mismatched
                # the handler, so cross-insights (incl. C1 haov_verified rules) never
                # arrived — the apply channel was structurally dead. Reply with the
                # type+shape the async consumers expect.
                _send_msg(send_queue, bus.QUERY_RESPONSE, name,
                          msg.get("src", ""),
                          {"action": "get_cross_insights",
                           "consumer": consumer, "insights": insights},
                          msg.get("rid"))

        # ── MODULE_SHUTDOWN ───────────────────────────────────────
        # ── Microkernel v2 Phase B.1 §6 — shadow swap dispatch ────
        if _b1_reporter.handles(msg_type):
            _b1_reporter.handle(msg)
            if _b1_reporter.should_exit():
                break
            continue

        # ── Microkernel v2 Phase B.2.1 — supervision-transfer dispatch ──
        from titan_hcl.core import worker_swap_handler as _swap
        if _swap.maybe_dispatch_swap_msg(msg):
            continue

        elif msg_type == bus.SAVE_NOW:
            # Graceful checkpoint requested by orchestrator stop(save_first=True)
            # before a restart (SAVE_NOW → SAVE_DONE → SIGTERM). cgn previously
            # had NO SAVE_NOW handler (every other persistence worker does), so
            # the Guardian's save_first was a silent no-op for cgn → all learning
            # lost on graceful restart. Emit SAVE_DONE{module,request_id} so the
            # orchestrator doesn't block the full save_timeout waiting on us.
            _save_rid = (payload or {}).get("request_id")
            _save_t0 = time.time()
            _save_ok, _save_errs = True, 0
            try:
                cgn._save_state()
                _write_full_shm(cgn, shm_writer)
            except Exception as _save_now_err:  # noqa: BLE001
                _save_ok, _save_errs = False, 1
                logger.warning(
                    "[CGNWorker] SAVE_NOW checkpoint failed: %s", _save_now_err)
            try:
                _send_msg(send_queue, bus.SAVE_DONE, name, "guardian", {
                    "module": name, "request_id": _save_rid,
                    "saved": _save_ok, "errors": _save_errs,
                    "duration_ms": int((time.time() - _save_t0) * 1000),
                })
            except Exception:  # noqa: BLE001
                pass
            continue

        elif msg_type == bus.MODULE_SHUTDOWN:
            logger.info("[CGNWorker] Shutdown requested: %s",
                        payload.get("reason", "?"))
            # Final save before exit
            try:
                cgn._save_state()
                _write_full_shm(cgn, shm_writer)
                logger.info("[CGNWorker] Final state saved")
            except Exception as _swallow_exc:
                swallow_warn('[modules.cgn_worker] cgn_worker_main: cgn._save_state()', _swallow_exc,
                             key='modules.cgn_worker.cgn_worker_main.line645', throttle=100)
            break


# ── Helpers ────────────────────────────────────────────────────────────

def _serialize_verified_action_rules(cgn, top_k: int = 8) -> bytes:
    """INV-LOOP-7 (RFP_cgn_loop_closure §7.D) — pack each consumer's verified,
    ACTION-BEARING HAOV rules (confidence > 0.5, an ``action`` in action_context)
    into the SHM weights `extra` blob so the consumer process can apply them as a
    bounded prior on its learned policy. Impasse/concept-only rules (no ``action``)
    are excluded — those still flow via the C2/C3 text channel. Compact msgpack
    {consumer: [[action_idx, conf, predicted_magnitude, rule_name], ...]}; top-K
    per consumer by conf·magnitude. Returns b"" if nothing qualifies."""
    try:
        import msgpack
        out: dict = {}
        for name, tracker in getattr(cgn, "_haov_trackers", {}).items():
            rules = []
            for h in getattr(tracker, "_verified_rules", []):
                ac = h.action_context or {}
                if h.confidence > 0.5 and "action" in ac:
                    rules.append([int(ac["action"]),
                                  round(float(h.confidence), 4),
                                  round(float(h.predicted_magnitude), 4),
                                  str(h.rule)])
            if rules:
                rules.sort(key=lambda r: r[1] * r[2], reverse=True)
                out[name] = rules[:top_k]
        return msgpack.packb(out, use_bin_type=True) if out else b""
    except Exception as e:
        swallow_warn('[CGNWorker] verified-rule serialize failed', e,
                     key="modules.cgn_worker.verified_rule_serialize_failed", throttle=100)
        return b""


def _write_full_shm(cgn, shm_writer):
    """Write complete weight snapshot to /dev/shm."""
    try:
        consumer_nets = {
            name: net.state_dict()
            for name, net in cgn._action_nets.items()
        }
        shm_writer.write_full(
            cgn._value_net.state_dict(),
            consumer_nets,
            extra=_serialize_verified_action_rules(cgn))
    except Exception as e:
        swallow_warn('[CGNWorker] SHM write failed', e,
                     key="modules.cgn_worker.shm_write_failed", throttle=100)


def _write_incremental_shm(cgn, shm_writer):
    """Write V(s) + consumer nets to /dev/shm (same as full for now).

    Future optimization: write only V(s) for Sigma micro-updates
    since Q(s,a) only changes during dream consolidation.
    """
    _write_full_shm(cgn, shm_writer)


def _compute_affinity_matrix(cgn) -> dict:
    """Compute consumer affinity matrix from reward correlations.

    For each consumer pair, compute the correlation between their
    reward signals over recent transitions. High correlation means
    the consumers are learning in sync — they benefit from similar states.

    Returns: {("language","social"): 0.72, ("language","knowledge"): 0.45, ...}
    encoded as {"language:social": 0.72, ...} for JSON serialization.
    """
    affinity = {}
    consumers = list(cgn._consumers.keys())
    if len(consumers) < 2:
        return affinity

    # Collect reward timeseries per consumer
    reward_series = {}
    for c in consumers:
        transitions = cgn._buffer.get_consumer_transitions(c, max_count=100)
        rewarded = [t.reward for t in transitions if t.reward != 0.0]
        if len(rewarded) >= 5:
            reward_series[c] = rewarded

    # Compute pairwise correlation
    for i, c1 in enumerate(consumers):
        for c2 in consumers[i + 1:]:
            if c1 not in reward_series or c2 not in reward_series:
                continue
            r1 = reward_series[c1]
            r2 = reward_series[c2]
            # Align to same length (truncate to shorter)
            min_len = min(len(r1), len(r2))
            if min_len < 5:
                continue
            r1 = r1[:min_len]
            r2 = r2[:min_len]
            # Pearson correlation (manual — avoid numpy dependency)
            mean1 = sum(r1) / len(r1)
            mean2 = sum(r2) / len(r2)
            cov = sum((a - mean1) * (b - mean2) for a, b in zip(r1, r2))
            std1 = (sum((a - mean1) ** 2 for a in r1)) ** 0.5
            std2 = (sum((b - mean2) ** 2 for b in r2)) ** 0.5
            if std1 > 0 and std2 > 0:
                corr = round(cov / (std1 * std2), 3)
            else:
                corr = 0.0
            affinity[f"{c1}:{c2}"] = corr

    if affinity:
        # Persist to telemetry
        try:
            import json
            entry = {
                "type": "affinity_matrix",
                "ts": time.time(),
                "pairs": affinity,
            }
            with open("./data/cgn/affinity_history.jsonl", "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as _swallow_exc:
            swallow_warn('[modules.cgn_worker] _compute_affinity_matrix: import json', _swallow_exc,
                         key='modules.cgn_worker._compute_affinity_matrix.line735', throttle=100)

    return affinity


def _check_impasses(cgn, send_queue, name):
    """Run SOAR impasse detection for all consumers."""
    for consumer_name in cgn._consumers:
        try:
            impasse = cgn.detect_impasse(consumer_name)
            if impasse:
                # INTENTIONAL_BROADCAST: dst=all telemetry consumed by
                # frontend WebSocket (/v4/events stream). I-004 verified.
                _send_msg(send_queue, "CGN_IMPASSE", name, "all", impasse)
                logger.info("[CGNWorker] Impasse detected: %s %s (sev=%.2f)",
                            consumer_name, impasse.get("type"),
                            impasse.get("severity", 0))
                # ── META-CGN producer #9: knowledge.impasse_resolved ──
                # v3 Phase D rollout (rFP § 12 row 9). Fires when knowledge-path
                # impasse detected (detect_impasse() returns non-None + generates
                # HAOV subgoal hypothesis = the "resolution attempt"). BREAK 0.75
                # + HYPOTHESIZE 0.70 — BREAK is THE anti-monoculture primitive
                # (T1/T3 at 2.6% currently). No EdgeDetector (impasse detection
                # is already gated by condition — rate naturally bounded).
                if consumer_name == "knowledge":
                    try:
                        from titan_hcl.bus import emit_meta_cgn_signal
                        _p9_sev = float(impasse.get("severity", 0.5))
                        _p9_type = str(impasse.get("type", "stuck"))
                        _p9_sent = emit_meta_cgn_signal(
                            send_queue,
                            src="knowledge",
                            consumer="knowledge",
                            event_type="impasse_resolved",
                            intensity=min(1.0, max(0.1, _p9_sev)),
                            domain=_p9_type,
                            reason=f"SOAR impasse on knowledge path: {_p9_type} (severity={_p9_sev:.2f}) "
                                   f"— HAOV subgoal hypothesis generated",
                        )
                        if _p9_sent:
                            logger.info(
                                "[META-CGN] knowledge.impasse_resolved EMIT — type=%s severity=%.2f",
                                _p9_type, _p9_sev)
                        else:
                            logger.warning(
                                "[META-CGN] Producer #9 knowledge.impasse_resolved DROPPED by bus "
                                "— type=%s severity=%.2f (rate-gate or queue-full)",
                                _p9_type, _p9_sev)
                        # Record stage (rFP_experience_distillation_phase_c §5) —
                        # a knowledge-path impasse resolution attempt is a
                        # knowledge experience. cognitive_worker enriches + records.
                        from titan_hcl.bus import emit_experience_record
                        emit_experience_record(
                            send_queue, "knowledge",
                            domain="knowledge",
                            action_taken=f"impasse:{_p9_type}",
                            outcome_score=min(1.0, max(0.5, _p9_sev)),
                            context={"type": _p9_type, "severity": _p9_sev,
                                     "impasse_resolved": True,
                                     "source": "impasse_resolved"},
                            coalesce_key="knowledge:impasse",
                        )
                    except Exception as _p9_err:
                        logger.warning(
                            "[META-CGN] Producer #9 knowledge.impasse_resolved emit FAILED "
                            "— type=%s err=%s (signal missed)",
                            impasse.get("type", "?"), _p9_err)
        except Exception as _swallow_exc:
            swallow_warn('[modules.cgn_worker] _check_impasses: impasse = cgn.detect_impasse(consumer_name)', _swallow_exc,
                         key='modules.cgn_worker._check_impasses.line789', throttle=100)


def _send_msg(send_queue, msg_type, src, dst, payload, rid=None):
    try:
        send_queue.put_nowait({
            "type": msg_type, "src": src, "dst": dst,
            "ts": time.time(), "rid": rid, "payload": payload,
        })
    except Exception:
        from titan_hcl.bus import record_send_drop
        record_send_drop(src, dst, msg_type)


def _send_response(send_queue, src, dst, payload, rid):
    _send_msg(send_queue, bus.RESPONSE, src, dst, payload, rid)


# Heartbeat throttle (Phase E Fix 2): 3s min interval per process.
_last_hb_ts: float = 0.0


def _send_heartbeat(send_queue, name):
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
    _send_msg(send_queue, bus.MODULE_HEARTBEAT, name, "guardian",
              {"rss_mb": round(rss_mb, 1)})
