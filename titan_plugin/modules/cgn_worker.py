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
import time
from queue import Empty

logger = logging.getLogger(__name__)


def cgn_worker_main(recv_queue, send_queue, name: str, config: dict) -> None:
    """Main loop for the CGN Cognitive Kernel process.

    Args:
        recv_queue: receives messages from DivineBus (bus->worker)
        send_queue: sends messages back to DivineBus (worker->bus)
        name: module name ("cgn")
        config: dict from [cgn] config section
    """
    project_root = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    logger.info("[CGNWorker] Initializing Cognitive Kernel...")
    init_start = time.time()

    # ── Load CGN ────────────────────────────────────────────────────────
    from titan_plugin.logic.cgn import (
        ConceptGroundingNetwork, CGNConsumerConfig, CGNTransition)
    from titan_plugin.logic.cgn_shm_protocol import ShmWeightWriter

    state_dir = config.get("state_dir", "data/cgn")
    db_path = config.get("db_path", "data/inner_memory.db")
    shm_path = config.get("shm_path", "/dev/shm/cgn_live_weights.bin")

    # Load HAOV config with per-Titan profile overrides
    # NOTE: config dict comes from config.toml via v5_core. HAOV profiles
    # live in titan_params.toml, so we load directly from there.
    haov_config = {}
    try:
        import json as _hc_json
        try:
            import tomllib as _hc_toml
        except ModuleNotFoundError:
            import toml as _hc_toml  # type: ignore
        _params_path = os.path.join(
            os.path.dirname(__file__), "..", "titan_params.toml")
        with open(_params_path, "rb") as _pf:
            _params = _hc_toml.load(_pf)
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

    cgn = ConceptGroundingNetwork(
        db_path=db_path, state_dir=state_dir, haov_config=haov_config)
    # _load_state() called in __init__

    # ── Pre-register "reasoning" consumer for ARC (runs as standalone script) ──
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

    # ── Initialize /dev/shm weight writer ──────────────────────────────
    shm_writer = ShmWeightWriter(shm_path)
    _write_full_shm(cgn, shm_writer)
    logger.info("[CGNWorker] SHM weights written to %s (v=%d)",
                shm_path, shm_writer.get_version())

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
    _online_consolidation_counter = 0
    _online_consolidation_threshold = config.get(
        "online_consolidation_every", 50)

    init_ms = (time.time() - init_start) * 1000
    logger.info("[CGNWorker] Ready in %.0fms (consumers=%s, buffer=%d, "
                "shm=%s)", init_ms,
                list(cgn._consumers.keys()),
                cgn._buffer.size(), shm_path)
    _send_msg(send_queue, "MODULE_READY", name, "guardian", {})

    # ── Main loop ──────────────────────────────────────────────────────
    while True:
        try:
            msg = recv_queue.get(timeout=_heartbeat_interval)
        except Empty:
            # Heartbeat on idle
            _send_heartbeat(send_queue, name)
            _last_heartbeat = time.time()
            continue

        msg_type = msg.get("type", "")
        payload = msg.get("payload", {})

        # ── Heartbeat check ──
        now = time.time()
        if now - _last_heartbeat > _heartbeat_interval:
            _send_heartbeat(send_queue, name)
            _last_heartbeat = now

        # ── CGN_TRANSITION — experience from any consumer ──────────
        if msg_type == "CGN_TRANSITION":
            try:
                if payload.get("type") == "outcome":
                    # Delayed reward for existing transition
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
                            # Route to correct bus module (consumer→module mapping)
                            _haov_dest = {"language": "language",
                                          "social": "spirit",
                                          "reasoning": "spirit",
                                          "knowledge": "knowledge",
                                          "coding": "spirit"}.get(
                                _haov_consumer, _haov_consumer)
                            _send_msg(send_queue, "CGN_HAOV_VERIFY_REQ",
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
                _send_msg(send_queue, "CGN_WEIGHTS_MAJOR", name, "all", {
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
                send_queue.put({"type": "TIMECHAIN_COMMIT", "src": name,
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
                logger.debug("[CGNWorker] Surprise error: %s", e)

        # ── CGN_HAOV_VERIFY_RSP — verification from consumer ─────
        elif msg_type == "CGN_HAOV_VERIFY_RSP":
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
                    send_queue.put({"type": "TIMECHAIN_COMMIT", "src": name,
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
                logger.debug("[CGNWorker] HAOV verify error: %s", e)

        # ── CGN_INFERENCE_REQ — policy inference for remote processes ─
        # API_STUB: handler ready, awaits remote process senders (T2/T3
        # delegate path). Tracked I-003.
        elif msg_type == "CGN_INFERENCE_REQ":
            try:
                result = cgn.infer_social_action(
                    __import__("titan_plugin.logic.cgn",
                               fromlist=["SensoryContext"]).SensoryContext(
                        neuromods=payload.get("neuromods", {}),
                        epoch=payload.get("epoch", 0)),
                    user_features=payload.get("user_features"))
                _send_response(send_queue, name, msg.get("src", ""),
                               result, msg.get("rid"))
            except Exception as e:
                logger.debug("[CGNWorker] Inference error: %s", e)

        # ── CGN_KNOWLEDGE_REQ — route to knowledge worker ─────────
        elif msg_type == "CGN_KNOWLEDGE_REQ":
            # Forward to knowledge worker (when it exists)
            _send_msg(send_queue, "CGN_KNOWLEDGE_REQ", name,
                      "knowledge", payload)

        # ── QUERY — stats and diagnostics ─────────────────────────
        elif msg_type == "QUERY":
            from titan_plugin.core.profiler import handle_memory_profile_query
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
                _send_response(send_queue, name, msg.get("src", ""),
                               {"insights": insights}, msg.get("rid"))

        # ── MODULE_SHUTDOWN ───────────────────────────────────────
        elif msg_type == "MODULE_SHUTDOWN":
            logger.info("[CGNWorker] Shutdown requested: %s",
                        payload.get("reason", "?"))
            # Final save before exit
            try:
                cgn._save_state()
                _write_full_shm(cgn, shm_writer)
                logger.info("[CGNWorker] Final state saved")
            except Exception:
                pass
            break


# ── Helpers ────────────────────────────────────────────────────────────

def _write_full_shm(cgn, shm_writer):
    """Write complete weight snapshot to /dev/shm."""
    try:
        consumer_nets = {
            name: net.state_dict()
            for name, net in cgn._action_nets.items()
        }
        shm_writer.write_full(
            cgn._value_net.state_dict(),
            consumer_nets)
    except Exception as e:
        logger.debug("[CGNWorker] SHM write failed: %s", e)


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
        except Exception:
            pass

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
                        from titan_plugin.bus import emit_meta_cgn_signal
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
                    except Exception as _p9_err:
                        logger.warning(
                            "[META-CGN] Producer #9 knowledge.impasse_resolved emit FAILED "
                            "— type=%s err=%s (signal missed)",
                            impasse.get("type", "?"), _p9_err)
        except Exception:
            pass


def _send_msg(send_queue, msg_type, src, dst, payload, rid=None):
    try:
        send_queue.put_nowait({
            "type": msg_type, "src": src, "dst": dst,
            "ts": time.time(), "rid": rid, "payload": payload,
        })
    except Exception:
        from titan_plugin.bus import record_send_drop
        record_send_drop(src, dst, msg_type)


def _send_response(send_queue, src, dst, payload, rid):
    _send_msg(send_queue, "RESPONSE", src, dst, payload, rid)


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
    _send_msg(send_queue, "MODULE_HEARTBEAT", name, "guardian",
              {"rss_mb": round(rss_mb, 1)})
