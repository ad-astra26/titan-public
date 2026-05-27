"""social_worker — Python L2 module hosting X-posting + social pressure.

Per PLAN_microkernel_phase_c_s9_social_worker_extraction.md (parent rFP §4.C
of rFP_titan_hcl_l2_separation_strategy). Closes the migration gap left by
C-S8: cognitive_worker shipped 2026-05-05 turned spirit_worker into a
HEARTBEAT-ONLY STUB whenever microkernel.l0_rust_enabled=true, but the
X-posting code (SocialXGateway boot + ArchetypeDispatcher + SocialPressureMeter
+ catalyst accumulation) stayed in spirit_worker. This worker takes ownership.

ACTIVE UNDER: ``microkernel.social_worker_enabled = true`` ONLY.
Under the flag = false, the legacy ``spirit_worker`` (or its slim shim) owns
the X-posting code path per Maker D3(b) parity discipline. social_worker exits
early (after MODULE_READY) so no double-poster work runs simultaneously.

Owns (after chunks 9B-9F land):
  - ``SocialXGateway`` (X/Twitter posting; `titan_hcl/logic/social_x_gateway.py`)
  - ``ArchetypeDispatcher`` (`titan_hcl/logic/social_x/dispatcher.py`)
  - ``SocialPressureMeter`` (`titan_hcl/logic/social_pressure.py`)
  - Catalyst fan-in from 5 SOCIAL_CATALYST_* bus events
  - SHM slot ``data/social_x_state.bin`` (single writer, G21)
  - Bus publisher: ``X_POST_PUBLISHED``, ``SOCIAL_GRAPH_UPDATE``
  - Per-Titan independence — each Titan boots its own social_worker with its
    own gateway, own actions DB, own archetype_pool_scores

Per-Titan polling mode (chunks 9M-9P):
  - ``social_x.canonical_poller_titan_id`` = "T1" (default fleet mode) →
    only T1 polls X, broadcasts MENTION_RECEIVED / FELT_EXPERIENCE_CAPTURED /
    ENGAGEMENT_SNAPSHOT_TAKEN; T2/T3 subscribe + write to local DBs.
  - ``""`` or matches own titan_id → solo mode, this Titan polls itself.

Subscribe contract (full set after all chunks land):
  - EXPRESSION_FIRED (SOCIAL composite triggers post)
  - SOCIAL_RECEIVED (DM/mention)
  - KIN_SIGNAL (kin resonance)
  - SOCIAL_CATALYST_EUREKA / DREAM_SUMMARY / KIN_RESONANCE / ART_GENERATED / EMOTION_SHIFT
  - MENTION_RECEIVED / FELT_EXPERIENCE_CAPTURED / ENGAGEMENT_SNAPSHOT_TAKEN
    (non-canonical Titans, fleet mode)
  - MODULE_SHUTDOWN, SAVE_NOW

Publish contract:
  - X_POST_PUBLISHED (after every successful post)
  - SOCIAL_GRAPH_UPDATE (community_registry / followed-author state changed)
  - MENTION_RECEIVED / FELT_EXPERIENCE_CAPTURED / ENGAGEMENT_SNAPSHOT_TAKEN
    (canonical poller, fleet mode)

CHUNK SCOPE (9A — this commit): boot section (setup_worker_bus + pdeathsig +
flag gate) + ModuleSpec registration + heartbeat-only main loop. Engine init,
bus dispatch, SHM publisher, polling — all in subsequent chunks.

ARG ORDER per cognitive_worker template note: Guardian-spawned L2 workers
follow ``(recv_queue, send_queue, name, config)``. Stale docstring in
worker_bus_bootstrap.py shows wrong order — do not follow.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from queue import Empty

from titan_hcl import bus
from titan_hcl.core.state_registry import resolve_titan_id
from titan_hcl.core.module_error_handler import with_error_envelope
from titan_hcl.errors import Severity as _phase11_sev

logger = logging.getLogger(__name__)

# Heartbeat cadence per PLAN §2.1 — slower than cognitive_worker's 10s
# (social_worker is lower priority + lower update cadence).
_HEARTBEAT_INTERVAL_S = 30.0
_POLL_INTERVAL_S = 0.2
# Phase C-S9 chunk 9Q — post-dispatch orchestration tick cadence. Matches
# legacy spirit_worker which gated the same block on
# `_msl_tick_count % 30 == 0` at 1 Hz tick → ~30s. Configurable via
# `[social_x].post_dispatch_tick_interval_seconds` in titan_hcl/config.toml.
_DEFAULT_POST_DISPATCH_TICK_INTERVAL_S = 30.0

# Subscribe topics — chunk 9D adds the §4.C-spec'd subscriptions + 5 catalyst
# types. Polling-broadcast subscriptions (chunk 9O) join non-canonically.
_SOCIAL_WORKER_SUBSCRIBE_TOPICS = [
    # Lifecycle (chunk 9A)
    bus.MODULE_SHUTDOWN,
    bus.SAVE_NOW,
    # rFP §4.C-spec'd subscriptions (chunk 9D)
    bus.EXPRESSION_FIRED,
    bus.SOCIAL_RECEIVED,
    bus.KIN_SIGNAL,
    # ONE generic catalyst event — type in payload (replaces in-process
    # _x_catalysts.append flow at 8 producer sites in spirit_worker)
    bus.SOCIAL_CATALYST,
    # D-SPEC-66 v1.11.0 PLAN §1.5 — Maker force-post via /maker/x-force-post
    # API endpoint. api/maker.py publishes with dst="social" (was "spirit" —
    # legacy spirit_worker handler at L7995 dead under fleet-wide Phase C
    # 2026-05-14). Handler converts payload to internal
    # meter.on_catalyst_event with force_ungrounded=True (D8-3 site #6).
    bus.X_FORCE_POST,
    # Polling broadcasts — non-canonical Titans only (chunk 9O wires per-mode)
    bus.MENTION_RECEIVED,
    bus.FELT_EXPERIENCE_CAPTURED,
    bus.ENGAGEMENT_SNAPSHOT_TAKEN,
    # SPEC v1.12.0 §9.B health_monitor_worker — D-SPEC-67. social_worker
    # is the first owning worker for the HEAL_REQUEST contract: handles
    # action="refresh_session" against the live SocialXGateway in-proc
    # session state. Reply with HEAL_RESULT(dst="health_monitor"). This
    # preserves SOLE-sanctioned-X-path — health_monitor never instantiates
    # a second SocialXGateway in its own process.
    bus.HEAL_REQUEST,
]


@with_error_envelope(module_name="social_worker", subsystem="entry", severity=_phase11_sev.FATAL)
def social_worker_main(recv_queue, send_queue, name: str, config: dict) -> None:
    """Main loop for the social_worker subprocess.

    Chunk 9A skeleton — heartbeat-only main loop. Engine init (9B-9C),
    bus dispatcher (9D), SHM publisher (9E), publishers (9F), recency boost
    (9G), Observatory route (9H), polling (9M-9O) land in subsequent commits.
    """
    # === BOILERPLATE: spawn-mode sys.path bootstrap ===
    project_root = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # === BOILERPLATE: Phase B.2 §C7 socket-mode bus client setup ===
    from titan_hcl.core.worker_bus_bootstrap import setup_worker_bus
    try:
        recv_queue, send_queue, _bus_client = setup_worker_bus(
            name, recv_queue, send_queue,
            topics=_SOCIAL_WORKER_SUBSCRIBE_TOPICS,
        )
    except Exception as _err:
        logger.error(
            "[SocialWorker] setup_worker_bus failed: %s — exiting", _err,
            exc_info=True)
        return

    # === BOILERPLATE: pdeathsig installation ===
    try:
        from titan_hcl.core.worker_lifecycle import install_parent_death_signal
        install_parent_death_signal()
    except Exception as _err:
        logger.debug("[SocialWorker] pdeathsig install skipped: %s", _err)

    titan_id = resolve_titan_id()
    boot_ts = time.time()

    # === Flag-gated activation ===
    # Under social_worker_enabled=false: spirit_worker (or its slim shim) owns
    # X-posting per Maker D3(b). Skeleton no-ops so guardian doesn't restart-loop.
    # Registration in legacy_core.py is also flag-gated; this check is defensive.
    flag_on = bool(
        (config or {}).get("microkernel", {}).get("social_worker_enabled", False))
    if not flag_on:
        logger.info(
            "[SocialWorker] microkernel.social_worker_enabled=false — "
            "legacy spirit_worker owns X-posting in this mode. "
            "Entering heartbeat-only no-op loop.")
        _send_msg(send_queue, bus.MODULE_READY, name, "guardian", {
            "titan_id": titan_id, "ts": boot_ts, "flag_off_noop": True,
            "chunk": "9A",
        })
        _heartbeat_loop(recv_queue, send_queue, name, flag_off=True)
        return

    logger.info(
        "[SocialWorker] Booting (titan_id=%s, social_worker_enabled=true) — "
        "chunk 9A skeleton. Engine init / dispatcher / SHM / publishers / "
        "polling land in chunks 9B-9P.", titan_id)

    # === MODULE-SPECIFIC: engine init (chunks 9B-9C land here) ===
    # state_refs dict shape per cognitive_worker template — one key per
    # engine instance or None on init failure.
    state_refs: dict = _init_social_x_gateway(config, name, send_queue)
    state_refs["social_pressure_meter"] = _init_pressure_meter(config)
    state_refs["titan_id"] = titan_id
    state_refs["boot_ts"] = boot_ts

    # === MODULE-SPECIFIC: per-Titan polling-mode determination (chunk 9M) ===
    social_x_cfg = (config or {}).get("social_x", {}) or {}
    canonical_poller = social_x_cfg.get("canonical_poller_titan_id", "T1")
    is_canonical_poller = (canonical_poller == "" or canonical_poller == titan_id)
    state_refs["is_canonical_poller"] = is_canonical_poller
    logger.info(
        "[SocialWorker] polling mode: %s (canonical_poller_titan_id=%r, "
        "this titan=%s)",
        "CANONICAL (this Titan polls X)" if is_canonical_poller
        else "FLEET-CONSUMER (subscribes to broadcasts from %s)" % canonical_poller,
        canonical_poller, titan_id,
    )

    _send_msg(send_queue, bus.MODULE_READY, name, "guardian", {
        "titan_id": titan_id, "ts": boot_ts,
        "chunk": "9E",
        "is_canonical_poller": is_canonical_poller,
    })

    # === SHM state publisher (chunk 9E) ===
    # 1Hz daemon thread writing social_x_state.bin per G21 single-writer rule.
    # Consumed by /v4/social Observatory route + dim-live producers
    # (ANANDA[11]/[36]/[38]). Best-effort — publisher init failure is non-fatal,
    # worker continues without SHM state visibility.
    _start_state_publisher(state_refs, titan_id)

    # === Post-dispatch orchestrator (chunk 9Q) ===
    # Migrates spirit_worker:7770-8400 in here. The orchestrator owns the
    # SOLE sanctioned X-post path (gateway.post()) under
    # social_worker_enabled=true per
    # `feedback_social_x_gateway_post_is_sole_sanctioned_x_path.md`. Init is
    # non-fatal — if the gateway is missing the orchestrator runs but every
    # tick no-ops harmlessly. is_canonical_poller decides whether this
    # social_worker also broadcasts MENTION_RECEIVED to other social_workers
    # (chunk 9N).
    state_refs["post_dispatch_orchestrator"] = _init_post_dispatch_orchestrator(
        state_refs, name, send_queue, titan_id,
        is_canonical_poller=is_canonical_poller)

    # === Main loop with bus dispatcher (chunks 9A skeleton + 9D dispatcher) ===
    _main_loop(recv_queue, send_queue, name, state_refs, config)


def _init_post_dispatch_orchestrator(state_refs: dict, name: str,
                                       send_queue, titan_id: str,
                                       *,
                                       is_canonical_poller: bool = True):
    """Boot the PostDispatchOrchestrator. Returns the orchestrator or None
    on init failure (worker continues; tick path becomes a no-op)."""
    gateway = state_refs.get("social_x_gateway")
    meter = state_refs.get("social_pressure_meter")
    if gateway is None or meter is None:
        logger.warning(
            "[SocialWorker] PostDispatchOrchestrator NOT started — "
            "gateway=%s meter=%s. Catalysts will accumulate but no posts "
            "will fire. Investigate engine init failure above.",
            gateway is not None, meter is not None)
        return None
    try:
        from titan_hcl.logic.social_worker_post_dispatch import (
            PostDispatchOrchestrator,
        )
        # Phase 3 Chunk ω-bis (D-SPEC-88, 2026-05-18) — orchestrator needs
        # api_base + internal_key for /v4/llm-distill compose round-trip.
        # Same pattern as the metabolism gate URL construction above.
        _api_port = 7777
        _internal_key = ""
        try:
            from titan_hcl.config_loader import load_titan_config
            _cfg = load_titan_config() or {}
            _api_port = int(_cfg.get("api", {}).get("port", 7777))
            _internal_key = _cfg.get("api", {}).get("internal_key", "") or ""
        except Exception:
            pass
        orchestrator = PostDispatchOrchestrator(
            gateway=gateway, meter=meter, titan_id=titan_id,
            send_queue=send_queue, worker_name=name,
            is_canonical_poller=is_canonical_poller,
            api_base=f"http://127.0.0.1:{_api_port}",
            internal_key=_internal_key)
        logger.info(
            "[SocialWorker] PostDispatchOrchestrator started "
            "(canonical_poller=%s, api_base=:%d, internal_key=%s)",
            is_canonical_poller, _api_port,
            "set" if _internal_key else "MISSING")
        return orchestrator
    except Exception as _err:
        logger.warning(
            "[SocialWorker] PostDispatchOrchestrator init failed: %s — "
            "tick path will no-op until restart", _err, exc_info=True)
        return None


def _start_state_publisher(state_refs: dict, titan_id: str) -> None:
    """Start the social_x_state.bin SHM publisher daemon thread (chunk 9E)."""
    try:
        from titan_hcl.logic.social_x_state_publisher import (
            SocialXStatePublisher)
        publisher = SocialXStatePublisher(titan_id=titan_id)
    except Exception as _err:
        logger.warning(
            "[SocialWorker] SHM state publisher init failed: %s — "
            "social_x_state.bin will not be written. Observatory + dim-live "
            "consumers will see stale/empty data.", _err, exc_info=True)
        return

    import threading

    def _publisher_loop():
        # Writes at 1Hz per PLAN §2.5 + SOCIAL_X_STATE_REFRESH_HZ default.
        while True:
            try:
                publisher.publish(state_refs)
            except Exception as _err:
                logger.debug(
                    "[SocialWorker] publisher tick failed (non-fatal): %s",
                    _err)
            time.sleep(1.0)

    threading.Thread(
        target=_publisher_loop,
        daemon=True, name="social-x-state-publisher",
    ).start()
    logger.info("[SocialWorker] SHM state publisher started "
                "(slot=social_x_state, 1Hz)")


def _heartbeat_loop(recv_queue, send_queue, name: str, *, flag_off: bool) -> None:
    """Heartbeat-only main loop — handles MODULE_SHUTDOWN cleanly. Used under
    flag-false (no-op mode). Chunk 9A scope."""
    last_heartbeat = 0.0
    shutdown_requested = False

    while not shutdown_requested:
        now = time.time()
        if now - last_heartbeat >= _HEARTBEAT_INTERVAL_S:
            _send_heartbeat(send_queue, name, extra={
                "flag_off": flag_off, "chunk": "9A",
            })
            last_heartbeat = now
        try:
            msg = recv_queue.get(timeout=_POLL_INTERVAL_S)
        except Empty:
            continue
        msg_type = msg.get("type") if isinstance(msg, dict) else None
        if msg_type == bus.MODULE_SHUTDOWN:
            logger.info("[SocialWorker] MODULE_SHUTDOWN received — exiting.")
            shutdown_requested = True


def _main_loop(recv_queue, send_queue, name: str, state_refs: dict,
               config: dict | None = None) -> None:
    """Active main loop with bus dispatcher. Chunk 9D adds catalyst event
    handling. Chunks 9F/9N/9O extend with publishers + polling subscribers.
    Chunk 9G/9H consume archetype dispatch + Observatory state.

    Chunk 9Q (post-dispatch orchestration): drives
    ``PostDispatchOrchestrator.run_tick()`` every
    `[social_x].post_dispatch_tick_interval_seconds` (default 30s).
    """
    last_heartbeat = 0.0
    last_post_dispatch = 0.0
    last_emotion_check = 0.0
    shutdown_requested = False
    meter = state_refs.get("social_pressure_meter")
    orchestrator = state_refs.get("post_dispatch_orchestrator")

    # D-SPEC-66 v1.11.0 PLAN §1.4 — emotion-shift catalyst (D8-3 site
    # #4 close). Reuses existing infrastructure: ShmReaderBank.read_
    # neuromod() returns levels by name; detect_emotion_from_levels
    # (extracted helper in logic/neuromodulator.py) derives the
    # current_emotion byte-identically to NeuromodulatorSystem.
    # _detect_emotion. No SHM schema change; no new slots; no new
    # bus events. Was spirit_worker.py:6561 dead under fleet-wide
    # Phase C heartbeat-stub since 2026-05-14.
    _emotion_reader = None
    _emotion_check_interval_s = 5.0  # bounded — spirit_worker original
                                     # was ~1Hz MSL tick; 5s is enough
                                     # to catch transitions (which
                                     # happen on minute timescale).
    try:
        from titan_hcl.api.shm_reader_bank import ShmReaderBank
        _emotion_reader = ShmReaderBank(
            titan_id=state_refs.get("titan_id"))
        logger.info(
            "[SocialWorker] emotion-shift reader initialized — "
            "emotion check every %.1fs via ShmReaderBank.read_neuromod",
            _emotion_check_interval_s)
    except Exception as _emo_init_err:
        logger.warning(
            "[SocialWorker] emotion-shift reader init failed: %s — "
            "emotion_shift catalyst will not fire", _emo_init_err)
    state_refs.setdefault("_prev_emotion", "neutral")

    # Resolve tick interval once at loop start — re-resolved only on
    # restart per `feedback_no_quick_patches` (config-driven cadence is
    # set up front, not per-tick).
    tick_interval = _DEFAULT_POST_DISPATCH_TICK_INTERVAL_S
    try:
        social_x_cfg = (config or {}).get("social_x", {}) or {}
        tick_interval = float(social_x_cfg.get(
            "post_dispatch_tick_interval_seconds",
            _DEFAULT_POST_DISPATCH_TICK_INTERVAL_S))
    except Exception:
        pass

    while not shutdown_requested:
        now = time.time()
        if now - last_heartbeat >= _HEARTBEAT_INTERVAL_S:
            _send_heartbeat(send_queue, name, extra={
                "chunk": "9Q",
                "gateway_alive": state_refs.get("social_x_gateway") is not None,
                "meter_alive": meter is not None,
                "orchestrator_alive": orchestrator is not None,
            })
            last_heartbeat = now

        # Post-dispatch tick — chunk 9Q. Inline-runs the migrated
        # spirit_worker:7770-8400 orchestration loop. Best-effort: errors
        # inside the tick are logged but never raise out — they must
        # never block bus dispatch / heartbeat.
        if (orchestrator is not None
                and now - last_post_dispatch >= tick_interval):
            last_post_dispatch = now
            try:
                orchestrator.run_tick()
            except Exception as _err:
                logger.warning(
                    "[SocialWorker] post-dispatch tick raised: %s",
                    _err, exc_info=True)

        # D-SPEC-66 v1.11.0 PLAN §1.4 emotion-shift tick (D8-3 site #4).
        # Read neuromod levels via ShmReaderBank, derive current_emotion
        # via byte-identical helper, on transition call meter
        # .on_catalyst_event(type=emotion_shift) in-process.
        if (meter is not None and _emotion_reader is not None
                and now - last_emotion_check >= _emotion_check_interval_s):
            last_emotion_check = now
            try:
                _nm_payload = _emotion_reader.read_neuromod()
                if _nm_payload is not None:
                    from titan_hcl.logic.neuromodulator import (
                        detect_emotion_from_levels)
                    _levels = {
                        name: float(meta.get("level", 0.5))
                        for name, meta in (
                            _nm_payload.get("modulators") or {}).items()
                    }
                    if _levels:
                        _cur_emo, _ = detect_emotion_from_levels(_levels)
                        _prev_emo = state_refs.get(
                            "_prev_emotion", "neutral")
                        if _cur_emo != _prev_emo:
                            _handle_catalyst_event(meter, {
                                "type": "emotion_shift",
                                "significance": 0.5,
                                "content":
                                    f"{_prev_emo} → {_cur_emo}",
                                "data": {"from": _prev_emo,
                                         "to": _cur_emo},
                            })
                            state_refs["_prev_emotion"] = _cur_emo
                            logger.info(
                                "[SocialWorker] emotion_shift catalyst "
                                "emitted: %s → %s",
                                _prev_emo, _cur_emo)
            except Exception as _emo_err:
                logger.warning(
                    "[SocialWorker] emotion-shift tick raised: %s",
                    _emo_err)

        try:
            msg = recv_queue.get(timeout=_POLL_INTERVAL_S)
        except Empty:
            continue

        msg_type = msg.get("type") if isinstance(msg, dict) else None
        if msg_type == bus.MODULE_SHUTDOWN:
            logger.info("[SocialWorker] MODULE_SHUTDOWN received — exiting.")
            shutdown_requested = True
            continue
        if msg_type == bus.SAVE_NOW:
            # Chunk 9E adds SHM-slot write here.
            continue

        # === Catalyst event dispatch (chunk 9D + 9I — generic SOCIAL_CATALYST) ===
        # Producer-side: chunk 9I wires all 8 spirit_worker callsites to
        # publish SOCIAL_CATALYST under flag-true (legacy in-process append
        # under flag-false, with D8 markers for retirement). Payload shape
        # matches the legacy _x_catalysts.append dict: {type, significance,
        # content, data, [force_ungrounded]}.
        if msg_type == bus.SOCIAL_CATALYST and meter is not None:
            _handle_catalyst_event(meter, msg.get("payload") or {})
            continue

        # D-SPEC-66 v1.11.0 PLAN §1.5 — Maker X_FORCE_POST handler.
        # Convert /maker/x-force-post API payload into internal
        # SOCIAL_CATALYST(type=<catalyst_type>, force_ungrounded=True)
        # via direct meter.on_catalyst_event call (D8-3 site #6 close).
        # Was spirit_worker.py:7995 dead under Phase C heartbeat-stub.
        if msg_type == bus.X_FORCE_POST and meter is not None:
            _fp = msg.get("payload") or {}
            _fp_topic = str(_fp.get("topic", "")).strip()
            _fp_text = str(_fp.get("text_hint", "")).strip()
            _fp_type = str(_fp.get("catalyst_type", "maker_force")).strip() \
                       or "maker_force"
            if _fp_topic:
                _handle_catalyst_event(meter, {
                    "type": _fp_type,
                    "content": _fp_text or
                               f"Maker requested a post about: {_fp_topic}",
                    "significance": 1.0,
                    "force_ungrounded": True,
                    "data": {"topic": _fp_topic,
                             "text_hint": _fp_text},
                })
                logger.info("[SocialWorker] X_FORCE_POST queued maker "
                            "catalyst: topic=%r type=%s",
                            _fp_topic, _fp_type)
            continue

        # T1 canary closure — EXPRESSION_FIRED carries SOCIAL composite urge.
        # Spirit_worker:9001-9006 used to call meter.on_social_fire(urge) when
        # SOCIAL composite fired. Under flag-true the meter is None in
        # spirit_worker; this bus subscription replaces that callsite. Without
        # this, urge never accumulates → meter.should_post never trips →
        # social_worker never posts despite catalysts flowing. Verified empirically
        # at T1 canary 2026-05-12 10:58 UTC (urge stuck at 22.5 / threshold 50).
        if msg_type == bus.EXPRESSION_FIRED:
            if meter is not None:
                payload = msg.get("payload") or {}
                composite = payload.get("composite", "")
                if composite == "SOCIAL":
                    urge = float(payload.get("urge", 1.0))
                    meter.on_social_fire(urge)
            continue
        if msg_type in (bus.SOCIAL_RECEIVED, bus.KIN_SIGNAL):
            logger.debug("[SocialWorker] received %s (handler TBD)", msg_type)
            continue

        # SPEC v1.12.0 §9.B health_monitor_worker — D-SPEC-67. Handles
        # HEAL_REQUEST(dst="social"); dispatches by action against the
        # live in-proc SocialXGateway state; replies HEAL_RESULT to
        # health_monitor. Best-effort — exceptions become reply=failure;
        # never crash the dispatcher.
        if msg_type == bus.HEAL_REQUEST:
            _handle_heal_request(
                msg.get("payload") or {}, state_refs,
                send_queue, name)
            continue

        # Polling broadcasts — chunk 9O consumer side.
        # The canonical poller (default T1) broadcasts these after each
        # successful poll cycle / DB write. Non-canonical Titans
        # subscribe and INSERT OR IGNORE the row into their own local
        # DB so each Titan owns a complete (eventually-consistent) copy.
        # Idempotency: tweet_id (mention_tracking) / (post_tweet_id,
        # snapshot_time) / id (felt_experiences) UNIQUE constraints
        # turn re-broadcasts into no-ops.
        if msg_type == bus.MENTION_RECEIVED:
            if not state_refs.get("is_canonical_poller", True):
                _ingest_mention_received(
                    state_refs.get("social_x_gateway"),
                    msg.get("payload") or {})
            continue
        if msg_type == bus.FELT_EXPERIENCE_CAPTURED:
            if not state_refs.get("is_canonical_poller", True):
                _ingest_felt_experience(msg.get("payload") or {})
            continue
        if msg_type == bus.ENGAGEMENT_SNAPSHOT_TAKEN:
            if not state_refs.get("is_canonical_poller", True):
                _ingest_engagement_snapshot(msg.get("payload") or {})
            continue


def _ingest_mention_received(gateway, payload: dict) -> None:
    """Phase C-S9 chunk 9O. Non-canonical Titan consumer for
    MENTION_RECEIVED bus event. INSERT OR IGNORE the broadcast row into
    the local mention_tracking table so per-Titan independence holds —
    each social_worker owns a complete copy of mention state.

    Idempotency: ``UNIQUE`` on ``tweet_id`` in mention_tracking turns
    re-broadcasts (or canonical-poller restarts that re-emit) into
    no-ops. Best-effort: a missing DB / locked DB just logs DEBUG."""
    if gateway is None:
        return
    try:
        db = gateway._db()
        db.execute(
            "INSERT OR IGNORE INTO mention_tracking "
            "(tweet_id, author, author_handle, text, our_post_id, "
            "titan_id, status, relevance_score, discovered_at) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (
                payload.get("tweet_id", ""),
                payload.get("author", ""),
                payload.get("author_handle", ""),
                payload.get("text", "")[:500],
                payload.get("our_post_id", ""),
                payload.get("titan_id", ""),
                payload.get("status", "pending"),
                float(payload.get("relevance_score", 0.0) or 0.0),
                float(payload.get("discovered_at", time.time()) or
                      time.time()),
            ))
        db.commit()
        db.close()
    except Exception as _err:
        logger.debug(
            "[SocialWorker] MENTION_RECEIVED ingest failed: %s", _err)


def _ingest_felt_experience(payload: dict) -> None:
    """Phase C-S9 chunk 9O. Non-canonical consumer for
    FELT_EXPERIENCE_CAPTURED. Writes the broadcast row into local
    ``data/events_teacher.db`` ``felt_experiences`` table with idempotent
    INSERT OR IGNORE. Schema mirrors events_teacher writer
    (events_teacher.py:374-386).

    EventsTeacher itself is the canonical writer on the canonical Titan.
    On non-canonical Titans we use the same DB file shape so reads from
    the local DB (e.g. by social_narrator, archetype dispatcher) see
    the same data the canonical Titan sees.

    Idempotency: ``id`` is the canonical primary key; with
    INSERT OR IGNORE re-broadcasts are no-ops."""
    import sqlite3
    try:
        db = sqlite3.connect("data/events_teacher.db", timeout=5)
        db.execute(
            "INSERT OR IGNORE INTO felt_experiences "
            "(id, titan_id, source, author, topic, sentiment, arousal, "
            "relevance, concept_signals, semantic_concepts, "
            "felt_summary, contagion_type, mode, window_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                payload.get("id"),
                payload.get("titan_id", ""),
                payload.get("source", ""),
                payload.get("author", ""),
                payload.get("topic", ""),
                float(payload.get("sentiment", 0.0) or 0.0),
                float(payload.get("arousal", 0.0) or 0.0),
                float(payload.get("relevance", 0.0) or 0.0),
                payload.get("concept_signals", "[]"),
                payload.get("semantic_concepts", "[]"),
                payload.get("felt_summary", ""),
                payload.get("contagion_type", ""),
                payload.get("mode", ""),
                int(payload.get("window_id", 0) or 0),
                float(payload.get("created_at", time.time())
                      or time.time()),
            ))
        db.commit()
        db.close()
    except Exception as _err:
        logger.debug(
            "[SocialWorker] FELT_EXPERIENCE_CAPTURED ingest failed: %s",
            _err)


def _ingest_engagement_snapshot(payload: dict) -> None:
    """Phase C-S9 chunk 9O. Non-canonical consumer for
    ENGAGEMENT_SNAPSHOT_TAKEN. Writes the broadcast row into local
    ``data/events_teacher.db`` ``engagement_snapshots`` table. Schema
    mirrors events_teacher writer (events_teacher.py:432-440)."""
    import sqlite3
    try:
        db = sqlite3.connect("data/events_teacher.db", timeout=5)
        db.execute(
            "INSERT OR IGNORE INTO engagement_snapshots "
            "(id, titan_id, tweet_id, likes, replies, quotes, "
            "delta_likes, delta_replies, delta_quotes, checked_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                payload.get("id"),
                payload.get("titan_id", ""),
                payload.get("tweet_id", ""),
                int(payload.get("likes", 0) or 0),
                int(payload.get("replies", 0) or 0),
                int(payload.get("quotes", 0) or 0),
                int(payload.get("delta_likes", 0) or 0),
                int(payload.get("delta_replies", 0) or 0),
                int(payload.get("delta_quotes", 0) or 0),
                float(payload.get("checked_at", time.time())
                      or time.time()),
            ))
        db.commit()
        db.close()
    except Exception as _err:
        logger.debug(
            "[SocialWorker] ENGAGEMENT_SNAPSHOT_TAKEN ingest failed: %s",
            _err)


def _handle_catalyst_event(meter, payload: dict) -> None:
    """Translate a SOCIAL_CATALYST bus event payload into a CatalystEvent
    instance and feed it to the meter. Payload shape mirrors the legacy
    spirit_worker _x_catalysts.append dict verbatim:
        {type: str, significance: float, content: str, data: dict,
         [force_ungrounded: bool]}
    Backward-compatible with all 8 producer sites in spirit_worker."""
    try:
        from titan_hcl.logic.social_pressure import CatalystEvent
        catalyst_type = str(payload.get("type", "unknown"))
        event = CatalystEvent(
            type=catalyst_type,
            significance=float(payload.get("significance", 0.5)),
            content=str(payload.get("content", "")),
            data=payload.get("data") or {},
        )
        meter.on_catalyst_event(event)
    except Exception as _err:
        logger.warning("[SocialWorker] catalyst dispatch failed: %s payload=%s",
                       _err, payload)


# === MODULE-SPECIFIC: SocialXGateway + ArchetypeDispatcher init ===
# Chunk 9B (PLAN §2.2). Identical wiring to spirit_worker's previous boot
# block (lines 1259-1355) — copied verbatim then adapted: log prefix +
# bus-publish src field. Same injection chain: OutputVerifier → VCB →
# metabolism gate. ArchetypeDispatcher constructed lazily inside the
# gateway's `_ensure_archetype_dispatcher()` (per the existing pattern at
# social_x_gateway.py:234) — no separate construction here.
def _init_social_x_gateway(config: dict, name: str, send_queue) -> dict:
    """Boot the SocialXGateway + dependency injection chain.

    Returns a state_refs dict with `social_x_gateway` populated (or None on
    init failure — boot is non-fatal so worker continues heartbeating). The
    dispatcher is lazy-constructed inside the gateway on first probe.
    """
    state_refs: dict = {
        "social_x_gateway": None,
        "archetype_dispatcher": None,  # populated lazily by gateway on first probe
    }

    titan_id = resolve_titan_id()
    data_dir = (config.get("data_dir") or "./data")
    cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "config.toml")

    try:
        from titan_hcl.logic.social_x_gateway import SocialXGateway
        gateway = SocialXGateway(
            db_path=os.path.join(data_dir, "social_x.db"),
            config_path=cfg_path,
            telemetry_path=os.path.join(data_dir, "social_x_telemetry.jsonl"),
        )
    except Exception as _gw_err:
        logger.warning("[SocialWorker] SocialXGateway boot failed: %s", _gw_err,
                       exc_info=True)
        return state_refs

    # Inject OutputVerifier for security gating of X posts/replies
    try:
        from titan_hcl.logic.output_verifier import OutputVerifier
        wallet_path = (config.get("network", {}) or {}).get(
            "wallet_keypair_path", "data/titan_identity_keypair.json")
        gateway.set_output_verifier(OutputVerifier(
            titan_id=titan_id, data_dir="data/timechain",
            keypair_path=wallet_path))
        logger.info("[SocialWorker] OVG injected into SocialXGateway")
    except Exception as _ovg_err:
        logger.warning("[SocialWorker] OVG injection failed: %s", _ovg_err)

    # Inject VerifiedContextBuilder for memory-enriched replies
    try:
        from titan_hcl.logic.verified_context_builder import VerifiedContextBuilder
        known_users = []
        try:
            import sqlite3
            sg = sqlite3.connect(
                os.path.join(data_dir, "social_graph.db"), timeout=5)
            known_users = [r[0] for r in sg.execute(
                "SELECT user_id FROM users ORDER BY interaction_count DESC LIMIT 100"
            ).fetchall()]
            sg.close()
        except Exception:
            pass
        gateway.set_context_builder(VerifiedContextBuilder(
            data_dir=data_dir, known_users=known_users))
        logger.info("[SocialWorker] VCB injected into SocialXGateway "
                    "(known_users=%d)", len(known_users))
    except Exception as _vcb_err:
        logger.warning("[SocialWorker] VCB injection failed: %s", _vcb_err)

    # Mainnet Lifecycle Wiring rFP (2026-04-20): metabolism gate via HTTP.
    try:
        import httpx
        api_port = 7777
        try:
            from titan_hcl.config_loader import load_titan_config
            api_port = int(load_titan_config().get("api", {}).get("port", 7777))
        except Exception:
            pass
        gate_url = f"http://127.0.0.1:{api_port}/v4/metabolism/evaluate-gate"

        def _metabolism_gate_call(feature: str, caller: str):
            try:
                r = httpx.get(gate_url,
                              params={"feature": feature, "caller": caller},
                              timeout=2.0)
                if r.status_code == 200:
                    d = r.json().get("data", {})
                    return (bool(d.get("should_proceed", True)),
                            float(d.get("rate_multiplier", 1.0)))
            except Exception:
                pass
            return (True, 1.0)  # fail-open

        gateway.set_metabolism_gate(_metabolism_gate_call)
        logger.info("[SocialWorker] Metabolism gate injected into SocialXGateway")
    except Exception as _mge_err:
        logger.warning("[SocialWorker] Metabolism gate injection failed: %s",
                       _mge_err)

    # Chunk 9F — X_POST_PUBLISHED bus event publisher injection.
    # Gateway invokes this callback after every successful post (verified or
    # posted). Subscribers consume X_POST_PUBLISHED for: events_teacher
    # engagement reaper (+12h scoring), Observatory live timeline, future
    # KIN_RESONANCE cross-Titan coordination (Phase 2c).
    def _x_post_published_callback(*, tweet_id: str, titan_id: str,
                                     post_type: str, archetype: str,
                                     pool: str, source_id: str, status: str):
        try:
            send_queue.put({
                "type": bus.X_POST_PUBLISHED, "src": name, "dst": "all",
                "payload": {
                    "tweet_id": tweet_id, "titan_id": titan_id,
                    "post_type": post_type, "archetype": archetype,
                    "pool": pool, "source_id": source_id, "status": status,
                },
                "ts": time.time(),
            })
        except Exception as _err:
            logger.debug("[SocialWorker] X_POST_PUBLISHED publish failed: %s",
                         _err)

    gateway.set_post_success_callback(_x_post_published_callback)
    logger.info("[SocialWorker] X_POST_PUBLISHED bus publisher injected")

    state_refs["social_x_gateway"] = gateway
    logger.info("[SocialWorker] SocialXGateway v3 booted: db=%s/social_x.db",
                data_dir)
    return state_refs


# Chunk 9C (PLAN §2.3) — SocialPressureMeter init.
# Same instantiation shape as the legacy spirit_worker block (line 1383-1385,
# now flag-skipped under social_worker_enabled=true). Reads [social_presence]
# config — the meter's ctor restores its own state from
# data/social_pressure_state.json automatically. Each Titan boots its OWN
# meter with its OWN per-Titan state file (per Maker Q1 — per-Titan
# independence). on_social_relief subscriber bus-wiring lands in chunk 9D.
def _init_pressure_meter(config: dict):
    """Boot the SocialPressureMeter. Returns the meter or None on failure
    (boot is non-fatal — on_social_relief becomes unavailable but worker
    keeps running)."""
    try:
        from titan_hcl.logic.social_pressure import SocialPressureMeter
        sp_cfg = (config or {}).get("social_presence", {}) or {}
        meter = SocialPressureMeter(sp_cfg)
        logger.info("[SocialWorker] SocialPressureMeter booted "
                    "(threshold=%.1f, urge_restored=%.1f, catalysts_restored=%d)",
                    meter.post_threshold, meter.urge_accumulator,
                    len(meter.catalyst_events))
        return meter
    except Exception as _err:
        logger.warning("[SocialWorker] SocialPressureMeter boot failed: %s",
                       _err)
        return None


# === BOILERPLATE: helpers (extract to shared _worker_skeleton.py when 4th
# L2 worker lands per cognitive_worker note line 117) ===

def _send_msg(send_queue, msg_type: str, src: str, dst: str,
              payload: dict) -> None:
    """Send a typed message via the worker's send_queue. Best-effort —
    swallows queue errors so worker boot never crashes on bus hiccups."""
    try:
        send_queue.put({
            "type": msg_type, "src": src, "dst": dst,
            "payload": payload, "ts": time.time(),
        })
    except Exception as _err:
        logger.debug("[SocialWorker] _send_msg(%s) failed: %s", msg_type, _err)


def _send_heartbeat(send_queue, name: str, extra: dict | None = None) -> None:
    """Emit MODULE_HEARTBEAT for guardian liveness."""
    payload = {"ts": time.time()}
    if extra:
        payload.update(extra)
    _send_msg(send_queue, bus.MODULE_HEARTBEAT, name, "guardian", payload)


# ── SPEC v1.12.0 §9.B health_monitor_worker — D-SPEC-67 HEAL_REQUEST handler ──


def _handle_heal_request(payload: dict, state_refs: dict,
                          send_queue, name: str) -> None:
    """Dispatch HEAL_REQUEST(dst="social") against the live SocialXGateway.

    Per SPEC v1.12.0 §8.7 HEAL_REQUEST + HEAL_RESULT contract: replies
    HEAL_RESULT(dst="health_monitor") with success + reason; correlation_id
    is echoed verbatim from the request.

    Supported actions:
      - "refresh_session" — call SocialXGateway._refresh_session(api_key,
        proxy); success=True iff a non-empty session string returned.

    Unknown actions reply success=False reason="unknown_action_<name>".
    Gateway-absent (worker boot incomplete) replies success=False
    reason="gateway_not_initialized".
    """
    correlation_id = str(payload.get("correlation_id", ""))
    action = str(payload.get("action", ""))
    plugin_name = str(payload.get("plugin", ""))

    def _reply(success: bool, reason: str) -> None:
        _send_msg(send_queue, bus.HEAL_RESULT, name, "health_monitor", {
            "plugin": plugin_name,
            "action": action,
            "success": success,
            "reason": reason,
            "correlation_id": correlation_id,
            "ts": time.time(),
        })

    gateway = state_refs.get("social_x_gateway")
    if gateway is None:
        logger.warning(
            "[SocialWorker] HEAL_REQUEST plugin=%s action=%s — "
            "gateway not initialized; replying failure.",
            plugin_name, action)
        _reply(False, "gateway_not_initialized")
        return

    if action == "refresh_session":
        try:
            # CANONICAL resolution: gateway._load_config() is the production
            # source-of-truth for the merged social_x config — it does the
            # full 3-layer merge AND applies the gateway's own fallback
            # chains (sx.api_key → stealth_sage.twitterapi_io_key;
            # sx.proxy → twitter_social.webshare_static_url per
            # logic/social_x_gateway.py:472-473). Using this auto-tracks
            # any future schema changes the gateway adopts.
            #
            # Phase 1.10 (2026-05-17): the prior implementation re-did
            # resolution by hand and missed the `webshare_static_url`
            # proxy fallback → twitterapi.io returned
            # `{"detail":"proxy is required"}` HTTP 400 → _refresh_session
            # returned empty even when credentials were fine. Verified
            # live by diagnostic on T2 2026-05-17 11:30 UTC.
            api_key = ""
            proxy = ""
            try:
                # gateway._load_config() returns a FLAT dict (NOT nested
                # under "social_x" — verified by reading
                # logic/social_x_gateway.py:451-493). Keys like 'api_key'
                # and 'proxy' are at the top level with the gateway's
                # full fallback resolution already applied
                # (sx.api_key → stealth_sage.twitterapi_io_key;
                #  sx.proxy → twitter_social.webshare_static_url).
                gw_cfg = gateway._load_config()
                api_key = gw_cfg.get("api_key", "") or ""
                proxy = gw_cfg.get("proxy", "") or ""
            except Exception as _cfg_err:
                logger.debug(
                    "[SocialWorker] HEAL refresh_session: "
                    "gateway._load_config() failed (%s) — "
                    "falling back to load_titan_config",
                    _cfg_err)
                # Defense-in-depth fallback: direct config loader
                # mirroring the gateway's resolution chain by hand.
                try:
                    from titan_hcl.config_loader import (
                        load_titan_config)
                    full_cfg = load_titan_config(force_reload=True)
                    sx_cfg = full_cfg.get("social_x") or {}
                    sage_cfg = full_cfg.get("stealth_sage") or {}
                    tw_cfg = full_cfg.get("twitter_social") or {}
                    api_key = (sx_cfg.get("api_key", "")
                                or sage_cfg.get("twitterapi_io_key", "")
                                or "")
                    proxy = (sx_cfg.get("proxy", "")
                              or tw_cfg.get("webshare_static_url", "")
                              or "")
                except Exception:
                    pass
            # Final test-fixture fallback: gateway with config dict attr
            # (mocks).
            _gw_cfg = getattr(gateway, "config", {}) or {}
            if not api_key:
                api_key = _gw_cfg.get("api_key", "")
            if not proxy:
                proxy = _gw_cfg.get("proxy", "")
            if not api_key:
                _reply(False, "api_key_missing")
                return
            new_session = gateway._refresh_session(api_key, proxy)
            if new_session:
                logger.info(
                    "[SocialWorker] HEAL_REQUEST refresh_session SUCCEEDED "
                    "(plugin=%s) — new session installed",
                    plugin_name)
                _reply(True, "session_refreshed")
            else:
                logger.warning(
                    "[SocialWorker] HEAL_REQUEST refresh_session returned "
                    "empty (plugin=%s) — credentials missing or "
                    "twitterapi.io login failed",
                    plugin_name)
                _reply(False, "refresh_returned_empty")
        except Exception as e:
            logger.warning(
                "[SocialWorker] HEAL_REQUEST refresh_session raised: %s "
                "(plugin=%s)", e, plugin_name, exc_info=True)
            _reply(False, f"refresh_exception:{type(e).__name__}")
        return

    logger.warning(
        "[SocialWorker] HEAL_REQUEST unknown action=%r plugin=%s — "
        "replying failure.", action, plugin_name)
    _reply(False, f"unknown_action:{action}")


__all__ = ("social_worker_main",)
