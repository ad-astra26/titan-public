"""expression_worker — Python L2 module hosting the ExpressionManager instance.

Per rFP_titan_hcl_l2_separation_strategy.md §4.B (LOCKED 2026-05-05;
SHIPPED 2026-05-15) + SPEC §9.B expression_worker block + D-SPEC-NN.

ACTIVE: always-on autostart under l0_rust_enabled=true. Owns the 6
EXPRESSION composites (SPEAK / ART / MUSIC / SOCIAL / KIN_SENSE /
LONGING) extracted from cognitive_worker per §4.B Track 3 ("included
inside cognitive_worker for #1 cognitive_worker (avoid double surgery);
split out once cognitive_worker is parity-verified" — closure 2026-05-15).

Owns:
  - ExpressionManager (titan_hcl/logic/expression_composites.py) +
    the 6 composites registered at boot via create_speak / create_art /
    create_music / create_social / create_kin_sense / create_longing.
  - Composite ledger: per-composite (urge, level, last_fired_ts,
    fire_count, intensity) state snapshot for /v4/expression-composites.
  - expression_composites_state.bin SHM slot (G21 single-writer; 1 Hz;
    payload per SPEC §7.1).
  - Composite META-CGN EdgeDetector (relocated from
    cognitive_worker._composite_meta_cgn_edge — was coordinator-attached
    in process-local memory).

Bus subscriptions:
  REQUIRED — bus.KERNEL_EPOCH_TICK   (drives evaluate_all per Maker
                                       decision Q2 — adaptive 1–30s
                                       cognitive epoch coupling)
             + bus.MODULE_SHUTDOWN
             + bus.SAVE_NOW          (persist EdgeDetector state)
  OPTIONAL — bus.EXPRESSION_RELIEF   (composite-ledger relief deltas;
                                       wired when producers reactivated
                                       post-D8)
             + bus.DREAM_STATE_CHANGED (gate evaluate_all when dreaming)
             + bus.REASONING_STATS_UPDATED (transitional bridge for
                                       chain-driven composer signals)

Bus publications:
  - EXPRESSION_FIRED                (per Tier-2 composite fire; producer
                                     field updated cognitive_worker →
                                     expression_worker per SPEC §8.7)
  - EXPRESSION_COMPOSITES_UPDATED   (1 Hz coalesced; full ledger via SHM)
  - SPEAK_REQUEST_PENDING           (per Tier-1 SPEAK detection; consumed
                                     by cognitive_worker which assembles
                                     the language-pipeline SPEAK_REQUEST
                                     using consciousness/msl/exp_orch
                                     state in its own address space)
  - COMPOSITION_READY               (reserved — wired when /v4/composer/*
                                     pipeline lands per rFP §4.B)
  - SOCIAL_CATALYST(type=strong_composition) (D8-3 catalyst-site #8
                                     closure — composite fire with
                                     level ≥ 7 + confidence ≥ 0.8
                                     publishes SOCIAL_CATALYST event;
                                     consumed by social_worker.meter)
  - NS_REWARD                       (per composite fire; cognitive_worker
                                     subscribes + calls record_outcome
                                     on its NeuralNervousSystem instance)
  - META_CGN_SIGNAL via emit_meta_cgn_signal (per composite fire that
                                     crosses EdgeDetector threshold)
  - MODULE_READY / MODULE_HEARTBEAT / MODULE_SHUTDOWN

Implementation reference: social_graph_worker.py (CANONICAL §9.B
TEMPLATE) for `=== BOILERPLATE ===` sections + memory_worker.py
`_periodic_publish_loop` for the 1Hz SHM publisher.

ARG ORDER: Guardian-spawned L2 workers follow
``(recv_queue, send_queue, name, config)``.

Migration map per SPEC §9.B vX.Y.Z:
  cognitive_worker.py:843-862  → REMOVED (ExpressionManager init block;
                                   expression_worker owns)
  cognitive_worker.py:2183-2329 → REMOVED (Block 8 evaluate_all;
                                   expression_worker drives on
                                   KERNEL_EPOCH_TICK)
  cognitive_worker.py:1582      → REMOVED (expression_manager state_refs
                                   read)
  cognitive_worker.py:1636      → REMOVED (expression_manager.get_stats
                                   in /v4/expression-composites path;
                                   route reads SHM directly via
                                   ExpressionCompositesShmReader)
  spirit_worker.py:12273        → DEPRECATED (strong_composition
                                   catalyst-site #8 emit; expression_worker
                                   now publishes SOCIAL_CATALYST direct.
                                   spirit_worker site stays gated under
                                   l0_rust=false fallback until D8-3.)
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time
from queue import Empty
from typing import Any, Optional

from titan_hcl import bus
from titan_hcl.core.module_error_handler import with_error_envelope
from titan_hcl.errors import Severity as _phase11_sev

logger = logging.getLogger(__name__)

# Module name (matches Guardian registry per SPEC §9.B).
MODULE_NAME = "expression_worker"

# Cadence + lifecycle constants.
_HEARTBEAT_INTERVAL_S = 10.0            # SPEC §10.B MODULE_HEARTBEAT_INTERVAL_S
_POLL_INTERVAL_S = 0.2                  # recv loop poll cadence
_SHM_PUBLISH_INTERVAL_S = 1.0           # expression_composites_state.bin 1 Hz
_STATE_CHECKPOINT_INTERVAL_S = 300.0   # periodic edge_detector disk checkpoint
_STATS_NOTIFY_INTERVAL_S = 5.0          # EXPRESSION_COMPOSITES_UPDATED bus notify cadence

# Internal evaluate_all cadence. Originally the PLAN proposed KERNEL_EPOCH_TICK
# (Maker Q2) as the trigger, on the assumption that the Rust pi_heartbeat
# emission would fan out to this worker reliably. Live verification on T3
# (2026-05-15 08:43 UTC cascade) showed the bus event is published but the
# worker's recv_queue never observed it under the current broker fanout —
# 0 received in 3 minutes. As a robustness fix, expression_worker drives
# evaluate_all from its own internal timer at 5s cadence (matches the
# composites' slow accumulation dynamics — Tier-2 urges build over minutes,
# not sub-second). The KERNEL_EPOCH_TICK subscriber path remains active as
# a secondary trigger when the bus event does arrive, so the slower paths
# of /v4/expression-composites converge in either case.
_EVALUATE_ALL_INTERVAL_S = 5.0

# Thresholds for catalyst-site #8 — strong_composition SOCIAL_CATALYST emit.
# Mirrors spirit_worker.py:12273 site that this worker absorbs per the
# D8 RETIREMENT PREREQUISITE block at the top of
# rFP_titan_hcl_l2_separation_strategy.md.
_STRONG_COMPOSITION_LEVEL_GATE: int = 7
_STRONG_COMPOSITION_CONFIDENCE_GATE: float = 0.8

# EdgeDetector intensity threshold for META-CGN signal emit (matches the
# value used in the legacy spirit_worker.py block — `0.2` urge/3 floor).
_META_CGN_INTENSITY_GATE: float = 0.2


# Topics expression_worker subscribes to (per SPEC §9.B).
_EXPRESSION_WORKER_SUBSCRIBE_TOPICS: list[str] = [
    bus.KERNEL_EPOCH_TICK,          # drives evaluate_all per Maker Q2
    bus.MODULE_SHUTDOWN,
    bus.SAVE_NOW,                   # persist EdgeDetector state
]
# Optional subscriptions added at runtime IFF the constants exist
# (forward-compat — these are NULL-safe if not yet defined in bus.py):
_OPTIONAL_TOPICS = [
    "EXPRESSION_RELIEF",
    "DREAM_STATE_CHANGED",
    "REASONING_STATS_UPDATED",
    # L3 housekeeping closure 2026-05-26: cross-process translator
    # stats bridge — parent (translator owner under l0_rust=true)
    # publishes a snapshot every ~5s; we cache it for the next 1Hz
    # SHM publish so translator-derived fields stop reading defaults.
    "EXPRESSION_TRANSLATOR_STATS_UPDATED",
]


# ── Lifecycle helpers (mirror social_graph_worker template) ───────────


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


def _send_heartbeat(send_queue, name: str, extra: Optional[dict] = None) -> None:
    """Emit MODULE_HEARTBEAT to guardian_HCL with current RSS."""
    try:
        import resource
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        rss_mb = 0.0
    payload = {"alive": True, "ts": time.time(), "rss_mb": round(rss_mb, 1)}
    if extra:
        payload.update(extra)
    _send_msg(send_queue, bus.MODULE_HEARTBEAT, name, "guardian", payload)


# ── ExpressionManager bootstrap ───────────────────────────────────────


def _init_expression_manager():
    """Construct ExpressionManager + register all 6 composites.

    Mirrors cognitive_worker.py:843-862 (the boot block that this worker
    replaces). Returns ExpressionManager or None on failure (worker
    exits non-zero so guardian respawns).
    """
    try:
        from titan_hcl.logic.expression_composites import (
            ExpressionManager,
            create_speak, create_art, create_music,
            create_social, create_kin_sense, create_longing,
        )
        em = ExpressionManager()
        em.register(create_speak())
        em.register(create_art())
        em.register(create_music())
        em.register(create_social())
        em.register(create_kin_sense())
        em.register(create_longing())
        logger.info(
            "[ExpressionWorker] ExpressionManager booted with 6 composites: "
            "SPEAK, ART, MUSIC, SOCIAL, KIN_SENSE, LONGING")
        return em
    except Exception as e:
        logger.error(
            "[ExpressionWorker] ExpressionManager init FAILED: %s",
            e, exc_info=True)
        return None


# ── META-CGN EdgeDetector (relocated from coordinator._composite...) ──


class _CompositeMetaCgnEdgeHolder:
    """Wraps EdgeDetector + lazy-init + persistence restoration.

    Replaces the coordinator._composite_meta_cgn_edge + _init flag pair
    that lived on InnerTrinityCoordinator in cognitive_worker's address
    space. Now process-local to expression_worker — owns its own state
    via edge_detector_persistence.
    """

    def __init__(self) -> None:
        self._detector = None
        self._initialized = False

    def observe(self, consumer: str, intensity: float, threshold: float
                ) -> bool:
        """Returns True iff this is a fresh crossing for `consumer`."""
        if not self._initialized:
            self._lazy_init()
        if self._detector is None:
            return False
        try:
            return bool(self._detector.observe(consumer, intensity, threshold))
        except Exception:
            return False

    def state_dict(self) -> dict:
        """Snapshot for persistence (caller saves via
        edge_detector_persistence.save_edge_detector_state)."""
        if self._detector is None:
            return {}
        try:
            getter = getattr(self._detector, "to_dict", None)
            if callable(getter):
                return getter()
        except Exception:
            pass
        return {}

    def _lazy_init(self) -> None:
        try:
            from titan_hcl.logic.meta_cgn import EdgeDetector
            from titan_hcl.logic.edge_detector_persistence import (
                load_edge_detector_state,
            )
            self._detector = EdgeDetector()
            persisted = (load_edge_detector_state() or {}).get(
                "composite_meta_cgn")
            if persisted:
                try:
                    self._detector.load_dict(persisted)
                    logger.info(
                        "[ExpressionWorker] Composite EdgeDetector state "
                        "restored (%d consumer(s) previously crossed)",
                        sum(1 for v in persisted.get("crossed", {}).values()
                            if v))
                except Exception as _ler:
                    logger.warning(
                        "[ExpressionWorker] EdgeDetector load_dict failed: "
                        "%s — starting clean", _ler)
            self._initialized = True
        except Exception as _ier:
            logger.warning(
                "[ExpressionWorker] EdgeDetector lazy-init raised: %s — "
                "META-CGN edge emits suppressed this run", _ier)
            self._initialized = True


# ── Tier-2 evaluate_all + Tier-1 SPEAK detection + per-fire processing ──


def _drive_evaluate_all(
    expression_manager,
    hormone_levels: dict[str, float],
    edge_holder: "_CompositeMetaCgnEdgeHolder",
    send_queue,
    name: str,
    *,
    vocabulary_confidence: float = 1.0,
    developmental_age: int = 0,
    dream_state_gates_fire: bool = False,
) -> dict:
    """Drive ExpressionManager.evaluate_all + Tier-1 SPEAK detection +
    per-fire bus emissions (EXPRESSION_FIRED + NS_REWARD + META_CGN_SIGNAL
    + SOCIAL_CATALYST(strong_composition)).

    Returns a dict {tier2_fired: list, speak_pending: bool} for the
    caller's bookkeeping (kept symmetric with the legacy cognitive_worker
    Block 8 return shape so cognitive_worker can mirror the locals it
    used to stash into state_refs).
    """
    tier2_fired: list = []
    speak_pending = False

    if expression_manager is None:
        return {"tier2_fired": tier2_fired, "speak_pending": speak_pending}

    if not hormone_levels:
        # No hormone state yet — nothing to evaluate. evaluate_all would
        # short-circuit on zero hormones anyway; skip cleanly.
        return {"tier2_fired": tier2_fired, "speak_pending": speak_pending}

    if dream_state_gates_fire:
        # Dream state suppresses Tier-2 fires (matches the legacy gate
        # in spirit_worker — composites accumulate urge but don't fire
        # while consciousness is in non-awake state). Still evaluate
        # Tier-1 SPEAK without firing so we don't lose detection.
        logger.debug(
            "[ExpressionWorker] dream_state active — Tier-2 fires gated")
        return {"tier2_fired": tier2_fired, "speak_pending": speak_pending}

    try:
        tier2_fired = expression_manager.evaluate_all(
            hormone_levels,
            vocabulary_confidence=vocabulary_confidence,
            developmental_age=developmental_age,
            hormonal_system=None,  # cross-process; no in-proc consumption
            exclude={"SPEAK"},  # SPEAK fires via Tier-1 path (below)
        )

        # Tier-1 SPEAK trigger detection — evaluate without firing.
        speak_comp = expression_manager.composites.get("SPEAK")
        if speak_comp:
            try:
                speak_eval = speak_comp.evaluate(
                    hormone_levels, vocabulary_confidence, developmental_age)
                if speak_eval.get("should_fire"):
                    speak_pending = True
            except Exception as _sper:
                logger.debug(
                    "[ExpressionWorker] SPEAK Tier-1 eval raised: %s", _sper)

        # Per-fire bus emissions (one bus event per fire).
        for tf in tier2_fired:
            composite_name = tf.get("composite", "")
            urge = float(tf.get("urge", 0.0))
            intensity = float(tf.get("intensity", 0.0))

            # 1) EXPRESSION_FIRED (primary signal — consumed by
            #    social_worker.meter / outer_interface_worker / language).
            ef_payload = {
                "composite": composite_name,
                "urge": round(urge, 3),
                "intensity": round(intensity, 3),
                "helper": tf.get("action_helper", ""),
                "ts": time.time(),
            }
            _send_msg(send_queue, bus.EXPRESSION_FIRED, name, "all",
                      ef_payload)
            # TimeChain explicit dst — dst=all may not reach
            # subprocesses subscribed via separate topic.
            _send_msg(send_queue, bus.EXPRESSION_FIRED, name, "timechain",
                      ef_payload)

            # 1b) HORMONE_CONSUME — deplete the driving hormones in their
            #     owner (hormonal_worker). Restores the consumption→refractory
            #     loop the Phase C split severed: evaluate_all runs here with
            #     hormonal_system=None, so the in-proc depletion never fires
            #     and composites would otherwise re-fire every tick
            #     (EXPRESSION.SOCIAL runaway, 2026-06-01). hormonal_worker
            #     applies HormonalPressure.consume() per hormone; the lowered
            #     levels feed back via HormonalShmReader next tick → urge drops
            #     below threshold → natural pause until hormones rebuild.
            _consumption = tf.get("consumption", {}) or {}
            if _consumption:
                _send_msg(send_queue, bus.HORMONE_CONSUME, name,
                          "hormonal_module", {
                              "consumption": _consumption,
                              "composite": composite_name,
                              "src": name,
                              "ts": time.time(),
                          })

            # 2) NS_REWARD — cognitive_worker subscribes and calls
            #    record_outcome on its NeuralNervousSystem instance.
            ce_program = None
            if composite_name in ("ART", "MUSIC"):
                ce_program = "CREATIVITY"
            elif composite_name in ("SOCIAL", "KIN_SENSE", "LONGING"):
                ce_program = "EMPATHY"
            if ce_program and urge > 0:
                ns_reward_payload = {
                    "reward": min(1.0, urge),
                    "program": ce_program,
                    "source": f"composite.{composite_name.lower()}",
                    "ts": time.time(),
                }
                _send_msg(send_queue, bus.NS_REWARD, name, "all",
                          ns_reward_payload)

                # 3) META_CGN_SIGNAL — emit_meta_cgn_signal helper handles
                #    the bus emission; EdgeDetector gates on intensity
                #    crossing.
                try:
                    from titan_hcl.bus import emit_meta_cgn_signal
                    ce_consumer = ce_program.lower()
                    # urge can be > 1; normalize to ≤ 1 for intensity.
                    ce_intensity = min(1.0, urge / 3.0)
                    if edge_holder.observe(
                            ce_consumer, ce_intensity,
                            _META_CGN_INTENSITY_GATE):
                        emit_meta_cgn_signal(
                            send_queue,
                            src=ce_consumer,
                            consumer=ce_consumer,
                            event_type="fired",
                            intensity=ce_intensity,
                            domain=composite_name.lower(),
                            reason=(
                                f"{ce_program} via {composite_name} composite "
                                f"urge={urge:.2f}"))
                except Exception as _mcerr:
                    logger.debug(
                        "[ExpressionWorker] META_CGN emit raised: %s",
                        _mcerr)

            # 4) SOCIAL_CATALYST(type=strong_composition) — D8-3
            #    catalyst-site #8 closure. Fires when composite level ≥ 7
            #    AND a confidence proxy ≥ 0.8. Confidence proxy =
            #    min(1.0, intensity) since composites don't carry an
            #    independent confidence channel today. Matches the
            #    semantics of spirit_worker.py:12273 site.
            try:
                composite_level = int(tf.get("level", 0) or 0)
                composite_conf = min(1.0, max(0.0, intensity))
                if (composite_level >= _STRONG_COMPOSITION_LEVEL_GATE
                        and composite_conf >= _STRONG_COMPOSITION_CONFIDENCE_GATE):
                    catalyst_payload = {
                        "type": "strong_composition",
                        "significance": composite_conf,
                        "content": (
                            f"Strong {composite_name} composition "
                            f"(level={composite_level}, "
                            f"intensity={intensity:.2f})"),
                        "data": {
                            "composite": composite_name,
                            "level": composite_level,
                            "intensity": intensity,
                            "urge": urge,
                        },
                        "ts": time.time(),
                    }
                    _send_msg(
                        send_queue, bus.SOCIAL_CATALYST, name, "all",
                        catalyst_payload)
                    logger.info(
                        "[ExpressionWorker] SOCIAL_CATALYST emitted: "
                        "strong_composition via %s (L%d, conf=%.2f) — "
                        "D8-3 catalyst-site #8 closure",
                        composite_name, composite_level, composite_conf)
            except Exception as _caterr:
                logger.debug(
                    "[ExpressionWorker] strong_composition catalyst raised: "
                    "%s", _caterr)

            # 5) Info log for non-SPEAK fires (matches legacy semantics).
            if composite_name != "SPEAK":
                logger.info(
                    "[EXPRESSION.%s] FIRED — urge=%.3f, helper=%s, "
                    "intensity=%.3f",
                    composite_name, urge, tf.get("action_helper", ""),
                    intensity)

    except Exception as e:
        logger.warning(
            "[ExpressionWorker] evaluate_all raised: %s", e, exc_info=True)

    return {"tier2_fired": tier2_fired, "speak_pending": speak_pending}


# ── Main entry ────────────────────────────────────────────────────────


# Phase 11 §11.I.3 / §11.I.5 (Chunk 11N) — module-level readiness sentinel.
# Flipped True after ExpressionManager + HormonalShmReader + state publisher
# init complete. Gates SHM-slot heartbeat so titan_hcl's 1Hz poll sees real
# liveness rather than the boot-time "subscribed-but-not-warm" lie.
_WORKER_READY: bool = False


@with_error_envelope(module_name="expression_worker", subsystem="entry", severity=_phase11_sev.FATAL)
def expression_worker_main(recv_queue, send_queue, name: str,
                           config: dict) -> None:
    """Main loop for the expression_worker subprocess.

    Hosts ExpressionManager + 6 composites + composite-ledger.
    Subscribes to KERNEL_EPOCH_TICK; on each tick reads hormonal_state.bin
    via HormonalShmReader and runs evaluate_all (Tier-2) + SPEAK Tier-1
    detection. Publishes EXPRESSION_FIRED / NS_REWARD / META_CGN_SIGNAL /
    SOCIAL_CATALYST per fire + SPEAK_REQUEST_PENDING on Tier-1 detection
    + EXPRESSION_COMPOSITES_UPDATED at 5s coalesced + composite ledger to
    expression_composites_state.bin at 1 Hz.
    """
    # Phase 11 §11.I.5 (Chunk 11N) — readiness flag reset per entry.
    global _WORKER_READY
    _WORKER_READY = False

    # === BOILERPLATE: spawn-mode sys.path bootstrap ===
    project_root = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # ── Phase 11 §11.I.5 / Chunk 11N — SHM state-slot writer (G21 per worker) ──
    # Constructed BEFORE slow ExpressionManager + HormonalShmReader init so
    # the slot publishes state="starting" immediately. Periodic + heartbeat
    # paths republish state_writer.heartbeat() so guardian_hcl staleness
    # detector survives the boot window.
    _state_writer = None
    try:
        from titan_hcl.core.module_state import (
            BootPriority,
            ModuleStateWriter,
        )
        _state_writer = ModuleStateWriter(
            module_name="expression_worker",
            layer="L2",
            boot_priority=BootPriority.OPTIONAL_POST_BOOT,
        )
        _state_writer.write_state("starting")
    except Exception as _sw_err:  # noqa: BLE001
        logger.warning(
            "[ExpressionWorker] Phase 11 ModuleStateWriter init failed "
            "(continuing on legacy path): %s", _sw_err)

    # Build subscribe-topics list, conditionally adding optionals if the
    # bus.py constants exist (forward-compat per template).
    subscribe_topics = list(_EXPRESSION_WORKER_SUBSCRIBE_TOPICS)
    for _opt in _OPTIONAL_TOPICS:
        if hasattr(bus, _opt):
            subscribe_topics.append(getattr(bus, _opt))

    # === BOILERPLATE: Phase B.2 §C7 socket-mode bus client setup ===
    from titan_hcl.core.worker_bus_bootstrap import setup_worker_bus
    try:
        recv_queue, send_queue, _bus_client = setup_worker_bus(
            name, recv_queue, send_queue, topics=subscribe_topics,
        )
    except Exception as _err:
        logger.error(
            "[ExpressionWorker] setup_worker_bus failed: %s — exiting",
            _err, exc_info=True)
        return

    # === BOILERPLATE: pdeathsig installation ===
    try:
        from titan_hcl.core.worker_lifecycle import install_parent_death_signal
        install_parent_death_signal()
    except Exception as _err:
        logger.debug(
            "[ExpressionWorker] pdeathsig install skipped: %s", _err)

    from titan_hcl.core.state_registry import resolve_titan_id
    titan_id = (
        (config.get("info_banner", {}) or {}).get("titan_id")
        or resolve_titan_id()
    )
    boot_ts = time.time()

    logger.info(
        "[ExpressionWorker] Booting (titan_id=%s) — rFP §4.B + D-SPEC-NN",
        titan_id)

    # === MODULE-SPECIFIC: ExpressionManager init ===
    expression_manager = _init_expression_manager()
    if expression_manager is None:
        logger.error(
            "[ExpressionWorker] ExpressionManager init failed — exiting "
            "non-zero so guardian respawns")
        sys.exit(1)

    # === MODULE-SPECIFIC: HormonalShmReader init ===
    hormonal_reader = None
    try:
        from titan_hcl.logic.hormonal_shm_reader import HormonalShmReader
        hormonal_reader = HormonalShmReader(titan_id=titan_id)
    except Exception as _hrerr:
        logger.error(
            "[ExpressionWorker] HormonalShmReader BOOT FAILED — composites "
            "will see empty hormone dict and never fire: %s",
            _hrerr, exc_info=True)

    # === MODULE-SPECIFIC: SHM publisher init (reuses existing
    # expression_state.bin slot per logic/expression_state_publisher.py;
    # cleanly transfers G21 single-writer ownership from main plugin to
    # this worker under l0_rust_enabled=true. Parent's
    # _expression_state_publish_loop is gated off in core/plugin.py to
    # preserve G21.). ExpressionTranslator state remains in main plugin
    # process for now — translator fields published as defaults; full
    # translator migration deferred to follow-up rFP.
    state_publisher = None
    try:
        from titan_hcl.logic.expression_state_publisher import (
            ExpressionStatePublisher,
        )
        state_publisher = ExpressionStatePublisher(titan_id=titan_id)
    except Exception as _spub_err:
        logger.error(
            "[ExpressionWorker] ExpressionStatePublisher BOOT FAILED — "
            "/v4/expression-composites will see absent slot + use cold "
            "defaults: %s",
            _spub_err, exc_info=True)

    # === MODULE-SPECIFIC: META-CGN EdgeDetector holder ===
    edge_holder = _CompositeMetaCgnEdgeHolder()

    # Pi-monitor developmental_age + dream-state cache. These are bus-
    # event-driven (DREAM_STATE_CHANGED) or implicit defaults (developmental
    # age 0 if pi_monitor not yet reachable via bus events). Cached locally.
    _dream_active = False
    _developmental_age = 0  # default; updated when pi_monitor publishes

    # L3 housekeeping closure 2026-05-26: cross-process ExpressionTranslator
    # stats cache. Main plugin emits EXPRESSION_TRANSLATOR_STATS_UPDATED
    # every ~5s under l0_rust_enabled=true; we cache the most recent
    # snapshot here and feed it into the 1Hz SHM publish so the
    # translator-derived fields on the slot (sovereignty_ratio,
    # learned_actions, llm_actions, total_actions, top_mappings,
    # total_learned_pairs, posture_authenticity_ratio_30) carry real
    # values instead of the default stubs. None means no snapshot has
    # arrived yet (cold start) — publisher falls back to defaults.
    _translator_stats_cache: dict | None = None

    # Phase 11 §11.I.2 — slot transition: starting → booted (D-SPEC-141 / v1.65.0).
    # MODULE_READY bus emit DELETED per locked D2 (no shim, no dual-publish).
    # EXPRESSION_WORKER_READY remains as a peer-broadcast (not a guardian
    # readiness probe — informational only).
    _WORKER_READY = True
    if _state_writer is not None:
        try:
            _state_writer.write_state("booted")
            logger.info(
                "[ExpressionWorker] Phase 11 §11.I.2 — SHM slot state=booted "
                "(awaiting MODULE_PROBE_REQUEST from titan_hcl)")
        except Exception as _swb_err:  # noqa: BLE001
            logger.warning(
                "[ExpressionWorker] Phase 11 write_state(booted) failed: %s",
                _swb_err)
    if hasattr(bus, "EXPRESSION_WORKER_READY"):
        _send_msg(send_queue, bus.EXPRESSION_WORKER_READY, name, "all", {
            "titan_id": titan_id, "ts": boot_ts,
        })

    # === MODULE-SPECIFIC: 1Hz SHM publisher thread + 5s STATS notify +
    # 5s evaluate_all driver thread (live-verified necessary 2026-05-15 —
    # see _EVALUATE_ALL_INTERVAL_S rationale at the top of the file).
    _periodic_stop = threading.Event()

    def _periodic_publish_loop():
        last_shm = 0.0
        last_stats_notify = 0.0
        last_evaluate = 0.0
        while not _periodic_stop.is_set():
            try:
                now = time.time()
                # 1) 5s evaluate_all driver — primary firing path.
                if now - last_evaluate > _EVALUATE_ALL_INTERVAL_S:
                    hormone_levels_p: dict[str, float] = {}
                    if hormonal_reader is not None:
                        try:
                            hormone_levels_p = hormonal_reader.get_hormone_levels()
                        except Exception as _hrerr:
                            logger.debug(
                                "[ExpressionWorker] hormonal_reader.get "
                                "raised (periodic): %s", _hrerr)
                    result_p = _drive_evaluate_all(
                        expression_manager,
                        hormone_levels_p,
                        edge_holder,
                        send_queue,
                        name,
                        vocabulary_confidence=1.0,
                        developmental_age=_developmental_age,
                        dream_state_gates_fire=_dream_active,
                    )
                    if result_p["speak_pending"]:
                        speak_comp_p = expression_manager.composites.get(
                            "SPEAK")
                        speak_urge_p = (
                            float(getattr(speak_comp_p, "_last_urge", 0.0)
                                  or 0.0)
                            if speak_comp_p else 0.0)
                        spk_topic_p = (
                            bus.SPEAK_REQUEST_PENDING
                            if hasattr(bus, "SPEAK_REQUEST_PENDING")
                            else "SPEAK_REQUEST_PENDING")
                        _send_msg(
                            send_queue, spk_topic_p, name, "cognitive_worker",
                            {
                                "urge": speak_urge_p,
                                "hormones": hormone_levels_p,
                                "developmental_age": _developmental_age,
                                "tier2_fired_composites": [
                                    f["composite"]
                                    for f in result_p["tier2_fired"]],
                                "ts": time.time(),
                            })
                    last_evaluate = now
                # 2) 1Hz SHM publish — composite-ledger snapshot.
                if state_publisher is not None and \
                        now - last_shm > _SHM_PUBLISH_INTERVAL_S:
                    try:
                        # L3 housekeeping 2026-05-26: pass the cached
                        # translator_stats snapshot (received every ~5s
                        # via EXPRESSION_TRANSLATOR_STATS_UPDATED from
                        # main plugin) so translator-derived fields on
                        # the SHM slot stop reading defaults. None →
                        # cold-start or l0_rust=false fallback (parent
                        # publishes the slot directly in that mode and
                        # this worker isn't writing).
                        state_publisher.publish(
                            None, expression_manager,
                            translator_stats=_translator_stats_cache)
                    except Exception as _shmerr:
                        logger.warning(
                            "[ExpressionWorker] state publish raised: %s",
                            _shmerr, exc_info=True)
                    last_shm = now
                # 3) 5s STATS notify — informational only.
                if now - last_stats_notify > _STATS_NOTIFY_INTERVAL_S:
                    _send_msg(
                        send_queue,
                        bus.EXPRESSION_COMPOSITES_UPDATED if hasattr(
                            bus, "EXPRESSION_COMPOSITES_UPDATED")
                            else "EXPRESSION_COMPOSITES_UPDATED",
                        name, "all", {"ts": now})
                    last_stats_notify = now
            except Exception as _per_err:
                logger.warning(
                    "[ExpressionWorker] periodic publish thread error: %s",
                    _per_err)
            _periodic_stop.wait(0.5)

    _periodic_thread = threading.Thread(
        target=_periodic_publish_loop,
        daemon=True,
        name="expression-periodic-publish",
    )
    _periodic_thread.start()

    # === Main recv loop ===
    last_heartbeat = time.time()
    last_state_checkpoint = time.time()  # first checkpoint ~5min after boot
    _ckpt_thread = [None]                 # single-slot non-blocking writer
    while True:
        now = time.time()
        if now - last_heartbeat > _HEARTBEAT_INTERVAL_S:
            _send_heartbeat(send_queue, name)
            # Phase 11 §11.I.5 — SHM-slot heartbeat sidecar.
            if _state_writer is not None and _WORKER_READY:
                try:
                    _state_writer.heartbeat()
                except Exception:  # noqa: BLE001
                    pass
            last_heartbeat = now

        # ── Periodic edge_detector disk checkpoint (survives ANY crash) ──
        # SAVE_NOW persists composite_meta_cgn only on graceful shutdown, so an
        # ungraceful death (shm_pid_dead / SIGKILL) loses all edge-detector
        # learning since boot. Snapshot state_dict() on the main thread (cheap,
        # consistent), then offload the disk load+merge+write to a single-slot
        # daemon thread so it never blocks the heartbeat under IO/swap pressure.
        if (now - last_state_checkpoint > _STATE_CHECKPOINT_INTERVAL_S
                and (_ckpt_thread[0] is None or not _ckpt_thread[0].is_alive())):
            last_state_checkpoint = now
            try:
                _edge_snapshot = edge_holder.state_dict()
            except Exception:  # noqa: BLE001
                _edge_snapshot = None
            if _edge_snapshot is not None:
                def _do_ckpt(_snap=_edge_snapshot):
                    try:
                        from titan_hcl.logic.edge_detector_persistence import (
                            save_edge_detector_state, load_edge_detector_state,
                        )
                        blob = load_edge_detector_state() or {}
                        blob["composite_meta_cgn"] = _snap
                        save_edge_detector_state(blob)
                    except Exception as _ckpt_err:  # noqa: BLE001
                        logger.warning(
                            "[ExpressionWorker] periodic checkpoint failed: %s",
                            _ckpt_err)
                _ckpt_thread[0] = threading.Thread(
                    target=_do_ckpt, daemon=True, name="expression-checkpoint")
                _ckpt_thread[0].start()

        try:
            msg = recv_queue.get(timeout=_POLL_INTERVAL_S)
        except Empty:
            continue
        except (KeyboardInterrupt, SystemExit):
            break

        msg_type = msg.get("type", "")

        # ── Phase 11 §11.I.3 — MODULE_PROBE_REQUEST handler ─────────────
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
                    "[ExpressionWorker] MODULE_PROBE_REQUEST handler failed: %s",
                    _probe_err)
            continue

        # B.2.1 supervision-transfer dispatch (boilerplate).
        try:
            from titan_hcl.core import worker_swap_handler as _swap
            if _swap.maybe_dispatch_swap_msg(msg):
                continue
        except Exception:
            pass

        if msg_type == bus.MODULE_SHUTDOWN:
            logger.info(
                "[ExpressionWorker] Shutdown: %s",
                msg.get("payload", {}).get("reason"))
            break

        if msg_type == bus.SAVE_NOW:
            # Persist EdgeDetector state. Best-effort.
            try:
                from titan_hcl.logic.edge_detector_persistence import (
                    save_edge_detector_state, load_edge_detector_state,
                )
                blob = load_edge_detector_state() or {}
                blob["composite_meta_cgn"] = edge_holder.state_dict()
                save_edge_detector_state(blob)
            except Exception as _saverr:
                logger.debug(
                    "[ExpressionWorker] SAVE_NOW EdgeDetector persist "
                    "raised: %s", _saverr)
            continue

        if msg_type == bus.KERNEL_EPOCH_TICK:
            # Drive evaluate_all on each cognitive epoch (per Maker Q2).
            payload = msg.get("payload", {}) or {}
            try:
                _developmental_age = int(
                    payload.get("developmental_age", _developmental_age)
                    or _developmental_age)
            except Exception:
                pass

            # Read hormone levels via SHM (per Maker Q1).
            hormone_levels: dict[str, float] = {}
            if hormonal_reader is not None:
                try:
                    hormone_levels = hormonal_reader.get_hormone_levels()
                except Exception as _hrerr:
                    logger.debug(
                        "[ExpressionWorker] hormonal_reader.get raised: %s",
                        _hrerr)

            result = _drive_evaluate_all(
                expression_manager,
                hormone_levels,
                edge_holder,
                send_queue,
                name,
                vocabulary_confidence=1.0,
                developmental_age=_developmental_age,
                dream_state_gates_fire=_dream_active,
            )

            # SPEAK_REQUEST_PENDING — cognitive_worker subscribes and
            # constructs the language-pipeline SPEAK_REQUEST using its
            # in-proc consciousness / msl / exp_orchestrator state.
            if result["speak_pending"]:
                speak_comp = expression_manager.composites.get("SPEAK")
                speak_urge = (
                    float(getattr(speak_comp, "_last_urge", 0.0) or 0.0)
                    if speak_comp else 0.0)
                spk_topic = (bus.SPEAK_REQUEST_PENDING
                             if hasattr(bus, "SPEAK_REQUEST_PENDING")
                             else "SPEAK_REQUEST_PENDING")
                _send_msg(send_queue, spk_topic, name, "cognitive_worker", {
                    "urge": speak_urge,
                    "hormones": hormone_levels,
                    "developmental_age": _developmental_age,
                    "tier2_fired_composites": [
                        f["composite"] for f in result["tier2_fired"]],
                    "ts": time.time(),
                })

            continue

        # Optional bus events — cache dream state, etc.
        if hasattr(bus, "DREAM_STATE_CHANGED") and \
                msg_type == bus.DREAM_STATE_CHANGED:
            try:
                _dream_active = bool(
                    (msg.get("payload", {}) or {}).get("dreaming", False))
                logger.info(
                    "[ExpressionWorker] DREAM_STATE_CHANGED: dreaming=%s",
                    _dream_active)
            except Exception:
                pass
            continue

        # L3 housekeeping 2026-05-26: cache translator stats snapshot
        # from main plugin so the next 1Hz SHM publish carries real
        # values for translator-derived fields. ~5s emit cadence from
        # parent under l0_rust_enabled=true.
        if hasattr(bus, "EXPRESSION_TRANSLATOR_STATS_UPDATED") and \
                msg_type == bus.EXPRESSION_TRANSLATOR_STATS_UPDATED:
            try:
                pl = msg.get("payload") or {}
                if isinstance(pl, dict):
                    _translator_stats_cache = dict(pl)
            except Exception as _tserr:
                logger.debug(
                    "[ExpressionWorker] EXPRESSION_TRANSLATOR_STATS_UPDATED "
                    "cache update raised: %s", _tserr)
            continue

        logger.debug(
            "[ExpressionWorker] Unhandled msg_type=%s — ignoring", msg_type)

    # === Clean shutdown ===
    logger.info(
        "[ExpressionWorker] Exiting — stopping publisher thread + EdgeDetector "
        "persist")
    _periodic_stop.set()
    try:
        from titan_hcl.logic.edge_detector_persistence import (
            save_edge_detector_state, load_edge_detector_state,
        )
        blob = load_edge_detector_state() or {}
        blob["composite_meta_cgn"] = edge_holder.state_dict()
        save_edge_detector_state(blob)
    except Exception as _saverr:
        logger.debug(
            "[ExpressionWorker] shutdown EdgeDetector persist raised: %s",
            _saverr)
    logger.info("[ExpressionWorker] Exit complete")
