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
Note: the docstring inside ``titan_plugin/core/worker_bus_bootstrap.py``
shows ``(name, recv_q, send_q, config)`` — that is STALE and contradicts
all production workers (outer_body_worker, mind_worker, body_worker,
spirit_worker, etc.). Use the production order shown here, not the
bootstrap docstring. Future extractions: do not copy the wrong order.

GENERIC HELPERS BELOW — ``_send_msg``, ``_send_heartbeat``,
``_load_toml_section`` exist in 5+ workers in slightly different shapes.
When the 3rd L2 extraction lands (per L2 separation strategy rFP §5),
extract these to a shared ``titan_plugin/modules/_worker_skeleton.py``
helper module. YAGNI today; plan ahead.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time
from queue import Empty

from titan_plugin import bus
from titan_plugin._phase_c_constants import (
    COGNITIVE_EPOCH_DEFAULT_INTERVAL_S,
    COGNITIVE_EPOCH_MAX_INTERVAL_S,
    COGNITIVE_EPOCH_MIN_INTERVAL_S,
    COGNITIVE_PERSIST_EVERY_N_EPOCHS,
)

logger = logging.getLogger(__name__)

# Heartbeat cadence per SPEC §9.C (10s — guardian_HCL liveness contract).
_HEARTBEAT_INTERVAL_S = 10.0
# Main loop poll cadence (kept tight so MODULE_SHUTDOWN is responsive).
_POLL_INTERVAL_S = 0.2

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
    bus.CONVERSATION_STIMULUS,     # chat → reasoning_engine.observe_stimulus
    bus.EXPERIENCE_STIMULUS,       # experience replay → reasoning_engine
    bus.MEDITATION_COMPLETE,       # meditation phase tracking via coordinator
    bus.MODULE_SHUTDOWN,           # clean shutdown
    bus.SAVE_NOW,                  # B.1 shadow_swap orchestrator (when re-enabled)
]


# ARG ORDER (template-critical — see module docstring): every Guardian-spawned
# L2 worker entry follows (recv_queue, send_queue, name, config).
def cognitive_worker_main(recv_queue, send_queue, name: str, config: dict) -> None:
    """Main loop for the cognitive_worker subprocess.

    Chunk 8E skeleton — heartbeat-only main loop. Bus dispatcher (chunk 8F),
    consciousness epoch driver (chunk 8G), and snapshot publishers
    (chunk 8H) land in subsequent commits.
    """
    # === BOILERPLATE: spawn-mode sys.path bootstrap ===
    # Spawn mode starts a fresh Python interpreter without inheriting the
    # parent's sys.path. Re-add the project root so `from titan_plugin.X
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
    from titan_plugin.core.worker_bus_bootstrap import setup_worker_bus
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

    # === BOILERPLATE: pdeathsig installation ===
    # Linux PR_SET_PDEATHSIG: kernel delivers SIGTERM if titan_HCL parent
    # dies, so this worker can't outlive its supervisor (matches A.8
    # graduated-spawn worker pattern). Failure is non-fatal — the
    # parent_watcher fallback in worker_lifecycle handles it.
    try:
        from titan_plugin.core.worker_lifecycle import install_parent_death_signal
        install_parent_death_signal()
    except Exception as _err:
        logger.debug("[CognitiveWorker] pdeathsig install skipped: %s", _err)

    titan_id = (config.get("info_banner", {}) or {}).get("titan_id") or "T1"
    boot_ts = time.time()

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
        _send_msg(send_queue, bus.MODULE_READY, name, "guardian", {
            "titan_id": titan_id, "ts": boot_ts, "flag_off_noop": True,
            "chunk": "8E",
        })
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
        from titan_plugin.modules.spirit_loop import _init_consciousness
        state_refs["consciousness"] = _init_consciousness(config)
    except Exception as _err:
        logger.warning("[CognitiveWorker] _init_consciousness failed: %s", _err)
        state_refs["consciousness"] = None

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

    # Signal MODULE_READY to guardian_HCL.
    _send_msg(send_queue, bus.MODULE_READY, name, "guardian", {
        "titan_id": titan_id, "ts": boot_ts, "chunk": "8G",
    })
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
        from titan_plugin.modules.spirit_loop import start_snapshot_builder_threads
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

    # === BOILERPLATE: Phase B.1 readiness reporter ===
    # Lets shadow_swap orchestrator drain this worker's state before
    # kernel swap. The `save_state_cb` is MODULE-SPECIFIC — return a
    # list of file paths that shadow_swap should persist. For chunk 8E
    # skeleton this is `[]`; chunks 8G/8I wire reasoning_totals.json,
    # dreaming_state.json, pi_heartbeat_state.json, neural_ns/* etc.
    try:
        from titan_plugin.core.readiness_reporter import trivial_reporter
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
    while True:
        now = time.time()
        if now - last_heartbeat_ts >= _HEARTBEAT_INTERVAL_S:
            _send_heartbeat(send_queue, name)
            last_heartbeat_ts = now

        try:
            msg = recv_queue.get(timeout=_POLL_INTERVAL_S)
        except Empty:
            continue
        except (KeyboardInterrupt, SystemExit):
            break
        except Exception:
            continue

        msg_type = msg.get("type")

        # Phase B.1 shadow swap dispatch.
        if _b1_reporter is not None and _b1_reporter.handles(msg_type):
            _b1_reporter.handle(msg)
            if _b1_reporter.should_exit():
                break
            continue

        # Phase B.2.1 supervision-transfer dispatch — preserves spawn-mode adoption.
        try:
            from titan_plugin.core import worker_swap_handler as _swap
            if _swap.maybe_dispatch_swap_msg(msg):
                continue
        except Exception:
            pass

        if msg_type == bus.MODULE_SHUTDOWN:
            logger.info("[CognitiveWorker] Shutdown received — stopping epoch driver + persisting state")
            _epoch_stop_event.set()
            _epoch_early_fire_event.set()  # unblock the wait in epoch loop
            _epoch_thread.join(timeout=5.0)
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

            elif msg_type in (bus.CONVERSATION_STIMULUS, bus.EXPERIENCE_STIMULUS):
                _dispatch_stimulus(state_refs, msg_type, payload)

            elif msg_type == bus.MEDITATION_COMPLETE:
                _dispatch_meditation_complete(state_refs, payload)

            elif msg_type == bus.SAVE_NOW:
                # Persist all engine state (chunk 8G persistence cadence
                # is the 100-epoch tick; SAVE_NOW is the on-demand path
                # for shadow_swap B.1 readiness).
                _persist_engine_state(state_refs)

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
    from titan_plugin.modules._cognitive_init import (
        _init_t2_state_registries, _init_observable_engine,
        _init_neural_nervous_system, _init_coordinator,
    )

    inner_state = spirit_state = observable_engine = None
    neural_nervous_system = coordinator = None
    pi_monitor = reasoning_engine = meta_engine = None

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

    try:
        from titan_plugin.logic.pi_heartbeat import PiHeartbeatMonitor
        pi_monitor = PiHeartbeatMonitor(
            min_cluster_size=3, min_gap_size=2,
            state_path="./data/pi_heartbeat_state.json")
    except Exception as _err:
        logger.warning("[CognitiveWorker] PiHeartbeatMonitor init failed: %s", _err)

    try:
        from titan_plugin.logic.reasoning import ReasoningEngine
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

    try:
        from titan_plugin.logic.meta_reasoning import MetaReasoningEngine
        _meta_cfg = _load_toml_section("meta_reasoning")
        if _meta_cfg.get("enabled", True):
            meta_engine = MetaReasoningEngine(
                config=_meta_cfg, send_queue=send_queue)
            if coordinator is not None:
                coordinator._meta_engine = meta_engine
    except Exception as _err:
        logger.warning("[CognitiveWorker] MetaReasoningEngine init failed: %s", _err)

    # ExpressionManager — composite-ledger that drives /v4/expression-composites
    # snapshot publisher (EXPRESSION_COMPOSITES_UPDATED → cache.expression.composites).
    # Mirrors legacy spirit_worker.py:1142-1148 init pattern.
    expression_manager = None
    try:
        from titan_plugin.logic.expression_composites import (
            ExpressionManager, create_speak, create_art, create_music,
            create_social, create_kin_sense, create_longing,
        )
        expression_manager = ExpressionManager()
        expression_manager.register(create_speak())
        expression_manager.register(create_art())
        expression_manager.register(create_music())
        expression_manager.register(create_social())
        expression_manager.register(create_kin_sense())
        expression_manager.register(create_longing())
        if coordinator is not None:
            coordinator._expression_manager = expression_manager
    except Exception as _err:
        logger.warning("[CognitiveWorker] ExpressionManager init failed: %s", _err)

    # === chunk 8M.6 — LifeForceEngine (Chi) init ===
    # Per rFP_phase_c_observatory_data_pipeline.md §3.6 + Gap F (§2.6): under
    # l0_rust_enabled=true, chi has no live Python publisher (legacy
    # spirit_worker_main is a heartbeat-only stub per chunk 8I). Instantiate
    # the engine here so cognitive_worker can publish chi.state via the
    # snapshot builder. The Rust kernel-rs chi_state.bin shm slot remains
    # the canonical source — chi_snapshot from shm wins via build_coordinator_snapshot
    # (chunk 8M.5), this engine provides a Python-side chi reading when shm
    # is unavailable AND drives metabolic accumulation logic that's not yet
    # in Rust.
    life_force_engine = None
    try:
        from titan_plugin.logic.life_force import LifeForceEngine
        life_force_engine = LifeForceEngine()
        # Wire drain passive decay from titan_params.toml [dreaming] section
        # (mirrors spirit_worker.py:1817-1829 init pattern).
        try:
            import tomllib as _tom_lfe
            _lfe_path = os.path.join(
                os.path.dirname(__file__), "..", "titan_params.toml")
            if os.path.exists(_lfe_path):
                with open(_lfe_path, "rb") as _lfe_f:
                    _lfe_cfg = _tom_lfe.load(_lfe_f).get("dreaming", {})
                if "drain_passive_decay" in _lfe_cfg:
                    life_force_engine._drain_passive_decay = float(
                        _lfe_cfg["drain_passive_decay"])
        except Exception as _lfe_decay_err:
            logger.debug(
                "[CognitiveWorker] LifeForce drain_passive_decay load failed: %s",
                _lfe_decay_err)
        # Restore persisted metabolic_drain from coordinator.dreaming if available.
        if coordinator is not None and getattr(coordinator, "dreaming", None):
            _persisted = getattr(coordinator.dreaming, "_persisted_drain", 0.0)
            if _persisted > 0.001:
                life_force_engine._metabolic_drain = _persisted
                logger.info(
                    "[CognitiveWorker] Restored metabolic_drain=%.4f from "
                    "dreaming state", _persisted)
        if coordinator is not None:
            coordinator._life_force_engine = life_force_engine
        logger.info(
            "[CognitiveWorker] LifeForceEngine (Chi) booted — 3×3 Trinity "
            "matrix (drain_decay=%.4f, drain=%.4f)",
            life_force_engine._drain_passive_decay,
            life_force_engine._metabolic_drain)
    except Exception as _err:
        logger.warning(
            "[CognitiveWorker] LifeForceEngine init failed: %s — chi.state "
            "will fall back to shm-only reads", _err)

    # === chunk 8M.7 — Multisensory Synthesis Layer (MSL) init ===
    # Per rFP §3.7 + Gap F (§2.6): MSL hosts the L2 perception engine
    # (concept grounding, attention weighting, i-confidence). Under
    # l0_rust=true the legacy spirit_worker stub doesn't init it.
    # Mirrors spirit_worker.py:1949-1979 init pattern + same titan_params.toml
    # [msl] section gate.
    msl = None
    try:
        from titan_plugin.logic.msl import MultisensorySynthesisLayer
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

    return {
        "coordinator": coordinator,
        "neural_nervous_system": neural_nervous_system,
        "pi_monitor": pi_monitor,
        "reasoning_engine": reasoning_engine,
        "meta_engine": meta_engine,
        "observable_engine": observable_engine,
        "expression_manager": expression_manager,
        "inner_state": inner_state,
        "spirit_state": spirit_state,
        # chunk 8M.6 + 8M.7: chi + MSL engines accessible via state_refs so
        # spirit_loop.build_coordinator_snapshot picks them up under
        # l0_rust_enabled=true.
        "life_force_engine": life_force_engine,
        "msl": msl,
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

    Reuses ``titan_plugin.api.shm_reader_bank.ShmReaderBank`` — same bank
    api_subprocess uses (defense in depth: cognitive_worker write path +
    api_subprocess fallback path both go through one set of typed
    accessors that own per-registry SeqLock-validated reads).

    Returns ``None`` if the bank can't be constructed (titan-rust kernel
    not running, shm root missing, etc.) — the epoch driver tolerates a
    None bank by falling back to bus-cache slots only (chunk 8G behavior).
    """
    try:
        from titan_plugin.api.shm_reader_bank import ShmReaderBank
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
        from titan_plugin.core.state_registry import NEUROMOD_STATE, RegistryBank
        bank = RegistryBank()
        if not bank.is_enabled(NEUROMOD_STATE):
            return None
        reader = bank.reader(NEUROMOD_STATE)

        def _read():
            arr = reader.read()
            if arr is None or len(arr) < 6:
                return None
            return {
                "DA": float(arr[0]),
                "5HT": float(arr[1]),
                "NE": float(arr[2]),
                "ACh": float(arr[3]),
                "Endorphin": float(arr[4]),
                "GABA": float(arr[5]),
            }
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
    from titan_plugin.modules.spirit_loop import _run_consciousness_epoch

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
    from titan_plugin.modules.spirit_loop import _run_consciousness_epoch

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
                config, outer_state=outer_state_dict)
        except Exception as _err:
            logger.debug("[CognitiveWorker] _run_consciousness_epoch failed: %s", _err)

    latest = (consciousness or {}).get("latest_epoch") or {}
    epoch_id = latest.get("epoch_id", 0)
    curvature = latest.get("curvature", 0.0)

    # 5. Drive PiHeartbeatMonitor.observe(curvature, epoch_id).
    if pi_monitor is not None and epoch_id > 0:
        try:
            pi_monitor.observe(curvature=curvature, epoch_id=epoch_id)
        except Exception as _err:
            logger.debug("[CognitiveWorker] pi_monitor.observe failed: %s", _err)

    # 6. Step ReasoningEngine if it has an active chain.
    if reasoning_engine is not None:
        try:
            has_active = getattr(reasoning_engine, "has_active_chain", None)
            step = getattr(reasoning_engine, "step", None)
            if callable(has_active) and callable(step) and has_active():
                step()
        except Exception as _err:
            logger.debug("[CognitiveWorker] reasoning_engine.step failed: %s", _err)

    # 7. Tick MetaReasoningEngine — feeds latest_epoch state.
    if meta_engine is not None:
        try:
            tick_fn = getattr(meta_engine, "tick", None)
            if callable(tick_fn):
                # MetaReasoningEngine.tick(state_132d, neuromods, reasoning_engine, ...)
                # signature varies — pass best-effort kwargs and tolerate
                # AttributeError if the engine wants different args.
                tick_fn(
                    state_132d=latest.get("state_vector"),
                    neuromods=neuromod_reader() if neuromod_reader else None,
                    reasoning_engine=reasoning_engine,
                )
        except Exception as _err:
            logger.debug("[CognitiveWorker] meta_engine.tick failed: %s", _err)


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
    coordinator = state_refs.get("coordinator")

    for engine, name_, method in (
        (reasoning_engine, "reasoning_engine", "save_state"),
        (pi_monitor, "pi_monitor", "_save_state"),
        (neural_nervous_system, "neural_nervous_system", "save_all"),
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
    (titan_plugin/modules/). Returns {} on any failure (file missing,
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


# === BOILERPLATE: heartbeat + messaging helpers ====================
# These three helpers (_heartbeat_loop, _send_heartbeat, _send_msg) are
# generic L2 worker boilerplate. They're inlined here for chunk 8E so
# cognitive_worker is self-contained, but should be promoted to a shared
# `titan_plugin/modules/_worker_skeleton.py` module when the 3rd L2
# extraction lands (per L2 separation strategy rFP §5 sequencing). Until
# then, copy verbatim into each new L2 worker — change only the
# `[CognitiveWorker]` log prefix + the `chunk` payload field.


def _heartbeat_loop(recv_queue, send_queue, name: str, *, flag_off: bool) -> None:
    """Heartbeat-only loop for the l0_rust_enabled=false defensive branch.

    Exits cleanly on MODULE_SHUTDOWN. No engine init, no dispatcher.
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
        if msg.get("type") == bus.MODULE_SHUTDOWN:
            logger.info("[CognitiveWorker] Shutdown received (flag_off branch)")
            return


def _send_heartbeat(send_queue, name: str, extra: dict | None = None) -> None:
    """Emit MODULE_HEARTBEAT to guardian_HCL with current RSS."""
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
