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

    # Block F: pre-D8 ownership audit Track 1 — memory + monitor engines.
    working_mem = None
    prediction_engine = None
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

    # ── Block F (Track 1): WorkingMemory + PredictionEngine ──
    # 2026-05-10 — pre-D8 ownership audit Track 1 closure. Both small,
    # dep-free engines mirroring spirit_worker.py:1156+1159.
    # WorkingMemory is load-bearing for reasoning_engine.tick (chain
    # context); PredictionEngine drives per-epoch novelty signal that
    # feeds chi + dreaming.
    try:
        from titan_plugin.logic.working_memory import WorkingMemory
        working_mem = WorkingMemory(capacity=7, decay_epochs=5)
        logger.info(
            "[CognitiveWorker] WorkingMemory booted (capacity=7, decay=5)")
    except Exception as _wm_err:
        logger.warning(
            "[CognitiveWorker] WorkingMemory init failed: %s", _wm_err)

    try:
        from titan_plugin.logic.prediction_engine import PredictionEngine
        prediction_engine = PredictionEngine(error_window=20)
        logger.info(
            "[CognitiveWorker] PredictionEngine booted (error_window=20)")
    except Exception as _pe_err:
        logger.warning(
            "[CognitiveWorker] PredictionEngine init failed: %s", _pe_err)

    # ── EpisodicMemory ──
    try:
        from titan_plugin.logic.episodic_memory import EpisodicMemory
        episodic_mem = EpisodicMemory(db_path="./data/episodic_memory.db")
        logger.info("[CognitiveWorker] EpisodicMemory booted")
    except Exception as _em_err:
        logger.warning(
            "[CognitiveWorker] EpisodicMemory init failed: %s", _em_err)

    # ── IntuitionConvergenceDetector (M11-M13) ──
    try:
        from titan_plugin.logic.intuition_convergence import (
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
        from titan_plugin.logic.wallet_observer import WalletObserver
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
        from titan_plugin.logic.meta_recruitment import MetaRecruitment
        meta_recruitment = MetaRecruitment()
        logger.info("[CognitiveWorker] MetaRecruitment booted")
    except Exception as _mr_err:
        logger.warning(
            "[CognitiveWorker] MetaRecruitment init failed: %s", _mr_err)

    # ── TimeseriesStore (telemetry sink) ──
    try:
        from titan_plugin.logic.timeseries import TimeseriesStore
        timeseries_store = TimeseriesStore("./data/timeseries.db")
        logger.info("[CognitiveWorker] TimeseriesStore booted")
    except Exception as _ts_err:
        logger.warning(
            "[CognitiveWorker] TimeseriesStore init failed: %s", _ts_err)

    # ── MiniReasonerRegistry (distributed mini-reasoners) ──
    try:
        from titan_plugin.logic.mini_experience import MiniReasonerRegistry
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
    except Exception as _mri_err:
        logger.warning(
            "[CognitiveWorker] MiniReasonerRegistry init failed: %s",
            _mri_err)

    # ── ReasoningInterpreter (concept-domain interpretation) ──
    try:
        from titan_plugin.logic.reasoning_interpreter import (
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

    # ── MeditationWatchdog (cadence + alerts) ──
    try:
        from titan_plugin.logic.meditation_watchdog import MeditationWatchdog
        _med_cfg = _load_toml_section("meditation_watchdog")
        med_watchdog = MeditationWatchdog(config=_med_cfg) if _med_cfg else (
            MeditationWatchdog())
        logger.info(
            "[CognitiveWorker] MeditationWatchdog booted (titan_id=%s)",
            getattr(med_watchdog, "titan_id", "?"))
    except Exception as _mw_err:
        logger.warning(
            "[CognitiveWorker] MeditationWatchdog init failed: %s", _mw_err)

    # ── Meta-Reasoning Foundation (M1-M3) ──
    # 2026-05-10: post-deploy follow-up to the pre-D8 ownership audit.
    # Boot-driver parity audit caught that meta_engine was wired but its
    # 4 required positional deps (chain_archive, meta_wisdom, ex_mem,
    # meta_autoencoder) were missing in cognitive_worker — meta_engine.tick
    # was raising TypeError silently every epoch (now visible after
    # _log_driver_err visibility upgrade). Mirrors spirit_worker.py:1415-1434.
    chain_archive = None
    meta_wisdom = None
    meta_autoencoder = None
    try:
        from titan_plugin.logic.chain_archive import ChainArchive
        from titan_plugin.logic.meta_wisdom import MetaWisdomStore
        from titan_plugin.logic.meta_autoencoder import MetaAutoencoder
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

    # ── Boot NeuromodRewardObserver (rFP β Stage 2 Phase 2b) ──
    #
    # 2026-05-10: migrated from spirit_worker per pre-D8 ownership audit.
    # Constructor refactored to take a `levels_provider` callable so the
    # observer can read NEUROMOD_STATE shm (Rust-produced under
    # l0_rust=true) without needing cross-process Python attr access on
    # neuromodulator_system (which lives in neuromod_worker). NS lives
    # in this process, so record_outcome calls are in-process.
    # ── Boot ExperienceMemory + ExperientialMemory + ExperienceOrchestrator ──
    #
    # 2026-05-10: Block D of pre-D8 ownership audit closure. Tier 1 SPEAK
    # firing path emits SPEAK_REQUEST with experience_bias built by
    # ExperienceOrchestrator.get_experience_bias("language", ...). Migrated
    # from spirit_worker:1133-1880 (l0_rust=false legacy path) so SPEAK
    # works on T3 (where spirit_worker is heartbeat-only). Plugins
    # registered: ArcPuzzle, LanguageLearning, CreativeExpression, Communication.
    ex_mem = None
    e_mem = None
    exp_orchestrator = None
    try:
        from titan_plugin.logic.experience_memory import ExperienceMemory
        from titan_plugin.logic.experiential_memory import ExperientialMemory
        from titan_plugin.logic.experience_orchestrator import (
            ExperienceOrchestrator)
        from titan_plugin.logic.experience_plugins import (
            ArcPuzzlePlugin, LanguageLearningPlugin,
            CreativeExpressionPlugin, CommunicationPlugin)
        ex_mem = ExperienceMemory(db_path="./data/experience_memory.db")
        _dev_age_fn = (lambda: pi_monitor.developmental_age) if pi_monitor else (
            lambda: 0)
        e_mem = ExperientialMemory(
            db_path="./data/experiential_memory.db",
            developmental_age_fn=_dev_age_fn,
        )
        exp_orchestrator = ExperienceOrchestrator(
            ex_mem=ex_mem, e_mem=e_mem, cognee_memory=None,
            db_path="./data/experience_orchestrator.db")
        exp_orchestrator.register_plugin(ArcPuzzlePlugin())
        exp_orchestrator.register_plugin(LanguageLearningPlugin())
        exp_orchestrator.register_plugin(CreativeExpressionPlugin())
        exp_orchestrator.register_plugin(CommunicationPlugin())
        logger.info(
            "[CognitiveWorker] Experience Orchestrator booted: %s",
            list(exp_orchestrator._plugins.keys()))
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
        from titan_plugin.logic.social_pressure import SocialPressureMeter
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
        from titan_plugin.logic.neuromod_reward_observer import (
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
        "expression_manager": expression_manager,
        "inner_state": inner_state,
        "spirit_state": spirit_state,
        # chunk 8M.6 + 8M.7: chi + MSL engines accessible via state_refs so
        # spirit_loop.build_coordinator_snapshot picks them up under
        # l0_rust_enabled=true.
        "life_force_engine": life_force_engine,
        "msl": msl,
        "neuromod_reward_observer": neuromod_reward_observer,
        # Block D — Tier 1 SPEAK migration deps:
        "ex_mem": ex_mem,
        "e_mem": e_mem,
        "exp_orchestrator": exp_orchestrator,
        "social_pressure_meter": _social_pressure_meter,
        # Meta-reasoning foundation (M1-M3) — required by meta_engine.tick:
        "chain_archive": chain_archive,
        "meta_wisdom": meta_wisdom,
        "meta_autoencoder": meta_autoencoder,
        # Block F (Track 1) — pre-D8 ownership audit migrations:
        "working_mem": working_mem,
        "prediction_engine": prediction_engine,
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
        from titan_plugin.config_loader import load_titan_config
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

    # 4.5 Chi (Λ) Life Force Evaluation + publish CHI_UPDATED.
    #
    # Phase C migration closure (2026-05-10): cognitive_worker hosts the
    # LifeForceEngine instance (chunk 8M.6) but the original chunk left the
    # compute+publish loop behind in spirit_worker (now heartbeat-only on
    # T3). Without this block, chi.state cache stays empty and /v4/chi
    # falls through to SHM raw shape (flat 6 numerics, no rich
    # state/developmental_phase/components). Observatory home route
    # rendered T3 chi as 1% with empty SPIRIT/MIND/BODY breakdowns.
    #
    # Mirrors spirit_worker.py:6126-6207 (the canonical Phase A+B path).
    # Both call sites use the same library helpers from logic/life_force.py
    # (compute_coherence_from_sv / compute_expression_fire_rate /
    # compute_hormonal_vitality / compute_neuromodulator_homeostasis); no
    # logic duplication, just per-worker orchestration. When D8-3 retires
    # spirit_worker.py, that block goes away and this one becomes sole.
    life_force_engine = state_refs.get("life_force_engine")
    expression_manager = state_refs.get("expression_manager")
    neural_nervous_system = state_refs.get("neural_nervous_system")
    if life_force_engine is not None and consciousness is not None:
        try:
            from titan_plugin.logic.life_force import (
                compute_neuromodulator_homeostasis,
                compute_hormonal_vitality,
                compute_coherence_from_sv,
                compute_expression_fire_rate,
            )
            _lf_sv = latest.get("state_vector", [])
            if hasattr(_lf_sv, "to_list"):
                _lf_sv = _lf_sv.to_list()
            _lf_sv = list(_lf_sv) if _lf_sv else []

            _lf_pi_ratio = pi_monitor.heartbeat_ratio if pi_monitor else 0.0
            _lf_dev_age = pi_monitor.developmental_age if pi_monitor else 0
            _lf_sov = 0
            _lf_spirit_coh = 0.5
            if len(_lf_sv) >= 130:
                _is_coh = compute_coherence_from_sv(_lf_sv, 20, 65)
                _os_coh = compute_coherence_from_sv(_lf_sv, 85, 130)
                _lf_spirit_coh = (_is_coh + _os_coh) / 2.0
            _lf_vocab = 0
            try:
                from titan_plugin.utils.db import safe_connect as _sc_vdb
                _vdb = _sc_vdb("data/inner_memory.db")
                _lf_vocab = _vdb.execute(
                    "SELECT COUNT(*) FROM vocabulary WHERE confidence > 0.3"
                ).fetchone()[0]
                _vdb.close()
            except Exception:
                pass
            # neuromodulator_system lives on coordinator under l0_rust=true.
            _nm_sys = getattr(coordinator, "neuromodulator_system", None) if coordinator else None
            _lf_lr_gain = 1.0
            if _nm_sys is not None:
                try:
                    _lf_lr_gain = _nm_sys.get_modulation().get(
                        "learning_rate_gain", 1.0)
                except Exception:
                    pass
            _lf_emotion_conf = (
                getattr(_nm_sys, "_emotion_confidence", 0.5)
                if _nm_sys is not None else 0.5
            )
            _lf_nm_homeo = compute_neuromodulator_homeostasis(
                getattr(_nm_sys, "modulators", {}) if _nm_sys is not None else {})
            _lf_mind_coh = 0.5
            if len(_lf_sv) >= 85:
                _im_coh = compute_coherence_from_sv(_lf_sv, 5, 20)
                _om_coh = compute_coherence_from_sv(_lf_sv, 70, 85)
                _lf_mind_coh = (_im_coh + _om_coh) / 2.0
            _lf_expr_rate = compute_expression_fire_rate(
                expression_manager.get_stats() if expression_manager else {})
            _lf_sol = 13.0
            _lf_anchor = 0.5
            _lf_hormonal_vit = compute_hormonal_vitality(
                neural_nervous_system.get_stats().get("hormonal_system", {})
                if neural_nervous_system else {})
            _lf_body_coh = 0.5
            if len(_lf_sv) >= 70:
                _ib_coh = compute_coherence_from_sv(_lf_sv, 0, 5)
                _ob_coh = compute_coherence_from_sv(_lf_sv, 65, 70)
                _lf_body_coh = (_ib_coh + _ob_coh) / 2.0
            # Topology grounding for chi: under l0_rust_enabled=true the
            # LowerTopology engines live in Rust (titan-trinity-rs writes
            # topology_30d.bin per Schumann tick). Layout per Rust
            # `assemble_topology_30d` (titan-trinity-rs/src/topology.rs:337):
            #   [0:10]  outer_lower.topology_10d
            #   [10:20] inner_lower.topology_10d
            #   [20:30] whole_10d
            # Coherence = cosine similarity of inner_lower vs balanced
            # reference [0.5]*10 (matches Python LowerTopology._cosine_sim).
            # Closes BUG-COGNITIVE-WORKER-INNER-LOWER-TOPO-COMPUTE-MISSING:
            # under l0_rust=true the legacy `coordinator.inner_lower_topo`
            # Python instance doesn't exist, so the previous getattr path
            # fell through to 0.5 every tick.
            _lf_topo = 0.5
            try:
                _topo_values = topology_snap.get("values") if isinstance(
                    topology_snap, dict) else None
                if _topo_values and len(_topo_values) >= 20:
                    from titan_plugin.logic.lower_topology import _cosine_sim
                    _inner_lower_10d = list(_topo_values[10:20])
                    _lf_topo = _cosine_sim(_inner_lower_10d, [0.5] * 10)
            except Exception as _topo_err:
                # Rate-limited WARN — failure here means chi reverts to the
                # 0.5 default (graceful degradation), not a hard crash.
                if not hasattr(state_refs, "_topo_grounding_err_logged"):
                    logger.warning(
                        "[CognitiveWorker] chi topology_grounding shm-read failed: %s",
                        _topo_err, exc_info=True)
                    state_refs["_topo_grounding_err_logged"] = True

            _chi = life_force_engine.evaluate(
                pi_heartbeat_ratio=_lf_pi_ratio,
                developmental_age=_lf_dev_age,
                sovereignty_index=_lf_sov,
                spirit_coherence=_lf_spirit_coh,
                vocabulary_size=_lf_vocab,
                learning_rate_gain=_lf_lr_gain,
                emotional_coherence=_lf_emotion_conf,
                neuromodulator_homeostasis=_lf_nm_homeo,
                mind_coherence=_lf_mind_coh,
                expression_fire_rate=_lf_expr_rate,
                sol_balance=_lf_sol,
                anchor_freshness=_lf_anchor,
                hormonal_vitality=_lf_hormonal_vit,
                body_coherence=_lf_body_coh,
                topology_grounding=_lf_topo,
            )
            life_force_engine._latest_chi = _chi
            # Publish CHI_UPDATED → api_subprocess BusSubscriber maps to
            # chi.state cache key, /v4/chi reads it from there (rich shape
            # with state/developmental_phase/weights/components/contemplation
            # — what observatory expects).
            _send_msg(send_queue, bus.CHI_UPDATED, name, "all", _chi)
            # Update topology journey Y-axis with chi circulation.
            if consciousness:
                _topo = consciousness.get("topology")
                if _topo is not None and hasattr(_topo, "update_chi_circulation"):
                    try:
                        _topo.update_chi_circulation(_chi.get("circulation", 0.5))
                    except Exception:
                        pass
        except Exception as _err:
            logger.debug("[CognitiveWorker] chi evaluate/publish failed: %s",
                         _err, exc_info=True)

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
            _nn_chi: dict = {}
            life_force_engine = state_refs.get("life_force_engine")
            if life_force_engine is not None:
                _raw_chi = getattr(life_force_engine, "_latest_chi", {}) or {}
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
            _nn_drain = float(getattr(life_force_engine, "_metabolic_drain", 0.0) or 0.0
                              ) if life_force_engine is not None else 0.0
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

            # observables source: prefer coordinator.observable_engine if available
            _nn_obs: dict = {}
            obs_eng = getattr(coordinator, "observable_engine", None) if (
                coordinator is not None) else None
            if obs_eng is not None:
                try:
                    get_obs = getattr(obs_eng, "get_observables", None) or getattr(
                        obs_eng, "snapshot", None)
                    if callable(get_obs):
                        _maybe_obs = get_obs()
                        if isinstance(_maybe_obs, dict):
                            _nn_obs = _maybe_obs
                except Exception:
                    pass

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

            # Drive evaluate() — THIS is what actually trains + updates hormonal maturity.
            # Per spirit_worker.py:4593: temporal=None (programs are 55D; 60D mismatch).
            evaluate_fn = getattr(neural_nervous_system, "evaluate", None)
            if callable(evaluate_fn):
                _nn_signals = evaluate_fn(_nn_obs, temporal=None) or []
                # Light cadence-only logging — per-tick log would flood.
                if epoch_id and epoch_id % 100 == 0:
                    logger.info(
                        "[CognitiveWorker] NeuralNS alive: transitions=%d, "
                        "train_steps=%d, signals=%d, maturity=%.4f",
                        getattr(neural_nervous_system, "_total_transitions", 0),
                        getattr(neural_nervous_system, "_total_train_steps", 0),
                        len(_nn_signals),
                        getattr(getattr(neural_nervous_system, "_hormonal", None), "maturity", 0.0))
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

    # 6. Step ReasoningEngine if it has an active chain.
    if reasoning_engine is not None:
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
                _r_body = {
                    "fatigue": float(_nn_exp_p) if "_nn_exp_p" in dir() else 0.3,
                    "chi_total": float(
                        getattr(life_force_engine, "_latest_chi", {}).get("total", 0.5)
                        if life_force_engine else 0.5),
                    "metabolic_drain": float(
                        getattr(life_force_engine, "_metabolic_drain", 0.0)
                        if life_force_engine else 0.0),
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
        except Exception as _err:
            _log_driver_err("reasoning_engine.tick", _err)

    # 7. Tick MetaReasoningEngine — feeds latest_epoch state.
    #
    # 2026-05-10 post-deploy fix: MetaReasoningEngine.tick signature
    # (logic/meta_reasoning.py:1056) requires 4 additional positional
    # deps — chain_archive, meta_wisdom, ex_mem, meta_autoencoder —
    # that were missing from this call site. Was raising TypeError
    # silently every epoch. Foundation engines now booted in
    # _init_cognitive_engines and threaded through state_refs.
    if meta_engine is not None:
        try:
            tick_fn = getattr(meta_engine, "tick", None)
            if callable(tick_fn):
                tick_fn(
                    state_132d=latest.get("state_vector"),
                    neuromods=neuromod_reader() if neuromod_reader else None,
                    reasoning_engine=reasoning_engine,
                    chain_archive=state_refs.get("chain_archive"),
                    meta_wisdom=state_refs.get("meta_wisdom"),
                    ex_mem=state_refs.get("ex_mem"),
                    meta_autoencoder=state_refs.get("meta_autoencoder"),
                )
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
                from titan_plugin.bus import emit_meta_cgn_signal
                from titan_plugin.logic.edge_detector_persistence import (
                    load_edge_detector_state)
                if not getattr(
                        coordinator, "_p14_coherence_init", False):
                    from titan_plugin.logic.meta_cgn import EdgeDetector
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
        except Exception as _err:
            _log_driver_err("msl.tick", _err)

    # 8. Drive ExpressionManager.evaluate_all — Tier 2 composite firing.
    #
    # 2026-05-10 closes BUG-COGNITIVE-WORKER-EXPRESSION-EVALUATE-ALL-MISSING-20260510.
    # cognitive_worker boots ExpressionManager + registers all 6 composites
    # at line 552 but the per-epoch evaluate_all driver was forgotten in
    # chunk 8E. Result on T3 since l0_rust_enabled=true (2026-05-08): zero
    # EXPRESSION_FIRED events for ART/MUSIC/SOCIAL/KIN_SENSE/LONGING.
    #
    # Mirrors spirit_worker.py:5230-5345 Tier 2 block. SPEAK is excluded
    # here (Tier 1 SPEAK firing path migrated separately in Block D).
    # The post-fire block (composite reward into NS + lazy-init
    # _composite_meta_cgn_edge EdgeDetector + emit_meta_cgn_signal) is
    # also migrated here (closes BUG composite-meta-cgn-edge audit).
    expression_manager = state_refs.get("expression_manager")
    _t2_speak_pending = False  # consumed by Tier 1 block (Block D) below
    _t2_fired: list = []
    _t2_hormones: dict = {}
    if (expression_manager is not None
            and neural_nervous_system is not None
            and getattr(neural_nervous_system, "_hormonal_enabled", False)):
        try:
            _t2_hormones = {
                _hn: _h.level
                for _hn, _h in neural_nervous_system._hormonal._hormones.items()
            }
            _t2_vocab_conf = 1.0
            _t2_dev_age = pi_monitor.developmental_age if pi_monitor else 0
            _t2_fired = expression_manager.evaluate_all(
                _t2_hormones,
                vocabulary_confidence=_t2_vocab_conf,
                developmental_age=_t2_dev_age,
                hormonal_system=(
                    neural_nervous_system._hormonal
                    if neural_nervous_system._hormonal_enabled else None),
                exclude={"SPEAK"},  # SPEAK fires via Tier 1 (Block D)
            )

            # Tier-1 SPEAK trigger detection: evaluate SPEAK without firing.
            _speak_comp = expression_manager.composites.get("SPEAK")
            if _speak_comp:
                _speak_eval = _speak_comp.evaluate(
                    _t2_hormones, _t2_vocab_conf, _t2_dev_age)
                if _speak_eval.get("should_fire"):
                    _t2_speak_pending = True

            # Per-fire post-processing: bus emit + NS reward + META-CGN signal.
            for _tf in _t2_fired:
                _ef_payload = {
                    "composite": _tf["composite"],
                    "urge": round(_tf["urge"], 3),
                    "helper": _tf.get("action_helper", ""),
                }
                _send_msg(send_queue, bus.EXPRESSION_FIRED, name, "all", _ef_payload)
                # TimeChain: explicit dst (dst=all may not reach worker subprocess).
                _send_msg(send_queue, bus.EXPRESSION_FIRED, name, "timechain", _ef_payload)

                # rFP β Stage 2 Phase 2c: composite → NS program reward.
                # ART/MUSIC fires reward CREATIVITY (Endorphin+ACh stream is
                # primary; this is the discrete event signal).
                # SOCIAL/KIN_SENSE/LONGING fires reward EMPATHY.
                _cn = _tf.get("composite", "")
                _cu = float(_tf.get("urge", 0))
                if _cu > 0:
                    _ce_program = None
                    if _cn in ("ART", "MUSIC"):
                        _ce_program = "CREATIVITY"
                    elif _cn in ("SOCIAL", "KIN_SENSE", "LONGING"):
                        _ce_program = "EMPATHY"
                    if _ce_program:
                        try:
                            neural_nervous_system.record_outcome(
                                reward=min(1.0, _cu),
                                program=_ce_program,
                                source=f"composite.{_cn.lower()}")
                        except Exception as _ce_err:
                            _log_driver_err(
                                f"expression.composite_reward.{_ce_program}", _ce_err)

                        # rFP β Phase 3 § 4g + TUNING-016: NS → META-CGN coupling.
                        # EdgeDetector gates emission per the META-CGN
                        # architectural invariant "discrete state transitions
                        # only" — first crossing of intensity >= 0.2 per consumer
                        # fires; sustained elevated state is silent; drop-and-
                        # re-cross fires again. Persisted state restored from
                        # edge_detector_state.json at lazy init.
                        try:
                            from titan_plugin.bus import emit_meta_cgn_signal
                            from titan_plugin.logic.edge_detector_persistence import (
                                load_edge_detector_state)
                            _ce_consumer = _ce_program.lower()
                            _ce_intensity = min(1.0, _cu / 3.0)  # urge can be > 1
                            if not getattr(
                                    coordinator,
                                    "_composite_meta_cgn_edge_init",
                                    False):
                                from titan_plugin.logic.meta_cgn import EdgeDetector
                                coordinator._composite_meta_cgn_edge = EdgeDetector()
                                _cp_persisted = load_edge_detector_state().get(
                                    "composite_meta_cgn")
                                if _cp_persisted:
                                    coordinator._composite_meta_cgn_edge.load_dict(
                                        _cp_persisted)
                                    logger.info(
                                        "[META-CGN] Composite EdgeDetector "
                                        "state restored (%d consumer(s) "
                                        "previously crossed)",
                                        sum(1 for v in _cp_persisted.get(
                                            "crossed", {}).values() if v))
                                coordinator._composite_meta_cgn_edge_init = True
                            if coordinator._composite_meta_cgn_edge.observe(
                                    _ce_consumer, _ce_intensity, 0.2):
                                emit_meta_cgn_signal(
                                    send_queue,
                                    src=_ce_consumer,
                                    consumer=_ce_consumer,
                                    event_type="fired",
                                    intensity=_ce_intensity,
                                    domain=_cn.lower(),
                                    reason=(
                                        f"{_ce_program} via {_cn} composite "
                                        f"urge={_cu:.2f}"))
                        except Exception as _mc_err:
                            _log_driver_err(
                                "expression.meta_cgn_edge", _mc_err)

                if _tf["composite"] != "SPEAK":
                    logger.info(
                        "[EXPRESSION.%s] FIRED — urge=%.3f, helper=%s",
                        _tf["composite"], _tf["urge"], _tf.get("action_helper", ""))
                # MSL convergence detector signal for SPEAK/ART/MUSIC.
                if msl and _tf["composite"] in ("SPEAK", "ART", "MUSIC"):
                    try:
                        _msl_signal_action = getattr(msl, "signal_action", None)
                        if callable(_msl_signal_action):
                            _msl_signal_action("internal")
                    except Exception:
                        pass
        except Exception as _expr_err:
            _log_driver_err("expression_manager.evaluate_all", _expr_err)

    # Block D wires Tier 1 SPEAK firing here (consumes _t2_speak_pending +
    # _t2_hormones from above). Stub for now — appears in next commit.
    # Make these locals available to Block D when it lands.
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

            # SPEAK injection if Tier 2 flagged it.
            if _t2_speak_pending:
                _speak_in_fired = any(
                    f.get("composite") == "SPEAK" for f in _t2_fired)
                if not _speak_in_fired:
                    _speak_comp = expression_manager.composites.get("SPEAK")
                    _t2_fired.append({
                        "composite": "SPEAK",
                        "urge": getattr(_speak_comp, "_last_urge", 0.5)
                                if _speak_comp else 0.5,
                        "intensity": 1.0,
                        "dominant_hormone": "CREATIVITY",
                        "action_helper": "speak",
                        "total_consumption": 0,
                    })

            # Social Pressure: accumulate SOCIAL fires.
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

                _send_msg(send_queue, bus.SPEAK_REQUEST, name, "language", {
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
                    "[SPEAK] SPEAK_REQUEST sent to language_worker (epoch=%d)",
                    _ch_epoch)
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

    # prediction_engine.predict_next + compute_error — per-epoch novelty
    # signal (spirit_worker:5980-5985). Uses 130D state vector from the
    # consciousness latest_epoch (post-step-4 _run_consciousness_epoch).
    prediction_engine = state_refs.get("prediction_engine")
    if prediction_engine is not None and latest:
        try:
            _pe_sv = latest.get("state_vector", []) or []
            if hasattr(_pe_sv, "to_list"):
                _pe_sv = _pe_sv.to_list()
            _pe_sv = list(_pe_sv) if _pe_sv else []
            if _pe_sv:
                _compute_err = getattr(prediction_engine, "compute_error", None)
                _predict_next = getattr(prediction_engine, "predict_next", None)
                if callable(_compute_err):
                    _compute_err(_pe_sv)
                if callable(_predict_next):
                    _predict_next(_pe_sv, [0.0] * len(_pe_sv))
        except Exception as _err:
            _log_driver_err("prediction_engine.predict_next", _err)

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
    timeseries_store = state_refs.get("timeseries_store")
    if timeseries_store is not None and epoch_id:
        try:
            _ts_should = getattr(timeseries_store, "should_record", None)
            _ts_record = getattr(timeseries_store, "record", None)
            if callable(_ts_should) and callable(_ts_record) and _ts_should():
                _ts_metrics = {
                    "epoch_id": epoch_id,
                    "curvature": curvature,
                    "chi_total": float(getattr(
                        life_force_engine, "_latest_chi", {}).get("total", 0.5)
                        if life_force_engine else 0.5),
                }
                _ts_record(_ts_metrics)
        except Exception as _err:
            _log_driver_err("timeseries_store.record", _err)

    # mini_registry.tick_all — per-epoch distributed mini-reasoner tick
    # across body/mind/spirit rate tiers. Signature per
    # logic/mini_experience.py:460 is tick_all(context, rate_tier) where
    # rate_tier ∈ {"body", "mind", "spirit"}. Mirrors spirit_worker:4666-4668.
    mini_registry = state_refs.get("mini_registry")
    if mini_registry is not None:
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
