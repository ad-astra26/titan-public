"""
outer_interface_worker — L2 Subprocess (SPEC v1.2.1 §9.B + D-SPEC-38).

Track 2 of `rFP_phase_c_self_improvement_subsystem_migration.md` (rewritten
2026-05-11, SPEC-aligned to `rFP_titan_hcl_l2_separation_strategy.md` §4.F
LOCKED 2026-05-05).

Owns under `microkernel.l0_rust_enabled=true`:
  • `OuterInterface` (composition_engine + narrator + advisor + decoder),
  • dynamic word-recipes registry (`data/word_recipes.json`),
  • kin signature + kin society broadcast surface,
  • self-exploration cadence driver.

Closes the T3 SPEAK quality regression observed since 2026-05-10 deploy
(un-perturbed words, advisor refractory bypassed) by extending the
`SPEAK_REQUEST` Python L2 event with `word_perturbations` (§8.5).

# Boot signals

  → (MODULE_READY retired — Phase 11 §11.I.2 SHM slot state=booted is the contract)
  → `MODULE_HEARTBEAT`    — every HEARTBEAT_INTERVAL_S (10s per SPEC §10.B)

# Subscribed types (SPEC §9.B `outer_interface_worker` row + §2.A.3):

  REQUIRED — REASONING_STATS_UPDATED, NEUROMOD_STATE, KERNEL_EPOCH_TICK,
             EXPRESSION_FIRED, CONVERSATION_STIMULUS, MODULE_SHUTDOWN
  OPTIONAL — EMOT_KIN_STATE, SAVE_NOW

# Published types (chunks A7 + A8 wire the actual publishers):

  OUTER_INTERFACE_STATS_UPDATED   (2.5s coalesced) → cache.outer_interface.stats
  KIN_SIGNATURE_UPDATED           (2.5s coalesced) → cache.kin.signature
  KIN_SOCIETY_UPDATED             (10s coalesced)  → cache.kin.society
  WORD_PERTURBATION_HINT          (per SPEAK_REQUEST_PENDING precursor)
  ADVISOR_REFRACTORY_STATE        (on change; coalesce-by-titan_id)

# Persisted state (§11.H.1 critical-data, v1.2.1):

  data/word_recipes.json          — narrator's dynamic learned recipes
  data/outer_interface_state.json — advisor refractory + composition state

# Flag-gating

When `microkernel.outer_interface_worker_enabled = false` OR
`microkernel.l0_rust_enabled = false`, this worker enters a heartbeat-only
no-op loop (legacy `spirit_worker_main` owns OuterInterface in those
modes). guardian skips registration entirely when `_worker_enabled = false`
(see `legacy_core.py`); this in-worker check is defensive for the case
where l0_rust=false but the worker_enabled flag was left true.

# Boot chunk sequence (mirrors cognitive_worker chunks 8E → 8L pattern):

  A4 (this commit): boot section + heartbeat-only main loop
  A5 (next):        ModuleSpec registration in legacy_core.py
  A6:               bus subscription dispatcher (8 inbound handlers)
  A7:               publishers + WORD_PERTURBATION_HINT/ADVISOR_REFRACTORY_STATE
  A8:               cognitive_worker SPEAK gate consumer + language_worker
                    word_perturbations consumer
  A9:               dashboard /v4/self-exploration + /v4/kin-* routes
  A10:              unit + integration tests
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

logger = logging.getLogger("outer_interface_worker")

# Module name (matches Guardian registry per SPEC §9.B v1.2.1 outer_interface_worker row).
MODULE_NAME = "outer_interface_worker"

# Phase 11 §11.I.3 / §11.I.5 (Chunk 11N) — module-level readiness sentinel
# mirrored to per-process SHM slot via ModuleStateWriter. Set False at
# import; flipped True after OuterInterface init + publisher start.
from titan_hcl.modules._heartbeat_grace import (
    boot_deadline_from_now, shm_heartbeat_allowed,
)

_WORKER_READY: bool = False
_BOOT_DEADLINE = None  # boot-grace deadline (monotonic); None=no grace

# Cadence + lifecycle constants (defaults — per-titan overridable via [outer_interface]).
HEARTBEAT_INTERVAL_S = 10.0           # SPEC §10.B MODULE_HEARTBEAT_INTERVAL_S
POLL_INTERVAL_S = 0.2
SAVE_RECIPES_DEFAULT_S = 300.0        # narrator.save_dynamic_recipes cadence
SELF_EXPLORATION_DEFAULT_S = 30.0     # tick_self_exploration cadence
PUBLISHER_DEFAULT_S = 2.5             # OUTER_INTERFACE_STATS_UPDATED cadence

# Subscribed bus message types — every event the worker's dispatcher handles
# (A6 wires the actual handlers). Broker filters dst="all" broadcasts at
# publish time so only listed types reach this subscriber (closes the
# per-subscriber flood class identified 2026-04-30).
#
# Note: SPEC §9.B v1.2.1 lists `NEUROMOD_STATE` as REQUIRED — this is a SHM
# SLOT (neuromod_state.bin), NOT a bus event (per cognitive_worker.py
# line 117: "NEUROMOD_STATE is a SHM SLOT not a bus event — read via shm").
# The worker reads neuromod GABA from the slot at each KERNEL_EPOCH_TICK
# tick — wiring lands in chunk A6 (mirrors cognitive_worker's
# _make_neuromod_reader pattern). `NEUROMOD_STATS_UPDATED` (real bus event,
# published by cognitive_worker every 2.5s) is also subscribed as the bus-
# layer fallback when shm slot isn't fresh.
#
# `EMOT_KIN_STATE` cited in SPEC §9.B (OPTIONAL) does not yet exist as a
# bus constant — chunk A6 either adds it (if emot_cgn_module produces a
# kin-emotion broadcast) or uses the existing GREAT_KIN_PULSE wire.
_OUTER_INTERFACE_WORKER_SUBSCRIBE_TOPICS: list[str] = [
    # Composition + advisor feeds
    bus.REASONING_STATS_UPDATED,   # cognitive_worker → composition word-boost feed
    bus.NEUROMOD_STATS_UPDATED,    # cognitive_worker (2.5s) → GABA bus-layer fallback
    bus.CHI_UPDATED,               # cognitive_worker (per epoch) → chi for tick_self_exploration
    bus.KERNEL_EPOCH_TICK,         # 1 Hz tick — gates tick_self_exploration()
    bus.EXPRESSION_FIRED,          # cognitive_worker (composite ∈ SPEAK/SOCIAL/ART/MUSIC/KIN)
    bus.CONVERSATION_STIMULUS,     # chat / outer dispatch → narrator perturbation pre-compute
    # SPEAK gating precursor (v1.2.1 — closes T3 SPEAK regression)
    bus.SPEAK_REQUEST_PENDING,     # cognitive_worker → compute WORD_PERTURBATION_HINT
    # Kin resonance signal — substantive kin-related bus event today
    bus.GREAT_KIN_PULSE,           # cross-titan kin resonance (post-maturity gate)
    # Lifecycle
    bus.MODULE_SHUTDOWN,
    bus.SAVE_NOW,                  # B.1 shadow_swap orchestrator (when re-enabled)
]


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


def _send_heartbeat(send_queue, name: str, extra: Optional[dict] = None,
                    state_writer: Optional[Any] = None) -> None:
    """Emit MODULE_HEARTBEAT to guardian_HCL with current RSS.

    Phase 11 §11.I.5 (Chunk 11N): also publishes ModuleStateWriter.heartbeat()
    on the SHM slot when `state_writer` is provided AND `_WORKER_READY` is
    True so guardian_HCL's SHM-staleness detector + observatory /v6/readiness
    see fresh data on the same cadence as the legacy bus path.
    """
    try:
        import resource
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        rss_mb = 0.0
    payload = {"alive": True, "ts": time.time(), "rss_mb": round(rss_mb, 1),
               "chunk": "A4"}
    if extra:
        payload.update(extra)
    _send_msg(send_queue, bus.MODULE_HEARTBEAT, name, "guardian", payload)
    if state_writer is not None and shm_heartbeat_allowed(_WORKER_READY, _BOOT_DEADLINE):
        try:
            state_writer.heartbeat()
        except Exception:  # noqa: BLE001 — never crash heartbeat
            pass


def _heartbeat_loop(recv_queue, send_queue, name: str, *, flag_off: bool,
                    state_writer: Optional[Any] = None) -> None:
    """Heartbeat-only loop for flag-off / defensive-noop branches.

    Exits cleanly on MODULE_SHUTDOWN. No OuterInterface init, no dispatcher,
    no publishers. Used when:
      • microkernel.outer_interface_worker_enabled = false (rare — guardian
        normally skips registration entirely in that mode);
      • microkernel.l0_rust_enabled = false (legacy spirit_worker_main
        owns OuterInterface).

    Phase 11 §11.I.5: also handles MODULE_PROBE_REQUEST via the supplied
    state_writer so flag-off no-op workers still answer probes from
    titan_hcl (probe gets trivial-pass since worker is intentionally
    idle).
    """
    last_heartbeat_ts = 0.0
    while True:
        now = time.time()
        if now - last_heartbeat_ts >= HEARTBEAT_INTERVAL_S:
            _send_heartbeat(send_queue, name, extra={"flag_off_noop": flag_off},
                            state_writer=state_writer)
            last_heartbeat_ts = now
        try:
            msg = recv_queue.get(timeout=POLL_INTERVAL_S)
        except Empty:
            continue
        except Exception:
            continue
        mtype = msg.get("type") if isinstance(msg, dict) else None
        if mtype == bus.MODULE_PROBE_REQUEST and state_writer is not None:
            try:
                from titan_hcl.core.probe_dispatcher import (
                    handle_module_probe_request,
                )
                handle_module_probe_request(
                    msg, probe_fn=None, send_queue=send_queue,
                    module_name=name, state_writer=state_writer,
                )
            except Exception as _probe_err:  # noqa: BLE001
                logger.warning(
                    "[OuterInterfaceWorker] MODULE_PROBE_REQUEST handler "
                    "failed (flag_off branch): %s", _probe_err)
            continue
        if mtype == bus.MODULE_SHUTDOWN:
            logger.info(
                "[OuterInterfaceWorker] Shutdown received (flag_off branch)")
            return


def _init_outer_interface(config: dict, titan_id: str):
    """Construct an OuterInterface with the canonical 3-engine cluster
    (decoder + narrator + advisor + composition_engine via narrator).

    Returns the OuterInterface instance or None on init failure (worker
    enters defensive heartbeat loop in that case so guardian doesn't
    restart-loop on a deterministic init bug).
    """
    try:
        from titan_hcl.logic.outer_interface import OuterInterface
    except Exception as e:
        logger.error(
            "[OuterInterfaceWorker] OuterInterface import failed: %s", e,
            exc_info=True)
        return None

    oi_cfg = (config.get("outer_interface", {}) or {})
    word_recipe_dir = oi_cfg.get("word_recipe_dir", "data")

    # Resolve absolute path against project root so the worker subprocess
    # writes to the canonical data/ subtree (not subprocess cwd).
    project_root = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", ".."))
    if not os.path.isabs(word_recipe_dir):
        word_recipe_dir = os.path.join(project_root, word_recipe_dir)

    # kin.dna params (neurochemical reward params for kin resonance — fed
    # through advisor's GABA-governed cooldown surface).
    dna_params = (config.get("kin", {}) or {}).get("dna", {}) or None

    try:
        outer_interface = OuterInterface(
            word_recipe_dir=word_recipe_dir,
            inner_memory=None,           # worker uses bus events for memory, not direct attr access (per G18)
            dna_params=dna_params,
            params_config=config,
        )
    except Exception as e:
        logger.error(
            "[OuterInterfaceWorker] OuterInterface init failed: %s — "
            "entering defensive heartbeat loop", e, exc_info=True)
        return None

    logger.info(
        "[OuterInterfaceWorker] OuterInterface booted — word_recipe_dir=%s "
        "decoder=%s narrator=%s advisor=%s (self-exploration enabled)",
        word_recipe_dir,
        outer_interface.decoder is not None,
        outer_interface.narrator is not None,
        outer_interface.advisor is not None,
    )
    return outer_interface


# ARG ORDER (template-canonical — see cognitive_worker.py:135-137): every
# Guardian-spawned L2 worker entry follows (recv_queue, send_queue, name, config).
@with_error_envelope(module_name="outer_interface_worker", subsystem="entry", severity=_phase11_sev.FATAL)
def outer_interface_worker_main(recv_queue, send_queue, name: str, config: dict) -> None:
    """Main loop for the outer_interface_worker subprocess.

    Chunk A4 skeleton — boot section + heartbeat-only main loop. Bus
    dispatcher (A6), publishers + WORD_PERTURBATION_HINT/ADVISOR_REFRACTORY_STATE
    (A7), and cognitive_worker/language_worker consumer updates (A8) land in
    subsequent commits per the rFP §8 chunk sequence.
    """
    # === BOILERPLATE: spawn-mode sys.path bootstrap ===
    project_root = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # === BOILERPLATE: Phase B.2 §C7 socket-mode bus client setup ===
    # Falls back to mp.Queue in legacy mode. The `topics` list is module-
    # specific and enumerates every event type the worker's dispatcher
    # handles (A6 wires the handlers; broker uses this list at subscribe
    # time to filter dst="all" broadcasts).
    from titan_hcl.core.worker_bus_bootstrap import setup_worker_bus
    try:
        recv_queue, send_queue, _bus_client = setup_worker_bus(
            name, recv_queue, send_queue,
            topics=_OUTER_INTERFACE_WORKER_SUBSCRIBE_TOPICS,
        )
    except Exception as _err:
        logger.error(
            "[OuterInterfaceWorker] setup_worker_bus failed: %s — exiting",
            _err, exc_info=True)
        return

    # === BOILERPLATE: pdeathsig installation ===
    try:
        from titan_hcl.core.worker_lifecycle import install_parent_death_signal
        install_parent_death_signal()
    except Exception as _err:
        logger.debug("[OuterInterfaceWorker] pdeathsig install skipped: %s", _err)

    # Phase 11 §11.I.5 (Chunk 11N) — reset module-level readiness sentinel.
    global _WORKER_READY, _BOOT_DEADLINE
    _WORKER_READY = False
    _BOOT_DEADLINE = boot_deadline_from_now()

    # Canonical titan_id resolution (per feedback_titan_id_canonical_resolve.md
    # — SPEC §23.17 R-PORT-1; T2/T3 deployments may have missing
    # info_banner.titan_id, fall back to resolve_titan_id() which probes
    # canonical /dev/shm/titan_{T1,T2,T3}/ directories).
    from titan_hcl.core.state_registry import resolve_titan_id
    titan_id = (
        (config.get("info_banner", {}) or {}).get("titan_id")
        or resolve_titan_id()
        or "T1"
    )
    boot_ts = time.time()

    # ── Phase 11 §11.I.5 / Chunk 11N — SHM state-slot writer (G21) ──
    # Built BEFORE the flag-gate checks + OuterInterface init so all
    # boot exit paths (flag-off, init-fail, happy) can transition the
    # slot through starting → booted.
    _state_writer = None
    try:
        from titan_hcl.core.module_state import (
            BootPriority,
            ModuleStateWriter,
        )
        _state_writer = ModuleStateWriter(
            module_name=name,
            layer="L2",
            boot_priority=BootPriority.OPTIONAL_POST_BOOT,
        )
        _state_writer.write_state("starting")
    except Exception as _sw_err:  # noqa: BLE001
        logger.warning(
            "[OuterInterfaceWorker] Phase 11 ModuleStateWriter init failed "
            "(continuing — SHM slot disabled): %s", _sw_err)

    # === BOILERPLATE: flag-gated activation ===
    # Two flags gate this worker's behavior:
    #   1. microkernel.l0_rust_enabled — when false, legacy spirit_worker_main
    #      owns OuterInterface. We enter heartbeat-only noop.
    #   2. microkernel.outer_interface_worker_enabled — defensive; normally
    #      guardian skips registration when false, but this in-worker check
    #      catches the case where the flag was flipped post-registration.
    microkernel_cfg = (config or {}).get("microkernel", {}) or {}
    l0_rust = bool(microkernel_cfg.get("l0_rust_enabled", False))
    worker_enabled = bool(microkernel_cfg.get("outer_interface_worker_enabled", True))
    oi_section_enabled = bool(
        (config.get("outer_interface", {}) or {}).get("enabled", True))

    if not l0_rust:
        logger.info(
            "[OuterInterfaceWorker] microkernel.l0_rust_enabled=false — "
            "legacy spirit_worker_main owns OuterInterface in this mode. "
            "Entering heartbeat-only no-op loop.")
        # Phase 11 §11.I.2 — flag-off branch transitions SHM slot to booted
        # immediately (worker is intentionally idle, but must still answer
        # probes). Legacy MODULE_READY bus emit retired per locked D2.
        _WORKER_READY = True
        if _state_writer is not None:
            try:
                _state_writer.write_state("booted")
            except Exception as _swb_err:  # noqa: BLE001
                logger.warning(
                    "[OuterInterfaceWorker] Phase 11 write_state(booted) "
                    "(flag_off l0_rust=false) failed: %s", _swb_err)
        _heartbeat_loop(recv_queue, send_queue, name, flag_off=True,
                        state_writer=_state_writer)
        return

    if not (worker_enabled and oi_section_enabled):
        logger.info(
            "[OuterInterfaceWorker] worker_enabled=%s outer_interface.enabled=%s "
            "— entering heartbeat-only no-op loop.",
            worker_enabled, oi_section_enabled)
        _WORKER_READY = True
        if _state_writer is not None:
            try:
                _state_writer.write_state("booted")
            except Exception as _swb_err:  # noqa: BLE001
                logger.warning(
                    "[OuterInterfaceWorker] Phase 11 write_state(booted) "
                    "(flag_off worker_disabled) failed: %s", _swb_err)
        _heartbeat_loop(recv_queue, send_queue, name, flag_off=True,
                        state_writer=_state_writer)
        return

    logger.info(
        "[OuterInterfaceWorker] Booting (titan_id=%s, l0_rust=true) — chunk A4 "
        "skeleton. Bus dispatcher / publishers / SPEAK consumer updates land "
        "in chunks A6–A8.", titan_id)

    # === MODULE-SPECIFIC: OuterInterface init ===
    outer_interface = _init_outer_interface(config, titan_id)
    if outer_interface is not None:
        # AUDIT §C fix (rFP §P2): restore advisor refractory + composition +
        # mode/stats from disk on boot. Previously restore_state() was never
        # called → fresh blank state every respawn (the save side was also a
        # NOP, now fixed). No-op on first boot (file absent).
        try:
            outer_interface.load_state()
        except Exception as _ld_err:  # noqa: BLE001
            logger.warning(
                "[OuterInterfaceWorker] outer_interface.load_state() failed: %s",
                _ld_err)
    if outer_interface is None:
        # Defensive — _init_outer_interface logs the failure. Fall into
        # heartbeat-only loop so guardian doesn't restart-loop us.
        logger.warning(
            "[OuterInterfaceWorker] OuterInterface init returned None — "
            "entering defensive heartbeat loop.")
        # Phase 11 §11.I.2 — init-failure branch transitions slot to
        # `unhealthy` so titan_hcl + observatory surface the degraded
        # state rather than treating it as a healthy boot.
        if _state_writer is not None:
            try:
                _state_writer.write_state("unhealthy")
            except Exception as _swb_err:  # noqa: BLE001
                logger.warning(
                    "[OuterInterfaceWorker] Phase 11 write_state(unhealthy) "
                    "(init_failed) failed: %s", _swb_err)
        _heartbeat_loop(recv_queue, send_queue, name, flag_off=True,
                        state_writer=_state_writer)
        return

    # state_refs dict for snapshot publishers (A7) — populated here so the
    # template-canonical shape consumed by future snapshot builder threads
    # is established at boot.
    state_refs: dict = {
        "outer_interface": outer_interface,
        "_last_neuromod_gaba": 0.5,    # NEUROMOD_STATE handler updates this (A6)
        "_lang_boosts": {},            # REASONING_STATS_UPDATED handler populates (A6)
        "_lang_bias": {},              # REASONING_STATS_UPDATED handler populates (A6)
        "_advisor_state_cache": None,  # outbound ADVISOR_REFRACTORY_STATE prev value
        "_save_recipes_ts": time.time(),
        "_publisher_ts": time.time(),
        "_self_exploration_ts": time.time(),
    }

    # Cadences (from [outer_interface] params with defaults).
    oi_cfg = (config.get("outer_interface", {}) or {})
    save_recipes_every_s = float(oi_cfg.get(
        "save_recipes_every_s", SAVE_RECIPES_DEFAULT_S))
    self_exploration_cadence_s = float(oi_cfg.get(
        "self_exploration_cadence_s", SELF_EXPLORATION_DEFAULT_S))
    publisher_cadence_s = float(oi_cfg.get(
        "publisher_cadence_s", PUBLISHER_DEFAULT_S))

    # === MODULE-SPECIFIC: launch cadence-driven publisher daemon (chunk A7) ===
    # Daemon thread fires every publisher_cadence_s (default 2.5s) and
    # emits: OUTER_INTERFACE_STATS_UPDATED, KIN_SIGNATURE_UPDATED (2.5s),
    # KIN_SOCIETY_UPDATED (10s), ADVISOR_REFRACTORY_STATE (on change).
    # Also gates self_exploration tick (default 30s) and narrator
    # save_dynamic_recipes (default 5 min).
    _publisher_stop_event = threading.Event()
    state_refs["_publisher_stop_event"] = _publisher_stop_event
    kin_signature_cadence_s = float((config.get("outer_interface", {}) or {}).get(
        "kin_signature_cadence_s", PUBLISHER_DEFAULT_S))
    kin_society_cadence_s = float((config.get("outer_interface", {}) or {}).get(
        "kin_society_cadence_s", 10.0))
    _publisher_thread = threading.Thread(
        target=_publisher_loop,
        args=(state_refs, send_queue, name, titan_id, _publisher_stop_event,
              publisher_cadence_s, kin_signature_cadence_s,
              kin_society_cadence_s, self_exploration_cadence_s,
              save_recipes_every_s),
        name=f"outer_interface_publisher_{titan_id}",
        daemon=True,
    )
    _publisher_thread.start()

    # ── Phase 11 §11.I.2 — slot transition: starting → booted ─────────
    # Replaces the legacy MODULE_READY bus emit per locked D2 — the SHM
    # slot state=booted is the contract.
    _WORKER_READY = True
    if _state_writer is not None:
        try:
            _state_writer.write_state("booted")
            logger.info(
                "[OuterInterfaceWorker] Phase 11 §11.I.2 — SHM slot "
                "state=booted (awaiting MODULE_PROBE_REQUEST from titan_hcl)")
        except Exception as _swb_err:  # noqa: BLE001
            logger.warning(
                "[OuterInterfaceWorker] Phase 11 write_state(booted) "
                "failed: %s", _swb_err)
    logger.info(
        "[OuterInterfaceWorker] online — publisher cadences: "
        "stats=%.2fs kin_sig=%.2fs kin_soc=%.2fs self_explore=%.2fs save_recipes=%.0fs",
        publisher_cadence_s, kin_signature_cadence_s, kin_society_cadence_s,
        self_exploration_cadence_s, save_recipes_every_s)

    # === BOILERPLATE: main loop — chunk A4 = heartbeat-only ===
    # A6 wires the actual bus message dispatcher; A7 wires the snapshot
    # publishers + WORD_PERTURBATION_HINT/ADVISOR_REFRACTORY_STATE emit
    # paths. Until then, drain the recv_queue (so the broker doesn't slow-
    # consume us) but act only on MODULE_SHUTDOWN.
    last_heartbeat_ts = 0.0
    while True:
        now = time.time()

        # Periodic heartbeat.
        if now - last_heartbeat_ts >= HEARTBEAT_INTERVAL_S:
            _send_heartbeat(send_queue, name, state_writer=_state_writer)
            last_heartbeat_ts = now

        # Pull next message (poll-timeout for heartbeat tick).
        try:
            msg = recv_queue.get(timeout=POLL_INTERVAL_S)
        except Empty:
            continue
        except Exception:
            continue

        msg_type = msg.get("type")

        # ── Phase 11 §11.I.3 — MODULE_PROBE_REQUEST handler ────────
        if msg_type == bus.MODULE_PROBE_REQUEST:
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
                    "[OuterInterfaceWorker] MODULE_PROBE_REQUEST handler "
                    "failed: %s", _probe_err)
            continue

        # B.2.1 supervision-transfer dispatch (matches hormonal_worker:302-307).
        try:
            from titan_hcl.core import worker_swap_handler as _swap
            if _swap.maybe_dispatch_swap_msg(msg):
                continue
        except Exception:
            pass

        if msg_type == bus.MODULE_SHUTDOWN:
            logger.info(
                "[OuterInterfaceWorker] Shutdown received — stopping publisher "
                "daemon + saving narrator recipes + advisor state + exiting")
            stop_evt = state_refs.get("_publisher_stop_event")
            if stop_evt is not None:
                stop_evt.set()
            _save_state_on_shutdown(state_refs)
            return

        # Bus dispatcher (chunk A6) — one branch per subscribed topic. Each
        # handler is best-effort (catches exceptions, logs at WARNING, never
        # raises — the main loop must keep running so guardian heartbeat
        # doesn't trip). State mutations land in state_refs; OuterInterface
        # method calls happen here too. Inbound triggers that emit outbound
        # events (notably SPEAK_REQUEST_PENDING → WORD_PERTURBATION_HINT) call
        # publish helpers defined alongside the cadence-driven publishers in
        # chunk A7.
        try:
            _dispatch_msg(msg, msg_type, state_refs, send_queue, name)
        except Exception as e:
            logger.warning(
                "[OuterInterfaceWorker] dispatch failed for %s: %s",
                msg_type, e, exc_info=False)


def _save_state_on_shutdown(state_refs: dict) -> None:
    """Persist OuterInterface state on shutdown — narrator dynamic recipes
    + advisor refractory + composition state.

    Called from the MODULE_SHUTDOWN branch of the main loop. Best-effort:
    failures log but do not raise (worker must exit cleanly so guardian
    doesn't escalate).
    """
    outer_interface = state_refs.get("outer_interface")
    if outer_interface is None:
        return

    # Narrator dynamic recipes (data/word_recipes.json per §11.H.1).
    try:
        narrator = getattr(outer_interface, "narrator", None)
        if narrator is not None and hasattr(narrator, "save_dynamic_recipes"):
            narrator.save_dynamic_recipes()
            logger.info(
                "[OuterInterfaceWorker] narrator.save_dynamic_recipes() ok")
    except Exception as e:
        logger.warning(
            "[OuterInterfaceWorker] narrator.save_dynamic_recipes() failed: %s",
            e)

    # Advisor + composition state persistence (data/outer_interface_state.json
    # per §11.H.1). The OuterInterface class doesn't yet have a single
    # save_state method — A7 may add one when the publishers land. For the
    # skeleton we no-op gracefully if no save method exists.
    try:
        if hasattr(outer_interface, "save_state"):
            outer_interface.save_state()
            logger.info("[OuterInterfaceWorker] outer_interface.save_state() ok")
    except Exception as e:
        logger.warning(
            "[OuterInterfaceWorker] outer_interface.save_state() failed: %s",
            e)


# ── Bus dispatcher (chunk A6) ───────────────────────────────────────────────


def _dispatch_msg(msg: dict, msg_type: str, state_refs: dict,
                  send_queue, name: str) -> None:
    """Route an inbound bus message to its handler.

    Per rFP §2.A.3 handler table + SPEC v1.2.1 §9.B outer_interface_worker
    Bus subscriptions row. Each handler:
      • is best-effort (caller wraps in try/except — handlers may still
        log warnings on internal sub-step failures);
      • mutates state_refs and/or calls OuterInterface methods;
      • emits outbound events synchronously when the handler completes a
        chain (notably SPEAK_REQUEST_PENDING → WORD_PERTURBATION_HINT).

    MODULE_SHUTDOWN is handled by the main loop directly (terminal); it
    never reaches the dispatcher.
    """
    outer_interface = state_refs.get("outer_interface")
    if outer_interface is None:
        return  # defensive — should not happen post-boot

    payload = msg.get("payload") or {}

    if msg_type == bus.REASONING_STATS_UPDATED:
        _handle_reasoning_stats(payload, state_refs, outer_interface)
        return

    if msg_type == bus.NEUROMOD_STATS_UPDATED:
        _handle_neuromod_stats(payload, state_refs, outer_interface)
        return

    if msg_type == bus.CHI_UPDATED:
        _handle_chi_updated(payload, state_refs)
        return

    if msg_type == bus.KERNEL_EPOCH_TICK:
        _handle_epoch_tick(payload, state_refs, outer_interface)
        return

    if msg_type == bus.EXPRESSION_FIRED:
        _handle_expression_fired(payload, state_refs, outer_interface)
        return

    if msg_type == bus.CONVERSATION_STIMULUS:
        _handle_conversation_stimulus(payload, state_refs, outer_interface)
        return

    if msg_type == bus.SPEAK_REQUEST_PENDING:
        _handle_speak_request_pending(payload, state_refs, outer_interface,
                                       send_queue, name)
        return

    if msg_type == bus.GREAT_KIN_PULSE:
        _handle_great_kin_pulse(payload, state_refs, outer_interface)
        return

    if msg_type == bus.SAVE_NOW:
        # Shadow-swap orchestrator (B.1 readiness) — best-effort persist.
        _save_state_on_shutdown(state_refs)
        return

    # Unhandled types (subscribed but no logic yet): silently no-op.
    # Future SPEC §9.B revisions or new subscribe topics land their handlers
    # here.


def _handle_reasoning_stats(payload: dict, state_refs: dict,
                             outer_interface) -> None:
    """Cache `_lang_boosts` + `_lang_bias` from cognitive_worker for the
    next SPEAK_REQUEST_PENDING-triggered WORD_PERTURBATION_HINT computation.

    Per rFP §2.A.3 the legacy spirit_worker also set
    `outer_interface._composition_engine._reasoning_word_boosts /
    _reasoning_template_bias` directly via hasattr-guarded patch (see
    spirit_worker.py:5088-5091). The canonical composition_engine actually
    lives in language_worker per language_pipeline.py:161-184 — the
    OuterInterface's narrator handles word-perturbation, which is where
    the boosts are most useful to us here. We mirror the legacy hasattr-
    guarded patch for backwards compatibility (no-op if attribute absent).
    """
    interpreted = (payload.get("interpreted") or {})
    # Prefer interpreted.* (cognitive_worker's processed form) when present;
    # else fall back to top-level payload fields. The `or` short-circuit
    # would never fall through when interpreted has a literal "default"
    # string for template_bias, so use explicit `in` checks instead.
    if "word_boost" in interpreted:
        lang_boosts = interpreted.get("word_boost") or []
    else:
        lang_boosts = payload.get("word_boost", []) or []
    if "template_bias" in interpreted:
        lang_bias = interpreted.get("template_bias") or "default"
    else:
        lang_bias = payload.get("template_bias", "default")
    state_refs["_lang_boosts"] = lang_boosts
    state_refs["_lang_bias"] = lang_bias

    # Legacy parity: patch the composition_engine attribute if it exists.
    composition_engine = getattr(outer_interface, "_composition_engine", None)
    if composition_engine is not None and lang_boosts:
        try:
            composition_engine._reasoning_word_boosts = lang_boosts
            composition_engine._reasoning_template_bias = lang_bias
        except Exception as e:
            logger.debug(
                "[OuterInterfaceWorker] composition_engine patch skipped: %s", e)


def _handle_chi_updated(payload: dict, state_refs: dict) -> None:
    """Cache chi state from cognitive_worker's per-epoch CHI_UPDATED publish.

    Per SPEC §8 (Python L2 message) + spirit_loop._publish_coord_subdomains.
    Payload shape: `{circulation, total, ...}` (legacy spirit chi publisher
    contract). Surfaces to _tick_self_exploration so advisor refractory
    receives real chi values instead of a 0.5/0.5 placeholder.

    Closes deferral D1 from Phase A+B audit (Prime Directive #1).
    """
    if not isinstance(payload, dict):
        return
    state_refs["_chi_state"] = {
        "circulation": float(payload.get("circulation", 0.5) or 0.5),
        "total": float(payload.get("total", 0.5) or 0.5),
    }


def _handle_neuromod_stats(payload: dict, state_refs: dict,
                            outer_interface) -> None:
    """Extract GABA level → cache for next tick's advisor.check_resume() +
    self_exploration tick. The payload is cognitive_worker's 2.5s coalesced
    neuromod state snapshot — looks like `{neuromods: {GABA: {level: 0.x}}}`
    or flatter `{GABA: 0.x, DA: 0.x, ...}` depending on producer.
    """
    gaba = None
    neuromods = payload.get("neuromods") or payload
    if isinstance(neuromods, dict):
        gaba_entry = neuromods.get("GABA")
        if isinstance(gaba_entry, dict):
            gaba = gaba_entry.get("level")
        elif isinstance(gaba_entry, (int, float)):
            gaba = float(gaba_entry)
    if gaba is not None:
        state_refs["_last_neuromod_gaba"] = float(gaba)

    # Resume-check is cheap — call inline whenever we get fresh GABA so the
    # advisor can flip out of EXTERNAL_PASSTHROUGH as soon as the cooldown
    # window expires (rather than waiting for the next epoch tick).
    try:
        outer_interface.check_resume(state_refs["_last_neuromod_gaba"])
    except Exception as e:
        logger.debug("[OuterInterfaceWorker] check_resume failed: %s", e)


def _handle_epoch_tick(payload: dict, state_refs: dict,
                        outer_interface) -> None:
    """1 Hz tick. Two responsibilities:
      1. Track tick count for self-exploration cadence (default 30s).
      2. Cache circadian_phase if payload carries it (some upstream emitters
         attach phase float; safe to skip if absent).

    Actual `outer_interface.tick_self_exploration(...)` call happens in the
    cadence-driven publisher daemon (A7) since it needs expression_fires +
    chi snapshot which are aggregated there. The epoch tick here just gates
    whether the cadence is due.
    """
    state_refs["_last_epoch_tick_ts"] = time.time()
    phase = payload.get("circadian_phase")
    if isinstance(phase, (int, float)):
        state_refs["_circadian_phase"] = float(phase)


def _handle_expression_fired(payload: dict, state_refs: dict,
                              outer_interface) -> None:
    """A composite fired from cognitive_worker's expression_manager.

    Per rFP §2.A.3: filter on composite ∈ {SPEAK, SOCIAL, ART, MUSIC, KIN}
    and call on_external_interaction() + process_action_result(). The
    composite type tells us this is an action Titan is taking on the world —
    OuterInterface needs to know so it can switch to EXTERNAL_PASSTHROUGH
    mode and let the advisor's refractory cycle kick in.
    """
    composite = payload.get("composite") or payload.get("type") or ""
    if composite not in {"SPEAK", "SOCIAL", "ART", "MUSIC", "KIN"}:
        return  # silently skip non-matching composites

    try:
        outer_interface.on_external_interaction()
    except Exception as e:
        logger.debug("[OuterInterfaceWorker] on_external_interaction failed: %s", e)

    # process_action_result: feed the fire payload back so narrator can
    # learn the action↔word coupling.
    try:
        result = {
            "composite": composite,
            "action_helper": payload.get("action_helper", ""),
            "outcome": payload.get("outcome", "unknown"),
            "ts": payload.get("ts", time.time()),
        }
        outer_interface.process_action_result(composite, result)
    except Exception as e:
        logger.debug("[OuterInterfaceWorker] process_action_result failed: %s", e)


def _handle_conversation_stimulus(payload: dict, state_refs: dict,
                                    outer_interface) -> None:
    """Chat / outer dispatch arriving — pause self-exploration + give
    narrator a chance to pre-compute word perturbations from message text.
    """
    try:
        outer_interface.on_external_interaction()
    except Exception as e:
        logger.debug(
            "[OuterInterfaceWorker] on_external_interaction (chat) failed: %s",
            e)

    # Optional: pre-warm narrator perturbation cache from incoming text.
    text = payload.get("text") or payload.get("message") or ""
    if text and outer_interface.narrator is not None:
        try:
            # Tokenize naively — narrator.get_word_perturbation expects bare
            # words. The narrator's internal logic handles unknowns gracefully.
            for word in str(text).split()[:32]:  # cap to avoid pathological inputs
                outer_interface.narrator.get_word_perturbation(word)
        except Exception as e:
            logger.debug("[OuterInterfaceWorker] narrator pre-warm failed: %s", e)


def _handle_speak_request_pending(payload: dict, state_refs: dict,
                                    outer_interface, send_queue,
                                    name: str) -> None:
    """cognitive_worker SPEAK_REQUEST_PENDING precursor → compute
    narrator.get_word_perturbation per candidate word → emit
    WORD_PERTURBATION_HINT (P2, TTL≤200ms consumer-side).

    This is the load-bearing handler that closes the T3 SPEAK quality
    regression. Without this wire, language_worker emits SPEAK_REQUEST with
    no perturbations and the same words fire over and over.
    """
    request_id = payload.get("request_id")
    candidate_words = payload.get("candidate_words") or []
    if not request_id or not candidate_words:
        return

    narrator = getattr(outer_interface, "narrator", None)
    if narrator is None:
        return

    perturbations: dict[str, float] = {}
    for word in candidate_words[:64]:  # cap to avoid pathological inputs
        try:
            p = narrator.get_word_perturbation(word)
        except Exception:
            p = None
        if isinstance(p, dict):
            # narrator returns a dict — extract a scalar perturbation
            # weight per word for the language_worker's downstream use.
            val = p.get("perturbation") or p.get("weight") or p.get("delta")
            if isinstance(val, (int, float)):
                perturbations[word] = float(val)
        elif isinstance(p, (int, float)):
            perturbations[word] = float(p)
        # Words not in narrator's vocabulary → skipped (consumer falls back
        # to un-perturbed for those words).

    _send_msg(send_queue, bus.WORD_PERTURBATION_HINT, name, "all", {
        "request_id": request_id,
        "words": list(perturbations.keys()),
        "perturbations": perturbations,
        "ts": time.time(),
    })


def _handle_great_kin_pulse(payload: dict, state_refs: dict,
                              outer_interface) -> None:
    """Cross-titan kin resonance signal (post-maturity-gate). Feeds the
    kin signature/society broadcast logic in A7 — for A6 we just cache the
    most-recent pulse so the next periodic KIN_SIGNATURE_UPDATED publisher
    has fresh state to surface.
    """
    state_refs["_last_kin_pulse"] = {
        "resonance_score": payload.get("resonance_score"),
        "peer": payload.get("peer") or payload.get("src"),
        "ts": payload.get("ts", time.time()),
    }


# ── Cadence-driven publishers (chunk A7) ────────────────────────────────────


def _publisher_loop(state_refs: dict, send_queue, name: str, titan_id: str,
                    stop_event: threading.Event,
                    publisher_cadence_s: float,
                    kin_signature_cadence_s: float,
                    kin_society_cadence_s: float,
                    self_exploration_cadence_s: float,
                    save_recipes_every_s: float) -> None:
    """Background daemon — fires every publisher_cadence_s (default 2.5s)
    and dispatches the four cadence-driven outputs per rFP §2.A.4 + SPEC
    v1.2.1 §9.B outer_interface_worker Bus publications:

      • OUTER_INTERFACE_STATS_UPDATED (2.5s) → /v4/self-exploration
      • KIN_SIGNATURE_UPDATED          (2.5s) → /v4/kin-signature
      • KIN_SOCIETY_UPDATED            (10s)  → /v4/kin-society
      • ADVISOR_REFRACTORY_STATE       (on change) → cognitive_worker SPEAK gate

    Plus two cadence-gated side-effects:
      • tick_self_exploration()        (30s default)
      • narrator.save_dynamic_recipes() (300s default; data/word_recipes.json
        per §11.H.1 — atomic via narrator's own writer)

    Exits when stop_event.is_set() (MODULE_SHUTDOWN). The loop tolerates
    OuterInterface method failures by logging at debug + continuing — no
    single publish failure should stop the cadence (e.g. /v4/self-exploration
    might miss one frame but recover on the next).
    """
    last_stats_publish = 0.0
    last_kin_sig_publish = 0.0
    last_kin_soc_publish = 0.0
    last_self_explore = 0.0
    last_save_recipes = time.time()
    last_advisor_state_hash: Optional[tuple] = None
    sleep_interval = min(publisher_cadence_s, kin_signature_cadence_s) / 2.0
    sleep_interval = max(0.5, sleep_interval)  # don't busy-loop

    logger.debug(
        "[OuterInterfaceWorker] publisher_loop online — sleep_interval=%.2fs",
        sleep_interval)

    while not stop_event.is_set():
        now = time.time()
        outer_interface = state_refs.get("outer_interface")
        if outer_interface is None:
            # Worker init failed or shutdown in flight — back off.
            stop_event.wait(sleep_interval)
            continue

        # ── OUTER_INTERFACE_STATS_UPDATED (publisher_cadence_s, 2.5s) ────
        if now - last_stats_publish >= publisher_cadence_s:
            _publish_outer_interface_stats(
                outer_interface, state_refs, send_queue, name, titan_id)
            last_stats_publish = now

        # ── KIN_SIGNATURE_UPDATED (kin_signature_cadence_s, 2.5s) ────────
        if now - last_kin_sig_publish >= kin_signature_cadence_s:
            _publish_kin_signature(state_refs, send_queue, name, titan_id)
            last_kin_sig_publish = now

        # ── KIN_SOCIETY_UPDATED (kin_society_cadence_s, 10s) ─────────────
        if now - last_kin_soc_publish >= kin_society_cadence_s:
            _publish_kin_society(state_refs, send_queue, name, titan_id)
            last_kin_soc_publish = now

        # ── ADVISOR_REFRACTORY_STATE (on change) ──────────────────────────
        new_hash = _advisor_state_hash(outer_interface)
        if new_hash is not None and new_hash != last_advisor_state_hash:
            _publish_advisor_refractory_state(
                outer_interface, send_queue, name, titan_id)
            last_advisor_state_hash = new_hash

        # ── tick_self_exploration cadence (default 30s) ──────────────────
        if now - last_self_explore >= self_exploration_cadence_s:
            _tick_self_exploration(outer_interface, state_refs)
            last_self_explore = now

        # ── narrator.save_dynamic_recipes (default 300s) ─────────────────
        if now - last_save_recipes >= save_recipes_every_s:
            _save_dynamic_recipes_safely(outer_interface)
            last_save_recipes = now

        # Sleep until the next earliest deadline (or stop_event fires).
        stop_event.wait(sleep_interval)

    logger.debug("[OuterInterfaceWorker] publisher_loop exiting (stop_event set)")


def _publish_outer_interface_stats(outer_interface, state_refs: dict,
                                     send_queue, name: str,
                                     titan_id: str) -> None:
    """Emit OUTER_INTERFACE_STATS_UPDATED with outer_interface.get_stats()."""
    try:
        stats = outer_interface.get_stats()
    except Exception as e:
        logger.debug("[OuterInterfaceWorker] get_stats failed: %s", e)
        return
    _send_msg(send_queue, bus.OUTER_INTERFACE_STATS_UPDATED, name, "all", {
        "titan_id": titan_id,
        "stats": stats,
        "ts": time.time(),
    })


def _publish_kin_signature(state_refs: dict, send_queue, name: str,
                             titan_id: str) -> None:
    """Emit KIN_SIGNATURE_UPDATED with the most-recent kin pulse snapshot
    (cached by _handle_great_kin_pulse on inbound GREAT_KIN_PULSE).

    Per existing kin-signature payload contract: peer + resonance_score +
    ts. When no pulse has been received yet, publishes empty signature
    (consumer-side renders gracefully).
    """
    last_pulse = state_refs.get("_last_kin_pulse") or {}
    _send_msg(send_queue, bus.KIN_SIGNATURE_UPDATED, name, "all", {
        "titan_id": titan_id,
        "peer": last_pulse.get("peer"),
        "resonance_score": last_pulse.get("resonance_score"),
        "last_pulse_ts": last_pulse.get("ts"),
        "ts": time.time(),
    })


def _publish_kin_society(state_refs: dict, send_queue, name: str,
                           titan_id: str) -> None:
    """Emit KIN_SOCIETY_UPDATED — coarse-grained 10s view of all known
    peers (from [kin.peers] config) and their last-observed pulse state.
    For now publishes the config'd peer list; richer social-graph state
    can come in a follow-up rFP without re-touching the SPEC contract.
    """
    last_pulse = state_refs.get("_last_kin_pulse") or {}
    _send_msg(send_queue, bus.KIN_SOCIETY_UPDATED, name, "all", {
        "titan_id": titan_id,
        # Surface the most-recent pulse here too so consumers don't need to
        # cross-reference KIN_SIGNATURE_UPDATED.
        "most_recent_pulse": last_pulse,
        "ts": time.time(),
    })


def _advisor_state_hash(outer_interface) -> Optional[tuple]:
    """Compute a stable hash of the advisor's refractory-relevant state so
    we only publish ADVISOR_REFRACTORY_STATE when it actually changes.

    Returns a tuple of (sorted base_refractory items + sorted last_action_time
    items rounded to 1s) or None if advisor is unavailable.
    """
    advisor = getattr(outer_interface, "advisor", None)
    if advisor is None:
        return None
    try:
        base = tuple(sorted((advisor._base_refractory or {}).items()))
        last_actions = tuple(
            sorted((k, round(v, 1)) for k, v in (
                advisor._last_action_time or {}).items()))
        return (base, last_actions)
    except Exception:
        return None


def _publish_advisor_refractory_state(outer_interface, send_queue, name: str,
                                        titan_id: str) -> None:
    """Emit ADVISOR_REFRACTORY_STATE — consumed by cognitive_worker's
    SPEAK gate (A8): if `advisor_state[action_type].next_allowed_ts > now()`,
    cognitive_worker skips emitting SPEAK_REQUEST.

    Per SPEC §8.5 payload schema:
      {
        action_refractory: {
          action_type: {next_allowed_ts: float, cooldown_multiplier: float}
        },
        ts: float,
      }
    """
    advisor = getattr(outer_interface, "advisor", None)
    if advisor is None:
        return
    try:
        base_refractory = advisor._base_refractory or {}
        last_action_time = advisor._last_action_time or {}
        cooldown_multiplier = float(getattr(outer_interface,
                                              "_cooldown_multiplier", 9.0))
    except Exception as e:
        logger.debug("[OuterInterfaceWorker] advisor introspection failed: %s", e)
        return

    now = time.time()
    action_refractory: dict[str, dict] = {}
    for action_type, base in base_refractory.items():
        last_ts = float(last_action_time.get(action_type, 0.0))
        # Use a simple GABA-naive refractory window estimate here — the
        # actual gate cognitive_worker applies just checks
        # `next_allowed_ts > now`. A precise model lives inside advisor.
        next_allowed = last_ts + float(base) * cooldown_multiplier if last_ts > 0 else 0.0
        action_refractory[action_type] = {
            "next_allowed_ts": next_allowed,
            "base_refractory_s": float(base),
        }

    _send_msg(send_queue, bus.ADVISOR_REFRACTORY_STATE, name, "all", {
        "titan_id": titan_id,
        "action_refractory": action_refractory,
        "cooldown_multiplier": cooldown_multiplier,
        "ts": now,
    })


def _tick_self_exploration(outer_interface, state_refs: dict) -> None:
    """Fire outer_interface.tick_self_exploration with the latest cached
    neuromod GABA + cached chi state + an empty expression_fires list
    (the worker doesn't aggregate composite fires here — EXPRESSION_FIRED
    handler already drives on_external_interaction; tick_self_exploration
    is the periodic advisor poll, not the composite-fire path).

    chi values come from CHI_UPDATED bus subscription (D1 fix). Falls back
    to 0.5/0.5 defaults BEFORE the first CHI_UPDATED arrives at boot —
    not a permanent placeholder; updated on the very next publish cycle
    (~per consciousness epoch, every 1-30s adaptive).
    """
    try:
        gaba = float(state_refs.get("_last_neuromod_gaba", 0.5))
        neuromods = {"GABA": {"level": gaba}}
        chi_state = state_refs.get("_chi_state") or {"circulation": 0.5, "total": 0.5}
        outer_interface.tick_self_exploration(
            expression_fires=[],
            neuromodulators=neuromods,
            chi=chi_state,
            hormonal_system=None,
        )
    except Exception as e:
        logger.debug("[OuterInterfaceWorker] tick_self_exploration failed: %s", e)


def _save_dynamic_recipes_safely(outer_interface) -> None:
    """Periodic narrator.save_dynamic_recipes() (5 min default; data/word_recipes.json
    per §11.H.1). Best-effort; failures log at WARNING (not debug — losing
    recipe persistence is a real availability concern over a long uptime).
    """
    narrator = getattr(outer_interface, "narrator", None)
    if narrator is None or not hasattr(narrator, "save_dynamic_recipes"):
        return
    try:
        narrator.save_dynamic_recipes()
        logger.debug(
            "[OuterInterfaceWorker] periodic save_dynamic_recipes() ok")
    except Exception as e:
        logger.warning(
            "[OuterInterfaceWorker] periodic save_dynamic_recipes() failed: %s",
            e)
