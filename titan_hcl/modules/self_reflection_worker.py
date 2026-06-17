"""
self_reflection_worker — L2 Subprocess (SPEC v1.2.1 §9.B + D-SPEC-38).

Track 2 of `rFP_phase_c_self_improvement_subsystem_migration.md` (rewritten
2026-05-11, SPEC-aligned to `rFP_titan_hcl_l2_separation_strategy.md` §4.E
LOCKED 2026-05-05).

Owns under `microkernel.l0_rust_enabled=true`:
  • SelfReasoningEngine        — introspective chains, predictions, dream profile
  • CodingExplorer             — sandboxed code experimentation
  • CodingSandboxHelper        — Python subprocess child (PR_SET_PDEATHSIG cascade)
  • PredictionEngine           — novelty + EMA tracker (relocated from
                                 cognitive_worker per Track 1 drift correction —
                                 see rFP §0 table + commit 51e5cfbf)

# Boot signals

  (Phase 11 §11.I.2 D2: legacy boot-signal bus emit DELETED — SHM slot
   `module_self_reflection_worker_state.bin` state=booted is the contract)
  → MODULE_HEARTBEAT            — every HEARTBEAT_INTERVAL_S (10s per SPEC §10.B)

# Subscribed types (SPEC §9.B self_reflection_worker Bus subscriptions row):

  REQUIRED — REASONING_STATS_UPDATED (canonical proxy for REASONING_COMMIT
             until granular commit event ships per rFP §2.B.3),
             META_REASONING_STATS_UPDATED, EXPERIENCE_STIMULUS,
             DREAMING_STATE_UPDATED (filtered on payload.state ∈
             {dream_start, dream_end} per rFP §2.B.5),
             KERNEL_EPOCH_TICK, MODULE_SHUTDOWN
  OPTIONAL — CGN_CROSS_INSIGHT, SAVE_NOW

# Published types (chunks B7 wires the actual publishers):

  SELF_REFLECTION_STATS_UPDATED   (2.5s coalesced) → cache.self_reflection.state
  SELF_REASONING_INSIGHT          (on insight; consumed by cognitive_worker meta-feed)
  CODING_EXPLORER_STATS_UPDATED   (5s coalesced)   → cache.coding_explorer.state
  CODING_INSIGHT                  (on insight; consumed by cgn_module)
  PREDICTION_STATS_UPDATED        (2.5s coalesced) → cache.prediction.state
  PREDICTION_GENERATED            (on prediction; consumed by cognitive_worker
                                   for novelty-driven exploration)

# Persisted state (§11.H.1 critical-data, v1.2.1):

  data/inner_memory.db           — self_insights, self_predictions,
                                   coding_experiments tables (WAL mode, shared
                                   RO with memory_module)
  data/prediction/novelty_state.json  — EMA novelty state (saved every 100
                                   prediction cycles + on shutdown)

# Flag-gating

When `microkernel.self_reflection_worker_enabled = false` OR
`microkernel.l0_rust_enabled = false`, this worker enters a heartbeat-only
no-op loop. Under l0_rust=false the legacy spirit_worker_main owns
SelfReasoning + CodingExplorer; PredictionEngine stays in cognitive_worker
(Track 1 drift not yet corrected — drift correction lands in chunk B8).

# Boot chunk sequence (mirrors cognitive_worker chunks 8E → 8L pattern):

  B3 (this commit): boot section + 3-engine init + sandbox setup +
                    heartbeat-only main loop
  B4 (next):        ModuleSpec registration in legacy_core.py
  B5:               bus subscription dispatcher (8 inbound handlers)
  B6:               sandbox subprocess lifecycle (30s health + 60s orphan)
  B7:               cadence-driven publishers + on-event emit paths
  B8:               cognitive_worker prediction_engine drift correction
                    (Track 1 corrected — subscribe to PREDICTION_GENERATED)
  B9:               dashboard /v4/self-reflection + /v4/coding-explorer +
                    /v4/prediction routes
  B10:              unit + integration tests + SPEC parity test
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time
from queue import Empty
from typing import Optional

from titan_hcl import bus
from titan_hcl.core.module_error_handler import with_error_envelope
from titan_hcl.errors import Severity as _phase11_sev

logger = logging.getLogger("self_reflection_worker")

# Module name (matches Guardian registry per SPEC §9.B v1.2.1 self_reflection_worker row).
MODULE_NAME = "self_reflection_worker"

# Cadence + lifecycle constants (defaults — overridable via [self_reflection] params).
HEARTBEAT_INTERVAL_S = 10.0           # SPEC §10.B MODULE_HEARTBEAT_INTERVAL_S
_PREDICTION_CHECKPOINT_INTERVAL_S = 300.0  # periodic disk checkpoint — survives crash
POLL_INTERVAL_S = 0.2
PUBLISHER_DEFAULT_S = 2.5             # SELF_REFLECTION_STATS_UPDATED + PREDICTION_STATS_UPDATED
CODING_EXPLORER_PUBLISHER_S = 5.0     # CODING_EXPLORER_STATS_UPDATED
SANDBOX_HEALTH_CHECK_S = 30.0         # CodingSandboxHelper liveness probe
ORPHAN_CHECK_S = 60.0                 # extra-sandbox-PID detection

# Subscribed bus message types (chunk B5 wires the actual handlers).
# Using *_STATS_UPDATED transitional channels where the rFP's _COMMIT variants
# don't yet exist as bus constants (REASONING_COMMIT, META_REASONING_COMMIT
# would be on-commit edge events; STATS_UPDATED carries cumulative counts
# at 2.5s cadence which is sufficient for the consume-and-feed-engine path).
_SELF_REFLECTION_WORKER_SUBSCRIBE_TOPICS: list[str] = [
    bus.REASONING_STATS_UPDATED,       # cognitive_worker → self_reasoning.observe_chain
    bus.META_REASONING_STATS_UPDATED,  # cognitive_worker → self_reasoning.observe_meta
    bus.EXPERIENCE_STIMULUS,           # various L2 → prediction_engine.observe
    bus.DREAMING_STATE_UPDATED,        # cognitive_worker (state transitions per B2)
    bus.CGN_CROSS_INSIGHT,             # language_worker / cgn_module → coding_explorer
    bus.KERNEL_EPOCH_TICK,             # tick cadence for tick_cooldown() + prediction tick
    bus.MODULE_SHUTDOWN,               # clean shutdown
    bus.SAVE_NOW,                      # B.1 shadow_swap orchestrator (when re-enabled)
    # Session 3 (RFP_meta-reasoning_CGN_FIX.md §4.2 rows 8/9):
    # self_reflection_worker hosts PredictionEngine + SelfReasoning per
    # SPEC §9.B (§4.E shipped 2026-05-12). Receives CGN_KNOWLEDGE_REQ with
    # payload.kind ∈ {prediction, self_reasoning} from meta_service
    # resolvers and publishes CGN_KNOWLEDGE_RESP back via correlation_id.
    bus.CGN_KNOWLEDGE_REQ,
    # rFP_meta_reasoning_self_reasoning_resolver_migration / SPEC §9.B
    # + D-SPEC-70 v1.15.0 — cognitive_worker's _prim_introspect publishes
    # META_INTROSPECT_REQUEST (fire-and-forget per §8.0.ter D-SPEC-48) per
    # META INTROSPECT action; this worker runs sr.introspect(**payload),
    # persists to data/inner_memory.db.self_insights via
    # SelfReasoningEngine._persist_insight() (existing wiring — was working,
    # just never reached pre-fix), then writes the result to
    # inner_self_insight.bin SHM slot for cognitive_worker's next tick.
    # Closes F-8 fleet-wide.
    bus.META_INTROSPECT_REQUEST,
    bus.MODULE_PROBE_REQUEST,          # Phase 11 §11.I.3 probe handler
]


# Phase 11 §11.I.5 (Chunk 11N) — module-level readiness sentinel; gates
# SHM-slot heartbeat() (legacy bus heartbeat fires unconditionally for
# the boot window so guardian_HCL's stale-heartbeat detector doesn't
# kill a slow boot).
from titan_hcl.modules._heartbeat_grace import (
    boot_deadline_from_now, shm_heartbeat_allowed,
)
from titan_hcl.params import get_params

_WORKER_READY: bool = False
_BOOT_DEADLINE = None  # boot-grace deadline (monotonic); None=no grace


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
                    state_writer: Optional[object] = None) -> None:
    """Emit MODULE_HEARTBEAT to guardian_HCL with current RSS.

    Phase 11 §11.I.5: also publishes state_writer.heartbeat() on the SHM
    slot once _WORKER_READY is True. SHM writes are best-effort.
    """
    try:
        import resource
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        rss_mb = 0.0
    payload = {"alive": True, "ts": time.time(), "rss_mb": round(rss_mb, 1),
               "chunk": "B3"}
    if extra:
        payload.update(extra)
    _send_msg(send_queue, bus.MODULE_HEARTBEAT, name, "guardian", payload)
    if state_writer is not None and shm_heartbeat_allowed(_WORKER_READY, _BOOT_DEADLINE):
        try:
            state_writer.heartbeat()
        except Exception:  # noqa: BLE001 — never crash the heartbeat
            pass


def _heartbeat_loop(recv_queue, send_queue, name: str, *, flag_off: bool,
                    state_writer: Optional[object] = None) -> None:
    """Heartbeat-only loop for flag-off / defensive-noop branches.

    Used when:
      • microkernel.self_reflection_worker_enabled = false (rare — guardian
        normally skips registration entirely);
      • microkernel.l0_rust_enabled = false (legacy spirit_worker_main owns
        SelfReasoning + CodingExplorer; PredictionEngine stays in
        cognitive_worker under Track 1 drift state);
      • engine init failed (defensive — guardian doesn't restart-loop us).
    """
    last_heartbeat_ts = 0.0
    while True:
        now = time.time()
        if now - last_heartbeat_ts >= HEARTBEAT_INTERVAL_S:
            _send_heartbeat(
                send_queue, name, extra={"flag_off_noop": flag_off},
                state_writer=state_writer)
            last_heartbeat_ts = now
        try:
            msg = recv_queue.get(timeout=POLL_INTERVAL_S)
        except Empty:
            continue
        except Exception:
            continue
        _mt = msg.get("type")
        # ── Phase 11 §11.I.3 — MODULE_PROBE_REQUEST handler (flag-off branch) ──
        if _mt == bus.MODULE_PROBE_REQUEST and state_writer is not None:
            try:
                from titan_hcl.core.probe_dispatcher import (
                    handle_module_probe_request,
                )
                handle_module_probe_request(
                    msg,
                    probe_fn=None,
                    send_queue=send_queue,
                    module_name=name,
                    state_writer=state_writer,
                )
            except Exception as _probe_err:  # noqa: BLE001
                logger.warning(
                    "[SelfReflectionWorker] MODULE_PROBE_REQUEST handler failed: %s",
                    _probe_err)
            continue
        if _mt == bus.MODULE_SHUTDOWN:
            logger.info(
                "[SelfReflectionWorker] Shutdown received (flag_off branch)")
            return


def _resolve_data_path(config: dict, key: str, default: str) -> str:
    """Resolve a config-driven path against project root (so the subprocess
    writes to the canonical data/ subtree, not subprocess cwd)."""
    raw = (get_params("self_reflection") or {}).get(key, default)
    if os.path.isabs(raw):
        return raw
    project_root = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(project_root, raw)


def _init_self_reasoning(config: dict, db_path: str):
    """Construct SelfReasoningEngine. Returns instance or None on init failure."""
    try:
        from titan_hcl.logic.self_reasoning import SelfReasoningEngine
    except Exception as e:
        logger.error(
            "[SelfReflectionWorker] SelfReasoningEngine import failed: %s",
            e, exc_info=True)
        return None
    sr_cfg = (get_params("self_reasoning") or {})
    try:
        sr = SelfReasoningEngine(config=sr_cfg, db_path=db_path)
        logger.info(
            "[SelfReflectionWorker] SelfReasoningEngine booted (db=%s)",
            db_path)
        return sr
    except Exception as e:
        logger.error(
            "[SelfReflectionWorker] SelfReasoningEngine init failed: %s",
            e, exc_info=True)
        return None


def _init_coding_explorer(config: dict, db_path: str, send_queue):
    """Construct CodingExplorer (+ CodingSandboxHelper child subprocess).

    Per rFP §4.4 backwards-compat shim: read BOTH
    [self_reflection.coding_explorer] (preferred) and [cgn.coding]
    (fallback). Merge with preferred-on-conflict.
    """
    try:
        from titan_hcl.logic.coding_explorer import CodingExplorer
    except Exception as e:
        logger.error(
            "[SelfReflectionWorker] CodingExplorer import failed: %s",
            e, exc_info=True)
        return None
    # Backwards-compat shim — prefer [self_reflection.coding_explorer], fall
    # back to [cgn.coding] for keys missing in the new section.
    ce_cfg_new = (get_params("self_reflection") or {}).get(
        "coding_explorer", {}) or {}
    ce_cfg_old = (get_params("cgn") or {}).get("coding", {}) or {}
    ce_cfg = {**ce_cfg_old, **ce_cfg_new}  # new wins on conflict
    try:
        ce = CodingExplorer(send_queue=send_queue, config=ce_cfg, db_path=db_path)
        sandbox_status = "alive" if ce._sandbox is not None else "absent"
        logger.info(
            "[SelfReflectionWorker] CodingExplorer booted (db=%s, sandbox=%s)",
            db_path, sandbox_status)
        return ce
    except Exception as e:
        logger.error(
            "[SelfReflectionWorker] CodingExplorer init failed: %s — coding "
            "exercise surface disabled", e, exc_info=True)
        return None


def _init_prediction_engine(config: dict):
    """Construct PredictionEngine. Returns instance or None on init failure.

    Per rFP §2.B.6 drift correction: relocated from cognitive_worker. Data
    path stays unchanged (data/prediction/novelty_state.json per §11.H.1).
    """
    try:
        from titan_hcl.logic.prediction_engine import PredictionEngine
    except Exception as e:
        logger.error(
            "[SelfReflectionWorker] PredictionEngine import failed: %s",
            e, exc_info=True)
        return None
    pe_cfg = (config.get("prediction", {}) or {})
    error_window = pe_cfg.get("error_window")  # None → load from TOML default
    # Resolve data_dir against project root.
    data_dir_raw = pe_cfg.get("data_dir", "data/prediction")
    if os.path.isabs(data_dir_raw):
        data_dir = data_dir_raw
    else:
        project_root = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", ".."))
        data_dir = os.path.join(project_root, data_dir_raw)
    try:
        pe = PredictionEngine(
            error_window=error_window,
            data_dir=data_dir,
            load_state=True,
        )
        logger.info(
            "[SelfReflectionWorker] PredictionEngine booted "
            "(data_dir=%s, prev_state_loaded=%s)",
            data_dir,
            (pe._total_predictions > 0))
        return pe
    except Exception as e:
        logger.error(
            "[SelfReflectionWorker] PredictionEngine init failed: %s",
            e, exc_info=True)
        return None


# ── G18 state readers (NEUROMOD_STATE + EPOCH_COUNTER) ──────────────────────
# Phase 3A/3B (rFP_haov_efficacy_closure): the restored coding_explorer.explore()
# driver + self_reasoning.check_predictions() driver both need live neuromod
# levels + the consciousness epoch. Per SPEC G18 these are SHM *state* reads —
# never bus/RPC. Mirrors the meditation_worker NEUROMOD_STATE reader precedent
# (meditation_worker.py:187) and the ShmReaderBank EPOCH_COUNTER read.
# NEUROMOD_NAMES axis order matches state_registry.NEUROMOD_STATE (D-SPEC-54):
# (DA, 5HT, NE, ACh, Endorphin, GABA); field 0 = level.
_NEUROMOD_NAMES = ("DA", "5HT", "NE", "ACh", "Endorphin", "GABA")
_NM_FIELD_LEVEL = 0


def _init_shm_readers(titan_id: str) -> dict:
    """Attach NEUROMOD_STATE + EPOCH_COUNTER readers (G18 state reads).

    Best-effort, reattach-tolerant: each slot independently optional; callers
    degrade to defaults ({} neuromods / epoch 0) and retry next tick.
    """
    readers: dict = {"neuromod": None, "epoch": None}
    try:
        from titan_hcl.core.state_registry import (
            EPOCH_COUNTER, NEUROMOD_STATE, StateRegistryReader,
            ensure_shm_root,
        )
        shm_root = ensure_shm_root(titan_id)
        try:
            readers["neuromod"] = StateRegistryReader(NEUROMOD_STATE, shm_root)
        except Exception as e:
            logger.info(
                "[SelfReflectionWorker] neuromod_state.bin reader unavailable "
                "(coding/self_model drivers use neutral neuromods): %s", e)
        try:
            readers["epoch"] = StateRegistryReader(EPOCH_COUNTER, shm_root)
        except Exception as e:
            logger.info(
                "[SelfReflectionWorker] epoch_counter.bin reader unavailable "
                "(epoch-gated drivers will idle): %s", e)
    except Exception as e:
        logger.warning(
            "[SelfReflectionWorker] SHM reader init failed: %s — coding/"
            "self_model epoch+neuromod drivers degraded this run", e)
    return readers


def _read_neuromods(state_refs: dict) -> dict:
    """Read 6 modulator levels from NEUROMOD_STATE → {name: level}. {} on miss."""
    reader = (state_refs.get("_shm_readers") or {}).get("neuromod")
    if reader is None:
        return {}
    try:
        arr = reader.read()
        if arr is None or getattr(arr, "shape", None) != (6, 4):
            return {}
        return {_NEUROMOD_NAMES[i]: float(arr[i, _NM_FIELD_LEVEL])
                for i in range(6)}
    except Exception:
        return {}


def _read_epoch(state_refs: dict) -> int:
    """Read current consciousness epoch from EPOCH_COUNTER SHM. 0 on miss."""
    reader = (state_refs.get("_shm_readers") or {}).get("epoch")
    if reader is None:
        return 0
    try:
        arr = reader.read()
        if arr is None or len(arr) < 1:
            return 0
        return int(arr[0])
    except Exception:
        return 0


# ARG ORDER (template-canonical): every Guardian-spawned L2 worker entry
# follows (recv_queue, send_queue, name, config).
@with_error_envelope(module_name="self_reflection_worker", subsystem="entry", severity=_phase11_sev.FATAL)
def self_reflection_worker_main(recv_queue, send_queue, name: str,
                                  config: dict) -> None:
    """Main loop for the self_reflection_worker subprocess.

    Chunk B3 skeleton — boot section + 3-engine init + heartbeat-only main
    loop. Bus dispatcher (B5), sandbox lifecycle (B6), publishers + drift
    correction + dashboard routes (B7/B8/B9), tests (B10) land in
    subsequent commits per rFP §8 chunk sequence.
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
            topics=_SELF_REFLECTION_WORKER_SUBSCRIBE_TOPICS,
        )
    except Exception as _err:
        logger.error(
            "[SelfReflectionWorker] setup_worker_bus failed: %s — exiting",
            _err, exc_info=True)
        return

    # === BOILERPLATE: pdeathsig installation ===
    try:
        from titan_hcl.core.worker_lifecycle import install_parent_death_signal
        install_parent_death_signal()
    except Exception as _err:
        logger.debug(
            "[SelfReflectionWorker] pdeathsig install skipped: %s", _err)

    global _WORKER_READY, _BOOT_DEADLINE
    _WORKER_READY = False
    _BOOT_DEADLINE = boot_deadline_from_now()

    # Canonical titan_id resolution.
    from titan_hcl.core.state_registry import resolve_titan_id
    titan_id = (
        (get_params("info_banner") or {}).get("titan_id")
        or resolve_titan_id()
        or "T1"
    )
    boot_ts = time.time()

    # ── Phase 11 §11.I.5 (Chunk 11N) — SHM state-slot writer ──
    # Constructed BEFORE the slow flag-resolution + 3-engine init so the
    # slot publishes state="starting" immediately. Heartbeats keep
    # last_heartbeat fresh during the cold-boot window.
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
            "[SelfReflectionWorker] Phase 11 ModuleStateWriter init failed: %s",
            _sw_err)

    # === BOILERPLATE: two-flag gating ===
    microkernel_cfg = get_params("microkernel") or {}
    l0_rust = bool(microkernel_cfg.get("l0_rust_enabled", False))
    worker_enabled = bool(microkernel_cfg.get(
        "self_reflection_worker_enabled", True))
    sr_section_enabled = bool(
        (get_params("self_reflection") or {}).get("enabled", True))

    if not l0_rust:
        logger.info(
            "[SelfReflectionWorker] microkernel.l0_rust_enabled=false — "
            "legacy spirit_worker_main owns SelfReasoning + CodingExplorer; "
            "PredictionEngine stays in cognitive_worker (Track 1 drift state). "
            "Entering heartbeat-only no-op loop.")
        # Phase 11 §11.I.2 — slot=booted (no-op branch is "booted, idle")
        _WORKER_READY = True
        if _state_writer is not None:
            try:
                _state_writer.write_state("booted")
            except Exception:  # noqa: BLE001
                pass
        _heartbeat_loop(recv_queue, send_queue, name, flag_off=True,
                        state_writer=_state_writer)
        return

    if not (worker_enabled and sr_section_enabled):
        logger.info(
            "[SelfReflectionWorker] worker_enabled=%s self_reflection.enabled=%s "
            "— entering heartbeat-only no-op loop.",
            worker_enabled, sr_section_enabled)
        # Phase 11 §11.I.2 — slot=booted (no-op branch is "booted, idle")
        _WORKER_READY = True
        if _state_writer is not None:
            try:
                _state_writer.write_state("booted")
            except Exception:  # noqa: BLE001
                pass
        _heartbeat_loop(recv_queue, send_queue, name, flag_off=True,
                        state_writer=_state_writer)
        return

    logger.info(
        "[SelfReflectionWorker] Booting (titan_id=%s, l0_rust=true) — chunk B3 "
        "skeleton. Bus dispatcher / sandbox lifecycle / publishers / drift "
        "correction land in chunks B5–B8.", titan_id)

    # === MODULE-SPECIFIC: 3-engine init ===
    db_path = _resolve_data_path(config, "db_path", "data/inner_memory.db")
    self_reasoning = _init_self_reasoning(config, db_path)
    coding_explorer = _init_coding_explorer(config, db_path, send_queue)
    prediction_engine = _init_prediction_engine(config)

    # RFP_supervision_lifecycle §7.D / Phase D.1 — bus-INDEPENDENT save of the
    # PredictionEngine's learned state on any shutdown (SIGTERM/control-group/
    # PDEATHSIG). _save_state is atomic + idempotent; guarded against None init.
    from titan_hcl.core.worker_shutdown import register_shutdown_save
    register_shutdown_save(
        name,
        lambda: prediction_engine._save_state() if prediction_engine is not None else None,
    )

    if self_reasoning is None and coding_explorer is None and prediction_engine is None:
        # All 3 engines failed to init — defensive heartbeat-only.
        logger.warning(
            "[SelfReflectionWorker] all 3 engines failed to init — "
            "entering defensive heartbeat loop.")
        # Phase 11 §11.I.2 — slot=unhealthy is more honest than booted here;
        # downstream probe will report degraded. Use booted to remain
        # available for shutdown/probe RPC; the engine fault is surfaced via
        # absence of *_STATS_UPDATED publications.
        _WORKER_READY = True
        if _state_writer is not None:
            try:
                _state_writer.write_state("booted")
            except Exception:  # noqa: BLE001
                pass
        _heartbeat_loop(recv_queue, send_queue, name, flag_off=True,
                        state_writer=_state_writer)
        return

    # state_refs dict — template-canonical shape for B5 dispatcher + B7
    # publishers + B6 sandbox lifecycle.
    state_refs: dict = {
        "self_reasoning": self_reasoning,
        "coding_explorer": coding_explorer,
        "prediction_engine": prediction_engine,
        "_db_path": db_path,
        # Dream-cycle state cache (B5 DREAMING_STATE_UPDATED handler updates).
        "_last_dream_state": "awake",
        "_last_dream_profile": None,
        # Sandbox lifecycle state (B6 health check + orphan detection).
        "_sandbox_last_check_ts": time.time(),
        "_sandbox_last_orphan_check_ts": time.time(),
        "_sandbox_disabled": coding_explorer is None,
        # Publisher cadence trackers (B7).
        "_publisher_ts": time.time(),
        "_coding_publisher_ts": time.time(),
        "_prediction_publisher_ts": time.time(),
        # Phase 3A/3B (rFP_haov_efficacy_closure) — G18 SHM readers for the
        # restored coding_explorer.explore() + self_reasoning.check_predictions()
        # drivers, + epoch tracker for the 100-epoch self-prediction cadence.
        "_shm_readers": _init_shm_readers(titan_id),
        "_last_self_pred_check_epoch": 0,
    }

    # Cadences (from [self_reflection] params with defaults).
    sr_cfg = (get_params("self_reflection") or {})
    publisher_cadence_s = float(sr_cfg.get(
        "publisher_cadence_s", PUBLISHER_DEFAULT_S))
    coding_publisher_cadence_s = float(sr_cfg.get(
        "coding_explorer_publisher_cadence_s", CODING_EXPLORER_PUBLISHER_S))
    prediction_publisher_cadence_s = float(sr_cfg.get(
        "prediction_publisher_cadence_s", PUBLISHER_DEFAULT_S))
    sandbox_health_check_s = float(sr_cfg.get(
        "sandbox_health_check_s", SANDBOX_HEALTH_CHECK_S))
    orphan_check_s = float(sr_cfg.get("orphan_check_s", ORPHAN_CHECK_S))

    # === MODULE-SPECIFIC: launch cadence-driven publisher daemon (chunk B7) ===
    # Daemon thread fires every min(publisher_cadence_s, coding_publisher_cadence_s)/2
    # and emits *_STATS_UPDATED + PREDICTION_GENERATED (when novelty detected).
    _publisher_stop_event = threading.Event()
    state_refs["_publisher_stop_event"] = _publisher_stop_event
    _publisher_thread = threading.Thread(
        target=_publisher_loop,
        args=(state_refs, send_queue, name, titan_id, _publisher_stop_event,
              publisher_cadence_s, coding_publisher_cadence_s,
              prediction_publisher_cadence_s),
        name=f"self_reflection_publisher_{titan_id}",
        daemon=True,
    )
    _publisher_thread.start()

    # ── Phase 11 §11.I.2 — slot transition: starting → booted ──
    # (legacy boot-signal bus emit deleted per locked D2 / no-shim policy)
    sandbox_status = ("active" if coding_explorer is not None
                      and coding_explorer._sandbox is not None else "absent")
    _WORKER_READY = True
    if _state_writer is not None:
        try:
            _state_writer.write_state("booted")
        except Exception as _swb_err:  # noqa: BLE001
            logger.warning(
                "[SelfReflectionWorker] Phase 11 write_state(booted) failed: %s",
                _swb_err)
    logger.info(
        "[SelfReflectionWorker] online — SelfReasoning=%s + CodingExplorer=%s "
        "+ PredictionEngine=%s booted (sandbox=%s)",
        self_reasoning is not None, coding_explorer is not None,
        prediction_engine is not None, sandbox_status)

    # === BOILERPLATE: main loop — chunk B3 = heartbeat-only ===
    # B5 wires the actual bus message dispatcher (with DREAMING_STATE_UPDATED
    # dream_start / dream_end handlers per rFP §2.B.5); B6 wires the sandbox
    # lifecycle (30s health + 60s orphan); B7 wires the cadence-driven +
    # on-event publishers. Until then, drain recv_queue and act only on
    # MODULE_SHUTDOWN. The skeleton sets SHM slot=booted + heartbeats so
    # guardian classifies us as online during the soak between B3 and B5.
    last_heartbeat_ts = 0.0
    last_prediction_checkpoint = time.time()  # first checkpoint ~5min after boot
    _prediction_ckpt_thread = [None]          # single-slot non-blocking writer
    while True:
        now = time.time()

        if now - last_heartbeat_ts >= HEARTBEAT_INTERVAL_S:
            _send_heartbeat(send_queue, name, state_writer=_state_writer)
            last_heartbeat_ts = now

        # ── Periodic disk checkpoint (survives ANY crash) — NON-BLOCKING ──
        # PredictionEngine self-persists every _save_every predictions, but that
        # is COUNT-based: when predictions are sparse, novelty_state.json can go
        # stale for a long time (observed ~16d). This adds a TIME-based 5-min
        # flush via the engine's atomic _save_state(), offloaded to a single-slot
        # daemon thread so the disk write never blocks the heartbeat.
        if (prediction_engine is not None
                and now - last_prediction_checkpoint
                > _PREDICTION_CHECKPOINT_INTERVAL_S):
            last_prediction_checkpoint = now
            if (_prediction_ckpt_thread[0] is None
                    or not _prediction_ckpt_thread[0].is_alive()):
                def _do_pred_ckpt(_pe=prediction_engine):
                    try:
                        _pe._save_state()
                    except Exception as _ckpt_err:  # noqa: BLE001
                        logger.warning(
                            "[SelfReflectionWorker] periodic checkpoint "
                            "failed: %s", _ckpt_err)
                _prediction_ckpt_thread[0] = threading.Thread(
                    target=_do_pred_ckpt, daemon=True,
                    name="self_reflection-checkpoint")
                _prediction_ckpt_thread[0].start()

        # Sandbox subprocess lifecycle polled cadences (chunk B6 per
        # rFP §2.B.6). CodingSandboxHelper.status() is a lightweight PATH
        # check (no subprocess call per coding_sandbox.py:214-226 docstring)
        # so polling from the main loop is fine — no separate daemon thread
        # needed for these low-cadence (30s + 60s) checks.
        if not state_refs.get("_sandbox_disabled", False):
            if now - state_refs["_sandbox_last_check_ts"] >= sandbox_health_check_s:
                _check_sandbox_health(state_refs)
                state_refs["_sandbox_last_check_ts"] = now
            if now - state_refs["_sandbox_last_orphan_check_ts"] >= orphan_check_s:
                _check_sandbox_orphans(state_refs)
                state_refs["_sandbox_last_orphan_check_ts"] = now

        try:
            msg = recv_queue.get(timeout=POLL_INTERVAL_S)
        except Empty:
            continue
        except Exception:
            continue

        msg_type = msg.get("type")

        # ── Phase 11 §11.I.3 — MODULE_PROBE_REQUEST handler ──
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
                    "[SelfReflectionWorker] MODULE_PROBE_REQUEST handler failed: %s",
                    _probe_err)
            continue

        # B.2.1 supervision-transfer dispatch.
        try:
            from titan_hcl.core import worker_swap_handler as _swap
            if _swap.maybe_dispatch_swap_msg(msg):
                continue
        except Exception:
            pass

        if msg_type == bus.MODULE_SHUTDOWN:
            logger.info(
                "[SelfReflectionWorker] Shutdown received — stopping publisher "
                "daemon + saving engine state + terminating sandbox + exiting")
            stop_evt = state_refs.get("_publisher_stop_event")
            if stop_evt is not None:
                stop_evt.set()
            _save_state_on_shutdown(state_refs)
            _terminate_sandbox(state_refs)
            return

        # Bus dispatcher (chunk B5) — one branch per subscribed topic. Each
        # handler is best-effort (top-level try/except catches; handler internals
        # log debug on sub-step failure, never raise).
        try:
            _dispatch_msg(msg, msg_type, state_refs, send_queue, name)
        except Exception as e:
            logger.warning(
                "[SelfReflectionWorker] dispatch failed for %s: %s",
                msg_type, e, exc_info=False)


def _save_state_on_shutdown(state_refs: dict) -> None:
    """Persist all 3 engine states on shutdown — best-effort."""
    pe = state_refs.get("prediction_engine")
    if pe is not None:
        try:
            # PredictionEngine's persist method is _save_state (atomic
            # tmp+rename). The prior guard checked hasattr(pe, "save_state")
            # — a method that does NOT exist — so the graceful-shutdown flush
            # silently never ran (novelty_state.json stale on every clean stop).
            if hasattr(pe, "_save_state"):
                pe._save_state()
                logger.info(
                    "[SelfReflectionWorker] prediction_engine._save_state() ok")
        except Exception as e:
            logger.warning(
                "[SelfReflectionWorker] prediction_engine save failed: %s", e)
    # SelfReasoningEngine + CodingExplorer persist via the shared
    # data/inner_memory.db (DB checkpoint on connection close per SPEC
    # §11.H.3). Connection close happens via the standard lifecycle.


def _terminate_sandbox(state_refs: dict) -> None:
    """Explicit sandbox subprocess termination on MODULE_SHUTDOWN.

    Per rFP §2.B.6: terminate(timeout=5s) → terminate() → SIGKILL fallback.
    Wait for child reap. PR_SET_PDEATHSIG cascade catches anything missed.
    """
    ce = state_refs.get("coding_explorer")
    if ce is None or getattr(ce, "_sandbox", None) is None:
        return
    sandbox = ce._sandbox
    try:
        if hasattr(sandbox, "terminate"):
            sandbox.terminate()
            logger.info("[SelfReflectionWorker] sandbox.terminate() ok")
        elif hasattr(sandbox, "close"):
            sandbox.close()
            logger.info("[SelfReflectionWorker] sandbox.close() ok")
    except Exception as e:
        logger.warning(
            "[SelfReflectionWorker] sandbox termination failed: %s", e)


# ── Bus dispatcher (chunk B5) ────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────
# Session 3 CGN_KNOWLEDGE_REQ response builders
# RFP_meta-reasoning_CGN_FIX.md §4.2 rows 8/9. Produce real output dicts
# from the engines self_reflection_worker hosts post Track 2 v1.2.1 commit
# B8 (PredictionEngine + SelfReasoning canonical home).
# ──────────────────────────────────────────────────────────────────────


def _build_prediction_response(prediction_engine, name: str,
                               payload: dict) -> dict:
    """Build a prediction-kind response. The `name` is the post-dot
    recruiter portion (typically "default"). Returns PredictionEngine's
    novelty surrogate + recent prediction stats so the consumer's outcome
    computer can compare predicted-vs-actual deltas.
    """
    if prediction_engine is None:
        return {"engine": "unavailable", "name": name,
                "suggested_action": "fallback_static"}
    try:
        stats = prediction_engine.get_stats() or {}
    except Exception:
        stats = {}
    try:
        novelty = float(prediction_engine.get_novelty_signal()) \
            if hasattr(prediction_engine, "get_novelty_signal") else 0.0
    except Exception:
        novelty = 0.0
    # Fix4 — suggested_action from novelty + surprise rate
    total_p = int(stats.get("total_predictions", 0))
    total_s = int(stats.get("total_surprises", 0))
    surprise_rate = total_s / max(1, total_p)
    if novelty > 0.5:
        sugg = "explore_novel_trajectory"
    elif surprise_rate > 0.3:
        sugg = "recalibrate_predictions"
    elif novelty < 0.1:
        sugg = "exploit_known_pattern"
    else:
        sugg = "continue_current_trajectory"
    return {
        "engine": "prediction",
        "name": name,
        "novelty_signal": novelty,
        "total_predictions": total_p,
        "total_surprises": total_s,
        "avg_error": float(stats.get("avg_error", 0.0)),
        "question_type": payload.get("question_type", ""),
        "consumer_id": payload.get("consumer_id", ""),
        "suggested_action": sugg,
    }


def _build_self_reasoning_response(self_reasoning, name: str,
                                   payload: dict) -> dict:
    """Build a self_reasoning-kind response. The `name` is the post-dot
    recruiter portion ("predict", "meta_audit", etc.). Returns the engine's
    introspection state + recent observation counters so the consumer's
    outcome computer has differentiated input.
    """
    if self_reasoning is None:
        return {"engine": "unavailable", "name": name,
                "suggested_action": "fallback_static"}
    try:
        stats = self_reasoning.get_stats() if hasattr(self_reasoning, "get_stats") \
            else {}
    except Exception:
        stats = {}
    # Fix4 — suggested_action from introspection depth
    depth = float(stats.get("introspection_depth", 0.0))
    if name == "meta_audit":
        sugg = "deep_audit" if depth > 0.5 else "shallow_audit"
    elif depth < 0.3:
        sugg = "deepen_introspection"
    elif depth > 0.7:
        sugg = "synthesize_self_insight"
    else:
        sugg = "continue_observation"
    return {
        "engine": "self_reasoning",
        "name": name,
        "introspection_depth": depth,
        "observations_total": int(stats.get("observations_total", 0)),
        "audit_count": int(stats.get("audit_count", 0)),
        "last_audit_ts": float(stats.get("last_audit_ts", 0.0)),
        "question_type": payload.get("question_type", ""),
        "consumer_id": payload.get("consumer_id", ""),
        "suggested_action": sugg,
    }


def _dispatch_msg(msg: dict, msg_type: str, state_refs: dict,
                  send_queue, name: str) -> None:
    """Route an inbound bus message to its handler per rFP §2.B.3.

    The rFP names some handler-callees that don't exist as engine methods
    today (observe_chain, observe_meta, note_chain) — those are aspirational
    granular hooks that would need engine extensions. This dispatcher wires
    the methods that DO exist on each engine today:

      self_reasoning.tick_cooldown()        — per-epoch refractory tick
      self_reasoning.consolidate_training() — dream-end consolidation
      self_reasoning._last_dream_profile    — dream profile cache (attr write)
      coding_explorer.tick_cooldown()       — per-epoch refractory tick
      coding_explorer.on_dream_start()      — dream-start callback
      prediction_engine.predict_next(...)   — input-driven (handled in B7 publisher path)
      prediction_engine.compute_error(...)  — input-driven (handled in B7 publisher path)

    For events that don't have a clean engine method today (REASONING_STATS_UPDATED,
    META_REASONING_STATS_UPDATED, CGN_CROSS_INSIGHT), we cache the payload in
    state_refs so future engine extensions (or the B7 publishers) can consume
    them. This honours the rFP intent without inventing methods that don't exist.
    """
    payload = msg.get("payload") or {}

    if msg_type == bus.REASONING_STATS_UPDATED:
        # Cache cognitive_worker's reasoning stats (counters, last_chain_at,
        # commit_rate). Future granular REASONING_COMMIT event would carry
        # chain-content; this proxy carries cumulative stats sufficient for
        # self_reasoning observability + dream-end consolidation context.
        state_refs["_last_reasoning_stats"] = dict(payload) if isinstance(payload, dict) else None
        # D5 (v1.2.1): feed observation into engines per rFP §2.B.3
        # contract closure. observe_chain / note_chain are additive
        # surfaces (counter + last-ts + cached payload) — they do NOT
        # require chain body content, just the stats summary.
        sr = state_refs.get("self_reasoning")
        if sr is not None and hasattr(sr, "observe_chain"):
            try:
                sr.observe_chain(payload or {})
            except Exception as e:
                logger.debug("[SelfReflectionWorker] sr.observe_chain failed: %s", e)
        ce = state_refs.get("coding_explorer")
        if ce is not None and hasattr(ce, "note_chain"):
            try:
                ce.note_chain(payload or {})
            except Exception as e:
                logger.debug("[SelfReflectionWorker] ce.note_chain failed: %s", e)
        return

    if msg_type == bus.META_REASONING_STATS_UPDATED:
        state_refs["_last_meta_reasoning_stats"] = dict(payload) if isinstance(payload, dict) else None
        # D5 (v1.2.1): feed meta-observation into self_reasoning per
        # rFP §2.B.3 contract closure.
        sr = state_refs.get("self_reasoning")
        if sr is not None and hasattr(sr, "observe_meta"):
            try:
                sr.observe_meta(payload or {})
            except Exception as e:
                logger.debug("[SelfReflectionWorker] sr.observe_meta failed: %s", e)
        return

    if msg_type == bus.EXPERIENCE_STIMULUS:
        _handle_experience_stimulus(payload, state_refs)
        return

    if msg_type == bus.DREAMING_STATE_UPDATED:
        _handle_dreaming_state_updated(payload, state_refs)
        return

    if msg_type == bus.CGN_CROSS_INSIGHT:
        _handle_cgn_cross_insight(payload, state_refs)
        return

    if msg_type == bus.KERNEL_EPOCH_TICK:
        _handle_epoch_tick(payload, state_refs, send_queue, name)
        return

    if msg_type == bus.SAVE_NOW:
        _save_state_on_shutdown(state_refs)
        return

    if msg_type == bus.META_INTROSPECT_REQUEST:
        # rFP_meta_reasoning_self_reasoning_resolver_migration / SPEC §9.B
        # + D-SPEC-70 v1.15.0 — fire-and-forget trigger from
        # cognitive_worker._prim_introspect. Runs SelfReasoningEngine.introspect
        # with the full ctx echoed from cognitive_worker; persists insight to
        # data/inner_memory.db.self_insights via _persist_insight() (existing
        # wiring); writes result to inner_self_insight.bin SHM slot for
        # cognitive_worker's next pre-warmed-cache read per G20.
        # No response publish (cognitive_worker's tick is fire-and-forget;
        # result flows back via SHM, not bus).
        _handle_meta_introspect_request(payload, state_refs, name, send_queue)
        return

    if msg_type == bus.CGN_KNOWLEDGE_REQ:
        # Session 3 live-dispatch handler — RFP_meta-reasoning_CGN_FIX.md
        # §4.2 rows 8/9. Receives requests from meta_service resolvers
        # tagged payload.kind ∈ {prediction, self_reasoning}. Publishes
        # CGN_KNOWLEDGE_RESP back with the same correlation_id.
        # SPEC anchors: §8.2 D-SPEC-52 targeted routing; §8.0.ter
        # D-SPEC-48 non-blocking publish; Preamble G19 bounded handler.
        _ckr_corr = payload.get("correlation_id")
        _ckr_kind = payload.get("kind", "")
        _ckr_name = payload.get("name", "")
        _ckr_src = msg.get("src", "meta_service")
        # Skip non-Session-3 envelopes — no correlation_id or wrong kind.
        if not _ckr_corr or _ckr_kind not in ("prediction", "self_reasoning"):
            return
        _ckr_output: dict
        _ckr_failure = None
        try:
            if _ckr_kind == "prediction":
                _ckr_output = _build_prediction_response(
                    state_refs.get("prediction_engine"), _ckr_name, payload)
            elif _ckr_kind == "self_reasoning":
                _ckr_output = _build_self_reasoning_response(
                    state_refs.get("self_reasoning"), _ckr_name, payload)
            else:
                _ckr_output = {}
                _ckr_failure = "unknown_kind"
        except Exception as _ckr_err:
            logger.warning(
                "[SelfReflectionWorker] CGN_KNOWLEDGE_REQ kind=%s name=%s "
                "handler error: %s", _ckr_kind, _ckr_name, _ckr_err)
            _ckr_output = {"error": str(_ckr_err)}
            _ckr_failure = "handler_error"

        _resp_payload = {
            "correlation_id": _ckr_corr,
            "kind": _ckr_kind,
            "name": _ckr_name,
            "output": _ckr_output,
            "ts": time.time(),
        }
        if _ckr_failure:
            _resp_payload["failure"] = _ckr_failure
        _send_msg(send_queue, bus.CGN_KNOWLEDGE_RESP, name,
                  _ckr_src, _resp_payload)
        return

    # Unknown types — broker filter prevents these but defense in depth.


def _get_or_init_inner_self_insight_writer(state_refs: dict):
    """Lazy-init the StateRegistryWriter for inner_self_insight.bin.

    Mirrors the trajectory_state.bin precedent at
    cognitive_worker.py:2451-2456 — cache the writer in state_refs,
    log first-attach + every-Nth-error per directive_error_visibility.

    Returns None on init failure (defensive — handler still runs
    sr.introspect + _persist_insight even if SHM write fails; next
    successful write warms the cache).
    """
    if "_inner_self_insight_writer" in state_refs:
        return state_refs["_inner_self_insight_writer"]
    try:
        from titan_hcl.core.state_registry import (
            INNER_SELF_INSIGHT, StateRegistryWriter, resolve_shm_root,
            resolve_titan_id,
        )
        titan_id = resolve_titan_id()
        writer = StateRegistryWriter(
            INNER_SELF_INSIGHT, resolve_shm_root(titan_id))
        state_refs["_inner_self_insight_writer"] = writer
        state_refs["_inner_self_insight_write_count"] = 0
        state_refs["_inner_self_insight_err_count"] = 0
        logger.info(
            "[SelfReflectionWorker] inner_self_insight.bin writer attached "
            "(titan_id=%s) — D-SPEC-70 v1.15.0 / closes F-8", titan_id)
        return writer
    except Exception as e:
        logger.warning(
            "[SelfReflectionWorker] inner_self_insight.bin writer init "
            "failed: %s — SHM write disabled for this run", e, exc_info=True)
        state_refs["_inner_self_insight_writer"] = None
        return None


def _handle_meta_introspect_request(payload: dict, state_refs: dict,
                                      name: str, send_queue=None) -> None:
    """rFP_meta_reasoning_self_reasoning_resolver_migration / SPEC §9.B
    + D-SPEC-70 v1.15.0 — handler for META_INTROSPECT_REQUEST from
    cognitive_worker._prim_introspect.

    Flow:
      1. Extract introspect kwargs from payload (sub_mode, epoch,
         neuromods, msl_data, reasoning_stats, language_stats,
         coordinator_data, state_132d).
      2. Run state_refs["self_reasoning"].introspect(**kwargs). This
         persists to data/inner_memory.db.self_insights via the engine's
         existing _persist_insight() wiring (unchanged — that wiring
         was correct, just never reached in Phase C pre-fix because
         meta_reasoning._self_reasoning was always None).
      3. Build SHM payload = result + payload-echo metadata
         (sub_mode, effective_sub_mode, neuromods, epoch, ts,
         cold_start=False).
      4. Msgpack-encode + write to inner_self_insight.bin SHM slot.

    No response publish — cognitive_worker's tick is fire-and-forget;
    result flows back via SHM read on the next tick (G20 pre-warmed
    cache).

    Closes F-8 fleet-wide.
    """
    sr = state_refs.get("self_reasoning")
    if sr is None:
        # SelfReasoningEngine failed to init at boot; nothing to do.
        # Cognitive_worker keeps hitting the cold-start placeholder path
        # in _prim_introspect — no regression vs broken state.
        logger.debug(
            "[SelfReflectionWorker] META_INTROSPECT_REQUEST received but "
            "self_reasoning is None (engine init failed at boot) — skip")
        return

    sub_mode = payload.get("sub_mode", "state_audit")
    epoch = int(payload.get("epoch", 0))
    neuromods = payload.get("neuromods") or {}
    state_132d = payload.get("state_132d") or []

    # Neuromod-coupled mode override — matches the pre-fix in-process
    # behavior at meta_reasoning.py:4217-4220 ("sub_mode='state_audit' →
    # sr.select_introspection_mode(neuromods) may upgrade to a more
    # specific mode"). Keep the override here so cognitive_worker's
    # next-tick read sees the upgraded mode label.
    effective_sub = sub_mode
    if sub_mode == "state_audit":
        try:
            if hasattr(sr, "select_introspection_mode"):
                suggested = sr.select_introspection_mode(neuromods)
                if suggested and suggested != "state_audit":
                    effective_sub = suggested
        except Exception as e:
            logger.debug(
                "[SelfReflectionWorker] select_introspection_mode failed: "
                "%s — falling back to sub_mode=%s", e, sub_mode)

    # Run introspect — this is the call that was dead pre-fix.
    try:
        result = sr.introspect(
            sub_mode=effective_sub,
            epoch=epoch,
            neuromods=neuromods,
            msl_data=payload.get("msl_data"),
            reasoning_stats=payload.get("reasoning_stats"),
            language_stats=payload.get("language_stats"),
            coordinator_data=payload.get("coordinator_data"),
            state_132d=state_132d,
        )
    except Exception as e:
        logger.warning(
            "[SelfReflectionWorker] sr.introspect raised: %s "
            "(sub_mode=%s effective=%s epoch=%d) — SHM slot NOT updated",
            e, sub_mode, effective_sub, epoch, exc_info=True)
        return

    # Build SHM payload — echo trigger metadata + introspect output.
    # cognitive_worker's _prim_introspect reads this dict directly on the
    # next tick and populates META-CGN Producer #13/#14 queues from
    # confidence + chi_coh.
    if not isinstance(result, dict):
        logger.warning(
            "[SelfReflectionWorker] sr.introspect returned non-dict "
            "(%s) — SHM slot NOT updated", type(result).__name__)
        return

    # ── 3A Layer-A: gap-driven coding exploration (rFP §3 Phase 3A) ──
    # Faithful to the legacy spirit_worker path: after a coherence-check
    # introspect populates _coherence_gaps, derive exploration triggers and
    # drive a coding exercise on the highest-urgency one. get_exploration_triggers
    # self-gates (returns [] when no gaps) → no-op outside coherence gaps. The
    # always-on Layer-B time-fallback lives in _handle_epoch_tick.
    ce = state_refs.get("coding_explorer")
    if (send_queue is not None and ce is not None and ce.can_explore
            and hasattr(sr, "get_exploration_triggers")):
        try:
            _triggers = sr.get_exploration_triggers()
        except Exception as _gt_err:
            logger.debug(
                "[SelfReflectionWorker] get_exploration_triggers failed: %s",
                _gt_err)
            _triggers = []
        if _triggers:
            _drive_coding_exploration(
                state_refs, send_queue, name,
                trigger=_triggers[0],
                epoch=epoch,
                neuromods=neuromods or _read_neuromods(state_refs),
                context=_build_coding_explore_context(
                    state_refs, sr, payload.get("msl_data")))

    shm_payload = {
        "primitive": str(result.get("primitive", "INTROSPECT")),
        "sub_mode": sub_mode,
        "effective_sub_mode": effective_sub,
        "confidence": float(result.get("confidence", 0.0)),
        "mode_trigger": str(result.get("mode_trigger", "default")),
        "inner_avg": float(result.get("inner_avg", 0.0)),
        "outer_avg": float(result.get("outer_avg", 0.0)),
        "neuromods": {str(k): float(v) for k, v in neuromods.items()},
        # chi_coh — present only on coherence_check sub-mode, per
        # meta_reasoning Producer #14 contract. Read from msl_data.chi_coherence
        # echoed in the trigger payload (the same source the in-process
        # code path used pre-D8-3).
        "chi_coh": _extract_chi_coh(payload),
        "epoch": epoch,
        "ts": time.time(),
        "cold_start": False,
    }

    # Optional fields from introspect result (mode-specific — kept under
    # cap, dropped silently if oversized).
    for k in ("alignment_score", "bond_health", "dialogue_summary", "note"):
        if k in result:
            shm_payload[k] = result[k]

    # Write SHM slot.
    writer = _get_or_init_inner_self_insight_writer(state_refs)
    if writer is None:
        return  # init failed; engine still ran + persisted.

    try:
        import msgpack
        raw = msgpack.packb(shm_payload, use_bin_type=True)
        writer.write_variable(raw)
        state_refs["_inner_self_insight_write_count"] = \
            state_refs.get("_inner_self_insight_write_count", 0) + 1
    except ValueError as e:
        # Payload overflow (>1024B) — drop optional fields + retry once.
        logger.warning(
            "[SelfReflectionWorker] inner_self_insight payload overflow "
            "(%s) — retrying without optional fields", e)
        for k in ("alignment_score", "bond_health", "dialogue_summary", "note"):
            shm_payload.pop(k, None)
        try:
            import msgpack
            raw = msgpack.packb(shm_payload, use_bin_type=True)
            writer.write_variable(raw)
            state_refs["_inner_self_insight_write_count"] = \
                state_refs.get("_inner_self_insight_write_count", 0) + 1
        except Exception as e2:
            state_refs["_inner_self_insight_err_count"] = \
                state_refs.get("_inner_self_insight_err_count", 0) + 1
            if state_refs["_inner_self_insight_err_count"] % 100 == 1:
                logger.warning(
                    "[SelfReflectionWorker] inner_self_insight write "
                    "FAILED after retry: %s (err_count=%d)",
                    e2, state_refs["_inner_self_insight_err_count"])
    except Exception as e:
        state_refs["_inner_self_insight_err_count"] = \
            state_refs.get("_inner_self_insight_err_count", 0) + 1
        if state_refs["_inner_self_insight_err_count"] % 100 == 1:
            logger.warning(
                "[SelfReflectionWorker] inner_self_insight write failed: "
                "%s (err_count=%d)", e,
                state_refs["_inner_self_insight_err_count"])


def _extract_chi_coh(payload: dict):
    """Pull chi_coherence from echoed msl_data for SHM payload (Producer #14
    coherence_gain consumer). Returns None if msl_data missing the key —
    matches the pre-D8-3 _prim_introspect behavior at meta_reasoning.py:4262.
    """
    try:
        msl_data = payload.get("msl_data") or {}
        chi = msl_data.get("chi_coherence")
        if chi is None:
            return None
        return float(chi)
    except (TypeError, ValueError):
        return None


def _handle_experience_stimulus(payload: dict, state_refs: dict) -> None:
    """Cache the latest experience stimulus for prediction_engine to consume
    on the next KERNEL_EPOCH_TICK + publisher cycle. PredictionEngine's API
    is predict_next(state, trajectory) + compute_error(actual) — both need
    real state vectors which the periodic publisher (B7) will pair from this
    cache + the most-recent stimulus.
    """
    state_refs["_last_experience_stimulus"] = dict(payload) if isinstance(payload, dict) else {}


def _handle_dreaming_state_updated(payload: dict, state_refs: dict) -> None:
    """Per rFP §2.B.5 + chunk B2 cognitive_worker DREAMING_STATE_UPDATED
    state-transition publish:
      • state="dream_start" → coding_explorer.on_dream_start()
      • state="dream_end"   → self_reasoning.consolidate_training() +
                              _last_dream_profile = payload.dream_profile

    Cache the current state regardless so B7 publishers can surface it.
    """
    state = payload.get("state", "")
    state_refs["_last_dream_state"] = state

    if state == "dream_start":
        ce = state_refs.get("coding_explorer")
        if ce is not None and hasattr(ce, "on_dream_start"):
            try:
                ce.on_dream_start()
                logger.info(
                    "[SelfReflectionWorker] coding_explorer.on_dream_start() ok")
            except Exception as e:
                logger.warning(
                    "[SelfReflectionWorker] coding_explorer.on_dream_start() "
                    "failed: %s", e)
        return

    if state == "dream_end":
        sr = state_refs.get("self_reasoning")
        if sr is not None:
            # Set _last_dream_profile before consolidate_training so the engine
            # can reference it during consolidation.
            profile = payload.get("dream_profile")
            if profile is not None:
                try:
                    sr._last_dream_profile = profile
                except Exception:
                    pass  # attribute may be slotted; non-fatal
            if hasattr(sr, "consolidate_training"):
                try:
                    consolidate_result = sr.consolidate_training()
                    logger.info(
                        "[SelfReflectionWorker] self_reasoning."
                        "consolidate_training() ok")
                    # L2 / Sub-phase E (housekeeping 2026-05-26):
                    # push the two INTROSPECT signals to meta-reasoning's
                    # subsystem cache so the INTROSPECT compound reward
                    # at meta_reasoning_rewards.py:305 stops reading the
                    # stub 0.0 default (line 317 comment retired).
                    introspect_signals = (
                        consolidate_result.get("introspect_signals")
                        if isinstance(consolidate_result, dict) else None
                    )
                    if introspect_signals:
                        meta_engine = state_refs.get("meta_engine")
                        if meta_engine is not None and hasattr(
                                meta_engine, "update_subsystem_cache"):
                            try:
                                meta_engine.update_subsystem_cache(
                                    self_prediction_accuracy=introspect_signals.get(
                                        "self_prediction_accuracy"),
                                    self_profile_divergence=introspect_signals.get(
                                        "self_profile_divergence"),
                                )
                                logger.info(
                                    "[SelfReflectionWorker] INTROSPECT signals "
                                    "pushed to meta_engine: accuracy=%.3f "
                                    "divergence=%.3f",
                                    introspect_signals.get(
                                        "self_prediction_accuracy", 0.0),
                                    introspect_signals.get(
                                        "self_profile_divergence", 0.0))
                            except Exception as e:
                                logger.warning(
                                    "[SelfReflectionWorker] meta_engine."
                                    "update_subsystem_cache(INTROSPECT) "
                                    "failed: %s", e)
                except Exception as e:
                    logger.warning(
                        "[SelfReflectionWorker] self_reasoning."
                        "consolidate_training() failed: %s", e)


def _handle_cgn_cross_insight(payload: dict, state_refs: dict) -> None:
    """Feed cross-consumer insight into coding_explorer's CGN client per
    rFP §2.B.3. The engine may have a `_cgn_client` attribute with a
    note_incoming_cross_insight method — defensively check.
    """
    ce = state_refs.get("coding_explorer")
    if ce is None:
        return
    cgn_client = getattr(ce, "_cgn_client", None)
    if cgn_client is None:
        return
    note_fn = getattr(cgn_client, "note_incoming_cross_insight", None)
    if callable(note_fn):
        try:
            note_fn(payload)
        except Exception as e:
            logger.debug(
                "[SelfReflectionWorker] note_incoming_cross_insight failed: %s",
                e)


def _build_coding_explore_context(state_refs: dict, sr,
                                  msl_data: dict = None) -> dict:
    """Best-effort context for coding_explorer.explore() action selection.

    Phase-C-faithful reconstruction of the legacy spirit_worker context build:
    the in-process objects (msl / language_stats / reasoning_engine) no longer
    live here, so we assemble from what the worker already caches
    (REASONING_STATS_UPDATED) + the introspect payload's msl_data + the
    in-process prediction EMA. explore() degrades gracefully on missing keys.
    """
    ctx: dict = {}
    msl_data = msl_data or {}
    for _k in ("chi_coherence", "i_confidence"):
        _v = msl_data.get(_k)
        if _v is not None:
            try:
                ctx[_k] = float(_v)
            except (TypeError, ValueError):
                pass
    rstats = state_refs.get("_last_reasoning_stats") or {}
    if rstats:
        try:
            ctx["total_chains"] = int(rstats.get("total_chains", 0) or 0)
            ctx["commit_rate"] = float(rstats.get("commit_rate", 0.0) or 0.0)
        except (TypeError, ValueError):
            pass
    if sr is not None:
        try:
            ctx["prediction_accuracy"] = float(sr._prediction_accuracy_ema)
        except Exception:
            pass
    return ctx


def _drive_coding_exploration(state_refs: dict, send_queue, name: str, *,
                              trigger: dict, epoch: int, neuromods: dict,
                              context: dict) -> None:
    """Run ONE coding_explorer.explore() exercise — the restored dropped loop.

    rFP_haov_efficacy_closure Phase 3A / BUG-CODING-EXPLORER-EXPLORE-CALLER-
    DROPPED: commit 72f95a6b deleted both explore() call sites with spirit_worker.
    explore() internally records the result → emits the consumer="coding"
    CGN_TRANSITION (the HAOV `coding` consumer feed) + META-CGN producers #5/#6
    (problem_solved / test_failed). This is what wakes the dormant `coding`
    HAOV consumer.

    Faithful-restore scope (Maker 2026-05-30, option A): CGN/HAOV/META-CGN only;
    NO synthesis procedural-fork logging (deferred to a separate rFP). The legacy
    Layer-A neuromod nudge (apply_external_nudge) is intentionally NOT recreated:
    neuromods are SHM-owned by neuromod_worker under Phase C, so a cross-worker
    nudge would need its own bus command + greenlight — out of scope here.
    """
    ce = state_refs.get("coding_explorer")
    if ce is None or not ce.can_explore:
        return
    try:
        result = ce.explore(trigger=trigger, epoch=epoch,
                            neuromods=neuromods, context=context)
        if result is not None:
            logger.info(
                "[SelfReflectionWorker→CodingExplore] %s/%s → %s reward=%.3f "
                "tests=%d/%d (trigger=%s)",
                result.action, result.concept,
                "PASS" if result.sandbox_success else "FAIL",
                result.reward, result.tests_passed, result.tests_total,
                trigger.get("gap_metric", trigger.get("action", "?")))
    except Exception as e:
        logger.warning(
            "[SelfReflectionWorker] coding_explorer.explore() failed: %s", e)


def _drive_self_prediction_check(state_refs: dict, send_queue, name: str,
                                 epoch: int, neuromods: dict,
                                 msl_data: dict = None,
                                 language: dict = None) -> None:
    """Run self_reasoning.check_predictions() + emit self_model CGN_TRANSITIONs.

    Restores dropped loop #2 (rFP_haov_efficacy_closure Phase 3B; commit
    72f95a6b removed the caller). check_predictions is the ONLY updater of
    `_prediction_accuracy_ema` (frozen at 0.5 since retirement) and the resolver
    of SelfPredictions created by introspect→_make_prediction.

    For each resolved prediction we emit the legacy ground-truth effects that
    still have a LIVE destination:
      • CGN_TRANSITION consumer="self_model" (the producer the rFP §1.3 inventory
        flagged as entirely MISSING — this is what wakes the self_model HAOV
        consumer). Reward = legacy contract 0.5 confirmed / -0.1 falsified.
      • TIMECHAIN_COMMIT meta-fork (self-insight provenance — timechain_worker
        is a live subscriber).
      • emit_chain_outcome_insight peer cross-insight (Phase C §23.1).
    The legacy also emitted SELF_PREDICTION_VERIFIED, but that constant has NO
    subscriber (dead even pre-retirement — spirit_worker only emitted it, never
    consumed it); recreating it would reintroduce the F4 dead-route anti-pattern,
    so it is intentionally dropped.
    """
    sr = state_refs.get("self_reasoning")
    if sr is None or not hasattr(sr, "check_predictions"):
        return
    try:
        verifications = sr.check_predictions(
            epoch, neuromods, msl_data or {}, language or {})
    except Exception as e:
        logger.warning(
            "[SelfReflectionWorker] self_reasoning.check_predictions() "
            "failed: %s", e)
        return
    if not verifications:
        return
    for v in verifications:
        target = v.get("target", "unknown")
        confirmed = bool(v.get("confirmed", False))
        reward = 0.5 if confirmed else -0.1
        # (1) self_model CGN_TRANSITION — the missing HAOV producer (rFP §1.3).
        _send_msg(send_queue, "CGN_TRANSITION", name, "cgn", {
            "type": "experience",  # (b) self-pred is simultaneous → record_experience → observe_for (DEFERRED G1)
            "consumer": "self_model",
            "concept_id": f"self_pred_{target}",
            "reward": reward,
            "outcome_context": {
                "source": "self_prediction",
                "metric": str(target)[:100],
                "predicted": v.get("predicted", 0),
                "actual": v.get("actual", 0),
                "error": v.get("error", 0),
                "confirmed": confirmed,
            },
        })
        # (2) meta-fork TimeChain commit (live: timechain_worker).
        _send_msg(send_queue, bus.TIMECHAIN_COMMIT, name, "timechain", {
            "fork": "meta", "thought_type": "meta",
            "source": "self_reasoning",
            "content": {"event": "SELF_PREDICTION_VERIFIED",
                        "prediction": str(target)[:100],
                        "error": v.get("error", 0),
                        "confirmed": confirmed},
            "significance": 0.6, "novelty": 0.5, "coherence": 0.5,
            "tags": ["self_insight", "prediction_verified"],
        })
        # (3) peer cross-insight (informative-only gate inside; falsified only).
        try:
            from titan_hcl.logic.cgn_consumer_client import (
                emit_chain_outcome_insight)
            emit_chain_outcome_insight(
                send_queue, name, "self_model", float(reward),
                ctx={"source": "self_prediction", "confirmed": confirmed})
        except Exception:
            pass
    logger.info(
        "[SelfReflectionWorker] self_model: %d prediction(s) verified "
        "(EMA accuracy=%.3f)", len(verifications),
        getattr(sr, "_prediction_accuracy_ema", 0.5))


def _handle_epoch_tick(payload: dict, state_refs: dict, send_queue,
                       name: str) -> None:
    """Per-epoch tick — drive engines' cooldown bookkeeping + the restored
    coding-exploration time-fallback (3A) + self-prediction check (3B).

    Per rFP §2.B.3: tick_cooldown for self_reasoning + coding_explorer.
    PredictionEngine doesn't have a tick() — it's input-driven via
    predict_next/compute_error from the publisher daemon (B7).
    """
    sr = state_refs.get("self_reasoning")
    if sr is not None and hasattr(sr, "tick_cooldown"):
        try:
            sr.tick_cooldown()
        except Exception as e:
            logger.debug(
                "[SelfReflectionWorker] self_reasoning.tick_cooldown() failed: %s",
                e)

    ce = state_refs.get("coding_explorer")
    if ce is not None and hasattr(ce, "tick_cooldown"):
        try:
            ce.tick_cooldown()
        except Exception as e:
            logger.debug(
                "[SelfReflectionWorker] coding_explorer.tick_cooldown() failed: %s",
                e)

    # ── 3A Layer-B: coding-exploration time-fallback (rFP §3 Phase 3A) ──
    # The dominant activation path on saturated-stable Titans: the gap-driven
    # Layer-A (introspect coherence gaps) rarely fires, so this 6h time-fallback
    # guarantees coding activity. Cheap timestamp gate first; only read SHM +
    # build context when it actually fires.
    if ce is not None and ce.can_explore:
        _now = time.time()
        try:
            _fire = ce.should_fire_fallback(_now)
        except Exception:
            _fire = False
        if _fire:
            try:
                _trigger = ce.build_fallback_trigger(_now)
            except Exception as _bft_err:
                logger.warning(
                    "[SelfReflectionWorker] build_fallback_trigger failed: %s",
                    _bft_err)
                _trigger = None
            if _trigger is not None:
                logger.info(
                    "[SelfReflectionWorker] CodingExplorer time-fallback fired "
                    "(silence threshold %.0fs)", ce._time_fallback_seconds)
                _drive_coding_exploration(
                    state_refs, send_queue, name,
                    trigger=_trigger,
                    epoch=_read_epoch(state_refs),
                    neuromods=_read_neuromods(state_refs),
                    context=_build_coding_explore_context(state_refs, sr))

    # ── 3B: self-prediction check every 100 epochs (rFP §3 Phase 3B) ──
    # check_predictions resolves introspect-created SelfPredictions + is the only
    # updater of _prediction_accuracy_ema. Legacy cadence = every 100 epochs.
    if sr is not None and hasattr(sr, "check_predictions"):
        _epoch = _read_epoch(state_refs)
        _last = state_refs.get("_last_self_pred_check_epoch", 0)
        if _epoch > 0 and (_epoch - _last) >= 100:
            state_refs["_last_self_pred_check_epoch"] = _epoch
            _drive_self_prediction_check(
                state_refs, send_queue, name, _epoch,
                _read_neuromods(state_refs))


# ── Sandbox subprocess lifecycle (chunk B6) ─────────────────────────────────


def _check_sandbox_health(state_refs: dict) -> None:
    """30s health probe of CodingSandboxHelper. Per rFP §2.B.6:
    if status reports unavailable, attempt single restart; if restart fails,
    log CRITICAL + disable coding_explorer (PredictionEngine + SelfReasoning
    continue running).

    Note: CodingSandboxHelper.status() is a lightweight `which python3`
    check (coding_sandbox.py:214-226). Real subprocess errors surface at
    sandbox.run() call sites, not here.
    """
    ce = state_refs.get("coding_explorer")
    if ce is None:
        return
    sandbox = getattr(ce, "_sandbox", None)
    if sandbox is None:
        return
    try:
        status = sandbox.status() if hasattr(sandbox, "status") else "unknown"
    except Exception as e:
        logger.warning(
            "[SelfReflectionWorker] sandbox.status() raised: %s", e)
        status = "unknown"

    prev_status = state_refs.get("_sandbox_last_status")
    state_refs["_sandbox_last_status"] = status

    if status == "available":
        if prev_status not in (None, "available"):
            logger.info(
                "[SelfReflectionWorker] sandbox recovered: %s → available",
                prev_status)
        return

    # Status unavailable / unknown — attempt single restart.
    restart_attempts = state_refs.get("_sandbox_restart_attempts", 0)
    if restart_attempts >= 1:
        # Already tried once; disable coding_explorer to prevent restart loop.
        logger.critical(
            "[SelfReflectionWorker] sandbox restart failed (status=%s after "
            "1 attempt) — disabling coding_explorer; "
            "SelfReasoning + PredictionEngine continue running",
            status)
        state_refs["_sandbox_disabled"] = True
        return

    logger.warning(
        "[SelfReflectionWorker] sandbox status=%s — attempting restart "
        "(attempt %d/1)", status, restart_attempts + 1)
    state_refs["_sandbox_restart_attempts"] = restart_attempts + 1
    # CodingSandboxHelper has no explicit restart() — re-instantiate.
    try:
        from titan_hcl.logic.agency.helpers.coding_sandbox import (
            CodingSandboxHelper)
        ce._sandbox = CodingSandboxHelper()
        new_status = ce._sandbox.status() if hasattr(ce._sandbox, "status") else "available"
        if new_status == "available":
            logger.info(
                "[SelfReflectionWorker] sandbox restart succeeded — "
                "status=%s", new_status)
            state_refs["_sandbox_restart_attempts"] = 0  # reset on success
        else:
            logger.warning(
                "[SelfReflectionWorker] sandbox restart returned status=%s",
                new_status)
    except Exception as e:
        logger.critical(
            "[SelfReflectionWorker] sandbox restart raised: %s — disabling "
            "coding_explorer", e, exc_info=True)
        state_refs["_sandbox_disabled"] = True


def _check_sandbox_orphans(state_refs: dict) -> None:
    """60s orphan-PID belt-and-suspenders detection per rFP §2.B.6.

    CodingSandboxHelper is per-call subprocess.run, NOT a daemon child —
    so under normal operation there should be at most 0 long-lived
    python3 children. This check catches the edge case where a sandbox.run()
    got stuck (e.g. python3 hung on stdin) and left an orphan.

    Threshold: >1 python3 child for >60s = anomalous. Logs WARN and
    terminates extras. Best-effort: psutil may be unavailable on minimal
    deploys; skip gracefully.
    """
    try:
        import psutil  # noqa: WPS433 — lazy import; psutil may not be installed
    except Exception:
        return

    try:
        proc = psutil.Process()
        children = proc.children(recursive=True)
        py_children = [
            c for c in children
            if (c.name() or "").startswith("python") or "python" in (c.name() or "").lower()
        ]
    except Exception as e:
        logger.debug(
            "[SelfReflectionWorker] orphan check psutil failed: %s", e)
        return

    # Threshold: more than 1 python3 child indicates an orphan. (Some
    # transient child during sandbox.run is OK; persistent multi-children
    # 60s after the previous check is the anomaly.)
    if len(py_children) > 1:
        logger.warning(
            "[SelfReflectionWorker] %d python child processes detected "
            "(expected ≤1) — terminating extras", len(py_children))
        # Keep the youngest, terminate the rest (oldest are most likely orphans).
        try:
            sorted_by_age = sorted(py_children, key=lambda c: c.create_time())
            # Terminate all but the most recent.
            for c in sorted_by_age[:-1]:
                try:
                    c.terminate()
                    logger.warning(
                        "[SelfReflectionWorker] terminated orphan PID=%d "
                        "name=%s", c.pid, c.name())
                except Exception as e:
                    logger.debug(
                        "[SelfReflectionWorker] orphan terminate %d failed: %s",
                        c.pid, e)
        except Exception as e:
            logger.warning(
                "[SelfReflectionWorker] orphan termination loop failed: %s", e)


# ── Cadence-driven publishers (chunk B7) ────────────────────────────────────


def _publisher_loop(state_refs: dict, send_queue, name: str, titan_id: str,
                    stop_event: threading.Event,
                    publisher_cadence_s: float,
                    coding_publisher_cadence_s: float,
                    prediction_publisher_cadence_s: float) -> None:
    """Background daemon — fires every min cadence and dispatches per
    rFP §2.B.4 + SPEC v1.2.1 §9.B self_reflection_worker Bus publications:

      • SELF_REFLECTION_STATS_UPDATED  (2.5s) → /v4/self-reflection
      • CODING_EXPLORER_STATS_UPDATED  (5s)   → /v4/coding-explorer
      • PREDICTION_STATS_UPDATED       (2.5s) → /v4/prediction
      • PREDICTION_GENERATED           (on prediction; tracks
                                        _prediction_engine._total_predictions
                                        delta and emits when it grows)

    On-event SELF_REASONING_INSIGHT + CODING_INSIGHT emit from the
    handler/engine paths (not periodic) — those are reserved for the
    follow-up that wires the engine's internal insight callbacks.

    Exits when stop_event.is_set() (MODULE_SHUTDOWN). All publish paths
    are best-effort; failures log at debug + continue.
    """
    last_self_reflection = 0.0
    last_coding = 0.0
    last_prediction = 0.0
    last_predictions_total = 0
    # Track 2 D4 (v1.2.1) — on-insight emit detector state. Mirrors the
    # PREDICTION_GENERATED counter-delta pattern: we poll the engine's
    # cumulative insight counter each publisher cycle and emit
    # SELF_REASONING_INSIGHT / CODING_INSIGHT when the counter grows.
    last_introspections_total = 0
    last_exercises_total = 0
    sleep_interval = min(publisher_cadence_s, coding_publisher_cadence_s,
                          prediction_publisher_cadence_s) / 2.0
    sleep_interval = max(0.5, sleep_interval)

    logger.debug(
        "[SelfReflectionWorker] publisher_loop online — sleep_interval=%.2fs",
        sleep_interval)

    while not stop_event.is_set():
        now = time.time()

        # ── SELF_REFLECTION_STATS_UPDATED (2.5s) ────────────────────────
        if now - last_self_reflection >= publisher_cadence_s:
            _publish_self_reflection_stats(state_refs, send_queue, name, titan_id)
            # Track 2 D4 — SELF_REASONING_INSIGHT on counter delta.
            sr = state_refs.get("self_reasoning")
            if sr is not None:
                cur = int(getattr(sr, "_total_introspections", 0))
                if cur > last_introspections_total:
                    _publish_self_reasoning_insight(
                        sr, send_queue, name, titan_id, cur,
                        last_introspections_total)
                    last_introspections_total = cur
            last_self_reflection = now

        # ── CODING_EXPLORER_STATS_UPDATED (5s) ──────────────────────────
        if now - last_coding >= coding_publisher_cadence_s:
            _publish_coding_explorer_stats(state_refs, send_queue, name, titan_id)
            # Track 2 D4 — CODING_INSIGHT on exercise-counter delta.
            ce = state_refs.get("coding_explorer")
            if ce is not None:
                cur = int(getattr(ce, "_total_exercises", 0))
                if cur > last_exercises_total:
                    _publish_coding_insight(
                        ce, send_queue, name, titan_id, cur,
                        last_exercises_total)
                    last_exercises_total = cur
            last_coding = now

        # ── PREDICTION_STATS_UPDATED (2.5s) ─────────────────────────────
        if now - last_prediction >= prediction_publisher_cadence_s:
            _publish_prediction_stats(state_refs, send_queue, name, titan_id)
            # Detect new predictions since last publish — emit PREDICTION_GENERATED
            # on edge so cognitive_worker novelty consumer fires.
            pe = state_refs.get("prediction_engine")
            if pe is not None:
                total = int(getattr(pe, "_total_predictions", 0))
                if total > last_predictions_total:
                    _publish_prediction_generated(
                        pe, state_refs, send_queue, name, titan_id, total)
                    last_predictions_total = total
            last_prediction = now

        stop_event.wait(sleep_interval)

    logger.debug("[SelfReflectionWorker] publisher_loop exiting (stop_event set)")


def _publish_self_reflection_stats(state_refs: dict, send_queue, name: str,
                                     titan_id: str) -> None:
    """SELF_REFLECTION_STATS_UPDATED — aggregate snapshot of SelfReasoningEngine
    state (the rFP groups all three engines under self_reflection_worker but
    the canonical "self_reflection.state" cache key surfaces the SelfReasoning
    surface specifically — coding_explorer + prediction have their own keys).
    """
    sr = state_refs.get("self_reasoning")
    if sr is None or not hasattr(sr, "get_stats"):
        return
    try:
        stats = sr.get_stats()
    except Exception as e:
        logger.debug(
            "[SelfReflectionWorker] self_reasoning.get_stats failed: %s", e)
        return
    _send_msg(send_queue, bus.SELF_REFLECTION_STATS_UPDATED, name, "all", {
        "titan_id": titan_id,
        "stats": stats,
        "last_dream_state": state_refs.get("_last_dream_state"),
        "ts": time.time(),
    })


def _publish_coding_explorer_stats(state_refs: dict, send_queue, name: str,
                                     titan_id: str) -> None:
    """CODING_EXPLORER_STATS_UPDATED — CodingExplorer.get_stats snapshot."""
    ce = state_refs.get("coding_explorer")
    if ce is None or not hasattr(ce, "get_stats"):
        return
    try:
        stats = ce.get_stats()
    except Exception as e:
        logger.debug(
            "[SelfReflectionWorker] coding_explorer.get_stats failed: %s", e)
        return
    _send_msg(send_queue, bus.CODING_EXPLORER_STATS_UPDATED, name, "all", {
        "titan_id": titan_id,
        "stats": stats,
        "sandbox_disabled": state_refs.get("_sandbox_disabled", False),
        "sandbox_last_status": state_refs.get("_sandbox_last_status"),
        "ts": time.time(),
    })


def _publish_prediction_stats(state_refs: dict, send_queue, name: str,
                                titan_id: str) -> None:
    """PREDICTION_STATS_UPDATED — PredictionEngine.get_stats snapshot.

    Closes Track 1 drift: pre-Track-2, this event was emitted from
    cognitive_worker (where prediction_engine was drift-relocated). Post-
    Track-2, self_reflection_worker is the canonical producer per
    rFP §0 table + drift correction. /v4/prediction route reads from
    cache.prediction.state.
    """
    pe = state_refs.get("prediction_engine")
    if pe is None or not hasattr(pe, "get_stats"):
        return
    try:
        stats = pe.get_stats()
    except Exception as e:
        logger.debug(
            "[SelfReflectionWorker] prediction_engine.get_stats failed: %s", e)
        return
    _send_msg(send_queue, bus.PREDICTION_STATS_UPDATED, name, "all", {
        "titan_id": titan_id,
        "stats": stats,
        "ts": time.time(),
    })


def _publish_prediction_generated(prediction_engine, state_refs: dict,
                                    send_queue, name: str, titan_id: str,
                                    total: int) -> None:
    """PREDICTION_GENERATED — emit when _total_predictions counter grows.

    Per rFP §2.B.4: consumed by cognitive_worker for novelty-driven exploration.
    Payload includes the latest prediction surrogate (last_prediction attribute
    if accessible) + total counter for delta tracking.
    """
    last_prediction = getattr(prediction_engine, "_last_prediction", None)
    total_surprises = int(getattr(prediction_engine, "_total_surprises", 0))
    _send_msg(send_queue, bus.PREDICTION_GENERATED, name, "all", {
        "titan_id": titan_id,
        "total_predictions": total,
        "total_surprises": total_surprises,
        "last_prediction": last_prediction if isinstance(
            last_prediction, (list, tuple, type(None))) else None,
        "ts": time.time(),
    })


def _publish_self_reasoning_insight(self_reasoning, send_queue, name: str,
                                      titan_id: str, total: int,
                                      prev_total: int) -> None:
    """SELF_REASONING_INSIGHT — emit when _total_introspections counter grows.

    Track 2 D4 (v1.2.1) closure per rFP §2.B.4 + Prime Directive #1. Mirrors
    the PREDICTION_GENERATED counter-delta pattern: poll the cumulative
    introspection counter each publisher cycle; on delta, emit with the
    last sub-mode profile snapshot if accessible.

    Consumed by cognitive_worker for meta-reasoning feed (per rFP).
    """
    last_profile = getattr(self_reasoning, "_last_profile", None)
    last_profile_epoch = int(getattr(
        self_reasoning, "_last_profile_epoch", 0))
    prediction_accuracy_ema = float(getattr(
        self_reasoning, "_prediction_accuracy_ema", 0.0))
    _send_msg(send_queue, bus.SELF_REASONING_INSIGHT, name, "all", {
        "titan_id": titan_id,
        "total_introspections": total,
        "delta": total - prev_total,
        "last_profile_epoch": last_profile_epoch,
        "prediction_accuracy_ema": round(prediction_accuracy_ema, 4),
        # last_profile may carry dataclass/dict fields; serialize defensively.
        "profile_kind": getattr(last_profile, "kind", None) if last_profile is not None else None,
        "ts": time.time(),
    })


def _publish_coding_insight(coding_explorer, send_queue, name: str,
                              titan_id: str, total: int,
                              prev_total: int) -> None:
    """CODING_INSIGHT — emit when _total_exercises counter grows.

    Track 2 D4 (v1.2.1) closure per rFP §2.B.4 + Prime Directive #1.
    Counter-delta pattern. Consumed by cgn_module for cross-insight
    propagation (per rFP §2.B.4).
    """
    total_successes = int(getattr(coding_explorer, "_total_successes", 0))
    success_rate = (total_successes / max(1, total)) if total else 0.0
    _send_msg(send_queue, bus.CODING_INSIGHT, name, "all", {
        "titan_id": titan_id,
        "total_exercises": total,
        "delta": total - prev_total,
        "total_successes": total_successes,
        "success_rate": round(success_rate, 4),
        "ts": time.time(),
    })
