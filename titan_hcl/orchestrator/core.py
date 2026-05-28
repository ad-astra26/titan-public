"""
titan_hcl.orchestrator.core — Orchestrator class (Phase 11 §11.I.1 role split).

Phase 6 (D-SPEC-135) carved Guardian out of titan_hcl/guardian.py into
guardian_hcl/core.py. Phase 11 §11.I.1 (D-SPEC-141 / v1.65.0) splits the
monolithic Guardian's role into:

  * Orchestrator — owns module spawn / dep activation / probe execution /
    hot-reload spawn / lazy-start (this file).
  * Supervisor (titan_hcl.supervisor.core) — owns fault detection +
    ModuleError aggregation + MODULE_RESTART_REQUEST emission (locked D5).

For 11E.b.1 (single-process Python class split), the Orchestrator class also
retains the supervision methods (monitor_tick / _process_guardian_messages /
_module_ready_publish_loop) verbatim — the Supervisor class wraps an
orchestrator reference and adds fault-detection → MODULE_RESTART_REQUEST
emission. The kernel-rs peer-spawn (11E.b.2) is the gate that moves the
supervision methods physically into the Supervisor process.

For back-compat with all 30+ existing test + callsite imports
(`from titan_hcl.guardian_hcl import Guardian`), the bottom of this file
exposes `Guardian = Orchestrator` and `titan_hcl/guardian_hcl/core.py`
re-exports from here.

Dataclasses (ModuleState / ModuleSpec / ModuleInfo / ReloadState) live in
`titan_hcl/orchestrator/module_registry.py`. Mixins:
  * OrchestratorReloadMixin (titan_hcl/reload.py) — D-SPEC-50 hot reload
  * OrchestratorDepActivationMixin (titan_hcl/orchestrator/dep_activation.py)
    — D-SPEC-90 §11.G.2.5 ENSURE_RUNNING
"""
"""
Guardian — Module supervisor for Titan V4.0 microkernel.

Manages the lifecycle of supervised module processes:
  - Start/stop/restart individual modules
  - Monitor heartbeats (kill/restart on timeout)
  - Track RSS per module (restart on threshold breach)
  - Provide module status to Core via the Divine Bus
  - Sliding-window restart tracking (prevents infinite restart loops)
  - Per-module heartbeat timeout (Spirit needs longer for heavy V4 work)

Each module runs as a separate multiprocessing.Process with its own
memory space, communicating exclusively through the Divine Bus.
"""
import asyncio
import logging
import os
import queue as _queue_mod
import signal
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from multiprocessing import Process
from queue import Empty
from typing import Callable, Optional

from titan_hcl.bus import (
    AnyQueue,
    BUS_PEER_DIED,
    BUS_WORKER_ADOPT_ACK,
    BUS_WORKER_ADOPT_REQUEST,
    DivineBus,
    MODULE_CRASHED,
    MODULE_HEARTBEAT,
    MODULE_PROBE_REQUEST,
    MODULE_RELOAD_ACK,
    MODULE_RELOAD_REQUEST,
    MODULE_SHUTDOWN,
    SUPERVISION_CHILD_DOWN,
    SUPERVISION_CHILD_RESTARTED,
    SUPERVISION_DEPENDENCY_ACTIVATING,
    SUPERVISION_DEPENDENCY_BLOCKED,
    SUPERVISION_DEPENDENCY_DEGRADED,
    SUPERVISION_DEPENDENCY_RECOVERED,
    SUPERVISION_ESCALATION,
    make_msg,
)
from titan_hcl import bus
from titan_hcl._phase_c_constants import (
    ADOPTION_TIMEOUT_S,
    MODULE_RELOAD_DEFAULT_TIMEOUT_S,
    MODULE_RELOAD_HAPPY_PATH_S,
    SUPERVISION_DEPENDENCY_ACTIVATION_TIMEOUT_S,
)
from titan_hcl.supervision import (
    Dependency,
    DependencyAction,
    DependencyKind,
    DependencySeverity,
    EscalationDecision,
    ReasonRecord,
    SupervisionReason,
    classify_exit_code,
    kernel_default_decision,
    most_common_reason,
)

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────

HEARTBEAT_INTERVAL = 10.0       # seconds between expected heartbeats
HEARTBEAT_TIMEOUT = 90.0        # seconds before declaring a module dead (mainnet-safe: ~Schumann×27)
DEFAULT_RSS_LIMIT_MB = 1500     # per-module RSS limit (MB)
RESTART_BACKOFF_BASE = 2.0      # exponential backoff base (seconds)
MAX_RESTARTS_IN_WINDOW = 5      # max restarts allowed in the sliding window
RESTART_WINDOW_SECONDS = 600.0  # 10-minute sliding window for restart tracking
SUSTAINED_UPTIME_RESET = 300.0  # 5 minutes of uptime before restart count resets
REENABLE_COOLDOWN_S = 600.0    # 10 minutes before auto-re-enabling a disabled module
# CPU-aware heartbeat (added 2026-04-21) — when heartbeat times out, sample
# /proc/<pid>/stat CPU time. If CPU grew ≥ MIN_CPU_DELTA_FOR_ALIVE since last
# sample, the module is alive-but-CPU-starved (not deadlocked). Defer restart
# for up to MAX_STARVED_CYCLES wallclock heartbeat windows; then force-restart
# (bounded grace prevents runaway hang on a truly stuck module).
MIN_CPU_DELTA_FOR_ALIVE = 1.0   # seconds of CPU time per heartbeat window proves liveness
MAX_STARVED_CYCLES = 5          # how many consecutive starved-but-alive cycles to tolerate
# Bumped 3 → 5 on 2026-04-21 after observing both T2+T3 media modules hit
# grace-exhausted-restart once each during the same 75-min ARC iter-3 slot.
# 5 cycles ≈ 5 minutes wallclock grace under monitor_tick=5s — should bridge
# typical ARC tail without leaving truly-stuck modules hanging too long.



from titan_hcl.orchestrator.module_registry import (
    ModuleState,
    ModuleSpec,
    ModuleInfo,
    ReloadState,
    _append_meta_cgn_emission_log,
)
from titan_hcl.reload import OrchestratorReloadMixin
from titan_hcl.orchestrator.dep_activation import OrchestratorDepActivationMixin


class Orchestrator(OrchestratorReloadMixin, OrchestratorDepActivationMixin):
    """
    Orchestrator (Phase 11 §11.I.1 D-SPEC-141): supervises module processes —
    starts, monitors, restarts. Owns the lifecycle surface; the Supervisor
    class (titan_hcl.supervisor.core) wraps an Orchestrator and emits
    MODULE_RESTART_REQUEST on fault detection per locked D5.

    V4 improvements over V3:
      - Sliding-window restart tracking (no more infinite restart loops)
      - Per-module heartbeat_timeout (Spirit gets 120s for heavy V4 work)
      - Restart count only resets after sustained uptime (5 min)
      - Single-instance enforcement via process liveness check before start

    Usage:
        bus = DivineBus()
        guardian = Guardian(bus)
        guardian.register(ModuleSpec("memory", memory_worker_fn, config={...}))
        guardian.register(ModuleSpec("recorder", recorder_worker_fn, rss_limit_mb=2500))
        guardian.start_all()      # starts autostart=True modules
        guardian.start("memory")  # start a specific module on demand
        guardian.monitor_tick()   # call periodically (e.g. every 5s)
    """

    def __init__(self, bus: DivineBus, config: dict | None = None):
        self.bus = bus
        self._modules: dict[str, ModuleInfo] = {}
        self._module_recv_queues: dict[str, AnyQueue] = {}  # name → recv queue (for bus routing)
        # Option B (2026-04-29): declare the msg_types Guardian actually
        # consumes via this queue. All four are typically sent with
        # dst="guardian" (targeted) and therefore bypass the broadcast
        # filter regardless — but we still declare them so the contract is
        # explicit and arch_map can verify it. The real win: every other
        # broadcast (~150 msg types — SPHERE_PULSE, *_UPDATED, EXPRESSION_*,
        # etc.) is now dropped at publish, freeing the guardian queue from
        # the dst="all" flood that was causing 87 MODULE_HEARTBEAT drops
        # per ~500-line window on T1 (false-restart risk).
        self._guardian_queue = bus.subscribe(
            "guardian",
            types=[
                MODULE_HEARTBEAT,
                MODULE_SHUTDOWN,
                BUS_WORKER_ADOPT_REQUEST,
                # Phase B.2 §D9 (2026-05-02) — broker → Guardian peer-death
                # signal. Broker detects peer PID dead via os.kill(pid, 0)
                # and publishes BUS_PEER_DIED. Guardian triggers immediate
                # restart for named workers (faster than 1Hz polling).
                BUS_PEER_DIED,
                # SPEC §8.3 Phase B (D-SPEC-49) — Maker CLI / future D9
                # Guardian initiates per-module hot-reload via this targeted
                # P0 message (dst="guardian"). Routed in
                # `_process_guardian_messages` to `_dispatch_reload_request`.
                MODULE_RELOAD_REQUEST,
                # SAVE_DONE is targeted (dst="guardian") so it bypasses the
                # filter regardless, but list it explicitly so arch_map can
                # see the contract.
            ],
        )
        self._stop_requested = False
        self._module_lock = threading.RLock()  # serialize start/stop/restart to prevent duplicate spawns

        # Option B (2026-05-02) — restart executor offload.
        # Pre-fix: monitor_tick called self.restart() synchronously, which
        # blocks for up to 30s (SAVE_NOW wait) PER worker. Multi-restart
        # bursts (e.g. cascade after a stall) blocked monitor_tick for
        # 6×30s ≈ 180s, starving heartbeat processing → MORE timeouts →
        # cascade. Option A processes heartbeats inline during the wait
        # (mitigation); Option B moves restart() off monitor_tick entirely
        # so the queue keeps draining at 50/sec even during a long
        # SAVE_NOW wait.
        # max_workers=4 → up to 4 concurrent restarts, more than enough for
        # any realistic burst; module_lock in restart() serializes per-name.
        # See BUG-GUARDIAN-STOP-SAVE-NOW-HEARTBEAT-CASCADE-20260502.
        import concurrent.futures
        self._restart_executor: concurrent.futures.ThreadPoolExecutor = (
            concurrent.futures.ThreadPoolExecutor(
                max_workers=4, thread_name_prefix="guardian-restart")
        )
        self._restarts_in_flight: set[str] = set()
        self._restart_lock = threading.Lock()
        # SPEC §11.B.3 (D-SPEC-49) — per-module hot-reload state. Keyed by
        # module name; a non-None entry indicates a reload is mid-flight
        # for that module. The orchestrator runs on _restart_executor (a
        # separate worker) so it does not block monitor_tick / message
        # drain. _reload_lock serializes mutation of the dict + the
        # `info.reload_in_flight` flag transition; the lock is held only
        # for the entry/exit transitions (NOT for the entire orchestration)
        # to avoid blocking other reloads on different modules.
        self._reloads_in_flight: dict[str, ReloadState] = {}
        self._reload_lock = threading.Lock()
        # Microkernel v2 Phase A retrofit (2026-04-27): swap-aware kernel ref.
        # Kernel sets `self.guardian._kernel_ref = self` after Guardian
        # construction. start()/restart() consult kernel.is_shadow_swap_active()
        # to block lazy-starts during swap (prevents proxy-driven mid-swap
        # worker resurrection that holds DB locks). None in legacy mode
        # (in-process; no kernel split) → swap interlock degrades to no-op.
        self._kernel_ref = None

        # D-SPEC-123 follow-up (2026-05-23, Option B): SHM-based module-
        # ready state for cross-process liveness checks. Replaces the
        # tactical guardian=None tolerance in proxies/_start_safe.py with
        # a proper Phase-C-canonical mechanism (G18 watermark pattern).
        # Guardian publishes the full {name: state} snapshot to
        # module_ready.bin every 1s; subprocess proxies that don't hold a
        # Guardian reference (MemoryProxy / SocialGraphProxy / RLProxy
        # constructed with guardian=None in agno_worker_plugin.py) read
        # this slot for liveness checks. Best-effort construct — if SHM
        # init fails the proxies fall back to the optimistic-True path
        # (still no crash).
        self._module_ready_shm_writer = None
        self._module_ready_publisher_stop = threading.Event()
        self._module_ready_publisher_thread: Optional[threading.Thread] = None
        try:
            from titan_hcl.core.module_ready_shm import ModuleReadyShmWriter
            self._module_ready_shm_writer = ModuleReadyShmWriter()
            self._module_ready_publisher_thread = threading.Thread(
                target=self._module_ready_publish_loop,
                name="guardian-module-ready-shm",
                daemon=True,
            )
            self._module_ready_publisher_thread.start()
            logger.info(
                "[Guardian] module_ready.bin SHM publisher started "
                "(D-SPEC-123 follow-up Option B — 1Hz snapshot)")
        except Exception as _mr_err:
            logger.warning(
                "[Guardian] module_ready.bin SHM writer init failed: %s "
                "(proxies fall back to optimistic-True liveness path)",
                _mr_err)

        # [guardian] toml plumbing — 2026-04-16. Previously Guardian(bus)
        # was constructed with no config, so the module-level constants
        # HEARTBEAT_TIMEOUT / DEFAULT_RSS_LIMIT_MB / MAX_RESTARTS_IN_WINDOW
        # etc. could not be tuned via titan_params.toml. Constants remain
        # as fallbacks to preserve behavior when config is absent.
        cfg = config or {}
        self._heartbeat_timeout = float(cfg.get("heartbeat_timeout_default", HEARTBEAT_TIMEOUT))
        self._heartbeat_timeout_spirit = float(cfg.get("heartbeat_timeout_spirit", 120.0))
        self._max_restarts_in_window = int(cfg.get("max_restarts_in_window", MAX_RESTARTS_IN_WINDOW))
        self._restart_window_seconds = float(cfg.get("restart_window", RESTART_WINDOW_SECONDS))
        self._sustained_uptime_reset = float(cfg.get("sustained_uptime_reset", SUSTAINED_UPTIME_RESET))
        # Phase 9 Chunk 9L (RFP §3F.2.6) — staggered boot to eliminate
        # cold-boot CPU contention. ~40 autostart modules booting in parallel
        # on a 4-core VPS oversubscribed CPU 5-6×, causing cascade
        # heartbeat-timeouts (audit AUDIT_agno_chat_hang_diagnosis_20260527).
        # Stagger start() calls by N seconds so workers' import + Agent-init
        # phases don't all hit the GIL/CPU at the same instant. Phase 11
        # replaces the flat walk with a two-phase pipeline (§11.I.7); within
        # each phase the stagger is applied between consecutive spawns.
        self._boot_stagger_delay_s = max(
            0.0, float(cfg.get("boot_stagger_delay_s", 1.5)))
        # Phase 11 §11.I.7 / §11.I.8 — Phase B (OPTIONAL_POST_BOOT) uses a
        # longer default stagger (5s) so the post-boot wave does not pile
        # CPU pressure on top of an already-serving fleet.
        self._post_boot_stagger_delay_s = max(
            0.0, float(cfg.get("post_boot_stagger_delay_s", 5.0)))
        # Phase 11 §11.I.7 boot concurrency cap (Maker 2026-05-28). Non-blocking
        # concurrent boot can oversubscribe a small box (cold imports of torch /
        # models all at once → load 20-32 on the 4-core T1 box). This bounds the
        # number of modules IN-FLIGHT (spawned but not yet state=running) at
        # once; the 1Hz probe poller drives in-flight→running, freeing a slot
        # for the next spawn. Caps the cold-boot CPU spike (≈cap concurrent
        # cold-boots) while keeping full boot under ~3 min. 0 = uncapped.
        self._boot_concurrency_cap = int(cfg.get("boot_concurrency_cap", 8))
        # Phase 11 §11.I.3 — per-module probe-wait budget. After spawn, the
        # orchestrator polls the worker's `module_<name>_state.bin` slot
        # waiting for state=RUNNING. Workers without a SHM writer (legacy,
        # pre-11I migration) simply never reach `running` via SHM — the
        # orchestrator falls back to today's MODULE_READY bus path via
        # `info.ready_time`. The probe wait timeout bounds the wait either
        # way so a single slow worker can never stall the whole boot.
        # No lower floor — tests may opt into very short waits; production
        # tuning comes via titan_params.toml [guardian.probe_wait_timeout_s].
        self._probe_wait_timeout_s = max(
            0.0, float(cfg.get("probe_wait_timeout_s", 30.0)))
        # Phase 11 §11.I.7 — when True (default), Orchestrator.start_all()
        # uses the new BootPriority-partitioned pipeline + writes its own
        # `titan_hcl_state.bin` SHM slot for `fleet_ready` /
        # `fleet_optional_ready` (kernel-rs gates on this). When False,
        # falls back to the Phase 9 9L flat staggered walk + skips
        # publication (useful for unit tests + in-process Orchestrators
        # that do not own the orchestrator SHM slot).
        self._phase_11_pipeline_enabled = bool(cfg.get(
            "phase_11_pipeline_enabled", True))
        # Phase 11 §11.I.7 — fleet-state SHM writer (orchestrator-owned slot,
        # G21 single-writer). Lazy-init to keep test fixtures that construct
        # an Orchestrator without a SHM root happy. Writes are best-effort:
        # init failure is logged + start_all proceeds without SHM
        # publication (kernel-rs falls back to its existing process-health
        # check). See `feedback_phase_c_async_only_state_lookup`.
        self._titan_hcl_state_writer = None  # set lazily on first phase write
        # Phase 11 §11.I.7 / RFP §3H Chunk 11D — continuous 1Hz SHM probe
        # poller. THE single, always-on prober: reads every registered module
        # slot at 1Hz and, on observing state=BOOTED, dispatches
        # MODULE_PROBE_REQUEST (worker → probing → running). Covers cold-boot,
        # post-restart, AND lazy-activation with one mechanism — the piece
        # whose absence left workers stuck `booted` forever (live T1
        # 2026-05-28: 30/40 modules alive+heartbeating but never promoted
        # because the ONLY prober was the per-module boot wait). Started in
        # start_all (only the titan_hcl orchestrator calls start_all;
        # guardian_hcl never does, so the poller never runs in the supervisor
        # process → no double-probe). Stopped in stop_all.
        self._probe_poll_interval_s: float = max(
            0.2, float(cfg.get("probe_poll_interval_s", 1.0)))
        # Per-module re-fire window: a worker that missed the first probe (bus
        # subscribe bootstrap race) gets re-probed, but we never flood. Cleared
        # the instant a slot leaves `booted` so a future re-boot re-probes
        # promptly.
        self._probe_refire_interval_s: float = max(
            1.0, float(cfg.get("probe_refire_interval_s", 3.0)))
        self._probe_dispatched_at: dict[str, float] = {}
        self._probe_poller_stop = threading.Event()
        self._probe_poller_thread: Optional[threading.Thread] = None

    def register(self, spec: ModuleSpec) -> None:
        """Register a module specification. Does not start the module."""
        # Microkernel v2 Phase A §A.5 — validate layer before registering.
        from titan_hcl._layer_canon import validate_layer
        validate_layer(spec.layer)
        if spec.name in self._modules:
            logger.warning("[Guardian] Module '%s' already registered, updating spec", spec.name)
        self._modules[spec.name] = ModuleInfo(spec=spec)
        logger.info("[Guardian] Registered module '%s' [%s] (autostart=%s, lazy=%s, rss_limit=%dMB, hb_timeout=%.0fs)",
                     spec.name, spec.layer, spec.autostart, spec.lazy, spec.rss_limit_mb, spec.heartbeat_timeout)

    # ── Microkernel v2 Phase B.2.1 — worker supervision transfer ──────────

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        """Check liveness without sending a signal (signal 0 = existence check)."""
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            # PID exists but we don't own it (different uid). For our purposes,
            # alive — we still tracked an externally-spawned worker.
            return True

    def adopt_worker(self, name: str, pid: int,
                     spec: Optional[ModuleSpec] = None) -> bool:
        """Phase B.2.1 — register an externally-spawned worker without spawning.

        Used during shadow swap when a worker migrates from the old kernel to
        this (shadow) kernel. The worker process already exists at `pid`; we:
          - Verify the PID is alive
          - Verify a ModuleSpec is registered (either passed in or pre-registered
            via Guardian.register at boot)
          - Register a ModuleInfo with adopted=True (changes _cleanup behavior;
            queue-cleanup path uses os.kill instead of mp.Process.kill)
          - Start heartbeat tracking fresh from now (no clock continuity from
            the prior kernel — first heartbeat from the adopted worker resets
            our wall-clock view)

        Args:
            name:  Module name; must match the key registered via .register()
                   on this kernel, or the spec passed in.
            pid:   PID of the live worker process.
            spec:  Optional override spec. If None, uses self._modules[name].spec.

        Returns:
            True on successful adoption.
            False on: unknown name (and no spec passed), dead PID, or already-
            running module (state=RUNNING and adopted=False — fresh worker
            already spawned for this slot, can't double-claim).

        Thread-safe via _module_lock.
        """
        with self._module_lock:
            existing = self._modules.get(name)
            if spec is None and existing is None:
                logger.warning("[Guardian] adopt_worker: unknown name '%s'", name)
                return False
            if not self._pid_alive(pid):
                logger.warning("[Guardian] adopt_worker: pid %d not alive", pid)
                return False
            # If a fresh worker is already running here, reject — double-claim
            # would corrupt _modules state.
            if existing is not None and existing.state == ModuleState.RUNNING \
                    and not existing.adopted:
                logger.warning(
                    "[Guardian] adopt_worker: '%s' already running fresh "
                    "(pid=%s, state=%s) — rejecting adoption of pid=%d",
                    name, existing.pid, existing.state.value, pid,
                )
                return False
            now = time.time()
            actual_spec = spec or existing.spec
            info = ModuleInfo(spec=actual_spec)
            info.pid = pid
            info.process = None  # we don't own a multiprocessing.Process
            info.state = ModuleState.RUNNING
            info.start_time = now
            info.last_heartbeat = now
            info.ready_time = now  # adopted = ready by definition
            info.adopted = True
            info.adopt_ts = now
            self._modules[name] = info
            logger.info(
                "[Guardian] Adopted worker '%s' (pid=%d) from prior kernel "
                "[layer=%s, start_method=%s]",
                name, pid, actual_spec.layer, actual_spec.start_method,
            )
            return True

    def start(self, name: str, activate_deps: bool = True) -> bool:
        """Start a specific module process. Thread-safe via _module_lock.

        `activate_deps` (default True) runs the §11.G.2.5 ENSURE_RUNNING
        dep-activation, which BLOCKS up to SUPERVISION_DEPENDENCY_ACTIVATION_
        TIMEOUT_S per critical dep waiting for it to reach RUNNING. That wait is
        correct for LAZY / on-demand (proxy-driven) starts where the caller
        needs the dep available NOW. During the COORDINATED boot pipeline
        (start_all), the orchestrator already spawns every module in dep+layer
        topo order and the 1Hz probe poller drives them all booted→running
        concurrently — so the boot path passes `activate_deps=False` to avoid
        serializing the otherwise-concurrent boot on per-dep 30s waits (the
        cause of the ~90s Phase A on cold 4-core boot, Maker 2026-05-28)."""
        # Microkernel v2 Phase A retrofit (2026-04-27): autonomous swap
        # interlock. If a shadow swap is in flight, block the calling
        # thread until the swap completes — prevents proxy lazy-starts
        # from resurrecting workers mid-swap (which re-acquire DB locks
        # and fail shadow_boot on locks_not_released). No exception
        # thrown; caller (proxy thread) waits up to 60s, then proceeds
        # against whichever kernel won the swap. Caller is guaranteed
        # eventual completion; no user-visible retry needed.
        if (self._kernel_ref is not None
                and hasattr(self._kernel_ref, "is_shadow_swap_active")
                and self._kernel_ref.is_shadow_swap_active()):
            logger.info(
                "[Guardian] start('%s') deferred — shadow swap in flight; "
                "waiting for completion (max 60s)", name,
            )
            try:
                self._kernel_ref.wait_for_swap_completion(timeout=60.0)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "[Guardian] start('%s') swap-wait failed: %s; "
                    "proceeding anyway", name, e,
                )

        # SPEC §11.G.2.5 (D-SPEC-90, v1.29.0) — dependency-driven activation.
        # Pre-start any ENSURE_RUNNING critical MODULE deps BEFORE acquiring
        # _module_lock so the recursive start(dep_name) call below does not
        # deadlock on its own lock re-acquire. DAG-acyclicity is enforced by
        # SPEC §11.G.7 (arch_map verify --check-deps); recursive activation
        # terminates by induction on DAG depth. Skipped on the boot path
        # (activate_deps=False) — see start() docstring.
        if activate_deps:
            self._activate_dependencies(name)

        with self._module_lock:
            info = self._modules.get(name)
            if not info:
                logger.error("[Guardian] Cannot start unknown module '%s'", name)
                return False

            if info.state in (ModuleState.RUNNING, ModuleState.STARTING):
                # Double-check process is actually alive
                if info.process and info.process.is_alive():
                    logger.debug("[Guardian] Module '%s' already %s (pid=%s)", name, info.state.value, info.pid)
                    return True
                else:
                    logger.warning("[Guardian] Module '%s' marked %s but process dead — cleaning up",
                                   name, info.state.value)
                    self._cleanup_module(name)

            if info.state == ModuleState.DISABLED:
                logger.warning("[Guardian] Module '%s' is disabled, not starting", name)
                return False

            # Safety: kill any orphaned process before spawning
            if info.process is not None:
                try:
                    if info.process.is_alive():
                        logger.warning("[Guardian] Pre-spawn cleanup: killing orphaned process '%s' (pid=%s)",
                                       name, info.pid)
                        info.process.kill()
                        info.process.join(timeout=3.0)
                    info.process.close()
                except Exception:
                    pass
                info.process = None
                info.pid = None

            # Clean up stale bus subscriptions before re-registering
            if info.queue:
                self.bus.unsubscribe(name, info.queue)
            if name in self._module_recv_queues:
                del self._module_recv_queues[name]

            # Create module's bus queues (bidirectional).
            # Microkernel v2 Phase A §A.3 (S6): spec.start_method selects
            # fork (default, byte-identical) vs spawn (fresh interpreter,
            # ~200 MB RSS savings per worker). Unknown values fall back
            # to fork with a WARNING — Guardian never crashes on a
            # misconfigured spec.
            import multiprocessing
            method = info.spec.start_method
            if method not in ("fork", "spawn"):
                logger.warning(
                    "[Guardian] Module '%s' has unknown start_method=%r — "
                    "falling back to 'fork'", name, method)
                method = "fork"
            ctx = multiprocessing.get_context(method)
            # B.3 Stage 1 (2026-05-02) — fork-at-locked-mp.Queue avoidance.
            # Under socket mode, workers rebind to SocketQueue via
            # `setup_worker_bus`. The mp.Queues we used to allocate here
            # were never used in the worker, but they introduced a hazard:
            # mp.Queue's internal feeder thread holds a Condition lock; if
            # parent's feeder owned that lock at fork moment, the child
            # inherited a phantom-locked Lock with no thread to release it.
            # Any code path in worker that transitively touched the
            # inherited Queue (libraries holding closure refs, exception
            # paths, etc.) would block forever — exactly the symptom we saw
            # for spirit/media/language hanging post-init, never sending
            # BUS_SUBSCRIBE. Skipping allocation when broker is attached
            # eliminates the hazard entirely.
            #
            # Legacy fallback (no broker, unit tests, or legacy mode) still
            # gets the mp.Queues — the bug only manifests under fork with
            # the broker active.
            #
            # See: titan-docs/PLAN_microkernel_phase_b3_legacy_path_cleanup.md §1
            if self.bus.has_socket_broker:
                info.queue = None
                info.send_queue = None
            else:
                info.queue = ctx.Queue(maxsize=10000)
                info.send_queue = ctx.Queue(maxsize=10000)

            # Subscribe in the bus so targeted messages get routed.
            # When broker is attached, info.queue is None → skip both the
            # `_module_recv_queues` bookkeeping and the in-process subscriber
            # registration. Workers receive via SocketQueue (broker-routed).
            if info.queue is not None:
                self._module_recv_queues[name] = info.queue
                if not self.bus.has_socket_broker:
                    self.bus._subscribers.setdefault(name, []).append(info.queue)
            self.bus._modules.add(name)
            # reply_only modules skip dst="all" broadcasts — they only receive
            # targeted messages (QUERY dst=name, MODULE_SHUTDOWN, etc.)
            if info.spec.reply_only:
                self.bus._reply_only.add(name)

            # Spawn process
            try:
                proc = ctx.Process(
                    target=_module_wrapper,
                    args=(
                        info.spec.entry_fn,
                        name,
                        info.queue,
                        info.send_queue,
                        info.spec.config,
                        info.spec.start_method,
                        info.spec.broadcast_topics,
                        info.spec.reply_only,  # SPEC §8.2 v1.4.0 D-SPEC-42
                    ),
                    name=f"titan-{name}",
                    daemon=True,
                )
                proc.start()
                info.process = proc
                info.pid = proc.pid
                info.state = ModuleState.STARTING
                info.start_time = time.time()
                info.last_heartbeat = time.time()
                logger.info("[Guardian] Started module '%s' (pid=%d)", name, proc.pid)
                return True
            except Exception as e:
                logger.error("[Guardian] Failed to start module '%s': %s", name, e)
                info.state = ModuleState.CRASHED
                return False

    def _cleanup_module(self, name: str) -> None:
        """Clean up a module's state, force-killing any surviving process and its children."""
        info = self._modules.get(name)
        if not info:
            return

        # Microkernel v2 Phase B.2.1 — adopted workers don't own .process.
        # Use os.kill SIGTERM → 2s grace → os.kill SIGKILL. Gentler than the
        # pgid-based path below; gives the worker's graceful SIGTERM handlers
        # (flush WAL / release locks) a chance to run. We then fall through
        # to the queue + state cleanup at the end (same path for both).
        if getattr(info, 'adopted', False):
            self._kill_adopted_process(info, name)
        else:
            self._kill_owned_process(info, name)
        self._finalize_module_cleanup(info, name)

    def _kill_adopted_process(self, info: ModuleInfo, name: str) -> None:
        """Phase B.2.1 — gentle SIGTERM → 2s grace → SIGKILL for adopted workers."""
        if info.pid is None:
            return
        try:
            os.kill(info.pid, signal.SIGTERM)
        except ProcessLookupError:
            return  # already gone
        except OSError as e:
            logger.warning("[Guardian] adopted '%s' SIGTERM failed: %s", name, e)
            return
        # 2s graceful grace
        deadline = time.time() + 2.0
        while time.time() < deadline:
            try:
                os.kill(info.pid, 0)
            except ProcessLookupError:
                return  # exited cleanly
            time.sleep(0.1)
        # Still alive — force
        try:
            os.kill(info.pid, signal.SIGKILL)
            logger.info("[Guardian] adopted '%s' (pid=%s) SIGKILL after grace", name, info.pid)
        except ProcessLookupError:
            pass

    def _kill_owned_process(self, info: ModuleInfo, name: str) -> None:
        """Pre-B.2.1 cleanup path — extracted from _cleanup_module 2026-04-27.

        Force-kill any surviving child processes of the worker.
        CRITICAL: must NOT killpg if the worker shares our process group
        (which it does today — no setpgrp in worker startup). A naive
        killpg(worker_pgid) when worker_pgid == our_pgid is parent-suicide:
        titan_hcl dies and the API goes dark until watchdog reboot.
        Observed 2026-04-14 on T1 when spirit worker didn't respond to
        SIGTERM in 15s — killpg fired and took titan_hcl with it.
        Safe pattern: only killpg if worker is in a DIFFERENT group.
        """
        if info.pid is not None:
            try:
                worker_pgid = os.getpgid(info.pid)
                my_pgid = os.getpgid(0)
                if worker_pgid != my_pgid:
                    os.killpg(worker_pgid, signal.SIGKILL)
                    logger.info("[Guardian] Killed process group for '%s' (pid=%s, pgid=%s)",
                                name, info.pid, worker_pgid)
                # If same pgid: rely on info.process.kill() below to target
                # the worker only. Orphaned grandchildren (e.g. DuckDB procs)
                # get reparented to init and exit cleanly on their own.
            except (ProcessLookupError, PermissionError, OSError):
                pass
        if info.process is not None:
            try:
                if info.process.is_alive():
                    logger.warning("[Guardian] Cleanup: force-killing surviving process '%s' (pid=%s)", name, info.pid)
                    info.process.kill()
                    info.process.join(timeout=3.0)
                # Reap zombie even if not alive
                info.process.close()
            except Exception:
                pass

    def _finalize_module_cleanup(self, info: ModuleInfo, name: str) -> None:
        """Queue + state cleanup — runs after both adopted-kill and owned-kill paths.

        Queue cleanup uses cancel_join_thread() instead of join_thread().
        Why: the consumer child was SIGKILL'd above, but forked siblings
        may still hold inherited read-end FDs on the pipe (phantom FDs),
        keeping the pipe open. A pending os.write() inside the queue's
        feeder thread then blocks indefinitely on a full pipe instead of
        receiving EPIPE, deadlocking Guardian's asyncio loop (observed
        2026-04-14 on T1, cascading API hang; matches I-018 on T2).
        cancel_join_thread() is the documented Python fix for exactly
        this case — we accept the loss of any unflushed bytes, which
        were destined for a SIGKILL'd process anyway.
        """
        if info.queue:
            self.bus.unsubscribe(name, info.queue)
            try:
                info.queue.cancel_join_thread()
                info.queue.close()
            except Exception:
                pass
        if info.send_queue:
            try:
                info.send_queue.cancel_join_thread()
                info.send_queue.close()
            except Exception:
                pass
        if name in self._module_recv_queues:
            del self._module_recv_queues[name]
        info.state = ModuleState.STOPPED
        info.process = None
        info.pid = None
        info.queue = None
        info.send_queue = None
        # B.2.1 — clear adoption sentinel on cleanup (next start_module
        # via spawn would set adopted=False fresh anyway, but be explicit).
        info.adopted = False
        info.adopt_ts = 0.0

    def stop(self, name: str, reason: str = "requested",
             save_first: bool = True, save_timeout: float = 30.0) -> None:
        """Gracefully stop a module (SAVE_NOW → SAVE_DONE → SIGTERM → wait → SIGKILL).

        Thread-safe via _module_lock.

        Args:
            save_first: If True (default), publish SAVE_NOW and wait for the
                module to publish SAVE_DONE (or save_timeout) before sending
                MODULE_SHUTDOWN. Modules that don't handle SAVE_NOW (most
                workers as of 2026-04-13) simply ignore it; we still wait
                briefly then proceed. Modules that DO handle it (spirit_worker
                today; expandable to others) get a clean checkpoint window.
            save_timeout: Max seconds to wait for SAVE_DONE.
        """
        with self._module_lock:
            info = self._modules.get(name)
            if not info or not info.process:
                return

            logger.info("[Guardian] Stopping module '%s' (reason: %s, save_first=%s)",
                        name, reason, save_first)

            # ── Phase 1: graceful checkpoint via SAVE_NOW ──
            # Publish SAVE_NOW and drain the guardian queue looking for the
            # matching SAVE_DONE. If it doesn't come within save_timeout,
            # proceed with shutdown anyway — better to lose unsaved state than
            # to hang the restart pipeline.
            if save_first and info.process.is_alive():
                import uuid as _uuid
                save_rid = _uuid.uuid4().hex[:8]
                try:
                    self.bus.publish(make_msg(
                        bus.SAVE_NOW, "guardian", name,
                        {"module": name, "request_id": save_rid,
                         "reason": reason}))
                except Exception as _pub_err:
                    logger.warning("[Guardian] SAVE_NOW publish failed: %s",
                                   _pub_err)
                # Drain guardian queue waiting for SAVE_DONE matching rid.
                #
                # 2026-05-02 (Option A) — process MODULE_HEARTBEAT and
                # MODULE_READY INLINE so OTHER modules don't appear stale
                # while we're blocked here. Pre-fix behavior was to stash
                # all non-matching messages in `drained_msgs` and
                # re-publish them at the end (40+ seconds later under
                # multi-restart cascade). During those seconds,
                # `info.last_heartbeat` for other modules wasn't being
                # updated → Guardian falsely concluded they had timed out
                # → triggered MORE restarts → cascade.
                #
                # Heartbeats and READYs are idempotent state updates;
                # processing them inline here is safe and prevents the
                # cascade. Other message types (SAVE_DONE for OTHER
                # modules, BUS_PEER_DIED, BUS_WORKER_ADOPT_REQUEST) are
                # rare during SAVE_NOW wait and still re-published at the
                # end for the main `_process_guardian_messages` to handle.
                #
                # See BUG-GUARDIAN-STOP-SAVE-NOW-HEARTBEAT-CASCADE-20260502.
                save_deadline = time.time() + save_timeout
                save_done_seen = False
                drained_msgs: list = []  # re-enqueue non-heartbeat-class msgs
                inline_processed = 0
                while time.time() < save_deadline:
                    try:
                        m = self._guardian_queue.get(timeout=0.5)
                    except Exception:
                        continue
                    _mt = m.get("type")
                    if (_mt == bus.SAVE_DONE
                            and m.get("payload", {}).get("module") == name
                            and m.get("payload", {}).get("request_id") == save_rid):
                        _p = m.get("payload", {})
                        logger.info("[Guardian] SAVE_DONE from '%s': "
                                    "saved=%s errors=%s (%dms)",
                                    name, _p.get("saved"), _p.get("errors"),
                                    _p.get("duration_ms", 0))
                        save_done_seen = True
                        break
                    # Inline heartbeat/ready handler (Option A) — no need
                    # to round-trip through the queue + monitor_tick later.
                    if _mt == MODULE_HEARTBEAT:
                        _src = m.get("src", "")
                        _info = self._modules.get(_src)
                        if _info is not None:
                            _info.last_heartbeat = time.time()
                            _rss = m.get("payload", {}).get("rss_mb", 0)
                            if _rss:
                                _info.rss_mb = _rss
                            inline_processed += 1
                        continue  # don't stash; consumed inline
                    # Other types (rare during SAVE_NOW): stash for re-publish
                    drained_msgs.append(m)
                # Re-publish remaining drained messages so we don't lose them
                for _dm in drained_msgs:
                    try:
                        self.bus.publish(_dm)
                    except Exception:
                        pass
                if inline_processed:
                    logger.info(
                        "[Guardian] processed %d inline heartbeat/ready msg(s) "
                        "during '%s' SAVE_NOW wait (Option A — prevents "
                        "cascade-restart starvation)",
                        inline_processed, name)
                if not save_done_seen:
                    logger.info("[Guardian] No SAVE_DONE from '%s' within "
                                "%.1fs — module may not handle SAVE_NOW; "
                                "proceeding with SHUTDOWN (post-loop cleanup "
                                "still runs as fallback save)",
                                name, save_timeout)

            # ── Phase 2: shutdown signal via bus ──
            # D-SPEC-93 / §11.B.3.1 pid-targeting: pin the shutdown to the OLD
            # pid so a freshly-spawned same-name worker — which subscribes
            # under the same dst=name and may receive this still-buffered frame
            # after it boots — DROPS it (bus_socket._handle_inbound) instead of
            # self-exiting on a restart it is not the target of. Without this,
            # restart = stop(broadcast SHUTDOWN) → start(new pid) → new pid
            # honors the old SHUTDOWN → exits → heartbeat stale → restart loop.
            # Mirrors reload.py's targeted shutdown.
            self.bus.publish(make_msg(MODULE_SHUTDOWN, "guardian", name, {
                "reason": reason, "target_pid": info.process.pid}))

            # SIGTERM first — bumped 5s → 15s 2026-04-13 to give post-loop
            # cleanup enough time to complete (FAISS save can take 3-5s alone).
            try:
                if info.process.is_alive():
                    info.process.terminate()
                    info.process.join(timeout=15.0)
            except Exception:
                pass

            # SIGKILL if still alive. Same pgid-guard as _cleanup_module —
            # never killpg our own process group (parent-suicide).
            try:
                if info.process.is_alive():
                    logger.warning("[Guardian] Module '%s' didn't terminate, sending SIGKILL", name)
                    try:
                        import signal
                        worker_pgid = os.getpgid(info.pid)
                        my_pgid = os.getpgid(0)
                        if worker_pgid != my_pgid:
                            os.killpg(worker_pgid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError, OSError):
                        pass
                    info.process.kill()  # SIGKILL to worker PID only
                    info.process.join(timeout=3.0)
            except Exception:
                pass

            # Reap the process object to prevent zombies
            try:
                if info.process is not None:
                    info.process.close()
            except Exception:
                pass

            # Cleanup
            self._cleanup_module(name)

    def restart_async(self, name: str, reason: str = "requested"):
        """Option B (2026-05-02) — schedule restart() on a separate executor
        thread so monitor_tick doesn't block.

        Idempotent: if a restart for this module is already in flight,
        returns the existing Future (does NOT submit a duplicate). This
        prevents monitor_tick from queueing the same restart twice while
        the first is still mid-stop().

        Returns Future for the restart task, or None if this module is
        already restarting (caller should treat None as "in flight, fine").

        SPEC §11.B.3 (D-SPEC-49) — if a per-module hot-reload is in flight
        for this name, restart_async returns None and emits a debug log.
        The reload orchestrator owns OLD/NEW lifecycle until it reaches a
        terminal status. Bounded by MODULE_RELOAD_DEFAULT_TIMEOUT_S so this
        suppression is always recoverable.

        See BUG-GUARDIAN-STOP-SAVE-NOW-HEARTBEAT-CASCADE-20260502 +
        Guardian.__init__ comment block on _restart_executor for context.
        """
        info = self._modules.get(name)
        if info is not None and info.reload_in_flight:
            logger.debug(
                "[Guardian] restart_async('%s', reason=%s) skipped — "
                "reload in flight (SPEC §11.B.3)", name, reason)
            return None
        with self._restart_lock:
            if name in self._restarts_in_flight:
                logger.debug(
                    "[Guardian] restart_async('%s', reason=%s) skipped — "
                    "already in flight", name, reason)
                return None
            self._restarts_in_flight.add(name)

        def _run() -> bool:
            try:
                return self.restart(name, reason=reason)
            finally:
                with self._restart_lock:
                    self._restarts_in_flight.discard(name)

        return self._restart_executor.submit(_run)

    def restart(self, name: str, reason: str = "requested") -> bool:
        """Stop then start a module, with sliding-window restart limit. Thread-safe via _module_lock.

        Phase C C-S7 (2026-05-05) cross-language unification: emits
        SUPERVISION_CHILD_DOWN before stop + SUPERVISION_CHILD_RESTARTED
        after successful start. When max_restarts_in_window is exceeded,
        runs the SPEC §11.B.1 escalation handshake (in-process via
        kernel_default_decision — same simplification as the Rust kernel
        applies for kernel-self escalations).
        """
        with self._module_lock:
            info = self._modules.get(name)
            if not info:
                return False

            now = time.time()

            # Phase C C-S7 — pre-respawn dependency check (§11.G.2). Only
            # runs when the module declares dependencies; default-empty
            # path is byte-identical to today.
            if info.spec.dependencies:
                blocked_dep = self._check_critical_dependencies(name, info)
                if blocked_dep is not None:
                    info.blocked_dependency = blocked_dep
                    if info.blocked_since == 0.0:
                        info.blocked_since = now
                    self.bus.publish(make_msg(
                        SUPERVISION_DEPENDENCY_BLOCKED, "guardian", "kernel", {
                            "child_name": name,
                            "supervisor": "guardian_HCL",
                            "blocked_dependency": blocked_dep,
                            "since_ts": info.blocked_since,
                            "since_s": now - info.blocked_since,
                        }))
                    logger.warning(
                        "[Guardian] Module '%s' respawn blocked on critical "
                        "dependency '%s' (since %.0fs ago)",
                        name, blocked_dep, now - info.blocked_since)
                    return False
                # All critical deps OK — clear blocked state if previously set.
                if info.blocked_dependency is not None:
                    self.bus.publish(make_msg(
                        SUPERVISION_DEPENDENCY_RECOVERED, "guardian", "kernel", {
                            "child_name": name,
                            "supervisor": "guardian_HCL",
                            "dependency_name": info.blocked_dependency,
                            "total_blocked_s": now - info.blocked_since,
                        }))
                    info.blocked_dependency = None
                    info.blocked_since = 0.0

            # Sliding window: count restarts in the last _restart_window_seconds
            # Prune old timestamps
            while info.restart_timestamps and (now - info.restart_timestamps[0]) > self._restart_window_seconds:
                info.restart_timestamps.popleft()

            if len(info.restart_timestamps) >= self._max_restarts_in_window:
                # Phase C C-S7 — escalation handshake per SPEC §11.B.1.
                # In-process short-circuit via kernel_default_decision (same
                # approach the Rust kernel takes for its own children).
                self._handle_escalation(name, info, reason, now)
                return False

            # Phase C C-S7 — emit SUPERVISION_CHILD_DOWN BEFORE stop.
            # Reason classified from the `reason` string (which today is
            # things like "heartbeat_timeout", "rss_400mb", etc.). Best-effort
            # mapping to canonical SupervisionReason — exit-code-based mapping
            # happens in monitor_tick where exitcode is observable.
            sup_reason = self._reason_string_to_canonical(reason)
            info.reason_buffer.append(ReasonRecord.make(
                sup_reason, detail=f"reason={reason}", exit_code=None))
            self.bus.publish(make_msg(
                SUPERVISION_CHILD_DOWN, "guardian", "kernel", {
                    "child_name": name,
                    "supervisor": "guardian_HCL",
                    "reason": sup_reason.value,
                    "reason_detail": reason,
                    "restart_count": len(info.restart_timestamps),
                    "ts": now,
                }))

            # Exponential backoff based on recent restart count
            recent_count = len(info.restart_timestamps)
            backoff = RESTART_BACKOFF_BASE ** min(recent_count, 5)  # cap at 32s
            since_last = now - info.last_restart if info.last_restart > 0 else float('inf')
            if since_last < backoff:
                logger.debug("[Guardian] Module '%s' restart backoff (%.1fs remaining)",
                             name, backoff - since_last)
                return False

            self.stop(name, reason=f"restart:{reason}")
            info.restart_timestamps.append(now)
            info.restart_count += 1
            info.last_restart = now
            ok = self.start(name)
            if ok:
                # Phase C C-S7 — emit SUPERVISION_CHILD_RESTARTED on success.
                self.bus.publish(make_msg(
                    SUPERVISION_CHILD_RESTARTED, "guardian", "kernel", {
                        "child_name": name,
                        "supervisor": "guardian_HCL",
                        "restart_count": info.restart_count,
                        "reason": sup_reason.value,
                        "ts": time.time(),
                    }))
            return ok

    def restart_module(self, name: str, reason: str = "requested",
                       start_method: str | None = None) -> dict:
        """Public, kernel_rpc-exposable wrapper around `restart()`.

        Closes BUG-HOT-RELOAD-CODE-LOADING: the legacy
        `/v4/admin/restart-module/{name}` endpoint called `guardian.restart`
        + `guardian._modules` directly, but in V6 api_subprocess mode those
        names go through `GuardianAccessor` which has neither — silent
        no-op (`restart`) and AttributeError (`_modules`). This method
        does the entire flow in the parent process and returns a dict
        the api_subprocess can serialize directly.

        Parameters:
          name: module name (must be in caller's allowlist)
          reason: free-form audit string
          start_method: optional one-shot override ("fork" or "spawn").
                        When set, temporarily flips info.spec.start_method
                        for this single restart and restores after — lets
                        callers force a fresh-interpreter spawn (true
                        code reload from disk) without persistent config
                        drift. None = use whatever spec was registered with.

        Returns:
          {"ok": bool, "module": name, "restart_initiated": bool,
           "process_alive": bool, "start_method_used": str | None,
           "error": str | None}
        """
        info = self._modules.get(name)
        if info is None:
            return {"ok": False, "module": name, "restart_initiated": False,
                    "process_alive": False, "start_method_used": None,
                    "error": f"unknown module: {name}"}

        original_method = None
        if start_method is not None and start_method != info.spec.start_method:
            if start_method not in ("fork", "spawn"):
                return {"ok": False, "module": name, "restart_initiated": False,
                        "process_alive": False, "start_method_used": None,
                        "error": f"invalid start_method: {start_method!r} "
                                 f"(must be 'fork' or 'spawn')"}
            original_method = info.spec.start_method
            info.spec.start_method = start_method
            logger.info(
                "[Guardian] restart_module %s: start_method override "
                "%r → %r (one-shot)", name, original_method, start_method)

        try:
            ok = self.restart(name, reason=reason)
        finally:
            if original_method is not None:
                info.spec.start_method = original_method

        # Re-fetch info; restart may have mutated process refs
        info = self._modules.get(name)
        is_alive = bool(info and info.process and info.process.is_alive())
        return {
            "ok": bool(ok),
            "module": name,
            "restart_initiated": bool(ok),
            "process_alive": is_alive,
            "start_method_used": start_method or (info.spec.start_method if info else None),
            "error": None,
        }

    def _write_disabled_to_slot(self, name: str) -> None:
        """Publish the terminal `disabled` state to a module's SHM slot after
        the orchestrator disables it (SPEC §11.I.4).

        The worker was just stopped (pid dead) so this is G21-safe. Without it,
        guardian_hcl's 1Hz SHM poll keeps reading the slot's stale
        `running`/`booted` + dead pid → endlessly re-requests restarts of a
        module the orchestrator already gave up on. Writing `disabled` makes the
        supervisor skip it cleanly (`sstate in ('stopped','disabled') → continue`).
        """
        try:
            from titan_hcl.core.module_state import write_terminal_module_state
            from titan_hcl.core.state_registry import resolve_titan_id
            write_terminal_module_state(
                name, "disabled", titan_id=resolve_titan_id())
        except Exception as e:  # noqa: BLE001 — best-effort terminal write
            logger.debug(
                "[Orchestrator] disabled-slot write for '%s' failed: %s",
                name, e)

    def _handle_escalation(
        self, name: str, info: "ModuleInfo", reason: str, now: float,
    ) -> None:
        """SPEC §11.B.1 escalation handshake. In-process short-circuit via
        kernel_default_decision (same as Rust kernel's kernel-self path).

        Emits SUPERVISION_ESCALATION (audit record + kernel signal) then
        applies the policy decision directly:
          - CONTINUE → reset counter; module retries on next monitor_tick
          - TERMINATE → disable the offending module locally and signal the
            kernel via SUPERVISION_ESCALATION. The orchestrator NEVER
            self-terminates: per the Phase 11 role split (§11.I) only the
            kernel (L0, Rust) may kill/restart the titan_hcl orchestrator peer;
            self-exiting here would take down all sibling modules. (Supersedes
            the pre-split exit-64 self-terminate reading of §11.B.1 step 6b.)
          - HALT → disable module; Maker must intervene
        """
        import uuid
        escalation_id = str(uuid.uuid4())
        info.last_escalation_id = escalation_id
        # Compute most_common_reason from per-child rolling buffer.
        common_reason = most_common_reason(info.reason_buffer)
        last_detail = (
            info.reason_buffer[-1].detail
            if info.reason_buffer else f"reason={reason}"
        )
        reasons_observed = [r.reason.value for r in info.reason_buffer]

        # Audit emit — kernel writes this to supervision.jsonl when it
        # subscribes to SUPERVISION_ESCALATION (today: live observability).
        self.bus.publish(make_msg(
            SUPERVISION_ESCALATION, "guardian", "kernel", {
                "escalation_id": escalation_id,
                "child_name": name,
                "supervisor_name": "guardian_HCL",
                "restart_count": len(info.restart_timestamps),
                "window_s": self._restart_window_seconds,
                "reasons_observed": reasons_observed,
                "most_common_reason": common_reason.value,
                "last_reason_detail": last_detail,
                "ts": now,
            }))

        # Apply default policy in-process (Maker-confirmed simplification —
        # bus round-trip would time out anyway since Rust kernel doesn't
        # implement an escalation-response server yet).
        decision = kernel_default_decision(common_reason)
        logger.error(
            "[Guardian] Module '%s' [%s] hit max_restarts (%d in %.0fs) — "
            "escalation %s most_common_reason=%s decision=%s",
            name, info.spec.layer, len(info.restart_timestamps),
            self._restart_window_seconds, escalation_id,
            common_reason.value, decision.value,
        )

        if decision == EscalationDecision.CONTINUE:
            # Reset counter so module gets fresh restart budget. Keep state
            # RUNNING/CRASHED depending on current; Guardian's monitor_tick
            # will respawn naturally.
            info.restart_timestamps.clear()
            info.reason_buffer.clear()
            logger.warning(
                "[Guardian] Continue policy — resetting restart counter "
                "for '%s'; will retry on next monitor_tick", name)
        elif decision == EscalationDecision.TERMINATE:
            # Phase 11 role split (§11.I): the titan_hcl orchestrator has NO
            # authority to terminate itself on a module escalation — doing so
            # would take down all sibling modules. Only the kernel (L0, Rust)
            # may kill/restart the orchestrator peer. The SUPERVISION_ESCALATION
            # emitted above IS the kernel's signal; the orchestrator's local
            # action is to disable the offending module so it stops looping.
            # The kernel, as sole owner of orchestrator lifecycle, recycles the
            # peer if its own policy decides to. (Supersedes the legacy
            # exit-64 "supervisor terminates self" reading of §11.B.1
            # step 6b, which predates the orchestrator/supervisor split.)
            logger.critical(
                "[Guardian] Terminate escalation for '%s' [%s] — disabling "
                "module locally; orchestrator does NOT self-exit (kernel holds "
                "sole authority to recycle the orchestrator per §11.I). "
                "escalation_id=%s", name, info.spec.layer, escalation_id)
            self.stop(name, reason="escalation_terminate")
            info.state = ModuleState.DISABLED
            info.disabled_at = now
            self._write_disabled_to_slot(name)
            self.bus.publish(make_msg(MODULE_CRASHED, "guardian", "core", {
                "module": name, "reason": "escalation_terminate",
                "restarts": len(info.restart_timestamps),
                "window_seconds": self._restart_window_seconds,
                "escalation_id": escalation_id,
            }))
        else:  # HALT
            self.stop(name, reason="escalation_halt")
            info.state = ModuleState.DISABLED
            info.disabled_at = now
            self._write_disabled_to_slot(name)
            self.bus.publish(make_msg(MODULE_CRASHED, "guardian", "core", {
                "module": name, "reason": "escalation_halt",
                "restarts": len(info.restart_timestamps),
                "window_seconds": self._restart_window_seconds,
                "escalation_id": escalation_id,
            }))
            logger.warning(
                "[Guardian] Halt policy — '%s' disabled; Maker must "
                "intervene (auto-re-enable in %.0fs)",
                name, REENABLE_COOLDOWN_S)

    def enable(self, name: str) -> bool:
        """Re-enable a disabled module, reset restart counters, and start it. Thread-safe via _module_lock."""
        with self._module_lock:
            info = self._modules.get(name)
            if not info:
                logger.error("[Guardian] Cannot enable unknown module '%s'", name)
                return False
            if info.state != ModuleState.DISABLED:
                logger.info("[Guardian] Module '%s' is not disabled (state=%s)", name, info.state)
                return True  # already enabled
            logger.info("[Guardian] Re-enabling module '%s' — resetting restart counters", name)
            info.state = ModuleState.STOPPED
            info.restart_count = 0
            info.restart_timestamps.clear()
            return self.start(name)

    def _compute_boot_order(self, names: list[str]) -> list[str]:
        """Phase 9 Chunk 9L — order autostart modules: dependency-correct
        topological sort, then layer-ascending (L1→L2→L3) + alphabetic
        within each topo level. Returns a flat list — Guardian.start_all
        walks it with `boot_stagger_delay_s` between calls.

        Algorithm:
          1. Topological sort honoring ModuleSpec.dependencies (SPEC §11.G).
             A module ships only after every MODULE-kind dep ships.
          2. Within each topo level, sort by (ModuleSpec.layer L1→L2→L3,
             then name) so light infra fires before heavy workers, then
             alphabetic for determinism.

        Cycle protection: §11.G.7 enforces DAG via arch_map; if a cycle is
        still present at runtime, fall back to alphabetic and log WARN —
        boot continues correctness-first.
        """
        name_set = set(names)
        incoming: dict[str, set[str]] = {n: set() for n in names}
        for n in names:
            spec = self._modules[n].spec
            for dep in spec.dependencies:
                if dep.kind == DependencyKind.MODULE and dep.name in name_set:
                    incoming[n].add(dep.name)

        ordered: list[str] = []
        remaining = set(names)
        while remaining:
            ready = [n for n in remaining if not (incoming[n] & remaining)]
            if not ready:
                logger.warning(
                    "[Guardian] boot dep cycle detected among %s — "
                    "flat-ordering remaining modules", sorted(remaining))
                ready = sorted(remaining)
            ready.sort(key=lambda n: (self._modules[n].spec.layer, n))
            ordered.extend(ready)
            for n in ready:
                remaining.discard(n)
        return ordered

    # ── Phase 11 §11.I.7 — BootPriority partition helper ──────────────

    def _partition_autostart_by_boot_priority(
        self,
        names: list[str],
    ) -> tuple[list[str], list[str], list[str]]:
        """Phase 11 §11.I.8 — split autostart names into
        (mandatory, post_boot, lazy) buckets per ModuleSpec.boot_priority.

        Returns three lists; each preserves the input ordering so the
        downstream `_compute_boot_order` topological sort + layer
        ordering applies the same way per bucket.

        Per §11.I.8: any LAZY autostart module is logged as
        warning-then-skipped — LAZY by definition is NEVER auto-started
        (`autostart=True + boot_priority=lazy` is a config mistake; we
        do not silently re-classify it).
        """
        mandatory: list[str] = []
        post_boot: list[str] = []
        lazy: list[str] = []
        for n in names:
            spec = self._modules[n].spec
            bp = (spec.boot_priority or "mandatory").lower()
            if bp == "mandatory":
                mandatory.append(n)
            elif bp == "post_boot":
                post_boot.append(n)
            elif bp == "lazy":
                lazy.append(n)
                logger.warning(
                    "[Orchestrator] Module '%s' has autostart=True + "
                    "boot_priority=lazy — Phase 11 §11.I.8 requires LAZY "
                    "modules to be pre-activated via §11.G.2.5 instead. "
                    "Skipping autostart.", n)
            else:
                logger.warning(
                    "[Orchestrator] Module '%s' has unknown boot_priority="
                    "%r — defaulting to MANDATORY (Phase 11 §11.I.8)",
                    n, spec.boot_priority)
                mandatory.append(n)
        return mandatory, post_boot, lazy

    # ── Phase 11 §11.I.7 / RFP 11D — continuous probe poller (sole prober) ──

    def _maybe_dispatch_probe(self, name: str, sstate: str, now: float) -> None:
        """Dispatch MODULE_PROBE_REQUEST to `name` iff its slot is at `booted`
        and the per-module re-fire window has elapsed (SPEC §11.I.2 / §11.I.7).

        Shared by the continuous poller AND `_wait_for_module_running` /
        `_wait_for_wave_running` via the single `self._probe_dispatched_at`
        throttle map, so the two paths can never double-probe: whichever fires
        first stamps the timestamp; the other respects the window. Fire-and-
        forget publish (non-blocking, §8.0.ter); the SHM slot's transition to
        `running` is the authoritative readiness signal (locked D1), so no bus
        reply is awaited here.
        """
        if sstate != "booted":
            # Not awaiting a probe (starting / probing / running / terminal):
            # clear the marker so a future re-boot of this slot re-probes
            # without waiting out a stale window.
            self._probe_dispatched_at.pop(name, None)
            return
        last = self._probe_dispatched_at.get(name, 0.0)
        if now - last < self._probe_refire_interval_s:
            return
        self._probe_dispatched_at[name] = now
        try:
            self.bus.publish(make_msg(
                MODULE_PROBE_REQUEST, "titan_hcl", name,
                {"name": name, "probe_id": uuid.uuid4().hex}))
        except Exception:  # noqa: BLE001 — probe publish must never crash poll
            pass

    def _ensure_probe_poller_started(self) -> None:
        """Start the continuous 1Hz probe poller thread once (idempotent).

        Called at the top of start_all. Only the titan_hcl orchestrator process
        calls start_all, so the poller is exclusive to it (guardian_hcl builds a
        metadata-only Orchestrator and never calls start_all → never probes).
        """
        if (self._probe_poller_thread is not None
                and self._probe_poller_thread.is_alive()):
            return
        self._probe_poller_stop.clear()
        t = threading.Thread(
            target=self._probe_poll_loop,
            name="orchestrator-probe-poller",
            daemon=True,
        )
        self._probe_poller_thread = t
        t.start()
        logger.info(
            "[Orchestrator] probe poller started — 1Hz SHM scan, probes any "
            "module observed at state=booted (SPEC §11.I.7 / RFP 11D)")

    def _probe_poll_loop(self) -> None:
        """1Hz loop: SPEC §11.I.7 'titan_hcl detects BOOTED via 1Hz SHM poll →
        emits MODULE_PROBE_REQUEST', applied continuously (boot + steady-state).
        Also refreshes titan_hcl_state.bin live fleet counters (Fix 4). Errors
        are swallowed + throttled — the poller must NEVER die (a dead poller =
        modules stuck booted again)."""
        bank = self._ensure_module_state_reader_bank()
        if bank is None:
            logger.info(
                "[Orchestrator] probe poller idle — no SHM reader bank "
                "(test fixture / no /dev/shm); modules self-attest via boot wait")
            return
        last_err_log = 0.0
        while not self._probe_poller_stop.wait(self._probe_poll_interval_s):
            try:
                self._probe_poll_tick(bank)
            except Exception as exc:  # noqa: BLE001
                now = time.time()
                if now - last_err_log > 60.0:
                    logger.warning(
                        "[Orchestrator] probe poll tick failed: %s", exc)
                    last_err_log = now

    def _probe_poll_tick(self, bank) -> None:
        """One poll pass: probe every `booted` slot + recompute live fleet
        counters from SHM truth (kills the boot-tally vs SHM divergence)."""
        now = time.time()
        mandatory_running = 0
        post_boot_running = 0
        for name, info in list(self._modules.items()):
            # Intentionally-down modules own no live slot to probe.
            if info.state == ModuleState.DISABLED:
                continue
            try:
                entry = bank.read(name)
            except Exception:  # noqa: BLE001 — a bad read must not stop the scan
                entry = None
            sstate = (entry.state if entry is not None else "") or ""
            # Fix 1 — probe any booted slot (boot, post-restart, lazy-activate).
            self._maybe_dispatch_probe(name, sstate, now)
            # Fix 4 — live fleet counters from SHM, not boot-time tally.
            if sstate == "running":
                bp = (info.spec.boot_priority or "mandatory").lower()
                if bp == "mandatory":
                    mandatory_running += 1
                elif bp == "post_boot":
                    post_boot_running += 1
        writer = getattr(self, "_titan_hcl_state_writer", None)
        if writer is not None:
            try:
                # Only the live counters — fleet_ready / boot_phase remain the
                # boot-phase latches set by start_all / _run_phase_b.
                writer.update(
                    mandatory_ready=mandatory_running,
                    post_boot_ready=post_boot_running)
            except Exception:  # noqa: BLE001
                pass

    # ── Phase 11 §11.I.7 / RFP 11F — wave-based boot ──────────────────────

    def _wait_for_wave_running(
        self, names: list[str], timeout_s: float,
    ) -> int:
        """Wait until every module in `names` shows state=running in SHM (or
        timeout). The continuous poller drives booted→running concurrently;
        this just observes + mirrors SHM truth into in-process ModuleInfo. Also
        nudges a probe per pass (shared throttle → no double-probe) so wave
        readiness never depends on poll-loop timing. Returns the count that
        reached RUNNING.
        """
        bank = self._ensure_module_state_reader_bank()
        if bank is None:
            return len(names)  # no SHM (tests) — assume ready so boot proceeds
        deadline = time.time() + timeout_s
        pending = set(names)
        ready: set[str] = set()
        while pending and time.time() < deadline:
            now = time.time()
            for name in list(pending):
                info = self._modules.get(name)
                if info is None:
                    pending.discard(name)
                    continue
                try:
                    entry = bank.read(name)
                except Exception:  # noqa: BLE001
                    entry = None
                sstate = (entry.state if entry is not None else "") or ""
                self._maybe_dispatch_probe(name, sstate, now)
                if sstate == "running":
                    info.state = ModuleState.RUNNING
                    if info.ready_time == 0.0:
                        info.ready_time = now
                    ready.add(name)
                    pending.discard(name)
                elif sstate in ("disabled", "crashed"):
                    # Terminal for boot purposes — stop waiting (supervisor owns
                    # recovery). `unhealthy` is NOT terminal: the poller keeps
                    # re-probing; a transient probe failure may still recover.
                    pending.discard(name)
            if pending:
                time.sleep(0.1)
        return len(ready)

    def _spawn_capped(
        self, order: list[str], stagger: float, ready_timeout: float,
    ) -> int:
        """Spawn `order` (dep+layer topo) with BOUNDED CONCURRENCY, then wait
        for all to reach RUNNING (Maker 2026-05-28).

        Never more than `self._boot_concurrency_cap` modules are IN-FLIGHT
        (spawned but not yet state=running) at once: the 1Hz probe poller drives
        in-flight→running, and as each reaches running it frees a slot for the
        next spawn. This bounds the cold-boot CPU spike on a small box while
        keeping full boot fast. The gate has a bounded wait (`ready_timeout`) so
        a single stuck module can never deadlock the boot. Returns count RUNNING.

        cap <= 0 → uncapped (spawn all staggered). bank is None (test fixtures,
        no /dev/shm) → uncapped (the gate is a no-op there).
        """
        bank = self._ensure_module_state_reader_bank()
        cap = self._boot_concurrency_cap

        def _running(n: str) -> bool:
            if bank is None:
                return True
            try:
                e = bank.read(n)
                return bool(e) and e.state == "running"
            except Exception:  # noqa: BLE001
                return False

        spawned: list[str] = []
        for name in order:
            # Gate: bound in-flight (spawned, not yet running) to `cap`. Bounded
            # by ready_timeout so a stuck module can't stall the whole boot.
            if cap > 0 and bank is not None:
                gate_deadline = time.time() + ready_timeout
                while (time.time() < gate_deadline
                       and sum(1 for n in spawned if not _running(n)) >= cap):
                    time.sleep(0.3)
            if spawned and stagger > 0:
                time.sleep(stagger)
            self.start(name, activate_deps=False)  # poller drives readiness
            spawned.append(name)
        return self._wait_for_wave_running(order, ready_timeout)

    def _wait_for_module_running(
        self,
        name: str,
        timeout_s: Optional[float] = None,
    ) -> bool:
        """Phase 11 §11.I.2/§11.I.7 — drive the module through the probe
        contract and block until its SHM slot reports `state=running`.

        SHM is the SOLE readiness source (locked D1/D2 — the legacy
        MODULE_READY bus path is DELETED). Mechanics:
          1. Poll the worker's `module_<name>_state.bin` slot at 10 Hz.
          2. When the worker reaches `state=booted` it is awaiting a probe;
             we publish MODULE_PROBE_REQUEST (non-blocking, §8.0.ter, dst=worker).
             The worker's `handle_module_probe_request` runs its probe_fn (or
             trivial pass per §11.I.2) and writes `probing→running` to its OWN
             slot. We re-fire every 2s to beat the subscribe bootstrap race.
          3. On observed `state=running` we mirror it into the in-process
             ModuleInfo (state + ready_time) so the rest of the orchestrator
             stays consistent WITHOUT any bus broadcast.

        Returns True if the slot reached RUNNING within the timeout, else
        False. False does NOT stop the boot — the supervisor's SHM
        liveness/heartbeat detector catches a truly-stuck worker post-boot.
        """
        deadline = time.time() + (timeout_s if timeout_s is not None
                                  else self._probe_wait_timeout_s)
        reader_bank = self._ensure_module_state_reader_bank()
        if reader_bank is None:
            # No SHM (test fixtures without /dev/shm): cannot verify via the
            # canonical path. Treat as ready so boot proceeds; real fleets
            # always have SHM provisioned.
            return True
        while time.time() < deadline:
            info = self._modules.get(name)
            if info is None:
                return False
            try:
                entry = reader_bank.read(name)
            except Exception:  # noqa: BLE001
                entry = None
            sstate = entry.state if entry is not None else None
            if sstate == "running":
                # Mirror SHM truth into the in-process info dict (no bus event).
                info.state = ModuleState.RUNNING
                if info.ready_time == 0.0:
                    info.ready_time = time.time()
                return True
            # Drive the probe via the shared dispatcher (same throttle map as
            # the continuous poller → no double-probe). Used by dep-activation
            # (§11.G.2.5) + reload (§11.I.6) which may run before/independent of
            # a steady-state poll; the poller covers everything else.
            self._maybe_dispatch_probe(name, sstate or "", time.time())
            time.sleep(0.1)
        return False

    def _ensure_module_state_reader_bank(self):
        """Lazy-construct the per-Titan ModuleStateReaderBank.

        Cached on the Orchestrator instance. Returns None if the SHM
        root cannot be resolved (test fixtures without /dev/shm
        provisioning) — callers fall back to the in-process info path.
        """
        existing = getattr(self, "_module_state_reader_bank", None)
        if existing is not None:
            return existing
        try:
            from titan_hcl.core.module_state import ModuleStateReaderBank
            from titan_hcl.core.state_registry import resolve_titan_id
            titan_id = resolve_titan_id()
            bank = ModuleStateReaderBank(titan_id=titan_id)
        except Exception as e:  # noqa: BLE001
            logger.info(
                "[Orchestrator] module_state.bin reader bank unavailable "
                "(%s) — Phase 11 probe-wait falls back to in-process info",
                e)
            self._module_state_reader_bank = None
            return None
        self._module_state_reader_bank = bank
        return bank

    def _ensure_titan_hcl_state_writer(self):
        """Lazy-construct the orchestrator's own SHM state writer.

        Per SPEC G21 (one slot, one writer): only the canonical orchestrator
        process writes `titan_hcl_state.bin`. Any other process that
        constructs an Orchestrator (e.g. the api subprocess's mini-
        orchestrator, test fixtures, sub-supervisors) is rejected by env-var
        gate `TITAN_HCL_STATE_WRITER_CANONICAL=1` — set ONLY by
        scripts/guardian_hcl.py (today the canonical orchestrator owner;
        post-11E.b.2 physical split, scripts/titan_hcl.py) before calling
        start_all. Non-canonical callers get None back and silently skip
        the publish, leaving the slot to the canonical writer.

        Returns None on first-call failure (test fixtures, missing
        /dev/shm, non-canonical process) so start_all() proceeds without
        publishing `fleet_ready` (kernel-rs falls back to its existing
        process-health gate).
        """
        existing = getattr(self, "_titan_hcl_state_writer", None)
        if existing is not None:
            return existing
        # G21 single-writer enforcement.
        if os.environ.get("TITAN_HCL_STATE_WRITER_CANONICAL", "") != "1":
            logger.info(
                "[Orchestrator] titan_hcl_state.bin writer suppressed: "
                "TITAN_HCL_STATE_WRITER_CANONICAL!=1 in this process "
                "(G21 single-writer — only the canonical orchestrator "
                "publishes fleet_ready)")
            self._titan_hcl_state_writer = None
            return None
        try:
            from titan_hcl.core.titan_hcl_state import TitanHclStateWriter
            from titan_hcl.core.state_registry import resolve_titan_id
            titan_id = resolve_titan_id()
            writer = TitanHclStateWriter(titan_id=titan_id)
        except Exception as e:  # noqa: BLE001
            logger.info(
                "[Orchestrator] titan_hcl_state.bin writer unavailable "
                "(%s) — Phase 11 fleet_ready publication disabled "
                "(in-process boot continues)", e)
            self._titan_hcl_state_writer = None
            return None
        self._titan_hcl_state_writer = writer
        return writer

    def start_all(self) -> None:
        """Phase 11 §11.I.7 — orchestrator boot pipeline.

        Two-phase walk:

          * **Phase A — MANDATORY** (gates `fleet_ready=true`):
            partition autostart by `ModuleSpec.boot_priority == "mandatory"`
            → topological + layer-ordered + alphabetic per
            `_compute_boot_order` → spawn each with
            `boot_stagger_delay_s` (default 1.5s) between calls →
            probe-gated wait via `_wait_for_module_running` (caps at
            `probe_wait_timeout_s`, default 30s) so the next module
            does not spawn until the previous one reaches RUNNING.
          * **Phase B — OPTIONAL_POST_BOOT** (informational
            `fleet_optional_ready`): same walk on the post_boot bucket
            with `post_boot_stagger_delay_s` (default 5s) between
            spawns. Runs in a background daemon thread so start_all
            returns once Phase A is complete + fleet_ready is published.
          * **LAZY**: never auto-started; pre-activated on-demand by
            consumer dep-activation per §11.G.2.5.

        Phase 11 §11.I.7 SHM publication: after Phase A completes
        titan_hcl writes `fleet_ready=true` to its own
        `titan_hcl_state.bin` SHM slot (G21 single-writer). kernel-rs +
        guardian_hcl + observatory + api detect this via 1Hz SHM poll
        (NO bus broadcast — locked D1). Once Phase B completes,
        `fleet_optional_ready=true` is published (informational).

        Microkernel v2 Phase B.2.1 (2026-04-27 PM): when env var
        TITAN_B2_1_ADOPTION_PENDING=1 is set, spawn-mode autostart
        modules are SKIPPED here (Phase A AND Phase B). They reconnect
        to shadow's broker and register via BUS_WORKER_ADOPT_REQUEST →
        Orchestrator.adopt_worker.

        Back-compat: when `_phase_11_pipeline_enabled=False` (config
        knob, default True), falls back to the Phase 9 Chunk 9L flat
        staggered walk so legacy tests + alternate boot harnesses keep
        their behaviour bit-identical.
        """
        adoption_pending = (
            os.environ.get("TITAN_B2_1_ADOPTION_PENDING", "") == "1"
        )
        skipped_b2_1: list[str] = []
        autostart_names: list[str] = []
        for name, info in self._modules.items():
            if not info.spec.autostart:
                continue
            if adoption_pending and info.spec.start_method == "spawn":
                skipped_b2_1.append(name)
                continue
            autostart_names.append(name)
        if skipped_b2_1:
            logger.info(
                "[Orchestrator] B.2.1 adoption-pending: skipped autostart "
                "for spawn-mode workers (%d): %s — awaiting "
                "BUS_WORKER_ADOPT_REQUEST",
                len(skipped_b2_1), sorted(skipped_b2_1))

        # Back-compat: flat walk for fixtures that opt out of the
        # Phase 11 pipeline.
        if not self._phase_11_pipeline_enabled:
            order = self._compute_boot_order(autostart_names)
            start_all_t0 = time.time()
            logger.info(
                "[Orchestrator] start_all (legacy flat walk): %d modules; "
                "stagger %.1fs", len(order), self._boot_stagger_delay_s)
            for idx, name in enumerate(order, start=1):
                if idx > 1 and self._boot_stagger_delay_s > 0:
                    time.sleep(self._boot_stagger_delay_s)
                self.start(name)
            logger.info(
                "[Orchestrator] start_all (legacy) complete in %.1fs",
                time.time() - start_all_t0)
            return

        # Phase 11 §11.I.7 / §11.I.8 — partition autostart by boot_priority.
        mandatory, post_boot, _lazy_skipped = (
            self._partition_autostart_by_boot_priority(autostart_names))
        # Lazy bucket (autostart=True + boot_priority=lazy) was already
        # logged in the partition helper; nothing else to do here. LAZY
        # modules with autostart=False fall through §11.G.2.5 on first use.
        lazy_total = sum(
            1 for n, info in self._modules.items()
            if (info.spec.boot_priority or "").lower() == "lazy")

        writer = self._ensure_titan_hcl_state_writer()
        if writer is not None:
            writer.update(
                boot_phase="booting_a",
                mandatory_total=len(mandatory),
                post_boot_total=len(post_boot),
                lazy_total=lazy_total)

        # Continuous probe poller live BEFORE the first spawn so it drives
        # every wave's booted→running concurrently (SPEC §11.I.7 / RFP 11D).
        self._ensure_probe_poller_started()

        # Phase A — MANDATORY (non-blocking CONCURRENT boot — Maker 2026-05-28).
        # Spawn ALL mandatory modules staggered in dep+layer order WITHOUT
        # blocking between dependency waves; the continuous probe poller drives
        # every slot booted→running concurrently. Then ONE bounded readiness
        # wait for the whole mandatory set before publishing fleet_ready —
        # replacing the 3 sequential per-wave blocking waits that cost ~90s on
        # cold 4-core boot (the slowest module governed each wave, serially).
        # Dependency correctness holds via the topo+layer spawn ORDER + start()'s
        # recursive _activate_dependencies; the poller + supervisor finish any
        # straggler past the readiness wait. Restores monolith-class concurrent
        # boot while keeping Phase 11 probe-verified readiness.
        phase_a_t0 = time.time()
        order = self._compute_boot_order(mandatory)
        logger.info(
            "[Orchestrator] Phase A (MANDATORY) start: %d modules CONCURRENT "
            "(dep+layer order); stagger %.1fs; cap %d in-flight; readiness wait "
            "≤%.0fs", len(order), self._boot_stagger_delay_s,
            self._boot_concurrency_cap, self._probe_wait_timeout_s)
        ready_count = self._spawn_capped(
            order, self._boot_stagger_delay_s, self._probe_wait_timeout_s)
        phase_a_dt = time.time() - phase_a_t0
        if ready_count < len(order):
            logger.warning(
                "[Orchestrator] Phase A: %d/%d MANDATORY RUNNING in %.1fs "
                "(readiness wait ≤%.0fs) — publishing fleet_ready; poller keeps "
                "probing stragglers", ready_count, len(order), phase_a_dt,
                self._probe_wait_timeout_s)
        else:
            logger.info(
                "[Orchestrator] Phase A complete in %.1fs — %d/%d MANDATORY "
                "RUNNING; publishing fleet_ready",
                phase_a_dt, ready_count, len(order))
        if writer is not None:
            writer.update(
                fleet_ready=True,
                boot_phase="phase_a_done",
                mandatory_ready=ready_count)

        # Phase B — OPTIONAL_POST_BOOT in background daemon thread.
        if post_boot:
            t = threading.Thread(
                target=self._run_phase_b,
                args=(post_boot,),
                name="orchestrator-phase-b",
                daemon=True,
            )
            t.start()
            logger.info(
                "[Orchestrator] Phase B (OPTIONAL_POST_BOOT) scheduled in "
                "background: %d modules; stagger %.1fs; thread=%s",
                len(post_boot), self._post_boot_stagger_delay_s, t.name)
        else:
            if writer is not None:
                writer.update(
                    fleet_optional_ready=True,
                    boot_phase="quiescent")
            logger.info(
                "[Orchestrator] Phase B skipped — no OPTIONAL_POST_BOOT "
                "modules registered; quiescent")

    def _run_phase_b(self, names: list[str]) -> None:
        """Phase 11 §11.I.7 — Phase B (OPTIONAL_POST_BOOT) background
        walk. Same probe-gated semantics as Phase A but with the longer
        `post_boot_stagger_delay_s` between spawns. Runs in a daemon
        thread so start_all() can return once Phase A's fleet_ready
        publication completes.

        Errors during a single post-boot module spawn are logged + skipped
        (post-boot is by definition optional — a degraded module here
        does not regress fleet_ready).
        """
        writer = self._ensure_titan_hcl_state_writer()
        if writer is not None:
            writer.update(boot_phase="booting_b")
        # Poller is already live from Phase A; ensure it (defensive).
        self._ensure_probe_poller_started()
        # Non-blocking CONCURRENT post-boot (Maker 2026-05-28): spawn all
        # post-boot modules staggered in dep+layer order, then ONE readiness
        # wait (poller drives booted→running concurrently). Mirrors Phase A.
        order = self._compute_boot_order(names)
        t0 = time.time()
        # Bounded-concurrency spawn (cap shared with Phase A) — Phase B is the
        # heavy set (incl. memory/agno/cognitive moved here), so the cap is what
        # keeps the post-boot cold-import spike off the already-serving fleet.
        ready_count = self._spawn_capped(
            order, self._post_boot_stagger_delay_s, self._probe_wait_timeout_s)
        dt = time.time() - t0
        logger.info(
            "[Orchestrator] Phase B complete in %.1fs — %d/%d "
            "OPTIONAL_POST_BOOT modules RUNNING; publishing "
            "fleet_optional_ready", dt, ready_count, len(order))
        if writer is not None:
            writer.update(
                fleet_optional_ready=True,
                boot_phase="quiescent",
                post_boot_ready=ready_count)

    def _module_ready_publish_loop(self) -> None:
        """1Hz loop: snapshot {name: state.value} for every module and
        publish to module_ready.bin SHM (D-SPEC-123 follow-up Option B).

        Errors swallowed at this level — a stale SHM slot is a graceful
        degrade (readers fall back to optimistic-True) so we must NEVER
        kill the publisher thread.
        """
        while not self._module_ready_publisher_stop.is_set():
            try:
                snapshot = {
                    name: info.state.value
                    for name, info in list(self._modules.items())
                }
                if self._module_ready_shm_writer is not None:
                    self._module_ready_shm_writer.publish(snapshot)
            except Exception as exc:
                # Throttled — one log per 60s.
                now = time.time()
                last = getattr(
                    self, "_module_ready_publish_last_err_log", 0.0)
                if now - last > 60.0:
                    logger.warning(
                        "[Guardian] module_ready.bin publish failed: %s",
                        exc)
                    self._module_ready_publish_last_err_log = now
            # 1Hz cadence. Use Event.wait so shutdown is responsive.
            self._module_ready_publisher_stop.wait(1.0)

    def stop_all(self, reason: str = "shutdown") -> None:
        """Gracefully stop all running modules."""
        self._stop_requested = True
        # Stop the continuous probe poller (Phase 11 §11.I.7 / RFP 11D).
        try:
            self._probe_poller_stop.set()
        except Exception:
            pass
        # Stop module-ready SHM publisher.
        try:
            self._module_ready_publisher_stop.set()
            if self._module_ready_shm_writer is not None:
                self._module_ready_shm_writer.close()
        except Exception:  # noqa: BLE001
            pass
        # Option B (2026-05-02) — shut down restart executor cleanly so
        # in-flight async restart tasks finish before we shut down modules.
        # wait=False because we're about to forcibly stop all modules anyway;
        # no point waiting for a SAVE_NOW that's racing with shutdown.
        try:
            self._restart_executor.shutdown(wait=False, cancel_futures=True)
        except Exception:  # noqa: BLE001
            pass
        for name, info in self._modules.items():
            if info.state in (ModuleState.RUNNING, ModuleState.STARTING, ModuleState.UNHEALTHY):
                self.stop(name, reason=reason)

    def fast_kill(self, name: str) -> bool:
        """Fast SIGTERM-then-SIGKILL on a module — NO SAVE_NOW dance.

        Used by shadow_orchestrator AFTER pause() to clean up stragglers:
        workers that didn't ack HIBERNATE within the layer timeout but
        are still alive holding DB locks. Without this, locks_not_released
        would block shadow_boot indefinitely (workers don't auto-exit).

        Skips stop()'s SAVE_NOW + 30s wait. Goes straight to:
          _kill_owned_process: pgid-based killpg + .process.kill (~2s)
          _finalize_module_cleanup: queue cleanup + state reset

        Returns True if module was killed (or already dead). False on
        unknown name.
        """
        info = self._modules.get(name)
        if info is None:
            logger.warning("[Guardian] fast_kill: unknown module '%s'", name)
            return False
        if info.process is None or not info.process.is_alive():
            # Already exited via HIBERNATE — make sure state is clean
            self._finalize_module_cleanup(info, name)
            return True
        logger.info("[Guardian] fast_kill('%s') — straggler cleanup post-HIBERNATE", name)
        if getattr(info, 'adopted', False):
            self._kill_adopted_process(info, name)
        else:
            self._kill_owned_process(info, name)
        self._finalize_module_cleanup(info, name)
        return True

    def pause(self) -> None:
        """Mute monitor_tick + auto-restart WITHOUT iterating modules.

        Microkernel v2 Phase B fast-hibernate (2026-04-27): the original
        _phase_hibernate called stop_all() which iterated every module +
        published SAVE_NOW + waited up to 30s per module for SAVE_DONE
        from already-dead workers. That cost ~8.5 minutes per swap on a
        17-module fleet — with no benefit, since workers had already
        exited via HIBERNATE_ACK.

        pause() just sets _stop_requested=True (the kill switch). Workers
        are unchanged: those that exited via HIBERNATE stay exited;
        those still running keep running. Auto-restart is muted so dead
        workers don't get respawned by monitor_tick mid-swap.

        Symmetric counterpart: resume() (which clears the flag) must be
        called from rollback paths so workers respawn after a failed swap.

        Saves ~8.5 minutes per swap. The HIBERNATE_ACK collection that
        runs BEFORE pause() already gives workers their chance to save
        state cleanly; SAVE_NOW-then-30s-wait was redundant after that.
        """
        if self._stop_requested:
            logger.debug("[Guardian] pause() — already paused, no-op")
            return
        self._stop_requested = True
        logger.info("[Guardian] paused — monitor_tick muted, auto-restart "
                    "disabled (resume() to restore)")

    def resume(self) -> None:
        """Re-enable monitor_tick + auto-restart after a stop_all() pause.

        2026-04-27 Phase B.1 unwind-path bug fix: shadow_orchestrator's
        _phase_hibernate calls stop_all() to prevent Guardian from auto-
        respawning workers while the shadow kernel boots. Without a
        symmetric resume() call, _stop_requested stays True forever after
        a rollback, so monitor_tick becomes a permanent no-op and workers
        that exited (via HIBERNATE) are never restarted — Titan goes dark.

        Called from every shadow_orchestrator rollback path AFTER publishing
        HIBERNATE_CANCEL. After resume(), the caller should also invoke
        start_all() to actually respawn the autostart modules that exited.

        Idempotent: calling on an already-running Guardian is a no-op log.
        """
        if not self._stop_requested:
            logger.info("[Guardian] resume() called but already running — no-op")
            return
        self._stop_requested = False
        logger.info("[Guardian] resumed — monitor_tick re-enabled, "
                    "auto-restart back online")

    # Phase 11 §11.I.1 / D-SPEC-141 — Orchestrator.monitor_tick DELETED
    # per Maker 2026-05-27: "we dont need backward compatibility". The
    # supervisory loop is owned exclusively by
    # `titan_hcl.supervisor.core.Supervisor.monitor_tick`. All callers
    # MUST construct a Supervisor and drive supervisor.monitor_tick().

    def _process_guardian_messages(self) -> None:
        """Drain guardian queue and process control messages."""
        msgs = self.bus.drain(self._guardian_queue, max_msgs=50)
        for msg in msgs:
            msg_type = msg.get("type")
            src = msg.get("src", "")

            if msg_type == MODULE_HEARTBEAT:
                info = self._modules.get(src)
                if info:
                    info.last_heartbeat = time.time()
                    # Update RSS from heartbeat payload if provided
                    rss = msg.get("payload", {}).get("rss_mb", 0)
                    if rss:
                        info.rss_mb = rss

            elif msg_type == BUS_PEER_DIED:
                # Phase B.2 §D9 (2026-05-02) — broker detected peer process
                # is dead via os.kill(pid, 0) (authoritative OS signal).
                # Faster than Guardian's 1Hz process.is_alive() polling.
                # Idempotent w.r.t. polling-path restart via _module_lock.
                payload = msg.get("payload", {}) or {}
                name = payload.get("name", "")
                peer_pid = payload.get("pid")
                was_anon = bool(payload.get("was_anon", False))
                silent_for_s = float(payload.get("silent_for_s", 0.0))
                if was_anon:
                    # Pre-BUS_SUBSCRIBE death — broker can't tell which module
                    # was on this connection. Guardian's polling will discover
                    # via info.process.is_alive() within ~1s; nothing to do
                    # here besides logging. Useful for forensics if a worker
                    # crashes during init before sending BUS_SUBSCRIBE.
                    logger.warning(
                        "[Guardian] anon connection died pre-subscribe "
                        "(pid=%s, silent=%.1fs); polling-path will discover "
                        "the actual module within ~1s", peer_pid, silent_for_s)
                    continue
                # Named worker died. Trigger restart now (faster than polling).
                info = self._modules.get(name)
                if info is None:
                    logger.warning(
                        "[Guardian] BUS_PEER_DIED for unknown module '%s' "
                        "(pid=%s) — ignoring", name, peer_pid)
                    continue
                if not info.spec.restart_on_crash:
                    logger.info(
                        "[Guardian] '%s' (pid=%s) died but restart_on_crash="
                        "False — leaving stopped", name, peer_pid)
                    continue
                logger.error(
                    "[Guardian] '%s' (pid=%s) DIED — broker peer-dead signal "
                    "(silent=%.1fs); restart triggered (idempotent w.r.t. "
                    "polling path)", name, peer_pid, silent_for_s)
                # Mark crashed + restart. restart() acquires _module_lock,
                # so a concurrent polling-path restart becomes a no-op.
                # Option B (2026-05-02) — async so this code path (which
                # runs INSIDE _process_guardian_messages → monitor_tick)
                # doesn't block the queue drain.
                info.state = ModuleState.CRASHED
                self.bus.publish(make_msg(MODULE_CRASHED, "guardian", "core", {
                    "module": name, "exitcode": None,
                    "source": "broker_peer_dead",
                }))
                self.restart_async(name, reason="broker_peer_dead")

            elif msg_type == BUS_WORKER_ADOPT_REQUEST:
                # Microkernel v2 Phase B.2.1 — worker requesting adoption
                # by this (shadow) Guardian. Validate + register without
                # spawning + reply with BUS_WORKER_ADOPT_ACK (rid-matched).
                payload = msg.get("payload", {}) or {}
                worker_name = payload.get("name")
                worker_pid = payload.get("pid")
                rid = msg.get("rid")
                if not worker_name or not isinstance(worker_pid, int):
                    logger.warning(
                        "[Guardian] BUS_WORKER_ADOPT_REQUEST malformed: %r", payload,
                    )
                    continue
                # SPEC §11.B.3 (D-SPEC-49) — if this adoption corresponds to
                # an in-flight per-module reload, route to the reload
                # orchestrator's queue instead of the cross-kernel adopt
                # path (which would reject the same-name case as
                # "already_running"). The orchestrator emits its own
                # BUS_WORKER_ADOPT_ACK (rid-matched) per §4.3 step 4.
                #
                # Closed BUG-PHASE-B-FIRST-RELOAD-ADOPTION-ROUTING-MISS-20260514:
                # an earlier `rs.new_pid == worker_pid` guard had a memory-
                # visibility race between the orchestrator thread (running on
                # _restart_executor — sets rs.new_pid post-spawn) and the
                # message-handler thread (reads rs.new_pid here). Even though
                # the GIL makes individual reads atomic on CPython, with no
                # explicit synchronization the reader could observe the
                # initial None value while the writer was still pending in
                # the executor's queue. Name-based routing is sufficient:
                # the orchestrator validates `rs.new_pid == worker_pid` AND
                # swap_id in the payload when it picks up the frame from
                # adoption_q (defensive pid-validation moved into
                # _reload_module_sync step 4). Mismatches (cross-kernel
                # or stale) get rejected by the orchestrator with reason
                # logged, not silently mis-routed here.
                rs = self._reloads_in_flight.get(worker_name)
                if rs is not None:
                    try:
                        rs.adoption_q.put_nowait(msg)
                    except _queue_mod.Full:
                        pass
                    continue
                ok = self.adopt_worker(worker_name, worker_pid)
                ack_payload = {
                    "name": worker_name,
                    "pid": worker_pid,
                    "shadow_pid": os.getpid(),
                }
                if ok:
                    ack_payload["status"] = "adopted"
                    ack_payload["reason"] = None
                else:
                    ack_payload["status"] = "rejected"
                    # Best-effort distinguishability for logs/tests:
                    if not self._pid_alive(worker_pid):
                        ack_payload["reason"] = "pid_not_alive"
                    elif worker_name not in self._modules:
                        ack_payload["reason"] = "unknown_name"
                    else:
                        ack_payload["reason"] = "already_running"
                self.bus.publish(make_msg(
                    BUS_WORKER_ADOPT_ACK,
                    "guardian",
                    worker_name,
                    ack_payload,
                    rid=rid,
                ))

            elif msg_type == MODULE_RELOAD_REQUEST:
                # SPEC §8.3 + §11.B.3 (D-SPEC-49) — per-module hot-reload
                # initiated by Maker CLI / future D9 Guardian. Dispatch
                # asynchronously on _restart_executor so this queue drain
                # remains responsive — the reload sequence takes up to
                # MODULE_RELOAD_DEFAULT_TIMEOUT_S=30s and MUST NOT block
                # heartbeat processing.
                self._dispatch_reload_request(msg)

    @staticmethod
    def _get_rss_mb(pid: int) -> float:
        """Read RSS from /proc/{pid}/status (Linux only)."""
        try:
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024.0  # kB → MB
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
        return 0.0

    @staticmethod
    def _get_cpu_time_seconds(pid: int) -> float:
        """Read total CPU time (utime+stime) from /proc/{pid}/stat in seconds.

        Used by CPU-aware heartbeat check: if a module didn't send a
        heartbeat in time but its CPU time grew, it's alive-but-starved
        rather than deadlocked.
        """
        try:
            with open(f"/proc/{pid}/stat") as f:
                fields = f.read().split()
            # Fields per proc(5): utime=14, stime=15 (1-indexed). Account for
            # the comm field which can contain spaces wrapped in parentheses.
            # Easier: take last close-paren and slice from there.
            raw = open(f"/proc/{pid}/stat").read()
            tail = raw.rsplit(")", 1)[1].split()
            # After ')' the indices shift: state=0, ppid=1, ..., utime=11, stime=12.
            utime = int(tail[11])
            stime = int(tail[12])
            ticks_per_sec = os.sysconf("SC_CLK_TCK") or 100
            return (utime + stime) / float(ticks_per_sec)
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError, IndexError, OSError):
            pass
        return 0.0

    def get_status(self) -> dict:
        """Return a dict of module statuses for the Observatory API."""
        result = {}
        for name, info in self._modules.items():
            result[name] = {
                "state": info.state.value,
                "pid": info.pid,
                "rss_mb": round(info.rss_mb, 1),
                "uptime": round(time.time() - info.start_time, 1) if info.start_time else 0,
                "restart_count": info.restart_count,
                "restarts_in_window": len(info.restart_timestamps),
                "last_heartbeat_age": round(time.time() - info.last_heartbeat, 1) if info.last_heartbeat else -1,
                # Microkernel v2 Phase A §A.5 — layer exposed per module.
                "layer": info.spec.layer,
                # Microkernel v2 Phase A §A.3 (S6) — start method for hybrid policy.
                "start_method": info.spec.start_method,
                # Microkernel v2 Phase B.2.1 — adoption sentinel + timestamp.
                "adopted": info.adopted,
                "adopt_ts": info.adopt_ts if info.adopted else 0.0,
            }
        return result

    # ── Microkernel v2 Phase A §A.5 — layer queries ─────────────────────

    def get_layer(self, name: str) -> str | None:
        """Return the layer tag for a registered module, or None."""
        info = self._modules.get(name)
        return info.spec.layer if info else None

    def get_modules_by_layer(self, layer: str) -> list[str]:
        """Return sorted list of module names registered at the given layer."""
        return sorted(
            n for n, info in self._modules.items() if info.spec.layer == layer
        )

    def layer_stats(self) -> dict:
        """
        Return per-layer counters (total + per-state) for dashboard exposure.
        Used by /v4/guardian-status and arch_map layers.
        """
        stats: dict[str, dict[str, int]] = {
            "L0": {"total": 0, "running": 0, "crashed": 0, "disabled": 0},
            "L1": {"total": 0, "running": 0, "crashed": 0, "disabled": 0},
            "L2": {"total": 0, "running": 0, "crashed": 0, "disabled": 0},
            "L3": {"total": 0, "running": 0, "crashed": 0, "disabled": 0},
        }
        for info in self._modules.values():
            bucket = stats.get(info.spec.layer)
            if bucket is None:
                continue  # unknown layer — shouldn't happen after register() validation
            bucket["total"] += 1
            if info.state == ModuleState.RUNNING:
                bucket["running"] += 1
            elif info.state == ModuleState.CRASHED:
                bucket["crashed"] += 1
            elif info.state == ModuleState.DISABLED:
                bucket["disabled"] += 1
        return stats

    def is_running(self, name: str) -> bool:
        info = self._modules.get(name)
        return info is not None and info.state == ModuleState.RUNNING

    def is_started(self, name: str) -> bool:
        """True if the module's worker process is spawned (RUNNING or STARTING).

        Distinct from is_running which requires state==RUNNING. A worker in
        STARTING state has its process spawned and bus QUERY handling active
        — calling guardian.start() again is redundant. This predicate is the
        correct gate for proxy._ensure_started's "do we need to spawn?" check
        (see _start_safe.py). Avoids spurious "First use" log spam + extra
        guardian.start() calls during heavy-boot warmup (memory's 9-min
        FAISS+Kuzu+DuckDB load on T1).
        """
        info = self._modules.get(name)
        return info is not None and info.state in (
            ModuleState.RUNNING, ModuleState.STARTING)

    def query_module(self, name: str, action: str, payload: dict | None = None,
                     timeout: float = 5.0) -> dict | None:
        """Send a QUERY to a running module and wait for its RESPONSE.

        Returns the response payload dict, or None on timeout/error.
        Used by the profiling endpoint to collect child-process tracemalloc data.

        Phase B.2 §D12 audit (2026-05-02): under socket mode, info.queue +
        info.send_queue are inert (worker reads/writes via SocketQueue, not
        these mp.Queues). The legacy code path put a QUERY on info.queue
        that the worker never reads, then waited on info.send_queue that
        the worker never writes to → silent timeout, return None.

        Fix: under socket mode, route via bus.request() with a transient
        reply queue (the same path proxies use). Legacy mp.Queue path
        retained for the rare case Guardian runs without a broker (tests,
        legacy fallback) — gated on `self.bus.has_socket_broker`.
        """
        import uuid
        info = self._modules.get(name)
        if info is None or info.state != ModuleState.RUNNING:
            return None

        # Socket-mode path: route via bus.request (broker delivers QUERY,
        # worker publishes RESPONSE back to "guardian_query_<rid>", we
        # consume one message off that reply queue).
        if self.bus.has_socket_broker:
            rid = str(uuid.uuid4())
            reply_name = f"guardian_query_{rid[:8]}"
            reply_q = self.bus.subscribe(reply_name, reply_only=True)
            try:
                req_msg = {
                    "type": "QUERY",
                    "src": reply_name,  # worker echoes RESPONSE back to here
                    "dst": name,
                    "rid": rid,
                    "payload": {"action": action, **(payload or {})},
                    "ts": time.time(),
                }
                self.bus.publish(req_msg)
                deadline = time.time() + timeout
                while time.time() < deadline:
                    try:
                        resp = reply_q.get(timeout=min(0.2, max(0.01, deadline - time.time())))
                    except Empty:
                        continue
                    except Exception:
                        return None
                    if resp.get("rid") == rid:
                        return resp.get("payload")
                return None
            finally:
                # Always tear down the transient reply subscription.
                try:
                    self.bus.unsubscribe(reply_name, reply_q)
                except Exception:
                    pass

        # Legacy (no-broker) path — kept for tests + legacy fallback only.
        if info.queue is None:
            return None
        rid = str(uuid.uuid4())
        msg = {
            "type": "QUERY",
            "src": "guardian",
            "dst": name,
            "rid": rid,
            "payload": {"action": action, **(payload or {})},
            "ts": time.time(),
        }
        try:
            info.queue.put_nowait(msg)
        except Exception:
            return None
        deadline = time.time() + timeout
        stashed: list = []
        result = None
        while time.time() < deadline:
            try:
                resp = info.send_queue.get(timeout=0.2)
                if resp.get("rid") == rid:
                    result = resp.get("payload")
                    break
                stashed.append(resp)
            except Exception:
                continue
        for m in stashed:
            try:
                info.send_queue.put_nowait(m)
            except Exception:
                pass
        return result

    def drain_send_queues(self) -> int:
        """
        Drain all module send queues and publish their messages to the bus.
        Called by Core in its monitor loop to route worker responses back.
        Returns total messages routed.

        Also observes META_CGN_SIGNAL messages passing through and records
        them in the parent's BusHealthMonitor. Necessary because workers
        are forked subprocesses — their record_emission() calls update
        only the child's copy of the monitor singleton, not the parent's
        where /v4/bus-health reads. Intercepting at drain time is the
        single canonical point where parent sees every worker emission.

        Additionally persists each META_CGN_SIGNAL to
        `data/meta_cgn_emissions.jsonl` (append-only JSONL) — survives
        restarts so `arch_map producers` can show history across time.
        Deque in BusHealthMonitor is in-memory only (maxlen=2000, resets
        on restart). At current rate budget ~0.05 Hz the file grows ~4.3
        KB/Titan/day — negligible.

        B.3 Stage 1 (2026-05-02): under socket mode, info.send_queue is
        None for all modules (mp.Queue allocation skipped to avoid
        fork-at-locked-Queue hazard). The for-loop body is gated on
        `info.send_queue is None` already; this comment explains that the
        function becomes a no-op iteration in socket mode rather than
        being short-circuited at entry — kept as no-op to preserve the
        BusHealthMonitor singleton init + simplify the eventual full
        deletion in Stage 3.
        """
        from queue import Empty
        from titan_hcl.bus import META_CGN_SIGNAL
        try:
            from titan_hcl.core.bus_health import get_global_monitor
            _bh_monitor = get_global_monitor()
        except Exception:
            _bh_monitor = None

        total = 0
        for name, info in self._modules.items():
            if info.send_queue is None:
                continue
            for _ in range(100):  # max 100 msgs per module per tick
                try:
                    msg = info.send_queue.get_nowait()
                    # Observe META_CGN_SIGNAL in the parent so /v4/bus-health
                    # reflects actual worker emissions.
                    if msg.get("type") == META_CGN_SIGNAL:
                        p = msg.get("payload", {}) or {}
                        if _bh_monitor is not None:
                            try:
                                _bh_monitor.record_emission(
                                    producer_src=msg.get("src", "unknown"),
                                    consumer=str(p.get("consumer", "")),
                                    event_type=str(p.get("event_type", "")),
                                    intensity=float(p.get("intensity", 1.0)),
                                )
                            except Exception as _rec_err:
                                logger.debug(
                                    "[Guardian] BusHealthMonitor.record_emission "
                                    "failed: %s", _rec_err)
                        # Persistent append to data/meta_cgn_emissions.jsonl
                        try:
                            _append_meta_cgn_emission_log(msg, p)
                        except Exception as _log_err:
                            # Best-effort: don't let disk issues break bus drain.
                            # WARN because silent failure would hide an observability
                            # gap (e.g. disk full, permission issue).
                            logger.warning(
                                "[Guardian] Persistent emission log append failed: "
                                "%s (emission still bus-delivered)", _log_err)
                    self.bus.publish(msg)
                    total += 1
                except Empty:
                    break
        return total


def _module_wrapper(entry_fn: Callable, name: str, recv_queue, send_queue,
                    config: dict, start_method: str = "fork",
                    broadcast_topics: Optional[list] = None,
                    reply_only: bool = False) -> None:
    """
    Wrapper that runs in the child process.
    Sets up logging, lifecycle protection, B.2 bus bootstrap, B.2.1 swap-
    handler bootstrap, then calls the entry function and handles crashes.

    Phase B.2.1 (2026-04-27): centralizes setup_worker_bus + SwapHandlerState
    + supervision daemon here so per-worker change reduces to 3 elif branches
    in the main loop (chunk C2). `start_method` is passed by Guardian's
    spawn site from ModuleSpec.start_method; controls whether this worker
    takes the spawn-mode "true outlive" path or the fork-mode improved-B.1
    fallback (worker dies with old kernel, shadow's Guardian respawns it).
    """
    # Configure logging for child process
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [%(levelname)s] [{name}] %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(name)
    logger.info("Module '%s' process started (pid=%d)", name, os.getpid())

    # Orphan prevention: when titan_hcl is SIGKILL'd (OOM, kernel panic),
    # its signal handlers don't run and child workers reparent to systemd,
    # accumulate state, and become memory leaks until OOM-killed themselves
    # (see 2026-04-27 cascade incident: T2 IMW orphan grew to 11.4 GB).
    # PR_SET_PDEATHSIG asks the kernel to deliver SIGTERM the moment our
    # parent dies — survives parent SIGKILL because the kernel is the
    # messenger. Combined with a getppid()-poll watcher as backup defense.
    # Worker entry_fns install their own SIGTERM handlers afterward, which
    # catch this signal and perform graceful shutdown (flush WAL, etc.).
    from titan_hcl.core.worker_lifecycle import install_full_protection
    _wl = install_full_protection()
    logger.info(
        "Module '%s' lifecycle protection: pdeathsig=%s watcher=%s",
        name, _wl["pdeathsig_installed"], _wl["watcher_started"],
    )

    # Phase B.2.1 — bus bootstrap + swap-handler bootstrap. setup_worker_bus
    # rebinds (recv_queue, send_queue) to SocketQueue when env vars indicate
    # bus_ipc_socket_enabled mode; otherwise returns the original mp.Queue
    # handles unchanged (legacy behavior, no socket overhead).
    #
    # A.8.2 §3.5 (2026-04-28): the supervision daemon thread is only needed by
    # spawn-mode (graduated) workers — they take the "true outlive" path during
    # shadow swap and require BUS_HANDOFF / ADOPT_ACK / CANCELED dispatch.
    # Fork-mode workers die-with-parent (PR_SET_PDEATHSIG) and get respawned
    # by the shadow's Guardian, so they never participate in adoption — the
    # supervision tick is dead code for them. Skip start_supervision_thread()
    # for fork-mode entirely (saves the redundant no-op Thread object that
    # worker_swap_handler.start_supervision_thread() returns for fork-mode).
    # The SwapHandlerState is still registered process-globally so other
    # adoption-protocol entrypoints (HANDOFF dispatch from the bus message
    # path) remain reachable via get_active_swap_state(); they'll early-return
    # for fork-mode at their own dispatch sites.
    # SPEC §11.B.3 Phase B (D-SPEC-49): when Guardian.reload_module()
    # spawns this worker as a Phase B reload, _spawn_for_reload injects
    # `_phase_b_reload_swap_id` into a deep-copy of config. Pop it out so
    # the worker's entry_fn sees its normal config dict. Pass to
    # setup_worker_bus so the initial subscribe-ack callback also emits
    # ADOPTION_REQUEST (in addition to MODULE_READY) per rFP §4.3 step 4.
    phase_b_reload_swap_id = None
    if isinstance(config, dict):
        phase_b_reload_swap_id = config.pop("_phase_b_reload_swap_id", None)

    bus_client = None
    try:
        from titan_hcl.core.worker_bus_bootstrap import setup_worker_bus
        recv_queue, send_queue, bus_client = setup_worker_bus(
            name, recv_queue, send_queue,
            topics=broadcast_topics,
            reply_only=reply_only,  # SPEC §8.2 v1.4.0 D-SPEC-42
            phase_b_reload_swap_id=phase_b_reload_swap_id,
        )
        if bus_client is not None:
            from titan_hcl.core.worker_swap_handler import (
                SwapHandlerState,
                set_active_swap_state,
                start_supervision_thread,
            )
            swap_state = SwapHandlerState(
                name=name,
                start_method=start_method,
                watcher_state=_wl["watcher_state"],
                bus_client=bus_client,
            )
            set_active_swap_state(swap_state)
            if start_method == "spawn":
                start_supervision_thread(swap_state)
                logger.info(
                    "Module '%s' B.2.1 wiring active (start_method=%s, "
                    "supervision daemon ticking)",
                    name, start_method,
                )
            else:
                logger.info(
                    "Module '%s' B.2.1 wiring active (start_method=%s, "
                    "supervision skipped — fork-mode dies with parent, "
                    "adoption not applicable; A.8.2 §3.5)",
                    name, start_method,
                )
    except Exception as e:  # noqa: BLE001 — never crash worker boot on wiring
        logger.warning(
            "Module '%s' B.2.1 wiring init failed: %s — continuing in legacy mode",
            name, e, exc_info=True,
        )

    try:
        entry_fn(recv_queue, send_queue, name, config)
    except KeyboardInterrupt:
        logger.info("Module '%s' interrupted", name)
    except Exception as e:
        logger.error("Module '%s' crashed: %s", name, e, exc_info=True)
        raise
    finally:
        if bus_client is not None:
            try:
                bus_client.stop()
            except Exception:  # noqa: BLE001
                pass
        try:
            from titan_hcl.core.worker_swap_handler import set_active_swap_state
            set_active_swap_state(None)
        except Exception:  # noqa: BLE001
            pass
        logger.info("Module '%s' process exiting", name)


# Phase 11 §11.I.1 / D-SPEC-141 back-compat alias — preserves every callsite
# `from titan_hcl.guardian_hcl import Guardian` (30+ in tests + scripts +
# titan_hcl/core/plugin.py + titan_hcl/proxies/studio_proxy.py + …). The 11E.b.2
# process peer-split removes the alias once all callsites migrate to
# `from titan_hcl.orchestrator import Orchestrator`.
Guardian = Orchestrator
