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
import logging
import os
import signal
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from multiprocessing import Process
from typing import Callable, Optional

from .bus import (
    AnyQueue,
    BUS_WORKER_ADOPT_ACK,
    BUS_WORKER_ADOPT_REQUEST,
    DivineBus,
    MODULE_CRASHED,
    MODULE_HEARTBEAT,
    MODULE_READY,
    MODULE_SHUTDOWN,
    make_msg,
)
from titan_plugin import bus

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


class ModuleState(Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    UNHEALTHY = "unhealthy"
    CRASHED = "crashed"
    DISABLED = "disabled"


@dataclass
class ModuleSpec:
    """Specification for a supervised module."""
    name: str
    entry_fn: Callable  # function(bus_queue, config) → runs in child process
    config: dict = field(default_factory=dict)
    rss_limit_mb: int = DEFAULT_RSS_LIMIT_MB
    autostart: bool = False      # start immediately on Guardian boot
    lazy: bool = True            # start on first use (via proxy)
    restart_on_crash: bool = True
    heartbeat_timeout: float = HEARTBEAT_TIMEOUT  # per-module override
    reply_only: bool = False     # if True, skip dst="all" broadcasts (only receive targeted msgs)
    # Microkernel v2 Phase A §A.5 (2026-04-24): layer assignment.
    # Canonical values in titan_plugin._layer_canon.LAYER_CANON.
    # Validated in Guardian.register(). Used by arch_map, dashboard,
    # and layer-aware crash logging.
    layer: str = "L3"
    # Microkernel v2 Phase A §A.3 (2026-04-25, S6): spawn vs fork
    # start method. Default "fork" preserves Guardian's current
    # byte-identical behavior. When set to "spawn", Guardian boots
    # the worker via multiprocessing.get_context("spawn") — fresh
    # interpreter, no parent COW inheritance. Saves ~200 MB RSS per
    # worker (fork baseline ~265 MB → spawn baseline ~50-80 MB).
    # Unknown values fall back to "fork" with a WARNING (Guardian
    # never crashes boot on a misconfigured ModuleSpec).
    start_method: str = "fork"
    # Microkernel v2 Phase B.2.1 §M5 (2026-04-27 PM): adoption criticality.
    # When True (default): worker MUST adopt for shadow swap to succeed —
    # holds heavy in-process state (FAISS, DuckDB, audit chain, neural
    # nets, vocabulary) that would be expensive to re-load. When False:
    # nice-to-adopt — orchestrator declares swap successful regardless;
    # if this worker doesn't adopt by timeout, it's left to self-SIGTERM
    # via supervision daemon's bus-as-supervision check, and shadow's
    # Guardian respawns it fresh post-swap (light-state workers like
    # autonomous writers, observability aggregators, periodic backup
    # daemons).
    b2_1_swap_critical: bool = True


@dataclass
class ModuleInfo:
    """Runtime state for a supervised module."""
    spec: ModuleSpec
    state: ModuleState = ModuleState.STOPPED
    process: Optional[Process] = None
    pid: Optional[int] = None
    queue: Optional[AnyQueue] = None     # module's receive queue (bus→worker)
    send_queue: Optional[AnyQueue] = None  # module's send queue (worker→bus)
    last_heartbeat: float = 0.0
    start_time: float = 0.0
    restart_count: int = 0
    last_restart: float = 0.0
    rss_mb: float = 0.0
    restart_timestamps: deque = field(default_factory=lambda: deque(maxlen=MAX_RESTARTS_IN_WINDOW + 1))
    ready_time: float = 0.0  # when MODULE_READY was received
    disabled_at: float = 0.0  # when module was disabled (for auto-re-enable cooldown)
    # CPU-aware heartbeat (added 2026-04-21 after iter-3 ARC load triggered
    # cascading media-module restart loops on shared T2/T3 VPS — modules
    # were CPU-starved, not deadlocked, but wallclock heartbeat fired anyway).
    last_cpu_time: float = 0.0           # /proc/<pid>/stat utime+stime sample (seconds)
    last_cpu_sample_ts: float = 0.0      # when last_cpu_time was sampled
    consecutive_starved_cycles: int = 0  # heartbeat misses where CPU grew (alive-but-starved)
    # Microkernel v2 Phase B.2.1 (2026-04-27): worker supervision-transfer.
    # When True, this ModuleInfo refers to an externally-spawned worker
    # (adopted from a prior kernel via BUS_WORKER_ADOPT_REQUEST). We do NOT
    # own info.process (it stays None); cleanup uses os.kill instead of
    # process.kill(). First heartbeat after adoption resets clocks fresh.
    adopted: bool = False
    adopt_ts: float = 0.0                # when adoption completed (for telemetry)


class Guardian:
    """
    Supervises module processes — starts, monitors, restarts.

    V4 improvements over V3:
      - Sliding-window restart tracking (no more infinite restart loops)
      - Per-module heartbeat_timeout (Spirit gets 120s for heavy V4 work)
      - Restart count only resets after sustained uptime (5 min)
      - Single-instance enforcement via process liveness check before start

    Usage:
        bus = DivineBus()
        guardian = Guardian(bus)
        guardian.register(ModuleSpec("memory", memory_worker_fn, config={...}))
        guardian.register(ModuleSpec("rl", rl_worker_fn, rss_limit_mb=2500))
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
                MODULE_READY,
                MODULE_SHUTDOWN,
                BUS_WORKER_ADOPT_REQUEST,
                # SAVE_DONE is targeted (dst="guardian") so it bypasses the
                # filter regardless, but list it explicitly so arch_map can
                # see the contract.
            ],
        )
        self._stop_requested = False
        self._module_lock = threading.RLock()  # serialize start/stop/restart to prevent duplicate spawns
        # Microkernel v2 Phase A retrofit (2026-04-27): swap-aware kernel ref.
        # Kernel sets `self.guardian._kernel_ref = self` after Guardian
        # construction. start()/restart() consult kernel.is_shadow_swap_active()
        # to block lazy-starts during swap (prevents proxy-driven mid-swap
        # worker resurrection that holds DB locks). None in legacy mode
        # (in-process; no kernel split) → swap interlock degrades to no-op.
        self._kernel_ref = None

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

    def register(self, spec: ModuleSpec) -> None:
        """Register a module specification. Does not start the module."""
        # Microkernel v2 Phase A §A.5 — validate layer before registering.
        from ._layer_canon import validate_layer
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

    def start(self, name: str) -> bool:
        """Start a specific module process. Thread-safe via _module_lock."""
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
            info.queue = ctx.Queue(maxsize=10000)
            info.send_queue = ctx.Queue(maxsize=10000)

            # Subscribe in the bus so targeted messages get routed
            self._module_recv_queues[name] = info.queue
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
        titan_main dies and the API goes dark until watchdog reboot.
        Observed 2026-04-14 on T1 when spirit worker didn't respond to
        SIGTERM in 15s — killpg fired and took titan_main with it.
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
                # Drain guardian queue waiting for SAVE_DONE matching rid
                save_deadline = time.time() + save_timeout
                save_done_seen = False
                drained_msgs: list = []  # re-enqueue non-matching messages
                while time.time() < save_deadline:
                    try:
                        m = self._guardian_queue.get(timeout=0.5)
                    except Exception:
                        continue
                    if (m.get("type") == bus.SAVE_DONE
                            and m.get("payload", {}).get("module") == name
                            and m.get("payload", {}).get("request_id") == save_rid):
                        _p = m.get("payload", {})
                        logger.info("[Guardian] SAVE_DONE from '%s': "
                                    "saved=%s errors=%s (%dms)",
                                    name, _p.get("saved"), _p.get("errors"),
                                    _p.get("duration_ms", 0))
                        save_done_seen = True
                        break
                    drained_msgs.append(m)
                # Re-publish drained messages so we don't lose them
                for _dm in drained_msgs:
                    try:
                        self.bus.publish(_dm)
                    except Exception:
                        pass
                if not save_done_seen:
                    logger.info("[Guardian] No SAVE_DONE from '%s' within "
                                "%.1fs — module may not handle SAVE_NOW; "
                                "proceeding with SHUTDOWN (post-loop cleanup "
                                "still runs as fallback save)",
                                name, save_timeout)

            # ── Phase 2: shutdown signal via bus ──
            self.bus.publish(make_msg(MODULE_SHUTDOWN, "guardian", name, {"reason": reason}))

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

    def restart(self, name: str, reason: str = "requested") -> bool:
        """Stop then start a module, with sliding-window restart limit. Thread-safe via _module_lock."""
        with self._module_lock:
            info = self._modules.get(name)
            if not info:
                return False

            now = time.time()

            # Sliding window: count restarts in the last _restart_window_seconds
            # Prune old timestamps
            while info.restart_timestamps and (now - info.restart_timestamps[0]) > self._restart_window_seconds:
                info.restart_timestamps.popleft()

            if len(info.restart_timestamps) >= self._max_restarts_in_window:
                logger.error("[Guardian] Module '%s' [%s] exceeded %d restarts in %.0fs window — disabling (auto-re-enable in %.0fs)",
                             name, info.spec.layer, self._max_restarts_in_window, self._restart_window_seconds, REENABLE_COOLDOWN_S)
                self.stop(name, reason="max_restarts_disabled")
                info.state = ModuleState.DISABLED
                info.disabled_at = time.time()
                self.bus.publish(make_msg(MODULE_CRASHED, "guardian", "core", {
                    "module": name, "reason": "max_restarts_in_window",
                    "restarts": len(info.restart_timestamps),
                    "window_seconds": self._restart_window_seconds,
                }))
                return False

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
            return self.start(name)

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

    def start_all(self) -> None:
        """Start all modules that have autostart=True.

        Microkernel v2 Phase B.2.1 (2026-04-27 PM): when env var
        TITAN_B2_1_ADOPTION_PENDING=1 is set (passed by the shadow_swap
        orchestrator to a freshly-spawned shadow kernel), spawn-mode
        autostart modules are SKIPPED here. Those workers are the ones
        that survived the kernel swap; they reconnect to shadow's broker
        and register via BUS_WORKER_ADOPT_REQUEST → Guardian.adopt_worker.
        Spawning fresh copies would create duplicates fighting for
        shm/locks/sockets.

        Fork-mode workers + non-graduated specials (imw, observatory_writer,
        api_subprocess) still start normally — they died with the old
        kernel and need the shadow Guardian to respawn them. Once the
        adoption window closes (orchestrator times out or accepts), any
        spawn-mode worker that DIDN'T get adopted can be started via
        explicit start(name) calls — that fallback isn't automatic here
        to keep the boot-time logic simple.
        """
        adoption_pending = (
            os.environ.get("TITAN_B2_1_ADOPTION_PENDING", "") == "1"
        )
        skipped: list[str] = []
        for name, info in self._modules.items():
            if not info.spec.autostart:
                continue
            if adoption_pending and info.spec.start_method == "spawn":
                skipped.append(name)
                continue
            self.start(name)
        if skipped:
            logger.info(
                "[Guardian] B.2.1 adoption-pending: skipped autostart for "
                "spawn-mode workers (%d): %s — awaiting "
                "BUS_WORKER_ADOPT_REQUEST",
                len(skipped), sorted(skipped),
            )

    def stop_all(self, reason: str = "shutdown") -> None:
        """Gracefully stop all running modules."""
        self._stop_requested = True
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

    def monitor_tick(self) -> None:
        """
        Called periodically by Core to check module health.
        Processes heartbeats, checks RSS, restarts dead modules.
        Thread-safe: acquires _module_lock for state mutations.
        """
        if self._stop_requested:
            return

        # Process incoming messages on guardian queue
        self._process_guardian_messages()

        now = time.time()
        for name, info in self._modules.items():
            # Auto-re-enable disabled modules after cooldown period
            if info.state == ModuleState.DISABLED and info.disabled_at > 0:
                elapsed = now - info.disabled_at
                if elapsed >= REENABLE_COOLDOWN_S:
                    logger.info("[Guardian] Auto-re-enabling module '%s' after %.0fs cooldown", name, elapsed)
                    self.enable(name)
                continue

            if info.state not in (ModuleState.RUNNING, ModuleState.STARTING):
                continue

            # Check if process is still alive
            if info.process and not info.process.is_alive():
                with self._module_lock:
                    # Re-check under lock (process may have been restarted by proxy thread)
                    if info.process and not info.process.is_alive():
                        exitcode = info.process.exitcode
                        logger.warning("[Guardian] Module '%s' died (exitcode=%s)", name, exitcode)
                        info.state = ModuleState.CRASHED
                        self.bus.publish(make_msg(MODULE_CRASHED, "guardian", "core", {
                            "module": name, "exitcode": exitcode,
                        }))
                        if info.spec.restart_on_crash:
                            self.restart(name, reason=f"died_exitcode_{exitcode}")
                continue

            # Heartbeat timeout check — CPU-aware (2026-04-21).
            #
            # Wallclock heartbeat alone misclassifies CPU-starved modules
            # as deadlocked. On shared T2/T3 VPS during iter-3 ARC runs,
            # the media module's heartbeat thread was preempted >180s
            # repeatedly even though the module itself was making progress.
            # Restart loops every 3 min added MORE CPU pressure, worsening
            # the cascade.
            #
            # Algorithm: when wallclock timeout fires, sample /proc/<pid>/stat
            # CPU time. If CPU grew since last sample → module is alive
            # but starved → log + skip restart + count cycle. After
            # MAX_STARVED_CYCLES consecutive starved cycles, force restart
            # anyway (gives up — bounded grace prevents runaway hang).
            if info.state == ModuleState.RUNNING:
                hb_timeout = info.spec.heartbeat_timeout
                if now - info.last_heartbeat > hb_timeout:
                    cpu_now = self._get_cpu_time_seconds(info.pid) if info.pid else 0.0
                    cpu_grew = (info.last_cpu_time > 0.0
                                and cpu_now - info.last_cpu_time >= MIN_CPU_DELTA_FOR_ALIVE)
                    if cpu_grew and info.consecutive_starved_cycles < MAX_STARVED_CYCLES:
                        info.consecutive_starved_cycles += 1
                        logger.warning(
                            "[Guardian] Module '%s' heartbeat timeout (%.1fs > %.0fs) but "
                            "CPU grew +%.2fs — alive-but-starved cycle %d/%d, deferring restart",
                            name, now - info.last_heartbeat, hb_timeout,
                            cpu_now - info.last_cpu_time,
                            info.consecutive_starved_cycles, MAX_STARVED_CYCLES)
                        info.last_cpu_time = cpu_now
                        info.last_cpu_sample_ts = now
                        continue
                    # Either CPU didn't grow (truly stuck) or we exhausted grace cycles.
                    reason = ("heartbeat_timeout_starved_grace_exhausted"
                              if info.consecutive_starved_cycles >= MAX_STARVED_CYCLES
                              else "heartbeat_timeout")
                    # Microkernel v2 Phase A §A.5 — L1 crashes are architecturally
                    # unexpected (Trinity daemons should be rock-solid); log at
                    # ERROR level so they surface distinctly from L2/L3 restarts.
                    lvl = logging.ERROR if info.spec.layer == "L1" else logging.WARNING
                    logger.log(lvl,
                               "[Guardian] Module '%s' [%s] heartbeat timeout (%.1fs > %.0fs limit) — restart reason=%s",
                               name, info.spec.layer, now - info.last_heartbeat, hb_timeout, reason)
                    with self._module_lock:
                        info.state = ModuleState.UNHEALTHY
                        info.consecutive_starved_cycles = 0
                        if info.spec.restart_on_crash:
                            self.restart(name, reason=reason)
                    continue
                # Heartbeat fresh — refresh CPU sample so next timeout check
                # has a recent baseline (window ≈ monitor_tick interval, ~5s).
                # Without this, last_cpu_time would be sampled only at boot
                # and at recovery, making the cpu_grew comparison stale.
                if info.pid:
                    info.last_cpu_time = self._get_cpu_time_seconds(info.pid)
                    info.last_cpu_sample_ts = now
                if info.consecutive_starved_cycles > 0:
                    info.consecutive_starved_cycles = 0

            # Reset restart count after sustained uptime
            if info.state == ModuleState.RUNNING and info.ready_time > 0:
                if now - info.ready_time > self._sustained_uptime_reset and info.restart_count > 0:
                    logger.info("[Guardian] Module '%s' sustained uptime %.0fs — resetting restart count",
                                name, now - info.ready_time)
                    info.restart_count = 0
                    info.restart_timestamps.clear()

            # RSS check
            if info.pid:
                rss = self._get_rss_mb(info.pid)
                info.rss_mb = rss
                if rss > info.spec.rss_limit_mb:
                    logger.warning("[Guardian] Module '%s' RSS %.0fMB > limit %dMB",
                                   name, rss, info.spec.rss_limit_mb)
                    if info.spec.restart_on_crash:
                        self.restart(name, reason=f"rss_{rss:.0f}mb")

    def _process_guardian_messages(self) -> None:
        """Drain guardian queue and process control messages."""
        msgs = self.bus.drain(self._guardian_queue, max_msgs=50)
        for msg in msgs:
            msg_type = msg.get("type")
            src = msg.get("src", "")

            if msg_type == MODULE_READY:
                info = self._modules.get(src)
                if info:
                    info.state = ModuleState.RUNNING
                    info.last_heartbeat = time.time()
                    info.ready_time = time.time()
                    # Do NOT reset restart_count here — only after sustained uptime
                    logger.info("[Guardian] Module '%s' is READY (pid=%s, restarts=%d)",
                                src, info.pid, info.restart_count)

            elif msg_type == MODULE_HEARTBEAT:
                info = self._modules.get(src)
                if info:
                    info.last_heartbeat = time.time()
                    # Update RSS from heartbeat payload if provided
                    rss = msg.get("payload", {}).get("rss_mb", 0)
                    if rss:
                        info.rss_mb = rss

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

    def query_module(self, name: str, action: str, payload: dict | None = None,
                     timeout: float = 5.0) -> dict | None:
        """Send a QUERY to a running module and wait for its RESPONSE.

        Returns the response payload dict, or None on timeout/error.
        Used by the profiling endpoint to collect child-process tracemalloc data.
        """
        import uuid
        info = self._modules.get(name)
        if info is None or info.state != ModuleState.RUNNING or info.queue is None:
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
        # Poll send_queue for the response (interleaved with other messages)
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
        # Put back any messages we consumed that weren't our response
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
        """
        from queue import Empty
        from .bus import META_CGN_SIGNAL
        try:
            from .core.bus_health import get_global_monitor
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


# ──────────────────────────────────────────────────────────────────────
# Persistent META-CGN emission log
# ──────────────────────────────────────────────────────────────────────
# Single canonical append-only JSONL file. Written from Guardian drain loop
# (the only point where parent sees every worker emission). Readable by
# `arch_map producers --history N` for cross-Titan cross-time analysis.
# Schema is versioned so we can evolve; schema_version=1 as of 2026-04-15.
_META_CGN_EMISSION_LOG_PATH = os.environ.get(
    "TITAN_META_CGN_EMISSION_LOG", "./data/meta_cgn_emissions.jsonl"
)


def _append_meta_cgn_emission_log(msg: dict, payload: dict) -> None:
    """Append one META_CGN_SIGNAL emission event to the persistent JSONL log.

    Called from the Guardian drain loop after each drained META_CGN_SIGNAL.
    Schema v1: {ts, src, consumer, event_type, intensity, domain, reason,
                schema_version}. Missing optional fields recorded as null.
    """
    import json
    event = {
        "ts": float(msg.get("ts", time.time())),
        "src": str(msg.get("src", "unknown")),
        "consumer": str(payload.get("consumer", "")),
        "event_type": str(payload.get("event_type", "")),
        "intensity": float(payload.get("intensity", 1.0)),
        "domain": (str(payload["domain"])[:40] if payload.get("domain") else None),
        "reason": (str(payload["reason"])[:200] if payload.get("reason") else None),
        "schema_version": 1,
    }
    # Ensure parent dir exists (cheap, idempotent)
    _dir = os.path.dirname(_META_CGN_EMISSION_LOG_PATH)
    if _dir and not os.path.isdir(_dir):
        os.makedirs(_dir, exist_ok=True)
    with open(_META_CGN_EMISSION_LOG_PATH, "a") as f:
        f.write(json.dumps(event, separators=(",", ":")) + "\n")


def _module_wrapper(entry_fn: Callable, name: str, recv_queue, send_queue,
                    config: dict, start_method: str = "fork") -> None:
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

    # Orphan prevention: when titan_main is SIGKILL'd (OOM, kernel panic),
    # its signal handlers don't run and child workers reparent to systemd,
    # accumulate state, and become memory leaks until OOM-killed themselves
    # (see 2026-04-27 cascade incident: T2 IMW orphan grew to 11.4 GB).
    # PR_SET_PDEATHSIG asks the kernel to deliver SIGTERM the moment our
    # parent dies — survives parent SIGKILL because the kernel is the
    # messenger. Combined with a getppid()-poll watcher as backup defense.
    # Worker entry_fns install their own SIGTERM handlers afterward, which
    # catch this signal and perform graceful shutdown (flush WAL, etc.).
    from titan_plugin.core.worker_lifecycle import install_full_protection
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
    bus_client = None
    try:
        from titan_plugin.core.worker_bus_bootstrap import setup_worker_bus
        recv_queue, send_queue, bus_client = setup_worker_bus(
            name, recv_queue, send_queue,
        )
        if bus_client is not None:
            from titan_plugin.core.worker_swap_handler import (
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
            from titan_plugin.core.worker_swap_handler import set_active_swap_state
            set_active_swap_state(None)
        except Exception:  # noqa: BLE001
            pass
        logger.info("Module '%s' process exiting", name)
