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
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from multiprocessing import Process
from typing import Callable, Optional

from .bus import (
    AnyQueue,
    DivineBus,
    MODULE_CRASHED,
    MODULE_HEARTBEAT,
    MODULE_READY,
    MODULE_SHUTDOWN,
    make_msg,
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
        self._guardian_queue = bus.subscribe("guardian")
        self._stop_requested = False
        self._module_lock = threading.RLock()  # serialize start/stop/restart to prevent duplicate spawns

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
        if spec.name in self._modules:
            logger.warning("[Guardian] Module '%s' already registered, updating spec", spec.name)
        self._modules[spec.name] = ModuleInfo(spec=spec)
        logger.info("[Guardian] Registered module '%s' (autostart=%s, lazy=%s, rss_limit=%dMB, hb_timeout=%.0fs)",
                     spec.name, spec.autostart, spec.lazy, spec.rss_limit_mb, spec.heartbeat_timeout)

    def start(self, name: str) -> bool:
        """Start a specific module process. Thread-safe via _module_lock."""
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

            # Create module's bus queues (bidirectional)
            import multiprocessing
            ctx = multiprocessing.get_context("fork")
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
                    args=(info.spec.entry_fn, name, info.queue, info.send_queue, info.spec.config),
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
        # Force-kill any surviving child processes of the worker.
        # CRITICAL: must NOT killpg if the worker shares our process group
        # (which it does today — no setpgrp in worker startup). A naive
        # killpg(worker_pgid) when worker_pgid == our_pgid is parent-suicide:
        # titan_main dies and the API goes dark until watchdog reboot.
        # Observed 2026-04-14 on T1 when spirit worker didn't respond to
        # SIGTERM in 15s — killpg fired and took titan_main with it.
        # Safe pattern: only killpg if worker is in a DIFFERENT group.
        if info.pid is not None:
            try:
                import signal
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
        # Queue cleanup — use cancel_join_thread() instead of join_thread().
        # Why: the consumer child was SIGKILL'd above, but forked siblings
        # may still hold inherited read-end FDs on the pipe (phantom FDs),
        # keeping the pipe open. A pending os.write() inside the queue's
        # feeder thread then blocks indefinitely on a full pipe instead of
        # receiving EPIPE, deadlocking Guardian's asyncio loop (observed
        # 2026-04-14 on T1, cascading API hang; matches I-018 on T2).
        # cancel_join_thread() is the documented Python fix for exactly
        # this case — we accept the loss of any unflushed bytes, which
        # were destined for a SIGKILL'd process anyway.
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
                        "SAVE_NOW", "guardian", name,
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
                    if (m.get("type") == "SAVE_DONE"
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
                logger.error("[Guardian] Module '%s' exceeded %d restarts in %.0fs window — disabling (auto-re-enable in %.0fs)",
                             name, self._max_restarts_in_window, self._restart_window_seconds, REENABLE_COOLDOWN_S)
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
        """Start all modules that have autostart=True."""
        for name, info in self._modules.items():
            if info.spec.autostart:
                self.start(name)

    def stop_all(self, reason: str = "shutdown") -> None:
        """Gracefully stop all running modules."""
        self._stop_requested = True
        for name, info in self._modules.items():
            if info.state in (ModuleState.RUNNING, ModuleState.STARTING, ModuleState.UNHEALTHY):
                self.stop(name, reason=reason)

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

            # Heartbeat timeout check — uses per-module timeout
            if info.state == ModuleState.RUNNING:
                hb_timeout = info.spec.heartbeat_timeout
                if now - info.last_heartbeat > hb_timeout:
                    logger.warning("[Guardian] Module '%s' heartbeat timeout (%.1fs > %.0fs limit)",
                                   name, now - info.last_heartbeat, hb_timeout)
                    with self._module_lock:
                        info.state = ModuleState.UNHEALTHY
                        if info.spec.restart_on_crash:
                            self.restart(name, reason="heartbeat_timeout")
                    continue

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
            }
        return result

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


def _module_wrapper(entry_fn: Callable, name: str, recv_queue, send_queue, config: dict) -> None:
    """
    Wrapper that runs in the child process.
    Sets up logging, calls the entry function, handles crashes.
    """
    # Configure logging for child process
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [%(levelname)s] [{name}] %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(name)
    logger.info("Module '%s' process started (pid=%d)", name, os.getpid())

    try:
        entry_fn(recv_queue, send_queue, name, config)
    except KeyboardInterrupt:
        logger.info("Module '%s' interrupted", name)
    except Exception as e:
        logger.error("Module '%s' crashed: %s", name, e, exc_info=True)
        raise
    finally:
        logger.info("Module '%s' process exiting", name)
