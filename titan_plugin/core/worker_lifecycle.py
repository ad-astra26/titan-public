"""
Worker lifecycle protections — kernel-level orphan prevention.

When the Guardian-supervised parent (titan_main) is killed gracefully (SIGTERM),
its existing signal handler runs `_kill_stragglers` via atexit — child workers
get a clean shutdown signal. But when the parent is killed UNGRACEFULLY
(SIGKILL from OOM-killer, kernel panic, kill -9), no Python signal handler
runs and no atexit hook fires. Child workers are reparented to systemd
(PID 1) and continue executing — orphans. Single-writer daemons (IMW,
observatory_writer) accumulate WAL state, hold DB locks, and grow until
they're OOM-killed themselves, often taking sibling Titans down with them.

This module installs two complementary defenses in every worker subprocess:

1. **PR_SET_PDEATHSIG** — Linux kernel-level: when our parent dies (any
   reason, including SIGKILL), the kernel sends us SIGTERM. Our existing
   per-worker SIGTERM handlers then perform graceful shutdown (flush WAL,
   release locks, exit). Survives parent SIGKILL because the kernel is
   the messenger, not Python.

2. **Parent watcher thread** — backup polling defense. Every `interval`
   seconds, check `os.getppid() == 1`. If true, we've been reparented to
   init → SIGTERM ourselves. Catches edge cases where PDEATHSIG is unset
   or stripped (e.g., across exec() boundaries) and provides a defense on
   non-Linux platforms (no-op there since prctl is Linux-only).

Both defenses route through SIGTERM, so workers' existing graceful
shutdown handlers (e.g., persistence_entry.py:62, body_worker, etc.)
continue to work unchanged. No worker code needs to know about this
module — Guardian's `_module_wrapper` calls `install_full_protection()`
at child-process start, before invoking `entry_fn`.

Phase B.2.1 extension (2026-04-27 mid-day):
- `clear_parent_death_signal()` / `reset_parent_death_signal()` — workers
  strip their PDEATHSIG temporarily during a coordinated kernel swap so
  they outlive the old kernel. Bus-as-supervision (worker_swap_handler.py)
  is the replacement supervision signal during/after swap.
- `pause_parent_watcher(state)` / `resume_parent_watcher(state, relaxed=)`
  — suspend the 30s polling during swap-pending; resume in relaxed mode
  post-adoption (getppid()==1 alone no longer triggers self-SIGTERM).
- `WatcherState` — small mutable state carrier so the watcher reads pause
  and relaxed-mode flags on each iteration; allows external pause/resume
  without restarting the thread.

Why this is the canonical fix vs. ad-hoc watchdog reaping:
- Defense lives IN the protected process, not in an external watchdog
  (which itself can die / be missing / lag behind cron cadence).
- Synchronous with parent death (kernel signal at exit moment), not
  polling at 5-min cron resolution.
- Composes with shadow-swap (workers correctly die when their kernel is
  shut down, not when the OTHER kernel is). Per-process lineage, not
  global pgrep-by-name.
- Closes BUG-B1-SHARED-LOCKS partially: when shadow-boot fails partway
  and original kernel restarts workers via Guardian, leftover original
  workers don't become orphans — they got killed when original kernel
  dropped them.

Related: titan_plugin/guardian.py:_module_wrapper (call site);
scripts/titan_main.py:512 (existing parent SIGTERM handler — unchanged
by this module; complements it for the OOM/SIGKILL case it cannot cover);
titan_plugin/core/worker_swap_handler.py (B.2.1 supervision-transfer
state machine that drives pause/resume/relaxed-mode transitions).
"""
from __future__ import annotations

import ctypes
import logging
import os
import signal
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Linux prctl constant — kernel API stable since 2.6.
_PR_SET_PDEATHSIG = 1

# B.2.1: bus-as-supervision timeout — worker self-SIGTERMs if relaxed-mode
# AND ppid==1 AND bus has been unreachable for this many seconds. Tunable
# via microkernel.b2_1_supervision_timeout_seconds in titan_params.toml.
B2_1_DEFAULT_SUPERVISION_TIMEOUT_S = 30.0


@dataclass
class WatcherState:
    """Mutable state for parent_watcher; readable on every poll iteration.

    Allows external pause/resume + B.2.1 relaxed-mode transitions without
    tearing down + recreating the watcher thread. Kept tiny + lock-free
    (single-writer in test code or worker_swap_handler; single-reader in
    the watcher loop) — Python's GIL makes flag reads atomic for our needs.
    """
    stop_event: threading.Event
    interval: float = 30.0
    sig: int = signal.SIGTERM
    thread: Optional[threading.Thread] = None
    # B.2.1 runtime flags
    _paused: bool = False
    _b2_1_relaxed_mode: bool = False
    _bus_unreachable_since: Optional[float] = None
    _supervision_timeout_s: float = B2_1_DEFAULT_SUPERVISION_TIMEOUT_S

    def is_alive(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def join(self, timeout: float | None = None) -> None:
        """Backward-compat: pre-B.2.1 callers used start_parent_watcher().join()."""
        if self.thread is not None:
            self.thread.join(timeout=timeout)

    @property
    def name(self) -> str:
        """Backward-compat: pre-B.2.1 callers asserted on .name."""
        return self.thread.name if self.thread is not None else "parent_watcher"

    @property
    def daemon(self) -> bool:
        """Backward-compat: pre-B.2.1 callers asserted on .daemon."""
        return self.thread.daemon if self.thread is not None else True

    def mark_bus_unreachable(self) -> None:
        """Called by worker_swap_handler when bus client reports disconnect."""
        if self._bus_unreachable_since is None:
            self._bus_unreachable_since = time.time()

    def mark_bus_healthy(self) -> None:
        """Called by worker_swap_handler when bus client reports reconnect."""
        self._bus_unreachable_since = None


def install_parent_death_signal(sig: int = signal.SIGTERM) -> bool:
    """Linux-only: ask kernel to deliver `sig` when parent process dies.

    Survives parent SIGKILL because the signal is dispatched by the
    kernel at parent-exit, not by any Python-level handler. Returns
    True if the prctl call succeeded; False on any failure (non-Linux,
    libc unavailable, prctl rejected). Failures are logged at DEBUG —
    the parent_watcher thread provides a fallback.

    Race correctness: a child created via fork()/spawn() can race the
    parent's death — parent might die between the fork() syscall and
    the prctl() call below. We handle this by re-checking
    `os.getppid() == 1` immediately after prctl and self-signaling if
    we were already reparented. Without this re-check, a worker started
    moments before parent OOM would silently miss the death notice.

    Args:
        sig: Signal to receive on parent death. Default SIGTERM, which
             every Guardian-supervised worker already handles gracefully.
             Use signal.SIGKILL only if graceful shutdown is impossible
             (workers won't get to flush state — last resort).
    """
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
    except OSError:
        logger.debug("[worker_lifecycle] libc.so.6 unavailable — skipping prctl")
        return False

    rc = libc.prctl(_PR_SET_PDEATHSIG, sig, 0, 0, 0)
    if rc != 0:
        errno = ctypes.get_errno()
        logger.debug("[worker_lifecycle] prctl(PR_SET_PDEATHSIG) rc=%d errno=%d", rc, errno)
        return False

    # Race check: parent may have died between fork()/spawn() and now.
    # In that case ppid is already 1; PDEATHSIG won't fire (parent already
    # gone). Self-signal explicitly so we still exit cleanly.
    if os.getppid() == 1:
        logger.warning(
            "[worker_lifecycle] parent already dead at startup (race) — self-signal %s",
            sig,
        )
        os.kill(os.getpid(), sig)
    return True


def clear_parent_death_signal() -> bool:
    """Phase B.2.1: clear PDEATHSIG so worker survives parent death.

    Workers strip their death signal temporarily during a coordinated
    kernel swap so they outlive the old kernel. They do NOT re-arm
    against shadow (Linux PDEATHSIG resolves the *current* parent at
    prctl-call time; post-reparent the parent is init/PID 1, useless
    protection). Bus-as-supervision (worker_swap_handler) is the
    replacement supervision signal for adopted workers.

    Returns True on success; False if prctl unavailable or rejected
    (non-Linux). Failure is non-fatal: worker stays armed and will
    take the improved-B.1 path (die with old, get respawned by shadow).
    """
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
    except OSError:
        logger.debug("[worker_lifecycle] libc.so.6 unavailable — clear_parent_death_signal noop")
        return False

    rc = libc.prctl(_PR_SET_PDEATHSIG, 0, 0, 0, 0)
    if rc != 0:
        errno = ctypes.get_errno()
        logger.debug(
            "[worker_lifecycle] prctl(PR_SET_PDEATHSIG, 0) rc=%d errno=%d",
            rc, errno,
        )
        return False
    return True


def reset_parent_death_signal(sig: int = signal.SIGTERM) -> bool:
    """Phase B.2.1: re-arm PDEATHSIG against the CURRENT parent PID.

    Used in P-2c unwind path when swap is aborted: workers re-arm
    against old kernel (still alive) to restore strict supervision.
    NOT used in happy-path post-adopt — that path leaves PDEATHSIG
    cleared and uses bus-as-supervision permanently (D2/D8).
    """
    return install_parent_death_signal(sig=sig)


def pause_parent_watcher(state: WatcherState) -> None:
    """Phase B.2.1: suspend parent_watcher polling during swap-pending."""
    if state is not None:
        state._paused = True


def resume_parent_watcher(state: WatcherState, *, relaxed: bool = False) -> None:
    """Phase B.2.1: resume parent_watcher polling.

    Args:
        relaxed: If True (post-adoption), getppid()==1 alone no longer
                 triggers self-SIGTERM; supervision becomes the bus
                 connection — worker self-SIGTERMs only if BOTH ppid==1
                 AND bus unreachable for ≥30s. If False (post-cancel),
                 strict semantics restored.
    """
    if state is None:
        return
    state._paused = False
    state._b2_1_relaxed_mode = relaxed


def start_parent_watcher(interval: float = 30.0,
                         sig: int = signal.SIGTERM,
                         stop_event: threading.Event | None = None,
                         supervision_timeout_s: float = B2_1_DEFAULT_SUPERVISION_TIMEOUT_S,
                         ) -> WatcherState:
    """Start a daemon thread that polls getppid() and self-signals if reparented.

    Backup defense for the cases PDEATHSIG cannot cover:
    - Non-Linux platforms (prctl unavailable)
    - Race after exec() that strips PDEATHSIG (Linux clears it across exec)
    - Buggy/old kernels where PDEATHSIG behavior is unreliable

    Phase B.2.1 extension: the watcher reads `state._paused` (skip iteration
    if True) and `state._b2_1_relaxed_mode` (only signal if ppid==1 AND
    bus unreachable for ≥supervision_timeout_s). Both flags are flipped
    via pause_parent_watcher/resume_parent_watcher; bus health is signaled
    via state.mark_bus_unreachable / mark_bus_healthy.

    Polling cadence is intentionally coarse (default 30s) — orphan detection
    doesn't need to be instant; the goal is bounded cleanup time, not zero
    latency. Daemon=True so this thread doesn't block process exit.

    Args:
        interval: Poll cadence in seconds.
        sig: Signal to deliver to self when ppid==1 detected.
        stop_event: Optional Event the watcher honors as a clean shutdown
                    signal (mainly for tests to avoid thread leaks between
                    cases). If set, the watcher exits without firing.
        supervision_timeout_s: Bus-as-supervision timeout for relaxed mode.

    Returns the WatcherState carrying the thread + B.2.1 flags. Pre-B.2.1
    callers can ignore everything but `.is_alive()`.
    """
    state = WatcherState(
        stop_event=stop_event or threading.Event(),
        interval=interval,
        sig=sig,
        _supervision_timeout_s=supervision_timeout_s,
    )

    def _watch() -> None:
        while True:
            if state.stop_event.is_set():
                return
            if state._paused:
                state.stop_event.wait(state.interval)
                continue
            if os.getppid() == 1:
                should_signal = True
                if state._b2_1_relaxed_mode:
                    # Post-adoption: only self-SIGTERM if bus is ALSO
                    # unreachable for ≥supervision_timeout_s. ppid==1
                    # alone is legitimate (we're an adopted worker; old
                    # kernel exited; init is our parent now; bus is
                    # supervision).
                    bus_dead_since = state._bus_unreachable_since
                    if bus_dead_since is None:
                        should_signal = False
                    elif (time.time() - bus_dead_since) < state._supervision_timeout_s:
                        should_signal = False
                if should_signal:
                    logger.warning(
                        "[worker_lifecycle] parent_watcher: ppid=1 (reparented to init) — self-signal %s%s",
                        state.sig,
                        " (relaxed-mode supervision-via-bus timeout)" if state._b2_1_relaxed_mode else "",
                    )
                    try:
                        os.kill(os.getpid(), state.sig)
                    except OSError:
                        # Last-ditch: hard exit. We're an orphan; nothing else matters.
                        os._exit(1)
                    return
            state.stop_event.wait(state.interval)

    state.thread = threading.Thread(
        target=_watch, name="parent_watcher", daemon=True,
    )
    state.thread.start()
    return state


def install_full_protection(sig: int = signal.SIGTERM,
                            watcher_interval: float = 30.0,
                            supervision_timeout_s: float = B2_1_DEFAULT_SUPERVISION_TIMEOUT_S,
                            ) -> dict:
    """Install both PDEATHSIG + parent-watcher in a single call.

    Idempotent — calling twice is harmless (prctl is set-or-replace; a
    second watcher thread is wasteful but not incorrect). In practice
    Guardian's _module_wrapper calls this exactly once per worker, before
    invoking the worker's own entry function.

    Returns a small dict for observability:
        {"pdeathsig_installed": bool, "watcher_started": bool,
         "watcher_state": WatcherState}
    Tests + Guardian boot logs read this to confirm protection is in place.
    Phase B.2.1: the `watcher_state` key is the WatcherState object that
    callers (worker_swap_handler) pass to pause/resume helpers.
    """
    pdeathsig_ok = install_parent_death_signal(sig=sig)
    watcher_state = start_parent_watcher(
        interval=watcher_interval,
        sig=sig,
        supervision_timeout_s=supervision_timeout_s,
    )
    return {
        "pdeathsig_installed": pdeathsig_ok,
        "watcher_started": watcher_state.is_alive(),
        "watcher_state": watcher_state,
    }
