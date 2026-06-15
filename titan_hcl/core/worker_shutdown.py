"""
Worker shutdown self-save — bus-INDEPENDENT graceful persistence on SIGTERM.

RFP_supervision_lifecycle §7.D / Phase D.1 (2026-06-16). Closes the
graceful-shutdown ORDERING bug: a full restart used to hang → SIGKILL because
the only path that persisted worker state on shutdown was the orchestrator's
``SAVE_NOW``→``SAVE_DONE`` handshake **over the bus**, and the bus (owned by
the Rust L0 kernel) is torn down before/concurrent with the Python module
drain. Every remaining module's save-handshake then blocked on a dead bus
until ``TimeoutStopSec`` → SIGKILL, and workers SIGTERM'd directly by systemd's
``KillMode=control-group`` died with NO checkpoint at all (they had no SIGTERM
handler — the long-standing "workers install their own SIGTERM handlers"
comment in worker_lifecycle.py / orchestrator.core was STALE for the
titan_hcl/modules/* workers; only the persistence layer actually did).

The fix makes worker persistence depend on NOTHING but the worker's own
process receiving SIGTERM:

1. ``install_worker_sigterm_handler()`` — installed by Guardian's
   ``_module_wrapper`` for every spawned worker, right after
   ``install_full_protection()``. It is the FLOOR handler: it converts SIGTERM
   into ``KeyboardInterrupt`` (same mechanism as scripts/titan_hcl.py's parent
   handler), which unwinds the worker's blocking ``recv_queue.get(timeout=…)``
   loop so the wrapper's ``finally`` runs. A worker that installs its own,
   richer SIGTERM handler later (e.g. persistence_entry.py's asyncio
   stop_event) simply overrides this floor — no conflict.

2. ``register_shutdown_save(name, fn)`` — a worker that holds bus-independent
   critical state (a DuckDB/SQLite WAL, an on-disk state file) registers its
   EXISTING checkpoint/save callable ONCE at startup. The callable MUST be
   idempotent (it may also run on an orderly MODULE_SHUTDOWN) and MUST NOT
   touch the bus.

3. ``run_shutdown_saves()`` — called by ``_module_wrapper``'s ``finally`` on
   EVERY worker exit path (KeyboardInterrupt/SIGTERM, normal return, crash).
   Runs every registered callback in the worker's normal execution context
   (NOT inside the signal handler — DB work in a signal handler is the bug
   persistence_entry.py:64 deliberately avoids). Run-once guarded so an
   orderly shutdown that already saved + a trailing finally don't double-pay
   unboundedly; per-callback try/except so one failing saver can't starve the
   rest. SPEC §11.B (G16: no shutdown may corrupt persistent data) +
   SPEC §18.4 (ordered, bounded drain).

Scope fence: this module does NOT remove the bus ``SAVE_NOW`` path — that path
remains load-bearing for shadow-swap (B.1) and orderly MODULE_SHUTDOWN. This
adds a bus-independent FLOOR so a dead/slow bus can never cost data or hang the
drain.
"""
from __future__ import annotations

import logging
import os
import signal
import threading
from typing import Callable, List, Tuple

logger = logging.getLogger(__name__)

# Process-global registry. Each worker process is single-threaded for these
# purposes (one entry_fn); registration happens at startup, execution at
# shutdown — no concurrent mutation in practice. A plain list + lock is ample.
_REGISTRY: List[Tuple[str, Callable[[], None]]] = []
_REGISTRY_LOCK = threading.Lock()
_SAVES_RAN = False  # run-once guard for run_shutdown_saves()


def register_shutdown_save(name: str, fn: Callable[[], None]) -> None:
    """Register a bus-INDEPENDENT, idempotent save/checkpoint callable.

    Called once by a stateful worker at startup. ``fn`` takes no args and
    persists this worker's critical state without touching the bus (e.g. a
    ``PRAGMA wal_checkpoint(FULL)`` or an atomic state-file write). It runs on
    every shutdown path via ``run_shutdown_saves()``.

    Idempotency is required: ``fn`` may run after the worker already saved on
    an orderly MODULE_SHUTDOWN.
    """
    with _REGISTRY_LOCK:
        _REGISTRY.append((name, fn))
    logger.info("[worker_shutdown] registered bus-independent save for '%s'", name)


def run_shutdown_saves() -> int:
    """Run every registered save callback once. Returns the count that ran.

    Idempotent across calls (run-once guarded). Per-callback exceptions are
    caught + logged so one failure can't starve the rest. Safe to call from the
    worker's ``finally`` on any exit path; a no-op if nothing was registered.
    """
    global _SAVES_RAN
    with _REGISTRY_LOCK:
        if _SAVES_RAN:
            return 0
        _SAVES_RAN = True
        callbacks = list(_REGISTRY)
    ran = 0
    for name, fn in callbacks:
        try:
            fn()
            ran += 1
            logger.info("[worker_shutdown] bus-independent save complete for '%s'", name)
        except Exception as e:  # noqa: BLE001 — a failing saver must not block the rest
            logger.error(
                "[worker_shutdown] bus-independent save FAILED for '%s': %s",
                name, e, exc_info=True,
            )
    return ran


def _sigterm_to_keyboardinterrupt(_signum, _frame):
    """FLOOR SIGTERM handler: raise KeyboardInterrupt to unwind the worker loop.

    Minimal + signal-safe (no I/O, no DB work) — the actual persistence runs
    later in the wrapper's ``finally`` via ``run_shutdown_saves()``, in normal
    context. Mirrors scripts/titan_hcl.py's parent SIGTERM→KeyboardInterrupt
    conversion so the worker's blocking ``recv_queue.get(timeout=…)`` returns
    and the graceful-exit path runs instead of the process dying mid-loop with
    no checkpoint.
    """
    raise KeyboardInterrupt("SIGTERM received (worker floor handler)")


def install_worker_sigterm_handler() -> bool:
    """Install the floor SIGTERM→KeyboardInterrupt handler for this worker.

    Idempotent (set-or-replace). Returns True if installed. A worker that
    installs its own SIGTERM handler afterward (persistence_entry.py) overrides
    this — intended; that handler is richer (asyncio stop_event) and still
    reaches a graceful save.
    """
    try:
        signal.signal(signal.SIGTERM, _sigterm_to_keyboardinterrupt)
        return True
    except (ValueError, OSError) as e:
        # ValueError: not in main thread (shouldn't happen at worker entry).
        logger.debug("[worker_shutdown] SIGTERM handler install skipped (pid=%d): %s",
                     os.getpid(), e)
        return False
