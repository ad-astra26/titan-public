"""
Disk health monitoring + graceful-shutdown-on-full protocol.

Background: on 2026-04-14 the shared T2/T3 VPS disk filled to 100% (driven
by unbounded telemetry-in-git leak + unpruned studio exports). When a
write landed during that window, FAISS created a file and then ran out of
space before writing content — leaving a 0-byte file that blocked memory
worker boot in a crash loop. The cascade propagated through Guardian and
took T2+T3 offline.

This module adds a defence-in-depth runtime monitor that:
  - Polls shutil.disk_usage() on a background OS thread (NOT asyncio — a
    blocked event loop must not delay disk safety observations).
  - Maintains a 4-state machine (HEALTHY / WARNING / CRITICAL / EMERGENCY)
    with hysteresis so transient fluctuations don't flap the bus.
  - Publishes events ONLY on state transitions — edge-detected, discrete.
    Respects the canonical bus-clean invariant (no continuous telemetry
    masquerading as events).
  - On EMERGENCY: requests graceful shutdown via DISK_EMERGENCY event.
    Guardian handles the consumer side and invokes stop_all() cleanly.

Thresholds are configurable but default to:
  HEALTHY   > 5 GB free
  WARNING   2 GB – 5 GB free
  CRITICAL  0.5 GB – 2 GB free
  EMERGENCY < 0.5 GB free

Hysteresis: a 10% band around each boundary prevents flapping when free
space hovers near a threshold. Upgrade (toward EMERGENCY) uses the lower
number; downgrade (toward HEALTHY) uses upper.
"""
import logging
import shutil
import threading
import time
from enum import Enum
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class DiskState(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


# Threshold boundaries in bytes. Used with hysteresis: upgrading to a worse
# state uses the stricter (lower) boundary; downgrading to a better state
# requires free space to rise 10% above that boundary.
_GB = 1024 ** 3
DEFAULT_THRESHOLDS = {
    # upgrade boundaries — free < this triggers entry into the state
    "warning_enter": 5 * _GB,
    "critical_enter": 2 * _GB,
    "emergency_enter": int(0.5 * _GB),
    # downgrade boundaries — free > this triggers exit (hysteresis = +10%)
    "warning_exit": int(5.5 * _GB),
    "critical_exit": int(2.2 * _GB),
    "emergency_exit": int(0.55 * _GB),
}

# Poll interval in seconds. Frequent enough to catch fast disk fill under
# backup/meditation workloads; infrequent enough to be negligible cost.
DEFAULT_POLL_INTERVAL_S = 60.0


class DiskHealthMonitor:
    """Background-thread disk monitor publishing edge-detected events to the bus.

    Usage:
        monitor = DiskHealthMonitor(
            path="/home/antigravity/projects/titan",
            publish_fn=lambda state, free_bytes: bus.publish(...),
            shutdown_fn=lambda reason: guardian.stop_all(reason),
        )
        monitor.start()
        ...
        monitor.stop()

    The monitor only CALLS publish_fn / shutdown_fn — it does not import
    bus/guardian directly, to keep this module free of circular imports
    and testable in isolation."""

    def __init__(
        self,
        path: str,
        publish_fn: Optional[Callable[[DiskState, int], None]] = None,
        shutdown_fn: Optional[Callable[[str], None]] = None,
        thresholds: Optional[dict] = None,
        poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    ):
        self._path = path
        self._publish_fn = publish_fn
        self._shutdown_fn = shutdown_fn
        self._thresholds = thresholds or DEFAULT_THRESHOLDS
        self._poll_interval_s = poll_interval_s

        self._state = DiskState.HEALTHY
        self._last_free_bytes = 0
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # State machine with hysteresis
    # ------------------------------------------------------------------
    def _compute_state(self, free_bytes: int) -> DiskState:
        """Given current free bytes and last state, compute the next state.
        Uses hysteresis on downgrades so we don't flap at threshold boundaries."""
        t = self._thresholds
        cur = self._state

        # Upgrade path (free decreasing): use strict (lower) boundaries
        if free_bytes < t["emergency_enter"]:
            return DiskState.EMERGENCY
        if free_bytes < t["critical_enter"] and cur in (DiskState.HEALTHY, DiskState.WARNING):
            return DiskState.CRITICAL
        if free_bytes < t["warning_enter"] and cur == DiskState.HEALTHY:
            return DiskState.WARNING

        # If currently EMERGENCY, require free > emergency_exit to leave
        if cur == DiskState.EMERGENCY:
            if free_bytes < t["emergency_exit"]:
                return DiskState.EMERGENCY
            # Recovered enough to leave emergency — fall through to normal eval
        # If currently CRITICAL, require free > critical_exit to downgrade
        if cur == DiskState.CRITICAL:
            if free_bytes < t["critical_exit"]:
                return DiskState.CRITICAL
        # If currently WARNING, require free > warning_exit to downgrade
        if cur == DiskState.WARNING:
            if free_bytes < t["warning_exit"]:
                return DiskState.WARNING

        # Free space comfortably clear of all boundaries
        if free_bytes >= t["warning_exit"]:
            return DiskState.HEALTHY
        if free_bytes >= t["critical_exit"]:
            return DiskState.WARNING
        if free_bytes >= t["emergency_exit"]:
            return DiskState.CRITICAL
        return DiskState.EMERGENCY

    # ------------------------------------------------------------------
    # Polling loop (runs on background OS thread)
    # ------------------------------------------------------------------
    def _loop(self):
        """Main polling loop. Runs until stop() is called."""
        logger.info(
            "[DiskHealth] monitor started (path=%s, interval=%.0fs)",
            self._path, self._poll_interval_s,
        )
        while not self._stop_event.is_set():
            try:
                usage = shutil.disk_usage(self._path)
                self._last_free_bytes = usage.free
                new_state = self._compute_state(usage.free)
                if new_state != self._state:
                    self._transition(self._state, new_state, usage.free)
                    self._state = new_state
            except Exception as e:
                # Never let the monitor itself crash. Log and retry next tick.
                logger.warning("[DiskHealth] poll error: %s", e)

            # Wait with cancellation awareness
            self._stop_event.wait(self._poll_interval_s)
        logger.info("[DiskHealth] monitor stopped")

    def _transition(self, old: DiskState, new: DiskState, free_bytes: int):
        """Handle a state transition: log, publish bus event, invoke shutdown
        if EMERGENCY is entered from anywhere else."""
        free_gb = free_bytes / _GB
        logger.warning(
            "[DiskHealth] state change %s → %s (free=%.2f GB)",
            old.value, new.value, free_gb,
        )

        # Publish edge-detected event on bus. Always — architecture invariant:
        # state transitions are the event semantics.
        if self._publish_fn is not None:
            try:
                self._publish_fn(new, free_bytes)
            except Exception as e:
                logger.warning("[DiskHealth] publish_fn error: %s", e)

        # EMERGENCY entry triggers graceful shutdown (unless we're already there).
        if new == DiskState.EMERGENCY and old != DiskState.EMERGENCY:
            logger.error(
                "[DiskHealth] EMERGENCY — disk free %.2f GB — requesting graceful shutdown",
                free_gb,
            )
            if self._shutdown_fn is not None:
                try:
                    self._shutdown_fn(
                        f"disk_emergency: {free_gb:.2f}GB free at {self._path}"
                    )
                except Exception as e:
                    logger.error("[DiskHealth] shutdown_fn error: %s", e)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start(self):
        """Start the monitor on a daemon thread. Idempotent."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, name="DiskHealthMonitor", daemon=True,
        )
        self._thread.start()

    def stop(self):
        """Signal the loop to exit. Does not join — daemon thread dies with process."""
        self._stop_event.set()

    def snapshot(self) -> dict:
        """Current state + free bytes, for /health endpoint consumers."""
        return {
            "state": self._state.value,
            "free_bytes": self._last_free_bytes,
            "free_gb": round(self._last_free_bytes / _GB, 3),
            "path": self._path,
        }


# ----------------------------------------------------------------------
# Boot-time sanity check (Layer 3)
# ----------------------------------------------------------------------

def assert_disk_bootable(path: str, min_free_bytes: int = int(0.5 * _GB)) -> None:
    """Refuse to boot if disk is too full to safely initialise subsystems.

    Called from titan_main.py before heavy imports / state loading. If we
    detect a disk crisis at boot, we log LOUD and raise SystemExit(2). The
    watchdog will see 'not running' but the disk is too broken for a
    restart to help — human intervention required. Exit code 2 is distinct
    from generic crashes (exit 1) so operators know the cause."""
    try:
        usage = shutil.disk_usage(path)
    except Exception as e:
        logger.warning("[DiskHealth] boot check skipped (unreachable): %s", e)
        return
    if usage.free < min_free_bytes:
        free_mb = usage.free / (1024 * 1024)
        limit_mb = min_free_bytes / (1024 * 1024)
        logger.error(
            "=" * 70 + "\n"
            "[DiskHealth] BOOT ABORT — insufficient disk space\n"
            "  path:     %s\n"
            "  free:     %.1f MB\n"
            "  required: %.1f MB\n"
            "  Refusing to start; booting on a full disk corrupts persistent\n"
            "  state (2026-04-14 FAISS 0-byte incident). Free disk, then restart.\n"
            + "=" * 70,
            path, free_mb, limit_mb,
        )
        raise SystemExit(2)
    logger.info(
        "[DiskHealth] boot check OK — %.2f GB free at %s",
        usage.free / _GB, path,
    )
