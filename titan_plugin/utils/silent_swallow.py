"""
titan_plugin/utils/silent_swallow.py — Pattern C helper for the
"silent-swallow remediation" effort (initiated 2026-04-25 after
BUG-T1-CONSCIOUSNESS-67D-STATE-VECTOR was discovered to have been
silently active for 37 days).

The pre-existing pattern across ~442 sites was:

    try:
        risky_thing()
    except Exception as e:
        logger.debug("...", e)

This swallows exceptions at DEBUG level, which is below the threshold of
all our scanners (arch_map errors, brain log triage, watchdog) — so a
real failure can latently corrupt cognition for weeks before anyone
notices. The originating bug (outer_trinity._collect_extended) silently
collapsed Trinity 132D consciousness epochs to 67D on T1 for 5+ weeks.

Pattern C — "WARN + counter + safe fallback":

    from titan_plugin.utils.silent_swallow import swallow_warn

    try:
        risky_thing()
    except Exception as e:
        swallow_warn("[Module] context", e, key="module.context",
                     throttle=100)
        # ... existing safe fallback continues here ...

What `swallow_warn` does:
  1. Bumps the per-key failure counter in the global SILENT_SWALLOW_COUNTERS
     registry (read by warning_monitor_worker + arch_map silent-swallows).
  2. Logs at WARNING (not DEBUG) on the first occurrence, then every
     `throttle`-th occurrence, with exc_info on the throttled emits to keep
     stack traces available without flooding logs.
  3. Records last_failure_ts + last_failure_msg for diagnostic surfacing.

Per directive_error_visibility.md (codified 2026-04-25): NO silent
exceptions in critical paths. Every WARNING+ event MUST be visible to
the warning_monitor_worker.

Per fundamental_symmetry_requirements.md: dim/symmetry-critical paths
MUST emit WARNING on fallback so the symmetry collapse is detectable.
"""

import logging
import threading
import time
from collections import defaultdict
from typing import Optional

logger = logging.getLogger(__name__)

# Global registry — one entry per (key) call site.
# Read by warning_monitor_worker via get_swallow_state().
# Thread-safe via _LOCK; read-mostly so contention should be negligible.
SILENT_SWALLOW_COUNTERS: dict[str, dict] = defaultdict(lambda: {
    "count": 0,
    "first_seen_ts": 0.0,
    "last_seen_ts": 0.0,
    "last_msg": "",
    "exc_types": defaultdict(int),
})

_LOCK = threading.Lock()


def swallow_warn(prefix: str, exc: BaseException, *,
                 key: Optional[str] = None,
                 throttle: int = 100,
                 first_warn_exc_info: bool = True,
                 _logger: Optional[logging.Logger] = None) -> None:
    """Record a swallowed exception per Pattern C.

    Args:
        prefix: Log prefix, e.g. "[OuterTrinity] _collect_extended failed".
            Should be unique enough that the WARNING is identifiable.
        exc: The caught exception.
        key: Stable identifier for this call site (e.g. "outer_trinity.extended").
            If None, uses prefix. The key is what warning_monitor groups by.
        throttle: After the first WARNING, only emit every Nth occurrence
            to avoid flood. Default 100. Counter still increments every call.
        first_warn_exc_info: Include exc_info=True on first WARNING so the
            stack trace is captured. Throttled re-warns include exc_info too.
        _logger: Optional logger override (else use the caller's via stack
            inspection — currently always falls back to module logger).

    Returns:
        None. Caller proceeds to its safe-fallback code path.

    Pattern usage:
        try:
            risky_thing()
        except Exception as e:
            swallow_warn("[Module] op failed", e, key="module.op")
            # Safe fallback continues...
    """
    k = key or prefix
    now = time.time()
    msg_str = f"{type(exc).__name__}: {exc}"

    with _LOCK:
        entry = SILENT_SWALLOW_COUNTERS[k]
        entry["count"] += 1
        if entry["first_seen_ts"] == 0.0:
            entry["first_seen_ts"] = now
        entry["last_seen_ts"] = now
        entry["last_msg"] = msg_str
        entry["exc_types"][type(exc).__name__] += 1
        count = entry["count"]

    log = _logger or logger
    is_first = count == 1
    is_throttled = count > 1 and (count % throttle == 0)

    if is_first or is_throttled:
        log.warning(
            "%s — %s (count=%d, key=%s, throttle=%d). "
            "See arch_map silent-swallows --runtime + warning_monitor.",
            prefix, msg_str, count, k, throttle,
            exc_info=first_warn_exc_info,
        )


def get_swallow_state() -> dict:
    """Snapshot of SILENT_SWALLOW_COUNTERS for warning_monitor + arch_map.

    Returns a deep copy (defaultdict→dict, exc_types defaultdict→dict)
    so caller can serialize to JSON without leaking defaultdict semantics.
    """
    with _LOCK:
        return {
            k: {
                "count": v["count"],
                "first_seen_ts": v["first_seen_ts"],
                "last_seen_ts": v["last_seen_ts"],
                "last_msg": v["last_msg"],
                "exc_types": dict(v["exc_types"]),
            }
            for k, v in SILENT_SWALLOW_COUNTERS.items()
        }


def reset_swallow_state() -> None:
    """Test-only — reset registry. Production code never calls this."""
    with _LOCK:
        SILENT_SWALLOW_COUNTERS.clear()
