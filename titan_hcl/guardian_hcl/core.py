"""
titan_hcl.guardian_hcl.core — Phase 11 §11.I.1 back-compat re-export shim.

The Orchestrator class (formerly Guardian) lives in
`titan_hcl.orchestrator.core` per D-SPEC-141 / v1.65.0. This module preserves
the legacy import path `from titan_hcl.guardian_hcl.core import Guardian, ...`
used by a handful of tests and `scripts/_phase6_carve_guardian.py`.

Remove in 11E.b.2 once all callsites migrate to
`from titan_hcl.orchestrator import ...`.
"""
from titan_hcl.orchestrator.core import (  # noqa: F401
    DEFAULT_RSS_LIMIT_MB,
    HEARTBEAT_INTERVAL,
    HEARTBEAT_TIMEOUT,
    MAX_RESTARTS_IN_WINDOW,
    MAX_STARVED_CYCLES,
    MIN_CPU_DELTA_FOR_ALIVE,
    REENABLE_COOLDOWN_S,
    RESTART_BACKOFF_BASE,
    RESTART_WINDOW_SECONDS,
    SUSTAINED_UPTIME_RESET,
    Guardian,
    Orchestrator,
    _module_wrapper,
)

__all__ = [
    "Guardian",
    "Orchestrator",
    "_module_wrapper",
    "DEFAULT_RSS_LIMIT_MB",
    "HEARTBEAT_INTERVAL",
    "HEARTBEAT_TIMEOUT",
    "MAX_RESTARTS_IN_WINDOW",
    "MAX_STARVED_CYCLES",
    "MIN_CPU_DELTA_FOR_ALIVE",
    "REENABLE_COOLDOWN_S",
    "RESTART_BACKOFF_BASE",
    "RESTART_WINDOW_SECONDS",
    "SUSTAINED_UPTIME_RESET",
]
