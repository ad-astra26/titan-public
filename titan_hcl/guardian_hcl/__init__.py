"""
titan_hcl.guardian_hcl — Phase 11 §11.I.1 back-compat layer (D-SPEC-141 / v1.65.0).

Phase 6 (D-SPEC-135) carved the in-process Guardian into the standalone
`guardian_hcl` package. Phase 11 §11.I.1 splits the role into Orchestrator
(spawn/start/stop — `titan_hcl.orchestrator`) and Supervisor (fault
detection + restart-trigger — `titan_hcl.supervisor`).

This package is now a BACK-COMPAT RE-EXPORT layer so the 30+ existing
callsites + tests that do `from titan_hcl.guardian_hcl import Guardian,
ModuleSpec, ...` keep working unchanged. The 11E.b.2 kernel-rs peer-spawn
chunk migrates the callsites and removes this layer.

New code should import from the canonical locations:
    from titan_hcl.orchestrator import Orchestrator, ModuleSpec, ...
    from titan_hcl.supervisor  import Supervisor
"""
from titan_hcl.orchestrator import (
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
    ModuleInfo,
    ModuleSpec,
    ModuleState,
    Orchestrator,
    OrchestratorDepActivationMixin,
    OrchestratorReloadMixin,
    ReloadState,
    _append_meta_cgn_emission_log,
    _module_wrapper,
)

# Phase 11 §11.I.1 back-compat alias for the pre-rename mixin name.
GuardianDepActivationMixin = OrchestratorDepActivationMixin

__all__ = [
    "Guardian",
    "Orchestrator",
    "OrchestratorReloadMixin",
    "OrchestratorDepActivationMixin",
    "GuardianDepActivationMixin",  # back-compat alias
    "ModuleState",
    "ModuleSpec",
    "ModuleInfo",
    "ReloadState",
    "_module_wrapper",
    "_append_meta_cgn_emission_log",
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
