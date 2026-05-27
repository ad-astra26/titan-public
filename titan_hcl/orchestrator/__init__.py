"""
titan_hcl.orchestrator — Phase 11 §11.I.1 orchestrator package (D-SPEC-141 / v1.65.0).

Owns module spawn / start / stop / restart / dep activation (§11.G.2.5) /
hot-reload spawn (§8.3) / probe execution / lazy-start. Counterpart to
`titan_hcl.supervisor` (fault detection + ModuleError aggregation +
MODULE_RESTART_REQUEST emission per locked D5).

Public surface:
    from titan_hcl.orchestrator import (
        Orchestrator, ModuleSpec, ModuleInfo, ModuleState, ReloadState,
    )

Back-compat: `titan_hcl.guardian_hcl` re-exports `Guardian = Orchestrator`
and the mixin/dataclass symbols so existing callsites + 30+ tests keep
working unchanged. The 11E.b.2 kernel-rs peer-spawn chunk removes the
guardian_hcl back-compat layer once all callsites migrate.
"""
from titan_hcl.orchestrator.core import (
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
from titan_hcl.orchestrator.dep_activation import OrchestratorDepActivationMixin
from titan_hcl.orchestrator.module_registry import (
    ModuleInfo,
    ModuleSpec,
    ModuleState,
    ReloadState,
    _append_meta_cgn_emission_log,
)
from titan_hcl.reload import OrchestratorReloadMixin

__all__ = [
    "Orchestrator",
    "Guardian",  # back-compat alias = Orchestrator
    "OrchestratorReloadMixin",
    "OrchestratorDepActivationMixin",
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
