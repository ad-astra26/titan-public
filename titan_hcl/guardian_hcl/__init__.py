"""
titan_hcl.guardian_hcl — Guardian L1 supervisor package.

Carved from titan_hcl/guardian.py per SPEC §11.B.4 / D-SPEC-135 / v1.62.0.
Public surface re-exports preserved so existing call sites keep working:
    from titan_hcl.guardian_hcl import Guardian, ModuleSpec, ModuleInfo, ...

Internal organization:
    core.py             — class Guardian (lifecycle, monitor_tick, supervision)
    reload.py [MOVED to titan_hcl/reload.py per Phase 11 §11.I.6 / D-SPEC-141 / locked D6] — OrchestratorReloadMixin (D-SPEC-50 reload_module + 7-step seq)
    dep_activation.py   — GuardianDepActivationMixin (D-SPEC-90 §11.G.2.5)
    module_registry.py  — ModuleState / ModuleSpec / ModuleInfo / ReloadState

Public API per RFP §3C.3 6C is bus messages (MODULE_RELOAD_REQUEST/ACK,
MODULE_RESTART_REQUEST, SUPERVISION_*), not Python imports. The imports
below remain for legacy in-process callers until the standalone process
cutover (chunk 6F) replaces them with thin bus clients.
"""
from titan_hcl.guardian_hcl.core import (
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
    _module_wrapper,
)
from titan_hcl.guardian_hcl.module_registry import (
    ModuleState,
    ModuleSpec,
    ModuleInfo,
    ReloadState,
    _append_meta_cgn_emission_log,
)
from titan_hcl.reload import OrchestratorReloadMixin
from titan_hcl.guardian_hcl.dep_activation import GuardianDepActivationMixin

__all__ = [
    "Guardian",
    "OrchestratorReloadMixin",
    "GuardianDepActivationMixin",
    "ModuleState",
    "ModuleSpec",
    "ModuleInfo",
    "ReloadState",
    "_module_wrapper",
    "_append_meta_cgn_emission_log",
    # Heartbeat / restart-window tunables re-exported for legacy callers
    # (tests, arch_map). Pre-6C carve these were attributes on
    # titan_hcl.guardian module-level; restored at package-level here.
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
