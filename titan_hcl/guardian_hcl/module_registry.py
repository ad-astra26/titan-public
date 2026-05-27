"""
titan_hcl.guardian_hcl.module_registry — Phase 11 §11.I.1 back-compat re-export shim.

Canonical location: `titan_hcl.orchestrator.module_registry` (D-SPEC-141 /
v1.65.0). Remove in 11E.b.2 once all callsites migrate.
"""
from titan_hcl.orchestrator.module_registry import (  # noqa: F401
    ModuleInfo,
    ModuleSpec,
    ModuleState,
    ReloadState,
    _append_meta_cgn_emission_log,
)

__all__ = [
    "ModuleInfo",
    "ModuleSpec",
    "ModuleState",
    "ReloadState",
    "_append_meta_cgn_emission_log",
]
