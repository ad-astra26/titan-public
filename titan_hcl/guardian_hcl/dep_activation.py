"""
titan_hcl.guardian_hcl.dep_activation — Phase 11 §11.I.1 back-compat re-export shim.

Canonical location: `titan_hcl.orchestrator.dep_activation`
(`OrchestratorDepActivationMixin`). The legacy mixin name
`GuardianDepActivationMixin` is preserved as an alias.

Remove in 11E.b.2 once all callsites migrate.
"""
from titan_hcl.orchestrator.dep_activation import (  # noqa: F401
    OrchestratorDepActivationMixin,
)

# Back-compat alias for the pre-Phase-11 name.
GuardianDepActivationMixin = OrchestratorDepActivationMixin

__all__ = [
    "OrchestratorDepActivationMixin",
    "GuardianDepActivationMixin",
]
