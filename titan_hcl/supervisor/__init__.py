"""
titan_hcl.supervisor — Phase 11 §11.I.1 supervisor package (D-SPEC-141 / v1.65.0).

Counterpart to `titan_hcl.orchestrator`. Owns fault detection
(heartbeat-stale, RSS overflow, process-dead, MODULE_ERROR severity=FATAL)
and ModuleError aggregation. Emits MODULE_RESTART_REQUEST(name, reason) per
locked D5 instead of calling the orchestrator's restart paths directly —
the routing target is the orchestrator's existing MODULE_RESTART_REQUEST
subscriber wired in `scripts/guardian_hcl.py._handle_module_lifecycle_requests`.

In 11E.b.1 (single-process) the Supervisor wraps an Orchestrator instance
and reads its in-process `_modules` dict for state observation. In 11E.b.2
(kernel-rs peer-spawn) the Supervisor becomes the standalone process body —
it stops sharing an Orchestrator object reference and starts reading the
per-module SHM slots (G18 / §11.I.5) at 1Hz instead.
"""
from titan_hcl.supervisor.core import Supervisor

__all__ = ["Supervisor"]
