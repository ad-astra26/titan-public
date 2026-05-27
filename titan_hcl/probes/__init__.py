"""
titan_hcl.probes — Phase 11 §11.I.3 / D-SPEC-141 / v1.65.0 per-module probes (Chunk 11H).

One probe_fn per "heaviest worker" per RFP §3H.2 / §3H.10:

    agno_worker, cognitive_worker, memory, cgn, synthesis,
    observatory, social_worker, expression_worker, meditation,
    output_verifier

Probe contract (SPEC §11.I.3):
  * Signature: `Callable[[BusClient], ProbeResult]` — called inside the
    worker process by `handle_module_probe_request` when a
    MODULE_PROBE_REQUEST arrives. `bus_client` is the worker's
    DivineBus client (or None when the worker doesn't expose one).
  * Wall-time budget: ≤2s. The worker-side handler enforces this with
    a hard check + `PROBE_BUDGET_EXCEEDED` ModuleError.
  * Pure observation — must NOT mutate worker state, take main-thread
    locks, or block the asyncio loop.

11H scope: every probe in this module is a thin liveness shell. The
shells let the 10 ModuleSpec registrations declare their probe_fn now;
the bodies are filled in alongside the 11I worker-entry migration when
each worker grows the recv-loop hook + sets the module-level state
indicators the probe inspects. Until 11I lands per worker, the probes
return `ProbeResult.ok_()` — equivalent to the legacy trivial-pass
contract per §11.I.2.

Importing this module never touches worker-side heavy state — every
probe lazy-imports any worker module it inspects so the orchestrator
can register the probe_fn without paying the worker's import cost.
"""
from __future__ import annotations

from titan_hcl.probes.agno import agno_worker_probe
from titan_hcl.probes.cgn import cgn_probe
from titan_hcl.probes.cognitive import cognitive_worker_probe
from titan_hcl.probes.expression import expression_worker_probe
from titan_hcl.probes.meditation import meditation_probe
from titan_hcl.probes.memory import memory_probe
from titan_hcl.probes.observatory import observatory_probe
from titan_hcl.probes.output_verifier import output_verifier_probe
from titan_hcl.probes.social import social_worker_probe
from titan_hcl.probes.synthesis import synthesis_probe

__all__ = [
    "agno_worker_probe",
    "cgn_probe",
    "cognitive_worker_probe",
    "expression_worker_probe",
    "meditation_probe",
    "memory_probe",
    "observatory_probe",
    "output_verifier_probe",
    "social_worker_probe",
    "synthesis_probe",
]


# Convenience: name → probe_fn map for the §3H.2 11H roster. Used by
# the matrix-driven catalog wiring in titan_hcl.module_catalog and by
# tests that iterate over the canonical 10.
PROBE_REGISTRY: dict[str, object] = {
    "agno_worker":       agno_worker_probe,
    "cognitive_worker":  cognitive_worker_probe,
    "memory":            memory_probe,
    "cgn":               cgn_probe,
    "synthesis":         synthesis_probe,
    "observatory":       observatory_probe,
    "social_worker":     social_worker_probe,
    "expression_worker": expression_worker_probe,
    "meditation":        meditation_probe,
    "output_verifier":   output_verifier_probe,
}
