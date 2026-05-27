"""titan_hcl.probes.cognitive — cognitive_worker probe (Chunk 11N body).

Phase 11 §11.I.3 contract: returns `ProbeResult.ok_()` IFF cognitive_worker's
in-process readiness sentinel (`_WORKER_READY`) is True. False ⇒ failing
probe with a typed `ModuleError` envelope so titan_hcl + observatory
`/v6/errors` surface the cause.

The sentinel lives on the worker process's module-level state
(`titan_hcl.modules.cognitive_worker._WORKER_READY`); the probe runs INSIDE
the worker process (called by `titan_hcl.core.probe_dispatcher.handle_module_probe_request`
after a `MODULE_PROBE_REQUEST` arrives in the worker's recv loop). Lazy
import keeps this module orchestrator-side-safe.
"""
from __future__ import annotations

import time
from typing import Any, Optional

from ..core.module_state import ProbeResult
from ..errors import ModuleError, ModuleErrorCode, Severity


def cognitive_worker_probe(bus_client: Optional[Any] = None) -> ProbeResult:
    """Probe cognitive_worker — epoch loop tick ≤500ms.

    Returns `ProbeResult.ok_()` only if the module-level `_WORKER_READY`
    sentinel has been flipped True by the cognitive_worker entry function
    (which happens AFTER the per-mode bring-up — full cognitive engine
    init OR the lazy-disabled fast path — completes). Any other state ⇒
    probe fails with a `ModuleError` envelope explaining the cause.
    """
    t0 = time.perf_counter()
    try:
        from titan_hcl.modules import cognitive_worker as _cog_mod
        worker_ready = bool(getattr(_cog_mod, "_WORKER_READY", False))
    except Exception as e:  # noqa: BLE001
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        err = ModuleError(
            module_name="cognitive_worker",
            subsystem="probe.import",
            error_code=str(ModuleErrorCode.PROBE_FAILED.value),
            severity=Severity.ERROR,
            message=f"probe could not import cognitive_worker module: {e}",
            detail=repr(e),
        )
        return ProbeResult.fail(error=err, latency_ms=elapsed_ms)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if worker_ready:
        return ProbeResult.ok_(latency_ms=elapsed_ms)

    err = ModuleError(
        module_name="cognitive_worker",
        subsystem="probe.readiness",
        error_code=str(ModuleErrorCode.PROBE_FAILED.value),
        severity=Severity.WARN,
        message="cognitive_worker not ready: _WORKER_READY still False",
        detail=(
            "_WORKER_READY=True requires successful cognitive_worker entry — "
            "either full cognitive engine init OR lazy-disabled fast-path. "
            "Init failed at boot — check /v6/errors for the upstream FATAL."
        ),
        suggested_remediation=(
            "Inspect the cognitive_worker boot log for [CognitiveWorker] "
            "init / engine construction error lines."),
    )
    return ProbeResult.fail(error=err, latency_ms=elapsed_ms)
