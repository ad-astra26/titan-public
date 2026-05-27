"""titan_hcl.probes.cgn — cgn_worker probe (Chunk 11N body).

Phase 11 §11.I.3 contract: returns `ProbeResult.ok_()` IFF cgn_worker's
in-process readiness sentinel (`_WORKER_READY`) is True. False ⇒ failing
probe with a typed `ModuleError` envelope so titan_hcl + observatory
`/v6/errors` surface the cause.

Lazy import keeps this module orchestrator-side-safe — importing
`titan_hcl.probes.cgn` from titan_hcl never pulls in the torch / CGN
model dependency stack.
"""
from __future__ import annotations

import time
from typing import Any, Optional

from ..core.module_state import ProbeResult
from ..errors import ModuleError, ModuleErrorCode, Severity


def cgn_probe(bus_client: Optional[Any] = None) -> ProbeResult:
    """Probe cgn — model.forward(zero_input) ok ≤1s.

    Returns `ProbeResult.ok_()` only if the module-level `_WORKER_READY`
    sentinel has been flipped True by the cgn_worker entry function
    (which happens AFTER the CGN model load + warmup completes).
    """
    t0 = time.perf_counter()
    try:
        from titan_hcl.modules import cgn_worker as _cgn_mod
        worker_ready = bool(getattr(_cgn_mod, "_WORKER_READY", False))
    except Exception as e:  # noqa: BLE001
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        err = ModuleError(
            module_name="cgn",
            subsystem="probe.import",
            error_code=str(ModuleErrorCode.PROBE_FAILED.value),
            severity=Severity.ERROR,
            message=f"probe could not import cgn_worker module: {e}",
            detail=repr(e),
        )
        return ProbeResult.fail(error=err, latency_ms=elapsed_ms)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if worker_ready:
        return ProbeResult.ok_(latency_ms=elapsed_ms)

    err = ModuleError(
        module_name="cgn",
        subsystem="probe.readiness",
        error_code=str(ModuleErrorCode.PROBE_FAILED.value),
        severity=Severity.WARN,
        message="cgn_worker not ready: _WORKER_READY still False",
        detail=(
            "_WORKER_READY=True requires successful cgn_worker boot — "
            "model checkpoint load + forward-pass warmup. "
            "One of those failed — check /v6/errors for the upstream FATAL."
        ),
        suggested_remediation=(
            "Inspect the cgn_worker boot log for [CGNWorker] model load / "
            "warmup error lines."),
    )
    return ProbeResult.fail(error=err, latency_ms=elapsed_ms)
