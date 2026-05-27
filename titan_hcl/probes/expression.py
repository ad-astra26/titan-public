"""titan_hcl.probes.expression — expression_worker probe (Chunk 11N body).

Phase 11 §11.I.3 contract: returns `ProbeResult.ok_()` IFF expression_worker's
in-process readiness sentinel (`_WORKER_READY`) is True. False ⇒ failing
probe with a typed `ModuleError` envelope so titan_hcl + observatory
`/v6/errors` surface the cause.

Lazy import keeps this module orchestrator-side-safe — importing
`titan_hcl.probes.expression` from titan_hcl never pulls in the
ExpressionManager / composer dependency stack.
"""
from __future__ import annotations

import time
from typing import Any, Optional

from ..core.module_state import ProbeResult
from ..errors import ModuleError, ModuleErrorCode, Severity


def expression_worker_probe(bus_client: Optional[Any] = None) -> ProbeResult:
    """Probe expression_worker — ExpressionManager.evaluate_all dry-run ≤500ms.

    Returns `ProbeResult.ok_()` only if the module-level `_WORKER_READY`
    sentinel has been flipped True by the expression_worker entry function
    (which happens AFTER ExpressionManager init completes).
    """
    t0 = time.perf_counter()
    try:
        from titan_hcl.modules import expression_worker as _expr_mod
        worker_ready = bool(getattr(_expr_mod, "_WORKER_READY", False))
    except Exception as e:  # noqa: BLE001
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        err = ModuleError(
            module_name="expression_worker",
            subsystem="probe.import",
            error_code=str(ModuleErrorCode.PROBE_FAILED.value),
            severity=Severity.ERROR,
            message=f"probe could not import expression_worker module: {e}",
            detail=repr(e),
        )
        return ProbeResult.fail(error=err, latency_ms=elapsed_ms)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if worker_ready:
        return ProbeResult.ok_(latency_ms=elapsed_ms)

    err = ModuleError(
        module_name="expression_worker",
        subsystem="probe.readiness",
        error_code=str(ModuleErrorCode.PROBE_FAILED.value),
        severity=Severity.WARN,
        message="expression_worker not ready: _WORKER_READY still False",
        detail=(
            "_WORKER_READY=True requires successful expression_worker boot — "
            "ExpressionManager construction + initial evaluate_all dry-run. "
            "One of those failed — check /v6/errors for the upstream FATAL."
        ),
        suggested_remediation=(
            "Inspect the expression_worker boot log for [ExpressionWorker] "
            "ExpressionManager / composer init error lines."),
    )
    return ProbeResult.fail(error=err, latency_ms=elapsed_ms)
