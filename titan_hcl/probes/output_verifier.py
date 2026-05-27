"""titan_hcl.probes.output_verifier — output_verifier probe (Chunk 11N body).

Phase 11 §11.I.3 contract: returns `ProbeResult.ok_()` IFF output_verifier_worker's
in-process readiness sentinel (`_WORKER_READY`) is True. False ⇒ failing
probe with a typed `ModuleError` envelope so titan_hcl + observatory
`/v6/errors` surface the cause.

Lazy import keeps this module orchestrator-side-safe — importing
`titan_hcl.probes.output_verifier` from titan_hcl never pulls in the
Ed25519 / OutputVerifier dependency stack.
"""
from __future__ import annotations

import time
from typing import Any, Optional

from ..core.module_state import ProbeResult
from ..errors import ModuleError, ModuleErrorCode, Severity


def output_verifier_probe(bus_client: Optional[Any] = None) -> ProbeResult:
    """Probe output_verifier — Ed25519 sign a 1-byte payload ≤200ms.

    Returns `ProbeResult.ok_()` only if the module-level `_WORKER_READY`
    sentinel has been flipped True by the output_verifier_worker entry
    function (which happens AFTER Ed25519 key load + sign self-test
    completes).
    """
    t0 = time.perf_counter()
    try:
        from titan_hcl.modules import output_verifier_worker as _ov_mod
        worker_ready = bool(getattr(_ov_mod, "_WORKER_READY", False))
    except Exception as e:  # noqa: BLE001
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        err = ModuleError(
            module_name="output_verifier",
            subsystem="probe.import",
            error_code=str(ModuleErrorCode.PROBE_FAILED.value),
            severity=Severity.ERROR,
            message=f"probe could not import output_verifier_worker module: {e}",
            detail=repr(e),
        )
        return ProbeResult.fail(error=err, latency_ms=elapsed_ms)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if worker_ready:
        return ProbeResult.ok_(latency_ms=elapsed_ms)

    err = ModuleError(
        module_name="output_verifier",
        subsystem="probe.readiness",
        error_code=str(ModuleErrorCode.PROBE_FAILED.value),
        severity=Severity.WARN,
        message="output_verifier_worker not ready: _WORKER_READY still False",
        detail=(
            "_WORKER_READY=True requires successful output_verifier_worker boot — "
            "Ed25519 key load + sign self-test. "
            "One of those failed — check /v6/errors for the upstream FATAL."
        ),
        suggested_remediation=(
            "Inspect the output_verifier_worker boot log for [OutputVerifierWorker] "
            "Ed25519 key load / sign error lines."),
    )
    return ProbeResult.fail(error=err, latency_ms=elapsed_ms)
