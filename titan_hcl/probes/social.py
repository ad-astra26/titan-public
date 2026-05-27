"""titan_hcl.probes.social — social_worker probe (Chunk 11N body).

Phase 11 §11.I.3 contract: returns `ProbeResult.ok_()` IFF social_worker's
in-process readiness sentinel (`_WORKER_READY`) is True. False ⇒ failing
probe with a typed `ModuleError` envelope so titan_hcl + observatory
`/v6/errors` surface the cause.

Lazy import keeps this module orchestrator-side-safe — importing
`titan_hcl.probes.social` from titan_hcl never pulls in the
SocialXGateway / tweepy dependency stack.
"""
from __future__ import annotations

import time
from typing import Any, Optional

from ..core.module_state import ProbeResult
from ..errors import ModuleError, ModuleErrorCode, Severity


def social_worker_probe(bus_client: Optional[Any] = None) -> ProbeResult:
    """Probe social_worker — SocialXGateway gate-status read ≤500ms.

    Returns `ProbeResult.ok_()` only if the module-level `_WORKER_READY`
    sentinel has been flipped True by the social_worker entry function
    (which happens AFTER SocialXGateway construction + gate-status
    initialization completes — including the disabled-path fast bring-up).
    """
    t0 = time.perf_counter()
    try:
        from titan_hcl.modules import social_worker as _soc_mod
        worker_ready = bool(getattr(_soc_mod, "_WORKER_READY", False))
    except Exception as e:  # noqa: BLE001
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        err = ModuleError(
            module_name="social_worker",
            subsystem="probe.import",
            error_code=str(ModuleErrorCode.PROBE_FAILED.value),
            severity=Severity.ERROR,
            message=f"probe could not import social_worker module: {e}",
            detail=repr(e),
        )
        return ProbeResult.fail(error=err, latency_ms=elapsed_ms)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if worker_ready:
        return ProbeResult.ok_(latency_ms=elapsed_ms)

    err = ModuleError(
        module_name="social_worker",
        subsystem="probe.readiness",
        error_code=str(ModuleErrorCode.PROBE_FAILED.value),
        severity=Severity.WARN,
        message="social_worker not ready: _WORKER_READY still False",
        detail=(
            "_WORKER_READY=True requires successful social_worker boot — "
            "SocialXGateway construction + gate-status init "
            "(or disabled-path fast bring-up). "
            "One of those failed — check /v6/errors for the upstream FATAL."
        ),
        suggested_remediation=(
            "Inspect the social_worker boot log for [SocialWorker] "
            "SocialXGateway / gate-status init error lines."),
    )
    return ProbeResult.fail(error=err, latency_ms=elapsed_ms)
