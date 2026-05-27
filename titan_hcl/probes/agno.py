"""titan_hcl.probes.agno — agno_worker probe (Chunk 11H + 11N body).

Phase 11 §11.I.3 contract: returns `ProbeResult.ok_()` IFF agno_worker's
in-process readiness sentinels (`_AGENT_READY` + `_OVG_READY`) are both
True. Either False ⇒ failing probe with a typed `ModuleError` envelope
so titan_hcl + observatory `/v6/errors` surface the cause.

The sentinels live on the worker process's module-level state
(`titan_hcl.modules.agno_worker._AGENT_READY` / `_OVG_READY`); the
probe runs INSIDE the worker process (called by
`titan_hcl.core.probe_dispatcher.handle_module_probe_request` after a
`MODULE_PROBE_REQUEST` arrives in the worker's recv loop). Lazy import
keeps this module orchestrator-side-safe — importing
`titan_hcl.probes.agno` from titan_hcl never pulls in the Agno
framework / OutputVerifier / OVG dependency stack.
"""
from __future__ import annotations

import time
from typing import Any, Optional

from ..core.module_state import ProbeResult
from ..errors import ModuleError, ModuleErrorCode, Severity


def agno_worker_probe(bus_client: Optional[Any] = None) -> ProbeResult:
    """Probe agno_worker — Agent + OVG must both be initialized.

    Returns `ProbeResult.ok_()` only if both module-level sentinels
    have been flipped True by the agno_worker entry function (which
    happens AFTER `_init_worker_plugin_and_agent` returns successfully
    AND after the eager OVG warmup completes). Any other state ⇒ probe
    fails with a `ModuleError` envelope explaining which sentinel is
    not yet True.
    """
    t0 = time.perf_counter()
    try:
        from titan_hcl.modules import agno_worker as _agno_mod
        agent_ready = bool(getattr(_agno_mod, "_AGENT_READY", False))
        ovg_ready = bool(getattr(_agno_mod, "_OVG_READY", False))
    except Exception as e:  # noqa: BLE001
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        err = ModuleError(
            module_name="agno_worker",
            subsystem="probe.import",
            error_code=str(ModuleErrorCode.PROBE_FAILED.value),
            severity=Severity.ERROR,
            message=f"probe could not import agno_worker module: {e}",
            detail=repr(e),
        )
        return ProbeResult.fail(error=err, latency_ms=elapsed_ms)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if agent_ready and ovg_ready:
        return ProbeResult.ok_(latency_ms=elapsed_ms)

    # Surface which sentinel is missing so /v6/errors makes the cause obvious.
    missing = []
    if not agent_ready:
        missing.append("_AGENT_READY")
    if not ovg_ready:
        missing.append("_OVG_READY")
    err = ModuleError(
        module_name="agno_worker",
        subsystem="probe.readiness",
        error_code=str(ModuleErrorCode.PROBE_FAILED.value),
        severity=Severity.WARN,
        message=f"agno_worker not ready: {', '.join(missing)} still False",
        detail=(
            "_AGENT_READY=True requires successful _init_worker_plugin_and_agent; "
            "_OVG_READY=True requires successful eager OVG warmup. "
            "Either failed at boot — check /v6/errors for the upstream FATAL."
        ),
        suggested_remediation=(
            "Inspect the agno_worker boot log for [AgnoWorker] Agent "
            "construction / OVG warmup error lines."),
    )
    return ProbeResult.fail(error=err, latency_ms=elapsed_ms)
