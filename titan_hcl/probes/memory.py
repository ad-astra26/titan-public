"""titan_hcl.probes.memory — memory_worker probe (Chunk 11N body).

Phase 11 §11.I.3 contract: returns `ProbeResult.ok_()` IFF memory_worker's
in-process readiness sentinel (`_WORKER_READY`) is True. False ⇒ failing
probe with a typed `ModuleError` envelope so titan_hcl + observatory
`/v6/errors` surface the cause.

The sentinel lives on the worker process's module-level state
(`titan_hcl.modules.memory_worker._WORKER_READY`); the probe runs INSIDE
the worker process. Lazy import keeps this module orchestrator-side-safe —
importing `titan_hcl.probes.memory` from titan_hcl never pulls in the
FAISS / Kuzu / DuckDB dependency stack.
"""
from __future__ import annotations

import time
from typing import Any, Optional

from ..core.module_state import ProbeResult
from ..errors import ModuleError, ModuleErrorCode, Severity


def memory_probe(bus_client: Optional[Any] = None) -> ProbeResult:
    """Probe memory — FAISS zero-vector query ≤500ms.

    Returns `ProbeResult.ok_()` only if the module-level `_WORKER_READY`
    sentinel has been flipped True by the memory_worker entry function
    (which happens AFTER the FAISS/Kuzu/DuckDB backend init completes).
    """
    t0 = time.perf_counter()
    try:
        from titan_hcl.modules import memory_worker as _mem_mod
        worker_ready = bool(getattr(_mem_mod, "_WORKER_READY", False))
    except Exception as e:  # noqa: BLE001
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        err = ModuleError(
            module_name="memory",
            subsystem="probe.import",
            error_code=str(ModuleErrorCode.PROBE_FAILED.value),
            severity=Severity.ERROR,
            message=f"probe could not import memory_worker module: {e}",
            detail=repr(e),
        )
        return ProbeResult.fail(error=err, latency_ms=elapsed_ms)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if worker_ready:
        return ProbeResult.ok_(latency_ms=elapsed_ms)

    err = ModuleError(
        module_name="memory",
        subsystem="probe.readiness",
        error_code=str(ModuleErrorCode.PROBE_FAILED.value),
        severity=Severity.WARN,
        message="memory_worker not ready: _WORKER_READY still False",
        detail=(
            "_WORKER_READY=True requires successful memory_worker boot — "
            "FAISS index load + Kuzu graph open + DuckDB connection. "
            "One of those failed — check /v6/errors for the upstream FATAL."
        ),
        suggested_remediation=(
            "Inspect the memory_worker boot log for [MemoryWorker] FAISS / "
            "Kuzu / DuckDB init error lines."),
    )
    return ProbeResult.fail(error=err, latency_ms=elapsed_ms)
