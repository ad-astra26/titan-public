"""titan_hcl.probes._common — shared probe primitives (11H Chunk).

`shell_probe(module_name)` returns a fast-path ProbeResult.ok_() and
records wall-time latency. Used by every 11H probe shell until 11I
migrates the worker to a real per-module check.

When a worker is migrated in 11I, its probe shell body is replaced
with a real liveness check (e.g., `_agent is not None` for agno_worker,
FAISS zero-vector query for memory). The signature stays
`Callable[[Optional[BusClient]], ProbeResult]` so the orchestrator-side
wiring + the worker-side `handle_module_probe_request` adapter need no
further change at probe-replacement time.
"""
from __future__ import annotations

import time
from typing import Any, Optional

from ..core.module_state import ProbeResult


def shell_probe(module_name: str, bus_client: Optional[Any] = None) -> ProbeResult:
    """Phase 11 §11.I.3 trivial-pass probe shell — returns ok_().

    Used as the body of each 11H probe BEFORE the worker is migrated in
    11I. Records wall-time latency so the orchestrator's `last_probe_result`
    accounting still gets a real ms reading.
    """
    t0 = time.perf_counter()
    # Light no-op so the probe takes a measurable (non-zero) latency even
    # on the shell path. This makes the orchestrator's probe-budget
    # histograms meaningful before per-worker migration lands.
    _ = module_name
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return ProbeResult.ok_(latency_ms=elapsed_ms)
