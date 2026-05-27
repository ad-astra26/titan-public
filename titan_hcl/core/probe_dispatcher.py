"""titan_hcl/core/probe_dispatcher.py — Phase 11 Chunk 11D probe bus-RPC infra.

Per SPEC §11.I.2 / §11.I.3 (D-SPEC-141 / v1.65.0):

- **Worker side** (`handle_module_probe_request`): when a worker receives
  `MODULE_PROBE_REQUEST(name, probe_id)`, it (1) writes `state=probing` to
  its own SHM slot, (2) runs `probe_fn(bus)` with wall-time budget ≤2s,
  (3) writes `state=running` or `state=unhealthy` plus `last_probe_result`
  to its SHM slot, and (4) replies `MODULE_PROBE_RESPONSE(probe_id, result)`
  to titan_hcl. The dual SHM-write + bus-RPC reply lets titan_hcl verify
  via BOTH channels per §11.I.2.

- **Orchestrator side** (`ProbeDispatcher`): titan_hcl's 1Hz SHM poll
  detects a slot's state transitioning to `booted` → constructs + dispatches
  `MODULE_PROBE_REQUEST` via async bus-RPC with `asyncio.wait_for` timeout
  per locked D3 → awaits `MODULE_PROBE_RESPONSE` → returns `ProbeResult`.
  The 11F orchestrator wraps this with the 3-state escalation logic per
  locked D4 (1 retry → restart-trigger → DISABLED).

NO bus broadcasts for state transitions are emitted by either side per
locked D1/D2. Bus carries only the request/response pair + command events.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Callable, Optional

from ..bus import MODULE_PROBE_REQUEST, MODULE_PROBE_RESPONSE
from ..errors import ModuleError, ModuleErrorCode, Severity
from .module_state import ModuleStateWriter, ProbeResult

logger = logging.getLogger(__name__)


# ── Probe budget (locked D3) ──────────────────────────────────────────────────
PROBE_TIMEOUT_S: float = 2.0


# ── Worker-side: handle MODULE_PROBE_REQUEST ─────────────────────────────────

def handle_module_probe_request(
    msg: dict,
    *,
    probe_fn: Optional[Callable[..., ProbeResult]],
    send_queue: Any,
    module_name: str,
    state_writer: ModuleStateWriter,
    bus_client: Any = None,
) -> ProbeResult:
    """Worker-side handler for an inbound MODULE_PROBE_REQUEST envelope.

    Called from the worker's recv-loop when it receives:
        {"type": "MODULE_PROBE_REQUEST", "src": "titan_hcl",
         "dst": "<module_name>", "rid": <probe_id>, "payload": {"name": ...,
         "probe_id": <probe_id>}}

    The handler runs the worker's `probe_fn(bus_client)` if provided, writes
    the resulting state + ProbeResult to the worker's SHM slot, AND sends
    MODULE_PROBE_RESPONSE back on `send_queue`.

    Modules without a `probe_fn` (the common case for legacy workers) get a
    trivial pass-through `ProbeResult.ok_()` per SPEC §11.I.2 backward-compat
    contract.

    Args:
        msg: the raw MODULE_PROBE_REQUEST bus message dict.
        probe_fn: optional callable returning a ProbeResult. Called with
                  `bus_client` if provided; pass None for the trivial-pass
                  contract.
        send_queue: worker→broker send_queue (the MODULE_PROBE_RESPONSE goes here).
        module_name: this worker's canonical name (carried in the response payload).
        state_writer: the worker's ModuleStateWriter (writes state transitions
                      to its SHM slot).
        bus_client: optional bus-client object passed to probe_fn for any
                    needed bus access during probe (typically the worker's
                    DivineBus client).

    Returns:
        The ProbeResult that was sent (useful for tests + logging).
    """
    probe_id = msg.get("rid") or (msg.get("payload") or {}).get("probe_id") or str(uuid.uuid4())

    # 1. Transition to PROBING in SHM (single-writer per slot).
    try:
        state_writer.write_state("probing")
    except Exception as e:
        logger.warning(
            "[handle_module_probe_request] %s: SHM write_state(probing) failed: %s",
            module_name, e,
        )

    # 2. Run the probe (or trivial-pass) with wall-time budget enforcement.
    t0 = time.perf_counter()
    result: ProbeResult
    if probe_fn is None:
        result = ProbeResult.ok_(latency_ms=(time.perf_counter() - t0) * 1000.0)
    else:
        try:
            raw = probe_fn(bus_client)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if elapsed_ms > PROBE_TIMEOUT_S * 1000.0:
                # Over budget — treat as failed per §11.I.3 ("Probes >2s log
                # a WARN + treated as failed").
                logger.warning(
                    "[handle_module_probe_request] %s: probe budget exceeded "
                    "(%.1fms > %.1fms)",
                    module_name, elapsed_ms, PROBE_TIMEOUT_S * 1000.0,
                )
                err = ModuleError(
                    module_name=module_name,
                    subsystem="probe",
                    error_code=str(ModuleErrorCode.PROBE_BUDGET_EXCEEDED.value),
                    severity=Severity.ERROR,
                    message=f"probe exceeded {PROBE_TIMEOUT_S}s budget ({elapsed_ms:.1f}ms)",
                    detail=f"probe_fn={getattr(probe_fn, '__name__', repr(probe_fn))}",
                )
                result = ProbeResult.fail(error=err, latency_ms=elapsed_ms)
            elif isinstance(raw, ProbeResult):
                # Caller already returned a ProbeResult — use as-is but ensure
                # latency reflects actual wall time (caller might have left at 0.0).
                if raw.latency_ms <= 0.0:
                    result = ProbeResult(
                        ok=raw.ok,
                        latency_ms=elapsed_ms,
                        error_envelope=raw.error_envelope,
                    )
                else:
                    result = raw
            else:
                # Probe returned something weird — treat as failed.
                err = ModuleError(
                    module_name=module_name,
                    subsystem="probe",
                    error_code=str(ModuleErrorCode.PROBE_FAILED.value),
                    severity=Severity.ERROR,
                    message=f"probe_fn returned {type(raw).__name__}, expected ProbeResult",
                    detail=f"probe_fn={getattr(probe_fn, '__name__', repr(probe_fn))} returned {raw!r}",
                )
                result = ProbeResult.fail(error=err, latency_ms=elapsed_ms)
        except Exception as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            err = ModuleError.from_exception(
                e,
                module_name=module_name,
                subsystem="probe",
                error_code=ModuleErrorCode.PROBE_FAILED,
                severity=Severity.ERROR,
            )
            result = ProbeResult.fail(error=err, latency_ms=elapsed_ms)

    # 3. Publish state to SHM atomically with last_probe_result + last_error.
    next_state = "running" if result.ok else "unhealthy"
    try:
        state_writer.write_state(
            next_state,
            last_probe_result=result,
            last_error=result.error_envelope,
        )
    except Exception as e:
        logger.warning(
            "[handle_module_probe_request] %s: SHM write_state(%s) failed: %s",
            module_name, next_state, e,
        )

    # 4. Send MODULE_PROBE_RESPONSE back on send_queue (correlation_id-routed).
    response_msg = {
        "type": MODULE_PROBE_RESPONSE,
        "src": module_name,
        "dst": "titan_hcl",
        "ts": time.time(),
        "rid": probe_id,
        "payload": {
            "probe_id": probe_id,
            "name": module_name,
            "result": result.as_wire_dict(),
        },
    }
    try:
        if hasattr(send_queue, "put_nowait"):
            send_queue.put_nowait(response_msg)
        elif hasattr(send_queue, "publish"):
            send_queue.publish(response_msg)
        else:
            logger.warning(
                "[handle_module_probe_request] %s: send_queue has neither put_nowait "
                "nor publish — MODULE_PROBE_RESPONSE dropped (rid=%s)",
                module_name, probe_id,
            )
    except Exception as e:
        logger.warning(
            "[handle_module_probe_request] %s: send MODULE_PROBE_RESPONSE "
            "(rid=%s) failed: %s",
            module_name, probe_id, e,
        )

    return result


# ── Orchestrator-side: dispatch MODULE_PROBE_REQUEST ──────────────────────────

class ProbeDispatcher:
    """titan_hcl-side probe dispatcher.

    Holds a DivineBus reference + dispatches MODULE_PROBE_REQUEST async to
    target workers. The 11F orchestrator owns the boot-pipeline integration;
    this class is the bus-RPC primitive on top of which the 3-state
    escalation per locked D4 is built.

    Usage:

        dispatcher = ProbeDispatcher(bus)
        result = await dispatcher.dispatch_probe("agno_worker", timeout_s=2.0)
        if result.ok:
            ...
        else:
            ...  # 11F runs retry → restart-trigger → DISABLED escalation
    """

    def __init__(self, bus: Any, *, src: str = "titan_hcl") -> None:
        self._bus = bus
        self._src = src

    async def dispatch_probe(
        self,
        module_name: str,
        *,
        timeout_s: float = PROBE_TIMEOUT_S,
        probe_id: Optional[str] = None,
    ) -> ProbeResult:
        """Dispatch a probe RPC to `module_name` and await the reply.

        Returns:
            ProbeResult: the worker's reply. On bus failure / timeout, returns
            a synthetic failed ProbeResult with appropriate ModuleError.
        """
        pid = probe_id or str(uuid.uuid4())
        request_payload = {
            "type": MODULE_PROBE_REQUEST,
            "src": self._src,
            "dst": module_name,
            "ts": time.time(),
            "rid": pid,
            "payload": {
                "name": module_name,
                "probe_id": pid,
            },
        }

        t0 = time.perf_counter()
        try:
            reply = await asyncio.wait_for(
                self._bus.request_async(
                    src=self._src,
                    dst=module_name,
                    payload=request_payload,
                    timeout=timeout_s,
                ),
                timeout=timeout_s + 0.5,  # outer guard against bus stalls
            )
        except asyncio.TimeoutError:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            err = ModuleError(
                module_name=module_name,
                subsystem="probe.dispatch",
                error_code=str(ModuleErrorCode.PROBE_TIMEOUT.value),
                severity=Severity.ERROR,
                message=f"probe RPC timed out after {timeout_s}s",
                detail=f"probe_id={pid}",
            )
            return ProbeResult.fail(error=err, latency_ms=elapsed_ms)
        except Exception as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            err = ModuleError.from_exception(
                e,
                module_name=module_name,
                subsystem="probe.dispatch",
                error_code=ModuleErrorCode.BUS_PUBLISH_FAILED,
                severity=Severity.ERROR,
            )
            return ProbeResult.fail(error=err, latency_ms=elapsed_ms)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if not isinstance(reply, dict):
            err = ModuleError(
                module_name=module_name,
                subsystem="probe.dispatch",
                error_code=str(ModuleErrorCode.PROBE_FAILED.value),
                severity=Severity.ERROR,
                message=f"probe RPC reply was {type(reply).__name__}, expected dict",
                detail=f"probe_id={pid}",
            )
            return ProbeResult.fail(error=err, latency_ms=elapsed_ms)

        payload = reply.get("payload") if isinstance(reply.get("payload"), dict) else reply
        result_d = (payload or {}).get("result")
        if not isinstance(result_d, dict):
            err = ModuleError(
                module_name=module_name,
                subsystem="probe.dispatch",
                error_code=str(ModuleErrorCode.PROBE_FAILED.value),
                severity=Severity.ERROR,
                message="probe RPC reply missing `result` dict",
                detail=f"probe_id={pid}, reply_keys={list(reply.keys())}",
            )
            return ProbeResult.fail(error=err, latency_ms=elapsed_ms)
        try:
            result = ProbeResult.from_wire_dict(result_d)
        except Exception as e:
            err = ModuleError.from_exception(
                e,
                module_name=module_name,
                subsystem="probe.dispatch",
                error_code=ModuleErrorCode.PROBE_FAILED,
                severity=Severity.ERROR,
                message="probe RPC reply could not be deserialized into ProbeResult",
            )
            return ProbeResult.fail(error=err, latency_ms=elapsed_ms)
        # If the worker omitted latency_ms, fill from dispatcher-side wall time.
        if result.latency_ms <= 0.0:
            result = ProbeResult(
                ok=result.ok,
                latency_ms=elapsed_ms,
                error_envelope=result.error_envelope,
            )
        return result
