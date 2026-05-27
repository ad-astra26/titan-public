"""
titan_hcl.supervisor.core — Supervisor class (Phase 11 §11.I.1 D-SPEC-141 / v1.65.0).

Wraps an Orchestrator instance and adds fault-detection → restart-trigger
routing per locked D5: instead of calling `orchestrator.restart_async(name)`
directly, the Supervisor publishes `MODULE_RESTART_REQUEST(name, reason)` to
the bus. The orchestrator's existing `_handle_module_lifecycle_requests`
subscriber (`scripts/guardian_hcl.py`) translates the bus event into a
local `orchestrator.restart_module(name, reason)` call.

This routing is bus-mediated even though the Orchestrator + Supervisor are
co-resident in the same process for 11E.b.1 — the indirection is the
load-bearing contract that makes 11E.b.2 (kernel-rs peer-spawn) a pure
process-boundary change with no behavioural delta.

For backward compatibility, the legacy `orchestrator.monitor_tick()` path
(which calls `restart_async` directly when restart_on_crash=True) is
preserved. The Supervisor surface is OPT-IN: only callers that explicitly
construct a Supervisor see the D5 bus-mediated routing. Today's tests +
`scripts/guardian_hcl.py` keep using `orchestrator.monitor_tick()` directly,
which still works. Migration to bus-mediated routing in scripts/guardian_hcl.py
happens in 11E.b.2 alongside the kernel-rs peer-spawn.

Status query surface is forwarded to the Orchestrator unchanged.
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Optional

from titan_hcl.bus import (
    DivineBus,
    MODULE_RESTART_REQUEST,
    make_msg,
)

if TYPE_CHECKING:
    from titan_hcl.orchestrator import Orchestrator

logger = logging.getLogger(__name__)


class Supervisor:
    """Phase 11 §11.I.1 / D-SPEC-141 supervisor — fault-detect + restart-trigger.

    Co-resident with the Orchestrator in 11E.b.1; standalone process in
    11E.b.2. The class is intentionally minimal — heartbeat tracking, RSS
    sampling, and process-liveness polling all live on the Orchestrator
    (`monitor_tick`) and are read FROM the orchestrator until the SHM-slot
    contract from 11I module migration lets the Supervisor read state
    directly from `module_<name>_state.bin`.

    Public surface (D5-routed):
      * publish_module_restart_request(name, reason) — emits MODULE_RESTART_REQUEST
      * monitor_tick() — delegates to orchestrator.monitor_tick() in 11E.b.1
      * is_running / is_started / get_status — forwarded to orchestrator
    """

    def __init__(
        self,
        bus: DivineBus,
        orchestrator: "Orchestrator",
        config: Optional[dict] = None,
    ):
        self.bus = bus
        self.orchestrator = orchestrator
        self._config = config or {}

    def publish_module_restart_request(
        self,
        name: str,
        reason: str = "supervisor_request",
        **extra,
    ) -> None:
        """Locked D5 — emit MODULE_RESTART_REQUEST(name, reason) to the bus.

        Destination is the orchestrator's existing subscriber
        (`_handle_module_lifecycle_requests` in `scripts/guardian_hcl.py`).
        Routing is by name and `reply_only` semantics — the orchestrator
        executes `restart_module(name, reason)` synchronously on its
        lifecycle thread.

        `extra` keyword arguments pass through to `orchestrator.restart_module`
        (e.g. `save_first=False` to skip the SAVE_NOW dance).
        """
        payload = {"name": name, "reason": reason, **extra}
        self.bus.publish(make_msg(
            MODULE_RESTART_REQUEST,
            src="supervisor",
            dst="guardian_hcl_lifecycle",
            payload=payload,
        ))
        logger.info(
            "[Supervisor] published MODULE_RESTART_REQUEST(name=%s, reason=%s)",
            name, reason,
        )

    def monitor_tick(self) -> None:
        """Forward to orchestrator.monitor_tick() in 11E.b.1.

        In 11E.b.2 this method gains the SHM-slot poller + emits
        MODULE_RESTART_REQUEST via `publish_module_restart_request` on
        fault detection instead of calling restart_async directly.
        """
        self.orchestrator.monitor_tick()

    def is_running(self, name: str) -> bool:
        return self.orchestrator.is_running(name)

    def is_started(self, name: str) -> bool:
        return self.orchestrator.is_started(name)

    def get_status(self) -> dict:
        return self.orchestrator.get_status()

    def get_layer(self, name: str) -> Optional[str]:
        return self.orchestrator.get_layer(name)

    def layer_stats(self) -> dict:
        return self.orchestrator.layer_stats()
