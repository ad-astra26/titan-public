"""
titan_hcl.supervisor.core — Supervisor class (Phase 11 §11.I.1 D-SPEC-141 / v1.65.0).

The Supervisor owns FAULT DETECTION + RESTART TRIGGERING per locked D5:

  * Heartbeat-staleness detection (SHM `last_heartbeat` per §11.I.5 / locked
    D1, augmented by the legacy bus MODULE_HEARTBEAT path during the W3
    cascade transition).
  * Process-liveness check (PID alive via `multiprocessing.Process.is_alive`).
  * RSS budget enforcement.
  * Disabled-module re-enable after cooldown.
  * On fault: emits `MODULE_RESTART_REQUEST(name, reason)` to the bus
    (`publish_module_restart_request`) — the Orchestrator's existing
    subscriber (`_handle_module_lifecycle_requests` in
    `scripts/guardian_hcl.py`) translates the bus event into
    `orchestrator.restart_module(name, reason)`.

The Supervisor holds an `orchestrator` reference for 11E.b.1 (co-resident in
one process). Accessing `orchestrator._modules` for state mutations is the
documented Phase 11 transitional path; 11E.b.2 (kernel-rs peer-spawn) replaces
this with SHM-slot reads + bus-mediated state-change events. The bus
indirection is already in place via `publish_module_restart_request` so
11E.b.2 is a pure process-boundary change with no behavioural delta.

What lives WHERE post-Phase-11 §11.I.1 split:

  Orchestrator                              Supervisor
  ────────────────────────────────────────  ──────────────────────────────────
  register / start / stop / restart_*       monitor_tick (this file)
  start_all / Phase A + B pipeline          publish_module_restart_request
  probe dispatch / probe wait               heartbeat-timeout detection
  hot-reload spawn (D-SPEC-50)              process-liveness check
  dep activation (§11.G.2.5 D-SPEC-90)      RSS budget enforcement
  adopt_worker (B.2.1)                      disabled→enabled cooldown
  _process_guardian_messages                publishes MODULE_RESTART_REQUEST
    (BUS_PEER_DIED + MODULE_HEARTBEAT       on detected fault
     + BUS_WORKER_ADOPT_REQUEST +
     MODULE_RELOAD_REQUEST)
  fleet_ready SHM publish (G21 gated)
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


# Constants imported lazily from orchestrator/core to avoid circular import
# at module load time. Phase 11 §11.I.5 lazy-binding keeps the supervisor
# module standalone-loadable for the future 11E.b.2 split.
_MIN_CPU_DELTA_FOR_ALIVE: Optional[float] = None
_MAX_STARVED_CYCLES: Optional[int] = None
_REENABLE_COOLDOWN_S: Optional[float] = None


def _load_constants() -> None:
    global _MIN_CPU_DELTA_FOR_ALIVE, _MAX_STARVED_CYCLES, _REENABLE_COOLDOWN_S
    if _MIN_CPU_DELTA_FOR_ALIVE is None:
        from titan_hcl.orchestrator.core import (
            MAX_STARVED_CYCLES,
            MIN_CPU_DELTA_FOR_ALIVE,
            REENABLE_COOLDOWN_S,
        )
        _MIN_CPU_DELTA_FOR_ALIVE = MIN_CPU_DELTA_FOR_ALIVE
        _MAX_STARVED_CYCLES = MAX_STARVED_CYCLES
        _REENABLE_COOLDOWN_S = REENABLE_COOLDOWN_S


class Supervisor:
    """Phase 11 §11.I.1 supervisor — fault detection + restart trigger.

    Co-resident with the Orchestrator in 11E.b.1; standalone process in
    11E.b.2. The split is bus-mediated (`publish_module_restart_request`)
    so 11E.b.2 is a pure process-boundary change with no behavioural delta.

    Public surface:
      * monitor_tick()                           — periodic supervision sweep
      * publish_module_restart_request(name, …)  — D5-routed restart trigger
      * is_running / is_started / get_status     — forwarded to orchestrator
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

    # ── D5 restart-trigger emission ──────────────────────────────────────────

    def publish_module_restart_request(
        self,
        name: str,
        reason: str = "supervisor_request",
        **extra,
    ) -> None:
        """Locked D5 — emit MODULE_RESTART_REQUEST(name, reason) to the bus.

        Destination is the orchestrator's existing subscriber
        (`_handle_module_lifecycle_requests` in `scripts/guardian_hcl.py`).
        The orchestrator executes `restart_module(name, reason, **extra)`
        synchronously on its lifecycle thread.
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

    # ── Supervisory loop (Phase 11 §11.I.1, monitor_tick) ────────────────────

    def monitor_tick(self) -> None:
        """Periodic supervisory sweep — fault detection + restart triggering.

        Per-module checks:
          1. DISABLED + cooldown elapsed       → orchestrator.enable(name)
          2. process.is_alive() == False       → MODULE_CRASHED + restart-trigger
          3. SHM/bus heartbeat timeout         → restart-trigger (CPU-aware grace)
          4. RSS > rss_limit_mb                → restart-trigger
          5. Reset restart_count after sustained uptime

        Bus message ingestion (BUS_PEER_DIED, MODULE_HEARTBEAT, MODULE_READY,
        MODULE_RELOAD_REQUEST, BUS_WORKER_ADOPT_REQUEST) remains on the
        Orchestrator's `_process_guardian_messages` for 11E.b.1 — drained
        once per tick at the start of this method. In 11E.b.2 the BUS_PEER_DIED
        + MODULE_HEARTBEAT subscriptions move here; the adopt/reload paths
        stay on the Orchestrator (they're spawn-side actions).

        Thread-safety: mutations to `orchestrator._modules[name]` are guarded
        by `orchestrator._module_lock` for state transitions to UNHEALTHY /
        CRASHED. Counter updates (cpu samples, starved cycle count) are
        single-writer per module per tick — no lock needed.
        """
        from titan_hcl.bus import MODULE_CRASHED  # lazy, avoid circular
        from titan_hcl.orchestrator.module_registry import ModuleState

        _load_constants()
        orch = self.orchestrator

        if orch._stop_requested:
            return

        # Drain orchestrator's bus message queue (BUS_PEER_DIED + adopts +
        # reloads + bus MODULE_HEARTBEAT/READY updates). 11E.b.1 transitional:
        # this lives on the orchestrator because it mutates orchestrator state.
        # 11E.b.2 splits BUS_PEER_DIED + MODULE_HEARTBEAT to a supervisor-side
        # subscriber + emits state via bus to the orchestrator.
        orch._process_guardian_messages()

        now = time.time()
        for name, info in orch._modules.items():
            # 1. Auto-re-enable disabled modules after cooldown.
            if info.state == ModuleState.DISABLED and info.disabled_at > 0:
                elapsed = now - info.disabled_at
                if elapsed >= _REENABLE_COOLDOWN_S:
                    logger.info(
                        "[Supervisor] Auto-re-enabling module '%s' after "
                        "%.0fs cooldown", name, elapsed)
                    orch.enable(name)
                continue

            if info.state not in (ModuleState.RUNNING, ModuleState.STARTING):
                continue

            # SPEC §11.B.3 (D-SPEC-49) — supervision suppressed during
            # reload-in-flight; orchestrator owns lifecycle of OLD pid.
            if info.reload_in_flight:
                continue

            # 2. Process liveness check.
            if info.process and not info.process.is_alive():
                with orch._module_lock:
                    if info.process and not info.process.is_alive():
                        exitcode = info.process.exitcode
                        logger.warning(
                            "[Supervisor] Module '%s' died (exitcode=%s)",
                            name, exitcode)
                        info.state = ModuleState.CRASHED
                        self.bus.publish(make_msg(
                            MODULE_CRASHED, "supervisor", "core",
                            {"module": name, "exitcode": exitcode}))
                        if info.spec.restart_on_crash:
                            self.publish_module_restart_request(
                                name, reason=f"died_exitcode_{exitcode}")
                continue

            # 3. Heartbeat-timeout check (CPU-aware, 2026-04-21).
            #
            # Phase 11 §11.I.5 / locked D1: read the worker's SHM slot
            # `last_heartbeat` field and take the MAX of (SHM, bus) signals.
            # Workers migrated to the Phase 11 ModuleStateWriter contract
            # may stop sending bus MODULE_HEARTBEAT and only update their
            # SHM slot; without this read they'd be killed mid-cascade.
            # Once W3 cascade lands fleet-wide and bus MODULE_HEARTBEAT is
            # retired in workers, `info.last_heartbeat` becomes dead and
            # this collapses to SHM-only.
            if info.state == ModuleState.RUNNING:
                hb_timeout = info.spec.heartbeat_timeout
                effective_last_heartbeat = info.last_heartbeat
                try:
                    _bank = orch._ensure_module_state_reader_bank()
                    if _bank is not None:
                        _entry = _bank.read(name)
                        if _entry is not None and _entry.last_heartbeat > 0.0:
                            if _entry.last_heartbeat > effective_last_heartbeat:
                                effective_last_heartbeat = _entry.last_heartbeat
                except Exception:  # noqa: BLE001 — SHM read must never crash
                    pass

                if now - effective_last_heartbeat > hb_timeout:
                    cpu_now = (orch._get_cpu_time_seconds(info.pid)
                               if info.pid else 0.0)
                    cpu_grew = (info.last_cpu_time > 0.0
                                and cpu_now - info.last_cpu_time
                                >= _MIN_CPU_DELTA_FOR_ALIVE)
                    if (cpu_grew
                            and info.consecutive_starved_cycles
                            < _MAX_STARVED_CYCLES):
                        info.consecutive_starved_cycles += 1
                        logger.warning(
                            "[Supervisor] Module '%s' heartbeat timeout "
                            "(%.1fs > %.0fs) but CPU grew +%.2fs — "
                            "alive-but-starved cycle %d/%d, deferring restart",
                            name, now - effective_last_heartbeat, hb_timeout,
                            cpu_now - info.last_cpu_time,
                            info.consecutive_starved_cycles,
                            _MAX_STARVED_CYCLES)
                        info.last_cpu_time = cpu_now
                        info.last_cpu_sample_ts = now
                        continue
                    # Stuck or grace exhausted.
                    reason = ("heartbeat_timeout_starved_grace_exhausted"
                              if info.consecutive_starved_cycles
                              >= _MAX_STARVED_CYCLES
                              else "heartbeat_timeout")
                    # L1 crashes are architecturally unexpected — log ERROR.
                    lvl = (logging.ERROR if info.spec.layer == "L1"
                           else logging.WARNING)
                    logger.log(
                        lvl,
                        "[Supervisor] Module '%s' [%s] heartbeat timeout "
                        "(%.1fs > %.0fs limit) — restart reason=%s",
                        name, info.spec.layer,
                        now - effective_last_heartbeat, hb_timeout, reason)
                    with orch._module_lock:
                        info.state = ModuleState.UNHEALTHY
                        info.consecutive_starved_cycles = 0
                    if info.spec.restart_on_crash:
                        self.publish_module_restart_request(
                            name, reason=reason)
                    continue
                # Heartbeat fresh — refresh CPU sample baseline.
                if info.pid:
                    info.last_cpu_time = orch._get_cpu_time_seconds(info.pid)
                    info.last_cpu_sample_ts = now
                if info.consecutive_starved_cycles > 0:
                    info.consecutive_starved_cycles = 0

            # 4. Reset restart count after sustained uptime.
            if info.state == ModuleState.RUNNING and info.ready_time > 0:
                if (now - info.ready_time > orch._sustained_uptime_reset
                        and info.restart_count > 0):
                    logger.info(
                        "[Supervisor] Module '%s' sustained uptime %.0fs — "
                        "resetting restart count",
                        name, now - info.ready_time)
                    info.restart_count = 0
                    info.restart_timestamps.clear()

            # 5. RSS budget check.
            if info.pid:
                rss = orch._get_rss_mb(info.pid)
                info.rss_mb = rss
                if rss > info.spec.rss_limit_mb:
                    logger.warning(
                        "[Supervisor] Module '%s' RSS %.0fMB > limit %dMB",
                        name, rss, info.spec.rss_limit_mb)
                    if info.spec.restart_on_crash:
                        self.publish_module_restart_request(
                            name, reason=f"rss_{rss:.0f}mb")

    # ── Status forwarders ────────────────────────────────────────────────────

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
