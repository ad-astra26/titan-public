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

# Cascade guard (Phase 11 §11.I.2). Min seconds between successive
# MODULE_RESTART_REQUESTs for the SAME module. Must comfortably exceed a
# normal stop(SAVE_NOW ≤30s)+respawn+first-SHM-write so the worker has
# rewritten STARTING into its slot (→ state-machine dedup takes over) before
# this backstop expires. 90s covers memory's ~52s cold boot + 30s stop with
# headroom, and stays under the orchestrator's max_restarts window (600s) so
# a genuine crash-loop still escalates → DISABLED.
_RESTART_REQUEST_COOLDOWN_S: float = 90.0


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
        # Phase 11 §11.I.2 cascade guard — per-module timestamp of the last
        # MODULE_RESTART_REQUEST we published. A restart needs ~stop(30s
        # SAVE_NOW)+respawn+boot before the worker rewrites a fresh STARTING
        # state into its SHM slot. Re-requesting at 1 Hz during that window
        # floods the orchestrator's lifecycle queue → serial 30s restart
        # backlog that never drains (live T1 storm 2026-05-28). We suppress
        # re-requests for RESTART_REQUEST_COOLDOWN_S; the SHM state-machine
        # (STARTING/BOOTED/PROBING suppression) is the primary dedup, this is
        # the backstop for the pid-write race.
        self._restart_requested_at: dict[str, float] = {}

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
        # SPEC §11.I.2 (D-SPEC-141) — the per-module SHM slot
        # (module_<name>_state.bin, ModuleStateWriter) is the AUTHORITATIVE
        # source for state/pid/last_heartbeat. We do NOT consult the legacy
        # in-memory `info.state` (which is fed by the retired MODULE_READY
        # bus broadcast via _process_guardian_messages) for health/restart
        # decisions — D1/D2: "NO bus broadcasts for state transitions". The
        # bus drain above still services adopt/reload (spawn-side) paths.
        bank = None
        try:
            bank = orch._ensure_module_state_reader_bank()
        except Exception:  # noqa: BLE001
            bank = None

        for name, info in orch._modules.items():
            # 1. Auto-re-enable disabled modules after cooldown (orchestrator
            # lifecycle state, not a SHM-slot transition).
            if info.state == ModuleState.DISABLED and info.disabled_at > 0:
                elapsed = now - info.disabled_at
                if elapsed >= _REENABLE_COOLDOWN_S:
                    logger.info(
                        "[Supervisor] Auto-re-enabling module '%s' after "
                        "%.0fs cooldown", name, elapsed)
                    orch.enable(name)
                continue

            # SPEC §11.B.3 (D-SPEC-49) — supervision suppressed during
            # reload-in-flight; orchestrator owns lifecycle of OLD pid.
            if info.reload_in_flight:
                continue

            # Modules the orchestrator never autostarted / deliberately stopped
            # have no live slot to police.
            if not info.spec.autostart and info.state in (
                    ModuleState.STOPPED, ModuleState.DISABLED):
                continue

            # ── Authoritative health read from the SHM slot ──
            entry = None
            if bank is not None:
                try:
                    entry = bank.read(name)
                except Exception:  # noqa: BLE001 — SHM read must never crash tick
                    entry = None
            if entry is None:
                # No slot yet: worker hasn't begun writing (pre-boot) or this
                # module doesn't use ModuleStateWriter. Nothing to assess.
                continue

            sstate = entry.state or ""
            spid = int(entry.pid or 0)
            shb = float(entry.last_heartbeat or 0.0)

            # STOPPED / DISABLED are intentional — never police them.
            if sstate in ("stopped", "disabled"):
                continue

            # ── Fault detection ──
            fault_reason: Optional[str] = None

            # Liveness FIRST, for ALL live-intent states. A dead pid is CRASHED
            # from ANY state per SPEC §11.I.2 ("* → CRASHED"). This is checked
            # even for the transitional STARTING/BOOTED/PROBING states so a
            # module that dies mid-boot (e.g. blocked in a boot task, then
            # killed) is detected and restarted instead of being left stuck in
            # a transitional state forever — the supervisor used to `continue`
            # past these states and never noticed the death (live T1 bug
            # 2026-05-28: backup/meta_teacher stuck 'starting' after dying).
            if spid > 0:
                import os as _os
                import errno as _errno
                try:
                    _os.kill(spid, 0)  # liveness: signal-0
                except OSError as _ke:
                    if _ke.errno == _errno.ESRCH:
                        fault_reason = "shm_pid_dead"
                    # EPERM → exists under another uid → treat as alive.

            # Explicit terminal/degraded states (only if pid wasn't already dead).
            if fault_reason is None and sstate == "crashed":
                fault_reason = "shm_state_crashed"
            elif fault_reason is None and sstate == "unhealthy":
                fault_reason = "shm_state_unhealthy"

            # Transitional states (starting/booted/probing) legitimately have
            # not reached steady-state heartbeat/RSS — they are the
            # orchestrator-owned boot+probe progression + the just-restarted
            # window (primary cascade dedup). They are policed for a DEAD PID
            # only (above); heartbeat-staleness and RSS checks below apply to
            # RUNNING only. So a still-booting live module is never restarted
            # for a not-yet-fresh heartbeat.
            if sstate in ("starting", "booted", "probing") and fault_reason is None:
                continue

            # Heartbeat-staleness (CPU-aware grace) — only for a live RUNNING
            # process that passed the liveness check above.
            if fault_reason is None and sstate == "running" and spid > 0:
                hb_timeout = info.spec.heartbeat_timeout
                if now - shb > hb_timeout:
                    cpu_now = orch._get_cpu_time_seconds(spid)
                    cpu_grew = (info.last_cpu_time > 0.0
                                and cpu_now - info.last_cpu_time
                                >= _MIN_CPU_DELTA_FOR_ALIVE)
                    if (cpu_grew and info.consecutive_starved_cycles
                            < _MAX_STARVED_CYCLES):
                        info.consecutive_starved_cycles += 1
                        info.last_cpu_time = cpu_now
                        info.last_cpu_sample_ts = now
                        logger.warning(
                            "[Supervisor] Module '%s' heartbeat stale "
                            "(%.1fs > %.0fs) but CPU grew +%.2fs — alive-but-"
                            "starved cycle %d/%d, deferring restart",
                            name, now - shb, hb_timeout,
                            cpu_now - info.last_cpu_time,
                            info.consecutive_starved_cycles, _MAX_STARVED_CYCLES)
                    else:
                        fault_reason = (
                            "heartbeat_timeout_starved_grace_exhausted"
                            if info.consecutive_starved_cycles
                            >= _MAX_STARVED_CYCLES else "heartbeat_timeout")
                else:
                    # Fresh heartbeat — refresh CPU baseline + reset starve +
                    # clear any restart-request marker (module recovered).
                    info.last_cpu_time = orch._get_cpu_time_seconds(spid)
                    info.last_cpu_sample_ts = now
                    info.consecutive_starved_cycles = 0
                    self._restart_requested_at.pop(name, None)
                    # Reset restart count after sustained uptime.
                    if (info.ready_time > 0
                            and now - info.ready_time > orch._sustained_uptime_reset
                            and info.restart_count > 0):
                        info.restart_count = 0
                        info.restart_timestamps.clear()

            # ── RSS budget (live RUNNING only) ──
            if fault_reason is None and sstate == "running" and spid > 0:
                rss = orch._get_rss_mb(spid)
                info.rss_mb = rss
                if rss > info.spec.rss_limit_mb:
                    fault_reason = f"rss_{rss:.0f}mb"

            if fault_reason is None:
                continue

            # ── Restart-request emission with cooldown dedup ──
            # The SHM state-machine suppression above is the primary dedup;
            # this cooldown is the backstop for the window between publishing
            # the request and the worker rewriting STARTING into its slot
            # (stop+respawn+first-write ≈ tens of seconds). Without it the
            # 1 Hz tick re-fires every second → serial 30s restart backlog
            # (live T1 storm 2026-05-28).
            if not info.spec.restart_on_crash:
                continue
            last_req = self._restart_requested_at.get(name, 0.0)
            if now - last_req < _RESTART_REQUEST_COOLDOWN_S:
                continue
            self._restart_requested_at[name] = now
            if fault_reason in ("shm_pid_dead", "shm_state_crashed"):
                self.bus.publish(make_msg(
                    MODULE_CRASHED, "supervisor", "core",
                    {"module": name, "exitcode": None, "source": fault_reason}))
            lvl = (logging.ERROR if info.spec.layer == "L1"
                   else logging.WARNING)
            logger.log(
                lvl,
                "[Supervisor] Module '%s' [%s] fault=%s (shm_state=%s pid=%d "
                "hb_age=%.1fs) — requesting restart",
                name, info.spec.layer, fault_reason, sstate, spid,
                (now - shb) if shb > 0 else -1.0)
            self.publish_module_restart_request(name, reason=fault_reason)

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
