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
import queue as _queue
import time
from collections import deque
from typing import TYPE_CHECKING, Optional

from titan_hcl.bus import (
    DivineBus,
    MODULE_ERROR,
    MODULE_RESTART_REQUEST,
    make_msg,
)
from titan_hcl.errors import Severity, is_unrecoverable

if TYPE_CHECKING:
    from titan_hcl.orchestrator import Orchestrator

logger = logging.getLogger(__name__)


# Constants imported lazily from orchestrator/core to avoid circular import
# at module load time. Phase 11 §11.I.5 lazy-binding keeps the supervisor
# module standalone-loadable for the future 11E.b.2 split.
_MIN_CPU_DELTA_FOR_ALIVE: Optional[float] = None
_MAX_STARVED_CYCLES: Optional[int] = None
_REENABLE_COOLDOWN_S: Optional[float] = None
# RFP_supervision_lifecycle §7.B/§7.F — after this many consecutive cycles (≈1 Hz)
# of RssAnon over rss_limit, a THROTTLE warning escalates to a SUSTAINED line
# (genuine memory pressure, surfaced for root-cause). It still does NOT respawn —
# throttle beats respawn (INV-SUP-2); a real leak is a root-cause issue to surface,
# not flap.
_RSS_SUSTAINED_OVER_CYCLES: int = 30

# Cascade guard (Phase 11 §11.I.2). Min seconds between successive
# MODULE_RESTART_REQUESTs for the SAME module. Must comfortably exceed a
# normal stop(SAVE_NOW ≤30s)+respawn+first-SHM-write so the worker has
# rewritten STARTING into its slot (→ state-machine dedup takes over) before
# this backstop expires. 90s covers memory's ~52s cold boot + 30s stop with
# headroom, and stays under the orchestrator's max_restarts window (600s) so
# a genuine crash-loop still escalates → DISABLED.
_RESTART_REQUEST_COOLDOWN_S: float = 90.0

# RFP_supervision_lifecycle §7.F — taxonomy-driven DISABLE gate.
# A module that emits this many FATAL ModuleErrors (recoverable error_code)
# within the restart window is a genuine crash-loop the §7.A path would also
# catch via repeated restarts — but the taxonomy gate disables it FAST, with a
# typed reason, and a single greppable MODULE_CRITICAL_DOWN line. A FATAL with
# an UNRECOVERABLE error_code (errors.is_unrecoverable) disables on the FIRST
# occurrence — restart is futile. Resource conditions are NOT ModuleErrors
# (§7.B THROTTLE) and never reach this gate. The window mirrors the orchestrator
# restart window so the two gates agree on "what's a crash-loop".
_FATAL_MODULE_ERROR_DISABLE_THRESHOLD: int = 3
# Max ModuleError messages drained per monitor_tick — backstop against a flood
# (the bus already rate-limits to 100/s per (module,code); this bounds journal
# render + processing cost on a 1 Hz tick to a known ceiling).
_MODULE_ERROR_DRAIN_CAP_PER_TICK: int = 200


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

        # ── RFP §7.F — structured-error taxonomy consumer ────────────────────
        # Flags ship DEFAULT-ON (feedback_all_flags_default_on); the `=false`
        # form is a kill-switch only.
        self._module_error_journal_render = bool(
            self._config.get("module_error_journal_render", True))
        self._taxonomy_disable_gate = bool(
            self._config.get("taxonomy_fatal_disable_gate", True))
        self._fatal_disable_threshold = int(
            self._config.get("fatal_module_error_disable_threshold",
                             _FATAL_MODULE_ERROR_DISABLE_THRESHOLD))
        # Per-module rolling timestamps of FATAL ModuleErrors (window = the
        # orchestrator restart_window). deque so old entries age out cheaply.
        self._fatal_error_ts: dict[str, deque] = {}
        # Dedicated MODULE_ERROR queue — SEPARATE from the orchestrator's
        # "guardian" liveness queue (D-SPEC-151: keeping flooding broadcasts off
        # that queue protects MODULE_HEARTBEAT). The Rust broker forwards
        # MODULE_ERROR to this process because guardian_hcl.py declares it in
        # build_bus_and_client(broadcast_topics=[...]); the in-process dispatcher
        # routes it here by the types= filter.
        self._module_error_queue = None
        if self._module_error_journal_render or self._taxonomy_disable_gate:
            try:
                self._module_error_queue = bus.subscribe(
                    "guardian_module_errors", types=[MODULE_ERROR])
            except Exception as e:  # noqa: BLE001 — never crash supervision boot
                logger.warning(
                    "[Supervisor] MODULE_ERROR subscription failed: %s — "
                    "taxonomy journal/disable gate inactive this boot", e)

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

    # ── RFP §7.F — structured-error taxonomy consumer ────────────────────────

    def _process_module_errors(self) -> None:
        """Drain the dedicated MODULE_ERROR queue (RFP §7.F).

          (b) JOURNAL CASCADE — render EVERY ModuleError to the kernel journal as
              a STABLE greppable tag `[ERR][<module>][<code>][<severity>] <msg>`
              so an operator greps ONE tag instead of scanning 1000 lines.
          (c) DISABLE GATE — a FATAL ModuleError with an UNRECOVERABLE error_code
              disables the module on the FIRST occurrence (restart is futile); a
              FATAL with a recoverable code disables once it crosses
              `_fatal_disable_threshold` within the orchestrator restart window
              (the crash-loop case). A resource condition is NOT a ModuleError
              (it is a §7.B THROTTLE) so it can never reach this gate — DISABLE is
              taxonomy-driven, never rss-fault-driven (INV-SUP-1/8).

        This is ADDITIVE to the §7.A restart-window escalation (the infra-fault
        backstop for crashes that emit no ModuleError, e.g. an external SIGKILL or
        a native SEGV) — both gates coexist.
        """
        q = self._module_error_queue
        if q is None:
            return
        window = float(getattr(self.orchestrator, "_restart_window_seconds", 60.0))
        drained = 0
        while drained < _MODULE_ERROR_DRAIN_CAP_PER_TICK:
            try:
                msg = q.get_nowait()
            except _queue.Empty:
                break
            except Exception:  # noqa: BLE001 — a bad msg must never break the tick
                break
            drained += 1
            payload = msg.get("payload") if isinstance(msg, dict) else None
            if not isinstance(payload, dict):
                continue
            module = str(payload.get("module_name") or payload.get("module") or "unknown")
            code = str(payload.get("error_code") or "UNKNOWN")
            severity = str(payload.get("severity") or "ERROR")
            summary = str(payload.get("message") or "")

            # (b) greppable journal tag — one stable shape per ModuleError.
            if self._module_error_journal_render:
                logger.error("[ERR][%s][%s][%s] %s", module, code, severity, summary)

            # (c) DISABLE gate — FATAL only.
            if not self._taxonomy_disable_gate or severity != Severity.FATAL.value:
                continue
            now = time.time()
            buf = self._fatal_error_ts.get(module)
            if buf is None:
                buf = deque()
                self._fatal_error_ts[module] = buf
            buf.append(now)
            while buf and (now - buf[0]) > window:
                buf.popleft()

            unrecoverable = is_unrecoverable(code)
            if unrecoverable or len(buf) >= self._fatal_disable_threshold:
                reason = ("unrecoverable_fatal_module_error" if unrecoverable
                          else "repeated_fatal_module_error")
                logger.critical(
                    "[Supervisor] DISABLE gate fired for '%s' — %s "
                    "(error_code=%s, fatal_count=%d/%d in %.0fs window)",
                    module, reason, code, len(buf),
                    self._fatal_disable_threshold, window)
                try:
                    self.orchestrator.disable(
                        module, reason, error_code=code,
                        severity=severity, count=len(buf))
                except Exception as e:  # noqa: BLE001 — disable failure must not break the tick
                    logger.error(
                        "[Supervisor] disable('%s') failed: %s", module, e,
                        exc_info=True)
                buf.clear()  # reset so auto-re-enable (§7.C) gets a clean window

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

        # RFP §7.F — drain the dedicated MODULE_ERROR queue: render greppable
        # journal tags + run the taxonomy DISABLE gate. Separate queue from the
        # liveness drain above, so a ModuleError flood never crowds heartbeats.
        self._process_module_errors()

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

            # SPEC §11.B.4 INV-PROC-5 / §11.B.5 (2026-06-02) — kernel-supervised
            # peer (the L3 api). Its liveness + respawn are owned SOLELY by
            # titan-kernel-rs (kernel_supervisor.rs). The L1 Supervisor must NOT
            # police it: the kernel's zero-downtime swap transiently leaves the
            # canonical module_api_state.bin holding OLD's (dead) pid until NEW
            # self-promotes, and a shm_pid_dead check here would race the kernel
            # → a spurious MODULE_RESTART_REQUEST → a doomed orchestrator spawn
            # that loses the port and zombies. The slot stays enumerable for
            # /v6/* readouts; only the restart-policing is the kernel's job.
            if getattr(info.spec, "kernel_supervised", False):
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

            # SPEC §11.B.3 — only police the pid the orchestrator currently expects
            # (info.pid). During a kill→respawn (restart-module / restart()) the SHM
            # slot transiently holds the OLD (now-dead) pid for a few seconds until
            # the NEW process boots and overwrites the slot. Reading that stale dead
            # pid as a fault issues a SPURIOUS restart that races the in-flight
            # respawn → a flap cascade (live: agency_worker restart-module flap,
            # 2026-06-24; agency is reply_only so its 30s SAVE_NOW-timeout widens the
            # window). A slot pid that doesn't match the expected pid is a stale
            # pre-respawn slot, NOT a fault — skip until the new process writes it.
            # (Only when an expected pid is known; info.pid==0 ⇒ best-effort police.)
            _expected_pid = int(info.pid or 0)
            if spid > 0 and _expected_pid > 0 and spid != _expected_pid:
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
            # process that passed the liveness check above. shb > 0 guard: a
            # last_heartbeat of 0 means "not yet heartbeated" (just transitioned
            # to running), NOT stale — without it, now-0 ≫ timeout falsely fires
            # an instant heartbeat_timeout on every freshly-probed module. Real
            # death is caught by the pid-liveness check above, not here.
            if fault_reason is None and sstate == "running" and spid > 0 and shb > 0:
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
            # INV-SUP-1/2/8 (RFP §7.B/§7.F): a RESOURCE condition THROTTLES and
            # is surfaced — it NEVER drives a kill→respawn. `rss` is RssAnon (real
            # private memory) post the #1 enforcement fix, so an over-limit here
            # is genuine pressure, not VmRSS mmap inflation. Respawning a worker
            # over its real limit just loses its state and reboots into the same
            # pressure (and feeds the load→restart→load cascade) — throttle +
            # surface for root-cause instead. Only genuine critical faults (dead
            # pid / hung heartbeat above; FATAL ModuleError) restart. The legacy
            # rss→restart is behind a kill-switch (`rss_over_limit_throttles_
            # not_restarts=false`) for emergency revert only.
            if fault_reason is None and sstate == "running" and spid > 0:
                rss = orch._get_rss_mb(spid)   # RssAnon (real memory)
                info.rss_mb = rss
                if rss > info.spec.rss_limit_mb:
                    if getattr(orch, "_rss_over_throttles", True):
                        info.consecutive_rss_over_cycles += 1
                        _n = info.consecutive_rss_over_cycles
                        _sustained = _n >= _RSS_SUSTAINED_OVER_CYCLES
                        if _n == 1 or _n % _RSS_SUSTAINED_OVER_CYCLES == 0:
                            logger.warning(
                                "[Supervisor] Module '%s' RssAnon %.0fMB over "
                                "rss_limit %dMB for %d cycle(s) — THROTTLE, not "
                                "restart (resource conditions never drive "
                                "lifecycle; INV-SUP-1/2). %s",
                                name, rss, info.spec.rss_limit_mb, _n,
                                "SUSTAINED — genuine memory pressure, surfacing "
                                "for root-cause (still no respawn)" if _sustained
                                else "rechecking — letting it settle")
                        # (§7.B future: apply a cgroup memory.high soft-limit here
                        #  once Delegate=yes / leaf-cgroup layout is in place.)
                    else:
                        # kill-switch: legacy resource-fault→restart behavior.
                        fault_reason = f"rss_{rss:.0f}mb"
                else:
                    info.consecutive_rss_over_cycles = 0   # under-limit / recovered

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
