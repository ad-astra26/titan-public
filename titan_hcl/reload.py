"""
titan_hcl.reload — D-SPEC-50 §8.3 per-module hot-reload (7-step sequence + pid-targeting + rollback).

See SPEC §8.3 + §11.B.3 + §11.B.3.1 + D-SPEC-50 / D-SPEC-93.

Mixed into class Orchestrator(OrchestratorReloadMixin, OrchestratorDepActivationMixin)
in `titan_hcl/orchestrator/core.py` — `self` attributes (.bus, ._modules,
._reload_lock, etc.) come from Orchestrator.__init__.

Phase 11 §11.I.6 relocation (D-SPEC-141 / v1.65.0): this file moved from
titan_hcl/guardian_hcl/reload.py → titan_hcl/reload.py per locked D6 so
hot-reload spawn becomes owned by titan_hcl orchestrator rather than the
guardian_hcl supervisor. Class renamed GuardianReloadMixin →
OrchestratorReloadMixin same commit (11E.a).

Phase 11 §11.I.1 follow-up (11E.b.1): the host class renamed from
`Guardian` to `Orchestrator` and the dep-activation mixin renamed from
`GuardianDepActivationMixin` to `OrchestratorDepActivationMixin`. The
back-compat alias `Guardian = Orchestrator` keeps every legacy callsite
working unchanged.
"""
from __future__ import annotations  # PEP-563 lazy annotations (Phase 11 §11.I.6 circular-import break)
import asyncio
import logging
import os
import queue as _queue_mod
import signal
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from multiprocessing import Process
from queue import Empty
from typing import Callable, Optional

from titan_hcl.bus import (
    AnyQueue,
    BUS_PEER_DIED,
    BUS_WORKER_ADOPT_ACK,
    BUS_WORKER_ADOPT_REQUEST,
    DivineBus,
    MODULE_CRASHED,
    MODULE_HEARTBEAT,
    MODULE_READY,
    MODULE_RELOAD_ACK,
    MODULE_RELOAD_REQUEST,
    MODULE_SHUTDOWN,
    SUPERVISION_CHILD_DOWN,
    SUPERVISION_CHILD_RESTARTED,
    SUPERVISION_DEPENDENCY_ACTIVATING,
    SUPERVISION_DEPENDENCY_BLOCKED,
    SUPERVISION_DEPENDENCY_DEGRADED,
    SUPERVISION_DEPENDENCY_RECOVERED,
    SUPERVISION_ESCALATION,
    make_msg,
)
from titan_hcl import bus
from titan_hcl._phase_c_constants import (
    ADOPTION_TIMEOUT_S,
    MODULE_RELOAD_DEFAULT_TIMEOUT_S,
    MODULE_RELOAD_HAPPY_PATH_S,
    SUPERVISION_DEPENDENCY_ACTIVATION_TIMEOUT_S,
)
from titan_hcl.supervision import (
    Dependency,
    DependencyAction,
    DependencyKind,
    DependencySeverity,
    EscalationDecision,
    ReasonRecord,
    SupervisionReason,
    classify_exit_code,
    kernel_default_decision,
    most_common_reason,
)

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────

HEARTBEAT_INTERVAL = 10.0       # seconds between expected heartbeats
HEARTBEAT_TIMEOUT = 90.0        # seconds before declaring a module dead (mainnet-safe: ~Schumann×27)
DEFAULT_RSS_LIMIT_MB = 1500     # per-module RSS limit (MB)
RESTART_BACKOFF_BASE = 2.0      # exponential backoff base (seconds)
MAX_RESTARTS_IN_WINDOW = 5      # max restarts allowed in the sliding window
RESTART_WINDOW_SECONDS = 600.0  # 10-minute sliding window for restart tracking
SUSTAINED_UPTIME_RESET = 300.0  # 5 minutes of uptime before restart count resets
REENABLE_COOLDOWN_S = 600.0    # 10 minutes before auto-re-enabling a disabled module
# CPU-aware heartbeat (added 2026-04-21) — when heartbeat times out, sample
# /proc/<pid>/stat CPU time. If CPU grew ≥ MIN_CPU_DELTA_FOR_ALIVE since last
# sample, the module is alive-but-CPU-starved (not deadlocked). Defer restart
# for up to MAX_STARVED_CYCLES wallclock heartbeat windows; then force-restart
# (bounded grace prevents runaway hang on a truly stuck module).
MIN_CPU_DELTA_FOR_ALIVE = 1.0   # seconds of CPU time per heartbeat window proves liveness
MAX_STARVED_CYCLES = 5          # how many consecutive starved-but-alive cycles to tolerate
# Bumped 3 → 5 on 2026-04-21 after observing both T2+T3 media modules hit
# grace-exhausted-restart once each during the same 75-min ARC iter-3 slot.
# 5 cycles ≈ 5 minutes wallclock grace under monitor_tick=5s — should bridge
# typical ARC tail without leaving truly-stuck modules hanging too long.



# Phase 11 §11.I.6 / D-SPEC-141: this file was relocated from
# titan_hcl/guardian_hcl/reload.py per locked D6. To break the import cycle
# (titan_hcl.guardian_hcl/__init__.py imports core.py → core.py imports
# OrchestratorReloadMixin from here → here imports module_registry from
# the still-initializing guardian_hcl package), the module_registry symbols
# (ModuleInfo, ModuleSpec, ModuleState, ReloadState) are imported via
# TYPE_CHECKING for type hints + lazy local imports inside the methods that
# construct or compare against them.
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from titan_hcl.orchestrator.module_registry import (
        ModuleInfo,
        ModuleSpec,
        ModuleState,
        ReloadState,
    )


def _mr_symbols():
    """Return (ModuleInfo, ModuleSpec, ModuleState, ReloadState) lazily."""
    from titan_hcl.orchestrator.module_registry import (
        ModuleInfo, ModuleSpec, ModuleState, ReloadState,
    )
    return ModuleInfo, ModuleSpec, ModuleState, ReloadState


class OrchestratorReloadMixin:
    """Mixin providing D-SPEC-50 §8.3 per-module hot-reload (7-step sequence + pid-targeting + rollback) — see See SPEC §8.3 + §11.B.3 + §11.B.3.1 + D-SPEC-50 / D-SPEC-93.."""

    async def reload_module(self, module_name: str,
                            new_module_path: Optional[str] = None,
                            timeout_s: float = MODULE_RELOAD_DEFAULT_TIMEOUT_S
                            ) -> dict:
        """SPEC §8.3 + §11.B.3 — initiate per-module hot-reload.

        Per `rFP_phase_c_bus_delivery_continuity_and_hot_reload.md` §4.4.
        Orchestrates spawn-NEW → adopt → kill-OLD reusing §8.4 ADOPTION
        protocol + §8.0.bis boot-buffer for delivery continuity across
        the transfer window.

        Args:
            module_name: registered ModuleSpec.name
            new_module_path: path to new module file (None = same-source
                in-place reload of stuck module)
            timeout_s: max wait for terminal status (default
                MODULE_RELOAD_DEFAULT_TIMEOUT_S=30s)

        Returns:
            {swap_id, module_name, status, reason, total_elapsed_ms, ts}
            where status ∈ {ready, failed, rolled_back}.

        Idempotent — re-issuing during in-flight returns status="failed"
        with reason="reload_in_flight" per §4.4.

        Async coroutine wrapping a synchronous orchestrator delegated to
        `_restart_executor` so FastAPI/dashboard callers can `await`
        without blocking their event loop. The orchestrator itself is
        synchronous because mp.Process spawn + queue.get + os.kill are
        all sync I/O.
        """
        swap_id = str(uuid.uuid4())
        return await asyncio.to_thread(
            self._reload_module_sync,
            module_name, new_module_path, timeout_s, swap_id,
        )

    def _reload_module_sync(self, module_name: str,
                            new_module_path: Optional[str],
                            timeout_s: float, swap_id: str) -> dict:
        """Synchronous orchestrator for `reload_module()` + bus
        MODULE_RELOAD_REQUEST entry point. Runs on `_restart_executor`.

        Implements the §4.3 8-step sequence with rollback on adoption
        timeout. Always clears `info.reload_in_flight` and pops the
        `_reloads_in_flight` entry in `finally` so supervision authority
        is always recoverable.
        """
        # Phase 11 §11.I.6: lazy import to break the
        # titan_hcl.reload ↔ titan_hcl.guardian_hcl circular import.
        from titan_hcl.orchestrator.module_registry import (
            ModuleState, ReloadState,
        )
        started_ts = time.time()

        # ── Step 1: validate + register reload state ──────────────────
        with self._reload_lock:
            info = self._modules.get(module_name)
            if info is None:
                self._emit_reload_ack(
                    swap_id, module_name, "failed",
                    "unknown_module", started_ts)
                return self._reload_result(
                    swap_id, module_name, "failed",
                    "unknown_module", started_ts)
            if module_name in self._reloads_in_flight:
                self._emit_reload_ack(
                    swap_id, module_name, "failed",
                    "reload_in_flight", started_ts)
                return self._reload_result(
                    swap_id, module_name, "failed",
                    "reload_in_flight", started_ts)
            if info.state != ModuleState.RUNNING:
                self._emit_reload_ack(
                    swap_id, module_name, "failed",
                    f"not_running:state={info.state.value}", started_ts)
                return self._reload_result(
                    swap_id, module_name, "failed",
                    f"not_running:state={info.state.value}", started_ts)
            if info.process is None or info.pid is None:
                self._emit_reload_ack(
                    swap_id, module_name, "failed",
                    "no_process", started_ts)
                return self._reload_result(
                    swap_id, module_name, "failed",
                    "no_process", started_ts)
            old_process = info.process
            old_pid = info.pid
            rs = ReloadState(
                swap_id=swap_id,
                module_name=module_name,
                old_pid=old_pid,
                new_module_path=new_module_path,
                started_ts=started_ts,
            )
            self._reloads_in_flight[module_name] = rs
            info.reload_in_flight = True

        deadline = started_ts + timeout_s
        try:
            # ── Step 2: ACK status="spawning" ──────────────────────────
            self._emit_reload_ack(
                swap_id, module_name, "spawning", None, started_ts)

            # ── Step 3: spawn NEW alongside OLD ────────────────────────
            spawn_err = self._spawn_for_reload(rs, info)
            if spawn_err is not None:
                return self._rollback_reload(
                    rs, info, old_process, "spawn", spawn_err, started_ts)

            # ── Step 4: wait ADOPTION_REQUEST from NEW pid ────────────
            # Drains adoption_q until we get a frame whose payload.pid matches
            # rs.new_pid (defensive — race-free routing in
            # _process_guardian_messages places frames by name only; here we
            # validate the actual sender). Defensive pid-validation moved
            # here per BUG-PHASE-B-FIRST-RELOAD-ADOPTION-ROUTING-MISS-20260514
            # closure (race-free Guardian-side routing relies on this
            # validation site, not the broker-routing site).
            adoption_msg: dict | None = None
            while True:
                timeout_left = max(
                    0.5, min(ADOPTION_TIMEOUT_S, deadline - time.time()))
                if timeout_left <= 0.5 and deadline - time.time() <= 0:
                    break
                try:
                    candidate = rs.adoption_q.get(timeout=timeout_left)
                except _queue_mod.Empty:
                    break
                cand_pid = (candidate.get("payload") or {}).get("pid")
                if cand_pid == rs.new_pid:
                    adoption_msg = candidate
                    break
                logger.warning(
                    "[Guardian] reload '%s' (swap_id=%s) ignoring stale "
                    "ADOPTION_REQUEST from pid=%s (expected rs.new_pid=%s) — "
                    "likely fanout from a sibling/prior reload",
                    module_name, rs.swap_id, cand_pid, rs.new_pid)
            if adoption_msg is None:
                return self._rollback_reload(
                    rs, info, old_process,
                    "adoption", "adoption_timeout", started_ts)

            # ── Step 5: emit ADOPTION_ACK + ACK status="adopted" ──────
            rid = adoption_msg.get("rid")
            self.bus.publish(make_msg(
                BUS_WORKER_ADOPT_ACK, "guardian", module_name, {
                    "name": module_name,
                    "pid": rs.new_pid,
                    "shadow_pid": os.getpid(),
                    "status": "adopted",
                    "reason": None,
                }, rid=rid))
            rs.status = "adopted"
            self._emit_reload_ack(
                swap_id, module_name, "adopted", None, started_ts)

            # ── Step 6: send MODULE_SHUTDOWN to OLD + grace + SIGKILL ─
            self.bus.publish(make_msg(
                MODULE_SHUTDOWN, "guardian", module_name, {
                    "reason": "reload",
                    "swap_id": swap_id,
                    "target_pid": old_pid,
                }))
            # SUPERVISION_SHUTDOWN_GRACE_S=10s implicit per SPEC §11.A —
            # match Guardian.stop() semantics (gentle SIGTERM → 2s grace
            # for adopted workers; we have explicit grace here).
            shutdown_grace_deadline = time.time() + 10.0
            while time.time() < shutdown_grace_deadline:
                if not self._pid_alive(old_pid):
                    break
                time.sleep(0.1)
            if self._pid_alive(old_pid):
                logger.warning(
                    "[Guardian] Reload OLD pid=%s for '%s' did not exit "
                    "gracefully within 10s — SIGKILL", old_pid, module_name)
                try:
                    os.kill(old_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "[Guardian] SIGKILL of OLD pid=%s failed: %s",
                        old_pid, e)
                # Brief wait for SIGKILL to take effect before swap
                t_end = time.time() + 1.0
                while time.time() < t_end:
                    if not self._pid_alive(old_pid):
                        break
                    time.sleep(0.05)

            # ── Step 7: atomic swap of info.process / info.pid / queues ─
            with self._module_lock:
                if old_process is not None:
                    try:
                        old_process.join(timeout=2.0)
                    except Exception:  # noqa: BLE001
                        pass
                    try:
                        old_process.close()
                    except (ValueError, OSError):
                        # Process already closed or never started cleanly —
                        # benign in this code path.
                        pass
                info.process = rs.new_process
                info.pid = rs.new_pid
                # Queues: in socket-broker mode, both are None (worker
                # rebinds via setup_worker_bus). In legacy mp.Queue mode,
                # _spawn_for_reload pre-allocated rs.new_queue/send_queue
                # and registered with the bus.
                info.queue = rs.new_queue
                info.send_queue = rs.new_send_queue
                info.start_time = time.time()
                info.last_heartbeat = time.time()
                info.state = ModuleState.STARTING
                # Keep reload_in_flight=True until MODULE_READY arrives so
                # NEW gets boot grace without monitor_tick heartbeat-timeout
                # restart-cycling it. Cleared in `finally`.

            # ── Step 8: wait MODULE_READY from NEW ────────────────────
            timeout_left = max(
                1.0, min(MODULE_RELOAD_HAPPY_PATH_S, deadline - time.time()))
            try:
                rs.ready_q.get(timeout=timeout_left)
            except _queue_mod.Empty:
                # NEW didn't emit MODULE_READY within budget. OLD is dead,
                # NEW is alive — this is NOT a rollback (no recovery path).
                # Leave NEW running as the slot's new owner; supervision
                # will handle it normally via heartbeat-timeout if it never
                # boots. Emit failed status so initiator knows.
                self._emit_reload_ack(
                    swap_id, module_name, "failed",
                    "ready_timeout", started_ts)
                return self._reload_result(
                    swap_id, module_name, "failed",
                    "ready_timeout", started_ts)

            # NEW emitted MODULE_READY — finalize state=RUNNING explicitly
            # here regardless of whether _process_guardian_messages
            # already transitioned it. Race window: MODULE_READY may
            # arrive BEFORE Step 7's atomic swap (NEW boots faster than
            # Step 6's 10s SIGKILL grace). In that case
            # _process_guardian_messages set state=RUNNING first, then
            # Step 7 overwrote it back to STARTING. Without this final
            # transition the module stays stuck at state=starting forever
            # despite being fully alive (heartbeats arriving, requests
            # processed). Live-discovered 2026-05-19 during T3 cascade
            # of D-SPEC-93 (knowledge_worker pid=1090355 stuck STARTING
            # after pid=1090080 SIGKILL). Part of D-SPEC-93 closure.
            with self._module_lock:
                if info.state != ModuleState.RUNNING:
                    info.state = ModuleState.RUNNING
                    info.ready_time = time.time()
                    info.last_heartbeat = time.time()
                    logger.info(
                        "[Guardian] Module '%s' state finalized RUNNING "
                        "post-reload (pid=%s, swap_id=%s)",
                        module_name, info.pid, swap_id)
            self._emit_reload_ack(
                swap_id, module_name, "ready", None, started_ts)
            return self._reload_result(
                swap_id, module_name, "ready", None, started_ts)
        finally:
            # Always release supervision authority on this module so
            # monitor_tick can resume per §11.B.3 contract.
            with self._reload_lock:
                info.reload_in_flight = False
                self._reloads_in_flight.pop(module_name, None)

    def _spawn_for_reload(self, rs: "ReloadState",
                          info: ModuleInfo) -> Optional[str]:
        """Spawn NEW subprocess alongside OLD, populating rs.new_process /
        rs.new_pid / rs.new_queue / rs.new_send_queue. Returns None on
        success, an error string on failure.

        Mirrors `start()`'s spawn block but does NOT touch `info.process`
        — OLD stays the registered owner until the atomic swap in step 7.
        """
        import copy as _copy
        import multiprocessing

        method = info.spec.start_method
        if method not in ("fork", "spawn"):
            method = "fork"
        ctx = multiprocessing.get_context(method)

        # Mirror start() queue-allocation logic. In socket-broker mode
        # the worker rebinds via setup_worker_bus() so both queues are
        # None. In legacy mode we allocate ctx.Queue() for bidirectional
        # bus routing.
        if self.bus.has_socket_broker:
            rs.new_queue = None
            rs.new_send_queue = None
        else:
            rs.new_queue = ctx.Queue(maxsize=10000)
            rs.new_send_queue = ctx.Queue(maxsize=10000)

        # SPEC §11.B.3 Phase B — deep-copy config + inject swap_id so the
        # NEW worker's setup_worker_bus emits ADOPTION_REQUEST on initial
        # subscribe-ack. deep-copy prevents OLD's spec.config from being
        # mutated (which would race with concurrent reloads on other
        # modules). _module_wrapper pops the key before passing config
        # down to entry_fn so worker code sees its normal config dict.
        reload_config = _copy.deepcopy(info.spec.config) \
            if isinstance(info.spec.config, dict) else {}
        reload_config["_phase_b_reload_swap_id"] = rs.swap_id

        try:
            proc = ctx.Process(
                target=_module_wrapper,
                args=(
                    info.spec.entry_fn,
                    info.spec.name,
                    rs.new_queue,
                    rs.new_send_queue,
                    reload_config,
                    info.spec.start_method,
                    info.spec.broadcast_topics,
                    info.spec.reply_only,
                ),
                name=f"titan-{info.spec.name}-reload",
                daemon=True,
            )
            proc.start()
        except Exception as e:  # noqa: BLE001
            logger.error(
                "[Guardian] reload spawn failed for '%s': %s",
                info.spec.name, e)
            return f"spawn_exception:{e!r}"

        rs.new_process = proc
        rs.new_pid = proc.pid
        logger.info(
            "[Guardian] Reload: spawned NEW '%s' pid=%d alongside OLD pid=%d "
            "(swap_id=%s)",
            info.spec.name, proc.pid, rs.old_pid, rs.swap_id)
        return None

    def _rollback_reload(self, rs: "ReloadState", info: ModuleInfo,
                         old_process: Optional[Process],
                         failed_step: str, reason: str,
                         started_ts: float) -> dict:
        """Rollback path — kill NEW (still alive at this point), leave OLD
        untouched as sole owner. Caller is responsible for clearing
        `info.reload_in_flight` (done in `_reload_module_sync` `finally`).

        Only valid for failures BEFORE step 6 (MODULE_SHUTDOWN to OLD) —
        after that, OLD is dead and there's no recovery path. Step 8
        ready_timeout is handled separately as a SOFT failure.
        """
        logger.warning(
            "[Guardian] Reload rollback for '%s' (swap_id=%s): step=%s "
            "reason=%s; killing NEW pid=%s, OLD pid=%s resumes as owner",
            rs.module_name, rs.swap_id, failed_step, reason,
            rs.new_pid, rs.old_pid)
        # Kill NEW if it spawned
        if rs.new_process is not None:
            try:
                rs.new_process.kill()
                rs.new_process.join(timeout=2.0)
                rs.new_process.close()
            except (ValueError, OSError):
                pass
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "[Guardian] Reload rollback NEW kill failed for '%s': "
                    "%s", rs.module_name, e)
        # Also belt-and-suspenders: SIGKILL by pid if we have one
        if rs.new_pid is not None and self._pid_alive(rs.new_pid):
            try:
                os.kill(rs.new_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except Exception:  # noqa: BLE001
                pass
        self._emit_reload_ack(
            rs.swap_id, rs.module_name, "rolled_back",
            f"{failed_step}:{reason}", started_ts)
        return self._reload_result(
            rs.swap_id, rs.module_name, "rolled_back",
            f"{failed_step}:{reason}", started_ts)

    def _emit_reload_ack(self, swap_id: str, module_name: str, status: str,
                         reason: Optional[str], started_ts: float) -> None:
        """Publish a MODULE_RELOAD_ACK frame on the bus. dst="all" is the
        practical broadcast because the initiator subscription is not
        named (Maker CLI / future D9 Guardian). MODULE_RELOAD_ACK is
        pre-listed in BOOT_BUFFERED_TYPES so transient subscription gaps
        on the initiator side don't lose the terminal status (§8.0.bis).

        `total_elapsed_ms` is recorded on every ACK so observers can
        track end-to-end timing per acceptance gate §4.6 #1.
        """
        elapsed_ms = int((time.time() - started_ts) * 1000)
        try:
            self.bus.publish(make_msg(
                MODULE_RELOAD_ACK, "guardian", "all", {
                    "swap_id": swap_id,
                    "module_name": module_name,
                    "status": status,
                    "reason": reason,
                    "total_elapsed_ms": elapsed_ms,
                    "ts": time.time(),
                }))
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "[Guardian] MODULE_RELOAD_ACK publish failed for '%s' "
                "(swap_id=%s, status=%s): %s",
                module_name, swap_id, status, e)

    @staticmethod
    def _reload_result(swap_id: str, module_name: str, status: str,
                       reason: Optional[str], started_ts: float) -> dict:
        """Build the dict returned by `reload_module()` / `_reload_module_sync`.
        Mirrors MODULE_RELOAD_ACK payload shape per SPEC §8.3."""
        return {
            "swap_id": swap_id,
            "module_name": module_name,
            "status": status,
            "reason": reason,
            "total_elapsed_ms": int((time.time() - started_ts) * 1000),
            "ts": time.time(),
        }

    def _reason_string_to_canonical(self, reason: str) -> SupervisionReason:
        """Map Guardian's free-form reason strings to canonical
        SupervisionReason enum values per SPEC §11.B step 2.

        Best-effort heuristic — exit-code-based classification (used by
        monitor_tick when an exitcode is observable) is more accurate.
        """
        r = reason.lower()
        if "heartbeat" in r or "starved" in r or "stall" in r:
            return SupervisionReason.HANG
        if "rss" in r or "oom" in r or "memory" in r:
            return SupervisionReason.OOM
        if "exitcode" in r:
            # Try to extract trailing integer
            import re as _re
            m = _re.search(r"(\d+)", reason)
            if m:
                return classify_exit_code(int(m.group(1)))
            return SupervisionReason.PANIC
        if "config" in r:
            return SupervisionReason.CONFIG_ERROR
        if "boot" in r:
            return SupervisionReason.BOOT_FAILURE
        if "killed" in r or "sigkill" in r:
            return SupervisionReason.KILLED
        if "broker_peer_dead" in r or "peer_died" in r:
            return SupervisionReason.PANIC  # broker observed peer death
        if "dependency" in r:
            return SupervisionReason.DEPENDENCY_BLOCKED
        return SupervisionReason.OTHER

    def _dispatch_reload_request(self, msg: dict) -> None:
        """SPEC §8.3 entry point — submit a MODULE_RELOAD_REQUEST to the
        reload orchestrator on _restart_executor.

        Validates the request shape + dispatches; emits an immediate
        MODULE_RELOAD_ACK status="failed" for malformed/unknown-module
        requests without spawning anything. Successful dispatch returns
        the executor Future — caller's request-progress visibility is
        via subsequent MODULE_RELOAD_ACK frames on the bus."""
        payload = msg.get("payload", {}) or {}
        module_name = payload.get("module_name")
        new_module_path = payload.get("new_module_path")
        swap_id = payload.get("swap_id") or str(uuid.uuid4())
        if not module_name or not isinstance(module_name, str):
            logger.warning(
                "[Guardian] MODULE_RELOAD_REQUEST malformed: %r", payload)
            self._emit_reload_ack(
                swap_id, str(module_name or ""),
                status="failed",
                reason="malformed_request",
                started_ts=time.time())
            return
        # Submit to _restart_executor — reuses the same thread pool that
        # owns restart() so we share its 4-worker capacity (more than
        # enough for realistic concurrent reload requests).
        try:
            self._restart_executor.submit(
                self._reload_module_sync,
                module_name,
                new_module_path,
                MODULE_RELOAD_DEFAULT_TIMEOUT_S,
                swap_id,
            )
        except Exception as e:  # noqa: BLE001
            logger.error(
                "[Guardian] MODULE_RELOAD_REQUEST dispatch failed for "
                "'%s': %s", module_name, e)
            self._emit_reload_ack(
                swap_id, module_name,
                status="failed",
                reason=f"dispatch_error:{e!r}",
                started_ts=time.time())
