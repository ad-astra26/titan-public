"""
titan_hcl.orchestrator.dep_activation — D-SPEC-90 §11.G.2.5 dep-activation + critical-deps gating.

Phase 6 (D-SPEC-135) carved this out of titan_hcl/guardian.py into
guardian_hcl/dep_activation.py. Phase 11 §11.I.1 (D-SPEC-141 / v1.65.0)
relocates it under titan_hcl/orchestrator/ because dep-activation is an
Orchestrator concern (locked D5: start/stop owned by titan_hcl). See
SPEC §11.G + D-SPEC-90.

Mixed into class Orchestrator(OrchestratorReloadMixin, OrchestratorDepActivationMixin)
in core.py — `self` attributes (.bus, ._modules, ._reload_lock, etc.) come
from Orchestrator.__init__. Method bodies move verbatim — no logic change.
"""
"""
Guardian — Module supervisor for Titan V4.0 microkernel.

Manages the lifecycle of supervised module processes:
  - Start/stop/restart individual modules
  - Monitor heartbeats (kill/restart on timeout)
  - Track RSS per module (restart on threshold breach)
  - Provide module status to Core via the Divine Bus
  - Sliding-window restart tracking (prevents infinite restart loops)
  - Per-module heartbeat timeout (Spirit needs longer for heavy V4 work)

Each module runs as a separate multiprocessing.Process with its own
memory space, communicating exclusively through the Divine Bus.
"""
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
REENABLE_COOLDOWN_S = 180.0    # RFP_supervision_lifecycle §7.C — 3min (was 600) auto-re-enable cooldown
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



from titan_hcl.orchestrator.module_registry import (
    ModuleInfo,
    ModuleSpec,
    ModuleState,
    ReloadState,
)

class OrchestratorDepActivationMixin:
    """Mixin providing D-SPEC-90 §11.G.2.5 dep-activation + critical-deps gating — see SPEC §11.G + D-SPEC-90.

    Renamed from GuardianDepActivationMixin in Phase 11 §11.I.1 (D-SPEC-141 /
    v1.65.0) per the orchestrator/supervisor role split (locked D5).
    `GuardianDepActivationMixin = OrchestratorDepActivationMixin` back-compat
    alias is exported by `titan_hcl/guardian_hcl/__init__.py`.
    """

    def _activate_dependencies(self, name: str) -> None:
        """SPEC §11.G.2.5 (D-SPEC-90, v1.29.0) — pre-start dependency activation.

        For each `MODULE`-kind `CRITICAL`-severity dep declared with
        `action=ENSURE_RUNNING`, recursively start the dep if it is registered
        and currently STOPPED, then wait up to
        SUPERVISION_DEPENDENCY_ACTIVATION_TIMEOUT_S (30s) for the dep to reach
        RUNNING + emit MODULE_READY.

        Closes the lazy-start chicken-and-egg discovered post-§4.D
        meditation_worker extraction: `memory` was `autostart=False, lazy=True`
        and no subprocess could wake it from the parent's
        `MemoryProxy._ensure_started` bridge. Pre-§4.D it worked because
        plugin.py main process called `_ensure_started`; post-§4.D the
        subprocess can only emit bus events, leaving the lazy dep stranded.

        Runs at every Guardian.start(name) — including autostart-driven first
        start, lazy-wake start, and post-crash respawn. Soft and PROBE deps
        are ignored here (they go through §11.G.2 respawn check unchanged).

        DAG-acyclicity is enforced by SPEC §11.G.7
        (`arch_map phase-c verify --check-deps`); recursion terminates by
        induction on DAG depth.

        On dep-not-registered → emit SUPERVISION_DEPENDENCY_BLOCKED + log
        ERROR + continue (do not block dependent — let §11.G.2 catch the
        persistent failure mode if dep stays down).

        On activation-timeout → log WARNING + continue (do not block
        dependent — start proceeds and the dependent's own readiness probe
        absorbs the late-arrival, with §11.G.2 catching truly-down deps).
        """
        info = self._modules.get(name)
        if not info or not info.spec.dependencies:
            return
        for dep in info.spec.dependencies:
            if dep.action != DependencyAction.ENSURE_RUNNING:
                continue
            if dep.severity != DependencySeverity.CRITICAL:
                continue
            if dep.kind != DependencyKind.MODULE:
                continue

            dep_info = self._modules.get(dep.name)
            if dep_info is None:
                logger.error(
                    "[Guardian] ENSURE_RUNNING dep '%s' for module '%s' is "
                    "not registered — cannot activate (SPEC §11.G.2.5)",
                    dep.name, name)
                self.bus.publish(make_msg(
                    SUPERVISION_DEPENDENCY_BLOCKED, "guardian", "kernel", {
                        "child_name": name,
                        "supervisor": "guardian_HCL",
                        "blocked_dependency": dep.name,
                        "dependency_kind": dep.kind.value,
                        "severity": dep.severity.value,
                        "reason": "unregistered_dep",
                        "ts": time.time(),
                    }))
                continue

            if dep_info.state == ModuleState.RUNNING:
                continue  # Already up; nothing to do.

            if dep_info.state == ModuleState.DISABLED:
                logger.warning(
                    "[Guardian] ENSURE_RUNNING dep '%s' for module '%s' is "
                    "DISABLED — skipping activation (SPEC §11.G.2.5)",
                    dep.name, name)
                continue

            # Dep is registered + STOPPED/CRASHED/UNHEALTHY/STARTING →
            # announce + recursively start.
            logger.info(
                "[Guardian] Activating dep '%s' for module '%s' "
                "(SPEC §11.G.2.5 ENSURE_RUNNING; current state=%s)",
                dep.name, name, dep_info.state.value)
            self.bus.publish(make_msg(
                SUPERVISION_DEPENDENCY_ACTIVATING, "guardian", "kernel", {
                    "child_name": name,
                    "supervisor": "guardian_HCL",
                    "dependency_name": dep.name,
                    "dependency_kind": dep.kind.value,
                    "severity": dep.severity.value,
                    "ts": time.time(),
                }))

            # Recursive start — itself runs §11.G.2.5 on the dep's own deps.
            # Re-acquires _module_lock; the outer start() has not yet acquired
            # it (this method is called BEFORE the `with self._module_lock`
            # block in start()), so no deadlock.
            self.start(dep.name)

            # Phase 11 §11.I.2: readiness is SHM-driven. Reuse the canonical
            # probe-gated wait — it dispatches MODULE_PROBE_REQUEST, polls the
            # dep's own SHM slot for state=running, and mirrors RUNNING +
            # ready_time into ModuleInfo. The legacy MODULE_READY→info.state
            # path this used to poll was DELETED per D1/D2; without this the
            # dep wait ALWAYS timed out (~90s per dependency on cold boot —
            # live T1 18-min boot 2026-05-28).
            became_ready = self._wait_for_module_running(
                dep.name,
                timeout_s=SUPERVISION_DEPENDENCY_ACTIVATION_TIMEOUT_S)

            if not became_ready:
                logger.warning(
                    "[Guardian] Dep '%s' did not reach READY in %.0fs for "
                    "module '%s' — proceeding with dependent start anyway "
                    "(§11.G.2 respawn check catches persistent failure)",
                    dep.name, SUPERVISION_DEPENDENCY_ACTIVATION_TIMEOUT_S,
                    name)

    def _check_critical_dependencies(
        self, name: str, info: "ModuleInfo",
    ) -> Optional[str]:
        """Run pre-respawn dep check per SPEC §11.G.2. Returns the name of
        the first blocking critical dependency, or None if all are healthy.

        Soft deps that fail emit SUPERVISION_DEPENDENCY_DEGRADED (informational)
        but don't block. Custom check callables that raise are treated as
        "down" — fail closed for safety."""
        for dep in info.spec.dependencies:
            if dep.check is None:
                continue  # framework declared but no probe wired yet
            try:
                healthy = bool(dep.check())
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "[Guardian] dep probe '%s' for module '%s' raised %s — "
                    "treating as down (fail-closed)", dep.name, name, e)
                healthy = False
            if not healthy:
                if dep.severity == DependencySeverity.CRITICAL:
                    return dep.name
                # Soft dep failed — emit degraded event + continue.
                self.bus.publish(make_msg(
                    SUPERVISION_DEPENDENCY_DEGRADED, "guardian", "kernel", {
                        "child_name": name,
                        "supervisor": "guardian_HCL",
                        "dependency_name": dep.name,
                        "kind": dep.kind.value,
                        "severity": dep.severity.value,
                    }))
        return None
