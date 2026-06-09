"""Shared boot-grace heartbeat gate (BUG-BOOT-HEARTBEAT-SUPPRESSION-FALSE-CRASH).

A worker's SHM heartbeat (``ModuleStateWriter.heartbeat()``) must keep its slot
fresh DURING boot, not just after readiness — otherwise a slow boot under a
guardian restart cascade emits no SHM heartbeat → the slot goes stale →
guardian's ``* → CRASHED`` proxy (SPEC §11.I.2 "dead via PID + last_heartbeat
staleness") false-fires ``shm_pid_dead`` and kills an alive-but-booting worker.
This was the ~3 h T1 mainnet synthesis outage + the 90 ``shm_pid_dead``/min
cascade (2026-06-04); synthesis_worker was the prototype fix.

``heartbeat()`` PRESERVES the slot state (§11.I.5), so emitting it during boot is
safe: the slot stays ``starting``/``booted`` and readiness stays probe-gated
(state → running only via the recv-loop ``MODULE_PROBE_REQUEST`` handler). Past
the boot-grace window a still-not-ready worker stops heartbeating, so a genuinely
stuck boot is still caught by EMPTY/CRASHED supervision — no hidden hang.

Uniform rollout contract (one shared decision instead of 30+ hand-copied gates):
each worker keeps its module-level ``_WORKER_READY`` flag and adds a sibling
``_BOOT_DEADLINE`` (a ``time.monotonic()`` timestamp) set to
``time.monotonic() + BOOT_HEARTBEAT_GRACE_S`` at the same point it sets
``_WORKER_READY = False`` (i.e. before the heartbeat thread starts), then gates
its SHM ``heartbeat()`` on ``shm_heartbeat_allowed(_WORKER_READY, _BOOT_DEADLINE)``.
``_BOOT_DEADLINE=None`` degrades to the old readiness-only gate (a safe no-op).
"""
from __future__ import annotations

import time
from typing import Optional

# Max wall-clock a worker may emit its SHM heartbeat during boot before it must
# be READY. Matches the synthesis prototype's BOOT_HEARTBEAT_GRACE_S (2026-06-04):
# generous enough for a heavy cold boot under a cascade, short enough that a
# genuinely stuck boot still falls into CRASHED supervision afterwards.
BOOT_HEARTBEAT_GRACE_S: float = 300.0


def boot_deadline_from_now() -> float:
    """A boot-grace deadline ``BOOT_HEARTBEAT_GRACE_S`` seconds from now
    (``time.monotonic()`` clock). Call where the worker sets
    ``_WORKER_READY = False`` (before the heartbeat thread starts) and store the
    result in the module's ``_BOOT_DEADLINE``; pass it to
    :func:`shm_heartbeat_allowed`. Keeps ``time`` + the constant out of every
    worker module.
    """
    return time.monotonic() + BOOT_HEARTBEAT_GRACE_S


def shm_heartbeat_allowed(worker_ready: bool,
                          boot_deadline: Optional[float]) -> bool:
    """Return True if the worker should emit its SHM heartbeat right now.

    Emit once READY (steady state) OR during boot while still within the
    boot-grace window (``boot_deadline``, a ``time.monotonic()`` timestamp set
    when the heartbeat thread starts). ``boot_deadline=None`` → no grace
    (degrades to the old readiness-only gate; a safe no-op).
    """
    if worker_ready:
        return True
    return boot_deadline is not None and time.monotonic() < boot_deadline
