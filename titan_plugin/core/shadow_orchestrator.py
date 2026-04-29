"""
Microkernel v2 Phase B.1 §7 — Shadow Core Swap orchestrator.

In-kernel orchestration logic for state-preserving atomic restart.
Runs synchronously inside the kernel process so it has direct bus
access (publish + temporarily subscribe to collect responses).

Triggered via POST /v4/maker/shadow-swap (api_subprocess endpoint
which calls kernel.shadow_swap_orchestrate via kernel_rpc, §8).
CLI wrapper at scripts/shadow_swap.py provides Maker UX.

Phase flow (mirrors PLAN §architecture):
  0. PREFLIGHT — generate event_id, snapshot pre-state, publish QUEUED
  1. READINESS WAIT — poll workers every 1s for ≤120s grace; deferred
     on timeout (NO force flag — Maker's design call)
  2. HIBERNATE — broadcast HIBERNATE, collect HIBERNATE_ACK per-layer
     (L1=10s / L2=20s / L3=30s); rollback if any required worker fails
  3. SHADOW BOOT — spawn `titan_main.py --server --shadow-port N
     --restore-from PATH`; wait for new /health (max 60s)
  4. NGINX SWAP — sed upstream port + nginx -s reload + verify external
  5. SHUTDOWN OLD — SIGTERM kernel + workers; cleanup snapshot file

Failure modes return a result dict with `phase`, `outcome`, `reason`,
`event_id`, `gap_seconds`, `audit_log_path`. CLI parses + reports.

PLAN: titan-docs/PLAN_microkernel_phase_b1_shadow_swap.md §7
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import queue
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Optional

from titan_plugin import bus
from titan_plugin.bus import make_msg
from titan_plugin.core import shadow_protocol as sp


logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────

#: Audit log — append-only one-line-JSON per swap attempt.
AUDIT_LOG_PATH = Path("data/shadow_swap_history.jsonl")

#: Active port file — atomic-rename single-int file. CLI helper reads.
ACTIVE_PORT_PATH = Path("data/active_api_port")

#: Default API port (used when active_port file missing).
DEFAULT_API_PORT = 7777


# ── Multi-criterion health (BUG-B1-WEAK-HEALTH-CHECK fix) ───────────
#
# Pre-fix: shadow-boot success = HTTP 200 on /health. Discovered B.1
# deploy test 7 on T1: shadow's API bound 7779 + returned /health 200
# while memory_worker had crash-looped (DuckDB lock conflict). Orchestrator
# declared outcome=OK; nginx swapped to a degraded backend.
#
# Fix: a shadow kernel is "healthy" only when ALL of:
#   1. /health returns 200
#   2. >= min_modules_running modules in state="running"
#   3. ALL critical_modules in state="running"
#   4. NO module's restart_count exceeds max_restart_count
#   5. ALL running modules have heartbeat age < max_heartbeat_age_s
#   6. (optional smoke) a dynamic state field advances between two samples
#      separated by smoke_interval_s — proves the kernel isn't merely
#      bound + responsive but actually computing
#
# Defaults err on the side of caution. Per-call overrides allow tests to
# loosen thresholds and ops to tighten them when shadow_swap is rolled
# out to T2/T3.

@dataclasses.dataclass(frozen=True)
class HealthCriteria:
    """Knobs for multi-criterion shadow health gate.

    Used by `_check_multi_criterion_health()` and `_wait_for_health()`.
    All defaults chosen for production prod swap on a healthy fleet —
    relax for tests via constructor args; tighten via per-deploy override.
    """
    #: Minimum module count in state="running" (of ~18 total). Slack of
    #: 4 covers L3 modules that race the boot (api, observatory_writer,
    #: meta_teacher, etc.) — they sometimes report state=starting briefly.
    min_modules_running: int = 14

    #: These workers MUST be in state="running". Missing/stopped/disabled
    #: any of them = unsafe to swap (kernel can't function meaningfully).
    #: Order matches L0→L1→L2→L3 dependency direction.
    critical_modules: tuple[str, ...] = (
        "spirit", "body", "mind",       # L1 Trinity (consciousness loop)
        "timechain",                    # L2 substrate (audit/identity)
        "memory",                       # L2 (state durability)
        "imw",                          # L1 (single-writer for memory)
        "api",                          # L3 (the /health endpoint itself)
    )

    #: Hard ceiling on restart_count per module. >0 = crash-loop in flight,
    #: not safe to swap onto. Set to 1+ if a flaky non-critical module
    #: legitimately respawns once during clean boot — prefer to fix the
    #: flake instead.
    max_restart_count: int = 0

    #: How fresh a module's last heartbeat must be (seconds). Too generous
    #: misses dead-but-not-yet-crashed workers; too tight false-fails on
    #: large GC pauses. 30s = ~3× heartbeat interval (10s typical) per the
    #: heartbeat thread cadence in persistence_entry.py:97.
    max_heartbeat_age_s: float = 30.0

    #: Enable end-to-end smoke step: read a dynamic state field twice
    #: separated by smoke_interval_s; the field MUST advance. Confirms
    #: the kernel is computing, not just serving cached /health responses.
    smoke_test_enabled: bool = True

    #: Seconds between the two smoke-test reads (must be > expected
    #: minimum dynamic-state cadence; spirit.uptime advances every
    #: second so 3s gives 3-second margin).
    smoke_interval_s: float = 3.0

    #: Field path within /v4/state to sample for smoke (dotted notation).
    #: spirit.uptime increments monotonically every second once spirit
    #: is running — most reliable smoke field.
    smoke_field_path: str = "data.guardian.spirit.uptime"

    # ── Phase B.2 — bus_socket broker criteria ──────────────────────────
    # Activated only when the kernel has a bus_socket broker attached
    # (microkernel.bus_ipc_socket_enabled=true). Defaults disable the gate
    # so a Phase B.1-only kernel still passes — BUT the kernel-side check
    # respects these whenever a broker IS present, regardless of flag.

    #: Minimum count of workers connected to the bus_socket broker.
    #: 0 = disabled (default — works in Phase B.1-only mode and during
    #: B.2 first-deploy soak before workers have all flipped). Set to the
    #: expected fleet size (e.g., 14) per-deploy to catch missing reconnects
    #: after a swap.
    min_connected_bus_workers: int = 0

    #: Maximum tolerated bus drop rate per subscriber (percent).
    #: 100.0 = disabled. Below 5.0 in steady state per OBS-mkernel-b2-bus-soak.
    #: Used to gate Phase 4 nginx swap: if any subscriber is drop-thrashing,
    #: refuse the swap and unwind.
    max_bus_drop_rate_60s_pct: float = 100.0

    # ── Phase B.2.1 — supervision-transfer criterion (workers outlive swap) ──
    # When B.2.1 wiring is active (any spawn-mode worker has the swap_handler
    # wired + bus_ipc_socket_enabled=true), the orchestrator polls shadow's
    # /v4/state.guardian for adopted=True modules and gates Phase 4 on this
    # count meeting min_adopted_workers. Default 0 = disabled (back-compat
    # with B.1-only and B.2-only deploys where no worker has yet wired the
    # adoption protocol).
    min_adopted_workers: int = 0

    # ── Phase B.3 — Shadow DB integrity gate (Layer 2 corruption-prevention) ──
    # 2026-04-27 PM addition (T2-shadow-swap-fix session): even with Fix B
    # (per-DB-type aware shadow data dir) blocking the inode-share corruption
    # mechanism, an unanticipated case could still produce a malformed DB
    # in the shadow before nginx_swap. This gate runs `PRAGMA quick_check`
    # on every top-level SQLite (+ timechain/index.db) in shadow's data dir
    # right before the swap commits. Any malformed DB → fail the gate →
    # orchestrator unwinds → swap is refused. The shadow process exits;
    # old kernel keeps running unchanged. Cost: ~1-3s for ~30 DBs at boot
    # time when no contention. Default True — we always want this.
    #
    # Concrete incident this protects against: 2026-04-26 T1 "successful"
    # shadow swap committed with timechain/index.db silently corrupt. The
    # corruption manifested 1+ hour later as TCStatusWarmer cascade failure.
    # With this gate, that swap would have aborted at health-check phase.
    shadow_db_integrity_check_enabled: bool = True


def _kill_shadow_process_tree(proc) -> None:
    """Kill the shadow's entire process group on rollback.

    The shadow is spawned with `start_new_session=True` so it has its own
    process group (its workers are children). `proc.terminate()` only
    SIGTERMs the shadow's main pid — its workers can survive as orphans.
    Use `os.killpg(SIGKILL)` to nuke the entire tree atomically.

    Codified 2026-04-28 PM after the first T1 shadow swap rollback left 6
    stale `--shadow-port 7779` worker processes alive after rollback.
    """
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        # Already dead, or wasn't a session leader — fall back to plain
        # SIGKILL on the main pid (plus any direct children we can find).
        try:
            proc.kill()
        except Exception:
            pass
    except Exception:
        # Last-resort: best-effort SIGTERM to avoid raising during rollback.
        try:
            proc.terminate()
        except Exception:
            pass


def _get_dotted(d: Any, path: str) -> Any:
    """Walk a nested dict via dotted path; return None on any miss."""
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _fetch_state_json(port: int, timeout: float = 5.0) -> Optional[dict]:
    """GET /v4/state on the kernel; return parsed JSON or None on error."""
    import urllib.request
    url = f"http://127.0.0.1:{port}/v4/state"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            if resp.status != 200:
                return None
            return json.loads(resp.read())
    except Exception:
        return None


def _check_multi_criterion_health(
    port: int, criteria: HealthCriteria,
) -> tuple[bool, dict]:
    """One-shot health check against the multi-criterion gate.

    Returns (passed, diagnosis). The diagnosis dict is stable enough to
    log into shadow_swap_history.jsonl + surface to the polling CLI for
    Maker visibility — every failure mode names which criterion failed
    and gives the observed value vs. expected.

    Does not retry; callers (`_wait_for_health`) wrap this in a poll loop.
    """
    diag: dict[str, Any] = {
        "port": port,
        "checked_at": time.time(),
        "checks": {},
    }

    # 1. /health returns 200 (cheap pre-check; if this fails, the kernel
    # is definitely not ready and there's no point in any further check)
    import urllib.request
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/health", timeout=2
        ) as resp:
            health_ok = resp.status == 200
    except Exception as e:
        diag["checks"]["health_endpoint"] = {"pass": False, "error": str(e)[:200]}
        return False, diag
    diag["checks"]["health_endpoint"] = {"pass": health_ok}
    if not health_ok:
        return False, diag

    # 2-5. State endpoint gives us module roster + per-module guardian state
    state = _fetch_state_json(port)
    if state is None:
        diag["checks"]["state_endpoint"] = {"pass": False, "error": "state_fetch_failed"}
        return False, diag
    diag["checks"]["state_endpoint"] = {"pass": True}

    modules = _get_dotted(state, "data.guardian") or {}
    if not isinstance(modules, dict):
        diag["checks"]["modules_dict"] = {"pass": False, "error": "no guardian.modules"}
        return False, diag

    # 2. Min modules running
    running = [n for n, info in modules.items()
               if isinstance(info, dict) and info.get("state") == "running"]
    diag["checks"]["min_modules_running"] = {
        "pass": len(running) >= criteria.min_modules_running,
        "running_count": len(running),
        "required": criteria.min_modules_running,
        "running_names": sorted(running),
    }
    if len(running) < criteria.min_modules_running:
        return False, diag

    # 3. All critical modules running
    crit_running = [m for m in criteria.critical_modules
                    if isinstance(modules.get(m), dict)
                    and modules[m].get("state") == "running"]
    crit_missing = [m for m in criteria.critical_modules if m not in crit_running]
    diag["checks"]["critical_modules_running"] = {
        "pass": not crit_missing,
        "missing": crit_missing,
        "required": list(criteria.critical_modules),
    }
    if crit_missing:
        return False, diag

    # 4. No crash-loops in flight (restart_count above threshold for ANY module)
    crashing = [
        (n, info.get("restart_count", 0))
        for n, info in modules.items()
        if isinstance(info, dict) and int(info.get("restart_count", 0) or 0) > criteria.max_restart_count
    ]
    diag["checks"]["no_crash_loops"] = {
        "pass": not crashing,
        "modules_above_threshold": [{"name": n, "count": c} for n, c in crashing],
        "threshold": criteria.max_restart_count,
    }
    if crashing:
        return False, diag

    # 5. Heartbeats fresh on all running modules
    stale = [
        (n, float(info.get("last_heartbeat_age", -1) or -1))
        for n, info in modules.items()
        if isinstance(info, dict) and info.get("state") == "running"
        and float(info.get("last_heartbeat_age", 0) or 0) > criteria.max_heartbeat_age_s
    ]
    diag["checks"]["heartbeats_fresh"] = {
        "pass": not stale,
        "stale_modules": [{"name": n, "age_s": age} for n, age in stale],
        "max_age_s": criteria.max_heartbeat_age_s,
    }
    if stale:
        return False, diag

    # 6. (Phase B.2) bus_socket broker criteria — only checked if the gate
    # is enabled (criteria values different from disabled defaults).
    bus_diag = _check_bus_broker_criteria(state, criteria)
    if bus_diag is not None:
        diag["checks"]["bus_broker"] = bus_diag
        if not bus_diag.get("pass", False):
            return False, diag

    # 7. (Phase B.2.1) Adopted-workers criterion — verify the spawn-mode
    # workers we expected to outlive the swap are registered as adopted=True
    # in shadow's Guardian. Disabled (gate skipped) when min_adopted_workers
    # = 0, the default.
    adopted_diag = _check_adopted_workers_criteria(state, criteria)
    if adopted_diag is not None:
        diag["checks"]["adopted_workers"] = adopted_diag
        if not adopted_diag.get("pass", False):
            return False, diag

    # 8. (Phase B.3 Layer 2) Shadow DB integrity — refuse swap if any of
    # shadow's SQLite/DuckDB DBs are malformed. Closes the failure mode
    # from 2026-04-26 where a swap committed with corrupt timechain/index.db
    # that surfaced 1+ hour later as cascading observatory outage.
    db_integrity_diag = _check_shadow_db_integrity(port, criteria)
    if db_integrity_diag is not None:
        diag["checks"]["shadow_db_integrity"] = db_integrity_diag
        if not db_integrity_diag.get("pass", False):
            return False, diag

    diag["all_pre_smoke_passed"] = True
    return True, diag


def _check_shadow_db_integrity(
    shadow_port: int, criteria: HealthCriteria,
) -> Optional[dict]:
    """Phase B.3 Layer 2 — verify shadow's SQLite DBs are not malformed.

    Runs `PRAGMA quick_check` against every top-level *.db in the shadow's
    data dir, plus the deep `timechain/index.db` (since it's the one that
    actually corrupted yesterday). Read-only via `mode=ro` URI so it's safe
    to run while the shadow's writers are also bound to the file.

    Returns None when the gate is disabled (criteria.shadow_db_integrity_check_enabled=False).
    Otherwise returns a diag dict with pass bit + per-DB results. Failure
    is "any DB returns non-ok from quick_check" or "open fails for reasons
    other than expected lock-conflict".

    DuckDB files (titan_memory.duckdb) are NOT checked here — DuckDB
    forbids concurrent read-only opens while a writer holds the lock,
    so we'd get a lock-conflict false positive. Fix B prevents inode-share
    corruption regardless; if a real DuckDB corruption surfaces it'll
    manifest at write time and Guardian's restart loop will catch it.
    """
    if not getattr(criteria, "shadow_db_integrity_check_enabled", True):
        return None

    import sqlite3
    from pathlib import Path
    from . import shadow_data_dir as sdd

    project_root = Path(__file__).resolve().parents[2]
    shadow_dir = sdd.shadow_data_dir_for_port(shadow_port, root=project_root)
    if not shadow_dir.exists():
        return {
            "pass": False,
            "error": f"shadow_dir_missing: {shadow_dir}",
            "note": "shadow data dir not found; cannot verify DB integrity",
        }

    # Collect SQLite DBs to check. Top-level *.db + the known-critical
    # timechain/index.db (the one that actually corrupted in the 2026-04-26
    # incident). Skip *.db-wal / *.db-shm (those are SQLite internals,
    # checked indirectly by quick_check on the main file).
    db_files: list[Path] = []
    for f in sorted(shadow_dir.glob("*.db")):
        if f.is_file():
            db_files.append(f)
    tc_index = shadow_dir / "timechain" / "index.db"
    if tc_index.exists() and tc_index.is_file():
        db_files.append(tc_index)

    if not db_files:
        # No DBs to check — pass silently (e.g., empty shadow dir during tests)
        return {"pass": True, "checked_count": 0, "note": "no SQLite DBs found"}

    failed: list[dict] = []
    checked: list[str] = []
    for db_path in db_files:
        rel = db_path.relative_to(shadow_dir).as_posix()
        try:
            conn = sqlite3.connect(
                f"file:{db_path}?mode=ro", uri=True, timeout=5.0,
            )
            try:
                result = conn.execute("PRAGMA quick_check").fetchall()
            finally:
                conn.close()
            if result == [("ok",)]:
                checked.append(rel)
            else:
                failed.append({
                    "name": rel,
                    "result": [str(r)[:200] for r in result[:3]],
                })
        except Exception as e:
            failed.append({"name": rel, "error": str(e)[:200]})

    return {
        "pass": not failed,
        "checked_count": len(checked),
        "failed_count": len(failed),
        "failed": failed,
        "shadow_dir": str(shadow_dir),
    }


def _check_adopted_workers_criteria(state: dict,
                                    criteria: HealthCriteria) -> Optional[dict]:
    """Phase B.2.1 — verify expected count of workers were adopted by shadow.

    Returns None when min_adopted_workers <= 0 (gate disabled — back-compat
    with B.1/B.2 deploys before any worker has wired the adoption protocol).
    Otherwise returns a diagnosis dict with pass bit + observed count.

    Reads /v4/state.guardian.<name>.adopted as set by Guardian.get_status()
    after Guardian.adopt_worker() (C3).
    """
    if criteria.min_adopted_workers <= 0:
        return None
    guardian_status = _get_dotted(state, "data.guardian")
    if not isinstance(guardian_status, dict):
        return {
            "pass": False,
            "error": "guardian_status_absent",
            "note": "criteria gate active but /v4/state.guardian missing",
        }
    adopted_modules = sorted(
        name for name, info in guardian_status.items()
        if isinstance(info, dict) and info.get("adopted") is True
    )
    return {
        "pass": len(adopted_modules) >= criteria.min_adopted_workers,
        "adopted_count": len(adopted_modules),
        "min_required": criteria.min_adopted_workers,
        "adopted_modules": adopted_modules,
    }


def _check_bus_broker_criteria(state: dict, criteria: HealthCriteria) -> Optional[dict]:
    """Phase B.2 — verify the kernel's bus_socket broker (if any) is healthy.

    Returns None if both gates are disabled (defaults — back-compat with
    Phase B.1-only kernels). Otherwise returns a diagnosis dict with pass
    bit + per-criterion observations.

    The broker stats live in /v4/state under data.bus_broker (added when
    the broker is attached). Absence of that key means broker not present
    on this kernel — pass any disabled gate (caller decides if that's OK).
    """
    gate_active = (criteria.min_connected_bus_workers > 0
                   or criteria.max_bus_drop_rate_60s_pct < 100.0)
    if not gate_active:
        return None
    broker_stats = _get_dotted(state, "data.bus_broker")
    if not isinstance(broker_stats, dict):
        # Gate active but no broker on this kernel — that's a config mismatch
        return {
            "pass": False,
            "error": "bus_broker_stats_absent",
            "note": "criteria gate active but kernel reports no broker — flag mismatch?",
        }
    sub_count = int(broker_stats.get("subscriber_count", 0) or 0)
    subs = broker_stats.get("subscribers") or []
    # Drop-rate computation: max across subscribers (worst case)
    worst_drop_rate = 0.0
    worst_sub = None
    for s in subs:
        if not isinstance(s, dict):
            continue
        recv = max(int(s.get("recv_count_60s", 0) or 0), 1)
        drops = int(s.get("drop_count_60s", 0) or 0)
        rate = (drops / recv) * 100.0
        if rate > worst_drop_rate:
            worst_drop_rate = rate
            worst_sub = s.get("name", "?")
    return {
        "pass": (sub_count >= criteria.min_connected_bus_workers
                 and worst_drop_rate <= criteria.max_bus_drop_rate_60s_pct),
        "subscriber_count": sub_count,
        "min_required": criteria.min_connected_bus_workers,
        "worst_drop_rate_pct": worst_drop_rate,
        "max_allowed_pct": criteria.max_bus_drop_rate_60s_pct,
        "worst_subscriber": worst_sub,
    }


def _check_smoke_advancing(
    port: int, criteria: HealthCriteria,
) -> tuple[bool, dict]:
    """End-to-end smoke: sample a dynamic field twice; require monotonic advance.

    Confirms the kernel is doing work, not just serving stale /health.
    spirit.uptime increments every second the spirit worker is alive; any
    monotonic advance over smoke_interval_s confirms the supervised
    process is truly progressing.
    """
    sample1 = _fetch_state_json(port)
    v1 = _get_dotted(sample1, criteria.smoke_field_path) if sample1 else None
    time.sleep(criteria.smoke_interval_s)
    sample2 = _fetch_state_json(port)
    v2 = _get_dotted(sample2, criteria.smoke_field_path) if sample2 else None

    diag = {
        "field": criteria.smoke_field_path,
        "interval_s": criteria.smoke_interval_s,
        "sample1": v1,
        "sample2": v2,
    }

    if v1 is None or v2 is None:
        diag["pass"] = False
        diag["error"] = "field_unreadable"
        return False, diag

    try:
        advanced = float(v2) > float(v1)
    except (TypeError, ValueError):
        diag["pass"] = False
        diag["error"] = "field_not_numeric"
        return False, diag

    diag["pass"] = advanced
    if not advanced:
        diag["error"] = "field_did_not_advance"
    return advanced, diag


# ── Result envelope ─────────────────────────────────────────────────

class SwapResult:
    """Mutable result accumulator for the orchestrator phases.

    Phase methods append to .audit (per-phase events) and set .phase /
    .outcome / .reason. Final dict is written to AUDIT_LOG_PATH +
    returned to caller (api_subprocess endpoint → CLI).
    """

    def __init__(self, event_id: str, reason: str):
        self.event_id = event_id
        self.reason = reason
        self.phase = "preflight"
        self.outcome = "in_progress"  # "ok" | "deferred" | "rollback" | "error"
        self.failure_reason: Optional[str] = None
        self.started_at = time.time()
        self.gap_seconds: float = 0.0
        self.blockers_waited_on: list[dict] = []
        self.hibernate_acks: list[dict] = []
        self.shadow_port: Optional[int] = None
        self.kernel_version_to: Optional[str] = None
        self.audit: list[dict] = []

    def event(self, msg: str, **kw) -> None:
        entry = {"ts": time.time(), "phase": self.phase, "msg": msg, **kw}
        self.audit.append(entry)
        logger.info("[shadow_swap] phase=%s %s %s", self.phase, msg, kw)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "reason": self.reason,
            "phase": self.phase,
            "outcome": self.outcome,
            "failure_reason": self.failure_reason,
            "started_at": self.started_at,
            "elapsed_seconds": time.time() - self.started_at,
            "gap_seconds": self.gap_seconds,
            "blockers_waited_on": self.blockers_waited_on,
            "hibernate_acks": self.hibernate_acks,
            "shadow_port": self.shadow_port,
            "kernel_version_to": self.kernel_version_to,
            "audit_events": len(self.audit),
        }


# ── Phase C C-S2 (BUG-SERVICES-WATCHDOG-SHADOW-SWAP-RACE-20260428) ──
#
# Per PLAN_microkernel_phase_c_s2_kernel.md §17.1: shadow_orchestrator
# writes /tmp/titan${N}_restart.lock at swap start, heartbeats it every
# 10s during the swap, and removes it on completion. services_watchdog.sh
# reads this file to skip its duplicate-kill heuristic during the swap
# window (and force-cleans if the lock is expired = orchestrator crashed).
#
# Format:
#   {
#     "pid": <orchestrator process pid>,
#     "swap_id": "<event_id>",
#     "started_at": <epoch float>,
#     "expected_end_at": <epoch float>,
#     "heartbeat_at": <epoch float>,    # updated every 10s
#     "writer": "shadow_orchestrator"   # distinguishes from safe_restart.sh
#   }
#
# Backward-compat: services_watchdog.sh's pre-existing 90s plain-epoch
# logic still works for safe_restart.sh / t{2,3}_manage.sh writers; the
# new JSON form is detected by the leading "{" character.

#: Default expected swap window — orchestrator publishes one full window
#: of "expected" rather than rolling. Overrides via SHADOW_SWAP_LOCK_TTL_S.
SHADOW_SWAP_LOCK_DEFAULT_TTL_S = 90.0
#: Heartbeat cadence for the lockfile while swap is in flight.
SHADOW_SWAP_LOCK_HEARTBEAT_S = 10.0


def _restart_lock_path(titan_id: str) -> Path:
    """Return /tmp/titan${N}_restart.lock for given titan_id (T1 → titan1)."""
    n = "".join(ch for ch in titan_id.lower() if ch.isdigit()) or "1"
    return Path(f"/tmp/titan{n}_restart.lock")


def _kernel_titan_id(kernel: Any) -> str:
    """Best-effort resolution of the running Titan's id (T1/T2/T3)."""
    val = getattr(kernel, "titan_id", None)
    if isinstance(val, str) and val:
        return val
    val = os.environ.get("TITAN_KERNEL_TITAN_ID") or os.environ.get("TITAN_ID")
    return val or "T1"


def write_swap_lock(
    titan_id: str,
    swap_id: str,
    *,
    ttl_s: float = SHADOW_SWAP_LOCK_DEFAULT_TTL_S,
) -> Path:
    """Atomic-rename write of the shadow-swap lockfile. Returns path."""
    path = _restart_lock_path(titan_id)
    now = time.time()
    payload = {
        "pid": os.getpid(),
        "swap_id": swap_id,
        "started_at": now,
        "expected_end_at": now + ttl_s,
        "heartbeat_at": now,
        "writer": "shadow_orchestrator",
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(path)
    return path


def heartbeat_swap_lock(path: Path) -> None:
    """Refresh `heartbeat_at` (and slide `expected_end_at`) atomically.

    Best-effort: silently no-ops if file disappeared (orchestrator vs
    cleanup race).
    """
    try:
        text = path.read_text()
        body = json.loads(text)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return
    now = time.time()
    body["heartbeat_at"] = now
    body["expected_end_at"] = now + SHADOW_SWAP_LOCK_DEFAULT_TTL_S
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(body))
        tmp.replace(path)
    except OSError:
        pass


def remove_swap_lock(path: Path) -> None:
    """Delete the lockfile (best-effort; missing file is fine)."""
    try:
        path.unlink()
    except FileNotFoundError:
        pass


class _SwapLockHeartbeat:
    """Background thread that calls `heartbeat_swap_lock` at SHADOW_SWAP_LOCK_HEARTBEAT_S
    cadence. Call `stop()` from the swap finally-block to terminate."""

    def __init__(self, path: Path):
        self.path = path
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="shadow-swap-lock-heartbeat", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self, *, remove: bool = True) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        if remove:
            remove_swap_lock(self.path)

    def _run(self) -> None:
        while not self._stop.wait(SHADOW_SWAP_LOCK_HEARTBEAT_S):
            heartbeat_swap_lock(self.path)


# ── Active-port file helpers ────────────────────────────────────────

def read_active_port() -> int:
    """Return the API port currently serving traffic (default 7777)."""
    try:
        return int(ACTIVE_PORT_PATH.read_text().strip())
    except (FileNotFoundError, ValueError):
        return DEFAULT_API_PORT


def write_active_port(port: int) -> None:
    """Atomic-rename write of single-int port number."""
    ACTIVE_PORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = ACTIVE_PORT_PATH.with_suffix(".tmp")
    tmp.write_text(str(int(port)))
    tmp.replace(ACTIVE_PORT_PATH)


def pick_shadow_port(current: int) -> int:
    """Return the OTHER ping-pong port. Validates current ∈ {7777, 7779}."""
    a, b = sp.PING_PONG_PORTS
    if current == a:
        return b
    if current == b:
        return a
    # If config has overridden current to something custom, fall back to b.
    return b


# ── Audit log ───────────────────────────────────────────────────────

def append_audit(result: SwapResult) -> None:
    AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(result.to_dict())
    with AUDIT_LOG_PATH.open("a") as f:
        f.write(line + "\n")


# ── Bus helpers (subscribe + drain) ─────────────────────────────────

def _subscribe_temp(bus_obj, name: str = "shadow_swap"):
    """Subscribe a temporary 'shadow_swap' module to the bus.

    Returns the queue object; caller must unsubscribe via bus_obj
    when done (or just leave it — subscribers go away with the bus).
    """
    return bus_obj.subscribe(name)


def _drain_messages(q, msg_types: set[str], timeout: float,
                    kernel=None) -> list[dict]:
    """Collect messages of specified types within timeout (seconds).

    The orchestrator runs synchronously in the kernel main thread (via
    kernel_rpc). While blocked here, the kernel's normal Guardian-drain
    cycle doesn't run — meaning worker subprocess send_queues fill up
    but their messages never reach the bus's in-process subscriber
    queues. We drive the drain ourselves between get() retries so worker
    HIBERNATE_ACK / UPGRADE_READINESS_REPORT messages actually arrive.
    """
    deadline = time.monotonic() + timeout
    out = []
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        # Drive Guardian's drain so worker messages reach the bus → subscriber.
        if kernel is not None and getattr(kernel, "guardian", None) is not None:
            try:
                kernel.guardian.drain_send_queues()
            except Exception:
                pass
        try:
            msg = q.get(timeout=min(0.2, remaining))
        except queue.Empty:
            continue
        if msg.get("type") in msg_types:
            out.append(msg)
    return out


# ── Phase 1: Readiness wait ─────────────────────────────────────────

def _phase_readiness_wait(
    bus_obj,
    inbox,
    result: SwapResult,
    *,
    grace: float = sp.READINESS_GRACE_SECONDS,
    kernel=None,
) -> bool:
    """Poll workers for readiness. Return True if all-clear, False if deferred.

    Publishes UPGRADE_READINESS_QUERY every poll interval; collects
    UPGRADE_READINESS_REPORT replies; classifies blockers; emits
    SYSTEM_UPGRADE_PENDING with the union of blockers every 5s.

    Stops as soon as ALL responding workers report ready=True AND no
    new workers report blockers in the last cycle.
    """
    result.phase = "readiness_wait"
    deadline = time.monotonic() + grace
    last_pending_emit = 0.0

    while time.monotonic() < deadline:
        # Publish a new readiness query
        bus_obj.publish(make_msg(
            bus.UPGRADE_READINESS_QUERY,
            src="shadow_swap", dst="all",
            payload={"event_id": result.event_id},
        ))

        # Wait for reports for one poll interval
        reports = _drain_messages(
            inbox, {bus.UPGRADE_READINESS_REPORT},
            timeout=sp.READINESS_REPORT_TIMEOUT,
            kernel=kernel,
        )

        # Aggregate blockers across all responding workers
        all_hard: list[dict] = []
        all_soft: list[dict] = []
        any_not_ready = False
        responders = set()
        for r in reports:
            payload = r.get("payload", {})
            responders.add(payload.get("src", r.get("src", "?")))
            if not payload.get("ready", False):
                any_not_ready = True
            for b in payload.get("hard", []):
                all_hard.append({**b, "from": payload.get("src", "?")})
            for b in payload.get("soft", []):
                all_soft.append({**b, "from": payload.get("src", "?")})

        if not any_not_ready and responders:
            result.event("readiness_clear",
                         responders=sorted(responders))
            return True

        # Periodic PENDING broadcast (Maker UX + spirit re-thought-cb)
        now = time.monotonic()
        if now - last_pending_emit >= sp.READINESS_PENDING_NOTIFY_INTERVAL:
            last_pending_emit = now
            bus_obj.publish(make_msg(
                bus.SYSTEM_UPGRADE_PENDING,
                src="shadow_swap", dst="all",
                payload={
                    "event_id": result.event_id,
                    "elapsed_seconds": time.time() - result.started_at,
                    "grace_seconds": grace,
                    "blockers": all_hard + all_soft,
                },
            ))
            result.event("pending_emit",
                         hard=len(all_hard), soft=len(all_soft))

        # Brief tick before next query
        time.sleep(sp.READINESS_POLL_INTERVAL)

    # Grace exceeded — defer
    result.outcome = "deferred"
    result.failure_reason = "readiness_grace_exceeded"
    result.blockers_waited_on = all_hard + all_soft
    bus_obj.publish(make_msg(
        bus.SYSTEM_UPGRADE_PENDING_DEFERRED,
        src="shadow_swap", dst="all",
        payload={
            "event_id": result.event_id,
            "blockers": all_hard + all_soft,
            "elapsed_seconds": time.time() - result.started_at,
        },
    ))
    result.event("readiness_deferred",
                 blockers_total=len(all_hard) + len(all_soft))
    return False


# ── Phase 2: Hibernate ──────────────────────────────────────────────

def _phase_hibernate(
    bus_obj,
    inbox,
    result: SwapResult,
    *,
    expected_workers: list[str],
    kernel=None,
) -> bool:
    """Broadcast HIBERNATE; collect ACKs per-layer.

    Returns True on success, False on rollback. Sets result.hibernate_acks
    to the aggregated ACK payloads. Sets failure_reason on timeout.
    """
    result.phase = "hibernate"

    # SYSTEM_UPGRADE_STARTING — last call for self-thoughts
    bus_obj.publish(make_msg(
        bus.SYSTEM_UPGRADE_STARTING,
        src="shadow_swap", dst="all",
        payload={"event_id": result.event_id},
    ))

    # Phase B.2 — BUS_HANDOFF self-awareness signal. Informs workers a
    # kernel swap is in flight so spirit can emit a thought ("I sense the
    # kernel changing under me; my body keeps moving"). The actual socket
    # reattach is mechanical: workers see EOF on the broker socket and
    # auto-reconnect via BusSocketClient's state machine. This message is
    # for self-awareness + observability only — workers do not block on it.
    # No-op when bus_ipc_socket_enabled=false (workers won't receive it
    # via socket; the legacy mp.Queue path is unaffected).
    bus_obj.publish(make_msg(
        bus.BUS_HANDOFF,
        src="kernel", dst="all",
        payload={
            "event_id": result.event_id,
            "reason": "shadow_swap",
            "expected_downtime_ms": 500,
        },
    ))

    # HIBERNATE — per-worker targeted publish (NOT dst="all").
    #
    # 2026-04-27 PM fix (T2 shadow-swap investigation): the prior
    # `dst="all"` broadcast skipped reply_only modules per
    # `bus.py:512` (broadcast loop excludes `_reply_only`). IMW and
    # observatory_writer are reply_only=True (legacy mp.Queue path,
    # no socket-broker subscription) — they NEVER received HIBERNATE,
    # so they never hibernated. Old-kernel IMW stayed alive holding
    # the unix socket + WAL lock on `data/inner_memory.db` while
    # shadow's IMW tried to bind the same socket / open the same DB
    # inode (cp -al hardlinks share inodes for SQLite mainfiles since
    # WAL writes don't break the link). Result on T2: shadow IMW hit
    # SQLITE_CANTOPEN, cascade collapsed the whole swap.
    #
    # Per-worker dst publish bypasses `_reply_only` filtering at
    # `bus.py:518-527` (dst-specific path doesn't apply that filter),
    # guaranteeing every registered worker — including reply_only
    # ones — receives the death signal and can exit cleanly.
    for worker_name in expected_workers:
        bus_obj.publish(make_msg(
            bus.HIBERNATE,
            src="shadow_swap", dst=worker_name,
            payload={"event_id": result.event_id},
        ))

    # M1 (2026-04-27 PM audit): split ack collection by start_method.
    # Spawn-mode workers (B.2.1 outlive) reply with BUS_HANDOFF_ACK and
    # do NOT exit. Fork-mode workers (improved-B.1) reply with HIBERNATE_ACK
    # and exit. Both messages have dst="shadow_swap" so the inbox collects
    # them in a single drain. Classifying missing acks per-mode prevents
    # straggler_kill from SIGTERM'ing live spawn-mode workers.
    spawn_expected: set[str] = set()
    fork_expected: set[str] = set()
    if kernel is not None and getattr(kernel, "guardian", None) is not None:
        modules = getattr(kernel.guardian, "_modules", {}) or {}
        for name in expected_workers:
            info = modules.get(name)
            if info is None:
                continue
            method = getattr(info.spec, "start_method", "fork")
            if method == "spawn":
                spawn_expected.add(name)
            else:
                fork_expected.add(name)

    # Collect ACKs — wait the longest per-layer timeout (L3=30s).
    # Drain BOTH HIBERNATE_ACK (fork-mode) and BUS_HANDOFF_ACK (spawn-mode).
    max_timeout = max(sp.HIBERNATE_ACK_TIMEOUT_BY_LAYER.values())
    acks = _drain_messages(
        inbox, {bus.HIBERNATE_ACK, bus.BUS_HANDOFF_ACK},
        timeout=max_timeout, kernel=kernel,
    )

    result.hibernate_acks = [a.get("payload", {}) for a in acks]
    hibernate_ackers: set[str] = set()
    handoff_ackers: set[str] = set()
    for a in acks:
        atype = a.get("type")
        src = a.get("src") or a.get("payload", {}).get("src")
        if not src:
            continue
        if atype == bus.HIBERNATE_ACK:
            hibernate_ackers.add(src)
        elif atype == bus.BUS_HANDOFF_ACK:
            handoff_ackers.add(src)
    fork_missing = fork_expected - hibernate_ackers
    spawn_missing = spawn_expected - handoff_ackers
    missing = fork_missing | spawn_missing  # combined for logging
    all_ackers = hibernate_ackers | handoff_ackers

    result.event(
        "hibernate_acks_collected",
        acked=sorted(all_ackers),
        hibernate_acked=sorted(hibernate_ackers),
        handoff_acked=sorted(handoff_ackers),
        missing=sorted(missing),
        fork_missing=sorted(fork_missing),
        spawn_missing=sorted(spawn_missing),
    )

    # Disable Guardian auto-restart so exited workers DON'T respawn while
    # shadow kernel is booting. Without this, original kernel's Guardian
    # would re-spawn memory_worker / etc., re-acquiring DuckDB locks
    # before the shadow kernel can claim them. (Discovered B.1 deploy
    # test 6: shadow's memory_worker DuckDB IOException due to lock held
    # by an original-kernel-Guardian-respawned worker.)
    if kernel is not None and getattr(kernel, "guardian", None) is not None:
        try:
            # Microkernel v2 Phase B fast-hibernate (2026-04-27): use
            # Guardian.pause() instead of stop_all(save_first=True). pause()
            # just sets _stop_requested=True (mute monitor_tick) without
            # iterating modules + waiting 30s per module for SAVE_DONE
            # that already-dead workers can't send. Saves ~8.5 minutes
            # per swap. Workers that exited via HIBERNATE stay exited;
            # those still running keep running. Symmetric resume() in
            # rollback paths via _revive_guardian_after_unwind.
            kernel.guardian.pause()
            result.event("guardian_paused",
                         note="monitor_tick muted; no auto-restart during shadow boot")
        except Exception as e:
            logger.warning("[shadow_swap] guardian.pause failed: %s", e)

        # M1 (2026-04-27 PM audit): straggler_kill ONLY fork-mode workers
        # that didn't ack HIBERNATE. Spawn-mode workers in B.2.1 swap_pending
        # state are alive and awaiting adoption — killing them defeats the
        # whole point of graduation. spawn_missing means a spawn-mode worker
        # didn't ack BUS_HANDOFF either; that's a different failure (e.g.,
        # worker crashed pre-swap or wiring broken) and we just log it.
        if fork_missing:
            killed = []
            for name in sorted(fork_missing):
                try:
                    if kernel.guardian.fast_kill(name):
                        killed.append(name)
                except Exception as e:
                    logger.warning(
                        "[shadow_swap] fast_kill('%s') failed: %s", name, e)
            if killed:
                result.event("stragglers_killed",
                             modules=killed,
                             mode="fork",
                             note="post-HIBERNATE fork-mode non-ackers SIGTERM'd to release locks")
        if spawn_missing:
            result.event(
                "spawn_handoff_ack_missing",
                modules=sorted(spawn_missing),
                note="spawn-mode workers expected to ack BUS_HANDOFF — adoption may fail "
                     "for these; not killing (they may still be alive)",
            )

    # Tolerate missing acks if it's a small minority — fork-mode workers
    # were SIGKILL'd above, spawn-mode missing will surface in adoption_wait.
    # Rollback only if MORE THAN HALF of ALL workers failed to ack their
    # respective protocol message.
    if len(missing) > len(expected_workers) // 2:
        result.outcome = "rollback"
        result.failure_reason = f"too_many_workers_missing_ack:{sorted(missing)}"
        bus_obj.publish(make_msg(
            bus.HIBERNATE_CANCEL,
            src="shadow_swap", dst="all",
            payload={"event_id": result.event_id, "reason": "ack_timeout"},
        ))
        return False
    return True


# ── Phase 3: Shadow boot ────────────────────────────────────────────

def _wait_for_health(
    port: int, timeout: float,
    criteria: Optional[HealthCriteria] = None,
) -> tuple[bool, dict]:
    """Poll multi-criterion health gate until pass or timeout.

    Returns (passed, last_diagnosis). Diagnosis is the full dict from
    `_check_multi_criterion_health()` (and smoke if enabled), capturing
    every per-criterion pass/fail with observed values. Even on failure
    the diagnosis is non-empty — useful for orchestrator audit log.

    Backward-compat: if `criteria=None`, uses default HealthCriteria
    (which is the strict production gate). Tests pass a relaxed criteria
    instance.
    """
    if criteria is None:
        criteria = HealthCriteria()

    deadline = time.monotonic() + timeout
    last_diag: dict = {"never_polled": True}

    while time.monotonic() < deadline:
        passed, last_diag = _check_multi_criterion_health(port, criteria)
        if passed:
            # Optional smoke step — last gate before declaring healthy.
            if criteria.smoke_test_enabled:
                smoke_ok, smoke_diag = _check_smoke_advancing(port, criteria)
                last_diag["checks"]["smoke_advancing"] = smoke_diag
                if smoke_ok:
                    return True, last_diag
                # Smoke failed → don't return success; keep polling within
                # the outer timeout. The kernel may need a few more seconds
                # for dynamic state to start advancing.
            else:
                return True, last_diag
        time.sleep(1.0)

    return False, last_diag


# ── Phase 3.6 (Microkernel v2 B.2.1) — Adoption wait ────────────────────


def _phase_b2_1_wait_adoption(
    kernel,
    expected_workers: list[str],
    shadow_port: int,
    result: SwapResult,
    *,
    timeout: float = 15.0,
) -> bool:
    """Wait for spawn-mode workers to be adopted by shadow's Guardian.

    Polls shadow's /v4/state.guardian periodically; returns True when every
    spawn-mode worker in `expected_workers` reports adopted=True on the
    shadow side. Returns False on timeout (caller unwinds via
    _unwind_b2_1_handoff).

    Only meaningful when there ARE spawn-mode workers in the live ModuleSpec
    set with the swap_handler wired. With no such workers (today's default),
    the function trivially returns True after one poll (empty expected_set).

    Args:
        kernel:           old kernel — used to read ModuleSpec start_method
                          per worker (today's specs).
        expected_workers: ModuleSpec names that should adopt; the function
                          filters to spawn-mode internally.
        shadow_port:      shadow's /v4/state listen port.
        result:           SwapResult for event logging.
        timeout:          max seconds to wait; default 15s (~3× shadow boot).
    """
    result.phase = "b2_1_wait_adoption"
    # Filter to spawn-mode workers — fork-mode workers won't adopt; they
    # take the improved-B.1 path (die with old, respawn fresh on shadow).
    #
    # M4 (2026-04-27 PM, post-M1+M2+M3+M1.5 swap test): also exclude
    # workers that aren't actually running on the OLD kernel side. Workers
    # with autostart=False+lazy=True (e.g. rl, only spawned when ARC
    # needs it) will be in ModuleSpec registry as start_method='spawn' but
    # have state=STOPPED + pid=None. They can't be adopted because they
    # were never alive to ack the handoff. Filter requires info.pid is not
    # None (worker has been spawned at least once and is alive or recently
    # alive). Without this filter, adoption_wait waits the full 30s for
    # a worker that will never adopt because it doesn't exist.
    spawn_expected: set[str] = set()
    spawn_critical: set[str] = set()
    spawn_optional: set[str] = set()
    spawn_skipped_not_running: list[str] = []
    if kernel is not None and getattr(kernel, "guardian", None) is not None:
        modules = getattr(kernel.guardian, "_modules", {}) or {}
        for name in expected_workers:
            info = modules.get(name)
            if info is None:
                continue
            if getattr(info.spec, "start_method", "fork") != "spawn":
                continue
            if info.pid is None:
                spawn_skipped_not_running.append(name)
                continue
            spawn_expected.add(name)
            # M5 (2026-04-27 PM): split by criticality for tolerance
            if getattr(info.spec, "b2_1_swap_critical", True):
                spawn_critical.add(name)
            else:
                spawn_optional.add(name)
    if spawn_skipped_not_running:
        result.event(
            "b2_1_skipped_not_running_spawn_workers",
            modules=sorted(spawn_skipped_not_running),
            note="spawn-mode workers with pid=None (autostart=False or never started) "
                 "excluded from adoption expectation — nothing to adopt",
        )
    if not spawn_expected:
        result.event(
            "b2_1_no_spawn_mode_workers",
            note="adoption phase no-op (no spawn-mode workers in ModuleSpec set)",
        )
        return True
    result.event(
        "b2_1_adoption_classification",
        critical=sorted(spawn_critical),
        optional=sorted(spawn_optional),
        note="M5: swap declares success when all CRITICAL spawn workers adopt; "
             "OPTIONAL workers nice-to-adopt — orphans self-SIGTERM via "
             "supervision daemon bus-as-supervision check (~30s)",
    )

    deadline = time.time() + timeout
    last_seen: set[str] = set()
    while time.time() < deadline:
        state = _fetch_state_json(shadow_port)
        if state is not None:
            guardian = _get_dotted(state, "data.guardian") or {}
            if isinstance(guardian, dict):
                last_seen = {
                    name for name, info in guardian.items()
                    if isinstance(info, dict) and info.get("adopted") is True
                }
                # M5 (2026-04-27 PM): success = all CRITICAL adopted (not all expected)
                if spawn_critical.issubset(last_seen):
                    optional_missing = sorted(spawn_optional - last_seen)
                    if optional_missing:
                        result.event(
                            "b2_1_optional_workers_orphan",
                            modules=optional_missing,
                            note="non-critical spawn workers didn't adopt — they will "
                                 "self-SIGTERM via bus-as-supervision; shadow Guardian "
                                 "respawns them fresh post-swap",
                        )
                    result.event(
                        "b2_1_adoption_acks_collected",
                        adopted=sorted(last_seen),
                        expected=sorted(spawn_expected),
                        critical_adopted=sorted(spawn_critical & last_seen),
                        optional_adopted=sorted(spawn_optional & last_seen),
                    )
                    return True
        time.sleep(0.2)

    result.event(
        "b2_1_adoption_timeout",
        timeout_s=timeout,
        adopted=sorted(last_seen),
        expected=sorted(spawn_expected),
        missing=sorted(spawn_expected - last_seen),
        critical_missing=sorted(spawn_critical - last_seen),
        optional_missing=sorted(spawn_optional - last_seen),
        note="M5: failure means at least one CRITICAL worker didn't adopt",
    )
    return False


def _revive_guardian_after_unwind(kernel, result: SwapResult) -> None:
    """Phase B.1 unwind-path bug fix (2026-04-27): re-enable Guardian after
    a swap rollback so workers that exited via HIBERNATE respawn cleanly.

    The original _phase_hibernate calls kernel.guardian.stop_all() to prevent
    auto-restart while shadow boots. stop_all sets _stop_requested=True
    which permanently mutes monitor_tick. Without a symmetric resume(),
    a rollback leaves Titan in a broken state: workers gone, Guardian
    silent, only the kernel + api_subprocess processes remain.

    This helper:
      1. Calls guardian.resume() to flip the kill switch
      2. Calls guardian.start_all() to respawn autostart=True modules
         (workers exited via HIBERNATE-then-stop, so state==STOPPED)

    No-op when kernel.guardian is None (legacy mode without Guardian).

    Discovered during T1 first-flag-flip swap test 2026-04-27 ~13:25 UTC:
    locks_not_released → shadow_boot_failed → unwind without revive →
    T1 down for ~5min until --force restart restored it.
    """
    if kernel is None or getattr(kernel, "guardian", None) is None:
        return
    g = kernel.guardian
    # Phase A retrofit hot-fix #2 (2026-04-27): the proxy ↔ swap interlock
    # in Guardian.start() blocks any caller while is_shadow_swap_active().
    # That's correct for EXTERNAL callers (proxy lazy-starts), but the
    # orchestrator's OWN start_all in revive would deadlock waiting on the
    # done_event that won't be set until orchestrate_shadow_swap returns
    # AFTER revive completes. Without this clear-before-revive, each
    # start() blocks 60s waiting for swap completion → 17 modules = up to
    # 17 minutes of recovery time after a rollback.
    #
    # Clear the active flag + set the done event BEFORE start_all so the
    # interlock sees no swap and proceeds immediately. _run_swap.finally
    # is idempotent (checks `if active == event_id` so double-clear is fine).
    try:
        with kernel._shadow_swap_lock:
            kernel._shadow_swap_active = None
        kernel._shadow_swap_done_event.set()
    except AttributeError:
        # Legacy kernel without swap-completion machinery — proceed anyway
        pass
    try:
        g.resume()
        g.start_all()
        result.event("guardian_revived",
                     note="resume + start_all after unwind")
    except Exception as e:  # noqa: BLE001
        result.event("guardian_revive_failed", error=str(e),
                     severity="CRITICAL — Titan may be dark; manual restart needed")
        logger.exception("[shadow_swap] guardian revive failed: %s", e)


def _unwind_b2_1_handoff(
    kernel, bus_obj, result: SwapResult, *, reason: str,
) -> None:
    """Publish BUS_HANDOFF_CANCELED so spawn-mode workers re-arm PDEATHSIG.

    Symmetric with HIBERNATE_CANCEL (which fork-mode workers respond to via
    readiness_reporter). Called from rollback paths after a failed B.2.1
    swap step; spawn-mode workers' on_bus_handoff_canceled handler (C1)
    re-arms PDEATHSIG and restores strict watcher semantics.

    No-op if bus has no socket broker (no spawn-mode handler is listening
    on the new bus message types — workers in legacy mp.Queue mode rely on
    HIBERNATE_CANCEL which the existing rollback paths already publish).
    """
    if not getattr(bus_obj, "has_socket_broker", False):
        return
    bus_obj.publish(make_msg(
        bus.BUS_HANDOFF_CANCELED,
        src="kernel", dst="all",
        payload={"event_id": result.event_id, "reason": reason},
    ))
    result.event("b2_1_handoff_canceled", reason=reason)


def _phase_shadow_boot(
    kernel,
    result: SwapResult,
    *,
    snapshot_path: str,
    b2_1_active: bool = False,
) -> Optional[subprocess.Popen]:
    """Spawn shadow kernel; return the Popen handle on success, None on fail.

    `b2_1_active=True` (Phase B.2.1, 2026-04-27 PM) tells shadow's Guardian
    to skip-start spawn-mode workers — they'll be ADOPTED via
    BUS_WORKER_ADOPT_REQUEST instead of respawned, which is the load-bearing
    promise of "workers literally outlive a kernel swap". Two effects:
      1. Sets env TITAN_B2_1_ADOPTION_PENDING=1 in the shadow subprocess.
         Guardian.start_all() reads this and skips spawn_method='spawn' modules.
      2. Relaxes the `min_modules_running` health-gate threshold: the 9
         graduated workers won't appear in shadow's Guardian until adoption,
         so insisting on 14 fails before adoption can run. Adoption-phase
         coverage handles the rest. The strict gate is the post-adoption
         broker subscriber count check (added next).
    """
    result.phase = "shadow_boot"

    current_port = read_active_port()
    shadow_port = pick_shadow_port(current_port)
    result.shadow_port = shadow_port

    # Build command. We invoke titan_main.py directly via the same Python.
    project_root = Path(__file__).resolve().parents[2]
    cmd = [
        "python", "-u",
        str(project_root / "scripts" / "titan_main.py"),
        "--server",
        "--shadow-port", str(shadow_port),
        "--restore-from", snapshot_path,
    ]

    log_path = Path("/tmp") / f"titan_shadow_{shadow_port}.log"
    log_fp = log_path.open("a")
    log_fp.write(f"\n=== Shadow boot at {time.time()} (event_id={result.event_id}) ===\n")
    log_fp.flush()

    # BUG-B1-SHARED-LOCKS fix: per-shadow data dir.
    # Hardlink-copy data/ → data_shadow_<port>/ so shadow has its own
    # data files. DuckDB/SQLite/FAISS break the hardlink on first write
    # (copy-on-write semantics) → each kernel gets its own inode → no
    # lock contention regardless of whether original kernel is alive.
    # Set TITAN_DATA_DIR env so workers honor the shadow path.
    #
    # 2026-04-27 hot-fix (post-swap-#4): REMOVED lock_polling check.
    # It was defense-in-depth from B.1 first-deploy when this code path
    # was new + paranoid — but per-shadow data_dir + COW makes the
    # check architecturally redundant. Worse, lock_polling created a
    # false-failure mode: original-kernel children (api_subprocess,
    # IMW workers) holding inner_memory.db handles → poll fails → swap
    # rolls back even though shadow would have booted fine on its own
    # data_dir copy. Every swap on T1 today (#1, #3, #4) failed here.
    # The hardlinks-break-on-write semantics is the actual fix.
    from titan_plugin.core import shadow_data_dir as sdd
    shadow_data = sdd.shadow_data_dir_for_port(shadow_port, root=project_root)
    sdd.cleanup_shadow_dir(shadow_data)  # ensure fresh state on retry

    # Phase B.3 Layer 3 (2026-04-27 PM): break leftover hardlinks on
    # canonical data/ DBs BEFORE cp -al. Without this, data/<db> and
    # data.OLD.<prev_ts>/<db> share inodes from prior swaps; the upcoming
    # cp -al would extend the chain to data_shadow_<port>/<db> too.
    # Breaking pre-cp ensures shadow's data dir starts as the ONLY new
    # link (1) and data.OLD chain stays decoupled from anything live.
    canonical_data = project_root / "data"
    canonical_broken = sdd.break_canonical_db_hardlinks(canonical_data)
    if canonical_broken:
        result.event(
            "canonical_db_hardlinks_broken",
            count=canonical_broken,
            note="pre-cp Layer 3: decoupled canonical DBs from data.OLD inode chain")

    copied, method = sdd.copy_data_dir(project_root / "data", shadow_data)
    if not copied:
        result.outcome = "rollback"
        result.failure_reason = f"shadow_data_dir_copy_failed:{method}"
        result.event("shadow_data_copy_failed", reason=method)
        return None
    result.event("shadow_data_copy_ok",
                 shadow_dir=str(shadow_data), method=method)

    env = os.environ.copy()
    env["OPENROUTER_API_KEY"] = ""
    env["TITAN_API_PORT"] = str(shadow_port)
    env["TITAN_DATA_DIR"] = str(shadow_data)
    if b2_1_active:
        # Tell shadow's Guardian.start_all() to skip spawn-mode workers
        # — they'll be adopted via BUS_WORKER_ADOPT_REQUEST instead.
        env["TITAN_B2_1_ADOPTION_PENDING"] = "1"

    try:
        proc = subprocess.Popen(
            cmd, stdout=log_fp, stderr=subprocess.STDOUT,
            cwd=str(project_root), env=env,
            start_new_session=True,
        )
    except Exception as e:
        result.outcome = "rollback"
        result.failure_reason = f"shadow_spawn_failed:{e}"
        sdd.cleanup_shadow_dir(shadow_data)
        return None

    result.event("shadow_spawned", pid=proc.pid, port=shadow_port,
                 log=str(log_path))

    # Wait for multi-criterion health (BUG-B1-WEAK-HEALTH-CHECK fix).
    # Default HealthCriteria enforces module roster + critical workers
    # running + no crash-loops + fresh heartbeats + dynamic-state advance.
    #
    # Phase B.2.1 (2026-04-27 PM): when b2_1_active, the 9 graduated
    # spawn-mode workers won't appear in shadow's Guardian during shadow
    # boot — they get adopted in the next phase. Drop min_modules_running
    # to a threshold the fork-mode + always-respawn set (api_subprocess,
    # imw, observatory_writer, spirit, memory, timechain, rl, emot_cgn,
    # backup) can satisfy. Backup is graduated but uses its own pre-existing
    # spawn flag (S6 reference) — also outlives, also excluded from the
    # threshold. Adoption phase covers the rest.
    # B.2.1 mode: graduated workers don't appear in shadow's Guardian
    # until the adoption phase runs (post-shadow-boot). Memory + timechain
    # + emot_cgn + rl are now graduated (outlive the swap, retain
    # FAISS/DuckDB/state in-process), so they're excluded from this gate
    # too — they show up in the adoption_wait phase, not here.
    # Critical set in B.2.1: spirit (consciousness loop, fork-mode), imw
    # (single-writer for inner_memory.db), api (the /health endpoint host).
    #
    # 2026-04-28 PM late-3: when api_process_separation_enabled=false, api
    # runs in-process inside the kernel (not as a Guardian module) — the
    # check `running_modules.get("api").state == "running"` always fails
    # because Guardian has no "api" entry. The /health endpoint already
    # validates api liveness via HTTP (health_endpoint check), so it's safe
    # to drop "api" from critical_modules in legacy in-process mode.
    api_separated = (
        kernel._config.get("microkernel", {})
        .get("api_process_separation_enabled", False)
    )
    b2_1_critical = ("spirit", "imw", "api") if api_separated else ("spirit", "imw")
    health_criteria = (
        HealthCriteria(
            min_modules_running=3,
            critical_modules=b2_1_critical,
        )
        if b2_1_active else
        None  # default = strict pre-B.2.1 gate
    )
    passed, diag = _wait_for_health(
        shadow_port, sp.SHADOW_BOOT_TIMEOUT, criteria=health_criteria,
    )
    if not passed:
        result.outcome = "rollback"
        result.failure_reason = "shadow_health_multi_criterion_failed"
        result.event("shadow_health_failed", diagnosis=diag)
        _kill_shadow_process_tree(proc)
        return None

    result.event("shadow_health_ok", port=shadow_port, diagnosis=diag)
    return proc


# ── Phase 4: Nginx swap ─────────────────────────────────────────────

def _phase_nginx_swap(
    result: SwapResult,
    *,
    old_port: int,
    new_port: int,
) -> bool:
    """Update nginx upstream + reload. Best-effort: failure surfaces but
    doesn't auto-revert (orchestrator caller decides rollback)."""
    result.phase = "nginx_swap"

    project_root = Path(__file__).resolve().parents[2]
    helper = project_root / "scripts" / "nginx_swap_upstream.sh"
    if not helper.exists():
        # Helper not yet shipped (B.1 §9 bundle). Soft-pass: log + continue.
        # Production deploy MUST have the helper for actual nginx swap.
        result.event("nginx_helper_missing", helper=str(helper))
        write_active_port(new_port)  # reflect the intended state anyway
        return True

    try:
        # Invoke the script directly (it has #!/bin/bash shebang).
        # Calling "sudo bash <script>" would require NOPASSWD on /bin/bash
        # which is too permissive; "sudo <script>" matches the sudoers
        # entry that allows just this specific helper.
        completed = subprocess.run(
            ["sudo", "-n", str(helper), str(old_port), str(new_port)],
            capture_output=True, text=True, timeout=30,
        )
    except Exception as e:
        result.outcome = "rollback"
        result.failure_reason = f"nginx_helper_exception:{e}"
        return False

    if completed.returncode != 0:
        result.outcome = "rollback"
        result.failure_reason = f"nginx_helper_rc={completed.returncode}:{completed.stderr[:200]}"
        return False

    write_active_port(new_port)
    result.event("nginx_swapped", old=old_port, new=new_port)
    return True


# ── Phase 5: Shutdown old kernel ────────────────────────────────────

def _phase_shutdown_old(
    kernel,
    result: SwapResult,
) -> None:
    """SIGTERM old kernel + workers; SIGKILL stragglers after 30s.

    Note: this method runs INSIDE the old kernel process. We can't
    SIGTERM ourselves without abrupt termination; we publish
    MODULE_SHUTDOWN bus events for workers + let the OS reap us
    when our parent (titan_main.py) catches the new kernel's
    SYSTEM_RESUMED signal.

    For B.1, we send MODULE_SHUTDOWN to all guardian-managed modules
    (workers will exit) + log an instruction for the parent script
    (or external supervisor) to SIGTERM us via PID file.
    """
    result.phase = "shutdown_old"
    if kernel is not None and getattr(kernel, "guardian", None) is not None:
        try:
            kernel.guardian.stop_all(reason="shadow_swap_complete")
        except Exception as e:
            result.event("shutdown_workers_partial", error=str(e))

    # Don't actually os._exit ourselves — let the host script
    # (or systemd supervisor) handle final exit. Just signal complete.
    result.event("shutdown_signaled")


# ── Top-level orchestrator ──────────────────────────────────────────

def orchestrate_shadow_swap(
    kernel,
    *,
    reason: str = "manual",
    grace: float = sp.READINESS_GRACE_SECONDS,
    event_id: str = "",
    b2_1_forced: bool = False,
    b2_1_adoption_timeout_s: float = 30.0,
) -> dict[str, Any]:
    """Run the full B.1 shadow swap protocol.

    Designed to run in a BACKGROUND THREAD spawned by
    kernel.shadow_swap_orchestrate. Returns the final result dict; the
    background-thread wrapper stores it in kernel._shadow_swap_history.

    Updates kernel._shadow_swap_progress[event_id] at every phase
    transition so /maker/upgrade-status can report live state to the
    polling CLI.

    Always appends to data/shadow_swap_history.jsonl audit log.
    """
    if not event_id:
        event_id = sp.new_event_id()
    result = SwapResult(event_id=event_id, reason=reason)

    # Live-progress callback — kernel updates the shared dict so
    # /maker/upgrade-status reads can see in-flight state.
    def _publish_progress():
        try:
            with getattr(kernel, "_shadow_swap_lock", threading.Lock()):
                if hasattr(kernel, "_shadow_swap_progress"):
                    kernel._shadow_swap_progress[event_id] = result.to_dict()
        except Exception:
            pass

    # Wrap result.event so each phase-emit also publishes progress.
    _orig_event = result.event
    def _event_with_progress(msg, **kw):
        _orig_event(msg, **kw)
        _publish_progress()
    result.event = _event_with_progress  # type: ignore[method-assign]
    _publish_progress()  # initial preflight state

    bus_obj = kernel.bus
    inbox = _subscribe_temp(bus_obj, name="shadow_swap")

    # Phase C C-S2: write services_watchdog coordination lock so the
    # 5-min cron skips its duplicate-titan_main kill heuristic during
    # the swap window. Heartbeat thread refreshes the lock every 10s
    # to extend the window if the swap takes longer than the default
    # TTL. Per PLAN_microkernel_phase_c_s2_kernel.md §17.1 +
    # BUG-SERVICES-WATCHDOG-SHADOW-SWAP-RACE-20260428.
    swap_lock_path = write_swap_lock(_kernel_titan_id(kernel), event_id)
    swap_lock_hb = _SwapLockHeartbeat(swap_lock_path)
    swap_lock_hb.start()

    try:
        # Phase 0 — preflight
        result.event("preflight_begin")
        bus_obj.publish(make_msg(
            bus.SYSTEM_UPGRADE_QUEUED,
            src="shadow_swap", dst="all",
            payload={"event_id": event_id, "reason": reason},
        ))

        # Snapshot the kernel runtime (paired with hibernate_runtime call)
        # — we hibernate AFTER readiness wait, so the snapshot is fresh
        # at that point. But we collect expected workers now.
        expected_workers: list[str] = []
        if kernel.guardian is not None:
            expected_workers = sorted(getattr(kernel.guardian, "_modules", {}).keys())
        result.event("preflight_done", expected_workers=len(expected_workers))

        # Phase 1 — readiness wait
        ready = _phase_readiness_wait(bus_obj, inbox, result,
                                      grace=grace, kernel=kernel)
        if not ready:
            return result.to_dict()  # deferred — outcome already set

        # Phase 2 — hibernate
        snapshot_path = kernel.hibernate_runtime(event_id)
        ok = _phase_hibernate(bus_obj, inbox, result,
                              expected_workers=expected_workers,
                              kernel=kernel)
        if not ok:
            # Hot-fix #3 (2026-04-27 afternoon): _phase_hibernate's internal
            # rollback path (too_many_workers_missing_ack) was missing the
            # revive call — left Guardian paused, workers never respawned.
            # Symmetric to the 4 rollback branches in earlier fix (5970949e).
            _revive_guardian_after_unwind(kernel, result)
            return result.to_dict()  # rollback — outcome already set

        # Phase 3 — shadow boot
        # Compute B.2.1 active state BEFORE shadow_boot — used to (a) tell
        # shadow's Guardian to skip-spawn graduated workers via env var, and
        # (b) relax the min_modules_running health threshold since those
        # workers won't appear in shadow's Guardian until adoption.
        b2_1_active = getattr(bus_obj, "has_socket_broker", False) or b2_1_forced
        shadow_proc = _phase_shadow_boot(
            kernel, result, snapshot_path=snapshot_path,
            b2_1_active=b2_1_active,
        )
        if shadow_proc is None:
            # Rollback — restore old kernel
            bus_obj.publish(make_msg(
                bus.HIBERNATE_CANCEL,
                src="shadow_swap", dst="all",
                payload={"event_id": event_id, "reason": "shadow_boot_failed"},
            ))
            _unwind_b2_1_handoff(kernel, bus_obj, result, reason="shadow_boot_failed")
            _revive_guardian_after_unwind(kernel, result)
            return result.to_dict()

        # Phase 3.6 (Microkernel v2 B.2.1) — adoption wait. Only runs if the
        # kernel has a bus_socket broker attached (production flag-on path)
        # OR if b2_1_forced=True (CLI isolation testing). With no spawn-mode
        # workers wired, this is a no-op (returns True after one poll).
        # b2_1_active was computed above (before shadow boot).
        if b2_1_active:
            # Disconnect spawn-mode worker subscribers from the OLD kernel's
            # broker so they reconnect to shadow's broker (now bound on the
            # same /tmp/titan_bus_<id>.sock path). Without this, workers'
            # existing FD connections to old kernel's broker persist
            # (TCP-like) and they never reach shadow → adoption_wait times
            # out with adopted=[]. Workers' BusSocketClient retries on EOF
            # → connect() succeeds against shadow → SwapHandlerState's
            # supervision daemon observes reconnect_count++ → calls
            # request_adoption() → shadow Guardian.adopt_worker registers it.
            old_broker = getattr(bus_obj, "_broker", None)
            if old_broker is not None and hasattr(old_broker, "disconnect_subscribers"):
                spawn_names = [
                    name for name in expected_workers
                    if (info := getattr(kernel.guardian, "_modules", {}).get(name))
                    and getattr(info.spec, "start_method", "fork") == "spawn"
                ]
                if spawn_names:
                    purged = old_broker.disconnect_subscribers(
                        spawn_names, reason="b2_1_handoff",
                    )
                    result.event(
                        "b2_1_workers_disconnected",
                        purged=sorted(purged),
                        target=sorted(spawn_names),
                    )

            adopted_ok = _phase_b2_1_wait_adoption(
                kernel, expected_workers, result.shadow_port, result,
                timeout=b2_1_adoption_timeout_s,
            )
            if not adopted_ok:
                # Adoption failed — kill shadow + unwind so old kernel resumes
                # (with spawn-mode workers re-arming PDEATHSIG via _unwind helper).
                result.outcome = "rollback"
                result.failure_reason = "b2_1_adoption_timeout"
                _kill_shadow_process_tree(shadow_proc)
                bus_obj.publish(make_msg(
                    bus.HIBERNATE_CANCEL,
                    src="shadow_swap", dst="all",
                    payload={"event_id": event_id,
                             "reason": "b2_1_adoption_timeout"},
                ))
                _unwind_b2_1_handoff(kernel, bus_obj, result,
                                     reason="b2_1_adoption_timeout")
                _revive_guardian_after_unwind(kernel, result)
                return result.to_dict()

        # Phase 3.5 — pre-nginx-swap re-check (BUG-B1-WEAK-HEALTH-CHECK).
        # Phase 3 confirmed healthy at boot; but a kernel can degrade between
        # boot and nginx swap (worker crash-loop start, memory pressure, etc.).
        # Re-run the multi-criterion gate as the LAST gate before user-visible
        # traffic redirects.
        #
        # M5.1 (2026-04-27 PM, T2 swap): in B.2.1 mode the adoption phase
        # adds ~30s of "shadow alive but kernel busy" time during which the
        # api_subprocess's MODULE_HEARTBEAT can stale (api proves liveness
        # via /health endpoint, not a periodic heartbeat). The strict 30s
        # heartbeat threshold misclassified this as degradation. Use the
        # same relaxed criteria as shadow_boot for B.2.1 mode + bump
        # max_heartbeat_age to 120s to absorb the adoption-phase quiescence.
        result.phase = "pre_swap_recheck"
        # 2026-04-28 PM late-3: same conditional as _phase_shadow_boot —
        # when api_process_separation_enabled=false api is in-process inside
        # kernel, so it never appears in Guardian's module roster.
        api_separated_recheck = (
            kernel._config.get("microkernel", {})
            .get("api_process_separation_enabled", False)
        )
        recheck_critical = (
            ("spirit", "imw", "api") if api_separated_recheck else ("spirit", "imw")
        )
        recheck_criteria = (
            HealthCriteria(
                min_modules_running=3,
                critical_modules=recheck_critical,
                max_heartbeat_age_s=120.0,
                smoke_test_enabled=False,  # already verified at shadow_boot
            )
            if b2_1_active else
            HealthCriteria()
        )
        recheck_ok, recheck_diag = _check_multi_criterion_health(
            result.shadow_port, recheck_criteria,
        )
        if not recheck_ok:
            result.outcome = "rollback"
            result.failure_reason = "shadow_degraded_before_nginx_swap"
            result.event("pre_swap_recheck_failed", diagnosis=recheck_diag)
            _kill_shadow_process_tree(shadow_proc)
            bus_obj.publish(make_msg(
                bus.HIBERNATE_CANCEL,
                src="shadow_swap", dst="all",
                payload={"event_id": event_id,
                         "reason": "pre_swap_recheck_failed"},
            ))
            _unwind_b2_1_handoff(kernel, bus_obj, result,
                                 reason="pre_swap_recheck_failed")
            _revive_guardian_after_unwind(kernel, result)
            return result.to_dict()
        result.event("pre_swap_recheck_ok", diagnosis=recheck_diag)

        # Phase 4 — nginx swap
        old_port = read_active_port()
        nginx_ok = _phase_nginx_swap(result,
                                     old_port=old_port, new_port=result.shadow_port)
        if not nginx_ok:
            # Kill shadow + rollback
            _kill_shadow_process_tree(shadow_proc)
            bus_obj.publish(make_msg(
                bus.HIBERNATE_CANCEL,
                src="shadow_swap", dst="all",
                payload={"event_id": event_id, "reason": "nginx_swap_failed"},
            ))
            _unwind_b2_1_handoff(kernel, bus_obj, result,
                                 reason="nginx_swap_failed")
            _revive_guardian_after_unwind(kernel, result)
            return result.to_dict()

        # Phase 5 — shutdown old
        _phase_shutdown_old(kernel, result)

        # Phase 5b — promote shadow data dir to canonical (BUG-B1-SHARED-LOCKS).
        # Original kernel is gone (Phase 5); shadow is now the live kernel
        # reading from data_shadow_<port>/. Rotate filesystem state so future
        # boots see the new state at the canonical `data/` path. Old data/
        # is preserved as data.OLD.<ts>/ for one swap cycle before cleanup.
        # If this rename fails, the swap still SUCCEEDED (shadow is serving
        # traffic) — we just have a directory naming inconsistency that
        # ops can fix manually. We log it but don't roll back.
        try:
            from titan_plugin.core import shadow_data_dir as sdd
            project_root = Path(__file__).resolve().parents[2]
            shadow_dir = sdd.shadow_data_dir_for_port(
                result.shadow_port, root=project_root,
            )
            ok, msg = sdd.swap_data_dirs(project_root / "data", shadow_dir)
            if ok:
                result.event("data_dirs_promoted", message=msg)
                # Bound disk usage: prune older backups beyond 2 most-recent
                pruned = sdd.cleanup_old_backups(
                    project_root / "data", keep_count=2,
                )
                if pruned:
                    result.event("data_dir_backups_pruned", count=pruned)
            else:
                result.event("data_dirs_promotion_failed", reason=msg,
                             severity="warn — swap succeeded but ops should rotate dirs manually")
        except Exception as e:
            result.event("data_dirs_promotion_exception", error=str(e),
                         severity="warn — swap succeeded but dir promotion errored")

        result.gap_seconds = time.time() - result.started_at
        result.outcome = "ok"
        result.event("swap_complete", gap_seconds=result.gap_seconds)
        return result.to_dict()

    except Exception as e:
        logger.exception("[shadow_swap] orchestrator exception: %s", e)
        result.outcome = "error"
        result.failure_reason = f"orchestrator_exception:{e}"
        result.event("orchestrator_exception", error=str(e))
        # B.1 unwind-path bug fix: even on uncaught exceptions, attempt to
        # revive Guardian if hibernate already disabled it. Otherwise Titan
        # may be left dark with no auto-recovery path.
        try:
            _revive_guardian_after_unwind(kernel, result)
        except Exception as revive_err:  # noqa: BLE001
            logger.warning(
                "[shadow_swap] revive after exception failed: %s", revive_err)
        return result.to_dict()

    finally:
        # Cleanup shadow data dir on FAILURE (any non-ok outcome).
        # On success the dir was already promoted via swap_data_dirs above.
        if result.outcome != "ok" and result.shadow_port is not None:
            try:
                from titan_plugin.core import shadow_data_dir as sdd
                project_root = Path(__file__).resolve().parents[2]
                shadow_dir = sdd.shadow_data_dir_for_port(
                    result.shadow_port, root=project_root,
                )
                if sdd.cleanup_shadow_dir(shadow_dir):
                    result.event("shadow_data_cleanup_ok",
                                 shadow_dir=str(shadow_dir))
            except Exception as e:
                logger.warning("[shadow_swap] shadow_data cleanup failed: %s", e)

        try:
            append_audit(result)
        except Exception as e:
            logger.warning("[shadow_swap] audit append failed: %s", e)

        # Phase C C-S2: stop heartbeat thread + remove the
        # services_watchdog coordination lock. Always runs (success +
        # rollback + exception paths) so a stale lock never persists
        # past the swap window. Per PLAN §17.1.
        try:
            swap_lock_hb.stop(remove=True)
        except Exception as e:
            logger.warning("[shadow_swap] swap-lock cleanup failed: %s", e)
