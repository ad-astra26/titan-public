"""titan_hcl/modules/health_monitor_worker.py — pluggable L3 health-
monitor framework.

Per SPEC v1.12.0 §9.B `health_monitor_worker` block + D-SPEC-67 + rFP
`titan-docs/rFP_health_monitor_worker.md` (Maker greenlit 2026-05-17).

Architecture in one paragraph:
  - Boot: discover every HealthCheckPlugin subclass under
    `titan_hcl.health.*`, filter by `applies_on` against this Titan's
    role, instantiate each surviving plugin with its config dict.
  - Tick (1Hz internal sleep loop): for every plugin whose
    `next_fire_time` has elapsed, submit `plugin.check()` to a bounded
    ThreadPoolExecutor with `Future.result(timeout=HEALTH_CHECK_TIMEOUT_S)`
    enforcement. Each returned HealthResult is emitted as
    `HEALTH_CHECK_RESULT(dst="all")` and appended to the journal.
  - Heal: if `plugin.heal(result)` returns a non-None action AND the
    daily cap + cooldown gates pass, emit
    `HEAL_REQUEST(dst=plugin.owning_worker)` with a fresh correlation_id.
    Bus dispatcher correlates the matching HEAL_RESULT (60s timeout =
    failure). After receipt, emit `HEALTH_HEAL_ATTEMPT(dst="all")`. On
    daily-cap exhaustion OR `HEALTH_HEAL_CONSECUTIVE_FAILURE_THRESHOLD`
    consecutive failures, emit `HEALTH_HEAL_FAILED(dst="all")` P1
    (Maker Telegram alert via existing pattern).
  - State: persist `data/health_monitor/state.json` atomic-write per
    §11.H.2 every 60s + on shutdown. Append-only journal
    `data/health_monitor/events.jsonl` (50MB rotation).

SOLE-sanctioned heal path = bus → owning worker. health_monitor never
instantiates a second SocialXGateway or any other owning-worker class.
"""
from __future__ import annotations

import importlib
import inspect
import json
import logging
import os
import pkgutil
import sys
import tempfile
import threading
import time
import uuid
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutTimeout
from pathlib import Path
from queue import Empty
from typing import Any, Optional

from titan_hcl import bus
from titan_hcl.bus import (
    HEAL_REQUEST,
    HEAL_RESULT,
    HEALTH_CHECK_RESULT,
    HEALTH_HEAL_ATTEMPT,
    HEALTH_HEAL_FAILED,
    MODULE_HEARTBEAT,
    MODULE_SHUTDOWN,
    make_msg,
)
from titan_hcl.health import (
    HEALTH_CHECK_TIMEOUT_S,
    HEALTH_HEAL_CONSECUTIVE_FAILURE_THRESHOLD,
    HEALTH_HEAL_REPLY_TIMEOUT_S,
    HealthCheckPlugin,
    HealthResult,
)
from titan_hcl.core.module_error_handler import with_error_envelope
from titan_hcl.errors import Severity as _phase11_sev

logger = logging.getLogger("health_monitor_worker")

HEARTBEAT_INTERVAL_S = 30.0
TICK_INTERVAL_S = 1.0
STATE_PERSIST_INTERVAL_S = 60.0
RECV_POLL_TIMEOUT_S = 0.2
JOURNAL_ROTATION_BYTES = 50 * 1024 * 1024  # 50MB
DAILY_WINDOW_S = 24 * 3600.0

_STATE_DIR = "data/health_monitor"
_STATE_FILE = "state.json"
_JOURNAL_FILE = "events.jsonl"
_JOURNAL_ROTATED = "events.jsonl.1"


# ── Plugin discovery ─────────────────────────────────────────────────


def _discover_plugins(titan_id: str,
                       config: dict) -> list[HealthCheckPlugin]:
    """Import every `titan_hcl.health.*` submodule, find every
    concrete HealthCheckPlugin subclass, filter by applies_on, instantiate.

    Skips:
      - The package __init__ (the contract module).
      - Modules that fail to import (logged WARN; one bad plugin must
        not take out the worker).
      - Subclasses where applies_on excludes this Titan.

    Returns the live list of plugin instances ready to schedule."""
    import titan_hcl.health as health_pkg

    plugins: list[HealthCheckPlugin] = []
    canonical_poller = ((config or {}).get("social_x") or {}).get(
        "canonical_poller_titan_id", "T1")
    is_canonical_poller = (canonical_poller == ""
                           or canonical_poller == titan_id)
    mainnet_cluster = ((config or {}).get("solana") or {}).get(
        "cluster", "")
    is_mainnet = mainnet_cluster == "mainnet-beta"

    for finder, mod_name, ispkg in pkgutil.iter_modules(
            health_pkg.__path__):
        if ispkg:
            continue
        full_name = f"titan_hcl.health.{mod_name}"
        try:
            module = importlib.import_module(full_name)
        except Exception as e:
            logger.warning(
                "[HealthMonitor] plugin module %s failed to import: %s "
                "— skipping", full_name, e, exc_info=True)
            continue
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls is HealthCheckPlugin:
                continue
            if not issubclass(cls, HealthCheckPlugin):
                continue
            if inspect.isabstract(cls):
                continue
            # applies_on filter
            applies = getattr(cls, "applies_on", "all")
            if applies == "canonical_poller" and not is_canonical_poller:
                logger.info(
                    "[HealthMonitor] skip plugin %s — "
                    "applies_on=canonical_poller, this titan=%s "
                    "(canonical=%s)",
                    cls.name, titan_id, canonical_poller)
                continue
            if applies == "mainnet_only" and not is_mainnet:
                logger.info(
                    "[HealthMonitor] skip plugin %s — "
                    "applies_on=mainnet_only, cluster=%s",
                    cls.name, mainnet_cluster)
                continue
            # Per-plugin config dict — config.toml [health_monitor.<name>]
            # AND fall through to top-level [<owning_section>] for legacy
            # keys (social_x reads from [social_x] for db/key resolution).
            plugin_cfg = (((config or {}).get("health_monitor") or {})
                          .get(cls.name) or {})
            # Pass the FULL config too — plugins may need to resolve
            # owning-section keys (e.g. social_x reads [social_x]).
            merged_cfg = dict((config or {}).get(cls.name, {}))
            merged_cfg.update(plugin_cfg)
            # Special: social_x plugin needs the [social_x] section
            # for api_key/user_name/db_path resolution.
            if cls.name == "social_x":
                sx = (config or {}).get("social_x") or {}
                for k in ("api_key", "user_name", "db_path"):
                    if k in sx and k not in merged_cfg:
                        merged_cfg[k] = sx[k]
            try:
                instance = cls(merged_cfg)
            except Exception as e:
                logger.warning(
                    "[HealthMonitor] plugin %s init failed: %s — "
                    "skipping", cls.name, e, exc_info=True)
                continue
            plugins.append(instance)
            logger.info(
                "[HealthMonitor] loaded plugin: name=%s cadence_s=%.1f "
                "applies_on=%s owning_worker=%s",
                instance.name, instance.cadence_s,
                instance.applies_on, instance.owning_worker)
    return plugins


# ── Per-plugin runtime state ─────────────────────────────────────────


class _PluginRuntime:
    """Per-plugin runtime state — schedule + heal history + pending
    correlation IDs.

    NOT serialized verbatim; the persistor builds a lean dict from
    this for state.json.
    """

    def __init__(self, plugin: HealthCheckPlugin,
                  now: float, *,
                  next_fire_time: float | None = None,
                  heal_history_24h: list[dict] | None = None,
                  last_heal_at: float | None = None,
                  consecutive_failures: int = 0,
                  last_heal_failed_emit_ts: float | None = None,
                  last_result: dict | None = None) -> None:
        self.plugin = plugin
        self.next_fire_time: float = (
            next_fire_time if next_fire_time is not None else now)
        # Each entry: {ts: float, action: str, result: str ("success" |
        # "failed" | "timeout"), reason: str}
        self.heal_history_24h: deque[dict] = deque(
            heal_history_24h or [], maxlen=200)
        self.last_heal_at: float | None = last_heal_at
        self.consecutive_failures: int = consecutive_failures
        # last_heal_failed_emit_ts gates the once-per-24h cap-exhaustion
        # alert so we don't spam HEALTH_HEAL_FAILED every tick.
        self.last_heal_failed_emit_ts: float | None = (
            last_heal_failed_emit_ts)
        # AUDIT §C fix (rFP §P2): restore last_result on boot. It was written to
        # state.json (to_state_dict) but never passed back into the constructor
        # → permanently None until the next check completed, losing the persisted
        # last-known health status across respawn. to_dict() keys match the
        # HealthResult dataclass fields, so HealthResult(**saved) round-trips.
        self.last_result: HealthResult | None = None
        if isinstance(last_result, dict):
            try:
                self.last_result = HealthResult(**last_result)
            except (TypeError, ValueError):
                self.last_result = None
        # Maps correlation_id → {sent_ts, action, details} for in-flight
        # HEAL_REQUEST awaiting HEAL_RESULT.
        self.pending_heals: dict[str, dict] = {}

    def to_state_dict(self) -> dict:
        return {
            "next_fire_time": self.next_fire_time,
            "heal_history_24h": list(self.heal_history_24h),
            "last_heal_at": self.last_heal_at,
            "consecutive_failures": self.consecutive_failures,
            "last_heal_failed_emit_ts": self.last_heal_failed_emit_ts,
            "last_result": (self.last_result.to_dict()
                            if self.last_result else None),
        }

    def heal_attempts_in_window(self, now: float) -> int:
        cutoff = now - DAILY_WINDOW_S
        return sum(1 for h in self.heal_history_24h
                   if h.get("ts", 0) >= cutoff)

    def prune_heal_history(self, now: float) -> None:
        cutoff = now - DAILY_WINDOW_S
        # deque pop-left until first entry is in-window
        while (self.heal_history_24h
               and self.heal_history_24h[0].get("ts", 0) < cutoff):
            self.heal_history_24h.popleft()

    def cooldown_ok(self, now: float) -> bool:
        """Has the per-plugin cooldown elapsed since the last heal?

        Uses success-cooldown after the most recent success, failure-
        cooldown after the most recent failure/timeout. If no history,
        cooldown is satisfied."""
        if self.last_heal_at is None or not self.heal_history_24h:
            return True
        last = self.heal_history_24h[-1]
        elapsed = now - float(last.get("ts", 0))
        if last.get("result") == "success":
            return elapsed >= self.plugin.heal_cooldown_after_success_s
        return elapsed >= self.plugin.heal_cooldown_after_failure_s


# ── State persistence ────────────────────────────────────────────────


def _state_dir() -> Path:
    """Resolve absolute data/health_monitor/ path."""
    root = Path(__file__).resolve().parents[2]
    d = root / _STATE_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _atomic_write_json(path: Path, payload: dict) -> None:
    """§11.H.2-compliant atomic write: tempfile + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # tempfile + os.replace per §11.H.2
    with tempfile.NamedTemporaryFile(
            "w", dir=str(path.parent), prefix=path.name + ".",
            suffix=".tmp", delete=False) as tf:
        json.dump(payload, tf, indent=2, sort_keys=True)
        tf.flush()
        os.fsync(tf.fileno())
        tmp_name = tf.name
    os.replace(tmp_name, str(path))


def _load_state(state_path: Path) -> dict:
    if not state_path.exists():
        return {}
    try:
        with open(state_path, "r") as f:
            return json.load(f) or {}
    except Exception as e:
        logger.warning(
            "[HealthMonitor] state.json load failed: %s — starting fresh",
            e)
        return {}


def _save_state(state_path: Path,
                runtimes: dict[str, _PluginRuntime]) -> None:
    payload = {
        "plugins": {name: rt.to_state_dict()
                    for name, rt in runtimes.items()},
        "updated_at": time.time(),
    }
    try:
        _atomic_write_json(state_path, payload)
    except Exception as e:
        logger.warning(
            "[HealthMonitor] state.json save failed: %s", e, exc_info=True)


# ── Journal ──────────────────────────────────────────────────────────


def _append_journal(journal_path: Path, event_kind: str,
                     payload: dict) -> None:
    """Append-only JSONL writer. Rotates at JOURNAL_ROTATION_BYTES."""
    try:
        journal_path.parent.mkdir(parents=True, exist_ok=True)
        # Rotation check (cheap stat — only on the open path).
        if journal_path.exists():
            try:
                if journal_path.stat().st_size >= JOURNAL_ROTATION_BYTES:
                    rotated = journal_path.with_name(_JOURNAL_ROTATED)
                    if rotated.exists():
                        rotated.unlink()
                    journal_path.rename(rotated)
            except OSError:
                pass
        line = json.dumps({
            "kind": event_kind, "ts": time.time(), "payload": payload,
        }, separators=(",", ":"))
        with open(journal_path, "a") as f:
            f.write(line + "\n")
    except Exception as e:
        # Journal write failure must NEVER crash the worker; just log.
        logger.debug(
            "[HealthMonitor] journal append failed: %s", e)


# ── Bus helpers ──────────────────────────────────────────────────────


def _send(send_queue, msg_type: str, src: str, dst: str,
          payload: dict, rid: Optional[str] = None) -> None:
    """Non-blocking publish per §8.0.ter D-SPEC-48."""
    try:
        send_queue.put_nowait(make_msg(msg_type, src, dst, payload,
                                        rid=rid))
    except Exception as e:
        logger.debug(
            "[HealthMonitor] _send %s → %s failed: %s",
            msg_type, dst, e)


# Phase 11 §11.I.3/§11.I.5 — module-level readiness sentinel mirrored to
# the SHM slot (`module_health_monitor_state.bin`) via ModuleStateWriter so
# titan_hcl's 1Hz SHM poll + the orchestrator's MODULE_PROBE_REQUEST
# dispatcher see real liveness rather than a boot-time "subscribed-but-
# not-warm" lie. Flipped True only after plugin discovery + state load
# complete.
_WORKER_READY: bool = False


def _heartbeat_loop(send_queue, name: str,
                     stop_event: threading.Event,
                     state_writer: Optional[Any] = None) -> None:
    """Daemon thread — MODULE_HEARTBEAT every 30s.

    Phase 11 §11.I.5: also publishes ModuleStateWriter.heartbeat() on the
    SHM slot when `state_writer` is provided AND `_WORKER_READY` is True,
    so guardian_hcl's SHM-staleness detector + observatory /v6/readiness
    see fresh data on the same cadence as the legacy bus path. During the
    boot window the slot keeps state="starting"/"booted" instead of
    prematurely asserting state="running".
    """
    while not stop_event.is_set():
        _send(send_queue, MODULE_HEARTBEAT, name, "guardian",
              {"ts": time.time()})
        if state_writer is not None and _WORKER_READY:
            try:
                state_writer.heartbeat()
            except Exception:  # noqa: BLE001 — never crash the heartbeat
                pass
        stop_event.wait(HEARTBEAT_INTERVAL_S)


# ── Heal escalation logic ────────────────────────────────────────────


def _record_heal_outcome(rt: _PluginRuntime, *, action: str,
                          result: str, reason: str,
                          send_queue, name: str) -> None:
    """Record one heal outcome into the rolling history + emit
    HEALTH_HEAL_ATTEMPT + check escalation triggers.

    `result` ∈ {"success", "failed", "timeout"}.
    """
    now = time.time()
    rt.heal_history_24h.append({
        "ts": now, "action": action,
        "result": result, "reason": reason,
    })
    rt.last_heal_at = now
    if result == "success":
        rt.consecutive_failures = 0
    else:
        rt.consecutive_failures += 1

    attempts_today = rt.heal_attempts_in_window(now)
    _send(send_queue, HEALTH_HEAL_ATTEMPT, name, "all", {
        "plugin": rt.plugin.name,
        "action": action,
        "result": result,
        "attempt_n": attempts_today,
        "reason": reason,
        "ts": now,
    })

    # Escalation: 3 consecutive failures OR daily-cap exhaustion → P1
    should_escalate = False
    escalation_reason = ""
    if (rt.consecutive_failures
            >= HEALTH_HEAL_CONSECUTIVE_FAILURE_THRESHOLD):
        should_escalate = True
        escalation_reason = "consecutive_failures"
    elif attempts_today >= rt.plugin.max_heal_attempts_per_24h:
        # Only emit the cap-exhaustion alert once per 24h window.
        last_emit = rt.last_heal_failed_emit_ts or 0
        if now - last_emit >= DAILY_WINDOW_S:
            should_escalate = True
            escalation_reason = "daily_cap_exhausted"

    if should_escalate:
        rt.last_heal_failed_emit_ts = now
        _send(send_queue, HEALTH_HEAL_FAILED, name, "all", {
            "plugin": rt.plugin.name,
            "reason": escalation_reason,
            "attempts_today": attempts_today,
            "consecutive_failures": rt.consecutive_failures,
            "ts": now,
        })
        logger.warning(
            "[HealthMonitor] HEALTH_HEAL_FAILED plugin=%s reason=%s "
            "attempts_today=%d consecutive_failures=%d",
            rt.plugin.name, escalation_reason, attempts_today,
            rt.consecutive_failures)


def _maybe_emit_heal_request(rt: _PluginRuntime,
                              last_result: HealthResult,
                              send_queue, name: str,
                              journal_path: Path) -> bool:
    """If plugin's heal() returns a non-None action AND cap+cooldown
    gates pass, emit HEAL_REQUEST + track correlation_id. Returns True
    on emit, False on skip."""
    now = time.time()
    rt.prune_heal_history(now)

    # Daily cap check first — if we already emitted HEALTH_HEAL_FAILED
    # this window, silently skip further attempts.
    attempts_today = rt.heal_attempts_in_window(now)
    if attempts_today >= rt.plugin.max_heal_attempts_per_24h:
        # Re-emit HEALTH_HEAL_FAILED ONCE per window even on subsequent
        # would-be triggers — handled inside _record_heal_outcome via
        # the last_heal_failed_emit_ts gate.
        last_emit = rt.last_heal_failed_emit_ts or 0
        if now - last_emit >= DAILY_WINDOW_S:
            rt.last_heal_failed_emit_ts = now
            _send(send_queue, HEALTH_HEAL_FAILED, name, "all", {
                "plugin": rt.plugin.name,
                "reason": "daily_cap_exhausted",
                "attempts_today": attempts_today,
                "consecutive_failures": rt.consecutive_failures,
                "ts": now,
            })
        return False

    if not rt.cooldown_ok(now):
        return False

    try:
        action, details = rt.plugin.heal(last_result)
    except Exception as e:
        logger.warning(
            "[HealthMonitor] plugin %s heal() raised: %s — treating "
            "as no-heal", rt.plugin.name, e, exc_info=True)
        return False

    if not action:
        return False

    correlation_id = uuid.uuid4().hex
    rt.pending_heals[correlation_id] = {
        "sent_ts": now, "action": action, "details": dict(details),
    }
    _send(send_queue, HEAL_REQUEST, name, rt.plugin.owning_worker, {
        "plugin": rt.plugin.name,
        "action": action,
        "details": dict(details),
        "correlation_id": correlation_id,
        "ts": now,
    })
    _append_journal(journal_path, "heal_request", {
        "plugin": rt.plugin.name, "action": action,
        "correlation_id": correlation_id,
        "owning_worker": rt.plugin.owning_worker,
    })
    logger.info(
        "[HealthMonitor] HEAL_REQUEST plugin=%s action=%s → %s "
        "correlation_id=%s",
        rt.plugin.name, action, rt.plugin.owning_worker,
        correlation_id)
    return True


def _check_pending_heal_timeouts(
        runtimes: dict[str, _PluginRuntime],
        send_queue, name: str) -> None:
    """Sweep in-flight heal correlations; on >timeout, record as
    timeout outcome."""
    now = time.time()
    for rt in runtimes.values():
        if not rt.pending_heals:
            continue
        expired: list[str] = []
        for cid, meta in rt.pending_heals.items():
            if now - meta["sent_ts"] >= HEALTH_HEAL_REPLY_TIMEOUT_S:
                expired.append(cid)
        for cid in expired:
            meta = rt.pending_heals.pop(cid)
            _record_heal_outcome(
                rt, action=meta["action"], result="timeout",
                reason=(f"no HEAL_RESULT in "
                        f"{HEALTH_HEAL_REPLY_TIMEOUT_S:.0f}s"),
                send_queue=send_queue, name=name)


# ── Main loop ────────────────────────────────────────────────────────


@with_error_envelope(module_name="health_monitor", subsystem="entry", severity=_phase11_sev.FATAL)
def health_monitor_worker_main(recv_queue, send_queue, name: str,
                                config: dict) -> None:
    """L3 module entry — Guardian supervised.

    Boot sequence:
      1. sys.path bootstrap (spawn mode).
      2. setup_worker_bus (subscribe HEAL_RESULT + MODULE_SHUTDOWN).
      3. resolve titan_id.
      4. discover plugins, filter by applies_on, instantiate.
      5. load state.json (restores schedule + heal history).
      6. start heartbeat thread (30s).
      7. Phase 11 §11.I.2 — SHM slot state=booted (replaces legacy
         MODULE_READY bus emit per D-SPEC-141 / v1.65.0 locked D2).
      8. main tick loop: per-plugin scheduler + bus drain.
    """
    # Phase 11 §11.I.5 — reset module-level readiness sentinel for both
    # fork-mode (parent's True is inherited) and spawn-mode (fresh False).
    global _WORKER_READY
    _WORKER_READY = False

    # === BOILERPLATE: spawn-mode sys.path bootstrap ===
    project_root = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # === BOILERPLATE: socket-mode bus client setup ===
    from titan_hcl.core.worker_bus_bootstrap import setup_worker_bus
    try:
        recv_queue, send_queue, _bus_client = setup_worker_bus(
            name, recv_queue, send_queue,
            topics=[MODULE_SHUTDOWN],
        )
    except Exception as e:
        logger.error(
            "[HealthMonitor] setup_worker_bus failed: %s — exiting",
            e, exc_info=True)
        return

    # === BOILERPLATE: pdeathsig installation ===
    try:
        from titan_hcl.core.worker_lifecycle import (
            install_parent_death_signal)
        install_parent_death_signal()
    except Exception as e:
        logger.debug(
            "[HealthMonitor] pdeathsig install skipped: %s", e)

    from titan_hcl.core.state_registry import resolve_titan_id
    titan_id = resolve_titan_id()
    boot_ts = time.time()

    # ── Phase 11 §11.I.5 — SHM state-slot writer (G21 single-writer) ──
    # Constructed BEFORE plugin discovery + state load so the slot
    # publishes state="starting" immediately. titan_hcl's 1Hz SHM poll
    # sees the worker is alive while it warms. health_monitor is
    # MANDATORY per §3H.10 — it MUST be alive before any worker errors
    # emit so the L3 self-heal pipeline can react.
    _state_writer = None
    try:
        from titan_hcl.core.module_state import (
            BootPriority, ModuleStateWriter,
        )
        _state_writer = ModuleStateWriter(
            module_name=name, layer="L3",
            boot_priority=BootPriority.MANDATORY,
        )
        _state_writer.write_state("starting")
    except Exception as _sw_err:  # noqa: BLE001
        logger.warning(
            "[HealthMonitor] Phase 11 ModuleStateWriter init failed "
            "(continuing — SHM slot will be absent): %s", _sw_err)

    # === Heartbeat thread (started EARLY per Phase 11 §11.I.5 — covers ==
    # === the slow plugin-discovery boot window with both bus + SHM) ===
    stop_event = threading.Event()
    hb_thread = threading.Thread(
        target=_heartbeat_loop,
        args=(send_queue, name, stop_event, _state_writer),
        daemon=True, name="health-monitor-heartbeat")
    hb_thread.start()

    # === Plugin discovery + state load ===
    plugins = _discover_plugins(titan_id, config)
    state_path = _state_dir() / _STATE_FILE
    journal_path = _state_dir() / _JOURNAL_FILE
    saved_state = _load_state(state_path)
    saved_plugins = (saved_state.get("plugins") or {})

    now = time.time()
    runtimes: dict[str, _PluginRuntime] = {}
    for plugin in plugins:
        saved = saved_plugins.get(plugin.name) or {}
        runtimes[plugin.name] = _PluginRuntime(
            plugin, now,
            next_fire_time=saved.get("next_fire_time"),
            heal_history_24h=saved.get("heal_history_24h"),
            last_heal_at=saved.get("last_heal_at"),
            consecutive_failures=int(
                saved.get("consecutive_failures", 0)),
            last_heal_failed_emit_ts=saved.get(
                "last_heal_failed_emit_ts"),
            last_result=saved.get("last_result"),
        )

    logger.info(
        "[HealthMonitor] booted titan_id=%s plugins_loaded=%d "
        "(names=%s)",
        titan_id, len(runtimes),
        sorted(runtimes.keys()))

    # === Phase 11 §11.I.2 — SHM slot transition starting → booted ====
    # (heartbeat thread already started above before plugin discovery)
    # MODULE_READY bus emit DELETED per D-SPEC-141 / v1.65.0 locked D2.
    _WORKER_READY = True
    if _state_writer is not None:
        try:
            _state_writer.write_state("booted")
            logger.info(
                "[HealthMonitor] Phase 11 §11.I.2 — SHM slot state=booted "
                "(awaiting MODULE_PROBE_REQUEST from titan_hcl) "
                "titan_id=%s plugin_count=%d",
                titan_id, len(runtimes))
        except Exception as _swb_err:  # noqa: BLE001
            logger.warning(
                "[HealthMonitor] Phase 11 write_state(booted) failed: %s",
                _swb_err)

    # === Bounded executor for check() runs ===
    # max_workers=4 — small bound; checks are 30s-capped, one per plugin
    # in flight is sufficient with current plugin set (1).
    executor = ThreadPoolExecutor(
        max_workers=4, thread_name_prefix="health-check")
    in_flight_checks: dict[str, Future] = {}

    last_state_save = boot_ts
    shutdown_requested = False

    try:
        while not shutdown_requested:
            now = time.time()

            # ── 1. Sweep completed check() futures ──
            done_plugins: list[str] = []
            for pname, fut in list(in_flight_checks.items()):
                if not fut.done():
                    continue
                done_plugins.append(pname)
                rt = runtimes.get(pname)
                if rt is None:
                    continue
                try:
                    results = fut.result(timeout=0)
                except FutTimeout:
                    # Should not happen (we just checked done()) — defensive.
                    continue
                except Exception as e:
                    logger.warning(
                        "[HealthMonitor] plugin %s check() raised: %s",
                        pname, e, exc_info=True)
                    results = [HealthResult(
                        plugin=pname, layer="meta", status="DOWN",
                        reason=f"check_exception:{type(e).__name__}",
                        details={"exception": str(e)[:200]},
                        heal_recommended=False,
                    )]
                if not isinstance(results, list) or not results:
                    logger.warning(
                        "[HealthMonitor] plugin %s check() returned %r — "
                        "must be non-empty list[HealthResult]; coercing",
                        pname, results)
                    results = [HealthResult(
                        plugin=pname, layer="meta", status="DOWN",
                        reason="check_returned_empty_or_non_list",
                        heal_recommended=False,
                    )]
                for r in results:
                    if not isinstance(r, HealthResult):
                        continue
                    rt.last_result = r
                    _send(send_queue, HEALTH_CHECK_RESULT, name, "all",
                          r.to_dict())
                    _append_journal(
                        journal_path, "check_result", r.to_dict())
                    # Maybe heal — only the most-recent heal_recommended
                    # layer gets a heal attempt per pass (avoid emitting
                    # multiple HEAL_REQUESTs per check).
                    if r.heal_recommended:
                        _maybe_emit_heal_request(
                            rt, r, send_queue, name, journal_path)
            for pname in done_plugins:
                in_flight_checks.pop(pname, None)

            # ── 2. Schedule new check() submissions ──
            for pname, rt in runtimes.items():
                if pname in in_flight_checks:
                    continue
                if now < rt.next_fire_time:
                    continue
                # Submit check() to executor with timeout enforcement.
                def _run_check(p=rt.plugin) -> list[HealthResult]:
                    return p.check()
                fut = executor.submit(_run_check)
                in_flight_checks[pname] = fut
                rt.next_fire_time = now + rt.plugin.cadence_s
                # Async timeout enforcement: spawn a watchdog that
                # cancels the future if it runs over HEALTH_CHECK_TIMEOUT_S.
                # (ThreadPoolExecutor can't actually kill the thread; we
                # let it complete in the background but stop awaiting it
                # by recording a timeout result and dropping the future
                # from in_flight on the next sweep.)
                def _watchdog(pname=pname, fut=fut) -> None:
                    try:
                        fut.result(timeout=HEALTH_CHECK_TIMEOUT_S)
                    except FutTimeout:
                        logger.warning(
                            "[HealthMonitor] plugin %s check() exceeded "
                            "%.1fs — emitting timeout HealthResult; "
                            "real future continues in background",
                            pname, HEALTH_CHECK_TIMEOUT_S)
                        timeout_result = HealthResult(
                            plugin=pname, layer="meta", status="DOWN",
                            reason=(f"check_timeout_"
                                    f"{HEALTH_CHECK_TIMEOUT_S:.0f}s"),
                            heal_recommended=False,
                        )
                        # Emit timeout result; sweep will replace it with
                        # the real result when the future eventually
                        # completes (or be dropped on the next pass since
                        # we removed it from in_flight here).
                        _send(send_queue, HEALTH_CHECK_RESULT, name,
                              "all", timeout_result.to_dict())
                        _append_journal(
                            journal_path, "check_result",
                            timeout_result.to_dict())
                        in_flight_checks.pop(pname, None)
                    except Exception:
                        pass
                threading.Thread(
                    target=_watchdog, daemon=True,
                    name=f"health-watchdog-{pname}").start()

            # ── 3. Sweep heal-reply timeouts ──
            _check_pending_heal_timeouts(runtimes, send_queue, name)

            # ── 4. Drain bus (HEAL_RESULT + lifecycle) ──
            drain_deadline = now + RECV_POLL_TIMEOUT_S
            while time.time() < drain_deadline:
                try:
                    msg = recv_queue.get(timeout=RECV_POLL_TIMEOUT_S)
                except Empty:
                    break
                msg_type = (msg.get("type") if isinstance(msg, dict)
                            else None)
                if msg_type == MODULE_SHUTDOWN:
                    logger.info(
                        "[HealthMonitor] MODULE_SHUTDOWN received — "
                        "exiting cleanly.")
                    shutdown_requested = True
                    break
                # ── Phase 11 §11.I.3 — MODULE_PROBE_REQUEST handler ─────
                if msg_type == "MODULE_PROBE_REQUEST":
                    try:
                        from titan_hcl.core.probe_dispatcher import (
                            handle_module_probe_request,
                        )
                        handle_module_probe_request(
                            msg, send_queue=send_queue, module_name=name,
                            state_writer=_state_writer, probe_fn=None,
                        )
                    except Exception as _probe_err:  # noqa: BLE001
                        logger.warning(
                            "[HealthMonitor] MODULE_PROBE_REQUEST handler "
                            "failed: %s", _probe_err)
                    continue
                if msg_type == HEAL_RESULT:
                    payload = msg.get("payload") or {}
                    plugin_name = payload.get("plugin")
                    cid = payload.get("correlation_id")
                    rt = runtimes.get(plugin_name)
                    if rt is None:
                        logger.debug(
                            "[HealthMonitor] HEAL_RESULT for unknown "
                            "plugin %r — dropping", plugin_name)
                        continue
                    if cid not in rt.pending_heals:
                        logger.debug(
                            "[HealthMonitor] HEAL_RESULT for unknown "
                            "correlation_id %r plugin=%s — dropping "
                            "(likely already-timed-out)",
                            cid, plugin_name)
                        continue
                    meta = rt.pending_heals.pop(cid)
                    success = bool(payload.get("success", False))
                    reason = str(payload.get("reason", ""))
                    _record_heal_outcome(
                        rt, action=meta["action"],
                        result=("success" if success else "failed"),
                        reason=reason,
                        send_queue=send_queue, name=name)
                    _append_journal(journal_path, "heal_result", {
                        "plugin": plugin_name,
                        "action": meta["action"],
                        "success": success, "reason": reason,
                        "correlation_id": cid,
                    })
                    continue

            # ── 5. Periodic state persistence ──
            if (time.time() - last_state_save
                    >= STATE_PERSIST_INTERVAL_S):
                _save_state(state_path, runtimes)
                last_state_save = time.time()

            # ── 6. Sleep ──
            time.sleep(TICK_INTERVAL_S)
    finally:
        stop_event.set()
        # Final state save on shutdown.
        try:
            _save_state(state_path, runtimes)
        except Exception as e:
            logger.warning(
                "[HealthMonitor] final state save failed: %s", e)
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        logger.info("[HealthMonitor] exited cleanly.")


__all__ = (
    "HEARTBEAT_INTERVAL_S",
    "TICK_INTERVAL_S",
    "STATE_PERSIST_INTERVAL_S",
    "DAILY_WINDOW_S",
    "_PluginRuntime",
    "_discover_plugins",
    "_state_dir",
    "_load_state",
    "_save_state",
    "_atomic_write_json",
    "_append_journal",
    "_record_heal_outcome",
    "_maybe_emit_heal_request",
    "_check_pending_heal_timeouts",
    "health_monitor_worker_main",
)
