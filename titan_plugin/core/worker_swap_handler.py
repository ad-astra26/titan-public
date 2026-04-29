"""
worker_swap_handler — Phase B.2.1 worker-side supervision-transfer state machine.

Owns the worker-side message flow for "workers literally outlive kernel swap":

  BUS_HANDOFF receipt          → strip PDEATHSIG (spawn-mode) + pause watcher + ack
  bus reconnect to new broker  → request_adoption() sends BUS_WORKER_ADOPT_REQUEST
  BUS_WORKER_ADOPT_ACK         → exit swap-pending; resume watcher in relaxed mode
  BUS_HANDOFF_CANCELED         → re-arm PDEATHSIG; restore strict watcher (P-2c unwind)
  supervision_check (periodic) → if swap-pending AND broker unreachable ≥ N seconds → self-SIGTERM

Hybrid fork/spawn fallback (D3 of PLAN_microkernel_phase_b2_1_supervision.md):
  • spawn-mode workers: take the "true outlive" path (strip PDEATHSIG, adopt)
  • fork-mode workers: log + ignore HANDOFF — die with old, get respawned by
    shadow's Guardian. Improved-B.1 path (no DB-lock contention thanks to B.2;
    cleaner bus reattach; same restart pattern overall).

Bus-as-supervision is the post-adopt supervision signal: a healthy bus
connection to a healthy broker IS the supervision invariant. If the bus
client cannot reach a broker for `_supervision_timeout_s` (default 30s)
while in swap-pending state, the worker self-SIGTERMs to avoid permanent
orphan-hood. This is intentionally generous (≈2× the 15s ping/pong
stale-detection window from B.2) to allow for slow shadow boot.

Wiring (per worker entry_fn that opts in — currently spawn-mode workers only):

    state = SwapHandlerState(
        name=name,
        start_method="spawn",     # from env var TITAN_WORKER_START_METHOD or ModuleSpec
        watcher_state=protection["watcher_state"],
        bus_client=client,        # BusSocketClient or None for fork-mode
    )

    # In main loop dispatch:
    if msg_type == bus.BUS_HANDOFF:
        worker_swap_handler.on_bus_handoff(state, msg)
    elif msg_type == bus.BUS_WORKER_ADOPT_ACK:
        worker_swap_handler.on_bus_adopt_ack(state, msg)
    elif msg_type == bus.BUS_HANDOFF_CANCELED:
        worker_swap_handler.on_bus_handoff_canceled(state, msg)

    # Periodic (every 5–10s) — main loop ticker, or its own thread:
    worker_swap_handler.supervision_check(state)

    # When BusSocketClient signals reconnect after EOF (bus.py disconnect handler):
    worker_swap_handler.request_adoption(state)

The handler functions are pure — they only mutate `state` + call into the
opaque `bus_client` and `watcher_state`. Tests inject mocks for both.

Phase C portability: this module is worker-side Python; workers stay Python
in Phase C (PyO3 bridge to Rust L0). Wire-format messages it consumes and
produces are msgpack-framed via the same `core/_frame.py` parity vectors —
Rust workers (Phase D) speak the same protocol byte-for-byte.
"""
from __future__ import annotations

import logging
import os
import signal
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Optional

from titan_plugin import bus
from titan_plugin.core.worker_lifecycle import (
    WatcherState,
    clear_parent_death_signal,
    install_parent_death_signal,
    pause_parent_watcher,
    resume_parent_watcher,
)

logger = logging.getLogger(__name__)

# Process-global active swap-handler state. Each worker is its own process;
# at most one SwapHandlerState exists per process. Set by _module_wrapper at
# bootstrap; read by worker entry_fn dispatch branches.
_ACTIVE_STATE: Optional["SwapHandlerState"] = None
_ACTIVE_STATE_LOCK = threading.Lock()


@dataclass
class SwapHandlerState:
    """Per-worker B.2.1 supervision-transfer state.

    Created at worker bootstrap; survives across the swap window. Single-
    threaded mutation by the worker's main loop dispatcher; the
    parent_watcher reads `_bus_unreachable_since` via WatcherState (already
    GIL-atomic for our purposes).
    """
    name: str                    # module name — must match Guardian's spec_registry key
    start_method: str            # "fork" or "spawn" — from ModuleSpec / env
    watcher_state: WatcherState  # from install_full_protection["watcher_state"]
    bus_client: object           # BusSocketClient or compatible (.is_connected, .publish)
    # Internal flags — set/cleared by handler functions
    _swap_pending: bool = False
    _adopted: bool = False
    _handoff_event_id: Optional[str] = None
    _adopt_rid: Optional[str] = None
    # Bus-as-supervision tracking (mirrored into watcher_state for the watcher's relaxed-mode check)
    _bus_unreachable_since: Optional[float] = None


def _is_spawn_mode(state: SwapHandlerState) -> bool:
    """Hybrid policy: only spawn-mode workers take the B.2.1 fast-path.

    Fork-mode workers stay PDEATHSIG-armed and die with old kernel; shadow's
    Guardian respawns them per ModuleSpec (improved-B.1 fallback per D3).
    """
    return state.start_method == "spawn"


def _publish(state: SwapHandlerState, msg: dict) -> bool:
    """Publish via the worker's bus client. Returns False if client is missing
    or not connected (caller decides whether to retry / log).
    """
    client = state.bus_client
    if client is None:
        return False
    pub = getattr(client, "publish", None)
    if not callable(pub):
        return False
    try:
        return bool(pub(msg))
    except Exception:  # noqa: BLE001 — broker disconnects raise opaque types
        return False


def on_bus_handoff(state: SwapHandlerState, msg: dict) -> None:
    """Handle BUS_HANDOFF receipt.

    Spawn-mode workers: clear PDEATHSIG, pause watcher, set _swap_pending,
    reply with BUS_HANDOFF_ACK so orchestrator's Phase 2 collector can
    confirm the spawn-mode fleet is ready before kernel detaches broker.

    Fork-mode workers: log + return (improved-B.1 path; PDEATHSIG stays
    armed; worker dies with old kernel; shadow respawns it).
    """
    if not _is_spawn_mode(state):
        logger.info(
            "[%s] BUS_HANDOFF received; fork-mode worker, ignoring (improved-B.1 path)",
            state.name,
        )
        return

    payload = msg.get("payload", {}) or {}
    state._handoff_event_id = payload.get("event_id")

    if not clear_parent_death_signal():
        logger.warning(
            "[%s] clear_parent_death_signal failed; staying PDEATHSIG-armed "
            "(this worker will fall back to improved-B.1 even though spawn-mode)",
            state.name,
        )
        return

    pause_parent_watcher(state.watcher_state)
    state._swap_pending = True

    # Best-effort BUS_HANDOFF_ACK so the orchestrator can collect.
    # M1 (2026-04-27 PM audit): dst="shadow_swap" — same destination as
    # HIBERNATE_ACK so the orchestrator's _drain_messages collects both
    # via the single "shadow_swap" inbox subscription. Previously "kernel"
    # which had no in-process subscriber → silently dropped. Changing to
    # "shadow_swap" makes spawn-mode workers' acks reach the swap path's
    # collector, enabling the split ack-by-start_method M1 fix.
    ack = bus.make_msg(
        bus.BUS_HANDOFF_ACK,
        state.name,
        "shadow_swap",
        {
            "event_id": state._handoff_event_id,
            "pid": os.getpid(),
            "start_method": state.start_method,
        },
    )
    _publish(state, ack)
    logger.info(
        "[%s] BUS_HANDOFF acked; PDEATHSIG cleared; swap_pending; awaiting adoption",
        state.name,
    )


def request_adoption(state: SwapHandlerState) -> None:
    """Send BUS_WORKER_ADOPT_REQUEST after BusSocketClient reconnects to new broker.

    Called by the worker's reconnect-completed hook (or its main-loop ticker
    when it observes the client transitioning from disconnected → connected
    while in swap-pending state). Idempotent: re-sending while already
    adopted is harmless (shadow Guardian rejects with "already_registered").
    """
    if not _is_spawn_mode(state) or not state._swap_pending:
        return
    rid = str(uuid.uuid4())
    state._adopt_rid = rid
    # M1.5 (2026-04-27 PM): dst="guardian" — Guardian subscribes to bus
    # as "guardian" (guardian.py:156: bus.subscribe("guardian")) and
    # processes BUS_WORKER_ADOPT_REQUEST in _process_guardian_messages
    # (line 948). Sending to dst="kernel" had no in-process subscriber,
    # message was silently dropped → adoption never completed even when
    # all the rest of the protocol worked.
    req = bus.make_msg(
        bus.BUS_WORKER_ADOPT_REQUEST,
        state.name,
        "guardian",
        {
            "name": state.name,
            "pid": os.getpid(),
            "start_method": state.start_method,
            "boot_ts": time.time(),
        },
        rid=rid,
    )
    if _publish(state, req):
        logger.info("[%s] sent BUS_WORKER_ADOPT_REQUEST rid=%s", state.name, rid[:8])
    else:
        logger.warning(
            "[%s] BUS_WORKER_ADOPT_REQUEST publish failed; will retry on next reconnect",
            state.name,
        )


def on_bus_adopt_ack(state: SwapHandlerState, msg: dict) -> None:
    """Handle BUS_WORKER_ADOPT_ACK.

    On status="adopted": exit swap-pending; resume watcher in relaxed mode
    (getppid()==1 alone no longer triggers self-SIGTERM; supervision is now
    the bus connection).

    On status="rejected": log + self-SIGTERM. Shadow Guardian will respawn
    via ModuleSpec — improved-B.1 path for this worker.
    """
    if not _is_spawn_mode(state):
        return  # fork-mode never sent a request; ACK is spurious

    payload = msg.get("payload", {}) or {}
    incoming_rid = msg.get("rid")
    if state._adopt_rid is not None and incoming_rid != state._adopt_rid:
        logger.debug(
            "[%s] BUS_WORKER_ADOPT_ACK rid mismatch (got %s, expected %s); ignoring",
            state.name,
            (incoming_rid or "")[:8],
            state._adopt_rid[:8],
        )
        return

    status = payload.get("status")
    if status != "adopted":
        reason = payload.get("reason") or "unspecified"
        logger.error(
            "[%s] adoption REJECTED (reason=%s) — self-SIGTERM",
            state.name, reason,
        )
        os.kill(os.getpid(), signal.SIGTERM)
        return

    state._adopted = True
    state._swap_pending = False
    resume_parent_watcher(state.watcher_state, relaxed=True)
    logger.info(
        "[%s] adopted by shadow_pid=%s; bus-as-supervision active",
        state.name, payload.get("shadow_pid"),
    )


def on_bus_handoff_canceled(state: SwapHandlerState, msg: dict) -> None:
    """Handle BUS_HANDOFF_CANCELED (P-2c unwind).

    Re-arm PDEATHSIG against the (still-alive) old kernel and restore strict
    watcher semantics. State returns to pre-HANDOFF.
    """
    if not _is_spawn_mode(state) or not state._swap_pending:
        return
    install_parent_death_signal(sig=signal.SIGTERM)
    resume_parent_watcher(state.watcher_state, relaxed=False)
    state._swap_pending = False
    state._handoff_event_id = None
    state._adopt_rid = None
    logger.info(
        "[%s] BUS_HANDOFF_CANCELED — PDEATHSIG re-armed; strict watcher restored",
        state.name,
    )


def supervision_check(state: SwapHandlerState, *, now: Optional[float] = None) -> None:
    """Periodic check (every 5–10s).

    During swap-pending (no adoption ACK yet), if the bus client has been
    unreachable for ≥ supervision_timeout_s seconds, self-SIGTERM. After
    adoption, the watcher's relaxed-mode handles the supervision-via-bus
    invariant directly; this function is a fast-path safety net for the
    pre-adoption window.

    `now` is injectable for tests; defaults to time.time().
    """
    if not _is_spawn_mode(state):
        return
    if not state._swap_pending:
        # Either pre-HANDOFF or post-adoption — nothing to do here.
        # (Post-adoption supervision lives in WatcherState relaxed-mode.)
        state._bus_unreachable_since = None
        if state.watcher_state is not None:
            state.watcher_state.mark_bus_healthy()
        return

    now = now if now is not None else time.time()
    is_connected = bool(getattr(state.bus_client, "is_connected", False))
    if is_connected:
        state._bus_unreachable_since = None
        if state.watcher_state is not None:
            state.watcher_state.mark_bus_healthy()
        return

    if state._bus_unreachable_since is None:
        state._bus_unreachable_since = now
        if state.watcher_state is not None:
            state.watcher_state.mark_bus_unreachable()
        return

    elapsed = now - state._bus_unreachable_since
    threshold = state.watcher_state._supervision_timeout_s if state.watcher_state else 30.0
    if elapsed >= threshold:
        logger.error(
            "[%s] supervision-via-bus timeout (%.1fs without broker contact, swap-pending) — self-SIGTERM",
            state.name, elapsed,
        )
        os.kill(os.getpid(), signal.SIGTERM)


def maybe_dispatch_swap_msg(msg: dict) -> bool:
    """Phase B.2.1 — single-call worker-side dispatch helper.

    If `msg` is one of the three B.2.1 supervision-transfer messages
    (BUS_HANDOFF, BUS_WORKER_ADOPT_ACK, BUS_HANDOFF_CANCELED), routes it
    to the appropriate handler against the active SwapHandlerState set
    by Guardian._module_wrapper at worker bootstrap, and returns True.
    Otherwise returns False so the worker continues its normal dispatch
    chain.

    Per-worker wiring (one block right after `msg_type = msg.get("type", "")`,
    placed immediately after the B.1 readiness_reporter handles-block):

        # ── Microkernel v2 Phase B.2.1 — supervision-transfer dispatch ──
        from titan_plugin.core import worker_swap_handler as _swap
        if _swap.maybe_dispatch_swap_msg(msg):
            continue

    No-op (returns False) when no SwapHandlerState is registered — legacy
    mp.Queue mode, or any worker process where B.2.1 bootstrap did not run.
    """
    msg_type = msg.get("type", "")
    if msg_type not in (
        bus.BUS_HANDOFF,
        bus.BUS_WORKER_ADOPT_ACK,
        bus.BUS_HANDOFF_CANCELED,
    ):
        return False
    state = get_active_swap_state()
    if state is None:
        return False
    if msg_type == bus.BUS_HANDOFF:
        on_bus_handoff(state, msg)
    elif msg_type == bus.BUS_WORKER_ADOPT_ACK:
        on_bus_adopt_ack(state, msg)
    elif msg_type == bus.BUS_HANDOFF_CANCELED:
        on_bus_handoff_canceled(state, msg)
    return True


def set_active_swap_state(state: Optional["SwapHandlerState"]) -> None:
    """Register the per-process swap-handler state.

    Called from Guardian._module_wrapper at worker bootstrap. Worker
    entry_fns dispatch BUS_HANDOFF / BUS_WORKER_ADOPT_ACK / BUS_HANDOFF_CANCELED
    via get_active_swap_state(); the indirection keeps per-worker change
    minimal (3 elif branches) and lets the state live in the daemon thread
    without entry_fn signature changes.
    """
    global _ACTIVE_STATE
    with _ACTIVE_STATE_LOCK:
        _ACTIVE_STATE = state


def get_active_swap_state() -> Optional["SwapHandlerState"]:
    """Worker entry_fns use this in their dispatch branch."""
    with _ACTIVE_STATE_LOCK:
        return _ACTIVE_STATE


def start_supervision_thread(
    state: SwapHandlerState,
    *,
    interval: float = 5.0,
    stop_event: Optional[threading.Event] = None,
) -> threading.Thread:
    """Start the per-worker supervision daemon (Phase B.2.1).

    Every `interval` seconds:
      • Call supervision_check(state) — handles self-SIGTERM at timeout
      • If state._swap_pending and bus_client.reconnect_count just increased
        while is_connected → call request_adoption(state)
      • Exit cleanly once state._adopted is True (post-adopt supervision
        lives in WatcherState relaxed-mode; daemon's job is done)

    Daemon thread: never blocks process exit. Errors are caught + logged;
    a buggy supervision tick MUST NOT crash the worker.

    For fork-mode workers a no-op daemon is returned (uniform API surface;
    fork-mode dies with old kernel + gets respawned via improved-B.1).
    """
    if state.start_method != "spawn":
        t = threading.Thread(
            target=lambda: None,
            name=f"swap_handler_noop_{state.name}",
            daemon=True,
        )
        t.start()
        return t

    stop = stop_event if stop_event is not None else threading.Event()

    def _initial_reconnect_count() -> int:
        client = state.bus_client
        if client is None:
            return 0
        try:
            return int(getattr(client, "reconnect_count", 0))
        except (TypeError, ValueError):
            return 0

    # M3 (2026-04-27 PM audit): tick faster during the swap window so the
    # adoption-wait latency is bounded by ~1s instead of ~5s. Outside the
    # window we keep the lazy `interval` tick (low overhead in steady state).
    # Tests may pass `interval` smaller than 1.0 — in that case keep their
    # tighter cadence (min). Production interval default is 5.0.
    fast_interval = min(interval, 1.0)

    def _effective_interval() -> float:
        return fast_interval if state._swap_pending and not state._adopted else interval

    def _loop() -> None:
        last_rc = _initial_reconnect_count()
        while not stop.is_set():
            try:
                supervision_check(state)
                client = state.bus_client
                if state._swap_pending and client is not None:
                    try:
                        cur_rc = int(getattr(client, "reconnect_count", 0))
                        is_conn = bool(getattr(client, "is_connected", False))
                    except (TypeError, ValueError):
                        cur_rc, is_conn = last_rc, False
                    if cur_rc > last_rc and is_conn:
                        request_adoption(state)
                        last_rc = cur_rc
                if state._adopted:
                    return
            except Exception as e:  # noqa: BLE001 — never crash the worker
                logger.warning(
                    "[%s] supervision daemon tick error: %s",
                    state.name, e,
                )
            stop.wait(_effective_interval())

    t = threading.Thread(
        target=_loop,
        name=f"swap_handler_{state.name}",
        daemon=True,
    )
    t.start()
    return t
