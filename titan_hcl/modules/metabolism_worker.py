"""metabolism_worker — Python L2 module hosting the MetabolismController instance.

Per rFP_titan_hcl_l2_separation_strategy.md §4.J (LOCKED 2026-05-05;
SHIPPED 2026-05-14) + SPEC v1.7.2 §9.B metabolism_worker block +
D-SPEC-51.

ACTIVE: always-on autostart. No flag-gate (replaces titan_HCL inline
wire at plugin.py:1612 _wire_metabolism per §0 vision target — "if we
can make our titan_HCL lighter it will run faster, more stable").

Owns:
  - MetabolismController (titan_hcl/core/metabolism.py — 484 LOC, 24
    methods: SOL-balance gating, metabolic-tier evaluation, gates_enforced
    kill-switch, per-feature gate decisions, growth metrics).
  - In-memory ring buffer of last 64 evaluate_gate results
    (authoritative writer per Maker-locked 2026-05-14 G19-strict design;
    consumed by future /v4/metabolism/gate-history via
    GATE_DECISION_RECORDED bus broadcast).
  - In-memory tier-transition history (last 24h; consumed by future
    /v4/metabolism/tier-history).
  - metabolism_state.bin SHM slot (G21 single-writer; 1 Hz; payload per
    SPEC §7.1 v1.7.2).
  - MetabolismProxy dispatch handler (9 actions per
    phase_c_rpc_exemptions.yaml::work_rpc_sites — all ≤5s per G19).

Bus subscriptions:
  REQUIRED — bus.QUERY (dst=metabolism) for MetabolismProxy dispatch
             + SOL_BALANCE_CHANGED + MODULE_SHUTDOWN + SAVE_NOW
  OPTIONAL — LIFE_FORCE_UPDATED (cross-worker bridge from §4.G future;
             NULL-safe until §4.G ships)
             + SWAP_HANDOFF / ADOPTION_REQUEST (B.2.1 supervision-transfer)

Bus publications:
  - METABOLIC_TIER_CHANGED              (on every tier transition)
  - GATE_DECISION_RECORDED              (per evaluate_gate call)
  - METABOLIC_STATS_UPDATED             (1Hz coalesced; bulk via SHM)
  - MODULE_HEARTBEAT / MODULE_SHUTDOWN  (standard per §11; legacy MODULE_READY
                                         retired per Phase 11 §11.I.2 — SHM
                                         slot state=booted is the contract)

Implementation reference: social_graph_worker.py (CANONICAL §9.B
TEMPLATE) for `=== BOILERPLATE ===` sections + memory_worker.py
`_periodic_publish_loop` for the 1Hz SHM publisher thread.

Migration map per SPEC §9.B v1.7.2:
  plugin.py:1612 `_wire_metabolism`            → REMOVED (proxy
                                                  instantiation only)
  plugin.py:1670 `self.soul.set_metabolism(...)` → REMOVED (Soul reads
                                                  SHM directly via
                                                  MetabolismShmReader)
  Soul.allow_mint                              → uses
                                                  MetabolismShmReader
                                                  for tier read + sync
                                                  proxy
                                                  evaluate_gate_sync
                                                  for gate decision
  memo_inscribe.send_memo                      → uses sync proxy
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time
from collections import deque
from queue import Empty
from typing import Any, Optional

from titan_hcl import bus
from titan_hcl.core.module_error_handler import with_error_envelope
from titan_hcl.errors import Severity as _phase11_sev

logger = logging.getLogger(__name__)

# Module name (matches Guardian registry per SPEC §9.B v1.7.2).
MODULE_NAME = "metabolism"

# Phase 11 §11.I.3 / §11.I.5 (Chunk 11N) — module-level readiness sentinel
# mirrored to the per-process SHM slot via ModuleStateWriter. Set False at
# import time; flipped True after MetabolismController + SHM publisher
# init complete. Heartbeat publishes to the SHM slot only once True so the
# slot stays at "starting"/"booted" during the slow boot window.
from titan_hcl.modules._heartbeat_grace import (
    boot_deadline_from_now, shm_heartbeat_allowed,
)
from titan_hcl.params import get_params

_WORKER_READY: bool = False
_BOOT_DEADLINE = None  # boot-grace deadline (monotonic); None=no grace

# Cadence + lifecycle constants.
_HEARTBEAT_INTERVAL_S = 10.0            # SPEC §10.B MODULE_HEARTBEAT_INTERVAL_S
_POLL_INTERVAL_S = 0.2                  # recv loop poll cadence
_SHM_PUBLISH_INTERVAL_S = 1.0           # metabolism_state.bin 1 Hz per SPEC §7.1
_STATS_NOTIFY_INTERVAL_S = 1.0          # METABOLIC_STATS_UPDATED bus notification cadence
_BALANCE_REFRESH_INTERVAL_S = 30.0      # auto SOL balance refresh cadence
                                        # (also triggered explicitly by
                                        # SOL_BALANCE_CHANGED bus event)
_RING_BUFFER_MAX = 64                   # last N evaluate_gate results
_TIER_HISTORY_WINDOW_S = 86400.0        # last 24h of tier transitions

# Topics subscribed by metabolism_worker (per SPEC §9.B v1.7.2 +
# v1.8.3 §4.G — LIFE_FORCE_UPDATED soft-dep from v1.7.2 finally
# activates since life_force_worker now publishes it).
_METABOLISM_WORKER_SUBSCRIBE_TOPICS: list[str] = [
    bus.QUERY,                     # MetabolismProxy dispatch (dst=metabolism)
    bus.SOLANA_BALANCE_UPDATED,    # triggers tier re-evaluation
    bus.LIFE_FORCE_UPDATED,        # §4.G producer SHIPPED v1.8.3 — tier
                                   # weighting cross-worker bridge per
                                   # Maker-locked 2026-05-14 SPEC §9.B
                                   # life_force_worker "OPTIONAL" subscription
                                   # promoted to wired-but-best-effort below
    bus.MODULE_SHUTDOWN,
    bus.SAVE_NOW,
]


# ── Lifecycle helpers (mirror social_graph_worker template) ───────────


def _send_msg(send_queue, msg_type: str, src: str, dst: str, payload: dict,
              rid=None) -> None:
    """Best-effort enqueue helper — never raises (heartbeat path)."""
    try:
        msg = {"type": msg_type, "src": src, "dst": dst, "payload": payload,
               "ts": time.time()}
        if rid is not None:
            msg["rid"] = rid
        send_queue.put(msg)
    except Exception:
        pass


def _send_heartbeat(send_queue, name: str, extra: Optional[dict] = None,
                    state_writer: Optional[Any] = None) -> None:
    """Emit MODULE_HEARTBEAT to guardian_HCL with current RSS.

    Phase 11 §11.I.5 (Chunk 11N): also publishes ModuleStateWriter.heartbeat()
    on the SHM slot when `state_writer` is provided AND `_WORKER_READY` is
    True so guardian_HCL's SHM-staleness detector + observatory
    /v6/readiness see fresh data on the same cadence as the legacy bus path.
    """
    try:
        import resource
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        rss_mb = 0.0
    payload = {"alive": True, "ts": time.time(), "rss_mb": round(rss_mb, 1)}
    if extra:
        payload.update(extra)
    _send_msg(send_queue, bus.MODULE_HEARTBEAT, name, "guardian", payload)
    if state_writer is not None and shm_heartbeat_allowed(_WORKER_READY, _BOOT_DEADLINE):
        try:
            state_writer.heartbeat()
        except Exception:  # noqa: BLE001 — never crash heartbeat
            pass


def _send_response(send_queue, src_name: str, dst: str, payload: dict,
                   rid) -> None:
    """Emit RESPONSE to the bus.QUERY caller. rid is required."""
    if rid is None:
        return
    try:
        send_queue.put({
            "type": bus.RESPONSE,
            "src": src_name,
            "dst": dst,
            "rid": rid,
            "payload": payload,
            "ts": time.time(),
        })
    except Exception as e:
        logger.warning(
            "[MetabolismWorker] send_response enqueue failed (rid=%s): %s",
            rid, e)


# ── MetabolismController init (replicates plugin.py:_wire_metabolism inline) ─


def _build_growth_overrides(titan_id: str) -> dict:
    """Build the 3 async growth callables that bind metabolism's growth
    metrics to memory_worker stats via memory_state.bin SHM slot.

    Replicates the monkey-patch trio from plugin.py:_wire_metabolism
    (_v3_learning_velocity / _v3_directive_alignment / _v3_social_density)
    but reads from SHM directly per G18 instead of via MemoryProxy
    (which requires a DivineBus instance that this subprocess does not
    have access to). The values are identical to those memory_proxy
    would return — memory_worker publishes the same growth_metrics dict
    into memory_state.bin at 1Hz, so a sub-ms SHM read replaces a
    proxy.get_growth_metrics() work-RPC.

    NOTE: this closes the regression introduced when this worker was
    first extracted with `MemoryProxy(bus_client_factory=None)` — the
    proxy signature is (bus, guardian), so that call silently failed
    and the growth getters returned 0.5 constants. SHM-direct read is
    the SPEC-correct closure per G18.
    """
    import msgpack
    from titan_hcl.core.state_registry import (
        StateRegistryReader, ensure_shm_root,
    )
    from titan_hcl.logic.memory_state_specs import MEMORY_STATE_SPEC

    shm_root = ensure_shm_root(titan_id)
    reader = StateRegistryReader(MEMORY_STATE_SPEC, shm_root)

    def _read_memory_state() -> dict:
        try:
            raw = reader.read_variable()
            if raw is None:
                return {}
            decoded = msgpack.unpackb(raw, raw=False)
            return decoded if isinstance(decoded, dict) else {}
        except Exception:
            return {}

    async def _v3_learning_velocity() -> float:
        return float(_read_memory_state().get("learning_velocity", 0.5) or 0.5)

    async def _v3_directive_alignment() -> float:
        return float(_read_memory_state().get("directive_alignment", 0.5) or 0.5)

    async def _v3_social_density() -> float:
        # Mirror the pre-extraction inline math at plugin.py:1656-1658 —
        # rough estimate from persistent memory count, capped at 1.0.
        persistent = int(_read_memory_state().get("persistent_count", 0) or 0)
        return min(1.0, persistent / 100.0)

    return {
        "get_learning_velocity": _v3_learning_velocity,
        "get_directive_alignment": _v3_directive_alignment,
        "get_social_density": _v3_social_density,
    }


def _init_metabolism(config: dict, titan_id: str):
    """Construct MetabolismController instance.

    Mirrors plugin.py:_wire_metabolism — same kernel-side soul/network
    handles, but constructed INSIDE the worker process so the
    controller is self-contained here (G21 single-owner). The pre-
    extraction MemoryProxy + SocialGraphProxy bridges are replaced by
    direct SHM reads of memory_state.bin + social_graph_state.bin per
    G18 (subprocess context — proxies need DivineBus + Guardian which
    don't exist here).

    Returns MetabolismController or None on init failure.
    """
    growth_cfg = (get_params("growth_metrics") or {})

    # ── Soul + Network (kernel handles needed by MetabolismController) ──
    #
    # `soul` is stored on the controller (`self.soul = soul`) but never
    # accessed afterwards in titan_hcl/core/metabolism.py — verified
    # by grep of `self\.soul\.`. So we can safely pass None.
    #
    # `network` is used for `await self.network.get_balance()` at
    # metabolism.py:155 + :354 — must be a working Network instance
    # with Solana RPC client.
    soul = None

    try:
        from titan_hcl.core.network import HybridNetworkClient as Network
        network_cfg = (get_params("network") or {})
        # HybridNetworkClient.__init__(self, mood_engine=None, config=None) —
        # it takes NO `soul` kwarg, and `config` is the 2nd param. The prior
        # call `Network(network_cfg, soul=soul)` bound network_cfg to
        # mood_engine and passed an invalid `soul=` kwarg → TypeError crashed
        # _init_metabolism every boot fleet-wide. metabolism only needs
        # network.get_balance(), which uses config (RPC urls). Pass config only.
        network = Network(config=network_cfg)
    except Exception as e:
        logger.error(
            "[MetabolismWorker] Network init failed: %s — controller "
            "can still boot in degraded mode (get_balance will raise; "
            "tier falls back to last cached value)", e, exc_info=True)
        network = None

    # ── MetabolismController ──────────────────────────────────────────
    try:
        from titan_hcl.core.metabolism import MetabolismController
        metabolism = MetabolismController(
            soul=soul,
            network=network,
            memory=None,         # growth metrics replaced via monkey-patch below
            config=growth_cfg,
            social_graph=None,   # social_density replaced via monkey-patch below
        )
        overrides = _build_growth_overrides(titan_id=titan_id)
        for attr, fn in overrides.items():
            setattr(metabolism, attr, fn)
        logger.info(
            "[MetabolismWorker] MetabolismController booted (growth metrics "
            "via memory_state.bin SHM, social_density via persistent_count "
            "from same slot)")
    except Exception as e:
        logger.error(
            "[MetabolismWorker] MetabolismController init failed: %s",
            e, exc_info=True)
        return None

    return metabolism


# ── Ring-buffer + tier-history (in-memory, worker-owned per G19-strict) ─


class _GateDecisionRing:
    """Bounded ring of last N evaluate_gate decisions.

    Authoritative writer per Maker-locked 2026-05-14 — no client-side
    feature→bool lookup. Each entry: {feature, proceed, rate_mult,
    caller, tier, reason, ts}.
    """

    def __init__(self, max_size: int = _RING_BUFFER_MAX):
        self._dq: deque = deque(maxlen=max_size)
        self._lock = threading.Lock()

    def record(self, entry: dict) -> None:
        with self._lock:
            self._dq.append(entry)

    def snapshot(self, limit: int = 64) -> list[dict]:
        with self._lock:
            data = list(self._dq)
        return data[-limit:] if limit > 0 else data

    def __len__(self) -> int:
        with self._lock:
            return len(self._dq)


class _TierHistory:
    """Bounded tier-transition history (last 24h, in-memory)."""

    def __init__(self, window_s: float = _TIER_HISTORY_WINDOW_S):
        self._dq: deque = deque()
        self._window_s = window_s
        self._lock = threading.Lock()

    def record(self, tier_from: str, tier_to: str, ts: float,
               balance_pct: float, gates_enforced: bool) -> None:
        with self._lock:
            self._dq.append({
                "tier_from": tier_from,
                "tier_to": tier_to,
                "ts": ts,
                "balance_pct": balance_pct,
                "gates_enforced": gates_enforced,
            })
            self._gc_locked(now=ts)

    def _gc_locked(self, now: float) -> None:
        cutoff = now - self._window_s
        while self._dq and self._dq[0]["ts"] < cutoff:
            self._dq.popleft()

    def snapshot(self) -> list[dict]:
        with self._lock:
            self._gc_locked(now=time.time())
            return list(self._dq)


# ── Main entry ────────────────────────────────────────────────────────


@with_error_envelope(module_name="metabolism", subsystem="entry", severity=_phase11_sev.FATAL)
def metabolism_worker_main(recv_queue, send_queue, name: str,
                           config: dict) -> None:
    """Main loop for the metabolism_worker subprocess.

    Hosts MetabolismController + serves bus.QUERY work-RPC dispatch +
    publishes metabolism_state.bin SHM slot at 1 Hz via dedicated thread.
    """
    # === BOILERPLATE: spawn-mode sys.path bootstrap ===
    project_root = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # === BOILERPLATE: Phase B.2 §C7 socket-mode bus client setup ===
    # LIFE_FORCE_UPDATED constant ships in v1.8.3 §4.G — runtime gate
    # retired (was: hasattr(bus, "LIFE_FORCE_UPDATED") check).
    from titan_hcl.core.worker_bus_bootstrap import setup_worker_bus
    try:
        recv_queue, send_queue, _bus_client = setup_worker_bus(
            name, recv_queue, send_queue,
            topics=list(_METABOLISM_WORKER_SUBSCRIBE_TOPICS),
        )
    except Exception as _err:
        logger.error(
            "[MetabolismWorker] setup_worker_bus failed: %s — exiting",
            _err, exc_info=True)
        return

    # === BOILERPLATE: pdeathsig installation ===
    try:
        from titan_hcl.core.worker_lifecycle import install_parent_death_signal
        install_parent_death_signal()
    except Exception as _err:
        logger.debug(
            "[MetabolismWorker] pdeathsig install skipped: %s", _err)

    # Phase 11 §11.I.5 (Chunk 11N) — reset module-level readiness sentinel
    # on every entry (fork-mode re-entries inherit parent's True;
    # spawn-mode re-spawns get fresh False; explicit reset covers both).
    global _WORKER_READY, _BOOT_DEADLINE
    _WORKER_READY = False
    _BOOT_DEADLINE = boot_deadline_from_now()

    from titan_hcl.core.state_registry import resolve_titan_id
    titan_id = (
        (get_params("info_banner") or {}).get("titan_id")
        or resolve_titan_id()
    )
    boot_ts = time.time()

    logger.info(
        "[MetabolismWorker] Booting (titan_id=%s) — rFP §4.J + D-SPEC-51",
        titan_id)

    # ── Phase 11 §11.I.5 / Chunk 11N — SHM state-slot writer (G21) ──
    # Built BEFORE the slow MetabolismController init so titan_hcl's 1Hz
    # SHM poll sees the worker is alive while it warms.
    _state_writer = None
    try:
        from titan_hcl.core.module_state import (
            BootPriority,
            ModuleStateWriter,
        )
        _state_writer = ModuleStateWriter(
            module_name=name,
            layer="L2",
            boot_priority=BootPriority.OPTIONAL_POST_BOOT,
        )
        _state_writer.write_state("starting")
    except Exception as _sw_err:  # noqa: BLE001
        logger.warning(
            "[MetabolismWorker] Phase 11 ModuleStateWriter init failed "
            "(continuing — SHM slot disabled): %s", _sw_err)

    # === MODULE-SPECIFIC: MetabolismController init ===
    metabolism = _init_metabolism(config, titan_id=titan_id)
    if metabolism is None:
        logger.error(
            "[MetabolismWorker] MetabolismController init failed — exiting "
            "non-zero so guardian respawns")
        sys.exit(1)

    # === MODULE-SPECIFIC: ring buffer + tier history ===
    gate_ring = _GateDecisionRing(max_size=_RING_BUFFER_MAX)
    tier_history = _TierHistory(window_s=_TIER_HISTORY_WINDOW_S)
    last_published_tier = {"tier": str(metabolism.get_metabolic_tier())}

    # === MODULE-SPECIFIC: SHM publisher init ===
    state_publisher = None
    try:
        from titan_hcl.logic.metabolism_state_publisher import (
            MetabolismStatePublisher,
        )
        state_publisher = MetabolismStatePublisher(titan_id=titan_id)
    except Exception as _shm_err:
        logger.error(
            "[MetabolismWorker] MetabolismStatePublisher BOOT FAILED — "
            "worker continues without SHM visibility (consumers see "
            "absent slot + use cold defaults): %s",
            _shm_err, exc_info=True)

    # ── Phase 11 §11.I.2 — slot transition: starting → booted ─────────
    # Replaces the legacy MODULE_READY bus emit per locked D2 — the SHM
    # slot state=booted is the contract. /v4/metabolism endpoint reads
    # metabolism_state.bin SHM with cold-default tolerance.
    _WORKER_READY = True
    if _state_writer is not None:
        try:
            _state_writer.write_state("booted")
            logger.info(
                "[MetabolismWorker] Phase 11 §11.I.2 — SHM slot state=booted "
                "(awaiting MODULE_PROBE_REQUEST from titan_hcl)")
        except Exception as _swb_err:  # noqa: BLE001
            logger.warning(
                "[MetabolismWorker] Phase 11 write_state(booted) failed: %s",
                _swb_err)

    # === MODULE-SPECIFIC: 1 Hz SHM publisher thread + tier-transition detector ===
    _periodic_stop = threading.Event()
    last_gate_reason_box = {"reason": ""}  # mutable closure cell

    def _periodic_publish_loop():
        last_shm = 0.0
        last_stats_notify = 0.0
        last_balance_refresh = 0.0
        social_gravity_score = 0.0
        while not _periodic_stop.is_set():
            try:
                now = time.time()

                # SOL balance refresh (also triggered by SOL_BALANCE_CHANGED
                # bus event in the main loop; this is the fallback cadence).
                if now - last_balance_refresh > _BALANCE_REFRESH_INTERVAL_S:
                    _refresh_balance_and_tier(
                        metabolism, tier_history, send_queue, name,
                        last_published_tier
                    )
                    last_balance_refresh = now
                    # social_gravity_score: forward-looking field in the
                    # SHM schema (plugin.py:1629 stale comment refers to a
                    # method that doesn't exist on the controller today).
                    # Stays 0.0 until a producer for it is identified.
                    pass

                # SHM publish (1Hz)
                if state_publisher is not None and \
                        now - last_shm > _SHM_PUBLISH_INTERVAL_S:
                    try:
                        state_publisher.publish(
                            metabolism,
                            social_gravity_score=social_gravity_score,
                            last_gate_decision_reason=last_gate_reason_box["reason"],
                        )
                    except Exception as _shm_err:
                        logger.warning(
                            "[MetabolismWorker] state publish raised at "
                            "top level: %s", _shm_err, exc_info=True)
                    last_shm = now

                # 1Hz METABOLIC_STATS_UPDATED notification (bulk via SHM)
                if now - last_stats_notify > _STATS_NOTIFY_INTERVAL_S:
                    _send_msg(
                        send_queue, bus.METABOLIC_STATS_UPDATED, name,
                        "all", {"ts": now},
                    )
                    last_stats_notify = now
            except Exception as _per_err:
                logger.warning(
                    "[MetabolismWorker] periodic publish thread error: %s",
                    _per_err)
            _periodic_stop.wait(0.5)

    _periodic_thread = threading.Thread(
        target=_periodic_publish_loop,
        daemon=True,
        name="metabolism-periodic-publish",
    )
    _periodic_thread.start()

    # === Main recv loop ===
    last_heartbeat = time.time()
    while True:
        now = time.time()
        if now - last_heartbeat > _HEARTBEAT_INTERVAL_S:
            _send_heartbeat(send_queue, name, extra={
                "tier": last_published_tier["tier"],
                "gate_ring_size": len(gate_ring),
            }, state_writer=_state_writer)
            last_heartbeat = now

        try:
            msg = recv_queue.get(timeout=_POLL_INTERVAL_S)
        except Empty:
            continue
        except (KeyboardInterrupt, SystemExit):
            break

        # B.2.1 supervision-transfer dispatch
        try:
            from titan_hcl.core import worker_swap_handler as _swap
            if _swap.maybe_dispatch_swap_msg(msg):
                continue
        except Exception:
            pass

        msg_type = msg.get("type", "")

        # ── Phase 11 §11.I.3 — MODULE_PROBE_REQUEST handler ────────
        if msg_type == bus.MODULE_PROBE_REQUEST:
            try:
                from titan_hcl.core.probe_dispatcher import (
                    handle_module_probe_request,
                )
                handle_module_probe_request(
                    msg,
                    probe_fn=None,
                    send_queue=send_queue,
                    module_name=name,
                    state_writer=_state_writer,
                )
            except Exception as _probe_err:  # noqa: BLE001
                logger.warning(
                    "[MetabolismWorker] MODULE_PROBE_REQUEST handler "
                    "failed: %s", _probe_err)
            continue

        if msg_type == bus.MODULE_SHUTDOWN:
            logger.info(
                "[MetabolismWorker] Shutdown: %s",
                msg.get("payload", {}).get("reason"))
            break

        if msg_type == bus.SAVE_NOW:
            # No persistent DB to checkpoint; log ring summary for forensic value.
            logger.info(
                "[MetabolismWorker] SAVE_NOW — gate_ring_size=%d "
                "last_tier=%s",
                len(gate_ring), last_published_tier["tier"])
            continue

        if msg_type == bus.SOLANA_BALANCE_UPDATED:
            _refresh_balance_and_tier(
                metabolism, tier_history, send_queue, name,
                last_published_tier
            )
            continue

        if msg_type == bus.LIFE_FORCE_UPDATED:
            # Cross-worker bridge from §4.G v1.8.3 — life_force_worker
            # publishes the 1Hz notification; bulk payload lives in
            # life_force_state.bin SHM slot (G18 sub-µs read available
            # via LifeForceShmReader). Currently no-op here (consumed
            # for future tier-weighting math). Producer is live (the
            # §4.J SOFT-dep finally has its event-source).
            continue

        if msg_type == bus.QUERY:
            _handle_query(
                msg, metabolism, gate_ring, tier_history,
                last_published_tier, last_gate_reason_box,
                send_queue, name,
            )
            continue

        logger.debug(
            "[MetabolismWorker] Unhandled msg_type=%s — ignoring", msg_type)

    # === Clean shutdown ===
    logger.info(
        "[MetabolismWorker] Exiting — stopping publisher thread "
        "(gate_ring_size=%d last_tier=%s)",
        len(gate_ring), last_published_tier["tier"])
    _periodic_stop.set()
    logger.info("[MetabolismWorker] Exit complete")


# ── Tier-transition detector ──────────────────────────────────────────


def _refresh_balance_and_tier(metabolism, tier_history, send_queue, name,
                              last_published_tier: dict) -> None:
    """Re-read SOL balance + re-evaluate tier; publish METABOLIC_TIER_CHANGED
    if the tier transitioned.

    Called from:
      - SOL_BALANCE_CHANGED bus event (responsive path)
      - periodic thread @ _BALANCE_REFRESH_INTERVAL_S (fallback cadence)
    """
    try:
        prev_tier = str(last_published_tier.get("tier") or "HEALTHY")
        # _last_balance_pct is a @property (cached pct from last balance fetch)
        balance_pct = float(metabolism._last_balance_pct)
        new_tier = str(metabolism.get_metabolic_tier())
        if new_tier != prev_tier:
            now = time.time()
            try:
                gates_enforced = bool(metabolism.get_gates_enforced())
            except Exception:
                gates_enforced = False
            tier_history.record(
                tier_from=prev_tier,
                tier_to=new_tier,
                ts=now,
                balance_pct=balance_pct,
                gates_enforced=gates_enforced,
            )
            last_published_tier["tier"] = new_tier
            _send_msg(
                send_queue, bus.METABOLIC_TIER_CHANGED, name, "all",
                {
                    "tier_from": prev_tier,
                    "tier_to": new_tier,
                    "balance_pct": balance_pct,
                    "gates_enforced": gates_enforced,
                    "ts": now,
                },
            )
            logger.info(
                "[MetabolismWorker] METABOLIC_TIER_CHANGED %s → %s "
                "(balance_pct=%.3f, gates_enforced=%s)",
                prev_tier, new_tier, balance_pct, gates_enforced)
    except Exception as e:
        logger.warning(
            "[MetabolismWorker] tier-refresh failed: %s", e)


# ── Action dispatch ───────────────────────────────────────────────────


def _handle_query(
    msg: dict,
    metabolism,
    gate_ring: _GateDecisionRing,
    tier_history: _TierHistory,
    last_published_tier: dict,
    last_gate_reason_box: dict,
    send_queue,
    name: str,
) -> None:
    """Dispatch QUERY action to the appropriate MetabolismController method.

    Every action listed here corresponds to a row in
    phase_c_rpc_exemptions.yaml::work_rpc_sites under metabolism_proxy:.
    All actions return RESPONSE payloads when rid is present.

    Per G19: each call is bounded by caller timeout (≤5s on proxy side).
    MetabolismController methods are sub-second on a healthy stack
    (SOL balance read can stall on RPC; tier eval is in-memory).
    """
    import asyncio

    payload = msg.get("payload", {}) or {}
    action = payload.get("action", "")
    rid = msg.get("rid")
    src = msg.get("src", "")

    try:
        # ── Hot reads (also exposed via SHM, kept here for bus-RPC path) ──
        if action == "get_metabolic_tier":
            tier = str(metabolism.get_metabolic_tier())
            _send_response(send_queue, name, src, {"result": tier}, rid)
            return

        if action == "get_gates_enforced":
            enforced = bool(metabolism.get_gates_enforced())
            _send_response(send_queue, name, src, {"result": enforced}, rid)
            return

        if action == "get_tier_info":
            info = metabolism.get_tier_info()
            _send_response(send_queue, name, src, {"result": info}, rid)
            return

        if action == "get_last_gate_decision_reason":
            reason = str(metabolism.get_last_gate_decision_reason() or "")
            _send_response(send_queue, name, src, {"result": reason}, rid)
            return

        # ── Gate decision (authoritative ring-buffer writer per
        #    Maker-locked 2026-05-14 G19-strict) ──────────────────────
        if action == "evaluate_gate":
            feature = str(payload.get("feature", ""))
            caller = str(payload.get("caller", ""))
            proceed, rate_mult = metabolism.evaluate_gate(feature, caller=caller)
            reason = str(metabolism.get_last_gate_decision_reason() or "")
            tier = str(metabolism.get_metabolic_tier())
            now = time.time()
            entry = {
                "feature": feature,
                "proceed": bool(proceed),
                "rate_mult": float(rate_mult),
                "caller": caller,
                "tier": tier,
                "reason": reason,
                "ts": now,
            }
            gate_ring.record(entry)
            last_gate_reason_box["reason"] = reason
            _send_msg(
                send_queue, bus.GATE_DECISION_RECORDED, name, "all", entry,
            )
            _send_response(
                send_queue, name, src,
                {"result": [bool(proceed), float(rate_mult)]},
                rid,
            )
            return

        if action == "get_gate_decision_summary":
            summary = metabolism.get_gate_decision_summary()
            _send_response(send_queue, name, src, {"result": summary}, rid)
            return

        if action == "get_emergency_duration":
            dur = float(metabolism.get_emergency_duration())
            _send_response(send_queue, name, src, {"result": dur}, rid)
            return

        if action == "can_use_feature":
            feature = str(payload.get("feature", ""))
            allowed = bool(metabolism.can_use_feature(feature))
            _send_response(send_queue, name, src, {"result": allowed}, rid)
            return

        if action == "get_service_gate":
            feature = str(payload.get("feature", ""))
            ok, rate_mult, reason = metabolism.get_service_gate(feature)
            _send_response(
                send_queue, name, src,
                {"result": [bool(ok), float(rate_mult), str(reason)]},
                rid,
            )
            return

        # ── Async actions (run in worker's own loop) ───────────────────
        if action == "get_current_state":
            state = asyncio.run(metabolism.get_current_state())
            _send_response(send_queue, name, src, {"result": str(state)}, rid)
            return

        if action == "can_afford":
            cost = float(payload.get("cost", 0.0))
            can = asyncio.run(metabolism.can_afford(cost))
            _send_response(send_queue, name, src, {"result": bool(can)}, rid)
            return

        if action == "get_metabolic_health":
            health = asyncio.run(metabolism.get_metabolic_health())
            _send_response(send_queue, name, src, {"result": float(health)}, rid)
            return

        if action == "get_learning_velocity":
            lv = asyncio.run(metabolism.get_learning_velocity())
            _send_response(send_queue, name, src, {"result": float(lv)}, rid)
            return

        if action == "get_directive_alignment":
            da = asyncio.run(metabolism.get_directive_alignment())
            _send_response(send_queue, name, src, {"result": float(da)}, rid)
            return

        if action == "get_social_density":
            sd = asyncio.run(metabolism.get_social_density())
            _send_response(send_queue, name, src, {"result": float(sd)}, rid)
            return

        # ── Diagnostic reads (gate-history / tier-history) ────────────
        if action == "get_gate_ring":
            limit = int(payload.get("limit", 64))
            data = gate_ring.snapshot(limit=limit)
            _send_response(send_queue, name, src, {"result": data}, rid)
            return

        if action == "get_tier_history":
            data = tier_history.snapshot()
            _send_response(send_queue, name, src, {"result": data}, rid)
            return

        # ── Unknown action ────────────────────────────────────────────
        logger.warning(
            "[MetabolismWorker] Unknown action: %s (payload=%s)",
            action, payload)
        _send_response(
            send_queue, name, src,
            {"error": f"unknown action: {action}"},
            rid,
        )

    except Exception as e:
        logger.error(
            "[MetabolismWorker] action=%s raised: %s",
            action, e, exc_info=True)
        _send_response(
            send_queue, name, src, {"error": str(e)}, rid,
        )
