"""social_graph_worker — Python L2 module hosting the SocialGraph instance.

Per rFP_titan_hcl_l2_separation_strategy.md §4.P (LOCKED 2026-05-05;
SHIPPED 2026-05-14) + SPEC v1.7.1 §9.B social_graph_worker block +
D-SPEC-50.

ACTIVE: always-on autostart. No flag-gate (replaces the legacy
`_proxies["social_graph"] = _proxies["mind"]` alias rot — Maker
2026-05-12: *"no right to be in microkernel Phase C architecture that
must be lean and fast"*).

Owns:
  - SocialGraph (titan_hcl/core/social_graph.py — 9 SQLite tables,
    50 methods: Phase 13 Sage Socialite — cross-session user profiles,
    interaction scoring, social edges, donations/inspirations, X
    engagement ledger, per-Titan social preferences, community registry).
  - data/social_graph.db (WAL mode; G16 critical-data; routed writes via
    social_graph_writer IMW daemon when [persistence_social_graph]
    enabled — existing pattern preserved).
  - social_graph_state.bin SHM slot (G21 single-writer; 1 Hz; payload
    {users/edges/donations/total_donated_sol/inspirations/engagement_
    ledger_today/schema_version/ts} per SPEC §7.1 v1.7.1).
  - SocialGraphProxy dispatch handler (~25 actions per
    phase_c_rpc_exemptions.yaml::work_rpc_sites — 14 writes + 11
    parameterized reads, all ≤5s per G19).

Bus subscriptions:
  REQUIRED — bus.QUERY (dst=social_graph) for SocialGraphProxy dispatch
             + MODULE_SHUTDOWN + SAVE_NOW
  OPTIONAL — SWAP_HANDOFF / ADOPTION_REQUEST (B.2.1 supervision-transfer)

Bus publications:
  - SOCIAL_GRAPH_READY                  (once on boot)
  - SOCIAL_GRAPH_STATS_UPDATED          (5s coalesced; bulk via SHM)
  - SOCIAL_INTERACTION_RECORDED         (per record_interaction write)
  - SOCIAL_DONATION_RECORDED            (per record_donation write)
  - SOCIAL_INSPIRATION_RECORDED         (per record_inspiration write)
  - MODULE_HEARTBEAT / MODULE_SHUTDOWN (Phase 11 §11.I.2 D2:
    legacy boot-signal emit DELETED — SHM slot state=booted is the contract)

Closes BUG-MINDPROXY-MISSING-RECORD-INTERACTION-ASYNC-20260514 (chat
post-hook AttributeError fleet-wide silently breaking KnownUserResolver
familiarity scoring; root cause = MindProxy alias only exposed sync
subset of the SocialGraph API).

Closes G22 violation: mind_worker's `get_social_stats` orphan-handler
(documented in phase_c_rpc_exemptions.yaml::orphan_handler_allowlist
with rationale "SocialGraph stats; full migration deferred"). Stats now
read from social_graph_state.bin SHM directly per G18.

ARG ORDER per cognitive_worker template note: Guardian-spawned L2
workers follow ``(recv_queue, send_queue, name, config)``. Stale
docstring in worker_bus_bootstrap.py shows wrong order — do not follow.

Implementation reference: cognitive_worker.py (CANONICAL L2 WORKER
TEMPLATE) for `=== BOILERPLATE ===` sections (spawn-mode sys.path
bootstrap, setup_worker_bus, pdeathsig install, Phase 11 SHM-slot
state="booted", heartbeat) + memory_worker.py `_periodic_publish_loop`
for the 1Hz SHM publisher thread.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time
from queue import Empty
from typing import Any, Optional

from titan_hcl import bus
from titan_hcl.core.module_error_handler import with_error_envelope
from titan_hcl.errors import Severity as _phase11_sev

logger = logging.getLogger(__name__)

# Module name (matches Guardian registry per SPEC §9.B v1.7.1).
MODULE_NAME = "social_graph"

# Cadence + lifecycle constants.
_HEARTBEAT_INTERVAL_S = 10.0            # SPEC §10.B MODULE_HEARTBEAT_INTERVAL_S
_POLL_INTERVAL_S = 0.2                  # recv loop poll cadence
_SHM_PUBLISH_INTERVAL_S = 1.0           # social_graph_state.bin 1 Hz per SPEC §7.1
_STATS_NOTIFY_INTERVAL_S = 5.0          # SOCIAL_GRAPH_STATS_UPDATED bus notification cadence

# === MODULE-SPECIFIC: subscribe topics list ===
# social_graph_worker is primarily a work-RPC server: bus.QUERY messages
# arrive with dst="social_graph" routed by the broker. Beyond that, we
# only need lifecycle topics. ADOPTION_REQUEST / SWAP_HANDOFF handled by
# the shared worker_swap_handler boilerplate (delegated below).
_SOCIAL_GRAPH_WORKER_SUBSCRIBE_TOPICS: list[str] = [
    bus.QUERY,                  # SocialGraphProxy dispatch (dst=social_graph)
    bus.MODULE_SHUTDOWN,        # clean shutdown
    bus.SAVE_NOW,               # B.1 shadow_swap orchestrator (WAL checkpoint)
    bus.MODULE_PROBE_REQUEST,   # Phase 11 §11.I.3 probe handler
]


# Phase 11 §11.I.5 (Chunk 11N) — module-level readiness sentinel; gates
# SHM-slot heartbeat() (legacy bus heartbeat fires unconditionally for
# the boot window so guardian_HCL's stale-heartbeat detector doesn't
# kill a slow boot).
from titan_hcl.modules._heartbeat_grace import (
    boot_deadline_from_now, shm_heartbeat_allowed,
)

_WORKER_READY: bool = False
_BOOT_DEADLINE = None  # boot-grace deadline (monotonic); None=no grace


# ── Lifecycle helpers ─────────────────────────────────────────────────


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
                    state_writer: Optional[object] = None) -> None:
    """Emit MODULE_HEARTBEAT to guardian_HCL with current RSS.

    Phase 11 §11.I.5: also publishes state_writer.heartbeat() on the SHM
    slot once _WORKER_READY is True. SHM writes are best-effort.
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
        except Exception:  # noqa: BLE001 — never crash the heartbeat
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
            "[SocialGraphWorker] send_response enqueue failed (rid=%s): %s",
            rid, e)


# ── SocialGraph init ──────────────────────────────────────────────────


def _resolve_db_path(config: dict) -> str:
    """Resolve data/social_graph.db against project root."""
    sg_cfg = (config.get("social_graph", {}) or {})
    raw = sg_cfg.get("db_path", "data/social_graph.db")
    if os.path.isabs(raw):
        return raw
    project_root = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(project_root, raw)


def _init_social_graph(db_path: str):
    """Construct SocialGraph instance with WAL + boot integrity check.

    Returns SocialGraph or None on init failure (worker exits non-zero
    so guardian knows). G16 boot-time integrity check is implicit via
    SQLite's _init_db (CREATE TABLE IF NOT EXISTS); explicit PRAGMA
    integrity_check happens on read errors per the SocialGraph
    `_route_write` fallback path.
    """
    try:
        from titan_hcl.core.social_graph import SocialGraph
        sg = SocialGraph(db_path=db_path)
        logger.info(
            "[SocialGraphWorker] SocialGraph booted (db=%s, "
            "writer_routing=%s)",
            db_path, "via social_graph_writer" if sg._writer else "direct")
        return sg
    except Exception as e:
        logger.error(
            "[SocialGraphWorker] SocialGraph init failed (db=%s): %s",
            db_path, e, exc_info=True)
        return None


# ── Main entry ────────────────────────────────────────────────────────


@with_error_envelope(module_name="social_graph", subsystem="entry", severity=_phase11_sev.FATAL)
def social_graph_worker_main(recv_queue, send_queue, name: str,
                             config: dict) -> None:
    """Main loop for the social_graph_worker subprocess.

    Hosts SocialGraph + serves bus.QUERY work-RPC dispatch + publishes
    social_graph_state.bin SHM slot at 1 Hz via dedicated thread.
    """
    # === BOILERPLATE: spawn-mode sys.path bootstrap ===
    project_root = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # === BOILERPLATE: Phase B.2 §C7 socket-mode bus client setup ===
    from titan_hcl.core.worker_bus_bootstrap import setup_worker_bus
    try:
        recv_queue, send_queue, _bus_client = setup_worker_bus(
            name, recv_queue, send_queue,
            topics=_SOCIAL_GRAPH_WORKER_SUBSCRIBE_TOPICS,
        )
    except Exception as _err:
        logger.error(
            "[SocialGraphWorker] setup_worker_bus failed: %s — exiting",
            _err, exc_info=True)
        return

    # === BOILERPLATE: pdeathsig installation ===
    try:
        from titan_hcl.core.worker_lifecycle import install_parent_death_signal
        install_parent_death_signal()
    except Exception as _err:
        logger.debug(
            "[SocialGraphWorker] pdeathsig install skipped: %s", _err)

    global _WORKER_READY, _BOOT_DEADLINE
    _WORKER_READY = False
    _BOOT_DEADLINE = boot_deadline_from_now()

    from titan_hcl.core.state_registry import resolve_titan_id
    titan_id = (
        (config.get("info_banner", {}) or {}).get("titan_id")
        or resolve_titan_id()
    )
    boot_ts = time.time()

    logger.info(
        "[SocialGraphWorker] Booting (titan_id=%s) — rFP §4.P + D-SPEC-50",
        titan_id)

    # ── Phase 11 §11.I.5 (Chunk 11N) — SHM state-slot writer ──
    # Constructed BEFORE the slow SocialGraph init so the slot publishes
    # state="starting" immediately.
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
            "[SocialGraphWorker] Phase 11 ModuleStateWriter init failed: %s",
            _sw_err)

    # === MODULE-SPECIFIC: SocialGraph init ===
    db_path = _resolve_db_path(config)
    social_graph = _init_social_graph(db_path)
    if social_graph is None:
        logger.error(
            "[SocialGraphWorker] SocialGraph init failed — exiting non-zero "
            "so guardian respawns")
        sys.exit(1)

    # === MODULE-SPECIFIC: SHM publisher init ===
    state_publisher = None
    try:
        from titan_hcl.logic.social_graph_state_publisher import (
            SocialGraphStatePublisher,
        )
        state_publisher = SocialGraphStatePublisher(titan_id=titan_id)
    except Exception as _shm_err:
        logger.error(
            "[SocialGraphWorker] SocialGraphStatePublisher BOOT FAILED — "
            "worker continues without SHM visibility (consumers see "
            "absent slot + use cold defaults): %s",
            _shm_err, exc_info=True)

    # ── Phase 11 §11.I.2 — slot transition: starting → booted ──
    # (legacy boot-signal bus emit deleted per locked D2 / no-shim policy)
    _WORKER_READY = True
    if _state_writer is not None:
        try:
            _state_writer.write_state("booted")
        except Exception as _swb_err:  # noqa: BLE001
            logger.warning(
                "[SocialGraphWorker] Phase 11 write_state(booted) failed: %s",
                _swb_err)
    _send_msg(send_queue, bus.SOCIAL_GRAPH_READY, name, "all", {
        "titan_id": titan_id, "ts": boot_ts,
    })

    # === MODULE-SPECIFIC: 1 Hz SHM publisher thread + 5s STATS_UPDATED notify ===
    _periodic_stop = threading.Event()

    def _periodic_publish_loop():
        last_shm = 0.0
        last_stats_notify = 0.0
        while not _periodic_stop.is_set():
            try:
                now = time.time()
                if state_publisher is not None and \
                        now - last_shm > _SHM_PUBLISH_INTERVAL_S:
                    try:
                        state_publisher.publish(social_graph)
                    except Exception as _shm_err:
                        logger.warning(
                            "[SocialGraphWorker] state publish raised at "
                            "top level: %s", _shm_err, exc_info=True)
                    last_shm = now
                if now - last_stats_notify > _STATS_NOTIFY_INTERVAL_S:
                    _send_msg(
                        send_queue, bus.SOCIAL_GRAPH_STATS_UPDATED, name,
                        "all", {"ts": now},
                    )
                    last_stats_notify = now
            except Exception as _per_err:
                logger.warning(
                    "[SocialGraphWorker] periodic publish thread error: %s",
                    _per_err)
            _periodic_stop.wait(0.5)

    _periodic_thread = threading.Thread(
        target=_periodic_publish_loop,
        daemon=True,
        name="social-graph-periodic-publish",
    )
    _periodic_thread.start()

    # === Main recv loop ===
    last_heartbeat = time.time()
    while True:
        now = time.time()
        if now - last_heartbeat > _HEARTBEAT_INTERVAL_S:
            _send_heartbeat(send_queue, name, state_writer=_state_writer)
            last_heartbeat = now

        try:
            msg = recv_queue.get(timeout=_POLL_INTERVAL_S)
        except Empty:
            continue
        except (KeyboardInterrupt, SystemExit):
            break

        msg_type = msg.get("type", "")

        # ── Phase 11 §11.I.3 — MODULE_PROBE_REQUEST handler ──
        if msg_type == bus.MODULE_PROBE_REQUEST and _state_writer is not None:
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
                    "[SocialGraphWorker] MODULE_PROBE_REQUEST handler failed: %s",
                    _probe_err)
            continue

        # B.2.1 supervision-transfer dispatch
        try:
            from titan_hcl.core import worker_swap_handler as _swap
            if _swap.maybe_dispatch_swap_msg(msg):
                continue
        except Exception:
            pass  # swap handler not installed — fine

        if msg_type == bus.MODULE_SHUTDOWN:
            logger.info(
                "[SocialGraphWorker] Shutdown: %s",
                msg.get("payload", {}).get("reason"))
            break

        if msg_type == bus.SAVE_NOW:
            _checkpoint_wal(social_graph)
            continue

        if msg_type == bus.QUERY:
            _handle_query(msg, social_graph, send_queue, name)
            continue

        logger.debug(
            "[SocialGraphWorker] Unhandled msg_type=%s — ignoring",
            msg_type)

    # === Clean shutdown ===
    logger.info(
        "[SocialGraphWorker] Exiting — stopping publisher thread + WAL checkpoint")
    _periodic_stop.set()
    _checkpoint_wal(social_graph)
    logger.info("[SocialGraphWorker] Exit complete")


def _checkpoint_wal(social_graph) -> None:
    """G16: explicit PRAGMA wal_checkpoint(FULL) on shutdown / SAVE_NOW.

    Direct connection to the canonical DB (NOT via writer client) because
    checkpoint is a session-bound operation, not a write to persist.
    """
    try:
        import sqlite3
        with sqlite3.connect(social_graph._db_path, timeout=5) as conn:
            conn.execute("PRAGMA wal_checkpoint(FULL)")
        logger.info(
            "[SocialGraphWorker] WAL checkpoint complete (db=%s)",
            social_graph._db_path)
    except Exception as e:
        logger.warning(
            "[SocialGraphWorker] WAL checkpoint failed: %s",
            e, exc_info=True)


# ── Action dispatch ───────────────────────────────────────────────────


def _serialize_profile(profile) -> dict:
    """Serialize a UserProfile (or its _data dict) to a flat bus-payload dict."""
    if profile is None:
        return {}
    if hasattr(profile, "to_dict"):
        try:
            return profile.to_dict()
        except Exception:
            pass
    if isinstance(profile, dict):
        return dict(profile)
    # Best-effort attribute pull (defensive)
    keys = (
        "user_id", "platform", "display_name", "sol_address",
        "first_seen", "last_seen", "interaction_count", "like_score",
        "dislike_score", "total_donated_sol", "engagement_level", "notes",
    )
    return {k: getattr(profile, k, None) for k in keys}


def _profile_from_dict(payload_profile: dict):
    """Build a UserProfile from a bus-payload dict (for _save_profile path)."""
    from titan_hcl.core.social_graph import UserProfile
    if not isinstance(payload_profile, dict):
        return None
    return UserProfile(payload_profile)


def _handle_query(msg: dict, social_graph, send_queue, name: str) -> None:
    """Dispatch QUERY action to the appropriate SocialGraph method.

    Every action listed here corresponds to a row in
    phase_c_rpc_exemptions.yaml::work_rpc_sites under
    social_graph_proxy:. Reads return RESPONSE payloads with the
    computed shape; writes return RESPONSE {"ok": True} when rid is
    present (fire-and-forget callers omit rid and skip the response).

    Per G19: each call is bounded by the caller's timeout (≤5s on the
    proxy side). The handler itself does NOT enforce timeouts — the
    SocialGraph methods are sub-second on a healthy DB. Defensive
    error handling per `directive_error_visibility.md`: surface as
    RESPONSE {"error": str} when rid present; log otherwise.
    """
    payload = msg.get("payload", {}) or {}
    action = payload.get("action", "")
    rid = msg.get("rid")
    src = msg.get("src", "")

    try:
        # ── Writes / upserts ────────────────────────────────────────
        if action == "record_interaction":
            user_id = str(payload.get("user_id", ""))
            quality = float(payload.get("quality", 0.5))
            social_graph.record_interaction(user_id, quality)
            profile = social_graph._cache.get(user_id)
            engagement_level = (
                float(profile.engagement_level) if profile is not None else 0.0
            )
            _send_msg(send_queue, bus.SOCIAL_INTERACTION_RECORDED, name, "all", {
                "user_id": user_id, "quality": quality,
                "engagement_level": engagement_level, "ts": time.time(),
            })
            _send_response(send_queue, name, src, {"ok": True}, rid)

        elif action == "get_or_create_user":
            user_id = str(payload.get("user_id", ""))
            platform = str(payload.get("platform", "unknown"))
            display_name = str(payload.get("display_name", ""))
            profile = social_graph.get_or_create_user(
                user_id, platform=platform, display_name=display_name)
            _send_response(send_queue, name, src, {
                "profile": _serialize_profile(profile),
            }, rid)

        elif action == "should_engage":
            user_id = str(payload.get("user_id", ""))
            level = social_graph.should_engage(user_id)
            _send_response(send_queue, name, src, {"level": level}, rid)

        elif action == "save_profile":
            profile_data = payload.get("profile", {}) or {}
            profile = _profile_from_dict(profile_data)
            if profile is not None:
                social_graph._save_profile(profile)
                social_graph._cache[profile.user_id] = profile
            _send_response(send_queue, name, src, {"ok": profile is not None}, rid)

        elif action == "record_edge":
            user_a = str(payload.get("user_a", ""))
            user_b = str(payload.get("user_b", ""))
            social_graph.record_edge(user_a, user_b)
            _send_response(send_queue, name, src, {"ok": True}, rid)

        elif action == "record_donation":
            tx_signature = str(payload.get("tx_signature", ""))
            sender_address = str(payload.get("sender_address", ""))
            amount_sol = float(payload.get("amount_sol", 0.0))
            memo = str(payload.get("memo", ""))
            matched_user = social_graph.record_donation(
                tx_signature, sender_address, amount_sol, memo)
            mood_delta, memory_weight = social_graph.get_donation_mood_boost(
                amount_sol)
            _send_msg(send_queue, bus.SOCIAL_DONATION_RECORDED, name, "all", {
                "user_id": matched_user.user_id if matched_user else None,
                "amount_sol": amount_sol, "mood_delta": mood_delta,
                "memory_weight": memory_weight,
                "tx_signature": tx_signature, "ts": time.time(),
            })
            _send_response(send_queue, name, src, {
                "matched_user": _serialize_profile(matched_user) if matched_user else None,
            }, rid)

        elif action == "record_inspiration":
            tx_signature = str(payload.get("tx_signature", ""))
            sender_address = str(payload.get("sender_address", ""))
            message = str(payload.get("message", ""))
            amount_sol = float(payload.get("amount_sol", 0.0))
            matched_user = social_graph.record_inspiration(
                tx_signature, sender_address, message, amount_sol)
            _send_msg(send_queue, bus.SOCIAL_INSPIRATION_RECORDED, name, "all", {
                "user_id": matched_user.user_id if matched_user else None,
                "message": message, "amount_sol": amount_sol,
                "tx_signature": tx_signature, "ts": time.time(),
            })
            _send_response(send_queue, name, src, {
                "matched_user": _serialize_profile(matched_user) if matched_user else None,
            }, rid)

        elif action == "link_sol_address":
            user_id = str(payload.get("user_id", ""))
            sol_address = str(payload.get("sol_address", ""))
            social_graph.link_sol_address(user_id, sol_address)
            _send_response(send_queue, name, src, {"ok": True}, rid)

        elif action == "mark_inspiration_processed":
            tx_signature = str(payload.get("tx_signature", ""))
            outcome = str(payload.get("outcome", ""))
            social_graph.mark_inspiration_processed(tx_signature, outcome)
            _send_response(send_queue, name, src, {"ok": True}, rid)

        # set_titan_preference / sync_community / mark_checked /
        # update_last_tweet — cross-process kin-protocol write handlers
        # RETIRED 2026-05-23 (D-SPEC-120, BUG-SOCIAL-GRAPH-WIRING re-triage).
        # Zero external callers; in-process `SocialGraph` API remains.

        elif action == "ledger_record":
            tweet_id = str(payload.get("tweet_id", ""))
            user_name = str(payload.get("user_name", ""))
            action_kind = str(payload.get("action_kind", ""))
            mention_text = str(payload.get("mention_text", ""))
            social_graph.ledger_record(
                tweet_id, user_name, action_kind, mention_text=mention_text)
            _send_response(send_queue, name, src, {"ok": True}, rid)

        elif action == "ledger_cleanup":
            max_age_seconds = float(payload.get("max_age_seconds", 172800))
            removed = social_graph.ledger_cleanup(max_age_seconds=max_age_seconds)
            _send_response(send_queue, name, src, {"removed": int(removed)}, rid)

        # ── Parameterized reads ─────────────────────────────────────
        elif action == "get_top_users":
            limit = int(payload.get("limit", 10))
            users = social_graph.get_top_users(limit=limit)
            _send_response(send_queue, name, src, {
                "users": [_serialize_profile(u) for u in (users or [])],
            }, rid)

        elif action == "get_user_connections":
            user_id = str(payload.get("user_id", ""))
            conns = social_graph.get_user_connections(user_id)
            _send_response(send_queue, name, src,
                           {"connections": list(conns or [])}, rid)

        # get_community / get_titan_favorites / get_accounts_to_check —
        # cross-process kin-protocol read handlers RETIRED 2026-05-23
        # (D-SPEC-120). Zero external callers; in-process `SocialGraph`
        # read API remains.

        elif action == "get_pending_inspirations":
            limit = int(payload.get("limit", 10))
            pending = social_graph.get_pending_inspirations(limit=limit)
            _send_response(send_queue, name, src,
                           {"inspirations": list(pending or [])}, rid)

        elif action == "find_user_by_sol_address":
            sol_address = str(payload.get("sol_address", ""))
            profile = social_graph.find_user_by_sol_address(sol_address)
            _send_response(send_queue, name, src, {
                "profile": _serialize_profile(profile) if profile else None,
            }, rid)

        elif action == "ledger_has_tweet":
            tweet_id = str(payload.get("tweet_id", ""))
            action_kind = payload.get("action_kind")
            has = social_graph.ledger_has_tweet(tweet_id, action=action_kind)
            _send_response(send_queue, name, src, {"has": bool(has)}, rid)

        elif action == "ledger_user_reply_count":
            user_name = str(payload.get("user_name", ""))
            window_seconds = float(payload.get("window_seconds", 0.0))
            count = social_graph.ledger_user_reply_count(
                user_name, window_seconds)
            _send_response(send_queue, name, src, {"count": int(count)}, rid)

        elif action == "ledger_last_reply_to_user":
            user_name = str(payload.get("user_name", ""))
            ts_val = social_graph.ledger_last_reply_to_user(user_name)
            _send_response(send_queue, name, src,
                           {"ts": float(ts_val or 0.0)}, rid)

        elif action == "ledger_total_today":
            action_kind = payload.get("action_kind")
            count = social_graph.ledger_total_today(action=action_kind)
            _send_response(send_queue, name, src, {"count": int(count)}, rid)

        else:
            logger.warning(
                "[SocialGraphWorker] Unknown action: %r (src=%s, rid=%s)",
                action, src, rid)
            if rid is not None:
                _send_response(send_queue, name, src, {
                    "error": f"unknown_action: {action}",
                }, rid)

    except Exception as e:
        logger.error(
            "[SocialGraphWorker] Error handling action=%r (src=%s, rid=%s): %s",
            action, src, rid, e, exc_info=True)
        if rid is not None:
            _send_response(send_queue, name, src, {"error": str(e)}, rid)
