#!/usr/bin/env python3
"""scripts/titan_hcl_api.py — Phase 11 §11.I.1 kernel-rs peer (L3 api).

Per Maker 2026-05-27 — api becomes a kernel-spawned peer to titan_hcl
+ guardian_hcl (INV-PROC-3 / INV-PROC-5). Self-bootstraps the standard
worker bus contract (recv/send queues backed by SocketQueue via the
canonical `setup_worker_bus` path used by all 39 spawn-mode workers)
and delegates to `titan_hcl.api.api_main.entry` for the unchanged
api_subprocess body.

Why setup_worker_bus and not _build_bus_and_client + manual shims:
  * api_main.entry → api_subprocess_main expects the worker queue
    contract (send_queue.put_nowait + recv_queue.get(timeout=...))
    that SocketQueue already implements end-to-end.
  * setup_worker_bus auto-emits MODULE_READY on every BUS_SUBSCRIBE
    ack (§8.0.bis worker contract C2) and registers SwapHandlerState
    — same wiring _module_wrapper applies to every other worker.
  * Subscriber name stays "api" so dst="api" routing (OBSERVATORY_EVENT
    bridge + CHAT_RESPONSE / CHAT_STREAM_CHUNK rid replies) keeps
    working unchanged — see titan_hcl/module_catalog.py:1957.

ps identity = "titan_hcl_api" (INV-PROC-1). ModuleStateWriter("api",
layer="L3", BootPriority.MANDATORY) is established inside
api_main.entry so /v6/readiness sees the slot the same way it does
for the 39 worker slots migrated in commit 58761482.
"""
import logging
import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

# Phase 11 §11.I.5 / Chunk 11L — MALLOC_ARENA_MAX defensive default.
# kernel-rs sets this in build_child_env (spawn.rs:127) for the
# production fleet boot path; the setdefault load-bears for standalone
# dev runs + test fixtures importing this script directly.
os.environ.setdefault("MALLOC_ARENA_MAX", "2")

# INV-PROC-1 — ps identity as first I/O.
try:
    import setproctitle as _spt
    _spt.setproctitle("titan_hcl_api")
except ImportError:
    pass


def setup_logging() -> None:
    """Configure logging based on merged config plugin_log_level."""
    try:
        from titan_hcl.params import load_titan_params as load_titan_config
        level_str = load_titan_config().get("openclaw", {}).get("plugin_log_level", "INFO")
    except Exception:
        level_str = "INFO"
    level = getattr(logging, level_str.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] [titan_hcl_api] %(message)s",
        datefmt="%H:%M:%S",
    )


def run() -> int:
    setup_logging()
    logger = logging.getLogger(__name__)

    from titan_hcl.params import load_titan_params as load_titan_config
    from titan_hcl.core.state_registry import resolve_titan_id

    cfg = load_titan_config()
    titan_id = resolve_titan_id()
    logger.info("[titan_hcl_api] booting for titan_id=%s pid=%d",
                titan_id, os.getpid())

    # Build the same sub_config dict that module_catalog hands to the
    # Guardian-module entry today (titan_hcl/module_catalog.py:1944-1955).
    # Forwarding [api] + [microkernel] + [network] + [frontend] keeps
    # the api endpoints' vault/RPC/frontend-mode behaviour unchanged.
    api_cfg = cfg.get("api", {}) or {}
    sub_config = {
        "api": api_cfg,
        "microkernel": cfg.get("microkernel", {}),
        "network": cfg.get("network", {}),
        "frontend": cfg.get("frontend", {}),
    }

    # SPEC §11.B.4 / D-SPEC-135: subscriber name MUST remain "api" so
    # dst="api" routing keeps working (OBSERVATORY_EVENT bridge +
    # CHAT_RESPONSE / CHAT_STREAM_CHUNK rid-routed replies). ps identity
    # differentiation comes from setproctitle above.
    subscriber_name = "api"

    # Phase D D-SPEC-82 + the module_catalog reply_only=True invariant
    # — api consumes ONLY targeted dst="api" messages (no broadcasts);
    # broker silently skips broadcast fan-out per SPEC §8.2 v1.4.0
    # D-SPEC-42. Mirror the catalog entry verbatim.
    broadcast_topics: list = []
    reply_only = True

    # Use the canonical worker_bus_bootstrap path so SocketQueue handles
    # both send + recv — same wiring `_module_wrapper` applies to every
    # spawn-mode worker. The mp.Queue we pass below is the legacy-mode
    # fallback (unused under Phase C — setup_worker_bus hard-fails if
    # broker env vars are missing).
    import multiprocessing as mp
    legacy_recv = mp.Queue()
    legacy_send = mp.Queue()

    # kernel-rs spawns this peer directly and seeds only the CANONICAL
    # broker env vars (TITAN_KERNEL_BUS_SOCKET_PATH + TITAN_BUS_TITAN_ID).
    # setup_worker_bus reads the legacy TITAN_BUS_SOCKET_PATH +
    # TITAN_BUS_KEYPAIR_PATH names — under the old guardian-spawns-workers
    # topology those were seeded into the parent env and inherited; the
    # kernel-rs peer never ran that path, so seed them now from config +
    # bus_sock_path(titan_id). (Live T3 fix 2026-05-28.)
    from scripts._titan_bus_client_helpers import seed_broker_env
    seed_broker_env(titan_id, cfg)

    from titan_hcl.core.worker_bus_bootstrap import setup_worker_bus
    bus_client = None
    try:
        recv_queue, send_queue, bus_client = setup_worker_bus(
            subscriber_name, legacy_recv, legacy_send,
            topics=broadcast_topics,
            reply_only=reply_only,
        )
        # Mirror _module_wrapper's SwapHandlerState wiring so the api
        # peer participates in worker_swap_handler dispatch consistently
        # with every other spawn-mode worker (HANDOFF / ADOPT_ACK /
        # CANCELED + supervision daemon tick). install_full_protection
        # also installs PR_SET_PDEATHSIG so a kernel-rs SIGKILL of this
        # process's parent tree cascades a clean shutdown here.
        from titan_hcl.core.worker_lifecycle import install_full_protection
        _wl = install_full_protection()
        logger.info(
            "[titan_hcl_api] worker lifecycle protection: pdeathsig=%s watcher=%s",
            _wl["pdeathsig_installed"], _wl["watcher_started"],
        )
        from titan_hcl.core.worker_swap_handler import (
            SwapHandlerState, set_active_swap_state, start_supervision_thread,
        )
        swap_state = SwapHandlerState(
            name=subscriber_name,
            start_method="spawn",
            watcher_state=_wl["watcher_state"],
            bus_client=bus_client,
        )
        set_active_swap_state(swap_state)
        start_supervision_thread(swap_state)
        logger.info(
            "[titan_hcl_api] bus client attached (name=%s, reply_only=True) "
            "+ swap-handler supervision active",
            subscriber_name)
    except Exception as e:  # noqa: BLE001
        # Hard-fail: under fleet-wide Phase C the broker is the only
        # supported transport; setup_worker_bus raises if env vars are
        # missing. Surface and exit non-zero so kernel-rs's supervision
        # log records the failure.
        logger.error("[titan_hcl_api] setup_worker_bus failed: %s",
                     e, exc_info=True)
        return 2

    try:
        from titan_hcl.api.api_main import entry as api_entry
        api_entry(recv_queue, send_queue, subscriber_name, sub_config)
        return 0
    except KeyboardInterrupt:
        logger.info("[titan_hcl_api] interrupted")
        return 0
    except Exception as e:  # noqa: BLE001
        logger.error("[titan_hcl_api] crashed: %s", e, exc_info=True)
        return 1
    finally:
        if bus_client is not None:
            try:
                bus_client.stop()
            except Exception:
                pass


def main() -> None:
    sys.exit(run())


if __name__ == "__main__":
    main()
