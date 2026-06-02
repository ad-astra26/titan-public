#!/usr/bin/env python3
"""scripts/_titan_bus_client_helpers.py — shared bus-client builder for the
Phase 11 §11.I.1 / D-SPEC-141 peer-spawn process trio (guardian_hcl,
titan_hcl, titan_hcl_api).

Factored out of `scripts/guardian_hcl.py` so that each of the three
kernel-rs peers can build its own DivineBus + outbound BusSocketClient
without duplicating the ~70-LOC handshake (config → keypair → authkey →
sock_path → BusSocketClient.start → bus.attach_client → broker env-var
seed for descendant fork).

Mirrors `TitanKernel._start_bus_socket_clients` but standalone — used by
processes that don't carry a full kernel (the L1 supervisor + L3 api
peers).
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Optional

logger = logging.getLogger(__name__)


def _resolve_wallet_path(config: dict) -> str:
    """Resolve the identity keypair path from config, absolutized to repo root."""
    network_cfg = config.get("network", {})
    wallet_path = network_cfg.get(
        "wallet_keypair_path", "data/titan_identity_keypair.json")
    # Expand ~ FIRST: a "~/.config/solana/id.json" value is not os.path.isabs(),
    # so without this it would be joined onto the repo root as a literal "~"
    # path (ENXIO at load). expanduser makes it the real absolute home path.
    wallet_path = os.path.expanduser(wallet_path)
    if not os.path.isabs(wallet_path):
        wallet_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", wallet_path))
    return wallet_path


def seed_broker_env(titan_id: str, config: dict) -> None:
    """Seed the legacy broker env vars that `setup_worker_bus` reads.

    kernel-rs `build_child_env` (spawn.rs) sets the CANONICAL names
    (TITAN_KERNEL_BUS_SOCKET_PATH + TITAN_BUS_TITAN_ID alias) but NOT
    TITAN_BUS_SOCKET_PATH / TITAN_BUS_KEYPAIR_PATH. Under the legacy
    guardian-spawns-workers topology those were seeded into the parent
    env by `_build_bus_and_client` and inherited by every mp.Process
    worker. A kernel-rs peer (titan_hcl_api) spawned DIRECTLY by the
    kernel never ran that path, so `setup_worker_bus` hard-failed on
    "missing TITAN_BUS_SOCKET_PATH,TITAN_BUS_KEYPAIR_PATH" (live T3
    2026-05-28). Seed them here from config + bus_sock_path(titan_id)
    so the canonical worker bus bootstrap works for the api peer too.
    """
    from titan_hcl.core.bus_socket import bus_sock_path
    from titan_hcl.core.worker_bus_bootstrap import (
        ENV_BUS_SOCKET_PATH, ENV_BUS_TITAN_ID, ENV_BUS_KEYPAIR_PATH,
    )
    os.environ[ENV_BUS_SOCKET_PATH] = str(bus_sock_path(titan_id))
    os.environ[ENV_BUS_TITAN_ID] = str(titan_id)
    os.environ[ENV_BUS_KEYPAIR_PATH] = str(_resolve_wallet_path(config))


def build_bus_and_client(
    titan_id: str,
    config: dict,
    *,
    subscriber_name: str,
    broadcast_topics: Optional[list] = None,
    reply_only: bool = False,
):
    """Construct DivineBus + outbound BusSocketClient.

    Args:
        titan_id:        canonical Titan id (resolve via state_registry.resolve_titan_id).
        config:          merged titan_hcl config dict (load_titan_config()).
        subscriber_name: broker subscriber name (matched against MsgHeader.dst).
                         e.g. "guardian" for the L1 supervisor process,
                         "api" for the L3 api process (so dst="api" routes there).
        broadcast_topics: per-subscriber broadcast filter (None = subscribe-all;
                          [] = reply_only-style empty; non-empty = filtered).
        reply_only:      SPEC §8.2 v1.4.0 D-SPEC-42 — True = consumes only
                          targeted dst=<name> messages, broadcasts silently
                          skipped by the broker. Use for RPC-shaped subscribers.

    Returns:
        (bus, client) tuple. Caller must `client.stop()` on shutdown.
    """
    from titan_hcl.bus import DivineBus
    from titan_hcl.core.bus_authkey import derive_bus_authkey
    from titan_hcl.core.bus_socket import BusSocketClient, bus_sock_path
    from titan_hcl.core.worker_bus_bootstrap import _try_load_identity_secret

    wallet_path = _resolve_wallet_path(config)

    identity_secret = _try_load_identity_secret(wallet_path)
    if identity_secret is None:
        raise RuntimeError(
            f"{subscriber_name} cannot start — identity keypair unreadable "
            f"at '{wallet_path}'. Phase C broker requires the Solana keypair "
            f"to derive the bus authkey (HKDF-SHA256).")

    authkey = derive_bus_authkey(identity_secret)
    # bus_sock_path returns pathlib.Path; coerce to str so env-var
    # assignments + BusSocketClient (which accepts either, but we set
    # os.environ below which requires str) are happy.
    sock_path = str(bus_sock_path(titan_id))

    bus = DivineBus(maxsize=10000)

    client = BusSocketClient(
        titan_id=titan_id,
        authkey=authkey,
        name=subscriber_name,
        sock_path=sock_path,
        topics=broadcast_topics,
        reply_only=reply_only,
    )
    client.start()
    bus.attach_client(client)

    # Seed broker env vars so any subprocess fork inherits broker context
    # (mirrors TitanKernel._start_bus_socket_clients + the legacy
    # guardian_hcl._build_bus_and_client behaviour). os.environ requires
    # str values; sock_path + wallet_path may be pathlib.PosixPath under
    # newer code paths — coerce explicitly (caught 2026-05-26 T1).
    from titan_hcl.core.worker_bus_bootstrap import (
        ENV_BUS_SOCKET_PATH, ENV_BUS_TITAN_ID, ENV_BUS_KEYPAIR_PATH,
    )
    os.environ[ENV_BUS_SOCKET_PATH] = str(sock_path)
    os.environ[ENV_BUS_TITAN_ID] = str(titan_id)
    os.environ[ENV_BUS_KEYPAIR_PATH] = str(wallet_path)

    return bus, client


def start_inbound_dispatcher(bus, client, stop_event: threading.Event) -> threading.Thread:
    """Drain the BusSocketClient inbound deque into the local DivineBus.

    Mirrors TitanKernel._bus_client_inbound_dispatcher + the legacy
    guardian_hcl._start_inbound_dispatcher. Blocks on the client's
    `_wake_cond.wait_for(predicate)` until inbound data arrives or either
    stop_event fires, then drains everything pending and re-publishes
    each message on the local DivineBus via `publish_in_process` (no
    broker echo).
    """
    inbound_lock = client._inbound_lock
    wake_cond = client._wake_cond
    inbound_deque = client._inbound
    stop_evt = client._stop_event  # client's own stop

    def _loop():
        predicate = lambda: bool(inbound_deque) or stop_evt.is_set() or stop_event.is_set()
        while not stop_event.is_set():
            with inbound_lock:
                wake_cond.wait_for(predicate, timeout=1.0)
                drained = list(inbound_deque)
                inbound_deque.clear()
            for msg in drained:
                try:
                    bus.publish_in_process(msg)
                except Exception as e:  # noqa: BLE001
                    logger.debug("[%s] inbound re-publish error: %s",
                                 client.name, e)

    t = threading.Thread(
        target=_loop, name=f"{client.name}-inbound", daemon=True)
    t.start()
    return t
