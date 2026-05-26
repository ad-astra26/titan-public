#!/usr/bin/env python3
"""
guardian_hcl.py — Titan L1 supervisor standalone process (Phase 6 / D-SPEC-135 / v1.62.0).

INV-PROC-1: ps identity = `guardian_hcl` (setproctitle first I/O).
INV-PROC-3: this process boots BEFORE titan_hcl and titan_hcl_api; spawns and
            supervises them as L2/L3 children alongside the rest of the module
            catalog (per SPEC §11.B.4).
INV-PROC-5: independent crash domain — kill -9 titan_hcl does not affect
            guardian_hcl, and vice versa. titan_hcl_api stays UP through
            titan_hcl restart.

Architecture (SPEC §9.B guardian_hcl block):
  - kernel-rs (Rust L0) spawns scripts/guardian_hcl.py
  - guardian_hcl opens a BusSocketClient to the Rust broker
    (/tmp/titan_bus_<id>.sock) under name "guardian"
  - guardian_hcl loads the canonical module catalog
    (titan_hcl/module_catalog.py:build_catalog) — 51 ModuleSpec registrations
    moved verbatim from titan_hcl/core/plugin.py:_register_modules
  - guardian_hcl runs Guardian.start_all() to spawn autostart modules and
    drives the supervision loop (Guardian.monitor_tick at 1 Hz)

Workers connect to the same broker via setup_worker_bus → BusSocketClient
in worker_bus_bootstrap.py. The kernel-rs broker fans messages across
processes by subscriber name (Phase B.2 IPC dual-mode, l0_rust_enabled=true).
"""
import asyncio
import logging
import os
import signal
import sys
import threading
import time

# Ensure project root is on path
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

# ── INV-PROC-1 (SPEC §11.B.4 / D-SPEC-135 / v1.62.0): set ps identity as
# first I/O after import resolution so `ps -ef` distinguishes the L1 supervisor
# from `titan_hcl` (L2 plugin) and `titan_hcl_api` (L3). Same soft-fallback
# pattern as scripts/titan_hcl.py:30-34.
try:
    import setproctitle as _spt
    _spt.setproctitle("guardian_hcl")
except ImportError:
    pass


def setup_logging() -> None:
    """Configure logging based on merged config plugin_log_level."""
    try:
        from titan_hcl.config_loader import load_titan_config
        level_str = load_titan_config().get("openclaw", {}).get("plugin_log_level", "INFO")
    except Exception:
        level_str = "INFO"
    level = getattr(logging, level_str.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] [guardian_hcl] %(message)s",
        datefmt="%H:%M:%S",
    )


def _acquire_pid_lock(titan_id: str) -> bool:
    """Prevent multiple guardian_hcl instances per Titan via fcntl.flock.

    Mirrors scripts/titan_hcl.py:_acquire_pid_lock — same defensive
    O_RDWR|O_CREAT (no truncate-before-lock) pattern. PID file lives at
    data/guardian_hcl_<titan_id>.pid so T1+T2+T3 can share /dev/shm + data
    without colliding on a shared lock.
    """
    import fcntl
    global _pid_lock_fd  # noqa: PLW0603

    pid_path = os.path.join(
        os.path.dirname(__file__), "..", "data",
        f"guardian_hcl_{titan_id}.pid",
    )
    pid_path = os.path.normpath(pid_path)
    os.makedirs(os.path.dirname(pid_path), exist_ok=True)

    try:
        fd = os.open(pid_path, os.O_RDWR | os.O_CREAT, 0o644)
        lock_fp = os.fdopen(fd, "r+")
        fcntl.flock(lock_fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_fp.seek(0)
        lock_fp.truncate()
        lock_fp.write(str(os.getpid()))
        lock_fp.flush()
        _pid_lock_fd = lock_fp
        return True
    except (IOError, OSError):
        old_pid = "?"
        try:
            with open(pid_path) as f:
                content = f.read().strip()
                if content:
                    old_pid = content
        except Exception:
            pass
        print(
            f"\n  *** ABORT: Another guardian_hcl is already running "
            f"(PID {old_pid}) ***\n  If stale, remove {pid_path} and retry.\n"
        )
        return False


_pid_lock_fd = None  # Module-level for finally-release


def _release_pid_lock(titan_id: str) -> None:
    """Release PID lock and remove file on clean shutdown."""
    import fcntl
    global _pid_lock_fd  # noqa: PLW0603

    if _pid_lock_fd is not None:
        try:
            fcntl.flock(_pid_lock_fd, fcntl.LOCK_UN)
            _pid_lock_fd.close()
        except Exception:
            pass
        _pid_lock_fd = None

    pid_path = os.path.join(
        os.path.dirname(__file__), "..", "data",
        f"guardian_hcl_{titan_id}.pid",
    )
    pid_path = os.path.normpath(pid_path)
    try:
        os.remove(pid_path)
    except FileNotFoundError:
        pass


def _build_bus_and_client(titan_id: str, config: dict):
    """Construct DivineBus + outbound BusSocketClient mirroring kernel pattern.

    Mirrors TitanKernel._start_bus_socket_clients but for the guardian_hcl
    process. Returns (bus, client, identity_secret) tuple.

    The DivineBus serves as the in-process hub for Guardian's local subscribers
    (the "guardian" subscription registered in Guardian.__init__). The
    BusSocketClient connects to the Rust broker so:
      • Workers publishing dst="guardian" reach Guardian via broker fan-out
      • Guardian's outbound bus.publish() reaches the broker via attached client
    """
    from titan_hcl.bus import DivineBus
    from titan_hcl.core.bus_authkey import derive_bus_authkey
    from titan_hcl.core.bus_socket import BusSocketClient, bus_sock_path
    from titan_hcl.core.worker_bus_bootstrap import _try_load_identity_secret

    network_cfg = config.get("network", {})
    wallet_path = network_cfg.get(
        "wallet_keypair_path", "data/titan_identity_keypair.json")
    if not os.path.isabs(wallet_path):
        wallet_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", wallet_path))

    identity_secret = _try_load_identity_secret(wallet_path)
    if identity_secret is None:
        raise RuntimeError(
            f"guardian_hcl cannot start — identity keypair unreadable at "
            f"'{wallet_path}'. Phase C broker requires the Solana keypair "
            f"to derive the bus authkey (HKDF-SHA256).")

    authkey = derive_bus_authkey(identity_secret)
    # bus_sock_path returns a pathlib.Path; coerce to str so env-var
    # assignments + BusSocketClient (which accepts either, but we set
    # os.environ below which requires str) are happy.
    sock_path = str(bus_sock_path(titan_id))

    bus = DivineBus(maxsize=10000)

    # guardian_hcl's connection to the broker. Subscribes to the L1 supervision
    # topic set per SPEC §9.B guardian_hcl block. broadcast_topics enumerates
    # the broadcast types Guardian consumes; targeted dst="guardian" messages
    # bypass the broadcast filter and are routed by name.
    from titan_hcl.bus import (
        MODULE_HEARTBEAT, MODULE_READY, MODULE_SHUTDOWN, MODULE_CRASHED,
        MODULE_RELOAD_REQUEST, BUS_WORKER_ADOPT_REQUEST, BUS_PEER_DIED,
    )
    broadcast_topics = [
        MODULE_HEARTBEAT, MODULE_READY, MODULE_SHUTDOWN, MODULE_CRASHED,
        MODULE_RELOAD_REQUEST, BUS_WORKER_ADOPT_REQUEST, BUS_PEER_DIED,
    ]

    client = BusSocketClient(
        titan_id=titan_id,
        authkey=authkey,
        name="guardian",  # the canonical L1 supervisor subscriber name
        sock_path=sock_path,
        topics=broadcast_topics,
        reply_only=False,
    )
    client.start()
    bus.attach_client(client)

    # Set env vars so any subprocess fork inherits broker context.
    from titan_hcl.core.worker_bus_bootstrap import (
        ENV_BUS_SOCKET_PATH, ENV_BUS_TITAN_ID, ENV_BUS_KEYPAIR_PATH,
    )
    os.environ[ENV_BUS_SOCKET_PATH] = sock_path
    os.environ[ENV_BUS_TITAN_ID] = titan_id
    os.environ[ENV_BUS_KEYPAIR_PATH] = wallet_path

    return bus, client


def _start_inbound_dispatcher(bus, client, stop_event: threading.Event) -> threading.Thread:
    """Drain the BusSocketClient inbound queue into the local DivineBus.

    Mirrors TitanKernel._bus_client_inbound_dispatcher (kernel.py). Inbound
    messages from the broker arrive on client.inbound_q; we re-publish them
    onto the local DivineBus so Guardian's local subscribers (the "guardian"
    queue registered in Guardian.__init__) receive them.
    """
    def _loop():
        from queue import Empty
        while not stop_event.is_set():
            try:
                msg = client.inbound_q.get(timeout=0.5)
            except Empty:
                continue
            except Exception:
                continue
            if msg is None:
                continue
            try:
                # Re-inject into local bus. publish_in_process avoids the
                # client-outbound path so we don't loop the message back to
                # the broker.
                bus.publish_in_process(msg)
            except Exception as e:  # noqa: BLE001
                logging.getLogger(__name__).debug(
                    "[guardian_hcl] inbound re-publish error: %s", e)

    t = threading.Thread(
        target=_loop, name="guardian-hcl-inbound", daemon=True)
    t.start()
    return t


def _publish_guardian_state(guardian, titan_id: str, stop_event: threading.Event) -> threading.Thread:
    """Publish guardian_state.bin SHM slot at 1 Hz (Phase A.4 / D-SPEC-70).

    Migrated from TitanKernel._guardian_loop — under Phase 6, guardian_hcl
    is the canonical writer of guardian_state.bin (G21 single-writer). The
    api_subprocess + arch_map continue to read this slot unchanged.
    """
    pub = None
    try:
        from titan_hcl.logic.guardian_state_publisher import GuardianStatePublisher
        pub = GuardianStatePublisher(titan_id=titan_id)
        pub.publish(guardian)  # cold-boot first publish
        logging.getLogger(__name__).info(
            "[guardian_hcl] GuardianStatePublisher attached (G21 single-writer)")
    except Exception as e:  # noqa: BLE001
        logging.getLogger(__name__).warning(
            "[guardian_hcl] GuardianStatePublisher init failed: %s — "
            "api_subprocess will read cold-boot stubs from guardian_state.bin", e)

    def _loop():
        while not stop_event.is_set():
            try:
                if pub is not None:
                    pub.publish(guardian)
            except Exception as e:  # noqa: BLE001
                logging.getLogger(__name__).debug(
                    "[guardian_hcl] state publish error: %s", e)
            stop_event.wait(timeout=1.0)

    t = threading.Thread(
        target=_loop, name="guardian-hcl-state-publish", daemon=True)
    t.start()
    return t


def _handle_module_lifecycle_requests(bus, guardian, stop_event: threading.Event) -> threading.Thread:
    """Subscribe to MODULE_*_REQUEST messages from GuardianHCLClient in plugin process.

    Phase 6 § 3C.3 6F bus contract:
      - MODULE_START_REQUEST   {name}                 → guardian.start(name)
      - MODULE_STOP_REQUEST    {name, reason}         → guardian.stop(name, reason)
      - MODULE_RESTART_REQUEST {name, reason, **kw}   → guardian.restart_module(name, reason, **kw)
      - MODULE_RELOAD_REQUEST already handled by Guardian._process_guardian_messages
    """
    from titan_hcl.bus import (
        MODULE_START_REQUEST, MODULE_STOP_REQUEST, MODULE_RESTART_REQUEST,
    )
    q = bus.subscribe(
        "guardian_hcl_lifecycle",
        types=[MODULE_START_REQUEST, MODULE_STOP_REQUEST, MODULE_RESTART_REQUEST],
        reply_only=True,
    )

    def _loop():
        from queue import Empty
        while not stop_event.is_set():
            try:
                msg = q.get(timeout=0.5)
            except Empty:
                continue
            except Exception:
                continue
            mtype = msg.get("type")
            payload = msg.get("payload", {}) or {}
            name = payload.get("name")
            if not name:
                continue
            try:
                if mtype == MODULE_START_REQUEST:
                    guardian.start(name)
                elif mtype == MODULE_STOP_REQUEST:
                    guardian.stop(name, reason=payload.get("reason", "requested"))
                elif mtype == MODULE_RESTART_REQUEST:
                    extra = {k: v for k, v in payload.items()
                             if k not in ("name", "reason")}
                    guardian.restart_module(
                        name,
                        reason=payload.get("reason", "requested"),
                        **extra,
                    )
            except Exception as e:  # noqa: BLE001
                logging.getLogger(__name__).warning(
                    "[guardian_hcl] lifecycle request %s for '%s' failed: %s",
                    mtype, name, e)

    t = threading.Thread(
        target=_loop, name="guardian-hcl-lifecycle", daemon=True)
    t.start()
    return t


def run() -> int:
    setup_logging()
    logger = logging.getLogger(__name__)

    # ── Load config + titan_id ───────────────────────────────────────
    from titan_hcl.config_loader import load_titan_config
    from titan_hcl.core.state_registry import resolve_titan_id

    config = load_titan_config()
    titan_id = resolve_titan_id()
    logger.info("[guardian_hcl] booting for titan_id=%s pid=%d", titan_id, os.getpid())

    # ── PID lock ─────────────────────────────────────────────────────
    if not _acquire_pid_lock(titan_id):
        return 1

    # ── Process group leader so SIGTERM cascades to all module children ──
    try:
        os.setpgrp()
    except OSError:
        pass

    stop_event = threading.Event()

    def _sigterm_handler(_signum, _frame):
        logger.info("[guardian_hcl] SIGTERM received — initiating shutdown")
        stop_event.set()

    signal.signal(signal.SIGTERM, _sigterm_handler)
    signal.signal(signal.SIGINT, _sigterm_handler)

    bus = None
    client = None
    guardian = None
    try:
        # ── Bus client to Rust broker ────────────────────────────────
        bus, client = _build_bus_and_client(titan_id, config)
        logger.info("[guardian_hcl] bus client connected (name=guardian)")

        # ── Guardian instance ────────────────────────────────────────
        from titan_hcl.guardian_hcl import Guardian
        guardian = Guardian(bus, config=config.get("guardian", {}))
        # _kernel_ref = None: cross-process swap interlock degrades to no-op
        # (per Guardian.start docstring "None in legacy mode → swap interlock
        # degrades to no-op").
        guardian._kernel_ref = None
        logger.info("[guardian_hcl] Guardian instance constructed")

        # ── Module catalog (51 ModuleSpec registrations) ─────────────
        from titan_hcl.module_catalog import build_catalog
        build_catalog(bus, guardian, config, titan_id=titan_id)
        logger.info(
            "[guardian_hcl] module catalog built — %d modules registered",
            len(guardian._modules))

        # ── Background loops ─────────────────────────────────────────
        inbound_t = _start_inbound_dispatcher(bus, client, stop_event)
        state_t = _publish_guardian_state(guardian, titan_id, stop_event)
        lifecycle_t = _handle_module_lifecycle_requests(bus, guardian, stop_event)

        # ── Boot autostart modules ───────────────────────────────────
        guardian.start_all()
        logger.info(
            "[guardian_hcl] start_all complete — modules: %s",
            list(guardian._modules.keys()))

        # ── Main supervision loop ────────────────────────────────────
        # 1 Hz monitor tick + drain_send_queues. drain_send_queues is a
        # no-op when workers communicate via socket (which they do under
        # has_socket_broker=True / l0_rust_enabled=true), but the call is
        # still safe + serves legacy mp.Queue fallback.
        logger.info("[guardian_hcl] supervision loop entered")
        while not stop_event.is_set():
            try:
                guardian.monitor_tick()
                guardian.drain_send_queues()
                guardian._process_guardian_messages()
            except Exception as e:  # noqa: BLE001
                logger.error("[guardian_hcl] supervision tick error: %s", e, exc_info=True)
            stop_event.wait(timeout=1.0)

        logger.info("[guardian_hcl] supervision loop exited — stopping modules")
        try:
            guardian.stop_all(reason="shutdown")
        except Exception as e:  # noqa: BLE001
            logger.warning("[guardian_hcl] stop_all error: %s", e)

        return 0

    except Exception as e:  # noqa: BLE001
        logger.error("[guardian_hcl] fatal boot error: %s", e, exc_info=True)
        return 2

    finally:
        try:
            if guardian is not None:
                # _module_ready_publisher thread + restart_executor cleanup
                guardian._module_ready_publisher_stop.set()
                try:
                    guardian._restart_executor.shutdown(wait=False)
                except Exception:
                    pass
        except Exception:
            pass
        try:
            if client is not None:
                client.stop()
        except Exception:
            pass
        try:
            _release_pid_lock(titan_id)
        except Exception:
            pass


def main() -> None:
    sys.exit(run())


if __name__ == "__main__":
    main()
