#!/usr/bin/env python3
"""
guardian_hcl.py — Titan L1 supervisor peer process (Phase 11 §11.I.1 / D-SPEC-141 / v1.65.0).

INV-PROC-1: ps identity = `guardian_hcl` (setproctitle first I/O).
INV-PROC-3: kernel-rs spawns guardian_hcl as a peer to titan_hcl + titan_hcl_api
            (NOT as their parent). Independent crash domain — kill -9 titan_hcl
            does not affect guardian_hcl, and vice versa.
INV-PROC-5: titan_hcl_api stays UP through titan_hcl restart.

Phase 11 §11.I.1 split — what lives where:

  Orchestrator (lives in `scripts/titan_hcl.py`):
    - register / start / stop / restart_module / start_all
    - lifecycle subscriber for MODULE_*_REQUEST
    - module spawn + Phase A/B pipeline + probe dispatch + hot-reload
    - fleet_ready SHM publish (G21 single-writer)

  Supervisor (this process):
    - heartbeat-stale + RSS + PID-dead fault detection
    - publishes MODULE_RESTART_REQUEST(dst="guardian_hcl_lifecycle")
    - holds a metadata-only Orchestrator instance for ModuleSpec lookups
      (heartbeat_timeout, layer, rss_limit_mb, restart_on_crash, …)

Architecture (SPEC §9.B guardian_hcl block):
  - kernel-rs (Rust L0) spawns scripts/guardian_hcl.py
  - guardian_hcl opens a BusSocketClient to the Rust broker
    (/tmp/titan_bus_<id>.sock) under name "guardian" so MODULE_HEARTBEAT
    / MODULE_READY / MODULE_CRASHED / BUS_PEER_DIED fan out to BOTH
    titan_hcl (Orchestrator) and this process (Supervisor) — the broker
    delivers targeted dst="guardian" frames to every subscriber whose
    name matches (titan-rust/.../broker.rs:653).
  - build_catalog populates `Orchestrator._modules` with ModuleSpec rows
    so Supervisor.monitor_tick has the heartbeat_timeout / rss_limit_mb
    metadata it needs. start_all is NOT called here — titan_hcl spawns
    the workers; this process just observes their bus events.
"""
import logging
import os
import signal
import sys
import threading

# Ensure project root is on path
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

# ── Phase 11 §11.I.5 / Chunk 11L — MALLOC_ARENA_MAX defensive default ──
# kernel-rs already sets this via build_child_env (titan-kernel-rs/spawn.rs:127)
# when it spawns guardian_hcl, so this `setdefault` is a no-op in the
# production fleet boot path. It load-bears under three independent
# scenarios where guardian_hcl boots WITHOUT the kernel-rs env:
#   (1) standalone dev runs (`python scripts/guardian_hcl.py`)
#   (2) systemd unit overrides that strip kernel-spawned env
#   (3) tests / fixtures that import scripts/guardian_hcl.py directly
os.environ.setdefault("MALLOC_ARENA_MAX", "2")

# ── INV-PROC-1: ps identity ──────────────────────────────────────────
try:
    import setproctitle as _spt
    _spt.setproctitle("guardian_hcl")
except ImportError:
    pass

# Native-crash visibility (SPEC §11.I.4) — dump a C+Python traceback to
# stderr→journal on a fatal native signal (SIGSEGV/SIGABRT/SIGBUS/SIGFPE);
# the @with_error_envelope cascade only catches Python exceptions, not signals.
try:
    import faulthandler as _faulthandler
    _faulthandler.enable()
except Exception:
    pass


def setup_logging() -> None:
    """Configure logging based on merged config plugin_log_level."""
    try:
        # RFP_config_as_shm_state §7.C/C.6: read [openclaw] from the SHM slot
        # (config-as-state, INV-CFG-7). SHM is live at B8 (daemon seeds at B3.5
        # before python spawn); SHM-absent → get_params bootstraps.
        from titan_hcl.params import get_params
        level_str = get_params("openclaw").get("plugin_log_level", "INFO")
    except Exception:
        level_str = "INFO"
    level = getattr(logging, level_str.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] [guardian_hcl] %(message)s",
        datefmt="%H:%M:%S",
    )


_pid_lock_fd = None  # Module-level for finally-release


def _acquire_pid_lock(titan_id: str) -> bool:
    """Prevent multiple guardian_hcl instances per Titan via fcntl.flock.

    Defensive O_RDWR|O_CREAT (no truncate-before-lock) pattern mirroring
    scripts/titan_hcl.py:_acquire_pid_lock — preserves live PID info on
    contention so the abort message is diagnosable.
    """
    import fcntl
    global _pid_lock_fd

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


def _release_pid_lock(titan_id: str) -> None:
    """Release PID lock and remove file on clean shutdown."""
    import fcntl
    global _pid_lock_fd

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


def _publish_guardian_state(orchestrator, titan_id: str, stop_event: threading.Event) -> threading.Thread:
    """Publish guardian_state.bin SHM slot at 1 Hz (Phase A.4 / D-SPEC-70).

    Phase 11 §11.I.1 / D-SPEC-141: under the split, guardian_hcl remains
    the canonical writer of guardian_state.bin (G21 single-writer). The
    orchestrator reference here is the metadata-only instance — its
    _modules dict is populated by build_catalog + updated by bus events
    received via the "guardian" subscription, so GuardianStatePublisher
    sees the same {name: state} view the live Orchestrator in titan_hcl
    publishes via its own per-module SHM slots.
    """
    pub = None
    try:
        from titan_hcl.logic.guardian_state_publisher import GuardianStatePublisher
        pub = GuardianStatePublisher(titan_id=titan_id)
        pub.publish(orchestrator)
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
                    pub.publish(orchestrator)
            except Exception as e:  # noqa: BLE001
                logging.getLogger(__name__).debug(
                    "[guardian_hcl] state publish error: %s", e)
            stop_event.wait(timeout=1.0)

    t = threading.Thread(
        target=_loop, name="guardian-hcl-state-publish", daemon=True)
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

    # ── Process group leader (kept for legacy SIGTERM forwarding) ────
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
    orchestrator = None
    try:
        # ── Bus client to Rust broker ────────────────────────────────
        # Subscribes as "guardian" so the broker fans MODULE_HEARTBEAT /
        # MODULE_READY / MODULE_CRASHED / BUS_PEER_DIED to BOTH this
        # process (Supervisor) AND titan_hcl (Orchestrator). Each
        # process's local DivineBus subscribers receive their own copy
        # via its inbound dispatcher → publish_in_process path.
        from titan_hcl.bus import (
            MODULE_HEARTBEAT, MODULE_READY, MODULE_SHUTDOWN, MODULE_CRASHED,
            MODULE_RELOAD_REQUEST, BUS_WORKER_ADOPT_REQUEST, BUS_PEER_DIED,
            MODULE_ERROR,
        )
        from scripts._titan_bus_client_helpers import (
            build_bus_and_client, start_inbound_dispatcher,
        )
        bus, client = build_bus_and_client(
            titan_id, config,
            subscriber_name="guardian",
            broadcast_topics=[
                MODULE_HEARTBEAT, MODULE_READY, MODULE_SHUTDOWN, MODULE_CRASHED,
                MODULE_RELOAD_REQUEST, BUS_WORKER_ADOPT_REQUEST, BUS_PEER_DIED,
                # RFP_supervision_lifecycle §7.F — forward MODULE_ERROR so the
                # Supervisor's taxonomy consumer (its own "guardian_module_errors"
                # in-process subscriber) renders the greppable journal cascade +
                # runs the FATAL-ModuleError DISABLE gate.
                MODULE_ERROR,
            ],
            reply_only=False,
        )
        logger.info("[guardian_hcl] bus client connected (name=guardian)")

        # ── Orchestrator (metadata-only) + Supervisor ────────────────
        # Phase 11 §11.I.1 / D-SPEC-141: in this peer-spawn topology the
        # Orchestrator here NEVER calls start_all — titan_hcl owns spawn.
        # build_catalog still runs so `_modules` carries the ModuleSpec
        # rows the Supervisor.monitor_tick consults for heartbeat_timeout,
        # rss_limit_mb, layer, restart_on_crash. State transitions arrive
        # via bus events (MODULE_READY → info.pid, MODULE_HEARTBEAT →
        # info.last_heartbeat) handled by `_process_guardian_messages`
        # at the top of monitor_tick.
        from titan_hcl.orchestrator import Orchestrator
        from titan_hcl.supervisor import Supervisor
        orchestrator = Orchestrator(bus, config=config.get("guardian", {}))
        orchestrator._kernel_ref = None
        supervisor = Supervisor(bus, orchestrator, config=config.get("guardian", {}))
        logger.info(
            "[guardian_hcl] Orchestrator (metadata-only) + Supervisor "
            "constructed (Phase 11 §11.I.1 peer-spawn)")

        # ── Module catalog (51 ModuleSpec registrations) ─────────────
        from titan_hcl.module_catalog import build_catalog
        build_catalog(bus, orchestrator, config, titan_id=titan_id)
        logger.info(
            "[guardian_hcl] module catalog built — %d modules registered "
            "(metadata only; titan_hcl owns spawn)",
            len(orchestrator._modules))

        # ── Background loops ─────────────────────────────────────────
        start_inbound_dispatcher(bus, client, stop_event)
        _publish_guardian_state(orchestrator, titan_id, stop_event)

        # ── Main supervision loop ────────────────────────────────────
        # Phase 11 §11.I.1 — supervisor.monitor_tick drains the
        # orchestrator's bus queue (BUS_PEER_DIED + MODULE_HEARTBEAT +
        # MODULE_READY + BUS_WORKER_ADOPT_REQUEST + MODULE_RELOAD_REQUEST),
        # then runs fault detection + RSS budget enforcement. On fault it
        # publishes MODULE_RESTART_REQUEST(dst="guardian_hcl_lifecycle")
        # which the lifecycle subscriber in titan_hcl translates back
        # into orchestrator.restart_module via _start_lifecycle_subscriber.
        # drain_send_queues remains a no-op here (no workers were spawned
        # by this orchestrator; info.send_queue is None for all modules).
        logger.info("[guardian_hcl] supervision loop entered (Supervisor-driven, Phase 11 §11.I.1)")
        while not stop_event.is_set():
            try:
                supervisor.monitor_tick()
            except Exception as e:  # noqa: BLE001
                logger.error("[guardian_hcl] supervision tick error: %s", e, exc_info=True)
            stop_event.wait(timeout=1.0)

        logger.info("[guardian_hcl] supervision loop exited")
        return 0

    except Exception as e:  # noqa: BLE001
        logger.error("[guardian_hcl] fatal boot error: %s", e, exc_info=True)
        return 2

    finally:
        try:
            if orchestrator is not None:
                # _module_ready_publisher thread + restart_executor cleanup.
                # restart_executor is dormant here (no restart_module calls
                # ever route to this orchestrator) but the ThreadPoolExecutor
                # was instantiated in __init__ so close it cleanly.
                orchestrator._module_ready_publisher_stop.set()
                try:
                    orchestrator._restart_executor.shutdown(wait=False)
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
