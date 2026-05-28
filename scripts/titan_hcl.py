#!/usr/bin/env python3
"""
titan_hcl.py — The Titan Sovereign Agent Launcher.

Boots the full Titan cognitive stack and creates an Agno-powered sovereign agent:
  - All subsystems (memory, metabolism, soul, sage, studio, observatory)
  - Circadian rhythms (meditation every 6h, rebirth every 24h)
  - Sovereign Observatory API on configured port (includes POST /chat endpoint)
  - Interactive stdin prompt loop using the Agno agent

Usage:
    titan-main                       # If installed via pip entry_points
    python scripts/titan_hcl.py     # Direct execution
    python scripts/titan_hcl.py --health-only  # Boot, health check, exit
"""
import asyncio
import logging
import os
import sys
import signal
import threading

# Ensure project root is on path
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

# ── Phase 11 §11.I.5 / Chunk 11L — MALLOC_ARENA_MAX defensive default ──
# Same rationale as scripts/guardian_hcl.py:38 — kernel-rs sets this in
# build_child_env when it spawns guardian_hcl (which in turn spawns this
# titan_hcl process via subprocess.Popen with `env={**os.environ, ...}`).
# The `setdefault` makes the cap survive any of: standalone dev runs,
# systemd unit env overrides, and tests/fixtures that import
# scripts/titan_hcl.py directly. Children (the 40 worker modules spawned
# via multiprocessing.Process inside this process) inherit the value
# transparently — fleet-wide glibc arena cap of 2 (RFP §3F.2.7 9F
# folded into Phase 11 §3H.2).
os.environ.setdefault("MALLOC_ARENA_MAX", "2")

# ── INV-PROC-1 (SPEC §11.B.4 / D-SPEC-135 / v1.62.0): set ps identity as
# first I/O after import resolution so `ps -ef` distinguishes the L2 plugin
# from `titan_hcl_api` (L3) and `guardian_hcl` (L1). Same soft-fallback
# pattern as `titan_hcl/persistence_entry.py:48` — setproctitle absence is
# observability degradation, not a functional fault.
try:
    import setproctitle as _spt
    _spt.setproctitle("titan_hcl")
except ImportError:
    pass

# ── tracemalloc: must start BEFORE any titan_hcl imports ──────────
# Captures all Python allocations from this point forward (including imports).
# Config-gated: reads [profiling].tracemalloc_enabled from titan_params.toml.
# nframes=1 keeps overhead at ~5%. Disable via config if unneeded.
_tracemalloc_started = False
try:
    import tomllib
    _params_path = os.path.join(os.path.dirname(__file__), "..", "titan_hcl", "titan_params.toml")
    with open(_params_path, "rb") as _pf:
        _prof_cfg = tomllib.load(_pf).get("profiling", {})
    if _prof_cfg.get("tracemalloc_enabled", True):
        import tracemalloc
        tracemalloc.start(_prof_cfg.get("tracemalloc_nframes", 1))
        _tracemalloc_started = True
except Exception:
    pass  # Config not found or parse error — skip silently


def setup_logging():
    """Configure logging based on merged config plugin_log_level."""
    try:
        from titan_hcl.config_loader import load_titan_config
        level_str = load_titan_config().get("openclaw", {}).get("plugin_log_level", "INFO")
    except Exception:
        level_str = "INFO"

    level = getattr(logging, level_str.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def print_banner():
    print()
    print("=" * 60)
    print("  TITAN — Sovereign Agent Mode")
    print("=" * 60)
    print()


def print_status(plugin):
    """Print a compact Bio-State summary."""
    try:
        mode = plugin._last_execution_mode
        limbo = "LIMBO" if plugin._limbo_mode else "ACTIVE"
        nodes = plugin.memory.get_persistent_count() if plugin.memory else 0
        mood = plugin.mood_engine.get_mood_label() if plugin.mood_engine else "N/A"
        meditating = plugin._is_meditating

        print(f"\n  Bio-State: {limbo} | Mood: {mood} | Nodes: {nodes} | Mode: {mode}")
        if meditating:
            print("  [Meditation cycle in progress...]")
        print()
    except Exception:
        print("  [Bio-State unavailable]")
        print()


async def prompt_loop(plugin, agent):
    """
    Interactive prompt loop using the Agno sovereign agent.
    Each input runs through the full cognitive pipeline:
      Guardian → pre-hook (memory/directives/gatekeeper) → LLM → post-hook (logging/RL)
    """
    print("  Type a prompt to interact with the Titan.")
    print("  Commands: /status, /health, /quit")
    print()

    loop = asyncio.get_event_loop()

    while True:
        try:
            line = await loop.run_in_executor(None, lambda: input("  You > "))
            line = line.strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Shutting down gracefully...")
            break

        if not line:
            continue

        if line == "/quit":
            print("  Shutting down gracefully...")
            break

        if line == "/status":
            print_status(plugin)
            continue

        if line == "/health":
            await _print_health(plugin)
            continue

        if plugin._limbo_mode:
            print("  [LIMBO] The Titan has no brain. Awaiting resurrection at POST /maker/resurrect.")
            continue

        # Run through Agno agent (full cognitive pipeline)
        try:
            run_output = await agent.arun(line)

            response_text = ""
            if hasattr(run_output, "content"):
                response_text = str(run_output.content)
            elif isinstance(run_output, str):
                response_text = run_output
            else:
                response_text = str(run_output)

            mode = plugin._last_execution_mode
            mood = plugin.mood_engine.get_mood_label() if plugin.mood_engine else "N/A"
            print(f"\n  [{mode}|{mood}] Titan > {response_text}\n")

        except ValueError as e:
            if "Sovereignty Violation" in str(e):
                print(f"\n  [GUARDIAN BLOCKED] {e}\n")
            else:
                logging.error("Agent error: %s", e)
                print(f"\n  [ERROR] {e}\n")
        except Exception as e:
            logging.error("Agent error: %s", e, exc_info=True)
            print(f"\n  [ERROR] {e}\n")


async def _print_health(plugin):
    """Print health status inline."""
    print()
    try:
        from titan_hcl.utils.solana_client import is_available as solana_ok

        checks = {
            "Solana SDK": "active" if solana_ok() else "degraded",
            "Memory": "active" if plugin.memory else "absent",
            "Metabolism": "active" if plugin.metabolism else "absent",
            "Soul": "active" if plugin.soul else "absent",
            "Guardian": "active" if plugin.guardian else "absent",
            "Studio": "active" if getattr(plugin, "studio", None) else "absent",
            "Social": "active" if plugin.social else "absent",
            "Agent": "active",
            "Observatory": "active",
        }

        for name, status in checks.items():
            symbol = "+" if status == "active" else ("~" if status == "degraded" else "-")
            print(f"  [{symbol}] {name}: {status}")

        if plugin.memory:
            nodes = plugin.memory.get_persistent_count()
            print(f"\n  Persistent nodes: {nodes}")

    except Exception as e:
        print(f"  Health check error: {e}")
    print()


def _resolve_asyncio_pool_size(config: dict | None = None) -> int:
    """Resolve the asyncio default-executor pool size from config.

    Pre-V6 (legacy monolith): hardcoded max_workers=64 to handle 122
    FastAPI endpoints in-parent under peak Observatory load.

    Post-V6 (microkernel.api_process_separation_enabled=true): FastAPI
    moved to `api_subprocess`; parent services only bus + background
    coroutines + ~140 `asyncio.to_thread` call sites in worker proxies.
    Each pool thread reserves ~8 MB of VmData stack; bounding at 16 saves
    ~400 MB of address space vs the old 64. See rFP_microkernel_phase_a8
    §A.8.2 §3.4 (parent thread-count target ≤25).

    Config knobs (`[microkernel]`):
      `asyncio_pool_bounded_enabled` (default true) — flag-on bounding
      `asyncio_pool_max_workers`     (default 16)   — bounded size

    Returns the effective max_workers integer.
    """
    if config is None:
        try:
            from titan_hcl.config_loader import load_titan_config
            config = load_titan_config()
        except Exception:
            config = {}
    mk = config.get("microkernel", {}) if isinstance(config, dict) else {}
    if not isinstance(mk, dict):
        mk = {}
    if not mk.get("asyncio_pool_bounded_enabled", True):
        return 64  # legacy escape hatch
    try:
        return int(mk.get("asyncio_pool_max_workers", 16))
    except (TypeError, ValueError):
        return 16


_orchestrator_stop_event = None  # threading.Event set in finally to halt helper threads
_orchestrator_client = None      # BusSocketClient stopped in finally
_orchestrator_ref = None         # Orchestrator instance stopped in finally


def _start_lifecycle_subscriber(bus, orchestrator, stop_event):
    """Subscribe to MODULE_*_REQUEST messages and dispatch to Orchestrator.

    Phase 11 §11.I.1 / D-SPEC-141: moved here from `scripts/guardian_hcl.py`
    because the Orchestrator now lives in `titan_hcl` post-split. The
    Supervisor in guardian_hcl emits MODULE_RESTART_REQUEST(dst=
    "guardian_hcl_lifecycle"); titan_hcl's "guardian" BusSocketClient (which
    ALSO matches guardian_hcl_lifecycle? — no, it doesn't) receives
    *broadcast* and dst="guardian" messages. To keep MODULE_RESTART_REQUEST
    routable in the split topology, the Supervisor's publish target keeps
    the legacy dst (`guardian_hcl_lifecycle`) and we mount a local
    DivineBus subscriber here — the inbound dispatcher re-publishes every
    inbound frame into the local DivineBus regardless of original dst, so
    a local types=[MODULE_*_REQUEST] subscriber catches everything that
    reached this process via the broker (broker routes by name+alias; we
    register `guardian_hcl_lifecycle` as an alias on the bus client
    elsewhere — see build_bus_and_client's BusSocketClient construction).
    """
    import logging as _logging
    from queue import Empty
    from titan_hcl.bus import (
        MODULE_START_REQUEST, MODULE_STOP_REQUEST, MODULE_RESTART_REQUEST,
    )
    log = _logging.getLogger(__name__)
    q = bus.subscribe(
        "guardian_hcl_lifecycle",
        types=[MODULE_START_REQUEST, MODULE_STOP_REQUEST, MODULE_RESTART_REQUEST],
        reply_only=True,
    )

    def _loop():
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
                    orchestrator.start(name)
                elif mtype == MODULE_STOP_REQUEST:
                    orchestrator.stop(name, reason=payload.get("reason", "requested"))
                elif mtype == MODULE_RESTART_REQUEST:
                    extra = {k: v for k, v in payload.items()
                             if k not in ("name", "reason")}
                    orchestrator.restart_module(
                        name,
                        reason=payload.get("reason", "requested"),
                        **extra,
                    )
            except Exception as e:  # noqa: BLE001
                log.warning(
                    "[titan_hcl] lifecycle request %s for '%s' failed: %s",
                    mtype, name, e)

    t = threading.Thread(
        target=_loop, name="titan-hcl-lifecycle", daemon=True)
    t.start()
    return t


async def run(health_only: bool = False, server_only: bool = False,
              restore_from: str | None = None, shadow_port: int | None = None):
    """Boot the Titan and enter the main loop.

    Guardian microkernel is the sole boot path (V2 monolith removed 2026-04-03).
    """
    setup_logging()

    # ── Bound asyncio default thread pool ────────────────────────────
    # See `_resolve_asyncio_pool_size` for sizing rationale (V6 microkernel
    # mode → 16 workers vs legacy hardcoded 64). Closes a load-bearing
    # contributor to BUG-PARENT-MEMORY-LEAK-HOST-OOM-20260428: 64 thread
    # stacks × 8 MB ≈ 512 MB VmData reservation, dominant share of the
    # parent's pre-fix 2.78 GB VmData watermark.
    import concurrent.futures
    _max_workers = _resolve_asyncio_pool_size()
    _executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=_max_workers, thread_name_prefix="asyncio")
    asyncio.get_running_loop().set_default_executor(_executor)
    logging.getLogger(__name__).info(
        "[asyncio] default executor bounded: max_workers=%d", _max_workers)

    # ── Disk sanity check (Layer 3) ──────────────────────────────────
    # Refuse to boot if the working disk is critically full. Writing to a
    # full disk corrupts persistent state (FAISS 0-byte incident 2026-04-14).
    # Exit code 2 distinguishes from generic crashes so watchdog / operators
    # can see the real cause.
    from titan_hcl.core.disk_health import assert_disk_bootable
    assert_disk_bootable(os.getcwd())

    # Load config for wallet path override
    wallet_path = os.path.join(os.path.dirname(__file__), "..", "authority.json")
    try:
        from titan_hcl.config_loader import load_titan_config
        cfg = load_titan_config()
        wallet_path = cfg.get("network", {}).get("wallet_keypair_path", wallet_path)
    except Exception:
        cfg = {}

    # ── Phase 11 §11.I.1 / D-SPEC-141 — Orchestrator boot (peer-spawn) ──
    # Pre-Phase-11 the Orchestrator + start_all + lifecycle subscriber
    # lived in scripts/guardian_hcl.py (which then Popen'd this script as
    # the L2 plugin). Under kernel-rs peer-spawn (titan_hcl is now a
    # kernel-rs child, sibling to guardian_hcl), the Orchestrator moves
    # HERE: this process owns module spawn + start_all + the
    # MODULE_*_REQUEST lifecycle subscriber. guardian_hcl is reduced to
    # the Supervisor role (fault detection + restart trigger). The kernel
    # below builds its own DivineBus + "titan_HCL" BusSocketClient
    # connection — that's the proxy-reply transport, separate from the
    # orchestrator bus constructed here (broker fans by client name; the
    # two connections coexist on the same Unix socket without conflict).
    from titan_hcl.config_loader import load_titan_config as _load_cfg
    from titan_hcl.core.state_registry import resolve_titan_id as _resolve_id
    from titan_hcl.bus import (
        MODULE_HEARTBEAT, MODULE_READY, MODULE_SHUTDOWN, MODULE_CRASHED,
        MODULE_RELOAD_REQUEST, BUS_WORKER_ADOPT_REQUEST, BUS_PEER_DIED,
    )
    from scripts._titan_bus_client_helpers import (
        build_bus_and_client as _build_bus_and_client,
        start_inbound_dispatcher as _start_inbound_dispatcher,
    )
    _orch_cfg = _load_cfg()
    _orch_titan_id = _resolve_id()
    _orch_stop = threading.Event()
    _orch_broadcast = [
        MODULE_HEARTBEAT, MODULE_READY, MODULE_SHUTDOWN, MODULE_CRASHED,
        MODULE_RELOAD_REQUEST, BUS_WORKER_ADOPT_REQUEST, BUS_PEER_DIED,
    ]
    _orch_bus, _orch_client = _build_bus_and_client(
        _orch_titan_id, _orch_cfg,
        subscriber_name="guardian",
        broadcast_topics=_orch_broadcast,
        reply_only=False,
    )
    # Phase 11 §11.I.1 routing fix — the Supervisor (guardian_hcl process)
    # publishes MODULE_RESTART_REQUEST with dst="guardian_hcl_lifecycle".
    # The lifecycle subscriber now lives HERE (titan_hcl), so register
    # "guardian_hcl_lifecycle" as a broker alias on this connection — the
    # broker then fans dst="guardian_hcl_lifecycle" frames to this client
    # (alongside dst="guardian"). Without this the restart request hits no
    # subscriber and Supervisor re-fires every tick (memory restart storm
    # observed live T3 2026-05-28). subscribe_alias persists across
    # reconnects (bus_socket.py:709).
    _orch_client.subscribe_alias("guardian_hcl_lifecycle")
    logging.info(
        "[titan_hcl] orchestrator bus client connected "
        "(name=guardian, alias=guardian_hcl_lifecycle)")

    from titan_hcl.orchestrator import Orchestrator
    from titan_hcl.module_catalog import build_catalog
    _orchestrator = Orchestrator(_orch_bus, config=_orch_cfg.get("guardian", {}))
    # _kernel_ref = None: cross-process swap interlock degrades to no-op
    # (per Orchestrator.start docstring "None in legacy mode → swap
    # interlock degrades to no-op"). The TitanKernel constructed below
    # is its OWN process-local kernel ref; the cross-process shadow
    # swap interlock isn't applicable to this orchestrator path because
    # the workers live as siblings in the kernel-rs process tree.
    _orchestrator._kernel_ref = None

    # Phase 11 §11.I.7 / G21 single-writer (D-SPEC-141) — mark this
    # process as the canonical writer of titan_hcl_state.bin so the
    # Orchestrator's _ensure_titan_hcl_state_writer creates the writer
    # here and ONLY here. Non-canonical processes (api subprocess
    # mini-orchestrators, test fixtures) inherit the absence of this
    # env var and silently skip the publish, eliminating the slot
    # clobbering observed live 2026-05-27. Moved from guardian_hcl.py
    # as part of the Phase 11 final split (this process now owns
    # start_all, so it MUST own the canonical state writer).
    os.environ["TITAN_HCL_STATE_WRITER_CANONICAL"] = "1"

    build_catalog(_orch_bus, _orchestrator, _orch_cfg, titan_id=_orch_titan_id)
    logging.info(
        "[titan_hcl] module catalog built — %d modules registered",
        len(_orchestrator._modules))

    _start_inbound_dispatcher(_orch_bus, _orch_client, _orch_stop)
    _start_lifecycle_subscriber(_orch_bus, _orchestrator, _orch_stop)

    _orchestrator.start_all()
    logging.info(
        "[titan_hcl] start_all complete — modules: %s",
        list(_orchestrator._modules.keys()))

    # Stash for the `finally` block at the bottom of run() so shutdown
    # tears down workers + bus client cleanly.
    global _orchestrator_stop_event, _orchestrator_client, _orchestrator_ref
    _orchestrator_stop_event = _orch_stop
    _orchestrator_client = _orch_client
    _orchestrator_ref = _orchestrator

    # ── Guardian Microkernel Boot (kernel/plugin split) ──────────────
    # Phase C is the SOLE boot path. The legacy TitanCore monolith
    # (kernel_plugin_split_enabled=false) was retired 2026-05-21 per
    # rFP_phase_c_titan_hcl_cleanup — production runs the kernel+plugin
    # split exclusively (l0_rust_enabled=true fleet-wide). The flag and
    # the legacy_core.py fallback are gone (no shims; old path deleted).
    print()
    print("=" * 60)
    print("  TITAN — Microkernel Mode (kernel/plugin split)")
    print("=" * 60)
    print()

    from titan_hcl.core.kernel import TitanKernel
    from titan_hcl.core.plugin import TitanHCL
    logging.info("Booting Titan microkernel v2 (kernel+plugin split)...")
    kernel = TitanKernel(wallet_path)

    # Microkernel v2 Phase B.1 §4 — Shadow Core Swap restore hook.
    # When orchestrator (scripts/shadow_swap.py) boots a shadow kernel
    # via `--restore-from PATH`, verify the snapshot's compatibility
    # with this kernel BEFORE workers spawn. On verify failure we log
    # + continue clean boot — refusing to boot would leave the system
    # without a kernel; orchestrator's HIBERNATE_CANCEL rollback path
    # restores the old kernel instead.
    _restore_result = None
    if restore_from:
        _restore_result = kernel.restore_from_snapshot(restore_from)
        if _restore_result.get("verified"):
            logging.info(
                "[B.1] Shadow restore VERIFIED — event_id=%s gap=%.1fs",
                _restore_result["event_id"][:8],
                _restore_result.get("age_seconds", 0.0),
            )
        else:
            logging.warning(
                "[B.1] Shadow restore REFUSED — reason=%s — clean boot",
                _restore_result.get("reason", "?"),
            )

    # Microkernel v2 Phase B.1 §4 — shadow port override.
    # Orchestrator passes --shadow-port N when booting a shadow kernel
    # on the alternate ping-pong port (7777 ↔ 7779). The API subprocess
    # reads TITAN_API_PORT from env if set, otherwise config.toml [api].
    if shadow_port:
        os.environ["TITAN_API_PORT"] = str(shadow_port)
        logging.info("[B.1] Shadow port set: API will listen on %d", shadow_port)

    # §G5.2 item 5 (D-SPEC-112) — publish the trinity-restoring gain sidecar
    # BEFORE booting workers so the 6 Rust trinity daemons retry-load real
    # titan_params.toml values within their first ~1 s. Daemon kernel-defaults
    # apply only while the sidecar is absent (cold boot < ~1 s). Errors are
    # surfaced and the boot continues — daemons stay on defaults rather than
    # blocking startup (`directive_error_visibility` + substrate continues).
    try:
        from titan_hcl.logic.trinity_restoring_publisher import (
            publish_trinity_restoring_cfg,
        )
        publish_trinity_restoring_cfg()
    except Exception as _e:
        logging.exception(
            "[trinity_restoring] publish failed; daemons will run on crate "
            "DEFAULT_* gains until next CONFIG_RELOAD: %s",
            _e,
        )

    core = TitanHCL(kernel)
    await core.boot()

    # After workers spawn + are ready, surface the restore event_id
    # to the bus so SystemForkBlock can be written by timechain_worker
    # and SYSTEM_RESUMED can be published by spirit_worker.
    if _restore_result and _restore_result.get("verified"):
        from titan_hcl.bus import make_msg, SYSTEM_RESUMED
        kernel.bus.publish(make_msg(
            SYSTEM_RESUMED, src="kernel", dst="all",
            payload={
                "event_id": _restore_result["event_id"],
                "kernel_version_from": _restore_result.get("kernel_version_from"),
                "kernel_version_to": kernel.kernel_version,
                "gap_seconds": _restore_result.get("age_seconds", 0.0),
            },
        ))

    # Downstream code (profiling baseline, agent creation, observatory
    # access, prompt loop, shutdown) uses `core` as the plugin root —
    # TitanHCL exposes the @property facade (bus, guardian, soul,
    # _limbo_mode, _observatory_app, etc.) it relies on.

    # ── Profiling baseline snapshot (after all boot allocations) ──
    if _tracemalloc_started:
        import tracemalloc as _tm
        from titan_hcl.core.profiler import TraceMallocCollector
        core._profiling_collector = TraceMallocCollector(
            cache_ttl=_prof_cfg.get("snapshot_cache_ttl_s", 30.0))
        core._profiling_collector.set_baseline(_tm.take_snapshot())
        logging.info("[Profiling] tracemalloc baseline captured (nframes=%d)",
                     _prof_cfg.get("tracemalloc_nframes", 1))
    else:
        core._profiling_collector = None

    if core._limbo_mode:
        logging.warning("Titan booted in LIMBO STATE.")
    else:
        logging.info("Titan Core online. Modules load on demand.")

    agent = core.create_agent()
    # Cache on plugin so plugin.run_chat() can reach it without going
    # through app.state — needed for the chat bus bridge (parent-side
    # _chat_handler_loop calls self._agent.arun() with no FastAPI Request
    # in scope). See BUG-CHAT-AGENT-NOT-INITIALIZED-API-SUBPROCESS.
    core._agent = agent
    if hasattr(core, '_observatory_app') and core._observatory_app:
        core._observatory_app.state.titan_agent = agent

    v3_status = core.get_v3_status()
    api_port = cfg.get("api", {}).get("port", 7777)
    print(f"  Bus modules:  {v3_status['bus_modules']}")
    print(f"  Bus stats:    {v3_status['bus_stats']}")
    print(f"  Observatory:  http://localhost:{api_port}")
    print()

    if health_only:
        print("  Health check: OK")
        return

    try:
        if server_only:
            print("  Running in API-only mode (--server). Modules load on demand.")
            print("  Press Ctrl+C to stop.\n")
            while True:
                await asyncio.sleep(3600)
        else:
            await prompt_loop(core, agent)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        # Phase 11 §11.I.1 / D-SPEC-141 — Orchestrator owns module
        # lifecycle in this process post-split. core.guardian (the L2
        # plugin's GuardianHCLClient proxy) only emits bus requests, so
        # the authoritative stop_all lives on the local orchestrator
        # instance constructed at run() start (stashed into module-level
        # globals earlier via the single `global` declaration there).
        if _orchestrator_ref is not None:
            try:
                logging.info("Stopping orchestrator modules...")
                _orchestrator_ref.stop_all(reason="shutdown")
            except Exception as _e:  # noqa: BLE001
                logging.warning("orchestrator stop_all error: %s", _e)
            try:
                _orchestrator_ref._module_ready_publisher_stop.set()
            except Exception:
                pass
            try:
                _orchestrator_ref._restart_executor.shutdown(wait=False)
            except Exception:
                pass
        if _orchestrator_stop_event is not None:
            _orchestrator_stop_event.set()
        if _orchestrator_client is not None:
            try:
                _orchestrator_client.stop()
            except Exception:
                pass
        logging.info("Titan session ended.")


_pid_lock_fd = None  # Module-level so finally block can release


def _acquire_pid_lock() -> bool:
    """Prevent multiple titan_hcl instances via atomic file lock (fcntl.flock).

    Returns True if lock acquired, False if another instance holds it.

    2026-04-21 bugfix: previously used `open(pid_path, "w")` which TRUNCATES
    the file BEFORE attempting flock. When another instance was holding the
    lock, truncate succeeded but flock failed — leaving the PID file EMPTY
    and the abort message showing "(PID )" blank. Subsequent restart attempts
    saw the empty file and couldn't diagnose whether it was stale or live,
    causing phantom "Another titan_hcl running" aborts that blocked
    legitimate restarts until manual PID-file removal.

    Fix: open with os.O_RDWR | os.O_CREAT (NO O_TRUNC), try flock FIRST,
    truncate+write PID only if lock acquired. This preserves the live PID
    info in the file when the lock check fails.
    """
    import fcntl
    global _pid_lock_fd

    pid_path = os.path.join(os.path.dirname(__file__), "..", "data", "titan_hcl.pid")
    pid_path = os.path.normpath(pid_path)
    os.makedirs(os.path.dirname(pid_path), exist_ok=True)

    fd = None
    try:
        # Open WITHOUT truncating — preserves existing PID if lock fails.
        fd = os.open(pid_path, os.O_RDWR | os.O_CREAT, 0o644)
        _pid_lock_fd = os.fdopen(fd, "r+")
        fcntl.flock(_pid_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Lock acquired — now safe to truncate and write our PID.
        _pid_lock_fd.seek(0)
        _pid_lock_fd.truncate()
        _pid_lock_fd.write(str(os.getpid()))
        _pid_lock_fd.flush()
        return True
    except (IOError, OSError):
        # Another instance holds the lock — DO NOT TRUNCATE.
        # Read live PID from the intact file.
        old_pid = "?"
        if _pid_lock_fd is not None:
            try:
                _pid_lock_fd.seek(0)
                content = _pid_lock_fd.read().strip()
                if content:
                    old_pid = content
            except Exception:
                pass
            try:
                _pid_lock_fd.close()
            except Exception:
                pass
            _pid_lock_fd = None
        elif fd is not None:
            try:
                os.close(fd)
            except Exception:
                pass
        print(f"\n  *** ABORT: Another titan_hcl is already running (PID {old_pid}) ***")
        print(f"  If this is stale, remove {pid_path} and retry.\n")
        return False


def _release_pid_lock():
    """Release PID file lock and remove file on clean shutdown."""
    import fcntl
    global _pid_lock_fd

    pid_path = os.path.join(os.path.dirname(__file__), "..", "data", "titan_hcl.pid")
    pid_path = os.path.normpath(pid_path)
    if _pid_lock_fd:
        try:
            fcntl.flock(_pid_lock_fd, fcntl.LOCK_UN)
            _pid_lock_fd.close()
        except Exception:
            pass
        _pid_lock_fd = None
    try:
        os.remove(pid_path)
    except FileNotFoundError:
        pass


def main():
    """Entry point for the titan-main console script."""
    import argparse

    parser = argparse.ArgumentParser(description="Titan Sovereign Agent Launcher")
    parser.add_argument("--health-only", action="store_true", help="Boot, run health check, and exit")
    parser.add_argument("--server", action="store_true", help="API-only mode (no interactive prompt loop)")
    # ── Microkernel v2 Phase B.1 — Shadow Core Swap (rFP §347-357) ─────
    parser.add_argument("--shadow-port", type=int, default=None,
        help="Boot kernel on a shadow API port (e.g. 7779) for shadow swap. "
             "Used by scripts/shadow_swap.py. Sets TITAN_API_PORT env var.")
    parser.add_argument("--restore-from", type=str, default=None,
        help="Path to runtime.msgpack from a hibernated kernel. Triggers "
             "compatibility verification + SYSTEM_RESUMED publish on bus.")
    # --v3 kept for backwards compatibility (ignored — Guardian is always used)
    parser.add_argument("--v3", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    # Auto-detect non-interactive stdin (cron, nohup, systemd) → server mode
    server_only = args.server or not os.isatty(sys.stdin.fileno())

    # Prevent duplicate instances (except health-only checks + shadow boots).
    # B.1 §7 — shadow boots (`--shadow-port`) MUST run alongside the
    # original kernel; the PID lock would block the swap. Use a
    # port-suffixed lock so two kernels (original + shadow) can coexist.
    if not args.health_only:
        if args.shadow_port:
            # Shadow kernel — use port-suffixed lock for parallel running.
            global _pid_lock_path  # type: ignore
            _pid_lock_path = os.path.join(
                os.path.dirname(__file__), "..", "data",
                f"titan_hcl.shadow_{args.shadow_port}.pid",
            )
            _pid_lock_path = os.path.normpath(_pid_lock_path)
            # Inline equivalent of _acquire_pid_lock but using the shadow path.
            try:
                import fcntl
                fd = os.open(_pid_lock_path, os.O_RDWR | os.O_CREAT, 0o644)
                lock_fp = os.fdopen(fd, "r+")
                fcntl.flock(lock_fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
                lock_fp.seek(0); lock_fp.truncate()
                lock_fp.write(str(os.getpid())); lock_fp.flush()
                # Stash for cleanup at shutdown
                global _pid_lock_fd  # type: ignore
                _pid_lock_fd = lock_fp
                logging.info("[B.1] Shadow PID lock acquired: %s", _pid_lock_path)
            except (IOError, OSError) as e:
                print(f"\n  *** ABORT: Shadow PID lock {_pid_lock_path} held by another shadow ({e})\n")
                sys.exit(1)
        else:
            if not _acquire_pid_lock():
                sys.exit(1)

    # ── Process group: ensure kill of parent cascades to all children ──
    # Become process group leader. All child processes (Guardian modules)
    # inherit this group. On SIGTERM, we forward to the entire group so
    # no orphans are left behind.
    try:
        os.setpgrp()
    except OSError:
        pass  # May fail if already group leader

    def _sigterm_handler(signum, frame):
        """Convert SIGTERM into clean shutdown via KeyboardInterrupt.

        This triggers the existing finally block in run() which calls
        guardian.stop_all() — the Guardian knows all child PIDs and kills
        them properly (SIGTERM → wait → SIGKILL including process groups).

        After clean shutdown, atexit handler kills any remaining stragglers.
        """
        # Convert to KeyboardInterrupt so asyncio loop exits cleanly
        # and the finally block in run() calls guardian.stop_all()
        raise KeyboardInterrupt("SIGTERM received")

    signal.signal(signal.SIGTERM, _sigterm_handler)
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    import atexit

    def _kill_stragglers():
        """Last-resort cleanup: SIGKILL our process group + any known children."""
        try:
            import psutil
            me = psutil.Process(os.getpid())
            children = me.children(recursive=True)
            for child in children:
                try:
                    child.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except Exception:
            pass
        try:
            os.killpg(os.getpgrp(), signal.SIGKILL)
        except OSError:
            pass

    atexit.register(_kill_stragglers)

    try:
        asyncio.run(run(health_only=args.health_only, server_only=server_only,
                        restore_from=args.restore_from, shadow_port=args.shadow_port))
    except KeyboardInterrupt:
        print("\n  Titan shutdown complete.")
    except SystemExit:
        pass
    finally:
        if not args.health_only:
            _release_pid_lock()


if __name__ == "__main__":
    main()
