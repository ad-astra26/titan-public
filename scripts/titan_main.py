#!/usr/bin/env python3
"""
titan_main.py — The Titan Sovereign Agent Launcher.

Boots the full Titan cognitive stack and creates an Agno-powered sovereign agent:
  - All subsystems (memory, metabolism, soul, sage, studio, observatory)
  - Circadian rhythms (meditation every 6h, rebirth every 24h)
  - Sovereign Observatory API on configured port (includes POST /chat endpoint)
  - Interactive stdin prompt loop using the Agno agent

Usage:
    titan-main                       # If installed via pip entry_points
    python scripts/titan_main.py     # Direct execution
    python scripts/titan_main.py --health-only  # Boot, health check, exit
"""
import asyncio
import logging
import os
import sys
import signal

# Ensure project root is on path
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

# ── tracemalloc: must start BEFORE any titan_plugin imports ──────────
# Captures all Python allocations from this point forward (including imports).
# Config-gated: reads [profiling].tracemalloc_enabled from titan_params.toml.
# nframes=1 keeps overhead at ~5%. Disable via config if unneeded.
_tracemalloc_started = False
try:
    import tomllib
    _params_path = os.path.join(os.path.dirname(__file__), "..", "titan_plugin", "titan_params.toml")
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
        from titan_plugin.config_loader import load_titan_config
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
        from titan_plugin.utils.solana_client import is_available as solana_ok

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


async def run(health_only: bool = False, server_only: bool = False,
              restore_from: str | None = None, shadow_port: int | None = None):
    """Boot the Titan and enter the main loop.

    Guardian microkernel is the sole boot path (V2 monolith removed 2026-04-03).
    """
    setup_logging()

    # ── Bump asyncio default thread pool ─────────────────────────────
    # Default is min(32, cpu_count+4) ≈ 36. After Phase E.2 wraps 90+
    # sync I/O sites in asyncio.to_thread, concurrent endpoints can
    # exhaust the pool and serialize. 64 workers gives ample headroom
    # for our 122 FastAPI endpoints under peak Observatory load.
    import concurrent.futures
    _executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=64, thread_name_prefix="asyncio")
    asyncio.get_running_loop().set_default_executor(_executor)

    # ── Disk sanity check (Layer 3) ──────────────────────────────────
    # Refuse to boot if the working disk is critically full. Writing to a
    # full disk corrupts persistent state (FAISS 0-byte incident 2026-04-14).
    # Exit code 2 distinguishes from generic crashes so watchdog / operators
    # can see the real cause.
    from titan_plugin.core.disk_health import assert_disk_bootable
    assert_disk_bootable(os.getcwd())

    # Load config for wallet path override
    wallet_path = os.path.join(os.path.dirname(__file__), "..", "authority.json")
    try:
        from titan_plugin.config_loader import load_titan_config
        cfg = load_titan_config()
        wallet_path = cfg.get("network", {}).get("wallet_keypair_path", wallet_path)
    except Exception:
        cfg = {}

    # ── Guardian Microkernel Boot ────────────────────────────────────
    # Microkernel v2 Phase A S3 — flag-branched boot path per
    # titan-docs/PLAN_microkernel_phase_a_s3.md §3 D3 + §4.4.
    # Default false → legacy TitanCore monolith (byte-identical to
    # pre-S3 behavior). Flip on for kernel+plugin split; per-Titan
    # cutover (T2 → T3 → T1) with 24h soak gates per PLAN §8.
    _split_enabled = bool(
        cfg.get("microkernel", {}).get("kernel_plugin_split_enabled", False)
    )

    print()
    print("=" * 60)
    if _split_enabled:
        print("  TITAN — Microkernel Mode (kernel/plugin split)")
    else:
        print("  TITAN — Microkernel Mode (legacy monolith)")
    print("=" * 60)
    print()

    if _split_enabled:
        from titan_plugin.core.kernel import TitanKernel
        from titan_plugin.core.plugin import TitanPlugin
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

        core = TitanPlugin(kernel)
        await core.boot()

        # After workers spawn + are ready, surface the restore event_id
        # to the bus so SystemForkBlock can be written by timechain_worker
        # and SYSTEM_RESUMED can be published by spirit_worker.
        if _restore_result and _restore_result.get("verified"):
            from titan_plugin.bus import make_msg, SYSTEM_RESUMED
            kernel.bus.publish(make_msg(
                SYSTEM_RESUMED, src="kernel", dst="all",
                payload={
                    "event_id": _restore_result["event_id"],
                    "kernel_version_from": _restore_result.get("kernel_version_from"),
                    "kernel_version_to": kernel.kernel_version,
                    "gap_seconds": _restore_result.get("age_seconds", 0.0),
                },
            ))
    else:
        from titan_plugin.legacy_core import TitanCore
        logging.info("Booting Titan microkernel (legacy monolith)...")
        core = TitanCore(wallet_path)
        await core.boot()

    # Downstream code (profiling baseline, agent creation, observatory
    # access, prompt loop, shutdown) uses `core` as the plugin root.
    # TitanPlugin's compat @property facade (bus, guardian, soul,
    # _limbo_mode, _observatory_app, etc.) makes it duck-type-identical
    # to TitanCore — zero further changes needed in this file.

    # ── Profiling baseline snapshot (after all boot allocations) ──
    if _tracemalloc_started:
        import tracemalloc as _tm
        from titan_plugin.core.profiler import TraceMallocCollector
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
        # Clean shutdown: stop all Guardian child processes
        if hasattr(core, 'guardian') and core.guardian:
            logging.info("Stopping Guardian modules...")
            core.guardian.stop_all(reason="shutdown")
        logging.info("Titan session ended.")


_pid_lock_fd = None  # Module-level so finally block can release


def _acquire_pid_lock() -> bool:
    """Prevent multiple titan_main instances via atomic file lock (fcntl.flock).

    Returns True if lock acquired, False if another instance holds it.

    2026-04-21 bugfix: previously used `open(pid_path, "w")` which TRUNCATES
    the file BEFORE attempting flock. When another instance was holding the
    lock, truncate succeeded but flock failed — leaving the PID file EMPTY
    and the abort message showing "(PID )" blank. Subsequent restart attempts
    saw the empty file and couldn't diagnose whether it was stale or live,
    causing phantom "Another titan_main running" aborts that blocked
    legitimate restarts until manual PID-file removal.

    Fix: open with os.O_RDWR | os.O_CREAT (NO O_TRUNC), try flock FIRST,
    truncate+write PID only if lock acquired. This preserves the live PID
    info in the file when the lock check fails.
    """
    import fcntl
    global _pid_lock_fd

    pid_path = os.path.join(os.path.dirname(__file__), "..", "data", "titan_main.pid")
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
        print(f"\n  *** ABORT: Another titan_main is already running (PID {old_pid}) ***")
        print(f"  If this is stale, remove {pid_path} and retry.\n")
        return False


def _release_pid_lock():
    """Release PID file lock and remove file on clean shutdown."""
    import fcntl
    global _pid_lock_fd

    pid_path = os.path.join(os.path.dirname(__file__), "..", "data", "titan_main.pid")
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
                f"titan_main.shadow_{args.shadow_port}.pid",
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
