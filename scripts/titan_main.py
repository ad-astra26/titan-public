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


async def run(health_only: bool = False, server_only: bool = False):
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
    from titan_plugin.v5_core import TitanCore

    print()
    print("=" * 60)
    print("  TITAN — Microkernel Mode")
    print("=" * 60)
    print()

    logging.info("Booting Titan microkernel...")
    core = TitanCore(wallet_path)
    await core.boot()

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
    """
    import fcntl
    global _pid_lock_fd

    pid_path = os.path.join(os.path.dirname(__file__), "..", "data", "titan_main.pid")
    pid_path = os.path.normpath(pid_path)
    os.makedirs(os.path.dirname(pid_path), exist_ok=True)

    try:
        _pid_lock_fd = open(pid_path, "w")
        fcntl.flock(_pid_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _pid_lock_fd.write(str(os.getpid()))
        _pid_lock_fd.flush()
        return True
    except (IOError, OSError):
        # Another instance holds the lock
        try:
            with open(pid_path) as f:
                old_pid = f.read().strip()
        except Exception:
            old_pid = "?"
        print(f"\n  *** ABORT: Another titan_main is already running (PID {old_pid}) ***")
        print(f"  If this is stale, remove {pid_path} and retry.\n")
        if _pid_lock_fd:
            _pid_lock_fd.close()
            _pid_lock_fd = None
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
    # --v3 kept for backwards compatibility (ignored — Guardian is always used)
    parser.add_argument("--v3", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    # Auto-detect non-interactive stdin (cron, nohup, systemd) → server mode
    server_only = args.server or not os.isatty(sys.stdin.fileno())

    # Prevent duplicate instances (except health-only checks)
    if not args.health_only:
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
        asyncio.run(run(health_only=args.health_only, server_only=server_only))
    except KeyboardInterrupt:
        print("\n  Titan shutdown complete.")
    except SystemExit:
        pass
    finally:
        if not args.health_only:
            _release_pid_lock()


if __name__ == "__main__":
    main()
