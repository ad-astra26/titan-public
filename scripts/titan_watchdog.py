"""
scripts/titan_watchdog.py — Self-healing watchdog for Titan process management.

Monitors the Titan backend with multi-layer health checks and auto-recovery.
Designed for production uptime — Titan should self-heal without human intervention.

Health Layers:
  1. Process alive (PID exists)
  2. HTTP health endpoint responsive (/health)
  3. System resources (RAM, CPU load, disk)
  4. Inference health (API error rate tracking)
  5. Heartbeat staleness (last successful response timestamp)

Recovery Actions:
  - Soft restart: SIGTERM → wait → SIGKILL if needed
  - Pre-crash detection: graceful shutdown before OOM
  - Backoff: exponential delay between restarts to avoid crash loops
  - Alert logging: structured events for monitoring

Usage:
    python scripts/titan_watchdog.py
"""
import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [Watchdog] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/tmp/titan_watchdog.log", mode="a"),
    ],
)
log = logging.getLogger("watchdog")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TITAN_CMD = [
    sys.executable, str(PROJECT_ROOT / "scripts" / "titan_main.py"), "--server"
]
TITAN_LOG = "/tmp/titan_agent.log"
TITAN_URL = "http://127.0.0.1:7777"
WATCHDOG_PORT = 7779  # 7778 is T3's API port

# ─── Thresholds ─────────────────────────────────────────────────────────────

HEALTH_CHECK_INTERVAL = 60        # seconds between health polls (mainnet-safe: reduced false restarts)
HEALTH_TIMEOUT = 15               # HTTP timeout for /health endpoint
MAX_CONSECUTIVE_FAILURES = 3      # failures before auto-restart
MAX_RESTARTS_PER_HOUR = 5         # rate limit: restarts per hour
MEMORY_WARN_PCT = 80              # log warning when RAM usage exceeds this %
MEMORY_CRITICAL_PCT = 92          # force restart when RAM usage exceeds this %
TITAN_RSS_CRITICAL_MB = 4200      # proactive restart when Titan RSS exceeds this (MB)
                                  # Note: ~2GB is TorchRL mmap (shared, not real heap)
LOAD_AVG_CRITICAL = 8.0           # force restart if 1-min load avg exceeds this
HEARTBEAT_STALE_SECONDS = 600     # restart if no successful /health for 10 minutes
RESTART_BACKOFF_BASE = 10         # base seconds for exponential backoff
RESTART_BACKOFF_MAX = 300         # max backoff: 5 minutes


# ─── State ──────────────────────────────────────────────────────────────────

@dataclass
class WatchdogState:
    titan_pid: int = 0
    restart_times: list = field(default_factory=list)
    consecutive_health_failures: int = 0
    last_successful_health: float = 0.0
    total_restarts: int = 0
    last_restart_time: float = 0.0
    boot_time: float = field(default_factory=time.time)
    last_health_status: dict = field(default_factory=dict)
    # Resource tracking
    peak_memory_pct: float = 0.0
    last_memory_pct: float = 0.0
    last_load_avg: float = 0.0
    # Inference error tracking
    inference_errors_window: list = field(default_factory=list)
    # FD tracking for log file handle
    log_fh: object = None

state = WatchdogState()

app = FastAPI()


# ─── System Resource Monitoring ─────────────────────────────────────────────

def _get_system_resources() -> dict:
    """Get current system resource usage."""
    resources = {
        "memory_pct": 0.0,
        "memory_available_mb": 0,
        "memory_total_mb": 0,
        "load_avg_1m": 0.0,
        "load_avg_5m": 0.0,
        "load_avg_15m": 0.0,
        "disk_free_gb": 0.0,
    }

    # Memory from /proc/meminfo
    try:
        with open("/proc/meminfo") as f:
            meminfo = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    meminfo[parts[0].rstrip(":")] = int(parts[1])
            total_kb = meminfo.get("MemTotal", 1)
            available_kb = meminfo.get("MemAvailable", total_kb)
            used_pct = ((total_kb - available_kb) / total_kb) * 100
            resources["memory_pct"] = round(used_pct, 1)
            resources["memory_available_mb"] = available_kb // 1024
            resources["memory_total_mb"] = total_kb // 1024
    except Exception:
        pass

    # Load average
    try:
        load1, load5, load15 = os.getloadavg()
        resources["load_avg_1m"] = round(load1, 2)
        resources["load_avg_5m"] = round(load5, 2)
        resources["load_avg_15m"] = round(load15, 2)
    except Exception:
        pass

    # Disk space
    try:
        st = os.statvfs(str(PROJECT_ROOT))
        free_gb = (st.f_bavail * st.f_frsize) / (1024 ** 3)
        resources["disk_free_gb"] = round(free_gb, 1)
    except Exception:
        pass

    return resources


def _get_titan_process_memory_mb() -> float:
    """Get Titan process RSS memory in MB from /proc."""
    if not state.titan_pid:
        return 0.0
    try:
        with open(f"/proc/{state.titan_pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024  # KB to MB
    except Exception:
        pass
    return 0.0


# ─── Process Management ────────────────────────────────────────────────────

def _start_titan() -> int:
    """Start the Titan backend process."""
    env = os.environ.copy()
    env.pop("OPENROUTER_API_KEY", None)

    # Close previous log file handle to avoid FD leak
    if state.log_fh is not None:
        try:
            state.log_fh.close()
        except Exception:
            pass
    state.log_fh = open(TITAN_LOG, "a")  # Append mode to preserve crash context
    proc = subprocess.Popen(
        TITAN_CMD,
        stdout=state.log_fh,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=str(PROJECT_ROOT),
    )
    state.titan_pid = proc.pid
    state.last_restart_time = time.time()
    state.consecutive_health_failures = 0
    log.info("Titan started with PID %d", state.titan_pid)
    return state.titan_pid


def _stop_titan(reason: str = "requested") -> None:
    """Stop ALL Titan backend processes (prevents zombie accumulation)."""
    log.info("Stopping Titan (reason: %s)", reason)

    # Kill the known PID first
    if state.titan_pid:
        try:
            os.kill(state.titan_pid, signal.SIGTERM)
            for _ in range(20):
                time.sleep(0.5)
                try:
                    os.kill(state.titan_pid, 0)
                except OSError:
                    log.info("Titan PID %d stopped gracefully", state.titan_pid)
                    break
            else:
                os.kill(state.titan_pid, signal.SIGKILL)
                log.info("Force-killed Titan PID %d", state.titan_pid)
                time.sleep(1)
        except OSError:
            log.info("Titan PID %d already stopped", state.titan_pid)

    # Sweep for any orphaned titan_main processes (zombie prevention)
    import subprocess
    my_pid = os.getpid()
    try:
        result = subprocess.run(
            ["pgrep", "-f", "titan_main.py --server"],
            capture_output=True, text=True, timeout=5,
        )
        orphan_pids = [int(p) for p in result.stdout.strip().split() if p]
        for pid in orphan_pids:
            if pid == my_pid:
                continue
            log.warning("Stopping orphaned titan_main PID %d (SIGTERM)", pid)
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                continue
        # Grace period: wait 3s then SIGKILL any survivors
        if orphan_pids:
            time.sleep(3)
            for pid in orphan_pids:
                if pid == my_pid:
                    continue
                try:
                    os.kill(pid, 0)  # check if still alive
                except OSError:
                    continue  # already dead
                log.warning("Force-killing orphaned titan_main PID %d (SIGKILL)", pid)
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
    except Exception:
        pass

    state.titan_pid = 0


def _check_rate_limit() -> bool:
    """Return True if restart is allowed within rate limits."""
    now = time.time()
    hour_ago = now - 3600
    state.restart_times = [t for t in state.restart_times if t > hour_ago]
    if len(state.restart_times) >= MAX_RESTARTS_PER_HOUR:
        return False
    state.restart_times.append(now)
    return True


def _get_backoff_delay() -> float:
    """Exponential backoff between restarts to avoid crash loops."""
    now = time.time()
    recent = [t for t in state.restart_times if now - t < 3600]
    n = len(recent)
    delay = min(RESTART_BACKOFF_BASE * (2 ** max(0, n - 1)), RESTART_BACKOFF_MAX)
    return delay


def _is_process_alive() -> bool:
    """Check if Titan process is still running."""
    if not state.titan_pid:
        return False
    try:
        os.kill(state.titan_pid, 0)
        return True
    except OSError:
        return False


# ─── Health Checking ────────────────────────────────────────────────────────

async def _check_titan_health() -> dict:
    """
    Multi-layer health check.
    Returns dict with health status and details.
    """
    result = {
        "alive": False,
        "http_ok": False,
        "resources_ok": True,
        "details": {},
        "action": None,  # "restart", "warn", or None
        "reason": "",
    }

    # Layer 1: Process alive
    result["alive"] = _is_process_alive()
    if not result["alive"]:
        result["action"] = "restart"
        result["reason"] = "process_dead"
        return result

    # Layer 2: HTTP health endpoint
    try:
        async with httpx.AsyncClient(timeout=HEALTH_TIMEOUT) as client:
            resp = await client.get(f"{TITAN_URL}/health")
            if resp.status_code == 200:
                result["http_ok"] = True
                result["details"] = resp.json().get("data", {})
                state.last_successful_health = time.time()
                state.consecutive_health_failures = 0

                # Check for inference errors in health response
                ollama = result["details"].get("ollama_cloud", {})
                if ollama:
                    state.last_health_status = result["details"]
            else:
                state.consecutive_health_failures += 1
                log.warning("Health check returned HTTP %d", resp.status_code)
    except httpx.TimeoutException:
        state.consecutive_health_failures += 1
        log.warning("Health check timed out (%ds)", HEALTH_TIMEOUT)
    except Exception as e:
        state.consecutive_health_failures += 1
        log.warning("Health check failed: %s", e)

    # Too many consecutive failures → restart
    if state.consecutive_health_failures >= MAX_CONSECUTIVE_FAILURES:
        result["action"] = "restart"
        result["reason"] = f"health_unresponsive_{state.consecutive_health_failures}_consecutive"
        return result

    # Layer 3: Heartbeat staleness
    if state.last_successful_health > 0:
        stale = time.time() - state.last_successful_health
        if stale > HEARTBEAT_STALE_SECONDS:
            result["action"] = "restart"
            result["reason"] = f"heartbeat_stale_{int(stale)}s"
            return result

    # Layer 4: System resources
    resources = _get_system_resources()
    state.last_memory_pct = resources["memory_pct"]
    state.last_load_avg = resources["load_avg_1m"]
    if resources["memory_pct"] > state.peak_memory_pct:
        state.peak_memory_pct = resources["memory_pct"]

    titan_rss_mb = _get_titan_process_memory_mb()

    # Proactive RSS-based restart — catch growth before Titan becomes unresponsive
    if titan_rss_mb > 0 and titan_rss_mb >= TITAN_RSS_CRITICAL_MB:
        result["action"] = "restart"
        result["reason"] = f"titan_rss_{titan_rss_mb:.0f}mb"
        result["resources_ok"] = False
        log.warning(
            "TITAN RSS CRITICAL: %.0fMB (threshold: %dMB). Proactive restart.",
            titan_rss_mb, TITAN_RSS_CRITICAL_MB,
        )
        return result

    if resources["memory_pct"] >= MEMORY_CRITICAL_PCT:
        result["action"] = "restart"
        result["reason"] = f"memory_critical_{resources['memory_pct']:.1f}%"
        result["resources_ok"] = False
        log.critical(
            "MEMORY CRITICAL: %.1f%% used (available: %dMB, Titan RSS: %.0fMB). Restarting.",
            resources["memory_pct"], resources["memory_available_mb"], titan_rss_mb,
        )
        return result
    elif resources["memory_pct"] >= MEMORY_WARN_PCT:
        result["resources_ok"] = False
        log.warning(
            "MEMORY WARNING: %.1f%% used (available: %dMB, Titan RSS: %.0fMB)",
            resources["memory_pct"], resources["memory_available_mb"], titan_rss_mb,
        )

    if resources["load_avg_1m"] >= LOAD_AVG_CRITICAL:
        result["resources_ok"] = False
        log.warning("LOAD WARNING: 1m avg = %.1f (threshold: %.1f)", resources["load_avg_1m"], LOAD_AVG_CRITICAL)

    if resources["disk_free_gb"] < 1.0:
        log.warning("DISK WARNING: Only %.1f GB free", resources["disk_free_gb"])

    return result


# ─── Background Health Monitor Loop ────────────────────────────────────────

async def _health_monitor():
    """
    Background loop: multi-layer health monitoring with auto-recovery.
    This is the core self-healing loop.
    """
    # Wait for initial boot
    await asyncio.sleep(30)
    log.info("Health monitor active. Polling every %ds.", HEALTH_CHECK_INTERVAL)

    while True:
        try:
            health = await _check_titan_health()

            if health["action"] == "restart":
                log.warning("AUTO-RESTART triggered: %s", health["reason"])

                if not _check_rate_limit():
                    log.error(
                        "Rate limit exceeded (%d/%d per hour). NOT restarting. Manual intervention needed.",
                        len(state.restart_times), MAX_RESTARTS_PER_HOUR,
                    )
                    await asyncio.sleep(60)
                    continue

                # Backoff delay
                delay = _get_backoff_delay()
                if delay > RESTART_BACKOFF_BASE:
                    log.info("Backoff: waiting %.0fs before restart (attempt %d)", delay, len(state.restart_times))
                    await asyncio.sleep(delay)

                _stop_titan(reason=health["reason"])
                await asyncio.sleep(3)
                _start_titan()
                state.total_restarts += 1

                log.info(
                    "Restart #%d complete. Reason: %s. Waiting 60s before next health check.",
                    state.total_restarts, health["reason"],
                )
                await asyncio.sleep(60)  # Grace period for Titan to boot
                continue

            # Periodic resource log (every 5 minutes = every 10th check)
            if int(time.time()) % 300 < HEALTH_CHECK_INTERVAL:
                resources = _get_system_resources()
                titan_rss = _get_titan_process_memory_mb()
                log.info(
                    "Resources: RAM %.1f%% (avail %dMB) | Titan RSS %.0fMB | Load %.1f | Disk %.1fGB",
                    resources["memory_pct"], resources["memory_available_mb"],
                    titan_rss, resources["load_avg_1m"], resources["disk_free_gb"],
                )

        except Exception as e:
            log.error("Health monitor error: %s", e)

        await asyncio.sleep(HEALTH_CHECK_INTERVAL)


# ─── API Endpoints ──────────────────────────────────────────────────────────

@app.post("/restart")
async def restart(request: Request):
    """Manual restart endpoint (local-only, rate-limited)."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    requester = body.get("requester", "manual")
    reason = body.get("reason", "manual_request")
    log.info("Manual restart requested by %s (reason: %s)", requester, reason)

    if not _check_rate_limit():
        return JSONResponse(
            status_code=429,
            content={"error": f"Too many restarts. Max {MAX_RESTARTS_PER_HOUR} per hour."},
        )

    _stop_titan(reason=reason)
    await asyncio.sleep(3)
    pid = _start_titan()
    state.total_restarts += 1

    return {"status": "restarting", "pid": pid}


@app.get("/health")
async def health():
    """Watchdog health + Titan status overview."""
    resources = _get_system_resources()
    titan_rss = _get_titan_process_memory_mb()

    return {
        "watchdog": "healthy",
        "uptime_seconds": round(time.time() - state.boot_time),
        "titan": {
            "pid": state.titan_pid,
            "alive": _is_process_alive(),
            "rss_mb": round(titan_rss),
            "last_health_ok": round(time.time() - state.last_successful_health) if state.last_successful_health else None,
            "consecutive_failures": state.consecutive_health_failures,
        },
        "restarts": {
            "total": state.total_restarts,
            "last_hour": len([t for t in state.restart_times if time.time() - t < 3600]),
            "remaining": MAX_RESTARTS_PER_HOUR - len([t for t in state.restart_times if time.time() - t < 3600]),
        },
        "resources": {
            "memory_pct": resources["memory_pct"],
            "memory_available_mb": resources["memory_available_mb"],
            "load_avg_1m": resources["load_avg_1m"],
            "disk_free_gb": resources["disk_free_gb"],
            "titan_rss_mb": round(titan_rss),
            "peak_memory_pct": state.peak_memory_pct,
        },
        "thresholds": {
            "memory_warn_pct": MEMORY_WARN_PCT,
            "memory_critical_pct": MEMORY_CRITICAL_PCT,
            "max_restarts_per_hour": MAX_RESTARTS_PER_HOUR,
            "heartbeat_stale_seconds": HEARTBEAT_STALE_SECONDS,
        },
    }


@app.get("/status")
async def status():
    """Detailed status with recent health history."""
    return {
        "titan_pid": state.titan_pid,
        "alive": _is_process_alive(),
        "total_restarts": state.total_restarts,
        "last_restart": state.last_restart_time,
        "consecutive_failures": state.consecutive_health_failures,
        "last_successful_health_ago": round(time.time() - state.last_successful_health) if state.last_successful_health else None,
        "memory_pct": state.last_memory_pct,
        "peak_memory_pct": state.peak_memory_pct,
        "load_avg": state.last_load_avg,
        "last_health_data": state.last_health_status,
    }


@app.on_event("startup")
async def startup():
    """Start Titan and health monitor on watchdog boot."""
    # Kill any orphaned titan_main processes from previous watchdog sessions
    _stop_titan(reason="watchdog_startup_cleanup")
    _start_titan()
    asyncio.create_task(_health_monitor())


if __name__ == "__main__":
    log.info("Starting Titan Watchdog on port %d (local-only)", WATCHDOG_PORT)
    log.info(
        "Thresholds: health_interval=%ds, max_failures=%d, memory_critical=%d%%, "
        "heartbeat_stale=%ds, max_restarts=%d/hr",
        HEALTH_CHECK_INTERVAL, MAX_CONSECUTIVE_FAILURES, MEMORY_CRITICAL_PCT,
        HEARTBEAT_STALE_SECONDS, MAX_RESTARTS_PER_HOUR,
    )
    uvicorn.run(app, host="127.0.0.1", port=WATCHDOG_PORT, log_level="warning")
