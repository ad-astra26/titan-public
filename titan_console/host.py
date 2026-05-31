"""Host resource readouts — stdlib only (no psutil).

Everything here reads Linux /proc + os + shutil so the Console Agent carries
zero third-party dependency surface. CPU pressure is reported as load average
(the meaningful Linux signal) plus a sampled busy-percent derived from two
/proc/stat reads.
"""
from __future__ import annotations

import os
import shutil
import time


def _read_meminfo() -> dict[str, int]:
    """Parse /proc/meminfo into a {key: kB} dict."""
    out: dict[str, int] = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split(":")
                if len(parts) != 2:
                    continue
                key = parts[0].strip()
                val = parts[1].strip().split()
                if val and val[0].isdigit():
                    out[key] = int(val[0])  # kB
    except OSError:
        pass
    return out


def _cpu_times() -> tuple[int, int] | None:
    """Return (idle, total) jiffies from /proc/stat's aggregate cpu line."""
    try:
        with open("/proc/stat") as f:
            for line in f:
                if line.startswith("cpu "):
                    fields = [int(x) for x in line.split()[1:]]
                    idle = fields[3] + (fields[4] if len(fields) > 4 else 0)
                    return idle, sum(fields)
    except (OSError, ValueError):
        pass
    return None


def cpu_busy_percent(sample_s: float = 0.2) -> float | None:
    """Sample /proc/stat twice `sample_s` apart → busy percent (0-100)."""
    a = _cpu_times()
    if a is None:
        return None
    time.sleep(max(0.05, sample_s))
    b = _cpu_times()
    if b is None:
        return None
    idle_d = b[0] - a[0]
    total_d = b[1] - a[1]
    if total_d <= 0:
        return None
    return round(100.0 * (1.0 - idle_d / total_d), 1)


def _uptime_s() -> float | None:
    try:
        with open("/proc/uptime") as f:
            return float(f.read().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def read_host_resources(disk_path: str = "/", *, sample_cpu: bool = True) -> dict:
    """Snapshot host CPU / memory / swap / disk / load / uptime (stdlib only).

    Sizes are bytes; percentages are 0-100 floats. Missing signals come back
    as None rather than raising — the agent must degrade, never crash.
    """
    mem = _read_meminfo()
    kb = 1024
    mem_total = mem.get("MemTotal", 0) * kb
    mem_avail = mem.get("MemAvailable", mem.get("MemFree", 0)) * kb
    mem_used = max(0, mem_total - mem_avail)
    swap_total = mem.get("SwapTotal", 0) * kb
    swap_free = mem.get("SwapFree", 0) * kb
    swap_used = max(0, swap_total - swap_free)

    try:
        du = shutil.disk_usage(disk_path)
        disk = {"total": du.total, "used": du.used, "free": du.free,
                "percent": round(100.0 * du.used / du.total, 1) if du.total else None}
    except OSError:
        disk = {"total": None, "used": None, "free": None, "percent": None}

    try:
        load1, load5, load15 = os.getloadavg()
    except (OSError, AttributeError):
        load1 = load5 = load15 = None

    ncpu = os.cpu_count() or 1
    return {
        "cpu": {
            "count": ncpu,
            "busy_percent": cpu_busy_percent() if sample_cpu else None,
            "load1": load1, "load5": load5, "load15": load15,
            "load1_per_core": round(load1 / ncpu, 2) if load1 is not None else None,
        },
        "memory": {
            "total": mem_total, "used": mem_used, "available": mem_avail,
            "percent": round(100.0 * mem_used / mem_total, 1) if mem_total else None,
        },
        "swap": {
            "total": swap_total, "used": swap_used,
            "percent": round(100.0 * swap_used / swap_total, 1) if swap_total else None,
        },
        "disk": disk,
        "uptime_s": _uptime_s(),
    }
