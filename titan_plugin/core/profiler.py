"""Live memory + CPU profiler for Titan.

Two-tier design:
  Tier 1 (zero overhead): /proc filesystem parsing for RSS, VmPeak, CPU time,
    I/O bytes, threads — parent + all Guardian module PIDs.
  Tier 2 (on-demand, ~5-10%): tracemalloc snapshots showing per-file Python
    allocations, with diff mode to show GROWTH since boot.

No external dependencies — stdlib tracemalloc + /proc only.

Usage:
  API:  GET /v4/admin/memory-profile
  CLI:  python scripts/arch_map.py profile [--all] [--diff] [--cpu]
"""
from __future__ import annotations

import logging
import os
import time
import tracemalloc
from typing import Optional

logger = logging.getLogger(__name__)

# Clock ticks per second (for CPU time conversion from /proc/stat jiffies)
try:
    _SC_CLK_TCK = os.sysconf("SC_CLK_TCK")
except (AttributeError, ValueError):
    _SC_CLK_TCK = 100  # Linux default


# ── Tier 1: /proc Filesystem Reader ──────────────────────────────────

class ProcStatReader:
    """Parse /proc/{pid}/* for memory, CPU, I/O, and thread info.

    Linux-only.  All reads are non-blocking (<1ms each — kernel memcpy).
    Returns empty dicts on error (process gone, permission denied).
    """

    @staticmethod
    def read_memory(pid: int) -> dict:
        """Read /proc/{pid}/status -> VmRSS, VmPeak, VmSize, VmSwap, VmData, Threads."""
        result: dict = {}
        try:
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        result["rss_kb"] = int(line.split()[1])
                    elif line.startswith("VmPeak:"):
                        result["vm_peak_kb"] = int(line.split()[1])
                    elif line.startswith("VmSize:"):
                        result["vm_size_kb"] = int(line.split()[1])
                    elif line.startswith("VmSwap:"):
                        result["vm_swap_kb"] = int(line.split()[1])
                    elif line.startswith("VmData:"):
                        result["vm_data_kb"] = int(line.split()[1])
                    elif line.startswith("Threads:"):
                        result["threads"] = int(line.split()[1])
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
        return result

    @staticmethod
    def read_cpu(pid: int) -> dict:
        """Read /proc/{pid}/stat -> user time, system time (seconds), state."""
        try:
            with open(f"/proc/{pid}/stat") as f:
                parts = f.read().split()
            # Fields: pid (comm) state utime stime ... (0-indexed after split)
            # utime=13, stime=14 (0-indexed)
            return {
                "state": parts[2],
                "utime_s": int(parts[13]) / _SC_CLK_TCK,
                "stime_s": int(parts[14]) / _SC_CLK_TCK,
                "num_threads": int(parts[19]),
            }
        except (FileNotFoundError, ProcessLookupError, PermissionError,
                IndexError, ValueError):
            return {}

    @staticmethod
    def read_io(pid: int) -> dict:
        """Read /proc/{pid}/io -> read_bytes, write_bytes."""
        result: dict = {}
        try:
            with open(f"/proc/{pid}/io") as f:
                for line in f:
                    key, val = line.split(":", 1)
                    key = key.strip()
                    if key in ("read_bytes", "write_bytes", "syscr", "syscw"):
                        result[key] = int(val.strip())
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
        return result

    @staticmethod
    def read_system_memory() -> dict:
        """Read /proc/meminfo -> total, available, swap."""
        result: dict = {}
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        result["total_kb"] = int(line.split()[1])
                    elif line.startswith("MemAvailable:"):
                        result["available_kb"] = int(line.split()[1])
                    elif line.startswith("SwapTotal:"):
                        result["swap_total_kb"] = int(line.split()[1])
                    elif line.startswith("SwapFree:"):
                        result["swap_free_kb"] = int(line.split()[1])
        except (FileNotFoundError, PermissionError):
            pass
        if "total_kb" in result and "available_kb" in result:
            used = result["total_kb"] - result["available_kb"]
            result["used_pct"] = round(used / result["total_kb"] * 100, 1)
        return result

    @staticmethod
    def sample_cpu_percent(pids: dict[str, int], duration_s: float = 1.0) -> dict[str, float]:
        """Sample CPU% for multiple PIDs over a time window.

        Returns {name: cpu_percent} where 100% = one full core.
        """
        # Snapshot t0
        t0_data: dict[str, tuple[float, float]] = {}
        for name, pid in pids.items():
            cpu = ProcStatReader.read_cpu(pid)
            if cpu:
                t0_data[name] = (cpu["utime_s"], cpu["stime_s"])

        time.sleep(duration_s)

        # Snapshot t1 + compute delta
        result: dict[str, float] = {}
        for name, pid in pids.items():
            if name not in t0_data:
                continue
            cpu = ProcStatReader.read_cpu(pid)
            if not cpu:
                continue
            dt_user = cpu["utime_s"] - t0_data[name][0]
            dt_sys = cpu["stime_s"] - t0_data[name][1]
            result[name] = round((dt_user + dt_sys) / duration_s * 100, 1)
        return result


# ── Tier 2: tracemalloc Snapshot Collector ────────────────────────────

class TraceMallocCollector:
    """Wraps tracemalloc snapshot operations with caching and diff support.

    Snapshots are expensive (~50-200ms).  Cache the last one for
    `cache_ttl` seconds to prevent concurrent API requests from
    hammering the tracer.
    """

    def __init__(self, cache_ttl: float = 30.0):
        self._baseline: Optional[tracemalloc.Snapshot] = None
        self._last_snapshot: Optional[tracemalloc.Snapshot] = None
        self._last_snapshot_ts: float = 0.0
        self._cache_ttl = cache_ttl

    @property
    def is_tracing(self) -> bool:
        return tracemalloc.is_tracing()

    def set_baseline(self, snapshot: tracemalloc.Snapshot) -> None:
        """Store the boot-time baseline for diff mode."""
        self._baseline = snapshot

    def take_snapshot(self, force: bool = False) -> Optional[tracemalloc.Snapshot]:
        """Take a new snapshot (or return cached if within TTL)."""
        if not tracemalloc.is_tracing():
            return None
        now = time.time()
        if not force and self._last_snapshot and (now - self._last_snapshot_ts) < self._cache_ttl:
            return self._last_snapshot
        self._last_snapshot = tracemalloc.take_snapshot()
        self._last_snapshot_ts = now
        return self._last_snapshot

    def get_top_stats(self, n: int = 25, key_type: str = "filename") -> list[dict]:
        """Top N allocations by size.  key_type: 'filename' or 'lineno'."""
        snap = self.take_snapshot()
        if snap is None:
            return []
        stats = snap.statistics(key_type)
        return [
            {"file": str(s.traceback), "size_bytes": s.size, "count": s.count}
            for s in stats[:n]
        ]

    def get_diff_stats(self, n: int = 25, key_type: str = "filename") -> list[dict]:
        """Compare current snapshot vs baseline.  Returns GROWTH since boot."""
        if self._baseline is None:
            return self.get_top_stats(n, key_type)
        snap = self.take_snapshot()
        if snap is None:
            return []
        diff = snap.compare_to(self._baseline, key_type)
        return [
            {
                "file": str(d.traceback),
                "size_diff_bytes": d.size_diff,
                "size_bytes": d.size,
                "count_diff": d.count_diff,
                "count": d.count,
            }
            for d in diff[:n]
        ]

    def get_summary(self) -> dict:
        """Return traced memory current/peak."""
        if not tracemalloc.is_tracing():
            return {"active": False}
        current, peak = tracemalloc.get_traced_memory()
        return {
            "active": True,
            "traced_current_bytes": current,
            "traced_peak_bytes": peak,
        }


# ── Profile Report Assembler ─────────────────────────────────────────

class ProfileReport:
    """Assembles a complete profiling report combining /proc + tracemalloc."""

    def __init__(self, collector: Optional[TraceMallocCollector] = None):
        self.collector = collector

    def collect_proc_report(self, guardian) -> dict:
        """Collect /proc stats for parent + all Guardian child PIDs."""
        parent_pid = os.getpid()
        parent_mem = ProcStatReader.read_memory(parent_pid)
        parent_cpu = ProcStatReader.read_cpu(parent_pid)
        parent_io = ProcStatReader.read_io(parent_pid)

        report = {
            "system": _kb_to_mb_dict(ProcStatReader.read_system_memory()),
            "parent": {
                "pid": parent_pid,
                **_kb_to_mb_dict(parent_mem),
                **parent_cpu,
                **parent_io,
            },
            "modules": {},
            "totals": {"total_rss_mb": 0.0, "process_count": 0},
        }

        total_rss = parent_mem.get("rss_kb", 0)
        count = 1  # parent

        # Per-module stats from Guardian
        if guardian is not None:
            status = guardian.get_status()
            for name, info in status.items():
                pid = info.get("pid")
                if not pid:
                    report["modules"][name] = {
                        "state": info.get("state", "unknown"),
                        "pid": None,
                    }
                    continue
                mem = ProcStatReader.read_memory(pid)
                cpu = ProcStatReader.read_cpu(pid)
                io_data = ProcStatReader.read_io(pid)
                report["modules"][name] = {
                    "state": info.get("state", "unknown"),
                    "pid": pid,
                    "rss_limit_mb": info.get("rss_limit_mb"),
                    "uptime_s": info.get("uptime", 0),
                    "restart_count": info.get("restart_count", 0),
                    **_kb_to_mb_dict(mem),
                    **cpu,
                    **io_data,
                }
                total_rss += mem.get("rss_kb", 0)
                count += 1

        report["totals"]["total_rss_mb"] = round(total_rss / 1024, 1)
        report["totals"]["process_count"] = count
        return report

    def collect_parent_tracemalloc(self, top_n: int = 25,
                                    diff: bool = False,
                                    key_type: str = "filename") -> dict:
        """Collect tracemalloc stats for the parent process."""
        if self.collector is None:
            return {"active": False}
        summary = self.collector.get_summary()
        if diff:
            top = self.collector.get_diff_stats(top_n, key_type)
        else:
            top = self.collector.get_top_stats(top_n, key_type)
        # Convert bytes to MB for readability
        for item in top:
            if "size_bytes" in item:
                item["size_mb"] = round(item["size_bytes"] / (1024 * 1024), 2)
            if "size_diff_bytes" in item:
                item["size_diff_mb"] = round(item["size_diff_bytes"] / (1024 * 1024), 2)
        summary_mb = {}
        if summary.get("active"):
            summary_mb = {
                "active": True,
                "traced_current_mb": round(summary["traced_current_bytes"] / (1024 * 1024), 1),
                "traced_peak_mb": round(summary["traced_peak_bytes"] / (1024 * 1024), 1),
            }
        else:
            summary_mb = {"active": False}
        return {**summary_mb, "top": top}

    def collect_cpu_sample(self, guardian, duration_s: float = 1.0) -> dict:
        """Sample CPU% for parent + all modules over a time window."""
        pids: dict[str, int] = {"parent": os.getpid()}
        if guardian is not None:
            status = guardian.get_status()
            for name, info in status.items():
                pid = info.get("pid")
                if pid:
                    pids[name] = pid
        return ProcStatReader.sample_cpu_percent(pids, duration_s)

    def full_report(self, guardian, top_n: int = 25, diff: bool = False,
                    key_type: str = "filename", include_cpu: bool = False,
                    cpu_duration: float = 1.0) -> dict:
        """Assemble the complete profiling report."""
        report = self.collect_proc_report(guardian)
        report["parent"]["tracemalloc"] = self.collect_parent_tracemalloc(
            top_n, diff, key_type)
        if include_cpu:
            cpu_pcts = self.collect_cpu_sample(guardian, cpu_duration)
            report["parent"]["cpu_pct"] = cpu_pcts.get("parent", 0.0)
            for name in report["modules"]:
                if name in cpu_pcts:
                    report["modules"][name]["cpu_pct"] = cpu_pcts[name]
        report["profile_ts"] = time.time()
        return report


# ── Child Process Profiling Utility ───────────────────────────────────

def handle_memory_profile_query(msg: dict, send_queue, name: str) -> bool:
    """Handle a 'get_memory_profile' QUERY in a worker process.

    Call this from any worker's QUERY handler.  Returns True if handled,
    False if the message is not a profiling request (caller should continue
    processing).

    Collects in-process tracemalloc snapshot + /proc/self stats and sends
    the response back via send_queue.
    """
    payload = msg.get("payload", {})
    if payload.get("action") != "get_memory_profile":
        return False

    top_n = int(payload.get("top_n", 20))
    key_type = str(payload.get("key_type", "filename"))
    diff = bool(payload.get("diff", False))

    result: dict = {"module": name, "pid": os.getpid()}

    # tracemalloc (inherited from parent fork — should be active)
    if tracemalloc.is_tracing():
        snapshot = tracemalloc.take_snapshot()
        stats = snapshot.statistics(key_type)
        current, peak = tracemalloc.get_traced_memory()
        result["tracemalloc"] = {
            "active": True,
            "traced_current_mb": round(current / (1024 * 1024), 1),
            "traced_peak_mb": round(peak / (1024 * 1024), 1),
            "top": [
                {"file": str(s.traceback), "size_mb": round(s.size / (1024 * 1024), 2),
                 "size_bytes": s.size, "count": s.count}
                for s in stats[:top_n]
            ],
        }
    else:
        result["tracemalloc"] = {"active": False}

    # /proc/self stats
    pid = os.getpid()
    mem = ProcStatReader.read_memory(pid)
    cpu = ProcStatReader.read_cpu(pid)
    result["proc"] = {**_kb_to_mb_dict(mem), **cpu}

    # Send response
    rid = msg.get("rid", "")
    src = msg.get("src", "")
    response = {
        "type": "RESPONSE",
        "src": f"{name}_proxy",
        "dst": src,
        "rid": rid,
        "payload": result,
        "ts": time.time(),
    }
    try:
        send_queue.put_nowait(response)
    except Exception:
        pass  # Queue full — drop silently
    return True


# ── Helpers ───────────────────────────────────────────────────────────

def _kb_to_mb_dict(d: dict) -> dict:
    """Convert any *_kb keys to *_mb (rounded to 1 decimal)."""
    out: dict = {}
    for k, v in d.items():
        if k.endswith("_kb"):
            out[k.replace("_kb", "_mb")] = round(v / 1024, 1)
        else:
            out[k] = v
    return out
