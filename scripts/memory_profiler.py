#!/usr/bin/env python3
"""
scripts/memory_profiler.py
Logs top-5 memory-consuming processes every N seconds during endurance tests.

Outputs TSV to stdout (pipe to file). Designed to be correlated with
titan_agent.log and endurance test logs by timestamp.

Usage:
  python3 scripts/memory_profiler.py > /tmp/titan_memory_profile.tsv
  python3 scripts/memory_profiler.py --interval 5 --duration 7200
"""
import argparse
import subprocess
import time
from datetime import datetime, timezone


def get_top_processes(n: int = 5) -> list[dict]:
    """Return top N processes by RSS memory."""
    result = subprocess.run(
        ["ps", "aux", "--sort=-rss"],
        capture_output=True, text=True, timeout=10,
    )
    lines = result.stdout.strip().split("\n")
    processes = []
    for line in lines[1:n + 1]:  # skip header
        parts = line.split(None, 10)
        if len(parts) < 11:
            continue
        processes.append({
            "user": parts[0],
            "pid": parts[1],
            "cpu": parts[2],
            "mem": parts[3],
            "vsz_kb": int(parts[4]),
            "rss_kb": int(parts[5]),
            "command": parts[10][:80],
        })
    return processes


def get_system_memory() -> dict:
    """Get system memory stats from /proc/meminfo."""
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith(("MemTotal:", "MemAvailable:", "SwapTotal:", "SwapFree:")):
                key, val = line.split(":")
                info[key.strip()] = int(val.strip().split()[0])  # kB
    return {
        "total_mb": info.get("MemTotal", 0) // 1024,
        "available_mb": info.get("MemAvailable", 0) // 1024,
        "used_pct": round(100 * (1 - info.get("MemAvailable", 0) / max(info.get("MemTotal", 1), 1)), 1),
        "swap_used_mb": (info.get("SwapTotal", 0) - info.get("SwapFree", 0)) // 1024,
    }


def main():
    parser = argparse.ArgumentParser(description="Memory profiler for Titan endurance tests")
    parser.add_argument("--interval", type=int, default=10, help="Seconds between samples (default: 10)")
    parser.add_argument("--duration", type=int, default=7200, help="Total profiling duration in seconds (default: 7200)")
    parser.add_argument("--top", type=int, default=5, help="Number of top processes to log (default: 5)")
    args = parser.parse_args()

    # Header
    print("timestamp\tsys_used_pct\tsys_avail_mb\tswap_mb\trank\tpid\trss_mb\tcpu%\tmem%\tcommand")

    start = time.time()
    while time.time() - start < args.duration:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        sys_mem = get_system_memory()
        processes = get_top_processes(args.top)

        for rank, proc in enumerate(processes, 1):
            rss_mb = proc["rss_kb"] / 1024
            print(
                f"{ts}\t{sys_mem['used_pct']}\t{sys_mem['available_mb']}\t{sys_mem['swap_used_mb']}"
                f"\t{rank}\t{proc['pid']}\t{rss_mb:.0f}\t{proc['cpu']}\t{proc['mem']}\t{proc['command']}",
                flush=True,
            )

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
