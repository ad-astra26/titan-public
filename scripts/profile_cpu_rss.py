#!/usr/bin/env python3
"""Per-worker CPU%% + RSS ranking probe (D-SPEC-143 self-identity era).

Runs on the shared VPS. Attributes each process by /proc/<pid>/cwd (T2 vs T3)
and names it from the INV-PROC-7 setproctitle (`titan_hcl:<name>` in cmdline).
CPU%% = (utime+stime delta over WINDOW) / clock_ticks / WINDOW * 100.

Usage: ssh root@HOST python3 - <repo_path> <window_s>   (script on stdin)
"""
import os, sys, glob, time, json

REPO = sys.argv[1] if len(sys.argv) > 1 else "/home/antigravity/projects/titan"
WINDOW = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
HZ = os.sysconf("SC_CLK_TCK")


def _name(pid):
    try:
        argv0 = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\x00")[0].decode("utf-8", "replace")
    except Exception:
        return None
    if argv0.startswith("titan_hcl:"):
        return argv0[len("titan_hcl:"):]
    if argv0 == "titan_hcl":
        return "(orchestrator-parent)"
    if argv0 == "titan_hcl_api":
        return "(api)"
    return None


def _stat_ticks(pid):
    # utime (14) + stime (15) — fields after the (comm) paren group.
    raw = open(f"/proc/{pid}/stat").read()
    fields = raw[raw.rindex(")") + 2:].split()
    return int(fields[11]) + int(fields[12])  # utime, stime (0-indexed post-comm)


def _rss_kb(pid):
    for line in open(f"/proc/{pid}/status"):
        if line.startswith("VmRSS"):
            return int(line.split()[1])
    return 0


# Pass 1
procs = {}
for pd in glob.glob("/proc/[0-9]*"):
    pid = pd.rsplit("/", 1)[-1]
    try:
        if os.readlink(pd + "/cwd") != REPO:
            continue
        nm = _name(pid)
        if nm is None:
            continue
        procs[pid] = {"name": nm, "t0": _stat_ticks(pid), "rss_kb": _rss_kb(pid)}
    except Exception:
        continue

time.sleep(WINDOW)

# Pass 2
out = []
for pid, p in procs.items():
    try:
        dt = _stat_ticks(pid) - p["t0"]
        cpu = (dt / HZ) / WINDOW * 100.0
        out.append({"pid": int(pid), "worker": p["name"],
                    "cpu_pct": round(cpu, 1),
                    "rss_mb": round(p["rss_kb"] / 1024, 1)})
    except Exception:
        continue

out.sort(key=lambda r: r["cpu_pct"], reverse=True)
total_cpu = round(sum(r["cpu_pct"] for r in out), 1)
total_rss = round(sum(r["rss_mb"] for r in out), 1)
print(json.dumps({"repo": REPO, "window_s": WINDOW, "n": len(out),
                  "total_cpu_pct": total_cpu, "total_rss_mb": total_rss,
                  "ranked": out}, indent=0))
