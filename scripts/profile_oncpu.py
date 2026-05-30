#!/usr/bin/env python3
"""
profile_oncpu.py — professional on-CPU profiler for Titan Phase C workers.

WHY THIS EXISTS
---------------
`py-spy record` (and `perf` without care) in their *default* modes are
WALL-CLOCK samplers: they sample every thread regardless of whether it is
running. A thread parked in ``time.sleep()`` or a blocking ``socket.recv()``
has *released the GIL* and is burning ZERO cpu, yet it shows up as a dominant
"hotspot" (its parked line). On 2026-05-30 this produced three false leads
(F1 ``recv_exact``, F3 publisher ``time.sleep``, and an inflated F5 builder)
before the methodology was corrected. This tool makes that mistake impossible.

WHAT IT MEASURES (on-CPU only — off-CPU is a separate concern, see --offcpu)
  1. ANCHOR  — /proc/<pid>/stat (utime+stime) delta over the window = absolute
               cpu-seconds → "% of ONE core". This decides whether a worker is
               even worth profiling, and grounds every relative number below.
  2. PYTHON  — `py-spy record --gil`: samples ONLY threads holding the GIL
               (= actually executing Python bytecode = on-CPU). sleep/blocked
               threads are excluded by construction. Per-frame self-time is
               reported as **% OF A CORE** (relative share × anchor), never raw
               wall-clock %.
  3. NATIVE  — optional `perf` cross-check (--perf): on-CPU by construction,
               catches C-extension / syscall time the GIL sampler under-counts.

PID RESOLUTION
  Workers self-title via prctl comm ``titan:<name>`` (INV-PROC-7). We resolve by
  EXACT comm match — never `pgrep -f "titan_hcl:<name>"`, which also matches a
  shell whose command line happens to contain that string (a real 2026-05-30 bug
  that mis-measured cognitive_worker as 0% by sampling a bash loop).

USAGE
  source test_env/bin/activate
  sudo -E python scripts/profile_oncpu.py mind timechain cognitive_worker
  sudo -E python scripts/profile_oncpu.py --all --duration 20
  sudo -E python scripts/profile_oncpu.py --pid 12345 --perf
  sudo -E python scripts/profile_oncpu.py mind --repo /home/antigravity/projects/titan3   # T3

  (sudo is required: ptrace_scope=1 means only a parent may ptrace; the worker's
   parent is the kernel, not your shell. `-E`/`env PATH=$PATH` preserves the venv
   so the bundled py-spy is found.)
"""
from __future__ import annotations

import argparse
import collections
import os
import subprocess
import sys
import time
from pathlib import Path

_CLK = os.sysconf("SC_CLK_TCK")
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _find_pyspy() -> str:
    """Locate py-spy. It lives in the venv bin/ alongside the interpreter, but
    test_env/bin/python is a symlink to system python, so do NOT resolve() it
    (that would point at /usr/bin). Try the unresolved interpreter dir, then
    $VIRTUAL_ENV, then the known repo venv, then PATH.
    """
    cands = [Path(sys.executable).parent / "py-spy"]
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        cands.append(Path(venv) / "bin" / "py-spy")
    cands.append(Path("/home/antigravity/projects/titan/test_env/bin/py-spy"))
    for c in cands:
        if c.exists():
            return str(c)
    import shutil
    return shutil.which("py-spy") or "py-spy"


_PYSPY = _find_pyspy()


def _read_cpu_ticks(pid: int) -> int:
    """utime+stime (clock ticks) from /proc/<pid>/stat. Fields 14,15 (1-based)."""
    with open(f"/proc/{pid}/stat") as f:
        # comm may contain spaces/parens → split after the trailing ')'
        data = f.read()
    rparen = data.rfind(")")
    fields = data[rparen + 2:].split()
    # after "pid (comm) ", state is field 3 → utime=field14 → index 11,12 here
    utime = int(fields[11])
    stime = int(fields[12])
    return utime + stime


def _comm(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/comm") as f:
            return f.read().strip()
    except OSError:
        return ""


def _cwd(pid: int) -> str:
    try:
        return os.readlink(f"/proc/{pid}/cwd")
    except OSError:
        return ""


def resolve_pid(worker: str, repo: str | None) -> int | None:
    """Resolve a worker's real pid by EXACT prctl comm match (titan:<name>),
    optionally constrained to a repo cwd (to disambiguate T1 vs T3 on one box).

    comm is truncated to 15 chars by the kernel, so match on the truncated form.
    """
    want = f"titan:{worker}"[:15]
    matches = []
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        pid = int(p)
        if _comm(pid) != want:
            continue
        if repo and _cwd(pid) != repo:
            continue
        matches.append(pid)
    if not matches:
        return None
    if len(matches) > 1:
        print(f"  ⚠ {worker}: {len(matches)} pids match comm={want} "
              f"(cwds: {[_cwd(p) for p in matches]}); using {matches[0]} "
              f"— pass --repo to disambiguate", file=sys.stderr)
    return matches[0]


def list_titan_workers(repo: str | None) -> list[tuple[str, int]]:
    """All running titan:<name> workers (optionally constrained to repo)."""
    out = []
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        pid = int(p)
        c = _comm(pid)
        if c.startswith("titan:") or c in ("titan_hcl", "titan_hcl_api"):
            if repo and _cwd(pid) != repo:
                continue
            out.append((c, pid))
    return sorted(out)


def pyspy_gil_folded(pid: int, duration: int, rate: int) -> str | None:
    """Run py-spy record --gil; return path to the raw folded file (or None)."""
    out = f"/tmp/oncpu_{pid}.folded"
    try:
        os.remove(out)
    except OSError:
        pass
    cmd = [str(_PYSPY), "record", "--gil", "-f", "raw", "-o", out,
           "--pid", str(pid), "--duration", str(duration), "--rate", str(rate)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if not os.path.exists(out):
        print(f"  py-spy failed: {r.stderr.strip()[:200]}", file=sys.stderr)
        return None
    return out


def parse_folded_selftime(path: str) -> tuple[collections.Counter, int]:
    """Aggregate self-time by leaf frame from a py-spy raw (folded) file."""
    leaf = collections.Counter()
    total = 0
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            try:
                stack, cnt = line.rsplit(" ", 1)
                cnt = int(cnt)
            except ValueError:
                continue
            total += cnt
            frames = stack.split(";")
            if frames:
                leaf[frames[-1]] += cnt
    return leaf, total


def profile_one(worker_label: str, pid: int, duration: int, rate: int,
                topn: int, run_perf: bool) -> dict:
    comm = _comm(pid)
    t0 = time.time()
    c0 = _read_cpu_ticks(pid)
    folded = pyspy_gil_folded(pid, duration, rate)
    c1 = _read_cpu_ticks(pid)
    t1 = time.time()

    cpu_cores = (c1 - c0) / _CLK / (t1 - t0)  # fraction of ONE core
    print(f"\n{'='*78}\n{worker_label}  (pid={pid}, comm={comm})")
    print(f"  ANCHOR — absolute on-CPU: {cpu_cores*100:.2f}% of one core "
          f"({(c1-c0)/_CLK:.2f} cpu-s / {t1-t0:.1f}s wall)")

    if cpu_cores * 100 < 0.5:
        print("  → essentially idle (<0.5% of a core); frame breakdown omitted "
              "(not worth optimizing).")
    elif folded:
        leaf, total = parse_folded_selftime(folded)
        if not total:
            print("  → py-spy --gil captured 0 GIL-held samples "
                  "(worker spends its CPU in released-GIL C or is sub-sample).")
        else:
            print(f"  ON-CPU Python (--gil, {total} samples) — top {topn} "
                  f"self-time frames as % OF A CORE:")
            for frame, cnt in leaf.most_common(topn):
                share = cnt / total
                print(f"    {share*cpu_cores*100:5.2f}% core  "
                      f"({share*100:4.1f}% on-cpu)  {frame[:64]}")

    if run_perf:
        _perf_native_split(pid)
    return {"worker": worker_label, "pid": pid, "cpu_cores_pct": cpu_cores * 100}


def _perf_native_split(pid: int) -> None:
    """perf on-CPU cross-check — native vs interpreter top symbols."""
    perf_out = f"/tmp/oncpu_{pid}.perf"
    subprocess.run(["perf", "record", "-F", "199", "-g", "-p", str(pid),
                    "-o", perf_out, "--", "sleep", "5"],
                   capture_output=True, text=True)
    r = subprocess.run(["perf", "report", "-i", perf_out, "--stdio",
                        "--sort", "symbol", "-g", "none"],
                       capture_output=True, text=True)
    print("  perf (native/kernel on-CPU cross-check) — top symbols:")
    n = 0
    for line in r.stdout.splitlines():
        line = line.strip()
        if line.startswith("#") or not line:
            continue
        print(f"    {line[:74]}")
        n += 1
        if n >= 8:
            break


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("workers", nargs="*", help="worker name(s), e.g. mind timechain")
    ap.add_argument("--all", action="store_true", help="profile every running titan:* worker")
    ap.add_argument("--pid", type=int, default=None, help="profile a specific pid")
    ap.add_argument("--duration", type=int, default=20, help="sample seconds (default 20)")
    ap.add_argument("--rate", type=int, default=100, help="samples/sec (default 100)")
    ap.add_argument("--top", type=int, default=12, help="top N frames (default 12)")
    ap.add_argument("--repo", default=None,
                    help="constrain to a repo cwd (T1=/home/antigravity/projects/titan, "
                         "T3=/home/antigravity/projects/titan3)")
    ap.add_argument("--perf", action="store_true", help="add perf native/kernel cross-check")
    args = ap.parse_args()

    if os.geteuid() != 0:
        print("⚠ Not root — py-spy/perf need ptrace (ptrace_scope=1). "
              "Re-run with: sudo -E python scripts/profile_oncpu.py ...",
              file=sys.stderr)

    targets: list[tuple[str, int]] = []
    if args.pid:
        targets.append((f"pid:{args.pid}", args.pid))
    elif args.all:
        targets = list_titan_workers(args.repo)
        print(f"Profiling {len(targets)} titan workers"
              f"{' in '+args.repo if args.repo else ''} "
              f"({args.duration}s each)...")
    else:
        for w in args.workers:
            pid = resolve_pid(w, args.repo)
            if pid is None:
                print(f"  ✗ {w}: no running process with comm=titan:{w[:9]}",
                      file=sys.stderr)
                continue
            targets.append((w, pid))

    if not targets:
        print("No targets. Usage: profile_oncpu.py <worker>... | --all | --pid N")
        return 1

    results = [profile_one(label, pid, args.duration, args.rate, args.top, args.perf)
               for label, pid in targets]

    if len(results) > 1:
        print(f"\n{'='*78}\nRANKING by absolute on-CPU (% of one core):")
        for r in sorted(results, key=lambda x: -x["cpu_cores_pct"]):
            print(f"  {r['cpu_cores_pct']:6.2f}%  {r['worker']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
