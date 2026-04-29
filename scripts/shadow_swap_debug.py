#!/usr/bin/env python3
"""
Shadow Swap Debug Driver — runtime arch_map for shadow-swap diagnostics.

Runs scripts/shadow_swap.py while concurrently tailing:
  - /tmp/titan_brain.log         (old kernel — has [shadow_swap] phase= events)
  - /tmp/titan_shadow_<port>.log (new kernel — emerges mid-swap)

Time-merges both streams + filters for shadow-swap-relevant lines + classifies
by source (KERNEL/SHADOW) + phase. Output is one consolidated stream so you can
see the orchestrator's phase progression alongside the shadow process's
boot/bus/worker activity in real time.

Usage:
    TITAN_MAKER_KEY=<key> python scripts/shadow_swap_debug.py [--reason TEXT]

Filters applied:
    KERNEL side: lines matching [shadow_swap]|HIBERNATE|BUS_HANDOFF|adoption|
                 b2_1_|spawn_handoff_ack
    SHADOW side: lines matching ERROR|WARN|Module|spawn|MODULE_READY|bus_socket|
                 DivineBus|FileNotFoundError|Traceback|connect|adopt|HIBERNATE

After the swap completes, prints:
    - Final outcome + phase + elapsed
    - Per-phase elapsed breakdown
    - First 5 errors from each side (if any)
    - Bus stats (subscribers connected/dropped/timed out)
    - Worker adoption table (if B.2.1 active)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
SHADOW_SWAP = REPO_ROOT / "scripts" / "shadow_swap.py"
KERNEL_LOG = Path("/tmp/titan_brain.log")


# ── Filter patterns ──────────────────────────────────────────────

# Lines from old kernel log we care about during a swap
KERNEL_PATTERNS = re.compile(
    r"\[shadow_swap\]|"
    r"HIBERNATE|"
    r"BUS_HANDOFF|"
    r"swap_pending|"
    r"adoption_|"
    r"b2_1_|"
    r"spawn_handoff|"
    r"adopted|"
    r"bus_socket broker|"
    r"disconnect_subscribers|"
    r"shadow_data|"
    r"shadow_health|"
    r"_revive_guardian",
    re.IGNORECASE,
)

# Lines from shadow process log
SHADOW_PATTERNS = re.compile(
    r"ERROR|WARN|Traceback|"
    r"Module '|"
    r"MODULE_READY|MODULE_HEARTBEAT|"
    r"bus_socket|DivineBus|"
    r"FileNotFoundError|ImportError|"
    r"connect|reconnect|"
    r"adopt|"
    r"\[Guardian\]|"
    r"is READY|"
    r"started \(pid=",
    re.IGNORECASE,
)


# ── Output formatting ────────────────────────────────────────────

class Stream:
    """Color-coded source labels for terminal output."""
    KERNEL = "\033[36mKERNEL\033[0m"   # cyan
    SHADOW = "\033[35mSHADOW\033[0m"   # magenta
    SWAP   = "\033[33mSWAP  \033[0m"   # yellow
    DRIVER = "\033[32mDRIVER\033[0m"   # green
    ERROR  = "\033[31m"
    WARN   = "\033[33m"
    RESET  = "\033[0m"


def now() -> str:
    return time.strftime("%H:%M:%S")


def emit(source: str, line: str) -> None:
    """Print one filtered line with source prefix + timestamp."""
    line = line.rstrip()
    color = ""
    if "ERROR" in line or "Traceback" in line:
        color = Stream.ERROR
    elif "WARN" in line.upper():
        color = Stream.WARN
    print(f"{now()}  {source}  {color}{line}{Stream.RESET}", flush=True)


# ── Log tailer threads ───────────────────────────────────────────

class _Tailer(threading.Thread):
    """Tail a file from current EOF, filter via pattern, push to emit()."""

    def __init__(self, path: Path, pattern: re.Pattern, source: str, stop: threading.Event):
        super().__init__(daemon=True)
        self.path = path
        self.pattern = pattern
        self.source = source
        self.stop = stop
        self.captured_errors: list[str] = []

    def run(self) -> None:
        # Wait for the file to appear (shadow log doesn't exist until shadow boots)
        while not self.stop.is_set() and not self.path.exists():
            time.sleep(0.2)
        if self.stop.is_set():
            return
        try:
            f = self.path.open("r")
        except OSError as e:
            emit(Stream.DRIVER, f"cannot open {self.path}: {e}")
            return
        # Seek to end so we only see new lines.
        f.seek(0, 2)
        while not self.stop.is_set():
            line = f.readline()
            if not line:
                time.sleep(0.1)
                continue
            if not self.pattern.search(line):
                continue
            emit(self.source, line)
            if "ERROR" in line or "Traceback" in line:
                self.captured_errors.append(line.strip())


# ── Shadow swap driver ───────────────────────────────────────────

def run_swap(reason: str, host: str, key: str) -> int:
    """Drive shadow_swap.py as subprocess, stream its stdout via SWAP source."""
    cmd = [sys.executable, str(SHADOW_SWAP), "--reason", reason, "--host", host]
    env = os.environ.copy()
    env["TITAN_MAKER_KEY"] = key
    emit(Stream.DRIVER, f"$ {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env=env, text=True, bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        emit(Stream.SWAP, line)
    proc.wait()
    return proc.returncode


def find_shadow_log() -> Optional[Path]:
    """Find /tmp/titan_shadow_*.log freshly written in the last 5 min."""
    now_ts = time.time()
    candidates = sorted(Path("/tmp").glob("titan_shadow_*.log"),
                        key=lambda p: p.stat().st_mtime if p.exists() else 0,
                        reverse=True)
    for p in candidates:
        try:
            age = now_ts - p.stat().st_mtime
            if age < 600:
                return p
        except OSError:
            continue
    return None


# ── Post-swap analysis ───────────────────────────────────────────

def parse_phase_timeline(captured: list[str]) -> dict[str, float]:
    """Walk SWAP-source lines for `→ phase=X outcome=Y elapsed=Z`."""
    phases: dict[str, float] = {}
    rx = re.compile(r"→ phase=(\w+).*?elapsed=([\d.]+)s")
    for line in captured:
        m = rx.search(line)
        if m:
            phases[m.group(1)] = float(m.group(2))
    return phases


def emit_summary(rc: int, kernel_t: _Tailer, shadow_t: _Tailer, swap_lines: list[str]) -> None:
    print()
    print("=" * 78)
    emit(Stream.DRIVER, f"shadow_swap.py exited rc={rc}")
    phases = parse_phase_timeline(swap_lines)
    if phases:
        print()
        emit(Stream.DRIVER, "Phase timeline:")
        prev = 0.0
        for phase, elapsed in phases.items():
            delta = elapsed - prev
            print(f"            {phase:25s}  cumulative={elapsed:6.1f}s  +{delta:5.1f}s")
            prev = elapsed

    if kernel_t.captured_errors:
        print()
        emit(Stream.DRIVER, f"KERNEL errors ({len(kernel_t.captured_errors)}):")
        for line in kernel_t.captured_errors[:5]:
            print(f"   {line}")

    if shadow_t.captured_errors:
        print()
        emit(Stream.DRIVER, f"SHADOW errors ({len(shadow_t.captured_errors)}):")
        for line in shadow_t.captured_errors[:8]:
            print(f"   {line}")
    print("=" * 78)


# ── Main ─────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description="Shadow swap debug driver")
    p.add_argument("--reason", default="debug-driven shadow swap")
    p.add_argument("--host", default="127.0.0.1")
    args = p.parse_args()

    key = os.environ.get("TITAN_MAKER_KEY") or os.environ.get("X_TITAN_INTERNAL_KEY")
    if not key:
        print("ERROR: TITAN_MAKER_KEY env var required", file=sys.stderr)
        return 1

    if not KERNEL_LOG.exists():
        print(f"ERROR: kernel log not found at {KERNEL_LOG}", file=sys.stderr)
        return 1

    stop = threading.Event()

    # Start kernel tailer first (always exists)
    kernel_t = _Tailer(KERNEL_LOG, KERNEL_PATTERNS, Stream.KERNEL, stop)
    kernel_t.start()

    # Find existing shadow log if any (post-rollback) OR start it lazily
    # by spawning a background tailer that polls for the file.
    swap_lines: list[str] = []

    # Wrap the swap driver so we can capture its stdout for phase parsing
    def _drive_with_capture():
        cmd = [sys.executable, str(SHADOW_SWAP), "--reason", args.reason, "--host", args.host]
        env = os.environ.copy()
        env["TITAN_MAKER_KEY"] = key
        emit(Stream.DRIVER, f"$ TITAN_MAKER_KEY=... python {SHADOW_SWAP.name} --reason '{args.reason}'")
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=env, text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            swap_lines.append(line)
            emit(Stream.SWAP, line)
        proc.wait()
        return proc.returncode

    # Once the kernel side logs `shadow_spawned`, the shadow log will appear.
    # Poll for it in a background thread + start tailing when found.
    shadow_t_holder: dict[str, _Tailer] = {}

    def _watch_for_shadow():
        while not stop.is_set():
            sl = find_shadow_log()
            if sl is not None:
                t = _Tailer(sl, SHADOW_PATTERNS, Stream.SHADOW, stop)
                t.start()
                shadow_t_holder["t"] = t
                emit(Stream.DRIVER, f"shadow log appeared: {sl}")
                return
            time.sleep(0.5)

    watcher = threading.Thread(target=_watch_for_shadow, daemon=True)
    watcher.start()

    rc = _drive_with_capture()

    # Give tailers a beat to flush final lines
    time.sleep(2.0)
    stop.set()
    time.sleep(0.5)

    shadow_t = shadow_t_holder.get("t")
    if shadow_t is None:
        # Make a sentinel for emit_summary
        class _Empty:
            captured_errors: list[str] = []
        shadow_t = _Empty()  # type: ignore

    emit_summary(rc, kernel_t, shadow_t, swap_lines)
    return rc


if __name__ == "__main__":
    sys.exit(main())
