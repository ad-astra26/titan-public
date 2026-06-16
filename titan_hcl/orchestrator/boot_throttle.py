"""RFP_supervision_lifecycle §7.B — predictive boot back-pressure (throttle, don't kill).

A heavy module booting on a memory-pressured box spikes its RSS during the load
phase (FAISS/Kuzu/DuckDB into heap). If that spike trips the guardian's per-module
rss limit while the box is starved, the old behavior kill→respawned it → repeated
flap → DISABLE → forced full restart (synthesis OOM incident 2026-06-15). A
respawn costs the whole system far more than letting the module page to swap and
finish booting.

This module gives the guardian a SMART, PREDICTIVE decision: before/as a heavy
module boots, read the WHOLE box state (MemAvailable / swap / load) against the
module's rss_limit + current RSS, and predict whether it can boot without
starving the box. If headroom is tight, THROTTLE it — set a per-module cgroup-v2
`memory.high` so the kernel gracefully reclaims/pages that module's memory to swap
instead of OOM-killing it ("swap-to-disk"). When cgroup delegation isn't available
(`Delegate=yes` not set on the unit), fall back to predictive-DEFER: the guardian
simply does not kill it during boot (Phase A boot-grace) and relies on OS swap.

Everything here is FAIL-OPEN: a throttle/predict error never blocks or breaks a
boot — worst case we behave exactly as Phase A (boot-grace, no throttle).
"""
from __future__ import annotations

import dataclasses
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Box-headroom floor: if booting this module to its rss_limit would leave the box
# with less than this much available memory, throttle it. ~256 MB keeps the box
# off the OOM cliff while not throttling on a healthy box.
DEFAULT_MEM_FLOOR_MB: float = 256.0
# memory.high is set this multiple above the module's rss_limit — a soft ceiling
# that triggers reclaim/paging before the (guardian) rss_limit hard-fault, giving
# the kernel room to page gradually rather than at a cliff.
MEMORY_HIGH_MULTIPLE: float = 1.10

CGROUP_ROOT = Path("/sys/fs/cgroup")


@dataclasses.dataclass(frozen=True)
class BoxPressure:
    """A point-in-time read of box-wide memory/CPU pressure."""
    mem_available_mb: float
    swap_used_mb: float
    swap_total_mb: float
    load1: float
    ncpu: int


def read_box_pressure() -> BoxPressure:
    """Read box pressure from /proc. Fail-open: missing fields → 0 (treated as
    'no pressure' by the predictor, i.e. don't throttle on a bad read)."""
    mem_avail = 0.0
    swap_total = 0.0
    swap_free = 0.0
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    mem_avail = float(line.split()[1]) / 1024.0
                elif line.startswith("SwapTotal:"):
                    swap_total = float(line.split()[1]) / 1024.0
                elif line.startswith("SwapFree:"):
                    swap_free = float(line.split()[1]) / 1024.0
    except Exception:  # noqa: BLE001
        pass
    load1 = 0.0
    try:
        with open("/proc/loadavg", "r") as f:
            load1 = float(f.read().split()[0])
    except Exception:  # noqa: BLE001
        pass
    return BoxPressure(
        mem_available_mb=mem_avail,
        swap_used_mb=max(0.0, swap_total - swap_free),
        swap_total_mb=swap_total,
        load1=load1,
        ncpu=os.cpu_count() or 1,
    )


def predict_needs_throttle(
    box: BoxPressure,
    rss_limit_mb: float,
    current_rss_mb: float,
    *,
    mem_floor_mb: float = DEFAULT_MEM_FLOOR_MB,
) -> bool:
    """Predict whether booting this module to its rss_limit would starve the box.

    Projected free = MemAvailable − (remaining growth to rss_limit). Throttle if
    that projection drops below the floor. A `mem_available_mb` of 0 (bad/zero
    read) yields projected_free ≤ 0 < floor → would always throttle, so guard it:
    a non-positive read is treated as 'unknown → don't throttle' (fail-open)."""
    if box.mem_available_mb <= 0.0:
        return False  # bad read — don't throttle on no information
    remaining_growth = max(0.0, rss_limit_mb - max(0.0, current_rss_mb))
    projected_free = box.mem_available_mb - remaining_growth
    return projected_free < mem_floor_mb


def _own_cgroup_dir() -> Optional[Path]:
    """Resolve THIS process's cgroup-v2 directory (the service cgroup the guardian
    runs in). cgroup v2 line format: `0::/system.slice/titan-tN.service`."""
    try:
        with open("/proc/self/cgroup", "r") as f:
            for line in f:
                # v2 unified hierarchy is the `0::` entry.
                if line.startswith("0::"):
                    rel = line.strip().split("::", 1)[1]
                    return CGROUP_ROOT / rel.lstrip("/")
    except Exception:  # noqa: BLE001
        pass
    return None


def cgroup_throttle_available() -> bool:
    """True if the guardian can create a memory-controlled sub-cgroup — i.e.
    cgroup v2 is mounted, the `memory` controller is delegated into our subtree
    (`cgroup.subtree_control` lists `memory`), and our cgroup dir is writable
    (`Delegate=yes` on the unit). Best-effort; any uncertainty → False (defer)."""
    own = _own_cgroup_dir()
    if own is None or not own.is_dir():
        return False
    if not os.access(own, os.W_OK):
        return False
    try:
        subtree = (own / "cgroup.subtree_control").read_text()
    except Exception:  # noqa: BLE001
        return False
    return "memory" in subtree.split()


def apply_memory_high(pid: int, module_name: str, high_mb: float) -> bool:
    """Place `pid` in a per-module sub-cgroup with `memory.high = high_mb`.

    Returns True on success. FAIL-OPEN: any error → False (caller falls back to
    defer). Idempotent: re-applying for a live sub-cgroup just rewrites
    memory.high + re-adds the pid."""
    own = _own_cgroup_dir()
    if own is None:
        return False
    child = own / f"throttle_{module_name}"
    try:
        child.mkdir(exist_ok=True)
        # memory.high BEFORE moving the pid in, so the cap applies immediately.
        (child / "memory.high").write_text(str(int(high_mb * 1024 * 1024)))
        # Move the process into the sub-cgroup (cgroup.procs takes one pid/write).
        (child / "cgroup.procs").write_text(str(int(pid)))
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "[BootThrottle] cgroup memory.high for '%s' (pid=%d) failed: %s "
            "— falling back to predictive-defer", module_name, pid, e)
        return False


@dataclasses.dataclass(frozen=True)
class ThrottleDecision:
    throttled: bool          # was any back-pressure applied / warranted?
    mechanism: str           # "cgroup" | "defer" | "none"
    box: BoxPressure
    high_mb: float           # memory.high set (cgroup) or 0


def evaluate_and_throttle(
    pid: int,
    module_name: str,
    rss_limit_mb: float,
    current_rss_mb: float,
    *,
    mem_floor_mb: float = DEFAULT_MEM_FLOOR_MB,
    cgroup_enabled: bool = True,
) -> ThrottleDecision:
    """The full §7.B decision for one (heavy) module at/after spawn.

    Predict box headroom; if tight, throttle via cgroup memory.high (when
    delegation is available) else mark for predictive-defer (Phase A boot-grace
    keeps it alive). FAIL-OPEN throughout."""
    box = read_box_pressure()
    if not predict_needs_throttle(
            box, rss_limit_mb, current_rss_mb, mem_floor_mb=mem_floor_mb):
        return ThrottleDecision(False, "none", box, 0.0)

    high_mb = rss_limit_mb * MEMORY_HIGH_MULTIPLE
    if cgroup_enabled and cgroup_throttle_available() and pid > 0:
        if apply_memory_high(pid, module_name, high_mb):
            return ThrottleDecision(True, "cgroup", box, high_mb)
    # Fallback: predictive-defer (no kill; Phase A boot-grace + OS swap).
    return ThrottleDecision(True, "defer", box, 0.0)
