#!/bin/bash
# t3_phase_c_cleanup.sh — Hard cleanup for T3 Phase C (kernel-rs).
#
# Wired as `ExecStopPost=` in `titan-t3.service` (idempotent — runs on
# every service stop, both clean and unclean). Kernel-rs's own pdeathsig
# chain SHOULD already kill all descendants when the kernel exits, but
# this script is a defense-in-depth safety net that catches:
#
#   1. Python multiprocessing-spawn children that don't inherit pdeathsig
#      from the Python parent (kernel orphans them on titan_HCL panic).
#   2. Legacy `python -u scripts/titan_main.py --server` instances
#      launched by `t3_manage.sh` (NOT a kernel-rs descendant — pdeathsig
#      doesn't reach it).
#   3. Stale `data/titan_main.pid` left by abrupt Python exit, which
#      causes kernel-rs's freshly-spawned `titan_main` to abort with
#      "Another titan_main is already running".
#   4. Stale UNIX sockets + `/dev/shm/titan_T3/` slot files from the
#      previous kernel generation (Phase C kernel rebuilds slots from
#      L0 snapshot every boot — preserving stale half-written slots
#      across restarts is a hazard).
#
# Strictly scoped to T3 paths (`/projects/titan3/`) so it can NEVER
# touch T1 (`/projects/titan/`) or T2 (`/projects/titan/`) processes
# on the shared VPS — see `feedback_pkill_f_titan_main_kills_t2.md`
# (Day 1 lesson 2 of `SESSION_20260506_phase_c_close_runtime_gaps`).
#
# Closes BUG-T3-LEGACY-PYTHON-COLLISION-2026-05-06 (filed this session).
set -u  # No -e: each step idempotent; we want all cleanup to run even
        # if one step has nothing to do.

T3_ROOT="/home/antigravity/projects/titan3"
T3_DATA="${T3_ROOT}/data"
T3_BIN_DIR="${T3_ROOT}/bin"
T3_SHM="/dev/shm/titan_T3"
T3_BUS_SOCK="/tmp/titan_bus_T3.sock"
T3_KERNEL_RPC_SOCK="/tmp/titan_kernel_T3.sock"
T3_PID_FILE="${T3_DATA}/titan_main.pid"

log() { echo "[t3-cleanup] $*"; }

# ── 1. Kill T3-scoped Python multiprocessing orphans + legacy main ─
# Pattern matches:
#   - /projects/titan3/test_env/bin/python (multiprocessing-fork etc)
#   - python -u scripts/titan_main.py --server (cwd: titan3) — legacy
# pgrep -f matches the full command line + cwd is implicit via path.
# `|| true` so we don't fail when nothing matches (post-clean-shutdown).
log "killing T3 Python orphans (cwd=titan3)"
pkill -9 -f "/projects/titan3/.*python.*titan_main" 2>/dev/null || true
pkill -9 -f "/projects/titan3/test_env/bin/python" 2>/dev/null || true

# ── 2. Kill any T3 Rust daemon processes that escaped pdeathsig ──
log "killing T3 Rust daemon orphans (binary=projects/titan3/bin/*)"
pkill -9 -f "${T3_BIN_DIR}/titan-" 2>/dev/null || true
# Also catch /usr/local/bin symlinks that point at projects/titan3/bin/*
# but where the running process is the symlink target.
for sym in /usr/local/bin/titan-*; do
    [ -L "$sym" ] || continue
    target="$(readlink "$sym" 2>/dev/null || true)"
    case "$target" in
        "${T3_BIN_DIR}/"*)
            pkill -9 -f "$sym" 2>/dev/null || true
            ;;
    esac
done

# ── 3. Remove stale lock / pid / socket files ──
log "removing stale pid + sockets"
rm -f "${T3_PID_FILE}" 2>/dev/null || true
rm -f "${T3_BUS_SOCK}" 2>/dev/null || true
rm -f "${T3_KERNEL_RPC_SOCK}" 2>/dev/null || true

# ── 4. Wipe /dev/shm/titan_T3 slot files ──
# Kernel-rs recreates from L0 snapshot at next boot per SPEC §10.A B7;
# stale half-written slot bytes from a panic'd kernel must NOT persist.
# State durability is in `data/snapshots/`, NOT /dev/shm.
if [ -d "${T3_SHM}" ]; then
    log "wiping ${T3_SHM}/* (kernel will rebuild from L0 snapshot)"
    rm -rf "${T3_SHM}"/* 2>/dev/null || true
fi

# ── 5. Brief grace period — let kernel zombie-reap any race-killed PIDs
# before systemd considers the unit fully stopped + accepts a restart. ─
sleep 1

log "cleanup done"
exit 0
