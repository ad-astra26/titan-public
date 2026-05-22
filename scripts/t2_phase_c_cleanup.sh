#!/bin/bash
# t2_phase_c_cleanup.sh — Hard cleanup for T2 Phase C (kernel-rs).
#
# Wired as ExecStartPre + ExecStopPost in `titan-t2.service` (idempotent —
# runs on every service start/stop, both clean and unclean). Defense-in-
# depth safety net for Phase C orphan-process scenarios (mirror of
# `t3_phase_c_cleanup.sh`):
#
#   1. Python multiprocessing-spawn children that don't inherit pdeathsig
#      from the Python parent (kernel orphans them on titan_HCL panic).
#   2. Legacy `python -u scripts/titan_hcl.py --server` instances
#      launched by `t2_manage.sh` BEFORE the Phase C migration to systemd
#      (NOT a kernel-rs descendant — pdeathsig doesn't reach it).
#   3. Stale `data/titan_hcl.pid` left by abrupt Python exit, which
#      causes kernel-rs's freshly-spawned `titan_hcl` to abort with
#      "Another titan_hcl is already running".
#   4. Stale UNIX sockets + `/dev/shm/titan_T2/` slot files from the
#      previous kernel generation (Phase C kernel rebuilds slots from
#      L0 snapshot every boot — preserving stale half-written slots
#      across restarts is a hazard).
#
# CRITICAL: scoped to T2 paths (`/projects/titan/`) NOT T3 (`/projects/titan3/`).
# The substring `/projects/titan/` does NOT appear in T3 process command
# lines (which all contain `/projects/titan3/`), so pkill patterns are
# safe on this shared VPS. T1 is on a DIFFERENT host (10.135.0.3) so its
# processes never appear in /proc here.
#
# Per Phase C T1+T2 migration session 2026-05-14. Mirror of `t3_phase_c_cleanup.sh`.
set -u  # No -e: each step idempotent; we want all cleanup to run even
        # if one step has nothing to do.

T2_ROOT="/home/antigravity/projects/titan"
T2_DATA="${T2_ROOT}/data"
T2_BIN_DIR="${T2_ROOT}/bin"
T2_SHM="/dev/shm/titan_T2"
T2_BUS_SOCK="/tmp/titan_bus_T2.sock"
T2_KERNEL_RPC_SOCK="/tmp/titan_kernel_T2.sock"
T2_PID_FILE="${T2_DATA}/titan_hcl.pid"

log() { echo "[t2-cleanup] $*"; }

# ── 1. Kill T2-scoped Python multiprocessing orphans + legacy main ─
# Pattern matches /projects/titan/ but NOT /projects/titan3/ (substring
# `/projects/titan/` includes the trailing slash that distinguishes).
log "killing T2 Python orphans (cwd=/projects/titan/, NOT titan3)"
pkill -9 -f "/projects/titan/.*python.*titan_hcl" 2>/dev/null || true
pkill -9 -f "/projects/titan/test_env/bin/python" 2>/dev/null || true

# ── 2. Kill any T2 Rust daemon processes that escaped pdeathsig ──
log "killing T2 Rust daemon orphans (binary=projects/titan/bin/*)"
pkill -9 -f "${T2_BIN_DIR}/titan-" 2>/dev/null || true
# Also catch /usr/local/bin symlinks that point at projects/titan/bin/*
# (none expected on T2 today, but defense-in-depth for future symlink convention).
for sym in /usr/local/bin/titan-*; do
    [ -L "$sym" ] || continue
    target="$(readlink "$sym" 2>/dev/null || true)"
    case "$target" in
        "${T2_BIN_DIR}/"*)
            pkill -9 -f "$sym" 2>/dev/null || true
            ;;
    esac
done

# ── 3. Remove stale lock / pid / socket files ──
log "removing stale pid + sockets"
rm -f "${T2_PID_FILE}" 2>/dev/null || true
rm -f "${T2_BUS_SOCK}" 2>/dev/null || true
rm -f "${T2_KERNEL_RPC_SOCK}" 2>/dev/null || true

# ── 4. Wipe /dev/shm/titan_T2 slot files ──
# Kernel-rs recreates from L0 snapshot at next boot per SPEC §10.A B7;
# stale half-written slot bytes from a panic'd kernel must NOT persist.
# State durability is in `data/snapshots/`, NOT /dev/shm.
if [ -d "${T2_SHM}" ]; then
    log "wiping ${T2_SHM}/* (kernel will rebuild from L0 snapshot)"
    rm -rf "${T2_SHM}"/* 2>/dev/null || true
fi

# ── 5. Brief grace period — let kernel zombie-reap any race-killed PIDs
# before systemd considers the unit fully stopped + accepts a restart. ─
sleep 1

log "cleanup done"
exit 0
