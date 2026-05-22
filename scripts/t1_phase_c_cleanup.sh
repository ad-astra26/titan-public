#!/bin/bash
# t1_phase_c_cleanup.sh — Hard cleanup for T1 Phase C (kernel-rs).
#
# Wired as ExecStartPre + ExecStopPost in `titan-t1.service` (idempotent —
# runs on every service start/stop, both clean and unclean). Defense-in-
# depth safety net for Phase C orphan-process scenarios (mirror of
# `t2_phase_c_cleanup.sh` + `t3_phase_c_cleanup.sh`):
#
#   1. Python multiprocessing-spawn children that don't inherit pdeathsig
#      from the Python parent (kernel orphans them on titan_HCL panic).
#   2. Legacy `python -u scripts/titan_hcl.py --server` instances
#      launched via the pre-migration `safe_restart.sh` + nohup path
#      (NOT a kernel-rs descendant — pdeathsig doesn't reach it).
#   3. Stale `data/titan_hcl.pid` left by abrupt Python exit, which
#      causes kernel-rs's freshly-spawned `titan_hcl` to abort with
#      "Another titan_hcl is already running".
#   4. Stale UNIX sockets + `/dev/shm/titan_T1/` slot files from the
#      previous kernel generation (Phase C kernel rebuilds slots from
#      L0 snapshot every boot — preserving stale half-written slots
#      across restarts is a hazard).
#
# CRITICAL: T1 lives on its own host (typically 10.135.0.3), separate
# from T2/T3 (10.135.0.6). Pattern-matching by `/projects/titan/` would
# be ambiguous if T1 and T2 shared a host — but they don't, so the
# patterns here are unique on T1's host. (T3 is `/projects/titan3/`,
# distinct prefix anyway.)
#
# Per Phase C T1+T2 migration session 2026-05-14. Mirror of `t2_phase_c_cleanup.sh`.
set -u

T1_ROOT="/home/antigravity/projects/titan"
T1_DATA="${T1_ROOT}/data"
T1_BIN_DIR="${T1_ROOT}/bin"
T1_SHM="/dev/shm/titan_T1"
T1_BUS_SOCK="/tmp/titan_bus_T1.sock"
T1_KERNEL_RPC_SOCK="/tmp/titan_kernel_T1.sock"
T1_PID_FILE="${T1_DATA}/titan_hcl.pid"

log() { echo "[t1-cleanup] $*"; }

# ── 1. Kill T1-scoped Python multiprocessing orphans + legacy main ─
log "killing T1 Python orphans (cwd=/projects/titan/)"
pkill -9 -f "/projects/titan/.*python.*titan_hcl" 2>/dev/null || true
pkill -9 -f "/projects/titan/test_env/bin/python" 2>/dev/null || true

# ── 2. Kill any T1 Rust daemon processes that escaped pdeathsig ──
log "killing T1 Rust daemon orphans (binary=projects/titan/bin/*)"
pkill -9 -f "${T1_BIN_DIR}/titan-" 2>/dev/null || true
for sym in /usr/local/bin/titan-*; do
    [ -L "$sym" ] || continue
    target="$(readlink "$sym" 2>/dev/null || true)"
    case "$target" in
        "${T1_BIN_DIR}/"*)
            pkill -9 -f "$sym" 2>/dev/null || true
            ;;
    esac
done

# ── 3. Remove stale lock / pid / socket files ──
log "removing stale pid + sockets"
rm -f "${T1_PID_FILE}" 2>/dev/null || true
rm -f "${T1_BUS_SOCK}" 2>/dev/null || true
rm -f "${T1_KERNEL_RPC_SOCK}" 2>/dev/null || true

# ── 4. Wipe /dev/shm/titan_T1 slot files ──
# Kernel-rs recreates from L0 snapshot at next boot per SPEC §10.A B7;
# stale half-written slot bytes from a panic'd kernel must NOT persist.
# State durability is in `data/snapshots/`, NOT /dev/shm.
if [ -d "${T1_SHM}" ]; then
    log "wiping ${T1_SHM}/* (kernel will rebuild from L0 snapshot)"
    rm -rf "${T1_SHM}"/* 2>/dev/null || true
fi

# ── 5. Brief grace period ─
sleep 1

log "cleanup done"
exit 0
