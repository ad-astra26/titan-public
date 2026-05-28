#!/bin/bash
# titan_phase_c_cleanup.sh — Generic hard cleanup for a Phase C (kernel-rs) Titan.
#
# Parameterized sibling of the fleet's per-Titan `t{1,2,3}_phase_c_cleanup.sh`.
# Where those hard-code T1/T2/T3 paths, this one reads two environment
# variables so a tester's single-Titan install (created by `setup_titan`)
# can reuse the exact same defense-in-depth safety net without a bespoke
# per-host script:
#
#   TITAN_ID    — the Titan id (e.g. T1). Default: T1 (matches
#                 titan_hcl.core.state_registry.resolve_titan_id's fallback).
#   TITAN_ROOT  — the install root (the cloned repo). Default: the repo this
#                 script lives in (two dirs up from scripts/).
#
# Wired as ExecStartPre + ExecStopPost in the generated `titan.service`
# (idempotent — runs on every service start/stop, clean or unclean). Mirrors
# t1_phase_c_cleanup.sh's four concerns:
#   1. Python multiprocessing-spawn orphans (no pdeathsig from the panic'd parent).
#   2. Stale data/titan_hcl.pid that aborts the freshly-spawned titan_hcl.
#   3. Stale UNIX sockets from the previous kernel generation.
#   4. Stale /dev/shm/titan_<ID>/ slot files (kernel rebuilds from L0 snapshot;
#      half-written slot bytes from a panic must NOT persist — durable state is
#      in data/snapshots/, never /dev/shm).
#
# A single-Titan tester box has no sibling Titan, so the cwd-scoped pkill
# patterns here are unambiguous. (The fleet's hosts co-reside T2+T3, which is
# why those scripts carry extra host-disambiguation comments.)
set -u

TITAN_ID="${TITAN_ID:-T1}"
# Default root = two levels up from this script (…/scripts/ -> repo root).
_SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TITAN_ROOT="${TITAN_ROOT:-$(cd "${_SELF_DIR}/.." && pwd)}"

TITAN_DATA="${TITAN_ROOT}/data"
TITAN_BIN_DIR="${TITAN_ROOT}/bin"
TITAN_SHM="/dev/shm/titan_${TITAN_ID}"
TITAN_BUS_SOCK="/tmp/titan_bus_${TITAN_ID}.sock"
TITAN_KERNEL_RPC_SOCK="/tmp/titan_kernel_${TITAN_ID}.sock"
TITAN_PID_FILE="${TITAN_DATA}/titan_hcl.pid"

log() { echo "[titan-cleanup ${TITAN_ID}] $*"; }

# ── 1. Kill this install's Python multiprocessing orphans + legacy main ──
log "killing Python orphans (cwd=${TITAN_ROOT})"
pkill -9 -f "${TITAN_ROOT}/.*python.*titan_hcl" 2>/dev/null || true
pkill -9 -f "${TITAN_ROOT}/test_env/bin/python" 2>/dev/null || true

# ── 2. Kill any Rust daemon processes that escaped pdeathsig ──
log "killing Rust daemon orphans (binary=${TITAN_BIN_DIR}/*)"
pkill -9 -f "${TITAN_BIN_DIR}/titan-" 2>/dev/null || true
for sym in /usr/local/bin/titan-*; do
    [ -L "$sym" ] || continue
    target="$(readlink "$sym" 2>/dev/null || true)"
    case "$target" in
        "${TITAN_BIN_DIR}/"*)
            pkill -9 -f "$sym" 2>/dev/null || true
            ;;
    esac
done

# ── 3. Remove stale lock / pid / socket files ──
log "removing stale pid + sockets"
rm -f "${TITAN_PID_FILE}" 2>/dev/null || true
rm -f "${TITAN_BUS_SOCK}" 2>/dev/null || true
rm -f "${TITAN_KERNEL_RPC_SOCK}" 2>/dev/null || true

# ── 4. Wipe /dev/shm/titan_<ID> slot files ──
# Kernel-rs recreates from L0 snapshot at next boot per SPEC §10.A B7; stale
# half-written slot bytes from a panic'd kernel must NOT persist. Durable
# state lives in data/snapshots/, NOT /dev/shm.
if [ -d "${TITAN_SHM}" ]; then
    log "wiping ${TITAN_SHM}/* (kernel will rebuild from L0 snapshot)"
    rm -rf "${TITAN_SHM}"/* 2>/dev/null || true
fi

# ── 5. Brief grace period ──
sleep 1

log "cleanup done"
exit 0
