#!/bin/bash
# titan_watchdog.sh — Unified watchdog for T1/T2/T3 (microkernel v2 + Phase B.1 aware)
#
# Replaces the per-Titan t1_watchdog.sh / scattered logic with a single
# script that's shadow-swap-aware and uses the canonical PID file path.
#
# Usage:
#   bash scripts/titan_watchdog.sh T1
#   bash scripts/titan_watchdog.sh T2
#   bash scripts/titan_watchdog.sh T3
#
# Cron (every 5 min):
#   T1: */5 * * * * bash /home/antigravity/projects/titan/scripts/titan_watchdog.sh T1 >> /tmp/titan1_watchdog.log 2>&1
#   T2: */5 * * * * bash /home/antigravity/projects/titan/scripts/titan_watchdog.sh T2 >> /tmp/titan2_watchdog.log 2>&1
#   T3: */5 * * * * bash /home/antigravity/projects/titan3/scripts/titan_watchdog.sh T3 >> /tmp/titan3_watchdog.log 2>&1
#
# Design (post 2026-04-27 cascade incident — see SESSION_20260427_*.md):
#
# 1. CANONICAL PID FILE: reads $PROJECT_DIR/data/titan_main.pid (the file
#    titan_main itself writes + checks at boot). Pre-fix used /tmp/titan1.pid
#    which got out of sync, causing the 4-hour restart-loop deadlock when
#    titan_main's lock check saw a stale PID it didn't recognize.
#
# 2. STALE PID DETECTION: explicit step. If PID file exists but kill -0
#    fails (process gone), remove the file BEFORE attempting any restart.
#    Prevents the lock-check ABORT loop.
#
# 3. SHADOW-SWAP AWARE: reads $PROJECT_DIR/data/active_api_port. If the
#    shadow port (the OTHER ping-pong port) is also listening, we're
#    mid-swap — DO NOT kill anything. Just log + exit.
#
# 4. PR_SET_PDEATHSIG TRUST: workers now have kernel-level parent-death
#    signal (titan_plugin/core/worker_lifecycle.py). When parent dies,
#    workers self-shutdown via SIGTERM. So we no longer need aggressive
#    pgrep-by-name orphan kills — we just verify the canonical kernel is
#    alive and clean any holdover PID files.
#
# 5. ORPHAN BACKSTOP: even with PDEATHSIG, a backstop exists for the case
#    where worker_lifecycle protection failed to install (e.g. fork race
#    before prctl). We check for processes with PPID=1 matching titan
#    names. If found AND the active kernel doesn't claim them as children,
#    they're true orphans and we kill them.
#
# 6. SAFE RESTART: uses scripts/safe_restart.sh with dream-state check
#    when API is up. Falls back to --force when API is dead (no way to
#    verify dream state on a corpse). Three-state-health-check directive
#    is honored at the safe_restart layer.

set -uo pipefail  # NO -e (single check failure shouldn't abort the whole watchdog)

TITAN_ID="${1:-T1}"
NOW=$(date -u '+%Y-%m-%d %H:%M:%S UTC')

# Per-Titan config
case "$TITAN_ID" in
    T1)
        PROJECT_DIR="/home/antigravity/projects/titan"
        API_PORT_DEFAULT=7777
        SHADOW_PORT=7779
        BRAIN_LOG="/tmp/titan_brain.log"
        ;;
    T2)
        PROJECT_DIR="/home/antigravity/projects/titan"
        API_PORT_DEFAULT=7777
        SHADOW_PORT=7779
        BRAIN_LOG="/tmp/titan2_brain.log"
        ;;
    T3)
        PROJECT_DIR="/home/antigravity/projects/titan3"
        API_PORT_DEFAULT=7778
        SHADOW_PORT=7780
        BRAIN_LOG="/tmp/titan3_brain.log"
        ;;
    *)
        echo "[$NOW] ERROR: unknown Titan ID '$TITAN_ID' (expected T1|T2|T3)"
        exit 1
        ;;
esac

PIDFILE="${PROJECT_DIR}/data/titan_main.pid"
ACTIVE_PORT_FILE="${PROJECT_DIR}/data/active_api_port"
VENV="${PROJECT_DIR}/test_env/bin/activate"

# ── Phase C C-S2: l0_rust_enabled flag check ────────────────────────
#
# Per PLAN_microkernel_phase_c_s2_kernel.md §12.2 + SPEC §14 + §3.0:
# default-false → byte-identical to today's Python boot path. When
# flag-on, the titan-kernel-rs binary owns the L0 boot, supervision,
# and bus broker — its own respawn supervisor (set up in C-S8) handles
# restart, NOT this cron watchdog. We just verify the binary exists +
# log; the kernel-rs supervisor handles liveness.
L0_RUST_FLAG=$(python3 -c "
import sys
try:
    import tomllib
except ImportError:
    import tomli as tomllib
try:
    with open('${PROJECT_DIR}/titan_plugin/config.toml','rb') as f:
        c = tomllib.load(f)
    print(bool(c.get('microkernel', {}).get('l0_rust_enabled', False)))
except Exception:
    print(False)
" 2>/dev/null || echo "False")

if [ "$L0_RUST_FLAG" = "True" ]; then
    KERNEL_BINARY="${PROJECT_DIR}/titan-rust/target/release/titan-kernel-rs"
    if [ ! -x "$KERNEL_BINARY" ]; then
        echo "[$NOW] ${TITAN_ID}: ERROR: l0_rust_enabled=true but $KERNEL_BINARY not found"
        exit 1
    fi
    echo "[$NOW] ${TITAN_ID}: l0_rust_enabled=true — kernel-rs supervisor (C-S8) handles lifecycle; cron watchdog noop"
    exit 0
fi
# Default-false path falls through unchanged (byte-identical to today).

# Resolve current canonical port (B.1 ping-pong: 7777↔7779 or 7778↔7780)
if [ -f "$ACTIVE_PORT_FILE" ]; then
    API_PORT=$(cat "$ACTIVE_PORT_FILE" 2>/dev/null | tr -d '[:space:]')
    [ -z "$API_PORT" ] && API_PORT=$API_PORT_DEFAULT
else
    API_PORT=$API_PORT_DEFAULT
fi

# Helper: is anything listening on a given port? (ss is faster than netstat)
_port_listening() {
    ss -tln 2>/dev/null | awk '{print $4}' | grep -qE ":$1\$"
}

# ── Step 0: Honor the manage-script restart lockfile ────────────────
#
# 2026-04-29 — closes BUG-DUPLICATE-KERNELS-FRAGMENT-BUS-20260428. When
# safe_restart.sh / t2_manage.sh / t3_manage.sh starts a restart it
# writes /tmp/titan{1,2,3}_restart.lock with a unix timestamp. While
# fresh (<90s), this watchdog MUST defer — otherwise it races the
# manage-script restart, kills the freshly-spawned parent, and the
# manage-script's `start` step then spawns a SECOND parent → 2 kernels
# coexist → DivineBus fragments across them → bus messages "disappear".
#
# Symptom we saw 2026-04-29 09:01 UTC after deploy_t2.sh --restart:
# T2 had two PPID=1 titan_main parents (1041563 + 1041871), each with
# its own children, fragmenting the bus.
#
# 90s window covers ~75s SIGTERM/SIGKILL grace + spawn (matches the
# safe_restart.sh trap-on-INT/TERM cleanup expectation). On crash the
# lockfile stays — we treat ages > 90s as expired so a crashed restart
# can't permanently disable the watchdog.
TITAN_NUM="${TITAN_ID#T}"
RESTART_LOCK="/tmp/titan${TITAN_NUM}_restart.lock"
LOCK_MAX_AGE=90
if [ -f "$RESTART_LOCK" ]; then
    LOCK_TS=$(cat "$RESTART_LOCK" 2>/dev/null | tr -d '[:space:]')
    if [[ "$LOCK_TS" =~ ^[0-9]+$ ]]; then
        NOW_TS=$(date +%s)
        LOCK_AGE=$((NOW_TS - LOCK_TS))
        if [ "$LOCK_AGE" -lt "$LOCK_MAX_AGE" ]; then
            echo "[$NOW] ${TITAN_ID}: restart lock fresh (age=${LOCK_AGE}s, max=${LOCK_MAX_AGE}s) — skip cycle (manage-script restart in flight)"
            exit 0
        fi
    fi
fi

# ── Step 1: Detect mid-shadow-swap state ────────────────────────────
#
# If both ports are listening, B.1 shadow swap is in flight (Phase 3 has
# spawned shadow but Phase 4 hasn't completed nginx swap yet, or Phase 5
# hasn't shut down old kernel). DO NOT touch anything — let the
# orchestrator finish.

if _port_listening "$API_PORT_DEFAULT" && _port_listening "$SHADOW_PORT"; then
    echo "[$NOW] ${TITAN_ID}: shadow swap in flight (both $API_PORT_DEFAULT + $SHADOW_PORT listening) — skip"
    exit 0
fi

# ── Step 2: Detect + clean stale PID file ───────────────────────────
#
# titan_main writes its leader PID to $PIDFILE at boot + checks it at
# next-boot. If the file references a dead PID, the next start attempt
# aborts with "Another titan_main is already running" — root cause of
# the 2026-04-27 cascade. Clean stale BEFORE any restart logic.

if [ -f "$PIDFILE" ]; then
    PID=$(cat "$PIDFILE" 2>/dev/null | tr -d '[:space:]')
    if [ -n "$PID" ] && ! kill -0 "$PID" 2>/dev/null; then
        echo "[$NOW] ${TITAN_ID}: stale PID file (PID $PID does not exist) — removing"
        rm -f "$PIDFILE"
    fi
fi

# ── Step 3: Determine if titan is alive ─────────────────────────────
#
# Sources of truth (in order):
#   1. PID file exists + process alive
#   2. /health endpoint responds 200
# Both must be FALSE for us to consider it dead. Either alone is enough
# proof of life.

API_STATUS=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 \
    "http://127.0.0.1:${API_PORT}/health" 2>/dev/null || echo "000")

PID_ALIVE="no"
if [ -f "$PIDFILE" ]; then
    PID=$(cat "$PIDFILE" 2>/dev/null | tr -d '[:space:]')
    if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
        PID_ALIVE="yes"
    fi
fi

# ── Step 3.5: API-worker-disabled detection ─────────────────────────
#
# BUG-API-WORKER-CRASH-LOOP-CIRCUIT-BREAKER (2026-04-27): when api
# worker crash-loops 5× in 600s, Guardian disables it for 600s before
# auto-re-enable. During those 10 minutes the kernel keeps writing
# brain log (so the existing hung-detection at Step 3 sees "log
# active — skip") but no /health endpoint exists. Observatory shows
# the Titan disconnected for the whole window.
#
# Detection: kernel alive (PID present) AND nothing listening on
# $API_PORT. /v4/state can't be queried directly because it's served
# BY api — port-listening is the only out-of-band signal we have.
#
# State machine: flag file `/tmp/titan_<id>_api_disabled.flag` stores
# first-seen-disabled epoch. With cron firing every 5 min, the
# worst-case recovery is ~5-10 min after first detection (faster than
# Guardian's 600s auto-re-enable).
#   - first detection  → write flag, exit (no restart yet — could be
#                        boot/transient)
#   - flag age >120s   → restart (api truly stuck disabled)
#   - api recovers     → clear flag

API_DISABLED_FLAG="/tmp/titan_${TITAN_ID,,}_api_disabled.flag"
if [ "$PID_ALIVE" = "yes" ] && ! _port_listening "$API_PORT"; then
    NOW_EPOCH=$(date -u +%s)
    if [ -f "$API_DISABLED_FLAG" ]; then
        FIRST_SEEN=$(cat "$API_DISABLED_FLAG" 2>/dev/null | tr -d '[:space:]')
        if [ -n "$FIRST_SEEN" ]; then
            DISABLED_AGE=$((NOW_EPOCH - FIRST_SEEN))
            if [ "$DISABLED_AGE" -gt 120 ]; then
                echo "[$NOW] ${TITAN_ID}: api worker disabled ${DISABLED_AGE}s (>120s) — full restart"
                # Forensic capture before kill — same pattern as Step 3
                # hung-restart diagnostic.
                DIAG="/tmp/titan_${TITAN_ID,,}_apidown_$(date -u +%Y%m%d_%H%M%S).diag"
                {
                    echo "=== ${TITAN_ID} api-disabled-restart diagnostic ==="
                    echo "Time:           $NOW"
                    echo "Trigger:        api port $API_PORT silent ${DISABLED_AGE}s while kernel PID=$PID alive"
                    [ -n "$PID" ] && {
                        echo "Kernel uptime:  $(ps -o etime= -p "$PID" 2>/dev/null | tr -d ' ')"
                        echo "Kernel RSS MB:  $(ps -o rss= -p "$PID" 2>/dev/null | awk '{printf "%.1f", $1/1024}')"
                    }
                    echo "--- Last 200 brain log lines ---"
                    tail -200 "$BRAIN_LOG" 2>/dev/null
                } > "$DIAG"
                echo "[$NOW] ${TITAN_ID}: diagnostic at $DIAG"
                if [ -n "$PID" ]; then
                    PGID=$(ps -o pgid= -p "$PID" 2>/dev/null | tr -d ' ')
                    [ -n "$PGID" ] && kill -- -"$PGID" 2>/dev/null
                    sleep 3
                    [ -n "$PGID" ] && kill -9 -- -"$PGID" 2>/dev/null
                fi
                rm -f "$PIDFILE" "$API_DISABLED_FLAG"
                # Fall through to dead-titan restart branch
                API_STATUS="000"
                PID_ALIVE="no"
            else
                echo "[$NOW] ${TITAN_ID}: api worker disabled ${DISABLED_AGE}s (waiting for >120s threshold)"
                exit 0
            fi
        fi
    else
        echo "$NOW_EPOCH" > "$API_DISABLED_FLAG"
        echo "[$NOW] ${TITAN_ID}: api worker disabled (kernel alive, port $API_PORT silent) — first detection"
        exit 0
    fi
elif [ -f "$API_DISABLED_FLAG" ]; then
    # API recovered — clean flag
    rm -f "$API_DISABLED_FLAG"
    echo "[$NOW] ${TITAN_ID}: api worker recovered — flag cleared"
fi

if [ "$PID_ALIVE" = "yes" ] || [ "$API_STATUS" = "200" ]; then
    # Alive. Do startup-grace check — if process is < 120s old, skip the
    # busy-but-slow /health re-test (boot is slow under load).
    if [ "$PID_ALIVE" = "yes" ]; then
        AGE=$(ps -o etimes= -p "$PID" 2>/dev/null | tr -d ' ')
        if [ -n "$AGE" ] && [ "$AGE" -lt 120 ]; then
            echo "[$NOW] ${TITAN_ID}: alive (PID=$PID age=${AGE}s — startup grace, no health check)"
            exit 0
        fi
    fi

    # Truly-hung detection: /health failing AND brain log stale > 60s.
    # The log mtime side-channel catches the case where /health is slow
    # (memory consolidation, FAISS save, etc.) but the process is doing
    # real work. Only restart when both checks confirm the process is
    # not making progress.
    if [ "$API_STATUS" != "200" ] && [ -f "$BRAIN_LOG" ]; then
        LOG_MTIME=$(stat -c%Y "$BRAIN_LOG" 2>/dev/null || echo 0)
        NOW_EPOCH=$(date -u +%s)
        LOG_AGE=$((NOW_EPOCH - LOG_MTIME))
        if [ "$LOG_AGE" -gt 60 ]; then
            echo "[$NOW] ${TITAN_ID}: HUNG (/health=$API_STATUS + log ${LOG_AGE}s stale) — restart"
            # Capture forensic diagnostic before kill
            DIAG="/tmp/titan_${TITAN_ID,,}_crash_$(date -u +%Y%m%d_%H%M%S).diag"
            {
                echo "=== ${TITAN_ID} hung-restart diagnostic ==="
                echo "Time:       $NOW"
                echo "Trigger:    /health=$API_STATUS + log ${LOG_AGE}s stale"
                echo "PID:        $PID"
                [ -n "$PID" ] && {
                    echo "Uptime:     $(ps -o etime= -p "$PID" 2>/dev/null | tr -d ' ')"
                    echo "RSS (MB):   $(ps -o rss= -p "$PID" 2>/dev/null | awk '{printf "%.1f", $1/1024}')"
                }
                echo "--- Last 200 brain log lines ---"
                tail -200 "$BRAIN_LOG" 2>/dev/null
            } > "$DIAG"
            echo "[$NOW] ${TITAN_ID}: diagnostic at $DIAG"
            # Kill the entire process group so children die too
            if [ -n "$PID" ]; then
                PGID=$(ps -o pgid= -p "$PID" 2>/dev/null | tr -d ' ')
                [ -n "$PGID" ] && kill -- -"$PGID" 2>/dev/null
                sleep 3
                [ -n "$PGID" ] && kill -9 -- -"$PGID" 2>/dev/null
            fi
            rm -f "$PIDFILE"
            # Fall through to restart below (intentional — set API_STATUS so
            # we re-enter the dead-titan branch)
            API_STATUS="000"
            PID_ALIVE="no"
        else
            echo "[$NOW] ${TITAN_ID}: /health slow ($API_STATUS) but log active (${LOG_AGE}s ago) — skip"
            exit 0
        fi
    else
        echo "[$NOW] ${TITAN_ID}: alive (PID=$PID api=$API_STATUS)"
        exit 0
    fi
fi

# ── Step 4: Titan is dead — orphan backstop ─────────────────────────
#
# Even with PR_SET_PDEATHSIG, defensively check for orphaned workers
# (PPID=1, titan-named) before we restart. They might predate the
# protection install, or the install might have raced. Killing them
# reclaims memory + removes lock contention.

ORPHANS=$(ps -eo pid,ppid,comm 2>/dev/null | awk '$2 == 1 && /titan-/ {print $1}')
if [ -n "$ORPHANS" ]; then
    echo "[$NOW] ${TITAN_ID}: orphan backstop killing PIDs: $(echo $ORPHANS | tr '\n' ' ')"
    echo "$ORPHANS" | xargs -r kill -9 2>/dev/null
    sleep 2
fi

# ── Step 5: Safe restart ────────────────────────────────────────────
#
# Use scripts/safe_restart.sh which honors three-state-health-check
# (dream state). When titan is fully dead, dream state is unknown —
# pass --force to override (no point waiting for a corpse to wake).

cd "$PROJECT_DIR" || {
    echo "[$NOW] ${TITAN_ID}: cannot cd to $PROJECT_DIR — abort"
    exit 1
}

# Per-Titan local restart entry points (LOCAL — must work without SSH).
# T1: safe_restart.sh
# T2: t2_manage.sh restart  (NOT scripts/t2 — that's T1's SSH wrapper)
# T3: t3_manage.sh restart
# All three accept --force.
case "$TITAN_ID" in
    T1)
        echo "[$NOW] T1: dead — invoking safe_restart.sh t1 --force"
        if [ -f "$VENV" ]; then source "$VENV"; fi
        bash "${PROJECT_DIR}/scripts/safe_restart.sh" t1 --force 2>&1
        ;;
    T2)
        # CRITICAL: scripts/t2 SSHs from T1 → won't work when this watchdog
        # runs ON the T2 host (no SSH key for self-loop). Use t2_manage.sh
        # which is the local-on-T2 management entrypoint.
        # --force needed: dream state is "unknown" when API is dead
        # (three-state-health-check refuses unknown without explicit override).
        echo "[$NOW] T2: dead — invoking t2_manage.sh restart --force"
        bash "${PROJECT_DIR}/scripts/t2_manage.sh" restart --force 2>&1
        ;;
    T3)
        echo "[$NOW] T3: dead — invoking t3_manage.sh restart --force"
        bash "${PROJECT_DIR}/scripts/t3_manage.sh" restart --force 2>&1
        ;;
esac

# Verify recovery
sleep 5
RECOVERY_STATUS=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 \
    "http://127.0.0.1:${API_PORT}/health" 2>/dev/null || echo "000")
echo "[$NOW] ${TITAN_ID}: post-restart /health = $RECOVERY_STATUS"

# Log rotation: keep brain.log under 100 MB
LOG_SIZE=$(stat -c%s "$BRAIN_LOG" 2>/dev/null || echo 0)
if [ "$LOG_SIZE" -gt 104857600 ]; then
    DAY_STAMP=$(date -u '+%Y%m%d')
    echo "[$NOW] ${TITAN_ID}: rotating brain log ($LOG_SIZE bytes)"
    cp "$BRAIN_LOG" "${BRAIN_LOG}.${DAY_STAMP}"
    : > "$BRAIN_LOG"
    gzip "${BRAIN_LOG}.${DAY_STAMP}" 2>/dev/null &
fi
