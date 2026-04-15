#!/bin/bash
# t2_watchdog.sh — Watchdog + telemetry for T2
# Run via cron every 5 minutes on T2 VPS:
#   */5 * * * * bash /home/antigravity/projects/titan/scripts/t2_watchdog.sh >> /tmp/titan2_watchdog.log 2>&1
#
# Functions:
#   1. Auto-restart T2 if crashed (with duplicate process protection)
#   2. Log rotation when brain.log > 100MB

TITAN_DIR="/home/antigravity/projects/titan"
BRAIN_LOG="/tmp/titan2_brain.log"
MANAGE="${TITAN_DIR}/scripts/t2_manage.sh"
VENV="${TITAN_DIR}/test_env/bin/activate"
PIDFILE="/tmp/titan2.pid"

# Reliable process check via PID file
t2_is_running() {
    [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null
}

# Match T1's ulimit (VS Code raises it; SSH/cron default is 1024 which causes fd exhaustion)
ulimit -n 1048576 2>/dev/null || ulimit -n 65536 2>/dev/null || true

NOW=$(date -u '+%Y-%m-%d %H:%M:%S UTC')
DAY_STAMP=$(date -u '+%Y%m%d')

# ── 1. AUTO-RESTART CHECK ──
if ! t2_is_running; then
    echo "[$NOW] T2 NOT RUNNING — auto-restarting..."
    rm -f "$PIDFILE"
    # Kill any orphan titan_main processes — only T2's (CWD-scoped)
    for pid in $(pgrep -f "python.*titan_main" 2>/dev/null); do
        PID_CWD=$(readlink /proc/$pid/cwd 2>/dev/null || echo "")
        if [ "$PID_CWD" = "$TITAN_DIR" ]; then
            kill -9 "$pid" 2>/dev/null
        fi
    done
    # Kill anything on port 7777
    fuser -k 7777/tcp 2>/dev/null
    sleep 3
    cd "$TITAN_DIR"
    source "$VENV" 2>/dev/null
    export OPENROUTER_API_KEY=
    export TITAN_KIN_ADDRESSES="https://iamtitan.tech"
    nohup python -u scripts/titan_main.py --server >> "$BRAIN_LOG" 2>&1 &
    echo "$!" > "$PIDFILE"
    echo "[$NOW] T2 restarted, PID=$!"
    sleep 15
    HTTP=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:7777/health 2>/dev/null)
    echo "[$NOW] T2 health after restart: $HTTP"
else
    # Startup grace period: don't force-restart if T2 started < 120s ago
    T2_PID=$(cat "$PIDFILE" 2>/dev/null)
    T2_AGE=$(ps -o etimes= -p "$T2_PID" 2>/dev/null | tr -d ' ')
    if [ -n "$T2_AGE" ] && [ "$T2_AGE" -lt 120 ]; then
        echo "[$NOW] T2 in startup grace period (${T2_AGE}s old) — skipping health check"
    else
        # Smart health check (2026-04-13): /health timeout alone isn't enough.
        # Meditation/FAISS save/memory consolidation can legitimately hold the
        # event loop for 15-60s while the process is alive. Use brain.log
        # mtime as side-channel liveness signal and capture diagnostic before
        # any kill. See t1_watchdog.sh for full rationale.
        HTTP=$(curl -s -o /dev/null -w "%{http_code}" --max-time 15 http://localhost:7777/health 2>/dev/null)
        if [ "$HTTP" != "200" ]; then
            LOG_MTIME=$(stat -c%Y "$BRAIN_LOG" 2>/dev/null || echo 0)
            NOW_EPOCH=$(date -u +%s)
            LOG_AGE=$((NOW_EPOCH - LOG_MTIME))
            if [ "$LOG_AGE" -lt 30 ]; then
                echo "[$NOW] T2 /health slow ($HTTP) but log active (${LOG_AGE}s ago) — skipping restart (busy, not hung)"
            else
                DIAG="/tmp/titan2_crash_$(date -u +%Y%m%d_%H%M%S).diag"
                {
                    echo "=== T2 crash diagnostic ==="
                    echo "Captured:   $NOW"
                    echo "Trigger:    /health=$HTTP + log stale (${LOG_AGE}s)"
                    echo "PID:        $T2_PID"
                    if [ -n "$T2_PID" ]; then
                        echo "Uptime:     $(ps -o etime= -p "$T2_PID" 2>/dev/null | tr -d ' ')"
                        echo "RSS (MB):   $(ps -o rss= -p "$T2_PID" 2>/dev/null | awk '{printf "%.1f", $1/1024}')"
                        echo "PGID:       $(ps -o pgid= -p "$T2_PID" 2>/dev/null | tr -d ' ')"
                        echo "Threads:    $(ls /proc/$T2_PID/task 2>/dev/null | wc -l)"
                    fi
                    echo "--- Last 150 log lines ---"
                    tail -150 "$BRAIN_LOG" 2>/dev/null
                    echo "--- End diagnostic ---"
                } > "$DIAG"
                echo "[$NOW] T2 truly hung (log ${LOG_AGE}s stale, /health=$HTTP) — diagnostic: $DIAG — force restart..."
                bash "$MANAGE" stop
                sleep 3
                cd "$TITAN_DIR"
                source "$VENV" 2>/dev/null
                export OPENROUTER_API_KEY=
                export TITAN_KIN_ADDRESSES="https://iamtitan.tech"
                echo "" >> "$BRAIN_LOG"
                echo "=== [$NOW] RESTART boundary — force-restart (hung, diag=$DIAG) ===" >> "$BRAIN_LOG"
                nohup python -u scripts/titan_main.py --server >> "$BRAIN_LOG" 2>&1 &
                echo "$!" > "$PIDFILE"
                echo "[$NOW] T2 force-restarted, PID=$!"
            fi
        fi
    fi
fi

# ── 2. LOG ROTATION (if > 100MB) ──
LOG_SIZE=$(stat -c%s "$BRAIN_LOG" 2>/dev/null || echo 0)
if [ "$LOG_SIZE" -gt 104857600 ]; then
    echo "[$NOW] Rotating brain log ($LOG_SIZE bytes)..."
    mv "$BRAIN_LOG" "${BRAIN_LOG}.${DAY_STAMP}"
    gzip "${BRAIN_LOG}.${DAY_STAMP}" 2>/dev/null &
    echo "[$NOW] Log rotated"
fi
