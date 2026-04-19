#!/bin/bash
# t1_watchdog.sh — Watchdog + auto-restart for T1
# Run via cron every 5 minutes on T1:
#   */5 * * * * bash /home/antigravity/projects/titan/scripts/t1_watchdog.sh >> /tmp/titan1_watchdog.log 2>&1
#
# Functions:
#   1. Auto-restart T1 if crashed (with CWD-scoped duplicate protection)
#   2. Health check with 120s startup grace period
#   3. Log rotation when brain.log > 100MB

TITAN_DIR="/home/antigravity/projects/titan"
BRAIN_LOG="/tmp/titan_brain.log"
VENV="${TITAN_DIR}/test_env/bin/activate"
PIDFILE="/tmp/titan1.pid"

# Reliable process check via PID file
t1_is_running() {
    [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null
}

# Match VS Code ulimit (cron default is 1024 which causes fd exhaustion)
ulimit -n 1048576 2>/dev/null || ulimit -n 65536 2>/dev/null || true

NOW=$(date -u '+%Y-%m-%d %H:%M:%S UTC')
DAY_STAMP=$(date -u '+%Y%m%d')

# ── 1. AUTO-RESTART CHECK ──
if ! t1_is_running; then
    echo "[$NOW] T1 NOT RUNNING — auto-restarting..."
    rm -f "$PIDFILE"
    # Kill any orphan titan_main processes — only T1's (CWD-scoped)
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
    # Append (>>) so crash forensics from previous instance are preserved.
    # Log rotation below handles size bound.
    echo "" >> "$BRAIN_LOG"
    echo "=== [$NOW] RESTART boundary — auto-restart from crash ===" >> "$BRAIN_LOG"
    nohup python -u scripts/titan_main.py --server >> "$BRAIN_LOG" 2>&1 &
    echo "$!" > "$PIDFILE"
    echo "[$NOW] T1 restarted, PID=$!"
    sleep 15
    HTTP=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:7777/health 2>/dev/null)
    echo "[$NOW] T1 health after restart: $HTTP"
else
    # Startup grace period: don't force-restart if T1 started < 120s ago
    T1_PID=$(cat "$PIDFILE" 2>/dev/null)
    T1_AGE=$(ps -o etimes= -p "$T1_PID" 2>/dev/null | tr -d ' ')
    if [ -n "$T1_AGE" ] && [ "$T1_AGE" -lt 120 ]; then
        echo "[$NOW] T1 in startup grace period (${T1_AGE}s old) — skipping health check"
    else
        # Quick health check — if running but API unresponsive, *maybe* restart.
        # 2026-04-09: bumped --max-time 10 → 15 because /health worst-case
        # latency is balance(5s) + vault(4s) + memory_status(3s) = 12s when
        # mainnet RPC is slow.
        #
        # 2026-04-13 smart-watchdog: /health failure alone ISN'T enough to kill
        # a working process. T3 incident showed meditation / FAISS save /
        # memory consolidation can legitimately hold the event loop for 15-60s
        # while the process is otherwise healthy. We now use brain.log mtime
        # as a side-channel liveness signal: if logs are still being written,
        # the process is alive and just busy. Only restart when /health fails
        # AND logs have been quiet for 30+ seconds. Capture diagnostic before
        # any kill so we have forensic evidence.
        HTTP=$(curl -s -o /dev/null -w "%{http_code}" --max-time 15 http://127.0.0.1:7777/health 2>/dev/null)
        if [ "$HTTP" != "200" ]; then
            LOG_MTIME=$(stat -c%Y "$BRAIN_LOG" 2>/dev/null || echo 0)
            NOW_EPOCH=$(date -u +%s)
            LOG_AGE=$((NOW_EPOCH - LOG_MTIME))
            if [ "$LOG_AGE" -lt 30 ]; then
                echo "[$NOW] T1 /health slow ($HTTP) but log active (${LOG_AGE}s ago) — skipping restart (busy, not hung)"
            else
                # Truly hung: /health failing AND log stale.
                # Capture diagnostic BEFORE killing so post-mortem is possible.
                DIAG="/tmp/titan1_crash_$(date -u +%Y%m%d_%H%M%S).diag"
                {
                    echo "=== T1 crash diagnostic ==="
                    echo "Captured:   $NOW"
                    echo "Trigger:    /health=$HTTP + log stale (${LOG_AGE}s)"
                    echo "PID:        $T1_PID"
                    if [ -n "$T1_PID" ]; then
                        echo "Uptime:     $(ps -o etime= -p "$T1_PID" 2>/dev/null | tr -d ' ')"
                        echo "RSS (MB):   $(ps -o rss= -p "$T1_PID" 2>/dev/null | awk '{printf "%.1f", $1/1024}')"
                        echo "PGID:       $(ps -o pgid= -p "$T1_PID" 2>/dev/null | tr -d ' ')"
                        echo "Threads:    $(ls /proc/$T1_PID/task 2>/dev/null | wc -l)"
                    fi
                    echo "--- Last 150 log lines ---"
                    tail -150 "$BRAIN_LOG" 2>/dev/null
                    echo "--- End diagnostic ---"
                } > "$DIAG"
                echo "[$NOW] T1 truly hung (log ${LOG_AGE}s stale, /health=$HTTP) — diagnostic: $DIAG — force restart..."
                # Kill entire process group (parent + all child workers)
                T1_PGID=$(ps -o pgid= -p "$T1_PID" 2>/dev/null | tr -d ' ')
                if [ -n "$T1_PGID" ]; then
                    kill -- -"$T1_PGID" 2>/dev/null
                    sleep 3
                    kill -9 -- -"$T1_PGID" 2>/dev/null
                else
                    kill -9 "$T1_PID" 2>/dev/null
                fi
                rm -f "$PIDFILE"
                # Also kill any CWD-scoped orphans not in the group
                for pid in $(pgrep -f "python.*titan_main" 2>/dev/null); do
                    PID_CWD=$(readlink /proc/$pid/cwd 2>/dev/null || echo "")
                    if [ "$PID_CWD" = "$TITAN_DIR" ]; then
                        kill -9 "$pid" 2>/dev/null
                    fi
                done
                fuser -k 7777/tcp 2>/dev/null
                sleep 3
                cd "$TITAN_DIR"
                source "$VENV" 2>/dev/null
                export OPENROUTER_API_KEY=
                # Append (>>) so crash forensics from previous instance are preserved.
                echo "" >> "$BRAIN_LOG"
                echo "=== [$NOW] RESTART boundary — force-restart (hung, diag=$DIAG) ===" >> "$BRAIN_LOG"
                nohup python -u scripts/titan_main.py --server >> "$BRAIN_LOG" 2>&1 &
                echo "$!" > "$PIDFILE"
                echo "[$NOW] T1 force-restarted, PID=$!"
            fi
        fi
    fi
fi

# ── 2. LOG ROTATION (if > 100MB) ──
LOG_SIZE=$(stat -c%s "$BRAIN_LOG" 2>/dev/null || echo 0)
if [ "$LOG_SIZE" -gt 104857600 ]; then
    echo "[$NOW] Rotating brain log ($LOG_SIZE bytes)..."
    cp "$BRAIN_LOG" "${BRAIN_LOG}.${DAY_STAMP}"
    : > "$BRAIN_LOG"  # truncate in-place (preserves running process fd with O_APPEND)
    gzip "${BRAIN_LOG}.${DAY_STAMP}" 2>/dev/null &
    echo "[$NOW] Log rotated"
fi
