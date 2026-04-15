#!/bin/bash
# t3_watchdog.sh — Watchdog + telemetry for T3 birth experiment
# Run via cron every 5 minutes on T2 VPS:
#   */5 * * * * bash /home/antigravity/projects/titan3/scripts/t3_watchdog.sh >> /tmp/titan3_watchdog.log 2>&1
#
# Functions:
#   1. Auto-restart T3 if crashed
#   2. Hourly developmental telemetry snapshot → data/telemetry/
#   3. Daily data backup → data/backups/
#   4. Log rotation when brain.log > 100MB

TITAN3_DIR="/home/antigravity/projects/titan3"
BRAIN_LOG="/tmp/titan3_brain.log"
TELEM_DIR="${TITAN3_DIR}/data/telemetry"
BACKUP_DIR="${TITAN3_DIR}/data/backups"
MANAGE="${TITAN3_DIR}/scripts/t3_manage.sh"
VENV="/home/antigravity/projects/titan3/test_env/bin/activate"  # dedicated T3 venv
PIDFILE="/tmp/titan3.pid"

# Reliable process check via PID file
t3_is_running() {
    [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null
}

# Match T1's ulimit (VS Code raises it; SSH/cron default is 1024 which causes fd exhaustion)
ulimit -n 1048576 2>/dev/null || ulimit -n 65536 2>/dev/null || true

NOW=$(date -u '+%Y-%m-%d %H:%M:%S UTC')
HOUR=$(date -u '+%H')
MINUTE=$(date -u '+%M')
DAY_STAMP=$(date -u '+%Y%m%d')
HOUR_STAMP=$(date -u '+%Y%m%d_%H%M')

# Ensure dirs exist
mkdir -p "$TELEM_DIR" "$BACKUP_DIR"

# ── 1. AUTO-RESTART CHECK ──
if ! t3_is_running; then
    echo "[$NOW] T3 NOT RUNNING — auto-restarting..."
    rm -f "$PIDFILE"
    # Kill any orphan titan_main processes — only T3's (CWD-scoped)
    for pid in $(pgrep -f "python.*titan_main" 2>/dev/null); do
        PID_CWD=$(readlink /proc/$pid/cwd 2>/dev/null || echo "")
        if [ "$PID_CWD" = "$TITAN3_DIR" ]; then
            kill -9 "$pid" 2>/dev/null
        fi
    done
    # Kill anything on port 7778
    fuser -k 7778/tcp 2>/dev/null
    sleep 2
    cd "$TITAN3_DIR"
    source "$VENV" 2>/dev/null
    export OPENROUTER_API_KEY=
    export TITAN_KIN_ADDRESSES="http://localhost:7777"
    nohup python -u scripts/titan_main.py --server >> "$BRAIN_LOG" 2>&1 &
    echo "$!" > "$PIDFILE"
    echo "[$NOW] T3 restarted, PID=$!"
    sleep 15
    HTTP=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:7778/health 2>/dev/null)
    echo "[$NOW] T3 health after restart: $HTTP"
else
    # Startup grace period: don't force-restart if T3 started < 120s ago
    T3_PID=$(cat "$PIDFILE" 2>/dev/null)
    T3_AGE=$(ps -o etimes= -p "$T3_PID" 2>/dev/null | tr -d ' ')
    if [ -n "$T3_AGE" ] && [ "$T3_AGE" -lt 120 ]; then
        echo "[$NOW] T3 in startup grace period (${T3_AGE}s old) — skipping health check"
    else
        # Smart health check (2026-04-13): /health timeout alone isn't enough.
        # Meditation/FAISS save/memory consolidation can legitimately hold the
        # event loop for 15-60s while the process is alive. Use brain.log
        # mtime as side-channel liveness signal and capture diagnostic before
        # any kill. The T3 incident on 2026-04-13 08:30 UTC (/health timeout
        # during meditation, lost v_history=495 to false-positive restart)
        # motivated this change. See t1_watchdog.sh for full rationale.
        HTTP=$(curl -s -o /dev/null -w "%{http_code}" --max-time 15 http://localhost:7778/health 2>/dev/null)
        if [ "$HTTP" != "200" ]; then
            LOG_MTIME=$(stat -c%Y "$BRAIN_LOG" 2>/dev/null || echo 0)
            NOW_EPOCH=$(date -u +%s)
            LOG_AGE=$((NOW_EPOCH - LOG_MTIME))
            if [ "$LOG_AGE" -lt 30 ]; then
                echo "[$NOW] T3 /health slow ($HTTP) but log active (${LOG_AGE}s ago) — skipping restart (busy, not hung)"
            else
                PID=$(cat "$PIDFILE" 2>/dev/null)
                DIAG="/tmp/titan3_crash_$(date -u +%Y%m%d_%H%M%S).diag"
                {
                    echo "=== T3 crash diagnostic ==="
                    echo "Captured:   $NOW"
                    echo "Trigger:    /health=$HTTP + log stale (${LOG_AGE}s)"
                    echo "PID:        $PID"
                    if [ -n "$PID" ]; then
                        echo "Uptime:     $(ps -o etime= -p "$PID" 2>/dev/null | tr -d ' ')"
                        echo "RSS (MB):   $(ps -o rss= -p "$PID" 2>/dev/null | awk '{printf "%.1f", $1/1024}')"
                        echo "PGID:       $(ps -o pgid= -p "$PID" 2>/dev/null | tr -d ' ')"
                        echo "Threads:    $(ls /proc/$PID/task 2>/dev/null | wc -l)"
                    fi
                    echo "--- Last 150 log lines ---"
                    tail -150 "$BRAIN_LOG" 2>/dev/null
                    echo "--- End diagnostic ---"
                } > "$DIAG"
                echo "[$NOW] T3 truly hung (log ${LOG_AGE}s stale, /health=$HTTP) — diagnostic: $DIAG — force restart..."
                # Kill entire process group (parent + all child workers)
                T3_PGID=$(ps -o pgid= -p "$PID" 2>/dev/null | tr -d ' ')
                if [ -n "$T3_PGID" ]; then
                    kill -- -"$T3_PGID" 2>/dev/null
                    sleep 3
                    kill -9 -- -"$T3_PGID" 2>/dev/null
                else
                    kill -9 "$PID" 2>/dev/null
                fi
                rm -f "$PIDFILE"
                # Also kill any CWD-scoped orphans not in the group
                for pid in $(pgrep -f "python.*titan_main" 2>/dev/null); do
                    PID_CWD=$(readlink /proc/$pid/cwd 2>/dev/null || echo "")
                    if [ "$PID_CWD" = "$TITAN3_DIR" ]; then
                        kill -9 "$pid" 2>/dev/null
                    fi
                done
                fuser -k 7778/tcp 2>/dev/null
                sleep 3
                cd "$TITAN3_DIR"
                source "$VENV" 2>/dev/null
                export OPENROUTER_API_KEY=
                export TITAN_KIN_ADDRESSES="http://localhost:7777"
                echo "" >> "$BRAIN_LOG"
                echo "=== [$NOW] RESTART boundary — force-restart (hung, diag=$DIAG) ===" >> "$BRAIN_LOG"
                nohup python -u scripts/titan_main.py --server >> "$BRAIN_LOG" 2>&1 &
                echo "$!" > "$PIDFILE"
                echo "[$NOW] T3 force-restarted, PID=$!"
            fi
        fi
    fi
fi

# ── 2. HOURLY TELEMETRY (at minute 0 or 5) ──
if [ "$MINUTE" = "00" ] || [ "$MINUTE" = "05" ]; then
    # Only run once per hour (check if this hour's file exists)
    TELEM_FILE="${TELEM_DIR}/snap_${HOUR_STAMP}.json"
    if [ ! -f "$TELEM_FILE" ]; then
        echo "[$NOW] Collecting telemetry snapshot..."
        cd "$TITAN3_DIR"
        source "$VENV" 2>/dev/null
        bash "$MANAGE" telemetry > "$TELEM_FILE" 2>/dev/null
        echo "[$NOW] Telemetry saved: $TELEM_FILE"
    fi
fi

# ── 3. DAILY BACKUP (at 03:00 or 03:05 UTC — slack window in case cron misses) ──
# 2026-04-14 fix: previous condition `[ HOUR=03 ] && [ MIN=00 ] || [ MIN=05 ]`
# was incorrectly parsed as `(HOUR=03 && MIN=00) || MIN=05` — firing every
# xx:05 of any hour. File-exists guard prevented duplicate tarballs but
# intent was clearly 03:00 only. Now using explicit grouping.
if [ "$HOUR" = "03" ] && { [ "$MINUTE" = "00" ] || [ "$MINUTE" = "05" ]; }; then
    BACKUP_FILE="${BACKUP_DIR}/t3_data_${DAY_STAMP}.tar.gz"
    if [ ! -f "$BACKUP_FILE" ]; then
        echo "[$NOW] Daily backup..."
        cd "$TITAN3_DIR"
        # 2026-04-14 fix: exclude studio_exports (generated art/music JPG+WAV,
        # don't compress, regenerable). Was inflating daily tarball from ~1GB
        # to ~5.8GB and filled T2/T3 154GB disk in 8 days. Keep backup scoped
        # to actual state: databases, neural weights, timechain, MSL, soul.
        tar -czf "$BACKUP_FILE" \
            --exclude='data/backups' \
            --exclude='data/telemetry' \
            --exclude='data/studio_exports' \
            --exclude='*.pyc' \
            --exclude='__pycache__' \
            data/ 2>/dev/null
        BACKUP_SIZE=$(du -h "$BACKUP_FILE" 2>/dev/null | cut -f1)
        echo "[$NOW] Backup complete: $BACKUP_FILE ($BACKUP_SIZE)"

        # Keep only last 3 days (was 7 — 7×5.8GB exhausted 154GB disk).
        # 3 × ~1GB (post-exclude) = well within budget.
        find "$BACKUP_DIR" -name "t3_data_*.tar.gz" -mtime +3 -delete 2>/dev/null
    fi
fi

# ── 4. LOG ROTATION (if > 100MB) ──
LOG_SIZE=$(stat -c%s "$BRAIN_LOG" 2>/dev/null || echo 0)
if [ "$LOG_SIZE" -gt 104857600 ]; then
    echo "[$NOW] Rotating brain log ($LOG_SIZE bytes)..."
    mv "$BRAIN_LOG" "${BRAIN_LOG}.${DAY_STAMP}"
    gzip "${BRAIN_LOG}.${DAY_STAMP}" 2>/dev/null &
    echo "[$NOW] Log rotated"
fi
