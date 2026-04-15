#!/bin/bash
# Observatory Watchdog — checks frontend + backend, alerts via Telegram
# Runs via cron every 2 minutes. Only alerts ONCE per incident (cooldown file).
#
# Monitors:
#   - Frontend (Next.js) on port 3000
#   - Backend API on port 7777
#   - Auto-restarts frontend if down

TELEGRAM_BOT_TOKEN="8531091229:AAElGsqbsLDvDfxaCwMwG1qGdBzyVlksI0c"
TELEGRAM_CHAT_ID="6345894322"
COOLDOWN_FILE="/tmp/observatory_watchdog_alerted"
COOLDOWN_SECONDS=600  # Don't re-alert within 10 minutes
LOG="/tmp/observatory_watchdog.log"
PROJECT_DIR="/home/antigravity/projects/titan"

send_telegram() {
    local msg="$1"
    curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
        -d chat_id="$TELEGRAM_CHAT_ID" \
        -d text="$msg" \
        -d parse_mode="Markdown" > /dev/null 2>&1
}

should_alert() {
    if [ ! -f "$COOLDOWN_FILE" ]; then
        return 0  # No cooldown, should alert
    fi
    local last_alert=$(cat "$COOLDOWN_FILE" 2>/dev/null || echo 0)
    local now=$(date +%s)
    local diff=$((now - last_alert))
    if [ "$diff" -ge "$COOLDOWN_SECONDS" ]; then
        return 0  # Cooldown expired
    fi
    return 1  # Still in cooldown
}

mark_alerted() {
    date +%s > "$COOLDOWN_FILE"
}

clear_cooldown() {
    rm -f "$COOLDOWN_FILE"
}

# --- Check Frontend (port 3000) ---
# Use 127.0.0.1 (not localhost) to avoid IPv6 resolution timeout
FRONTEND_STATUS=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 http://127.0.0.1:3000 2>/dev/null)
FRONTEND_OK=true
if [ "$FRONTEND_STATUS" != "200" ]; then
    FRONTEND_OK=false
    echo "$(date '+%Y-%m-%d %H:%M:%S') FRONTEND DOWN (HTTP $FRONTEND_STATUS)" >> "$LOG"

    # Try auto-restart
    echo "$(date '+%Y-%m-%d %H:%M:%S') Attempting frontend restart..." >> "$LOG"
    pkill -f "next-server" 2>/dev/null
    sleep 2
    cd "$PROJECT_DIR/titan-observatory" && nohup npx next start -p 3000 > /tmp/next_server.log 2>&1 &
    sleep 5

    # Re-check
    FRONTEND_STATUS=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 http://127.0.0.1:3000 2>/dev/null)
    if [ "$FRONTEND_STATUS" = "200" ]; then
        FRONTEND_OK=true
        echo "$(date '+%Y-%m-%d %H:%M:%S') Frontend auto-recovered" >> "$LOG"
        if should_alert; then
            send_telegram "⚡ *Observatory Alert*
Frontend was down — auto-restarted successfully.
Status: ✅ recovered"
            mark_alerted
        fi
    fi
fi

# --- Check Backend API (port 7777) ---
# Startup grace period: skip backend check if titan_main started < 120s ago
TITAN_PID=$(pgrep -o -f "titan_main.*--server" 2>/dev/null)
TITAN_AGE=""
[ -n "$TITAN_PID" ] && TITAN_AGE=$(ps -o etimes= -p "$TITAN_PID" 2>/dev/null | tr -d ' ')
BACKEND_OK=true
if [ -n "$TITAN_AGE" ] && [ "$TITAN_AGE" -lt 120 ] 2>/dev/null; then
    # Titan just started — API may not be ready yet, skip check
    true
else
    # Use 127.0.0.1 (not localhost) to avoid IPv6 resolution timeout.
    # 20s timeout: event loop may be briefly busy under load.
    BACKEND_STATUS=$(curl -s -o /dev/null -w "%{http_code}" --max-time 20 http://127.0.0.1:7777/health 2>/dev/null)
    if [ "$BACKEND_STATUS" != "200" ]; then
        # Retry once after 5s — transient event loop congestion
        sleep 5
        BACKEND_STATUS=$(curl -s -o /dev/null -w "%{http_code}" --max-time 20 http://127.0.0.1:7777/health 2>/dev/null)
        if [ "$BACKEND_STATUS" != "200" ]; then
            BACKEND_OK=false
            echo "$(date '+%Y-%m-%d %H:%M:%S') BACKEND DOWN (HTTP $BACKEND_STATUS)" >> "$LOG"
        fi
    fi
fi

# --- Alert if anything is down ---
if [ "$FRONTEND_OK" = false ] || [ "$BACKEND_OK" = false ]; then
    if should_alert; then
        # Check if processes are running to distinguish crash vs overload
        TITAN_RUNNING=$(pgrep -f "titan_main.*--server" >/dev/null 2>&1 && echo "yes" || echo "no")
        NEXT_RUNNING=$(pgrep -f "next-server" >/dev/null 2>&1 && echo "yes" || echo "no")

        MSG="🚨 *Observatory Issue*"
        [ "$FRONTEND_OK" = false ] && MSG="$MSG
Frontend (3000): ❌ HTTP $FRONTEND_STATUS (process: $NEXT_RUNNING)"
        [ "$BACKEND_OK" = false ] && MSG="$MSG
Backend (7777): ❌ HTTP $BACKEND_STATUS (process: $TITAN_RUNNING)"

        # Add actionable context
        if [ "$TITAN_RUNNING" = "no" ] && [ "$BACKEND_OK" = false ]; then
            MSG="$MSG

⚠️ _Backend process not running — likely crashed. Check VPS._"
        elif [ "$BACKEND_STATUS" = "000" ] && [ "$TITAN_RUNNING" = "yes" ]; then
            MSG="$MSG

⏳ _Backend running but not responding — event loop may be overloaded. Will retry._"
        else
            MSG="$MSG

_Check VPS if this persists._"
        fi
        send_telegram "$MSG"
        mark_alerted
    fi
else
    # Everything OK — clear cooldown so next incident alerts immediately
    clear_cooldown
fi

# Trim log
tail -100 "$LOG" > "${LOG}.tmp" 2>/dev/null && mv "${LOG}.tmp" "$LOG" 2>/dev/null
